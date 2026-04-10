# 8.3 模型导出与格式转换

前两节我们学习了两种优化模型推理速度的方法：`torch.compile()` 通过图编译来减少 Python 开销和融合算子，量化通过降低数值精度来减小体积并加速计算。但优化后的模型仍然以 PyTorch 的 `.pt` / `.bin` 格式存在——这是 PyTorch 生态内部的格式，如果你想把模型部署到生产环境（比如一个 FastAPI 服务、一个移动端应用、或者一个边缘设备），通常需要把它转换为其他更通用的或更适合特定场景的格式。这一节我们将学习主流的模型导出方案：ONNX（跨平台部署标准）、GGUF（llama.cpp/Ollama 格式）、SafeTensors（高效存储格式），以及如何在不同格式之间做选择。

## ONNX：跨平台推理的标准格式

**ONNX（Open Neural Network Exchange）** 是微软和 Facebook 联合推出的开放神经网络交换格式，已经成为深度学习模型部署的事实标准。几乎所有主流推理框架都支持 ONNX——TensorRT、ONNX Runtime、OpenVINO、CoreML、TensorFlow Lite 等。把 PyTorch 模型导出为 ONNX 格式，意味着你可以在几乎任何平台上运行它，而不依赖 PyTorch 运行时。

### 基本导出流程

```python
import torch
import torch.onnx


def export_to_onnx(model, tokenizer, output_path="model.onnx"):
    """将 GPT 模型导出为 ONNX 格式"""
    model.eval()
    
    dummy_input = torch.randint(0, 1000, (1, 128))
    
    torch.onnx.export(
        model,
        (dummy_input,),
        f=output_path,
        input_names=['input_ids'],
        output_names=['logits'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'logits':   {0: 'batch_size', 1: 'sequence_length'},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    
    import os
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    
    print(f"✓ ONNX model exported to {output_path}")
    print(f"  File size: {size_mb:.1f} MB")
    
    return output_path


export_to_onnx(your_model, your_tokenizer)
```

### 动态轴（Dynamic Axes）

注意上面代码中的 `dynamic_axes` 参数。LLM 推理中输入序列长度是可变的——用户的问题可能只有 10 个 token，也可能有 2048 个 token。如果不指定动态轴，ONNX 导出器会使用第一次追踪时的固定 shape（比如上面的 `torch.Size([1, 128])`），之后所有输入都必须严格匹配这个 shape，否则会报错。通过指定 `dynamic_axes`，我们告诉 ONNX "batch 维度和序列长度维度可以是任意大小"，这样导出的模型就能处理不同长度的输入了。

### 验证导出的 ONNX 模型

导出之后必须验证模型的正确性——确保 ONNX 版本的输出与 PyTorch 版本一致：

```python
def verify_onnx_export(model, onnx_path):
    """验证 ONNX 导出的正确性"""
    import onnxruntime as ort
    import numpy as np

    model.eval()
    
    test_input = torch.randint(0, 1000, (2, 64))
    
    with torch.no_grad():
        pytorch_output = model(test_input)['logits'].numpy()

    ort_session = ort.InferenceSession(onnx_path)
    ort_inputs = {'input_ids': test_input.numpy()}
    onnx_output = ort_session.run(None, ort_inputs)[0]

    max_diff = np.abs(pytorch_output - onnx_output).max()
    mean_diff = np.abs(pytorch_output - onnx_output).mean()

    print(f"ONNX Verification:")
    print(f"  PyTorch output shape: {pytorch_output.shape}")
    print(f"  ONNX output shape:   {onnx_output.shape}")
    print(f"  Max absolute diff:  {max_diff:.8f}")
    print(f"  Mean absolute diff: {mean_diff:.8f}")

    if max_diff < 1e-4:
        print(f"  ✓ Export verified successfully!")
    elif max_diff < 1e-2:
        print(f"  △ Acceptable (small numerical difference)")
    else:
        print(f"  ✗ Significant mismatch — check the export!")


verify_onnx_export(your_model, "model.onnx")

# 典型输出:
# ONNX Verification:
#   PyTorch output shape: (2, 64, 1000)
#   ONNX output shape:   (2, 64, 1000)
#   Max absolute diff:  0.00000123
#   Mean absolute diff: 0.00000004
#   ✓ Export verified successfully!
```

### ONNX 的局限性与 LLM 场景的特殊考虑

ONNX 在计算机视觉领域非常成熟，但在 LLM 推理场景下有一些需要注意的限制：

**不支持动态控制流**：标准的 ONNX 导出不支持 Python 层面的 if/else 或 for 循环（虽然 ONNX 有 Loop 和 If 算子，但它们的使用比较复杂）。对于自回归生成这种需要循环调用的场景，通常的做法是：
1. 只导出单步前向传播（one-step forward pass）为 ONNX
2. 在外部用 Python/C++ 循环调用这个 ONNX 模型来完成多步生成

**某些算子支持问题**：一些高级操作（如 RoPE、FlashAttention、某些自定义 CUDA kernel）可能在 ONNX 中没有对应的实现或实现效率不高。解决方案包括：
- 使用 `onnx_custom_ops` 注册自定义算子
- 使用 `onnxscript` 编写兼容的替代实现
- 或者干脆不用 ONNX，改用其他格式（如 GGUF）

## GGUF：llama.cpp / Ollama 的原生格式

如果说 ONNX 是通用的工业级部署格式，那 **GGUF** 就是本地 LLM 推理领域的王者。它是 **llama.cpp** 项目使用的模型格式，也是 **Ollama**、**LM Studio** 等流行本地推理工具的原生格式。GGUF 的核心优势：

- **极致压缩**：支持从 Q2 到 F16 的各种量化级别
- **CPU/GPU/MPS 全平台支持**：不需要 NVIDIA GPU 就能跑大模型
- **零依赖运行**：单个文件包含完整的模型权重 + 元数据
- **社区生态庞大**：大量预量化的模型可以直接下载使用

### 从 HF 模型导出到 GGUF

```python
def export_to_gguf(hf_model_dir, output_file, q_method="Q4_K_M"):
    """导出为 GGUF 格式"""
    import subprocess
    import os
    
    # 方法一：使用 llama.cpp 的 convert 工具
    cmd = [
        "python", "convert_hf_to_gguf.py",
        hf_model_dir,
        "--outfile", output_file,
        "--outtype", q_method,
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            size_mb = os.path.getsize(output_file) / 1024 / 1024
            print(f"✓ GGUF exported to {output_file} ({size_mb:.1f} MB)")
            print(f"  Quantization: {q_method}")
            
            # 显示 GGUF 文件中的元数据
            from gguf import GGUFReader
            reader = GGUFReader(output_file)
            metadata = reader.fields
            
            if 'general.architecture' in metadata:
                arch = str(metadata['general.architecture'])
                print(f"  Architecture: {arch}")
            if 'general.name' in metadata:
                name = str(metadata['general.name'])
                print(f"  Model name: {name}")
                
        else:
            print(f"✗ Export failed:\n{result.stderr}")
            
    except FileNotFoundError:
        print("convert_hf_to_gguf.py not found.")
        print("Install llama.cpp: git clone https://github.com/ggerganov/llama.cpp")


export_to_gguf("./output/qwen-merged", "./output/qwen.Q4_K_M.gguf", "Q4_K_M")
```

### GGUF 量化方法速查

| 方法 | 比特数 | 大小(7B) | 质量 | 适用场景 |
|-----|-------|----------|------|---------|
| Q2_K | ~2.5 bit | ~2.5 GB | 较差 | 极限压缩/测试 |
| Q3_K_M | ~3 bit | ~3.5 GB | 一般 | 存储极度受限 |
| Q4_0_4_4 | 4 bit | ~4.0 GB | 较好 | 标准推荐 |
| **Q4_K_M** | **~4.5 bit** | **~4.5 GB** | **好** | **默认推荐** |
| Q5_K_M | ~5.5 bit | ~5.5 GB | 很好 | 质量优先 |
| Q6_K | 6 bit | ~6.5 GB | 极好 | 高质量需求 |
| Q8_0 | 8 bit | ~7.5 GB | 几乎无损 | 基准对比 |
| F16 | 16 bit | ~14 GB | 完全无损 | 参考基准 |

**推荐选择**：大多数情况下 `Q4_K_M` 是最佳平衡点——体积约为原始 BF16 的 32%，质量损失在大多数 benchmark 上 < 2%。如果对质量有更高要求且磁盘空间允许，`Q5_K_M` 是更好的选择。

### 用 Ollama 加载 GGUF 模型

导出完成后，可以直接用 Ollama 加载运行：

```bash
# 创建 Modelfile
cat > Modelfile <<EOF
FROM ./output/qwen.Q4_K_M.gguf
PARAMETER temperature 0.7
PARAMETER top_p 0.9
TEMPLATE """{{- if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
""""
PARAMETER stop "<|im_end|>"
EOF

# 创建 Ollama 模型
ollama create my-qwen -f Modelfile

# 运行
ollama run my-qwen "你好，请介绍一下你自己"
```

## SafeTensors：HuggingFace 的新标准存储格式

在 HuggingFace 生态内部，**SafeTensors** 正在逐步取代传统的 PyTorch `.bin` / `.pt` 格式成为模型权重的标准存储格式。它的设计目标是解决 `.bin` 格式的安全问题——`.bin` 文件本质上是 `torch.save()` 的输出，加载时会触发 `pickle` 反序列化，而 pickle 可以执行任意代码（安全风险）。SafeTensors 通过只保存张量数据而不执行任何代码来消除这个风险。

```python
from safetensors.torch import save_file, load_file


def safetensors_demo(model, output_path="model.safetensors"):
    state_dict = model.state_dict()
    
    save_file(state_dict, output_path)
    
    loaded_state_dict = load_file(output_path)
    
    match = True
    for key in state_dict:
        if key not in loaded_state_dict:
            match = False
            break
        if not torch.equal(state_dict[key], loaded_state_dict[key]):
            match = False
            break
    
    import os
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    
    print(f"SafeTensors demo:")
    print(f"  Saved to: {output_path}")
    print(f"  Size: {size_mb:.1f} MB")
    print(f"  All keys match: {match}")
    print(f"  Keys count: {len(state_dict)}")


safetensors_demo(your_model)

# 输出:
# SafeTensors demo:
#   Saved to: model.safetensors
#   Size: 128.4 MB
#   All keys match: True
#   Keys count: 47
```

HF Trainer 默认使用 SafeTensors 格式保存 checkpoint。当你看到 Model Hub 上的模型文件列表中有 `.safetensors` 后缀的文件时，那就是 SafeTensors 格式。它和 `.bin` 格式可以互相转换，但建议新项目统一使用 SafeTensors。

## 格式选型决策树

面对众多的导出格式，如何选择？以下是一个实用的决策指南：

```
你的目标平台是什么？
│
├── Web 服务端 (Python 后端)
│   └── 🎯 **保持 PyTorch 格式 (.pt/.safetensors)** 
│       或 **TorchScript**
│       理由：Python 生态内最方便，无需额外转换
│
├── 高性能服务 (C++/Go/Rust 后端)
│   ├── 需要 GPU 加速？
│   │   ├── 是 → 🎯 **ONNX + TensorRT** (NVIDIA GPU)
│   │   │         或 **ONNX Runtime** (通用)
│   │   └── 否 → 🎯 **ONNX Runtime CPU** 或 **OpenVINO**
│   │
│   └── 需要极低延迟？
│       └── 🎯 **Triton / CUTLASS 自定义 kernel**
│           （终极性能，开发成本高）
│
├── 本地桌面 / 笔记本电脑
│   ├── 有 NVIDIA GPU?
│   │   └── 🎯 **GGUF + Ollama** 或 **llama.cpp**
│   │       （最简单的本地部署方式）
│   │
│   └── 只有 CPU / Apple Silicon?
│       └── 🎯 **GGUF (Q4_K_M) + Ollama**
│           （Apple Silicon 对 GGUF 有良好支持）
│
├── 移动设备 (iOS / Android)
│   ├── iOS → 🎯 **CoreML** (通过 coremltools 导出)
│   └── Android → 🎯 **TFLite / ONNX Runtime Mobile**
│
├── 边缘设备 / 嵌入式 (树莓派 / Jetson)
│   ├── NVIDIA Jetson → 🎯 **TensorRT** 或 **ONNX Runtime**
│   └── ARM Linux → 🎯 **GGUF (Q2/Q3)** 或 **ONNX Runtime ARM**
│
└── 分享给社区 / 上传到 Hub
    └── 🎯 **SafeTensors 格式上传到 HuggingFace Hub**
        （社区标准，自动支持）
```

下一节我们将综合前面所有的优化技术，做一个完整的性能分析实践——如何测量推理延迟、识别瓶颈、以及建立系统化的优化 Checklist。
