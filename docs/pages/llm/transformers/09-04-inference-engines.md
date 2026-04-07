# 推理加速引擎：ONNX Runtime / TensorRT / vLLM / TGI

## 这一节讲什么？

前面几节我们学习了量化（减少精度）、蒸馏（减少参数）、剪枝（移除冗余）——这些都是**模型层面的优化**。但即使模型已经优化到极致，如果推理框架不够高效，仍然无法充分发挥硬件性能。

这一节我们将学习工业界最主流的推理加速引擎：
1. **TorchScript 与 Torch Compile**——PyTorch 原生的编译优化
2. **ONNX Runtime**——跨平台推理的行业标准
3. **TensorRT**——NVIDIA 的终极 GPU 加速方案
4. **vLLM**——LLM 专用的高吞吐推理引擎
5. **TGI (Text Generation Inference)**——HuggingFace 的生产级服务框架

---

## 一、为什么需要专门的推理引擎？

## 1.1 PyTorch Eager Mode 的瓶颈

```python
def eager_mode_bottlenecks():
    """解释 PyTorch Eager Mode 的性能瓶颈"""

    bottlenecks = [
        {
            "瓶颈": "Python 解释器开销",
            "解释": "每一步操作都要经过 Python 解释器,"
                   "对于简单的逐元素运算, Python 开销可能超过计算本身",
            "占比": "小模型上可达 30~50% 的总时间",
        },
        {
            "瓶颈": "算子粒度太细",
            "解释": "Eager 模式下每个 tensor 操作都是独立的 kernel launch,"
                   "GPU 每执行一个操作就要调度一次",
            "占比": "kernel launch latency 累积",
        },
        {
            "瓶颈": "缺少算子融合",
            "解释": "Conv + BN + ReLU 本可以融合为一个 kernel,"
                   "但 Eager 模式下它们是三个独立的调用",
            "占比": "内存读写增加, 缓存利用率低",
        },
        {
            "瓶颈": "内存分配不可预测",
            "解释": "动态 shape 导致无法预分配内存,"
                   "每次前向传播都可能触发新的内存分配",
            "占比": "碎片化和额外的分配开销",
        },
    ]

    print("=" * 70)
    print("PyTorch Eager Mode 性能瓶颈")
    print("=" * 70)
    for b in bottlenecks:
        print(f"\n⚠️  {b['瓶颈']}")
        print(f"   原因: {b['解释']}")
        print(f"   影响: {b['占比']}")

eager_mode_bottlenecks()
```

## 1.2 推理优化技术全景

```python
def optimization_techniques():
    """推理优化技术分类"""

    techniques = [
        ("图级别", "将计算图静态化, 一次性优化整个图", 
         "TorchScript/TorchDynamo/ONNX"),
        ("算子融合", "将多个小算子合并为一个大算子 (如 Conv+BN+ReLU→CudnnConvolution)",
         "TensorRT/XLA"),
        ("内存优化", "预分配内存池, 重用中间结果, 减少分配次数",
         "ONNX RT/TensorRT"),
        ("内核优化", "针对特定硬件的手写高性能 kernel (如 FlashAttention)",
         "FlashAttention/Cutlass"),
        ("并行优化", "张量并行/流水线并行/专家并行",
         "vLLM/DeepSpeed/FastServe"),
        ("批处理优化", "动态批处理/连续批处理/PagedAttention",
         "vLLM/TGI"),
    ]

    print("=" * 80)
    print("推理优化技术全景")
    print("=" * 80)
    for level, desc, tools in techniques:
        print(f"\n🔧 [{level}] {desc}")
        print(f"   工具: {tools}")

optimization_techniques()
```

---

## 二、TorchScript 与 Torch Compile：PyTorch 原生加速

## 2.1 TorchScript：静态图追踪

```python
import torch
import torch.nn as nn
import time


class SimpleModel(nn.Module):
    """用于演示的简单模型"""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def torchscript_demo():
    """TorchScript 编译与对比"""

    model = SimpleModel().eval()
    dummy_input = torch.randn(64, 784)

    # ========== Eager Mode ==========
    print("=" * 60)
    print("TorchScript vs Eager Mode 对比")
    print("=" * 60)

    # 预热
    for _ in range(10):
        _ = model(dummy_input)

    # Eager Mode 计时
    start = time.perf_counter()
    for _ in range(1000):
        with torch.no_grad():
            _ = model(dummy_input)
    eager_time = time.perf_counter() - start

    # ========== TorchScript 编译 ==========
    scripted_model = torch.jit.script(model)

    # 预热
    for _ in range(10):
        _ = scripted_model(dummy_input)

    # Scripted 计时
    start = time.perf_counter()
    for _ in range(1000):
        with torch.no_grad():
            _ = scripted_model(dummy_input)
    script_time = time.perf_counter() - start

    print(f"\n{'模式':<15} {'耗时 (ms)':<12} {'加速比'}")
    print("-" * 45)
    print(f"{'Eager':<15} {eager_time*1000:.2f}{'':>8} {'1.00x':<12}")
    print(f"{'TorchScript':<15} {script_time*1000:.2f}{'':>8} {eager_time/script_time:.2f}x")

    print(f"\n💡 TorchScript 的优势:")
    print(f"   • 将 Python 代码转换为中间表示 (IR)")
    print(f"   • 可以序列化为文件, 无需原始 Python 代码即可加载")
    print(f"   • 在 C++/移动端部署时非常有用")

torchscript_demo()
```

## 2.2 Torch Compile (PyTorch 2.0+)

```python
def torch_compile_demo():
    """torch.compile 演示 (PyTorch 2.0+)"""

    import torch
    import torch.nn as nn
    import time

    class TransformerBlock(nn.Module):
        def __init__(self, d_model=256, nhead=8, dim_ff=512):
            super().__init__()
            self.self_attn = nn.MultiheadAttention(d_model, nhead)
            self.ff = nn.Sequential(
                nn.Linear(d_model, dim_ff),
                nn.ReLU(),
                nn.Linear(dim_ff, d_model),
            )
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)

        def forward(self, x):
            attn_out, _ = self.self_attn(x, x, x)
            x = self.norm1(x + attn_out)
            ff_out = self.ff(x)
            x = self.norm2(x + ff_out)
            return x

    model = TransformerBlock().eval()
    dummy = torch.randn(16, 32, 256)  # (batch, seq_len, d_model)

    # Eager baseline
    for _ in range(5):
        _ = model(dummy)

    start = time.perf_counter()
    for _ in range(500):
        with torch.no_grad():
            _ = model(dummy)
    eager_ms = (time.perf_counter() - start) * 1000

    # Compiled version
    try:
        compiled_model = torch.compile(model)

        for _ in range(5):  # warmup (首次编译有 overhead)
            _ = compiled_model(dummy)

        start = time.perf_counter()
        for _ in range(500):
            with torch.no_grad():
                _ = compiled_model(dummy)
        compile_ms = (time.perf_counter() - start) * 1000

        print("=" * 55)
        print("torch.compile 性能对比")
        print("=" * 55)
        print(f"\n{'Eager Mode':<18} {eager_ms:>8.2f} ms")
        print(f"{'torch.compile':<18} {compile_ms:>8.2f} ms")
        print(f"{'加速比':<18} {eager_ms/compile_ms:>8.2f}x")

        print(f"\n⚠️  注意事项:")
        print(f"   • 首次调用会有编译开销 (JIT compilation)")
        print(f"   • 动态 shape 可能导致 recompilation (graph break)")
        print(f"   • 不是所有模型都能被正确捕获 (某些 control flow)")

    except Exception as e:
        print(f"torch.compile 不可用: {e}")
        print("   (需要 PyTorch >= 2.0 且安装正确的依赖)")

torch_compile_demo()
```

---

## 三、ONNX Runtime：跨平台推理标准

## 3.1 ONNX 是什么？

ONNX (Open Neural Network Exchange) 是一个开放的模型交换格式，就像图片领域的 JPEG 或 PDF。它让模型可以在不同框架之间自由转换：

```
PyTorch (.pt) ──导出──→ ONNX (.onnx) ──推理──→ ONNX Runtime
                                          ├──推理──→ TensorRT
                                          ├──推理──→ OpenVINO
                                          └──推理──→ CoreML (macOS/iOS)
```

## 3.2 完整的 ONNX 导出与推理流程

```python
def onnx_export_and_infer_demo():
    """ONNX 导出与推理完整流程"""

    import torch
    import torch.nn as nn
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import os

    model_name = "distilbert-base-uncased-finetuned-sst-2-english"

    print("=" * 65)
    print("ONNX 导出与推理实战")
    print("=" * 65)

    # Step 1: 加载 PyTorch 模型
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).eval()

    # Step 2: 准备示例输入 (用于 tracing)
    text = "This movie was absolutely wonderful! I loved every minute of it."
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)

    print(f"\n[Step 1] 加载模型完成")
    print(f"  Model: {model_name}")

    # Step 3: 导出到 ONNX
    onnx_path = "./model.onnx"
    
    dynamic_axes = {
        'input_ids': {0: 'batch', 1: 'sequence'},
        'attention_mask': {0: 'batch', 1: 'sequence'},
    }

    torch.onnx.export(
        model,
        (inputs['input_ids'], inputs['attention_mask']),
        onnx_path,
        input_names=['input_ids', 'attention_mask'],
        output_names=['logits'],
        dynamic_axes=dynamic_axes,
        opset_version=14,
        do_constant_folding=True,  # 常量折叠优化
    )

    onnx_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"\n[Step 2] ONNX 导出完成")
    print(f"  文件: {onnx_path}")
    print(f"  大小: {onnx_size_mb:.1f} MB")

    # Step 4: 用 ONNX Runtime 推理
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(onnx_path)
        
        ort_inputs = {
            'input_ids': inputs['input_ids'].numpy(),
            'attention_mask': inputs['attention_mask'].numpy(),
        }
        
        outputs = session.run(None, ort_inputs)
        
        import numpy as np
        predicted_class_id = np.argmax(outputs[0][0])
        labels = ["negative", "positive"]

        print(f"\n[Step 3] ONNX Runtime 推理结果:")
        print(f"  输入文本: '{text}'")
        print(f"  预测标签: {labels[predicted_class_id]}")

        # Step 5: 性能对比
        import time
        
        # PyTorch 推理计时
        for _ in range(20):
            _ = model(**inputs)
        start = time.perf_counter()
        for _ in range(200):
            with torch.no_grad():
                _ = model(**inputs)
        pt_time = (time.perf_counter() - start) * 1000 / 200

        # ONNX 推理计时
        for _ in range(20):
            _ = session.run(None, ort_inputs)
        start = time.perf_counter()
        for _ in range(200):
            _ = session.run(None, ort_inputs)
        onnx_time = (time.perf_counter() - start) * 1000 / 200

        print(f"\n[Step 4] 性能对比 (单样本, CPU):")
        print(f"  PyTorch Eager:  {pt_time:.3f} ms")
        print(f"  ONNX Runtime: {onnx_time:.3f} ms")
        print(f"  加速比:        {pt_time/onnx_time:.2f}x")

    except ImportError:
        print("\n⚠️  onnxruntime 未安装。请运行: pip install onnxruntime")

onnx_export_and_infer_demo()
```

## 3.3 ONNX Graph Optimization

ONNX Runtime 内置了多轮图优化，了解它们有助于理解为什么 ONNX 比 PyTorch 快：

```python
def onnx_optimization_passes():
    """ONNX 图优化过程详解"""

    passes = [
        {"Pass 1": "常量折叠 (Constant Folding)",
         "作用": "5 * 3 → 15 (在编译期直接计算)"},
        {"Pass 2": "死代码消除 (Dead Code Elimination)",
         "作用": "移除不影响输出的分支"},
        {"Pass 3": "算子融合 (Operator Fusion)",
         "作用": "Conv + BN + ReLU → 单个 fused kernel"},
        {"Pass 4": "公共子表达式消除 (CSE)",
         "作用": "x * y + z * y → (x+z) * y"},
        {"Pass 5": "内存布局优化 (Memory Layout Opt)",
         "作用": "调整 tensor 存储顺序以适配硬件缓存行"},
    ]

    print("=" * 65)
    print("ONNX 图优化 Passes")
    print("=" * 65)
    for p in passes:
        name = list(p.keys())[0]
        desc = p[name]
        print(f"\n  {name}: {desc}")

onnx_optimization_passes()
```

---

## 四、TensorRT：NVIDIA 的终极武器

## 4.1 为什么 TensorRT 最快？

TensorRT 是 NVIDIA 专门为 GPU 推理优化的 SDK。它的核心优势在于：

```python
def tensorrt_advantages():
    """TensorRT 的核心优势"""

    advantages = """
┌──────────────────────────────────────────────────────────────┐
│                    TensorRT 核心优势                       │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Kernel Auto-Tuning                                       │
│     对于每种算子组合, 自动测试所有可能的实现方式             │
│     选择当前 GPU 型号和 batch size 下最快的那个               │
│                                                              │
│  2. Layer Fusion (深度算子融合)                              │
│     不只是 Conv+BN+ReLU                                     │
│     而是可以融合整个 Attention 层或整个 Transformer Block      │
│     → 大幅减少 memory I/O                                   │
│                                                              │
│  3. Precision Calibration (INT8 校准)                      │
│     自动收集校准数据集的激活值统计                           │
│     为每一层计算最优的量化缩放因子                             │
│     → INT8 推理精度接近 FP16                                  │
│                                                              │
│  4. Dynamic Shape 支持                                      │
│     支持可变输入长度 (对 NLP 任务至关重要!)                   │
│     通过 profile 信息自动选择最优 execution plan              │
│                                                              │
│  5. Multi-Stream Execution                                 │
│     并行处理多个请求流 (CUDA streams)                        │
│     隐藏 host-device 同步延迟                                │
│                                                              │
│  典型加速效果:                                              │
│  vs PyTorch FP32:  ~3-8x                                    │
│  vs PyTorch FP16:  ~1.5-3x                                  │
│  vs ONNX Runtime:  ~1.2-2x                                  │
└──────────────────────────────────────────────────────────────┘
"""
    print(advantages)

tensorrt_advantages()
```

## 4.2 使用 optimum 库导出 TensorRT

```python
def tensorrt_export_demo():
    """使用 optimum 导出 TensorRT 引擎"""

    try:
        from optimum.intel import INCQuantizer  # Intel Neural Compressor
        from transformers import pipeline
        import time

        print("=" * 60)
        print("TensorRT / inference 优化导出演示")
        print("=" * 60)

        # 方法 1: 通过 optimum 导出
        print("\n方法 1: optimum.export (通用接口)")
        print("""
from optimum import exporters
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 导出为 ONNX
exporter = exporters.OnnxConfig(task="text-generation-with-past")
exporter.model = model
exporter.save_pretrained("./onnx_model")

# 从 ONNX 转换为 TensorRT
trt_exporter = exporters.TensorRTConfig(
    task="text-generation-with-past",
    dtype="float16",           # FP16 推理
    max_batch_size=4,
    max_sequence_length=128,
)
trt_exporter.model = "./onnx_model"
trt_exporter.save_pretrained("./trt_model")
""")

        # 方法 2: 直接用 HF pipeline + device_map
        print("\n方法 2: pipeline + device_map='auto' (自动选择最优后端)")
        pipe = pipeline(
            "text-generation",
            model="gpt2",
            device="auto",          # 自动选择最优设备/后端
        )

        result = pipe("Hello, world!", max_new_tokens=20)
        print(f"生成结果: {result[0]['generated_text']}")

    except ImportError:
        print("optimum 未完全安装。部分功能需要额外依赖。")

tensorrt_export_demo()
```

---

## 五、vLLM：LLM 时代的推理王者

## 5.1 为什么 vLLM 这么重要？

vLLM 是目前 LLM 推理领域最热门的开源项目之一，它解决了 LLM 推理中的几个核心痛点：

```python
def explain_vllm():
    """vLLM 核心创新点解释"""

    explanation = """
┌───────────────────────────────────────────────────────────────────┐
│                         vLLM 架构概览                          │
├───────────────────────────────────────────────────────────────────┤
│                                                                   │
│  核心问题 1: KV Cache 内存碎片                                     │
│  传统做法:                                                     │
│    Request A: [████████████████████]  200 tokens                  │
│    Request B: [██████░░░░░░░░░░░░░░]  完成后释放 150 tokens       │
│    Request C: [████████████████████]  新来 200 tokens              │
│    问题: B 释放的空间不连续! 无法给 C 分配连续的 200 token 空间    │
│                                                                   │
│  vLLM 方案: PagedAttention (分页注意力)                        │
│    ┌─────────────────────────────────────────────────────────┐   │
│    │ 物理内存分成固定大小的 Page (如每页 16 tokens)           │   │
│    │ Request A: Page1, Page2, ..., Page13                       │   │
│    │ Request B: Page1, Page2 (完成后释放)                     │   │
│    │ Request C: 可以复用 B 释放的 Page! ✅                     │   │
│    └─────────────────────────────────────────────────────────┘   │
│                                                                   │
│  核心问题 2: Static Batching 效率低                               │
│  传统做法:                                                     │
│    |A █████████████████|  A 还没完, B 必须等...                │
│    |B ██████░░░░░░░░░░░|  B 已完成, 但 GPU 被 A 占着              │
│                                                                   │
│  vLLM 方案: Continuous Batching (连续批处理)                    │
│    |A █████████████████████|                                   │
│    |B ████████|              |                                   │
│    |C          |████████████████|  B 完成后立即插入 C!         │
│    |D                  |████████|                            │
│    → GPU 利用率从 ~30% 提升到 >90%!                            │
│                                                                   │
│  核心技术 3: FlashAttention-2                                   │
│    • IO-aware 精确注意力计算 (减少 HBM 读取)                   │
│    • PagedAttention 与 FlashAttention 的无缝集成               │
│    • 支持 FP8/BF16/INT8 多种精度                               │
│                                                                   │
│  典型性能 (Llama-2-7B, A100):                                  │
│    吞吐量: ~100x vs HuggingFace Transformers (batch=1)          │
│    首延迟: ~20ms (vs ~50ms)                                    │
│    显存效率: ~90%+ (vs ~50%)                                    │
└───────────────────────────────────────────────────────────────────┘
"""
    print(explanation)

explain_vllm()
```

## 5.2 vLLM 快速上手

```bash
# 安装
pip install vllm

# 启动 OpenAI 兼容的服务器
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-chat-hf \
    --port 8000 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.9

# 或者 Python 中直接使用
"""
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-2-7b-chat-hf")

params = SamplingParams(
    temperature=0.7,
    top_p=0.92,
    max_tokens=256,
)

outputs = llm.generate(["Hello!", "How are you?"], params)
for output in outputs:
    print(output.outputs[0].text)
"""
```

---

## 六、TGI：HuggingFace 的生产级推理服务

## 6.1 TGI 特性一览

```python
def tgi_features():
    """TGI (Text Generation Inference) 功能特性"""

    features = [
        ("OpenAI 兼容 API", "/v1/chat/completions, /v1/completions 等"),
        ("FlashAttention", "内置支持, 显著降低首延迟"),
        ("PagedAttention", "高并发下的显存高效利用"),
        ("Continuous Batching", "动态批处理, 高吞吐"),
        ("Safetensors 格式", "安全快速地加载大模型"),
        ("量化支持", "FP8/BF16/INT8/EXL2/GPTQ/AWQ"),
        ("Token Streaming", "SSE 流式输出"),
        ("Prometheus 监控", "内置 metrics endpoint"),
        ("Docker 部署", "一行命令启动服务"),
        ("负载均衡", "支持多 worker + router"),
    ]

    print("=" * 65)
    print("TGI 功能特性一览")
    print("=" * 65)
    for name, desc in features:
        print(f"  ✅ {name}: {desc}")

tgi_features()
```

## 6.2 TGI Docker 部署模板

```bash
# 标准 TGI 部署命令
docker run --gpus all -p 8080:80 \
  -v $PWD/data:/data \
  ghcr.io/huggingface/text-generation-inference:latest \
  --model-id meta-llama/Meta-Llama-3-8B-Instruct \
  --num-shard 4 \                    # 张量并行度
  --max-total-tokens 4096 \        # 最大上下文长度
  --dtype float16 \                 # 推理精度
  --enable-flash-attn \             # 启用 FlashAttention
  --enable-paged-attention \        # 启用 PagedAttention
  --max-batch-size 64 \             # 最大批大小
  --max-waiting-tokens 20 \          # 连续批处理等待窗口
```

---

## 七、全面 Benchmark 对比

```python
def comprehensive_benchmark():
    """各推理引擎全面对比"""

    engines = [
        ("HF Transformers\n(Eager)", "~50ms", "~1x", "~24GB", "最简单, 最慢",
         "原型验证/开发调试"),
        ("HF Transformers\n(TorchCompile)", "~35ms", "~1.5x", "~24GB", "简单, 有改善",
         "日常使用/中小模型"),
        ("ONNX Runtime", "~25ms", "~2x", "~22GB", "跨平台好",
         "生产部署/CPU 环境"),
        ("TensorRT", "~15ms", "~3-5x", "~18GB", "最快(NVIDIA)",
         "NVIDIA GPU 生产环境"),
        ("vLLM", "~20ms*", "~5-10x", "~14GB", "最高吞吐",
         "LLM API 服务/高并发"),
        ("TGI", "~20ms*", "~5-10x", "~14GB", "全功能服务",
         "企业级 LLM 服务"),
        ("llama.cpp", "~25ms*", "~3-5x", "~6GB", "本地/边缘部署",
         "消费级硬件"),
    ]

    print("=" * 85)
    print("推理引擎全面对比 (以 LLaMA-2-7B, A100 为参考)")
    print("=" * 85)
    print(f"\n{'引擎':<26} {'首延迟':<10} {'吞吐':<10}{'显存':<10}{'特点':<18}{'适用场景'}")
    print("-" * 95)
    for e in engines:
        print(f"{e[0]:<26}{e[1]:<10}{e[2]:<10}{e[3]:<10}{e[4]:<18}{e[5]}")

    print(f"\n注: * 表示 batch 推理的首 token 延迟; 后续 token 通常 < 5ms")
    print(f"    吞吐数据基于 batch_size=32, seq_len=128 的估算")

comprehensive_benchmark()
```

---

## 八、本章小结

这一节我们系统学习了推理加速引擎的完整生态：

| 引擎 | 核心优势 | 适用场景 | 学习曲线 |
|------|---------|---------|---------|
| **TorchScript** | 序列化/无 Python 依赖 | C++/边缘部署 | 低 |
| **Torch Compile** | 自动图优化 | PyTorch 2.0+ 用户 | 中 |
| **ONNX Runtime** | 跨平台/图优化 | CPU/多框架 | 中 |
| **TensorRT** | 终极 GPU 性能 | NVIDIA GPU 生产 | 高 |
| **vLLM** | LLM 专用高吞吐 | LLM API 服务 | 中 |
| **TGI** | 全功能企业级服务 | 企业级 LLM 平台 | 中 |

**核心要点**：
1. **PyTorch Eager Mode 是开发利器，但不是生产之选**——至少用 `torch.compile` 或导出 ONNX
2. **ONNX 是模型交换的标准格式**——一次导出，到处运行
3. **TensorRT 在 NVIDIA GPU 上无可匹敌**——如果你只用 NVIDIA GPU，它是首选
4. **vLLM / TGI 是 LLM 推理的事实标准**——新项目几乎都从这两个开始
5. **没有银弹**——根据你的硬件、框架、延迟要求选择合适的方案

下一节我们将学习 **系统级优化**——批处理策略、缓存管理、FlashAttention 等更深层的优化技术。
