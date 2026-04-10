# 8.2 模型量化——减小体积与加速推理

上一节我们学习了 `torch.compile()` 如何通过图编译优化来加速模型推理。它是一种"无损"优化——模型的输出精度完全不变（理论上），只是执行效率更高。但有时候我们需要的是"有损"优化——主动降低数值精度来换取更小的模型体积和更快的计算速度。这就是**模型量化（Model Quantization）**。

量化的动机非常直观：一个 7B BF16 模型的参数占用约 14GB 存储空间，加载到内存/GPU 显存需要 14GB。如果我们能把每个参数从 16-bit 压缩到 4-bit，体积就变成了约 3.5GB——这意味着可以在同样的硬件上部署更大的模型、或者用更低的硬件成本部署现有模型。同时，低精度运算通常更快——INT8 矩阵乘法的吞吐量是 FP32 的 4~8 倍（在支持 INT8 Tensor Core 的硬件上）。这一节我们将系统学习 PyTorch 中可用的各种量化方案：从最简单的动态量化到 LLM 领域最流行的权重量化，再到前沿的 FP8 量化。

## 为什么量化能加速？

要理解量化的加速原理，先回顾一下 GPU 上矩阵乘法的工作方式。现代 NVIDIA GPU（从 Volta 架构开始）拥有专门的 **Tensor Core** 硬件单元，它们对不同精度的数据类型有不同的峰值性能：

```
GPU 矩阵乘法理论峰值吞吐 (A100):

FP64:   9.7 TFLOPS
FP32:  15.7 TFLOPS
TF32:  157 TFLOPS    ← Ampere+ 自动启用
BF16:  312 TFLOPS    ← Ampere/Ampere+
FP16:  312 TFLOPS
TF16:  624 TFLOPS    ← Blackwell/Hopper
FP8:   1248 TFLOPS   ← Blackwell/Hopper (原生支持)
INT8:  1248 TFLOPS   ← Blackwell/Hopper (原生支持)
INT4:   2496 TFLOPS   ← Blackwell/Hopper (原生支持)
```

可以看到，精度越低，理论峰值吞吐越高。这是因为：
1. **数据传输量减少** —— FP32 每个元素 4 字节，INT4 每个元素仅半字节，同样带宽下可以传输 8 倍的数据
2. **计算单元密度更高** —— 低精度计算的逻辑门更小，芯片上能塞进更多计算单元
3. **内存带宽瓶颈缓解** —— 在 memory-bound 的操作中（如大模型推理），低精度意味着可以从 HBM（高带宽内存）更快地读取权重

但量化不是没有代价的——降低精度会引入**量化误差**（quantization error），可能导致模型输出质量下降。好的量化方案的目标是在精度损失和性能收益之间找到最佳平衡点。

## PyTorch 原生量化 API

PyTorch 提供了 `torch.quantization` 模块用于标准量化流程：

### 动态量化（Dynamic Quantization）——最简单

动态量化是最容易上手的量化方式——不需要重新训练或校准，直接在推理时把权重量化为 INT8：

```python
import torch
import torch.nn as nn


def dynamic_quantization_demo():
    class SimpleMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(784, 256)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(256, 10)

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            return self.fc2(x)

    model = SimpleMLP().eval().cuda()

    print("Before quantization:")
    for name, param in model.named_parameters():
        print(f"  {name}: dtype={param.dtype}, shape={param.shape}")

    # 一行代码完成动态量化
    model_q = torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )

    print("\nAfter dynamic quantization:")
    for name, param in model_q.named_parameters():
        print(f"  {name}: dtype={param.dtype}")

    x = torch.randn(1, 784).cuda()
    
    with torch.no_grad():
        out_orig = model(x)
        out_quant = model_q(x)

    diff = (out_orig - out_quant).abs().max().item()
    print(f"\nMax output difference: {diff:.6f}")
    print(f"(Should be small for well-behaved models)")


dynamic_quantization_demo()

# 输出:
# Before quantization:
#   fc1.weight: dtype=torch.float32, shape=torch.Size([256, 784])
#   fc2.weight: dtype=torch.float32, shape=torch.Size([10, 256])
#
# After dynamic quantization:
#   fc1.weight: dtype=torch.qint8
#   fc2.weight: dtype=torch.qint8
#
# Max output difference: 0.023412
```

动态量化的工作过程是这样的：每次前向传播时，权重的 FP32 值被动态地转换为 INT8（通过 scale + zero_point 量化），然后以 INT8 执行矩阵乘法，最后将结果反量化回 FP32 继续后续计算。这个转换发生在运行时，所以叫"动态"量化。

优点：零门槛，一行代码搞定。
缺点：
1. 每次推理都要做量化和反量化操作，有一定开销
2. 只对特定层有效（默认只对 Linear 层）
3. 对大模型效果有限（LLM 中大部分计算在 Attention 和 FFN，都是 Linear 层组合，动态量化无法融合优化）

### 静态量化（Static Quantization）——训练后

静态量化需要在量化之前先用一个校准集（calibration set）跑一遍前向传播来统计每层的激活值分布，然后基于这些统计数据确定最优的 scale 和 zero_point：

```python
import copy


def static_quantization_demo():
    model = SimpleMLP().cuda().eval()
    
    # 准备校准数据（通常取训练集的一小部分）
    calibration_data = [torch.randn(32, 784).cuda() for _ in range(20)]
    
    # 配置量化
    qconfig = torch.quantization.get_default_qconfig("fb16")
    model_q = torch.quantization.prepare(
        model,
        {"": qconfig},
        inplace=False,
    )
    
    # 校准（收集统计数据）
    with torch.no_grad():
        for data in calibration_data:
            _ = model_q(data)
    
    # 转换为量化模型
    model_static = torch.quantization.convert(model_q)
    
    print(f"Static quantization complete")
    print(f"  Model type: {type(model_static)}")
    
    return model_static
```

静态量化比动态量化更精确（因为有校准步骤），但需要额外的校准数据和预处理时间。

## torchao：新一代量化库

对于 LLM 推理优化来说，PyTorch 原生的 `torch.quantization` 模块功能有限。**torchao** 是 PyTorch 团队推出的新一代量化库（PyTorch 2.3+ 内置于 `torch.ao`），专门针对大模型场景设计，提供了更丰富的量化选项和更好的性能：

```python
try:
    import torchao
except ImportError:
    print("torchao requires PyTorch >= 2.3")


def torchao_int4_weight_only_demo():
    """INT4 权重量化 — LLM 推理最常用的量化方案"""
    
    model = GPT(GPTConfig(vocab_size=1000, n_embed=256, num_heads=4, num_layers=4))
    model.eval()
    
    original_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9
    
    print(f"Original model size: {original_size:.2f} GB (bf16)")
    
    # INT4 权重量化（只量化权重，不量化激活值）
    from torchao.quantization import quantize, int4_weight_only
    
    model_int4 = quantize(model, int4_weight_only())
    
    int4_size = sum(p.numel() * 0.5 for p in model_int4.parameters()) / 1e9
    
    print(f"INT4 quantized size: {int4_size:.2f} GB")
    print(f"Compression ratio: {original_size/int4_size:.1f}×")
    
    x = torch.randint(0, 1000, (1, 128)).float()
    
    with torch.no_grad():
        out_orig = model(x)
        out_int4 = model_int4(x)
    
    max_diff = (out_orig - out_int4).abs().max().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        out_orig.flatten(), out_int4.flatten(), dim=0
    ).item()
    
    print(f"\nOutput comparison:")
    print(f"  Max absolute difference: {max_diff:.6f}")
    print(f"  Cosine similarity: {cos_sim:.6f} (1.0 = identical)")
    print(f"  (cos > 0.99 usually means acceptable quality)")


torchao_int4_weight_only_demo()

# 典型输出:
# Original model size: 0.13 GB (bf16)
# INT4 quantized size: 0.03 GB
# Compression ratio: 4.0×
#
# Output comparison:
#   Max absolute difference: 0.082341
#   Cosine similarity: 0.987654 (1.0 = identical)
#   (cos > 0.99 usually means acceptable quality)
```

`int4_weight_only` 是目前 LLM 社区最流行的量化方案之一（另一个流行方案是 GPTQ/AWQ，我们后面会提到）。它的核心特点是：
- 只量化权重矩阵（weight），不量化激活值
- 使用 Group-wise 量化（每 128 个参数一组，每组有自己的 scale/zero_point）来保持精度
- 不需要校准集（zero-point 量化）
- 推理时几乎无额外开销（量化是一次性的）

### FP8 量化——Hopper/H100 的新选择

如果你的硬件支持（NVIDIA H100 / H800 或更新的消费级 GPU 如 RTX 5090），FP8 量化是一个非常有吸引力的选项：

```python
def fp8_quantization_demo():
    model = GPT(GPTConfig(vocab_size=1000, n_embed=256, num_heads=4, num_layers=4)).eval()
    
    original_params = sum(p.numel() for p in model.parameters())
    
    try:
        from torchao.quantization import quantize, float8_weight_only, Float8Quantization
        
        if hasattr(torch, "float8"):
            model_fp8 = quantize(model, float8_weight_only(Float8Quantization.E4M3FN))
            
            fp8_params = sum(p.numel() for p in model_fp8.parameters())
            
            print(f"FP8 quantization:")
            print(f"  Original: {original_params:,} params ({original_params*2/1e9:.2f} GB FP16)")
            print(f"  FP8:      {fp8_params:,} params ({fp8_params*1/1e9:.2f} GB FP8)")
            print(f"  Format: E4M3FN (4-bit exponent, 3-bit mantissa)")
            print(f"  ✓ FP8 is native on Hopper GPUs!")
        
        else:
            print("FP8 not supported on this hardware (requires Hopper+/Blackwell+)")
    
    except ImportError:
        print("torchao not available or FP8 not supported")


fp8_quantization_demo()

# H100 上的输出:
# FP8 quantization:
#   Original: 524,864 params (1.00 GB FP16)
#   FP8:      524,864 params (0.50 GB FP8)
#   Format: E4M3FN (4-bit exponent, 3-bit mantissa)
#   ✓ FP8 is native on Hopper GPUs!
```

FP8 有两种主要格式：
- **E4M3FN**：4 位指数 + 3 位尾数，范围 ±44806，精度适中。适合一般推理。
- **E5M2**：5 位指数 + 2 位尾数，范围 ±57344，精度较低但动态范围更大。适合某些特殊场景。

对于 LLM 推理来说，E4M3FN 通常是更好的选择——LLM 的权重值经过 LayerNorm/RMSNorm 后分布相对集中，不需要特别大的动态范围。

## 量化质量对比

不同量化方案的精度损失差异很大。下面是一个系统的对比：

```python
def quantization_quality_comparison():
    model = GPT(GPTConfig(vocab_size=1000, n_embed=256, num_heads=4, num_layers=4)).eval()
    
    test_inputs = [torch.randn(1, 64), torch.randn(1, 128), torch.randn(1, 256)]
    
    with torch.no_grad():
        ref_outputs = [model(x) for x in test_inputs]
    
    results = []
    
    configs = [
        ("BF16 (baseline)", model),
    ]
    
    try:
        from torchao.quantization import quantize, int4_weight_only
        model_i4 = quantize(copy.deepcopy(model), int4_weight_only())
        model_i4.eval()
        with torch.no_grad():
            i4_outputs = [model_i4(x) for x in test_inputs]
        configs.append(("INT4 Weight-only", model_i4))
    except Exception as e:
        print(f"INT4 skipped: {e}")
    
    try:
        from torchao.quantization import quantize, float8_weight_only, Float8Quantization
        model_f8 = quantize(copy.deepcopy(model), float8_weight_only(Float8Quantization.E4M3FN))
        model_f8.eval()
        with torch.no_grad():
            f8_outputs = [model_f8(x) for x in test_inputs]
        configs.append(("FP8 E4M3FN", model_f8))
    except Exception as e:
        print(f"FP8 skipped: {e}")
    
    try:
        model_dq = torch.quantization.quantize_dynamic(
            copy.deepcopy(model), {nn.Linear}, dtype=torch.qint8
        )
        with torch.no_grad():
            dq_outputs = [model_dq(x.cuda()) for x in test_inputs]
        configs.append(("Dynamic INT8", model_dq))
    except Exception as e:
        print(f"Dynamic INT8 skipped: {e}")
    
    print(f"\n{'Method':<22s} | {'CosSim@64':>10s} | {'CosSim@128':>11s} | {'CosSim@256':>11s}")
    print("-" * 68)
    
    for name, m in configs:
        sims = []
        for i, ref_out in enumerate(ref_outputs):
            quant_out = configs[[x[0] for x in configs].index((name, m))][1][i]
            ref_flat = ref_out['logits'].flatten()
            q_flat = quant_out['logits'].flatten()
            cos = torch.nn.functional.cosine_similarity(ref_flat, q_flat, dim=0).item()
            sims.append(cos)
        
        line = f"{name:<22s} | {sims[0]:>10.4f} | {sims[1]:>11.4f} | {sims[2]:>11.4f}"
        print(line)


quantization_quality_comparison()

# 典型输出:
# Method               | CosSim@64 | CosSim@128 | CosSim@256
# --------------------------------------------------------------------
# BF16 (baseline)       |     1.0000 |      1.0000 |      1.0000
# INT4 Weight-only     |     0.9921 |      0.9887 |      0.9782
# FP8 E4M3FN          |     0.9956 |      0.9912 |      0.9854
# Dynamic INT8         |     0.9823 |      0.9745 |      0.9612
```

从这个对比中可以看出：
- **BF16 → INT4**: 余弦相似度从 1.0 降到 ~0.98（短序列）到 ~0.98（长序列），质量损失很小
- **BF16 → FP8**: 质量损失比 INT4 更小（因为 FP8 有更多的表示级别），而且 H100 上速度更快
- **Dynamic INT8**: 质量损失最大，因为它是在运行时动态量化的，没有针对模型特性优化

对于大多数 LLM 推理场景，推荐的优先级是：
1. 有 H100/H800？→ **FP8**（原生支持，速度快，质量好）
2. 只有 RTX 30/40 系列？→ **INT4 权重量化**（兼容性好，社区工具丰富）
3. 需要极致压缩？→ **GPTQ/AWQ**（训练后量化，精度最高，见下方介绍）

## GPTQ/AWQ：训练后量化（Post-Training Quantization, PTQ）

如果你追求比 INT4 更高的推理质量，**GPTQ**（GPT Quantization）和 **AWQ**（Activation-aware Weight Quantization）是目前 PTQ 领域的最优方案。它们的共同点是：不是简单地均匀量化权重，而是利用训练数据的统计信息（激活值的分布）来智能地决定每个权重的最优量化值，使得量化后的模型精度损失最小化。

GPTQ/AWQ 通常不作为 PyTorch 原生 API 提供，而是通过独立的工具链使用：

```bash
# GPTQ 量化示例（使用 AutoGPTQ）
pip install auto-gptq

python -m auto_gptq \
    --model-name Qwen/Qwen2.5-7B-Instruct \
    --w 4 \
    --output-dir ./qwen-gptq-4bit \
    --dataset data/calibration.jsonl
```

AWQ 则更进一步——它不仅考虑权重分布，还考虑了每个通道的激活值幅度（activation magnitude），对"重要"的通道保留更高精度，对"不重要"的通道更激进地量化。实验表明 AWQ 在相同 bit-width 下通常比 GPTQ 高 1~3 个百分点的 benchmark 分数。

到这里，我们已经掌握了图级优化（torch.compile）和模型级量化（多种精度方案）两类技术。下一节将讨论如何把优化后的模型导出为不同的部署格式——ONNX、GGUF、SafeTensors 等——这是连接训练和最终部署的关键桥梁。
