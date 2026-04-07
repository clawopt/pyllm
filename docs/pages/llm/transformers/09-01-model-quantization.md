# 模型量化基础：从 FP32 到 INT4 的完整指南

## 这一节讲什么？

当你第一次尝试在本地运行一个 LLaMA-7B 模型时，可能会遇到这样的困境：模型文件 14GB（FP16），加载到显存需要约 28GB（训练时更多），而你只有一张 RTX 3090（24GB 显存）。这时候，**量化（Quantization）** 就是你的救星。

量化的核心思想非常直观：**用更少的比特数来表示模型的权重和激活值**。就像把一张高清照片压缩成 JPEG——质量略有下降，但体积大幅缩小。

这一节我们将系统学习：
1. **为什么量化能加速和省内存？**——从硬件层面理解
2. **PTQ vs QAT**——训练后量化 vs 量化感知训练
3. **各种量化格式**——FP16/BF16/INT8/INT4/NF4 的区别
4. **GPTQ/AWQ/GGUF**——LLM 专用的高效量化方法
5. **HF 实战**——使用 bitsandbytes 进行 4-bit/8-bit 推理

---

## 一、量化的直觉理解

## 1.1 从一个简单的例子开始

想象你要存储一组浮点数：`[3.14159, 2.71828, 1.41421, 0.57721]`

**FP32 (32-bit 浮点)**：
- 每个数占 4 字节 = 32 bit
- 总共：4 × 4 = 16 字节
- 精度：可以精确表示小数点后很多位

**INT8 (8-bit 整数)**：
- 先做线性映射（量化）：将范围 [0, 4] 映射到 [0, 255]
- `3.14 → ~201`, `2.72 → ~174`, `1.41 → ~90`, `0.58 → ~37`
- 每个数只占 1 字节 = 8 bit
- 总共：4 × 1 = 4 字节
- 精度损失：还原时 `201 → 3.15`（有微小误差）

**压缩比：4x！精度损失：通常 < 1%**

```python
def quantization_intuition():
    """量化的直觉演示"""

    import numpy as np

    original_weights = np.array([3.14159, 2.71828, 1.41421, 0.57721,
                                  -1.73205, -0.69315, 0.30103, 1.44270])

    print("=" * 65)
    print("量化直觉演示")
    print("=" * 65)

    print(f"\n原始权重 (FP32):")
    print(f"  {original_weights}")
    print(f"  内存占用: {original_weights.nbytes} bytes ({len(original_weights) * 4} bytes)")

    # 模拟 INT8 量化过程
    def quantize_to_int8(weights):
        q_min, q_max = -128, 127
        w_min, w_max = weights.min(), weights.max()
        scale = (w_max - w_min) / (q_max - q_min)
        zero_point = round(q_min - w_min / scale)

        quantized = np.round(weights / scale + zero_point).astype(np.int8)
        dequantized = (quantized.astype(np.float32) - zero_point) * scale

        return quantized, dequantized, scale, zero_point

    int8_weights, dequantized, scale, zp = quantize_to_int8(original_weights)

    print(f"\n量化后 (INT8):")
    print(f"  {int8_weights}")
    print(f"  内存占用: {int8_weights.nbytes} bytes ({len(int8_weights) * 1} byte)")

    print(f"\n反量化后 (恢复为 FP32):")
    print(f"  {dequantized}")

    print(f"\n误差分析:")
    mse = np.mean((original_weights - dequantized) ** 2)
    max_error = np.abs(original_weights - dequantized).max()
    relative_error = max_error / (np.abs(original_weights).max()) * 100

    print(f"  MSE (均方误差): {mse:.6f}")
    print(f"  最大绝对误差: {max_error:.6f}")
    print(f"  相对误差: {relative_error:.3f}%")

    print(f"\n💡 关键参数:")
    print(f"   Scale (缩放因子): {scale:.6f}")
    print(f"   Zero Point (零点偏移): {zp}")

quantization_intuition()
```

## 1.2 为什么量化能加速推理？

这不仅仅是因为模型变小了。更重要的是**硬件层面的支持**：

```python
def explain_hardware_acceleration():
    """解释量化的硬件加速原理"""

    reasons = [
        {
            "原因": "内存带宽瓶颈",
            "解释": "现代 GPU 的计算能力远超内存带宽。"
                   "一个 A100 有 312 TFLOPS (FP16) 但只有 2 TB/s 的内存带宽。"
                   "这意味着大部分时间花在'等数据从显存搬到计算单元'上。"
                   "量化后数据量减少 → 内存传输时间减少 → 整体加速",
            "例子": "FP16 矩阵乘法: 需要读取 2×N² 字节\n"
                    "INT8 矩阵乘法: 只需读取 N² 字节\n"
                    "内存需求减半!",
        },
        {
            "原因": "Tensor Core 专用加速",
            "解释": "NVIDIA 的 Tensor Core (Volta 架构起) 原生支持 INT8/FP16/BF16 计算。"
                   "INT8 矩阵乘法的吞吐量是 FP32 的 4~16 倍",
            "硬件": "A100: 624 TOPS (INT8) vs 19.5 TFLOPS (FP32)",
        },
        {
            "原因": "缓存命中率提升",
            "解释": "更小的数据意味着更多的权重可以放入 L1/L2 缓存。"
                   "缓存命中率的提升对性能的影响往往被低估",
            "效果": "L2 Cache 通常只有几 MB 到几十 MB,"
                   "量化后的模型更容易完全装入缓存",
        },
    ]

    print("=" * 75)
    print("为什么量化能加速推理?")
    print("=" * 75)
    for r in reasons:
        print(f"\n🔧 {r['原因']}")
        print(f"   解释: {r['解释']}")
        if '例子' in r:
            for line in r['例子'].split('\n'):
                print(f"   {line}")
        if '硬件' in r:
            print(f"   📊 {r['hardware']}")

explain_hardware_acceleration()
```

---

## 二、量化方法分类全景图

## 2.1 PTQ vs QAT

```python
def ptq_vs_qat():
    """PTQ 与 QAT 的全面对比"""

    comparison = {
        "维度": ["是否需要重新训练", "实现复杂度", "精度保持", "适用场景",
                 "所需资源", "典型工具"],
        "PTQ (Post-Training Quantization)": [
            "❌ 不需要", "⭐ 简单", "★★★☆☆ (有损失)",
            "快速部署 / 资源受限", "单卡几分钟", "bitsandbytes, GPTQ, AWQ"],
        "QAT (Quantization-Aware Training)": [
            "✅ 需要", "★★★☆☆ 复杂", "★★★★★ (接近原精度)",
            "追求极致精度的场景", "完整训练资源", "PyTorch QAT, BMT",
        ],
    }

    print("=" * 85)
    print("PTQ vs QAT 对比")
    print("=" * 85)
    dim = comparison["维度"]
    for key in list(comparison.keys())[1:]:
        row = comparison[key]
        print(f"\n📐 {key}:")
        for d, val in zip(dim, row):
            print(f"   {d}: {val}")

ptq_vs_qat()
```

## 2.2 PTQ 的两种模式：动态 vs 静态

```python
def dynamic_vs_static_quantization():
    """动态量化与静态量化的区别"""

    methods = [
        {
            "名称": "动态量化 (Dynamic Quantization)",
            "简称": "DQ",
            "时机": "推理时动态量化激活值",
            "特点": "权重是提前量化好的; 激活值在前向传播时实时量化",
            "优点": "简单易用; 无需校准集; 即插即用",
            "缺点": "每步都要做量化/反量化操作; 速度提升有限 (~2-3x)",
            "适用": "LLM 推理的入门选择",
            "HF API": "`torch.quantization.quantize_dynamic()` 或 `load_in_8bit=True`",
        },
        {
            "名称": "静态量化 (Static Quantization)",
            "简称": "SQ",
            "时机": "推理前离线完成全部量化",
            "特点": "权重和激活值都预先量化好; 需要校准集确定量化参数",
            "优点": "推理速度最快; 精度通常优于动态量化",
            "缺点": "需要准备校准集; 量化过程较慢; 可能不兼容某些输入分布",
            "适用": "生产环境部署; 追求极致性能",
            "HF API": "GPTQ/AWQ 等专用工具; optimum 库导出 ONNX 后量化",
        },
    ]

    print("=" * 80)
    print("动态量化 vs 静态量化")
    print("=" * 80)

    for m in methods:
        print(f"\n{'=' * 80}")
        print(f"📦 {m['名称']}")
        for k, v in m.items():
            if k != "名称":
                print(f"   {k}: {v}")

dynamic_vs_static_quantization()
```

---

## 三、精度格式详解：FP32 → INT4 全链路

## 3.1 各格式的数值范围对比

```python
def precision_formats_comparison():
    """各精度格式的详细对比"""

    formats = [
        ("FP32", 32, "±3.4e38", "~0 (机器精度)", "1.0x", "标准训练/推理"),
        ("FP16", 16, "±65504", "~6e-5 (半精度)", "2.0x", "训练加速/推理"),
        ("BF16", 16, "±3.4e38", "~7.7e-3 (低精度尾数)", "2.0x", "推荐用于 LLM"),
        ("INT8", 8, "-128~127", "1/256 ≈ 0.004", "4.0x", "通用量化"),
        ("INT4", 4, "-8~7 (有符号)", "1/16 = 0.0625", "8.0x", "极限压缩"),
        ("NF4", 4, "-1~1 (归一化)", "信息论最优", "8.0x", "QLoRA 推荐"),
    ]

    print("=" * 90)
    print("数值精度格式全览")
    print("=" * 90)
    print(f"\n{'格式':<8} {'Bit':<6} {'范围':<18} {'最小精度':<20} {'压缩比':<8} {'主要用途'}")
    print("-" * 90)
    for f in formats:
        print(f"{f[0]:<8} {f[1]:<6} {f[2]:<18} {f[3]:<20} {f[4]:<8} {f[5]}")

    print("\n💡 关键发现:")
    print("   • BF16 和 FP16 都是 16-bit, 但 BF16 的范围和 FP32 一样大!")
    print("   • 这就是为什么 LLM 训练/推理首选 BF16 而非 FP16")
    print("   • INT4 的精度只有 1/16, 对精度敏感的任务可能不够")

precision_formats_comparison()
```

## 3.2 FP16 vs BF16：为什么 LLM 更爱 BF16？

这是一个面试高频问题：

```python
def fp16_vs_bf16_deep_dive():
    """FP16 vs BF16 的深度对比"""

    print("=" * 70)
    print("FP16 vs BF16: 为什么 LLM 选择 BF16?")
    print("=" * 70)

    explanation = """
┌─────────────────────────────────────────────────────────────┐
│                     FP16 (Half Precision)                      │
│  ┌──────────┬─────────┐                                      │
│  │ 符号(1) │ 指数(5) │ 尾数(10)                              │
│  └──────────┴─────────┘                                      │
│  范围: ±65504 (指数只有 5 位!)                               │
│  问题: 梯度和激活值容易溢出!                                  │
│  例如: loss = 100000 → FP16 直接溢出为 inf/nan               │
├─────────────────────────────────────────────────────────────┤
│                     BF16 (Brain Float)                         │
│  ┌──────────┬─────────┬──────────┐                           │
│  │ 符号(1) │ 指数(8) │ 尾数(7)   │                           │
│  └──────────┴─────────┴──────────┘                           │
│  范围: ±3.4e38 (和 FP32 一样大!)                             │
│  代价: 尾数少了 3 位 (精度更低)                                │
│  优势: 几乎不会溢出!                                         │
├─────────────────────────────────────────────────────────────┤
│  为什么 LLM 选 BF16?                                         │
│  • LLM 训练中梯度值变化很大, FP16 容易溢出 → 需要 Loss Scaling │
│  • BF16 不需要 Loss Scaling (大多数情况下)                    │
│  • LLM 权重本身不需要很高精度 → 7 位尾数够用了                │
│  • NVIDIA Ampere+ 硬件原生支持 BF16 Tensor Core              │
└─────────────────────────────────────────────────────────────┘
"""
    print(explanation)

fp16_vs_bf16_deep_dive()
```

## 3.3 NF4：专为 LLM 设计的 4-bit 格式

**NF4 (NormalFloat 4-bit)** 是 QLoRA 论文提出的量化格式，专门针对正态分布优化的权重设计：

```python
import torch
import numpy as np


def explain_nf4():
    """NF4 原理解释"""

    print("=" * 70)
    print("NF4 (NormalFloat 4-bit) 详解")
    print("=" * 70)

    # NF4 的 16 个量化级别 (不是均匀分布!)
    nf4_levels = [-1.0, -0.69619277, -0.52507, -0.39491748,
                  -0.28444154, -0.18477386, -0.09105083, 0.0,
                  0.07958047, 0.16093051, 0.246112301, 0.33791521,
                  0.44070107, 0.55552627, 0.68797252, 1.0]

    # 普通 INT4 的 16 个级别 (均匀分布)
    int4_levels = list(range(-8, 8))

    # 模拟一个正态分布的权重
    np.random.seed(42)
    weights = np.random.randn(10000)

    print("\n📊 两种 4-bit 格式的量化级别:")
    print(f"\n{'等级':>6} {'NF4':>12} {'INT4':>12}")
    print("-" * 35)
    for i in range(16):
        print(f"{i:>6} {nf4_levels[i]:>12.8f} {int4_levels[i]:>12}")

    # 计算量化误差
    def compute_mse(data, levels):
        quantized = np.digitize(data, bins=np.array(levels)) - len(levels)//2
        # 简化: 取最近的级别
        closest = np.zeros_like(data)
        for i, v in enumerate(data):
            idx = np.argmin(np.abs(np.array(levels) - v))
            closest[i] = levels[idx]
        return np.mean((data - closest) ** 2)

    mse_nf4 = compute_mse(weights, nf4_levels)
    mse_int4 = compute_mse(weights, int4_levels)

    print(f"\n📈 正态分布权重的量化误差 (MSE):")
    print(f"   NF4:     {mse_nf4:.8f}")
    print(f"   INT4:    {mse_int4:.8f}")
    print(f"   NF4 优势: {((mse_int4 - mse_nf4)/mse_int4*100):.1f}% 误差更低!")

    print(f"\n💡 NF4 的关键洞察:")
    print(f"   • 预训练权重的分布近似正态分布 N(0, σ²)")
    print(f"   • NF4 的量化级别按正态分布的分位数排列")
    print(f"   • 在 0 和 ±1 附近分配了更多级别 (因为权重集中在这里)")
    print(f"   • 这使得 NF4 在相同 bit 数下比普通 INT4 精度更高")

explain_nf4()
```

---

## 四、GPTQ、AWQ 与 GGUF：LLM 专用量化方案

## 4.1 GPTQ：基于近似的二阶信息量化

GPTQ (2023) 是目前最流行的 LLM 4-bit 量化方法之一。它的核心思想是利用 Hessian 矩阵的二阶信息来指导量化，使得量化误差最小化。

```python
def explain_gptq():
    """GPTQ 原理解释"""

    print("=" * 70)
    print("GPTQ: Post-Training Quantization for GPT")
    print("=" * 70)

    explanation = """
算法流程:

  输入: FP16 权重矩阵 W ∈ R^{m×n}
  输出: INT4 量化矩阵 Ŵ + 反量化缩放因子

  Step 1: 将 W 分成列块 (每块 128 列)
  
  Step 2: 对于每一列块:
    a) 计算该块的 Hessian 信息 (二阶导数近似)
       H ≈ 2 × I + 1/N × X^T @ X  (X 为校准数据)
    
    b) 对每个量化层 (共 n/128 个):
       - 固定其他层, 只优化当前层的量化
       - 使用最优 BRP (Best-Rank Projection) 找最佳量化值
       - 更新剩余误差到下一层
  
  Step 3: 得到量化后的权重 Ŵ 和每层的 scale 因子

  核心数学:
    min ||W - Ŵ||²_H  (Hessian 加权的均方误差最小化)

  时间复杂度: O(n²d) per layer (n=隐藏维度, d=量化组大小)
  典型耗时: 7B 模型约 30 分钟 (A100), 70B 模型约 4 小时

  精度: PPL 通常增加 < 5% (相比 FP16)
  文件大小: 约 3.5 GB (7B 模型, 4-bit)
"""
    print(explanation)

explain_gptq()
```

## 4.2 AWQ：激活感知的权重量化

AWQ (Activation-Aware Weight Quantization, 2023) 的创新在于：**不是所有权重都同等重要**。那些对应到大激活值的权重通道应该被保护（保留更高精度），而小激活值的通道可以激进地量化。

```python
def explain_awq():
    """AWQ 原理解释"""

    print("=" * 70)
    print("AWQ: Activation-Aware Weight Quantization")
    print("=" * 70)

    explanation = """
核心洞察:
  传统量化对所有权重一视同仁 → 但不同权重对输出的影响差异巨大!
  
  AWQ 的做法:
  1. 用一小批校准数据跑一遍前向传播, 收集每层的激活值统计
  2. 计算每个权重通道 (channel) 的"重要性":
     salience_s = |W_s| × |X_s|  (权重幅度 × 对应激活幅度)
  3. 保护最重要的通道 (放大其 scale factor, 减少量化误差)
  4. 对不重要的通道进行更激进的量化

  直觉类比:
  就像搬家打包时, 易碎品 (重要权重) 用厚泡沫包裹 (精细量化),
  而衣服 (不重要权重) 可以塞进压缩袋 (粗略量化)

  vs GPTQ 的优势:
  • GPTQ: 基于 Hessian 的全局优化 (慢但精确)
  • AWQ: 基于激活值的局部启发式 (快且效果好)
  • AWQ 通常比 GPTQ 快 2-3x, 且在某些任务上精度更好
"""
    print(explanation)

explain_awq()
```

## 4.3 GGUF：llama.cpp 的原生格式

GGUF (GGML Universal Format) 是 llama.cpp 项目使用的量化格式，广泛用于本地部署：

```python
def explain_gguf():
    """GGUF 格式详解"""

    formats = [
        {"名称": "Q4_0", "描述": "4-bit, 均匀量化", "大小": "~4 GB (7B)", "速度": "快", "精度": "一般"},
        {"名称": "Q4_K_M", "描述": "4-bit, K-quant 混合 (中)", "大小": "~4.1 GB (7B)", "速度": "较快", "精度": "较好"},
        {"名称": "Q5_K_M", "描述": "5-bit, K-quant 混合 (中)", "大小": "~5 GB (7B)", "速度": "中等", "精度": "好"},
        {"名称": "Q6_K", "描述": "6-bit, K-quant", "大小": "~6 GB (7B)", "速度": "中等", "精度": "很好"},
        {"名称": "Q8_0", "描述": "8-bit", "大小": "~8 GB (7B)", "速度": "中等", "精度": "接近 FP16"},
    ]

    print("=" * 75)
    print("GGUF 量化级别速查表 (以 7B 模型为例)")
    print("=" * 75)
    print(f"\n{'格式':<12} {'描述':<25} {'大小':<14} {'速度':<8} {'精度'}")
    print("-" * 75)
    for f in formats:
        print(f"{f['名称']:<12} {f['描述']:<25} {f['大小']:<14} {f['速度']:<8} {f['精度']}")

    print(f"\n💡 GGUF 命名规则:")
    print(f"   Q = 量化")
    print(f"   数字 = 每个 weight 的平均 bit 数")
    print(f"   K = K-quant 方法 (对不同子空间使用不同量化策略)")
    print(f"   _M = Medium (混合策略的中等版本)")
    print(f"\n🎯 推荐选择:")
    print(f"   • 显存紧张 (< 8GB): Q4_K_M (最佳性价比)")
    print(f"   • 显存充足 (≥ 12GB): Q5_K_M 或 Q6_K")
    print(f"   • 追求精度: Q8_0 (接近无损)")

explain_gguf()
```

---

## 五、实战：使用 HF 进行量化推理

## 5.1 BitsAndBytes：最简单的 4-bit/8-bit 方案

```python
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)
import torch


def demo_bitsandbytes_quantization():
    """
    使用 BitsAndBytes 进行量化推理
    这是目前最简单、最流行的 LLM 量化方案
    """

    print("=" * 70)
    print("BitsAndBytes 量化实战")
    print("=" * 70)

    # ========== 配置 1: 8-bit 量化 ==========
    config_8bit = BitsAndBytesConfig(
        load_in_8bit=True,             # 启用 8-bit 加载
        llm_int8_threshold=6.0,        # 异常值阈值 (超过此值的 outlier 保持 FP16)
    )

    # ========== 配置 2: 4-bit NF4 量化 (QLoRA 标准) ==========
    config_4bit = BitsAndBytesConfig(
        load_in_4bit=True,                          # 启用 4-bit 加载
        bnb_4bit_quant_type="nf4",                  # NormalFloat4 格式
        bnb_4bit_compute_dtype=torch.bfloat16,      # 计算时用 BF16
        bnb_4bit_use_double_quant=True,             # 双重量化 (再省 ~0.4 bit/param)
    )

    configs = {
        "INT8 (8-bit)": config_8bit,
        "NF4 (4-bit)": config_4bit,
    }

    model_name = "huggingface/CodeLlama-7b-Instruct-hf"

    for name, config in configs.items():
        print(f"\n{'━' * 60}")
        print(f"配置: {name}")
        print(f"{'━' * 60}")

        # 加载量化模型
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=config,
            device_map="auto",
            trust_remote_code=True,
        )

        # 打印模型信息
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\n模型参数: {total_params / 1e9:.1f}B")

        # 估算显存占用
        if name.startswith("INT8"):
            est_memory_gb = total_params * 1 / (1024**3)  # 1 byte per param
        else:
            est_memory_gb = total_params * 0.5 / (1024**3)  # 0.5 byte per param (4-bit)
        print(f"估算显存: ~{est_memory_gb:.1f} GB (不含 KV cache)")

        # 测试推理
        prompt = "<s>[INST] Write a Python function to sort a list [/INST]"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        output = model.generate(**inputs, max_new_tokens=64, do_sample=True,
                                 temperature=0.7, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(output[0], skip_special_tokens=True)[len(prompt):]

        print(f"\n生成结果 ({name}):")
        print(f"{response}")

        # 清理显存
        del model
        torch.cuda.empty_cache()

demo_bitsandbytes_quantization()
```

## 5.2 不同量化方案的显存估算

```python
def memory_estimation_table():
    """不同模型规模和量化级别的显存估算"""

    models = [("7B", 7e9), ("13B", 13e9), ("34B", 34e9), ("70B", 70e9)]
    precisions = [
        ("FP16", 2, 1.0),
        ("BF16", 2, 1.0),
        ("INT8", 1, 0.55),
        ("NF4 (4-bit)", 0.5, 0.35),
        ("GGUF Q4_K_M", 0.5, 0.35),
    ]

    print("=" * 80)
    print("LLM 量化显存估算表 (仅模型权重, 不含 KV Cache)")
    print("=" * 80)
    
    header = f"{'模型':<8}"
    for p in precisions:
        header += f"{p[0]:<18}"
    print(f"\n{header}")
    print("-" * 98)

    for model_name, params in models:
        row = f"{model_name:<8}"
        for pname, bytes_per_param, kv_factor in precisions:
            weight_mem = params * bytes_per_param / (1024**3)
            # KV Cache 估算 (简化: seq_len=2048, hidden=4096, 2 layers)
            kv_mem = params * 2 * 2048 * 2 / (1024**3) * kv_factor  # 粗略估计
            total = weight_mem + kv_mem
            row += f"{total:>8.1f} GB      "
        print(row)

    print(f"\n💡 说明:")
    print(f"   • 以上为单 GPU 估算, 多 GPU 可通过 tensor parallelism 分摊")
    print(f"   • KV Cache 占据随序列长度线性增长 (seq_len=4096 时翻倍)")
    print(f"   • RTX 3090/4090 (24GB) 可运行 7B INT4 + 中等长度上下文")
    print(f"   • RTX 6000 Ada (48GB) 可运行 13B NF4 + 长上下文")

memory_estimation_table()
```

## 5.3 AutoGPTQ：GPTQ 量化的 HF 集成

```python
def demo_autogptq():
    """使用 AutoGPTQ 进行 GPTQ 量化"""

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

        print("=" * 65)
        print("AutoGPTQ 量化示例")
        print("=" * 65)

        model_name = "meta-llama/Llama-2-7b-hf"

        # Step 1: 加载原始模型
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoGPTQForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True,
        )

        # Step 2: 配置量化参数
        quantize_config = BaseQuantizeConfig(
            bits=4,                    # 4-bit 量化
            group_size=128,           # 量化组大小
            desc_act=False,            # 是否按激活值排序
            sym=False,                # 是否对称量化
        )

        # Step 3: 执行量化 (需要校准数据)
        print("\n正在执行 GPTQ 量化...")
        print("(实际使用中需要准备校准数据集)")

        # 校准数据 (这里只是示意, 实际需要几百条真实文本)
        calibration_data = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "Python is a popular programming language.",
        ] * 50  # 重复以凑够数量

        tokenized_data = [tokenizer(d, return_tensors="pt")["input_ids"] 
                          for d in calibration_data]

        model.quantize(tokenized_data, quantize_config=quantize_config)

        # Step 4: 保存量化后的模型
        save_dir = "./llama2_7b_gptq_4bit"
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        print(f"\n✅ GPTQ 量化完成! 已保存到 {save_dir}")

    except ImportError:
        print("AutoGPTQ 未安装。请运行: pip install auto-gptq")

demo_autogptq()
```

---

## 六、常见误区与最佳实践

## 6.1 五大误区

```python
def quantization_gotchas():
    """量化中的常见误区"""

    gotchas = [
        {
            "误区": "量化后精度一定大幅下降",
            "真相": "对于预训练良好的大模型, 4-bit 量化的精度损失通常 < 2%, "
                   "8-bit 甚至 < 0.5%。关键是选择正确的量化方法 (NF4 > 普通 INT4)",
        },
        {
            "误区": "所有模型都适合量化到 4-bit",
            "真相": "已经经过大量量化训练或蒸馏的小模型 (如 TinyLlama), "
                   "再次量化可能带来较大精度损失。建议先测试再决定",
        },
        {
            "误区": "量化后可以直接用于微调",
            "真相": "普通 4-bit 量化模型不能直接微调 (梯度无法正确传播)。"
                   "需要 QLoRA 技术 (NF4 + LoRA) 来实现 4-bit 微调",
        },
        {
            "误区": "量化一定能节省推理时间",
            "真相": "量化主要节省显存和内存带宽。如果瓶颈在计算而非访存, "
                   "加速可能不明显。配合 FlashAttention + vLLM 效果最好",
        },
        {
            "误区": "INT4 模型的文件大小就是 FP16 的 1/4",
            "真相": "还需要存储反量化的 scale 因子和零点, 以及 embedding 层等 "
                   "通常保持 FP16。实际压缩比约为 3x~4x",
        },
    ]

    print("=" * 78)
    print("量化五大误区")
    print("=" * 78)
    for g in gotchas:
        print(f"\n❌ {g['误区']}")
        print(f"   💡 {g['真相']}")

quantization_gotchas()
```

## 6.2 生产环境 Checklist

```python
def production_checklist():
    """量化部署检查清单"""

    checklist = [
        ("✅ 选择合适的量化级别", "根据可用显存和精度要求权衡"),
        ("✅ 验证量化后精度", "在测试集上对比 PPL 和任务指标"),
        ("✅ 测试边界情况", "超长输入/空输入/特殊字符"),
        ("✅ Benchmark 性能", "首 token 延迟 / 吞吐量 / 显存占用"),
        ("✅ 确认推理框架兼容性", "vLLM/TGI/Ollama 对不同格式支持不同"),
        ("✅ 监控异常值问题", "8-bit 量化注意 llm_int8_threshold 设置"),
    ]

    print("=" * 65)
    print("量化生产部署 Checklist")
    print("=" * 65)
    for item, desc in checklist:
        print(f"  {item}: {desc}")

production_checklist()
```

---

## 七、本章小结

这一节我们系统地学习了模型量化的完整知识体系：

| 方法 | Bit | 原理 | 精度损失 | 速度提升 | 适用场景 |
|------|-----|------|---------|---------|---------|
| **FP16** | 16 | 截断 FP32 | 极小 | 2x | 标准 GPU 推理 |
| **BF16** | 16 | 截断 FP32 (保留范围) | 极小 | 2x | LLM 首选 |
| **INT8 (Dynamic)** | 8 | 动态量化激活值 | < 0.5% | 2~3x | 快速入门 |
| **INT8 (Static)** | 8 | 离线校准量化 | < 0.5% | 3~4x | 生产部署 |
| **NF4 (QLoRA)** | 4 | 正态分布最优量化 | 1~3% | 3~5x | 消费级 GPU |
| **GPTQ** | 4 | Hessian 最优化 | 1~2% | 3~5x | 高质量 4-bit |
| **AWQ** | 4 | 激活感知保护 | 1~2% | 3~5x | 替代 GPTQ |
| **GGUF Q4_K_M** | 4 | K-quant 混合 | 1~3% | 4~6x | 本地部署 |

**核心要点**：
1. **量化的本质是精度换效率**——用可接受的精度损失换取显著的内存和速度收益
2. **BF16 是 LLM 的甜点位**——既有 FP32 的范围又有 FP16 的效率
3. **NF4 + LoRA (QLoRA) 是目前消费级 GPU 上微调大模型的最佳方案**
4. **选择量化方案时要考虑整个技术栈**——量化格式 × 推理框架 × 硬件的兼容性

下一节我们将学习 **知识蒸馏（Knowledge Distillation）**——另一种重要的模型压缩技术。
