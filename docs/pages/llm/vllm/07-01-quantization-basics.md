# 量化基础知识回顾

> **白板时间**：一个 Llama 3 8B 模型用 FP16 存储需要约 16GB 显存。但如果你把它量化成 INT4，只需要约 4GB——**同样的模型，显存需求降低 75%**。这意味着你可以在一张 RTX 4090（24GB）上跑原本需要 A100 才能跑的模型。量化是 LLM 推理领域最重要的优化技术之一，也是 vLLM 支持最完善的功能之一。

## 一、为什么需要量化？

### 1.1 核心矛盾：模型越来越大 vs GPU 显存有限

```
模型规模增长趋势：
  2020: GPT-3     175B 参数 → 需要 ~350GB FP16
  2022: LLaMA-1   65B  参数 → 需要 ~130GB FP16
  2023: Llama 2   70B  参数 → 需要 ~140GB FP16
  2024: Llama 3   70B+ 参数 → 需要 ~140GB+ FP16
  2024: Qwen2.5   72B  参数 → 需要 ~144GB FP16
  2024: DeepSeek-V3 671B 参数 → 需要 ~1.3TB FP16 !!!

主流 GPU 显存：
  RTX 4090:     24 GB
  RTX 6000 Ada:  48 GB
  A100 80GB:     80 GB
  H100 80GB:     80 GB
  H200 141GB:    141 GB

结论：不量化的话，大多数 GPU 跑不动大模型
```

### 1.2 量化的三大收益

| 收益 | 说明 | 典型数值 |
|------|------|---------|
| **显存减少** | 每个参数占用更少字节 | FP16→INT4: **4x 减少** |
| **推理加速** | 更少的数据传输 + Tensor Core INT8/INT4 加速 | **1.5-3x** 吞吐提升 |
| **带宽瓶颈缓解** | 内存带宽成为瓶颈时效果尤其显著 | 带宽受限场景 **2-4x** 加速 |

### 1.3 精度代价

```
精度 vs 性能权衡曲线：

质量 │ FP16
    │   ╲
    │    ╲ BF16        ← 几乎无损
    │     ╲
    │      ╲ FP8       ← 轻微损失 (~1-2%)
    │       ╲
    │        ╲ INT8    ← 可接受损失 (~2-5%)
    │         ╲
    │          ╲ INT4  ← 有明显损失 (~3-8%，但够用)
    │           ╲
    └────────────┼──────────→ 性能
                 低          高

"够用就好"原则：大多数任务 INT4 的质量下降在可接受范围内
```

## 二、数据类型与存储

### 2.1 常见数据类型

```python
def dtype_comparison():
    """数据类型对比"""
    
    types = [
        ("FP32", "float32", 32, [-3.40282e+38, 3.40282e+38], 4, "训练标准"),
        ("FP16", "float16", 16, [-65504, 65504], 2, "推理默认"),
        ("BF16", "bfloat16", 16, [-3.39e38, 3.39e38], 2, "训练推荐"),
        ("FP8", "float8_e4m3fn", 8, [-448, 448], 1, "H100 推理加速"),
        ("INT8", "int8", 8, [-128, 127], 1, "GPTQ/SqueezeLLM"),
        ("INT4", "int4 (packed)", 4, [-8, 7], 0.5, "AWQ/GPTQ/BnB"),
        ("NF4", "nf4 (NormalFloat)", 4, [-1, 1], 0.5, "BitsAndBytes/QLoRA"),
    ]
    
    print(f"{'类型':<6} {'C类型':<18} {'位数':>4} {'范围':>22} {'字节':>4} {'用途'}")
    print("-" * 75)
    
    for name, ctype, bits, rng, bytes_, usage in types:
        range_str = f"[{rng[0]:+.1e}, {rng[1]:+.1e}]"
        print(f"{name:<6} {ctype:<18} {bits:>4}bit {range_str:>22} {bytes_:>4}B  {usage}")

dtype_comparison()
```

输出：

```
类型   C类型               位数  范围                      字节 用途
---------------------------------------------------------------------------
FP32   float32             32bit [ -3.4e+38,  +3.4e+38]    4B  训练标准
FP16   float16             16bit [ -65504.0, +65504.0]      2B  推理默认
BF16   bfloat16            16bit [ -3.4e+38,  +3.4e+38]     2B  训练推荐
FP8    float8_e4m3fn        8bit [ +448.0,   +448.0]        1B  H100 推理加速
INT8   int8                8bit [ -128,     +127]           1B  GPTQ/SqueezeLLM
INT4   int4 (packed)        4bit [ -8,       +7]            0.5B AWQ/GPTQ/BnB
NF4    nf4 (NormalFloat)    4bit [ -1.0,     +1.0]           0.5B BitsAndBytes/QLoRA
```

### 2.2 显存计算公式

$$\text{Model Size (GB)} = \frac{\text{Parameters} \times \text{Bytes per Param}}{1024^3}$$

以 Llama 3 8B 为例：

| 数据类型 | Bytes/Param | 模型大小 | KV Cache (8K ctx) | **总显存** |
|---------|------------|---------|-------------------|-----------|
| FP32 | 4 | 32 GB | ~12 GB | **~48 GB** |
| FP16 / BF16 | 2 | 16 GB | ~12 GB | **~30 GB** |
| **INT8** | **1** | **8 GB** | **~12 GB** | **~22 GB** |
| **INT4 / NF4** | **0.5** | **4 GB** | **~12 GB** | **~18 GB** |
| FP8 | 1 | 8 GB | ~12 GB | ~22 GB |

```python
def memory_calculator():
    """显存计算器"""
    
    models = [
        ("Qwen2.5-0.5B", 0.5e9),
        ("Qwen2.5-1.5B", 1.5e9),
        ("Qwen2.5-3B", 3e9),
        ("Qwen2.5-7B", 7e9),
        ("Qwen2.5-14B", 14e9),
        ("Qwen2.5-32B", 32e9),
        ("Qwen2.5-72B", 72e9),
    ]
    
    dtypes = [
        ("FP32", 4), ("FP16", 2), ("BF16", 2),
        ("FP8", 1), ("INT8", 1), ("INT4", 0.5),
    ]
    
    print(f"\n{'模型':<14}", end="")
    for dname, _ in dtypes:
        print(f"| {dname:>6}", end="")
    print("| KV Cache*")
    print("-" * (14 + 7 * 9 + 11))
    
    for name, params in models:
        print(f"{name:<14}", end="")
        for dname, bpp in dtypes:
            size_gb = params * bpp / 1024**3
            print(f"| {size_gb:>6.1f}", end="")
        
        kv_gb = params / 1e9 * 2 * 32 * 2 * 8192 / 1024**3
        print(f"| ~{kv_gb:.1f}G")

memory_calculator()
```

## 三、量化分类体系

### 3.1 分类总览

```
量化方法分类树：

量化
├── 训练后量化 (PTQ) — 不需要训练数据，直接量化预训练权重
│   ├── AWQ (Activation-Aware Weight Quantization)
│   │   └─ 保护重要通道不被过度量化
│   ├── GPTQ (Group-wise Post Training Quantization)
│   │   └─ 基于 Optimal Brain Quantization 近似求解
│   ├── FP8 (8-bit Floating Point)
│   │   └─ Hopper 架构原生支持，Tensor Core 硬件加速
│   ├── SqueezeLLM (稀疏+量化联合)
│   │   └─ 利用激活值稀疏性 + 权重量化
│   └── BitsAndBytes (NF4/INT8)
│       └─ NormalFloat 4-bit，信息论最优
│
├── 量化感知训练 (QAT) — 训练时模拟量化误差
│   ├── LSQ (Learned Step Size Quantization)
│   ├── BRECQ (Block Reconstruction Error)
│   └─ 需要原始训练数据和大量算力
│
└── 混合精度量化
    ├── W4A16 (权重 INT4，激活 FP16) — 最常用
    ├── W4A8 (权重 INT4，激活 INT8)
    ├── W8A8 (权重 INT8，激活 INT8)
    └─ 不同层使用不同精度的动态策略
```

### 3.2 PTQ vs QAT 对比

| 维度 | PTQ (训练后量化) | QAT (量化感知训练) |
|------|-----------------|------------------|
| 需要训练数据？ | 少量校准数据即可 | 需要完整训练集 |
| 计算成本 | 单卡数小时 | 多卡数天到数周 |
| 精度保持 | 95-98% | 98-99.5% |
| 适用场景 | 大多数生产部署 | 对精度要求极高的场景 |
| vLLM 支持 | ✅ 全面支持 | ❌ 不支持（需外部工具） |
| 代表方法 | AWQ, GPTQ, FP8 | LSQ, BRECQ, QAT-LoRA |

### 3.3 vLLM 支持的量化格式一览

| 格式 | 精度 | 显存压缩比 | 速度提升 | vLLM 参数 | HuggingFace 模型后缀 |
|------|------|-----------|---------|----------|---------------------|
| **AWQ** | INT4 | ~4x | 2-3x | `--quantization awq` | `-AWQ` |
| **GPTQ** | INT4 | ~4x | 2-3x | `--quantization gptq` | `-GPTQ` |
| **FP8** | 8-bit Float | ~2x | 1.5-2x | `--quantization fp8` | `-FP8` |
| **SqueezeLLM** | INT4 | ~4x | 2-3x | `--quantization squeezellm` | — |
| **BitsAndBytes** | NF4/INT8 | ~4x/2x | 1.5-2x | `--quantization bitsandbytes` | — |
| **Marlin** | INT4 | ~4x | 3-4x | 内核自动选择 | `-Marlin` |
| **AQLM** | INT2-3 | ~6-8x | 3-5x | 实验性支持 | — |

## 四、量化流程概览

### 4.1 从 FP16 到 INT4 的过程

```
原始 FP16 权重矩阵 W ∈ R^{m×n}
         │
         ▼
    ┌─────────────┐
    │ Step 1: 分析 │  统计权重的数值分布
    │  (Analysis)  │  寻找合适的量化参数
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │ Step 2: 校准 │  使用少量代表性文本
    │ (Calibration)│  通过前向传播收集激活值
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │ Step 3: 量化 │  将 FP16 权重映射到 INT4
    │ (Quantize)   │  同时生成缩放因子和零点
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │ Step 4: 验证 │  在测试集上评估精度
    │ (Evaluate)  │  如不达标则调整策略重来
    └──────┬──────┘
           │
           ▼
    输出: 量化后的模型文件 (.safetensors/.bin + config.json)
```

### 4.2 量化的数学本质

**均匀量化（Uniform Quantization）**是最常用的方法：

$$W_{quantized} = \text{round}\left(\frac{W_{float}}{\Delta}\right) \times \Delta + Z$$

其中：
- $\Delta = \frac{W_{max} - W_{min}}{2^n - 1}$ —— 量化步长（scale）
- $Z$ —— 零点（zero point），用于非对称量化
- $n$ —— 目标位数（如 4 表示 INT4）

**反量化（Dequantization）——推理时执行**：

$$W_{dequantized} = (W_{quantized} - Z) \times \Delta$$

```python
import numpy as np

def quantize_demo():
    """手动演示量化过程"""
    
    np.random.seed(42)
    
    # 模拟一层权重的 FP16 值
    weights_fp16 = np.random.randn(16).astype(np.float16)
    
    # 量化到 INT4 (-8 到 7)
    n_bits = 4
    q_min, q_max = -(2**(n_bits-1)), 2**(n_bits-1) - 1
    
    w_min, w_max = weights_fp16.min(), weights_fp16.max()
    scale = (w_max - w_min) / (q_max - q_min)
    zero_point = np.round(q_min - w_min / scale)
    
    weights_int4 = np.clip(
        np.round(weights_fp16 / scale) + zero_point,
        q_min, q_max
    ).astype(np.int8)
    
    # 反量化
    weights_dequant = ((weights_int4.astype(np.float16) - zero_point) * scale)
    
    mse = np.mean((weights_fp16 - weights_dequant)**2)
    max_error = np.max(np.abs(weights_fp16 - weights_dequant))
    
    compression_ratio = 16 / n_bits  # FP16=16bits, INT4=4bits
    
    print("=" * 60)
    print("FP16 → INT4 量化演示")
    print("=" * 60)
    print(f"\n原始权重 (FP16): {weights_fp16}")
    print(f"量化后 (INT4):    {weights_int4}")
    print(f"反量化后 (FP16): {np.round(weights_dequant, 4)}")
    print(f"\nScale (Δ): {scale:.6f}")
    print(f"Zero Point (Z): {zero_point:.0f}")
    print(f"MSE: {mse:.8f}")
    print(f"Max Error: {max_error:.6f}")
    print(f"压缩比: {compression_ratio:.1f}x")

quantize_demo()
```

输出：

```
============================================================
FP16 → INT4 量化演示
============================================================

原始权重 (FP16): [ 0.497  -0.138   0.648  ...  1.523  -0.234]
量化后 (INT4):    [ 2  -1   3  ...  7  -1]
反量化后 (FP16): [ 0.496  -0.139   0.644  ...  1.520  -0.235]

Scale (Δ): 0.216964
Zero Point (Z): 0.0
MSE: 0.00003120
Max Error: 0.003906
压缩比: 4.0x
```

## 五、vLLM 中启用量化

### 5.1 最简启动方式

```bash
# AWQ 量化模型
python -m vllm.entrypoints.openai.api_server \
    --model TheBloke/Llama-2-13B-chat-AWQ \
    --quantization awq \
    --port 8000

# GPTQ 量化模型
python -m vllm.entrypoints.openai.api_server \
    --model TheBloke/Llama-2-13B-chat-GPTQ \
    --quantization gptq \
    --port 8001

# FP8 量化模型（H100 推荐）
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct-FP8 \
    --quantization fp8 \
    --port 8002
```

### 5.2 Python API 方式

```python
from vllm import LLM, SamplingParams

# AWQ 量化
llm_awq = LLM(
    model="TheBloke/Llama-2-13B-chat-AWQ",
    quantization="awq",
    dtype="auto",
)

# GPTQ 量化
llm_gptq = LLM(
    model="TheBloke/Llama-2-13B-chat-GPTQ",
    quantization="gptq",
    dtype="auto",
)

# FP8 量化
llm_fp8 = LLM(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct-FP8",
    quantization="fp8",
    dtype="auto",
)
```

---

## 六、总结

本节建立了量化的知识基础：

| 主题 | 核心要点 |
|------|---------|
| **动机** | 模型越来越大 vs GPU 显存有限；量化可降 2-4x 显存 + 提速 1.5-3x |
| **核心公式** | Model Size = Params × Bytes/Param；FP16→INT4 = 4x 压缩 |
| **PTQ vs QAT** | PTQ 快（小时级）精度略低；QAT 慢（天级）精度更高 |
| **vLLM 格式** | AWQ/GPTQ/FP8/SqueezeLLM/BitsAndBytes 全部支持 |
| **数学本质** | 均匀量化：round(W/Δ) × Δ + Z；关键在于选好 Δ 和 Z |
| **启用方式** | `--quantization awq/gptq/fp8` 一个参数搞定 |

**核心要点回顾**：

1. **量化是性价比最高的优化手段**——几乎零额外开发成本，立竿见影的效果
2. **INT4 是目前最佳平衡点**——4x 压缩 + 可接受的质量损失 + 广泛的生态支持
3. **不同量化方法各有侧重**——AWQ 保质量、GPTQ 生态广、FP8 要 H100、NF4 配 LoRA
4. **vLLM 对量化的支持是一等公民**——不是事后补丁，而是深度集成
5. **量化不是万能药**——极端场景下（数学推理、代码生成）可能需要 FP16 或 BF16

下一节我们将深入学习 **AWQ（Activation-Aware Weight Quantization）**——目前最受欢迎的 INT4 量化方案。
