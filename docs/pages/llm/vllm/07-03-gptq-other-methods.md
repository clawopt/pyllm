# GPTQ 与其他量化方案

> **白板时间**：AWQ 不是唯一的 INT4 量化方案。GPTQ 是更早出现、生态更成熟的替代品；FP8 是 H100 硬件原生支持的新星；SqueezeLLM 结合了稀疏性和量化；BitsAndBytes 的 NF4 则是 LoRA 微调的最佳搭档。今天我们逐一了解这些方案，并给出选型建议。

## 一、GPTQ（Group-wise Post Training Quantization）

### 1.1 原理简介

GPTQ 基于 **Optimal Brain Quantization (OBQ)** 的思想：

```
核心思路: 逐列量化，每次只关注一个权重列
         量化该列时，最小化对输出结果的影响

数学形式:
  给定权重矩阵 W ∈ R^{m×n}
  对于第 i 列 W[:,i]:
    1. 将其从 FP16 量化为 INT4: Ŵ[:,i] = Q(W[:,i])
    2. 更新剩余列来补偿误差: W[:,i+1:] ← W[:,i+1:] + Δ·correction
    3. 继续下一列

关键: "Group-wise" = 每 group_size=128 列共享一组量化参数
     这比逐参数量化更高效，比全层量化更灵活
```

### 1.2 AWQ vs GPTQ 对比

| 维度 | AWQ | GPTQ |
|------|-----|------|
| **发表时间** | 2023.8 | 2023.3 |
| **核心思想** | 保护重要通道（激活值感知） | OBQ 逐列最优量化 |
| **量化速度** | 快（分钟级） | 较慢（小时级） |
| **精度** | 通常略优 | 略低但差距很小 |
| **HF 模型数量** | 中等 | **最多**（生态最成熟） |
| **vLLM 支持** | ✅ 原生支持 | ✅ 原生支持 |
| **推荐场景** | 追求最佳质量 | 需要特定模型/最大兼容性 |

### 1.3 使用 GPTQ 模型

```bash
# HuggingFace 上 GPTQ 模型最多
# 搜索: TheBloke/*-GPTQ

python -m vllm.entrypoints.openai.api_server \
    --model TheBloke/Llama-2-13B-chat-GPTQ \
    --quantization gptq \
    --dtype auto \
    --port 8000
```

```python
from vllm import LLM, SamplingParams

# Python API
llm_gptq = LLM(
    model="TheBloke/Mistral-7B-Instruct-v0.2-GPTQ",
    quantization="gptq",
)

outputs = llm.generate(["你好"], SamplingParams(max_tokens=32))
print(outputs[0].outputs[0].text)
```

## 二、FP8 量化

### 2.1 为什么 FP8 特殊？

```
传统量化路径:
  FP16 → INT4 (软件模拟) → Tensor Core INT4 计算 → 反量化
  
FP8 路径 (Hopper/H100):
  FP16 → FP8 (硬件原生) → Tensor Core FP8 计算 → 直接输出
  
区别:
  - INT4: 需要"伪量化"（dequantize→compute→quantize 循环）
  - FP8: H100/H200 有原生的 FP8 Tensor Core！一步到位
```

### 2.2 FP8 的两种格式

| 格式 | 符号位 | 指数位 | 尾数位 | 最大值 | 适用场景 |
|------|--------|--------|--------|--------|---------|
| **E4M3fn** | 1 | 4 | 3 | ±448 | 训练 + 推理（权重量化） |
| **E5M2** | 1 | 5 | 2 | ±57344 | 推理（激活值，动态范围大） |

vLLM 通常使用 **E4M3fn** 作为权重量化格式。

### 2.3 FP8 使用条件与方式

```bash
# ⚠️ FP8 需要满足以下条件之一:
# 1. GPU 为 H100 / H200 / H800 (Hopper 架构)
# 2. CUDA ≥ 12.0 + Driver ≥ 535+
# 3. vLLM ≥ 0.5.0

# 启动 FP8 服务
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct-FP8 \
    --quantization fp8 \
    --dtype fp8 \
    --port 8000
```

```python
from vllm import LLM, SamplingParams

llm_fp8 = LLM(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct-FP8",
    quantization="fp8",
    dtype="fp8",  # 必须指定 dtype=fp8
)
```

### 2.4 FP8 性能参考（A100 vs H100）

| GPU | FP16 TTFT | FP8 TTFT | FP16 TPOT | FP8 TPOT | 加速比 |
|-----|----------|---------|----------|---------|-------|
| A100 80GB | 450ms | 420ms* | 42ms | 38ms* | ~1.1x |
| **H100 80GB** | 280ms | **150ms** | 28ms | **15ms** | **~1.8x** |

\* A100 不支持原生 FP8，需要软件模拟，收益有限。**H100 才是 FP8 的最佳拍档**。

## 三、SqueezeLLM

### 3.1 核心思想：稀疏 + 量化双重优化

```
SqueezeLLM 的两板斧:

1. 稀疏化 (Sparsity):
   ┌──────────────────────┐
   │ [0.523][  0 ][3.14] │  →  发现大量接近零的权重
   │ [0.001][-0.02][  0 ] │     将它们精确地设为零
   │ [2.71][  0 ][0.08] │     利用稀疏矩阵跳过计算
   └──────────────────────┘
   
2. 量化 (Quantization):
   对非零权重做 INT4 量化
   
组合效果: 
   稀疏(减少计算量) × 量化(减少存储和带宽) = 2-4x 加速
```

### 3.2 使用方式

```bash
python -m vllm.entrypoints.openai.api_server \
    --model model-name-squeezellm \
    --quantization squeezellm \
    --port 8000
```

**注意**：SqueezeLLM 需要模型本身经过特殊预处理（稀疏化），不能直接对任意模型使用。

## 四、BitsAndBytes (NF4)

### 4.1 NF4：信息论最优的 4-bit 数据类型

NF4 (NormalFloat4) 是 BitsAndBytes 团队设计的一种特殊 4-bit 数据类型：

```python
def nf4_explanation():
    """NF4 数据类型详解"""
    
    explanation = """
    NF4 的设计哲学:
    
    传统 INT4: [-8, -7, ..., 0, ..., 6, 7]
    → 均匀分布，但实际权重不是均匀的！
    
    权重分布特点 (预训练后):
    → 大部分值集中在 0 附近 (正态分布)
    → 极少数值很大 (长尾)
    
    NF4 的解决方案:
    → 将量化级别按正态分布排列
    → 0 附近有更多的量化级别（更精细）
    → 远端量化级别较少（可以接受更大误差）
    
    NF4 的 16 个量化值 (归一化到 [-1, 1]):
    [-1.0, -0.696, -0.526, -0.394, -0.280, -0.199, -0.130, -0.067,
       0.0,   0.067,  0.130,  0.199,  0.280,  0.394,  0.526,  0.696]
    
    注意: NF4 的值是不对称的！
    正半轴最大 0.696，负半轴最大 -1.0
    
    信息论意义: 在给定方差约束下，
    NF4 最大化了量化后信息的保留量。
    """
    print(explanation)

nf4_explanation()
```

### 4.2 NF4 与 QLoRA 的黄金搭档

```
QLoRA (Quantized Low-Rank Adaptation) = NF4 + LoRA

原始微调: 全量 FP16 参数 → 需要 7B × 2B = 14GB 显存 (仅权重)
QLoRA 微调:
  ├── Base Model: NF4 量化 → 7B × 0.5B = 3.5 GB
  ├── LoRA Adapter: FP16 低秩矩阵 → ~0.1 GB
  └── 总计: ~3.6 GB (在单张 RTX 3090 上即可微调 7B!)
  
推理时:
  ├── vLLM 加载 NF4 base model (--quantization bitsandbytes)
  ├── 加载训练好的 LoRA adapter (--enable-lora)
  └── 合并推理: W' = W_NF4 + BA_LoRA
```

### 4.3 vLLM 中使用 BitsAndBytes

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --quantization bitsandbytes \
    --load-format bitsandbytes \
    --port 8000
```

## 五、Marlin 内核：INT4 的极速引擎

### 5.1 什么是 Marlin？

**Marlin** 是一种专门为 INT4 量化推理设计的 CUDA kernel：

```
常规 INT4 推理流程:
  权重 (INT4) → 反量化为 FP16 → MatMul (FP16) → 输出
  ↑ 每次都要反量化，有额外开销

Marlin 流程:
  权重 (INT4) → 直接在 INT4 空间做混合精度 MatMul → 输出
  ↑ 无需反量化！内核内部直接处理 INT4×FP16 计算

效果: INT4 推理速度提升 2-3x（相比通用 GPTQ kernel）
```

### 5.2 Marlin 兼容性

```python
def marlin_compatibility():
    """Marlin 内核兼容性"""
    
    info = """
    Marlin 支持矩阵:
    ├── 输入: FP16/BF16 (激活值)
    ├── 权重: INT4 (量化后的权重)
    └─ 支持 Group Size: 64 和 128
    
    自动激活条件:
    ├── 量化格式: AWQ 或 GPTQ (INT4)
    ├── GPU 架构: Ampere+ (RTX 30+, A100, H100, ...)
    ├── vLLM 版本: ≥ 0.4.0
    └─ 模型架构: LLaMA / Mistral / Qwen 等主流模型
    
    如何确认 Marlin 生效?
    → 查看 vLLM 启动日志中是否有 "Using Marlin kernel" 字样
    → 或者设置 VLLM_LOGGING_LEVEL=DEBUG 查看详细信息
    
    手动指定:
    python -m vllm.entrypoints.openai.api_server \\
        --model xxx-AWQ \\
        --quantization awq \\
        --enable-marlin  # 显式启用
    """
    print(info)

marlin_compatibility()
```

## 六、完整对比与选型决策

### 6.1 一览表

```python
def quantization_comparison_table():
    """所有量化方案综合对比"""
    
    print(f"\n{'='*90}")
    print(f"vLLM 量化方案全面对比")
    print(f"{'='*90}")
    
    methods = [
        ("AWQ", "INT4", "~4x", "1.3-1.5x", "95-98%", "快(分钟)", "多", "⭐ 推荐"),
        ("GPTQ", "INT4", "~4x", "1.2-1.4x", "93-97%", "慢(小时)", "最多", "生态最广"),
        ("FP8", "FP8", "~2x", "1.5-2x*", "98-99%", "极快", "中等", "H100首选"),
        ("SqueezeLLM", "INT4", "~4x", "2-4x", "90-95%", "中等", "少", "特殊场景"),
        ("BnB/NF4", "INT4", "~4x", "1.2-1.3x", "92-96%", "极快", "多", "LoRA搭档"),
        ("Marlin", "INT4", "~4x", "2-3x", "同GPTQ", "N/A", "N/A", "加速内核"),
        ("无(FP16)", "FP16", "1x", "1x (基准)", "100%", "-", "-", "基准线"),
    ]
    
    header = f"{'方法':<12} {'精度':<6} {'压缩':>6} {'加速':>10} {'质量':>8} {'转换速度':>8} {'模型数':>6} {'备注'}"
    print(header)
    print("-" * 90)
    
    for m in methods:
        print(f"{m[0]:<12} {m[1]:<6} {m[2]:>6} {m[3]:>10} {m[4]:>8} {m[5]:>8} {m[6]:>6} {m[7]}")
    
    print("\n* FP8 在 H100 上可达 1.8x；在其他 GPU 上约 1.1x")

quantization_comparison_table()
```

### 6.2 选型决策树

```
你的情况是什么？
│
├─ 有 H100 / H200 GPU
│   └─→ 🎯 **FP8** （硬件原生支持，最佳性价比）
│
├─ 追求最佳 INT4 质量
│   └─→ 🎯 **AWQ** （通常优于 GPTQ）
│
├─ 需要最多的模型选择
│   └─→ 🎯 **GPTQ** （HF 上 GPTQ 模型最丰富）
│
├─ 要配合 LoRA 微调使用
│   └─→ 🎯 **BitsAndBytes NF4**
│
├─ 模型本身有高稀疏性
│   └─→ **SqueezeLLM** （稀疏+量化联合优化）
│
├─ 已有 GPTQ/AWQ 模型，想进一步提速
│   └─→ **Marlin Kernel** （自动或手动启用）
│
└─ 不确定 / 想要最稳妥的选择
    └─→ 🎯 **AWQ INT4** （默认推荐，兼顾质量和生态）
```

---

## 七、总结

本节学习了 vLLM 支持的所有量化方案：

| 方案 | 一句话总结 | 最佳场景 |
|------|-----------|---------|
| **AWQ** | 保护重要通道的 INT4 量化 | 大多数生产部署（默认推荐）|
| **GPTQ** | 基于 OBQ 的逐列最优量化 | 需要特定模型/最大兼容性 |
| **FP8** | H100 原生 8-bit 浮点量化 | H100/H200 硬件（性能之王）|
| **SqueezeLLM** | 稀疏+量化双重优化 | 高稀疏性模型 |
| **BnB/NF4** | 信息论最优 4-bit 类型 | QLoRA 微调搭档 |
| **Marlin** | INT4 极速 CUDA 内核 | 已有 GPTQ/AWQ 模型想再提速 |

**核心要点回顾**：

1. **没有"最好的"量化方案，只有"最适合的"**——根据你的硬件和需求选择
2. **H100 用户必用 FP8**——这是唯一能充分发挥硬件优势的量化方案
3. **非 H 用户首选 AWQ INT4**——质量好、速度快、社区活跃
4. **GPTQ 是保底选择**——几乎每个模型都有 GPTQ 版本
5. **Marlin 是免费的加速器**——如果你已经在用 GPTQ/AWQ，确保 Marlin 内核已启用
6. **量化方案可以组合**——比如 FP8 权重 + INT8 激活（W8A8）

至此，**Chapter 7（量化技术）还剩最后一节**。接下来学习 **量化模型性能对比与选型**。
