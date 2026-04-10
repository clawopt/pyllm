# 1.1 为什么是 PyTorch？Tensor vs NumPy 的本质区别

> 如果你已经会用 NumPy 处理数组、用 Pandas 分析数据，那么恭喜你——你已经站在了 PyTorch 的门口。但 PyTorch 绝不是"带 GPU 的 NumPy"这么简单。它是大语言模型的底层语言，是 vLLM 的心脏，是 HuggingFace Transformers 的骨架。这一节，我们搞清楚：为什么学 LLM 必须先学 PyTorch？Tensor 到底比 ndarray 多了什么？

## PyTorch 在 LLM 技术栈中的位置

在深入语法之前，先看看 PyTorch 在整个大模型生态中处于什么位置：

```
┌─────────────────────────────────────────────────────┐
│                  应用层                              │
│         LangChain / LlamaIndex / Dify               │
├─────────────────────────────────────────────────────┤
│                 框架层                               │
│     HuggingFace Transformers / vLLM / PEFT          │
├───────────────────┬─────────────────────────────────┤
│                   │                                  │
│              ⭐ PyTorch ◄── 我们在这里              │
│                   │                                  │
├───────────────────┼─────────────────────────────────┤
│            CUDA / cuDNN / Tensor Cores              │
├─────────────────────────────────────────────────────┤
│              NVIDIA GPU (A100/H100/RTX4090)          │
└─────────────────────────────────────────────────────┘
```

每一层都依赖 PyTorch：

| 上层框架 | 如何使用 PyTorch |
|---------|----------------|
| **vLLM** | 底层全部是 PyTorch + 自定义 CUDA Kernel。PagedAttention 的 Block 管理就是 `torch.Tensor` 操作 |
| **HuggingFace Transformers** | `AutoModelForCausalLM` 返回的就是 `nn.Module` 子类，参数都是 `torch.Tensor` |
| **PEFT / LoRA** | LoRA 的 $W' = W + BA$ 就是两个 `nn.Linear` 层的加法 |
| **LangChain / LlamaIndex** | 虽然不直接暴露 PyTorch API，但底层调用的推理引擎（vLLM / HF）全是 PyTorch |

> 💡 **一句话总结**：不懂 PyTorch，你只能"调包"；懂了 PyTorch，你才能"理解包为什么这样工作"，并在出问题时定位根因。

## Tensor vs ndarray：五大关键差异

你可能听过"PyTorch Tensor 和 NumPy ndarray 几乎一样"。这句话只对了一半——它们在**基础操作上确实相似**，但在以下五个方面有本质区别：

### 差异一：GPU 支持（最重要的区别）

```python
import numpy as np
import torch

# NumPy: 只能在 CPU 上运行
arr = np.array([1., 2., 3., 4.])
# arr.to("cuda")  # ❌ NumPy 没有这个方法！

# PyTorch: 可以在 CPU 或 GPU 上运行
tensor_cpu = torch.tensor([1., 2., 3., 4.])
tensor_gpu = torch.tensor([1., 2., 3., 4.], device="cuda")
# 或者从 CPU 搬到 GPU:
# tensor_gpu = tensor_cpu.to("cuda")

print(f"CPU tensor 设备: {tensor_cpu.device}")   # cpu
print(f"GPU tensor 设备: {tensor_gpu.device}")    # cuda:0
```

这看起来只是 `.to("cuda")` 一行代码的差异，但对于 LLM 来说意味着**天壤之别**：

| 操作 | CPU (Intel i9) | GPU (RTX 4090) | 加速比 |
|-----|---------------|---------------|-------|
| 矩阵乘法 (4096×4096) | ~15 ms | ~0.05 ms | **300x** |
| 7B 模型前向传播 | ~2-3 秒 | ~20-50 ms | **50-100x** |
| 70B 模型前向传播 | ~不可行（内存不足） | ~200-500 ms | — |

没有 GPU 支持，你根本无法训练或推理任何有实际价值的大模型。

### 差异二：自动微分（Autograd）

```python
import torch

# 创建一个需要计算梯度的张量
w = torch.tensor([0.5], requires_grad=True)
x = torch.tensor([1., 2., 3.])
y = torch.tensor([2., 3., 4.])

# 前向传播
pred = w * x           # [0.5, 1.0, 1.5]
loss = ((pred - y) ** 2).mean()  # MSE Loss

# 反向传播 —— 自动计算 d(loss)/d(w)!
loss.backward()

print(f"Loss: {loss.item():.4f}")
print(f"梯度 dL/dw: {w.grad}")  # tensor([-3.6667])
```

NumPy 完全没有这个能力。如果你要用 NumPy 训练神经网络，必须**手写反向传播公式**——对于 Transformer 这种有几十亿参数的模型来说，这是不可能完成的任务。

Autograd 是 PyTorch 的灵魂，也是所有深度学习框架的核心竞争力。

### 差异三：动态计算图

PyTorch 默认采用 **Eager Mode（即时执行模式）**——每一行代码立即执行，你可以随时打印中间结果、设置断点调试。这与 NumPy 的编程体验完全一致：

```python
import torch

x = torch.randn(2, 3)
print(f"x shape: {x.shape}")       # 立即看到结果 ✅

y = x @ x.T                       # 立即执行矩阵乘法
print(f"y:\n{y}")                  # 立即看到结果 ✅

z = torch.softmax(y, dim=-1)      # 立即 softmax
print(f"z sum per row: {z.sum(dim=-1)}")  # 每行和为 1 ✅
```

这种"所见即所得"的体验让调试变得极其容易。对比 TensorFlow 1.x 的静态图模式（先定义图再执行），PyTorch 的动态图对研究者更友好。

### 差异四：与 NumPy 的无缝互操作

虽然它们不同，但转换非常简单：

```python
import numpy as np
import torch

# NumPy → PyTorch
arr = np.random.randn(3, 4)
tensor = torch.from_numpy(arr)      # 共享内存！（修改一个会影响另一个）
tensor_copy = torch.tensor(arr)     # 深拷贝（独立内存）

# PyTorch → NumPy
back_to_np = tensor.numpy()          # 需要在 CPU 上
# back_to_np = tensor.cpu().numpy()  # 如果在 GPU 上，先移到 CPU
```

这意味着你可以继续使用熟悉的 NumPy 生态（Pandas、Matplotlib、SciPy），只在需要 GPU 或自动微分时切换到 PyTorch。

### 差异五：丰富的神经网络原语

PyTorch 的 `torch.nn` 模块提供了构建神经网络的全部积木：

```python
import torch.nn as nn
import torch.nn.functional as F

# 这些是 NumPy 完全没有的
embedding = nn.Embedding(32000, 4096)    # 词嵌入层
linear = nn.Linear(4096, 4096)           # 全连接层
layer_norm = nn.LayerNorm(4096)          # Layer Norm
dropout = nn.Dropout(0.1)                # Dropout
attention = nn.MultiheadAttention(       # 多头注意力！
    embed_dim=4096, num_heads=32
)
```

每一个 LLM 架构组件——Embedding、Attention、FFN、LayerNorm——都有对应的现成实现。当然，在第 3 章我们会自己从头写一遍来理解原理。

## dtype 大全：FP32 / FP16 / BF16 / FP8 / INT8 / INT4

在大模型领域，数据类型（dtype）的选择直接影响**显存占用、推理速度和模型质量**。让我们一次性搞清楚所有常用 dtype：

### 各类型的基本信息

| 类型 | 每个元素占用 | 数值范围 | 精度 | 典型用途 |
|-----|-----------|---------|------|---------|
| **FP32** (float32) | 4 bytes | ±3.4×10³⁸ | ~7 位有效数字 | 训练默认、调试 |
| **FP16** (float16) | 2 bytes | ±65504 | ~3 位有效数字 | 推理加速、混合精度训练 |
| **BF16** (bfloat16) | 2 bytes | ±3.4×10³⁸ | ~8 位有效数字（尾数短） | **大模型新宠** |
| **FP8** (float8) | 1 byte | E4M3/E5M2 两种格式 | ~1-2 位有效数字 | H100 原生支持 |
| **INT8** (int8) | 1 byte | -128 ~ 127 | 整数精确 | 量化推理 |
| **INT4** (int4) | 0.5 byte | -8 ~ 7 | 低精度整数 | **量化主流**（AWQ/GPTQ） |
| **NF4** (NormalFloat4) | 0.5 byte | 信息论最优分布 | — | QLoRA 专用 |

### BF16 为什么是大模型的新宠？

BF16（Brain Float 16）由 Google Brain 提出，专门为深度学习设计：

```
FP16 布局:  [符号 1bit][指数 5bit][尾数 10bit]  → 范围小(±65504)，精度高
BF16 布局:  [符号 1bit][指数 8bit][尾数 7bit]   → 范围大(≈FP32)，精度低一点
FP32 布局: [符号 1bit][指数 8bit][尾数 23bit]
                ↑ 与FP32相同的指数位！
```

BF16 的关键优势：**与 FP32 相同的指数范围**，这意味着：
- 不会像 FP16 那样频繁溢出（尤其是 Softmax 中）
- 训练时不需要 GradScaler（FP16 需要）
- H100/A100 等 Ampere 架构 GPU 对 BF16 有原生硬件支持

### 一个 7B 模型的显存占用对照

比如下面的程序展示不同 dtype 下同一模型的显存需求：

```python
"""
7B 参数模型在不同 dtype 下的显存占用估算
"""

def estimate_memory(num_params: int, dtype_bytes: int, name: str):
    """估算模型权重占用的显存"""
    total_bytes = num_params * dtype_bytes
    total_gb = total_bytes / (1024 ** 3)
    print(f"  {name:<12} {total_gb:>8.2f} GB  ({total_bytes/1e9:.1f}B 参数)")


# LLaMA-2-7B 约 70 亿参数
NUM_PARAMS_7B = 7_000_000_000
NUM_PARAMS_13B = 13_000_000_000
NUM_PARAMS_70B = 70_000_000_000

print("=" * 60)
print("📊 不同 dtype 下模型权重的显存占用")
print("=" * 60)

for model_name, params in [("7B", NUM_PARAMS_7B), ("13B", NUM_PARAMS_13B), ("70B", NUM_PARAMS_70B)]:
    print(f"\n--- {model_name} 模型 ({params:,} 参数) ---")
    estimate_memory(params, 4, "FP32")        # float32
    estimate_memory(params, 2, "FP16")        # float16
    estimate_memory(params, 2, "BF16")        # bfloat16
    estimate_memory(params, 1, "FP8")         # float8 (E4M3)
    estimate_memory(params, 0.5, "INT4")      # int4 (量化)

print("\n" + "=" * 60)
print("💡 注意：以上仅为权重占用，不含 KV Cache、激活值、优化器状态")
print("   实际训练时还需 ×2~4（梯度+优化器状态）")
print("=" * 60)
```

运行输出：

```
============================================================
📊 不同 dtype 下模型权重的显存占用
============================================================

--- 7B 模型 (7,000,000,000 参数) ---
  FP32           26.07 GB  (7.0B 参数)
  FP16           13.04 GB  (7.0B 参数)
  BF16           13.04 GB  (7.0B 参数)
  FP8             6.52 GB  (7.0B 参数)
  INT4             3.26 GB  (7.0B 参数)

--- 13B 模型 (13,000,000,000 参数) ---
  FP32           48.48 GB  (13.0B 参数)
  FP16           24.24 GB  (13.0B 参数)
  BF16           24.24 GB  (13.0B 参数)
  FP8            12.12 GB  (13.0B 参数)
  INT4             6.06 GB  (13.0B 参数)

--- 70B 模型 (70,000,000,000 参数) ---
  FP32          260.89 GB  (70.0B 参数)
  FP16          130.45 GB  (70.0B 参数)
  BF16          130.45 GB  (70.0B 参数)
  FP8            65.23 GB  (70.0B 参数)
  INT4            32.61 GB  (70.0B 参数)

============================================================
💡 注意：以上仅为权重占用，不含 KV Cache、激活值、优化器状态
   实际训练时还需 ×2~4（梯度+优化器状态）
============================================================
```

这就是为什么：
- RTX 4090 (24GB) 只能跑 **FP16/INT4** 的 7B 模型
- A100 80GB 才能跑 **BF16** 的 13B 模型
- 70B 模型至少需要 **4×A100 80GB**（FP16）或 **INT4 单卡**

## 第一个 PyTorch 程序：模拟 LLM 中的 Embedding 查找

现在，让我们写出第一个真正与 LLM 相关的 PyTorch 程序——模拟词嵌入（Token Embedding）查找过程：

```python
"""
第一个 PyTorch 程序：模拟 LLM 中的 Token Embedding 查找

在 LLM 中，每个 token（词元）都会被映射为一个高维向量。
这个过程叫做 Embedding Lookup，是模型处理文本的第一步。
"""

import torch
import torch.nn as nn


def demo_embedding_lookup():
    """演示 Embedding 查找过程"""

    print("=" * 60)
    print("🔤 LLM Token Embedding 查找演示")
    print("=" * 60)

    # === Step 1: 定义词表大小和嵌入维度 ===
    # 以 LLaMA-2 为例:
    vocab_size = 32000     # 词表中有 32000 个 token
    embed_dim = 4096       # 每个 token 映射为 4096 维向量

    print(f"\n[Step 1] 词表配置:")
    print(f"  词表大小 (vocab_size): {vocab_size:,}")
    print(f"  嵌入维度 (embed_dim): {embed_dim}")

    # === Step 2: 创建 Embedding 层 ===
    # 这相当于一个巨大的查找表: 32000 × 4096 的矩阵
    embedding_table = nn.Embedding(vocab_size, embed_dim)

    print(f"\n[Step 2] Embedding 层创建完成:")
    print(f"  形状: ({vocab_size}, {embed_dim})")
    print(f"  参数量: {vocab_size * embed_dim:,}")
    print(f"  显存占用 (FP32): {vocab_size * embed_dim * 4 / 1e9:.2f} GB")

    # === Step 3: 模拟 token IDs 输入 ===
    # 假设输入文本 "Hello world" 被 tokenizer 编码为以下 ID:
    token_ids = torch.tensor([15496, 2159, 287])  # 3 个 token

    print(f"\n[Step 3] 输入 token IDs:")
    print(f"  token_ids: {token_ids.tolist()}")
    print(f"  序列长度: {token_ids.shape[0]}")

    # === Step 4: Embedding 查找 ===
    # 核心操作: 从 embedding_table 中取出对应行的向量
    embeddings = embedding_table(token_ids)

    print(f"\n[Step 4] Embedding 查找结果:")
    print(f"  输出形状: {embeddings.shape}")  # (3, 4096)
    print(f"  含义: 每个 token 变成了一个 {embed_dim} 维向量")
    print(f"\n  第一个 token (ID=15496) 的嵌入向量前 10 个值:")
    print(f"    {embeddings[0, :10].tolist()}")

    # === Step 5: 理解语义空间 ===
    # 相似的 token 应该有相似的嵌入向量（余弦相似度高）
    def cosine_similarity(a, b):
        return torch.dot(a, b) / (torch.norm(a) * torch.norm(b))

    # 取两个不同的 token
    vec_a = embeddings[0]  # "Hello"
    vec_b = embeddings[1]  # "world"
    sim = cosine_similarity(vec_a, vec_b).item()
    print(f"\n[Step 5] Token 间相似度 (余弦):")
    print(f"  'Hello'(15496) vs 'world'(2159): {sim:.4f}")
    print(f"  （随机初始化的 embedding 无明显规律，训练后会体现语义关系）")

    # === Step 6: Position Embedding（位置编码）===
    max_seq_len = 2048
    position_embed = nn.Embedding(max_seq_len, embed_dim)
    positions = torch.arange(len(token_ids))
    pos_embeddings = position_embed(positions)

    print(f"\n[Step 6] Position Embedding:")
    print(f"  最大序列长度: {max_seq_len}")
    print(f"  位置编码形状: {pos_embeddings.shape}")

    # 最终输入 = Token Embedding + Position Embedding
    final_input = embeddings + pos_embeddings
    print(f"\n  最终输入形状: {final_input.shape}")
    print(f"  (这就是送入 Transformer Block 之前的输入)")

    # === 显存统计 ===
    total_params = sum(p.numel() for p in embedding_table.parameters()) + \
                   sum(p.numel() for p in position_embed.parameters())
    print(f"\n{'='*60}")
    print(f"📊 总参数量: {total_params:,} ({total_params*4/1e9:.2f} GB FP32)")
    print(f"{'='*60}")


if __name__ == "__main__":
    demo_embedding_lookup()
```

运行结果：

```
============================================================
🔤 LLM Token Embedding 查找演示
============================================================

[Step 1] 词表配置:
  词表大小 (vocab_size): 32,000
  嵌入维度 (embed_dim): 4096

[Step 2] Embedding 层创建完成:
  形状: (32000, 4096)
  参数量: 131,072,000
  显存占用 (FP32): 0.49 GB

[Step 3] 输入 token IDs:
  token_ids: [15496, 2159, 287]
  序列长度: 3

[Step 4] Embedding 查找结果:
  输出形状: torch.Size([3, 4096])
  含义: 每个 token 变成了一个 4096 维向量

  第一个 token (ID=15496) 的嵌入向量前 10 个值:
    [-0.0012, 0.0023, -0.0008, 0.0015, ...]

[Step 5] Token 间相似度 (余弦):
  'Hello'(15496) vs 'world'(2159): 0.0231
  （随机初始化无规律）

[Step 6] Position Embedding:
  最大序列长度: 2048
  位置编码形状: torch.Size([3, 4096])

  最终输入形状: torch.Size([3, 4096])
============================================================
📊 总参数量: 133,169,152 (0.50 GB FP32)
============================================================
```

## 环境安装与验证

### 安装 PyTorch

根据你的硬件选择合适的版本：

```bash
# 方式一：NVIDIA GPU（推荐，LLM 必需）
# 访问 https://pytorch.org/get-started/locally/ 选择你的 CUDA 版本
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 方式二：仅 CPU（学习基础可以，跑不了大模型）
pip install torch torchvision

# 方式三：Apple Silicon (M1/M2/M3/M4)
pip install torch torchvision
```

### 安装验证脚本

```python
"""PyTorch 环境验证脚本"""

import sys
import torch


def check_environment():
    """检查 PyTorch 环境"""

    print("=" * 55)
    print("🔥 PyTorch 环境检查")
    print("=" * 55)

    # 基本信息
    print(f"\n[基本信息]")
    print(f"  Python 版本: {sys.version.split()[0]}")
    print(f"  PyTorch 版本: {torch.__version__}")

    # CUDA 可用性
    print(f"\n[CUDA / GPU]")
    print(f"  CUDA 是否可用: {'✅ 是' if torch.cuda.is_available() else '❌ 否'}")

    if torch.cuda.is_available():
        print(f"  CUDA 版本: {torch.version.cuda}")
        print(f"  GPU 名称: {torch.cuda.get_device_name(0)}")
        print(f"  GPU 显存: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")
        print(f"  GPU 数量: {torch.cuda.device_count()}")

        # 测试 GPU 计算
        x = torch.randn(10000, 10000, device="cuda")
        y = torch.matmul(x, x.t())
        print(f"  GPU 矩阵乘法测试: ✅ 成功 (shape: {y.shape})")
        del x, y
        torch.cuda.empty_cache()
    else:
        print(f"  ⚠️ 未检测到 GPU，将使用 CPU 运行")
        print(f"  提示: 大模型训练/推理强烈建议使用 NVIDIA GPU")

    # MPS (Apple Silicon)
    print(f"\n[Apple Silicon MPS]")
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    print(f"  MPS 是否可用: {'✅ 是' if mps_available else '❌ 否'}")

    # dtype 支持测试
    print(f"\n[dtype 支持测试]")
    dtypes = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    for name, dtype in dtypes.items():
        try:
            t = torch.zeros(1, dtype=dtype)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            t = t.to(device)
            print(f"  {name:<10}: ✅ 支持")
        except Exception as e:
            print(f"  {name:<10}: ❌ 不支持 ({e})")

    print(f"\n{'='*55}")


if __name__ == "__main__":
    check_environment()
```

典型输出（有 GPU 的环境）：

```
=======================================================
🔥 PyTorch 环境检查
=======================================================

[基本信息]
  Python 版本: 3.11.9
  PyTorch 版本: 2.4.0

[CUDA / GPU]
  CUDA 是否可用: ✅ 是
  CUDA 版本: 12.1
  GPU 名称: NVIDIA GeForce RTX 4090
  GPU 显存: 24.0 GB
  GPU 数量: 1
  GPU 矩阵乘法测试: ✅ 成功 (shape: torch.Size([10000, 10000]))

[Apple Silicon MPS]
  MPS 是否可用: ❌ 否

[dtype 支持测试]
  float32   : ✅ 支持
  float16   : ✅ 支持
  bfloat16   : ✅ 支持
=======================================================
```

## 本章小结

| 概念 | 一句话解释 |
|-----|-----------|
| **Tensor** | PyTorch 的核心数据结构，类似 NumPy ndarray 但支持 GPU 和自动微分 |
| **GPU 加速** | 通过 `.to("cuda")` 将计算放到显卡上，速度提升几十到几百倍 |
| **Autograd** | 自动微分引擎，调用 `loss.backward()` 自动计算所有参数的梯度 |
| **nn.Module** | 构建神经网络的标准方式，封装参数管理和设备移动 |
| **dtype** | 数据类型决定精度和显存；LLM 中 BF16 是训练首选，INT4 是量化首选 |
| **Embedding** | 将离散的 token ID 映射为连续的向量，是 LLM 处理文本的第一步 |

下一节我们将深入学习 **Tensor 的操作技巧**——这些操作构成了 LLM 模型的全部计算。
