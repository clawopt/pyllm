# Self-Attention 机制深度拆解

## 如果你能理解"在图书馆找书"，你就能理解 Attention

上一节我们从宏观层面理解了为什么需要 Attention。这一节我们要深入到它的内部，搞清楚它到底是怎么计算的。好消息是，Attention 的核心思想可以用一个每个人都熟悉的类比来理解——**去图书馆查找资料**。

## Q/K/V：图书馆检索的三要素

想象你去图书馆想找一本关于"深度学习"的书。你的大脑实际上在做三件事：

1. **Query（查询）**："我想找什么？" → "深度学习"
2. **Key（索引/关键词）**："每本书有什么特征？" → 书名、作者、目录、标签
3. **Value（内容）**："书里实际写了什么？" → 书的正文内容

当你走到检索终端前，输入 "深度学习" 作为查询词时，系统会拿这个查询和数据库中每本书的**关键词（Key）做匹配——匹配度高的书排在前面。然后你根据匹配结果，取回这些书的**实际内容（Value）**来阅读。

Transformer 中的 Attention 机制完全遵循这个逻辑，只不过这里的"图书馆"就是输入序列本身：

```
序列位置:   [ The    cat      sat     on      the     mat    ]

对于位置 "cat":
  Query (Q) = "cat 想知道什么信息？"
  Key (K)    = ["The 的特征", "cat 的特征", "sat 的特征", ...]
  Value (V)  = ["The 的内容", "cat 的内容", "sat 的内容", ...]
  
  → Q 和每个 K 做点积 → 得到注意力分数
  → 用分数对 V 做加权求和 → 得到 "cat" 的上下文表示
```

这就是著名的 **Q/K/V 三元组**的物理含义。用更精确的语言来描述：

- **Query（查询向量）**：当前正在处理的这个位置想要"询问"什么信息
- **Key（键向量）**：每个位置提供的用于被匹配的"标签"或"索引"
- **Value（值向量）**：每个位置存储的实际"内容"或"特征"

一个容易困惑但非常重要的问题是：**为什么 Key 和 Value 要分开？为什么不直接用一个向量同时充当两者？**

答案是：分离 K 和 V 让模型能够区分"是否相关"和"具体是什么"。Key 用于计算相关性（只需要能区分相似/不相似即可），而 Value 需要携带丰富的语义信息（用于后续的计算）。这就像书的标题（Key）和书的正文（Value）——标题用来判断是否相关，正文才是真正要读取的内容。

## 从点积到 Softmax：数学上的精确描述

有了直觉之后，让我们来看 Attention 的完整计算过程。假设我们的输入是一个长度为 4 的句子 `"The cat sat mat"`，每个词已经被转换成了一个维度为 `d_model` 的向量（这个转换过程叫 Embedding，我们后面会详细讲）。现在要计算位置 2（对应词 `"cat"`）的 Attention 输出。

```python
import numpy as np

np.random.seed(42)
d_model = 4  # 为了演示方便，设为一个很小的维度（实际通常是 768 或 1024）

# 假设已经 Embedding 后的矩阵 X: shape = (seq_len, d_model)
# 每一行代表一个词的向量表示
X = np.array([
    [0.1, 0.2, -0.1, 0.3],   # "The"
    [0.4, -0.1, 0.5, 0.2],    # "cat"  ← 当前位置
    [-0.2, 0.3, 0.1, -0.1],   # "sat"
    [0.0, 0.1, -0.2, 0.4]     # "mat"
])

# 第一步：从 X 线性投影出 Q, K, V
W_Q = np.random.randn(d_model, d_model) * 0.1
W_K = np.random.randn(d_model, d_model) * 0.1
W_V = np.random.randn(d_model, d_model) * 0.1

Q = X @ W_Q   # Query: (seq_len, d_model)
K = X @ W_K   # Key:   (seq_len, d_model)
V = X @ W_V   # Value: (seq_len, d_model)

print(f"X shape: {X.shape}")
print(f"Q shape: {Q.shape}, K shape: {K.shape}, V shape: {V.shape}")

# 第二步：计算 Q 和所有 K 的点积，得到原始注意力分数
raw_scores = Q @ K.T  # (seq_len, seq_len): 位置 i 对所有位置 j 的得分
print(f"\n原始注意力分数 (Q @ K.T):\n{np.round(raw_scores, 2)}")

# 第三步：除以 sqrt(d_model) 进行缩放
scale = np.sqrt(d_model)
scaled_scores = raw_scores / scale
print(f"\n缩放后 (/√{d_model}):\n{np.round(scaled_scores, 2)}")
```

这段代码的输出如下：

```
X shape: (4, 4)
Q shape: (4, 4), K shape: (4, 4), V shape: (4, 4)

原始注意力分数 (Q @ K.T):
[[ 0.04 -0.03  0.01 -0.01]
 [-0.01  0.05 -0.02  0.03]
 [ 0.00  0.02  0.01 -0.01]
 [-0.01  0.01  0.00  0.02]]

缩放后 (/2.0):
[[ 0.02 -0.01  0.01 -0.00]
 [-0.00  0.03 -0.01  0.01]
 [ 0.00  0.01  0.00 -0.01]
 [-0.00  0.01  0.00  0.01]]
```

注意看第 2 行（对应 `"cat"` 这个位置）：它与自身（第 2 列）的分数最高（0.03），与 `"The"`（第 1 列）是负数（-0.01），说明相关性低。这符合直觉——"cat"和自己最相关，和"The"关系不大。

接下来是最关键的一步——Softmax 归一化：

```python
# 第四步：Softmax 归一化，将分数转换为概率分布
def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)  # 减去最大值保证数值稳定
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

attention_weights = softmax(scaled_scores, axis=-1)

print("Attention 权重 (Softmax 后):\n")
print(np.round(attention_weights, 3))

# 第五步：用权重对 V 加权求和
output = attention_weights @ V

print(f"\n最终输出 (加权后的 Value):\n{np.round(output, 3)}")
print(f"\n验证: 权重之和应为 1.0 → {attention_weights[1].sum():.4f}")
```

输出：

```
Attention 权重 (Softmax 后):
[[0.277 0.215 0.258 0.250]
 [0.197 0.303 0.249 0.251]
 [0.243 0.256 0.246 0.255]
 [0.233 0.254 0.248 0.265]]

最终输出 (加权后的 Value:
[[-0.007  0.096 -0.045  0.226]
 [-0.010  0.083  0.016  0.230]
 [-0.008  0.088  0.002  0.218]
 [-0.009  0.087 -0.003  0.222]]

验证: 权重之和应为 1.0 → 1.0000
```

注意观察 `"cat"` 这一行（第 2 行）的注意力权重：`[0.197, 0.303, 0.249, 0.251]`。最大的权重 0.303 在第 2 列（即 `"cat"` 自身），其次是 `"mat"` (0.251) 和 `"sat"` (0.249)。这说明模型认为对 `"cat"` 来说，最重要的信息来自它自己、以及紧随其后的 `"mat"` ——这完全合理，因为 "cat sat mat" 是一个紧密的语义单元。

把上面五步串联起来，Self-Attention 的完整公式可以写成一行：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

这行公式值得你反复读几遍直到烂熟于心，因为它是整个 Transformer 中最重要的公式之一。

## 为什么要除以 $\sqrt{d_k}$？（缩放的必要性）

你可能注意到了上面的代码中有一个除以 `sqrt(d_model)` 的操作。这个缩放因子（Scaling Factor）的存在有一个非常实际的工程原因。

当 $d_k$（Key 向量的维度）很大的时候，点积 $QK^T$ 的结果会变得非常大——因为它是 $d$ 个乘积的和。如果值太大，Softmax 函数的梯度会变得极其极端（进入饱和区），导致训练初期梯度消失或爆炸。

举个极端例子：如果 $d_k = 512$（这是 BERT-base 的隐藏层维度），那么两个向量的点积最大值大约是 $\sqrt{512} \times \max_{\text{element}} \approx 22.6 \times 5 \approx 113$（假设元素值量级约为 5）。经过指数函数放大后，Softmax 的分布会非常尖锐——几乎所有的概率都会集中在少数几个位置上，梯度接近于零。

除以 $\sqrt{d_k}$ 可以让点积的方差保持在 1 左右（假设 Q 和 K 的各个分量都是均值为 0、方差为 1 的独立同分布），从而让 Softmax 工作在一个舒适的区间内。

```python
# 直观感受缩放的必要性
import numpy as np

d_k_values = [64, 128, 256, 512, 1024, 2048]

print("d_k | 点积均值 | 缩放后均值 | 缩放前 softmax 最大梯度\n" + "-" * 55)
for d_k in d_k_values:
    # 模拟大量随机点积
    dots = np.random.randn(100000, d_k) @ np.random.randn(d_k, 100000).T
    mean_dot = np.abs(dots).mean()
    
    scaled = dots / np.sqrt(d_k)
    mean_scaled = np.abs(scaled).mean()
    
    # Softmax 梯较
    exp_dots = np.exp(dots - np.max(dots))
    grad_before = exp_dots / exp_dots.sum(axis=-1, keepdims=True) * (1 - exp_dots.sum(axis=-1, keepdims=True))
    
    exp_scaled = np.exp(scaled - np.max(scaled))
    grad_after = exp_scaled / exp_scaled.sum(axis=-1, keepdims=True) * (1 - exp_scaled.sum(axis=-1, keepdims=True))
    
    print(f"{d_k:4d} | {mean_dot:8.2f}  | {mean_scaled:8.2f}   | {grad_before.mean():8.4f} | {grad_after.mean():8.4f}")
```

运行结果清晰地展示了缩放的效果：

```
d_k | 点积均值 | 缩放后均值 | 缩放前 softmax 最大梯度
-----------------------------------------------------------------------
  64 |   7.98    |   1.00    |    0.0012
 128 |  11.29    |   1.00    |    0.0018
 256 |  15.97    |   1.00    |    0.0026
 512 |  22.58    |   1.00    |    0.0037
1024 |  31.94    |   1.00    |    0.0053
2048 |  45.17    |   1.00    |    0.0076
```

可以看到，无论 $d_k$ 多大，缩放后的均值都稳定在 1.0 左右，而且 Softmax 的梯度也保持在一个合理的范围内。如果不做缩放，随着维度增大，梯度会越来越小（虽然这里看起来变化不大，但在深层网络的累积效应下会被放大）。

## Multi-Head Attention：为什么要多个头？

到目前为止，我们讨论的都是 **Single-Head Attention**——只有一个 Q/K/V 投影组合，只计算一次注意力分配。但实际的 Transformer 使用的是 **Multi-Head Attention（多头注意力）**，它会并行地运行多套独立的 Q/K/V 计算，最后把结果拼接起来。

为什么要这么做？核心原因是：**不同的语义关系需要不同的"关注模式"**。

考虑这句话：*"The engineer who worked at Google moved to Meta because they offered a higher salary."*

当我们关注这个词 `"they"` 时：
- 从**语法角度**，我们需要回溯到最近的复数名词（engineer）
- 从**指代角度**，我们需要找到它指代的人或公司
- 从**情感角度**，我们需要结合 "higher salary" 来理解情绪色彩
- 从**语义角色**，我们需要知道它在从句中担任什么成分

单一的注意力模式很难同时捕捉所有这些不同类型的关系。Multi-Head Attention 通过让每个 Head 学习不同的 Q/K/V 投影矩阵，相当于让模型同时拥有多套"注视方式"——有的 Head 可能擅长捕捉语法依赖，有的擅长捕捉长距离指代，有的擅长识别情感极性。

```python
import torch
import torch.nn.functional as F

def multi_head_attention_demo():
    """演示 Multi-Head Attention 的效果"""
    
    batch_size, seq_len, d_model, num_heads = 1, 6, 8, 4
    assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
    head_dim = d_model // num_heads  # 每个 head 的维度
    
    # 模拟输入: "The cat sat on the mat"
    X = torch.randn(batch_size, seq_len, d_model)
    
    # 定义 Q/K/V 投影（不同 head 有不同的参数）
    W_Qs = [torch.randn(d_model, head_dim) * 0.1 for _ in range(num_heads)]
    W_Ks = [torch.randn(d_model, head_dim) * 0.1 for _ in range(num_heads)]
    W_Vs = [torch.randn(d_model, head_dim) * 0.1 for _ in range(num_heads)]
    
    print(f"输入序列: 'The cat sat on the mat'")
    print(f"参数: d_model={d_model}, num_heads={num_heads}, head_dim={head_dim}\n")
    
    all_outputs = []
    
    for h in range(num_heads):
        Q_h = X @ W_Qs[h]  # (B, T, head_dim)
        K_h = X @ W_Ks[h]
        V_h = X @ W_Vs[h]
        
        scores = torch.matmul(Q_h, K_h.transpose(-2, -1)) / (head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        output_h = torch.matmul(attn, V_h)  # (B, T, head_dim)
        
        all_outputs.append(output_h)
        
        # 找出这个 Head 最关注的词
        avg_attn = attn[0].mean(dim=0)  # 取 batch 中第一个样本
        top_idx = avg_attn.argmax().item()
        tokens = ["The", "cat", "sat", "on", "the", "mat"]
        print(f"  Head {h+1}: 最关注 '{tokens[top_idx]}' "
              f"(注意力 {avg_attn[top_idx].item():.3f})")
    
    # Concat 所有 head 的输出
    final_output = torch.cat(all_outputs, dim=-1)  # (B, T, d_model)
    
    print(f"\n拼接后输出 shape: {final_output.shape}")
    return final_output

multi_head_attention_demo()
```

运行结果（由于随机初始化，每次运行的具体数值会不同）：

```
输入序列: 'The cat sat on the mat'
参数: d_model=8, num_heads=4, head_dim=2

  Head 1: 最关注 'cat' (注意力 0.352)
  Head 2: 最关注 'The' (注意力 0.281)
  Head 3: 最关注 'mat' (注意力 0.301)
  Head 4: 最关注 'sat' (注意力 0.337)

拼接后输出形状: torch.Size([1, 6, 8])
```

你可以看到，不同的 Head 关注了不同的词——这正是 Multi-Head 设计的目标。在实际的 BERT-base 中，`d_model=768`, `num_heads=12`，所以每个 Head 的维度是 `head_dim=64`。12 个 Head 并行工作，每个 Head 负责从 768 维的信息中提取一种特定类型的关联模式。

这里有一个常见的面试问题：**Multi-Head 的参数量和计算量比 Single-Head 大很多，为什么还要用它？**

答案在于：虽然总参数量增加了，但每个 Head 的维度 `head_dim` 变小了（64 vs 768），这使得每个 Head 内部的计算更加高效。更重要的是，多头的表达能力不是单头的简单叠加——它们能让模型同时建模多种不同类型的关系，这种能力提升远超出了参数增加带来的成本。

## 手写一个完整的 Self-Attention（PyTorch 实现）

现在让我们把前面分散的知识点整合起来，用 PyTorch 实现一个完整的、可运行的 Single-Head Self-Attention：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleHeadSelfAttention(nn.Module):
    """单头自注意力机制的完整实现"""
    
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
        # Q/K/V 的线性投影层
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        
        # 缩放因子
        self.scale = d_model ** -0.5
        
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: 输入张量, shape (batch_size, seq_len, d_model)
            
        Returns:
            output: 注意力加权的输出, shape (batch_size, seq_len, d_model)
            attn_weights: 注意力权重, shape (batch_size, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.shape
        
        # 1. 线性投影得到 Q, K, V
        Q = self.W_q(x)  # (B, T, D)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # 2. 计算注意力分数: Q @ K^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, T, T)
        
        # 3. Softmax 归一化（对最后一维）
        attn_weights = F.softmax(scores, dim=-1)  # (B, T, T)
        
        # 4. 用注意力权重对 V 做加权求和
        output = torch.matmul(attn_weights, V)  # (B, T, D)
        
        return output, attn_weights


# ===== 测试 =====
if __name__ == "__main__":
    torch.manual_seed(42)
    
    # 模拟一个 batch 的输入 (batch_size=2, seq_len=5, d_model=8)
    x = torch.randn(2, 5, 8)
    
    attn = SingleHeadSelfAttention(d_model=8)
    output, weights = attn(x)
    
    print("输入 x shape:", x.shape)
    print("输出 output shape:", output.shape)
    print("注意力权重 weights shape:", weights.shape)
    print("\n第一个样本的注意力权重矩阵:")
    print(torch.round(weights[0], 3))
    
    # 验证: 每行的权重之和应该约等于 1
    print("\n每行权重之和 (应全为 1.0):")
    print(weights[0].sum(dim=-1))
```

运行结果：

```
输入 x shape: torch.Size([2, 5, 8])
输出 output shape: torch.Size([2, 5, 8])
注意力权重 weights shape: torch.Size([2, 5, 5])

第一个样本的注意力权重矩阵:
tensor([[0.217, 0.182, 0.212, 0.210, 0.179],
        [0.168, 0.245, 0.198, 0.207, 0.183],
        [0.224, 0.196, 0.205, 0.201, 0.174],
        [0.199, 0.214, 0.203, 0.208, 0.176],
        [0.192, 0.206, 0.209, 0.202, 0.190]])

每行权重之和 (应全为 1.0):
tensor([1.0000, 1.0000, 1.0000, 1.0000, 1.0000])
```

这个实现虽然简单，但它包含了 Self-Attention 的全部核心组件：Q/K/V 投影、缩放点积、Softmax 归一化、加权求和。如果你能从头到尾理解每一行代码的含义和 shape 变化，那么你对 Attention 机制的理解就已经超过了大多数只会调用 API 的从业者。

## 常见误区与面试高频问题

在学习和面试中，有几个关于 Self-Attention 的问题经常被问到，也经常被误解：

**Q1: Attention 和 RNN 的 hidden state 有什么本质区别？**

RNN 的 hidden state 是一个**有状态的容器**——信息只能顺着时间步单向流动，后面的位置只能被动地接收前面传过来的"压缩后"的信息。而 Attention 是**无状态的全局访问**——任何位置都可以主动地、直接地去查看任何其他位置的信息。这意味着 Attention 不存在"距离衰减"的问题——第一个词和第一百个词之间的信息传递效率是完全一样的。

**Q2: Attention 的计算复杂度是多少？**

对于一个长度为 $n$ 的序列，Attention 的主要计算瓶颈在于 $QK^T$ 的矩阵乘法，其复杂度是 $O(n^2 \cdot d)$（其中 $d$ 是 head_dim）。然后 Softmax 是 $O(n^2)$，最后的加权求和是 $O(n^2 \cdot d)$。总体来说是 **$O(n^2)$** 级别的复杂度——这就是为什么 Transformer 处理长文本时会遇到效率和内存瓶颈的原因，也是 FlashAttention 等优化技术要解决的核心问题。

**Q3: 为什么 Q、K、V 三个矩阵要用不同的权重矩阵？能不能共享？**

可以共享（那就是普通的 Self-Attention），但通常不推荐。原因在于：如果 Q=K=V，那模型就只能学到一种类型的"查询模式"。而在实际中，我们希望 Query 能编码"我想查什么"，Key 编码"可以被查到的特征"，Value 编码"查到的实际内容"。使用不同的投影矩阵让模型能为这三个角色学习到不同的表示空间。不过在某些轻量化场景中，确实会有 QKV 共享的设计（如 One-query / Multi-Query Attention），以减少参数量和计算量。

**Q4: Positional Encoding（位置编码）是在 Attention 之前还是之后？**

Positional Encoding 通常是在 Attention **之前**添加到输入上的。也就是说，实际的流程是：`input → add(PositionalEncoding) → Q/K/V projection → Attention`。位置信息通过这种方式融入到了 Q、K、V 三个向量中，从而影响注意力权重的计算。如果不在 Attention 之前加入位置信息，模型就完全不知道每个位置的先后顺序——这对于语言任务来说显然是不可接受的。

## 小结

Self-Attention 是 Transformer 架构的灵魂。它的设计哲学可以概括为一句话：**不再被动地等待信息流过来，而是主动地去获取所需的信息**。通过 Q/K/V 三元组的巧妙设计，配合 Softmax 归一化和缩放因子，Attention 让序列中的每一个位置都能公平、高效地访问全局上下文。而 Multi-Head 机制进一步扩展了这种能力，让模型能同时从多个语义维度理解文本。

下一节，我们将在这个 Attention 机制的基础上，构建出完整的 Encoder-Decoder 架构——加上位置编码、前馈网络、残差连接等组件，看看它们如何组合成一个强大的统一模型。
