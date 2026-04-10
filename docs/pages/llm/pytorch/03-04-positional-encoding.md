# 3.4 Positional Encoding——给 Transformer 位置感

在前面三节中，我们实现了 Self-Attention 的核心机制、多头注意力的高效实现方式，以及包含 Attention 和 FFN 的完整 Transformer Block。但如果你仔细思考一下，会发现有一个根本性的问题被回避了：**Self-Attention 是排列不变（Permutation Invariant）的**。这是什么意思？假设输入序列是三个词向量 $[v_1, v_2, v_3]$，经过 Attention 层后得到 $[o_1, o_2, o_3]$；如果我们把输入顺序打乱成 $[v_3, v_1, v_2]$，Attention 的输出会变成 $[o_3, o_1, o_2]$ —— 每个位置的输出只做了对应的排列变换，$o_i$ 和 $o_j$ 之间的关系完全没有变化。换句话说，纯 Attention 不关心"谁在第几个位置"，它只关心"哪些元素之间有多强的关联"。这对于理解"我爱你"和"你爱我"这样语序不同但含义完全不同的句子来说是一个致命的缺陷——模型必须知道每个 token 出现在序列中的哪个位置，才能正确地理解语法结构和语义关系。

解决这个问题的方法是在输入中显式地加入位置信息，这就是**位置编码（Positional Encoding, PE）**的核心思想。从原始 Transformer 论文到现在最先进的 LLM，位置编码方案经历了多次演进，每一种新方案都在试图解决前一种方案的某种局限性。这一节我们将系统地学习三种最重要的位置编码方案：绝对位置编码（Sinusoidal / Learned）、旋转位置编码（RoPE）——这是当前现代 LLM 的事实标准，以及 ALiBi 作为替代方案的思路。

## 方案一：绝对位置编码

最直观的想法是给每个位置分配一个唯一的向量，然后把位置向量和词嵌入向量相加（或拼接）。原始 Transformer 论文提出了两种做法：

**Sinusoidal（正弦）位置编码**使用预定义的数学公式生成位置向量，不需要学习：

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

其中 `pos` 是位置索引（0, 1, 2, ...），`i` 是维度索引（0, 1, ..., d/2-1），`d` 是总维度。这个公式的巧妙之处在于它让每个位置都有一个唯一的编码，而且不同位置的编码之间有规律可循——相邻位置的编码相似度高（因为 sin/cos 在小角度变化时平滑过渡），远距离位置的编码差异大。更重要的是，对于任意固定的偏移量 k，$PE(pos+k)$ 可以表示为 $PE(pos)$ 的线性函数，这意味着模型有可能学习到相对位置关系。

```python
class SinusoidalPositionalEncoding(nn.Module):
    """正弦位置编码 — 原始 Transformer 采用"""

    def __init__(self, max_seq_len: int, embed_dim: int):
        super().__init__()
        pe = torch.zeros(max_seq_len, embed_dim)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        T = x.size(1)
        return x + self.pe[:, :T]
```

**Learned（学习式）位置编码**则更简单粗暴：直接把位置编码当作一个可学习的参数矩阵（本质上就是一个 nn.Embedding 层），让模型自己训练时学习出合适的位置表示。GPT-2 采用的就是这种方式。

```python
class LearnedPositionalEncoding(nn.Module):
    """学习式位置编码 — GPT-2 采用"""

    def __init__(self, max_seq_len: int, embed_dim: int):
        super().__init__()
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)

    def forward(self, x):
        B, T, C = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0)
        return x + self.pos_embed(positions)
```

这两种绝对位置编码方案在实际中工作得不错，但它们有一个共同的严重缺陷：**外推性差（Poor Extrapolation）**。如果模型在训练时最多见过长度为 1024 的序列，那它在推理时处理长度为 2048 或 4096 的序列时效果通常会显著下降。原因是位置 1025 及之后的编码在训练期间从未出现过，模型不知道如何解释这些从未见过的位置信息。对于 Sinusoidal 编码来说，虽然理论上可以外推到任意长度，但高频分量在外推区域的行为不可控；对于 Learned 编码来说就更直接了——超出训练范围的位置根本没有对应的 embedding 参数。

## 方案二：RoPE——旋转位置编码

为了从根本上解决外推性问题，Su 等人在 2021 年提出了 **RoPE（Rotary Positional Encoding，旋转位置编码）**，这个方案很快成为了现代 LLM 的事实标准——LLaMA、Qwen、Mistral、PaLM、Claude 等几乎所有主流模型都采用了 RoPE 或其变体。

RoPE 的核心思想非常优雅：与其把位置信息作为一个独立的向量加到输入上，不如把位置信息编码到 Query 和 Key 向量的**相对旋转**中。具体来说，RoPE 把一个二维向量 $(x_i, x_{i+1})$ 看作复平面上的一个复数 $z = x_i + i \cdot x_{i+1}$，然后用乘以复数 $e^{i \cdot m \theta}$ 来实现旋转——其中 $m$ 是位置索引，$\theta$ 是固定的角度频率。复数乘法的几何意义就是旋转：乘以 $e^{i\phi}$ 等于将向量逆时针旋转 $\phi$ 角度。

对于 d 维的向量，我们把每两个连续的维度组成一对，分别做旋转操作：

$$\begin{bmatrix} x'_m \\ x'_{m+1} \end{bmatrix} =
\begin{bmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{bmatrix}
\begin{bmatrix} x_m \\ x_{m+1} \end{bmatrix}$$

经过 RoPE 变换后，位置 m 处的 Query 和位置 n 处的 Key 的点积可以分解出一个只依赖于相对位置 (m-n) 的项。这就是 RoPE 的关键性质——**注意力得分天然编码了相对位置信息，而不需要额外的绝对位置嵌入**。这也意味着 RoPE 具有一定的长度外推能力：即使测试时的序列长度超过了训练时的最大长度，只要相对位置关系在训练范围内出现过，模型就能合理地处理。

下面是 RoPE 的完整实现：

```python
import math


class RotaryEmbedding(nn.Module):
    """旋转位置编码 (RoPE) — LLaMA/Qwen/Mistral 标准"""

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0

        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2).float() / dim)
        )
        self.register_buffer('inv_freq', inv_freq)

        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)
        self.register_buffer('cos_cached', freqs.cos())
        self.register_buffer('sin_cached', freqs.sin())

    def forward(self, x):
        seq_len = x.shape[-2]
        cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)

        x1 = x[..., ::2]
        x2 = x[..., 1::2]

        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos

        out = torch.stack([out1, out2], dim=-1).flatten(-2)
        return out
```

让我们逐步拆解这个实现。首先 `inv_freq` 计算的是频率倒数序列：`[1/(base^(0/d)), 1/(base^(2/d)), 1/(base^(4/d)), ...]`。然后 `torch.outer(t, inv_freq)` 生成一个 `(max_seq_len, dim/2)` 的矩阵，其中 `[m][i]` 位置存储的就是第 m 个位置在第 i 对维度上的旋转角度 `m × θ_i`。对这个角度矩阵取 cos 和 sin 并缓存起来，前向传播时只需要查表即可。

在 forward 方法中，输入 x 的 shape 假设为 `(B, T, head_dim)`（head_dim 必须是偶数）。`x[..., ::2]` 取出所有偶数索引维度的值（即每对的第一个元素），`x[..., 1::2]` 取出奇数索引维度的值（每对的第二个元素）。然后应用二维旋转公式：
- 新的偶数位 = 原偶数位 × cos - 原奇数位 × sin
- 新的奇数位 = 原偶数位 × sin + 原奇数位 × cos

最后用 `torch.stack` 把两半交错拼回去，恢复成原始维度顺序。

```python
def demo_rope():
    dim = 8
    rope = RotaryEmbedding(dim=dim, max_seq_len=16)

    x = torch.randn(1, 4, dim)
    print(f"Input (pos 0): {x[0, 0].round(3).tolist()}")

    x_rotated = rope(x)
    print(f"After RoPE (pos 0): {x_rotated[0, 0].round(3).tolist()}")

    norm_before = x[0, 0].norm().item()
    norm_after = x_rotated[0, 0].norm().item()
    print(f"L2 norm before: {norm_before:.4f}, after: {norm_after:.4f}")
    print("(Should be equal — rotation preserves length)")

    diff_pos01 = ((x_rotated[0, 1] - x_rotated[0, 0]).abs().sum().item())
    diff_pos03 = ((x_rotated[0, 3] - x_rotated[0, 0]).abs().sum().item())
    print(f"|rope(x₁) - rope(x₀)| = {diff_pos01:.4f}")
    print(f"|rope(x₃) - rope(x₀)| = {diff_pos03:.4f}")
    print("(Farther positions → larger difference → relative info encoded)")


demo_rope()
```

运行上面的代码你会验证两个重要性质：第一，RoPE 是保范变换（rotation 不改变向量的长度）；第二，距离越远的位置经过 RoPE 后差异越大，说明相对位置信息确实被编码进了表示中。

## 如何将 RoPE 集成到 Multi-Head Attention 中

有了 RotaryEmbedding 模块之后，我们需要把它集成到之前实现的 MultiHeadAttention 中。具体来说，RoPE 应该应用到 Q 和 K 上（而不是 V），而且是在 Q/K 被投影出来之后、计算 attention score 之前。

```python
class MultiHeadAttentionWithRoPE(nn.Module):
    """集成 RoPE 的多头注意力"""

    def __init__(self, n_embed: int, num_heads: int, max_seq_len: int = 2048):
        super().__init__()
        assert n_embed % num_heads == 0

        self.n_embed = n_embed
        self.num_heads = num_heads
        self.head_dim = n_embed // num_heads

        self.qkv = nn.Linear(n_embed, 3 * n_embed, bias=False)
        self.proj = nn.Linear(n_embed, n_embed)

        self.rope = RotaryEmbedding(dim=self.head_dim, max_seq_len=max_seq_len)
        self.scale = self.head_dim ** -0.5

    def forward(self, x, mask=None):
        B, T, C = x.shape

        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q = self.rope(q.reshape(B * self.num_heads, T, self.head_dim))
        k = self.rope(k.reshape(B * self.num_heads, T, self.head_dim))

        q = q.view(B, self.num_heads, T, self.head_dim)
        k = k.view(B, self.num_heads, T, self.head_dim)

        attn_scores = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)

        out = (attn_weights @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out), attn_weights
```

关键的改动只有几行：在 Q/K 投影之后、reshape 为多头格式之后，调用 `self.rope(q)` 和 `self.rope(k)` 应用旋转变换。V 不需要 RoPE，因为 Value 只负责承载信息内容，不参与相似度计算。

## 方案三：ALiBi——另一种思路

除了 RoPE 之外，还有一种值得了解的位置编码方案叫 **ALiBi（Attention with Linear Biases，带线性偏置的注意力）**，由 Press 等人在 2021 年提出。ALiBi 的思路与 RoPE 完全不同：它不在输入端修改 Q/K/V，而是在 Attention score 矩阵上加一个与距离成正比的静态偏置。

具体做法是：定义一组斜率 $s = \{-m/(2H), -(m-1)/(2H), ..., m/(2H)\}$（其中 H 是头数），然后对每个 head h，将注意力分数矩阵加上偏置 $-s_h \times |i-j|$（i 和 j 是位置索引）。也就是说，距离越远的位置对之间的注意力分数被惩罚得越多（减去更大的值），而距离越近的惩罚越小。这种简单的线性偏置策略带来了一个令人惊讶的好处：**ALiBi 天然支持任意长度的外推**——因为它不依赖任何预计算的固定长度位置表，偏置值完全由相对距离决定，无论序列多长都能正确计算。

```python
def create_alibi_bias(num_heads: int, seq_len: int):
    """创建 ALiBi 偏置矩阵"""
    slopes = 1.0 / (2 ** (torch.arange(1, num_heads + 1).float()))
    slopes = slopes * -1

    positions = torch.arange(seq_len).float()
    relative_dist = torch.abs(positions.unsqueeze(0) - positions.unsqueeze(1))

    bias = slopes.unsqueeze(1).unsqueeze(2) * relative_dist.unsqueeze(0)
    return bias
```

ALiBi 的优势在于简洁性和外推能力，但它的缺点是对短距离建模不够精细（只有线性的距离衰减，没有 RoPE 那种丰富的旋转模式）。目前 ALiBi 主要被 BLOOM 等模型采用，而在 decoder-only LLM 领域 RoPE 仍然是主流选择。

## 位置编码选型总结

到这里，三种主要的位置编码方案都介绍完了。用一个对比表格来总结它们的特性：

| 特性 | Sinusoidal | Learned | **RoPE** | ALiBi |
|-----|-----------|---------|----------|-------|
| 外推能力 | 弱 | 无 | **中等-强** | **极强** |
| 参数开销 | 无（公式计算） | O(max_len × d) | 无（公式计算） | 无 |
| 相对位置 | 隐式（需学习） | 无 | **显式（天然）** | **显式** |
| 主流采用 | 原始 Transformer | GPT-2 | **LLaMA/Qwen/Mistral** | BLOOM |

对于你自己的项目，如果你的目标是构建一个能处理超长上下文（比如 32K+ tokens）的模型，RoPE 配合 NTK-aware 缩放或者 YaRN 等扩展技术是目前社区验证最多的路线；如果你追求极致的外推性能且不太关心短距离精度，ALiBi 值得一试；如果你在做快速实验且序列长度不超过训练最大长度，Learned PE 最简单直接。

下一节我们将把所有组件——Token Embedding、RoPE（或 Learned PE）、N 个 Transformer Block、最终的 LayerNorm、以及 LM Head——组装成一个完整的 GPT 模型，并逐行追踪一次完整的前向传播过程。
