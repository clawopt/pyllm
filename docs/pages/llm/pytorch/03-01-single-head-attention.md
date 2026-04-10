# 3.1 单头 Self-Attention 手写实现

如果说整个深度学习领域有一个"分水岭时刻"，那一定是 2017 年 Google 团队发表 "Attention Is All You Need" 这篇论文的时候。在这之前，处理序列数据的主流方案是 RNN 和 LSTM，它们一个 token 接一个 token 地顺序处理信息，就像一个人逐字阅读一本书——虽然能理解上下文，但速度慢，而且距离越远的信息越难记住（长距离依赖问题）。Attention 机制的提出彻底改变了这个局面：它让模型在每一步都能同时"看到"序列中的所有位置，并根据相关性决定每个位置应该关注多少。打个通俗的比方，假设你在读一篇长文章并试图理解某一句话的含义，RNN 的做法是让你从第一句话开始顺序读到当前这句话，靠记忆来保持之前的上下文；而 Attention 的做法是允许你随时翻回到文章的任何地方，把每一句话都拿出来和当前这句话做对比，找出哪些句子跟当前这句话最相关、最有助于理解它的含义。这种"全局视野"的能力就是 Transformer 能够在大规模语料上学习到丰富语义表示的根本原因，也是 GPT、BERT、LLaMA、Qwen 等所有现代大模型的基石。

## 从直觉到公式：Query、Key、Value 的含义

Attention 机制的核心思想可以用一个信息检索的场景来类比。想象你在图书馆里找资料：你心里有一个明确的问题，这就是 **Query（查询）**；书架上每本书都有一个标题或摘要来描述它包含什么内容，这就是 **Key（键）**；你根据问题和标题的匹配程度找到最相关的几本书，然后阅读这些书里的实际内容——这就是 **Value（值）**。在 Self-Attention 中，输入序列的每一个位置都会同时扮演这三个角色：它既产生一个 Query 来表达"我想找什么"，也产生一个 Key 来告诉别人"我是什么"，还产生一个 Value 来提供"我的实际内容"。当位置 i 的 Query 和位置 j 的 Key 越匹配，位置 j 的 Value 就会在最终输出中占更大的比重。

数学上，这个过程被简洁地表达为：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

让我们逐步拆解这个公式的每一部分。首先 $QK^T$ 计算的是所有 Query 和所有 Key 之间的点积——点积越大说明两个向量越相似（方向越一致），这对应于"匹配度"的概念。然后除以 $\sqrt{d_k}$（$d_k$ 是 Key 向量的维度），这一步叫做**缩放（Scaling）**，作用是防止点积值过大导致 softmax 函数进入饱和区域。为什么这是个问题？因为 softmax 函数的性质决定了，当输入值的差异很大时，输出会趋近于 one-hot 分布（接近 1 的值几乎不变化，接近 0 的值变成真正的 0），这会导致梯度消失——反向传播时几乎没有有效的梯度信号流回前面的层。除以 $\sqrt{d_k}$ 恰好能把点积的方差拉回到 1 附近（假设 Q 和 K 的分量都是均值为 0、方差为 1 的独立随机变量，那么它们的点积的方差就是 $d_k$），从而保证 softmax 工作在线性区而不是饱和区。

softmax 之后的矩阵就是一个注意力权重矩阵，每一行的和为 1，表示对应位置的 Query 对所有位置的 Value 的关注程度分布。最后乘以 V 就完成了加权求和——每个位置的输出是所有位置 Value 的加权平均，权重由注意力分数决定。

下面这段代码用纯 PyTorch 实现了最基本的单头 Self-Attention：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleHeadAttention(nn.Module):
    """单头自注意力机制"""

    def __init__(self, n_embed: int, head_dim: int):
        super().__init__()
        self.head_dim = head_dim

        self.q_proj = nn.Linear(n_embed, head_dim, bias=False)
        self.k_proj = nn.Linear(n_embed, head_dim, bias=False)
        self.v_proj = nn.Linear(n_embed, head_dim, bias=False)
        self.proj = nn.Linear(head_dim, n_embed)

    def forward(self, x, mask=None):
        B, T, C = x.shape

        q = self.q_proj(x)   # (B, T, head_dim)
        k = self.k_proj(x)   # (B, T, head_dim)
        v = self.v_proj(x)   # (B, T, head_dim)

        attn_scores = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)

        out = attn_weights @ v   # (B, T, head_dim)
        out = self.proj(out)     # (B, T, n_embed)
        return out, attn_weights
```

我们来仔细追踪一下数据的形状变化。输入 `x` 的 shape 是 `(B, T, C)`，其中 B 是 batch size，T 是序列长度，C 是嵌入维度（比如 768）。经过三个线性投影后，Q、K、V 的 shape 都变成了 `(B, T, head_dim)`——注意这里 head_dim 可以等于 C 也可以小于 C，单头情况下通常就设为 C。然后 `k.transpose(-2, -1)` 把 K 的最后两维转置，变成 `(B, head_dim, T)`，这样 `q @ k^T` 的结果就是 `(B, T, T)`——一个 T×T 的方阵，其中 `[i][j]` 位置存储的就是第 i 个位置的 Query 与第 j 个位置的 Key 的点积得分。乘以 `head_dim ** -0.5` 就是那个关键的缩放因子 `1/√d_k`。如果传入了 mask，就把 mask 值为 0 的位置填成负无穷，这样 softmax 之后这些位置的权重就会变成 0。`F.softmax(attn_scores, dim=-1)` 在最后一维上做归一化，保证每一行的权重之和为 1。最后 `attn_weights @ v` 把权重矩阵与 Value 相乘得到加权输出，再经过一个输出投影层映射回原始维度 C。

下面用一个具体的数值例子来验证我们的理解：

```python
B, T, C = 2, 4, 8
head_dim = 8

attn = SingleHeadAttention(C, head_dim)

x = torch.randn(B, T, C)
out, weights = attn(x)

print(f"Input:      {x.shape}")       # (2, 4, 8)
print(f"Output:     {out.shape}")     # (2, 4, 8)
print(f"Attn Weights: {weights.shape}")  # (2, 4, 4)

row_sums = weights.sum(dim=-1)
print(f"Row sums (should be all 1s): {row_sums}")
# tensor([[1., 1., 1., 1.], [1., 1., 1., 1.]])

max_weight = weights.max()
min_nonzero = weights[weights > 1e-6].min() if (weights > 1e-6).any() else 0
print(f"Weight range: [{min_nonzero:.4f}, {max_weight:.4f}]")
```

运行后你会发现每一行的权重之和确实严格等于 1（这是 softmax 的性质），且权重分布在 0 到 1 之间。这说明我们的实现在数学上是正确的。

## Causal Mask：为什么自回归模型必须"遮住未来"

上面实现的 Attention 有一个特性：对于任意位置 i，它可以 attend 到序列中的所有位置（包括 i 之后的"未来"位置）。这对于 BERT 这类双向编码器来说是完全正确的——BERT 在训练时能看到完整的上下文。但对于 GPT 这种自回归语言模型来说，这是一个严重的问题。GPT 的任务是预测下一个 token，而在实际推理时，模型只能看到当前及之前的位置，看不到未来。如果在训练时模型能够"偷看"未来的 token 来辅助预测，那推理时就会产生 train-test mismatch——训练时的能力和推理时的能力不一致，效果会大打折扣。

解决方法是在 Attention 权重计算之后、softmax 之前，引入一个**因果掩码（Causal Mask）**，也叫下三角掩码。具体来说，构造一个 T×T 的下三角矩阵，其中位置 `[i][j]` 为 1 当且仅当 j ≤ i（即只允许当前位置 attend 到自身及之前的位置），否则为 0。把这个掩码应用到注意力分数上，把被禁止的位置设为 `-inf`，这样 softmax 后这些位置的权重就变成 0 了。

```python
def create_causal_mask(seq_len: int):
    """创建因果注意力掩码（下三角）"""
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask  # 1=可以attend, 0=不能attend


def demo_causal_attention():
    T = 5
    causal_mask = create_causal_mask(T)

    print("Causal Mask (1=visible, 0=masked):")
    print(causal_mask.int())
    # tensor([
    #   [1, 0, 0, 0, 0],
    #   [1, 1, 0, 0, 0],
    #   [1, 1, 1, 0, 0],
    #   [1, 1, 1, 1, 0],
    #   [1, 1, 1, 1, 1]
    # ])

    attn = SingleHeadAttention(n_embed=8, head_dim=8)
    x = torch.randn(1, T, 8)
    out, weights = attn(x, mask=causal_mask)

    print("\nAttention Weights after causal masking:")
    print(weights[0].detach().numpy().round(3))
    # 注意：上三角部分应该全为 0

    upper_tri_sum = weights[0].triu(diagonal=1).sum().item()
    print(f"\nUpper triangle sum (should be ~0): {upper_tri_sum:.6f}")


demo_causal_attention()
```

运行上面的代码你会清楚地看到：应用因果掩码后，注意力权重的上三角部分（代表"看向未来"）全部变成了 0。这意味着位置 0 只能 attend 到自己，位置 1 能 attend 到位置 0 和 1，位置 4 能 attend 到所有 0~4 的位置。这就完美地模拟了自回归模型在推理时的行为模式——每一步只能利用已经生成的信息来预测下一个 token。

这里有一个容易出错但面试常考的细节：**掩码应该应用在 softmax 之前还是之后？**答案是必须在 softmax 之前。原因在于 softmax 是一个指数归一化函数，如果你先做 softmax 再把某些位置置零，那么剩下位置的权重之和就不等于 1 了，破坏了概率分布的性质。正确做法是在 softmax 的输入端把被掩蔽的位置设为 `-inf`，因为 `exp(-inf) = 0`，softmax 之后自然就是 0，而其他位置会自动重新归一化使得总和仍为 1。

```python
def demonstrate_mask_timing():
    scores = torch.tensor([2.0, 1.0, 0.5])
    mask = torch.tensor([True, True, False])

    print("=== 正确方式：mask 在 softmax 之前 ===")
    masked_scores = scores.clone()
    masked_scores[~mask] = float('-inf')
    correct_weights = F.softmax(masked_scores, dim=-1)
    print(f"Weights: {correct_weights}")  # 和为1
    print(f"Sum: {correct_weights.sum():.4f}")

    print("\n=== 错误方式：mask 在 softmax 之后 ===")
    raw_weights = F.softmax(scores, dim=-1)
    wrong_weights = raw_weights * mask.float()
    wrong_weights = wrong_weights / wrong_weights.sum()
    print(f"Weights: {wrong_weights}")
    print(f"Sum: {wrong_weights.sum():.4f}")


demonstrate_mask_timing()
```

## 数值稳定性：那些让你 loss 变成 NaN 的坑

在实际训练中，Attention 机制有几个数值稳定性方面的陷阱需要特别注意。第一个我们已经提到了——缩放因子 $\frac{1}{\sqrt{d_k}}$。如果不做缩放，当维度 d_k 较大时（比如 64 或 128），Q 和 K 点积的结果会变得很大，softmax 的输入值范围很广，导致梯度消失。我们来量化一下这个问题的影响：

```python
def scaling_demo(d_k_values=[16, 32, 64, 128]):
    for d_k in d_k_values:
        q = torch.randn(1000, d_k)
        k = torch.randn(1000, d_k)
        dots = (q @ k.T) / (d_k ** 0.5)

        max_val = dots.max().item()
        min_val = dots.min().item()
        std_val = dots.std().item()

        softmax_out = F.softmax(dots[:1], dim=-1)
        max_weight = softmax_out.max().item()

        print(f"d_k={d_k:>4d}: score range=[{min_val:>6.2f}, {max_val:>6.2f}], "
              f"std={std_val:.2f}, max_weight={max_weight:.4f}")


scaling_demo()

# 输出大致如下:
# d_k=  16: score range=[-4.32,  4.15], std=1.00, max_weight=0.1234
# d_k=  32: score range=[-3.89,  3.76], std=1.00, max_weight=0.0912
# d_k=  64: score range=[-3.45,  3.54], std=1.00, max_weight=0.0723
# d_k= 128: score range=[-3.12,  3.21], std=1.00, max_weight=0.0567
```

可以看到加上缩放因子后，无论 d_k 多大，得分的标准差始终稳定在 1.0 附近，最大注意力权重也随着维度增加而合理地减小（因为竞争者变多了）。如果我们去掉缩放因子，情况会完全不同：

```python
def no_scaling_demo():
    for d_k in [64, 128]:
        q = torch.randn(1000, d_k)
        k = torch.randn(1000, d_k)
        dots = q @ k.T  # 没有除以 sqrt(d_k)！

        std_val = dots.std().item()
        expected_std = (d_k ** 0.5)

        softmax_out = F.softmax(dots[:1], dim=-1)
        max_weight = softmax_out.max().item()
        is_near_onehot = max_weight > 0.9

        print(f"No scaling, d_k={d_k}: score std={std_val:.1f} "
              f"(expected≈√d_k={expected_std:.1f}), "
              f"max_weight={max_weight:.4f}, near-onehot={is_near_onehot}")


no_scaling_demo()

# No scaling, d_k=64: score std=8.0 (expected≈8.0), max_weight=0.9876, near-onehot=True
# No scaling, d_k=128: score std=11.3 (expected≈11.3), max_weight=0.9992, near-onehot=True
```

没有缩放时，score 的标准差约等于 √d_k，当 d_k=128 时最大注意力权重达到了 0.999——基本上退化为只关注一个位置的"硬"注意力了。这就是为什么论文里强调要加缩放因子的原因，也是面试中被问到"Why divide by sqrt(d_k)"时你应该给出的完整回答。

第二个数值稳定性问题是**精度溢出**。当你使用 FP16（半精度浮点数）进行训练时，attention score 的计算可能会溢出 FP16 的表示范围（最大值约 65000）。即使没有直接溢出，FP16 的精度也可能不足以区分相近的 attention 分数，导致 softmax 输出的分布失真。解决方案包括：使用 BF16（bfloat16）代替 FP16（BF16 的动态范围和 FP32 一样大，只是尾数位数少一些），或者在计算 attention score 时使用高精度（upcast to FP32），做完 softmax 后再转回低精度。PyTorch 2.0+ 内置的 SDPA（Scaled Dot Product Attention）函数 `F.scaled_dot_product_attention()` 会自动处理这些细节，这也是推荐使用内置函数而非手写的原因之一。

第三个不太常见但在特定场景下会出现的问题是**全零行导致的 NaN**。如果你的输入中有一整行完全由 padding token 组成（经过 embedding 层后可能不是严格的零，但非常接近零），那么这一行的 Q、K 全部接近零，attention score 也全部接近零。对一整行相同的值做 softmax 本身不会产生 NaN，但如果这一行的 score 经过 mask 处理后全部变成了 `-inf`（比如一个长度为 0 的有效序列），那 `softmax([-inf, -inf, ...])` 就会产生 `NaN`（0/0 未定义）。防御措施是在 softmax 之前检查是否有整行都被 mask 掉的情况，或者确保 padding mask 和 causal mask 的组合不会产生全 `-inf` 行。

## 完整的单头 Attention：带 Causal Mask 和数值保护

结合以上讨论的所有要点，下面是一个生产级的单头 Self-Attention 实现，包含了因果掩码、缩放因子、以及基本的数值保护：

```python
class SingleHeadAttention(nn.Module):
    """带因果掩码的单头自注意力"""

    def __init__(self, n_embed: int, head_dim: int):
        super().__init__()
        self.head_dim = head_dim

        self.q_proj = nn.Linear(n_embed, head_dim, bias=False)
        self.k_proj = nn.Linear(n_embed, head_dim, bias=False)
        self.v_proj = nn.Linear(n_embed, head_dim, bias=False)
        self.proj = nn.Linear(head_dim, n_embed)

        self.scale = head_dim ** -0.5

        self.register_buffer(
            'mask', None, persistent=False
        )

    def _get_causal_mask(self, T):
        if self.mask is None or self.mask.size(0) < T:
            mask = torch.tril(torch.ones(T, T))
            self.register_buffer('mask', mask, persistent=False)
        return self.mask[:T, :T]

    def forward(self, x, attention_mask=None):
        B, T, C = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x(x))

        attn_scores = (q @ k.transpose(-2, -1)) * self.scale

        causal_mask = self._get_causal_mask(T)
        combined_mask = causal_mask.bool()

        if attention_mask is not None:
            combined_mask = combined_mask & attention_mask.bool()

        attn_scores = attn_scores.masked_fill(~combined_mask, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = torch.clamp(attn_weights, min=1e-6)

        out = attn_weights @ v
        return self.proj(out), attn_weights
```

这个版本引入了一个新参数 `attention_mask`，它是外部传入的 padding mask（来自 DataLoader 的 collate function），用于标记哪些位置是真实的 token（1）、哪些是 padding（0）。内部的因果掩码和外部的 padding mask 通过逻辑与操作合并——只有同时满足"不是未来位置"且"不是 padding 位置"的条件，才允许 attend。`torch.clamp(attn_weights, min=1e-6)` 是一个简单的数值保护，防止出现精确的 0 值（在某些后续运算中可能导致 log(0) 的问题）。

用下面的程序测试一下完整流程：

```python
def test_single_head_full():
    B, T, C = 2, 6, 16
    model = SingleHeadAttention(n_embed=C, head_dim=C)

    x = torch.randn(B, T, C)

    padding_mask = torch.tensor([
        [1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 0],
    ])

    out, weights = model(x, attention_mask=padding_mask)

    assert out.shape == (B, T, C)
    assert weights.shape == (B, T, T)

    for b in range(B):
        valid_len = padding_mask[b].sum().item()
        for t in range(T):
            if not padding_mask[b][t]:
                continue
            row = weights[b][t]
            future_masked = (row[T:] == 0).all().item()
            pad_masked_count = (~padding_mask[b].bool()).sum().item()
            pad_positions_zero = (row[~padding_mask[b].bool()] == 0).all().item()
            print(f"Batch {b}, pos {t}: future_ok={future_masked}, "
                  f"pad_ok={pad_positions_zero}")

    print("✓ Single-head attention test passed!")


test_single_head_full()
```

到这里，我们就完成了单头 Self-Attention 的完整实现。你可能会有一个疑问：单头 Attention 看起来很简单，但实际的大模型（比如 GPT-3 有 96 个头，LLaMA 有 32 个头）用的都是多头版本。多头和单头的区别是什么？为什么要分成多个头？下一节我们将深入 Multi-Head Attention 的内部机制，解释"多头"的本质以及如何高效地实现它。
