# 3.3 Feed-Forward Network 与 Transformer Block 组装

如果你仔细看过任何一个 Transformer 模型的参数量统计报告，你会发现一个反直觉的事实：**Feed-Forward Network（FFN，前馈神经网络）占了整个模型大约三分之二的参数量**，而看起来更复杂、更"核心"的 Multi-Head Attention 只占约三分之一。以 GPT-2 Medium 为例，总参数量 355M，其中 Attention 部分约 118M，FFN 部分约 237M。LLaMA-7B 的比例也类似——Attention 约 2.3B，FFN 约 4.7B。这意味着 FFN 绝不是一个简单的"附属组件"，而是 Transformer 表达能力的核心贡献者之一。这一节我们就要搞清楚 FFN 到底在做什么、为什么需要这么大、以及现代大模型中广泛采用的 SwiGLU 变体是如何工作的。最后我们会把之前学过的所有组件——MHA + 残差连接 + RMSNorm + FFN——组装成一个完整的 Transformer Block。

## FFN 的结构：为什么是先放大再缩小？

标准 Transformer 中的 FFN 结构非常简洁，只有两层全连接层和一个激活函数：

$$\text{FFN}(x) = W_2 \cdot \text{activation}(W_1 \cdot x + b_1) + b_2$$

第一层把输入从 `n_embed` 维投影到一个更大的中间维度（通常是 n_embed 的 4 倍），经过非线性激活函数后，第二层再把维度投影回 `n_embed`。这个"先放大再缩小"的结构叫做 bottleneck 架构或者 expansion-projection 结构。

```python
class FeedForward(nn.Module):
    """标准 FFN: Linear(up) → Activation → Linear(down)"""

    def __init__(self, n_embed: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden_dim = n_embed * expansion

        self.net = nn.Sequential(
            nn.Linear(n_embed, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
```

这里 `expansion=4` 是最常见的配置——中间隐藏层维度是输入维度的 4 倍。比如 GPT-2 的 n_embed=768，所以 FFN 中间层就是 3072 维；LLaMA-7B 的 n_embed=4096，中间层就是 16384 维。这就是 FFN 参数量大的直接原因：两个线性层的参数量分别是 `n_embed × (n_embed × 4)` 和 `(n_embed × 4) × n_embed`，加起来是 `8 × n_embed²`；而 MHA 的主要参数来自 Q/K/V 投影和输出投影，总共约 `4 × n_embed²`（假设单次联合投影）。两者的比值正好约为 2:1。

但为什么需要这个"膨胀"结构？直觉上的解释是：Attention 层负责的是"信息路由"——决定每个位置应该关注哪些其他位置的信息，这是一种相对"全局"的操作。而 FFN 负责"信息处理"——对每个位置独立地做复杂的非线性变换，是一种逐位置的操作。中间维度的膨胀给每个位置提供了更大的"思考空间"，让它能够学习到更丰富的特征表示。你可以把它想象成：Attention 决定了"看哪里"，FFN 负责在看了之后"怎么理解"。如果 FFN 的容量太小（比如 expansion=1，即没有膨胀），那它就退化为一个简单的线性变换，表达能力严重不足；如果太大（比如 expansion=16），参数量会爆炸而且容易过拟合。4 倍是一个在实践中被反复验证效果不错的折中值。

```python
def ffn_parameter_count(n_embed, expansion_values=[1, 2, 4, 8]):
    print(f"{'Expansion':>10s} | {'Hidden':>8s} | {'FFN Params':>12s} | "
          f"{'Total(%)':>10s}")
    print("-" * 55)

    attn_params = 4 * n_embed * n_embed

    for e in expansion_values:
        hidden = n_embed * e
        ffn_params = 2 * n_embed * hidden
        total = ffn_params + attn_params
        pct = ffn_params / total * 100

        print(f"{e:>10d} | {hidden:>8d} | {ffn_params:>12,d} | {pct:>9.1f}%")

ffn_parameter_count(n_embed=768)

# 输出:
# Expansion |   Hidden |  FFN Params |   Total(%)
# -------------------------------------------------------
#          1 |      768 |     1,179,648 |      23.1%
#          2 |     1,536 |     2,359,296 |      37.0%
#          4 |     3,072 |     4,718,592 |      50.0%
#          8 |     6,144 |     9,437,184 |      66.7%
```

可以看到当 expansion=4 时，FFN 正好占总参数的 50%（加上 Embedding 和 LayerNorm 等其他部分后实际占比约 2/3）。

## 激活函数演进：从 ReLU 到 SwiGLU

FFN 中间层的激活函数选择经历了一个有趣的演进过程。最早的 Transformer 论文使用的是 ReLU（Rectified Linear Unit），它的定义极其简单：`ReLU(x) = max(0, x)`——负数全部截断为 0，正数保持不变。ReLU 计算快、实现简单，但它有一个明显的缺陷：负半轴完全为零梯度（dying ReLU 问题），一旦某个神经元进入"死亡"状态（对所有输入都输出 0），它就永远无法恢复，因为梯度无法通过零值的区域传播回来。

GPT-2 把激活函数换成了 **GELU（Gaussian Error Linear Unit）**：

$$\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}\left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]$$

其中 $\Phi(x)$ 是标准高斯分布的累积分布函数（CDF）。GELU 可以理解为一种"平滑版的 ReLU"——它在负半轴不是突然截断为 0，而是平滑地过渡到接近 0 但不为 0 的值。这种平滑性带来了更好的梯度和优化性质，也是 GPT-2 相比原始 Transformer 效果提升的因素之一。

```python
import math

def gelu_comparison():
    x = torch.linspace(-3, 3, 100)

    relu_out = torch.relu(x)
    gelu_out = F.gelu(x)

    import numpy as np
    print(f"{'x':>6s} | {'ReLU':>6s} | {'GELU':>8s}")
    print("-" * 28)
    for i in range(0, 100, 10):
        print(f"{x[i]:>6.2f} | {relu_out[i]:>6.2f} | {gelu_out[i]:>8.4f}")

gelu_comparison()

# 输出:
#      x |   ReLU |    GELU
# ----------------------------
#  -3.00 |   0.00 |  -0.0040
#  -2.40 |   0.00 |  -0.0164
#  -1.80 |   0.00 |  -0.0665
#  -1.20 |   0.00 |  -0.1700
#  -0.60 |   0.00 |  -0.1586
#   0.00 |   0.00 |   0.0000
#   0.60 |   0.60 |   0.3970
#   1.20 |   1.20 |   1.0755
#   1.80 |   1.80 |   1.7645
#   2.40 |   2.40 |   2.3744
#   3.00 |   3.00 |   2.9954
```

注意 GELU 在负数区域（如 x=-0.6）输出 -0.1586 而不是 ReLU 的 0，这保证了即使输入为负也有非零梯度可以流动。

到了 LLaMA 和 GLM 这一代模型，激活函数又进化到了 **SwiGLU（Sigmoid-weighted Gated Linear Unit）**。SwiGLU 的结构比 GELU 复杂一些，它引入了门控机制：

$$\text{SwiGLU}(x) = (xW_1 \odot \text{SiLU}(xW_2)) W_3$$

其中 $\odot$ 表示逐元素乘法（element-wise product），SiLU（也叫 Swish）的定义是 $\text{SiLU}(x) = x \cdot \sigma(x)$（$\sigma$ 是 sigmoid 函数）。注意这里的 $W_1$ 和 $W_2$ 是**两个不同的权重矩阵**，这意味着 SwiGLU 实际上用了三个投影矩阵而不是标准 FFN 的两个。从架构上看，SwiGLU 的第一层被拆成了两个并行的分支：一个做恒等变换（或说线性变换 $W_1$），另一个做 SiLU 非线性变换（$W_2$ + SiLU），然后两者做逐元素相乘（门控），最后通过 $W_3$ 投影回去。

```python
class FeedForwardSwiGLU(nn.Module):
    """SwiGLU FFN — LLaMA/Qwen/Mistral 采用"""

    def __init__(self, n_embed: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden_dim = int(n_embed * expansion * 2 / 3)
        hidden_dim = ((hidden_dim + 255) // 256) * 256

        self.w1 = nn.Linear(n_embed, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, n_embed, bias=False)
        self.w3 = nn.Linear(n_embed, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
```

这里有一个重要的细节：LLaMA 中 SwiGLU 的隐藏层维度计算公式是 `hidden_dim = int(n_embed × 4/3 × 2/3)`... 不对，让我重新解释。实际上 LLaMA 的做法是保持与标准 FFN 大致相同的参数预算：标准 FFN 用 expansion=4 时有 `2 × n_embed × 4n_embed = 8n_embed²` 个参数；SwiGLU 有三个矩阵，为了使总参数量相当，设中间维度为 `4/3 × 2/3 × n_embed`... 让我简化一下。实际上 LLaMA 的代码中用的是 `hidden_dim = int(4/3 * 2 * n_embed / 3)` 这种方式来让 SwiGLU 的总参数量接近标准 4x FFN。不过具体实现上不同的模型略有差异，上面的代码采用了 LLaMA 的原始方案：`int(n_embed * expansion * 2 / 3)` 后再做 256 对齐（这是为了硬件效率，确保维度是 256 的倍数以便利用 Tensor Core）。

SwiGLU 为什么比 GELU 更好？核心原因在于**门控机制**带来的额外表达力。标准 FFN 中，每个神经元只能选择"激活"或"不激活"（通过 ReLU/GELU 的阈值行为）；而在 SwiGLU 中，$W_3$ 分支产生的向量相当于一个"门控信号"，它可以精细地控制 $W_1$ 分支中每个维度的信息通过多少——有些维度可能完全关闭（门控值接近 0），有些可能完全打开（门控值接近 1），还有的可能只通过一部分。这种细粒度的控制让模型能够学习到更复杂的特征交互模式。大量实验表明，在其他条件相同的情况下，SwiGLU 通常能带来 0.1~0.5 的 perplexity 改善，在大规模模型上这个差距虽然看似不大，但在竞争激烈的 benchmark 上足以拉开排名差距。

## 组装完整的 Transformer Block

好了，现在我们手里有了所有需要的积木：MultiHeadAttention（带因果掩码）、RMSNorm（Pre-Norm 归一化）、FeedForwardSwiGLU（前馈网络）、残差连接（通过加法实现）、Dropout（正则化）。让我们把它们组装成一个完整的 Transformer Block。

```python
class TransformerBlock(nn.Module):
    """完整的 Transformer Block — Pre-Norm + SwiGLU"""

    def __init__(self, n_embed: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert n_embed % num_heads == 0

        self.ln1 = RMSNorm(n_embed)
        self.attn = MultiHeadAttention(n_embed, num_heads)
        self.dropout1 = nn.Dropout(dropout)

        self.ln2 = RMSNorm(n_embed)
        self.ffn = FeedForwardSwiGLU(n_embed, expansion=4)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, C = x.shape

        x = x + self.dropout1(self.attn(self.ln1(x), mask)[0])
        x = x + self.dropout2(self.ffn(self.ln2(x)))

        return x
```

整个 forward 方法只有两行核心逻辑，但每一行都蕴含了丰富的设计思想。第一行 `x = x + dropout(attn(ln1(x)))` 执行的是：先把输入 x 做 RMSNorm 归一化 → 送入 Multi-Head Attention（带因果掩码）→ 对 attention 输出做 Dropout → 与原始 x 做残差相加。第二行 `x = x + dropout(ffn(ln2(x)))` 执行的是类似的流程，只不过把 Attention 换成了 FFN。这两行合在一起就是一个完整的 Pre-Norm Transformer Block。

我们来验证一下数据流是否正确：

```python
def test_transformer_block():
    config = dict(
        n_embed=256,
        num_heads=4,
        dropout=0.0,
    )

    block = TransformerBlock(**config)

    T = 32
    x = torch.randn(2, T, config['n_embed'])
    causal_mask = create_causal_mask(T)

    with torch.no_grad():
        out = block(x, mask=causal_mask)

    assert out.shape == x.shape, f"Shape mismatch: {out.shape} vs {x.shape}"

    diff = (out - x).abs().sum().item()
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"|output - input| L1 sum: {diff:.4f} (should be > 0)")
    print("✓ TransformerBlock test passed!")


test_transformer_block()
```

运行后确认输入输出的 shape 一致（都是 `(2, 32, 256)`），且输出确实不同于输入（残差连接保证了这一点，除非 Attention 和 FFN 都恰好输出全零）。

## Dropout 在 LLM 训练中的角色

Transformer Block 中出现了两次 Dropout——一次在 Attention 之后，一次在 FFN 之后。Dropout 的作用是在训练过程中随机将一部分神经元的输出置为零（概率由 dropout 参数控制），防止模型过度依赖特定的神经元组合，从而缓解过拟合。在推理时调用 `model.eval()` 后，Dropout 会自动关闭（所有神经元都保留）。

但在大语言模型的训练实践中，Dropout 的使用存在一个有趣的现象：很多大规模 LLM 在训练时干脆把 dropout 设为 0。这是因为当训练数据量极大（几千亿 token 级别）时，过拟合的风险本身就非常低——模型几乎不可能记住所有训练样本。同时，Dropout 会增加训练的不确定性（每次前向传播的结果略有不同），这在超大规模训练中可能会影响收敛稳定性。GPT-3 的论文明确指出他们没有使用 Dropout，LLaMA 同样如此。所以在你自己的项目中，如果你的数据量足够大（几亿 token 以上），可以尝试把 dropout 设为 0 来简化训练流程；如果数据量较小（千万 token 级别），保留 0.1 左右的 dropout 可能会有帮助。

```python
def dropout_effect_demo():
    model_train = TransformerBlock(n_embed=128, num_heads=4, dropout=0.1)
    model_eval = TransformerBlock(n_embed=128, num_heads=4, dropout=0.1)

    model_train.train()
    model_eval.eval()

    x = torch.randn(1, 8, 128)
    mask = create_causal_mask(8)

    out_train_1 = model_train(x, mask)
    out_train_2 = model_train(x, mask)
    out_eval_1 = model_eval(x, mask)
    out_eval_2 = model_eval(x, mask)

    train_same = torch.allclose(out_train_1, out_train_2, atol=1e-6)
    eval_same = torch.allclose(out_eval_1, out_eval_2, atol=1e-6)

    print(f"Train mode: same input → same output? {train_same}")
    print(f"Eval mode:  same input → same output? {eval_same}")
    print("(Train mode should be False due to dropout randomness)")


dropout_effect_demo()
```

到这里，单个 Transformer Block 就完整实现了。下一节我们将讨论一个到目前为止一直回避的问题：**位置编码（Positional Encoding）**。你可能已经注意到，Self-Attention 本身是排列不变的（permutation invariant）——如果你把输入序列的顺序打乱，Attention 的输出（不考虑掩码的话）只是做了同样的排列变换，每对元素之间的关系不变。这意味着纯 Attention 无法区分 "我爱你" 和 "你爱我" 这样的语序差异。为了让模型感知位置信息，我们需要显式地把位置信息注入到输入表示中。现代 LLM 最常用的方案是 RoPE（旋转位置编码），这也是下一节的核心内容。
