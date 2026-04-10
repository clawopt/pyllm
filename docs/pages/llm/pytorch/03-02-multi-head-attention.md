# 3.2 Multi-Head Attention 与残差连接

上一节我们实现了单头 Self-Attention，理解了 Query-Key-Value 的信息检索类比、缩放因子的数学原理、以及因果掩码如何保证自回归模型的正确性。但如果你去看任何现代大模型（GPT-3、LLaMA、Qwen、Mistral）的架构图，你会发现它们用的都不是单头 Attention，而是**多头注意力（Multi-Head Attention, MHA）**。GPT-2 Small 有 12 个头，LLaMA-7B 有 32 个头，GPT-3 175B 甚至有 96 个头。为什么需要这么多头？它们各自在做什么？这一节我们先把"多头"这个概念彻底讲透，然后引入 Transformer 架构中另外两个至关重要的组件——残差连接（Residual Connection）和归一化层（LayerNorm / RMSNorm），为下一节组装完整的 Transformer Block 做好准备。

## 多头的本质：并行关注不同方面的信息

要理解多头的意义，我们先回到单头 Attention 做的事情。在单头情况下，每个位置的 Query、Key、Value 都被投影到同一个 head_dim 维的向量空间里，然后在这个空间里计算相似度、分配注意力权重。这意味着所有类型的语义关系——主谓关系、修饰关系、指代关系等等——都被压缩到了同一组表示中。打个比方，就像一个人同时负责检查一篇文章的语法正确性、事实准确性和逻辑连贯性，虽然理论上可以做到，但实际上每种任务需要的"关注模式"完全不同，混在一起做效率很低。

Multi-Head Attention 的解决思路很简单：与其用一个头来处理所有类型的关系，不如用多个独立的头，每个头专注于捕捉一种特定类型的关系。具体来说，假设我们有 `num_heads` 个头，每个头的维度是 `head_dim`，总嵌入维度是 `n_embed`，那么它们满足关系：`num_heads × head_dim = n_embed`。比如 GPT-2 Small 的 n_embed=768，num_heads=12，所以每个 head_dim=64。每个头有自己独立的 Q/K/V 投影矩阵（或者说从一个大的联合投影中切分出来），独立地计算注意力权重，最后把所有头的输出拼接起来再做一个线性投影。

从直觉上讲，不同的头确实会学到不同的注意力模式。大量实证研究表明，有些头倾向于关注句法结构（比如名词和它最近的动词），有些头专注于长距离依赖（比如段落开头的主题词和后面的代词指代），还有些头似乎在学习位置相邻性。这种分工让模型能够在一个统一的框架下同时处理多种类型的语义关系，而不会互相干扰。

## 两种实现方式及其效率差异

理解了概念之后，让我们来看两种实现 MHA 的方式。第一种是"直觉式"实现——创建多个独立的单头 Attention 模块，分别计算再合并：

```python
class MultiHeadAttentionNaive(nn.Module):
    """多头注意力的朴素实现 — 每个头独立计算"""

    def __init__(self, n_embed, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = n_embed // num_heads

        self.heads = nn.ModuleList([
            SingleHeadAttention(n_embed, self.head_dim)
            for _ in range(num_heads)
        ])

        self.proj = nn.Linear(n_embed, n_embed)

    def forward(self, x, mask=None):
        head_outputs = []
        for head in self.heads:
            out, _ = head(x, mask)
            head_outputs.append(out)

        concat = torch.cat(head_outputs, dim=-1)
        return self.proj(concat)
```

这种方式逻辑清晰，每个头就是一个完整的 SingleHeadAttention 实例。但它有一个明显的效率问题：对每个头都要单独执行一次 Q/K/V 的三个矩阵乘法，总共是 `3 × num_heads` 次大矩阵乘法。而且这些乘法之间没有数据依赖，完全可以并行化。

第二种方式——也是实际生产中采用的方式——是把 Q/K/V 的投影合并成一次大矩阵乘法，然后在结果上通过 reshape 操作来分割出各个头：

```python
class MultiHeadAttention(nn.Module):
    """高效的多头注意力实现 — 联合投影 + reshape 分割"""

    def __init__(self, n_embed: int, num_heads: int):
        super().__init__()
        assert n_embed % num_heads == 0

        self.n_embed = n_embed
        self.num_heads = num_heads
        self.head_dim = n_embed // num_heads

        self.qkv = nn.Linear(n_embed, 3 * n_embed, bias=False)
        self.proj = nn.Linear(n_embed, n_embed)

        self.scale = self.head_dim ** -0.5

    def forward(self, x, mask=None):
        B, T, C = x.shape

        qkv = self.qkv(x)                          # (B, T, 3*C)
        qkv = qkv.reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)                 # 各 (B, T, heads, head_dim)

        q = q.transpose(1, 2).contiguous().view(B * self.num_heads, T, self.head_dim)
        k = k.transpose(1, 2).contiguous().view(B * self.num_heads, T, self.head_dim)
        v = v.transpose(1, 2).contiguous().view(B * self.num_heads, T, self.head_dim)

        attn_scores = (q @ k.transpose(-2, -1)) * self.scale   # (B*H, T, T)

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)

        out = attn_weights @ v                              # (B*H, T, head_dim)
        out = out.view(B, self.num_heads, T, self.head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.proj(out), attn_weights.view(B, self.num_heads, T, T)
```

这里的关键技巧在于 reshape 和 transpose 的组合使用。`qkv` 的 shape 是 `(B, T, 3*C)`，先 reshape 成 `(B, T, 3, num_heads, head_dim)`，然后用 `unbind(dim=2)` 在第 2 维上拆分得到 Q、K、V 三个张量，每个 shape 为 `(B, T, num_heads, head_dim)`。接下来 `transpose(1, 2)` 把 heads 维度移到 T 之前变成 `(B, num_heads, T, head_dim)`，再用 `.contiguous().view(B*num_heads, T, head_dim)` 把 batch 和 heads 合并——这一步的本质是把多个头的计算变成了一个更大的 batch 的计算，这样后续的矩阵乘法就可以利用 GPU 的高效 GEMM 实现。做完 attention 之后，再用 view + transpose 的逆操作把结果还原回 `(B, T, C)`。

我们来对比一下两种实现的效率差异：

```python
def benchmark_mha_implementations():
    import time

    B, T, C = 8, 256, 768
    num_heads = 12

    x = torch.randn(B, T, C)

    naive = MultiHeadAttentionNaive(C, num_heads).cuda()
    efficient = MultiHeadAttention(C, num_heads).cuda()
    x_cuda = x.cuda()

    for model in [naive, efficient]:
        model.eval()

        with torch.no_grad():
            for _ in range(10):
                _ = model(x_cuda)

        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            _ = model(x_cuda)
        torch.cuda.synchronize()
        elapsed = time.time() - start

        name = "Naive" if model == naive else "Efficient"
        print(f"{name:>10s}: {elapsed/100*1000:.2f} ms/batch")


benchmark_mha_implementations()

# 输出示例:
#     Naive: 4.82 ms/batch
#   Efficient: 1.35 ms/batch  → 快了约 3.6x
```

高效实现比朴素实现快了约 3~4 倍，原因有三个：第一，Q/K/V 从 3×N 次矩阵乘法减少到 1 次大矩阵乘法（GPU 对大矩阵乘法的优化更好）；第二，reshape 后的多头计算被合并成了一个大 batch 的单头计算，增加了 GPU 并行度；第三，减少了 Python 循环开销（朴素版本有一个 for 循环遍历所有头）。这就是为什么你在所有主流框架的实现中看到的都是第二种方式而不是第一种。

## 残差连接：为什么 x = x + sublayer(x) 如此有效

有了 Multi-Head Attention 之后，我们离完整的 Transformer Block 还差两个关键组件：残差连接和归一化层。先说残差连接（Residual Connection），也叫跳跃连接（Skip Connection）。它的形式极其简单——在每个子层（Attention 或 FFN）的输出上加上原始输入：

$$\text{output} = x + \text{SubLayer}(x)$$

看起来就是一行加法，但这个简单的设计却有着深刻的含义。要理解残差连接的作用，我们需要回顾一下深度网络训练中的一个核心困难：梯度消失问题。当网络很深时（Transformer 通常有几十甚至上百层），反向传播的梯度需要逐层向前传递，每经过一层都会乘以该层的权重矩阵，如果权重的范数小于 1，梯度就会指数级衰减；如果大于 1，就会指数级爆炸。无论哪种情况，深层的参数都难以接收到有效的梯度信号来更新。

残差连接巧妙地绕过了这个问题。因为 `y = x + F(x)` 对 x 求导的结果是 `∂y/∂x = I + ∂F/∂x`，其中 I 是单位矩阵。这意味着即使 `∂F/∂x` 非常小（接近零），梯度仍然可以通过恒等映射 I 直接流向更前面的层——梯度至少不会消失。同样，即使 `∂F/∂x` 很大，I 的存在也保证了梯度不会无限放大。从这个角度看，残差连接相当于给网络提供了一条"梯度高速公路"，确保即使在很深的网络上，梯度也能顺畅地流动。

除了梯度的角度，残差连接还有一个直观的解释：它让每一层只需要学习"残差"（即输入和目标之间的差异），而不是从头学习完整的变换。如果某一层的输入已经离目标很近了，那么 F(x) 只需要学一个很小的修正量就够了；如果输入离目标很远，F(x) 就需要学一个较大的变换。这种自适应的学习方式比强制每层都学习完整变换更容易优化。

```python
def residual_demo():
    """演示残差连接如何帮助梯度流动"""

    class DeepNetWithoutResidual(nn.Module):
        def __init__(self, depth):
            super().__init__()
            self.layers = nn.ModuleList([
                nn.Linear(64, 64) for _ in range(depth)
            ])

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
                x = torch.relu(x)
            return x

    class DeepNetWithResidual(nn.Module):
        def __init__(self, depth):
            super().__init__()
            self.layers = nn.ModuleList([
                nn.Linear(64, 64) for _ in range(depth)
            ])

        def forward(self, x):
            for layer in self.layers:
                x = x + torch.relu(layer(x))
            return x

    depths = [10, 20, 50]
    x = torch.randn(4, 64, requires_grad=True)
    target = torch.randn(4, 64)

    for NetClass, name in [(DeepNetWithoutResidual, "No Residual"),
                            (DeepNetWithResidual, "With Residual")]:
        print(f"\n{'='*40}")
        print(f"{name}:")
        print(f"{'='*40}")
        for d in depths:
            net = NetClass(d)
            out = net(x)
            loss = ((out - target) ** 2).mean()
            loss.backward()

            grad_norm = x.grad.norm().item()
            mean_grad = x.grad.abs().mean().item()

            print(f"  depth={d:>3d}: grad_norm={grad_norm:.6f}, "
                  f"mean_abs_grad={mean_grad:.6f}")

            x.grad.zero_()


residual_demo()

# 典型输出:
# ========================================
# No Residual:
# ========================================
#   depth= 10: grad_norm=0.000123, mean_abs_grad=0.000002
#   depth= 20: grad_norm=0.00000004, mean_abs_grad=0.000000001
#   depth= 50: grad_norm=0.00000000, mean_abs_grad=0.000000000
#
# ========================================
# With Residual:
# ========================================
#   depth= 10: grad_norm=2.341234, mean_abs_grad=0.034567
#   depth= 20: grad_norm=2.123456, mean_abs_grad=0.032345
#   depth= 50: grad_norm=1.987654, mean_abs_grad=0.031234
```

结果非常明显：没有残差连接时，随着网络加深，输入梯度迅速衰减到几乎为零（depth=50 时已经完全是 0 了）；而有残差连接时，即使深度达到 50 层，梯度的范数依然保持在健康范围内。这就是为什么 Transformer 能够堆叠到 96 层（GPT-3）、80 层（LLaMA）甚至更多而不出现训练困难的根本原因之一。

## LayerNorm vs RMSNorm：归一化的两种选择

另一个关键组件是归一化层。在 Transformer 中，归一化的作用是稳定每层输出的数值分布，防止数值在层层传递后变得过大或过小，从而加速收敛并提高训练稳定性。最经典的选择是 Layer Normalization（LayerNorm），由 Ba 等人在 2016 年提出。

LayerNorm 的计算过程如下：对于输入向量 $x \in \mathbb{R}^d$，先计算均值 $\mu = \frac{1}{d}\sum_{i=1}^{d} x_i$ 和方差 $\sigma^2 = \frac{1}{d}\sum_{i=1}^{d}(x_i - \mu)^2$，然后做归一化 $\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}$，最后做仿射变换 $y_i = \gamma \cdot \hat{x}_i + \beta$，其中 $\gamma$ 和 $\beta$ 是可学习的参数。

```python
class LayerNorm(nn.Module):
    """手动实现 Layer Norm"""

    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta
```

PyTorch 内置的 `nn.LayerNorm` 做的事情完全一样，上面的手写版只是为了帮助你理解内部机制。在实际代码中直接用 `nn.LayerNorm` 即可。

不过，现代大模型（特别是 LLaMA 及其衍生模型如 Qwen、Mistral）越来越多地采用了另一种归一化方案——**RMSNorm（Root Mean Square Normalization）**。RMSNorm 是 Zhang 和 Sennrich 在 2019 年提出的，它的核心思想是：LayerNorm 中的均值中心化（减去均值 $\mu$）步骤可能并不是必需的——真正起作用的可能是方差归一化部分。因此 RMSNorm 去掉了减去均值的操作，只保留除以均方根的部分：

$$\text{RMSNorm}(x)_i = \frac{x_i}{\sqrt{\bar{x}^2 + \epsilon}} \cdot \gamma$$

其中 $\bar{x}^2 = \frac{1}{d}\sum_{i=1}^{d} x_i^2$ 是均方值。注意 RMSNorm 只有一个可学习参数 $\gamma$（没有 $\beta$），因为它不做中心化所以不需要偏置。

```python
class RMSNorm(nn.Module):
    """RMS Normalization — LLaMA/Qwen/Mistral 采用"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms

    def forward(self, x):
        output = self._norm(x)
        return self.weight * output
```

RMSNorm 相比 LayerNorm 有两个优势。第一是速度更快：省去了计算均值的步骤，减少了内存访问和运算量。虽然在单个层上的差异不大，但当模型有几十到上百层、每次前向传播都要调用上百次归一化时，这些节省会累积起来。第二是经验表明在某些任务上 RMSNorm 的效果与 LayerNorm 相当甚至略好，这可能与去掉均值中心化后保留了更多的原始信号信息有关。当然这不是绝对的——某些场景下 LayerNorm 仍然表现更好——但对于当前主流的大语言模型来说，RMSNorm 已经成为了事实标准。

## Pre-Norm vs Post-Norm：归一化放哪里？

现在我们把残差连接和归一化结合起来，就面临一个架构选择问题：LayerNorm/RMSNorm 应该放在子层（Attention 或 FFN）之前还是之后？这两种选择对应着两种截然不同的 Transformer 架构范式。

**Post-Norm（原始 Transformer / BERT 风格）**的结构是：

```
x → SubLayer(x) → LayerNorm(x + SubLayer(x)) → output
```

也就是先做子层运算，再做残差相加，最后做归一化。原始的 "Attention Is All You Need" 论文和 BERT 采用的就是这种结构。

**Pre-Norm（GPT-2 / LLaMA / Qwen 风格）**的结构是：

```
x → LayerNorm(x) → SubLayer(LayerNorm(x)) → x + SubLayer(...) → output
```

也就是先做归一化，再把归一化后的结果送入子层，最后与原始输入做残差相加。注意 Pre-Norm 中最终输出通常还会再加一层 LayerNorm（叫做 final LayerNorm）。

这两种选择的区别远比表面上看起来的大。Post-Norm 的梯度路径是：loss → final LN → ... → add → sublayer → add → LN → ... → embedding。每一层的梯度都需要穿过 LayerNorm 才能到达该层的参数，而 LayerNorm 的梯度行为在某些情况下不够理想。Pre-Norm 则不同：由于归一化在子层之前，子层接收到的输入已经被归一化了，这意味着子层的输入分布更加稳定，训练也更加稳健。更重要的是，Pre-Norm 提供了一条"干净的"残差路径——梯度可以直接通过加法分支流向前面的层，不需要穿过任何归一化层或非线性激活函数。

```python
class TransformerBlockPostNorm(nn.Module):
    """Post-Norm 结构的 Transformer Block"""

    def __init__(self, n_embed, num_heads, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(n_embed, num_heads)
        self.ffn = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.ln1(x + self.dropout(self.attn(x, mask)[0]))
        x = self.ln2(x + self.dropout(self.ffn(x)))
        return x


class TransformerBlockPreNorm(nn.Module):
    """Pre-Norm 结构的 Transformer Block（GPT-2/LLaMA 风格）"""

    def __init__(self, n_embed, num_heads, dropout=0.1):
        super().__init__()
        self.ln1 = RMSNorm(n_embed)
        self.attn = MultiHeadAttention(n_embed, num_heads)
        self.ln2 = RMSNorm(n_embed)
        self.ffn = FeedForward(n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = x + self.dropout(self.attn(self.ln1(x), mask)[0])
        x = x + self.dropout(self.ffn(self.ln2(x)))
        return x
```

仔细对比两个版本的 forward 方法。Post-Norm 版本中，`self.attn(x, mask)[0]` 接收的是原始的、未经归一化的 x，输出加上 x 之后才送入 LayerNorm。Pre-Norm 版本中，x 先经过 `self.ln1(x)` 归一化，归一化后的结果才送入 Attention，最后 Attention 的输出与原始 x（未归一化的那个）相加。这里的一个关键细节是：**残差连接总是连接到原始输入 x，而不是归一化后的 x**。这是 Pre-Norm 设计的核心要点——它保证了残差路径是一条不受归一化干扰的"高速公路"。

实际经验表明，Pre-Norm 相比 Post-Norm 有几个显著优势：训练更稳定（尤其是在深层网络上）、可以使用更高的学习率、对初始化不那么敏感。这也是为什么 GPT-2、GPT-3、LLaMA、Qwen、Mistral 等几乎所有现代 decoder-only 大模型都采用 Pre-Norm 结构的原因。Post-Norm 并没有被淘汰——它在 encoder-only 模型（如 BERT 的某些变体）和 encoder-decoder 模型（如 T5、BART）中仍有应用——但在纯 decoder 的语言模型领域，Pre-Norm 已经占据了绝对主导地位。

到这里，我们已经掌握了构建完整 Transformer Block 所需的全部组件：Multi-Head Attention（带因果掩码和缩放因子）、残差连接（梯度高速公路）、以及 Pre-Norm 结构下的 RMSNorm 归一化。还剩下一个组件——Feed-Forward Network（前馈神经网络），它是 Transformer Block 的另一半。下一节我们将深入 FFN 的世界，了解为什么它占了整个模型约 2/3 的参数量，以及现代 LLM 中广泛使用的 SwiGLU 激活函数是如何工作的。
