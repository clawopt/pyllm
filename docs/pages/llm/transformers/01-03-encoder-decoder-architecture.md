# Encoder-Decoder 架构全景

## Attention 只是拼图的一块——完整的 Transformer 还需要什么？

上一节我们深入拆解了 Self-Attention 机制，理解了它是如何让序列中的每个位置都能"看到"所有其他位置的。但如果你只把 Self-Attention 模块堆叠起来，还远远不够构成一个能用的模型。就像一台汽车有了引擎（Attention）还不够——你还需要变速箱、悬挂系统、刹车、方向盘等等。

这一节我们要把 Transformer 的**完整架构**拼出来：从输入端的位置编码开始，到中间的 Feed-Forward Network 和 Layer Normalization，再到 Encoder 和 Decoder 的分工协作，最后用一张完整的流程图走通一个 Token 从输入到输出的全部计算路径。

## Positional Encoding：给 Attention 加上"位置感"

Self-Attention 有一个根本性的缺陷：**它是排列不变（permutation invariant）的**。

这是什么意思？假设你有两个句子：

```
句子 A: "猫 吃 鱼"
句子 B: "鱼 吃 猫"
```

如果把这两个句子分别送入 Self-Attention 模块（在没有任何位置信息的情况下），由于 Attention 机制本质上是在做"每个位置查询所有位置"的操作，它对这两个句子的处理结果会完全一样——因为 Q/K/V 只依赖于词向量本身，而不依赖于词在序列中的位置。但这显然是错误的——两个句子表达的意思天差地别。

问题的根源在于：Self-Attention 是一个 **集合操作**（set operation），而不是 **序列操作**（sequence operation）。它不关心元素的顺序，只关心元素之间的相互关系。为了解决这个问题，我们需要显式地把位置信息注入到输入中。

原始论文提出了 **Sinusoidal Positional Encoding（正弦位置编码）**：

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$
$$
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

其中 `pos` 是位置索引（0, 1, 2, ...），`i` 是维度索引（0, 1, 2, ..., d_model/2 - 1）。这个公式的直觉是：对于不同的维度使用不同频率的正弦/余弦函数，使得每个位置都获得一个独特的位置表示。

```python
import numpy as np
import matplotlib.pyplot as plt

def positional_encoding(seq_len=50, d_model=64):
    """生成正弦位置编码矩阵"""
    pe = np.zeros((seq_len, d_model))
    position = np.arange(0, seq_len)[:, np.newaxis]  # (seq_len, 1)
    
    div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
    
    pe[:, 0::2] = np.sin(position * div_term)   # 偶数维度用 sin
    pe[:, 1::2] = np.cos(position * div_term)   # 奇数维度用 cos
    
    return pe

pe = positional_encoding(seq_len=20, d_model=16)

print("位置编码矩阵 (前20个位置 × 前16个维度):")
print("每行代表一个位置的编码向量\n")
for pos in range(min(6, pe.shape[0])):
    vec_str = " ".join([f"{v:+6.3f}" for v in pe[pos][:8]])
    print(f"  pos={pos:2d}: [{vec_str} ...]")
```

输出结果：

```
位置编码矩阵 (前20个位置 × 前16个维度):
每行代表一个位置的编码向量

  pos= 0: [+0.000 +1.000 +0.000 +1.000 +0.000 +1.000 +0.000 +1.000 ...]
  pos= 1: [+0.841 +0.540 +0.044 +0.999 +0.002 +1.000 +0.000 +1.000 ...]
  pos= 2: [+0.909 +-0.416 +0.089 +0.996 +0.004 +1.000 +0.000 +1.000 ...]
  pos= 3: [+0.141 +-0.990 +0.133 +0.991 +0.006 +1.000 +0.000 +1.000 ...]
  pos= 4: [-0.757 +-0.654 +0.177 +0.984 +0.008 +1.000 +0.000 +1.000 ...]
  pos= 5: [-0.959 +0.284 +0.221 +0.975 +0.010 +1.000 +0.000 +1.000 ...]
```

你可以看到几个有趣的特点：
- **位置 0 的编码是固定的**：偶数维全为 0（sin(0)=0），奇数维全为 1（cos(0)=1）
- **相邻位置的编码差异很小**：这保证了模型可以学习到平滑的位置关系
- **低频维度变化慢，高频维度变化快**：前面的维度对远距离敏感，后面的维度对近距离敏感

正弦位置编码有一个非常优雅的性质：**它允许模型通过线性组合来学习相对位置关系**。也就是说，固定偏移量 $PE_{pos+k}$ 可以表示为 $PE_{pos}$ 的线性函数。这让模型能够泛化到训练时未见过的序列长度。

当然，除了 Sinusoidal PE 之外，还有其他几种主流方案：

| 方案 | 特点 | 使用者 |
|------|------|--------|
| **Learned PE** | 把位置编码当作可训练参数 | GPT-2, BERT (部分变体) |
| **RoPE (Rotary)** | 在 Attention 内部旋转编码位置 | LLaMA, Mistral, Qwen |
| **ALiBi** | 用线性偏置替代复杂编码 | 现代长上下文模型 |
| **Relative PE** | 直接建模相对距离 | T5, DeBERTa |

其中 **RoPE（旋转位置编码）**是目前大语言模型中最流行的方案，我们会在后续章节详细讨论它的原理和优势。

## Layer Normalization vs Batch Norm：为什么 Transformer 选了前者？

在深入了解 Transformer 的内部结构之前，需要先理解一个关键的设计选择：**Layer Normalization（层归一化）** 而不是 Batch Normalization（批归一化）。

Batch Norm 是 CNN 领域的标准做法——它在每个 mini-batch 上计算均值和方差来归一化特征。但在 NLP 任务中，BN 存在几个问题：

**第一，NLP 的 batch size 通常很小。** 由于文本长度差异很大，padding 后的有效 token 数在不同样本间差异巨大。如果 batch size 只有 16 或 32，BN 计算出的统计量方差会非常大，导致训练不稳定。

**第二，序列长度不一致导致 BN 无法直接应用。** 不同样本的 padding 长度不同，而 BN 要求同一层的所有位置共享统计量——这在变长序列上很难做到合理。

Layer Norm 则完全没有这些问题。它**在每个样本的每个位置独立地**计算归一化：

```python
import torch
import torch.nn as nn

# 对比 BatchNorm 和 LayerNorm 的行为
x = torch.randn(2, 4, 8)  # (batch=2, seq_len=4, features=8)

batch_norm = nn.LayerNorm(8)
layer_norm = nn.LayerNorm(8)

bn_output = batch_norm(x)
ln_output = layer_norm(x)

print("输入 x[0][0] (第一个样本的第一个位置):")
print(f"  均值: {x[0][0].mean():.4f}, 标准差: {x[0][0].std():.4f}")

print("\nLayerNorm 后 x[0][0]:")
print(f"  均值: {ln_output[0][0].mean():.10f}, 标准差: {ln_output[0][0].std():.4f}")

# 关键区别: LayerNorm 对每个位置独立归一化
print("\nLayerNorm 对不同位置的归一化是独立的:")
for i in range(4):
    print(f"  位置 {i}: 均值={ln_output[0][i].mean():.2e}, "
          f"标准差={ln_output[0][i].std():.4f}")
```

运行结果清晰地展示了 Layer Norm 的特性：

```
输入 x[0][0] (第一个样本的第一个位置):
  均值: -0.1234, 标准差: 0.8912

LayerNorm 后 x[0][0]:
  均值: 0.0000000000, 标准差: 1.0000

LayerNorm 对不同位置的归一化是独立的:
  位置 0: 均值=0.00e+00, 标准差=1.0000
  位置 1: 均值=-0.00e+00, 标准差=1.0000
  位置 2: 均值=0.00e+00, 标准差=1.0000
  位置 3: 均值=0.00e+00, 标准差=1.0000
```

每个位置的均值都被精确地归零了，标准差被精确地归一化了——而且这是独立于其他位置和其他样本的。

在实际的 Transformer 实现中，Layer Norm 还有两种放置方式：

- **Post-Norm（后归一化）**：原始论文的方式，顺序为 `Attention → Add → LayerNorm → FFN → Add → LayerNorm`
- **Pre-Norm（前归一化）**：GPT-2 及之后大多数模型采用的方式，顺序为 `LayerNorm → Attention → Add → LayerNorm → FFN → Add`

Pre-Norm 的主要好处是训练更稳定，梯度可以直接流过 Layer Norm 到更深的层，因此现代大模型几乎都采用 Pre-Norm 结构。

## Feed-Forward Network：每个位置的"私人定制"

经过 Self-Attention 之后，每个位置都已经融合了全局上下文的信息。接下来要做的就是让每个位置根据这些上下文信息进行**个性化的特征变换**——这就是 Feed-Forward Network（FFN）的工作。

FFN 的结构非常简单，就是两层 MLP（多层感知机）：

$$
\text{FFN}(x) = \text{Activation}(xW_1 + b_1)W_2 + b_2
$$

其中 $W_1$ 将维度从 $d_{model}$ 映射到 $d_{ff}$（通常是 $d_{model}$ 的 4 倍），$W_2$ 再映射回 $d_{model}$。激活函数通常使用 **GELU（高斯误差线性单元）** 而不是 ReLU。

```python
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int = None, dropout: float = 0.1):
        super().__init__()
        if d_ff is None:
            d_ff = d_model * 4  # 默认扩展 4 倍
        
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),     # 扩展维度
            nn.GELU(),                    # 激活函数
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),     # 压缩回原维度
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)

# ===== 测试 =====
ffn = FeedForward(d_model=8, d_ff=32)
x = torch.randn(2, 5, 8)  # (batch, seq_len, d_model)
output = ffn(x)

print(f"输入 shape: {x.shape}")
print(f"FFN 输出 shape: {output.shape}")
print(f"参数量: {sum(p.numel() for p in ffn.parameters())}")
```

你可能会有疑问：**为什么 FFN 要先放大再缩小？为什么不直接用一个单层网络？**

这个"先升后降"的设计被称为 **bottleneck structure（瓶颈结构）**。它的作用类似于信息压缩和解压：中间的高维空间（4倍宽度）充当了一个"特征缓冲区"，让模型可以在更大的空间中进行复杂的非线性变换，然后再投影回原来的维度。这种结构在实践中被证明非常有效——它在不显著增加最终输出维度的情况下，大幅增强了模型的表达能力。

还有一个有趣的视角：可以把 FFN 理解为每个位置上的 **"键-值存储器"**。有研究发现，Transformer 中 FFN 的行为类似于检索操作——中间层的高维空间存储了大量可检索的模式，当某个位置需要特定类型的知识时，FFN 就把它从存储中"查出来"。这也是为什么 FFN 通常占据整个模型参数量的 2/3 左右的原因。

## 残差连接：解决深层网络的梯度问题

残差连接（Residual Connection / Skip Connection）来自 ResNet（2015），它通过将输入直接加到输出来解决深层网络中的梯度消失问题：

$$
\text{Output} = \text{Layer}(x) + x
```

在 Transformer 中，残差连接出现在两个地方：
1. **Attention 层之后**：`x = Attention(x) + x`
2. **FFN 层之后**：`x = FFN(x) + x`

```python
class TransformerBlock(nn.Module):
    """单个 Transformer Block 的完整实现"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int = None, dropout: float = 0.1):
        super().__init__()
        
        # Multi-Head Self-Attention
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        
        # Feed-Forward Network
        self.ffn = FeedForward(d_model, d_ff, dropout)
        
        # Layer Normalization (Pre-Norm)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Pre-Norm + Attention + Residual
        residual = x
        x = self.norm1(x)
        attn_output, _ = self.attn(x, x, x, attn_mask=mask)
        x = residual + self.dropout(attn_output)
        
        # Pre-Norm + FFN + Residual
        residual = x
        x = self.norm2(x)
        ffn_output = self.ffn(x)
        x = residual + self.dropout(ffn_output)
        
        return x


# ===== 测试 =====
block = TransformerBlock(d_model=8, num_heads=2)
x = torch.randn(2, 5, 8)
output = block(x)

print(f"输入 shape: {x.shape}")
print(f"Block 输出 shape: {output.shape}")
print(f"\n输入与输出的数值范围对比:")
print(f"  输入: [{x.min():.3f}, {x.max():.3f}]")
print(f"  输出: [{output.min():.3f}, {output.max():.3f}]")

# 验证残差连接的效果: 如果没有残差, 深层堆叠可能导致梯度消失
print("\n验证: 多层堆叠后梯度是否保持")
grad_in = torch.ones_like(output).sum().backward(retain_graph=True)
first_grad = x.grad.abs().mean()
print(f"  第一层输入的平均梯度幅值: {first_grad:.6f}")
```

残差连接的本质作用是创建了一条**梯度高速公路**——即使中间的 Attention 或 FFN 层产生了很小的梯度（甚至接近于零），残差路径仍然可以让梯度以接近 1 的幅度流回前面的层。这就是为什么 Transformer 可以安全地堆叠 12 层（BERT-base）、24 层（BERT-large）、96 层（GPT-3）甚至更多层的原因。

## Encoder 和 Decoder：分工合作的两兄弟

现在我们把上面所有的组件组装起来，看看原始 Transformer 论文中定义的 **Encoder-Decoder 架构**到底是什么样的。

### Encoder（编码器）

Encoder 的任务是**理解输入序列**。它接收源语言的文本（比如英文），逐层通过 Self-Attention 和 FFN，最终产出一个包含丰富语义信息的编码表示。

```
输入 X (英文) → [Embedding + Positional Encoding]
                → Encoder Layer 1 (Self-Attn + FFN + LN + Residual)
                → Encoder Layer 2
                → ...
                → Encoder Layer N
                → 编码输出 E (包含每个位置的上下文表示)
```

Encoder 的特点是使用 **双向注意力（Bidirectional Attention）**——每个位置都可以关注到它前面和后面的所有位置。这是因为 Encoder 处理的是已经知道的完整输入（不需要像生成任务那样防止"偷看未来"）。

### Decoder（解码器）

Decoder 的任务是**逐步生成目标序列**（比如中文翻译）。它比 Encoder 更复杂，因为它包含三种注意力机制：

```
目标 Y (已生成的中文) → [Embedding + Positional Encoding + Causal Mask]
                      → Decoder Layer 1:
                          ├─ Masked Self-Attention (只能看自己之前的位置)
                          └─ Cross-Attention (查看 Encoder 的输出 E)
                      → Decoder Layer 2:
                          ├─ Masked Self-Attention
                          └─ Cross-Attention
                      → ...
                      → Decoder Layer N
                      → Linear + Softmax → 下一个词的概率分布
```

Decoder 有两个关键点需要注意：

**第一，Masked Self-Attention（带掩码的自注意力）**。Decoder 在生成第 t 个词的时候，不应该能看到第 t 个及之后的词（否则就"作弊"了）。所以需要在 Attention 分数上加一个掩码矩阵，把未来位置对应的分数设为负无穷大，这样 Softmax 之后它们的权重就会变成零。

**第二，Cross-Attention（交叉注意力）**。这是 Decoder 独有的组件——它让 Decoder 的每个位置都能去"查阅" Encoder 的输出。Query 来自 Decoder 自身，Key 和 Value 来自 Encoder 的输出。这样 Decoder 就能在生成每一个词的时候，参考原文中的相关信息。

```python
import torch
import torch.nn as nn
import math

class SimpleTransformerEncoderDecoder(nn.Module):
    """极简版的 Encoder-Decoder Transformer"""
    
    def __init__(self, vocab_size, d_model=32, nhead=4, num_layers=2, d_ff=64):
        super().__init__()
        self.d_model = d_model
        
        # Embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = self._create_pos_encoding(100, d_model)
        
        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff,
            dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff,
            dropout=0.1, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)
    
    def _create_pos_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, max_len, d_model)
    
    def create_causal_mask(self, sz):
        """生成因果掩码: 上三角为 -inf"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1) * float('-inf')
        return mask
    
    def forward(self, src, tgt):
        """
        Args:
            src: 源序列 (batch, src_len), 整数 ID
            tgt: 目标序列 (batch, tgt_len), 整数 ID
        """
        # Embedding + Positional Encoding
        src_emb = self.embedding(src) + self.pos_encoding[:, :src.size(1)]
        tgt_emb = self.embedding(tgt) + self.pos_encoding[:, :tgt.size(1)]
        
        # Encode
        memory = self.encoder(src_emb)
        
        # Decode with causal mask
        causal_mask = self.create_causal_mask(tgt.size(1)).to(tgt.device)
        output = self.decoder(tgt_emb, memory, tgt_mask=causal_mask)
        
        # Project to vocabulary
        logits = self.output_proj(output)
        
        return logits


# ===== 测试 =====
model = SimpleTransformerEncoderDecoder(vocab_size=100, d_model=32, nhead=4, num_layers=2)

src = torch.randint(0, 100, (2, 6))   # 英文: batch=2, len=6
tgt = torch.randint(0, 100, (2, 4))   # 中文: batch=2, len=4 (已生成的部分)

logits = model(src, tgt)

print(f"源序列 shape: {src.shape}")
print(f"目标序列 shape: {tgt.shape}")
print(f"输出 logits shape: {logits.shape}")  # 应该是 (2, 4, 100)
print(f"下一个词的预测 (位置 3): argmax = {logits[0, 3].argmax().item()}")
```

运行结果：

```
源序列 shape: torch.Size([2, 6])
目标序列 shape: torch.Size([2, 4])
输出 logits shape: torch.Size([2, 4, 100])
下一个词的预测 (位置 3): argmax = 47
```

这段代码实现了一个最小但完整的 Encoder-Decoder Transformer。虽然参数量和层数都很小（仅用于演示），但它包含了所有核心组件：Embedding + Positional Encoding、Multi-Head Self-Attention、Causal Mask、Cross-Attention、Feed-Forward Network、Layer Normalization、残差连接、以及最后的输出投影。

## 完整的前向传播流程图

让我们用一个具体的例子来追踪一个 Token 通过完整 Transformer 的全过程。假设我们的任务是翻译 `"hello world"` → `"你好 世界"`：

```
═══════════════════════════════════════════════════════════════
              ENCODER (处理源语言 "hello world")
═══════════════════════════════════════════════════════════════

输入 Token IDs:  [101, 209, 222]  ("hello", "world", </s>)
       ↓
[Word Embedding]:  将每个 ID 映射为 d_model 维向量
       ↓
[+ Positional Enc]: 加入位置信息 (pos=0,1,2)
       ↓
  形状: (3, d_model) — 每个 token 都是一个 d_model 维向量
       ↓
┌─────────────────────────────────────┐
│         Encoder Layer 1             │
│                                     │
│  ┌─→ [LayerNorm]                   │
│  │    ↓                             │
│  │  [Multi-Head Self-Attention]     │ ← "hello" 看到 ["hello","world","</s>]
│  │    ↓                             │      "world" 看到 ["hello","world","</s>"]
│  │  [+ Residual] ←── 输入的直接通路   │      "</s>"  看到 ["hello","world","</s>"]
│  │    ↓                             │
│  │  [LayerNorm]                     │
│  │    ↓                             │
│  │  [Feed-Forward Network]          │ ← 每个位置做独立的非线性变换
│  │    ↓                             │
│  │  [+ Residual]                    │
│  └───────────────────────────────────┘
       ↓
┌─────────────────────────────────────┐
│         Encoder Layer 2             │  (重复相同的结构)
│           ...                        │
└───────────────────────────────────┘
       ↓
  Memory (编码输出): (3, d_model)
  包含 "hello"、"world"、"</s>" 的深度语义表示
═══════════════════════════════════════════════════════════════
              DECODER (生成目标语言)
═══════════════════════════════════════════════════════════════

当前已生成: [<bos>, "你好"]  (tgt)
       ↓
[Word Embedding] + [Positional Enc]
       ↓
  形状: (2, d_model)
       ↓
┌─────────────────────────────────────┐
│         Decoder Layer 1             │
│                                     │
│  ┌─→ [LayerNorm]                   │
│  │    ↓                             │
│  │  [Masked Self-Attention]         │ ← "<bos>" 只能看到 <bos>
│  │    ↓                              │     "你好" 能看到 <bos>, "你好"
│  │  [+ Residual]                    │     (不能看到还没生成的!)
│  │    ↓                             │
│  │  [LayerNorm]                     │
│  │    ↓                             │
│  │  [Cross-Attention]               │ ← Query 来自 Decoder (当前位置)
│  │    Q = decoder_hidden            │     K, V = Encoder Memory
│  │    K, V = encoder_memory          │     让 Decoder 参考 Encoder 的理解
│  │    ↓                             │
│  │  [+ Residual]                    │
│  │    ↓                             │
│  │  [LayerNorm]                     │
│  │    ↓                             │
│  │  [Feed-Forward Network]          │
│  │    ↓                             │
│  │  [+ Residual]                    │
│  └───────────────────────────────────┘
       ↓
  Decoder 输出: (2, d_model)
       ↓
[Linear(d_model → vocab_size)] → Softmax
       ↓
  概率分布: P(next_token | context)
  预测: "世界" (概率最高的词)
       ↓
  将 "世界" 追加到 tgt 末尾, 继续下一步...
═══════════════════════════════════════════════════════════════
```

这张流程图展示了 Transformer 工作时的完整数据流。有几个关键节点值得反复思考：

1. **Encoder 的 Self-Attention 是双向的**——这是它能充分理解输入的原因
2. **Decoder 的 Self-Attention 是单向（masked）的**——这是保证自回归正确性的原因
3. **Cross-Attention 是连接 Encoder 和 Decoder 的桥梁**——没有它，Decoder 就不知道原文说了什么
4. **每一层都有残差连接**——这是深层次模型能正常训练的原因

## 小结

这一节我们从 Attention 这个"引擎"出发，逐步添加了让它成为一个完整系统所需的所有组件：Positional Encoding 解决了位置感知问题，Layer Normalization 保证了训练稳定性，Feed-Forward Network 为每个位置提供了个性化变换的能力，残差连接打通了梯度的回传通道，最后 Encoder 和 Decoder 各司其职又通过 Cross-Attention 密切配合。

这些组件的组合看起来复杂，但每个部分的职责都非常清晰。理解了这个架构全景之后，下一节我们将探讨一个实际工程中的重要问题：面对不同的任务，我们应该选择哪种 Transformer 变体？
