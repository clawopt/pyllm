# LLM场景示例：位置编码广播综合实践

在前两节中，我们学习了广播规则和不同形状数组的自动扩展。在这一节中，让我们通过位置编码这个LLM中的核心组件来综合实践这些知识。位置编码是Transformer架构中不可或缺的组成部分，因为自注意力机制本身并不包含位置信息——它是"位置无关"的。通过添加位置编码，模型才能知道序列中每个token的位置。理解位置编码的实现不仅能帮助我们掌握NumPy的广播技巧，还能深入理解LLM的工作原理。

## 位置编码的作用

在Transformer出现之前，RNN通过其递归结构天然地处理了序列位置信息。但Transformer使用的是自注意力机制，注意力分数只与内容相关，与位置无关。换句话说，如果把序列中的token打乱，注意力分数矩阵只是相应地打乱了，但计算结果的结构完全相同。位置编码就是为了解决这个问题而引入的。

## 生成位置编码

让我们实现经典的正弦/余弦位置编码：

```python
import numpy as np

def positional_encoding(seq_len, d_model):
    """生成 sin/cos 位置编码

    根据 "Attention Is All You Need" 论文中的公式：
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Args:
        seq_len: 序列长度
        d_model: 模型隐藏层维度

    Returns:
        位置编码矩阵，形状 (seq_len, d_model)
    """
    position = np.arange(seq_len)[:, np.newaxis]  # (seq_len, 1)
    div_term = np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model)  # (d_model/2,)
    div_term = np.exp(div_term)  # (d_model/2,)

    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(position * div_term)  # 偶数维度
    pe[:, 1::2] = np.cos(position * div_term)  # 奇数维度

    return pe.astype(np.float32)

# 测试位置编码
seq_len = 50
d_model = 128
pe = positional_encoding(seq_len, d_model)

print(f"位置编码形状: {pe.shape}")
print(f"位置 0 的编码前 10 维: {pe[0, :10]}")
print(f"位置 25 的编码前 10 维: {pe[25, :10]}")
```

## 位置编码的广播

现在，让我们将位置编码添加到词嵌入中。这是最典型的广播应用场景：

```python
batch_size = 4
seq_len = 50
embed_dim = 128

# 1. 模拟词嵌入 (batch, seq, embed_dim)
word_embeddings = np.random.randn(batch_size, seq_len, embed_dim).astype(np.float32) * 0.02
print(f"词嵌入形状: {word_embeddings.shape}")

# 2. 位置编码 (seq, embed_dim)
position_encoding = positional_encoding(seq_len, embed_dim)
print(f"位置编码形状: {position_encoding.shape}")

# 3. 将位置编码添加到词嵌入
# 问题：word_embeddings 是 (batch, seq, embed_dim)
#       position_encoding 是 (seq, embed_dim)
#       两者形状不同，如何相加？

# 答案：使用广播！
# position_encoding 会被广播为 (1, seq, embed_dim)，然后扩展到 (batch, seq, embed_dim)
embeddings_with_pos = word_embeddings + position_encoding
print(f"相加后形状: {embeddings_with_pos.shape}")
```

这里发生的神奇的事情是：NumPy自动将 (seq, embed_dim) 的位置编码广播到了 (batch, seq, embed_dim)，无需我们显式地复制数据。

## 广播过程的详细解析

让我们详细分析一下这个广播过程：

```python
# word_embeddings: (batch=4, seq=50, embed=128)
# position_encoding: (seq=50, embed=128)

# 从右侧开始比较维度：
# embed: 128 vs 128 ✓ 相同
# seq: 50 vs 50 ✓ 相同
# batch: 4 vs (position_encoding没有batch维度，视为1) ✓ 兼容

# 所以 position_encoding 被视为形状 (1, 50, 128)
# 然后在 batch 维度上复制 4 份

# 实际上这不是真正的复制，而是通过 strides 实现的"假装"复制
# 但这对于使用者来说是透明的
```

## 使用 np.newaxis 显式广播

有时候，显式地使用 `np.newaxis` 可以让代码更清晰：

```python
# 方法1：隐式广播（上面的方式）
result1 = word_embeddings + position_encoding

# 方法2：显式添加维度
position_encoding_expanded = position_encoding[np.newaxis, :, :]  # (1, seq, embed)
result2 = word_embeddings + position_encoding_expanded

# 验证两种方式结果相同
print(f"两种方式结果相同: {np.allclose(result1, result2)}")
```

显式添加维度在某些情况下可以让代码意图更清晰。

## 处理不同批次大小的广播

在实际应用中，不同批次可能有不同的序列长度。让我看看如何处理这种情况：

```python
def add_positional_encoding(embeddings, max_seq_len=None):
    """将位置编码添加到嵌入向量

    Args:
        embeddings: (batch, seq_len, embed_dim)
        max_seq_len: 如果指定，会生成到该长度的位置编码并截取

    Returns:
        添加位置编码后的嵌入
    """
    batch_size, seq_len, embed_dim = embeddings.shape

    # 生成位置编码
    pe = positional_encoding(seq_len, embed_dim)

    # 广播添加
    return embeddings + pe

# 测试
batch_size = 4
seq_len = 30
embed_dim = 128

embeddings = np.random.randn(batch_size, seq_len, embed_dim).astype(np.float32)
result = add_positional_encoding(embeddings)
print(f"输入形状: {embeddings.shape}")
print(f"输出形状: {result.shape}")
```

## 位置编码在自注意力中的应用

位置编码不仅仅用于添加到嵌入向量，在一些变体中还会用在其他地方：

```python
def relative_position_encoding(seq_len, head_dim):
    """生成相对位置编码（用于Transformer-XL等）"""
    # 生成相对位置的索引
    positions = np.arange(seq_len)[:, np.newaxis] - np.arange(seq_len)[np.newaxis, :]
    # positions[i, j] = i - j 表示位置 i 和位置 j 的相对距离

    # 应用 sin/cos 编码
    div_term = np.exp(np.arange(0, head_dim, 2) * -(np.log(10000.0) / head_dim))

    pe = np.zeros((seq_len, seq_len, head_dim))
    pe[:, :, 0::2] = np.sin(positions[:, :, np.newaxis] * div_term)
    pe[:, :, 1::2] = np.cos(positions[:, :, np.newaxis] * div_term)

    return pe.astype(np.float32)

# 测试相对位置编码
seq_len = 10
head_dim = 64
rel_pe = relative_position_encoding(seq_len, head_dim)
print(f"相对位置编码形状: {rel_pe.shape}")
print(f"相对位置矩阵（前5个位置）:\n{rel_pe[:5, :5, 0]}")
```

## 批量处理中的广播技巧

在训练时，我们通常需要批量处理句子。让我展示一些实用的广播技巧：

```python
# 场景：计算每个位置的加权位置编码
batch_size = 8
seq_len = 50
embed_dim = 128

# 位置权重：每个位置有不同的重要性（假设）
position_weights = np.random.rand(batch_size, seq_len, 1).astype(np.float32)
position_weights = position_weights / position_weights.sum(axis=1, keepdims=True)

# 位置编码
pe = positional_encoding(seq_len, embed_dim)

# 加权位置编码
weighted_pe = pe * position_weights  # (seq, embed) * (batch, seq, 1) -> (batch, seq, embed)
print(f"加权位置编码形状: {weighted_pe.shape}")

# 再加上原始的词嵌入
word_embeddings = np.random.randn(batch_size, seq_len, embed_dim).astype(np.float32) * 0.02
final_embeddings = word_embeddings + weighted_pe
print(f"最终嵌入形状: {final_embeddings.shape}")
```

## 学习式位置编码 vs 固定位置编码

现代LLM（如GPT）通常使用学习式位置编码，而不是固定的sin/cos编码。让我展示如何实现：

```python
class LearnedPositionalEncoding:
    """学习式位置编码"""

    def __init__(self, max_seq_len, embed_dim):
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim

        # 初始化位置嵌入（可学习参数）
        np.random.seed(42)
        self.position_embeddings = np.random.randn(max_seq_len, embed_dim).astype(np.float32) * 0.02

    def forward(self, seq_len):
        """获取位置嵌入

        Args:
            seq_len: 当前序列长度

        Returns:
            位置嵌入 (seq_len, embed_dim)
        """
        return self.position_embeddings[:seq_len]

    def add_to_embeddings(self, word_embeddings):
        """添加到词嵌入

        Args:
            word_embeddings: (batch, seq_len, embed_dim)

        Returns:
            添加位置编码后的嵌入
        """
        batch_size, seq_len, embed_dim = word_embeddings.shape
        pos_emb = self.forward(seq_len)  # (seq_len, embed_dim)
        # 广播：pos_emb 自动扩展到 (batch, seq_len, embed_dim)
        return word_embeddings + pos_emb

# 测试学习式位置编码
batch_size = 4
seq_len = 50
embed_dim = 128

learned_pe = LearnedPositionalEncoding(max_seq_len=512, embed_dim=embed_dim)
word_embeddings = np.random.randn(batch_size, seq_len, embed_dim).astype(np.float32) * 0.02

result = learned_pe.add_to_embeddings(word_embeddings)
print(f"学习式位置编码输出形状: {result.shape}")
```

## 常见问题与解决方案

### 序列长度超过预生成位置编码

```python
# 如果位置编码是预生成的，序列超过时会出错
pe_precomputed = positional_encoding(100, 128)  # 只生成了100个位置

seq_len = 150  # 需要150个位置
embeddings = np.random.randn(4, seq_len, 128).astype(np.float32)

# 直接相加会报错
try:
    result = embeddings + pe_precomputed
except ValueError as e:
    print(f"长度不匹配: {e}")

# 解决方案：动态生成或截断
pe_needed = positional_encoding(seq_len, 128)  # 动态生成需要的数量
result = embeddings + pe_needed
print(f"动态生成后的形状: {result.shape}")
```

### dtype 不一致

```python
# 位置编码是 float32，嵌入是 float64
pe = positional_encoding(50, 128)  # float32
embeddings = np.random.randn(4, 50, 128).astype(np.float64)  # float64

# 相加时 dtype 会向上转换
result = embeddings + pe
print(f"结果 dtype: {result.dtype}")  # float64
```

## 小结

这一节我们通过位置编码这个实际案例综合运用了广播的知识。关键点包括：

1. **位置编码生成**：使用 sin/cos 函数生成固定的位置编码矩阵
2. **广播原理**：位置编码 (seq, embed) 如何自动广播到 (batch, seq, embed)
3. **显式 vs 隐式广播**：使用 np.newaxis 让广播意图更清晰
4. **学习式位置编码**：作为对比，展示了可学习的位置编码实现

位置编码是理解LLM架构的重要知识点，而广播是实现它的关键技术。

面试时需要能够解释位置编码的生成原理，以及详细描述位置编码如何通过广播添加到词嵌入中。
