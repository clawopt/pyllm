# LLM场景示例：Embedding Lookup综合实践

在前几节中，我们学习了各种索引方式。现在让我们把这些知识综合起来，以LLM中最核心的操作之一——Embedding Lookup为例，展示如何将各种索引技术结合起来解决实际问题。Embedding Lookup是LLM处理文本的第一步：将token IDs转换为稠密的向量表示。理解这个过程不仅能帮助我们掌握NumPy索引技巧，还能深入理解LLM的工作原理。

## 完整的Embedding Lookup实现

让我们从一个简单的场景开始，逐步构建一个完整的Embedding Lookup：

```python
import numpy as np

vocab_size = 10000
embed_dim = 256

# 初始化embedding表（通常使用预训练或随机初始化）
np.random.seed(42)
embedding_table = np.random.randn(vocab_size, embed_dim).astype(np.float32) * 0.02
print(f"Embedding表形状: {embedding_table.shape}")

# Token IDs（假设已经是 tokenized 的文本）
token_ids = np.array([101, 2003, 1014, 1029, 2003, 2997])  # "I like dogs I like"

# 使用花式索引进行embedding lookup
embeddings = embedding_table[token_ids]
print(f"输入token数量: {len(token_ids)}")
print(f"输出embeddings形状: {embeddings.shape}")
```

这段代码展示了最简单的embedding lookup：给定token IDs和embedding表，使用NumPy的花式索引（整数数组索引）直接获取对应的向量。

## 批量Embedding Lookup

实际应用中，处理是批量进行的：

```python
batch_size = 4
seq_len = 20
vocab_size = 30000
embed_dim = 768

# 模拟批量token IDs
np.random.seed(42)
token_ids_batch = np.random.randint(0, vocab_size, size=(batch_size, seq_len))
print(f"批量Token IDs形状: {token_ids_batch.shape}")

# 初始化embedding表
embedding_table = np.random.randn(vocab_size, embed_dim).astype(np.float32) * np.sqrt(2.0 / embed_dim)

# 批量embedding lookup
embeddings_batch = embedding_table[token_ids_batch]
print(f"批量Embeddings形状: {embeddings_batch.shape}")
```

由于 `token_ids_batch` 的形状是 `(batch_size, seq_len)`，所以 `embedding_table[token_ids_batch]` 的结果形状是 `(batch_size, seq_len, embed_dim)`。

## 添加位置编码

Embedding lookup后，通常需要添加位置编码：

```python
def positional_encoding(seq_len, d_model):
    """生成sin/cos位置编码"""
    position = np.arange(seq_len)[:, np.newaxis]
    div_term = np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model)
    div_term = np.exp(div_term)

    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)

    return pe.astype(np.float32)

# 生成位置编码
pos_encoding = positional_encoding(seq_len, embed_dim)
print(f"位置编码形状: {pos_encoding.shape}")

# 添加到embeddings
embeddings_with_pos = embeddings_batch + pos_encoding[np.newaxis, :, :]
print(f"添加位置编码后形状: {embeddings_with_pos.shape}")
```

这里使用了 `pos_encoding[np.newaxis, :, :]` 进行广播，将形状 `(seq_len, embed_dim)` 的位置编码广播到 `(batch_size, seq_len, embed_dim)`。

## 处理Padding

在处理变长序列时，需要创建padding mask：

```python
# 模拟变长序列的实际长度
actual_lengths = np.array([15, 20, 12, 18])  # 这批序列的实际长度

# 创建padding mask：True表示有效位置，False表示padding
padding_mask = np.zeros((batch_size, seq_len), dtype=bool)
for i, length in enumerate(actual_lengths):
    padding_mask[i, :length] = True

print(f"Padding Mask:\n{padding_mask.astype(np.int32)}")

# 在attention计算中应用mask
attention_scores = np.random.randn(batch_size, seq_len, seq_len).astype(np.float32) * 0.1

# 将padding位置的score设为一个很小的负数
masked_scores = np.where(
    padding_mask[:, np.newaxis, :] & padding_mask[:, :, np.newaxis],
    attention_scores,
    -1e9
)
print(f"应用mask后attention scores形状: {masked_scores.shape}")
```

这里使用了两次广播：`padding_mask[:, np.newaxis, :]` 形状变为 `(batch_size, 1, seq_len)`，`padding_mask[:, :, np.newaxis]` 形状变为 `(batch_size, seq_len, 1)`，两者 AND 操作后得到 `(batch_size, seq_len, seq_len)` 的掩码。

## 完整的文本处理流程

让我们把以上内容整合成一个完整的例子：

```python
class SimpleEmbedding:
    """简化的Embedding层"""

    def __init__(self, vocab_size, embed_dim):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        # 使用Xavier初始化
        np.random.seed(42)
        self.embedding_table = np.random.randn(vocab_size, embed_dim).astype(np.float32) * np.sqrt(2.0 / embed_dim)

    def forward(self, token_ids):
        """前向传播：token_ids -> embeddings"""
        return self.embedding_table[token_ids]

    def add_positional_encoding(self, embeddings):
        """添加位置编码"""
        seq_len = embeddings.shape[1]
        pos_encoding = positional_encoding(seq_len, self.embed_dim)
        return embeddings + pos_encoding[np.newaxis, :, :]

    def create_padding_mask(self, token_ids, pad_token_id=0):
        """创建padding mask"""
        return token_ids != pad_token_id

# 创建embedding层
embed_layer = SimpleEmbedding(vocab_size=10000, embed_dim=256)

# 输入token IDs
token_ids = np.array([[101, 2003, 1014, 1029, 2003, 2997, 0, 0]])  # 含padding

# 获取embeddings
embeddings = embed_layer.forward(token_ids)
print(f"Embeddings形状: {embeddings.shape}")

# 添加位置编码
embeddings_with_pos = embed_layer.add_positional_encoding(embeddings)
print(f"带位置编码的形状: {embeddings_with_pos.shape}")

# 创建padding mask
padding_mask = embed_layer.create_padding_mask(token_ids)
print(f"Padding Mask: {padding_mask.astype(np.int32)}")
```

## 从隐藏状态中提取特定信息

在分析LLM输出时，经常需要提取特定位置或特定头的隐藏状态：

```python
batch_size = 2
num_heads = 12
seq_len = 512
head_dim = 64
hidden_dim = num_heads * head_dim

# 模拟多头注意力的输出
mha_output = np.random.randn(batch_size, seq_len, hidden_dim).astype(np.float32)

# 提取特定头的信息（例如头0、3、6、9）
target_heads = np.array([0, 3, 6, 9])
head_indices = target_heads * head_dim + np.arange(head_dim)  # 展平后的索引
selected_heads = mha_output[:, :, head_indices]
print(f"选中头的形状: {selected_heads.shape}")

# 提取关键位置的hidden states
key_positions = np.array([0, 1, seq_len-1])  # [CLS], 第一个token, 最后一个token
key_hidden = mha_output[:, key_positions, :]
print(f"关键位置的hidden shape: {key_hidden.shape}")

# 组合：选中头在关键位置的hidden
target_head_indices = np.array([h * head_dim + np.arange(head_dim) for h in target_heads]).flatten()
key_head_hidden = mha_output[:, key_positions][:, :, target_head_indices]
print(f"关键位置+选中头的形状: {key_head_hidden.shape}")
```

## 常见的性能优化

### 使用视图而非副本

某些操作可以避免不必要的数据复制：

```python
# 场景：需要频繁访问embedding表的特定行
embedding_table = np.random.randn(30000, 768).astype(np.float32)

# 选取特定的vocab subset
sub_vocab_ids = np.array([100, 200, 300, 400, 500])
sub_embeddings = embedding_table[sub_vocab_ids]  # 这是副本

# 如果不需要修改，使用view更高效
# 注意：view只能用于连续访问的情况
sub_view = embedding_table[100:501]  # 范围切片是view
print(f"使用view: {sub_view.shape}")
```

### 避免在循环中进行索引

```python
# 低效：在循环中逐个索引
batch_size = 32
seq_len = 512
vocab_size = 30000

token_ids = np.random.randint(0, vocab_size, size=(batch_size, seq_len))
embed_dim = 768

# 不要这样做：
# for i in range(batch_size):
#     embeddings[i] = embedding_table[token_ids[i]]

# 高效：直接批量索引
embedding_table = np.random.randn(vocab_size, embed_dim).astype(np.float32)
embeddings = embedding_table[token_ids]  # 一次完成
print(f"批量lookup结果形状: {embeddings.shape}")
```

## 小结

这一节我们综合运用了前几节学到的各种索引技术，实现了LLM中最核心的Embedding Lookup操作。从基本的token到embedding的转换，到批量处理、添加位置编码、处理padding mask，再到从隐藏状态中提取特定信息，这些操作构成了理解和实现LLM的基础。

关键技术点：
1. **花式索引**用于从embedding表中批量获取token向量
2. **广播**用于添加位置编码和处理掩码
3. **布尔索引/掩码**用于处理变长序列的padding
4. **切片和花式索引组合**用于提取特定的hidden states

面试时需要能够完整描述一个Embedding Lookup的实现流程，理解各种索引方式的适用场景，以及注意性能优化的方法。
