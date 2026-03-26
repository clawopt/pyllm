# 词向量查找

词向量查找（Embedding Lookup）是 LLM 中的基础操作，它根据 token 的索引从嵌入矩阵中取出对应的向量。本质上，这是一个简单的索引查表操作，在 NumPy 中通过数组索引实现。虽然操作简单，但在大规模 LLM 中，嵌入查找往往是计算瓶颈之一，理解其实现对于优化模型性能很重要。

## 嵌入查找的基本原理

给定嵌入矩阵 E（形状为 (V, D)，V 是词汇表大小，D 是嵌入维度）和 token IDs，嵌入查找返回 E[token_ids]。

```python
import numpy as np

def embedding_lookup(embeddings, token_ids):
    """词向量查找

    参数:
        embeddings: 嵌入矩阵 (vocab_size, embed_dim)
        token_ids: token ID 数组 (batch_size, seq_len) 或 (seq_len,)
    返回:
        vectors: 嵌入向量 (batch_size, seq_len, embed_dim)
    """
    return embeddings[token_ids]

# 示例
np.random.seed(42)
embeddings = np.random.randn(10000, 768).astype(np.float32)

# 单个 token
token_id = 100
vector = embedding_lookup(embeddings, token_id)
print(f"单个 token 向量形状: {vector.shape}")

# 多个 token
token_ids = np.array([0, 1, 2, 3, 4])
vectors = embedding_lookup(embeddings, token_ids)
print(f"多个 token 向量形状: {vectors.shape}")

# Batch
batch_token_ids = np.array([[0, 1, 2], [3, 4, 5]])
batch_vectors = embedding_lookup(embeddings, batch_token_ids)
print(f"Batch 向量形状: {batch_vectors.shape}")
```

## 完整的嵌入层实现

将嵌入查找封装成可复用的类：

```python
class EmbeddingLookup:
    """词嵌入查找层"""

    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.vocab_size = embeddings.shape[0]
        self.embedding_dim = embeddings.shape[1]

    def forward(self, token_ids):
        """前向传播

        参数:
            token_ids: token ID 数组，支持任意形状
        返回:
            embeddings: 嵌入向量，形状为 token_ids.shape + (embed_dim,)
        """
        return self.embeddings[token_ids]

    def __call__(self, token_ids):
        """支持像函数一样调用"""
        return self.forward(token_ids)

    def __getitem__(self, token_ids):
        """支持索引访问"""
        return self.forward(token_ids)

# 示例
np.random.seed(42)
embeddings = np.random.randn(10000, 768).astype(np.float32)
lookup = EmbeddingLookup(embeddings)

# 测试
token_ids = np.array([0, 1, 2, 3, 4])
vectors = lookup(token_ids)
print(f"查找结果形状: {vectors.shape}")
```

## 批量嵌入查找

在训练时，通常需要批量处理：

```python
class BatchEmbeddingLookup:
    """支持批量处理的嵌入查找"""

    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.vocab_size = embeddings.shape[0]
        self.embedding_dim = embeddings.shape[1]

    def forward(self, token_ids, padding_mask=None):
        """批量嵌入查找

        参数:
            token_ids: (batch_size, seq_len)
            padding_mask: (batch_size, seq_len)，True 表示有效位置
        返回:
            embeddings: (batch_size, seq_len, embed_dim)
        """
        # 基本嵌入查找
        embeddings = self.embeddings[token_ids]

        # 如果有 padding mask，可以对 padding 位置清零
        if padding_mask is not None:
            embeddings = embeddings * padding_mask[:, :, np.newaxis]

        return embeddings

    def get_embeddings_slice(self, token_ids, start_pos, end_pos):
        """获取嵌入的切片（用于自回归生成）

        参数:
            token_ids: (batch_size, seq_len)
            start_pos: 起始位置
            end_pos: 结束位置
        返回:
            embeddings: (batch_size, end_pos - start_pos, embed_dim)
        """
        return self.embeddings[token_ids[:, start_pos:end_pos]]

# 示例
batch_lookup = BatchEmbeddingLookup(embeddings)
batch_ids = np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
embeddings_out = batch_lookup.forward(batch_ids)
print(f"批量查找结果形状: {embeddings_out.shape}")

# 带 mask 的查找
padding_mask = np.array([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]])
embeddings_masked = batch_lookup.forward(batch_ids, padding_mask)
print(f"带 mask 的结果形状: {embeddings_masked.shape}")
```

## 处理未知 token

当遇到词汇表中不存在的 token 时，通常返回 UNK 的嵌入或零向量：

```python
class SafeEmbeddingLookup:
    """安全的嵌入查找，处理未知 token"""

    def __init__(self, embeddings, unk_id=0):
        self.embeddings = embeddings
        self.unk_id = unk_id
        self.vocab_size = embeddings.shape[0]

    def forward(self, token_ids):
        """安全的嵌入查找

        参数:
            token_ids: token ID 数组
        返回:
            embeddings: 嵌入向量
        """
        # 将超出词汇表范围的 ID 替换为 UNK ID
        safe_ids = np.clip(token_ids, 0, self.vocab_size - 1)
        return self.embeddings[safe_ids]

    def forward_with_unk(self, token_ids, unk_token_id=None):
        """将指定的 token ID 替换为 UNK

        参数:
            token_ids: token ID 数组
            unk_token_id: 要替换为 UNK 的 token ID 列表
        返回:
            embeddings: 嵌入向量
        """
        safe_ids = token_ids.copy()
        if unk_token_id is not None:
            safe_ids = np.where(
                np.isin(token_ids, unk_token_id),
                self.unk_id,
                token_ids
            )
        return self.embeddings[safe_ids]

# 示例
safe_lookup = SafeEmbeddingLookup(embeddings, unk_id=0)
token_ids_with_unknown = np.array([0, 1, 99999, 3, 4])  # 99999 超出范围
safe_vectors = safe_lookup.forward(token_ids_with_unknown)
print(f"安全查找结果形状: {safe_vectors.shape}")
```

## 嵌入查找的变体

### 带权重的嵌入查找

```python
class WeightedEmbeddingLookup:
    """带权重的嵌入查找"""

    def __init__(self, embeddings):
        self.embeddings = embeddings

    def forward(self, token_ids, weights=None):
        """加权嵌入查找

        参数:
            token_ids: (batch_size, seq_len)
            weights: (batch_size, seq_len)，每个 token 的权重
        返回:
            weighted_embeddings: (batch_size, seq_len, embed_dim)
        """
        embeddings = self.embeddings[token_ids]

        if weights is not None:
            # 应用权重
            embeddings = embeddings * weights[:, :, np.newaxis]

        return embeddings

    def weighted_sum(self, token_ids, weights):
        """加权求和，返回单个向量

        参数:
            token_ids: (batch_size, seq_len)
            weights: (batch_size, seq_len)
        返回:
            weighted_sum: (batch_size, embed_dim)
        """
        embeddings = self.embeddings[token_ids]
        weighted_emb = embeddings * weights[:, :, np.newaxis]
        return np.sum(weighted_emb, axis=1)

# 示例
weighted_lookup = WeightedEmbeddingLookup(embeddings)
token_ids = np.array([[0, 1, 2, 3, 4]])
weights = np.array([[0.1, 0.2, 0.3, 0.2, 0.2]])
weighted_sum = weighted_lookup.weighted_sum(token_ids, weights)
print(f"加权求和形状: {weighted_sum.shape}")
```

### 上下文相关的嵌入

某些模型（如 ELMo）使用上下文相关的嵌入：

```python
def context_aware_lookup(token_embeddings, hidden_states, projection_matrix):
    """简化的上下文相关嵌入

    参数:
        token_embeddings: 基础嵌入 (batch, seq_len, embed_dim)
        hidden_states: 来自其他层的隐藏状态 (batch, seq_len, hidden_dim)
        projection_matrix: 投影矩阵 (hidden_dim, embed_dim)
    返回:
        context_embeddings: (batch, seq_len, embed_dim)
    """
    # 将隐藏状态投影到嵌入空间
    projected = np.einsum('bsd,hd->bsh', hidden_states, projection_matrix)

    # 与基础嵌入结合
    context_embeddings = token_embeddings + projected

    return context_embeddings
```

## 性能优化

### 向量化批量查找

NumPy 的数组索引本身就是高度向量化的：

```python
import time

np.random.seed(42)
embeddings = np.random.randn(50000, 768).astype(np.float32)

# 创建测试数据
batch_size = 32
seq_len = 512
token_ids = np.random.randint(0, 50000, size=(batch_size, seq_len))

# 测量查找时间
start = time.time()
for _ in range(100):
    result = embeddings[token_ids]
lookup_time = time.time() - start

print(f"嵌入查找耗时: {lookup_time*10:.2f}ms（100次平均）")
print(f"结果形状: {result.shape}")
print(f"内存带宽: {result.nbytes * 100 / lookup_time / 1e9:.2f} GB/s")
```

### 避免不必要的数据复制

切片操作可能返回视图或副本：

```python
# 普通索引返回视图
token_ids = np.array([0, 1, 2, 3, 4])
vectors = embeddings[token_ids]  # 返回视图或副本，取决于连续性

# 确保结果是连续的（如果是副本）
if not vectors.flags['C_CONTIGUOUS']:
    vectors = np.ascontiguousarray(vectors)
```

## 与 PyTorch 的对比

PyTorch 的 `nn.Embedding` 使用非常高效：

```python
# PyTorch 等效代码（伪代码）
# import torch
# embedding = torch.nn.Embedding(vocab_size, embed_dim)
# embeddings = embedding(token_ids)  # token_ids 是 torch.Tensor
#
# NumPy 实现
# embeddings = numpy_embeddings[token_ids]  # 直接索引
#
# 两者语义相同，但 PyTorch 会在内部维护索引到嵌入的映射
```

## 常见误区

**误区一：混淆索引维度**

token_ids 的形状决定了输出嵌入的形状：

```python
# (seq_len,) -> (seq_len, embed_dim)
token_ids_1d = np.array([0, 1, 2])
vectors_1d = embeddings[token_ids_1d]
print(f"1D 输入: {token_ids_1d.shape} -> {vectors_1d.shape}")

# (batch, seq_len) -> (batch, seq_len, embed_dim)
token_ids_2d = np.array([[0, 1, 2]])
vectors_2d = embeddings[token_ids_2d]
print(f"2D 输入: {token_ids_2d.shape} -> {vectors_2d.shape}")
```

**误区二：忽略 dtype 不匹配**

嵌入矩阵和 token IDs 的 dtype 可能不匹配：

```python
# 嵌入是 float32，token_ids 是 int64
embeddings_f32 = np.random.randn(10000, 768).astype(np.float32)
token_ids = np.array([0, 1, 2], dtype=np.int64)

# NumPy 会自动处理，但最好显式转换
vectors = embeddings_f32[token_ids]
print(f"向量 dtype: {vectors.dtype}")
```

**误区三：批量查找时忽视内存布局**

批量查找时，保持 C 连续布局可以提高后续操作的效率：

```python
# 确保结果是 C 连续的
vectors = embeddings[token_ids]
if not vectors.flags['C_CONTIGUOUS']:
    vectors = np.ascontiguousarray(vectors)
```

## API 总结

| 方法 | 描述 |
|------|------|
| `embeddings[token_ids]` | 基本嵌入查找 |
| `lookup.forward(token_ids)` | 封装后的前向传播 |
| `lookup.forward_with_unk()` | 安全查找，处理未知 token |
| `np.ascontiguousarray()` | 确保连续内存布局 |

词向量查找是 LLM 的基础操作，虽然实现简单，但在大规模应用中性能优化很重要。掌握这些技巧能帮助你更好地理解语言模型的内部机制。
