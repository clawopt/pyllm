# 创建词嵌入矩阵

词嵌入矩阵（Embedding Matrix）是 LLM 中将离散 token 映射为连续向量的核心组件。矩阵的每一行对应一个 token 的嵌入向量，整个矩阵的形状是 `(vocab_size, embedding_dim)`。本篇文章介绍如何使用 NumPy 创建词嵌入矩阵，包括随机初始化、预训练向量的加载，以及嵌入矩阵的各种操作。

## 嵌入矩阵的基本概念

嵌入矩阵 E 的形状是 (V, D)，其中 V 是词汇表大小，D 是嵌入维度（也叫 hidden size）。给定一个 token 的索引 i，嵌入向量就是 E[i]。

```python
import numpy as np

# 嵌入矩阵示例
vocab_size = 10000
embedding_dim = 768

# 创建一个简单的嵌入矩阵
E = np.random.randn(vocab_size, embedding_dim).astype(np.float32)
print(f"嵌入矩阵形状: {E.shape}")
print(f"嵌入向量形状: {E[0].shape}")
```

## 随机初始化嵌入矩阵

嵌入矩阵需要初始化。现代神经网络使用合适的初始化方法避免梯度问题：

```python
def xavier_init(shape):
    """Xavier 初始化"""
    fan_in, fan_out = shape[0], shape[1]
    std = np.sqrt(2.0 / (fan_in + fan_out))
    return np.random.randn(shape[0], shape[1]).astype(np.float32) * std

def normal_init(shape, std=0.02):
    """正态分布初始化（GPT-2 使用）"""
    return np.random.randn(shape[0], shape[1]).astype(np.float32) * std

def uniform_init(shape, range=0.1):
    """均匀分布初始化"""
    return np.random.uniform(-range, range, shape).astype(np.float32)
```

## 创建嵌入矩阵类

```python
class EmbeddingLayer:
    """简单的嵌入层实现"""

    def __init__(self, vocab_size, embedding_dim, init_method='normal'):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        # 根据初始化方法创建嵌入矩阵
        if init_method == 'xavier':
            self.weight = xavier_init((vocab_size, embedding_dim))
        elif init_method == 'normal':
            self.weight = normal_init((vocab_size, embedding_dim), std=0.02)
        elif init_method == 'uniform':
            self.weight = uniform_init((vocab_size, embedding_dim), range=0.1)
        else:
            raise ValueError(f"Unknown init method: {init_method}")

    def forward(self, token_ids):
        """前向传播：查找嵌入向量

        参数:
            token_ids: token ID 数组 (batch_size, seq_len) 或 (seq_len,)
        返回:
            embeddings: 嵌入向量 (batch_size, seq_len, embed_dim)
        """
        return self.weight[token_ids]

    def __getitem__(self, token_ids):
        """支持直接索引访问"""
        return self.forward(token_ids)

# 示例
np.random.seed(42)
embedding_layer = EmbeddingLayer(vocab_size=10000, embedding_dim=768, init_method='normal')
print(f"嵌入矩阵形状: {embedding_layer.weight.shape}")
print(f"嵌入矩阵均值: {embedding_layer.weight.mean():.6f}")
print(f"嵌入矩阵标准差: {embedding_layer.weight.std():.6f}")
```

## 添加特殊 token 的嵌入

特殊 token（如 PAD、UNK）通常需要特殊处理：

```python
def initialize_special_tokens(embedding_layer, special_token_ids):
    """初始化特殊 token 的嵌入

    参数:
        embedding_layer: EmbeddingLayer 实例
        special_token_ids: 特殊 token 的 ID 字典
    """
    embedding_dim = embedding_layer.embedding_dim

    # PAD token 通常初始化为零向量
    if 'pad' in special_token_ids:
        pad_id = special_token_ids['pad']
        embedding_layer.weight[pad_id] = np.zeros(embedding_dim, dtype=np.float32)

    # UNK token 可以使用平均嵌入或其他策略
    # 这里简化为小随机值
    if 'unk' in special_token_ids:
        unk_id = special_token_ids['unk']
        embedding_layer.weight[unk_id] = np.random.randn(embedding_dim).astype(np.float32) * 0.01

    return embedding_layer

# 示例
special_ids = {'pad': 0, 'unk': 1, 'bos': 2, 'eos': 3}
embedding_layer = initialize_special_tokens(embedding_layer, special_ids)

# 验证 PAD token 是零向量
print(f"PAD 嵌入是零向量: {np.allclose(embedding_layer.weight[0], 0)}")
```

## 预训练嵌入的加载

在实践中，嵌入矩阵通常从预训练模型加载：

```python
def load_pretrained_embeddings(filepath, vocab, embedding_dim):
    """加载预训练的嵌入向量

    参数:
        filepath: 嵌入文件路径
        vocab: Vocabulary 实例
        embedding_dim: 嵌入维度
    返回:
        embeddings: 嵌入矩阵 (vocab_size, embedding_dim)
    """
    # 假设文件格式：每行一个 token 和它的嵌入向量
    # token<TAB>vec1,vec2,...
    embeddings = np.random.randn(len(vocab), embedding_dim).astype(np.float32) * 0.01

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    token, vec_str = parts
                    if token in vocab.token_to_idx:
                        vec = np.fromstring(vec_str, sep=',', dtype=np.float32)
                        if len(vec) == embedding_dim:
                            idx = vocab.token_to_idx[token]
                            embeddings[idx] = vec
    except FileNotFoundError:
        print(f"文件不存在，使用随机初始化: {filepath}")

    return embeddings

# 示例
# embeddings = load_pretrained_embeddings('glove.100d.txt', vocab, 100)
```

## 嵌入矩阵的操作

### 提取子嵌入矩阵

```python
def get_subembedding(embeddings, token_ids):
    """提取子嵌入矩阵

    参数:
        embeddings: 完整嵌入矩阵 (vocab_size, embed_dim)
        token_ids: token ID 数组
    返回:
        sub_embeddings: (len(token_ids), embed_dim)
    """
    return embeddings[token_ids]

# 示例
np.random.seed(42)
embeddings = np.random.randn(10000, 768).astype(np.float32)
token_ids = np.array([0, 1, 2, 3, 4])
sub_emb = get_subembedding(embeddings, token_ids)
print(f"子嵌入矩阵形状: {sub_emb.shape}")
```

### 嵌入向量的归一化

```python
def normalize_embeddings(embeddings, axis=1):
    """L2 归一化嵌入向量

    参数:
        embeddings: 嵌入矩阵 (..., embed_dim)
        axis: 归一化的轴
    返回:
        normalized: 归一化后的嵌入矩阵
    """
    norms = np.linalg.norm(embeddings, axis=axis, keepdims=True)
    return embeddings / (norms + 1e-8)

# 示例
normalized = normalize_embeddings(embeddings, axis=1)
print(f"归一化后向量范数: {np.linalg.norm(normalized[0]):.6f}")
```

### 计算嵌入矩阵的统计量

```python
def embedding_statistics(embeddings):
    """计算嵌入矩阵的统计量

    返回:
        stats: 统计字典
    """
    stats = {
        'mean': np.mean(embeddings),
        'std': np.std(embeddings),
        'min': np.min(embeddings),
        'max': np.max(embeddings),
        'norm_mean': np.mean(np.linalg.norm(embeddings, axis=1)),
        'norm_std': np.std(np.linalg.norm(embeddings, axis=1)),
    }
    return stats

# 示例
stats = embedding_statistics(embeddings)
print("嵌入矩阵统计:")
for name, value in stats.items():
    print(f"  {name}: {value:.4f}")
```

## 在 LLM 中的使用

### 词嵌入与位置嵌入的结合

```python
class TokenAndPositionEmbedding:
    """词嵌入 + 位置嵌入"""

    def __init__(self, vocab_size, max_len, embed_dim):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        # 词嵌入
        self.token_embedding = EmbeddingLayer(vocab_size, embed_dim, init_method='normal')

        # 位置嵌入（初始化为零）
        self.position_embedding = np.zeros((max_len, embed_dim), dtype=np.float32)

    def forward(self, token_ids):
        """前向传播

        参数:
            token_ids: (batch_size, seq_len)
        返回:
            embeddings: (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len = token_ids.shape

        # 词嵌入
        token_emb = self.token_embedding.forward(token_ids)

        # 位置嵌入
        position_ids = np.arange(seq_len, dtype=np.int32)
        position_emb = self.position_embedding[position_ids]

        # 相加
        embeddings = token_emb + position_emb

        return embeddings

# 示例
np.random.seed(42)
combined_embedding = TokenAndPositionEmbedding(vocab_size=10000, max_len=512, embed_dim=768)
token_ids = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
embeddings = combined_embedding.forward(token_ids)
print(f"输入形状: {token_ids.shape}")
print(f"嵌入形状: {embeddings.shape}")
```

### 嵌入 Dropout

训练时对嵌入向量应用 Dropout：

```python
class EmbeddingDropout:
    """嵌入 Dropout"""

    def __init__(self, embed_layer, dropout_rate=0.1):
        self.embed_layer = embed_layer
        self.dropout_rate = dropout_rate

    def forward(self, token_ids, training=True):
        """前向传播

        参数:
            token_ids: token ID 数组
            training: 是否在训练模式
        """
        embeddings = self.embed_layer.forward(token_ids)

        if training and self.dropout_rate > 0:
            mask = np.random.binomial(1, 1 - self.dropout_rate, embeddings.shape)
            embeddings = embeddings * mask / (1 - self.dropout_rate)

        return embeddings

# 示例
dropout_layer = EmbeddingDropout(embedding_layer, dropout_rate=0.1)
embeddings = dropout_layer.forward(np.array([[1, 2, 3]]), training=True)
print(f"Dropout 后形状: {embeddings.shape}")
```

## 内存管理

嵌入矩阵通常占用大量内存：

```python
def estimate_embedding_memory(vocab_size, embedding_dim, dtype=np.float32):
    """估算嵌入矩阵的内存占用

    参数:
        vocab_size: 词汇表大小
        embedding_dim: 嵌入维度
        dtype: 数据类型
    返回:
        memory_mb: 内存占用（MB）
    """
    bytes_per_element = np.dtype(dtype).itemsize
    total_bytes = vocab_size * embedding_dim * bytes_per_element
    return total_bytes / 1024 / 1024

# 示例
vocab_sizes = [10000, 30000, 50257, 100000]
embed_dim = 768

print("嵌入矩阵内存占用:")
for vs in vocab_sizes:
    mem = estimate_embedding_memory(vs, embed_dim)
    print(f"  vocab_size={vs:,}: {mem:.1f} MB")
```

## 常见误区

**误区一：忽略 dtype 选择**

使用 float64 会浪费一半内存：

```python
# 错误：使用 float64
emb_f64 = np.random.randn(10000, 768)
print(f"float64 内存: {emb_f64.nbytes / 1024 / 1024:.1f} MB")

# 正确：使用 float32
emb_f32 = np.random.randn(10000, 768).astype(np.float32)
print(f"float32 内存: {emb_f32.nbytes / 1024 / 1024:.1f} MB")
```

**误区二：直接修改原始嵌入矩阵**

在微调时，如果只想更新部分嵌入，应该先复制：

```python
# 需要更新的 token
updated_token_ids = [0, 1, 2, 3, 4]

# 创建可学习的嵌入副本
trainable_embeddings = embeddings[updated_token_ids].copy()
```

**误区三：忽视嵌入的尺度**

嵌入向量的尺度会影响训练稳定性：

```python
# 合适的初始化标准差（如 0.02）比过大的标准差更好
small_std = np.random.randn(10000, 768).astype(np.float32) * 0.02
large_std = np.random.randn(10000, 768).astype(np.float32) * 0.5

print(f"小标准差向量范数均值: {np.mean(np.linalg.norm(small_std, axis=1)):.4f}")
print(f"大标准差向量范数均值: {np.mean(np.linalg.norm(large_std, axis=1)):.4f}")
```

## API 总结

| 类/函数 | 描述 |
|--------|------|
| `EmbeddingLayer` | 嵌入层封装 |
| `xavier_init` | Xavier 初始化 |
| `normalize_embeddings` | L2 归一化 |
| `get_subembedding` | 提取子嵌入 |
| `embedding_statistics` | 统计信息 |

创建和管理嵌入矩阵是 LLM 开发的基础技能。掌握这些技巧能帮助你更好地理解和使用语言模型。
