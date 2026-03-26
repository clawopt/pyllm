# 词汇表与索引映射

词嵌入（Word Embedding）是大型语言模型的基础组件之一，它将离散的 token（如单词、子词）映射到连续的向量空间。实现词嵌入的第一步是构建词汇表（Vocabulary）和 token 到索引的映射。本篇文章从零开始，使用纯 NumPy 实现一个完整的词汇表系统，包括特殊 token 的处理、词表到索引的双向映射，以及常见的词汇表操作。这为后续构建词嵌入矩阵和实现词向量查找打下基础。

## 词汇表的基本概念

词汇表是 LLM 的核心组件之一，它定义了模型能够处理的所有 token。在现代 LLM 中，词汇表通常包含：
- 普通单词（如 "hello", "world"）
- 子词（subwords，如 "hel", "lo"）
- 特殊 token（如 `[PAD]` 填充、`[UNK]` 未知、`[CLS]` 分类、`[SEP]` 分隔符、`[MASK]` 掩码）
- 数字和标点符号

词汇表的大小直接影响模型的能力和效率：太大的词汇表会增加模型参数和计算成本；太小的词汇表会导致过度切分，增加序列长度。

## 使用 NumPy 实现词汇表

### 基本词汇表类

```python
import numpy as np
from collections import OrderedDict

class Vocabulary:
    """简单的词汇表实现

    使用有序字典存储 token -> index 和 index -> token 的双向映射
    """

    def __init__(self):
        self.token_to_idx = OrderedDict()
        self.idx_to_token = OrderedDict()
        self.num_tokens = 0

        # 添加特殊 token
        self.pad_token = '[PAD]'
        self.unk_token = '[UNK]'
        self.bos_token = '[BOS]'  # Begin of Sequence
        self.eos_token = '[EOS]'  # End of Sequence
        self.mask_token = '[MASK]'

    def add_token(self, token):
        """添加 token 到词汇表"""
        if token not in self.token_to_idx:
            self.token_to_idx[token] = self.num_tokens
            self.idx_to_token[self.num_tokens] = token
            self.num_tokens += 1

    def add_tokens(self, tokens):
        """批量添加 tokens"""
        for token in tokens:
            self.add_token(token)

    def token_to_index(self, token):
        """将 token 转换为索引"""
        return self.token_to_idx.get(token, self.token_to_idx[self.unk_token])

    def index_to_token(self, idx):
        """将索引转换为 token"""
        return self.idx_to_token.get(idx, self.unk_token)

    def lookup_token(self, token):
        """查找 token（与 token_to_index 相同）"""
        return self.token_to_index(token)

    def __len__(self):
        return self.num_tokens

    def __contains__(self, token):
        return token in self.token_to_idx
```

### 创建词汇表实例

```python
def build_vocabulary():
    """构建示例词汇表"""
    vocab = Vocabulary()

    # 添加特殊 token
    vocab.add_token(vocab.pad_token)
    vocab.add_token(vocab.unk_token)
    vocab.add_token(vocab.bos_token)
    vocab.add_token(vocab.eos_token)
    vocab.add_token(vocab.mask_token)

    # 添加普通 tokens
    tokens = [
        'hello', 'world', 'the', 'quick', 'brown', 'fox', 'jumps',
        'over', 'lazy', 'dog', 'i', 'am', 'a', 'student',
        'natural', 'language', 'processing', 'is', 'fascinating'
    ]
    vocab.add_tokens(tokens)

    return vocab

# 示例
vocab = build_vocabulary()
print(f"词汇表大小: {len(vocab)}")
print(f"特殊 token: {[vocab.pad_token, vocab.unk_token, vocab.bos_token, vocab.eos_token]}")
print(f"\ntoken -> index 映射:")
for token, idx in list(vocab.token_to_idx.items())[:10]:
    print(f"  '{token}': {idx}")
```

## 索引映射操作

### 批量编码（将文本转为索引）

```python
def encode_text(text, vocab):
    """将文本编码为索引序列

    参数:
        text: 输入文本（字符串或 token 列表）
        vocab: Vocabulary 实例
    返回:
        indices: 索引列表
    """
    if isinstance(text, str):
        tokens = text.split()
    else:
        tokens = text

    indices = [vocab.token_to_index(token) for token in tokens]
    return indices

def decode_indices(indices, vocab):
    """将索引序列解码为文本

    参数:
        indices: 索引列表
        vocab: Vocabulary 实例
    返回:
        tokens: token 列表
    """
    tokens = [vocab.index_to_token(idx) for idx in indices]
    return tokens

# 示例
text = "hello world the quick brown fox"
indices = encode_text(text, vocab)
print(f"原文: {text}")
print(f"编码: {indices}")
decoded = decode_indices(indices, vocab)
print(f"解码: {' '.join(decoded)}")
```

### 处理未知 token

```python
def encode_text_with_unknown(text, vocab):
    """编码文本，将未知 token 映射到 [UNK]

    参数:
        text: 输入文本
        vocab: Vocabulary 实例
    返回:
        indices: 索引列表
    """
    if isinstance(text, str):
        tokens = text.split()
    else:
        tokens = text

    indices = []
    for token in tokens:
        if token in vocab:
            indices.append(vocab.token_to_index(token))
        else:
            indices.append(vocab.token_to_index(vocab.unk_token))
    return indices

# 测试未知 token
text_with_unknown = "hello python the world"
indices = encode_text_with_unknown(text_with_unknown, vocab)
print(f"原文: {text_with_unknown}")
print(f"编码: {indices}")
print(f"解码: {' '.join(decode_indices(indices, vocab))}")
```

## 词汇表的序列化与反序列化

保存和加载词汇表是重要的功能：

```python
def save_vocabulary(vocab, filepath):
    """保存词汇表到文件

    格式：每行一个 token
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        for idx in range(len(vocab)):
            token = vocab.idx_to_token[idx]
            f.write(f"{token}\n")
    print(f"词汇表已保存: {filepath}")

def load_vocabulary(filepath):
    """从文件加载词汇表"""
    vocab = Vocabulary()
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            token = line.strip()
            if token:
                vocab.add_token(token)
    return vocab

# 保存示例
save_vocabulary(vocab, 'vocab.txt')

# 加载示例
loaded_vocab = load_vocabulary('vocab.txt')
print(f"加载后词汇表大小: {len(loaded_vocab)}")
```

## 高级词汇表功能

### 添加词频统计

```python
class FrequencyVocabulary(Vocabulary):
    """带词频统计的词汇表"""

    def __init__(self):
        super().__init__()
        self.token_freq = OrderedDict()

    def add_token(self, token):
        """添加 token 并更新词频"""
        if token not in self.token_to_idx:
            self.token_to_idx[token] = self.num_tokens
            self.idx_to_token[self.num_tokens] = token
            self.token_freq[token] = 0
            self.num_tokens += 1
        self.token_freq[token] += 1

    def get_frequency(self, token):
        """获取 token 的词频"""
        return self.token_freq.get(token, 0)

    def most_common(self, k=10):
        """返回最常见的 k 个 token"""
        sorted_tokens = sorted(self.token_freq.items(), key=lambda x: x[1], reverse=True)
        return sorted_tokens[:k]

# 示例
freq_vocab = FrequencyVocabulary()
texts = [
    "the quick brown fox jumps over the lazy dog",
    "the quick brown fox jumps over the lazy dog",
    "hello world",
    "hello world",
    "hello world",
    "natural language processing is fascinating"
]

for text in texts:
    tokens = text.split()
    for token in tokens:
        freq_vocab.add_token(token)

print("词频统计:")
for token, freq in freq_vocab.most_common(5):
    print(f"  '{token}': {freq}")
```

### 词汇表剪裁

```python
def trim_vocabulary(vocab, min_freq=2):
    """剪裁词汇表，移除低频 token

    参数:
        vocab: FrequencyVocabulary 实例
        min_freq: 最小词频阈值
    返回:
        trimmed_vocab: 剪裁后的词汇表
    """
    trimmed_vocab = Vocabulary()

    # 添加特殊 token
    trimmed_vocab.add_token(trimmed_vocab.pad_token)
    trimmed_vocab.add_token(trimmed_vocab.unk_token)

    # 只添加高频 token
    for token, freq in vocab.token_freq.items():
        if freq >= min_freq and token not in [vocab.pad_token, vocab.unk_token]:
            trimmed_vocab.add_token(token)

    return trimmed_vocab

# 示例
trimmed = trim_vocabulary(freq_vocab, min_freq=2)
print(f"原始词汇表大小: {len(freq_vocab)}")
print(f"剪裁后词汇表大小: {len(trimmed)}")
```

## 与词嵌入矩阵的集成

词汇表通常与嵌入矩阵一起使用：

```python
def create_vocab_and_embeddings(texts, embed_dim=768, min_freq=1):
    """从文本创建词汇表和随机初始化的嵌入矩阵

    参数:
        texts: 文本列表
        embed_dim: 嵌入维度
        min_freq: 最小词频
    返回:
        vocab: 词汇表
        embeddings: 嵌入矩阵 (vocab_size, embed_dim)
    """
    # 构建词频词汇表
    freq_vocab = FrequencyVocabulary()
    for text in texts:
        tokens = text.split()
        for token in tokens:
            freq_vocab.add_token(token)

    # 剪裁词汇表
    vocab = trim_vocabulary(freq_vocab, min_freq)

    # 创建嵌入矩阵
    vocab_size = len(vocab)
    embeddings = np.random.randn(vocab_size, embed_dim).astype(np.float32) * 0.02

    # 初始化特殊 token 的嵌入
    embeddings[vocab.token_to_index(vocab.pad_token)] = np.zeros(embed_dim)

    return vocab, embeddings

# 示例
sample_texts = [
    "hello world",
    "hello world hello",
    "natural language processing",
    "language is fascinating"
]

vocab, embeddings = create_vocab_and_embeddings(sample_texts)
print(f"词汇表大小: {len(vocab)}")
print(f"嵌入矩阵形状: {embeddings.shape}")
```

## 常见误区

**误区一：特殊 token 处理不一致**

特殊 token 应该在词汇表构建初期就确定，并始终保持一致：

```python
# 错误：不同地方使用不同的特殊 token
vocab1 = Vocabulary()
vocab1.add_token('[PAD]')
vocab1.add_token('PAD')  # 不一致！

# 正确：统一使用相同的特殊 token
vocab = Vocabulary()
assert vocab.pad_token == '[PAD]'  # 始终一致
```

**误区二：忽略词汇表大小对模型的影响**

词汇表大小直接影响嵌入矩阵的参数量和计算效率：

```python
# vocab_size * embed_dim = 嵌入参数量
vocab_size = 50000
embed_dim = 768
num_params = vocab_size * embed_dim
print(f"嵌入参数量: {num_params:,} ({num_params / 1e6:.1f}M)")
```

**误区三：使用字符串作为索引的键**

在实际应用中，应该使用整数索引，而不是字符串：

```python
# 错误：使用字符串 key
token_ids = ["hello", "world"]

# 正确：使用整数索引
token_ids = [vocab.token_to_index("hello"), vocab.token_to_index("world")]
```

## API 总结

| 方法 | 描述 |
|------|------|
| `vocab.add_token(token)` | 添加单个 token |
| `vocab.add_tokens(tokens)` | 批量添加 tokens |
| `vocab.token_to_index(token)` | token -> 索引 |
| `vocab.index_to_token(idx)` | 索引 -> token |
| `vocab.__len__()` | 词汇表大小 |
| `vocab.__contains__(token)` | 检查 token 是否存在 |

构建词汇表是 LLM 数据处理的第一步。掌握词汇表的创建和操作，为后续的词嵌入查找和模型构建奠定基础。
