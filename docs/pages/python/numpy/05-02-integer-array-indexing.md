# 整数数组索引

整数数组索引是NumPy中最强大也最灵活的索引方式之一。想象一下，当你有一个包含所有词向量的大矩阵（词表），而你需要根据一系列token ID从中取出对应的词向量时；或者当你需要从一个数据集中挑选出特定索引的样本时，整数数组索引就是你的首选工具。与基础索引使用单个整数不同，整数数组索引使用一个整数数组作为索引，可以一次从数组中取出任意位置的多个元素，返回的是副本而非视图。理解整数数组索引对于实现embedding lookup等LLM核心操作至关重要。

## 基本用法

整数数组索引使用一个整数数组作为索引：

```python
import numpy as np

arr = np.array([10, 20, 30, 40, 50])
indices = np.array([0, 2, 4])

# 使用整数数组作为索引
selected = arr[indices]
print(f"原始数组: {arr}")
print(f"索引: {indices}")
print(f"选中的元素: {selected}")
```

这里 `arr[[0, 2, 4]]` 返回了索引0、2、4位置的元素组成的数组。

## 二维数组的整数数组索引

整数数组索引在二维数组上更加灵活，可以同时指定行和列的索引：

```python
matrix = np.array([[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12]])

# 提取对角线元素
row_indices = np.array([0, 1, 2])
col_indices = np.array([0, 1, 2])
diagonal = matrix[row_indices, col_indices]
print(f"对角线元素: {diagonal}")

# 提取任意位置的元素
rows = np.array([0, 0, 1, 2])
cols = np.array([1, 3, 0, 2])
selected = matrix[rows, cols]
print(f"选中的元素: {selected}")
```

当使用两个整数数组作为索引时，它们必须长度相同。NumPy会取对应位置的元素组成结果。

## 使用整数数组索引进行赋值

整数数组索引不仅可以用于取值，还可以用于赋值：

```python
arr = np.array([1, 2, 3, 4, 5])
indices = np.array([0, 2, 4])

# 将索引位置的值改为10
arr[indices] = 10
print(f"赋值后数组: {arr}")  # [10, 2, 10, 4, 10]

# 也可以用数组赋值
arr[indices] = np.array([100, 200, 300])
print(f"数组赋值后: {arr}")  # [100, 2, 200, 4, 300]
```

## 在LLM场景中的应用

### 词向量查找（Embedding Lookup）

这是LLM中最核心的操作之一：根据token ID从embedding表中查找对应的向量：

```python
vocab_size = 30000
embed_dim = 768

# 模拟词表（词向量矩阵）
embedding_table = np.random.randn(vocab_size, embed_dim).astype(np.float32) * 0.02
print(f"Embedding表形状: {embedding_table.shape}")

# Token IDs
token_ids = np.array([101, 2003, 1037, 3010, 102])  # "hello world"
print(f"Token IDs: {token_ids}")

# 使用整数数组索引查找embedding
embeddings = embedding_table[token_ids]
print(f"查找到的embeddings形状: {embeddings.shape}")
print(f"embeddings:\n{embeddings}")
```

这就是LLM中embedding lookup的核心实现。每次你输入一个句子，模型首先将每个token转换为对应的向量，这个转换就是通过整数数组索引完成的。

### 批量Embedding Lookup

在处理批量数据时，一次可以查找多个token的embedding：

```python
batch_size = 4
seq_len = 20
vocab_size = 30000
embed_dim = 768

# 模拟批量token IDs
batch_token_ids = np.random.randint(0, vocab_size, size=(batch_size, seq_len))
print(f"批量Token IDs形状: {batch_token_ids.shape}")

# 词向量表
embedding_table = np.random.randn(vocab_size, embed_dim).astype(np.float32) * 0.02

# 批量查找embeddings
batch_embeddings = embedding_table[batch_token_ids]
print(f"批量Embeddings形状: {batch_embeddings.shape}")
```

对于 (batch_size, seq_len) 的token IDs，查找结果是一个 (batch_size, seq_len, embed_dim) 的embeddings。

### 从隐藏状态中选取特定位置

在分析LLM的隐藏状态时，可能需要提取特定位置的表示：

```python
batch_size = 2
seq_len = 512
hidden_dim = 768

# 模拟隐藏状态
hidden_states = np.random.randn(batch_size, seq_len, hidden_dim).astype(np.float32)

# 选取关键位置：[CLS] token、标点符号、句首句尾
key_positions = np.array([0, 10, 50, 100, 200, 300, 400, 500, 511])

# 选取第一个样本的关键位置
key_hidden = hidden_states[0, key_positions, :]
print(f"关键位置hidden形状: {key_hidden.shape}")

# 选取所有样本的关键位置
all_key_hidden = hidden_states[:, key_positions, :]
print(f"所有样本关键位置形状: {all_key_hidden.shape}")
```

### 选择特定注意力头

在分析多头注意力时，可能需要选择特定的注意力头：

```python
batch_size = 1
num_heads = 12
seq_len = 128
head_dim = 64

# 模拟注意力输出
attention_output = np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np.float32)

# 选择最后4个头（通常包含更多语义信息）
last_heads = np.array([8, 9, 10, 11])
selected_heads = attention_output[:, last_heads, :, :]
print(f"选中头的形状: {selected_heads.shape}")

# 也可以用于分析特定的语义头
semantic_heads = np.array([0, 5, 10])
semantic_output = attention_output[:, semantic_heads, :, :]
print(f"语义头形状: {semantic_output.shape}")
```

## 常见误区与注意事项

### 误区一：索引数组形状与被索引数组维度不匹配

```python
matrix = np.array([[1, 2, 3], [4, 5, 6]])

# 如果有两个索引数组，它们必须长度相同
rows = np.array([0, 1])
cols = np.array([0, 1, 2])  # 长度不匹配！

try:
    result = matrix[rows, cols]
except IndexError as e:
    print(f"索引长度不匹配错误: {e}")
```

### 误区二：索引越界

```python
arr = np.array([1, 2, 3, 4, 5])

# 索引超出数组范围会报错
try:
    result = arr[np.array([0, 10])]
except IndexError as e:
    print(f"索引越界错误: {e}")
```

### 误区三：返回的是副本而非视图

```python
arr = np.array([10, 20, 30, 40, 50])
indices = np.array([1, 3])

selected = arr[indices]
print(f"选中的元素: {selected}")

# 修改选中元素
selected[0] = 99

# 原始数组不受影响
print(f"原始数组: {arr}")  # 仍然是 [10, 20, 30, 40, 50]
```

与切片不同，整数数组索引返回的是副本。修改返回的数组不会影响原始数组。

### 误区四：一维索引与二维索引混淆

```python
matrix = np.array([[1, 2, 3], [4, 5, 6]])

# 一维索引：取特定行
row = matrix[1]  # [4, 5, 6]

# 二维索引：取特定位置的元素
element = matrix[1, 2]  # 6

# 整数数组索引
elements = matrix[[1], [2]]  # [6] - 注意这是二维索引的数组版本
print(f"二维整数数组索引: {elements}")
```

## 底层原理

整数数组索引的底层实现与基础索引完全不同。当使用整数数组索引时，NumPy会创建一个新的数组（副本），然后将索引位置的数据复制到新数组中。这个过程涉及：

1. 解析索引数组，确定需要访问的位置
2. 分配新的内存空间
3. 将索引位置的数据复制到新空间

由于返回的是副本，整数数组索引不会共享内存，这是与切片的一个重要区别。

## 小结

整数数组索引是NumPy中最灵活的索引方式，使用整数数组作为索引可以一次从数组中取出任意位置的多个元素。与切片不同，整数数组索引返回的是副本而非视图。理解整数数组索引对于实现LLM中的embedding lookup等核心操作至关重要。

在LLM场景中，整数数组索引广泛用于：根据token ID从embedding表中查找词向量、从隐藏状态中选取特定位置、选择特定的注意力头等。

面试时需要能够解释整数数组索引与切片的区别（副本vs视图），理解二维整数数组索引的用法，以及注意索引数组长度必须匹配的要求。
