# 花式索引

花式索引（Fancy Indexing）是NumPy中一种强大但容易让人困惑的索引方式。"花式"这个词听起来很花哨，其实它只是指用整数数组作为索引的一种统称。想象一下，当你需要从一个大矩阵中选取任意分布的多个行或列时，或者当你需要按照特定顺序重新排列数组元素时，花式索引就是你的首选工具。与基础切片不同，花式索引可以选取任意不连续的元素位置；与布尔索引不同，花式索引可以使用整数数组精确指定要选取的位置。在LLM和深度学习中，花式索引常用于从embedding表中批量获取token向量、按特定顺序选取数据等场景。

## 基本用法

花式索引使用整数数组作为索引：

```python
import numpy as np

arr = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

# 使用整数数组作为索引
indices = np.array([0, 2, 4, 6, 8])
selected = arr[indices]
print(f"原始数组: {arr}")
print(f"索引: {indices}")
print(f"选中的元素: {selected}")
```

花式索引返回的是一个新的数组，包含了索引指定位置的所有元素。

## 使用负数索引

花式索引也支持负数索引，从数组末尾开始计数：

```python
arr = np.array([10, 20, 30, 40, 50])

# 使用负数索引
indices = np.array([0, -1, -2])
selected = arr[indices]
print(f"使用负数索引: {selected}")  # [10, 50, 40]
```

## 重复索引

花式索引允许重复使用同一个索引：

```python
arr = np.array([10, 20, 30, 40, 50])

# 重复索引
indices = np.array([0, 0, 0, 1, 1, 2])
selected = arr[indices]
print(f"重复索引: {selected}")  # [10, 10, 10, 20, 20, 30]
```

重复索引会返回对应位置的元素重复出现，这在某些数据增强场景中很有用。

## 二维花式索引

花式索引在二维数组上有两种主要用法：按行索引和按行列同时索引：

```python
matrix = np.array([[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12],
                   [13, 14, 15, 16]])

# 按行花式索引
rows = np.array([0, 2, 3])
selected_rows = matrix[rows]
print(f"选取行0,2,3:\n{selected_rows}")

# 同时指定行和列索引
row_indices = np.array([0, 1, 2, 3])
col_indices = np.array([0, 1, 2, 3])
diagonal = matrix[row_indices, col_indices]
print(f"对角线元素: {diagonal}")

# 任意位置组合
positions = np.array([(0, 0), (1, 2), (3, 3)])
rows = positions[:, 0]
cols = positions[:, 1]
selected = matrix[rows, cols]
print(f"选取(0,0),(1,2),(3,3): {selected}")
```

## 花式索引与切片的组合

花式索引可以与切片混合使用：

```python
matrix = np.arange(16).reshape(4, 4)
print(f"原始矩阵:\n{matrix}")

# 花式索引选取特定行，再用切片选取特定列
rows = np.array([0, 2, 3])
selected = matrix[rows, 1:3]
print(f"行0,2,3的列1,2:\n{selected}")
```

## 花式索引与视图

这是一个重要的区别：花式索引返回的是副本，而不是视图。这意味着修改花式索引的结果不会影响原始数组：

```python
arr = np.array([10, 20, 30, 40, 50])
indices = np.array([0, 2, 4])
selected = arr[indices]

# 修改选中的元素
selected[0] = 99

# 原始数组不受影响
print(f"原始数组: {arr}")  # 仍然是 [10, 20, 30, 40, 50]
```

这一点与基础切片不同，切片返回的是视图。

## 在LLM场景中的应用

### 从词嵌入矩阵中批量获取token向量

这是花式索引最常见的应用场景之一：

```python
vocab_size = 30000
embed_dim = 768

# 词嵌入矩阵
embedding_table = np.random.randn(vocab_size, embed_dim).astype(np.float32) * 0.02

# Token IDs（可以是任意顺序）
token_ids = np.array([100, 2000, 500, 10000, 25000])

# 使用花式索引批量获取embeddings
embeddings = embedding_table[token_ids]
print(f"获取的embeddings形状: {embeddings.shape}")
```

花式索引能够一次性从embedding表中取出任意多个token的向量，而不需要循环。

### 打乱数据顺序

在训练时打乱数据顺序：

```python
batch_size = 8
seq_len = 512
embed_dim = 768

# 模拟一个批次的数据
batch_embeddings = np.random.randn(batch_size, seq_len, embed_dim).astype(np.float32)

# 生成打乱的索引
shuffle_indices = np.arange(batch_size)
np.random.shuffle(shuffle_indices)

# 使用花式索引重新排列数据
shuffled_batch = batch_embeddings[shuffle_indices]
print(f"打乱后的batch形状: {shuffled_batch.shape}")
```

### 选择特定位置的隐藏状态

在分析模型行为时，可能需要选取特定位置的隐藏状态：

```python
batch_size = 4
seq_len = 512
hidden_dim = 768

# 模拟隐藏状态
hidden_states = np.random.randn(batch_size, seq_len, hidden_dim).astype(np.float32)

# 选取关键位置：句首、句尾、以及每隔100个位置
key_positions = np.array([0, 100, 200, 300, 400, 500, 511])

# 选取第一个样本的关键位置
key_hidden = hidden_states[0, key_positions, :]
print(f"关键位置hidden形状: {key_hidden.shape}")

# 选取所有样本的关键位置
all_key_hidden = hidden_states[:, key_positions, :]
print(f"所有样本关键位置形状: {all_key_hidden.shape}")
```

### 按样本ID列表选取数据

从数据集中按ID列表选取样本：

```python
num_samples = 10000
seq_len = 128
features = 256

# 模拟数据集
dataset = np.random.randn(num_samples, seq_len, features).astype(np.float32)

# 选取特定ID的样本
sample_ids = np.array([0, 100, 200, 300, 9999])
selected_samples = dataset[sample_ids]
print(f"选取的样本形状: {selected_samples.shape}")
```

## 花式索引与 np.take

NumPy提供了 `np.take` 函数，它可以实现类似花式索引的功能：

```python
arr = np.array([10, 20, 30, 40, 50])
indices = np.array([0, 2, 4])

# 使用花式索引
result1 = arr[indices]

# 使用 np.take
result2 = np.take(arr, indices)

print(f"花式索引结果: {result1}")
print(f"np.take结果: {result2}")
print(f"两者相等: {np.array_equal(result1, result2)}")
```

`np.take` 在某些情况下更直观，而且支持 `axis` 参数，可以沿指定轴进行选取。

## 常见误区与注意事项

### 误区一：混淆花式索引和整数数组索引

实际上，花式索引就是整数数组索引的另一个名称，本质上是相同的概念。

### 误区二：忘记花式索引返回副本

```python
arr = np.array([10, 20, 30, 40, 50])
indices = np.array([0, 2, 4])
selected = arr[indices]

selected[0] = 99
print(f"原始数组: {arr}")  # 不受影响
```

花式索引返回副本，布尔索引也返回副本，只有基础切片返回视图。

### 误区三：索引越界

```python
arr = np.array([10, 20, 30, 40, 50])
indices = np.array([0, 5])  # 5 越界

try:
    result = arr[indices]
except IndexError as e:
    print(f"索引越界错误: {e}")
```

### 误区四：花式索引的性能问题

```python
arr = np.random.randn(1000000)

# 大量花式索引操作可能较慢
indices = np.random.randint(0, 1000000, size=100000)

# 这种操作比切片慢很多
start = time.time()
result = arr[indices]
print(f"花式索引耗时: {time.time() - start:.4f}s")
```

花式索引涉及数据复制，对于非常大的数组可能成为性能瓶颈。

## 小结

花式索引（整数数组索引）是一种强大的索引方式，允许你用整数数组从数组中选取任意位置的元素。与切片返回视图不同，花式索引返回副本。理解花式索引对于实现LLM中的embedding lookup等核心操作至关重要。

在LLM场景中，花式索引广泛用于：从embedding表中批量获取token向量、按特定顺序选取数据、选取特定位置的隐藏状态等。掌握花式索引是进行高效NumPy编程的重要技能。

面试时需要能够解释花式索引与切片的区别（副本vs视图），理解花式索引的各种用法，以及注意索引越界和性能问题。
