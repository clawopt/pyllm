# 分割操作

分割操作是拼接操作的逆过程。想象一下，当你有一个完整的句子嵌入，但你需要将其分割成单独的词向量以便进行后续处理时；或者当你需要将一个大的批次分割成多个小批次以便进行分布式训练时；又或者当你需要将长序列切分成多个较短的片段（比如用于 Transformer-XL 的段落分割）时，这些场景都需要用到数组分割。NumPy 提供了 `np.split`、`np.array_split`、`np.vsplit`、`np.hsplit` 等函数来完成这些任务。理解这些分割函数对于正确处理批量数据和序列数据非常重要。

## np.split：均等分割

`split` 将数组分割成多个子数组：

```python
import numpy as np

arr = np.arange(12)
print(f"原始数组: {arr}")

# 均等分割成3份
split_arr = np.split(arr, 3)
print(f"分割成3份: {split_arr}")

# 在指定位置分割
split_at = np.split(arr, [3, 7])
print(f"在位置3和7分割: {split_at}")
```

`np.split(ary, indices_or_sections)` 的第二个参数可以是整数（表示要均分的份数）或索引位置列表。

## np.array_split：不等分割

`array_split` 允许你将数组分割成不等份：

```python
arr = np.arange(10)

# 分割成3份（尽可能均等）
split_arr = np.array_split(arr, 3)
print(f"不等分割: {split_arr}")
# 结果可能是 [array([0, 1, 2, 3]), array([4, 5, 6, 7]), array([8, 9])]
```

当数组长度不能被份数整除时，`array_split` 会尽可能均匀地分割。

## np.vsplit 与 np.hsplit

`vsplit` 沿轴0分割，`hsplit` 沿轴1分割：

```python
matrix = np.arange(12).reshape(4, 3)
print(f"原始矩阵:\n{matrix}")

# 沿轴0垂直分割成2份
split_v = np.vsplit(matrix, 2)
print(f"vsplit:\n{split_v}")

# 沿轴1水平分割成3份
split_h = np.hsplit(matrix, 3)
print(f"hsplit:\n{split_h}")
```

## 多维数组的分割

```python
arr_3d = np.arange(24).reshape(4, 3, 2)
print(f"三维数组形状: {arr_3d.shape}")

# 沿轴0分割成2份
split_axis0 = np.split(arr_3d, 2, axis=0)
print(f"沿轴0分割: {[s.shape for s in split_axis0]}")

# 沿轴1分割
split_axis1 = np.split(arr_3d, 3, axis=1)
print(f"沿轴1分割: {[s.shape for s in split_axis1]}")

# 在指定位置分割
split_pos = np.split(arr_3d, [1, 3], axis=0)
print(f"在位置[1,3]分割: {[s.shape for s in split_pos]}")
```

## 在深度学习中的应用

### 将长序列切分为chunks

在处理超长序列时，可能需要将其切分成多个较短的片段以便进行注意力计算：

```python
seq_len = 2048
chunk_len = 512
batch_size = 4
hidden_dim = 768

# 模拟一个长序列的隐藏状态
long_sequence = np.random.randn(batch_size, seq_len, hidden_dim).astype(np.float32)
print(f"长序列形状: {long_sequence.shape}")

# 计算片段数量
num_chunks = seq_len // chunk_len

# 分割成多个chunks
chunks = np.split(long_sequence, num_chunks, axis=1)
print(f"片段数量: {len(chunks)}")
print(f"每个片段形状: {chunks[0].shape}")
```

### 划分训练集和验证集

在机器学习中，经常需要将数据分割为训练集和验证集：

```python
data_size = 1000
batch_size = 32

# 模拟数据
all_data = np.random.randn(data_size, 512, 768).astype(np.float32)
print(f"全部数据形状: {all_data.shape}")

# 划分比例
train_ratio = 0.8
train_size = int(data_size * train_ratio)

# 分割数据
train_data = all_data[:train_size]
val_data = all_data[train_size:]

print(f"训练集形状: {train_data.shape}")
print(f"验证集形状: {val_data.shape}")
```

### 分割多头注意力的输出

在 Transformer 中，可能需要将多头注意力的输出分割给不同的处理模块：

```python
batch_size = 4
seq_len = 512
num_heads = 12
head_dim = 64

# 模拟多头注意力的输出
mha_output = np.random.randn(batch_size, seq_len, num_heads, head_dim).astype(np.float32)
print(f"MHA输出形状: {mha_output.shape}")

# 分割成两部分：一部分用于跳跃连接，一部分用于后续处理
split_point = num_heads // 2

# 沿num_heads维度分割
parts = np.split(mha_output, [split_point], axis=2)
print(f"第一部分形状: {parts[0].shape}")  # (4, 512, 6, 64)
print(f"第二部分形状: {parts[1].shape}")  # (4, 512, 6, 64)
```

### 使用 array_split 处理不均匀批次

在处理变长序列时，分割可能不均匀：

```python
batch_size = 5
seq_lengths = [100, 150, 80, 200, 120]
hidden_dim = 256

# 模拟变长序列的隐藏状态（padding到最大长度）
max_len = max(seq_lengths)
hidden_states = np.random.randn(batch_size, max_len, hidden_dim).astype(np.float32)
print(f"隐藏状态形状: {hidden_states.shape}")

# 为每个序列创建掩码
attention_masks = np.array([np.concatenate([np.ones(l), np.zeros(max_len - l)]) for l in seq_lengths])
print(f"注意力掩码形状: {attention_masks.shape}")
```

## 分割与视图

分割操作返回的是原始数据的视图，而不是副本：

```python
arr = np.arange(12)
parts = np.split(arr, 3)

# 分割的每个部分都与原始数组共享内存
print(f"第一部分共享内存: {np.shares_memory(arr, parts[0])}")  # True

# 修改分割后的数组会影响原始数组
parts[0][0] = 99
print(f"原始数组: {arr}")  # arr[0] 变成了 99
```

## 常见误区与注意事项

### 误区一：分割数量不能整除

```python
arr = np.arange(10)

# 分割成3份，10不能被3整除
try:
    split_arr = np.split(arr, 3)
    print(f"分割结果: {split_arr}")
except ValueError as e:
    print(f"分割错误: {e}")

# 使用 array_split 可以处理这种情况
split_arr = np.array_split(arr, 3)
print(f"array_split结果: {split_arr}")
```

### 误区二：索引位置越界

```python
arr = np.arange(12)

# 分割位置超出数组范围
try:
    split_arr = np.split(arr, [5, 10, 15])  # 15超出范围
    print(f"分割结果: {split_arr}")
except ValueError as e:
    print(f"分割位置错误: {e}")
```

### 误区三：分割后忘记保持维度信息

```python
arr_2d = np.arange(12).reshape(3, 4)
print(f"二维数组形状: {arr_2d.shape}")

# 分割后可能变成一维数组
parts = np.split(arr_2d, 3, axis=0)
print(f"分割后第一部分形状: {parts[0].shape}")  # (4,) - 变成一维了

# 如果需要保持维度，使用 expand_dims 或 keepdims
parts_with_dims = [np.expand_dims(p, axis=0) if p.ndim == 1 else p for p in parts]
print(f"保持维度后形状: {[p.shape for p in parts_with_dims]}")
```

## 与拼接的配合使用

分割和拼接经常配合使用，实现数据的重组：

```python
arr = np.arange(12)

# 分割后重新拼接（验证一致性）
parts = np.split(arr, 4)
reconstructed = np.concatenate(parts)
print(f"分割后拼接一致: {np.array_equal(arr, reconstructed)}")

# 更复杂的重组：交换前半部分和后半部分
arr_2d = np.arange(12).reshape(3, 4)
print(f"原始矩阵:\n{arr_2d}")

# 沿轴1分割成两部分，交换顺序后拼接
left, right = np.split(arr_2d, 2, axis=1)
reordered = np.concatenate([right, left], axis=1)
print(f"交换后:\n(reordered}")
```

## 小结

`np.split`、`np.array_split`、`np.vsplit` 和 `np.hsplit` 是进行数组分割的主要函数。`split` 可以均等分割或按位置分割，`array_split` 可以处理不均等分割，`vsplit` 和 `hsplit` 是沿特定轴分割的简化版本。分割操作返回的是视图而非副本，理解这一点对于避免意外的副作用非常重要。

在深度学习中，分割操作广泛用于：将长序列切分成chunks、划分训练集和验证集、分割多头注意力的输出、处理变长序列等场景。掌握这些分割函数是处理批量数据和序列数据的基础。

面试时需要能够解释分割和拼接的配合使用，理解分割返回视图而非副本的特性，以及注意分割位置必须有效的要求。
