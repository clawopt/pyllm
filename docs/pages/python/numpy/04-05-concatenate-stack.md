# 拼接与堆叠

在处理批量数据或合并多个数据源时，我们经常需要将多个数组拼接在一起。想象一下，当你需要将多个批次的训练数据合并成一个更大的批次时，或者当你想将多个上下文片段拼接成一个完整序列时，这些操作就显得非常重要。NumPy 提供了 `np.concatenate`、`np.vstack`、`np.hstack`、`np.stack` 等函数来完成这些任务。这些函数看似相似，但有着微妙的区别：`concatenate` 在现有轴上连接数组，而 `stack` 在新轴上堆叠数组。理解这些拼接和堆叠操作的原理对于正确处理多维数据至关重要。

## np.concatenate：沿轴连接

`concatenate` 是最基本的拼接函数，沿指定轴将多个数组连接在一起：

```python
import numpy as np

# 一维数组拼接
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
c = np.concatenate([a, b])
print(f"一维拼接: {c}")  # [1, 2, 3, 4, 5, 6]

# 二维数组沿轴0拼接（上下拼接）
matrix1 = np.array([[1, 2, 3], [4, 5, 6]])
matrix2 = np.array([[7, 8, 9], [10, 11, 12]])
concat_axis0 = np.concatenate([matrix1, matrix2], axis=0)
print(f"沿轴0拼接:\n{concat_axis0}")

# 二维数组沿轴1拼接（左右拼接）
concat_axis1 = np.concatenate([matrix1, matrix2], axis=1)
print(f"沿轴1拼接:\n{concat_axis1}")
```

`concatenate` 接受一个数组列表，将它们沿指定轴连接。关键要求是：除了连接轴之外，其他轴的大小必须相同。

## np.vstack：垂直堆叠

`vstack` 是沿轴0的简化拼接，等价于沿轴0拼接一维数组或沿轴0拼接二维数组：

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# vstack 拼接一维数组会创建一个二维数组
result = np.vstack([a, b])
print(f"vstack一维数组:\n{result}")

# vstack 拼接二维数组
matrix1 = np.array([[1, 2, 3]])
matrix2 = np.array([[4, 5, 6]])
result = np.vstack([matrix1, matrix2])
print(f"vstack二维数组:\n{result}")
```

`vstack` 可以接受任何形状兼容的数组，会自动将1维数组当作行向量来处理。

## np.hstack：水平堆叠

`hstack` 是沿轴1的简化拼接：

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# hstack 拼接一维数组
result = np.hstack([a, b])
print(f"hstack一维数组: {result}")  # [1, 2, 3, 4, 5, 6]

# 二维数组沿轴1拼接
matrix1 = np.array([[1, 2], [3, 4]])
matrix2 = np.array([[5, 6], [7, 8]])
result = np.hstack([matrix1, matrix2])
print(f"hstack二维数组:\n{result}")
```

## np.stack：沿新轴堆叠

`stack` 与 `concatenate` 的关键区别在于：`stack` 会创建一个新的轴，而 `concatenate` 是在现有轴上连接：

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# stack 创建新轴
stacked = np.stack([a, b])
print(f"stack结果:\n{stacked}")
print(f"stack形状: {stacked.shape}")  # (2, 3)

# 默认沿轴0堆叠
stacked_axis0 = np.stack([a, b], axis=0)
print(f"沿轴0堆叠: {stacked_axis0.shape}")

# 沿轴1堆叠
stacked_axis1 = np.stack([a, b], axis=1)
print(f"沿轴1堆叠: {stacked_axis1.shape}")  # (3, 2)
```

`stack` 会增加数组的维度：两个一维数组堆叠后变成二维，三个三维数组堆叠后变成四维。

## concatenate 与 stack 的区别

这是面试中经常被问到的问题。简单来说：
- `concatenate` 沿现有轴连接，不增加维度
- `stack` 沿新轴堆叠，增加一个新维度

```python
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

# concatenate 保持2维
concat = np.concatenate([a, b], axis=0)
print(f"concatenate形状: {concat.shape}")  # (4, 2)

# stack 增加一个新维度
stacked = np.stack([a, b], axis=0)
print(f"stack形状: {stacked.shape}")  # (2, 2, 2)

stacked_axis2 = np.stack([a, b], axis=2)
print(f"stack沿轴2形状: {stacked_axis2.shape}")  # (2, 2, 2)
```

## 在深度学习中的应用

### 合并多个批次的数据

在训练过程中，可能需要将多个 mini-batch 合并成一个较大的批次：

```python
batch1 = np.random.randn(32, 512, 768).astype(np.float32)
batch2 = np.random.randn(32, 512, 768).astype(np.float32)
batch3 = np.random.randn(32, 512, 768).astype(np.float32)

# 合并批次
combined_batch = np.concatenate([batch1, batch2, batch3], axis=0)
print(f"合并后形状: {combined_batch.shape}")  # (96, 512, 768)
```

### 拼接多个序列片段

在处理超长序列时，可能需要将多个片段拼接成一个完整序列：

```python
# 模拟多个序列片段
segment1 = np.random.randn(128, 512, 768).astype(np.float32)  # 第一段
segment2 = np.random.randn(128, 512, 768).astype(np.float32)  # 第二段
segment3 = np.random.randn(128, 512, 768).astype(np.float32)  # 第三段

# 拼接成一个完整序列
full_sequence = np.concatenate([segment1, segment2, segment3], axis=1)
print(f"完整序列形状: {full_sequence.shape}")  # (128, 1536, 768)
```

### 拼接词嵌入和位置编码

在 Transformer 中，需要将词嵌入和位置编码相加：

```python
batch_size = 4
seq_len = 512
embed_dim = 768

# 词嵌入
word_embeddings = np.random.randn(batch_size, seq_len, embed_dim).astype(np.float32)

# 位置编码（预先计算好的）
position_embeddings = np.random.randn(seq_len, embed_dim).astype(np.float32)

# 需要将位置编码扩展到批次维度
position_embeddings = np.expand_dims(position_embeddings, axis=0)  # (1, seq_len, embed_dim)
position_embeddings = np.broadcast_to(position_embeddings, (batch_size, seq_len, embed_dim))

# 拼接词嵌入和位置编码
# 注意：这里实际上是相加，但如果需要拼接不同信息可以用 concatenate
combined = np.concatenate([word_embeddings, position_embeddings], axis=-1)
print(f"拼接后形状: {combined.shape}")  # (4, 512, 1536)
```

### 使用 stack 创建批次

在创建训练批次时，stack 可以用来将多个样本组合成批次：

```python
# 假设我们有三个独立的样本
sample1 = np.random.randn(512, 768).astype(np.float32)
sample2 = np.random.randn(512, 768).astype(np.float32)
sample3 = np.random.randn(512, 768).astype(np.float32)

# 使用 stack 创建批次
batch = np.stack([sample1, sample2, sample3], axis=0)
print(f"批次形状: {batch.shape}")  # (3, 512, 768)

# 如果需要沿最后一个维度堆叠
batch_last = np.stack([sample1, sample2, sample3], axis=-1)
print(f"沿最后维度堆叠: {batch_last.shape}")  # (512, 768, 3)
```

## 常见误区与注意事项

### 误区一：混淆 concatenate 和 stack

```python
a = np.array([1, 2])
b = np.array([3, 4])

# concatenate 保持一维
c = np.concatenate([a, b])
print(f"concatenate: {c.shape}")  # (4,)

# stack 增加一维
s = np.stack([a, b])
print(f"stack: {s.shape}")  # (2, 2)
```

### 误区二：拼接时形状不兼容

```python
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6, 7]])  # 不同形状

# 沿轴0拼接会失败，因为列数不同
try:
    np.concatenate([a, b], axis=0)
except ValueError as e:
    print(f"形状不兼容错误: {e}")

# 沿轴1拼接会失败，因为行数不同
try:
    np.concatenate([a, b.reshape(2, 1, 1)], axis=1)  # 形状也不兼容
except ValueError as e:
    print(f"沿轴1形状不兼容: {e}")
```

### 误区三：忘记广播

```python
a = np.array([[1, 2, 3]])
b = np.array([[4, 5, 6]])

# vstack 自动处理一维数组
result = np.vstack([a, b])
print(f"vstack: {result.shape}")  # (2, 3)

# 但 concatenate 需要显式指定轴
result2 = np.concatenate([a, b], axis=0)
print(f"concatenate: {result2.shape}")  # (2, 3)
```

### 误区四：dtype 不一致时的处理

```python
a = np.array([1, 2, 3], dtype=np.int32)
b = np.array([4, 5, 6], dtype=np.float32)

# concatenate 不会自动转换类型
result = np.concatenate([a, b])
print(f"结果dtype: {result.dtype}")  # float32，会发生类型转换

# 可能会丢失精度
a_big = np.array([2**30], dtype=np.int32)
b_float = np.array([1.5], dtype=np.float32)
result = np.concatenate([a_big, b_float])
print(f"大整数结果: {result[0]}")  # 可能精度丢失
```

## 底层原理

`concatenate` 的实现涉及创建一个新的数组，然后将每个输入数组的数据复制到新数组的相应位置。这意味着对于大型数组，concatenate 可能是一个昂贵的操作，需要 O(n) 的时间和 O(n) 的内存。

`stack` 的实现类似，但还需要分配新的维度空间，然后将每个输入数组放到新维度的相应位置。

```python
# 查看拼接操作的内存行为
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# concatenate 不创建视图
c = np.concatenate([a, b])
print(f"concatenate 是新数组: {c.base is not a and c.base is not b}")  # True
```

## 小结

`np.concatenate`、`np.vstack`、`np.hstack` 和 `np.stack` 是进行数组拼接和堆叠的主要函数。`concatenate` 沿现有轴连接，`stack` 沿新轴堆叠。两者的关键区别在于维度是否增加。在深度学习中，这些操作广泛用于合并批次、拼接序列、处理多头注意力的输出等场景。

面试时需要能够解释 concatenate 和 stack 的区别，理解 vstack 和 hstack 是特定轴上的简化操作，以及注意拼接时形状兼容性的要求。
