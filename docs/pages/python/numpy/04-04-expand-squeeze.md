# 增加与删除维度

在深度学习的数据处理中，维度变换是一项非常频繁的操作。想象一下，当你有一个单独的样本（没有批次维度）需要输入到神经网络时，你可能需要给它增加一个批次维度；或者当你有一个批次的数据，但某个操作需要特定的维度格式时，你需要增加或删除某些维度。NumPy 提供了 `np.newaxis`、`np.expand_dims` 和 `np.squeeze` 这些工具来完成这些操作。理解这些维度变换操作对于正确处理神经网络输入输出至关重要，因为不同框架和模型对输入维度的要求可能不同。

## np.newaxis：增加维度

`np.newaxis` 是在索引中使用的一个特殊对象，它可以在指定位置插入一个新的维度：

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5])
print(f"原始数组形状: {arr.shape}")  # (5,)

# 在位置0增加一个维度
arr_expanded = arr[np.newaxis, :]
print(f"增加批次维度后形状: {arr_expanded.shape}")  # (1, 5)

# 在位置1增加一个维度
arr_expanded2 = arr[:, np.newaxis]
print(f"增加最后一维后形状: {arr_expanded2.shape}")  # (5, 1)

# 也可以使用 None，效果相同
arr_expanded3 = arr[None, :]
print(f"使用None: {arr_expanded3.shape}")  # (1, 5)
```

`np.newaxis` 的本质是在索引位置插入一个大小为 1 的维度。它不会复制任何数据，只是改变了数组的视图。

## np.expand_dims：更明确的增加维度

`np.expand_dims` 是增加维度的更显式的方式：

```python
arr = np.array([1, 2, 3, 4, 5])

# 在轴0增加维度
expanded = np.expand_dims(arr, axis=0)
print(f"在轴0展开: {expanded.shape}")  # (1, 5)

# 在轴1增加维度
expanded = np.expand_dims(arr, axis=1)
print(f"在轴1展开: {expanded.shape}")  # (5, 1)

# 对于高维数组
matrix = np.array([[1, 2, 3], [4, 5, 6]])
print(f"矩阵形状: {matrix.shape}")  # (2, 3)

# 在轴0增加维度
expanded = np.expand_dims(matrix, axis=0)
print(f"在轴0展开: {expanded.shape}")  # (1, 2, 3)

# 在轴2增加维度
expanded = np.expand_dims(matrix, axis=2)
print(f"在轴2展开: {expanded.shape}")  # (2, 3, 1)
```

`np.expand_dims` 的 `axis` 参数指定了新增维度的位置，可以是 0 到 n（数组原维度数）之间的任意值。

## np.squeeze：删除大小为1的维度

`squeeze` 操作与 `expand_dims` 相反，它删除所有大小为 1 的维度：

```python
arr = np.array([[[1, 2, 3]]])
print(f"原始形状: {arr.shape}")  # (1, 1, 3)

# 删除所有大小为1的维度
squeezed = np.squeeze(arr)
print(f"squeeze后形状: {squeezed.shape}")  # (3,)

# 删除特定的大小为1的维度
arr2 = np.array([[[1], [2], [3]]])
print(f"原始形状: {arr2.shape}")  # (1, 3, 1)

# 只删除轴0
squeezed = np.squeeze(arr2, axis=0)
print(f"只删除轴0: {squeezed.shape}")  # (3, 1)

# 删除轴2
squeezed = np.squeeze(arr2, axis=2)
print(f"只删除轴2: {squeezed.shape}")  # (1, 3)
```

## 在深度学习中的应用

### 为模型输入增加批次维度

在推理时，我们通常处理单个样本，需要为其增加批次维度以匹配训练时的批量输入格式：

```python
# 模拟单个样本的隐藏状态
single_hidden = np.random.randn(512, 768).astype(np.float32)
print(f"单个样本形状: {single_hidden.shape}")

# 增加批次维度 (batch_size=1)
batch_hidden = single_hidden[np.newaxis, :, :]
print(f"增加批次维度后: {batch_hidden.shape}")

# 或者使用 reshape
batch_hidden2 = single_hidden.reshape(1, 512, 768)
print(f"reshape后: {batch_hidden2.shape}")

# 增加批次维度更简洁的方式
batch_hidden3 = np.expand_dims(single_hidden, axis=0)
print(f"expand_dims后: {batch_hidden3.shape}")
```

### 适配不同模型的输入格式

某些模型可能需要四维输入，而你的数据可能是三维的：

```python
# 假设我们有一个 (batch, height, width) 的图像批次
images = np.random.randn(8, 224, 224).astype(np.float32)
print(f"图像批次形状: {images.shape}")

# 需要 (batch, height, width, channels) 格式
# 先增加 channels 维度
images_expanded = np.expand_dims(images, axis=3)  # 添加在最后
print(f"增加channels维度后: {images_expanded.shape}")  # (8, 224, 224, 1)

# 如果需要复制到多个 channels
images_multi_channel = np.repeat(images_expanded, 3, axis=3)
print(f"复制到3通道: {images_multi_channel.shape}")  # (8, 224, 224, 3)
```

### Layer Normalization 中的维度处理

在实现 Layer Normalization 时，需要注意 keepdims 参数的使用：

```python
batch_size = 4
seq_len = 512
hidden_dim = 768

hidden_states = np.random.randn(batch_size, seq_len, hidden_dim).astype(np.float32)

# 计算均值，keepdims 保持维度
mean = hidden_states.mean(axis=-1, keepdims=True)
print(f"均值形状（keepdims）: {mean.shape}")  # (4, 512, 1)

# 如果不使用 keepdims
mean_no_keep = hidden_states.mean(axis=-1)
print(f"均值形状（无keepdims）: {mean_no_keep.shape}")  # (4, 512)

# squeeze 可以移除不需要的维度
mean_squeezed = np.squeeze(mean)
print(f"squeeze后形状: {mean_squeezed.shape}")  # (4, 512)
```

### 处理 Variable Length Sequences

在处理变长序列时，经常需要为 padding 后的序列添加掩码：

```python
# 假设有三个长度不同的序列
seq1 = np.random.randn(10, 256)  # 长度10
seq2 = np.random.randn(8, 256)   # 长度8
seq3 = np.random.randn(12, 256)  # 长度12

# padding 到相同长度
max_len = 12
padded = np.zeros((3, max_len, 256))
padded[0, :10] = seq1
padded[1, :8] = seq2
padded[2, :] = seq3

print(f"padding后形状: {padded.shape}")

# 创建注意力掩码
lengths = [10, 8, 12]
masks = np.array([np.concatenate([np.ones(l), np.zeros(max_len - l)]) for l in lengths])
print(f"掩码形状: {masks.shape}")  # (3, 12)

# 为掩码增加维度以用于广播
masks_expanded = masks[:, :, np.newaxis]  # (3, 12, 1)
print(f"掩码扩展后形状: {masks_expanded.shape}")
```

## 常见误区与注意事项

### 误区一：newaxis 和 expand_dims 不会复制数据

```python
arr = np.array([1, 2, 3])
expanded = arr[np.newaxis, :]

# 修改扩展后的数组会影响原始数组吗？
print(f"共享内存: {np.shares_memory(arr, expanded)}")  # True

expanded[0, 0] = 99
print(f"原始数组: {arr}")  # 也会变成 [99, 2, 3]
```

`np.newaxis` 和 `np.expand_dims` 都只是创建了一个新的视图，共享底层数据。修改会影响原始数组。

### 误区二：squeeze 会删除所有大小为1的维度

```python
arr = np.array([[[1, 2, 3]]])
print(f"形状: {arr.shape}")  # (1, 1, 3)

# squeeze 会删除所有大小为1的维度
squeezed = np.squeeze(arr)
print(f"squeeze后: {squeezed.shape}")  # (3,)

# 如果只想保留某个特定维度的大小为1的维度
squeezed2 = np.squeeze(arr, axis=0)
print(f"只删除轴0: {squeezed2.shape}")  # (1, 3)
```

### 误区三：axis 参数范围错误

```python
arr = np.array([1, 2, 3])

# 轴0是正确的
expanded = np.expand_dims(arr, axis=0)
print(f"axis=0: {expanded.shape}")  # (1, 3)

# 轴1是正确的
expanded = np.expand_dims(arr, axis=1)
print(f"axis=1: {expanded.shape}")  # (3, 1)

# 轴2会报错，因为原数组只有2个维度
try:
    expanded = np.expand_dims(arr, axis=2)
    print(f"axis=2: {expanded.shape}")
except np.AxisError as e:
    print(f"轴错误: {e}")
```

### 误区四：reshape(-1) 与 newaxis 的混淆

```python
arr = np.array([1, 2, 3, 4])

# newaxis 增加一个维度
a = arr[np.newaxis, :]  # (1, 4)
b = arr[:, np.newaxis]  # (4, 1)

# reshape(-1, 1) 也可以增加维度，但方式不同
c = arr.reshape(-1, 1)  # (4, 1)
print(f"reshape(-1, 1): {c.shape}")

d = arr.reshape(1, -1)  # (1, 4)
print(f"reshape(1, -1): {d.shape}")
```

## 底层原理

`np.newaxis` 的实现非常巧妙。它实际上是一个继承自 `np.ndarray` 的特殊类，其唯一作用就是在索引表达式中返回一个带有修改后 strides 的视图。当你在索引中使用 `arr[np.newaxis, :]` 时，NumPy 会创建一个新的数组对象，其 shape 为 (1, n)，strides 中对应 newaxis 位置的值为 0。

`squeeze` 的实现则更简单，它返回一个删除了所有大小为 1 的维度的视图。

```python
arr = np.array([[[1, 2, 3]]])
squeezed = np.squeeze(arr)

print(f"原始 shape: {arr.shape}")
print(f"squeezed shape: {squeezed.shape}")
print(f"共享内存: {np.shares_memory(arr, squeezed)}")  # True
```

## 小结

`np.newaxis`、`np.expand_dims` 和 `np.squeeze` 是进行维度变换的三个基本工具。`np.newaxis` 在索引中使用，是增加维度的最简洁方式；`np.expand_dims` 更显式地指定要增加的维度位置；`np.squeeze` 删除所有大小为 1 的维度。这些操作都返回视图而不是副本。

在深度学习中，维度变换无处不在：为单个样本增加批次维度、适配不同模型的输入格式要求、处理变长序列的掩码等。掌握这些维度操作是处理神经网络数据的基础。

面试时需要能够解释 newaxis 和 expand_dims 的区别，理解 squeeze 会删除所有大小为1的维度，以及注意这些操作返回视图而非副本的特性。
