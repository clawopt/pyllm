# 查看形状与 reshape

在NumPy中，形状（shape）是数组最重要的属性之一。想象一下，当你处理一个批次的文本嵌入时，你首先需要知道这个数组的形状才能正确地进行矩阵乘法；或者当你需要将一个一维的序列转换为二维矩阵以便进行特定操作时，就需要用到 reshape。`shape` 属性告诉你数组各维度的大小，`reshape` 方法则允许你改变数组的形状而不改变其数据。理解这两个操作是处理多维数据的基础，无论是在数据分析还是在深度学习中都非常重要。

## 查看数组形状

每个NumPy数组都有一个 `shape` 属性，返回一个元组表示各维度的大小：

```python
import numpy as np

# 一维数组
arr_1d = np.array([1, 2, 3, 4, 5])
print(f"一维数组形状: {arr_1d.shape}")  # (5,)

# 二维数组
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
print(f"二维数组形状: {arr_2d.shape}")  # (2, 3)

# 三维数组（常见于图像数据）
arr_3d = np.random.randn(4, 32, 32)
print(f"三维数组形状: {arr_3d.shape}")  # (4, 32, 32)

# 四维数组（常见于卷积神经网络的批量图像）
arr_4d = np.random.randn(8, 3, 224, 224)  # batch_size, channels, height, width
print(f"四维数组形状: {arr_4d.shape}")  # (8, 3, 224, 224)
```

形状元组的每个元素依次表示第0轴、第1轴、第2轴...的大小。对于深度学习中常见的数据，理解形状含义非常重要。

## reshape：改变数组形状

`reshape` 可以在不改变数据的前提下将数组调整为新的形状：

```python
# 一维数组
arr = np.arange(12)
print(f"原始数组: {arr}")

# 改变为二维
matrix = arr.reshape(3, 4)
print(f"reshape(3, 4):\n{matrix}")

# 改变为三维
cube = arr.reshape(2, 3, 2)
print(f"reshape(2, 3, 2):\n{cube}")
```

`reshape` 总元素数必须与原数组相同。上例中 12 个元素可以变成 (3, 4)、(2, 6)、(12, 1) 等任意形状，但不能变成 (3, 5) 这样元素数不匹配的形状。

## -1 的自动推断

`reshape` 中可以使用 -1 让 NumPy 自动计算该维度的大小：

```python
arr = np.arange(24)

# 自动计算为 4
a = arr.reshape(-1, 6)
print(f"reshape(-1, 6) = {a.shape}")  # (4, 6)

# 自动计算为 6
b = arr.reshape(4, -1)
print(f"reshape(4, -1) = {b.shape}")  # (4, 6)

# 自动计算为 4
c = arr.reshape(2, 3, -1)
print(f"reshape(2, 3, -1) = {c.shape}")  # (2, 3, 4)
```

这是非常实用的功能，尤其在处理批量数据时。假设我们不知道数组的总元素数，但知道需要 8 行：

```python
arr = np.random.randn(100, 50)
flattened = arr.reshape(arr.shape[0], -1)
print(f"展平后形状: {flattened.shape}")  # (100, 50)
```

## reshape 与视图

这是一个极其重要的概念：`reshape` 返回的是视图，不是副本。这意味着修改 reshape 后的数组会影响原始数组：

```python
arr = np.arange(12)
reshaped = arr.reshape(3, 4)

print(f"reshape 是视图: {reshaped.base is arr}")  # True

reshaped[0, 0] = 99
print(f"修改后原始数组: {arr}")  # arr[0] 变成了 99
```

这个特性既是优点（节省内存）也是陷阱（可能意外修改数据）。如果需要副本，使用 `.copy()`。

## 在深度学习中的应用

### 展平嵌入向量

在处理词嵌入时，经常需要将嵌入向量展平以作为全连接层的输入：

```python
batch_size = 4
seq_len = 10
embed_dim = 768

# 模拟嵌入输出
embeddings = np.random.randn(batch_size, seq_len, embed_dim).astype(np.float32)
print(f"嵌入输出形状: {embeddings.shape}")

# 展平为 (batch, seq_len * embed_dim)
embeddings_flat = embeddings.reshape(batch_size, -1)
print(f"展平后形状: {embeddings_flat.shape}")
```

### Reshape 为多头注意力准备

在多头注意力中，需要将隐藏维度拆分为多个头：

```python
batch_size = 4
seq_len = 512
hidden_dim = 768
num_heads = 12

# 模拟注意力输入
hidden_states = np.random.randn(batch_size, seq_len, hidden_dim).astype(np.float32)

# Reshape 为 (batch, seq_len, num_heads, head_dim)
head_dim = hidden_dim // num_heads
hidden_reshaped = hidden_states.reshape(batch_size, seq_len, num_heads, head_dim)
print(f"多头注意力形状: {hidden_reshaped.shape}")

# 如果需要调整维度顺序
# (batch, seq_len, num_heads, head_dim) -> (batch, num_heads, seq_len, head_dim)
hidden_transposed = hidden_reshaped.transpose(0, 2, 1, 3)
print(f"转置后形状: {hidden_transposed.shape}")
```

### 图像数据的 reshape

在处理图像批次时，经常需要在不同形状之间转换：

```python
batch_size = 16
height, width = 224, 224
channels = 3

# 展平的图像数据（如从文件读取）
flat_pixels = np.random.randn(batch_size, height * width * channels).astype(np.float32)

# Reshape 回图像格式
images = flat_pixels.reshape(batch_size, channels, height, width)  # NCHW 格式
print(f"NCHW 形状: {images.shape}")

# 转换为 NHWC 格式（TensorFlow 默认）
images_nhwc = images.transpose(0, 2, 3, 1)
print(f"NHWC 形状: {images_nhwc.shape}")
```

## 常见误区与注意事项

### 误区一：reshape 后忘记总元素数必须匹配

```python
arr = np.arange(12)

# 错误：12 个元素不能变成 3x5=15 的形状
try:
    wrong = arr.reshape(3, 5)
except ValueError as e:
    print(f"形状不匹配错误: {e}")

# 正确
correct = arr.reshape(3, 4)  # 3x4=12
```

### 误区二：误以为 reshape 会复制数据

```python
arr = np.arange(12)
reshaped = arr.reshape(3, 4)

# 两者共享底层数据
reshaped[0, 0] = 99
print(f"原始数组: {arr}")  # 会被修改

# 如果需要独立的副本
independent_copy = arr.reshape(3, 4).copy()
independent_copy[0, 0] = 0
print(f"原始数组: {arr}")  # 不受影响
```

### 误区三：连续性与 C/F order

对于非连续数组，reshape 可能需要复制数据：

```python
# 非连续数组
arr = np.arange(12).reshape(3, 4).T  # 转置后变为非连续
print(f"是否连续: {arr.flags['C_CONTIGUOUS']}")

# reshape 可能需要先复制
reshaped = arr.reshape(2, 6)  # 会自动复制
print(f"reshaped 是新数组: {reshaped.base is not arr}")
```

## 底层原理

NumPy 的 `reshape` 操作通过改变数组的元数据（特别是 strides）来实现，而不需要复制底层数据。Strides 元组描述了如何从数据的起始位置计算每个维度的元素偏移量。

```python
arr = np.arange(12).reshape(3, 4)
print(f"shape: {arr.shape}")  # (3, 4)
print(f"strides: {arr.strides}")  # (32, 8) - 每行32字节，每元素8字节

arr2 = arr.reshape(4, 3)
print(f"reshape(4,3) strides: {arr2.strides}")  # (24, 8) - 不同 strides
```

当数组是连续的时候，reshape 非常高效；但如果数组不连续，NumPy 会自动创建一个副本。

## 小结

`shape` 属性和 `reshape` 方法是NumPy中最基础也最常用的形状操作。`shape` 告诉你数组各维度的大小，`reshape` 允许你改变数组形状而不改变数据。最重要的是理解 `reshape` 返回的是视图而不是副本，这在某些情况下是优势（节省内存），在另一些情况下可能导致意外修改。

在深度学习中，reshape 广泛用于：展平嵌入向量为全连接层做准备、将隐藏状态 reshape 为多头注意力的格式、处理图像数据的维度转换等。

面试时需要能够解释 reshape 与副本/视图的关系，理解 -1 参数的自动推断机制，以及注意连续性对 reshape 性能的影响。
