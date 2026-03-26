# 转置操作

转置是线性代数中最基础也是最重要的操作之一。想象一下，当你处理一个批次的图像数据时，你可能需要将通道维度从"通道优先"（CHW）转换为"通道在后"（HWC）格式；或者在多头注意力中，你需要交换批次维度和序列维度以便进行矩阵运算。NumPy 提供了多种进行转置的方法：`.T` 属性用于二维数组的简单转置，`np.transpose` 函数用于任意维度的转置，以及 `np.swapaxes` 用于交换两个特定的轴。理解这些转置操作的原理和应用场景，对于处理多维数据至关重要。

## .T 属性：简单的转置

对于二维数组，`.T` 是最简单的转置方式，它会交换行和列：

```python
import numpy as np

matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])
print(f"原始矩阵:\n{matrix}")
print(f"原始形状: {matrix.shape}")

transposed = matrix.T
print(f"转置后:\n{transposed}")
print(f"转置后形状: {transposed.shape}")
```

二维矩阵的转置将 (i, j) 位置的元素移动到 (j, i) 位置。对于 (2, 3) 的矩阵，转置后变成 (3, 2)。

## np.transpose：通用转置

`np.transpose` 允许你指定任意维度的排列顺序：

```python
# 二维数组
matrix = np.array([[1, 2, 3], [4, 5, 6]])
print(f"transpose(0, 1):\n{np.transpose(matrix, (0, 1))}")

# 三维数组
arr_3d = np.random.randn(2, 3, 4)
print(f"三维数组形状: {arr_3d.shape}")

# 交换最后两个维度
transposed_3d = np.transpose(arr_3d, (0, 2, 1))
print(f"转置后形状: {transposed_3d.shape}")

# 完全翻转维度顺序
reversed_3d = np.transpose(arr_3d, (2, 1, 0))
print(f"翻转后形状: {reversed_3d.shape}")
```

`transpose` 的参数是一个元组，指定了原始维度到新维度的映射关系。例如 `(0, 2, 1)` 表示：新数组的第0维是原始数组的第0维，新数组的第1维是原始数组的第2维，新数组的第2维是原始数组的第1维。

## np.swapaxes：交换两个轴

`np.swapaxes` 是 `transpose` 的简化版本，专门用于交换两个轴：

```python
arr_3d = np.random.randn(2, 3, 4)

# 交换第1和第2个轴
swapped = np.swapaxes(arr_3d, 1, 2)
print(f"交换后形状: {swapped.shape}")  # (2, 4, 3)

# 这等价于
transposed = np.transpose(arr_3d, (0, 2, 1))
print(f"等价于transpose: {np.array_equal(swapped, transposed)}")
```

当你只需要交换两个特定的维度时，`swapaxes` 比 `transpose` 更直观。

## 转置与视图

转置操作返回的是视图，而不是副本。这意味着修改转置后的数组会影响原始数组：

```python
matrix = np.array([[1, 2, 3], [4, 5, 6]])
transposed = matrix.T

print(f"转置是视图: {transposed.base is matrix}")  # True

transposed[0, 0] = 99
print(f"原始矩阵: {matrix}")  # matrix[0, 0] 变成了 99
```

这是因为转置只改变了数组的 strides 信息，而不复制数据。

## 在深度学习中的应用

### 多头注意力的维度转换

在 Transformer 的多头注意力实现中，维度转换是核心操作之一。假设我们有形状为 (batch, seq_len, hidden_dim) 的隐藏状态，需要转换为 (batch, num_heads, seq_len, head_dim) 以进行多头注意力计算：

```python
batch_size = 4
seq_len = 512
hidden_dim = 768
num_heads = 12

# 原始隐藏状态 (batch, seq_len, hidden_dim)
hidden_states = np.random.randn(batch_size, seq_len, hidden_dim).astype(np.float32)

# 重塑为 (batch, seq_len, num_heads, head_dim)
head_dim = hidden_dim // num_heads
hidden_reshaped = hidden_states.reshape(batch_size, seq_len, num_heads, head_dim)
print(f"重塑后形状: {hidden_reshaped.shape}")

# 转置为 (batch, num_heads, seq_len, head_dim)
# 需要将 seq_len 和 num_heads 的位置交换
mha_output = np.transpose(hidden_reshaped, (0, 2, 1, 3))
print(f"MHA 输出形状: {mha_output.shape}")
```

这是 Transformer 中计算多头注意力的标准维度变换。

### 图像格式转换

在计算机视觉中，图像数据有不同的存储格式：NCHW（通道优先）和 NHWC（通道在后）：

```python
batch_size = 8
channels = 3
height = 224
width = 224

# NCHW 格式 (batch, channels, height, width)
image_nchw = np.random.randn(batch_size, channels, height, width).astype(np.float32)
print(f"NCHW 形状: {image_nchw.shape}")

# 转换为 NHWC 格式 (batch, height, width, channels)
image_nhwc = np.transpose(image_nchw, (0, 2, 3, 1))
print(f"NHWC 形状: {image_nhwc.shape}")

# 再转回去
image_nchw_restored = np.transpose(image_nhwc, (0, 3, 1, 2))
print(f"还原后形状: {image_nchw_restored.shape}")
```

PyTorch 默认使用 NCHW 格式，而 TensorFlow 默认使用 NHWC 格式，理解这些转换在处理跨框架数据时非常重要。

### 自定义注意力权重的计算

在某些特殊的注意力机制变体中，需要对维度进行特定的变换：

```python
batch_size = 4
seq_len = 10
num_heads = 4

# 模拟 Q, K, V 形状 (batch, num_heads, seq_len, head_dim)
q = np.random.randn(batch_size, num_heads, seq_len, 8)
k = np.random.randn(batch_size, num_heads, seq_len, 8)
v = np.random.randn(batch_size, num_heads, seq_len, 8)

# 计算注意力分数：Q @ K^T
# 需要将 K 转置为 (batch, num_heads, head_dim, seq_len)
# 使用 swapaxes 交换最后两个维度
k_transposed = np.swapaxes(k, -2, -1)  # 等价于 np.swapaxes(k, 2, 3)

# 批量矩阵乘法
attention_scores = np.matmul(q, k_transposed)
print(f"注意力分数形状: {attention_scores.shape}")
```

## 转置与连续性

转置操作可能会使数组变得不连续，这会影响某些操作的性能：

```python
matrix = np.array([[1, 2, 3], [4, 5, 6]])
transposed = matrix.T

print(f"原始矩阵是否连续: {matrix.flags['C_CONTIGUOUS']}")  # True
print(f"转置后是否连续: {transposed.flags['C_CONTIGUOUS']}")  # False

# 如果需要连续数组，可以使用 np.ascontiguousarray
contiguous = np.ascontiguousarray(transposed)
print(f"强制连续后是否连续: {contiguous.flags['C_CONTIGUOUS']}")
```

某些操作（如某些 BLAS 例程）要求数组是连续的，这时需要使用 `np.ascontiguousarray` 强制转换。

## 常见误区与注意事项

### 误区一：混淆 transpose 的参数含义

```python
arr = np.random.randn(2, 3, 4)

# 参数 (0, 1, 2) 表示保持不变
same = np.transpose(arr, (0, 1, 2))
print(f"相同形状: {same.shape}")  # (2, 3, 4)

# 参数 (2, 1, 0) 表示完全翻转
flipped = np.transpose(arr, (2, 1, 0))
print(f"翻转后形状: {flipped.shape}")  # (4, 3, 2)

# 参数 (0, 2, 1) 表示交换第1和第2维
swapped = np.transpose(arr, (0, 2, 1))
print(f"交换后形状: {swapped.shape}")  # (2, 4, 3)
```

transpose 的参数是维度索引的排列，而不是简单的交换。

### 误区二：忘记转置返回的是视图

```python
matrix = np.array([[1, 2], [3, 4]])
transposed = matrix.T

# 修改转置后的元素
transposed[0, 0] = 99

# 原始矩阵也被修改了
print(f"原始矩阵: {matrix}")  # [[99, 2], [3, 4]]
```

### 误区三：对高维数组使用 .T

```python
arr_3d = np.random.randn(2, 3, 4)

# .T 只转置最后两个维度
t_arr = arr_3d.T
print(f".T 形状: {t_arr.shape}")  # (4, 3, 2)

# 如果想交换其他维度，需要用 transpose
swapped = np.transpose(arr_3d, (1, 0, 2))
print(f"transpose(1,0,2) 形状: {swapped.shape}")  # (3, 2, 4)
```

`.T` 只交换最后两个维度，对于更高维的数组，需要使用 `np.transpose`。

## 底层原理

转置操作的底层实现与 reshape 类似，只是改变了 strides 信息。当你对二维数组使用 `.T` 时，NumPy 创建一个新的数组对象，但其底层数据缓冲区是共享的。通过调整 strides，元组访问 `transposed[i, j]` 被映射到 `matrix[j, i]` 的数据。

```python
matrix = np.array([[1, 2, 3], [4, 5, 6]])
transposed = matrix.T

print(f"原始 strides: {matrix.strides}")  # (24, 8) - 每行24字节，每元素8字节
print(f"转置 strides: {transposed.strides}")  # (8, 24) - 反过来了
```

## 小结

NumPy 提供了多种转置操作：`.T` 用于简单的二维转置，`np.transpose` 用于任意维度的转置，`np.swapaxes` 用于交换两个特定维度。转置操作返回的是视图而不是副本，理解这一点对于避免意外的副作用非常重要。

在深度学习中，转置操作无处不在：多头注意力中的维度转换、图像格式的转换（NCHW 与 NHWC）、各种自定义层的维度变换等。掌握这些转置操作的原理和使用场景，是处理多维数据的基础。

面试时需要能够解释转置操作返回视图的原理，理解 `.T`、`transpose` 和 `swapaxes` 三者的区别和适用场景，以及注意转置可能导致的连续性问题。
