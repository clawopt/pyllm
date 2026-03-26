# 单位矩阵与对角矩阵

单位矩阵是线性代数中最基础也最重要的概念之一。想象一下，在数字世界中，单位矩阵就像数字1一样——任何矩阵乘以单位矩阵都等于它自己。NumPy提供了 `np.eye` 函数来创建单位矩阵，顾名思义，"eye" 发音和"I"（我，代表单位矩阵）相同。单位矩阵的特点是：主对角线上的元素全是1，其他位置全是0。在深度学习中，单位矩阵虽然不像全零或全一数组那样常用，但它在某些特定场景下非常重要，比如初始化某些特殊的变换矩阵，或者作为注意力机制中的基础操作。

## np.eye：创建单位矩阵

`np.eye` 是创建单位矩阵的标准方法：

```python
import numpy as np

# 创建 3x3 的单位矩阵
I3 = np.eye(3)
print(f"3x3单位矩阵:\n{I3}")

# 创建 NxN 的单位矩阵
I5 = np.eye(5)
print(f"5x5单位矩阵:\n{I5}")
```

单位矩阵的数学符号是 I，顾名思义，它就像一个"单位"，任何矩阵乘以它都不会改变。形式化地说，如果 A 是一个 m×n 的矩阵，I 是 k×k 的单位矩阵（其中 k 等于 n），那么 AI = A。

## np.eye 的参数

`np.eye` 有几个有用的参数，可以创建更复杂的"单位矩阵"：

```python
# 创建 4x3 的单位矩阵（矩形，不是方阵）
I_rect = np.eye(4, 3)
print(f"4x3单位矩阵:\n{I_rect}")

# 偏移主对角线
I_offset = np.eye(4, k=1)  # 主对角线上方偏移1
print(f"偏移1的单位矩阵:\n{I_offset}")

I_offset_neg = np.eye(4, k=-1)  # 主对角线下方偏移1
print(f"偏移-1的单位矩阵:\n{I_offset_neg}")
```

参数 `k` 控制对角线的偏移量：k=0 是主对角线，k>0 是主对角线上方，k<0 是主对角线下方。这种偏移单位矩阵在某些算法中很有用，比如创建三对角矩阵。

## np.identity：另一种创建单位矩阵的方法

NumPy还提供了 `np.identity` 函数，它只用于创建方阵：

```python
# np.identity 只接受一个参数（只能是方阵）
I3 = np.identity(3)
print(f"identity(3):\n{I3}")
```

`np.identity` 和 `np.eye(3)` 的效果是一样的，但 `np.eye` 更通用（支持矩形矩阵），而 `np.identity` 更简洁。如果你确定需要方阵，用 `np.identity` 会更清晰。

## np.diag：创建对角矩阵

对角矩阵是另一种特殊的矩阵，它只有主对角线上有非零元素。`np.diag` 可以创建对角矩阵，也可以从矩阵中提取对角线：

```python
# 创建对角矩阵
diagonal = np.array([1, 2, 3, 4])
D = np.diag(diagonal)
print(f"对角矩阵:\n{D}")

# 从数组创建指定位置的对角矩阵
D_offset = np.diag(diagonal, k=1)
print(f"偏移1的对角矩阵:\n{D_offset}")

# 从现有矩阵提取对角线
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
extracted_diag = np.diag(matrix)
print(f"提取的对角线: {extracted_diag}")

extracted_diag_offset = np.diag(matrix, k=1)
print(f"提取的偏移1对角线: {extracted_diag_offset}")
```

`np.diag` 是一个"双用途"函数：传入一维数组时创建对角矩阵，传入二维数组时提取对角线。面试时可能会问这个函数的用法，要注意区分两种用法。

## 在深度学习中的应用

### 注意力机制中的单位矩阵

在某些注意力机制的变体中，单位矩阵用于创建简单的基线：

```python
seq_len = 5

# 创建一个"恒等注意力"的基线
identity_attention = np.eye(seq_len, dtype=np.float32)
print(f"恒等注意力掩码:\n{identity_attention}")
```

恒等注意力是一种简化的注意力机制，它只关注相同位置的信息。这种注意力虽然简单，但可以作为实验的基线。

### 初始化残差连接的变换矩阵

在残差连接中，有时候需要初始化一个稍微偏离单位矩阵的变换：

```python
hidden_dim = 768

# 初始化为接近单位矩阵的权重
# 这种初始化被称为"残差初始化"
residual_weight = np.eye(hidden_dim, dtype=np.float32) * 1.0
print(f"残差权重形状: {residual_weight.shape}")
```

在某些Transformer的改进中，比如 Pre-LN Transformer，残差分支的权重会被初始化为接近单位矩阵的形式，这样可以确保训练初期各层能够更好地传递梯度。

### 创建掩码的辅助工具

单位矩阵可以用于创建某些特殊的注意力掩码：

```python
seq_len = 6

# 创建局部注意力的掩码（只关注邻近位置）
local_window = 2
local_mask = np.eye(seq_len, dtype=np.float32)  # 从单位矩阵开始

# 构建局部注意力模式
for i in range(seq_len):
    for j in range(seq_len):
        if abs(i - j) > local_window:
            local_mask[i, j] = 0  # 超出窗口的位置设为0

print(f"局部注意力掩码（窗口={local_window}）:\n{local_mask}")
```

## 常见误区与注意事项

### 误区一：混淆 np.eye 和 np.identity 的用法

```python
# 正确：np.eye 可以创建矩形矩阵
rect = np.eye(3, 4)
print(f"矩形单位矩阵形状: {rect.shape}")

# 错误：np.identity 只能创建方阵
try:
    wrong = np.identity(3, 4)  # 会报错
except TypeError as e:
    print(f"错误: {e}")
```

### 误区二：忘记 dtype 参数

```python
# 默认创建的是 float64
I_default = np.eye(3)
print(f"默认dtype: {I_default.dtype}")

# 深度学习中应该用 float32
I_float32 = np.eye(3, dtype=np.float32)
print(f"float32 dtype: {I_float32.dtype}")
```

### 误区三：对角线索引记错

```python
n = 5

# 创建一个上三角都是1的矩阵（包含主对角线）
upper = np.ones((n, n))
upper = np.triu(upper)  # 上三角
print(f"上三角矩阵:\n{upper}")

# 创建下三角都是1的矩阵（包含主对角线）
lower = np.ones((n, n))
lower = np.tril(lower)  # 下三角
print(f"下三角矩阵:\n{lower}")
```

## 底层原理

单位矩阵之所以特殊，是因为它是矩阵乘法的"单位元"。在数学中，如果 a 是实数，那么 a × 1 = a。对于矩阵，这个"1"就是单位矩阵 I。

单位矩阵还有一个重要性质：它的逆矩阵是它自己。即 I^(-1) = I。这是因为 I × I = I，所以 I 乘以 I 就得到 I 本身。

从数值计算的角度看，单位矩阵在内存中存储效率很高——只需要存储主对角线上的 n 个1，而不是 n² 个元素。NumPy 的 `np.eye` 函数内部也是这样优化的，它不会真的创建一个 n² 的二维数组然后填1，而是只存储对角线的信息（实际上存储的仍然是一个完整的二维数组，但某些操作可以识别其稀疏性）。

## np.eye 的其他用途

### 创建一个热编码矩阵

在某些场景下，可以用 `np.eye` 快速创建 one-hot 编码：

```python
num_classes = 5
indices = np.array([0, 2, 4, 1, 3])

# 将类别索引转换为 one-hot 编码
one_hot = np.eye(num_classes)[indices]
print(f"One-hot编码:\n{one_hot}")
```

这里的技巧是利用索引来从单位矩阵中选取行。`np.eye(num_classes)` 创建了一个 num_classes × num_classes 的单位矩阵，然后用索引数组作为行索引来选取。

### 创建掩码的快捷方式

```python
batch_size = 4
seq_len = 8

# 创建一个(batch, seq_len, seq_len)的注意力掩码，初始为单位矩阵
attention_mask = np.broadcast_to(np.eye(seq_len, dtype=np.float32), 
                                  (batch_size, seq_len, seq_len))
print(f"广播后的掩码形状: {attention_mask.shape}")
```

`np.broadcast_to` 是一个高效的函数，它不会真正复制数据，而是创建一个共享底层数据的新视图。

## 小结

单位矩阵和对角矩阵是线性代数中的基础概念。`np.eye` 用于创建单位矩阵，`np.identity` 是创建方阵单位矩阵的简化写法，`np.diag` 既是创建对角矩阵的工具，也是提取对角线的函数。在深度学习中，这些函数虽然不如 `np.zeros` 或 `np.ones` 那么常用，但在创建注意力掩码、初始化残差连接、构建特殊矩阵结构等场景中有重要应用。

面试时需要能够解释单位矩阵的数学意义，理解 `np.eye`、`np.identity`、`np.diag` 三个函数的区别和联系，以及能够使用这些函数解决实际问题。
