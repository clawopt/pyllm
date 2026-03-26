# 全零与全一数组

在NumPy中，创建数组最简单也最常用的方法之一就是用 `np.zeros` 和 `np.ones` 生成全零或全一的数组。你可能会问，为什么需要创建全零或全一的数组呢？想象一下，当你需要初始化神经网络的偏置项时，偏置通常初始化为零；当你需要创建一个掩码（mask）来标记某些位置需要被忽略时，全零或全一的数组就派上用场了。`np.zeros` 用于创建全是0的数组，`np.ones` 用于创建全是1的数组。这两个函数看起来很简单，但它们在深度学习中的应用非常广泛，从权重初始化到注意力掩码，从占位符到临时缓冲区，到处都能看到它们的身影。

## 基本用法

`np.zeros` 和 `np.ones` 的用法非常直观，只需要指定数组的形状就可以了：

```python
import numpy as np

# 创建一维全零数组
zeros_1d = np.zeros(5)
print(f"一维全零: {zeros_1d}")

# 创建二维全一数组
ones_2d = np.ones((3, 4))
print(f"二维全一:\n{ones_2d}")

# 创建指定数据类型的全零数组
zeros_float32 = np.zeros((2, 3), dtype=np.float32)
print(f"float32 全零: {zeros_float32.dtype}")
```

需要特别注意，`np.zeros` 和 `np.ones` 的第一个参数是形状（shape），它可以是一个整数（创建一维数组），也可以是一个元组（创建多维数组）。如果误把元组写成两个参数，会得到意想不到的结果。

## dtype 参数的重要性

在深度学习中，数据类型的选择直接影响内存占用和计算精度。`np.zeros` 和 `np.ones` 默认创建的是 `float64` 类型的数组，但在实际应用中，我们通常使用 `float32` 来节省内存：

```python
# 默认 float64
zeros_64 = np.zeros((1000, 1000))
print(f"float64 内存: {zeros_64.nbytes / 1024 / 1024:.2f} MB")

# 指定 float32
zeros_32 = np.zeros((1000, 1000), dtype=np.float32)
print(f"float32 内存: {zeros_32.nbytes / 1024 / 1024:.2f} MB")
```

内存减半！在处理大型模型时，这个差异非常显著。比如一个形状为 (batch_size=32, seq_len=512, hidden_dim=768) 的隐藏状态，float32 占用约 48MB，而 float64 占用约 96MB。

## 在深度学习中的典型应用

### 初始化神经网络偏置

神经网络的偏置通常初始化为零：

```python
batch_size = 32
seq_len = 512
num_heads = 12
head_dim = 64

# Attention 中的偏置初始化
attention_bias = np.zeros((num_heads, seq_len, seq_len), dtype=np.float32)
print(f"注意力偏置形状: {attention_bias.shape}")
```

为什么偏置初始化为零？这是因为在训练初期，我们希望模型的行为主要受到权重的影响，而不是偏置。如果偏置初始化过大，可能会导致激活值过大，从而引发梯度爆炸。

### 创建注意力掩码

在Transformer中，注意力掩码用于屏蔽某些位置（比如padding位置或未来位置）：

```python
seq_len = 20
padding_len = 5

# 创建padding掩码：有效位置为1，padding位置为0
padding_mask = np.ones(seq_len, dtype=np.float32)
padding_mask[-padding_len:] = 0
print(f"Padding掩码: {padding_mask}")

# 创建因果掩码：上三角为0（屏蔽未来位置），下三角为1
causal_mask = np.ones((seq_len, seq_len), dtype=np.float32)
causal_mask = np.triu(causal_mask, k=1)  # 上三角（包括对角线）设为0
causal_mask = 1 - causal_mask  # 反转：上三角变0，下三角变1
print(f"因果掩码形状: {causal_mask.shape}")
```

这里 `np.triu` 是一个很有用的函数，它返回上三角矩阵（包含对角线）。通过设置 k 参数可以控制包含多少个对角线。

### 初始化LayerNorm参数

LayerNorm 中有一些可学习参数，初始化时 γ（缩放参数）通常为1，β（偏移参数）通常为0：

```python
hidden_dim = 768

# LayerNorm 的初始化
gamma = np.ones(hidden_dim, dtype=np.float32)  # 缩放参数
beta = np.zeros(hidden_dim, dtype=np.float32)  # 偏移参数
print(f"LayerNorm参数形状: gamma={gamma.shape}, beta={beta.shape}")
```

## empty、full 与 full_like

NumPy还提供了其他创建数组的函数：

```python
# np.empty: 创建未初始化的数组（内存中的随机值）
empty_arr = np.empty((3, 4))
print(f"empty数组（可能有随机值）: {empty_arr[:2, :2]}")

# np.full: 创建指定值的数组
full_arr = np.full((3, 4), 7)
print(f"full数组（全为7）:\n{full_arr}")

# np.full_like: 创建与给定数组形状相同的指定值数组
arr = np.zeros((2, 3), dtype=np.float32)
full_like_arr = np.full_like(arr, 3.14)
print(f"full_like数组: {full_like_arr}")
```

`np.empty` 的优势是创建速度快，因为它不需要初始化内存。但使用前一定要记得赋值，否则数组中会有随机值，可能导致难以调试的bug。

## 常见误区与注意事项

### 误区一：忘记形状是元组

```python
# 错误写法
try:
    wrong = np.zeros(3, 4)  # 这会被当作两个参数处理
except Exception as e:
    print(f"错误: {e}")

# 正确写法
correct = np.zeros((3, 4))  # 元组
```

### 误区二：使用float64而不自知

在深度学习中，float64通常是多余的，而且会浪费一半的内存和一半的计算时间：

```python
# 检查默认类型
zeros_default = np.zeros((2, 2))
print(f"默认dtype: {zeros_default.dtype}")  # float64

# 在深度学习中通常应该用
zeros_for_dl = np.zeros((2, 2), dtype=np.float32)
print(f"DL常用dtype: {zeros_for_dl.dtype}")  # float32
```

### 误区三：与Python列表混淆

```python
# 这是一个包含5个元素的列表
python_list = [0, 0, 0, 0, 0]
print(f"Python列表: {python_list}")

# 这是形状为(5,)的NumPy数组
numpy_array = np.zeros(5)
print(f"NumPy数组: {numpy_array}")
```

两者的区别不仅仅是类型不同，NumPy数组支持向量化操作、广播等高级特性，而Python列表不支持。

## 底层原理

`np.zeros` 和 `np.ones` 的底层实现其实很有讲究。当你在NumPy中调用这些函数时，它实际上是在内存中分配一块连续的空间，然后用0或1填充。对于大型数组，这比用Python循环赋值要快得多，因为NumPy底层调用的是高度优化的C代码。

更深入一点，NumPy创建的数组默认是C-order（行优先）存储的。这意味着同一行的元素在内存中是连续存储的，访问时 locality 更好，CPU缓存利用率更高。如果你需要 Fortran-order（列优先），可以指定 `order='F'` 参数。

## 小结

`np.zeros` 和 `np.ones` 是NumPy中最基础也最常用的数组创建函数。它们在深度学习中有着广泛的应用：初始化偏置、创建掩码、初始化LayerNorm参数、创建临时缓冲区等。正确使用这些函数，特别是选择合适的dtype，是编写高效深度学习代码的基础。

面试时需要能够解释为什么要对 dtype 进行显式指定，了解 float32 与 float64 在深度学习中的取舍，以及理解 `np.zeros` 和 `np.ones` 在神经网络初始化中的作用原理。
