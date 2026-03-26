---
title: ndarray属性详解
---

# ndarray属性详解

ndarray（N-dimensional array）是NumPy的核心数据结构，它是一个多维、同质的数据容器，可以高效地存储和操作大规模数值数据。ndarray中的所有元素必须是相同的数据类型，这使得它在内存中连续存储，从而实现快速的向量化运算。

## 五个核心属性

每个NumPy数组都有五个核心属性，就像每个人都有身高、体重、年龄等属性一样：

```python
import numpy as np

arr = np.array([[1, 2, 3],
                [4, 5, 6]])

print(f"shape: {arr.shape}")      # (2, 3)
print(f"dtype: {arr.dtype}")      # int64
print(f"ndim: {arr.ndim}")       # 2
print(f"size: {arr.size}")       # 6
print(f"itemsize: {arr.itemsize}") # 8
```

## shape：形状

shape是最常用的属性，它是一个元组，描述数组各个维度的大小：

```python
# 一维数组
arr1d = np.array([1, 2, 3, 4, 5])
print(f"一维shape: {arr1d.shape}")  # (5,)

# 二维数组：2行3列
arr2d = np.array([[1, 2, 3],
                   [4, 5, 6]])
print(f"二维shape: {arr2d.shape}")  # (2, 3)

# 三维数组
arr3d = np.random.randn(2, 3, 4)  # 2个3×4的矩阵
print(f"三维shape: {arr3d.shape}")  # (2, 3, 4)
```

shape的含义是"每一维度的元素数量"。对于图像数据，常见shape是(batch, height, width, channels)；对于文本数据，常见shape是(batch, seq_len)。

## dtype：数据类型

dtype描述数组元素的数据类型：

```python
arr = np.array([1, 2, 3])
print(f"默认整数类型: {arr.dtype}")  # int64

arr = np.array([1.0, 2.0, 3.0])
print(f"默认浮点类型: {arr.dtype}")  # float64

arr = np.array([True, False, True])
print(f"布尔类型: {arr.dtype}")  # bool
```

dtype的字符编码可以用dt.char获取：

```python
dt = np.dtype('Float64')
print(dt.char)  # 'd'
```

不同类型有不同的字符码：'b'是int8，'h'是int16，'i'是int32，'q'是int64，'f'是float32，'d'是float64。

dtype的type属性对应数组元素的具体类型：

```python
dt = np.dtype('Float64')
print(dt.type)  # <class 'numpy.float64'>
```

dtype的str属性给出类型的字符串表示，通常是"字节序+类型字符+字节数"的形式：

```python
dt = np.dtype('Float64')
print(dt.str)  # '<f8'

dt = np.dtype('>f4')  # 大端序的float32
print(dt.str)  # '>f4'
```

这里的'>'表示大端序，'<'表示小端序。大端序是把高位字节放在前面，小端序是把低位字节放在前面。大部分现代电脑都是小端序，所以这个通常不需要关心。

## ndim：维度数

ndim是数组的维度数量：

```python
arr1d = np.array([1, 2, 3])
print(f"一维ndim: {arr1d.ndim}")  # 1

arr2d = np.array([[1, 2], [3, 4]])
print(f"二维ndim: {arr2d.ndim}")  # 2

arr3d = np.random.randn(2, 3, 4)
print(f"三维ndim: {arr3d.ndim}")  # 3
```

## size：元素总数

size是数组中所有元素的总数：

```python
arr = np.array([[1, 2, 3],
                [4, 5, 6]])
print(f"size: {arr.size}")  # 6
print(f"等于shape的乘积: {np.prod(arr.shape)}")  # 6
```

## itemsize：单个元素字节大小

itemsize是每个元素占用的字节数：

```python
arr_int64 = np.array([1, 2, 3], dtype=np.int64)
print(f"int64 itemsize: {arr_int64.itemsize}")  # 8

arr_int32 = np.array([1, 2, 3], dtype=np.int32)
print(f"int32 itemsize: {arr_int32.itemsize}")  # 4

arr_float32 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
print(f"float32 itemsize: {arr_float32.itemsize}")  # 4
```

## nbytes：总字节数

计算数组占用的总内存很方便：

```python
arr = np.random.randn(1000, 1000)
print(f"float64数组占内存: {arr.nbytes / 1024 / 1024:.2f} MB")

arr_f32 = arr.astype(np.float32)
print(f"float32数组占内存: {arr_f32.nbytes / 1024 / 1024:.2f} MB")
```

转换为float32可以节省一半内存。在大模型推理中常用。

## strides：步长

strides描述在各个维度上移动时需要跳过的字节数，这是一个稍微高级的属性：

```python
arr = np.array([[1, 2, 3],
                [4, 5, 6]], dtype=np.int32)
print(f"shape: {arr.shape}")   # (2, 3)
print(f"strides: {arr.strides}")  # (12, 4)
```

strides的含义是：想在第一维（行）移动到下一个元素，需要跳过12字节；想在第二维（列）移动到下一个元素，需要跳过4字节。

对于连续内存的数组，strides[0] = itemsize × shape[1]，strides[1] = itemsize。

## C顺序 vs Fortran顺序

NumPy数组在内存中可以是行优先（C顺序）或列优先（Fortran顺序）：

```python
arr_c = np.array([[1, 2, 3],
                   [4, 5, 6]], order='C')  # C顺序，行优先
arr_f = np.array([[1, 2, 3],
                   [4, 5, 6]], order='F')  # Fortran顺序，列优先

print(f"C顺序 strides: {arr_c.strides}")  # (24, 8) for float64
print(f"Fortran顺序 strides: {arr_f.strides}")  # (16, 8) for float64
```

在深度学习框架中，理解内存布局有助于避免不必要的数据复制。大多数情况下不需要关心这个，NumPy会自动选择最优方式。
