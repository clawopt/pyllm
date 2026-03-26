---
title: 数据类型详解
---

# 数据类型详解

## 为什么需要多种数据类型

计算机用二进制存储数据。不同的数据类型决定了如何解释这些二进制位，以及它们占用多少空间。

比如一个字节（8位），如果解释为无符号整数，范围是0到255；如果解释为有符号整数，范围是-128到127。NumPy提供了丰富的数据类型，让我们可以根据需要选择合适的解释方式。

## 整数类型

NumPy支持多种整数类型，它们占用不同字节数，因此范围也不同：

```python
import numpy as np

# 创建不同精度的整数数组
arr_i8 = np.array([127, -128], dtype=np.int8)
arr_i16 = np.array([32767, -32768], dtype=np.int16)
arr_i32 = np.array([2147483647, -2147483648], dtype=np.int32)
arr_i64 = np.array([9223372036854775807, -9223372036854775808], dtype=np.int64)

print(f"int8范围: {np.iinfo(np.int8).min} ~ {np.iinfo(np.int8).max}")
print(f"int16范围: {np.iinfo(np.int16).min} ~ {np.iinfo(np.int16).max}")
print(f"int32范围: {np.iinfo(np.int32).min} ~ {np.iinfo(np.int32).max}")
print(f"int64范围: {np.iinfo(np.int64).min} ~ {np.iinfo(np.int64).max}")
```

无符号整数只有正数，范围从0开始：

```python
print(f"uint8范围: 0 ~ {np.iinfo(np.uint8).max}")
print(f"uint16范围: 0 ~ {np.iinfo(np.uint16).max}")
```

在AI应用中，整数类型常用于存储token IDs（通常用int32或int64），或者做量化后的权重（用int8）。

## 浮点数类型

浮点数用于表示小数。NumPy支持三种精度的浮点数：

```python
print(f"float16: {np.finfo(np.float16).nbits}位, 范围 ~ {np.finfo(np.float16).max:.0e}")
print(f"float32: {np.finfo(np.float32).nbits}位, 范围 ~ {np.finfo(np.float32).max:.0e}")
print(f"float64: {np.finfo(np.float64).nbits}位, 范围 ~ {np.finfo(np.float64).max:.0e}")
```

float16是半精度，float32是单精度，float64是双精度。位数越多，精度越高，但占用内存也越大。

精度差异的实际表现：

```python
arr_f16 = np.array([0.12345678901234567890], dtype=np.float16)
arr_f32 = np.array([0.12345678901234567890], dtype=np.float32)
arr_f64 = np.array([0.12345678901234567890], dtype=np.float64)

print(f"float16: {arr_f16[0]:.20f}")
print(f"float32: {arr_f32[0]:.20f}")
print(f"float64: {arr_f64[0]:.20f}")
```

float16和float32在很长的数字后面会出现精度损失。

## 布尔类型

布尔类型只有两个值：True和False：

```python
bool_arr = np.array([True, False, True])
print(f"dtype: {bool_arr.dtype}")  # bool

# 布尔类型在索引中非常有用
arr = np.array([1, 2, 3, 4, 5])
mask = arr > 2
print(f"掩码: {mask}")  # [False False True True True]
print(f"筛选: {arr[mask]}")  # [3 4 5]
```

布尔类型在内存中只占1字节，不像有些语言用完整字节。

## 复数类型

NumPy支持复数，这在信号处理等领域很有用：

```python
complex_arr = np.array([1+2j, 3+4j], dtype=np.complex64)
print(f"复数dtype: {complex_arr.dtype}")
print(f"实部: {complex_arr.real}")
print(f"虚部: {complex_arr.imag}")

# 复数的模
print(f"模: {np.abs(complex_arr)}")  # [2.236..., 5.]
```

complex64用两个float32存储，complex128用两个float64存储。

## 模型量化

理解数据类型的一个实际应用是模型量化。量化是用更少位数表示模型参数的技术：

```python
# 假设有一个float32的权重矩阵
weights_fp32 = np.random.randn(1000, 1000).astype(np.float32)
print(f"FP32大小: {weights_fp32.nbytes / 1024:.1f} KB")

# 量化到int8
# 首先计算缩放因子
scale = np.max(np.abs(weights_fp32)) / 127
weights_int8 = (weights_fp32 / scale).astype(np.int8)
print(f"INT8大小: {weights_int8.nbytes / 1024:.1f} KB")

print(f"压缩比: {weights_fp32.nbytes / weights_int8.nbytes:.1f}x")
```

INT8量化可以将内存占用减少到原来的四分之一，同时大多数深度学习推理对这种精度损失不敏感。

## dtype转换

转换dtype用astype方法：

```python
arr = np.array([1.5, 2.7, 3.9])

# 转成整数：小数部分被截断
int_arr = arr.astype(np.int32)
print(f"截断: {int_arr}")  # [1 2 3]

# 转成字符串
str_arr = arr.astype(str)
print(f"字符串: {str_arr}")  # ['1.5' '2.7' '3.9']

# 如果想要四舍五入
rounded = np.round(arr).astype(np.int32)
print(f"四舍五入: {rounded}")  # [2 3 4]
```

astype总是返回一个新数组，不修改原数组。

## dtype的常见陷阱

dtype转换有个常见的坑：

```python
arr = np.array([256, 512, 1024], dtype=np.int32)

# 转成uint8：数据溢出！
uint8_arr = arr.astype(np.uint8)
print(f"溢出: {uint8_arr}")  # [0, 0, 0]
```

int32的范围是-2147483648到2147483647，而uint8的范围是0到255。当大数字转成uint8时，会发生溢出，结果变成0或其他奇怪的值。这个陷阱在处理图像像素时特别容易遇到，因为像素值范围是0-255。

## 查看数组的dtype

可以用dtype属性查看数组的类型：

```python
arr = np.array([1, 2, 3])
print(arr.dtype)  # int64

# 也可以用 isinstance 检查
print(isinstance(arr[0], np.int64))  # True
```

## dtype在AI中的选择

AI应用中选择dtype的经验法则：

训练阶段用float32，因为需要高精度。推理阶段可以用float16或int8来节省内存和加速。Token IDs通常用int32或int64，取决于词汇表大小。Attention mask通常用bool或int8。

理解dtype能帮你更好地优化模型和调试问题。
