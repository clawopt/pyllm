---
title: 从列表创建数组
---

# 从列表创建数组

大多数情况下，你的第一个NumPy数组都是从Python列表创建的。这一章详细讲解这个最基本的操作，以及其中的细节。

## 数组创建的起点

假设你有一段Python代码，里面有一些数据存在列表里：

```python
python_list = [1, 2, 3, 4, 5]
```

这个列表很普通，你可以对它做各种Python操作，但如果你想做数学计算——比如给每个元素平方然后求和——你得写循环：

```python
result = sum(x**2 for x in python_list)
print(result)
```

这能工作，但慢。NumPy的做法是先把列表转成数组：

```python
import numpy as np

python_list = [1, 2, 3, 4, 5]
arr = np.array(python_list)

print(arr)           # [1 2 3 4 5]
print(type(arr))     # <class 'numpy.ndarray'>
print(arr ** 2)      # [ 1  4  9 16 25]
print(np.sum(arr ** 2))  # 55
```

就这么简单。`np.array()`接受一个Python列表，返回一个NumPy数组。之后的运算就是向量化的了，速度比循环快几十倍。

## 创建二维数组

把嵌套列表传入np.array就得到多维数组：

```python
matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])

print(matrix)
# [[1 2 3]
#  [4 5 6]]
print(matrix.shape)  # (2, 3)
```

这里创建了一个2行3列的二维数组。NumPy会把它打印成矩阵的样式便于阅读。

三维数组也是类似的：

```python
cube = np.array([[[1, 2], [3, 4]],
                  [[5, 6], [7, 8]]])

print(cube.shape)  # (2, 2, 2)
```

## dtype：数据的类型

你可能注意到了，数组里面的元素会自动有某种"类型"。NumPy会根据你传入的数据推断出合适的类型：

```python
arr = np.array([1, 2, 3])
print(arr.dtype)  # int64

arr = np.array([1.0, 2.0, 3.0])
print(arr.dtype)  # float64

arr = np.array([True, False, True])
print(arr.dtype)  # bool
```

NumPy会在必要时自动向上转型——如果你列表里既有整数又有浮点数，整数会被转成浮点数。

dtype是NumPy数组最重要的属性之一。它决定了数据在内存中如何存储，以及能做哪些运算。你可以用dtype参数显式指定类型：

```python
arr = np.array([1, 2, 3], dtype=np.float32)
print(arr.dtype)  # float32

arr = np.array([1, 2, 3], dtype=np.complex64)
print(arr.dtype)  # complex64
```

## dtype的详细属性

dtype对象有很多有用的属性，理解它们能帮你更好地处理数据。

字符编码是最常用的属性之一：

```python
dt = np.dtype('Float64')
print(dt.char)  # 'd'
```

不同类型有不同的字符码：'b'是int8，'h'是int16，'i'是int32，'q'是int64，'f'是float32，'d'是float64，'g'是float128。

type属性对应数组元素的具体类型：

```python
dt = np.dtype('Float64')
print(dt.type)  # <class 'numpy.float64'>
```

这个type是Python内置的type对象，可以用isinstance检查：

```python
arr = np.array([1.0, 2.0], dtype=np.float64)
print(isinstance(arr[0], np.float64))  # True
```

str属性给出类型的字符串表示，通常是"字节序+类型字符+字节数"的形式：

```python
dt = np.dtype('Float64')
print(dt.str)  # '<f8'

dt = np.dtype('>f4')  # 大端序的float32
print(dt.str)  # '>f4'
```

这里的'>'表示大端序（big-endian），'<'表示小端序（little-endian）。大端序是把高位字节放在前面，小端序是把低位字节放在前面。大部分现代电脑都是小端序，所以这个通常不需要关心。

## 创建时的类型转换陷阱

如果你从一个列表创建数组，然后想转成整数，要小心：

```python
arr = np.array([1.5, 2.7, 3.9])
print(arr.dtype)  # float64

int_arr = arr.astype(np.int32)
print(int_arr)  # [1 2 3]
```

astype返回一个新的数组，小数部分被截断了。如果你想要四舍五入，要用np.round：

```python
arr = np.array([1.5, 2.5, 3.5])
print(arr.astype(np.int32))  # [1 2 3]  截断
print(np.round(arr).astype(np.int32))  # [2 2 4]  四舍五入
```

另一个常见陷阱是复数的处理：

```python
arr = np.array([1+2j, 3+4j])
print(arr.dtype)  # complex128

# 如果你只取实部
print(arr.real.dtype)  # float64
```

arr.real返回实部，是一个新的float64数组。

## 嵌套列表的坑

创建二维数组时，内部列表长度不一致会导致问题：

```python
# 正确：所有内部列表长度相同
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr.shape)  # (2, 3)

# 错误：内部列表长度不同
try:
    arr = np.array([[1, 2], [3, 4, 5]])
except ValueError as e:
    print(f"错误: {e}")
```

NumPy会尽最大努力创建数组，但如果形状不一致，它可能只创建部分数据或者报错。这个错误在处理真实数据时很常见，比如CSV文件每行长度不一致。

## asarray vs array

创建数组还有另一个函数np.asarray，它和np.array的区别在于：np.asarray尽量不复制数据：

```python
original = np.array([1, 2, 3])

# np.array总是创建副本
copy1 = np.array(original)
print(copy1 is original)  # False

# np.asarray尽量返回视图
view = np.asarray(original)
print(view is original)  # True
```

但这个区别其实没那么重要，因为大多数时候我们不会在意副本还是视图。真正需要关心的是当你修改数组时，原数据会不会受影响——那是另一回事。

## 创建数组后的修改

最后记住，创建数组后修改元素不会影响原列表：

```python
python_list = [1, 2, 3]
arr = np.array(python_list)

arr[0] = 100
print(arr)       # [100 2 3]
print(python_list)  # [1, 2, 3]  原列表不变
```

因为np.array会复制数据。但如果你的原数据本身就是ndarray，会有一些特殊情况涉及到视图和副本，这是下一章会详细讲的内容。
