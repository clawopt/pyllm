---
title: NumPy vs Python列表
---

# NumPy vs Python列表

这一章深入对比NumPy数组和Python列表的差异，解释为什么NumPy能快这么多。

## 一个速度对比实验

先来看一个真实的例子。我们计算1到1000万的平方和，看看两种方式差多少：

```python
import numpy as np
import time

size = 10_000_000

# Python列表的做法
python_list = list(range(size))
start = time.time()
python_sum = sum(x ** 2 for x in python_list)
python_time = time.time() - start

# NumPy数组的做法
numpy_array = np.arange(size, dtype=np.int64)
start = time.time()
numpy_sum = np.sum(numpy_array ** 2)
numpy_time = time.time() - start

print(f"Python列表: {python_time:.3f}秒, 结果={python_sum}")
print(f"NumPy数组: {numpy_time:.3f}秒, 结果={numpy_sum}")
print(f"NumPy快 {python_time/numpy_time:.1f} 倍")
```

在我这台电脑上，NumPy通常快50到100倍。如果你的电脑更快或者更慢，数字会有变化，但结论是一样的：NumPy碾压性的快。

## 为什么差这么多

要理解这个差异，得从计算机的底层说起。

Python列表存储的是"指针"，每个元素都是一个完整的Python对象。整数1在列表里不是单纯的1，而是一个完整的Python int对象，包含了类型信息、引用计数等额外数据。这些对象在内存中是分散存储的，就像一串珍珠——每颗珍珠（Python对象）单独存放，用线（指针）串起来。

NumPy数组完全不同。数组里的所有元素紧密排列在一个连续的内存块中。就像一长条巧克力，每个格子紧密相连，没有任何间隙。

打个比方：Python列表像是在停车场里找车——每辆车在不同的位置，你得一个个找；NumPy像是在军队列队——所有人在一排，喊一声报数就知道谁在哪。

这种存储方式带来的差异是根本性的。NumPy的运算可以一次性处理一大块连续内存，而Python列表需要逐个取出元素、计算、再存回去。

## 内存占用的差异

速度之外，内存占用也差很多：

```python
import sys

# Python列表：每个整数是一个完整对象
python_list = [1] * 1000
print(f"Python列表1000个int: {sys.getsizeof(python_list)} 字节")
# 加上每个int对象的内存...
# Python的int对象大约28字节，所以总共约28KB

# NumPy数组：紧密排列
numpy_array = np.ones(1000, dtype=np.int8)
print(f"NumPy int8数组: {numpy_array.nbytes} 字节")

numpy_array = np.ones(1000, dtype=np.int64)
print(f"NumPy int64数组: {numpy_array.nbytes} 字节")
```

NumPy数组的内存占用是严格可计算的：nbytes = 元素数量 × 每个元素的字节数。Python列表的内存占用取决于解释器的实现和你具体存了什么。

## 一个经典的坑

很多人刚学NumPy时会犯一个错误：用列表的思维去操作数组。比如想在数组末尾添加元素：

```python
# Python列表append很快
lst = []
for i in range(10000):
    lst.append(i)

# NumPy数组不能这样用——每次"append"都会创建新数组！
arr = np.array([])
for i in range(10000):
    arr = np.append(arr, i)  # 非常慢！每次都复制整个数组
```

正确的做法是预先分配空间：

```python
arr = np.zeros(10000, dtype=np.int64)
for i in range(10000):
    arr[i] = i
```

或者用列表收集完再转数组：

```python
lst = []
for i in range(10000):
    lst.append(i)
arr = np.array(lst)
```

## 类型统一的问题

Python列表可以放任意类型：

```python
mixed_list = [1, "hello", 3.14, True, [1, 2]]
```

NumPy数组必须类型统一：

```python
# NumPy数组只能是同一种类型
arr = np.array([1, 2, 3])  # int64
arr = np.array([1.0, 2.0, 3.0])  # float64
arr = np.array([True, False, True])  # bool
```

这个限制看似严格，实际上解放了你——你不需要检查每个元素是什么类型，NumPy保证了数据的一致性。

## 什么时候用什么

讲了这么多，什么时候该用Python列表，什么时候该用NumPy？

简单的判断标准是：需要做数学计算吗？需要处理大量同类型数据吗？如果是，用NumPy。其他情况，列表可能更灵活。

举个例子：你想存储5个温度读数，然后计算平均值。用列表还是NumPy？

```python
# 用列表：简单场景
temps = [22.5, 23.1, 21.8, 24.2, 22.9]
avg = sum(temps) / len(temps)
print(f"平均温度: {avg:.1f}")
```

这个场景列表就够了。但如果你要处理100万个温度读数，NumPy的优势就显现出来了。

另一个例子：如果你的数据天然是异构的——比如一个用户对象有姓名、年龄、地址——用字典列表比NumPy数组合适。

## 面试会问什么

面试中关于列表和NumPy的常见问题包括：为什么NumPy比列表快？什么是向量化？为什么NumPy数组必须类型统一？视图和副本的区别是什么？

第一个问题的答案就是这一章讲的内容：连续内存、SIMD指令、避免Python对象开销。

第二个问题的答案是：向量化就是用数组操作代替显式循环，NumPy会自动利用底层优化。

第三个问题的答案是：NumPy的底层是C实现的连续内存数组，类型统一才能做到这一点。

## 尾声

理解列表和NumPy的区别，是理解NumPy一切特性的基础。接下来的章节里，我们会看到更多这种差异如何影响代码编写和性能。
