---
title: 迭代器
---

# 迭代器

迭代器是 Python 中一种**"记住遍历位置"**的对象。你可以把它想象成一个**"智能指针"**，每次你问它"下一个是什么"，它就给你下一个元素，然后自己往后挪一步。

在代码中，如下是使用迭代器的一个例子：

```python
nums = [1, 2, 3]

it = iter(nums)  # 调用 __iter__
print(next(it))  # 1，调用 __next__
print(next(it))  # 2
print(next(it))  # 3
print(next(it))  # StopIteration
```

迭代器是 Python 最核心的概念之一，却也是最容易被忽视的概念。很多人用 Python 好几年，每天都在用 for 循环遍历列表、遍历文件、遍历数据库结果，却从来没有真正思考过：为什么这些对象可以被 for 循环遍历？是什么让它们变得可遍历？

## 可迭代对象与迭代器的区别

在 Python 中，不是所有对象都可以用 for 循环遍历的。只有"可迭代对象"才可以。那么什么是可迭代对象？

简单来说，可迭代对象就是实现了 `__iter__` 方法的对象。这个方法返回一个迭代器。当你在代码中写 `for x in obj` 时，Python 会在后台调用 `iter(obj)`，这会触发 `__iter__` 方法获取迭代器，然后反复调用迭代器的 `__next__` 方法获取元素，直到抛出 StopIteration 异常。

```python
nums = [1, 2, 3]

it = iter(nums)  # 调用 __iter__
print(next(it))  # 1，调用 __next__
print(next(it))  # 2
print(next(it))  # 3
print(next(it))  # StopIteration
```

区分可迭代对象和迭代器很重要。列表是 可迭代对象，但列表本身不是迭代器。列表有 `__iter__` 方法（返回迭代器），但没有 `__next__` 方法。迭代器则同时拥有 `__iter__` 和 `__next__` 方法。

文件对象也是可迭代对象，可以直接用 for 遍历每一行：

```python
with open("file.txt") as f:
    for line in f:  # f 是可迭代对象
        print(line)
```

这里 f 是可迭代对象，for 循环在内部帮我们获取了迭代器并不断调用 next 获取每一行。

## iter 和 next 的工作原理

迭代器的两个核心方法是 `__iter__` 和 `__next__`。

`__iter__` 返回迭代器本身。在 Python 的迭代器协议中，迭代器的 `__iter__` 方法总是返回 self，这意味着迭代器本身也是可迭代对象。

`__next__` 返回下一个元素。如果没有更多元素，应该抛出 StopIteration 异常。这个异常不是错误，而是告诉 for 循环"已经遍历完了"的信号。

手动模拟一个简单的迭代器：

```python
class Counter:
    def __init__(self, limit):
        self.limit = limit
        self.current = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.current < self.limit:
            result = self.current
            self.current += 1
            return result
        raise StopIteration

counter = Counter(3)
for i in counter:
    print(i)
```

输出是 0、1、2。迭代器消耗完一次后就空了，不能重新开始，必须创建新的迭代器：

```python
counter = Counter(3)
print(list(counter))  # [0, 1, 2]
print(list(counter))  # []，第二次是空的
```

这和生成器的行为一致，实际上生成器就是迭代器的一种。

## 迭代器协议与内幕

Python 的迭代器协议是一套完整的机制。当你执行 `for x in iterable` 时，Python 内部发生的事情是：

```python
it = iter(iterable)  # 获取迭代器
while True:
    try:
        x = next(it)  # 获取下一个元素
    except StopIteration:
        break  # 没有更多元素，退出循环
    # 这里是循环体
```

这就是 for 循环能够工作的幕后机制。for 循环之所以能遍历任何可迭代对象，正是因为这套协议的存在。

理解这套协议对于调试和高级编程很重要。比如你知道 for 循环会在 StopIteration 时自动退出，就不会被这个异常吓到；你知道迭代器是通过 iter() 获取的，就可以手动控制迭代过程。

## 迭代器的惰性求值

迭代器最重要的特性是惰性求值。迭代器不会在创建时计算所有元素，而是在每次调用 next 时才计算下一个元素。这种"按需计算"的模式在处理大规模数据时非常有价值。

想象你需要处理一个包含一千万行的日志文件。如果用列表一次性读取，所有数据都会加载到内存，可能导致内存溢出。但如果用迭代器，每一轮循环只读取和处理一行，内存占用始终很低：

```python
with open("huge_log.txt") as f:
    for line in f:  # f 是迭代器，惰性读取
        process(line)
```

即使文件有 10GB，这种方式的内存占用也微乎其微。

惰性求值还有另一个好处：可以表示无限序列。比如一个生成斐波那契数列的迭代器可以永不停止：

```python
def fibonacci():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

for i, fib in enumerate(fibonacci()):
    print(fib)
    if i > 10:
        break
```

这个迭代器可以永远运行下去，因为它不会预先计算所有值。

## 迭代器与生成器的关系

生成器是一种特殊的迭代器。生成器函数使用 yield 关键字来产生值，而不是 return。当调用生成器函数时，返回的是一个生成器对象。

```python
def count_up_to(n):
    i = 0
    while i < n:
        yield i
        i += 1

counter = count_up_to(3)
print(next(counter))  # 0
print(next(counter))  # 1
print(next(counter))  # 2
print(next(counter))  # StopIteration
```

生成器函数被调用时，函数体不会立即执行。只有当调用 next() 时，函数才开始运行，直到遇到 yield 暂停并返回值。下一次调用 next() 时，函数从暂停处继续执行，直到再次遇到 yield 或函数结束。

生成器表达式是另一种创建迭代器的方式，语法和列表推导式类似，但用圆括号：

```python
gen = (x ** 2 for x in range(5))
print(list(gen))  # [0, 1, 4, 9, 16]
```

生成器表达式和列表推导式的区别在于：列表推导式立即生成所有元素（ eager 求值），生成器表达式只创建生成器对象，每次 next 时才计算下一个（惰性求值）。

## 内置迭代器工具

Python 提供了一些很有用的内置迭代器和函数，可以和迭代器配合使用。

`enumerate` 在迭代时提供索引：

```python
fruits = ["apple", "banana", "cherry"]
for i, fruit in enumerate(fruits):
    print(f"{i}: {fruit}")
```

`zip` 并行迭代多个序列：

```python
names = ["Alice", "Bob"]
scores = [85, 92]

for name, score in zip(names, scores):
    print(f"{name}: {score}")
```

`itertools` 模块提供了更多强大的迭代器工具：

```python
import itertools

# 无限迭代器
for i in itertools.count(5, 2):  # 从5开始，每次+2
    print(i)
    if i > 15:
        break

# 循环迭代
for x in itertools.cycle([1, 2]):
    print(x)  # 1, 2, 1, 2, 1, 2, ...

# 累积
for c in itertools.accumulate([1, 2, 3, 4]):
    print(c)  # 1, 3, 6, 10
```

`itertools` 是 Python 中处理迭代问题瑞士军刀，掌握它可以大大简化代码。

## 迭代器的常见陷阱

迭代器最常见的陷阱是：迭代器消耗完后是空的。

```python
nums = iter([1, 2, 3])
print(list(nums))  # [1, 2, 3]
print(list(nums))  # []，什么都没了
```

这是因为迭代器是有状态的，一次性遍历完后状态就变了。要重新遍历，必须重新获取迭代器。

另一个陷阱是在迭代过程中修改容器：

```python
nums = [1, 2, 3, 4, 5]

for n in nums:
    if n % 2 == 0:
        nums.remove(n)
```

在迭代过程中修改容器会导致意外行为或错误。正确的做法是遍历容器的副本，或者使用列表推导式创建新列表。

还有一个陷阱是迭代器的返回值。如果你写了一个迭代器但忘记在 StopIteration 前返回值，迭代器会默默结束，没有任何错误提示。调试这类问题时，可以在迭代器中加日志来观察执行流程。

## 自定义迭代器

通过实现 `__iter__` 和 `__next__` 方法，可以创建自定义迭代器。

一个典型的例子是反向迭代器：

```python
class ReverseList:
    def __init__(self, data):
        self.data = data
        self.index = len(data)

    def __iter__(self):
        return self

    def __next__(self):
        if self.index <= 0:
            raise StopIteration
        self.index -= 1
        return self.data[self.index]

nums = ReverseList([1, 2, 3])
for n in nums:
    print(n)
```

输出是 3、2、1。迭代器的状态由 `self.index` 维护，每次 next 调用都会更新状态。

## 迭代器的适用场景

迭代器最适合以下场景：处理大规模数据（内存效率高）、处理无限序列（不需要预先知道数据量）、惰性计算（延迟到真正需要时才计算）、管道处理（多个处理步骤串联）。

不适合用迭代器的场景：需要多次遍历（迭代器只能用一次）、需要随机访问（应该用列表或索引访问）、数据量很小（迭代器的额外开销可能不值得）。

理解迭代器的适用场景，才能在合适的场合选择迭代器，在不适用的场合选择列表或其他数据结构。
