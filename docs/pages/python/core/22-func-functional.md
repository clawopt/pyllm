---
title: 函数式编程工具
---

# 函数式编程工具

前面两章我们学习了怎么定义函数、怎么传递参数。但函数作为一种对象，不只是被调用，还可以被传递、被操作。就像数字可以进行加减乘除运算，函数也可以被组合、被转换。这就是函数式编程的核心思想。

函数式编程是一种编程范式，它把计算看作是数学函数的求值，强调函数之间的组合和变换，而不是改变状态和 mutable 数据。Python 提供了几个内置工具来支持这种编程风格：lambda 表达式、map、filter、reduce，以及配合这些工具使用的 key 函数。

这一章解决的问题是：怎么把函数当作数据处理工具来用。这个思维在处理数据时非常有用，尤其是在配合推导式使用时。

## lambda 表达式

lambda 表达式是创建匿名函数的语法。在 Python 中，它的形式是 `lambda 参数列表: 表达式`：

```python
square = lambda x: x ** 2
print(square(5))  # 25
```

lambda 和普通 def 的本质区别在于：lambda 是表达式，而 def 是语句。这意味着 lambda 可以写在任何允许表达式的地方，比如作为另一个函数的参数：

```python
numbers = [3, 1, 4, 1, 5, 9, 2, 6]
numbers.sort(key=lambda x: -x)  # 按降序排序
print(numbers)
```

lambda 的限制是：只能写单个表达式，不能写语句（比如 if、for、while）。这是有意为之的设计——lambda 的目的不是替代 def，而是提供一个便捷的语法来创建简单的函数对象。

什么时候用 lambda？当需要一个简单函数作为参数时，lambda 比完整的 def 更简洁。比如 `sorted(words, key=lambda w: -len(w))` 比单独定义一个函数然后传给 sorted 更直观。

什么时候不该用 lambda？当逻辑复杂到需要多行代码时，应该用 def。lambda 应该保持简单和一目了然。

## map 函数

map 把一个函数应用到一个可迭代对象的每个元素上，返回结果组成的迭代器：

```python
numbers = [1, 2, 3, 4, 5]

# 用 lambda 平方每个数
squared = list(map(lambda x: x ** 2, numbers))
print(squared)  # [1, 4, 9, 16, 25]

# 用普通函数也可以
def square(x):
    return x ** 2

squared = list(map(square, numbers))
```

map 的第一个参数是函数，第二个参数是可迭代对象。map 会把这个函数应用到可迭代对象的每个元素上，返回所有结果。

在现代 Python 中，map 经常被列表推导式替代：

```python
# map 版本
squared = list(map(lambda x: x ** 2, numbers))

# 推导式版本
squared = [x ** 2 for x in numbers]
```

两者效果相似，但推导式更符合 Python 的风格。不过 map 的优势是可以处理无限序列或非常大的序列，因为它是惰性求值的。

## filter 函数

filter 根据条件过滤可迭代对象的元素，返回满足条件的元素组成的迭代器：

```python
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# 过滤偶数
evens = list(filter(lambda x: x % 2 == 0, numbers))
print(evens)  # [2, 4, 6, 8, 10]

# 过滤奇数
odds = list(filter(lambda x: x % 2 != 0, numbers))
print(odds)  # [1, 3, 5, 7, 9]
```

filter 的第一个参数是返回布尔值的函数（谓词），第二个参数是可迭代对象。filter 会保留所有让谓词返回 True 的元素。

同样，filter 也可以用推导式替代：

```python
# filter 版本
evens = list(filter(lambda x: x % 2 == 0, numbers))

# 推导式版本
evens = [x for x in numbers if x % 2 == 0]
```

## reduce 函数

reduce（来自 functools 模块）对可迭代对象的元素进行累积计算，把整个序列聚合成一个值：

```python
from functools import reduce

numbers = [1, 2, 3, 4, 5]

# 求和
total = reduce(lambda x, y: x + y, numbers)
print(total)  # 15

# 求积
product = reduce(lambda x, y: x * y, numbers)
print(product)  # 120
```

reduce 的工作方式是：先用函数处理前两个元素得到结果，然后用结果和第三个元素继续调用函数，直到处理完所有元素。

reduce 还可以接受一个初始值：

```python
from functools import reduce

numbers = [1, 2, 3, 4, 5]

# 有初始值 10
total = reduce(lambda x, y: x + y, numbers, 10)
print(total)  # 25，10 + 1 + 2 + 3 + 4 + 5
```

reduce 的使用场景包括：求和、求积、找最大值、字符串连接、构建复杂对象等。但在现代 Python 中，很多 reduce 的场景可以用内置函数替代——比如求和直接用 `sum()`，求积直接用 `math.prod()`。

## sorted 函数的 key 参数

sorted 函数可以对可迭代对象排序，返回新的排序后的列表：

```python
words = ["banana", "apple", "cherry", "date"]

# 按字母排序
print(sorted(words))  # ['apple', 'banana', 'cherry', 'date']

# 按长度排序
print(sorted(words, key=len))  # ['date', 'apple', 'banana', 'cherry']
```

key 参数接受一个函数，这个函数会应用到每个元素上，然后用函数返回值来决定排序顺序。

常见的 key 函数用法：

```python
students = [("Alice", 85), ("Bob", 92), ("Charlie", 85)]

# 按分数排序
print(sorted(students, key=lambda s: s[1]))  # [('Alice', 85), ('Charlie', 85), ('Bob', 92)]

# 按分数升序，分数相同按名字降序
print(sorted(students, key=lambda s: (s[1], -ord(s[0][0]))))
```

字符串排序时，默认是区分大小写的：

```python
words = ["Apple", "banana", "Cherry"]

# 默认按 ASCII 顺序，大写字母排在前
print(sorted(words))  # ['Apple', 'Cherry', 'banana']

# 忽略大小写排序
print(sorted(words, key=str.lower))  # ['Apple', 'banana', 'Cherry']
```

min 和 max 函数也支持 key 参数，用法相同。

## 组合使用

这些工具可以组合使用，形成数据处理管道：

```python
from functools import reduce

numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# 找出平方后能被 3 整除的数
result = list(filter(
    lambda x: x % 3 == 0,
    map(lambda x: x ** 2, numbers)
))
print(result)  # [9, 36, 81]
```

但这种写法可读性较差，更 Pythonic 的方式是使用推导式：

```python
result = [x ** 2 for x in numbers if x ** 2 % 3 == 0]
```

或者分成多步：

```python
squared = (x ** 2 for x in numbers)  # 生成器表达式，惰性
result = [x for x in squared if x % 3 == 0]
```

函数式工具和推导式各有适用场景：简单的映射和过滤用推导式更清晰，复杂的累积计算用 reduce 更合适。

## 常见误区

第一个误区是过度使用 lambda。lambda 的本意是创建一个简单的匿名函数，不是替代 def。如果逻辑复杂，应该用 def。

第二个误区是忘记 map/filter/reduce 返回的是迭代器，不是列表。需要用 list() 转换成列表。

第三个误区是在 reduce 中混淆累加和累积的含义。reduce 的函数参数顺序是 accumulator, element，如果用错了顺序，结果会完全错误。

第四个误区是在 sorted 的 key 中使用昂贵的计算。如果需要根据复杂条件排序，key 函数会被调用很多次，应该尽量保持简单。

理解这些函数式编程工具，以及它们和推导式的关系，能让你的数据处理代码更加简洁和灵活。
