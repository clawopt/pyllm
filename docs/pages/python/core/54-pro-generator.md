---
title: 生成器
---

# 生成器

在讲生成器之前，必须先把迭代器协议讲清楚。因为生成器本质上就是迭代器的一个特例。

迭代器协议规定：一个对象如果实现了 `__iter__` 和 `__next__` 方法，就可以被 for 循环遍历。手动实现迭代器需要写一个完整的类，但生成器让这件事变得简单得多——生成器是一种特殊的迭代器，用 `yield` 关键字来"暂停"函数执行，而不是返回一个值就结束。

这一章解决的问题是：怎么处理大数据流，怎么实现惰性计算。

## 迭代器协议

当你在 Python 中写 `for x in obj` 时，背后发生的事情是：Python 调用 `iter(obj)` 获取迭代器，然后反复调用 `next(iterator)` 获取元素，直到抛出 `StopIteration` 异常。

```python
nums = [1, 2, 3]
it = iter(nums)  # 获取迭代器

print(next(it))  # 1
print(next(it))  # 2
print(next(it))  # 3
print(next(it))  # StopIteration
```

列表是有 `__iter__` 方法的，所以它是可迭代对象。但列表本身不是迭代器——列表的 `__iter__` 返回的是一个迭代器对象。

迭代器协议的好处是：它提供了一种统一的方式来遍历任何集合。无论是列表、文件、数据库查询结果，还是自定义对象，只要实现了迭代器协议，就可以用 for 循环遍历。

## 生成器函数

生成器函数使用 `yield` 关键字而不是 `return`。当调用生成器函数时，函数不会立即执行，而是返回一个生成器对象。每次调用 `next()` 时，函数执行到下一个 `yield`，然后暂停，把 `yield` 的值作为 `next()` 的返回值。

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

生成器函数和普通函数的区别在于：普通函数调用一次执行完然后返回，生成器函数调用一次创建一个生成器对象，每次 `next()` 执行一部分然后暂停。

这种"惰性计算"的模式在处理大数据时非常有用。假设要读取一个 10GB 的日志文件：

```python
def read_large_file(filename):
    with open(filename) as f:
        for line in f:
            yield line.strip()

for line in read_large_file("huge_log.txt"):
    process(line)  # 每次只读取一行，内存占用很小
```

无论文件多大，内存占用始终只有一行。这就是生成器的价值。

## 生成器表达式

除了生成器函数，还有一种创建生成器的方式：生成器表达式。语法和列表推导式类似，但用圆括号：

```python
squares = (x ** 2 for x in range(5))
print(squares)  # <generator object at ...>

print(list(squares))  # [0, 1, 4, 9, 16]
```

生成器表达式和列表推导式的区别：

```python
# 列表推导式：立即创建所有元素
squares_list = [x ** 2 for x in range(5)]

# 生成器表达式：只创建生成器对象
squares_gen = (x ** 2 for x in range(5))
```

列表推导式创建列表时，所有元素立即计算并存储在内存中。生成器表达式只在调用 `next()` 时才计算下一个值，是惰性求值。

生成器表达式经常和 sum、max、min 等函数配合使用：

```python
total = sum(x ** 2 for x in range(1000))
max_square = max((x ** 2 for x in range(10)))
```

注意在这些情况下不需要 list()，因为这些函数自己会迭代生成器。

## 生成器的方法

生成器除了可以用 `next()` 获取值，还有几个有用的方法。

`send()` 允许向生成器发送值：

```python
def coro():
    result = None
    while True:
        value = yield result
        result = value * 2

c = coro()
next(c)  # 启动生成器
print(c.send(5))   # 10
print(c.send(10))  # 20
```

`send()` 的工作方式是：它把值发送给生成器，这个值成为 `yield` 表达式的结果，然后生成器继续执行。第一个 `next()` 调用是为了启动生成器，这叫做"预激"（priming）。

`throw()` 向生成器抛出异常：

```python
def numbers():
    for i in range(5):
        try:
            yield i
        except ValueError:
            print("ValueError received")

gen = numbers()
print(next(gen))  # 0
print(next(gen))  # 1
gen.throw(ValueError)  # 在当前位置抛出异常
```

`close()` 关闭生成器：

```python
def numbers():
    for i in range(5):
        yield i

gen = numbers()
print(next(gen))  # 0
gen.close()       # 关闭生成器
print(next(gen))  # StopIteration
```

## yield from

`yield from` 用于委托生成器，可以简化嵌套生成器的写法：

```python
def inner():
    yield from [1, 2, 3]

def outer():
    yield from inner()

for x in outer():
    print(x)  # 1, 2, 3
```

`yield from` 的语义是：把控制权委托给另一个生成器，让它产生所有值。这在写递归生成器时特别有用，比如遍历树结构：

```python
def traverse(node):
    if node is None:
        return
    yield node.value
    yield from traverse(node.left)
    yield from traverse(node.right)
```

没有 `yield from` 的版本需要手动迭代：

```python
def traverse(node):
    if node is None:
        return
    yield node.value
    for child in traverse(node.left):
        yield child
    for child in traverse(node.right):
        yield child
```

`yield from` 让代码更简洁，也更符合"委托"的语义。

## 生成器与迭代器的关系

生成器是迭代器的简化写法。手动实现一个迭代器需要写一个完整的类：

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
```

用生成器函数可以实现同样的效果，但代码简单得多：

```python
def count_up_to(limit):
    current = 0
    while current < limit:
        yield current
        current += 1
```

生成器函数被调用时返回的是一个生成器对象，这个对象实现了 `__iter__` 和 `__next__` 方法。所以生成器就是迭代器，只是写法更简洁。

## 常见误区

第一个误区是忘记生成器只能用一次。生成器迭代完后就空了，不能重新开始。如果需要多次遍历，应该用列表或者每次创建新的生成器。

```python
gen = (x for x in range(3))
print(list(gen))  # [0, 1, 2]
print(list(gen))  # []，空了
```

第二个误区是在生成器中修改可变对象。由于生成器是惰性求值的，如果在生成器中修改了可变对象，可能会导致意外行为。

第三个误区是混淆 `yield` 和 `return`。`yield` 暂停函数并产出值，`return` 结束函数并返回值。如果在同一个函数中混用两者，需要明确理解执行流程。

第四个误区是认为生成器会自动计算所有值。生成器是惰性的，只有调用 `next()` 时才会计算下一个值。在迭代完成之前，生成器不会一次性计算所有值。

生成器是 Python 中处理大数据流和惰性计算的重要工具。理解生成器的惰性求值特性，是写出高效 Python 代码的关键。
