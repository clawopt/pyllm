---
title: 魔术方法
---

# 魔术方法

Python 的魔术方法（也叫 dunder methods，因为方法名以双下划线开头和结尾，比如 `__init__`）让你能够自定义类的行为，让对象能够使用 Python 的内置语法和操作符。

比如，你可以让自定义类的对象能够用 `+` 相加、用 `[]` 索引访问、用 `len()` 获取长度。这让自定义类型可以像内置类型一样使用。

理解魔术方法，是理解 Python 面向对象高级特性的基础。很多 Python 的核心特性，比如上下文管理器、迭代器、描述器，都依赖于魔术方法。

## 什么是魔术方法

魔术方法是 Python 类的特殊方法，它们以双下划线开头和结尾（如 `__init__`、`__str__`）。当某个操作在对象上执行时，Python 会自动调用对应的魔术方法。

```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

person = Person("Alice", 30)
print(person)  # 自动调用 __str__
```

当你 `print(person)` 时，Python 实际执行的是 `print(person.__str__())`。这就是魔术方法的工作方式。

## __str__ 与 __repr__

`__str__` 和 `__repr__` 控制对象的字符串表示。

`__repr__` 是"官方"字符串表示，应该是 unambiguous 的，最好能通过它重建对象：

```python
class Dog:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"Dog(name='{self.name}')"
```

```python
dog = Dog("旺财")
print(repr(dog))  # Dog(name='旺财')
print(dog)         # Dog(name='旺财')，如果没有 __str__，会 fallback 到 __repr__
```

`__str__` 是"非正式"的字符串表示，应该 human-readable：

```python
class Dog:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return f"一只叫{self.name}的狗"

    def __repr__(self):
        return f"Dog(name='{self.name}')"
```

```python
dog = Dog("旺财")
print(str(dog))   # 一只叫旺财的狗
print(repr(dog))  # Dog(name='旺财')
```

## __eq__ 与 __hash__

`__eq__` 定义对象相等的比较，默认用身份（内存地址）比较：

```python
class Dog:
    def __init__(self, name):
        self.name = name

    def __eq__(self, other):
        if isinstance(other, Dog):
            return self.name == other.name
        return False
```

```python
dog1 = Dog("旺财")
dog2 = Dog("旺财")
print(dog1 == dog2)  # True，同名被认为是相等的
print(dog1 is dog2)   # False，不同的对象
```

如果定义了 `__eq__` 而不定义 `__hash__`，对象会变成不可哈希的（因为 Python 假定相等的对象应该有相同的哈希值）：

```python
dog = Dog("旺财")
print(hash(dog))  # 会报错！TypeError: unhashable type
```

如果想让对象可哈希，同时 `__eq__` 定义自定义相等逻辑，需要同时定义 `__hash__`：

```python
class Dog:
    def __init__(self, name):
        self.name = name

    def __eq__(self, other):
        if isinstance(other, Dog):
            return self.name == other.name
        return False

    def __hash__(self):
        return hash(self.name)  # 用 name 来计算哈希
```

## __lt__ 与比较方法

如果对象要支持比较操作（`<`、`>`、`<=`、`>=`），需要定义对应的魔术方法：

```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __lt__(self, other):
        return self.age < other.age

    def __le__(self, other):
        return self.age <= other.age

    def __gt__(self, other):
        return self.age > other.age

    def __ge__(self, other):
        return self.age >= other.age
```

```python
p1 = Person("Alice", 30)
p2 = Person("Bob", 25)

print(p1 < p2)   # False，30 > 25
print(p1 > p2)   # True
```

Python 3.7+ 支持 `functools.total_ordering` 装饰器，只需要定义 `__lt__` 和 `__eq__`，其他比较方法会自动生成。

## __call__ 可调用对象

`__call__` 让对象可以像函数一样被调用：

```python
class Counter:
    def __init__(self):
        self.count = 0

    def __call__(self):
        self.count += 1
        return self.count

counter = Counter()
print(counter())  # 1
print(counter())  # 2
print(counter())  # 3
```

当执行 `counter()` 时，Python 实际上调用的是 `counter.__call__()`。

`__call__` 的用途包括：创建可调用对象、实现装饰器、创建带状态的函数等。

## __enter__ 与 __exit__ 上下文管理器

上下文管理器让你可以用 `with` 语句来管理资源，确保资源在使用后被正确清理：

```python
class FileManager:
    def __init__(self, filename):
        self.filename = filename

    def __enter__(self):
        self.file = open(self.filename, 'w')
        return self.file

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()
        return False  # 不压制异常
```

```python
with FileManager('test.txt') as f:
    f.write('Hello!')
# 文件在这里自动关闭
```

`__exit__` 的三个参数是异常信息，如果 with 语句块内没有异常，都是 None。

Python 3.7+ 推荐使用 `@contextmanager` 装饰器来创建上下文管理器，而不是直接实现 `__enter__` 和 `__exit__`：

```python
from contextlib import contextmanager

@contextmanager
def file_manager(filename):
    f = open(filename, 'w')
    try:
        yield f
    finally:
        f.close()
```

## __len__ 与 __getitem__

`__len__` 让对象支持 `len()` 函数：

```python
class MyList:
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)
```

`__getitem__` 让对象支持索引访问：

```python
    def __getitem__(self, index):
        return self.items[index]
```

```python
my_list = MyList([1, 2, 3, 4, 5])
print(len(my_list))   # 5
print(my_list[0])     # 1
print(my_list[2:4])   # [3, 4]
```

## __iter__ 与 __next__

`__iter__` 和 `__next__` 让对象变成迭代器：

```python
class Countdown:
    def __init__(self, start):
        self.current = start

    def __iter__(self):
        return self

    def __next__(self):
        if self.current <= 0:
            raise StopIteration
        result = self.current
        self.current -= 1
        return result
```

```python
for i in Countdown(5):
    print(i)
# 输出：5, 4, 3, 2, 1
```

## 常见误区

第一个误区是定义了 `__eq__` 但忘记定义 `__hash__`。如果你需要把对象放进 set 或作为 dict 的 key，必须确保对象是可哈希的。定义了自定义 `__eq__` 后，对象默认变成不可哈希的，必须显式定义 `__hash__`。

第二个误区是混淆 `__str__` 和 `__repr__`。`__repr__` 是给开发者看的，应该是精确的；`__str__` 是给用户看的，应该是友好的。如果只定义了其中一个，缺失的会 fallback 到另一个。

第三个误区是在 `__exit__` 中忽略异常处理。`__exit__` 应该正确处理异常，通常返回 False 表示不压制异常，让异常继续传播。

第四个误区是让 `__iter__` 返回新对象而不是 self。如果想让对象本身是迭代器（可以多次迭代），让 `__iter__` 返回 self；如果每次迭代需要独立状态，让 `__iter__` 返回新对象。

理解魔术方法是掌握 Python 面向对象高级特性的关键。它们让自定义类型能够和 Python 的内置语法和操作符无缝集成。
