---
title: 命名元组
---

# 命名元组

当你需要表示一个简单的数据结构时，第一反应可能是写一个类：

```python
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

p = Point(3, 5)
print(p.x, p.y)  # 3 5
```

这个写法很直观，但如果你需要创建大量这样的小对象，就会发现每个 Point 实例都有一个 `__dict__`，占用额外的内存空间。而且定义一个类需要写好几行代码，显得有些啰嗦。

namedtuple 正是为了解决这些问题而设计的。它用一种更轻量的方式创建"带名字的元组"，既保留了元组的不可变特性和内存效率，又提供了通过名字访问字段的能力。

## namedtuple 的基本用法

namedtuple 是 tuple 的子类，它通过字段名来访问元素：

```python
from collections import namedtuple

Point = namedtuple("Point", ["x", "y"])
p = Point(3, 5)

print(p.x)      # 3，像访问属性一样
print(p[0])     # 3，像元组一样用索引访问
print(p.x == p[0])  # True，两者等价
```

namedtuple 的第一个参数是类型名，第二个参数是字段名列表（或空格分隔的字符串）。返回的是一个类，可以像普通类一样实例化。

## 它的底层实现原理

namedtuple 的实现非常巧妙。它动态创建一个类，这个类继承自 tuple，同时通过 `__slots__` 禁用实例字典，又通过 `__getattr__` 把字段名映射到对应的索引上。

你可以大致把它理解为这样的实现：

```python
class Point(tuple):
    __slots__ = ()  # 不创建实例字典

    @property
    def x(self):
        return self[0]

    @property
    def y(self):
        return self[1]
```

关键在于：namedtuple 创建的对象在内存布局上完全等同于 tuple，没有额外的实例字典。所以它既有元组的内存效率，又有普通类的可读性。

```python
from collections import namedtuple

Point = namedtuple("Point", ["x", "y"])
p = Point(3, 5)

# tuple 的特性它都有
print(isinstance(p, tuple))  # True
print(len(p))                # 2
print(p[0], p[1])           # 3 5

# 同时可以通过名字访问
print(p.x, p.y)             # 3 5
```

## 为什么 namedtuple 比普通类更节省内存

普通类创建的实例都有一个 `__dict__`，用来存储实例属性。这个字典本身就需要占用内存，而且每次属性访问都需要通过字典查找。

namedtuple 使用 `__slots__ = ()` 来禁用实例字典，所有的属性访问都变成索引访问，直接对应 tuple 的内存布局。这就是为什么 namedtuple 比普通类更节省内存。

```python
import sys
from collections import namedtuple

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

PointNT = namedtuple("Point", ["x", "y"])

p_class = Point(1, 2)
p_namedtuple = PointNT(1, 2)

print(sys.getsizeof(p_class))        # 普通类实例更大
print(sys.getsizeof(p_namedtuple))   # namedtuple 实例更小
```

当需要创建数百万个简单数据对象时，这个内存差异会变得非常显著。

## 不可变语义

namedtuple 继承自 tuple，所以具有不可变性——创建后不能修改字段：

```python
from collections import namedtuple

Point = namedtuple("Point", ["x", "y"])
p = Point(3, 5)

p.x = 10  # AttributeError: can't set attribute
```

这个限制既是约束也是保护。在很多场景下，数据一旦创建就不应该再改变——比如作为字典的 key，或者在并发环境中共享数据。不可变性可以防止意外修改导致的 bug。

如果需要修改字段，可以用 `_replace` 方法创建一个新的实例：

```python
from collections import namedtuple

Point = namedtuple("Point", ["x", "y"])
p = Point(3, 5)

p2 = p._replace(x=10)
print(p)   # Point(x=3, y=5)，原对象不变
print(p2)  # Point(x=10, y=5)，新对象
```

`_replace` 的实现方式是：创建一个新的 tuple，然后逐个复制原字段并替换需要修改的部分。

## 字段相关的方法

namedtuple 提供了几个有用的方法。

`_fields` 返回字段名元组：

```python
from collections import namedtuple

Point = namedtuple("Point", ["x", "y"])
print(Point._fields)  # ('x', 'y')
```

`_asdict` 返回字段和值的字典：

```python
p = Point(3, 5)
print(p._asdict())  # {'x': 3, 'y': 5}
```

这在需要序列化或转换成其他数据结构时很有用。

## 默认值与工厂方法

从 Python 3.7 开始，namedtuple 支持默认值：

```python
from collections import namedtuple

Point = namedtuple("Point", ["x", "y"], defaults=[0, 0])
p1 = Point(3, 5)
p2 = Point(3)  # y 使用默认值 0

print(p1)  # Point(x=3, y=5)
print(p2)  # Point(x=3, y=0)
```

defaults 从右向左应用，所以这里 x 必须提供，y 默认是 0。

## namedtuple 与字典的互相转换

namedtuple 和 dict 之间可以方便地互相转换：

```python
from collections import namedtuple

Point = namedtuple("Point", ["x", "y"])
p = Point(3, 5)

# namedtuple -> dict
d = p._asdict()
print(d)  # {'x': 3, 'y': 5}

# dict -> namedtuple
d = {"x": 10, "y": 20}
p2 = Point(**d)
print(p2)  # Point(x=10, y=20)
```

## namedtuple 和 dataclass 的选择

Python 3.7 引入了 dataclass，也经常被拿来和 namedtuple 对比。

dataclass 默认是可变的，而且每个实例都有 `__dict__`：

```python
from dataclasses import dataclass

@dataclass
class Point:
    x: int
    y: int

p = Point(3, 5)
p.x = 10  # 允许修改
print(sys.getsizeof(p))  # 更大
```

两者的主要区别：

| 特性 | namedtuple | dataclass |
|------|-----------|-----------|
| 可变性 | 不可变 | 默认可变 |
| 内存占用 | 小 | 大 |
| 实例字典 | 无 | 有 |
| 适用场景 | 轻量数据、结构 | 复杂对象 |

选择建议：如果你需要不可变的简单数据结构，用 namedtuple；如果你需要可变字段或更多功能（方法、类型注解、默认值），用 dataclass。

## 常见工程场景

### 数据库查询结果

namedtuple 非常适合表示数据库查询结果：

```python
from collections import namedtuple

Row = namedtuple("Row", ["id", "name", "email"])

for row in db.execute("SELECT id, name, email FROM users"):
    print(row.id, row.name, row.email)
```

相比普通 tuple，用名字访问更清晰；相比普通类，内存效率更高。

### CSV 数据处理

处理 CSV 数据时，namedtuple 可以提供列名访问：

```python
from collections import namedtuple

with open("data.csv") as f:
    headers = f.readline().strip().split(",")
    Row = namedtuple("Row", headers)

    for line in f:
        row = Row(*line.strip().split(","))
        print(row.name, row.email)
```

### 配置参数结构

用 namedtuple 定义配置结构比字典更清晰，而且不可变性可以防止配置被意外修改：

```python
from collections import namedtuple

Config = namedtuple("Config", ["host", "port", "debug"])
config = Config("localhost", 8080, True)
```

## 常见误区

第一个误区是认为 namedtuple 和普通类完全一样。它们底层都是 tuple，所以有 `__slots__` 的内存优势，也有不可变性的限制。

第二个误区是不理解 _replace 的用法。`_replace` 返回新实例，不修改原实例。如果你想"修改"namedtuple 的字段，需要用 `_replace` 的返回值创建新对象。

第三个误区是在循环中重复定义 namedtuple。`namedtuple()` 调用会在每次执行时创建一个新类，这在循环中会累积类定义，应该在循环外定义一次然后复用。

```python
# 错误
for _ in range(10):
    Point = namedtuple("Point", ["x", "y"])  # 每次都创建新类

# 正确
Point = namedtuple("Point", ["x", "y"])  # 定义一次
for _ in range(10):
    p = Point(1, 2)
```

第四个误区是混淆 namedtuple 的适用场景。namedtuple 适合简单数据结构，但如果需要复杂逻辑、方法、继承等，应该用普通类或 dataclass。

## 面试核心回答

面试中关于 namedtuple 常问的问题包括：

"namedtuple 和 tuple 有什么区别？"——答案是：namedtuple 是 tuple 的子类，添加了通过字段名访问的能力，但内存布局和 tuple 完全一样。

"namedtuple 为什么比普通类更节省内存？"——答案是：因为它使用 `__slots__` 禁用实例字典，所有属性访问都变成索引访问，内存布局和 tuple 完全一样。

"namedtuple 和 dataclass 的区别是什么？"——答案是：namedtuple 不可变、内存占用小，适合简单数据结构；dataclass 可变、有 `__dict__`，适合复杂对象。

"namedtuple 如何实现字段名访问？"——答案是：通过继承 tuple 并重写 `__getattr__`，将字段名映射到对应的索引位置。

理解这些问题的答案，说明你对 namedtuple 的理解已经足够深入了。
