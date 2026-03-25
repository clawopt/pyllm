---
title: 字典
---

# 字典

字典（dict）是 Python 中用来存储"键值对"的数据结构。你可以把它想象成一本真实的字典：你通过查找"单词"（键），找到对应的"解释"（值）。

在 Python 中，字典用花括号 `{}` 表示，键和值之间用冒号 `:` 分隔：

```python
d = {"name": "Tom", "age": 18}
```

举个生活中的例子：想象一个快递柜系统。每个快递柜格子都有一个编号，快递员通过输入取件码（键），系统就能立即定位到对应的格子（值），不需要逐个打开柜子查找。字典的工作原理与此类似——通过哈希函数将键转换成"编号"，直接定位到存储位置，实现快速存取。

如果说 list 是 Python 最常用的容器，那么 dict 就是 Python 最重要的容器。

这听起来有点夸张，但只要看看 Python 内部是如何工作的，你就会明白。模块的命名空间是 dict，对象的属性是 dict，函数的局部变量在底层也是 dict，甚至 Python 内部用来存储字节码的常量池也是 dict。dict 是 Python 运行时的基石。


## 哈希表的基本原理

字典的底层实现是哈希表（Hash Table）。要理解 dict，首先要理解哈希表的工作方式。

哈希表的核心思想是：通过一个哈希函数，把 key 转换成一个整数（哈希值），然后用这个整数去定位一个数组的槽位。理想情况下，每个 key 都能通过哈希函数找到一个唯一的位置，实现 O(1) 的查找速度。

用一个生活中的例子来理解：想象一个图书馆的书架，每本书都有一个唯一的编号。如果我知道编号，就能直接找到对应的书架位置，不需要一本一本地去找。哈希表就是这个原理——哈希函数就像是"编号生成器"，它把 key 转换成一个"编号"，然后直接去访问对应的"书架位置"。

```python
d = {"name": "Tom", "age": 18}

# 查找 "name" 时：
# 1. 计算 hash("name") -> 某个整数
# 2. 用这个整数定位到数组的某个位置
# 3. 读取该位置存储的 value
```

## 为什么 key 必须可哈希

在 dict 中，每个 key 都必须是可哈希的。可哈希的意思是：这个对象在创建后，其哈希值在生命周期内保持不变。

这个要求是哈希表工作原理的直接结果。当 dict 存储一个 key 时，它依赖 `hash(key)` 来计算存储位置。如果 key 的哈希值改变了，dict 用新的哈希值去查找，却发现原来的位置上什么都没有，就会出现"丢失"的问题。

不可哈希的对象（如 list、dict、set）不能作为 key，因为它们是可变的：

```python
d = {}

# list 不可哈希
d[[1, 2]] = "value"  # TypeError: unhashable type 'list'

# tuple 可哈希（如果内部元素都是可哈希的）
d[(1, 2)] = "value"  # 正常
```

这里有一个常见的陷阱：tuple 本身是不可变的，但如果 tuple 内部包含可变对象，那么这个 tuple 仍然不可哈希：

```python
d = {}

# 包含 list 的 tuple 不可哈希
d[([1, 2], 3)] = "value"  # TypeError: unhashable type 'list'
```

为什么 list、dict、set 不可哈希？因为它们可以被修改。如果允许它们作为 key，修改它们会导致哈希值改变，从而破坏 dict 的内部结构。

而 int、str、tuple（只包含可哈希元素）是可哈希的，因为它们不可变。

## 哈希冲突与开放寻址

哈希函数把无限的 key 空间映射到有限的数组位置，冲突是不可避免的。不同的 key 可能计算出相同的哈希值，这在哈希表中叫做"冲突"（collision）。

Python 使用"开放寻址法"（Open Addressing）来解决冲突。当一个槽位被占用时，dict 会按照某种探测序列（比如线性探测或扰动探测）去寻找下一个空位。

探测的过程大致是：先用 `hash(key)` 计算第一个位置，如果被占用，就探测下一个位置，再被占用就继续探测，直到找到空位为止。查找的时候也按照相同的探测序列进行。

这意味着在最坏情况下（大量冲突），查找可能退化到 O(n)。但正常情况下，哈希表的查找仍然是 O(1)。

Python 的哈希表实现还引入了"扰动"机制——它不只是简单地线性探测，而是混合使用多个探测步长，这能更好地分散冲突。这个设计非常精妙，使得 Python 的 dict 在处理各种 key 模式时都能保持良好的性能。

## dict 为什么是有序的

这是一个经典面试题：dict 是否有序？

答案是：Python 3.7 之前，dict 的有序性只是实现细节，不是语言承诺。Python 3.7 之后，有序性成为了语言规范。

在早期版本中，dict 的实现是：哈希表数组既存储 key-value 对，也存储部分索引信息。插入顺序大致能保持，但理论上可能因为哈希冲突而打乱。Python 3.6 的实现做了重大改进，变成"紧凑数组 + 索引表"分离的结构，才真正保证了严格的插入顺序。

Python 3.7 的 dict 底层结构大致可以理解为两个数组：

```
索引表: [hash1, hash2, None, hash3, ...]  # 存哈希值的某种编码
数据表: [(key1, value1), (key2, value2), ...]  # 按插入顺序排列
```

查找时用哈希定位索引，再通过索引找到实际数据。这样既保证了 O(1) 的查找性能，又保证了插入顺序。

面试时如果被问到 dict 的有序性，应该准确回答：3.7 之前是实现细节，3.7 起是语言保证。

## 常用接口详解

dict 有四种基本操作：访问、插入、更新、删除。

访问有两种方式：`d[key]` 和 `d.get(key)`。

`d[key]` 直接访问，如果 key 不存在会抛出 KeyError。`d.get(key)` 访问，如果 key 不存在会返回 None（或者指定的默认值），不会抛出异常。

```python
d = {"name": "Tom", "age": 18}

print(d["name"])      # "Tom"
print(d["city"])      # KeyError

print(d.get("name"))  # "Tom"
print(d.get("city"))  # None
print(d.get("city", "Unknown"))  # "Unknown"
```

工程中推荐使用 `get`，除非你确定 key 一定存在，或者 key 不存在是一个真正的异常情况。

插入和更新都是 `d[key] = value`，如果 key 已存在就更新，不存在就插入：

```python
d = {"name": "Tom"}

d["age"] = 18      # 插入
d["name"] = "Jerry"  # 更新

print(d)  # {"name": "Jerry", "age": 18}
```

删除可以用 `del d[key]`、`d.pop(key)`、`d.popitem()`。

`pop` 删除指定 key 并返回值，`popitem` 删除并返回最后一个插入的 key-value 对（3.7+ 保证）：

```python
d = {"a": 1, "b": 2}

print(d.pop("a"))  # 1，删除并返回值
print(d)           # {"b": 2}

d["c"] = 3
print(d.popitem())  # ("c", 3)
```

## 视图对象与迭代器

dict.keys()、dict.values()、dict.items() 返回的不是列表，而是"视图对象"（View Objects）。

视图对象是 dict 的动态视图——它们反映了 dict 的当前状态，但不占用额外内存：

```python
d = {"a": 1, "b": 2}

keys = d.keys()
print(type(keys))  # <class 'dict_keys'>

d["c"] = 3
print(list(keys))  # ['a', 'b', 'c']，自动反映新添加的 key
```

如果在遍历视图的同时修改 dict，会导致 RuntimeError。正确的做法是先转换成列表：

```python
d = {"a": 1, "b": 2}

for key in list(d.keys()):  # 先复制
    if key == "a":
        del d[key]
```

dict.items() 返回的视图尤其有用，它可以直接在 for 循环中解包：

```python
d = {"name": "Tom", "age": 18}

for key, value in d.items():
    print(f"{key}: {value}")
```

## defaultdict 的工作方式

defaultdict 是 dict 的一个子类，它解决了"key 不存在时自动创建默认值"的问题。

普通的 dict 访问不存在的 key 会报错，但 defaultdict 会在访问时自动创建一个默认值。这个默认值由 defaultdict 的第一个参数（default_factory）决定：

```python
from collections import defaultdict

d = defaultdict(list)

d["fruits"].append("apple")
print(d["fruits"])  # ["apple"]
print(d["vegetables"])  # []，自动创建空列表
```

它的实现原理是在 `__getitem__` 中拦截 KeyError，然后调用 default_factory 创建默认值。这个设计很巧妙——它不会影响 `__contains__` 等其他方法的行为。

defaultdict 常见的 default_factory 包括：list（用于分组）、int（用于计数）、set（用于去重分组）。

```python
from collections import defaultdict

# 计数
counter = defaultdict(int)
for char in "hello":
    counter[char] += 1
print(counter)  # {'h': 1, 'e': 1, 'l': 2, 'o': 1}

# 分组
groups = defaultdict(list)
for name, score in [("Tom", 85), ("Jerry", 90), ("Tom", 92)]:
    groups[name].append(score)
print(groups)  # {'Tom': [85, 92], 'Jerry': [90]}
```

## 字典推导式

字典推导式和列表推导式类似，但生成的是字典：

```python
squares = {x: x ** 2 for x in range(5)}
print(squares)  # {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}
```

字典推导式常用于：反转字典、批量转换数据、过滤数据。

```python
# 反转字典
original = {"a": 1, "b": 2, "c": 3}
reversed_d = {v: k for k, v in original.items()}
print(reversed_d)  # {1: 'a', 2: 'b', 3: 'c'}

# 注意：如果有重复的 value，反转会丢失数据
```

## 性能分析

dict 的各项操作时间复杂度如下：

| 操作 | 平均复杂度 | 最坏复杂度 |
|------|-----------|-----------|
| 查找 | O(1) | O(n) |
| 插入 | O(1) | O(n) |
| 删除 | O(1) | O(n) |
| 遍历 | O(n) | O(n) |

正常情况下是 O(1)，但极端情况下（大量哈希冲突）会退化为 O(n)。

dict 的内存占用大约是同等大小 list 的两倍，因为 dict 需要同时存储 key、value 和哈希信息。但考虑到它带来的查找效率提升，这个空间换时间是值得的。

```python
import sys

d = {f"key_{i}": i for i in range(1000)}
print(sys.getsizeof(d))  # 几万字节

l = [(f"key_{i}", i) for i in range(1000)]
print(sys.getsizeof(l))  # 小一些，但查找是 O(n)
```

## 常见的工程误区

第一个误区是用可变对象作为 key。虽然 tuple 本身不可变，但如果 tuple 里包含 list，那这个 tuple 仍然不可哈希。确保所有 key 都是真正不可变的。

第二个误区是在遍历 dict 时修改它。这会导致 RuntimeError，应该先转换成列表或使用字典的副本：

```python
d = {"a": 1, "b": 2, "c": 3}

# 错误
for key in d:
    if d[key] > 1:
        del d[key]  # RuntimeError

# 正确
for key in list(d.keys()):
    if d[key] > 1:
        del d[key]
```

第三个误区是误以为 dict 是线程安全的。dict 的单次操作是原子的（比如 `d[key] = value`），但复合操作不是。如果需要线程安全，应该使用 `queue.Queue` 或其他线程安全的数据结构，或者加锁保护。

第四个误区是在哈希冲突严重时性能退化。虽然 Python 的哈希表设计得很好，但在极端情况下（比如有人故意构造大量冲突的 key）仍然可能出现问题。

## dict 与其他语言对应结构的对比

如果你有其他语言的背景，理解 dict 在 Python 中的独特之处很重要。

在 Java 中，对应的结构是 HashMap。两者都是哈希表实现，但 Java 需要处理泛型，Python 的 dict 则更简洁。

在 JavaScript 中，对应的结构是对象 `{}` 或 Map。ES6 引入的 Map 更接近 Python 的 dict，支持任意类型的 key。

在 Go 中，对应的是 map，也是哈希表实现。

但 Python 的 dict 有一个独特之处：它是 Python 运行时的核心，几乎所有命名空间都用 dict 表示。这意味着 dict 的实现经过了极致的优化，是 CPython 中最复杂的 C 数据结构之一。

## 理解 dict 就是理解 Python 运行时

当你真正理解 dict，你就理解了 Python 的很多核心特性为什么那样设计。

为什么函数可以有任意数量的参数？因为 `*args` 和 `**kwargs` 底层就是用 tuple 和 dict 存储的。

为什么对象可以动态添加属性？因为对象的属性在底层就是用 dict 存储的，叫做 `__dict__`。

为什么模块可以动态导入？因为模块的命名空间就是 dict。

为什么 JSON 可以表示 Python 对象？因为 dict 是 JSON 的 object（`{}`）在 Python 中的等价物。

dict 不只是 Python 的一个容器，它是 Python 语言大厦的基石。理解 dict，就是理解 Python 的半壁江山。
