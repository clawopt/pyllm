---
title: defaultdict
---

# defaultdict

在 Python 中，当我们需要对数据进行分组时，最常见的写法是这样的：

```python
groups = {}
for key, value in data:
    if key not in groups:
        groups[key] = []
    groups[key].append(value)
```

这段代码有个问题：`if key not in groups` 这个判断在每次循环时都要执行。虽然不会出错，但总觉得有点啰嗦。

为了简化这种模式，有人会改用 `setdefault`：

```python
groups = {}
for key, value in data:
    groups.setdefault(key, []).append(value)
```

这确实简洁了一些。但 `setdefault` 有一个隐藏的性能问题：无论 key 是否存在，每次循环都会创建一个新的空列表 `[]`，只不过如果 key 已存在，这个新列表就被丢弃了。当数据量巨大时，这种无意义的对象创建会成为开销。

`defaultdict` 正是为了解决这个问题而设计的。它是 dict 的子类，但行为不同：当你访问一个不存在的 key 时，它不会报错，而是会调用你指定的 factory 函数创建默认值。

## defaultdict 的基本用法

使用 defaultdict 非常简单。只需要在创建时指定一个 factory 函数（叫做 `default_factory`），当访问不存在的 key 时，default_factory 就会被调用，生成一个默认值：

```python
from collections import defaultdict

groups = defaultdict(list)

for key, value in [("a", 1), ("b", 2), ("a", 3)]:
    groups[key].append(value)

print(dict(groups))  # {'a': [1, 3], 'b': [2]}
```

注意这里没有任何 if 判断。`groups["a"]` 在第一次访问时会自动创建一个空列表 `[]`，这就是 `default_factory=list` 的效果。

## 它的底层工作原理

理解 defaultdict 的关键是理解它什么时候创建默认值。

当你写 `groups["new_key"]` 时，Python 内部会调用 `__getitem__` 方法。如果 key 存在，直接返回 value。如果 key 不存在，defaultdict 会：

1. 调用 `default_factory()` 创建默认值
2. 把这个默认值插入字典
3. 返回这个默认值

注意这个过程是在"访问"时触发的，而不是在"创建"时触发。这意味着如果你从来不访问某个 key，这个 key 就不会被创建：

```python
d = defaultdict(list)

print("never_created" in d)  # False，key 从未被创建
print(d["never_accessed"])   # []，访问时触发了创建
print("never_accessed" in d) # True，访问后被创建了
```

这个"延迟创建"的机制是 defaultdict 性能优势的关键：它只在真正需要的时候才创建默认值。

## default_factory 可以是任何可调用对象

最常见的 default_factory 是 `list`、`int`、`set`，但理论上任何可调用对象都可以：

```python
from collections import defaultdict

# 计数器
counter = defaultdict(int)
for char in "hello":
    counter[char] += 1
print(counter)  # {'h': 1, 'e': 1, 'l': 2, 'o': 1}

# set 分组（自动去重）
unique = defaultdict(set)
for item in [1, 2, 2, 3, 3, 3]:
    unique["numbers"].add(item)
print(unique)  # {'numbers': {1, 2, 3}}

# 自定义工厂
def create_dict():
    return {"count": 0, "items": []}

d = defaultdict(create_dict)
d["category"]["count"] += 1
print(d)  # {'category': {'count': 1, 'items': []}}
```

当使用 `default_factory=int` 时，访问不存在的 key 会调用 `int()`，它返回 `0`。这就是为什么 `counter[char] += 1` 能正常工作——第一次访问时得到 0，然后加 1 得到 1。

## defaultdict 和 dict.get() 的区别

有人会问："我用 `dict.get(key, [])` 不也行吗？"

这个方法有个根本问题：`get` 返回的是默认值，但不会把默认值插入字典。所以如果你接着对这个返回值调用修改操作（比如 append），你是在修改一个临时对象，原字典并没有被修改：

```python
d = {}

value = d.get("key", [])
value.append(1)  # 修改的是临时列表，不是 d["key"]
print(d)          # {}，d 没有变化！

# 用 defaultdict 就没有这个问题
from collections import defaultdict
d = defaultdict(list)
d["key"].append(1)  # 正确：访问创建了默认值，并返回给调用者修改
print(d)            # {'key': [1]}
```

这是 defaultdict 和 `dict.get()` 的本质区别：`dict.get()` 返回默认值但不插入，`defaultdict` 返回默认值并插入。

## 常见的工程场景

### 邻接表

在表示图结构时，邻接表是一种常见的表示方法。用 defaultdict 可以优雅地构建邻接表：

```python
from collections import defaultdict

edges = [(1, 2), (1, 3), (2, 4), (3, 4), (4, 5)]

graph = defaultdict(list)
for src, dst in edges:
    graph[src].append(dst)

print(dict(graph))
# {1: [2, 3], 2: [4], 3: [4], 4: [5]}
```

如果不用 defaultdict，就需要先检查 key 是否存在：

```python
graph = {}
for src, dst in edges:
    if src not in graph:
        graph[src] = []
    graph[src].append(dst)
```

### 多级嵌套结构

有时需要创建多级嵌套的字典结构，defaultdict 可以优雅地处理：

```python
from collections import defaultdict

tree = defaultdict(lambda: defaultdict(list))

tree["user1"]["skills"].append("Python")
tree["user2"]["skills"].extend(["Java", "Go"])

print(dict(tree))
```

但要注意，多级 defaultdict 需要小心处理，因为访问 `tree["user1"]` 得到的是一个 defaultdict，而不是普通字典。

### 批量统计

当需要按多个维度统计数据时，defaultdict 非常有用：

```python
from collections import defaultdict

data = [
    ("2024-01", "A", 100),
    ("2024-01", "B", 200),
    ("2024-02", "A", 150),
    ("2024-02", "B", 180),
]

# 按月份统计
by_month = defaultdict(int)
for month, _, amount in data:
    by_month[month] += amount

# 按产品统计
by_product = defaultdict(int)
for _, product, amount in data:
    by_product[product] += amount

print(dict(by_month))    # {'2024-01': 300, '2024-02': 330}
print(dict(by_product))  # {'A': 250, 'B': 380}
```

## 一个容易踩的坑

defaultdict 在"访问"时会创建 key。这既是它的优势，也是一个容易忽略的陷阱。

```python
from collections import defaultdict

d = defaultdict(list)

# 只是访问了一下
print(d["not_exist"])  # []
print(dict(d))          # {'not_exist': []}，key 被创建了！

# 判断存在性应该用 in
d2 = defaultdict(list)
if "not_exist" in d2:   # False
    print(d2["not_exist"])
print(dict(d2))          # {}，没有创建任何 key
```

如果你只是想在 key 不存在时获得一个默认值用于判断，不应该用 defaultdict，否则会意外创建 key。

## 另一个常见的误用：返回同一个对象

defaultdict 的 default_factory 每次访问都会调用。如果 factory 返回的是一个可变对象，而且你直接修改它而不重新赋值，可能会出现意外情况：

```python
from collections import defaultdict

# 错误写法
shared = []
d = defaultdict(lambda: shared)
d["a"].append(1)
d["b"].append(2)
print(d["a"])  # [1, 2]，两个 key 指向同一个列表！
print(d["b"])  # [1, 2]

# 正确写法：每次返回新对象
d = defaultdict(list)
d["a"].append(1)
d["b"].append(2)
print(d["a"])  # [1]
print(d["b"])  # [2]
```

这是因为 lambda 返回的是同一个 list 对象引用。`defaultdict(list)` 是安全的，因为 `list()` 每次调用都会创建一个新的空列表。

## defaultdict 与 setdefault 的性能对比

虽然 defaultdict 和 setdefault 都能解决"key 不存在时创建默认值"的问题，但它们的性能特性不同。

`setdefault` 的问题是：即使 key 已存在，它每次都要调用 factory 函数创建默认值（虽然这个值会被丢弃）：

```python
d = {}

# 这两行代码，即使 "key" 已存在，仍然会创建一个新列表
for i in range(1000000):
    d.setdefault("key", []).append(i)
```

而 defaultdict 只在 key 真正不存在时才调用 factory：

```python
from collections import defaultdict

d = defaultdict(list)

# 只在第一次访问时调用 list()
for i in range(1000000):
    d["key"].append(i)
```

在大规模数据处理中，这个差异可能会影响性能。defaultdict 的设计更符合"延迟计算"的原则，性能也更好。

## defaultdict 和普通 dict 的选择

什么时候用 defaultdict，什么时候用普通 dict？

用 defaultdict：当你的逻辑本质上是"访问不存在的 key 时自动创建默认值"。这在分组、计数、构建图结构等场景下非常自然。

用普通 dict：当你不希望自动创建 key，或者需要区分"key 不存在"和"key 存在但值为某值"的情况。

```python
from collections import defaultdict

# 适合用 defaultdict 的场景：分组统计
groups = defaultdict(list)
for category, item in data:
    groups[category].append(item)

# 适合用普通 dict 的场景：需要区分存在性和具体值
if key in config:  # 只检查存在性
    do_something()
value = config.get(key, default_value)  # 只在需要时获取
```

## defaultdict 与 Counter 的关系

Counter 是 defaultdict(int) 的一个特化版本，它专门用于计数场景。

```python
from collections import Counter, defaultdict

# 这两者效果相似
c = Counter("hello")
d = defaultdict(int, {"h": 1, "e": 1, "l": 2, "o": 1})

# Counter 有更多统计功能
print(Counter("hello").most_common(2))  # [('l', 2), ('h', 1)]
```

但要注意它们的行为有一个关键区别：访问不存在的 key 时，Counter 返回 0 但不插入，defaultdict(int) 返回 0 并插入。

```python
from collections import Counter, defaultdict

c = Counter()
c["new"]  # 返回 0
print("new" in c)  # False，没有插入

d = defaultdict(int)
d["new"]  # 返回 0
print("new" in d)  # True，插入了
```

## 面试常问问题

面试中关于 defaultdict 有几个常见问题：

第一个问题是"defaultdict 和普通 dict 有什么区别？"——答案是 defaultdict 在访问不存在的 key 时会自动调用 default_factory 创建默认值，而普通 dict 会抛出 KeyError。

第二个问题是"default_factory 什么时候被调用？"——答案是仅在访问不存在的 key 时调用，已存在的 key 不会触发。

第三个问题是"setdefault 和 defaultdict 的性能区别是什么？"——答案是 setdefault 每次都调用 factory 创建默认值（即使 key 已存在），而 defaultdict 只在 key 不存在时调用。

第四个问题是"为什么 defaultdict(list) 安全但 lambda: [] 不安全？"——答案是 list() 每次调用创建新的空列表对象，而 lambda: [] 捕获的是同一个列表对象引用。

如果你能准确回答这些问题，说明你对 defaultdict 的理解已经足够深入了。
