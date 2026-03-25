---
title: Counter
---

# Counter

当你需要统计一个序列中各个元素出现的次数时，最直接的做法是这样：

```python
data = ["apple", "banana", "apple", "orange", "banana", "apple"]

counts = {}
for item in data:
    if item not in counts:
        counts[item] = 0
    counts[item] += 1

print(counts)  # {'apple': 3, 'banana': 2, 'orange': 1}
```

这段代码能完成任务，但写起来有点啰嗦。而且从语义上看，我们其实是在表达一种"计数"的意图，但普通 dict 并不能直接传达这个意图。

Counter 就是为这种场景设计的。它是 dict 的子类，但专门用于计数场景。使用 Counter，代码可以简化成：

```python
from collections import Counter

data = ["apple", "banana", "apple", "orange", "banana", "apple"]

counts = Counter(data)
print(counts)  # Counter({'apple': 3, 'banana': 2, 'orange': 1})
```

Counter 不只是语法糖，它有完整的数学模型支撑——它本质上是一个"多重集合"（multiset）的实现。

## 多重集合的数学模型

普通集合 `{a, b, c}` 只能表示"有没有"，不能表示"有多少"。但有时候我们需要表示"每个元素出现了多少次"。

多重集合就是在普通集合的基础上加上"计数"。比如多重集合 `{a×3, b×1, c×2}` 表示 a 出现了 3 次，b 出现了 1 次，c 出现了 2 次。

Counter 正是多重集合的 Python 实现。它的核心模型是：元素 -> 出现次数。这个映射关系和普通 dict 的 key-value 映射完全一致，所以 Counter 用 dict 来存储是很自然的设计。

## 基本用法

Counter 可以接收任何可迭代对象：

```python
from collections import Counter

# 从列表创建
c1 = Counter(["apple", "banana", "apple", "orange"])
print(c1)  # Counter({'apple': 2, 'banana': 1, 'orange': 1})

# 从字符串创建（字符计数）
c2 = Counter("hello")
print(c2)  # Counter({'l': 2, 'h': 1, 'e': 1, 'o': 1})

# 从字典创建
c3 = Counter({"a": 3, "b": 1})
print(c3)  # Counter({'a': 3, 'b': 1})

# 用关键字参数创建
c4 = Counter(a=3, b=1)
print(c4)  # Counter({'a': 3, 'b': 1})
```

空 Counter 可以用 `Counter()` 创建一个没有任何元素的计数器。

## 与普通 dict 的关键区别

Counter 和普通 dict 有一些细微但重要的区别，理解这些区别才能正确使用 Counter。

第一个区别是访问不存在的 key。普通 dict 访问不存在的 key 会抛出 KeyError，但 Counter 返回 0 而不是报错：

```python
from collections import Counter

c = Counter(["apple", "banana"])

print(c["apple"])    # 1
print(c["orange"])   # 0，不会抛出 KeyError
```

这是 Counter 重写了 `__missing__` 方法的结果。它让计数逻辑更流畅，不需要每次都检查 key 是否存在。

但要注意：Counter 返回 0 并不意味着 key 存在于 Counter 中。

```python
from collections import Counter

c = Counter()
print(c["new"])      # 0
print("new" in c)    # False，key 不存在
```

这和 defaultdict(int) 的行为不同：defaultdict(int) 访问不存在的 key 时会插入 key，而 Counter 不会。

## update 和加减运算符

Counter 支持一些特殊运算，这些运算让它更像一个数学集合。

`update` 方法用于增加计数：

```python
from collections import Counter

c = Counter(["apple", "banana"])
c.update(["apple", "orange"])
print(c)  # Counter({'apple': 3, 'banana': 1, 'orange': 1})
```

注意 update 是增加计数，不是覆盖。如果你想减少计数，可以使用 `-` 运算符：

```python
from collections import Counter

c = Counter(["apple", "banana", "orange"])
c.subtract(["apple", "banana", "banana"])
print(c)  # Counter({'apple': 2, 'banana': 0, 'orange': 1})
```

`subtract` 允许计数变成 0 或负数，而 `+` 运算符产生的 Counter 会丢弃负数和零：

```python
from collections import Counter

c1 = Counter(a=3, b=1)
c2 = Counter(a=1, b=2)

print(c1 - c2)  # Counter({'a': 2})，负数被丢弃
print(c1 + c2)  # Counter({'a': 4, 'b': 3})，所有计数相加
```

## most_common：找出最高频的元素

`most_common` 是 Counter 最常用的方法之一，它返回出现次数最多的前 n 个元素：

```python
from collections import Counter

c = Counter("hello world")

print(c.most_common(3))  # [('l', 3), ('o', 2), (' ', 1)]
print(c.most_common())   # 全部元素按频率排序
```

面试中常问的"Top-K 问题"可以用 most_common 简洁地解决：

```python
from collections import Counter

nums = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
print(Counter(nums).most_common(2))  # [(4, 4), (3, 3)]
```

`most_common` 的时间复杂度是 O(n log k)，其中 n 是元素总数，k 是要返回的个数。如果不指定 k，会对所有元素排序，时间复杂度是 O(n log n)。

## elements：展开计数

`elements` 方法返回计数器中每个元素重复对应次数的迭代器：

```python
from collections import Counter

c = Counter(a=3, b=1)
print(list(c.elements()))  # ['a', 'a', 'a', 'b']
```

注意 elements 会忽略计数为 0 或负数的元素：

```python
from collections import Counter

c = Counter(a=2, b=0, c=-1)
print(list(c.elements()))  # ['a', 'a']
```

这个方法在需要"展开"计数结果时很有用，比如生成验证码：

```python
from collections import Counter
import random

chars = Counter("aaabbc")
password = "".join(random.choice(list(chars.elements())) for _ in range(6))
```

## 集合运算：& 和 |

Counter 支持两个 Counter 之间的交集（&）和并集（|）运算。但这里的语义和普通集合不同。

`&` 运算符返回两个 Counter 中相同元素的最小计数：

```python
from collections import Counter

c1 = Counter(a=3, b=2, c=1)
c2 = Counter(a=1, b=2, d=4)

print(c1 & c2)  # Counter({'a': 1, 'b': 2})，取 min
```

`|` 运算符返回两个 Counter 中所有元素的最大计数：

```python
from collections import Counter

c1 = Counter(a=3, b=2, c=1)
c2 = Counter(a=1, b=2, d=4)

print(c1 | c2)  # Counter({'a': 3, 'b': 2, 'd': 4, 'c': 1})，取 max
```

这是多重集合的数学定义决定的：交集取较小值，并集取较大值。但很多初学者会误以为 `&` 是集合交集（类似 set），实际上它是"最小计数"运算。

## 为什么允许负计数

`subtract` 允许计数器变成负数。这在某些场景下是有意义的，比如"增量比较"：

```python
from collections import Counter

before = Counter("aabbc")
after = Counter("abccc")

diff = Counter(after) - Counter(before)
print(diff)  # Counter({'c': 1})，新增的元素
```

这里 `after - before` 会丢弃负数，只保留新增的部分。如果 `before` 比 `after` 多，结果会显示减少的元素：

```python
from collections import Counter

before = Counter("aabbc")
after = Counter("ab")

diff = Counter(before) - Counter(after)
print(diff)  # Counter({'c': 1})，减少的元素
```

但注意，两个 Counter 相减得到的结果会丢弃所有负数，而不是保留负数显示"少多少"：

```python
from collections import Counter

c1 = Counter(a=3, b=1)
c2 = Counter(a=1, b=3)

print(c1 - c2)  # Counter({'a': 2})，b 被丢弃因为结果是 0 或负数
```

如果你想看到"少多少"，需要自己实现逻辑，或者使用 `subtract` 方法直接修改原 Counter。

## 常见工程场景

### 统计字符或单词出现频率

这是 Counter 最直接的应用：

```python
from collections import Counter

text = "hello world hello python hello"
words = text.split()
print(Counter(words))  # Counter({'hello': 3, 'world': 1, 'python': 1})

# 统计字母频率
print(Counter(text.replace(" ", "")))  # 每个字符的计数
```

### Top-K 高频问题

Top-K 问题在面试和工程中都常见。Counter 加 most_common 是标准解法：

```python
from collections import Counter

def top_k(nums, k):
    return Counter(nums).most_common(k)
```

如果是流式数据或者数据量太大无法全部加载到内存，需要用堆或者分治法：

```python
import heapq
from collections import Counter

def top_k_stream(nums, k):
    counter = Counter()
    for num in nums:
        counter[num] += 1
        if len(counter) > k:
            pass
    return counter.most_common(k)
```

### 差异对比

Counter 可以用来对比两份数据的差异：

```python
from collections import Counter

old_users = ["a", "b", "c", "d"]
new_users = ["b", "c", "d", "e", "f"]

old_count = Counter(old_users)
new_count = Counter(new_users)

added = new_count - old_count  # 新增的
removed = old_count - new_count  # 移除的

print(list(added.elements()))  # ['e', 'f']
print(list(removed.elements()))  # ['a']
```

## 和 defaultdict(int) 的区别

虽然 Counter 和 `defaultdict(int)` 看起来差不多，但有一个关键区别：

访问不存在的 key 时，`defaultdict(int)` 会插入 key 并返回 0，而 Counter 只返回 0 但不插入：

```python
from collections import Counter, defaultdict

# defaultdict(int)
d = defaultdict(int)
print(d["new"])      # 0
print("new" in d)    # True，插入了

# Counter
c = Counter()
print(c["new"])      # 0
print("new" in c)    # False，没有插入
```

这个区别在实际使用中很重要。如果你需要判断一个 key 是否"曾经出现"过，应该用 Counter；如果你需要初始化计数，应该用 defaultdict(int)。

```python
# 用 defaultdict(int) 实现计数
d = defaultdict(int)
for item in data:
    d[item] += 1  # 自动初始化为 0 再加 1

# 用 Counter 实现计数
c = Counter(data)  # 直接从数据创建
```

## 性能分析

Counter 基于 dict，所以各项操作的时间复杂度和 dict 一致：

| 操作 | 时间复杂度 |
|------|-----------|
| 访问/更新计数 | O(1) |
| update | O(n) |
| most_common(k) | O(n log k) |
| elements | O(n) |

内存占用和 dict 基本一致，因为底层就是 dict。

## 常见误区

第一个误区是把 Counter 的 `&` 和 `|` 误认为是集合的交集并集。Counter 的 `&` 是"取较小计数"，`|` 是"取较大计数"，这是多重集合的数学语义。

第二个误区是混淆 `subtract` 和 `-` 运算符。`-` 运算符会丢弃结果中的负数和零，`subtract` 方法会保留负数。

第三个误区是不理解 Counter 的访问行为。Counter 访问不存在的 key 返回 0 但不插入，这有时候会导致意外：

```python
from collections import Counter

c = Counter(["a", "b"])
c["c"] += 1  # 这会正常工作，但 c 会被插入
print(c)  # Counter({'a': 1, 'b': 1, 'c': 1})
```

注意，虽然访问 `c["c"]` 返回 0，但如果对返回值执行 `+=` 操作，仍然会正确地初始化并增加计数。这是因为 `c["c"] += 1` 实际上是 `c["c"] = c["c"] + 1`，赋值操作会触发插入。

## 面试核心回答

面试中关于 Counter 常问的问题包括：

"Counter 和 defaultdict(int) 有什么区别？"——答案是：两者都能用于计数，但 defaultdict(int) 访问不存在的 key 时会插入，而 Counter 不会。

"Counter 的 & 运算符语义是什么？"——答案是：取两个 Counter 中相同元素的最小计数。

"Counter.most_common 的复杂度是什么？"——答案是：O(n log k)，其中 n 是元素总数，k 是要返回的个数。如果不指定 k，则是 O(n log n)。

"subtract 和 - 运算符有什么区别？"——答案是：- 运算符丢弃结果中的负数和零，subtract 保留负数。

理解这些问题的答案，说明你对 Counter 的理解已经足够深入了。
