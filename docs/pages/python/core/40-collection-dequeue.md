---
title: 双端队列
---

# 双端队列

双端队列（Double-Ended Queue，简称 deque）是一种特殊的线性数据结构，它允许你在队列的两端（头部和尾部）都能高效地进行添加和删除操作。你可以把它想象成一条双车道的高速公路，车辆可以从入口进入，也可以从出口离开，而且入口和出口都支持双向通行。

## 代码表现形式

在 Python 中，双端队列由 `collections` 模块提供：

```python
from collections import deque

d = deque()
```

当你第一次用 Python 的 deque 时，可能只是觉得它用起来和 list 差不多——都能 append，都能 pop。但如果你仔细想想，会发现一个奇怪的现象：deque 的 `appendleft` 和 `popleft` 为什么和 list 的对应操作名称不一样？

这个表面上的差异背后，藏着更深层的原理：list 和 deque 的底层数据结构完全不同。list 是动态数组，deque 是分块双向链表。理解了这个底层差异，你才能真正理解为什么在某些场景下 deque 是更好的选择，以及为什么它能实现 O(1) 的头部操作。

## 为什么 list 的头部操作是 O(n)

在讨论 deque 之前，我们先来理解 list 为什么在头部操作上效率低下。

假设有一个 list，它的内部结构大致是这样的：

```
[elem0, elem1, elem2, elem3, elem4, ...]
 ↑
 起始地址
```

当你执行 `list.insert(0, x)` 时，Python 必须把 elem0 到 elem4 整体向后移动一位，然后在位置 0 放入新元素。这个移动操作需要复制每一个元素的指针，对于长度为 n 的列表，这是 O(n) 的操作。

同样的，当你执行 `list.pop(0)` 时，Python 必须把 elem1 到 elemn-1 整体向前移动一位，这是也是 O(n) 的操作。

这个问题的本质在于：list 使用的是连续内存结构。就像电影院的一排座位，如果有人要坐到最左边所有人的都得挪一挪。但如果你用的是双端队列，就像是一条走廊，两端都可以进入，不需要移动中间的人。

## deque 的底层结构

deque（Double-Ended Queue）的全称是双端队列，它在底层使用的是"分块双向链式结构"。

你可以把它想象成这样的一种数据结构：每个块是一个固定大小的数组（Python 中默认是 64 个元素），块和块之间通过双向指针链接：

```
[block A] <-> [block B] <-> [block C] <-> [block D]
```

当你在右侧添加元素时，如果当前块还有空间，就直接写入；如果满了，就创建新块并链接到尾部。

当你在左侧添加元素时也是如此：如果当前块头部还有空间，就直接写入；如果满了，就创建新块并链接到头部。

关键在于：无论是头部还是尾部操作，都只需要操作当前块的固定位置，不需要移动其他块里的任何元素。这就是为什么 deque 的两端操作都是 O(1)。

```python
from collections import deque

d = deque()

# 两端操作都是 O(1)
d.append(1)      # 右端添加
d.appendleft(0)  # 左端添加
d.pop()          # 右端删除，O(1)
d.popleft()      # 左端删除，O(1)
```

## 为什么不用纯链表

既然 deque 这么好，为什么 Python 不直接用传统的双向链表（每个节点单独分配内存）？

答案在于内存效率。传统的双向链表每个节点都需要额外的指针空间，而且这些指针分散在内存各处，导致 CPU 缓存命中率低。

deque 的"分块"设计是一个精妙的折中：每个块是一个小数组（64 元素），块内元素是连续内存保证了缓存友好性，而块间的指针链接又保证了在两端的 O(1) 操作。

从内存局部性角度来看，遍历 deque 的 block 内的元素是缓存友好的，因为它们在内存上是连续的。只有在 block 边界才需要跟随指针跳转到下一个 block。这是性能和内存效率的完美平衡。

## maxlen：有界队列

deque 有一个独特的功能：可以设置最大长度。当队列满时，自动丢弃对侧的元素。

```python
from collections import deque

window = deque(maxlen=3)

for i in range(5):
    window.append(i)
    print(window)
```

输出：

```
deque([0], maxlen=3)
deque([0, 1], maxlen=3)
deque([0, 1, 2], maxlen=3)
deque([1, 2, 3], maxlen=3)
deque([2, 3, 4], maxlen=3)
```

当队列满后再添加元素，会自动从对侧删除最老的元素。这在实现滑动窗口、保留最近 N 条记录、限制缓冲区大小等场景时非常有用。

这个功能在流式处理和数据流控制中特别有用。比如你要实现一个实时监控系统，只想保留最近 100 条错误日志：

```python
recent_errors = deque(maxlen=100)
```

## rotate 的实现原理

deque 的 `rotate` 方法也是面试常问的考点。

`rotate(n)` 将队列中的元素向右旋转 n 位。如果 n 是正数，元素向右移动；如果 n 是负数，元素向左移动。

```python
d = deque([1, 2, 3, 4, 5])

d.rotate(2)
print(d)  # deque([4, 5, 1, 2, 3])

d.rotate(-1)
print(d)  # deque([5, 1, 2, 3, 4])
```

rotate 的实现原理并不是真的移动所有元素——它只是在逻辑上调整了头尾指针的位置。这使得 rotate 的时间复杂度是 O(k)，其中 k 是旋转的步数，而不是 O(n)。

理解这一点很重要：如果 rotate 需要真的移动所有元素，那它就会是 O(n) 的。但实际上它只是调整指针，所以无论 deque 有多长，rotate 的成本只和旋转的步数有关。

## 索引访问与中间操作

deque 支持索引访问：`d[0]` 获取第一个元素，`d[-1]` 获取最后一个元素。

但要注意，deque 不是连续内存结构，访问中间位置时需要跨 block 查找，时间复杂度是 O(n)。如果你需要大量的随机索引访问，list 仍然是更好的选择。

```python
d = deque([1, 2, 3, 4, 5])

print(d[0])   # O(1)
print(d[-1])  # O(1)
print(d[2])   # O(n)
```

deque 的 `insert` 和 `remove` 操作也是在中间位置进行的，时间复杂度是 O(n)。deque 的设计目标不是中间操作，而是两端操作——这是它的核心价值所在。

## 广度优先搜索中的典型应用

deque 最重要的应用场景之一是广度优先搜索（BFS）。

在 BFS 中，我们需要一个队列来存储待访问的节点。每当我们访问一个节点，就把它所有未访问的邻居加入队列。由于我们总是从队列头部取出节点（dequeue），并且从队列尾部添加新节点（enqueue），这正好对应了 deque 的 `popleft` 和 `append` 操作。

```python
from collections import deque

def bfs(graph, start):
    visited = set()
    queue = deque([start])
    visited.add(start)

    while queue:
        node = queue.popleft()
        print(node, end=" ")

        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
```

如果用 list 代替 deque 来实现 BFS，每次 popleft(0) 都是 O(n) 的操作（需要移动所有剩余元素），会导致 BFS 的时间复杂度从 O(V+E) 退化到 O(V²+E)。

## 单调队列与滑动窗口最大值

deque 另一个重要的应用是单调队列，可以用来解决"滑动窗口最大值"问题。

滑动窗口最大值的朴素解法是遍历窗口内的所有元素找最大值，时间复杂度是 O(nk)。但用单调递减队列可以把时间复杂度降到 O(n)。

核心思想是：维护一个从大到小递减的 deque，deque 的头部永远保存当前窗口的最大值。

```python
from collections import deque

def max_sliding_window(nums, k):
    dq = deque()  # 存储索引
    result = []

    for i, num in enumerate(nums):
        # 移除超出窗口范围的元素
        while dq and dq[0] <= i - k:
            dq.popleft()

        # 移除所有比当前元素小的元素（它们不可能是最大值）
        while dq and nums[dq[-1]] < num:
            dq.pop()

        # 添加当前元素
        dq.append(i)

        # 当窗口形成后记录答案
        if i >= k - 1:
            result.append(nums[dq[0]])

    return result
```

这个算法的精妙之处在于：deque 允许我们在两端高效地添加和删除元素，使得我们可以维护一个"单调递减"的队列——队列里的元素是递减的，所以头部永远是当前窗口的最大值。

## extendleft 的顺序陷阱

deque 的 `extendleft` 有一个容易踩的坑：它的行为和我们直觉可能不一样。

当你用 `extendleft` 添加一个可迭代对象时，这些元素是"反向"添加的——第一个元素会被放到最左边。

```python
d = deque([1, 2])
d.extendleft([3, 4, 5])
print(d)  # deque([5, 4, 3, 1, 2])
```

这是因为 `extendleft` 实际上是逐个执行 `appendleft`，每次都把新元素放到最左边。所以最后添加的元素反而在最右边。

理解这一点很重要：在需要按顺序添加多个元素到左侧时，应该先反转可迭代对象，或者使用其他方式。

## 时间复杂度总结

理解 deque 的性能特征，关键是记住它的设计目标：两端操作高效，中间操作低效。

| 操作 | 时间复杂度 | 说明 |
|------|-----------|------|
| append | O(1) | 右端添加 |
| appendleft | O(1) | 左端添加 |
| pop | O(1) | 右端删除 |
| popleft | O(1) | 左端删除 |
| 索引访问两端 | O(1) | d[0], d[-1] |
| 索引访问中间 | O(n) | d[i], i 不是两端 |
| insert | O(n) | 中间插入 |
| remove | O(n) | 中间删除 |
| rotate | O(k) | k 是旋转步数 |

关键理解：**deque 优化的是"两端"，不是"中间"**。如果你需要大量在中间位置插入或删除，list 仍然是更好的选择。deque 不是 list 的替代品，而是不同场景下的不同选择。

## 内存模型与缓存

deque 的分块结构使得它在遍历时有一定的缓存局部性优势——每个 block 内的元素在内存上是连续的，访问起来比纯链表的缓存命中率高。

但如果你的场景是纯粹顺序遍历（从头到尾），list 的表现通常会更好，因为 list 整个是连续内存，而 deque 在 block 边界需要跳转。

所以选择数据结构的标准是：**如果主要操作是两端，选 deque；如果主要操作是纯顺序遍历，选 list**。

## 常见误区

第一个误区是认为 deque 可以完全替代 list。deque 在中间操作上不如 list 高效，所以如果你需要频繁在列表中间插入或删除，或者需要随机访问，list 仍然是更好的选择。

第二个误区是忘记 extendleft 是反向添加。如果你不理解这个行为，可能会写出 bug。记住：extendleft 是逐个 appendleft，所以后添加的元素会在更右侧。

第三个误区是在 deque 上做大量中间操作。deque 的设计目标是两端操作，不是中间操作。在 deque 的中间位置插入或删除是 O(n) 的，和 list 相比没有优势。

第四个误区是不理解 maxlen 的行为。当 deque 满时，新元素会从对侧挤出老的元素。这个行为在某些场景下很有用，但在另一些场景下可能导致意外的数据丢失。

## 面试核心回答

面试中关于 deque 最常问的问题是："deque 和 list 的区别是什么？"

一个完整且深入的回答应该包含以下要点：

deque 使用分块双向链式结构，在两端操作（append、appendleft、pop、popleft）时只移动指针，不需要整体元素搬移，所以是 O(1)。而 list 使用连续内存结构，在头部操作时需要移动所有元素，所以是 O(n)。

deque 的这种结构牺牲了中间位置的随机访问性能——访问中间位置需要跨 block 查找，是 O(n)。而 list 的随机访问是 O(1)。

此外，deque 支持 maxlen 参数，可以限制队列长度，满队时自动挤出老元素，这在实现滑动窗口等功能时非常有用。而 list 没有这个功能。

选择使用哪个数据结构，取决于你的访问模式：如果主要是从两端添加删除，deque 更快；如果主要是随机访问或中间操作，list 更合适。
