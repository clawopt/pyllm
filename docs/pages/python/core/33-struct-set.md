---
title: 集合
---

# 集合

set 是 Python 中的一种数据类型，它就像是你生活中的一个"袋子"，专门用来装东西，但有三个特别的规则：

**1. 不重复** —— 同样的东西只能放一份
**2. 无序** —— 东西放进去没有固定的位置
**3. 确定性** —— 一个东西要么在袋子里，要么不在

想象你有一个**音乐播放列表**：

- 用 list（列表）：`["周杰伦", "周杰伦", "林俊杰", "周杰伦"]` —— 同一首歌可能出现多次，顺序很重要
- 用 set（集合）：`{"周杰伦", "林俊杰"}` —— 自动去重，不关心顺序，只关心"有没有这首歌"

很多人觉得 set 很简单——不就是"没有 value 的 dict"吗？

这个理解不算错，但太浅了。set 真正的价值在于它是"数学集合"的程序实现，它的意义不在于存储数据，而在于表达成员关系和做集合运算。

当你需要判断一个元素是否在某个集合里，当你需要对多个集合做交集并集差集，当你需要去重——set 是最自然的选择。但如果你不理解 set 的设计初衷和使用场景，就容易用它做不该做的事（比如用它做复杂运算，或者误以为它有序）。

理解 set，不能只记住 API，而要理解它和 dict 的关系、它的哈希表本质、它的运算语义、以及为什么它比 list 更适合做成员测试。

## 集合的数学模型

set 在计算机科学中对应的是"集合"这个数学概念。集合有三个核心特征：无序、不重复、确定性。

无序意味着 `{1, 2, 3}` 和 `{3, 2, 1}` 是同一个集合。Python 的 set 打印出来可能有顺序，但那只是实现细节，不应该依赖它。

不重复是 set 最重要的特征。当你往 set 里添加重复元素时，它只会保留一份。这不是 set 的限制，而是它存在的理由——去重是 set 的本职工作。

确定性意味着一个元素要么在集合里，要么不在，不存在"可能存在"的状态。这和概率论里的集合概念是一致的。

```python
s = {1, 2, 3, 2, 1}
print(s)  # {1, 2, 3}，自动去重
```

## set 和 dict 的关系

如果你理解了 dict 的实现，set 就很好理解。set 本质上就是"只有 key 没有 value 的 dict"。

dict 存储的是 `(key, value)` 对，set 存储的是 `key` 本身。它们的底层实现几乎完全一样：都用哈希表，都需要 key 可哈希，都有 O(1) 的平均查找复杂度。

唯一的区别是 set 不需要存储 value，所以它的空间利用率更高。但 Python 内部实际上复用了 dict 的很多实现代码，所以它们在内部结构上是非常接近的。

这个关系对于面试很重要。当面试官问"set 和 dict 有什么关系"时，你应该回答：set 是去掉了 value 的 dict，它们使用相同的哈希表实现，set 的空间更小，因为只存储 key。

## 创建集合的方式

创建 set 有两种方式：字面量和构造函数。

字面量用花括号，但要注意：`{}` 不是空 set，而是空字典。创建空 set 必须用 `set()`：

```python
s = {1, 2, 3}      # 非空 set
empty = set()       # 空 set
not_dict = {}       # 这是 dict，不是 set

print(type({}))     # <class 'dict'>
print(type(set()))  # <class 'set'>
```

构造函数可以接收任何可迭代对象，会自动去重：

```python
s1 = set([1, 2, 2, 3, 3])  # {1, 2, 3}
s2 = set("hello")           # {'h', 'e', 'l', 'o'}
s3 = set(range(5))          # {0, 1, 2, 3, 4}
```

这使得 set 是天然的去重工具，比 list 的去重方式要简洁得多。

## 成员测试：set 的核心价值

set 最核心的应用场景是成员测试。当我们需要频繁判断一个元素是否存在时，set 比 list 高效得多。

list 的成员测试是 O(n) 的，需要遍历所有元素。set 的成员测试是 O(1) 的，通过哈希直接定位。

```python
# list 查找 O(n)
nums_list = list(range(10000))
%timeit 9999 in nums_list  # 可能是几百微秒

# set 查找 O(1)
nums_set = set(range(10000))
%timeit 9999 in nums_set   # 可能是几十纳秒
```

差距可能达到上万倍。在处理大数据时，这个差距会直接影响程序性能。

```python
# 常见场景：过滤
valid_ids = {1, 5, 10, 20}
all_ids = range(100)

# 用 set 做过滤
filtered = [x for x in all_ids if x in valid_ids]

# 如果 valid_ids 是 list，这个过滤是 O(n*m)
# 如果 valid_ids 是 set，这个过滤是 O(n)
```

## 添加与删除操作

set 的添加操作有 `add` 和 `update` 两个方法。

`add` 添加单个元素，`update` 接收可迭代对象，把其中的每个元素添加进去：

```python
s = {1, 2}

s.add(3)           # {1, 2, 3}
s.update([4, 5])   # {1, 2, 3, 4, 5}
s.update("hello")  # 添加字符串中的字符
```

删除操作有三个：`remove`、`discard`、`pop`。

`remove` 删除指定元素，如果不存在会抛出 KeyError。`discard` 也是删除指定元素，但如果不存在不会报错。`pop` 删除并返回任意一个元素，set 为空时会抛出 KeyError：

```python
s = {1, 2, 3}

s.remove(2)      # {1, 3}
s.discard(100)   # 什么都不发生，不报错
s.discard(1)    # {3}

s.pop()          # 删除并返回 3，set 变为空
s.pop()          # KeyError: pop from an empty set
```

面试常问 `remove` 和 `discard` 的区别。答案是：`remove` 不存在时抛异常，`discard` 不存在时什么都不做。

## 集合运算：set 的真正强大之处

set 真正强大的地方在于它的数学集合运算：交集、并集、差集、对称差。

这些运算在 Python 中有两种写法：运算符和等价的方法调用。

```python
a = {1, 2, 3, 4}
b = {3, 4, 5, 6}
```

交集是两个集合的公共元素：

```python
print(a & b)        # {3, 4}
print(a.intersection(b))  # 同样效果
```

并集是所有元素合并去重：

```python
print(a | b)        # {1, 2, 3, 4, 5, 6}
print(a.union(b))   # 同样效果
```

差集是 a 有但 b 没有的元素：

```python
print(a - b)        # {1, 2}
print(a.difference(b))  # 同样效果
```

对称差是 a 和 b 互相没有的公共部分：

```python
print(a ^ b)        # {1, 2, 5, 6}
print(a.symmetric_difference(b))  # 同样效果
```

这些运算在处理"标签系统"、"权限管理"、"数据过滤"等场景时非常有用。

```python
# 场景：用户画像标签系统
user1_tags = {"python", "machine-learning", "deep-learning"}
user2_tags = {"python", "web-development", "data-analysis"}

# 共同标签
common = user1_tags & user2_tags
print(common)  # {"python"}

# 用户1独有的标签
unique_to_user1 = user1_tags - user2_tags
print(unique_to_user1)  # {"machine-learning", "deep-learning"}

# 所有标签
all_tags = user1_tags | user2_tags
print(all_tags)  # {"python", "machine-learning", "deep-learning", "web-development", "data-analysis"}
```

## 子集与超集判断

set 还支持子集和超集的判断。

`a <= b` 判断 a 是否是 b 的子集，`a < b` 判断是否是真子集（a 是 b 的子集且不等于 b）。`a >= b` 判断 a 是否是 b 的超集，`a > b` 判断是否是真超集：

```python
a = {1, 2}
b = {1, 2, 3}

print(a <= b)   # True，a 是 b 的子集
print(a < b)    # True，a 是 b 的真子集
print(a >= b)   # False
print(a > b)    # False
```

子集和超集在处理"权限系统"、"分类体系"时很有用。比如判断一个用户的权限集合是否是某个角色权限集合的子集。

```python
# 权限检查场景
required_permissions = {"read", "write", "delete"}
user_permissions = {"read", "write"}

# 检查用户权限是否满足要求
if user_permissions >= required_permissions:
    print("权限足够")
else:
    print("权限不足")
```

## frozenset：不可变集合

set 是可变的，所以不可哈希，不能作为 dict 的 key。但有时候我们需要不可变的集合，比如作为 dict 的 key，或者需要哈希集合的时候。

frozenset 就是这个需求的设计。它是 set 的不可变版本，创建后不能添加、删除元素：

```python
fs = frozenset([1, 2, 3])
print(fs)  # frozenset({1, 2, 3})

# frozenset 不可哈希内部元素
d = {}
d[fs] = "value"
print(d)  # {frozenset({1, 2, 3}): 'value'}

# set 不能作为 dict key
s = {1, 2, 3}
d[s] = "value"  # TypeError: unhashable type 'set'
```

frozenset 除了不可变之外，拥有普通 set 的所有功能：交集、并集、差集等运算都可以用。

frozenset 和 set 的关系，类似于 tuple 和 list 的关系——都是可变与不可变的对应。tuple 是不可变的 list，frozenset 是不可变的 set。

## 时间复杂度分析

set 的各项操作时间复杂度如下：

| 操作 | 时间复杂度 |
|------|-----------|
| 添加 | O(1) |
| 删除 | O(1) |
| 成员测试 | O(1) |
| 交集 | O(min(n, m)) |
| 并集 | O(n + m) |
| 差集 | O(n) |
| 对称差 | O(n + m) |

最坏情况下（大量哈希冲突），上述 O(1) 操作可能退化到 O(n)。

集合运算的时间复杂度需要注意：交集是 O(min(n, m))，因为只需要遍历较小的集合。但并集是 O(n + m)，因为需要合并两个集合的所有元素。

## 常见误区

第一个误区是认为 set 是有序的。虽然 Python 的 set 实现会维护某种顺序（尤其是在小规模数据上），但这不应该依赖。如果需要有序集合，应该使用其他数据结构。

第二个误区是忘记 set 的元素必须可哈希。set 本身是可变的，所以不可哈希。set 的元素必须是不可变对象：

```python
s = {1, 2, 3}     # 正确
s = {[1, 2]}      # 错误，list 不可哈希
s = {(1, 2)}      # 正确，tuple 可哈希
s = {{1, 2}}      # 错误，set 不可哈希
```

第三个误区是在遍历中修改 set。这和 dict 一样会导致 RuntimeError。应该先复制再修改，或者使用集合运算得到新集合。

```python
s = {1, 2, 3, 4, 5}

# 错误
for x in s:
    if x > 3:
        s.remove(x)  # RuntimeError

# 正确：遍历副本
for x in s.copy():
    if x > 3:
        s.remove(x)

# 更好：用集合运算得到新集合
s = {x for x in s if x <= 3}
```

第四个误区是混淆交集运算符 `&` 和逻辑运算符。set 的 `&` 是集合交集，不是逻辑与。但在某些情况下结果可能看起来相似，不要弄混。

## set 和 list 的选择

什么时候用 set，什么时候用 list？

用 set 的场景：需要去重、需要成员测试（频繁判断是否存在）、需要做集合运算（交集并集差集）。

用 list 的场景：需要保持顺序、需要通过索引访问、需要在中间位置插入或删除（虽然这效率不高）、数据量很小（list 的常量因子更小，小数据可能更快）。

```python
# 成员测试用 set
if x in large_collection:  # 改用 set 更快

# 去重用 set
unique = set(duplicated_data)

# 需要顺序用 list
ordered_data = list(set(original_data))  # 先去重，但顺序可能改变

# 需要索引访问用 list
third_element = data[2]
```

理解这些差异，才能在写代码时做出正确的选择。
