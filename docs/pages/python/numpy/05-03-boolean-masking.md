# 布尔索引与掩码

布尔索引是NumPy中最强大也最直观的数据筛选方式。在LLM和深度学习中，布尔索引通常被称为"掩码"（Mask），这是一种极其重要的概念。想象一下，当你需要屏蔽掉attention计算中的padding位置时，你需要创建一个布尔掩码来标记哪些位置是有效的；或者当你需要实现因果注意力（只关注当前位置及之前的token）时，你需要创建一个上三角的掩码。布尔索引允许你用一个布尔数组作为"掩码"，精确地选择数组中满足特定条件的元素。理解掩码的概念和使用方法是实现各种注意力机制的基础。

## 基本用法

布尔索引用一个与数组形状相同的布尔数组来选择元素：

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# 创建一个布尔掩码：哪些元素大于5
bool_mask = arr > 5
print(f"原始数组: {arr}")
print(f"布尔掩码: {bool_mask}")

# 使用布尔掩码选择元素
selected = arr[bool_mask]
print(f"选中的元素: {selected}")

# 简化写法：直接在括号中使用条件表达式
selected = arr[arr > 5]
print(f"arr > 5: {selected}")
```

布尔索引的语法非常直观：数组名[条件表达式]。条件表达式会返回一个布尔数组，然后NumPy选择其中 True 对应的元素。

## 比较运算符

NumPy支持多种比较运算符，这些运算符返回布尔数组：

```python
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# 各种比较运算
print(f"等于: {arr == 5}")
print(f"不等于: {arr != 5}")
print(f"大于: {arr > 5}")
print(f"大于等于: {arr >= 5}")
print(f"小于: {arr < 5}")
print(f"小于等于: {arr <= 5}")
```

## 组合条件

组合多个条件时，必须使用 `&`（AND）、`|`（OR）和 `~`（NOT）：

```python
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# AND: 两个条件都满足
print(f"(arr > 3) & (arr < 8): {(arr > 3) & (arr < 8)}")

# OR: 满足任一条件
print(f"(arr < 3) | (arr > 8): {(arr < 3) | (arr > 8)}")

# NOT: 条件不满足
print(f"~(arr > 5): {~(arr > 5)}")
```

注意，在NumPy中组合布尔条件时，必须使用 `&`、`|` 和 `~`，而不是Python的 `and`、`or` 和 `not`。这是因为NumPy需要对数组的每个元素进行操作，而不是对整个数组进行判断。

## np.where：条件选择的核心函数

`np.where` 是进行条件选择的核心函数：

```python
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# 找到满足条件的元素的索引
indices = np.where(arr > 5)
print(f"大于5的索引: {indices}")

# np.where 的三参数形式：条件成立时选x，不成立时选y
result = np.where(arr > 5, arr * 2, arr)
print(f"大于5的元素翻倍: {result}")

# 将小于等于5的元素设为0
result = np.where(arr > 5, arr, 0)
print(f"小于等于5设为0: {result}")
```

`np.where` 的三参数形式相当于向量化版的 if-else，在数据预处理中非常有用。

## np.where 与掩码的结合

在LLM场景中，经常需要结合布尔掩码和 `np.where` 来处理数据：

```python
hidden_states = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
attention_mask = np.array([True, True, True, False, False])  # True表示有效

# 只计算有效位置的attention
valid_states = np.where(attention_mask, hidden_states, 0.0)
print(f"有效位置的值: {valid_states}")

# 计算有效位置的总和
valid_sum = np.where(attention_mask, hidden_states, 0.0).sum()
print(f"有效位置的总和: {valid_sum}")
```

## 在LLM场景中的应用

### Padding Mask（填充掩码）

在处理变长序列时，不足最大长度的序列需要padding到相同长度。Padding位置不应该参与attention计算，需要被mask掉：

```python
batch_size = 4
max_seq_len = 10

# 模拟实际序列长度
seq_lengths = np.array([7, 5, 10, 3])

# 创建padding mask: True表示有效位置，False表示padding
padding_mask = np.zeros((batch_size, max_seq_len), dtype=bool)
for i, length in enumerate(seq_lengths):
    padding_mask[i, :length] = True

print(f"Padding掩码:\n{padding_mask.astype(np.int32)}")

# 在attention计算中应用mask
attention_scores = np.random.randn(batch_size, max_seq_len, max_seq_len).astype(np.float32)

# 将padding位置的score设为一个很大的负数，使其softmax后接近0
masked_scores = np.where(
    padding_mask[:, np.newaxis, :],  # 广播掩码
    attention_scores,
    -1e9
)
print(f"Mask后分数形状: {masked_scores.shape}")
```

Padding mask 的原理是：在 softmax 之前，将 padding 位置的分数设为一个很大的负数，这样 softmax 后的概率就会接近于 0。

### Causal Mask（因果掩码）

在自回归模型（如GPT）中，当前位置只能关注之前的位置，不能"看到"未来的内容。需要创建一个因果掩码：

```python
seq_len = 5

# 方法1：使用 np.triu 创建上三角掩码
positions = np.arange(seq_len)
causal_mask = positions[np.newaxis, :] <= positions[:, np.newaxis]
print(f"因果掩码（<=）:\n{causal_mask.astype(np.int32)}")

# 方法2：使用 np.triu 的上三角
causal_mask2 = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=0)
print(f"因果掩码（triu）:\n{causal_mask2.astype(np.int32)}")

# k=1 表示不含对角线的上三角
causal_mask3 = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
print(f"不含对角线:\n{causal_mask3.astype(np.int32)}")

# 将上三角（不含对角线）设为False，表示屏蔽未来位置
future_mask = ~causal_mask3
print(f"最终因果掩码:\n{future_mask.astype(np.int32)}")
```

因果掩码确保了在预测第 i 个位置时，模型只能看到位置 0 到 i-1 的信息，这是GPT等自回归模型的核心特性。

### 组合Padding Mask和Causal Mask

在实际应用中，通常需要同时使用两种掩码：

```python
batch_size = 2
seq_len = 6

# 模拟实际序列长度
seq_lengths = np.array([5, 3])

# 1. 创建padding mask
padding_mask = np.zeros((batch_size, seq_len), dtype=bool)
for i, length in enumerate(seq_lengths):
    padding_mask[i, :length] = True

# 2. 创建causal mask
causal_mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
causal_mask = ~causal_mask  # True表示可以关注

# 3. 组合两个掩码（AND操作）
combined_mask = padding_mask[:, np.newaxis, :] & causal_mask[np.newaxis, :, :]
print(f"组合掩码形状: {combined_mask.shape}")

# 应用到attention scores
attention_scores = np.random.randn(batch_size, seq_len, seq_len).astype(np.float32)
masked_scores = np.where(combined_mask, attention_scores, -1e9)
print(f"应用掩码后的attention scores")
```

### 注意力掩码的广播

在实际计算中，掩码需要进行广播以匹配attention scores的形状：

```python
batch_size = 4
num_heads = 12
seq_len = 512

# 创建padding mask: (batch_size, seq_len)
padding_mask = np.random.randint(0, 2, size=(batch_size, seq_len)).astype(bool)

# 广播为 (batch_size, 1, 1, seq_len) 以匹配attention scores
mask_4d = padding_mask[:, np.newaxis, np.newaxis, :]

# 模拟attention scores: (batch, num_heads, seq_len, seq_len)
attention_scores = np.random.randn(batch_size, num_heads, seq_len, seq_len).astype(np.float32)

# 应用掩码
masked_scores = np.where(mask_4d, attention_scores, -1e9)
print(f"掩码应用后形状: {masked_scores.shape}")
```

广播是NumPy的一个重要特性，让掩码可以自动扩展到与被掩码数据相同的形状。

## 常见误区与注意事项

### 误区一：使用Python的and/or/not而不是&/|/~

```python
arr = np.array([1, 2, 3, 4, 5])

# 错误：Python的逻辑运算符不能用于numpy数组
try:
    result = arr[(arr > 1) and (arr < 5)]
except ValueError as e:
    print(f"错误: {e}")

# 正确：使用位运算符
result = arr[(arr > 1) & (arr < 5)]
print(f"正确结果: {result}")
```

### 误区二：忘记掩码返回副本

```python
arr = np.array([1, 2, 3, 4, 5])
selected = arr[arr > 2]

# 修改selected不会影响原始数组
selected[0] = 99
print(f"原始数组: {arr}")  # 不会被修改
```

布尔索引返回的是副本，这与切片返回视图不同。

### 误区三：掩码形状不匹配

```python
matrix = np.array([[1, 2, 3], [4, 5, 6]])

# 二维掩码
bool_mask_2d = np.array([[True, False, True], [False, True, False]])
result = matrix[bool_mask_2d]
print(f"二维掩码结果: {result}")  # [1, 3, 5]

# 如果掩码是一维的，会被广播
bool_mask_1d = np.array([True, False, True])
result2 = matrix[bool_mask_1d]  # 按第一维度索引
```

## 小结

布尔索引（掩码）是LLM中最重要的数据筛选工具。理解padding mask和causal mask的原理对于实现各种注意力机制至关重要。`np.where` 是进行条件选择的核心函数，其三参数形式相当于向量化版的 if-else。在LLM场景中，掩码广泛用于：屏蔽padding位置、确保因果注意力、实现各种特殊的注意力模式等。

面试时需要能够解释padding mask和causal mask的区别，理解 np.where 的用法，以及注意组合条件时必须使用位运算符。
