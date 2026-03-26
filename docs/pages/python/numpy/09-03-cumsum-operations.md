# 累积运算

累积运算是指从数组的第一个元素开始，逐步计算累积和、累积积等。在 NumPy 中，`cumsum` 和 `cumprod` 是最常用的累积函数。累积运算在深度学习中有一些有趣的应用：计算累积注意力（用于相对位置编码）、Prefix Sum 算法（高效计算滑动窗口）、以及某些特殊的损失函数。理解累积运算的原理和应用场景，对于拓宽 NumPy 的使用技巧非常有帮助。

## cumsum：累积和

`cumsum` 计算累积和，即从第一个元素到当前元素的累加结果：

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5])

print(f"原始数组: {arr}")
print(f"累积和: {arr.cumsum()}")
```

累积和的数学定义是：`cumsum[i] = Σ arr[0:i+1]`。

## cumprod：累积积

`cumprod` 计算累积积，即从第一个元素到当前元素的累积乘积：

```python
print(f"累积积: {arr.cumprod()}")  # [1, 2, 6, 24, 120]
```

## 沿轴累积

累积运算也可以沿指定轴进行：

```python
matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])

# 沿 axis=0 累积（按列）
print(f"沿 axis=0 累积和:\n{matrix.cumsum(axis=0)}")
# [[1, 2, 3],
#  [5, 7, 9]]

# 沿 axis=1 累积（按行）
print(f"沿 axis=1 累积和:\n{matrix.cumsum(axis=1)}")
# [[1, 3, 6],
#  [4, 9, 15]]
```

## 在LLM场景中的应用

### 高效计算滑动窗口

累积和可以用来高效计算滑动窗口的累加值：

```python
def moving_sum(arr, window_size):
    """使用累积和高效计算滑动窗口的累加值"""
    cumsum = np.cumsum(np.insert(arr, 0, 0))  # 前面补0
    return (cumsum[window_size:] - cumsum[:-window_size])

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
window = 3

# 计算窗口为3的滑动累加和
moving = moving_sum(arr, window)
print(f"滑动窗口({window})累加和: {moving}")
```

这个算法的复杂度是 O(n)，比朴素的 O(n*window) 快得多。

### 累积注意力（Prefix Sum Attention）

在某些注意力机制的变体中，需要计算累积注意力：

```python
def prefix_sum_attention(attention_scores):
    """Prefix Sum 注意力

    用于计算累积注意力权重
    """
    batch_size, num_heads, seq_len, _ = attention_scores.shape

    # 计算累积和
    prefix_attn = np.cumsum(attention_scores, axis=-1)

    # 归一化
    lengths = np.arange(1, seq_len + 1)
    normalized = prefix_attn / lengths[np.newaxis, np.newaxis, :, np.newaxis]

    return normalized

# 测试
batch_size = 2
num_heads = 4
seq_len = 8

attention_scores = np.random.rand(batch_size, num_heads, seq_len, seq_len).astype(np.float32)
prefix_attn = prefix_sum_attention(attention_scores)
print(f"累积注意力形状: {prefix_attn.shape}")
```

### 相对位置编码中的累积

在某些相对位置编码的实现中，累积运算用于生成累积的位置信息：

```python
def relative_position_encoding(seq_len, head_dim, max_rel_pos=10):
    """生成相对位置编码

    使用累积运算生成位置关系
    """
    # 生成相对位置
    positions = np.arange(seq_len)[:, np.newaxis] - np.arange(seq_len)[np.newaxis, :]

    # 裁剪到合理范围
    positions = np.clip(positions, -max_rel_pos, max_rel_pos)

    # 计算累积的嵌入（模拟）
    div_term = np.exp(np.arange(0, head_dim, 2) * -(np.log(10000.0) / head_dim))
    pe = np.zeros((seq_len, seq_len, head_dim))

    for i in range(seq_len):
        for j in range(seq_len):
            rel_pos = positions[i, j]
            offset = rel_pos + max_rel_pos
            pe[i, j, 0::2] = np.sin(offset * div_term)
            pe[i, j, 1::2] = np.cos(offset * div_term)

    return pe

seq_len = 8
head_dim = 16
rel_pe = relative_position_encoding(seq_len, head_dim)
print(f"相对位置编码形状: {rel_pe.shape}")
```

## 与普通求和的区别

累积和与普通求和的结果形状不同：

```python
arr = np.array([1, 2, 3, 4, 5])

print(f"普通求和: {arr.sum()}")      # 15 (标量)
print(f"累积和: {arr.cumsum()}")     # [1, 3, 6, 10, 15] (数组)
```

普通求和返回所有元素的和，累积和返回累积的中间结果。

## 常见误区与注意事项

### 误区一：混淆 cumsum 和 sum

```python
arr = np.array([1, 2, 3, 4, 5])

# sum 返回标量
print(f"sum: {arr.sum()}")          # 15

# cumsum 返回数组
print(f"cumsum: {arr.cumsum()}")    # [1, 3, 6, 10, 15]
```

### 误区二：cumsum 不返回副本

`cumsum` 返回一个新数组，不共享原始数组的内存：

```python
arr = np.array([1, 2, 3, 4, 5])
cumsum_result = arr.cumsum()

print(f"共享内存: {np.shares_memory(arr, cumsum_result)}")  # False
```

### 误区三：cumsum 不保持原始数组的 dtype

```python
arr_int = np.array([1, 2, 3, 4, 5], dtype=np.int32)
cumsum_float = arr_int.cumsum()

print(f"原始 dtype: {arr_int.dtype}")     # int32
print(f"累积和 dtype: {cumsum_float.dtype}")  # int32（不会自动转为 float）
```

## 小结

累积运算（cumsum、cumprod）是 NumPy 中用于计算累积和、累积积的函数。在 LLM 场景中，累积运算可以用于高效计算滑动窗口、实现前缀和注意力，以及相对位置编码等。掌握累积运算可以让代码更高效。

面试时需要能够解释 cumsum 和 sum 的区别，以及能够描述如何使用 cumsum 高效计算滑动窗口。
