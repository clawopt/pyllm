# 不同形状数组的自动扩展

在上一节中，我们学习了广播的基本规则。在这一节中，让我们更深入地探讨不同形状数组如何通过广播进行自动扩展，以及这种机制在LLM中的实际应用。广播的本质是NumPy会自动识别哪些维度需要扩展，让代码保持简洁的同时不损失性能。理解广播的细节对于写出正确的向量化代码至关重要，因为在实际应用中，形状不匹配是导致bug的主要原因之一。

## 广播的维度扩展过程

当两个数组进行运算时，NumPy会经历以下过程来确定如何扩展数组：

```python
import numpy as np

# 示例 1：一维加二维
a = np.array([1, 2, 3])       # 形状 (3,)
b = np.arange(6).reshape(2, 3)  # 形状 (2, 3)

# 广播过程：
# 1. 比较形状：从右侧开始
#    a: (1, 3)  - NumPy会在前面补1 -> (1, 3)
#    b: (2, 3)
# 2. 扩展后：a 变成 (1, 3)，复制成 (2, 3)
# 3. 相加

result = a + b
print(f"(3,) + (2, 3) = {result.shape}")
print(f"结果:\n{result}")
```

## 维度补1的规则

当两个数组的维度数量不同时，NumPy会在前面补1：

```python
# 标量与数组
scalar = 10
arr = np.array([1, 2, 3])

# scalar 被视为形状 ()
# arr 被视为形状 (3,)
# 广播时 scalar 补1变成 (1, 3)，再复制
result = scalar + arr
print(f"标量 + 数组: {result}")

# 一维与三维
a = np.array([1, 2, 3])                      # (3,)
b = np.random.randn(4, 3, 5).astype(np.float32)  # (4, 3, 5)

# a 补1变成 (1, 1, 3)
# 广播后形状: (4, 3, 5)
result = a + b
print(f"一维 + 三维 = {result.shape}")
```

## 轴的概念与广播

理解"轴"（axis）的概念对于理解广播至关重要。在NumPy中，轴0是最外层的维度：

```python
# 二维数组的轴
matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])

print(f"轴0（行）: {matrix.shape[0]}")
print(f"轴1（列）: {matrix.shape[1]}")
print(f"轴2（如果存在）: 不存在")

# 三维数组的轴
arr_3d = np.random.randn(2, 3, 4)
print(f"三维数组形状: {arr_3d.shape}")
print(f"轴0: {arr_3d.shape[0]}")
print(f"轴1: {arr_3d.shape[1]}")
print(f"轴2: {arr_3d.shape[2]}")
```

## 批量维度与样本维度的广播

在深度学习中，我们处理的数据通常有批量维度和样本维度：

```python
batch_size = 8
seq_len = 512
hidden_dim = 768

# 批量隐藏状态
hidden_states = np.random.randn(batch_size, seq_len, hidden_dim).astype(np.float32)

# 每个样本的缩放因子
sample_scale = np.random.randn(batch_size, 1, 1).astype(np.float32)

# 广播：sample_scale 自动扩展到 hidden_states 的形状
scaled = hidden_states * sample_scale
print(f"缩放后形状: {scaled.shape}")
```

这里 sample_scale 的形状是 (batch_size, 1, 1)，它会在 seq_len 和 hidden_dim 维度上自动扩展。

## 逐样本操作

有时候需要对每个样本应用不同的操作：

```python
batch_size = 4
seq_len = 10
hidden_dim = 768

# 模拟4个样本的隐藏状态
hidden_states = np.random.randn(batch_size, seq_len, hidden_dim).astype(np.float32)

# 每个样本的均值和方差（用于LayerNorm）
means = hidden_states.mean(axis=(1, 2), keepdims=True)  # (batch, 1, 1)
variances = hidden_states.var(axis=(1, 2), keepdims=True)  # (batch, 1, 1)

print(f"均值形状: {means.shape}")
print(f"方差形状: {variances.shape}")

# 归一化时，广播自动处理每个样本
normalized = (hidden_states - means) / np.sqrt(variances + 1e-5)
print(f"归一化后形状: {normalized.shape}")
```

## 跨批量维度的操作

在训练过程中，可能需要跨样本计算某些统计量：

```python
batch_size = 32
num_heads = 12
seq_len = 512
head_dim = 64

# 注意力权重 (batch, heads, seq, seq)
attention_weights = np.random.rand(batch_size, num_heads, seq_len, seq_len).astype(np.float32)
attention_weights = attention_weights / attention_weights.sum(axis=-1, keepdims=True)

# 计算每个样本的平均注意力（跨头和序列）
avg_attention_per_sample = attention_weights.mean(axis=(1, 2, 3))  # (batch,)
print(f"每个样本平均注意力: {avg_attention_per_sample.shape}")
```

## 维度不匹配时的处理

当维度完全不匹配时，需要显式处理：

```python
# 场景：有两个数组，形状分别是 (batch, seq, hidden) 和 (hidden,)
# 需要确保它们可以广播

H = np.random.randn(4, 10, 256).astype(np.float32)  # (batch, seq, hidden)
bias = np.random.randn(256).astype(np.float32)        # (hidden,)

# 直接相加：可以！NumPy 会自动处理
result = H + bias
print(f"H + bias = {result.shape}")

# 但如果我们想按样本添加不同的偏置呢？
per_sample_bias = np.random.randn(4, 1, 256).astype(np.float32)  # (batch, 1, hidden)
result2 = H + per_sample_bias
print(f"H + per_sample_bias = {result2.shape}")
```

## np.newaxis 与广播的配合

`np.newaxis`（或 `None`）可以显式地增加维度来辅助广播：

```python
arr = np.array([1, 2, 3])  # (3,)

# 增加一个维度
arr_2d = arr[:, np.newaxis]  # (3, 1)
arr_2d_v2 = arr[np.newaxis, :]  # (1, 3)

matrix = np.arange(9).reshape(3, 3)

# 现在可以广播相乘
result = matrix * arr_2d  # (3, 3) * (3, 1) -> (3, 3)
print(f"按列缩放:\n{result}")

result2 = matrix * arr_2d_v2  # (3, 3) * (1, 3) -> (3, 3)
print(f"按行缩放:\n{result2}")
```

## 常见广播形状组合

在深度学习中，有一些常见的广播形状组合：

```python
# (batch, seq, hidden) + (hidden,) -> (batch, seq, hidden)
# 偏置相加

# (batch, seq, hidden) + (seq, hidden) -> (batch, seq, hidden)
# 位置相关偏置

# (batch, seq, hidden) + (batch, seq, hidden) -> (batch, seq, hidden)
# 逐样本操作

# (batch, 1, hidden) + (1, seq, hidden) -> (batch, seq, hidden)
# 完全广播

# (batch, heads, seq, seq) * (batch, 1, 1, seq) -> (batch, heads, seq, seq)
# 注意力掩码应用
```

## 性能考虑

广播虽然不复制数据（通过改变 strides 实现），但某些情况下仍然需要注意性能：

```python
# 场景：对大型数组进行多次广播操作

# 如果你知道需要重复使用，考虑预先扩展
large_arr = np.random.randn(1000, 1000, 512).astype(np.float32)
small_arr = np.random.randn(512).astype(np.float32)

# 多次使用时，预扩展可能更快
small_arr_expanded = np.broadcast_to(small_arr, large_arr.shape)

# 然后直接相加
result = large_arr + small_arr_expanded
```

## 小结

广播是NumPy最强大的特性之一，它允许不同形状的数组进行运算而无需显式复制数据。广播规则是从右向左比较维度，兼容条件是相等或为1。理解广播对于写出简洁高效的NumPy代码至关重要，特别是在处理深度学习中常见的多维数组时。

面试时需要能够解释广播规则，理解为什么广播是内存高效的，以及能够分析复杂形状组合是否能够广播。
