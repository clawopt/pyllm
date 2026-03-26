# 矩阵乘法

矩阵乘法是线性代数中最核心的运算，也是深度学习的数学基础。想象一下，当你输入一个token的embedding到神经网络的第一层时，它会与权重矩阵相乘得到新的表示；当你在计算自注意力时，需要将Query和Key相乘来得到注意力分数。几乎神经网络中的每一个计算步骤都离不开矩阵乘法。与逐元素运算不同，矩阵乘法有其独特的数学规则：两个矩阵相乘，左边矩阵的每一行与右边矩阵的每一列进行点积运算。NumPy使用 `@` 运算符或 `np.matmul` 函数来进行矩阵乘法，这种运算在深度学习中无处不在，其计算复杂度通常是 O(n³)，但通过各种优化技术，实际运行效率可以非常高。

## 基本矩阵乘法

对于二维矩阵，使用 `@` 运算符或 `np.matmul`：

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# 使用 @ 运算符
C = A @ B
print(f"A @ B:\n{C}")

# 使用 np.matmul
D = np.matmul(A, B)
print(f"np.matmul(A, B):\n{D}")

# 使用 np.dot（对于二维数组，与 @ 等价）
E = np.dot(A, B)
print(f"np.dot(A, B):\n{E}")
```

对于二维矩阵，`@`、`np.matmul` 和 `np.dot` 在大多数情况下是等价的。

## 矩阵乘法的数学原理

如果 C = A @ B，则 C[i,j] = Σ A[i,k] * B[k,j]。用代码表示就是：

```python
# 手动验证矩阵乘法的定义
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# 计算 C[0, 0]
c00 = A[0, 0] * B[0, 0] + A[0, 1] * B[1, 0]  # 1*5 + 2*7 = 19
# 计算 C[0, 1]
c01 = A[0, 0] * B[0, 1] + A[0, 1] * B[1, 1]  # 1*6 + 2*8 = 22
# 计算 C[1, 0]
c10 = A[1, 0] * B[0, 0] + A[1, 1] * B[1, 0]  # 3*5 + 4*7 = 43
# 计算 C[1, 1]
c11 = A[1, 0] * B[0, 1] + A[1, 1] * B[1, 1]  # 3*6 + 4*8 = 50

print(f"手动计算 C = [[{c00}, {c01}], [{c10}, {c11}]]")
print(f"A @ B = \n{A @ B}")
```

## 批量矩阵乘法

在深度学习中，我们通常处理批量数据，所以更多时候使用的是批量矩阵乘法。NumPy的广播机制会自动处理批量维度：

```python
# 批量矩阵乘法示例
batch_size = 4
seq_len = 10
input_dim = 768
output_dim = 3072

# 输入: (batch, seq_len, input_dim)
X = np.random.randn(batch_size, seq_len, input_dim).astype(np.float32)

# 权重: (output_dim, input_dim)
W = np.random.randn(output_dim, input_dim).astype(np.float32) * np.sqrt(2.0 / input_dim)

# 线性变换: Y = X @ W.T -> (batch, seq_len, output_dim)
Y = X @ W.T
print(f"批量矩阵乘法输出形状: {Y.shape}")
```

批量矩阵乘法将前面的维度视为"批量"，只对最后两个维度执行矩阵乘法。

## np.matmul vs np.dot

面试中经常被问到："np.matmul 和 np.dot 有什么区别？"对于二维数组，它们基本等价；但对于更高维度的数组，行为有所不同：

```python
# 二维矩阵乘法：等价
A = np.random.randn(3, 4)
B = np.random.randn(4, 5)
print(f"matmul: {np.matmul(A, B).shape}")  # (3, 5)
print(f"dot: {np.dot(A, B).shape}")  # (3, 5)

# 三维数组：行为不同
A = np.random.randn(2, 3, 4)
B = np.random.randn(2, 4, 5)

# np.matmul 将最后两个维度视为矩阵，前面的维度视为批量
print(f"matmul: {np.matmul(A, B).shape}")  # (2, 3, 5)

# np.dot 是广义的矩阵乘法，会对第一个数组的最后一个维度和第二个数组的倒数第二个维度进行乘法
print(f"dot: {np.dot(A, B).shape}")  # (2, 3, 5)
```

对于三维数组，`np.matmul` 和 `np.dot` 在大多数情况下结果相同，但 `np.matmul` 的语义更清晰——它明确地将批量维度放在前面，最后两个维度是矩阵。

## 在LLM场景中的应用

### 自注意力中的 Q·K^T 计算

自注意力的第一步是计算Query和Key的点积，这是矩阵乘法的典型应用：

```python
batch_size = 4
seq_len = 512
hidden_dim = 768
num_heads = 12
head_dim = hidden_dim // num_heads  # 64

# 模拟隐藏状态 (batch, seq_len, hidden_dim)
hidden_states = np.random.randn(batch_size, seq_len, hidden_dim).astype(np.float32)

# Q, K, V 的投影权重
W_q = np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * np.sqrt(2.0 / hidden_dim)
W_k = np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * np.sqrt(2.0 / hidden_dim)
W_v = np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * np.sqrt(2.0 / hidden_dim)

# 计算 Q, K, V
Q = hidden_states @ W_q.T  # (batch, seq_len, hidden_dim)
K = hidden_states @ W_k.T
V = hidden_states @ W_v.T
print(f"Q 形状: {Q.shape}")

# Reshape 为多头形式 (batch, num_heads, seq_len, head_dim)
Q = Q.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
K = K.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
V = V.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
print(f"多头 Q 形状: {Q.shape}")

# 计算注意力分数: Q @ K^T
# Q: (batch, heads, seq, head_dim)
# K^T: (batch, heads, head_dim, seq)
# 结果: (batch, heads, seq, seq)
attention_scores = Q @ K.transpose(0, 1, 3, 2)
print(f"注意力分数形状: {attention_scores.shape}")
```

### Softmax 归一化

得到注意力分数后，需要用 softmax 归一化：

```python
def softmax(x):
    exp_x = np.exp(x - x.max(axis=-1, keepdims=True))
    return exp_x / exp_x.sum(axis=-1, keepdims=True)

# 对注意力分数应用 softmax
attention_weights = softmax(attention_scores)
print(f"注意力权重形状: {attention_weights.shape}")

# 验证每行和为1
print(f"每行和应为1: {attention_weights[0, 0, 0, :].sum():.6f}")
```

### 应用注意力到 V

最后，用归一化后的注意力权重对 V 进行加权求和：

```python
# attention_weights: (batch, heads, seq, seq)
# V: (batch, heads, seq, head_dim)
# 结果: (batch, heads, seq, head_dim)

context = attention_weights @ V
print(f"上下文向量形状: {context.shape}")

# 转置回原始格式 (batch, seq, heads, head_dim)
context = context.transpose(0, 2, 1, 3)
print(f"转置后形状: {context.shape}")

# 合并多头 (batch, seq, hidden_dim)
context = context.reshape(batch_size, seq_len, hidden_dim)
print(f"合并多头后形状: {context.shape}")
```

## 常见误区与注意事项

### 误区一：混淆逐元素乘法和矩阵乘法

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# 逐元素乘法
print(f"A * B:\n{A * B}")

# 矩阵乘法
print(f"A @ B:\n{A @ B}")
```

两者完全不同！

### 误区二：维度不匹配

```python
A = np.random.randn(3, 4)
B = np.random.randn(5, 6)

try:
    C = A @ B
except ValueError as e:
    print(f"维度不匹配: {e}")
```

矩阵乘法的规则是：(m, n) @ (n, k) = (m, k)。

### 误区三：@ 和 np.dot 对高维数组行为不同

```python
A = np.random.randn(2, 3, 4)
B = np.random.randn(2, 4, 5)

# np.matmul 正确处理批量维度
print(f"matmul: {(np.matmul(A, B)).shape}")  # (2, 3, 5)

# np.dot 在某些情况下行为不同
print(f"dot: {(np.dot(A, B)).shape}")  # (2, 3, 5)
```

对于三维数组，通常推荐使用 `np.matmul` 或 `@`。

## 小结

矩阵乘法是深度学习最核心的运算。`@` 运算符和 `np.matmul` 函数是进行矩阵乘法的标准方式，`np.dot` 在二维情况下等价但对高维数组行为不同。在LLM的自注意力机制中，矩阵乘法用于计算 Q·K^T 和应用注意力到 V。理解矩阵乘法的维度规则和批量处理是实现深度学习算法的基础。

面试时需要能够解释矩阵乘法的维度规则，理解 np.matmul 和 np.dot 的区别，以及能够描述自注意力中矩阵乘法的完整流程。
