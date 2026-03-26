# 点积

点积（Dot Product）是线性代数中最基础也最重要的运算之一，又称为内积（Inner Product）。想象一下，当你计算两个向量之间的相似度时，最直观的方法就是将对应元素相乘后求和，这就是点积的本质。在深度学习中，点积无处不在：计算注意力分数时需要 Query 和 Key 的点积，计算余弦相似度时需要向量的点积，全连接层的计算本质上也是点积的叠加。理解点积的数学原理和 NumPy 实现，是掌握线性代数和深度学习的基础。

## 一维向量的点积

两个一维向量的点积就是对应元素相乘后求和：

```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# 使用 np.dot 计算点积
dot_product = np.dot(a, b)
print(f"向量 a: {a}")
print(f"向量 b: {b}")
print(f"点积: {dot_product}")  # 1*4 + 2*5 + 3*6 = 32
```

点积的数学定义是：a · b = Σ a_i * b_i。

## 二维矩阵的点积

对于二维矩阵，`np.dot` 执行矩阵乘法：

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# np.dot 对于二维数组执行矩阵乘法
C = np.dot(A, B)
print(f"矩阵 A:\n{A}")
print(f"矩阵 B:\n{B}")
print(f"A @ B:\n{C}")
```

注意：矩阵乘法的规则是 (m, n) @ (n, k) = (m, k)。

## @ 运算符与 np.matmul

现代 NumPy 中，更推荐使用 `@` 运算符或 `np.matmul` 来进行矩阵乘法：

```python
# @ 运算符
C1 = A @ B

# np.matmul
C2 = np.matmul(A, B)

# np.dot（对于二维数组，等价于 @）
C3 = np.dot(A, B)

print(f"三种方式结果相同: {np.allclose(C1, C2) and np.allclose(C2, C3)}")
```

## 在LLM场景中的应用

### 注意力分数计算

点积是计算注意力分数的核心操作：

```python
batch_size = 4
num_heads = 12
seq_len = 512
head_dim = 64

# 模拟 Q 和 K（多头注意力的形式）
Q = np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np.float32)
K = np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np.float32)

# 计算注意力分数：Q @ K^T
# 需要将 K 转置最后两个维度
K_transposed = K.transpose(0, 1, 3, 2)  # (batch, heads, head_dim, seq)

# 计算点积
attention_scores = np.matmul(Q, K_transposed)
print(f"注意力分数形状: {attention_scores.shape}")

# 验证：attention_scores[b, h, i, j] = Σ Q[b, h, i, k] * K[b, h, j, k]
# 即位置 i 对位置 j 的注意力分数
```

### 余弦相似度

余弦相似度是衡量两个向量方向的相似程度，取值范围是 [-1, 1]：

```python
def cosine_similarity(a, b):
    """计算余弦相似度

    cos(θ) = (a · b) / (||a|| * ||b||)
    """
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot / (norm_a * norm_b)

# 测试句子嵌入的相似度
embedding1 = np.random.randn(768).astype(np.float32)  # "今天天气真好"
embedding2 = np.random.randn(768).astype(np.float32)  # "今天阳光明媚"
embedding3 = np.random.randn(768).astype(np.float32)  # "机器学习很有趣"

sim_12 = cosine_similarity(embedding1, embedding2)
sim_13 = cosine_similarity(embedding1, embedding3)

print(f"句子1和2的相似度: {sim_12:.4f}")  # 可能较高（都是关于天气）
print(f"句子1和3的相似度: {sim_13:.4f}")  # 可能较低（主题不同）
```

### 词向量相似度

在 NLP 中，点积常用于计算词向量的相似度：

```python
vocab_size = 10000
embed_dim = 300

# 模拟词向量矩阵
word_vectors = np.random.randn(vocab_size, embed_dim).astype(np.float32) * 0.02

# 获取两个词的向量
word1_id = 100
word2_id = 200

vec1 = word_vectors[word1_id]
vec2 = word_vectors[word2_id]

# 直接用点积作为相似度（未归一化）
raw_similarity = np.dot(vec1, vec2)

# 归一化后的相似度（余弦相似度）
normalized_similarity = cosine_similarity(vec1, vec2)

print(f"原始相似度: {raw_similarity:.4f}")
print(f"余弦相似度: {normalized_similarity:.4f}")
```

## 点积的性质

### 交换律

点积满足交换律：a · b = b · a

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print(f"a · b = {np.dot(a, b)}")  # 32
print(f"b · a = {np.dot(b, a)}")  # 32
```

### 分配律

点积满足分配律：(a + b) · c = a · c + b · c

```python
c = np.array([7, 8, 9])

left = np.dot(a + b, c)
right = np.dot(a, c) + np.dot(b, c)

print(f"(a + b) · c = {left}")
print(f"a · c + b · c = {right}")
print(f"分配律成立: {np.isclose(left, right)}")
```

### 与矩阵乘法的关系

点积与矩阵乘法密切相关：矩阵第 i 行与第 j 列的点积，等于矩阵乘法的第 (i, j) 元素。

## 常见误区与注意事项

### 误区一：混淆 np.dot 和 @ 对高维数组的行为

```python
A = np.random.randn(2, 3, 4)
B = np.random.randn(2, 4, 5)

# @ 对最后两个维度执行矩阵乘法
result_matmul = A @ B
print(f"@ 结果形状: {result_matmul.shape}")  # (2, 3, 5)

# np.dot 的行为不同（对于高维数组）
result_dot = np.dot(A, B)
print(f"np.dot 结果形状: {result_dot.shape}")  # 也是 (2, 3, 5)，但计算方式不同
```

### 误区二：点积要求维度匹配

```python
a = np.array([1, 2, 3])
b = np.array([4, 5])  # 长度不同！

try:
    np.dot(a, b)
except ValueError as e:
    print(f"维度不匹配错误: {e}")
```

## 小结

点积是线性代数中最基础的运算之一，是计算注意力分数和余弦相似度的核心操作。在 LLM 场景中，点积用于：计算注意力分数（Q · K^T）、计算句子/词向量的相似度、以及全连接层的计算。理解点积的数学原理和 NumPy 实现，对于掌握深度学习至关重要。

面试时需要能够解释点积的数学定义，理解点积与矩阵乘法的关系，以及能够描述点积在注意力机制中的应用。
