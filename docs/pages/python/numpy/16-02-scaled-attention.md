# 缩放点积注意力

缩放点积注意力（Scaled Dot-Product Attention）是 Transformer 的核心机制。给定 Query、Key、Value 三个矩阵，注意力通过计算 Q 与 K 的点积得到注意力分数，然后使用 softmax 归一化，最后乘以 V 得到输出。本篇文章详细介绍缩放点积注意力的原理和纯 NumPy 实现，包括多头注意力的实现。

## 注意力机制的基本原理

Attention(Q, K, V) = softmax(QK^T / √d_k) × V

其中 d_k 是 Key 向量的维度。缩放因子 √d_k 用于防止点积值过大导致 softmax 梯度消失。

```python
import numpy as np

def softmax(x, axis=-1):
    """Softmax 函数

    softmax(x_i) = exp(x_i) / sum(exp(x_j))
    """
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def scaled_dot_product_attention(Q, K, V, mask=None):
    """缩放点积注意力

    参数:
        Q: Query 矩阵 (..., seq_len, d_k)
        K: Key 矩阵 (..., seq_len, d_k)
        V: Value 矩阵 (..., seq_len, d_v)
        mask: 掩码 (..., seq_len, seq_len)，True 表示需要 mask
    返回:
        output: 注意力输出 (..., seq_len, d_v)
        attention_weights: 注意力权重 (..., seq_len, seq_len)
    """
    d_k = Q.shape[-1]

    # 计算 QK^T
    scores = np.einsum('...nd,...md->...nm', Q, K)

    # 缩放
    scores = scores / np.sqrt(d_k)

    # 应用掩码（如果提供）
    if mask is not None:
        scores = np.where(mask, -1e9, scores)

    # Softmax 归一化
    attention_weights = softmax(scores, axis=-1)

    # 乘以 V
    output = np.einsum('...nm,...md->...nd', attention_weights, V)

    return output, attention_weights

# 示例
np.random.seed(42)
seq_len = 5
d_k = 64

Q = np.random.randn(seq_len, d_k).astype(np.float32)
K = np.random.randn(seq_len, d_k).astype(np.float32)
V = np.random.randn(seq_len, d_k).astype(np.float32)

output, attention_weights = scaled_dot_product_attention(Q, K, V)
print(f"Q 形状: {Q.shape}")
print(f"K 形状: {K.shape}")
print(f"V 形状: {V.shape}")
print(f"输出形状: {output.shape}")
print(f"注意力权重形状: {attention_weights.shape}")
```

## 多头注意力的实现

```python
class MultiHeadAttention:
    """多头注意力机制"""

    def __init__(self, hidden_size, num_heads):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        assert hidden_size % num_heads == 0

        # QKV 投影权重
        self.W_q = np.random.randn(hidden_size, hidden_size).astype(np.float32) * 0.02
        self.W_k = np.random.randn(hidden_size, hidden_size).astype(np.float32) * 0.02
        self.W_v = np.random.randn(hidden_size, hidden_size).astype(np.float32) * 0.02

        # 输出投影权重
        self.W_o = np.random.randn(hidden_size, hidden_size).astype(np.float32) * 0.02

    def split_heads(self, X, batch_size):
        """将 hidden_dim 分割为 num_heads 个头

        X: (batch, seq_len, hidden_size)
        返回: (batch, num_heads, seq_len, head_dim)
        """
        X = X.reshape(batch_size, -1, self.num_heads, self.head_dim)
        return X.transpose(0, 2, 1, 3)

    def forward(self, Q, K, V, mask=None):
        """前向传播

        参数:
            Q, K, V: (batch_size, seq_len, hidden_size)
            mask: (batch_size, seq_len, seq_len)
        返回:
            output: (batch_size, seq_len, hidden_size)
            attention_weights: (batch_size, num_heads, seq_len, seq_len)
        """
        batch_size = Q.shape[0]
        seq_len = Q.shape[1]

        # QKV 投影
        Q = Q @ self.W_q  # (batch, seq, hidden)
        K = K @ self.W_k
        V = V @ self.W_v

        # 分割为多头
        Q = self.split_heads(Q, batch_size)  # (batch, heads, seq, head_dim)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)

        # 计算注意力
        d_k = self.head_dim
        scores = np.einsum('bhnd,bhmd->bhnm', Q, K)  # (batch, heads, seq, seq)
        scores = scores / np.sqrt(d_k)

        # 应用掩码
        if mask is not None:
            # 调整掩码维度以匹配
            mask = mask[:, np.newaxis, :, :]  # (batch, 1, seq, seq)
            scores = np.where(mask, -1e9, scores)

        # Softmax
        attention_weights = softmax(scores, axis=-1)

        # 乘以 V
        context = np.einsum('bhnm,bhmd->bhnd', attention_weights, V)  # (batch, heads, seq, head_dim)

        # 合并多头
        context = context.transpose(0, 2, 1, 3)  # (batch, seq, heads, head_dim)
        context = context.reshape(batch_size, seq_len, self.hidden_size)  # (batch, seq, hidden)

        # 输出投影
        output = context @ self.W_o

        return output, attention_weights

# 示例
np.random.seed(42)
batch_size = 2
seq_len = 10
hidden_size = 768
num_heads = 12

mha = MultiHeadAttention(hidden_size, num_heads)
X = np.random.randn(batch_size, seq_len, hidden_size).astype(np.float32)

output, attention_weights = mha.forward(X, X, X)
print(f"输入形状: {X.shape}")
print(f"输出形状: {output.shape}")
print(f"注意力权重形状: {attention_weights.shape}")
```

## 简化版单头注意力

为了便于理解，这里提供简化版的实现：

```python
class SimpleAttention:
    """简化的注意力机制（单头）"""

    def __init__(self, d_model):
        self.d_model = d_model

        # 简化的 QKV 投影
        self.W_q = np.random.randn(d_model, d_model).astype(np.float32) * 0.02
        self.W_k = np.random.randn(d_model, d_model).astype(np.float32) * 0.02
        self.W_v = np.random.randn(d_model, d_model).astype(np.float32) * 0.02

    def forward(self, X, mask=None):
        """前向传播

        X: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = X.shape

        # QKV 投影
        Q = X @ self.W_q
        K = X @ self.W_k
        V = X @ self.W_v

        # 计算注意力分数
        scores = Q @ K.transpose(0, 2, 1)  # (batch, seq, seq)
        scores = scores / np.sqrt(self.d_model)

        # 应用掩码
        if mask is not None:
            scores = np.where(mask, -1e9, scores)

        # Softmax
        attention_weights = softmax(scores, axis=-1)

        # 加权求和
        output = attention_weights @ V

        return output, attention_weights

# 示例
simple_attn = SimpleAttention(d_model=512)
X_simple = np.random.randn(2, 10, 512).astype(np.float32)
output, weights = simple_attn.forward(X_simple)
print(f"\n简化注意力:")
print(f"输入形状: {X_simple.shape}")
print(f"输出形状: {output.shape}")
print(f"注意力权重形状: {weights.shape}")
```

## 注意力权重的可视化

注意力权重展示了每个位置对其他位置的"关注程度"：

```python
def visualize_attention(attention_weights, tokens, head_idx=0):
    """可视化注意力权重

    参数:
        attention_weights: (batch, heads, seq, seq) 或 (batch, seq, seq)
        tokens: token 列表
        head_idx: 要可视化的头索引
    """
    if len(attention_weights.shape) == 4:
        # 多头注意力，选择指定的头
        weights = attention_weights[0, head_idx]  # (seq, seq)
    else:
        weights = attention_weights[0]

    seq_len = len(tokens)
    print(f"\n=== 注意力权重可视化 (Head {head_idx}) ===\n")

    # 打印表头
    header = "         " + "".join([f"{t:>8}" for t in tokens[:seq_len]])
    print(header)
    print("-" * len(header))

    # 打印矩阵
    for i, token in enumerate(tokens[:seq_len]):
        row = f"{token:>8} |"
        for j in range(seq_len):
            w = weights[i, j]
            char = '*' if w > 0.5 else ('+' if w > 0.2 else ('.' if w > 0.05 else ' '))
            row += f"{char:>8}"
        print(row)

# 示例
simple_attn = SimpleAttention(d_model=64)
tokens = ['[CLS]', 'The', 'cat', 'eats', 'the', 'fish', '[SEP]']
X_demo = np.random.randn(1, len(tokens), 64).astype(np.float32)
_, attention_weights = simple_attn.forward(X_demo)

# 找出注意力最集中的位置
print(f"注意力权重范围: [{attention_weights.min():.3f}, {attention_weights.max():.3f}")
print(f"每行注意力之和（应为1）: {attention_weights[0].sum(axis=1)}")
```

## 注意力机制的特性

### 自注意力（Self-Attention）

自注意力是指 Q、K、V 都来自同一个输入序列：

```python
def self_attention(X, mask=None):
    """自注意力"""
    return scaled_dot_product_attention(X, X, X, mask)

# 示例
X = np.random.randn(2, 5, 64).astype(np.float32)
output, _ = self_attention(X)
print(f"自注意力输出形状: {output.shape}")
```

### 交叉注意力（Cross-Attention）

交叉注意力是指 Q 来自一个序列，K、V 来自另一个序列：

```python
def cross_attention(Q, K, V, mask=None):
    """交叉注意力

    例如：解码器关注编码器的输出
    Q: 解码器隐藏状态
    K, V: 编码器隐藏状态
    """
    return scaled_dot_product_attention(Q, K, V, mask)

# 示例
encoder_output = np.random.randn(2, 10, 64).astype(np.float32)
decoder_hidden = np.random.randn(2, 5, 64).astype(np.float32)
output, _ = cross_attention(decoder_hidden, encoder_output, encoder_output)
print(f"交叉注意力输出形状: {output.shape}")
```

## 数值稳定性问题

softmax 在输入值很大时可能溢出：

```python
def stable_attention(Q, K, V):
    """数值稳定的注意力实现

    使用 log-sum-exp 技巧
    """
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(0, 2, 1)
    scores = scores / np.sqrt(d_k)

    # 减去最大值提高数值稳定性
    scores_max = scores.max(axis=-1, keepdims=True)
    scores = scores - scores_max

    exp_scores = np.exp(scores)
    attention_weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)

    return attention_weights @ V

# 测试数值稳定性
Q = np.random.randn(2, 10, 64).astype(np.float32) * 10  # 大值
K = np.random.randn(2, 10, 64).astype(np.float32) * 10
V = np.random.randn(2, 10, 64).astype(np.float32)

output = stable_attention(Q, K, V)
print(f"数值稳定注意力输出形状: {output.shape}")
print(f"输出值范围: [{output.min():.4f}, {output.max():.4f}]")
```

## 常见误区

**误区一：忘记缩放因子**

不缩放会导致 softmax 梯度消失：

```python
# 错误：忘记缩放
scores = Q @ K.transpose(0, 2, 1)  # d_k 大时，scores 可能很大
# softmax 会变成接近 one-hot

# 正确：缩放
scores = scores / np.sqrt(d_k)
```

**误区二：混淆注意力权重的维度**

注意力权重的形状是 (batch, heads, seq, seq) 或 (batch, seq, seq)：

```python
# 多头注意力
# attention_weights: (batch, heads, query_seq, key_seq)

# 自注意力时，query_seq == key_seq == seq_len
# 但维度不能混淆
```

**误区三：掩码使用不当**

掩码应该将需要隐藏的位置设为 True 或 -inf：

```python
# 正确的掩码方式
mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)  # 上三角为 True
scores = np.where(mask, -1e9, scores)
```

## API 总结

| 函数/类 | 描述 |
|--------|------|
| `softmax(x, axis)` | Softmax 函数 |
| `scaled_dot_product_attention(Q, K, V)` | 缩放点积注意力 |
| `MultiHeadAttention` | 多头注意力 |
| `SimpleAttention` | 简化版单头注意力 |

缩放点积注意力是 Transformer 的核心。理解其原理和实现，对于掌握现代语言模型至关重要。
