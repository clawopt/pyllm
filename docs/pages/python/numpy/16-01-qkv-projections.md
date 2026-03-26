# QKV投影矩阵

在 Transformer 的自注意力机制中，Query、Key、Value（简称 QKV）投影是核心操作。输入的隐藏状态通过三个独立的线性变换得到 Q、K、V 三个矩阵，然后用于计算注意力分数。本篇文章介绍如何使用纯 NumPy 实现 QKV 投影，包括多头注意力中的并行 QKV 计算，帮助你深入理解 Transformer 的工作原理。

## QKV 投影的基本概念

在自注意力中：
- **Query (Q)**：当前位置的"查询"，询问"我应该关注什么"
- **Key (K)**：每个位置的"键"，提供可以被查询匹配的信息
- **Value (V)**：每个位置的"值"，当查询匹配到键时返回的信息

QKV 通过输入 X 与三个权重矩阵 Wq、Wk、Wv 相乘得到：

```
Q = X @ Wq
K = X @ Wk
V = X @ Wv
```

```python
import numpy as np

def he_initialization(fan_in, fan_out):
    """He 初始化（适用于 ReLU）"""
    std = np.sqrt(2.0 / fan_in)
    return np.random.randn(fan_in, fan_out).astype(np.float32) * std

def xavier_initialization(fan_in, fan_out):
    """Xavier 初始化（适用于 softmax/tanh）"""
    std = np.sqrt(2.0 / (fan_in + fan_out))
    return np.random.randn(fan_in, fan_out).astype(np.float32) * std
```

## 基本的 QKV 投影实现

```python
class QKVProjection:
    """QKV 投影层"""

    def __init__(self, hidden_size, num_heads):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        # QKV 权重矩阵
        self.W_q = xavier_initialization(hidden_size, hidden_size)
        self.W_k = xavier_initialization(hidden_size, hidden_size)
        self.W_v = xavier_initialization(hidden_size, hidden_size)

        # QKV 偏置
        self.b_q = np.zeros(hidden_size, dtype=np.float32)
        self.b_k = np.zeros(hidden_size, dtype=np.float32)
        self.b_v = np.zeros(hidden_size, dtype=np.float32)

    def forward(self, X):
        """前向传播

        参数:
            X: 输入张量 (batch_size, seq_len, hidden_size)
        返回:
            Q, K, V: (batch_size, seq_len, hidden_size)
        """
        # 线性投影
        Q = np.einsum('bsd,hd->bsh', X, self.W_q) + self.b_q
        K = np.einsum('bsd,hd->bsh', X, self.W_k) + self.b_k
        V = np.einsum('bsd,hd->bsh', X, self.W_v) + self.b_v

        return Q, K, V

# 示例
np.random.seed(42)
batch_size = 2
seq_len = 10
hidden_size = 768
num_heads = 12

qkv = QKVProjection(hidden_size, num_heads)
X = np.random.randn(batch_size, seq_len, hidden_size).astype(np.float32)

Q, K, V = qkv.forward(X)
print(f"输入 X 形状: {X.shape}")
print(f"Q 形状: {Q.shape}")
print(f"K 形状: {K.shape}")
print(f"V 形状: {V.shape}")
```

## 多头注意力的 QKV 投影

在多头注意力中，QKV 被分割成多个头，每个头独立处理：

```python
class MultiHeadQKVProjection:
    """多头 QKV 投影

    将 hidden_size 分割为 num_heads 个头
    """

    def __init__(self, hidden_size, num_heads):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        assert hidden_size % num_heads == 0

        # 投影到 QKV（每个的维度都是 hidden_size）
        self.W_q = xavier_initialization(hidden_size, hidden_size)
        self.W_k = xavier_initialization(hidden_size, hidden_size)
        self.W_v = xavier_initialization(hidden_size, hidden_size)

        self.b_q = np.zeros(hidden_size, dtype=np.float32)
        self.b_k = np.zeros(hidden_size, dtype=np.float32)
        self.b_v = np.zeros(hidden_size, dtype=np.float32)

    def forward(self, X):
        """前向传播

        参数:
            X: (batch_size, seq_len, hidden_size)
        返回:
            Q: (batch_size, num_heads, seq_len, head_dim)
            K: (batch_size, num_heads, seq_len, head_dim)
            V: (batch_size, num_heads, seq_len, head_dim)
        """
        batch_size, seq_len, _ = X.shape

        # 线性投影
        Q = np.einsum('bsd,hd->bsh', X, self.W_q) + self.b_q
        K = np.einsum('bsd,hd->bsh', X, self.W_k) + self.b_k
        V = np.einsum('bsd,hd->bsh', X, self.W_v) + self.b_v

        # 重塑为多头形式 (batch_size, seq_len, num_heads, head_dim)
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        # 调整维度顺序 (batch_size, num_heads, seq_len, head_dim)
        Q = Q.transpose(0, 2, 1, 3)
        K = K.transpose(0, 2, 1, 3)
        V = V.transpose(0, 2, 1, 3)

        return Q, K, V

# 示例
qkv = MultiHeadQKVProjection(hidden_size, num_heads)
Q, K, V = qkv.forward(X)
print(f"\n多头 QKV 投影:")
print(f"Q 形状: {Q.shape}")  # (batch, num_heads, seq_len, head_dim)
print(f"K 形状: {K.shape}")
print(f"V 形状: {V.shape}")
```

## 简化的单头 QKV（便于理解）

```python
class SimpleQKV:
    """简化的 QKV 投影（单头，无偏置）"""

    def __init__(self, input_dim, qkv_dim):
        self.input_dim = input_dim
        self.qkv_dim = qkv_dim

        # 随机初始化
        self.W_q = np.random.randn(input_dim, qkv_dim).astype(np.float32) * 0.02
        self.W_k = np.random.randn(input_dim, qkv_dim).astype(np.float32) * 0.02
        self.W_v = np.random.randn(input_dim, qkv_dim).astype(np.float32) * 0.02

    def forward(self, X):
        """前向传播

        X: (batch_size, seq_len, input_dim)
        返回:
            Q, K, V: (batch_size, seq_len, qkv_dim)
        """
        Q = np.einsum('bsd,di->bsi', X, self.W_q)
        K = np.einsum('bsd,di->bsi', X, self.W_k)
        V = np.einsum('bsd,di->bsi', X, self.W_v)

        return Q, K, V

# 示例
simple_qkv = SimpleQKV(input_dim=512, qkv_dim=512)
X_simple = np.random.randn(1, 4, 512).astype(np.float32)
Q, K, V = simple_qkv.forward(X_simple)
print(f"\n简化 QKV:")
print(f"输入形状: {X_simple.shape}")
print(f"Q 形状: {Q.shape}")
```

## QKV 投影的物理意义

### Query 的作用

Query 代表当前位置的"查询请求"。例如，在翻译任务中，当翻译"the cat"时，Query 会询问"我应该关注什么词来理解主语"。

```python
def demonstrate_query_concept():
    """演示 Query 的概念"""
    # 假设我们有一个简单的序列
    # 每个位置的 Query 会与其他位置的 Key 计算相似度

    # 简化的语义示例
    words = ['The', 'cat', 'eats', 'the', 'fish']

    # 假设这些是每个词的向量表示（实际来自神经网络）
    # Query: "我想找主语"
    # Key: "我能提供主语信息"

    print("Query-Key 匹配概念:")
    print("当处理 'eats' 时，它的 Query 会寻找匹配的 Key")
    print("'cat' 的 Key 与 'eats' 的 Query 高度匹配")
    print("所以 attention 会集中在 'cat' 上")

demonstrate_query_concept()
```

## QKV 投影的实现细节

### 使用 einsum 提高可读性

`np.einsum` 可以更清晰地表达矩阵乘法：

```python
# 传统的矩阵乘法
Q1 = np.matmul(X, W_q)  # (batch, seq, hidden) @ (hidden, hidden)

# einsum 版本
Q2 = np.einsum('bsd,hd->bsh', X, W_q)  # 更清晰

# 两者等价
print(f"Matmul vs Einsum: {np.allclose(Q1, Q2)}")
```

### 批量处理

QKV 投影天然支持批量处理：

```python
class BatchQKVProjection:
    """支持批量的 QKV 投影"""

    def __init__(self, hidden_size, num_heads):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.W_q = xavier_initialization(hidden_size, hidden_size)
        self.W_k = xavier_initialization(hidden_size, hidden_size)
        self.W_v = xavier_initialization(hidden_size, hidden_size)

    def forward(self, X):
        """X: (batch_size, seq_len, hidden_size)"""
        batch_size = X.shape[0]

        Q = X @ self.W_q
        K = X @ self.W_k
        V = X @ self.W_v

        Q = Q.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        return Q, K, V

# 测试批量
batch_qkv = BatchQKVProjection(768, 12)
X_batch = np.random.randn(8, 512, 768).astype(np.float32)
Q, K, V = batch_qkv.forward(X_batch)
print(f"\n批量 QKV (batch=8, seq=512):")
print(f"Q 形状: {Q.shape}")
```

## 常见误区

**误区一：混淆 QKV 维度**

QKV 的维度应该等于 hidden_size（或在多头注意力中等于 num_heads * head_dim）：

```python
# 错误：QKV 维度不匹配
# X: (batch, seq, 512)
# W_q: (256, 512)  # 错误！输出维度应该是 512
# Q = X @ W_q  # 形状变成 (batch, seq, 256)

# 正确：QKV 权重形状是 (hidden_size, hidden_size)
W_q_correct = np.random.randn(512, 512)
Q = X @ W_q_correct  # (batch, seq, 512)
```

**误区二：多头注意力时忘记重塑**

多头注意力需要正确地分割和重塑张量：

```python
# 假设 hidden_size=768, num_heads=12, head_dim=64
# Q: (batch, seq, hidden) = (2, 10, 768)

# 错误：直接分割
# Q_split = Q.reshape(2, 10, 12, 64)  # 维度顺序错误

# 正确：先移动 seq_len 到正确位置
# Q: (batch, seq, hidden) -> (batch, num_heads, seq, head_dim)
Q = Q.reshape(batch, seq_len, num_heads, head_dim)  # (2, 10, 12, 64)
Q = Q.transpose(0, 2, 1, 3)  # (2, 12, 10, 64)
```

**误区三：忽视数值稳定性**

QKV 投影后进行 softmax 时，值域可能很大，导致溢出：

```python
# QKV 值域可能很大
Q = np.random.randn(2, 12, 10, 64) * 10
K = np.random.randn(2, 12, 10, 64) * 10

# Q @ K^T 的值可能非常大
scores = np.einsum('bhnd,bhmd->bhnm', Q, K)
print(f"Attention scores 范围: [{scores.min():.2f}, {scores.max():.2f}]")
```

## API 总结

| 类 | 描述 |
|----|------|
| `QKVProjection` | 基本的 QKV 投影层 |
| `MultiHeadQKVProjection` | 多头 QKV 投影 |
| `SimpleQKV` | 简化的 QKV（单头，无偏置） |

QKV 投影是 Transformer 自注意力的核心。理解其实现原理，对于深入掌握 Transformer 架构至关重要。
