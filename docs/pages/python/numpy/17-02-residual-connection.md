# 残差连接

残差连接（Residual Connection）又称为跳跃连接（Skip Connection），是深度学习中最常用的技术之一，由 He 等人在 2015 年提出。残差连接通过将输入直接加到输出上，使得梯度能够直接流过网络，有效缓解了深层网络的梯度消失问题。本篇文章介绍残差连接的原理，以及如何在 Transformer 块中将其与自注意力和前馈网络结合。

## 残差连接的基本原理

残差连接的核心思想是：与其学习 H(x)，不如学习 F(x) + x，其中 H(x) 是期望的映射。公式如下：

```
output = LayerNorm(x + Sublayer(x))
```

其中 Sublayer(x) 是子层（如注意力或 FFN）的输出。这种设计使得：
1. 梯度能够直接回传到输入
2. 信息流动更加顺畅
3. 网络更容易优化

```python
import numpy as np

class ResidualConnection:
    """残差连接（Add & Norm）"""

    def __init__(self, hidden_size, dropout_rate=0.1):
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate

    def forward(self, x, sublayer_output, training=True):
        """前向传播

        参数:
            x: 残差连接的输入
            sublayer_output: 子层的输出
            training: 是否在训练模式
        返回:
            残差连接后的输出
        """
        # Dropout（如果启用）
        if training and self.dropout_rate > 0:
            mask = np.random.binomial(1, 1 - self.dropout_rate, sublayer_output.shape)
            sublayer_output = sublayer_output * mask / (1 - self.dropout_rate)

        # 残差连接
        output = x + sublayer_output

        return output

# 示例
np.random.seed(42)
batch_size = 2
seq_len = 5
hidden_size = 8

x = np.random.randn(batch_size, seq_len, hidden_size).astype(np.float32)
sublayer_out = np.random.randn(batch_size, seq_len, hidden_size).astype(np.float32)

residual = ResidualConnection(hidden_size)
output = residual.forward(x, sublayer_out)

print(f"输入形状: {x.shape}")
print(f"子层输出形状: {sublayer_out.shape}")
print(f"残差连接输出形状: {output.shape}")
print(f"\n验证残差连接:")
print(f"x 的均值: {x.mean():.4f}")
print(f"sublayer_out 的均值: {sublayer_out.mean():.4f}")
print(f"output 的均值: {output.mean():.4f}")
```

## 完整的 Transformer 块模拟

```python
def layer_norm(x, gamma, beta, eps=1e-6):
    """LayerNorm"""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return gamma * x_norm + beta

def softmax(x, axis=-1):
    """Softmax"""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

class TransformerBlock:
    """简化的 Transformer 块

    包含：Multi-Head Attention -> Add & Norm -> FFN -> Add & Norm
    """

    def __init__(self, hidden_size, num_heads, ff_dim):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # QKV 投影
        self.W_q = np.random.randn(hidden_size, hidden_size).astype(np.float32) * 0.02
        self.W_k = np.random.randn(hidden_size, hidden_size).astype(np.float32) * 0.02
        self.W_v = np.random.randn(hidden_size, hidden_size).astype(np.float32) * 0.02
        self.W_o = np.random.randn(hidden_size, hidden_size).astype(np.float32) * 0.02

        # LayerNorm 参数（Attention 后）
        self.gamma1 = np.ones(hidden_size, dtype=np.float32)
        self.beta1 = np.zeros(hidden_size, dtype=np.float32)

        # FFN
        self.W1 = np.random.randn(hidden_size, ff_dim).astype(np.float32) * 0.02
        self.b1 = np.zeros(ff_dim, dtype=np.float32)
        self.W2 = np.random.randn(ff_dim, hidden_size).astype(np.float32) * 0.02
        self.b2 = np.zeros(hidden_size, dtype=np.float32)

        # LayerNorm 参数（FFN 后）
        self.gamma2 = np.ones(hidden_size, dtype=np.float32)
        self.beta2 = np.zeros(hidden_size, dtype=np.float32)

    def multi_head_attention(self, X):
        """Multi-Head Attention"""
        batch_size, seq_len, _ = X.shape

        # QKV 投影
        Q = X @ self.W_q
        K = X @ self.W_k
        V = X @ self.W_v

        # 分割为多头
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # 注意力分数
        scores = np.einsum('bhnd,bhmd->bhnm', Q, K) / np.sqrt(self.head_dim)

        # 因果掩码（简化）
        causal_mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)
        scores = np.where(causal_mask[np.newaxis, np.newaxis, :, :], -1e9, scores)

        # Softmax
        attention_weights = softmax(scores, axis=-1)

        # 加权求和
        context = np.einsum('bhnm,bhmd->bhnd', attention_weights, V)

        # 合并多头
        context = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.hidden_size)

        # 输出投影
        output = context @ self.W_o

        return output, attention_weights

    def feed_forward(self, X):
        """前馈网络"""
        hidden = np.einsum('bsd,df->bsf', X, self.W1) + self.b1
        hidden = np.maximum(0, hidden)  # ReLU
        output = np.einsum('bsf,fd->bsd', hidden, self.W2) + self.b2
        return output

    def forward(self, X):
        """完整的前向传播

        X: (batch_size, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = X.shape

        # === Self-Attention ===
        attn_output, attention_weights = self.multi_head_attention(X)

        # Add & Norm (Post-LN)
        attn_output = layer_norm(X + attn_output, self.gamma1, self.beta1)

        # === Feed-Forward ===
        ffn_output = self.feed_forward(attn_output)

        # Add & Norm
        output = layer_norm(attn_output + ffn_output, self.gamma2, self.beta2)

        return output, attention_weights

# 示例
np.random.seed(42)
batch_size = 2
seq_len = 8
hidden_size = 64
num_heads = 4
ff_dim = 256

transformer = TransformerBlock(hidden_size, num_heads, ff_dim)
X = np.random.randn(batch_size, seq_len, hidden_size).astype(np.float32)

output, attention_weights = transformer.forward(X)

print(f"=== Transformer 块前向传播 ===")
print(f"输入形状: {X.shape}")
print(f"输出形状: {output.shape}")
print(f"注意力权重形状: {attention_weights.shape}")
print(f"\n输出统计:")
print(f"  均值: {output.mean():.4f}")
print(f"  标准差: {output.std():.4f}")
print(f"  范围: [{output.min():.4f}, {output.max():.4f}]")
```

## 残差连接的可视化

```python
def visualize_residual_flow():
    """可视化残差连接的梯度流动"""

    print("\n=== 残差连接的优势 ===\n")

    print("场景1: 没有残差连接的深层网络")
    print("  输入 x -> Layer1 -> Layer2 -> ... -> LayerN -> 输出")
    print("  梯度反向传播: ∂L/∂x = ∂L/∂LayerN * ∂LayerN/∂LayerN-1 * ... * ∂Layer1/∂x")
    print("  问题: 深层网络中，梯度连乘可能导致梯度消失或爆炸\n")

    print("场景2: 有残差连接的深层网络")
    print("  输入 x -> Layer1 -> Layer2 -> ... -> LayerN")
    print("                     |")
    print("                     +---> Add -> 输出")
    print("  输出 = x + F(x)，其中 F 是子层的组合")
    print("  梯度: ∂(x+F(x))/∂x = 1 + ∂F(x)/∂x")
    print("  即使 ∂F(x)/∂x 很小，梯度仍然有 1，不会完全消失\n")

    print("场景3: 恒等映射的极端情况")
    print("  如果子层学习到 F(x) ≈ 0，则 output ≈ x")
    print("  这意味着残差连接可以学习恒等映射")
    print("  这在深度网络中非常有用，因为深层的恒等变换至少不会让效果变差")

visualize_residual_flow()
```

## 残差连接的变体

### 预激活残差连接（Pre-Activation）

```python
class PreActivationResidualBlock:
    """预激活残差块（Pre-Activation Residual Block）"""

    def __init__(self, hidden_size):
        self.hidden_size = hidden_size
        self.gamma = np.ones(hidden_size, dtype=np.float32)
        self.beta = np.zeros(hidden_size, dtype=np.float32)

    def forward(self, x, sublayer_func):
        """前向传播

        参数:
            x: 输入
            sublayer_func: 子层函数，接受 norm(x) 作为输入
        """
        # 先 LayerNorm
        x_norm = layer_norm(x, self.gamma, self.beta)

        # 子层
        sublayer_output = sublayer_func(x_norm)

        # 残差连接（在 LayerNorm 之前）
        output = x + sublayer_output

        return output

# 示例
np.random.seed(42)
pre_residual = PreActivationResidualBlock(hidden_size=8)
x = np.random.randn(2, 5, 8).astype(np.float32)

def ffn(x):
    return x @ np.random.randn(8, 8).astype(np.float32) * 0.02

output = pre_residual.forward(x, ffn)
print(f"预激活残差块输出形状: {output.shape}")
```

## 残差连接的代码实现要点

```python
class AddNorm:
    """Add & Norm 操作"""

    def __init__(self, hidden_size, dropout_rate=0.0):
        self.norm = LayerNorm(hidden_size)
        self.dropout_rate = dropout_rate

    def forward(self, x, sublayer_output, training=False):
        """前向传播

        x + Sublayer(x) 然后 LayerNorm
        """
        # Dropout（可选）
        if training and self.dropout_rate > 0:
            mask = np.random.binomial(1, 1 - self.dropout_rate, sublayer_output.shape)
            sublayer_output = sublayer_output * mask / (1 - self.dropout_rate)

        # 残差连接
        added = x + sublayer_output

        # LayerNorm
        output = self.norm.forward(added)

        return output

# 示例
np.random.seed(42)
add_norm = AddNorm(hidden_size=8, dropout_rate=0.1)

x = np.random.randn(2, 5, 8).astype(np.float32)
sublayer_out = np.random.randn(2, 5, 8).astype(np.float32)

output_train = add_norm.forward(x, sublayer_out, training=True)
output_eval = add_norm.forward(x, sublayer_out, training=False)

print(f"训练模式输出均值: {output_train.mean():.4f}")
print(f"评估模式输出均值: {output_eval.mean():.4f}")
```

## 常见误区

**误区一：残差连接后忘记 LayerNorm**

残差连接后的 LayerNorm 是 Transformer 的标准做法：

```python
# 正确顺序：x + Sublayer(x) -> LayerNorm
output = layer_norm(x + sublayer_output, gamma, beta)

# 错误：LayerNorm(x) + Sublayer(x) -> 缺少最后的归一化
# output = layer_norm(x, gamma, beta) + sublayer_output  # 这是不同的架构！
```

**误区二：维度不匹配**

残差连接要求 x 和 sublayer_output 的维度相同：

```python
# 错误：维度不匹配
x = np.random.randn(2, 5, 8)  # hidden = 8
sublayer_out = np.random.randn(2, 5, 16)  # hidden = 16
# x + sublayer_out  # 广播可能产生意外结果！

# 正确：确保维度一致
assert x.shape == sublayer_output.shape
```

**误区三：混淆 Pre-LN 和 Post-LN**

Pre-LN 和 Post-LN 的残差连接位置不同：

```python
# Post-LN（原始 Transformer）：
# output = LayerNorm(x + Sublayer(x))

# Pre-LN（更稳定）：
# output = x + Sublayer(LayerNorm(x))
```

## API 总结

| 类/函数 | 描述 |
|--------|------|
| `ResidualConnection` | 残差连接（Add） |
| `AddNorm` | Add & Norm 操作 |
| `TransformerBlock` | 完整 Transformer 块 |
| `PreActivationResidualBlock` | 预激活残差块 |

残差连接是现代深度学习模型的基石。理解其原理和实现，对于构建高效的 Transformer 模型至关重要。
