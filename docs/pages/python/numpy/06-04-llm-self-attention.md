# LLM场景示例：自注意力综合实践

在前几节中，我们学习了逐元素运算、向量化和矩阵乘法。现在让我们把这些知识综合起来，实现一个简化版的自注意力机制。自注意力（Self-Attention）是Transformer架构的核心组件，也是LLM能够处理长距离依赖的关键技术。理解自注意力的实现原理，不仅能帮助我们掌握NumPy的各种操作技巧，还能深入理解LLM的工作原理。

## 自注意力的核心思想

自注意力的核心思想是：序列中的每个位置都可以关注序列中的所有其他位置，计算它们之间的相关性（attention），然后用这个相关性作为权重对值（Value）进行加权求和。

数学上，自注意力可以表示为：
```
Attention(Q, K, V) = softmax(Q · K^T / √d) · V
```

其中 Q（Query）是查询，K（Key）是键，V（Value）是值。Q·K^T 计算查询和键的相关性，除以 √d 是为了缩放防止梯度消失，softmax 将分数归一化为概率分布，最后用这个分布对 V 加权求和。

## 完整自注意力实现

让我们一步步实现一个简化版的自注意力：

```python
import numpy as np

def softmax(x):
    """Softmax 归一化"""
    exp_x = np.exp(x - x.max(axis=-1, keepdims=True))
    return exp_x / exp_x.sum(axis=-1, keepdims=True)

class SimplifiedSelfAttention:
    """简化版自注意力"""

    def __init__(self, hidden_dim, num_heads=1):
        assert hidden_dim % num_heads == 0, "hidden_dim 必须能被 num_heads 整除"
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # 初始化 Q, K, V 的投影权重（简化为单个投影）
        np.random.seed(42)
        self.W_q = np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * np.sqrt(2.0 / hidden_dim)
        self.W_k = np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * np.sqrt(2.0 / hidden_dim)
        self.W_v = np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * np.sqrt(2.0 / hidden_dim)

    def split_heads(self, x, batch_size, seq_len):
        """将 hidden_dim 分割为多个头"""
        # x: (batch, seq, hidden_dim)
        x = x.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        # 转置: (batch, num_heads, seq, head_dim)
        return x.transpose(0, 2, 1, 3)

    def forward(self, hidden_states):
        """前向传播

        Args:
            hidden_states: (batch_size, seq_len, hidden_dim)

        Returns:
            output: (batch_size, seq_len, hidden_dim)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # 1. 计算 Q, K, V
        Q = hidden_states @ self.W_q.T  # (batch, seq, hidden_dim)
        K = hidden_states @ self.W_k.T
        V = hidden_states @ self.W_v.T

        # 2. 分割多头
        Q = self.split_heads(Q, batch_size, seq_len)  # (batch, heads, seq, head_dim)
        K = self.split_heads(K, batch_size, seq_len)
        V = self.split_heads(V, batch_size, seq_len)

        # 3. 计算注意力分数: Q @ K^T
        # Q: (batch, heads, seq, head_dim)
        # K^T: (batch, heads, head_dim, seq)
        attention_scores = Q @ K.transpose(0, 1, 3, 2)  # (batch, heads, seq, seq)
        attention_scores = attention_scores / np.sqrt(self.head_dim)  # 缩放

        # 4. 应用 softmax
        attention_weights = softmax(attention_scores)

        # 5. 应用注意力到 V
        # attention_weights: (batch, heads, seq, seq)
        # V: (batch, heads, seq, head_dim)
        context = attention_weights @ V  # (batch, heads, seq, head_dim)

        # 6. 合并多头
        # context: (batch, heads, seq, head_dim) -> (batch, seq, heads, head_dim) -> (batch, seq, hidden_dim)
        context = context.transpose(0, 2, 1, 3)  # (batch, seq, heads, head_dim)
        context = context.reshape(batch_size, seq_len, self.hidden_dim)

        return context, attention_weights

# 测试
batch_size = 2
seq_len = 8
hidden_dim = 16

# 创建自注意力层
attention = SimplifiedSelfAttention(hidden_dim, num_heads=4)

# 模拟输入
hidden_states = np.random.randn(batch_size, seq_len, hidden_dim).astype(np.float32)

# 前向传播
output, attention_weights = attention.forward(hidden_states)

print(f"输入形状: {hidden_states.shape}")
print(f"输出形状: {output.shape}")
print(f"注意力权重形状: {attention_weights.shape}")
print(f"注意力权重（第一个样本第一个头）:\n{attention_weights[0, 0]}")
```

## 添加位置编码

自注意力本身不包含位置信息，需要通过位置编码来注入。让我们添加位置编码：

```python
def positional_encoding(seq_len, d_model):
    """生成 sin/cos 位置编码"""
    position = np.arange(seq_len)[:, np.newaxis]
    div_term = np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model)
    div_term = np.exp(div_term)

    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)

    return pe.astype(np.float32)

class SelfAttentionWithPositionalEncoding:
    """带位置编码的自注意力"""

    def __init__(self, hidden_dim, num_heads=1):
        self.attention = SimplifiedSelfAttention(hidden_dim, num_heads)

    def forward(self, hidden_states):
        """前向传播（带位置编码）"""
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # 生成位置编码
        pos_encoding = positional_encoding(seq_len, hidden_dim)
        pos_encoding = pos_encoding[np.newaxis, :, :]  # (1, seq, hidden_dim)

        # 添加位置编码
        hidden_states = hidden_states + pos_encoding

        # 通过自注意力层
        return self.attention.forward(hidden_states)

# 测试带位置编码的版本
attention_with_pos = SelfAttentionWithPositionalEncoding(hidden_dim, num_heads=4)
output, attention_weights = attention_with_pos.forward(hidden_states)
print(f"带位置编码的输出形状: {output.shape}")
```

## 添加掩码机制

在训练自回归模型时，需要屏蔽未来位置。让我添加因果掩码：

```python
def create_causal_mask(seq_len):
    """创建因果掩码（确保只能看到当前位置及之前的内容）"""
    # 创建一个下三角矩阵，对角线及以上为 True（可关注），下方为 False（需屏蔽）
    mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=0)
    return mask

def apply_mask(attention_scores, mask, mask_value=-1e9):
    """应用掩码"""
    # 将需要屏蔽的位置设为一个很小的负数
    masked_scores = np.where(mask, attention_scores, mask_value)
    return masked_scores

class MaskedSelfAttention(SimplifiedSelfAttention):
    """带掩码的自注意力"""

    def forward(self, hidden_states, use_causal_mask=True):
        batch_size, seq_len, _ = hidden_states.shape

        # 计算 Q, K, V
        Q = hidden_states @ self.W_q.T
        K = hidden_states @ self.W_k.T
        V = hidden_states @ self.W_v.T

        # 分割多头
        Q = self.split_heads(Q, batch_size, seq_len)
        K = self.split_heads(K, batch_size, seq_len)
        V = self.split_heads(V, batch_size, seq_len)

        # 计算注意力分数
        attention_scores = Q @ K.transpose(0, 1, 3, 2)
        attention_scores = attention_scores / np.sqrt(self.head_dim)

        # 应用因果掩码
        if use_causal_mask:
            causal_mask = create_causal_mask(seq_len)  # (seq, seq)
            # 广播掩码以匹配 attention_scores 的形状
            causal_mask = causal_mask[np.newaxis, np.newaxis, :, :]  # (1, 1, seq, seq)
            attention_scores = apply_mask(attention_scores, causal_mask)

        # Softmax
        attention_weights = softmax(attention_scores)

        # 应用注意力到 V
        context = attention_weights @ V
        context = context.transpose(0, 2, 1, 3)
        context = context.reshape(batch_size, seq_len, self.hidden_dim)

        return context, attention_weights

# 测试带掩码的版本
masked_attention = MaskedSelfAttention(hidden_dim, num_heads=4)
output, attention_weights = masked_attention.forward(hidden_states)
print(f"带掩码的输出形状: {output.shape}")
print(f"因果掩码后的注意力权重:\n{attention_weights[0, 0]}")
```

## 多层自注意力

实际应用中，通常会堆叠多个自注意力层：

```python
class MultiLayerSelfAttention:
    """多层自注意力"""

    def __init__(self, hidden_dim, num_heads, num_layers=2):
        self.layers = []
        for i in range(num_layers):
            self.layers.append(SimplifiedSelfAttention(hidden_dim, num_heads))

    def forward(self, hidden_states):
        """多层前向传播"""
        for i, layer in enumerate(self.layers):
            # 残差连接（实际应用中通常会在每个子层周围加上）
            residual = hidden_states
            hidden_states, attention_weights = layer.forward(hidden_states)
            hidden_states = hidden_states + residual  # 残差连接
            print(f"第{i+1}层输出形状: {hidden_states.shape}")
        return hidden_states, attention_weights

# 测试多层自注意力
multi_layer_attention = MultiLayerSelfAttention(hidden_dim, num_heads=4, num_layers=3)
output, final_attention = multi_layer_attention.forward(hidden_states)
print(f"多层自注意力最终输出形状: {output.shape}")
```

## 性能分析

让我们测量自注意力各步骤的计算量：

```python
import time

def measure_performance():
    """测量自注意力的性能"""
    batch_size = 8
    seq_len = 256
    hidden_dim = 512
    num_heads = 8

    attention = SimplifiedSelfAttention(hidden_dim, num_heads)
    hidden_states = np.random.randn(batch_size, seq_len, hidden_dim).astype(np.float32)

    # 预热
    _ = attention.forward(hidden_states)

    # 测量时间
    iterations = 10
    start = time.time()
    for _ in range(iterations):
        output, _ = attention.forward(hidden_states)
    elapsed = time.time() - start

    print(f"序列长度: {seq_len}, 批量大小: {batch_size}")
    print(f"隐藏维度: {hidden_dim}, 头数: {num_heads}")
    print(f"每轮耗时: {elapsed/iterations*1000:.2f} ms")

    # 计算理论计算量
    # Q, K, V 投影: 3 * (batch * seq * hidden_dim * hidden_dim)
    # 注意力分数: batch * num_heads * seq * seq * head_dim
    # 输出投影: batch * seq * hidden_dim * hidden_dim
    print(f"\n理论计算量（浮点运算次数）:")
    print(f"QKV投影: {3 * batch_size * seq_len * hidden_dim * hidden_dim:,}")
    print(f"注意力分数: {batch_size * num_heads * seq_len * seq_len * hidden_dim // num_heads:,}")
    print(f"输出投影: {batch_size * seq_len * hidden_dim * hidden_dim:,}")

measure_performance()
```

## 小结

这一节我们综合运用了前几节学到的各种NumPy操作，实现了一个简化版的自注意力机制。关键技术点包括：

1. **矩阵乘法**用于 Q、K、V 的投影计算和注意力分数的计算
2. **逐元素运算**用于缩放因子、除法等操作
3. **softmax 归一化**使用向量化实现
4. **Reshape 和转置**用于多头注意力的维度变换
5. **掩码机制**用于实现因果注意力

自注意力是 Transformer 和 LLM 的核心，理解其实现原理对于深入学习大语言模型至关重要。

面试时需要能够描述自注意力的完整计算流程，理解 Q、K、V 的作用，以及解释为什么要除以 √d 进行缩放。
