# 自注意力前向传播模拟

综合前面的知识，本篇文章模拟一个完整的简化版 Transformer 块的前向传播过程。我们将实现：词嵌入 → QKV 投影 → 缩放点积注意力（含因果掩码）→ 输出投影，构建一个可以处理短序列的简化自注意力层。通过这个实战，你可以看到各组件如何协同工作。

## 完整的自注意力模块

```python
import numpy as np

def he_initialization(fan_in, fan_out):
    """He 初始化"""
    std = np.sqrt(2.0 / fan_in)
    return np.random.randn(fan_in, fan_out).astype(np.float32) * std

def xavier_initialization(fan_in, fan_out):
    """Xavier 初始化"""
    std = np.sqrt(2.0 / (fan_in + fan_out))
    return np.random.randn(fan_in, fan_out).astype(np.float32) * std

def softmax(x, axis=-1):
    """Softmax 函数"""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

class SelfAttentionBlock:
    """简化的自注意力块"""

    def __init__(self, hidden_size, num_heads):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        assert hidden_size % num_heads == 0

        # QKV 投影
        self.W_q = xavier_initialization(hidden_size, hidden_size)
        self.W_k = xavier_initialization(hidden_size, hidden_size)
        self.W_v = xavier_initialization(hidden_size, hidden_size)

        # 输出投影
        self.W_o = xavier_initialization(hidden_size, hidden_size)

    def create_causal_mask(self, seq_len):
        """创建因果掩码"""
        return np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)

    def split_heads(self, X, batch_size):
        """分割为多头"""
        X = X.reshape(batch_size, -1, self.num_heads, self.head_dim)
        return X.transpose(0, 2, 1, 3)

    def forward(self, X):
        """前向传播

        参数:
            X: (batch_size, seq_len, hidden_size)
        返回:
            output: (batch_size, seq_len, hidden_size)
            attention_weights: (batch_size, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = X.shape

        # QKV 投影
        Q = X @ self.W_q  # (batch, seq, hidden)
        K = X @ self.W_k
        V = X @ self.W_v

        # 分割为多头
        Q = self.split_heads(Q, batch_size)  # (batch, heads, seq, head)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)

        # 计算注意力分数
        d_k = self.head_dim
        scores = np.einsum('bhnd,bhmd->bhnm', Q, K)  # (batch, heads, seq, seq)
        scores = scores / np.sqrt(d_k)

        # 应用因果掩码
        causal_mask = self.create_causal_mask(seq_len)  # True = mask
        causal_mask = causal_mask[np.newaxis, np.newaxis, :, :]  # 广播维度
        scores = np.where(causal_mask, -1e9, scores)

        # Softmax
        attention_weights = softmax(scores, axis=-1)

        # 加权求和
        context = np.einsum('bhnm,bhmd->bhnd', attention_weights, V)  # (batch, heads, seq, head)

        # 合并多头
        context = context.transpose(0, 2, 1, 3)  # (batch, seq, heads, head)
        context = context.reshape(batch_size, seq_len, self.hidden_size)

        # 输出投影
        output = context @ self.W_o

        return output, attention_weights

# 示例
np.random.seed(42)
batch_size = 2
seq_len = 8
hidden_size = 64
num_heads = 4

attention = SelfAttentionBlock(hidden_size, num_heads)
X = np.random.randn(batch_size, seq_len, hidden_size).astype(np.float32)

output, attention_weights = attention.forward(X)

print(f"输入形状: {X.shape}")
print(f"输出形状: {output.shape}")
print(f"注意力权重形状: {attention_weights.shape}")
print(f"注意力权重范围: [{attention_weights.min():.4f}, {attention_weights.max():.4f}]")
```

## 带词嵌入的完整流程

```python
class EmbeddingLayer:
    """词嵌入层"""

    def __init__(self, vocab_size, embed_dim):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        # 随机初始化
        self.weight = np.random.randn(vocab_size, embed_dim).astype(np.float32) * 0.02

    def forward(self, token_ids):
        """前向传播

        token_ids: (batch_size, seq_len)
        返回: (batch_size, seq_len, embed_dim)
        """
        return self.weight[token_ids]

class PositionalEmbedding:
    """位置编码（简化的可学习位置编码）"""

    def __init__(self, max_len, embed_dim):
        self.max_len = max_len
        self.embed_dim = embed_dim
        # 可学习的位置嵌入
        self.weight = np.random.randn(max_len, embed_dim).astype(np.float32) * 0.02

    def forward(self, seq_len):
        """前向传播

        seq_len: 序列长度
        返回: (1, seq_len, embed_dim)
        """
        return self.weight[:seq_len][np.newaxis, :, :]

class TransformerBlock:
    """简化的 Transformer 块（含嵌入、注意力）"""

    def __init__(self, vocab_size, hidden_size, num_heads, max_len=512):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        # 嵌入层
        self.token_embedding = EmbeddingLayer(vocab_size, hidden_size)
        self.position_embedding = PositionalEmbedding(max_len, hidden_size)

        # 自注意力层
        self.attention = SelfAttentionBlock(hidden_size, num_heads)

    def forward(self, token_ids):
        """前向传播

        token_ids: (batch_size, seq_len)
        返回:
            output: (batch_size, seq_len, hidden_size)
            attention_weights: (batch_size, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len = token_ids.shape

        # 词嵌入
        token_emb = self.token_embedding.forward(token_ids)

        # 位置嵌入
        position_emb = self.position_embedding.forward(seq_len)

        # 相加
        X = token_emb + position_emb

        # 自注意力
        output, attention_weights = self.attention.forward(X)

        return output, attention_weights

# 示例
np.random.seed(42)
batch_size = 2
seq_len = 10
vocab_size = 10000
hidden_size = 64
num_heads = 4

transformer = TransformerBlock(vocab_size, hidden_size, num_heads)

# 创建随机 token IDs
token_ids = np.random.randint(0, vocab_size, size=(batch_size, seq_len))

output, attention_weights = transformer.forward(token_ids)

print(f"\n=== Transformer 块前向传播 ===")
print(f"输入 token_ids 形状: {token_ids.shape}")
print(f"Token IDs 范围: [{token_ids.min()}, {token_ids.max()}]")
print(f"输出形状: {output.shape}")
print(f"注意力权重形状: {attention_weights.shape}")
```

## 可视化注意力权重

```python
def visualize_attention_weights(attention_weights, tokens, head_idx=0):
    """可视化指定头的注意力权重

    参数:
        attention_weights: (batch, heads, seq, seq)
        tokens: token 列表
        head_idx: 头索引
    """
    batch, heads, seq, _ = attention_weights.shape
    weights = attention_weights[0, head_idx]

    print(f"\n=== 注意力可视化 (Head {head_idx}) ===\n")
    header = "         " + "".join([f"{t[:5]:>8}" for t in tokens[:seq]])
    print(header)
    print("-" * len(header))

    for i in range(seq):
        row = f"{tokens[i][:5]:>8} |"
        for j in range(seq):
            w = weights[i, j]
            if w > 0.3:
                char = '###'
            elif w > 0.2:
                char = '##'
            elif w > 0.1:
                char = '#'
            elif w > 0.05:
                char = '.'
            else:
                char = ' '
            row += f"{char:>8}"
        print(row)

# 示例
tokens = ['[PAD]' if i >= 8 else ['The', 'cat', 'eats', 'the', 'fish', 'in', 'the', 'river'][i] for i in range(seq_len)]
visualize_attention_weights(attention_weights, tokens, head_idx=0)
```

## 模拟训练步骤

```python
def compute_loss(output, target):
    """简化的损失计算

    output: (batch, seq, vocab)
    target: (batch, seq)
    """
    batch_size, seq_len, vocab_size = output.shape

    # 交叉熵损失
    log_probs = output - np.log(np.sum(np.exp(output - np.max(output, axis=-1, keepdims=True)), axis=-1, keepdims=True))
    loss = 0.0
    for b in range(batch_size):
        for i in range(seq_len):
            loss -= log_probs[b, i, target[b, i]]

    return loss / (batch_size * seq_len)

def backward_pass(output, attention_weights, token_ids, target):
    """简化的反向传播（只展示梯度流动）

    实际实现需要自动微分，这里只是概念演示
    """
    batch_size, seq_len, hidden_size = output.shape

    # 假设梯度
    d_loss = np.random.randn(batch_size, seq_len, hidden_size).astype(np.float32) * 0.01

    print(f"\n=== 反向传播（梯度形状）===")
    print(f"输出梯度形状: {d_loss.shape}")
    print(f"梯度范围: [{d_loss.min():.6f}, {d_loss.max():.6f}]")

    return d_loss

# 示例
target = np.random.randint(0, vocab_size, size=(batch_size, seq_len))
loss = compute_loss(np.random.randn(batch_size, seq_len, vocab_size).astype(np.float32), target)
print(f"模拟损失: {loss:.4f}")

d_loss = backward_pass(output, attention_weights, token_ids, target)
```

## 完整的前向传播时间测量

```python
import time

def benchmark_forward_pass(batch_size, seq_len, vocab_size, hidden_size, num_heads, n_runs=100):
    """测量前向传播性能"""
    transformer = TransformerBlock(vocab_size, hidden_size, num_heads)
    token_ids = np.random.randint(0, vocab_size, size=(batch_size, seq_len))

    # 预热
    for _ in range(10):
        _, _ = transformer.forward(token_ids)

    # 测量
    start = time.time()
    for _ in range(n_runs):
        _, _ = transformer.forward(token_ids)
    elapsed = time.time() - start

    avg_time = elapsed / n_runs * 1000  # ms

    print(f"\n=== 性能基准测试 ===")
    print(f"Batch size: {batch_size}")
    print(f"Seq length: {seq_len}")
    print(f"Vocab size: {vocab_size}")
    print(f"Hidden size: {hidden_size}")
    print(f"Num heads: {num_heads}")
    print(f"平均耗时: {avg_time:.2f}ms")

# 测试不同配置
print("\n性能测试配置1: 小批量")
benchmark_forward_pass(batch_size=8, seq_len=64, vocab_size=10000, hidden_size=256, num_heads=4)

print("\n性能测试配置2: 长序列")
benchmark_forward_pass(batch_size=4, seq_len=256, vocab_size=10000, hidden_size=256, num_heads=4)
```

## 各组件的参数量

```python
def count_parameters():
    """计算各组件的参数量"""
    vocab_size = 10000
    hidden_size = 256
    num_heads = 4
    max_len = 512

    # 词嵌入
    token_emb_params = vocab_size * hidden_size
    pos_emb_params = max_len * hidden_size

    # QKV 投影
    qkv_params = 3 * hidden_size * hidden_size

    # 输出投影
    output_params = hidden_size * hidden_size

    # 总计
    total_params = token_emb_params + pos_emb_params + qkv_params + output_params

    print("\n=== 参数量统计 ===")
    print(f"Token 嵌入: {token_emb_params:,} ({token_emb_params/1e6:.2f}M)")
    print(f"位置嵌入: {pos_emb_params:,} ({pos_emb_params/1e6:.2f}M)")
    print(f"QKV 投影: {qkv_params:,} ({qkv_params/1e6:.2f}M)")
    print(f"输出投影: {output_params:,} ({output_params/1e6:.2f}M)")
    print(f"总计: {total_params:,} ({total_params/1e6:.2f}M)")

count_parameters()
```

## 常见问题与调试

### 检查注意力权重是否有效

```python
def check_attention_weights(attention_weights):
    """检查注意力权重是否有效"""
    batch, heads, seq, _ = attention_weights.shape

    print("\n=== 注意力权重检查 ===")

    # 检查是否每行和为1
    row_sums = attention_weights.sum(axis=-1)
    print(f"每行和的范围: [{row_sums.min():.4f}, {row_sums.max():.4f}]")

    # 检查是否有 NaN
    print(f"包含 NaN: {np.any(np.isnan(attention_weights))}")

    # 检查是否有过小的值
    min_val = attention_weights.min()
    print(f"最小值: {min_val:.6f}")

check_attention_weights(attention_weights)
```

### 检查梯度流动

```python
def check_gradient_flow(output, gradient_threshold=1e-7):
    """检查梯度是否正常流动"""
    print("\n=== 梯度流动检查 ===")
    print(f"输出范围: [{output.min():.4f}, {output.max():.4f}]")
    print(f"输出均值: {output.mean():.4f}")
    print(f"输出标准差: {output.std():.4f}")

    # 如果输出过小或过大，可能存在梯度问题
    if output.std() < 1e-6:
        print("警告: 输出过小，可能存在梯度消失")
    if output.std() > 1e3:
        print("警告: 输出过大，可能存在梯度爆炸")

check_gradient_flow(output)
```

这个实战项目综合运用了前几篇文章的知识，展示了完整的自注意力前向传播流程。通过逐步构建这个简化版 Transformer，你可以深入理解语言模型的核心机制。
