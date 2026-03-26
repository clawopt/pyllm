# Transformer解码器前向传播

Transformer 解码器是 GPT 等自回归语言模型的核心组件。与编码器不同，解码器在处理序列时需要使用因果掩码（causal mask）来确保每个位置只能看到当前位置及其之前的 token。本篇文章详细介绍如何使用纯 NumPy 实现一个简化但完整的 Transformer 解码器前向传播过程。

## 解码器与编码器的区别

Transformer 解码器与编码器的主要区别：
1. **因果掩码**：确保预测第 i 个 token 时只能使用 1 到 i-1 的信息
2. **交叉注意力**：在解码器中使用编码器的输出（用于翻译、摘要等任务）
3. **自回归生成**：一次生成一个 token

```python
import numpy as np

def demonstrate_encoder_decoder_difference():
    """演示编码器和解码器的区别"""
    print("=== 编码器 vs 解码器 ===\n")
    print("编码器：")
    print("  输入: [The, cat, eats, the, fish]")
    print("  每个位置可以看到所有其他位置")
    print("  Attention: 全双向 (bidirectional)\n")

    print("解码器：")
    print("  输入: [The, cat, eats, the, fish]")
    print("  位置 1 (cat) 只能看到 [The]")
    print("  位置 2 (eats) 只能看到 [The, cat]")
    print("  位置 3 (the) 只能看到 [The, cat, eats]")
    print("  Attention: 单向/因果 (unidirectional/causal)\n")

demonstrate_encoder_decoder_difference()
```

## 完整的 Transformer 解码器块

### 基础组件实现

```python
def layer_norm(x, gamma, beta, eps=1e-6):
    """LayerNorm"""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return gamma * ((x - mean) / np.sqrt(var + eps)) + beta

def softmax(x, axis=-1):
    """Softmax"""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def causal_mask(seq_len):
    """创建因果掩码"""
    return np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)

def xavier_init(fan_in, fan_out):
    """Xavier 初始化"""
    std = np.sqrt(2.0 / (fan_in + fan_out))
    return np.random.randn(fan_in, fan_out).astype(np.float32) * std
```

### GPT 解码器块实现

```python
class GPTDecoderBlock:
    """GPT Decoder Block

    包含：
    1. 多头因果自注意力
    2. Add & Norm
    3. 前馈网络
    4. Add & Norm
    """

    def __init__(self, hidden_size, num_heads, ff_dim):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.ff_dim = ff_dim

        # QKV 投影
        self.W_q = xavier_init(hidden_size, hidden_size)
        self.W_k = xavier_init(hidden_size, hidden_size)
        self.W_v = xavier_init(hidden_size, hidden_size)
        self.W_o = xavier_init(hidden_size, hidden_size)

        # LayerNorm 1
        self.gamma1 = np.ones(hidden_size, dtype=np.float32)
        self.beta1 = np.zeros(hidden_size, dtype=np.float32)

        # FFN
        self.W1 = xavier_init(hidden_size, ff_dim)
        self.b1 = np.zeros(ff_dim, dtype=np.float32)
        self.W2 = xavier_init(ff_dim, hidden_size)
        self.b2 = np.zeros(hidden_size, dtype=np.float32)

        # LayerNorm 2
        self.gamma2 = np.ones(hidden_size, dtype=np.float32)
        self.beta2 = np.zeros(hidden_size, dtype=np.float32)

    def split_heads(self, x, batch_size):
        """分割为多头"""
        x = x.reshape(batch_size, -1, self.num_heads, self.head_dim)
        return x.transpose(0, 2, 1, 3)

    def multi_head_attention(self, x, mask):
        """多头自注意力"""
        batch_size, seq_len, _ = x.shape

        # QKV 投影
        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v

        # 分割多头
        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)

        # 计算注意力分数
        scores = np.einsum('bhnd,bhmd->bhnm', Q, K) / np.sqrt(self.head_dim)

        # 应用因果掩码
        scores = np.where(mask, -1e9, scores)

        # Softmax
        attention_weights = softmax(scores, axis=-1)

        # 加权求和
        context = np.einsum('bhnm,bhmd->bhnd', attention_weights, V)

        # 合并多头
        context = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.hidden_size)

        # 输出投影
        output = context @ self.W_o

        return output, attention_weights

    def feed_forward(self, x):
        """前馈网络"""
        hidden = np.maximum(0, x @ self.W1 + self.b1)  # ReLU
        return hidden @ self.W2 + self.b2

    def forward(self, x):
        """前向传播

        x: (batch_size, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = x.shape

        # 创建因果掩码
        mask = causal_mask(seq_len)
        mask = mask[np.newaxis, np.newaxis, :, :]  # 广播维度

        # === Multi-Head Self-Attention ===
        attn_output, attention_weights = self.multi_head_attention(x, mask)

        # Add & Norm
        x = layer_norm(x + attn_output, self.gamma1, self.beta1)

        # === Feed Forward ===
        ffn_output = self.feed_forward(x)

        # Add & Norm
        output = layer_norm(x + ffn_output, self.gamma2, self.beta2)

        return output, attention_weights

# 示例
np.random.seed(42)
batch_size = 2
seq_len = 10
hidden_size = 256
num_heads = 8
ff_dim = 1024

decoder_block = GPTDecoderBlock(hidden_size, num_heads, ff_dim)
x = np.random.randn(batch_size, seq_len, hidden_size).astype(np.float32)

output, attention_weights = decoder_block.forward(x)

print(f"输入形状: {x.shape}")
print(f"输出形状: {output.shape}")
print(f"注意力权重形状: {attention_weights.shape}")
```

## 多层解码器堆叠

```python
class GPTModel:
    """GPT 模型（多层解码器堆叠）"""

    def __init__(self, vocab_size, hidden_size, num_heads, num_layers, ff_dim):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers

        # Token 嵌入
        self.token_embedding = np.random.randn(vocab_size, hidden_size).astype(np.float32) * 0.02

        # 位置嵌入
        self.max_seq_len = 512
        self.position_embedding = np.random.randn(self.max_seq_len, hidden_size).astype(np.float32) * 0.02

        # 解码器层
        self.decoder_blocks = [
            GPTDecoderBlock(hidden_size, num_heads, ff_dim)
            for _ in range(num_layers)
        ]

        # LayerNorm
        self.final_norm_gamma = np.ones(hidden_size, dtype=np.float32)
        self.final_norm_beta = np.zeros(hidden_size, dtype=np.float32)

        # LM Head
        self.lm_head = np.random.randn(hidden_size, vocab_size).astype(np.float32) * 0.02

    def forward(self, token_ids):
        """前向传播

        token_ids: (batch_size, seq_len)
        返回: (batch_size, seq_len, vocab_size) 的 logits
        """
        batch_size, seq_len = token_ids.shape

        # Token 嵌入
        token_emb = self.token_embedding[token_ids]

        # 位置嵌入
        position_ids = np.arange(seq_len)
        position_emb = self.position_embedding[position_ids]

        # 相加
        hidden_states = token_emb + position_emb

        # 通过解码器层
        all_attention_weights = []
        for block in self.decoder_blocks:
            hidden_states, attn_weights = block.forward(hidden_states)
            all_attention_weights.append(attn_weights)

        # 最终 LayerNorm
        hidden_states = layer_norm(
            hidden_states,
            self.final_norm_gamma,
            self.final_norm_beta
        )

        # LM Head（与 token embedding 权重绑定是常见的做法）
        logits = hidden_states @ self.lm_head

        return logits, all_attention_weights

# 示例
np.random.seed(42)
vocab_size = 10000
hidden_size = 256
num_heads = 8
num_layers = 6
ff_dim = 1024

gpt = GPTModel(vocab_size, hidden_size, num_heads, num_layers, ff_dim)

# 随机 token IDs
token_ids = np.random.randint(0, vocab_size, size=(2, 20))

logits, attention_weights = gpt.forward(token_ids)

print(f"\n=== GPT 模型前向传播 ===")
print(f"输入 token IDs 形状: {token_ids.shape}")
print(f"Logits 形状: {logits.shape}")
print(f"层数: {len(attention_weights)}")
print(f"每层注意力权重形状: {attention_weights[0].shape}")
```

## 提取下一个 token 的 logits

```python
def get_next_token_logits(logits, position):
    """获取指定位置的 logits（用于预测下一个 token）

    参数:
        logits: (batch_size, seq_len, vocab_size)
        position: 位置索引
    返回:
        next_logits: (batch_size, vocab_size)
    """
    return logits[:, position, :]

def predict_next_token(gpt, token_ids):
    """预测下一个 token

    参数:
        gpt: GPT 模型
        token_ids: 当前序列的 token IDs
    返回:
        next_token_logits: (vocab_size,) 每个 token 的得分
    """
    logits, _ = gpt.forward(token_ids)
    next_logits = get_next_token_logits(logits, -1)  # 取最后一个位置
    return next_logits[0]  # 假设 batch_size=1

# 示例
token_ids = np.array([[10, 20, 30, 40, 50]])  # 假设这是输入
next_logits = predict_next_token(gpt, token_ids)
print(f"\n下一个 token 的 logits 形状: {next_logits.shape}")
print(f"最高分的 token ID: {np.argmax(next_logits)}")
```

## 可视化注意力权重

```python
def visualize_attention_pattern(attention_weights, tokens, layer_idx=0, head_idx=0):
    """可视化注意力模式

    参数:
        attention_weights: 每层的注意力权重列表
        tokens: token 列表
        layer_idx: 要可视化的层索引
        head_idx: 要可视化的头索引
    """
    weights = attention_weights[layer_idx][0, head_idx]  # batch=0, head=head_idx

    seq_len = len(tokens)
    print(f"\n=== Layer {layer_idx}, Head {head_idx} 注意力模式 ===\n")

    # 打印
    header = "         " + "".join([f"{t[:5]:>8}" for t in tokens[:seq_len]])
    print(header)
    print("-" * len(header))

    for i, token in enumerate(tokens[:seq_len]):
        row = f"{token[:5]:>8} |"
        for j in range(seq_len):
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
tokens = ['[BOS]', 'hello', 'world', 'how', 'are', 'you', '[EOS]']
token_ids = np.array([[0, 10, 20, 30, 40, 50, 0]])  # 伪 token IDs

logits, attention_weights = gpt.forward(token_ids)
visualize_attention_pattern(attention_weights, tokens, layer_idx=0, head_idx=0)
```

## 常见问题与调试

### 检查输出是否合理

```python
def check_model_output(logits, attention_weights):
    """检查模型输出是否合理"""
    print("\n=== 模型输出检查 ===")

    # 检查 logits 范围
    print(f"Logits 范围: [{logits.min():.2f}, {logits.max():.2f}]")
    print(f"Logits 均值: {logits.mean():.2f}")

    # 检查注意力权重
    print(f"\n注意力权重范围: [{attention_weights[0].min():.4f}, {attention_weights[0].max():.4f}]")

    # 检查注意力权重是否每行和为1
    attn_sum = attention_weights[0].sum(axis=-1)
    print(f"注意力权重每行和范围: [{attn_sum.min():.4f}, {attn_sum.max():.4f}]")

    # 检查是否有 NaN 或 Inf
    print(f"\n包含 NaN: {np.any(np.isnan(logits))}")
    print(f"包含 Inf: {np.any(np.isinf(logits))}")

check_model_output(logits, attention_weights)
```

### 测量推理时间

```python
import time

def benchmark_inference(gpt, seq_len, n_runs=100):
    """测量推理时间"""
    token_ids = np.random.randint(0, gpt.vocab_size, size=(1, seq_len))

    # 预热
    for _ in range(10):
        gpt.forward(token_ids)

    # 测量
    start = time.time()
    for _ in range(n_runs):
        gpt.forward(token_ids)
    elapsed = time.time() - start

    avg_time = elapsed / n_runs * 1000

    print(f"\n=== 推理性能 ===")
    print(f"序列长度: {seq_len}")
    print(f"模型层数: {gpt.num_layers}")
    print(f"平均耗时: {avg_time:.2f}ms")

benchmark_inference(gpt, seq_len=20)
benchmark_inference(gpt, seq_len=50)
benchmark_inference(gpt, seq_len=100)
```

这篇文章详细介绍了 Transformer 解码器的前向传播实现。在下一篇文章中，我们将介绍如何实现温度采样来生成文本。
