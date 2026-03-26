# 简单文本序列生成

综合前面的知识，本篇文章实现一个完整的简化版 GPT-like 文本生成系统。这个系统包含：词汇表、嵌入层、简化的 Transformer 解码器、LM Head、温度采样器，以及完整的自回归生成循环。

## 完整实现

```python
import numpy as np

def softmax(x, axis=-1):
    """Softmax 函数"""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def xavier_init(fan_in, fan_out):
    """Xavier 初始化"""
    std = np.sqrt(2.0 / (fan_in + fan_out))
    return np.random.randn(fan_in, fan_out).astype(np.float32) * std

class SimpleVocabulary:
    """简单词汇表"""
    def __init__(self, tokens):
        self.token_to_idx = {t: i for i, t in enumerate(tokens)}
        self.idx_to_token = {i: t for i, t in enumerate(tokens)}
        self.vocab_size = len(tokens)

    def encode(self, tokens):
        return [self.token_to_idx.get(t, self.token_to_idx['[UNK]']) for t in tokens]

    def decode(self, indices):
        return [self.idx_to_token.get(i, '[UNK]') for i in indices]

class SimpleTextGenerator:
    """简化的文本生成器

    包含：嵌入、位置编码、简化 Transformer、LM Head
    """

    def __init__(self, vocab_size, embed_dim=64, num_heads=4, num_layers=2):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        # Token 嵌入
        self.token_embedding = np.random.randn(vocab_size, embed_dim).astype(np.float32) * 0.02

        # 位置编码（可学习）
        self.position_embedding = np.random.randn(512, embed_dim).astype(np.float32) * 0.02

        # QKV 投影（简化：所有层共享）
        self.W_q = xavier_init(embed_dim, embed_dim)
        self.W_k = xavier_init(embed_dim, embed_dim)
        self.W_v = xavier_init(embed_dim, embed_dim)
        self.W_o = xavier_init(embed_dim, embed_dim)

        # FFN
        self.W1 = xavier_init(embed_dim, embed_dim * 4)
        self.b1 = np.zeros(embed_dim * 4, dtype=np.float32)
        self.W2 = xavier_init(embed_dim * 4, embed_dim)
        self.b2 = np.zeros(embed_dim, dtype=np.float32)

        # LM Head
        self.lm_head = np.random.randn(embed_dim, vocab_size).astype(np.float32) * 0.02

        # LayerNorm
        self.norm_gamma = np.ones(embed_dim, dtype=np.float32)
        self.norm_beta = np.zeros(embed_dim, dtype=np.float32)

    def layer_norm(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True)
        return self.norm_gamma * (x - mean) / (std + 1e-6) + self.norm_beta

    def causal_mask(self, seq_len):
        return np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)

    def forward(self, token_ids):
        """前向传播

        token_ids: (seq_len,) 或 (batch, seq_len)
        """
        if len(token_ids.shape) == 1:
            token_ids = token_ids[np.newaxis, :]

        batch_size, seq_len = token_ids.shape

        # Token 嵌入
        token_emb = self.token_embedding[token_ids]

        # 位置嵌入
        position_ids = np.arange(seq_len)
        position_emb = self.position_embedding[position_ids]

        # 相加
        hidden = token_emb + position_emb

        # === Transformer 层 ===
        for _ in range(1):  # 简化：只一层
            # Self-Attention
            Q = hidden @ self.W_q
            K = hidden @ self.W_k
            V = hidden @ self.W_v

            # 注意力分数
            d_k = self.embed_dim
            scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_k)

            # 因果掩码
            mask = self.causal_mask(seq_len)
            scores = np.where(mask[np.newaxis, :, :], -1e9, scores)

            # Softmax
            attn_weights = softmax(scores, axis=-1)

            # 加权求和
            context = attn_weights @ V

            # 残差 + LayerNorm
            hidden = self.layer_norm(hidden + context)

            # FFN
            ffn_hidden = np.maximum(0, hidden @ self.W1 + self.b1)
            ffn_output = ffn_hidden @ self.W2 + self.b2

            # 残差 + LayerNorm
            hidden = self.layer_norm(hidden + ffn_output)

        # LM Head
        logits = hidden @ self.lm_head

        return logits

    def generate(self, start_tokens, max_length=50, temperature=1.0, top_k=None):
        """自回归生成

        参数:
            start_tokens: 起始 token 列表
            max_length: 最大生成长度
            temperature: 温度参数
            top_k: Top-K 采样
        """
        generated = list(start_tokens)

        for _ in range(max_length):
            # 编码
            token_ids = np.array(self.vocab.encode(generated))

            # 前向传播
            logits = self.forward(token_ids)

            # 取最后一个位置的 logits
            next_logits = logits[0, -1]

            # 温度采样
            if temperature == 0:
                next_token_id = np.argmax(next_logits)
            else:
                # 应用温度
                scaled_logits = next_logits / temperature

                # Top-K
                if top_k is not None:
                    top_k_indices = np.argpartition(scaled_logits, -top_k)[-top_k:]
                    masked_logits = np.full_like(scaled_logits, -1e9)
                    masked_logits[top_k_indices] = scaled_logits[top_k_indices]
                    scaled_logits = masked_logits

                # Softmax
                probs = softmax(scaled_logits)

                # 采样
                next_token_id = np.random.choice(self.vocab_size, p=probs)

            # 解码
            next_token = self.vocab.decode([next_token_id])
            generated.append(next_token)

            # 遇到 EOS 停止
            if next_token == '[EOS]':
                break

        return generated

# 创建词汇表
tokens = ['[PAD]', '[UNK]', '[BOS]', '[EOS]', 'hello', 'world', 'the', 'quick',
          'brown', 'fox', 'jumps', 'over', 'lazy', 'dog', 'how', 'are', 'you',
          'good', 'morning', 'good', 'night', 'thank', 'you', 'yes', 'no']

vocab = SimpleVocabulary(tokens)

# 创建生成器
generator = SimpleTextGenerator(vocab_size=len(tokens), embed_dim=64, num_heads=4, num_layers=2)
generator.vocab = vocab

# 生成文本
np.random.seed(42)
start = ['[BOS]']

print("=== 文本生成示例 ===\n")

print("温度=0.0 (贪心):")
generated = generator.generate(start, max_length=30, temperature=0.0)
print(f"  {' '.join(generated)}\n")

print("温度=0.5:")
generated = generator.generate(start, max_length=30, temperature=0.5)
print(f"  {' '.join(generated)}\n")

print("温度=1.0:")
generated = generator.generate(start, max_length=30, temperature=1.0)
print(f"  {' '.join(generated)}\n")

print("温度=1.5 (更多样):")
generated = generator.generate(start, max_length=30, temperature=1.5)
print(f"  {' '.join(generated)}\n")
```

## 生成过程可视化

```python
def visualize_generation_step():
    """可视化生成步骤"""
    print("\n=== 生成过程可视化 ===\n")

    print("步骤 1: 输入 [BOS]")
    print("  token_ids = [2]  (BOS 的索引)")
    print()

    print("步骤 2: 前向传播 -> 计算下一个 token 的 logits")
    print()

    print("步骤 3: 温度采样 -> 选择下一个 token")
    print()

    print("步骤 4: 输出并添加到序列")
    print()

    print("步骤 5: 如果是 [EOS]，停止；否则回到步骤 1")
    print()

    print("示例序列:")
    sequence = ['[BOS]', 'hello', 'world', '[EOS]']
    for i, token in enumerate(sequence):
        arrow = " -> " if i < len(sequence) - 1 else ""
        print(f"  {token}{arrow}", end="")
    print()

visualize_generation_step()
```

## 批量生成

```python
def batch_generate(generator, start_sequences, max_length=50, temperature=1.0):
    """批量生成多个序列

    参数:
        generator: SimpleTextGenerator 实例
        start_sequences: 起始序列列表
        max_length: 最大长度
        temperature: 温度
    返回:
        生成的序列列表
    """
    results = []

    for start_seq in start_sequences:
        generated = generator.generate(start_seq, max_length, temperature)
        results.append(generated)

    return results

# 示例
np.random.seed(None)  # 随机种子
start_sequences = [
    ['[BOS]'],
    ['[BOS]', 'hello'],
    ['[BOS]', 'how', 'are'],
]

print("\n=== 批量生成 ===\n")
for i, start_seq in enumerate(start_sequences):
    generated = generator.generate(start_seq, max_length=20, temperature=1.0)
    print(f"序列 {i+1} {start_seq} -> {' '.join(generated)}")
```

## 评估生成质量

```python
def evaluate_generation(generated_tokens, reference=None):
    """简单的生成质量评估"""
    print("\n=== 生成质量评估 ===\n")

    # 长度统计
    print(f"生成长度: {len(generated_tokens)} tokens")

    # 重复检测
    unique_ratio = len(set(generated_tokens)) / len(generated_tokens) if generated_tokens else 0
    print(f"唯一 token 比例: {unique_ratio:.2%}")

    # N-gram 重复率
    def get_ngram_repeat(tokens, n=2):
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        if not ngrams:
            return 0
        unique_ngrams = len(set(ngrams))
        return 1 - unique_ngrams / len(ngrams)

    print(f"2-gram 重复率: {get_ngram_repeat(generated_tokens, 2):.2%}")
    print(f"3-gram 重复率: {get_ngram_repeat(generated_tokens, 3):.2%}")

    # 计算生成文本的熵
    from collections import Counter
    token_counts = Counter(generated_tokens)
    total = sum(token_counts.values())
    entropy = -sum((count/total) * np.log2(count/total) for count in token_counts.values())
    print(f"Token 熵: {entropy:.2f} bits")

# 评估生成的序列
generated = generator.generate(['[BOS]'], max_length=30, temperature=1.0)
print(f"生成的序列: {' '.join(generated)}")
evaluate_generation(generated)
```

## 常见问题与调试

### 检查嵌入是否正确

```python
def check_embeddings(generator):
    """检查嵌入层"""
    print("\n=== 嵌入检查 ===")
    print(f"Token 嵌入形状: {generator.token_embedding.shape}")
    print(f"Token 嵌入均值: {generator.token_embedding.mean():.6f}")
    print(f"Token 嵌入标准差: {generator.token_embedding.std():.6f}")

check_embeddings(generator)
```

### 检查采样是否工作

```python
def test_sampling():
    """测试采样"""
    print("\n=== 采样测试 ===")

    # 创建随机 logits
    logits = np.random.randn(len(tokens)).astype(np.float32)

    print("不同温度下的采样分布:")
    for temp in [0.1, 0.5, 1.0, 2.0]:
        scaled = logits / temp
        probs = softmax(scaled)
        top_prob = probs.max()
        top_token = tokens[np.argmax(probs)]
        print(f"  T={temp}: 最高概率={top_prob:.4f} ({top_token})")

test_sampling()
```

这个实战项目综合运用了前面章节的知识，实现了一个完整但简化的文本生成系统。虽然与真实的 GPT 相去甚远，但它展示了语言模型生成文本的核心原理。
