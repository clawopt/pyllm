# GPT-like文本生成概述

GPT-like 模型是基于 Transformer 解码器的自回归语言模型，其核心思想是根据前面的 token 序列预测下一个 token。本章通过四个实战项目，从词嵌入开始，逐步构建一个简化但完整的文本生成系统。通过这个过程，你将理解 GPT 等大型语言模型是如何工作的。

## 自回归语言模型的基本原理

自回归语言模型的核心是：给定前面的 token 序列，预测下一个 token。训练时，我们最大化似然概率 P(x_{t+1} | x_1, ..., x_t)；推理时，我们根据预测的概率分布采样下一个 token，然后将新 token 加入序列，重复这个过程。

```python
import numpy as np

class Vocabulary:
    """简单词汇表"""

    def __init__(self, tokens):
        self.token_to_idx = {t: i for i, t in enumerate(tokens)}
        self.idx_to_token = {i: t for i, t in enumerate(tokens)}

    def encode(self, tokens):
        return [self.token_to_idx.get(t, 0) for t in tokens]

    def decode(self, indices):
        return [self.idx_to_token.get(i, '[UNK]') for i in indices]

# 示例词汇表
tokens = ['[PAD]', '[UNK]', '[BOS]', '[EOS]', 'hello', 'world', 'the', 'quick', 'brown', 'fox', 'jumps', 'over', 'lazy', 'dog']
vocab = Vocabulary(tokens)

print("词汇表大小:", len(vocab.token_to_idx))
print("编码 'hello world':", vocab.encode(['hello', 'world']))
print("解码 [4, 5]:", vocab.decode([4, 5]))
```

## 词嵌入回顾

在之前的章节中，我们已经讨论过词嵌入。这里简要回顾：

```python
def create_embedding(vocab_size, embed_dim, init_std=0.02):
    """创建随机初始化的嵌入矩阵"""
    return np.random.randn(vocab_size, embed_dim).astype(np.float32) * init_std

class Embedding:
    """词嵌入层"""
    def __init__(self, vocab_size, embed_dim):
        self.weight = create_embedding(vocab_size, embed_dim)

    def forward(self, token_ids):
        return self.weight[token_ids]

# 示例
vocab_size = len(tokens)
embed_dim = 64
embedding = Embedding(vocab_size, embed_dim)

token_ids = vocab.encode(['hello', 'world'])
token_ids = np.array(token_ids)
embeddings = embedding.forward(token_ids)

print(f"\nToken IDs: {token_ids}")
print(f"嵌入形状: {embeddings.shape}")
```

## 生成任务的数据流

GPT-like 文本生成的数据流如下：

1. 输入：已生成的 token 序列
2. 通过嵌入层：将 token 转换为向量
3. 通过 Transformer 解码器：计算下一个 token 的 logits
4. 通过 Softmax：将 logits 转换为概率
5. 采样：根据概率分布采样下一个 token
6. 重复

```python
def demonstrate_generation_flow():
    """演示文本生成的数据流"""
    print("\n=== GPT-like 文本生成流程 ===\n")
    print("1. 输入: token 序列 [BOS, hello, world]")
    print("2. 嵌入: [emb(BOS), emb(hello), emb(world)]")
    print("3. Transformer: 计算 logits")
    print("4. Softmax: 计算概率分布 P(next | context)")
    print("5. 采样: 根据概率选择下一个 token")
    print("6. 输出: 新 token，追加到序列")

demonstrate_generation_flow()
```

## 准备工作：定义模型超参数

```python
class ModelConfig:
    """模型配置"""
    def __init__(self):
        self.vocab_size = 10000
        self.embed_dim = 256
        self.num_heads = 8
        self.num_layers = 6
        self.seq_len = 128
        self.ff_dim = 1024
        self.dropout = 0.1

config = ModelConfig()
print("模型配置:")
print(f"  词汇表大小: {config.vocab_size}")
print(f"  嵌入维度: {config.embed_dim}")
print(f"  注意力头数: {config.num_heads}")
print(f"  层数: {config.num_layers}")
print(f"  序列长度: {config.seq_len}")
```

## 完整的文本生成器框架

```python
class SimpleTextGenerator:
    """简化的文本生成器"""

    def __init__(self, config, vocab):
        self.config = config
        self.vocab = vocab

        # 嵌入层
        self.token_embedding = Embedding(config.vocab_size, config.embed_dim)

        # 位置编码（简化版）
        self.position_embedding = create_embedding(config.seq_len, config.embed_dim, init_std=0.01)

        # Transformer 解码器（简化版）
        self.decoder_layers = self._build_decoder()

        # 输出层
        self.lm_head = create_embedding(config.embed_dim, config.vocab_size, init_std=0.02)

    def _build_decoder(self):
        """构建解码器层（简化版）"""
        layers = []
        for _ in range(self.config.num_layers):
            layer = {
                'attn_weights': None  # 存储注意力权重
            }
            layers.append(layer)
        return layers

    def forward(self, token_ids):
        """前向传播"""
        seq_len = len(token_ids)

        # 词嵌入
        token_emb = self.token_embedding.forward(token_ids)

        # 位置嵌入
        position_ids = np.arange(seq_len)
        position_emb = self.position_embedding[position_ids]

        # 相加
        hidden_states = token_emb + position_emb

        # 简化的 Transformer 处理（实际应该过 decoder layers）
        # 这里只做嵌入层面的演示
        logits = hidden_states @ self.lm_head.T

        return logits

    def generate(self, start_tokens, max_len=50, temperature=1.0):
        """生成文本"""
        generated = list(start_tokens)

        for _ in range(max_len):
            token_ids = np.array(self.vocab.encode(generated))
            logits = self.forward(token_ids)

            # 取最后一个 token 的 logits
            next_logits = logits[-1]

            # 温度采样
            probs = self._apply_temperature(next_logits, temperature)
            next_token_id = np.random.choice(len(probs), p=probs)

            next_token = self.vocab.decode([next_token_id])
            generated.append(next_token)

            if next_token == '[EOS]':
                break

        return generated

    def _apply_temperature(self, logits, temperature):
        """应用温度参数"""
        if temperature == 0:
            # 贪心解码
            return np.eye(len(logits))[np.argmax(logits)]

        # 温度采样
        exp_logits = np.exp(logits / temperature)
        probs = exp_logits / np.sum(exp_logits)
        return probs

# 示例
generator = SimpleTextGenerator(config, vocab)
start = ['[BOS]']
generated = generator.generate(start, max_len=20, temperature=1.0)
print(f"\n生成的文本: {' '.join(generated)}")
```

## 下一步预告

接下来的三篇文章将深入讲解：
- Transformer 解码器的具体实现
- 温度采样的详细原理和实现
- 完整的文本生成流程

让我们开始深入探索这些核心组件。
