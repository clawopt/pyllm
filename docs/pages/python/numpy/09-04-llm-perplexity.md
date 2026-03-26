# LLM场景示例：困惑度计算综合实践

在前几节中，我们学习了统计方法和各种数学函数。在这一节中，让我们通过计算语言模型的困惑度（Perplexity）来综合运用这些知识。困惑度是评估语言模型性能的最重要指标之一，它可以理解为模型预测下一个词时的"平均不确定性"。困惑度越低，模型越好。理解困惑度的计算不仅能帮助我们评估模型，还能加深对语言模型工作原理的理解。

## 困惑度的定义

困惑度（Perplexity）的数学定义是：

```
PPL = exp(-1/N * Σ log P(x_i))
    = exp(CE)
```

其中 CE 是交叉熵损失的平均值，N 是 token 的数量。

从直观上理解，困惑度可以解释为：模型在预测下一个词时，平均有多少种可能的词需要考虑。如果困惑度是 10，意味着模型在每个位置平均需要考虑约 10 个可能的词。

## 交叉熵损失

交叉熵是衡量两个分布差异的指标，在语言模型中，我们用交叉熵来衡量模型预测分布与真实分布的差异：

```python
def cross_entropy_loss(logits, labels):
    """计算交叉熵损失

    Args:
        logits: 模型输出的未归一化 logit，形状 (batch, seq_len, vocab_size)
        labels: 真实的 token ID，形状 (batch, seq_len)

    Returns:
        每个样本的平均交叉熵损失
    """
    vocab_size = logits.shape[-1]

    # Log-Softmax
    log_probs = logits - logits.max(axis=-1, keepdims=True)
    log_probs = log_probs - np.log(np.exp(log_probs).sum(axis=-1, keepdims=True) + 1e-10)

    # 收集每个位置的 log 概率
    batch_size, seq_len = labels.shape
    log_probs_for_labels = log_probs[np.arange(batch_size)[:, None], np.arange(seq_len), labels]

    # 交叉熵 = -log(p)
    ce_loss = -log_probs_for_labels

    # 返回平均损失
    return ce_loss.mean()
```

## 完整的困惑度计算

让我们实现一个完整的困惑度计算函数：

```python
def perplexity(logits, labels):
    """计算困惑度

    PPL = exp(-1/N * Σ log P(x_i))
        = exp(CE)

    Args:
        logits: 模型输出的未归一化 logit
        labels: 真实的 token ID

    Returns:
        困惑度
    """
    vocab_size = logits.shape[-1]

    # 计算 log-softmax
    log_probs = logits - logits.max(axis=-1, keepdims=True)
    log_probs = log_probs - np.log(np.exp(log_probs).sum(axis=-1, keepdims=True) + 1e-10)

    # 收集每个位置的 log 概率
    batch_size, seq_len = labels.shape
    log_probs_for_labels = log_probs[np.arange(batch_size)[:, None], np.arange(seq_len), labels]

    # 计算交叉熵损失
    nll_loss = -log_probs_for_labels  # NLL: Negative Log Likelihood

    # 平均
    avg_nll_loss = nll_loss.mean()

    # 困惑度
    ppl = np.exp(avg_nll_loss)

    return ppl, avg_nll_loss

# 测试
batch_size = 4
seq_len = 128
vocab_size = 30000

# 模拟模型输出（logits）
np.random.seed(42)
logits = np.random.randn(batch_size, seq_len, vocab_size).astype(np.float32)

# 模拟真实标签
labels = np.random.randint(0, vocab_size, size=(batch_size, seq_len))

ppl, nll = perplexity(logits, labels)
print(f"平均负对数似然 (NLL): {nll:.4f}")
print(f"困惑度 (PPL): {ppl:.2f}")
```

## 分段困惑度

在实际应用中，我们经常需要计算分段困惑度，以便更细致地评估模型：

```python
def segment_perplexity(logits, labels, segment_length=64):
    """计算分段困惑度

    将序列分成多个段，分别计算每个段的困惑度
    """
    batch_size, seq_len, vocab_size = logits.shape
    num_segments = seq_len // segment_length

    # 截断序列使其可以被整除
    logits = logits[:, :num_segments * segment_length, :]
    labels = labels[:, :num_segments * segment_length]

    # Reshape 为 (batch * num_segments, segment_length, vocab_size)
    logits_reshaped = logits.reshape(batch_size, num_segments, segment_length, vocab_size)
    logits_reshaped = logits_reshaped.reshape(batch_size * num_segments, segment_length, vocab_size)

    labels_reshaped = labels.reshape(batch_size, num_segments, segment_length)
    labels_reshaped = labels_reshaped.reshape(batch_size * num_segments, segment_length)

    # 计算每个段的困惑度
    segment_ppls = []
    for i in range(batch_size * num_segments):
        ppl, _ = perplexity(logits_reshaped[i:i+1], labels_reshaped[i:i+1])
        segment_ppls.append(ppl)

    return np.array(segment_ppls).reshape(batch_size, num_segments)

# 计算分段困惑度
segment_ppls = segment_perplexity(logits, labels, segment_length=32)
print(f"分段困惑度形状: {segment_ppls.shape}")
print(f"第一样本各段困惑度: {segment_ppls[0]}")
print(f"平均困惑度: {segment_ppls.mean():.2f}")
```

## 使用掩码计算有效长度的困惑度

在实际数据中，不同样本的序列长度可能不同，需要使用掩码：

```python
def masked_perplexity(logits, labels, attention_mask=None):
    """计算带掩码的困惑度

    只在有效 token 上计算困惑度
    """
    vocab_size = logits.shape[-1]

    # 计算 log-softmax
    log_probs = logits - logits.max(axis=-1, keepdims=True)
    log_probs = log_probs - np.log(np.exp(log_probs).sum(axis=-1, keepdims=True) + 1e-10)

    # 收集每个位置的 log 概率
    batch_size, seq_len = labels.shape
    log_probs_for_labels = log_probs[np.arange(batch_size)[:, None], np.arange(seq_len), labels]

    # 计算 NLL
    nll_loss = -log_probs_for_labels

    # 应用掩码
    if attention_mask is not None:
        # 将无效位置的损失设为 0
        nll_loss = nll_loss * attention_mask

        # 计算有效 token 的数量
        valid_tokens = attention_mask.sum()

        # 计算平均损失
        avg_nll_loss = nll_loss.sum() / valid_tokens
    else:
        avg_nll_loss = nll_loss.mean()

    # 困惑度
    ppl = np.exp(avg_nll_loss)

    return ppl, avg_nll_loss

# 测试带掩码的困惑度
attention_mask = np.ones((batch_size, seq_len))
# 假设第一个样本只有前 100 个 token 是有效的
attention_mask[0, 100:] = 0

ppl, nll = masked_perplexity(logits, labels, attention_mask)
print(f"带掩码的困惑度: {ppl:.2f}")
```

## 批量困惑度计算

对于大批量数据，需要高效地计算困惑度：

```python
def batch_perplexity(logits, labels):
    """批量计算困惑度

    支持批量输入，高效计算
    """
    # 转换为 float64 提高精度
    logits = logits.astype(np.float64)

    # 计算 log-softmax
    log_probs = logits - logits.max(axis=-1, keepdims=True)
    log_probs = log_probs - np.log(np.exp(log_probs).sum(axis=-1, keepdims=True) + 1e-10)

    # 收集标签对应的 log 概率
    batch_size, seq_len, vocab_size = logits.shape
    indices = np.arange(batch_size)[:, None], np.arange(seq_len), labels
    log_probs_for_labels = log_probs[indices]

    # NLL
    nll_loss = -log_probs_for_labels

    # 平均（跨 batch 和 seq_len）
    avg_nll_loss = nll_loss.mean()

    # 困惑度
    ppl = np.exp(avg_nll_loss)

    return ppl, avg_nll_loss

# 大批量测试
large_batch_size = 32
large_seq_len = 512
large_vocab_size = 50000

logits_large = np.random.randn(large_batch_size, large_seq_len, large_vocab_size).astype(np.float32)
labels_large = np.random.randint(0, large_vocab_size, size=(large_batch_size, large_seq_len))

ppl, nll = batch_perplexity(logits_large, labels_large)
print(f"大批量困惑度: {ppl:.2f}")
print(f"大批量 NLL: {nll:.4f}")
```

## 困惑度的解释

困惑度的取值范围通常是 1 到 vocab_size：

```python
# 完美模型的困惑度是 1
# 这意味着模型总是 100% 确信任务 token

# 困惑度等于词汇量意味着模型预测均匀分布
# 即模型对所有词的概率相同，完全随机

# 实际语言模型的困惑度通常在 10-100 之间
# GPT-3 的困惑度据报道约为 20-30
print("困惑度解释:")
print("  1: 完美预测")
print("  10: 平均考虑 10 个词")
print("  100: 模型非常不确定")
```

## 常见问题与解决方案

### 数值溢出

```python
# 当 log 概率很小时，直接 exp 可能会下溢
small_nll = -100  # log(0) 的情况

try:
    ppl = np.exp(small_nll)
    print(f"困惑度: {ppl}")  # 非常小
except:
    print("数值问题")

# 解决方案：使用 np.exp(-np.clip(nll_loss, 0, 700))
# numpy 的 exp 在超过 700 左右时会溢出
max_nll = 100
safe_ppl = np.exp(-np.clip(-small_nll, -max_nll, max_nll))
```

### 序列长度的影响

```python
# 不同序列长度计算出的困惑度不能直接比较
# 因为较长序列的平均 NLL 会趋向于稳定

# 解决方案：使用相同的序列长度进行对比
```

## 小结

这一节我们通过困惑度计算，综合运用了前几节学到的统计方法和数学函数。关键点包括：

1. **交叉熵损失**：-log P(x) 是基本单元
2. **平均 NLL**：需要对所有 token 的 NLL 取平均
3. **困惑度公式**：exp(NLL)
4. **数值稳定性**：需要注意 exp 的溢出问题

困惑度是评估语言模型的核心指标，理解其计算原理对于评估和优化 LLM 至关重要。

面试时需要能够解释困惑度的定义，理解困惑度与交叉熵的关系，以及能够实现完整的困惑度计算函数。
