# 困惑度计算

困惑度（Perplexity，PPL）是语言模型最常用的评估指标之一。它衡量模型对序列的预测能力，困惑度越低表示模型越好。本篇文章详细介绍困惑度的计算原理和纯 NumPy 实现。

## 困惑度的定义

困惑度定义为：

```
PPL = exp(-(1/N) * Σ log P(x_i))
```

或者等价于：

```
PPL = exp(average_negative_log_likelihood)
```

其中 N 是 token 数量，log P(x_i) 是模型对第 i 个 token 预测的对数概率。

```python
import numpy as np

def compute_perplexity(log_likelihoods):
    """计算困惑度

    参数:
        log_likelihoods: 每个 token 的对数似然，形状 (N,)
    返回:
        perplexity: 困惑度
    """
    # 平均负对数似然
    avg_nll = -np.mean(log_likelihoods)

    # 困惑度
    perplexity = np.exp(avg_nll)

    return perplexity

# 示例
log_probs = np.array([-0.5, -1.0, -0.8, -1.2, -0.6])
ppl = compute_perplexity(log_probs)

print(f"对数似然: {log_probs}")
print(f"平均负对数似然: {-np.mean(log_probs):.4f}")
print(f"困惑度: {ppl:.4f}")
```

## 交叉熵与困惑度的关系

在训练语言模型时，我们通常最小化交叉熵损失。困惑度是交叉熵的指数形式：

```python
def cross_entropy_to_perplexity(cross_entropy):
    """交叉熵转困惑度"""
    return np.exp(cross_entropy)

def perplexity_to_cross_entropy(perplexity):
    """困惑度转交叉熵"""
    return np.log(perplexity)

# 示例
cross_entropy = 3.5
ppl = cross_entropy_to_perplexity(cross_entropy)
print(f"交叉熵={cross_entropy:.2f} -> 困惑度={ppl:.2f}")

ppl = 33.0
ce = perplexity_to_cross_entropy(ppl)
print(f"困惑度={ppl:.2f} -> 交叉熵={ce:.2f}")
```

## 分段困惑度

对于长序列，可以计算分段困惑度：

```python
def compute_segment_perplexity(log_likelihoods, segment_length=128):
    """计算分段困惑度

    参数:
        log_likelihoods: 对数似然数组
        segment_length: 分段长度
    返回:
        segment_ppls: 每段的困惑度
    """
    n = len(log_likelihoods)
    segment_ppls = []

    for i in range(0, n, segment_length):
        segment = log_likelihoods[i:i+segment_length]
        ppl = compute_perplexity(segment)
        segment_ppls.append(ppl)

    return np.array(segment_ppls)

# 示例
log_probs = np.random.randn(512) * 2 - 1  # 模拟对数似然
segment_ppls = compute_segment_perplexity(log_probs, segment_length=128)

print("分段困惑度:")
for i, ppl in enumerate(segment_ppls):
    print(f"  段 {i+1}: {ppl:.2f}")
```

## 批量困惑度计算

```python
def compute_batch_perplexity(log_likelihoods, attention_mask=None):
    """计算批量困惑度

    参数:
        log_likelihoods: (batch, seq_len) 每个 token 的对数似然
        attention_mask: (batch, seq_len) 有效位置掩码
    返回:
        perplexity: 批量平均困惑度
    """
    if attention_mask is not None:
        # 只考虑有效位置
        masked_logits = log_likelihoods * attention_mask
        total_tokens = attention_mask.sum()
        avg_nll = -masked_logits.sum() / total_tokens
    else:
        avg_nll = -np.mean(log_likelihoods)

    return np.exp(avg_nll)

# 示例
batch_size = 8
seq_len = 128

log_probs = np.random.randn(batch_size, seq_len) * 2 - 1
mask = np.random.randint(0, 2, size=(batch_size, seq_len))

ppl = compute_batch_perplexity(log_probs, mask)
print(f"批量困惑度: {ppl:.2f}")
```

## 实际应用：评估语言模型

```python
def evaluate_model_perplexity(model, dataset):
    """评估模型在数据集上的困惑度

    参数:
        model: 语言模型（有 forward 方法）
        dataset: 数据集（NumPy 数组）
    返回:
        avg_ppl: 平均困惑度
    """
    total_loss = 0.0
    total_tokens = 0

    for i in range(len(dataset)):
        input_ids = dataset[i]

        # 获取模型预测
        logits = model.forward(input_ids)

        # 计算负对数似然
        loss = compute_negative_log_likelihood(logits, input_ids)
        total_loss += loss
        total_tokens += len(input_ids) - 1  # 预测下一个 token

    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)

    return perplexity

def compute_negative_log_likelihood(logits, target_ids):
    """计算负对数似然

    参数:
        logits: (seq_len, vocab_size) 模型输出
        target_ids: (seq_len,) 目标 token IDs
    返回:
        loss: 负对数似然
    """
    # 移位：预测第 i+1 个 token 时使用第 i 个 token 的 logits
    target_logits = logits[:-1]
    target_tokens = target_ids[1:]

    # 计算交叉熵
    log_probs = target_logits - np.log(np.sum(np.exp(target_logits - np.max(target_logits, axis=-1, keepdims=True)), axis=-1, keepdims=True))
    loss = -log_probs[np.arange(len(target_tokens)), target_tokens]

    return np.sum(loss)

# 示例
print("困惑度评估示例完成")
```

## 常见问题与处理

### 数值稳定性

```python
def stable_perplexity(log_likelihoods, eps=1e-9):
    """数值稳定的困惑度计算

    当对数似然非常负时，exp 可能下溢
    """
    avg_nll = -np.mean(log_likelihoods)

    # 限制范围避免溢出
    avg_nll = min(avg_nll, 700)  # exp(700) 约等于 inf

    return np.exp(avg_nll)

# 测试
log_probs = np.array([-100, -200, -300, -100])
ppl = stable_perplexity(log_probs)
print(f"数值稳定的困惑度: {ppl:.2f}")
```

### 困惑度的解释

```python
def interpret_perplexity(ppl):
    """解释困惑度"""
    print(f"\n=== 困惑度解释 ===")
    print(f"困惑度 = {ppl:.2f}\n")

    if ppl < 10:
        print("非常低的困惑度，模型预测非常准确")
    elif ppl < 30:
        print("较低的困惑度，模型表现良好")
    elif ppl < 100:
        print("中等的困惑度，模型表现一般")
    elif ppl < 300:
        print("较高的困惑度，模型预测较困难")
    else:
        print("非常高的困惑度，模型预测几乎随机")

interpret_perplexity(25.0)
```

困惑度是评估语言模型的核心指标。掌握其计算方法对于正确评估模型性能非常重要。
