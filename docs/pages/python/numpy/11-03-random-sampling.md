# 随机采样

随机采样是从已知集合中随机选择元素的过程，在深度学习中有广泛应用。在语言模型训练中，我们需要从词汇表中采样下一个 token；在数据加载时，我们需要从大规模数据集中随机采样 mini-batch；在模型评估时，我们可能使用随机采样来生成文本。NumPy 提供了多种随机采样函数，其中最核心的是 `np.random.choice`，它可以从一维数组或整数范围中随机采样。

## np.random.choice：基础随机采样

`np.random.choice` 是 NumPy 中最常用的随机采样函数，它有以下几个关键参数：

```python
import numpy as np

# 从数组中随机选择一个元素
np.random.seed(42)
choice = np.random.choice([1, 2, 3, 4, 5])
print(f"随机选择: {choice}")

# 从数组中随机选择 k 个元素（不替换，即不重复）
choices = np.random.choice([1, 2, 3, 4, 5], k=3)
print(f"随机选择3个: {choices}")

# 从数组中随机选择 k 个元素（替换，即允许重复）
choices_with_replacement = np.random.choice([1, 2, 3, 4, 5], k=10, replace=True)
print(f"有放回选择10个: {choices_with_replacement}")
```

### 参数详解

`np.random.choice(a, size=None, replace=True, p=None)` 的参数含义：

- `a`：如果是一维数组，从中进行采样；如果是整数，从 `np.arange(a)` 中采样。
- `size`：输出形状，如 `None` 返回单个元素，`(3,)` 返回一维数组，`(3, 4)` 返回二维数组。
- `replace`：是否允许重复采样。默认为 `True`（有放回采样）。
- `p`：每个元素被选中的概率分布。如果为 `None`，则是均匀分布。

```python
# 从整数范围采样
np.random.seed(42)
samples = np.random.choice(10, size=5)  # 等价于 np.random.randint(0, 10, size=5)
print(f"从 0-9 采样: {samples}")

# 指定概率分布
# 例如：在词汇表中，"the" 可能比 "ephemeral" 出现的概率高得多
vocab = ["the", "cat", "dog", "is", "on"]
# 假设词汇表频率
word_freq = [0.15, 0.1, 0.1, 0.1, 0.55]
samples = np.random.choice(vocab, size=10, p=word_freq)
print(f"按概率采样: {samples}")
```

## 在LLM场景中的应用

### 词汇表采样与温度

语言模型生成文本时，最简单的策略是选择概率最高的 token（贪心解码），但这往往导致重复内容。温度采样（Temperature Sampling）是一种常用的策略，它通过调整 softmax 的"温度"来控制输出的随机性。

```python
def temperature_sampling(logits, temperature=1.0):
    """温度采样

    参数:
        logits: 模型输出的原始 logits (vocab_size,)
        temperature: 温度参数，>1 增加随机性，<1 减少随机性
    返回:
        选中的 token id
    """
    if temperature == 0:
        return np.argmax(logits)

    # 应用温度：logits / temperature
    logits = logits / temperature

    # Softmax 得到概率分布
    exp_logits = np.exp(logits - np.max(logits))  # 数值稳定化
    probs = exp_logits / np.sum(exp_logits)

    # 按概率采样
    token_id = np.random.choice(len(logits), p=probs)
    return token_id

# 模拟 logits
np.random.seed(42)
logits = np.random.randn(50257)  # GPT-2 词汇表大小

# 不同温度的效果
print("温度采样示例:")
for temp in [0.1, 0.5, 1.0, 1.5, 2.0]:
    np.random.seed(42)
    selected = temperature_sampling(logits, temperature=temp)
    print(f"  温度={temp}: 选中 token {selected}")
```

**温度的理解**：温度参数源于统计物理，在 softmax 中相当于"模糊"或"锐化"概率分布。温度 T→0 时，分布趋近于 one-hot（确定性选择最大概率）；温度 T→∞ 时，分布趋近于均匀（完全随机）；温度 T=1 时，是原始 softmax 分布。

```python
# 温度对分布的影响可视化
logits = np.array([2.0, 1.0, 0.5, 0.1])

for temp in [0.1, 0.5, 1.0, 2.0]:
    scaled = logits / temp
    exp_logits = np.exp(scaled - np.max(scaled))
    probs = exp_logits / np.sum(exp_logits)
    print(f"温度={temp}: {probs}")
    # temp=0.1 时，最大概率接近 1
    # temp=2.0 时，分布更均匀
```

### Top-K 采样

Top-K 采样限制模型只从概率最高的 K 个 token 中选择，这可以避免采样到极低概率的 token，从而提高生成质量。

```python
def top_k_sampling(logits, k=50, temperature=1.0):
    """Top-K 采样

    参数:
        logits: 模型输出的 logits (vocab_size,)
        k: 只考虑概率最高的 k 个 token
        temperature: 温度参数
    返回:
        选中的 token id
    """
    # 应用温度
    if temperature != 1.0:
        logits = logits / temperature

    # 找 top-k 的索引
    top_k_indices = np.argpartition(logits, -k)[-k:]

    # 只保留 top-k 的 logits，其他设为 -inf
    masked_logits = np.full_like(logits, -np.inf)
    masked_logits[top_k_indices] = logits[top_k_indices]

    # 转为概率分布并采样
    exp_logits = np.exp(masked_logits - np.max(masked_logits))
    probs = exp_logits / np.sum(exp_logits)

    return np.random.choice(len(logits), p=probs)

# 测试 Top-K 采样
np.random.seed(42)
logits = np.random.randn(50257)

print("Top-K 采样示例:")
for k in [1, 10, 50, 200]:
    np.random.seed(42)
    selected = top_k_sampling(logits, k=k)
    print(f"  Top-{k}: 选中 token {selected}")
```

### Top-P（核采样）

Top-P 采样（又称核采样，Nucleus Sampling）是另一种常用策略。它不是固定 K 个 token，而是动态选择累积概率达到阈值 P 的最小 token 集合。

```python
def top_p_sampling(logits, p=0.9, temperature=1.0):
    """Top-P（核）采样

    参数:
        logits: 模型输出的 logits (vocab_size,)
        p: 累积概率阈值
        temperature: 温度参数
    返回:
        选中的 token id
    """
    if temperature != 1.0:
        logits = logits / temperature

    # 转为概率分布
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(probs)

    # 按概率排序
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]

    # 找到累积概率达到 p 的最小集合
    cumsum = np.cumsum(sorted_probs)
    cutoff_idx = np.searchsorted(cumsum, p) + 1

    # 只保留这个集合
    top_p_indices = sorted_indices[:cutoff_idx]
    top_p_probs = probs[top_p_indices]
    top_p_probs = top_p_probs / np.sum(top_p_probs)  # 重新归一化

    return np.random.choice(top_p_indices, p=top_p_probs)

# 注意：上面的实现有个 bug，正确版本如下：
def top_p_sampling_correct(logits, p=0.9, temperature=1.0):
    """Top-P（核）采样 - 正确实现"""
    if temperature != 1.0:
        logits = logits / temperature

    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)

    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]

    cumsum = np.cumsum(sorted_probs)
    cutoff_idx = np.searchsorted(cumsum, p) + 1
    top_p_indices = sorted_indices[:cutoff_idx]
    top_p_probs = probs[top_p_indices]
    top_p_probs = top_p_probs / np.sum(top_p_probs)

    return np.random.choice(top_p_indices, p=top_p_probs)
```

### 束搜索（Beam Search）简介

虽然不是随机采样，但束搜索是语言模型生成中另一种重要的解码策略。束搜索维护多个候选序列（称为束），而不是只选择一个。这比贪心解码质量更高，但比纯随机采样更确定性。

```python
def beam_search_decode(probs_func, beam_width=5, max_length=50):
    """简化的束搜索解码

    参数:
        probs_func: 返回下一个 token 概率分布的函数
        beam_width: 束宽度
        max_length: 最大生成长度
    """
    # 初始化
    sequences = [([2], 0.0)]  # [(token_ids, log_prob), ...]
    EOS_TOKEN = 2

    for _ in range(max_length):
        all_candidates = []

        for seq, score in sequences:
            if seq[-1] == EOS_TOKEN:
                all_candidates.append((seq, score))
                continue

            # 获取下一个 token 的概率分布
            probs = probs_func(seq)
            top_k = np.argsort(probs)[-beam_width:]

            for token in top_k:
                new_seq = seq + [token]
                new_score = score + np.log(probs[token])
                all_candidates.append((new_seq, new_score))

        # 选择 top beam_width
        sequences = sorted(all_candidates, key=lambda x: x[1])[-beam_width:]

    return sequences[-1][0]  # 返回最佳序列
```

## Mini-Batch 采样

在训练时，我们需要从大规模训练集中随机采样 mini-batch。NumPy 的随机采样功能可以用于实现这一目的。

```python
def sample_mini_batches(data, batch_size, num_batches, rng=None):
    """从数据中随机采样多个 mini-batch

    参数:
        data: 训练数据 (n_samples, ...)
        batch_size: 每个 batch 的大小
        num_batches: 需要采样的 batch 数量
        rng: 随机数生成器
    返回:
        batches: batch 列表
    """
    if rng is None:
        rng = np.random.default_rng()

    n_samples = len(data)
    indices = np.arange(n_samples)
    batches = []

    for _ in range(num_batches):
        # 有放回随机采样 batch_size 个索引
        batch_indices = rng.choice(indices, size=batch_size, replace=True)
        batches.append(data[batch_indices])

    return batches

# 示例
np.random.seed(42)
data = np.random.randn(10000, 768)  # 10000 个样本，768 维特征
rng = np.random.default_rng(42)
batches = sample_mini_batches(data, batch_size=32, num_batches=10, rng=rng)

print(f"采样了 {len(batches)} 个 batch")
print(f"每个 batch 的形状: {batches[0].shape}")
```

## 常见误区

**误区一：混淆 replace=True 和 replace=False**

`replace=True` 是有放回采样，元素可以被重复选中；`replace=False` 是无放回采样，每个元素最多被选中一次。在从词汇表采样生成 token 时应该使用 `replace=True`；在从训练集中采样 mini-batch 时可以使用 `replace=False`。

```python
# 有放回 vs 无放回
np.random.seed(42)
with_replacement = np.random.choice([1, 2, 3], size=5, replace=True)
without_replacement = np.random.choice([1, 2, 3], size=3, replace=False)

print(f"有放回: {with_replacement}")      # 可能包含重复
print(f"无放回: {without_replacement}")   # 不包含重复
```

**误区二：忘记概率分布必须归一化**

当使用 `p` 参数指定概率分布时，`p` 的和必须等于 1。浮点数精度可能导致轻微偏差，但通常不会造成问题。如果需要手动指定概率分布，确保先归一化。

```python
weights = np.array([0.1, 0.2, 0.3])
weights = weights / weights.sum()  # 归一化
samples = np.random.choice([1, 2, 3], size=10, p=weights)
```

**误区三：在推理时不设置种子，但在调试时忘记恢复种子**

在生成文本时，每次调用采样函数都应该是独立随机的。但在调试时，可能需要固定随机性以便复现问题。确保在使用时明确是否需要设置种子。

## API 总结

| 函数 | 描述 |
|------|------|
| `np.random.choice(a, size, replace, p)` | 从数组或整数中随机采样 |
| `np.random.randint(low, high, size)` | 随机整数采样（均匀分布） |
| `rng.choice(a, size, replace, p)` | 推荐使用局部的随机数生成器 |

随机采样是 LLMs 生成文本的核心机制。理解温度采样、Top-K、Top-P 等技术的原理和适用场景，对于调优语言模型的输出质量非常重要。
