# 文本多样性评估

文本多样性是评估生成文本质量的重要指标。高质量的生成文本应该既准确又多样，避免重复和单调。本篇文章介绍如何使用 NumPy 计算文本多样性的各种指标。

## N-gram 重复率

重复率是最基本的多样性指标：

```python
import numpy as np
from collections import Counter

def compute_ngram_repetition(tokens, n=2):
    """计算 n-gram 重复率

    参数:
        tokens: token 列表
        n: n-gram 大小
    返回:
        repetition_rate: 重复率 (0-1)
    """
    if len(tokens) < n:
        return 0.0

    # 生成 n-grams
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

    if not ngrams:
        return 0.0

    # 计算重复
    ngram_counts = Counter(ngrams)
    repeated = sum(1 for count in ngram_counts.values() if count > 1)
    total = len(ngrams)

    return repeated / total

# 示例
tokens = ['the', 'cat', 'eats', 'the', 'cat', 'food', 'the', 'cat']

for n in [1, 2, 3]:
    rep = compute_ngram_repetition(tokens, n)
    print(f"{n}-gram 重复率: {rep:.4f}")
```

## 唯一 token 比例

```python
def compute_unique_token_ratio(tokens):
    """计算唯一 token 比例

    参数:
        tokens: token 列表
    返回:
        unique_ratio: 唯一 token 比例
    """
    if not tokens:
        return 0.0

    unique = len(set(tokens))
    return unique / len(tokens)

# 示例
tokens1 = ['the', 'cat', 'eats', 'the', 'cat', 'food']
tokens2 = ['a', 'b', 'c', 'd', 'e', 'f']

print(f"重复文本唯一率: {compute_unique_token_ratio(tokens1):.4f}")
print(f"多样文本唯一率: {compute_unique_token_ratio(tokens2):.4f}")
```

## Token 熵

熵衡量分布的不确定性，高熵表示高多样性：

```python
def compute_token_entropy(tokens):
    """计算 token 分布的熵

    参数:
        tokens: token 列表
    返回:
        entropy: 熵 (bits)
    """
    if not tokens:
        return 0.0

    # 计算 token 频率
    counts = Counter(tokens)
    total = len(tokens)

    # 计算熵
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * np.log2(p)

    return entropy

# 示例
tokens = ['the', 'cat', 'eats', 'the', 'cat', 'food']
entropy = compute_token_entropy(tokens)
print(f"Token 熵: {entropy:.2f} bits")
```

## 序列级多样性

```python
def compute_sequence_diversity(sequences):
    """计算序列级别的多样性

    参数:
        sequences: 序列列表
    返回:
        diversity: 多样性分数
    """
    if not sequences:
        return 0.0

    # 计算序列间的平均距离
    diversities = []
    for i, seq1 in enumerate(sequences):
        for j, seq2 in enumerate(sequences):
            if i < j:
                # 使用 Jaccard 距离
                set1, set2 = set(seq1), set(seq2)
                jaccard = len(set1 & set2) / len(set1 | set2) if set1 | set2 else 0
                diversities.append(1 - jaccard)

    return np.mean(diversities) if diversities else 0.0

# 示例
sequences = [
    ['hello', 'world'],
    ['hello', 'world'],
    ['goodbye', 'world']
]

diversity = compute_sequence_diversity(sequences)
print(f"序列多样性: {diversity:.4f}")
```

## 综合多样性评估

```python
def evaluate_text_diversity(generated_texts):
    """综合评估文本多样性

    参数:
        generated_texts: 生成的文本列表
    返回:
        metrics: 多样性指标字典
    """
    metrics = {}

    # 1. 唯一 token 比例
    all_tokens = [token for text in generated_texts for token in text]
    metrics['unique_token_ratio'] = compute_unique_token_ratio(all_tokens)

    # 2. Token 熵
    metrics['token_entropy'] = compute_token_entropy(all_tokens)

    # 3. 2-gram 重复率
    metrics['bigram_repetition'] = compute_ngram_repetition(all_tokens, n=2)

    # 4. 序列多样性
    metrics['sequence_diversity'] = compute_sequence_diversity(generated_texts)

    return metrics

# 示例
generated = [
    ['hello', 'world', 'the', 'quick', 'brown', 'fox'],
    ['hello', 'world', 'the', 'lazy', 'dog'],
    ['goodbye', 'world', 'the', 'smart', 'cat']
]

metrics = evaluate_text_diversity(generated)
print("多样性指标:")
for name, value in metrics.items():
    print(f"  {name}: {value:.4f}")
```

掌握文本多样性评估对于生成高质量、多样化的文本非常重要。
