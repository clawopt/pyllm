# BLEU与ROUGE指标

BLEU（Bilingual Evaluation Understudy）和 ROUGE（Recall-Oriented Understudy for Gushing Evaluation）是文本生成评估中常用的指标。本篇文章介绍如何使用 NumPy 计算这些指标中的统计量。

## BLEU 指标概述

BLEU 通过计算 n-gram 精确度来衡量生成文本与参考文本的相似度：

```python
import numpy as np

def compute_ngram_precision(generated, reference, n):
    """计算 n-gram 精确度

    参数:
        generated: 生成的 token 列表
        reference: 参考 token 列表
        n: n-gram 大小
    返回:
        precision: n-gram 精确度
    """
    def get_ngrams(tokens, n):
        return tuple(tokens[i:i+n] for i in range(len(tokens)-n+1))

    gen_ngrams = get_ngrams(generated, n)
    ref_ngrams = get_ngrams(reference, n)

    if not gen_ngrams:
        return 0.0

    # 统计生成的 n-gram 在参考中出现的次数
    matches = sum(1 for ng in gen_ngrams if ng in ref_ngrams)

    return matches / len(gen_ngrams)

# 示例
generated = ['the', 'cat', 'eats', 'the', 'cat', 'food']
reference = ['the', 'cat', 'eats', 'food']

for n in range(1, 5):
    precision = compute_ngram_precision(generated, reference, n)
    print(f"{n}-gram 精确度: {precision:.4f}")
```

## ROUGE 指标概述

ROUGE 通过计算召回率来衡量生成文本的质量：

```python
def compute_rouge_l(generated, reference):
    """计算 ROUGE-L（最长公共子序列）

    参数:
        generated: 生成的 token 列表
        reference: 参考 token 列表
    返回:
        rouge_l: ROUGE-L 分数
    """
    def lcs_length(s1, s2):
        m, n = len(s1), len(s2)
        dp = np.zeros((m+1, n+1), dtype=np.int32)

        for i in range(1, m+1):
            for j in range(1, n+1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])

        return dp[m][n]

    lcs_len = lcs_length(generated, reference)

    # ROUGE-L 召回率
    recall = lcs_len / len(reference) if reference else 0

    # ROUGE-L 精确率
    precision = lcs_len / len(generated) if generated else 0

    # F1 分数
    if recall + precision > 0:
        f1 = 2 * recall * precision / (recall + precision)
    else:
        f1 = 0

    return {'precision': precision, 'recall': recall, 'f1': f1}

# 示例
generated = ['the', 'cat', 'eats', 'the', 'cat', 'food']
reference = ['the', 'cat', 'eats', 'food']

rouge = compute_rouge_l(generated, reference)
print(f"ROUGE-L: precision={rouge['precision']:.4f}, recall={rouge['recall']:.4f}, f1={rouge['f1']:.4f}")
```

## BLEU 分数计算

```python
def compute_bleu(generated, reference, max_n=4):
    """计算 BLEU 分数

    参数:
        generated: 生成的 token 列表
        reference: 参考 token 列表
        max_n: 最大 n-gram
    返回:
        bleu: BLEU 分数
    """
    # 计算各阶 n-gram 精确度
    precisions = []
    for n in range(1, max_n + 1):
        p = compute_ngram_precision(generated, reference, n)
        precisions.append(p)

    # 避免 log(0)
    precisions = [max(p, 1e-10) for p in precisions]

    # 几何平均
    log_precisions = np.log(precisions)
    avg_log_precision = np.mean(log_precisions)

    # 简短惩罚
    gen_len = len(generated)
    ref_len = len(reference)
    if gen_len < ref_len:
        bp = np.exp(1 - ref_len / gen_len) if gen_len > 0 else 0
    else:
        bp = 1.0

    bleu = bp * np.exp(avg_log_precision)

    return bleu

# 示例
generated = ['the', 'cat', 'eats', 'the', 'cat', 'food']
reference = ['the', 'cat', 'eats', 'food']

bleu = compute_bleu(generated, reference)
print(f"BLEU 分数: {bleu:.4f}")
```

掌握 BLEU 和 ROUGE 指标的计算方法对于评估文本生成模型非常重要。
