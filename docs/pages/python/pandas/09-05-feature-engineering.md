---
title: 特征工程实战：为 LLM 训练构造数据
description: 文本特征提取 / 数值分箱 / 交互特征 / SFT 数据集特征增强，从原始语料到训练就绪的特征矩阵
---
# 特征工程实战

前面几节我们学了列操作、apply 和数据重塑。这一节把它们全部用起来解决一个实际问题：**从 LLM 语料的原始文本中提取有意义的特征**——这些特征可以用于数据质量评估、模型选择、或者作为下游分类/回归模型的输入。

## 文本基础统计特征

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'text': [
        'Transformer 是一种基于自注意力机制的深度神经网络架构...',
        'BERT 通过掩码语言模型进行预训练，能够理解上下文语义',
        'GPT 系列采用自回归方式生成文本，从左到右逐 token 预测',
    ],
})

df['char_len'] = df['text'].str.len()
df['word_count'] = df['text'].str.split().str.len()
df['avg_word_len'] = (df['char_len'] / df['word_count']).round(1)

df['punct_count'] = df['text'].str.count(r'[，。！？、；：""''（）【】]')
df['punct_density'] = (df['punct_count'] / df['char_len']).round(4)

df['chinese_ratio'] = (
    df['text'].str.count(r'[\u4e00-\u9fff]') / df['char_len']
).round(3)
df['code_ratio'] = (df['text'].str.count(r'[a-zA-Z_{}]') / df['char_len']).round(3)

print(df[['char_len', 'word_count', 'chinese_ratio', 'code_ratio']])
```

这些基础统计特征看起来简单，但在实际数据质量评估中非常有价值。比如 `char_len` 可以帮你发现异常短的样本（可能只有一句"好的"），`punct_density` 偏高可能意味着文本包含大量格式标记而非自然语言。

## 数值分箱：把连续值转成离散类别

```python
import pandas as pd
import numpy as np

np.random.seed(42)
n = 100_000
df = pd.DataFrame({
    'quality': np.round(np.random.uniform(1, 5, n), 1),
    'tokens': np.random.randint(10, 4000, n),
})

df['length_tier'] = pd.cut(
    df['tokens'],
    bins=[0, 100, 500, 1500, float('inf')],
    labels=['极短', '短', '中等', '长'],
)
df['quality_bin'] = pd.qcut(df['quality'], q=5, labels=['很差', '较差', '一般', '良好', '优秀'])

print(df.groupby('length_tier', observed=True).size())
print(f"\n质量分箱分布:")
print(df['quality_bin'].value_counts().sort_index())
```

`pd.cut()` 按你指定的边界值分箱（等宽），`pd.qcut()` 按分位数分箱（每个箱的样本数大致相等）。在 LLM 场景中，分箱常用于：
- token 数 → 对话长度等级（决定是否需要截断）
- 质量分数 → 质量等级（决定是否纳入训练集）
- 困惑度（perplexity）→ 难度等级（用于课程学习 curriculum learning）

## 交互特征：列与列的组合

有时候单个列的信息不够，需要组合两列来创造更有区分度的特征：

```python
import numpy as np

n = 50_000
df = pd.DataFrame({
    'prompt_tokens': np.random.randint(10, 500, n),
    'response_tokens': np.random.randint(20, 2000, n),
    'quality': np.round(np.random.uniform(1, 5, n), 1),
})

df['total_tokens'] = df['prompt_tokens'] + df['response_tokens']
df['response_ratio'] = (df['response_tokens'] / df['total_tokens']).round(3)
df['quality_per_token'] = (df['quality'] / df['total_tokens']).round(4)

df['is_balanced'] = np.where(
    (df['response_ratio'] > 0.3) & (df['response_ratio'] < 0.8),
    'balanced', 'imbalanced'
).astype('category')

print(f"交互特征:\n{df[['total_tokens', 'response_ratio', 'is_balanced']].head()}")
```

交互特征的威力在于它能捕捉单列无法表达的模式。比如 `response_ratio`（回复长度占总长度的比例）能帮你在不看具体内容的情况下判断对话是否均衡——比例过低说明回复太简短，比例过高可能是 prompt 太短而 response 冗长。
