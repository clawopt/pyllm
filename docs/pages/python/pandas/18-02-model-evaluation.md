---
title: 模型评估数据处理
description: 多模型评测结果的合并、排名、对比分析、自动生成评估报告
---
# 模型评估：从原始分数到决策报告

当你同时评测了多个模型（GPT-4o、Claude、Llama、Qwen...）在多个基准测试上的表现后，你需要把分散的结果整合成一份有意义的报告来辅助模型选型。

## 数据准备与合并

```python
import pandas as pd
import numpy as np

np.random.seed(42)
models = ['GPT-4o', 'Claude', 'Llama', 'Qwen', 'DeepSeek']
benchmarks = ['MMLU', 'HumanEval', 'MATH', 'GPQA', 'BBH']

rows = []
for m in models:
    base = {'GPT-4o':88,'Claude':89,'Llama':84,'Qwen':83,'DeepSeek':87}[m]
    for b in benchmarks:
        noise = np.random.randn() * (6 if b in ('MATH','GPQA') else 3)
        rows.append({'model': m, 'benchmark': b, 'score': round(base + noise + np.random.randn(), 1)})

df = pd.DataFrame(rows)
pivot = df.pivot_table(index='model', columns='benchmark', values='score')
pivot['mean'] = pivot.mean(axis=1).round(1)
pivot['rank'] = pivot['mean'].rank(ascending=False, method='dense').astype(int)
pivot = pivot.sort_values('rank')
print(pivot.round(1))
```

## 综合评分

不同基准的量纲和难度不同——直接取均值可能不公平。常见的做法是**标准化后再加权平均**：

```python
normalized = (pivot[benchmarks] - pivot[benchmarks].mean()) / pivot[benchmarks].std()
weights = {'MMLU':0.2, 'HumanEval':0.2, 'MATH':0.25, 'GPQA':0.2, 'BBH':0.15}
weighted = normalized.mul(weights).sum(axis=1).round(2)
pivot['weighted_score'] = weighted
pivot['weighted_rank'] = weighted.rank(ascending=False, method='dense').astype(int)
print(pivot[['mean', 'rank', 'weighted_score', 'weighted_rank']].sort_values('weighted_rank'))
```
