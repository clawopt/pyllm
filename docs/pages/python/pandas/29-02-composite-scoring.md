---
title: 综合评分与排名
description: 加权评分计算、标准化方法、排名策略、置信区间
---
# 综合评分

```python
pivot = df.pivot_table(index='model', columns='benchmark', values='score')

normalized = (pivot - pivot.mean()) / pivot.std()
weights = {'MMLU':0.2, 'HumanEval':0.2, 'MATH':0.25, 'GPQA':0.2, 'BBH':0.15}
pivot['weighted_score'] = normalized.mul(weights).sum(axis=1).round(2)
pivot['mean'] = pivot.mean(axis=1).round(1)
pivot['rank'] = pivot['weighted_score'].rank(ascending=False, method='dense').astype(int)

result = pivot.sort_values('rank')
print("=== 模型综合排名 ===")
print(result.round(1).to_string())
```
