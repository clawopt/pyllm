---
title: 排序实战场景
description: 模型评估排行榜生成、SFT 数据质量排序、RAG 检索结果重排、时间序列数据排序
---
# 实战：LLM 场景中的排序应用

前面三节我们学了 `sort_values()`、`rank()` 和 `nlargest()`。这一节把它们用在三个真实的 LLM 工程场景中。

## 场景一：模型评测排行榜

```python
import pandas as pd
import numpy as np

np.random.seed(42)
models = ['GPT-4o', 'Claude', 'Llama', 'Qwen', 'DeepSeek', 'Gemini']
benchmarks = ['MMLU', 'HumanEval', 'MATH', 'GPQA']

rows = []
for m in models:
    base = {'GPT-4o':88,'Claude':89,'Llama':84,'Qwen':83,'DeepSeek':87,'Gemini':86}[m]
    for b in benchmarks:
        noise = np.random.randn() * (8 if b == 'MATH' else 4)
        rows.append({'model': m, 'benchmark': b, 'score': round(base + noise, 1)})

df = pd.DataFrame(rows)
pivot = df.pivot_table(index='model', columns='benchmark', values='score', aggfunc='max')
pivot['mean'] = pivot.mean(axis=1).round(1)
pivot['rank'] = pivot['mean'].rank(ascending=False, method='dense').astype(int)
pivot = pivot.sort_values('rank')

print("=== LLM Benchmark 排行榜 ===")
print(pivot.round(1).to_string())

print(f"\n各维度最佳:")
for col in benchmarks:
    best = pivot[col].idxmax()
    print(f"  {col}: {best} ({pivot.loc[best, col]:.1f})")
```

这个排行榜生成器做了几件事：从原始的 `(model, benchmark, score)` 长格式数据透视成宽格式（每个模型一行），计算平均分，用 dense 排名法分配名次，最后按名次排序输出。**这是 LLM 团队做模型选型时最常用的分析模式之一**。

## 场景二：SFT 数据按质量分层

```python
import pandas as pd
import numpy as np

np.random.seed(42)
n = 50_000
df = pd.DataFrame({
    'quality': np.round(np.random.uniform(1, 5, n), 1),
    'tokens': np.random.randint(20, 2000, n),
    'source': np.random.choice(['api', 'web'], n),
})

df['tier'] = pd.qcut(df['quality'], q=4,
                      labels=['D_低质量', 'C_一般', 'B_良好', 'A_优秀'])

tier_stats = df.groupby('tier', observed=True).agg(
    count=('quality', 'size'),
    avg_quality=('quality', 'mean'),
    avg_tokens=('tokens', 'mean'),
).round(2)

print(tier_stats)
```

`pd.qcut()` 按分位数把连续的质量分数分成四个等级——每个等级的样本数大致相等（各约 25%）。这比简单的阈值切分更合理，因为它自动适应数据的实际分布。
