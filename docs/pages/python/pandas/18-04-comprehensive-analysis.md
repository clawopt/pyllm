---
title: LLM 数据分析综合实战
description: 端到端项目：从数据加载到报告生成的完整数据分析流程
---
# 综合实战：LLM 数据分析全流程

这一节把前面所有章节的技术整合到一个完整的项目中：**对一份包含 100 万条对话记录的数据集做全面的质量分析和可视化报告**。

## 项目概览

```python
import pandas as pd
import numpy as np

np.random.seed(42)
N = 1_000_000

df = pd.DataFrame({
    'conversation_id': [f'conv_{i:06d}' for i in range(N)],
    'model': np.random.choice(['GPT-4o', 'Claude', 'Llama'], N),
    'source': np.random.choice(['api', 'web', 'export'], N),
    'prompt': [f'问题{i%500}' for i in range(N)],
    'response': [f'回答{i}' if i%20 != 0 else '' for i in range(N)],
    'quality_score': np.round(np.random.uniform(1, 5, N), 1),
    'token_count': np.random.randint(20, 4000, N),
    'latency_ms': np.random.randint(100, 3000, N),
    'timestamp': pd.date_range('2025-01-01', periods=N, freq='s'),
})

df['quality_tier'] = pd.qcut(df['quality_score'], q=4,
                              labels=['D_低质','C_一般','B_良好','A_优秀'])

print("=" * 60)
print(f"  对话语料全面分析 | 共 {N:,} 条")
print("=" * 60)

print("\n【模型维度】")
model_stats = df.groupby('model').agg(
    count=('conversation_id', 'count'),
    avg_quality=('quality_score', 'mean'),
    p95_latency=('latency_ms', lambda x: x.quantile(0.95)),
).round(2).sort_values('avg_quality', ascending=False)
print(model_stats)

print("\n【质量分布】")
tier_dist = df.groupby('quality_tier', observed=True).size()
for t, c in tier_dist.items():
    print(f"  {t}: {c:,} ({c/N*100:.1f}%)")

print("\n【时间趋势】")
daily = df.set_index('timestamp').resample('D').size()
print(f"日均: {daily.mean():,.0f} 条 | 峰值日: {daily.max():,} ({daily.idxmax().strftime('%Y-%m-%d')})")
```

这个综合案例展示了从数据构造到多维度分析的完整流程——它用到了 groupby、qcut 分箱、resample 重采样、quantile 统计等核心操作。**把它当作你自己的数据分析项目的模板**。
