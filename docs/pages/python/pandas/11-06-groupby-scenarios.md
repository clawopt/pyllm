---
title: groupby 实战场景
description: LLM 对话数据的多维分析、SFT 数据集质量报告生成、模型评估分组统计、RAG 知识库元数据分析
---
# 实战：用 groupby 解决真实的 LLM 数据分析问题

前面五节我们系统学习了 groupby 的基础、聚合、transform/filter、透视表和高级用法。这一节把它们全部串起来，解决一个完整的工程问题：**对一份 50 万条的 LLM 对话语料做多维度的质量分析报告**。

## 场景：对话语料的全面质量审计

```python
import pandas as pd
import numpy as np

np.random.seed(42)
N = 500_000

df = pd.DataFrame({
    'conversation_id': [f'conv_{i:06d}' for i in range(N)],
    'model': np.random.choice(['GPT-4o', 'Claude', 'Llama'], N),
    'source': np.random.choice(['api', 'web', 'export'], N),
    'task_type': np.random.choice(['chat', 'code', 'reasoning', 'math'], N),
    'quality_score': np.round(np.random.uniform(1, 5, N), 1),
    'token_count': np.random.randint(20, 4000, N),
    'latency_ms': np.random.randint(100, 3000, N),
})

print("=" * 60)
print("  LLM 对话语料质量审计报告")
print(f"  总样本量: {N:,}")
print("=" * 60)

print("\n--- 按模型的统计 ---")
model_stats = df.groupby('model').agg(
    count=('conversation_id', 'count'),
    avg_quality=('quality_score', 'mean'),
    avg_tokens=('token_count', 'mean'),
    avg_latency=('latency_ms', 'mean'),
).round(2)
print(model_stats.sort_values('avg_quality', ascending=False))

print("\n--- 按 来源 × 任务类型 的交叉统计 ---")
cross = df.pivot_table(
    index='source',
    columns='task_type',
    values='quality_score',
    aggfunc='mean',
).round(2)
print(cross)

print("\n--- 各质量等级分布 ---")
df['tier'] = pd.qcut(df['quality_score'], q=4,
                      labels=['D_低质', 'C_一般', 'B_良好', 'A_优秀'])
tier_dist = df.groupby('tier', observed=True).size()
for tier, cnt in tier_dist.items():
    bar = '#' * int(cnt / N * 60)
    print(f"  {bar} {tier}: {cnt:,} ({cnt/N*100:.1f}%)")

print("\n--- 各组内 Top3 高质量对话 ---")
top_per_model = df.groupby('model', group_keys=False)\
    .apply(lambda g: g.nlargest(3, 'quality_score'))
print(top_per_model[['model', 'quality_score', 'source']].head(12))
```

这份报告整合了本章学到的所有技术：
- `groupby().agg()` 做各模型的核心指标汇总
- `pivot_table()` 做"来源 × 任务类型"的二维交叉分析
- `pd.qcut()` + `groupby(size)` 做质量等级的分布可视化
- `groupby().apply(nlargest)` 做各组内的 TopN 提取

到这里，第十章（排序）和第十一章（分组聚合）就全部结束了。这两章覆盖了 Pandas 中数据整理和分析最核心的操作——掌握了它们，你就能处理绝大多数结构化的数据分析任务。接下来的章节将进入更专业的领域：数据合并、时间序列处理和可视化。
