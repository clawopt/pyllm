---
title: groupby() 分组基础
description: Split-Apply-Combine 范式、单列/多列分组、groupby 对象遍历、分组键的类型注意事项
---
# groupby：Pandas 最强大的操作之一

如果说 Pandas 有一个"必须彻底搞懂"的方法，那一定是 `groupby`。它是"拆分-应用-合并"（Split-Apply-Combine）范式的直接实现——把数据按某个键拆成若干组，对每组独立做计算，再把结果合并回来。这个模式覆盖了数据分析中至少 60% 的需求。

## 核心思想：三步走

```
原始 DataFrame (100万行)
    │
    ▼ 按 source 拆分 (Split)
┌─────────┬─────────┬──────────┐
│ api组   │ web组   │ export组 │
│ 33万行  │ 33万行  │ 34万行   │
└────┬────┴────┬────┴─────┬────┘
     ▼         ▼          ▼
  计算均值   计算均值    计算均值  (Apply)
     │         │          │
     └─────────┴──────────┘
            ▼
    合并结果 (Combine)
  api: 4.2, web: 3.8, export: 4.0
```

## 单列分组：最常用的模式

```python
import pandas as pd
import numpy as np

np.random.seed(42)
n = 100_000
df = pd.DataFrame({
    'model': np.random.choice(['GPT-4o', 'Claude', 'Llama'], n),
    'latency_ms': np.random.randint(100, 2000, n),
    'tokens': np.random.randint(20, 4000, n),
})

avg_latency = df.groupby('model')['latency_ms'].mean()
print("各模型平均延迟:")
print(avg_latency.round(1).sort_values(ascending=False))
```

注意 `df.groupby('model')['latency_ms']` 这一步：先按 model 分组，再只取 latency_ms 列。这比先分组整个 DataFrame 再聚合要高效得多——因为 Pandas 在分组阶段就可以忽略不需要的列。

## 多列分组

```python
df['source'] = np.random.choice(['api', 'web'], n)

multi = df.groupby(['model', 'source']).agg(
    avg_latency=('latency_ms', 'mean'),
    avg_tokens=('tokens', 'mean'),
    count=('latency_ms', 'size'),
)

print(multi.round(1))
```

多列分组的结果是一个**多层索引的 Series 或 DataFrame**——第一层是第一个分组键，第二层是第二个分组键。这在分析"模型 × 来源"交叉维度的统计时极其常用。
