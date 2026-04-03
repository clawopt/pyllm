---
title: groupby() 分组基础
description: 单列分组 / 多列分组 / groupby 对象遍历 / 分组键的类型注意事项
---
# GroupBy 基础


## groupby 的核心思想

`groupby` 的操作遵循 **"拆分-应用-合并"（Split-Apply-Combine）** 范式：

```
原始数据 → 按 key 拆分成组 → 每组独立计算 → 合并结果
```

```python
import pandas as pd
import numpy as np

np.random.seed(42)

df = pd.DataFrame({
    'model': ['GPT-4o']*30 + ['Claude']*30 + ['Llama']*30 + ['Qwen']*30,
    'task': np.random.choice(['chat', 'code', 'math', 'reasoning'], 120),
    'latency_ms': np.concatenate([
        np.random.normal(800, 150, 30),
        np.random.normal(650, 120, 30),
        np.random.normal(350, 80, 30),
        np.random.normal(400, 90, 30),
    ]).astype(int),
    'tokens_used': np.random.randint(100, 2000, 120),
    'success': np.random.choice([1, 0], 120, p=[0.95, 0.05]),
})
```

## 单列分组

```python
by_model = df.groupby('model')['latency_ms'].mean()
print("各模型平均延迟 (ms):")
print(by_model.round(1).sort_values(ascending=False))
```

## 多列分组

```python
multi_grouped = df.groupby(['model', 'task']).agg(
    avg_latency=('latency_ms', 'mean'),
    total_calls=('latency_ms', 'count'),
    success_rate=('success', 'mean'),
    total_tokens=('tokens_used', 'sum'),
).round(2)

print(multi_grouped)
```

## 遍历分组对象

```python
for name, group in df.groupby('model'):
    print(f"\n=== {name} ({len(group)} 条) ===")
    print(f"  平均延迟: {group['latency_ms'].mean():.1f} ms")
    print(f"  成功率:   {group['success'].mean()*100:.1f}%")
```

### 获取特定分组

```python
gpt_group = df.groupby('model').get_group('GPT-4o')
print(f"GPT-4o 组: {len(gpt_group)} 条记录")

groups_dict = dict(list(df.groupby('model')))
print(f"可用分组: {list(groups_dict.keys())}")
```

## 分组键的类型陷阱

### 常见问题：NaN 分组和混合类型

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'category': ['tech', 'finance', None, 'tech', 'medical', 'Finance', 'TECH'],
    'value': [10, 20, 30, 40, 50, 60, 70],
})

print("默认（dropna=True，排除 NaN）:")
print(df.groupby('category', dropna=True)['value'].sum())

print("\n保留 NaN 组:")
print(df.groupby('category', dropna=False)['value'].sum())
```

### 大小写敏感问题

```python
print("\n⚠️ 大小写不同 = 不同组:")
print(df.groupby('category')['value'].count())

df['category_clean'] = df['category'].str.lower()
print("\n统一小写后:")
print(df.groupby('category_clean')['value'].sum())
```

### 分类类型（Category）保证完整性

```python
import pandas as pd

df = pd.DataFrame({
    'tier': pd.Categorical(['A', 'B', 'A', 'C', 'B'], categories=['A', 'B', 'C', 'D']),
    'score': [88, 92, 85, 78, 95],
})

grouped = df.groupby('tier')['score'].agg(['count', 'mean'])
print(grouped)

```

## groupby 的常见属性

```python
import pandas as pd
import numpy as np

np.random.seed(42)

df = pd.DataFrame({
    'model': np.random.choice(['GPT-4o', 'Claude', 'Llama'], 100),
    'score': np.random.uniform(0.6, 0.98, 100),
})

gb = df.groupby('model')

print(f"分组数:       {gb.ngroups}")
print(f"各组大小:\n{gb.size()}")
print(f"分组键列表:   {list(gb.groups.keys())}")
print(f"第 0 组索引:   {gb.indices.get('GPT-4o', [])[:5]}")
```

## LLM 场景：按模型统计 API 调用指标

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

n = 500
base_time = datetime(2025, 3, 15)

api_logs = pd.DataFrame({
    'timestamp': [base_time + timedelta(minutes=np.random.randint(0, 1440)) for _ in range(n)],
    'model': np.random.choice(['gpt-4o', 'gpt-4o-mini', 'claude-sonnet', 'claude-haiku',
                               'deepseek-chat', 'qwen-plus'], n,
                             p=[0.20, 0.35, 0.12, 0.18, 0.10, 0.05]),
    'prompt_tokens': np.random.randint(50, 2000, n),
    'completion_tokens': np.random.randint(20, 3000, n),
    'latency_ms': np.random.exponential(800, n).astype(int) + 100,
    'status': np.random.choice(['success', 'rate_limited', 'error'], n, p=[0.94, 0.04, 0.02]),
})

api_logs['total_tokens'] = api_logs['prompt_tokens'] + api_logs['completion_tokens']
api_logs['hour'] = api_logs['timestamp'].dt.hour
api_logs['cost_usd'] = (
    api_logs['prompt_tokens'] * 0.00001 +
    api_logs['completion_tokens'] * 0.00003
).round(4)

model_stats = api_logs.groupby('model').agg(
    total_calls=('model', 'count'),
    success_rate=('status', lambda x: (x == 'success').mean()),
    avg_latency_ms=('latency_ms', 'mean'),
    p50_latency=('latency_ms', lambda x: x.quantile(0.5)),
    p99_latency=('latency_ms', lambda x: x.quantile(0.99)),
    total_tokens=('total_tokens', 'sum'),
    total_cost=('cost_usd', 'sum'),
    avg_prompt_len=('prompt_tokens', 'mean'),
    avg_completion_len=('completion_tokens', 'mean'),
).round(2)

model_stats['success_pct'] = (model_stats['success_rate'] * 100).round(1)
print("=== API 调用按模型统计 ===")
print(model_stats.sort_values('total_calls', ascending=False))

hourly = api_logs.groupby(['model', 'hour']).size().unstack(fill_value=0)
print(f"\n调用时间分布（前6小时）:")
print(hourly.iloc[:, :6])
```
