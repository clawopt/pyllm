---
title: 时间序列实战场景
description: 用户活跃度分析、API 调用趋势、模型版本对比、数据增长预测
---
# 时间序列综合实战

```python
import pandas as pd
import numpy as np

np.random.seed(42)
dates = pd.date_range('2025-01-01', periods=90, freq='D')
df = pd.DataFrame({
    'date': dates,
    'api_calls': np.random.poisson(1000, 90) + np.arange(90)*5,
    'active_users': np.random.randint(50, 200, 90) + np.arange(90),
    'model': ['GPT-4o']*45 + ['Claude']*45,
})

df['calls_per_user'] = (df['api_calls'] / df['active_users']).round(1)
df['wow_growth'] = df['api_calls'].pct_change(7).round(3)

weekly = df.set_index('date').resample('W-Mon').agg(
    total_calls=('api_calls', 'sum'),
    avg_users=('active_users', 'mean'),
)

print("=== 周报 ===")
print(weekly.head())

print(f"\n90天总调用量: {df['api_calls'].sum():,}")
print(f"日均活跃用户: {df['active_users'].mean():.0f}")
print(f"周均增长率: {df['wow_growth'].dropna().mean()*100:.1f}%")
```
