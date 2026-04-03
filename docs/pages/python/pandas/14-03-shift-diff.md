---
title: 时间序列偏移与差分
description: shift() / tshift() / diff() / pct_change() / LLM 场景：趋势分析与异常检测
---
# Shift 与 Diff 操作


## shift(): 值偏移

```python
import pandas as pd
import numpy as np

np.random.seed(42)

df = pd.DataFrame({
    'date': pd.date_range('2025-03-01', periods=10, freq='D'),
    'api_calls': [100, 120, 115, 130, 140, 135, 150, 160, 155, 170],
})

df['prev_day'] = df['api_calls'].shift(1)

df['next_day'] = df['api_calls'].shift(-1)

df['prev_7d'] = df['api_calls'].shift(7)

print(df[['date', 'api_calls', 'prev_day', 'next_day']].head(8))
```

## diff(): 差分

```python
import pandas as pd

df = pd.DataFrame({
    'date': pd.date_range('2025-03-01', periods=10, freq='D'),
    'api_calls': [100, 120, 115, 130, 140, 135, 150, 160, 155, 170],
})

df['diff_1'] = df['api_calls'].diff()

df['diff_2'] = df['api_calls'].diff(2)

df['pct_change'] = df['api_calls'].pct_change().round(3)

print(df.round(2).to_string(index=False))
```

## pct_change(): 变化率

```python
import pandas as pd
import numpy as np

np.random.seed(42)

daily_cost = pd.DataFrame({
    'date': pd.date_range('2025-03-01', periods=30, freq='D'),
    'cost_usd': np.cumsum(np.random.uniform(5, 50, 30)),
})

daily_cost['day_over_day'] = daily_cost['cost_usd'].pct_change() * 100

daily_cost['week_over_week'] = daily_cost['cost_usd'].pct_change(periods=7) * 100

print("=== 费用变化率 ===")
print(daily_cost[['date', 'cost_usd', 'day_over_day']].round(2).tail(10))
```

## 时间索引偏移：tshift() vs shift()

```python
import pandas as pd

ts = pd.Series(
    range(5),
    index=pd.date_range('2025-03-01', periods=5, freq='D')
)

val_shifted = ts.shift(1)

idx_shifted = ts.shift(1, freq='D')

print("原始:")
print(ts)
print("\nshift(1):")
print(val_shifted)
print("\nshift(1, freq='D'):")
print(idx_shifted)
```
