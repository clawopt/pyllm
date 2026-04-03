---
title: 时间索引与重采样
description: set_index() 时间索引 / resample() 降采样 / asfreq() / 滚动窗口统计
---
# 重采样与滚动窗口


## 设置时间索引

```python
import pandas as pd
import numpy as np

np.random.seed(42)

df = pd.DataFrame({
    'timestamp': pd.date_range('2025-03-01', periods=100, freq='h'),
    'value': np.random.randn(100).cumsum(),
})

df_indexed = df.set_index('timestamp')

march_3rd = df_indexed['2025-03-03']
print(f"3月3日数据: {len(march_3rd)} 条")

week1 = df_indexed['2025-03-01':'2025-03-07']
print(f"第一周: {len(week1)} 条")
```

## resample()：降采样（聚合）

```python
import pandas as pd
import numpy as np

np.random.seed(42)

hourly_data = pd.DataFrame({
    'timestamp': pd.date_range('2025-03-01', periods=168, freq='h'),
    'api_calls': np.random.poisson(50, 168),
    'latency_ms': np.random.exponential(500, 168).astype(int),
}).set_index('timestamp')

daily = hourly_data.resample('D').agg(
    total_calls=('api_calls', 'sum'),
    avg_latency=('latency_ms', 'mean'),
    max_latency=('latency_ms', 'max'),
    p95_latency=('latency_ms', lambda x: x.quantile(0.95)),
).round(1)

print("=== 日粒度汇总 ===")
print(daily.head(7))

weekly = hourly_data.resample('W').agg(
    total_calls=('api_calls', 'sum'),
    avg_latency=('latency_ms', 'mean'),
)
print("\n=== 周粒度汇总 ===")
print(weekly)

monthly = hourly_data.resame('ME').agg(
    total_calls=('api_calls', 'sum'),
)
print(f"\n月总计: {monthly['total_calls'].sum():,} 次")
```

## 常用重采样频率别名

| 别名 | 含义 | 示例 |
|------|------|------|
| `D` | 天 | 每天 |
| `h` | 小时 | 每小时 |
| `W` | 周 | 每周日 |
| `ME` | 月末 | 每月底 |
| `QE` | 季末 | 每季度末 |
| `YE` | 年末 | 每年底 |
| `6h` | 6小时 | 每6小时 |
| `30min` | 30分钟 | 每30分钟 |

## 升采样与填充

```python
import pandas as pd

daily_df = pd.DataFrame(
    {'value': [10, 20, 15, 25]},
    index=pd.date_range('2025-03-01', periods=4, freq='D')
)

hourly_up = daily_df.resample('h').asfreq()
print(f"升采样: {len(hourly_up)} 行 (原{len(daily_df)}行)")

ffilled = daily_df.resample('h').ffill()

interpolated = daily_df.resample('h').interpolate(method='linear')
```

## 滚动窗口：rolling()

```python
import pandas as pd
import numpy as np

np.random.seed(42)

ts = pd.DataFrame({
    'value': np.random.randn(100) * 10 + 50,
}, index=pd.date_range('2025-01-01', periods=100, freq='D'))

ts['ma_7'] = ts['value'].rolling(window=7).mean().round(1)

ts['std_7'] = ts['value'].rolling(window=7).std().round(2)

ts['max_7'] = ts['value'].rolling(window=7).max()

ts['ma_30'] = ts['value'].rolling(window=30).mean().round(1)

print("=== 滚动窗口结果 ===")
print(ts[['value', 'ma_7', 'std_7']].head(12))

ts['ma_7_min2'] = ts['value'].rolling(window=7, min_periods=2).mean()
```

## expanding(): 扩展窗口（累计）

```python
import pandas as pd
import numpy as np

np.random.seed(42)

df = pd.DataFrame({'score': np.random.uniform(70, 95, 20)},
                  index=pd.date_range('2025-03-01', periods=20, freq='D'))

df['expanding_mean'] = df['score'].expanding().mean().round(2)

df['expanding_max'] = df['score'].expanding().max()

df['expanding_std'] = df['score'].expanding().std().round(3)

print(df.head(10))
```
