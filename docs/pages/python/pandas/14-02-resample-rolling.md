---
title: 重采样 resample() 与滚动窗口 rolling()
description: 按时间粒度聚合、移动平均/求和、窗口统计、LLM API 调用量趋势分析
---
# 重采样与滚动窗口

当你有时间序列数据时，两个最常用的操作是：**把数据按更大的时间粒度聚合**（比如从每小时到每天）和**计算滚动统计量**（比如 7 天移动平均）。`resample()` 和 `rolling()` 就是为此设计的。

## resample()：按时间粒度聚合

```python
import pandas as pd
import numpy as np

dates = pd.date_range('2025-01-01', periods=720, freq='h')
df = pd.DataFrame({
    'ts': dates,
    'requests': np.random.poisson(200, 720),
})

df = df.set_index('ts')

daily = df.resample('D').agg(
    total_requests=('requests', 'sum'),
    avg_hourly=('requests', 'mean'),
    peak=('requests', 'max'),
)

print(daily.head())
```

`resample('D')` 把小时级数据聚合成天级（D=Day）。其他常见频率：`'H'`(小时)、`'W'`(周)、`'ME'`(月末)、`'QE'`(季度末)。配合 `agg()` 可以同时计算多个统计量。

## rolling()：滑动窗口统计

```python
df['ma_7d'] = df['requests'].rolling(window=7*24).mean()
df['ma_24h'] = df['requests'].rolling(window=24).mean()

print(df[['requests', 'ma_24h']].head(30))
```

`rolling(window=N)` 创建一个 N 个周期的滑动窗口。注意前 N-1 个位置会是 NaN（因为窗口还没填满）。这在 LLM 场景中常用于：
- **API 调用量的 7 日移动平均**——平滑日间波动，看清趋势方向
- **延迟的 P95 滚动值**——检测性能是否在持续恶化
- **错误率的指数加权移动平均（EWMA）**——对近期数据赋予更高权重
