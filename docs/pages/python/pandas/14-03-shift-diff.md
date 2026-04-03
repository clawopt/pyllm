---
title: shift() 与 diff()：时间偏移与差分
description: 数据的滞后/前移、环比/同比计算、LLM 场景下的增长率分析
---
# shift() 与 diff()：时间的偏移与变化

`shift()` 把数据沿时间轴向前或向后移动，`diff()` 计算相邻时间点的差值。这两个方法组合起来可以做很多有用的分析。

## shift()：时间偏移

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'date': pd.date_range('2025-01-01', periods=5, freq='D'),
    'users': [100, 120, 115, 130, 140],
})

df['yesterday'] = df['users'].shift(1)
df['mom_change'] = df['users'] - df['users'].shift(1)
df['mom_pct'] = (df['users'] / df['users'].shift(1) - 1).round(3)

print(df)
```

`shift(1)` 让数据"向下移一格"——今天的数据跑到明天的位置（或者说今天的"昨天"就是上一天的值）。这在计算**环比增长率**（和上一个时间点相比）时是标准操作。

## diff()：差分

```python
df['diff_1d'] = df['users'].diff()
df['pct_change'] = df['users'].pct_change().round(3)

print(df[['date', 'users', 'diff_1d', 'pct_change']])
```

`diff()` 是 `x - x.shift()` 的简写。`pct_change()` 则直接计算百分比变化 `(x - x.shift()) / x.shift()`——在计算日环比、周同比等指标时极其常用。
