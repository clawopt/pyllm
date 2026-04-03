---
title: 时间序列基础
description: pd.to_datetime() 解析、DatetimeIndex 操作、时间范围选择、LLM 日志按时间分析
---
# 时间序列基础

在 LLM 开发中，时间戳无处不在——API 调用日志有时间、模型评估结果有版本发布日期、训练数据有采集时间。Pandas 提供了完整的时间序列工具链来处理这些数据。

## 创建与解析时间

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'timestamp': ['2025-01-15 10:30:00', '2025-01-16 14:20:00',
                 '2025/02/01 09:00', 'bad_date', None],
    'model': ['GPT-4o', 'Claude', 'Llama', 'GPT-4o', 'Claude'],
    'latency_ms': [1200, 980, 450, 1300, 890],
})

df['ts'] = pd.to_datetime(df['timestamp'], errors='coerce')
print(f"成功解析 {df['ts'].notna().sum()} / {len(df)} 条")
```

`pd.to_datetime()` 是把各种格式的字符串/数字转成 Pandas 时间类型的通用方法。`errors='coerce'` 让无法解析的值变成 `NaT`（Not a Time），而不是抛出异常——这在清洗脏数据时是标准做法。

## 按时间筛选

```python
import pandas as pd

dates = pd.date_range('2025-01-01', periods=100, freq='h')
df = pd.DataFrame({
    'ts': dates,
    'requests': np.random.randint(50, 500, 100),
})

jan = df[df['ts'] < '2025-02-01']
week1 = df[(df['ts'] >= '2025-01-06') & (df['ts'] < '2025-01-13')]

print(f"一月数据: {len(jan)} 条")
print(f"第一周: {len(week1)} 条")
```

Pandas 允许你直接用字符串和 datetime 列做比较——内部会自动做类型转换。这比手动写 `datetime.strptime()` 简洁得多。
