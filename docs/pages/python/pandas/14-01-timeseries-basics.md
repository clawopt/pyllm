---
title: 时间序列基础：创建与解析
description: pd.to_datetime() / 日期范围生成 / 时间戳组件提取 / 时区处理 / LLM API 日志时间分析
---
# 时间序列基础


## 为什么 LLM 开发需要时间序列

- **API 调用日志**：按小时/天统计调用量、延迟、成本
- **模型评估追踪**：版本迭代后的分数变化趋势
- **SFT 训练监控**：loss 曲线、学习率调度
- **RAG 知识库更新**：文档时效性管理

## 创建时间序列

### 从字符串解析

```python
import pandas as pd

dates_str = ['2025-01-15', '2025/03/20', 'March 25, 2025',
             '2025-04-01 14:30:00', '20250405']

dates = pd.to_datetime(dates_str)
print("自动解析多种格式:")
for s, d in zip(dates_str, dates):
    print(f"  '{s}' → {d}")
```

### 指定格式加速（大数据量时推荐）

```python
import pandas as pd

raw_dates = ['2025-03-15 08:30:00'] * 100000

auto_parsed = pd.to_datetime(raw_dates)

fast_parsed = pd.to_datetime(raw_dates, format='%Y-%m-%d %H:%M:%S')
```

### 常用格式码速查

| 格式码 | 含义 | 示例 |
|--------|------|------|
| `%Y` | 四位年份 | 2025 |
| `%m` | 两位月份 | 03 |
| `%d` | 两位日期 | 15 |
| `%H` | 24 小时制时 | 14 |
| `%M` | 分钟 | 30 |
| `%S` | 秒 | 45 |

## 生成日期范围

```python
import pandas as pd

daily = pd.date_range('2025-03-01', periods=7, freq='D')
print(f"按天: {daily.tolist()[:3]}...")

hourly = pd.date_range('2025-03-01', periods=48, freq='h')
print(f"按小时: {len(hourly)} 个")

business = pd.date_range('2025-03-01', periods=10, freq='B')
print(f"工作日: {business.tolist()}")

monthly = pd.date_range('2025-01-01', periods=6, freq='ME')
print(f"月末: {monthly.tolist()}")

custom = pd.date_range('2025-03-01', periods=8, freq='6h')
print(f"每6小时: {custom.tolist()[:3]}...")
```

## 时间戳组件提取

```python
import pandas as pd
import numpy as np

np.random.seed(42)
n = 100
ts_df = pd.DataFrame({
    'timestamp': pd.date_range('2025-03-01', periods=n, freq='3h'),
    'api_calls': np.random.poisson(50, n),
    'avg_latency_ms': np.random.normal(500, 150, n).astype(int),
})

ts_df['date'] = ts_df['timestamp'].dt.date
ts_df['time'] = ts_df['timestamp'].dt.time
ts_df['hour'] = ts_df['timestamp'].dt.hour
ts_df['day_of_week'] = ts_df['timestamp'].dt.dayofweek
ts_df['day_name'] = ts_df['timestamp'].dt.day_name()
ts_df['week'] = ts_df['timestamp'].dt.isocalendar().week.astype(int)
ts_df['month'] = ts_df['timestamp'].dt.month
ts_df['is_weekend'] = ts_df['day_of_week'].isin([5, 6]).astype(int)

print("=== 时间戳组件 ===")
print(ts_df[['timestamp', 'hour', 'day_name', 'is_weekend']].head(8))
```

## 时区处理

```python
import pandas as pd

utc_time = pd.Timestamp('2025-03-15 14:30:00', tz='UTC')

beijing = utc_time.tz_convert('Asia/Shanghai')
print(f"UTC:     {utc_time}")
print(f"Beijing: {beijing}")

naive = pd.Timestamp('2025-03-15 14:30:00')
localized = naive.tz_localize('America/New_York')
print(f"NY:      {localized}")

dates = pd.date_range('2025-03-01', periods=5, freq='D', tz='UTC')
shanghai_dates = dates.tz_convert('Asia/Shanghai')
print(f"\n批量转换:")
for d, sd in zip(dates, shanghai_dates):
    print(f"  {d} → {sd}")
```

## LLM 场景：API 日志时间分析准备

```python
import pandas as pd
import numpy as np

class APILogTimeAnalyzer:
    """API 日志时间序列分析器"""

    def __init__(self):
        self.df = None

    def generate_sample_data(self, n=720, start_date='2025-03-01'):
        np.random.seed(42)
        timestamps = pd.date_range(start_date, periods=n, freq='1h')

        hourly_pattern = (
            30 + 25 * np.sin(2 * np.pi * np.arange(n) / 24) +
            np.random.randn(n) * 10
        )
        calls = np.random.poisson(np.clip(hourly_pattern, 5, 120)).astype(int)

        latency_base = 400 + 200 * (1 - np.cos(2 * np.pi * np.arange(n) / 24))
        latency = np.clip(
            np.random.exponential(latency_base, n) + 100,
            80, 5000
        ).astype(int)

        self.df = pd.DataFrame({
            'timestamp': timestamps,
            'api_calls': calls,
            'avg_latency_ms': latency,
            'errors': np.random.choice([0, 0, 0, 0, 1], n),
            'total_tokens': calls * np.random.randint(500, 3000, n),
        })
        return self

    def extract_features(self):
        self.df['hour'] = self.df['timestamp'].dt.hour
        self.df['dow'] = self.df['timestamp'].dt.dayofweek
        self.df['dom'] = self.df['timestamp'].dt.day
        self.df['is_peak_hour'] = self.df['hour'].isin([9, 10, 14, 15, 20, 21]).astype(int)
        self.df['is_work_hours'] = self.df['hour'].between(9, 18).astype(int)
        self.df['error_rate'] = self.df['errors'] / self.df['api_calls'].replace(0, 1)
        return self

    def summary(self):
        if self.df is None:
            raise ValueError("请先生成数据")

        print(f"=== API 日志时间概览 ===")
        print(f"时间范围: {self.df['timestamp'].min()} ~ {self.df['timestamp'].max()}")
        print(f"总记录:   {len(self.df)} 条")
        print(f"总调用:   {self.df['api_calls'].sum():,} 次")
        print(f"总错误:   {self.df['errors'].sum()} 次")
        print(f"平均延迟: {self.df['avg_latency_ms'].mean():.0f} ms")
        print(f"P99 延迟: {self.df['avg_latency_ms'].quantile(0.99):.0f} ms")

        peak = self.df[self.df['is_peak_hour'] == 1]
        off_peak = self.df[self.df['is_peak_hour'] == 0]
        print(f"\n高峰 vs 非高峰:")
        print(f"  高峰均调用量:   {peak['api_calls'].mean():.1f}/h")
        print(f"  非高峰均调用量: {off_peak['api_calls'].mean():.1f}/h")
        return self


analyzer = APILogTimeAnalyzer()
analyzer.generate_sample_data().extract_features().summary()
```
