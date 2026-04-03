---
title: 时间序列监控实战
description: API 调用量/延迟/错误率的实时监控面板、异常检测、告警阈值触发
---
# 实战：LLM 服务监控面板

这一节用时间序列工具链构建一个简易的 LLM API 服务监控面板——追踪调用量、平均延迟和错误率三个核心指标。

```python
import pandas as pd
import numpy as np

np.random.seed(42)
dates = pd.date_range('2025-01-01', periods=30*24, freq='h')
n = len(dates)

df = pd.DataFrame({
    'ts': dates,
    'requests': np.random.poisson(200, n) + np.sin(np.arange(n)/12*np.pi)*50,
    'latency_ms': np.random.normal(800, 150, n).clip(200, 3000),
    'errors': np.random.choice([0, 1], n, p=[0.97, 0.03]),
})

df['error_rate'] = df['errors'].rolling(24).mean()
df['latency_p95'] = df['latency_ms'].rolling(24).quantile(0.95)
df['requests_ma'] = df['requests'].rolling(24*7).mean()

daily = df.set_index('ts').resample('D').agg(
    total_req=('requests', 'sum'),
    avg_latency=('latency_ms', 'mean'),
    error_rate=('errors', 'mean'),
)

print("=== 每日汇总 ===")
print(daily.round(2).head())

print(f"\n=== 异常检测 ===")
high_latency_days = daily[daily['avg_latency'] > daily['avg_latency'].mean() + 2*daily['avg_latency'].std()]
print(f"高延迟日期: {len(high_latency_days)} 天")
```

这个监控面板做了几件事：计算滚动错误率和 P95 延迟（平滑短期波动），按天重采样得到每日汇总，最后用简单的统计规则（均值+2倍标准差）检测异常高延迟的日期。**这是生产环境 LLM 服务监控的最小可行版本**。
