---
title: 时间序列实战：API 监控与趋势分析
description: 调用量趋势 / 延迟监控 / 成本预测 / 异常检测 / SLA 报告生成
---
# 时间序列监控


## 场景一：API 调用仪表盘数据准备

```python
import pandas as pd
import numpy as np

class APIDashboardBuilder:
    """API 监控仪表盘数据构建器"""

    def __init__(self):
        self.raw = None

    def generate(self, days=30, start='2025-03-01'):
        np.random.seed(42)
        n = days * 24

        timestamps = pd.date_range(start, periods=n, freq='h')

        hour_pattern = np.sin(2 * np.pi * np.arange(n) / 24) * 0.6 + 1
        weekday_boost = np.array([1.0 if d.weekday() < 5 else 0.7 for d in timestamps])
        trend = np.linspace(1.0, 1.15, n)

        base_calls = 40 * hour_pattern * weekday_boost * trend
        calls = np.random.poisson(np.clip(base_calls, 10, 150)).astype(int)

        latency_base = 400 + 200 * (1 - hour_pattern)
        latency = np.clip(
            np.random.exponential(latency_base, n) + 100,
            80, 5000
        ).astype(int)

        errors = np.random.choice(
            [0]*95 + [1]*3 + [2]*2,
            n, p=[0.95, 0.03, 0.02]
        )

        self.raw = pd.DataFrame({
            'timestamp': timestamps,
            'api_calls': calls,
            'avg_latency_ms': latency,
            'error_count': errors,
            'total_tokens': calls * np.random.randint(500, 3000, n),
        })
        return self

    def build_hourly_report(self):
        df = self.raw.set_index('timestamp')

        report = pd.DataFrame({
            '调用量': df['api_calls'],
            '平均延迟(ms)': df['avg_latency_ms'].rolling(24).mean().round(0),
            'P99延迟(ms)': df['avg_latency_ms'].rolling(24).quantile(0.99).round(0),
            '错误数': df['error_count'],
            'Token总量(K)': (df['total_tokens'] / 1000).round(1),
            '7日均量': df['api_calls'].rolling(window=24*7).mean().round(1),
            '环比变化%': df['api_calls'].pct_change(periods=24) * 100,
        })

        report['timestamp'] = report.index
        return report.reset_index(drop=True)


dashboard = APIDashboardBuilder()
dashboard.generate(days=30)
report = dashboard.build_hourly_report()

print(f"=== API 监控报告（{len(report)} 小时）===")
print(f"时间范围: {dashboard.raw['timestamp'].min()} ~ {dashboard.raw['timestamp'].max()}")
print(f"\n最近 12 小时:")
print(report[['timestamp', '调用量', '平均延迟(ms)', 'P99延迟(ms)',
               '错误数', '环比变化%']].tail(12).to_string(index=False))

daily_summary = dashboard.raw.groupby(dashboard.raw['timestamp'].dt.date).agg(
    total_calls=('api_calls', 'sum'),
    avg_latency=('avg_latency_ms', 'mean'),
    max_latency=('avg_latency_ms', 'max'),
    total_errors=('error_count', 'sum'),
    total_tokens=('total_tokens', 'sum'),
).round(1)

daily_summary['error_rate_%'] = (
    daily_summary['total_errors'] / daily_summary['total_calls'] * 100
).round(3)

print(f"\n=== 日粒度汇总 ===")
print(daily_summary.tail(7))
```

## 场景二：简单异常检测（基于统计）

```python
import pandas as pd
import numpy as np

class AnomalyDetector:
    """时间序列异常检测器"""

    @staticmethod
    def detect_zscore(series, window=24, threshold=3.0):
        """Z-Score 异常检测"""
        rolling_mean = series.rolling(window=window, min_periods=1).mean()
        rolling_std = series.rolling(window=window, min_periods=1).std()
        z_scores = (series - rolling_mean) / rolling_std.replace(0, 1)
        anomalies = z_scores.abs() > threshold
        return z_scores.round(2), anomalies

    @staticmethod
    def detect_iqr(series, k=1.5):
        """IQR 方法异常检测"""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - k*IQR, Q3 + k*IQR
        anomalies = (series < lower) | (series > upper)
        return lower, upper, anomalies


np.random.seed(42)
normal_data = np.random.normal(500, 100, 168)
outlier_indices = [20, 50, 90, 120, 150]
for idx in outlier_indices:
    normal_data[idx] += np.random.choice([-1, 1]) * np.random.uniform(500, 800)

latency_series = pd.Series(
    normal_data.astype(int),
    index=pd.date_range('2025-03-01', periods=168, freq='h')
)

z_scores, z_anomalies = AnomalyDetector.detect_zscore(latency_series, window=24)
lower, upper, iqr_anomalies = AnomalyDetector.detect_iqr(latency_series)

detection_df = pd.DataFrame({
    'latency': latency_series,
    'z_score': z_scores,
    'is_z_anomaly': z_anomalies,
    'is_iqr_anomaly': iqr_anomalies,
})

anomalies_found = detection_df[detection_df['is_z_anomaly'] | detection_df['is_iqr_anomaly']]
print(f"=== 延迟异常检测结果 ===")
print(f"总记录: {len(detection_df)}")
print(f"Z-Score 异常: {z_anomalies.sum()} 个")
print(f"IQR 异常:   {iqr_anomalies.sum()} 个")

if len(anomalies_found) > 0:
    print(f"\n异常时刻:")
    for ts, row in anomalies_found.iterrows():
        reasons = []
        if row['is_z_anomaly']:
            reasons.append(f"Z={row['z_score']:.1f}")
        if row['is_iqr_anomaly']:
            reasons.append("超出IQR范围")
        print(f"  {ts} → 延迟={row['latency']}ms ({', '.join(reasons)})")
```

## 场景三：SLA 合规性报告

```python
import pandas as pd
import numpy as np

class SLAReporter:
    """SLA 服务等级协议报告"""

    SLA_TARGETS = {
        'latency_p50_ms': 500,
        'latency_p95_ms': 1200,
        'latency_p99_ms': 2500,
        'availability_pct': 99.9,
        'error_rate_pct': 0.5,
    }

    def __init__(self, logs_df):
        self.df = logs_df.copy()

    def check_latency_sla(self):
        results = {}
        for metric, target in [('p50', 50), ('p95', 95), ('p99', 99)]:
            actual = self.df['avg_latency_ms'].quantile(target/100)
            sla_key = f'latency_{metric}_ms'
            target_val = self.SLA_TARGETS[sla_key]
            results[sla_key] = {
                'target': target_val,
                'actual': round(actual, 0),
                'passed': actual <= target_val,
            }
        return results

    def check_availability(self):
        total_checks = len(self.df)
        failed = self.df[self.df['error_count'] > 0].shape[0]
        availability = ((total_checks - failed) / total_checks * 100) if total_checks > 0 else 0
        return {
            'availability_pct': {
                'target': self.SLA_TARGETS['availability_pct'],
                'actual': round(availability, 3),
                'passed': availability >= self.SLA_TARGETS['availability_pct'],
            }
        }

    def generate_report(self):
        print("=" * 55)
        print("SLA 合规性报告")
        print("=" * 55)

        latency_results = self.check_latency_sla()
        avail_results = self.check_availability()

        all_results = {**latency_results, **avail_results}

        passed = sum(r['passed'] for r in all_results.values())
        total = len(all_results)

        for metric, info in all_results.items():
            status = "✅ PASS" if info['passed'] else "❌ FAIL"
            print(f"  {metric:<25s} 目标: {info['target']:>8} | 实际: {info['actual']:>8} | {status}")

        print("-" * 55)
        overall_status = "✅ 全部通过" if passed == total else f"⚠️ {total-passed}/{total} 未达标"
        print(f"总体状态: {overall_status}")
        return all_results


np.random.seed(42)
n = 720
logs = pd.DataFrame({
    'timestamp': pd.date_range('2025-03-01', periods=n, freq='h'),
    'avg_latency_ms': np.clip(
        np.random.exponential(450, n).astype(int) + 100,
        80, 3000
    ),
    'error_count': np.random.choice([0]*96 + [1]*3 + [2]*1, n),
})

reporter = SLAReporter(logs)
results = reporter.generate_report()
```
