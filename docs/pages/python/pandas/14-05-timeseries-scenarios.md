---
title: 时间序列实战：训练日志与成本趋势
description: Loss 曲线分析 / 学习率调度可视化数据 / Token 用量趋势 / 多模型成本对比时间线
---
# 时间序列场景实战


## 场景一：SFT 训练 Loss 曲线分析

```python
import pandas as pd
import numpy as np

class TrainingLogAnalyzer:
    """训练日志分析器"""

    def __init__(self):
        self.logs = None

    def generate_sft_logs(self, n_steps=500, model_name='llama-3-8b-sft'):
        np.random.seed(42)
        steps = range(1, n_steps + 1)

        initial_loss = 2.5
        trend = initial_loss * np.exp(-np.arange(n_steps) / 150)
        noise = np.random.randn(n_steps) * 0.08 * trend

        loss = (trend + noise + 0.3 * np.sin(np.arange(n_steps) / 30)).clip(0.15, 3.0)

        lr = 2e-4 * np.cos(np.linspace(0, np.pi/2, n_steps))

        grad_norm = np.random.exponential(0.5, n_steps).clip(0.05, 3.0)

        self.logs = pd.DataFrame({
            'step': list(steps),
            'loss': loss.round(4),
            'learning_rate': lr.round(8),
            'grad_norm': grad_norm.round(4),
            'tokens_per_sec': np.random.randint(8000, 15000, n_steps),
            'model': model_name,
        })
        return self

    def analyze_convergence(self):
        """收敛性分析"""
        df = self.logs

        rolling_loss = df['loss'].rolling(window=20).mean()
        best_loss = df['loss'].min()
        best_step = df['loss'].idxmin() + 1

        recent = df['loss'].tail(50)
        convergence_metric = (recent.max() - recent.min()) / recent.mean()

        print("=== 训练收敛分析 ===")
        print(f"总步数:       {len(df)}")
        print(f"初始 Loss:   {df['loss'].iloc[0]:.4f}")
        print(f"最终 Loss:   {df['loss'].iloc[-1]:.4f}")
        print(f"最低 Loss:   {best_loss:.4f} (Step {best_step})")
        print(f"收敛指标:     {convergence_metric:.4f} (<0.05 表示已收敛)")
        print(f"状态:         {'✅ 已收敛' if convergence_metric < 0.05 else '⏳ 训练中'}")

        milestones = [0.25, 0.5, 1.0, 0.5]
        for target in [2.0, 1.5, 1.0, 0.5]:
            below = df[df['loss'] <= target]
            if len(below) > 0:
                first_step = below.index[0] + 1
                print(f"  Loss ≤ {target}: Step {first_step}")

        return {
            'best_loss': best_loss,
            'best_step': best_step,
            'convergence_metric': round(convergence_metric, 4),
            'final_loss': round(df['loss'].iloc[-1], 4),
        }

    def detect_anomalies(self, window=10, threshold=3.0):
        """检测训练异常（Loss 突升）"""
        df = self.logs.copy()
        df['rolling_mean'] = df['loss'].rolling(window=window).mean()
        df['rolling_std'] = df['loss'].rolling(window=window).std()
        df['deviation'] = ((df['loss'] - df['rolling_mean']) /
                           df['rolling_std'].replace(0, 1)).round(2)

        anomalies = df[df['deviation'].abs() > threshold]

        if len(anomalies) > 0:
            print(f"\n⚠️ 检测到 {len(anomalies)} 个异常点:")
            for _, row in anomalies.head(5).iterrows():
                direction = "📈 突升" if row['deviation'] > 0 else "📉 骤降"
                print(f"  Step {row['step']:>4d} | Loss={row['loss']:.4f} | "
                      f"Z={row['deviation']:+.2f} | {direction}")
        else:
            print("\n✅ 未检测到明显异常")

        return anomalies


analyzer = TrainingLogAnalyzer()
analyzer.generate_sft_logs(n_steps=500)
stats = analyzer.analyze_convergence()
anomalies = analyzer.detect_anomalies()
```

## 场景二：多模型 API 成本趋势对比

```python
import pandas as pd
import numpy as np

class CostTrendAnalyzer:
    """多模型成本趋势分析器"""

    def __init__(self):
        self.cost_data = None

    def generate_data(self, days=60, start='2025-02-01'):
        np.random.seed(42)
        models = ['gpt-4o', 'claude-sonnet', 'deepseek-chat', 'qwen-plus']
        base_costs = {'gpt-4o': 45, 'claude-sonnet': 38,
                       'deepseek-chat': 12, 'qwen-plus': 18}

        rows = []
        for day_offset in range(days):
            date = pd.Timestamp(start) + pd.Timedelta(days=day_offset)
            for model in models:
                base = base_costs[model]
                growth = 1 + day_offset * 0.008
                weekly_pattern = 1 + 0.15 * np.sin(2 * np.pi * day_offset / 7)
                daily_cost = base * growth * weekly_pattern * (1 + np.random.randn() * 0.1)

                rows.append({
                    'date': date,
                    'model': model,
                    'daily_cost_usd': max(daily_cost, 1),
                    'api_calls': int(daily_cost / base_costs[model] * 100),
                    'total_tokens': int(daily_cost * 10000),
                })

        self.cost_data = pd.DataFrame(rows)
        return self

    def build_trend_report(self):
        df = self.cost_data.copy()

        pivot_cost = df.pivot_table(
            index='date',
            columns='model',
            values='daily_cost_usd',
            aggfunc='sum'
        ).fillna(0)

        report = {}
        for model in pivot_cost.columns:
            series = pivot_cost[model]
            report[model] = {
                'total_spent': series.sum().round(2),
                'avg_daily': series.mean().round(2),
                '7d_moving_avg': series.rolling(7).mean().iloc[-1].round(2),
                'growth_rate': ((series.iloc[-7:].mean() - series.iloc[:7].mean())
                                / series.iloc[:7].mean() * 100).round(1),
                'max_day': series.idxmax().strftime('%Y-%m-%d'),
                'max_cost': series.max().round(2),
            }

        print("=== 多模型成本趋势报告 ===")
        print(f"{'Model':<16s} {'Total($)':>10s} {'Avg/d($)':>9s} "
              f"{'7dMA($)':>9s} {'Growth%':>9s} {'Peak Day':>12s}")
        print("-" * 70)
        for model, info in sorted(report.items(),
                                   key=lambda x: x[1]['total_spent'],
                                   reverse=True):
            print(f"{model:<16s} ${info['total_spent']:>9.2f} ${info['avg_daily']:>8.2f} "
                  f"${info['7d_moving_avg']:>8.2f} {info['growth_rate']:>+8.1f}% "
                  f"{info['max_day']}")

        total_all = sum(r['total_spent'] for r in report.values())
        print("-" * 70)
        print(f"{'TOTAL':<16s} ${total_all:>9.2f}")

        return report


trend = CostTrendAnalyzer()
trend.generate_data(days=60)
report = trend.build_trend_report()
```
