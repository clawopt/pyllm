---
title: 项目三：API 成本监控仪表盘
description: API 日志分析 / 实时成本追踪 / 预算告警 / 模型推荐 / 定期报告生成
---
# API 成本监控仪表盘


## 项目概述

构建一个 **LLM API 调用成本监控仪表盘**，自动分析调用日志、计算成本、发现异常、给出优化建议。

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class APICostDashboard:
    """API 成本监控仪表盘"""

    def __init__(self):
        self.logs = None
        self.price_table = pd.DataFrame({
            'model': ['gpt-4o', 'gpt-4o-mini', 'claude-sonnet', 'claude-haiku',
                      'deepseek-chat', 'qwen-plus'],
            'input_per_1M': [2.50, 0.15, 3.00, 0.80, 0.14, 0.40],
            'output_per_1M': [10.0, 0.60, 15.0, 4.00, 0.28, 1.20],
        }).set_index('model')

    def load_logs(self, logs_df):
        """加载 API 调用日志"""
        required = ['timestamp', 'model', 'prompt_tokens',
                    'completion_tokens']
        self.logs = logs_df[required].copy()
        self._compute_costs()
        print(f"加载 {len(self.logs):,} 条日志")
        return self

    def _compute_costs(self):
        """计算每条记录的成本"""
        merged = self.logs.merge(
            self.price_table,
            left_on='model',
            right_index=True,
            how='left'
        )

        merged['input_cost'] = (
            merged['prompt_tokens'] / 1e6 *
            merged['input_per_1M'].fillna(2.0)
        ).round(6)

        merged['output_cost'] = (
            merged['completion_tokens'] / 1e6 *
            merged['output_per_1M'].fillna(5.0)
        ).round(6)

        merged['total_cost'] = (merged['input_cost'] + merged['output_cost']).round(6)

        missing_model_mask = merged['input_per_1M'].isna()
        if missing_model_mask.any():
            default_input = 1.0
            default_output = 3.0
            merged.loc[missing_model_mask, 'input_cost'] = (
                merged.loc[missing_model_mask, 'prompt_tokens'] / 1e6 * default_input
            )
            merged.loc[missing_model_mask, 'output_cost'] = (
                merged.loc[missing_model_mask, 'completion_tokens'] / 1e6 * default_output
            )
            merged.loc[missing_model_mask, 'total_cost'] = (
                merged.loc[missing_model_mask, 'input_cost'] +
                merged.loc[missing_model_mask, 'output_cost']
            )

        self.logs = merged.drop(
            columns=['input_per_1M', 'output_per_1M'], errors='ignore'
        )
        return self

    def summary(self):
        """总览摘要"""
        if self.logs is None:
            raise ValueError("请先加载数据")

        total_cost = self.logs['total_cost'].sum()
        total_calls = len(self.logs)
        total_input_tokens = self.logs['prompt_tokens'].sum()
        total_output_tokens = self.logs['completion_tokens'].sum()

        print("=" * 55)
        print("API 成本监控总览")
        print("=" * 55)
        print(f"总调用量:     {total_calls:,} 次")
        print(f"总成本:       ${total_cost:.2f}")
        print(f"输入 Token:   {total_input_tokens:,}")
        print(f"输出 Token:   {total_output_tokens:,}")
        print(f"均次成本:      ${total_cost/total_calls:.4f}" if total_calls > 0 else "")
        print(f"时间范围:     {self.logs['timestamp'].min()} ~ "
              f"{self.logs['timestamp'].max()}")
        return {
            'total_cost': round(total_cost, 2),
            'total_calls': total_calls,
            'avg_cost': round(total_cost/max(total_calls,1), 4),
        }

    def model_breakdown(self):
        """按模型分解"""
        model_stats = self.logs.groupby('model').agg(
            calls=('model', 'count'),
            total_cost=('total_cost', 'sum'),
            avg_cost=('total_cost', 'mean'),
            input_tokens=('prompt_tokens', 'sum'),
            output_tokens=('completion_tokens', 'sum'),
            avg_latency=('latency_ms', 'mean') if 'latency_ms' in self.logs.columns else None,
        ).round(4).sort_values('total_cost', ascending=False)

        print("\n=== 各模型成本明细 ===")
        display_cols = ['calls', 'total_cost', 'avg_cost', 'input_tokens', 'output_tokens']
        cols_to_show = [c for c in display_cols if c in model_stats.columns]
        print(model_stats[cols_to_show].to_string())

        if 'total_cost' in model_stats.columns:
            pct_of_total = (model_stats['total_cost'] /
                           model_stats['total_cost'].sum() * 100).round(1)
            model_stats['cost_pct'] = pct_of_total

        return model_stats

    def daily_trend(self):
        """每日趋势"""
        if len(self.logs) == 0:
            return pd.DataFrame()

        self.logs['date'] = pd.to_datetime(self.logs['timestamp']).dt.date
        daily = self.logs.groupby('date').agg(
            calls=('model', 'count'),
            cost=('total_cost', 'sum'),
            tokens=('prompt_tokens', 'sum') + self.logs.groupby('date')['completion_tokens'].sum(),
        ).round(2)

        print("\n=== 每日趋势 ===")
        print(daily.tail(7))
        return daily

    def optimization_suggestions(self):
        """优化建议"""
        suggestions = []

        model_stats = self.model_breakdown()

        if 'avg_cost' in model_stats.columns and len(model_stats) > 1:
            most_expensive = model_stats['avg_cost'].idxmax()
            cheapest = model_stats['avg_cost'].idxmin()
            savings_potential = model_stats.loc[most_expensive, 'avg_cost'] - \
                               model_stats.loc[cheapest, 'avg_cost']

            if savings_potential > 0.5:
                suggestions.append({
                    'type': 'cost_reduction',
                    'title': f"成本优化建议",
                    'detail': f"{most_expensive} 的均次成本最高 "
                            f"(${model_stats.loc[most_expensive,'avg_cost']:.4f})，"
                            f"考虑将部分任务切换到 {cheapest}"
                            f"(${model_stats.loc[cheapest,'avg_cost']:.4f})",
                    'potential_monthly': f"${savings_potential * 30:.0f}/月"
                            f"(假设每天 {int(model_stats.loc[most_expensive,'calls'])} 次)",
                })

        if 'calls' in model_stats.columns:
            top_model = model_stats['calls'].idxmax()
            top_pct = model_stats.loc[top_model, 'calls'] / model_stats['calls'].sum() * 100
            if top_pct > 70:
                suggestions.append({
                    'type': 'diversification',
                    'title': "模型多样性建议",
                    'detail': f"{top_model} 占比 {top_pct:.0f}%，"
                           f"建议分散到多个模型以降低供应商风险",
                })

        if len(suggestions) > 0:
            print("\n=== 💡 优化建议 ===")
            for i, s in enumerate(suggestions, 1):
                print(f"\n  [{i}] {s['title']}")
                print(f"      {s['detail']}")
                if 'potential_monthly' in s:
                    print(f"      📊 {s['potential_monthly']}")

        return suggestions


np.random.seed(42)
n = 2000

api_logs = pd.DataFrame({
    'timestamp': pd.date_range('2025-03-01', periods=n, freq='10min'),
    'model': np.random.choice(['gpt-4o', 'gpt-4o-mini', 'claude-sonnet',
                                'deepseek-chat'], n,
                             p=[0.18, 0.35, 0.22, 0.25]),
    'prompt_tokens': np.random.randint(50, 4000, n),
    'completion_tokens': np.random.randint(20, 5000, n),
    'latency_ms': np.random.exponential(500, n).astype(int) + 100,
})

dashboard = APICostDashboard()
dashboard.load_logs(api_logs)
dashboard.summary()
dashboard.model_breakdown()
dashboard.daily_trend()
dashboard.optimization_suggestions()
```
