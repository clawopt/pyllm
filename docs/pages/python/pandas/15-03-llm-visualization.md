---
title: LLM 评估可视化实战
description: Benchmark 雷达图数据准备 / 模型对比热力图 / 成本-性能散点 / 训练曲线
---
# LLM 数据可视化


## 场景一：模型评估对比矩阵热力图

```python
import pandas as pd
import numpy as np

class EvalVisualizer:
    """评估数据可视化辅助器"""

    @staticmethod
    def prepare_benchmark_matrix():
        """生成 benchmark 对比矩阵（可直接用于 seaborn.heatmap）"""
        np.random.seed(42)
        models = ['GPT-4o', 'Claude-3.5-Sonnet', 'Llama-3.1-70B',
                  'Qwen2.5-72B', 'DeepSeek-V3']
        benchmarks = ['MMLU', 'HumanEval', 'MATH', 'GPQA', 'BBH',
                      'IFEval', 'CMMLU']

        base = {'GPT-4o': 88, 'Claude-3.5-Sonnet': 89, 'Llama-3.1-70B': 84,
                'Qwen2.5-72B': 83, 'DeepSeek-V3': 86}
        bm_mods = {'HumanEval': 5, 'MATH': -8, 'GPQA': -4,
                   'BBH': 2, 'IFEval': 1, 'CMMLU': 0}

        matrix_data = []
        for model in models:
            row = {'model': model}
            for bm in benchmarks:
                score = base[model] + bm_mods.get(bm, 0) + np.random.randn() * 4
                row[bm] = round(np.clip(score, 35, 99), 1)
            matrix_data.append(row)

        df = pd.DataFrame(matrix_data).set_index('model')

        rank_row = df.rank(ascending=False, method='dense').astype(int)
        rank_row.index = ['Rank']

        return pd.concat([df, rank_row])

    @staticmethod
    def prepare_cost_perf_scatter():
        """成本-性能散点图数据"""
        np.random.seed(42)
        data = []
        for model, params, base_score, price in [
            ('GPT-4o', 1760, 88, 12.50),
            ('Claude-3.5', 175, 89, 18.00),
            ('Llama-3.1-70B', 70, 84, 0.87),
            ('Qwen2.5-72B', 72, 83, 1.47),
            ('DeepSeek-V3', 37, 87, 0.42),
            ('Gemini-1.5-Pro', None, 86, 6.25),
        ]:
            for _ in range(10):
                data.append({
                    'model': model,
                    'params_B': params,
                    'avg_score': round(base_score + np.random.randn() * 2, 1),
                    'cost_per_1M': round(price * (1 + np.random.randn() * 0.1), 2),
                    'category': 'Closed' if price > 1 else 'Open',
                })
        return pd.DataFrame(data)



print("=== Benchmark 对比矩阵 (用于 heatmap) ===")
matrix = EvalVisualizer.prepare_benchmark_matrix()
bench_cols = [c for c in matrix.columns if c != 'Rank']
print(f"模型 × 基准: {matrix.shape[0]} × {len(bench_cols)}")
print("\n原始分数:")
print(matrix[bench_cols].round(1).to_string())

print("\n排名:")
print(matrix.loc['Rank'].to_string())

print("\n=== 成本-性能散点图数据 ===")
scatter_df = EvalVisualizer.prepare_cost_perf_scatter()
print(scatter_df.head(10).to_string(index=False))

summary = scatter_df.groupby('model').agg(
    avg_score=('avg_score', 'mean'),
    avg_cost=('cost_per_1M', 'mean'),
    category=('category', 'first'),
).round(2)
print("\n各模型汇总:")
print(summary.sort_values('avg_score', ascending=False))
```

## 场景二：训练 Loss 曲线数据

```python
import pandas as pd
import numpy as np

class TrainingCurveData:
    """训练曲线数据生成器"""

    @staticmethod
    def generate(n_steps=500):
        np.random.seed(42)
        steps = range(1, n_steps + 1)

        initial_loss = 2.5
        trend = initial_loss * np.exp(-np.arange(n_steps) / 150)
        noise = np.random.randn(n_steps) * 0.06 * trend
        loss = (trend + noise + 0.25 * np.sin(np.arange(n_steps) / 25)).clip(0.18, 2.8)

        lr = 2e-4 * np.cos(np.linspace(0, np.pi * 0.48, n_steps))
        grad_norm = np.clip(np.random.exponential(0.45, n_steps), 0.05, 2.5)

        val_loss = loss * (1 + 0.03 * np.arange(n_steps)/n_steps) + np.random.randn(n_steps) * 0.04
        val_loss = val_loss.clip(0.20, 3.0)

        return pd.DataFrame({
            'step': list(steps),
            'train_loss': loss.round(4),
            'val_loss': val_loss.round(4),
            'learning_rate': lr.round(8),
            'grad_norm': grad_norm.round(4),
        })


curve_df = TrainingCurveData.generate()

print("=== 训练曲线数据预览 ===")
print(curve_df[['step', 'train_loss', 'val_loss']].head(10).to_string(index=False))
print("...")
print(curve_df[['step', 'train_loss', 'val_loss']].tail(5).to_string(index=False))

for target in [2.0, 1.5, 1.0, 0.5]:
    below = curve_df[curve_df['train_loss'] <= target]
    if len(below) > 0:
        print(f"\nTrain Loss ≤ {target}: Step {below.iloc[0]['step']} "
              f"(Val Loss: {below.iloc[0]['val_loss']:.4f})")
```

## 场景三：API 调用分布与异常检测可视化数据

```python
import pandas as pd
import numpy as np

class APIMetricsVisualizer:
    """API 指标可视化数据"""

    @staticmethod
    def generate_hourly_distribution(days=7):
        np.random.seed(42)
        n = days * 24

        hour_pattern = 40 + 30 * np.sin(2 * np.pi * np.arange(n) / 24)
        calls = np.random.poisson(np.clip(hour_pattern, 10, 120), n)

        latency_base = 400 + 250 * (1 - np.sin(2 * np.pi * np.arange(n) / 24))
        latency = np.clip(
            np.random.exponential(latency_base, n) + 100,
            80, 4500
        ).astype(int)

        timestamps = pd.date_range('2025-03-10', periods=n, freq='h')
        hours = timestamps.hour

        return pd.DataFrame({
            'timestamp': timestamps,
            'hour': hours,
            'api_calls': calls,
            'avg_latency_ms': latency,
            'is_weekend': [d.weekday() >= 5 for d in timestamps],
        })

    @staticmethod
    def prepare_boxplot_data(df):
        """按小时分组的箱线图数据"""
        hourly_stats = df.groupby('hour')['avg_latency_ms'].agg(
            ['min', 'q1', 'median', 'q3', 'max', 'count']
        )
        hourly_stats.columns = ['Min', 'Q1', 'Median', 'Q3', 'Max', 'Count']

        outliers_list = []
        for hour in range(24):
            hour_data = df[df['hour'] == hour]['avg_latency_ms']
            Q1, Q3 = hour_data.quantile([0.25, 0.75])
            IQR = Q3 - Q1
            upper = Q3 + 1.5 * IQR
            outlier_vals = hour_data[hour_data > upper].tolist()
            if outlier_vals:
                outliers_list.append({'hour': hour, 'outliers': outlier_vals})

        return hourly_stats.round(0), outliers_list


dist_df = APIMetricsVisualizer.generate_hourly_distribution(days=7)
boxplot_data, outliers = APIMetricsVisualizer.prepare_boxplot_data(dist_df)

print("=== 各小时延迟箱线图数据 ===")
print(boxplot_data.to_string())

if outliers:
    print(f"\n⚠️ 异常值 ({len(outliers)} 个小时段):")
    for o in outliers[:5]:
        print(f"  Hour {o['hour']:02d}: {o['outliers']}")
```
