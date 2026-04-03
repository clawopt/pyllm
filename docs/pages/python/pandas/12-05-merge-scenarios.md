---
title: 合并与连接综合实战
description: 模型评估多表关联 / SFT 数据源整合 / API 日志与成本关联 / 增量数据同步
---
# 合并场景实战


## 场景一：模型评估完整报告（五表合并）

```python
import pandas as pd
import numpy as np

class ModelReportBuilder:
    """模型评估报告构建器——多表合并实战"""

    def __init__(self):
        self.tables = {}

    def add_table(self, name, df):
        self.tables[name] = df
        return self

    def build(self, model_col='model'):
        if 'meta' not in self.tables:
            raise ValueError("需要 meta 表（模型元信息）")

        result = self.tables['meta'].copy()

        merge_plan = [
            ('benchmarks', ['MMLU', 'HumanEval', 'MATH', 'GPQA']),
            ('cost', ['price_input', 'price_output']),
            ('perf', ['avg_latency', 'throughput', 'p99_latency']),
            ('reliability', ['uptime_pct', 'error_rate']),
        ]

        for table_name, expected_cols in merge_plan:
            if table_name not in self.tables:
                for col in expected_cols:
                    result[col] = np.nan
                continue

            src = self.tables[table_name]
            available_cols = [c for c in expected_cols if c in src.columns]
            if available_cols:
                result = pd.merge(
                    result,
                    src[[model_col] + available_cols],
                    on=model_col,
                    how='left'
                )

        bench_cols = [c for c in
                      ['MMLU', 'HumanEval', 'MATH', 'GPQA', 'BBH']
                      if c in result.columns]
        if len(bench_cols) >= 2:
            result['avg_bench'] = result[bench_cols].mean(axis=1).round(1)
        if 'price_input' in result.columns and 'price_output' in result.columns:
            result['total_price'] = result['price_input'] + result['price_output']
            if 'avg_bench' in result.columns:
                result['value_score'] = (
                    result['avg_bench'] / result['total_price'].replace(0, np.nan)
                ).round(2)

        return result.sort_values('avg_bench', ascending=False)


np.random.seed(42)
builder = ModelReportBuilder()

builder.add_table('meta', pd.DataFrame({
    'model': ['GPT-4o', 'Claude-3.5-Sonnet', 'Llama-3.1-70B',
              'Qwen2.5-72B', 'DeepSeek-V3'],
    'params_B': [1760, 175, 70, 72, 37],
    'vendor': ['OpenAI', 'Anthropic', 'Meta', 'Alibaba', 'DeepSeek'],
}))

builder.add_table('benchmarks', pd.DataFrame({
    'model': ['GPT-4o', 'Claude-3.5-Sonnet', 'Llama-3.1-70B',
              'Qwen2.5-72B', 'DeepSeek-V3'],
    'MMLU': [88.7, 89.2, 84.5, 83.5, 86.8],
    'HumanEval': [92.0, 93.1, 82.4, 78.6, 87.2],
    'MATH': [76.6, 71.1, 50.4, 60.8, 65.2],
    'GPQA': [53.9, 59.3, 34.2, 40.1, 45.6],
}))

builder.add_table('cost', pd.DataFrame({
    'model': ['GPT-4o', 'Claude-3.5-Sonnet', 'Llama-3.1-70B',
              'Qwen2.5-72B', 'DeepSeek-V3'],
    'price_input': [2.50, 3.00, 0.27, 0.27, 0.14],
    'price_output': [10.0, 15.0, 0.60, 1.20, 0.28],
}))

builder.add_table('perf', pd.DataFrame({
    'model': ['GPT-4o', 'Claude-3.5-Sonnet', 'Llama-3.1-70B',
              'Qwen2.5-72B', 'DeepSeek-V3'],
    'avg_latency_ms': [800, 650, 350, 400, 280],
    'throughput_tps': [45, 55, 120, 110, 150],
}))

report = builder.build()
print("=== 模型完整评估报告 ===")
display_cols = ['model', 'vendor', 'params_B', 'MMLU', 'HumanEval',
                'avg_bench', 'total_price', 'value_score']
print(report[[c for c in display_cols if c in report.columns]].to_string(index=False))
```

## 场景二：API 调用日志与价格表关联计费

```python
import pandas as pd
import numpy as np

class CostCalculator:
    """基于合并的 API 成本计算器"""

    PRICE_TABLE = pd.DataFrame({
        'model': ['gpt-4o', 'gpt-4o-mini', 'claude-sonnet', 'claude-haiku',
                  'deepseek-chat', 'qwen-plus'],
        'input_per_1M': [2.50, 0.15, 3.00, 0.80, 0.14, 0.40],
        'output_per_1M': [10.0, 0.60, 15.0, 4.00, 0.28, 1.20],
    })

    @classmethod
    def calculate(cls, logs_df):
        merged = pd.merge(
            logs_df,
            cls.PRICE_TABLE,
            on='model',
            how='left'
        )

        merged['input_cost'] = (
            merged['prompt_tokens'] / 1e6 * merged['input_per_1M'].fillna(1.0)
        )
        merged['output_cost'] = (
            merged['completion_tokens'] / 1e6 * merged['output_per_1M'].fillna(3.0)
        )
        merged['total_cost'] = (merged['input_cost'] + merged['output_cost']).round(6)

        return merged


np.random.seed(42)
n = 500
logs = pd.DataFrame({
    'timestamp': pd.date_range('2025-03-01', periods=n, freq='10min'),
    'model': np.random.choice(
        ['gpt-4o', 'gpt-4o-mini', 'claude-sonnet', 'deepseek-chat'], n,
        p=[0.18, 0.35, 0.22, 0.25]
    ),
    'prompt_tokens': np.random.randint(50, 4000, n),
    'completion_tokens': np.random.randint(20, 5000, n),
})

costed_logs = CostCalculator.calculate(logs)

print("=== API 调用成本明细（前10条）===")
print(costed_logs[['model', 'prompt_tokens', 'completion_tokens',
                    'total_cost']].head(10).to_string(index=False))

summary = costed_logs.groupby('model').agg(
    calls=('model', 'count'),
    total_cost=('total_cost', 'sum'),
    avg_cost=('total_cost', 'mean'),
    total_input_tokens=('prompt_tokens', 'sum'),
    total_output_tokens=('completion_tokens', 'sum'),
).round(4)

print(f"\n=== 各模型费用汇总 ===")
print(summary.sort_values('total_cost', ascending=False))
```
