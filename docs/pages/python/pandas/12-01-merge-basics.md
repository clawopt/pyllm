---
title: merge() 基础：内连接、外连接、左/右连接
description: 四种连接类型 / on 参数 / how 参数 / 多键合并 / LLM 场景：评估数据关联
---
# Merge 合并基础


## 为什么需要数据合并

在 LLM 开发中，你经常需要把来自不同来源的数据"拼"在一起：

- **模型评估分数** + **API 调用成本** → 完整性价比报告
- **SFT 训练数据** + **人工标注质量** → 带标签的数据集
- **RAG 检索日志** + **用户反馈** → 检索效果分析
- **Token 统计** + **价格表** → 费用计算

`merge()` 就是 Pandas 中实现这类需求的瑞士军刀。

## 四种连接类型图解

```
左表 (df_left)          右表 (df_right)
┌──────┬──────┐        ┌──────┬──────┐
│  key │ val_a│        │  key │ val_b│
├──────┼──────┤        ├──────┼──────┤
│   A  │   1  │        │   A  │  10  │
│   B  │   2  │        │   B  │  20  │
│   C  │   3  │        │   D  │  40  │
│   D  │   4  │        │   E  │  50  │
└──────┴──────┘        └──────┴──────┘

inner (交集):     left outer:       right outer:      full outer (并集):
A, B              A, B, C, D         A, B, D, E        A, B, C, D, E
```

## 基础用法

```python
import pandas as pd
import numpy as np

models_info = pd.DataFrame({
    'model': ['GPT-4o', 'Claude', 'Llama', 'Qwen', 'DeepSeek'],
    'params_B': [1760, 175, 70, 72, 37],
    'context_window': [128000, 200000, 131072, 131072, 65536],
})

eval_scores = pd.DataFrame({
    'model': ['GPT-4o', 'Claude', 'Llama', 'Qwen', 'Gemini'],
    'MMLU': [88.7, 89.2, 84.5, 83.5, 86.8],
    'HumanEval': [92.0, 93.1, 82.4, 78.6, 85.1],
})

inner = pd.merge(models_info, eval_scores, on='model', how='inner')
print("=== inner join（两边都有）===")
print(inner)

left = pd.merge(models_info, eval_scores, on='model', how='left')
print("\n=== left join（保留左表全部）===")
print(left)

right = pd.merge(models_info, eval_scores, on='model', how='right')
print("\n=== right join（保留右表全部）===")
print(right)

outer = pd.merge(models_info, eval_scores, on='model', how='outer')
print("\n=== outer join（并集）===")
print(outer)
```

## 多键合并

```python
import pandas as pd
import numpy as np

latency = pd.DataFrame({
    'model': ['GPT-4o', 'GPT-4o', 'Claude', 'Claude', 'Llama'],
    'task': ['chat', 'code', 'chat', 'code', 'chat'],
    'avg_latency_ms': [800, 1200, 650, 950, 350],
})

accuracy = pd.DataFrame({
    'model': ['GPT-4o', 'GPT-4o', 'Claude', 'Claude', 'Llama', 'Qwen'],
    'task': ['chat', 'code', 'chat', 'code', 'chat', 'code'],
    'accuracy': [0.92, 0.89, 0.91, 0.87, 0.84, 0.83],
})

multi_key = pd.merge(latency, accuracy, on=['model', 'task'], how='inner')
print("=== 多键合并 ===")
print(multi_key)
```

## 列名冲突处理

```python
import pandas as pd

df1 = pd.DataFrame({
    'id': [1, 2, 3],
    'name': ['Alice', 'Bob', 'Charlie'],
    'score': [88, 92, 75],
})

df2 = pd.DataFrame({
    'id': [1, 2, 4],
    'name': ['Alice', 'Bob', 'Diana'],
    'score': [90, 89, 80],
})

merged = pd.merge(df1, df2, on='id', how='outer')
print("默认后缀 (_x / _y):")
print(merged)

merged_custom = pd.merge(
    df1, df2, on='id',
    suffixes=('_exam1', '_exam2'),
    how='outer'
)
print("\n自定义后缀:")
print(merged_custom)
```

## indicator 参数：追踪来源

```python
import pandas as pd

merged = pd.merge(
    models_info, eval_scores,
    on='model',
    how='outer',
    indicator=True
)
print("=== 带 _merge 标记 ===")
print(merged[['model', 'params_B', 'MMLU', '_merge']])

only_left = merged[merged['_merge'] == 'left_only']
only_right = merged[merged['_merge'] == 'right_only']
print(f"\n只在左表: {only_left['model'].tolist()}")
print(f"只在右表: {only_right['model'].tolist()}")
```

## validate 参数：防止意外多对多

```python
import pandas as pd

df_a = pd.DataFrame({'key': [1, 1, 2], 'val_a': ['a1', 'a2', 'a3']})
df_b = pd.DataFrame({'key': [1, 2], 'val_b': ['b1', 'b2']})

result = pd.merge(df_a, df_b, on='key', validate='many_to_one')
print(f"many_to_one 验证通过: {len(result)} 行")

try:
    bad = pd.merge(df_a, df_a, on='key', validate='one_to_one')
except Exception as e:
    print(f"\n验证失败（预期行为）: {type(e).__name__}")
```

## LLM 场景：评估数据与元信息关联

```python
import pandas as pd
import numpy as np

class ModelEvaluator:
    """模型评估数据合并分析器"""

    def __init__(self):
        self.model_meta = pd.DataFrame({
            'model': ['GPT-4o', 'Claude-3.5-Sonnet', 'Llama-3.1-70B',
                      'Qwen2.5-72B', 'DeepSeek-V3', 'Gemini-1.5-Pro'],
            'params_B': [1760, 175, 70, 72, 37, 未知],
            'context_win': [128000, 200000, 131072, 131072, 65536, 1000000],
            'vendor': ['OpenAI', 'Anthropic', 'Meta', 'Alibaba', 'DeepSeek', 'Google'],
            'price_input_per_1M': [2.50, 3.00, 0.27, 0.27, 0.14, 1.25],
            'price_output_per_1M': [10.0, 15.0, 0.60, 1.20, 0.28, 5.00],
        })

    def generate_eval_data(self, n=300, seed=42):
        np.random.seed(seed)
        models = self.model_meta['model'].tolist()
        tasks = ['MMLU', 'HumanEval', 'MATH', 'GPQA', 'BBH']
        rows = []
        base_map = dict(zip(self.model_meta['model'], [88, 89, 84, 83, 86, 87]))
        for _ in range(n):
            m = np.random.choice(models)
            t = np.random.choice(tasks)
            base = base_map.get(m, 80)
            noise = 6 if t == 'MATH' else 3
            rows.append({
                'model': m,
                'benchmark': t,
                'score': round(np.clip(base + np.random.randn() * noise, 35, 99), 1),
                'n_shots': np.random.choice([0, 5]),
            })
        return pd.DataFrame(rows)

    def build_full_report(self, eval_df):
        full = pd.merge(
            self.model_meta,
            eval_df,
            on='model',
            how='left'
        )

        pivot = full.pivot_table(
            index=['model', 'vendor', 'params_B', 'context_win',
                   'price_input_per_1M', 'price_output_per_1M'],
            columns='benchmark',
            values='score',
            aggfunc='mean'
        ).round(1).reset_index()

        numeric_cols = pivot.select_dtypes(include=[np.number]).columns
        bench_cols = [c for c in numeric_cols if c not in
                      ['params_B', 'context_win', 'price_input_per_1M', 'price_output_per_1M']]
        if len(bench_cols) > 0:
            pivot['avg_score'] = pivot[bench_cols].mean(axis=1).round(1)
            pivot['total_price_per_1M'] = (
                pivot['price_input_per_1M'] + pivot['price_output_per_1M']
            )
            pivot['value_ratio'] = (
                pivot['avg_score'] / pivot['total_price_per_1M']
            ).round(2)

        return pivot.sort_values('avg_score', ascending=False)


evaluator = ModelEvaluator()
eval_df = evaluator.generate_eval_data()
report = evaluator.build_full_report(eval_df)

print("=== 模型完整评估报告（元信息 + 分数）===")
display_cols = ['model', 'vendor', 'params_B', 'avg_score', 'value_ratio']
if 'MMLU' in report.columns:
    display_cols.insert(4, 'MMLU')
if 'HumanEval' in report.columns:
    display_cols.insert(5, 'HumanEval')
print(report[display_cols].to_string(index=False))
```
