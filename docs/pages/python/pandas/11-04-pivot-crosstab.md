---
title: 数据透视表与交叉表
description: pivot_table() / crosstab() / 多级透视 / 填充值 / 边际汇总 / LLM 评估矩阵
---
# 透视表与交叉表


## pivot_table()：灵活的数据透视

### 基础透视

```python
import pandas as pd
import numpy as np

np.random.seed(42)

df = pd.DataFrame({
    'model': np.random.choice(['GPT-4o', 'Claude', 'Llama', 'Qwen'], 200),
    'task': np.random.choice(['chat', 'code', 'math', 'reasoning'], 200),
    'score': np.random.uniform(65, 95, 200),
    'latency_ms': np.random.randint(200, 3000, 200),
})

pt = df.pivot_table(
    index='model',
    columns='task',
    values='score',
    aggfunc='mean'
).round(2)

print("=== 模型 × 任务 平均分 ===")
print(pt)
```

### 多聚合函数

```python
multi_pt = df.pivot_table(
    index='model',
    columns='task',
    values='score',
    aggfunc=['mean', 'count', 'std']
)
print(multi_pt.round(2))
```

### 添加汇总行列（margins）

```python
margins_pt = df.pivot_table(
    index='model',
    columns='task',
    values='score',
    aggfunc='mean',
    margins=True,
    margins_name='总计'
).round(2)

print("=== 含边际汇总的透视表 ===")
print(margins_pt)
```

### 处理缺失值（fill_value）

```python
import pandas as pd

df_sparse = pd.DataFrame({
    'model': ['GPT-4o', 'GPT-4o', 'Claude', 'Claude', 'Llama'],
    'task': ['chat', 'code', 'chat', 'math', 'code'],
    'score': [90, 88, 87, 82, 78],
})

pt_nan = df_sparse.pivot_table(index='model', columns='task', values='score', aggfunc='mean')

pt_filled = df_sparse.pivot_table(
    index='model',
    columns='task',
    values='score',
    aggfunc='mean',
    fill_value=0.0
)
print(pt_filled)
```

## crosstab()：频率交叉表

### 基础用法

```python
import pandas as pd
import numpy as np

np.random.seed(42)

n = 300
survey = pd.DataFrame({
    'model_preference': np.random.choice(['GPT-4o', 'Claude', 'Llama'], n,
                                         p=[0.45, 0.35, 0.20]),
    'use_case': np.random.choice(['coding', 'writing', 'analysis', 'chat'], n,
                                 p=[0.30, 0.25, 0.25, 0.20]),
    'experience': np.random.choice(['beginner', 'intermediate', 'expert'], n,
                                  p=[0.30, 0.45, 0.25]),
})

ct = pd.crosstab(survey['model_preference'], survey['use_case'])
print("=== 模型偏好 × 使用场景 频数表 ===")
print(ct)
```

### 归一化（百分比）

```python
ct_row_pct = pd.crosstab(
    survey['model_preference'],
    survey['use_case'],
    normalize='index'
).round(3) * 100

print("=== 行百分比（各模型用户的使用场景分布）===")
print(ct_row_pct)

ct_col_pct = pd.crosstab(
    survey['model_preference'],
    survey['use_case'],
    normalize='columns'
).round(3) * 100

print("\n=== 列百分比（各场景的用户模型分布）===")
print(ct_col_pct)
```

### 多层交叉表 + 聚合

```python
import pandas as pd
import numpy as np

np.random.seed(42)

eval_data = pd.DataFrame({
    'model': np.random.choice(['GPT-4o', 'Claude', 'Llama', 'Qwen'], 150),
    'difficulty': np.random.choice(['easy', 'medium', 'hard'], 150),
    'domain': np.random.choice(['tech', 'finance', 'medical'], 150),
    'score': np.random.uniform(60, 98, 150),
})

ct_3d = pd.crosstab(
    [eval_data['model'], eval_data['difficulty']],
    eval_data['domain'],
    values=eval_data['score'],
    aggfunc='mean'
).round(1)

print(ct_3d)
```

### 带 margins 的交叉表

```python
ct_margins = pd.crosstab(
    survey['model_preference'],
    survey['use_case'],
    margins=True,
    margins_name='合计'
)
print(ct_margins)
```

## 多级索引透视表

```python
import pandas as pd
import numpy as np

np.random.seed(42)

df = pd.DataFrame({
    'model_family': ['OpenAI']*60 + ['Anthropic']*60 + ['Meta']*40 + ['Alibaba']*40,
    'model': ['GPT-4o']*30 + ['GPT-4o-mini']*30 +
             ['Claude-Sonnet']*35 + ['Claude-Haiku']*25 +
             ['Llama-70B']*20 + ['Llama-8B']*20 +
             ['Qwen-72B']*15 + ['Qwen-7B']*25,
    'metric': ['accuracy', 'latency', 'cost'] * 80,
    'value': np.random.uniform(0.5, 1.0, 200),
})

multi_idx_pt = df.pivot_table(
    index=['model_family', 'model'],
    columns='metric',
    values='value',
    aggfunc='mean'
).round(3)

print(multi_idx_pt)
```

## LLM 场景：完整 Benchmark 对比矩阵

```python
import pandas as pd
import numpy as np

class BenchmarkMatrix:
    """LLM Benchmark 对比矩阵生成器"""

    def __init__(self):
        self.models = {
            'GPT-4o': {'base': 89, 'price': 2.50},
            'Claude-3.5-Sonnet': {'base': 89, 'price': 3.00},
            'Llama-3.1-70B': {'base': 84, 'price': 0.27},
            'Qwen2.5-72B': {'base': 84, 'price': 0.27},
            'DeepSeek-V3': {'base': 86, 'price': 0.14},
            'Gemini-1.5-Pro': {'base': 87, 'price': 1.25},
        }
        self.benchmarks = {
            'MMLU': {'weight': 0.18, 'noise': 3},
            'HumanEval': {'weight': 0.16, 'noise': 4},
            'MATH': {'weight': 0.12, 'noise': 6},
            'GPQA': {'weight': 0.10, 'noise': 5},
            'BBH': {'weight': 0.12, 'noise': 4},
            'IFEval': {'weight': 0.08, 'noise': 3},
            'MUSR': {'weight': 0.08, 'noise': 5},
            'CMMLU': {'weight': 0.08, 'noise': 3},
            'C-Eval': {'weight': 0.08, 'noise': 4},
        }

    def generate_data(self, n_runs=5, seed=42):
        np.random.seed(seed)
        rows = []
        for model, info in self.models.items():
            for bm, bm_info in self.benchmarks.items():
                for run in range(n_runs):
                    score = round(info['base'] + np.random.randn() * bm_info['noise']
                                  + (5 if 'HumanEval' in bm else -3 if 'MATH' in bm else 0), 1)
                    rows.append({
                        'model': model,
                        'benchmark': bm,
                        'run_id': run + 1,
                        'score': max(30, min(99, score)),
                    })
        return pd.DataFrame(rows)

    def build_matrix(self, df):
        matrix = df.pivot_table(
            index='model',
            columns='benchmark',
            values='score',
            aggfunc='max'
        ).round(1)

        for model in matrix.index:
            weighted_sum = sum(
                matrix.loc[model, bm] * self.benchmarks[bm]['weight']
                for bm in self.benchmarks if bm in matrix.columns
            )
            total_weight = sum(self.benchmarks[bm]['weight'] for bm in self.benchmarks)
            matrix.loc[model, 'Weighted_Avg'] = round(weighted_sum / total_weight, 1)

        return matrix.sort_values('Weighted_Avg', ascending=False)

    def add_rankings(self, matrix):
        rank_matrix = matrix.rank(ascending=False, method='dense').astype(int)
        rank_matrix.columns = [f'{c}_rank' for c in rank_matrix.columns]
        combined = pd.concat([matrix, rank_matrix], axis=1)
        return combined

    def add_heat_annotations(self, matrix, top_n=3):
        """为 Top N 添加标注"""
        annotated = matrix.copy()
        for col in annotated.select_dtypes(include=[np.number]).columns:
            if col.endswith('_rank'):
                continue
            top_values = annotated[col].nlargest(top_n)
            for i, (idx, val) in enumerate(top_values.items()):
                medal = {0: ' 🥇', 1: ' 🥈', 2: ' 🥉'}.get(i, '')
                annotated.loc[idx, col] = f"{val}{medal}"
        return annotated


builder = BenchmarkMatrix()
raw_df = builder.generate_data(n_runs=5)

matrix = builder.build_matrix(raw_df)
full_report = builder.add_rankings(matrix)

print("=== LLM Benchmark 完整对比矩阵 ===")
print(full_report.to_string())

print(f"\n各维度最佳:")
for bm in builder.benchmarks:
    if bm in full_report.columns:
        best = full_report[bm].idxmax()
        print(f"  {bm}: {best} ({full_report.loc[best, bm]})")

price_perf = raw_df.pivot_table(
    index='model',
    columns='benchmark',
    values='score',
    aggfunc='mean'
).round(1)

prices = pd.Series({m: info['price'] for m, info in builder.models.items()}, name='price')
price_perf = price_perf.join(prices)

for model in price_perf.index:
    base_score = builder.models.get(model, {}).get('base', 80)
    price_perf.loc[model, 'value_score'] = round(base_score / price_perf.loc[model, 'price'], 1)

print(f"\n=== 性价比排名 ===")
print(price_perf[['price', 'value_score']].sort_values('value_score', ascending=False))
```
