---
title: 数据重塑：宽长格式转换
description: melt() 逆透视 / pivot() 透视 / wide_to_long() / stack() 与 unstack()
---
# 数据重塑操作


## 为什么需要数据重塑

在 LLM 数据处理中，你经常遇到两种格式：

**宽格式（Wide）**：每个变量一列，适合人类阅读和 Excel 展示
```python
import pandas as pd

wide_df = pd.DataFrame({
    'model': ['GPT-4o', 'Claude', 'Llama', 'Qwen'],
    'MMLU': [88.7, 89.2, 84.5, 83.5],
    'HumanEval': [92.0, 93.1, 82.4, 78.6],
    'MATH': [76.6, 71.1, 50.4, 60.8],
    'GPQA': [53.9, 59.3, 34.2, 40.1],
})
print("=== 宽格式 ===")
print(wide_df)
```

**长格式（Long）**：所有值堆在一列，用另一列标识变量名，适合绘图和分析
```python
long_df = pd.DataFrame({
    'model': ['GPT-4o']*4 + ['Claude']*4 + ['Llama']*4 + ['Qwen']*4,
    'metric': ['MMLU','HumanEval','MATH','GPQA']*4,
    'score': [88.7,92.0,76.6,53.9, 89.2,93.1,71.1,59.3,
              84.5,82.4,50.4,34.2, 83.5,78.6,60.8,40.1],
})
print("\n=== 长格式 ===")
print(long_df)
```

## melt()：从宽转长（逆透视）

### 基础用法

```python
import pandas as pd

df = pd.DataFrame({
    'model': ['GPT-4o', 'Claude', 'Llama'],
    'MMLU': [88.7, 89.2, 84.5],
    'HumanEval': [92.0, 93.1, 82.4],
    'MATH': [76.6, 71.1, 50.4],
})

melted = df.melt(
    id_vars=['model'],
    var_name='metric',
    value_name='score'
)

print(melted)
```

### 选择性融化部分列

```python
import pandas as pd

df = pd.DataFrame({
    'model': ['GPT-4o', 'Claude'],
    'params_B': [1760, 175],
    'MMLU': [88.7, 89.2],
    'HumanEval': [92.0, 93.1],
    'price': [2.50, 3.00],
})

partial_melt = df.melt(
    id_vars=['model', 'params_B', 'price'],
    value_vars=['MMLU', 'HumanEval'],
    var_name='benchmark',
    value_name='score'
)
print(partial_melt)
```

## pivot()：从长转宽（透视）

### 基础用法

```python
import pandas as pd

long_df = pd.DataFrame({
    'model': ['GPT-4o', 'GPT-4o', 'Claude', 'Claude', 'Llama', 'Llama'],
    'metric': ['MMLU', 'HumanEval', 'MMLU', 'HumanEval', 'MMLU', 'HumanEval'],
    'score': [88.7, 92.0, 89.2, 93.1, 84.5, 82.4],
})

pivoted = long_df.pivot(
    index='model',
    columns='metric',
    values='score'
)

print(pivoted)
```

### 处理重复值（pivot_table）

当 `index + columns` 组合有重复时，`pivot()` 会报错，需要用 `pivot_table()`：

```python
import pandas as pd
import numpy as np

long_df = pd.DataFrame({
    'model': ['GPT-4o', 'GPT-4o', 'GPT-4o', 'Claude', 'Claude'],
    'metric': ['MMLU', 'MMLU', 'HumanEval', 'MMLU', 'HumanEval'],
    'score': [88.7, 87.5, 92.0, 89.2, 93.1],
    'run_id': ['v1', 'v2', 'v1', 'v1', 'v1'],
})

pt = long_df.pivot_table(
    index='model',
    columns='metric',
    values='score',
    aggfunc='mean'
)
print(pt)

pt_multi = long_df.pivot_table(
    index='model',
    columns='metric',
    values='score',
    aggfunc=['mean', 'count']
)
print(pt_multi)
```

## wide_to_long()：处理规律命名的宽表

```python
import pandas as pd

df = pd.DataFrame({
    'model': ['A', 'B'],
    'score_2024Q1': [85, 90],
    'score_2024Q2': [87, 88],
    'score_2024Q3': [91, 92],
    'latency_2024Q1': [800, 650],
    'latency_2024Q2': [750, 600],
    'latency_2024Q3': [720, 580],
})

result = pd.wide_to_long(
    df,
    stubnames=['score', 'latency'],
    i='model',
    j='quarter',
    sep='_',
    suffix='.+'
).reset_index()

print(result.sort_values(['model', 'quarter']))
```

## stack() 与 unstack()

```python
import pandas as pd
import numpy as np

np.random.seed(42)

df = pd.DataFrame(
    np.random.randint(70, 95, (3, 4)),
    index=['GPT-4o', 'Claude', 'Llama'],
    columns=['MMLU', 'HumanEval', 'MATH', 'GPQA']
)
print("=== 原始 DataFrame ===")
print(df)

stacked = df.stack()
print("\n=== stack() 结果 ===")
print(stacked)

unstacked = stacked.unstack()
print("\n=== unstack() 还原 ===")
print(unstacked)

unstacked_level0 = stacked.unstack(level=0)
print("\n=== 按 level=0 展开 ===")
print(unstacked_level0)
```

## LLM 场景：评估报告的数据重塑

```python
import pandas as pd
import numpy as np

np.random.seed(42)

models = ['GPT-4o', 'Claude-3.5-Sonnet', 'Llama-3.1-70B', 'Qwen2.5-72B', 'DeepSeek-V3']
metrics = ['MMLU', 'HumanEval', 'MATH', 'GPQA', 'BBH', 'MUSR']

raw_eval = []
for model in models:
    for metric in metrics:
        base = {'GPT-4o': 88, 'Claude-3.5-Sonnet': 89, 'Llama-3.1-70B': 84,
                'Qwen2.5-72B': 83, 'DeepSeek-V3': 86}.get(model, 80)
        raw_eval.append({
            'model': model,
            'metric': metric,
            'score': round(base + np.random.randn() * 5, 1),
            'n_shots': np.random.choice([0, 5]),
        })

eval_df = pd.DataFrame(raw_eval)

print(f"原始数据: {len(eval_df)} 行 × {len(eval_df.columns)} 列")
print(eval_df.head(6))
```

### 步骤一：构建对比矩阵

```python
matrix = eval_df.pivot_table(
    index='model',
    columns='metric',
    values='score',
    aggfunc='max'
)

print("=== 模型 × 指标 对比矩阵 ===")
print(matrix.round(1))

matrix['mean'] = matrix.mean(axis=1).round(1)
matrix['rank'] = matrix.drop(columns='mean').mean(axis=1).rank(ascending=False).astype(int)
print(matrix.sort_values('rank'))
```

### 步骤二：熔化后绘制趋势图

```python
melted = eval_df.melt(
    id_vars=['model', 'n_shots'],
    var_name='benchmark',
    value_name='score'
)

summary = melted.groupby('model')['score'].agg(['mean', 'std', 'min', 'max']).round(2)
summary.columns = ['平均分', '标准差', '最低分', '最高分']
print(summary.sort_values('平均分', ascending=False))
```

### 步骤三：找出各模型最强/最弱项

```python
def find_extremes(group):
    best = group.loc[group['score'].idxmax()]
    worst = group.loc[group['score'].idxmin()]
    return pd.Series({
        '最强项': f"{best['metric']} ({best['score']:.1f})",
        '最弱项': f"{worst['metric']} ({worst['score']:.1f})",
        '分差': best['score'] - worst['score']
    })

extremes = eval_df.groupby('model').apply(find_extremes)
print("\n=== 各模型最强/最弱项 ===")
print(extremes)
```
