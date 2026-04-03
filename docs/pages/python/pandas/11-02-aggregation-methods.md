---
title: 聚合方法详解
description: agg() / aggregate()、多聚合函数、命名聚合、自定义聚合函数、transform vs agg
---
# 聚合方法详解


## 常用内置聚合函数速查

| 函数 | 说明 | 适用类型 | NaN 处理 |
|------|------|---------|----------|
| `count()` | 非空值计数 | 全部 | 自动跳过 |
| `sum()` | 求和 | 数值 | 跳过 NaN |
| `mean()` | 平均值 | 数值 | 跳过 NaN |
| `median()` | 中位数 | 数值 | 跳过 NaN |
| `std()` / `var()` | 标准差/方差 | 数值 | 跳过 NaN（ddof=1） |
| `min()` / `max()` | 最小/最大 | 可排序 | 跳过 NaN |
| `first()` / `last()` | 首/末值 | 全部 | 保留 NaN |
| `nunique()` | 唯一值数量 | 全部 | 不计 NaN |
| `idxmax()` / `idxmin()` | 最大/最小值索引 | 可排序 | 跳过 NaN |

## 单个聚合函数

```python
import pandas as pd
import numpy as np

np.random.seed(42)

df = pd.DataFrame({
    'model': ['GPT-4o']*50 + ['Claude']*50 + ['Llama']*50,
    'latency_ms': np.concatenate([
        np.random.normal(800, 150, 50),
        np.random.normal(650, 120, 50),
        np.random.normal(350, 80, 50),
    ]).astype(int),
    'tokens': np.random.randint(100, 2000, 150),
})

avg_latency = df.groupby('model')['latency_ms'].mean()
total_tokens = df.groupby('model')['tokens'].sum()
max_latency = df.groupby('model')['latency_ms'].max()

print("平均延迟:")
print(avg_latency.round(1))
```

## 多聚合函数（agg）

### 方式一：列表传入多个函数名

```python
import pandas as pd

multi_agg = df.groupby('model')['latency_ms'].agg(['count', 'mean', 'std', 'min', 'max'])
print(multi_agg.round(1))
```

### 方式二：字典对不同列指定不同聚合

```python
import pandas as pd

agg_dict = df.groupby('model').agg({
    'latency_ms': ['mean', 'std', 'median'],
    'tokens': ['sum', 'mean'],
})
print(agg_dict)
```

### 方式三：命名聚合（推荐，Pandas 0.25+）

```python
import pandas as pd

named_agg = df.groupby('model').agg(
    call_count=('latency_ms', 'count'),
    avg_latency=('latency_ms', 'mean'),
    p95_latency=('latency_ms', lambda x: x.quantile(0.95)),
    total_tokens=('tokens', 'sum'),
    max_latency=('latency_ms', 'max'),
)
print(named_agg.round(2))
```

**命名聚合的优势**：
- 输出列名清晰，不需要处理 MultiIndex
- 可以对同一列使用不同聚合函数并分别命名
- 结果是扁平的 DataFrame，便于后续操作

## 自定义聚合函数

```python
import pandas as pd
import numpy as np

def cv(series):
    """变异系数 = 标准差 / 均值"""
    return series.std() / series.mean() if series.mean() != 0 else np.nan

def iqr_range(series):
    """四分位距"""
    return series.quantile(0.75) - series.quantile(0.25)

def outlier_count(series, k=1.5):
    """异常值数量（IQR 方法）"""
    Q1, Q3 = series.quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower, upper = Q1 - k*IQR, Q3 + k*IQR
    return ((series < lower) | (series > upper)).sum()

custom_agg = df.groupby('model').agg(
    mean_lat=('latency_ms', 'mean'),
    std_lat=('latency_ms', 'std'),
    cv_lat=('latency_ms', cv),
    iqr_lat=('latency_ms', iqr_range),
    outliers=('latency_ms', lambda x: outlier_count(x)),
)
print(custom_agg.round(2))
```

## transform() vs agg() 的关键区别

```python
import pandas as pd
import numpy as np

np.random.seed(42)

df = pd.DataFrame({
    'category': ['A']*5 + ['B']*5 + ['C']*5,
    'value': [10, 12, 11, 13, 14, 20, 22, 21, 23, 24, 30, 35, 32, 33, 31],
})

agg_result = df.groupby('category')['value'].agg(['mean', 'std'])
print("agg() 结果 (3行 = 3组):")
print(agg_result)

df['group_mean'] = df.groupby('category')['value'].transform('mean')
df['z_score'] = df.groupby('category')['value'].transform(
    lambda x: (x - x.mean()) / x.std()
)
df['pct_of_group'] = df.groupby('category')['value'].transform(
    lambda x: x / x.sum() * 100
).round(1)

print("\ntransform() 结果 (15行 = 原数据行数):")
print(df)
```

## 同时使用 agg 和 transform

```python
import pandas as pd
import numpy as np

np.random.seed(42)

eval_df = pd.DataFrame({
    'model': ['GPT-4o']*20 + ['Claude']*20 + ['Llama']*20,
    'benchmark': ['MMLU','HumanEval','MATH','GPQA']*15,
    'score': np.random.uniform(70, 95, 60),
})

summary = eval_df.groupby('model').agg(
    avg_score=('score', 'mean'),
    std_score=('score', 'std'),
    count=('score', 'count'),
).round(2)

eval_df['model_avg'] = eval_df.groupby('model')['score'].transform('mean')
eval_df['above_avg'] = eval_df['score'] > eval_df['model_avg']
eval_df['deviation'] = (eval_df['score'] - eval_df['model_avg']).round(2)

print("=== 模型级汇总 ===")
print(summary.sort_values('avg_score', ascending=False))

print("\n=== 各模型内表现 ===")
for model in ['GPT-4o', 'Claude', 'Llama']:
    subset = eval_df[eval_df['model'] == model][['benchmark', 'score', 'deviation']]
    above = (subset['deviation'] > 0).sum()
    print(f"\n{model}: {above}/{len(subset)} 项高于均值")
```

## LLM 场景：评估数据的完整聚合分析

```python
import pandas as pd
import numpy as np

np.random.seed(42)

models = ['GPT-4o', 'Claude-3.5-Sonnet', 'Llama-3.1-70B', 'Qwen2.5-72B',
          'DeepSeek-V3', 'Gemini-1.5-Pro']
tasks = ['code_generation', 'math_reasoning', 'reading_comp',
         'translation_zh_en', 'translation_en_zh', 'instruction_following']

raw_eval = []
for _ in range(300):
    model = np.random.choice(models)
    task = np.random.choice(tasks)
    base = {'GPT-4o': 88, 'Claude-3.5-Sonnet': 89, 'Llama-3.1-70B': 84,
            'Qwen2.5-72B': 83, 'DeepSeek-V3': 87, 'Gemini-1.5-Pro': 86}.get(model, 80)
    raw_eval.append({
        'model': model,
        'task': task,
        'score': round(np.clip(base + np.random.randn() * 6, 40, 99), 1),
        'latency_s': round(abs(np.random.randn()) * 10 + 2, 2),
        'tokens_used': int(np.random.randint(500, 8000)),
    })

eval_data = pd.DataFrame(raw_eval)

full_report = eval_data.groupby('model').agg(
    n_samples=('score', 'count'),
    mean_score=('score', 'mean'),
    median_score=('score', 'median'),
    std_score=('score', 'std'),
    min_score=('score', 'min'),
    max_score=('score', 'max'),

    q25=('score', lambda x: x.quantile(0.25)),
    q75=('score', lambda x: x.quantile(0.75)),

    p95_latency=('latency_s', lambda x: x.quantile(0.95)),
    avg_tokens=('tokens_used', 'mean'),
    total_tokens=('tokens_used', 'sum'),

    cv_score=('score', lambda x: x.std()/x.mean() if x.mean()!=0 else np.nan),

    pass_rate_80=('score', lambda x: (x >= 80).mean()),
    pass_rate_90=('score', lambda x: (x >= 90).mean()),
).round(2)

full_report['pass_80_pct'] = (full_report['pass_rate_80'] * 100).round(1)
full_report['pass_90_pct'] = (full_report['pass_rate_90'] * 100).round(1)

print("=== LLM 完整评估报告 ===")
display_cols = ['n_samples', 'mean_score', 'median_score', 'std_score',
                'q25', 'q75', 'cv_score', 'pass_80_pct', 'pass_90_pct',
                'p95_latency', 'avg_tokens']
print(full_report[display_cols].sort_values('mean_score', ascending=False))

task_pivot = eval_data.pivot_table(
    index='model',
    columns='task',
    values='score',
    aggfunc='mean'
).round(1)

rank_matrix = task_pivot.rank(ascending=False, method='dense').astype(int)
print(f"\n=== 各任务排名矩阵 ===")
print(rank_matrix)
```
