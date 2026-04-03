---
title: 分组后拆分与导出
description: 按组拆分为字典 / 导出为多个文件 / 分层抽样 / 组间对比分析
---
# 分组拆分与导出


## 将分组结果拆分为字典

### 基础拆分

```python
import pandas as pd
import numpy as np

np.random.seed(42)

df = pd.DataFrame({
    'model': ['GPT-4o']*30 + ['Claude']*25 + ['Llama']*20 + ['Qwen']*15,
    'score': np.random.uniform(65, 95, 90),
    'latency_ms': np.random.randint(200, 3000, 90),
})

group_dict = {name: group for name, group in df.groupby('model')}
print(f"拆分成 {len(group_dict)} 组:")
for name, g in group_dict.items():
    print(f"  {name}: {len(g)} 行")

gpt_data = group_dict['GPT-4o']
print(f"\nGPT-4o 组前3行:")
print(gpt_data.head(3))
```

### to_dict() 的不同 orient 模式

```python
as_nested = df.head(5).to_dict(orient='dict')

as_list = df.head(5).to_dict(orient='list')

as_records = df.head(5).to_dict(orient='records')
print(as_records[0])
```

## 按组导出为多个文件

### 导出 CSV/JSONL

```python
import pandas as pd
import numpy as np
import os

np.random.seed(42)

output_dir = '/tmp/split_by_model'
os.makedirs(output_dir, exist_ok=True)

df = pd.DataFrame({
    'model': ['GPT-4o']*10 + ['Claude']*8 + ['Llama']*12,
    'prompt': [f'prompt_{i}' for i in range(30)],
    'response_len': np.random.randint(50, 2000, 30),
})

for name, group in df.groupby('model'):
    safe_name = name.replace('-', '_').replace('.', '')
    csv_path = f'{output_dir}/{safe_name}_data.csv'
    jsonl_path = f'{output_dir}/{safe_name}_data.jsonl'

    group.to_csv(csv_path, index=False)
    group.to_json(jsonl_path, orient='records', lines=True, force_ascii=False)
    print(f"✓ {name}: {len(group)} 行 → {csv_path}")

exported = {
    name: {
        'csv': f'{output_dir}/{name.replace("-","_")}.csv',
        'jsonl': f'{output_dir}/{name.replace("-","_")}.jsonl',
        'count': len(group),
    }
    for name, group in df.groupby('model')
}
```

### 导出 Parquet（推荐，高效）

```python
for name, group in df.groupby('model'):
    safe_name = name.replace('-', '_').replace('.', '')
    parquet_path = f'{output_dir}/{safe_name}.parquet'
    group.to_parquet(parquet_path, index=False)
    file_size = os.path.getsize(parquet_path)
    print(f"✓ {name}: {file_size/1024:.1f} KB")
```

## 分层抽样（Stratified Sampling）

### 等比例分层抽样

```python
import pandas as pd
import numpy as np

np.random.seed(42)

df = pd.DataFrame({
    'category': ['tech']*200 + ['finance']*150 + ['medical']*100 + ['legal']*50,
    'value': np.random.randn(500),
})

def stratified_sample(df, group_col, frac=0.1, min_samples=3, seed=42):
    """分层抽样：每组按比例抽取"""
    rng = np.random.RandomState(seed)
    sampled = []
    for name, group in df.groupby(group_col):
        n = max(int(len(group) * frac), min_samples)
        n = min(n, len(group))
        sampled.append(group.sample(n, random_state=rng))
    result = pd.concat(sampled, ignore_index=True)
    return result


sampled = stratified_sample(df, 'category', frac=0.1, min_samples=5)

print("=== 分层抽样前后对比 ===")
comparison = pd.DataFrame({
    '原始数量': df['category'].value_counts(),
    '抽样数量': sampled['category'].value_counts(),
    '抽样比例 (%)': (sampled['category'].value_counts() / df['category'].value_counts() * 100).round(1),
})
print(comparison.sort_index())
```

### 等量分层抽样（每组固定数量）

```python
def equal_stratified_sample(df, group_col, n_per_group=10, seed=42):
    """等量分层抽样：每组固定抽 N 个"""
    rng = np.random.RandomState(seed)
    sampled = []
    for name, group in df.groupby(group_col):
        if len(group) >= n_per_group:
            sampled.append(group.sample(n_per_group, random_state=rng))
        else:
            sampled.append(group)
    return pd.concat(sampled, ignore_index=True)


equal_sampled = equal_stratified_sample(df, 'category', n_per_group=15)
print(f"\n等量抽样: {len(equal_sampled)} 条")
print(equal_sampled['category'].value_counts().sort_index())
```

## 组间对比分析

### 两两对比矩阵

```python
import pandas as pd
import numpy as np
from scipy import stats

np.random.seed(42)

groups_data = []
for model in ['GPT-4o', 'Claude', 'Llama']:
    base = {'GPT-4o': 88, 'Claude': 87, 'Llama': 82}[model]
    groups_data.extend([{
        'model': model,
        'score': round(base + np.random.randn() * 5, 1)
    } for _ in range(50)])

scores_df = pd.DataFrame(groups_data)

models = scores_df['model'].unique()
pairwise_results = []

for i, m1 in enumerate(models):
    for m2 in models[i+1:]:
        s1 = scores_df[scores_df['model'] == m1]['score']
        s2 = scores_df[scores_df['model'] == m2]['score']

        t_stat, p_value = stats.ttest_ind(s1, s2)
        mean_diff = s1.mean() - s2.mean()

        pairwise_results.append({
            'model_a': m1,
            'model_b': m2,
            'mean_diff': round(mean_diff, 2),
            't_statistic': round(t_stat, 3),
            'p_value': round(p_value, 4),
            'significant': '***' if p_value < 0.001 else ('**' if p_value < 0.01 else ('*' if p_value < 0.05 else 'ns')),
        })

pairwise_df = pd.DataFrame(pairwise_results)
print("=== 模型两两对比 (t-test) ===")
print(pairwise_df.to_string(index=False))
print("\n显著性: *** p<0.001, ** p<0.01, * p<0.05, ns: 不显著")
```

### 多组方差分析概要

```python
def anova_summary(df, value_col, group_col):
    """单因素 ANOVA 摘要"""
    groups = [g[value_col].values for _, g in df.groupby(group_col)]
    f_stat, p_val = stats.f_oneway(*groups)

    overall_mean = df[value_col].mean()
    group_means = df.groupby(group_col)[value_col].mean()

    ss_between = sum(len(g) * (m - overall_mean)**2
                     for (_, g), m in zip(df.groupby(group_col), group_means))
    ss_total = ((df[value_col] - overall_mean) ** 2).sum()

    return {
        'F统计量': round(f_stat, 3),
        'p值': round(p_val, 6),
        '组间平方和': round(ss_between, 2),
        '总平方和': round(ss_total, 2),
        '解释变异比': round(ss_between / ss_total * 100, 2),
    }

result = anova_summary(scores_df, 'score', 'model')
print("\n=== ANOVA 分析摘要 ===")
for k, v in result.items():
    print(f"  {k}: {v}")
```
