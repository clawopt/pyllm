---
title: 分组转换与过滤
description: transform() 保持形状 / filter() 按条件过滤组 / apply() 灵活操作 / 填充分组统计值
---
# transform 与 filter


## transform()：分组计算后广播回原形状

### 核心特性

```python
import pandas as pd
import numpy as np

np.random.seed(42)

df = pd.DataFrame({
    'model': ['GPT-4o']*10 + ['Claude']*10 + ['Llama']*10,
    'latency_ms': np.concatenate([
        np.random.normal(800, 150, 10),
        np.random.normal(650, 120, 10),
        np.random.normal(350, 80, 10),
    ]).astype(int),
})

df['group_mean'] = df.groupby('model')['latency_ms'].transform('mean')
df['group_std'] = df.groupby('model')['latency_ms'].transform('std')
df['z_score'] = df.groupby('model')['latency_ms'].transform(
    lambda x: (x - x.mean()) / x.std()
)
df['pct_rank'] = df.groupby('model')['latency_ms'].transform(
    lambda x: x.rank(pct=True)
).round(3)

print(df[['model', 'latency_ms', 'group_mean', 'z_score', 'pct_rank']].head(9))
```

**`transform()` vs `agg()` 一句话区分**：
- `agg()` → 每组返回 **1 个值**（结果行数 = 组数）
- `transform()` → 每组返回 **等长序列**（结果行数 = 原数据行数）

### 实用转换模式

```python
import pandas as pd
import numpy as np

np.random.seed(42)

scores_df = pd.DataFrame({
    'subject': ['math', 'math', 'math', 'code', 'code', 'code',
                'lang', 'lang', 'lang'],
    'student': list('ABCDEFGHI'),
    'score': [88, 92, 75, 95, 82, 78, 90, 85, 70],
})

scores_df['centered'] = scores_df.groupby('subject')['score'].transform(
    lambda x: x - x.mean()
).round(1)

scores_df['normalized'] = scores_df.groupby('subject')['score'].transform(
    lambda x: (x - x.min()) / (x.max() - x.min())
).round(3)

scores_df['percentile'] = scores_df.groupby('subject')['score'].transform(
    lambda x: x.rank(pct=True)
).round(3)

scores_df['cum_pct'] = scores_df.groupby('subject')['score'].transform(
    lambda x: x.cumsum() / x.sum()
).round(3)

print(scores_df)
```

## filter()：按组级条件过滤

### 基础用法

```python
import pandas as pd
import numpy as np

np.random.seed(42)

df = pd.DataFrame({
    'category': ['A']*15 + ['B']*8 + ['C']*20 + ['D']*3 + ['E']*12,
    'value': np.random.randn(58),
})

filtered = df.groupby('category').filter(lambda x: len(x) >= 10)
print(f"原始: {len(df)} 行, {df['category'].nunique()} 组")
print(f"过滤后: {len(filtered)} 行, {filtered['category'].nunique()} 组")
print(f"被过滤的组: {set(df['category'].unique()) - set(filtered['category'].unique())}")
```

### 多条件过滤

```python
import pandas as pd
import numpy as np

np.random.seed(42)

eval_data = pd.DataFrame({
    'model': ['GPT-4o']*25 + ['Claude']*18 + ['Llama']*30 + ['Qwen']*8 + ['DeepSeek']*5,
    'score': np.random.uniform(60, 98, 86),
})

quality_filtered = eval_data.groupby('model').filter(
    lambda g: len(g) >= 10 and g['score'].mean() >= 75
)
print("满足条件的模型:")
print(quality_filtered.groupby('model')['score'].agg(['count', 'mean']).round(2))
```

### filter vs 先 agg 再 merge

```python
r1 = df.groupby('category').filter(lambda x: len(x) >= 10)

group_sizes = df.groupby('category').size()
valid_groups = group_sizes[group_sizes >= 10].index
r2 = df[df['category'].isin(valid_groups)]

assert len(r1) == len(r2)
```

## fillna() 用分组统计填充缺失值

```python
import pandas as pd
import numpy as np

np.random.seed(42)

df = pd.DataFrame({
    'department': ['Engineering']*8 + ['Marketing']*6 + ['Sales']*7 + ['HR']*4,
    'salary': [25000, 28000, np.nan, 30000, 27000, np.nan, 29000, 31000,
               18000, 20000, np.nan, 19000, np.nan, 21000,
               15000, 16000, np.nan, 17000, 15500, np.nan, 16500,
               12000, 13000, np.nan, 12500],
})

df['salary_filled_median'] = df.groupby('department')['salary'].transform(
    lambda x: x.fillna(x.median())
)

df['salary_filled_mean'] = df.groupby('department')['salary'].transform(
    lambda x: x.fillna(round(x.mean(), 0))
)

print("=== 缺失值填充对比 ===")
missing_rows = df[df['salary'].isna()]
print(missing_rows[['department', 'salary', 'salary_filled_median', 'salary_filled_mean']])
```

## apply()：最灵活的分组操作

```python
import pandas as pd
import numpy as np

np.random.seed(42)

df = pd.DataFrame({
    'model': ['GPT-4o']*20 + ['Claude']*20 + ['Llama']*20,
    'score': np.random.uniform(65, 96, 60),
    'latency': np.random.exponential(500, 60).astype(int) + 100,
})

def describe_group(group):
    """自定义的组描述函数"""
    return pd.Series({
        'count': len(group),
        'mean_score': group['score'].mean(),
        'median_latency': group['latency'].median(),
        'score_range': group['score'].max() - group['score'].min(),
        'slow_count': (group['latency'] > 1000).sum(),
        'slow_rate': (group['latency'] > 1000).mean(),
    })

result = df.groupby('model').apply(describe_group)
print(result.round(2))
```

### apply 返回 DataFrame

```python
def top_n_in_group(group, n=3):
    """返回每组的前 N 行"""
    return group.nlargest(n, 'score')

top_per_model = df.groupby('model', group_keys=False).apply(top_n_in_group, n=3)
print("各模型 Top 3:")
print(top_per_model[['model', 'score', 'latency']])
```

## LLM 场景：SFT 数据的分组质量分析

```python
import pandas as pd
import numpy as np

class SFTGroupAnalyzer:
    """SFT 数据集分组质量分析器"""

    def __init__(self, df):
        self.df = df.copy()

    def add_group_features(self):
        if 'source' in self.df.columns:
            self.df['source_mean_len'] = self.df.groupby('source')['response_len'].transform('mean')
            self.df['source_z_len'] = self.df.groupby('source')['response_len'].transform(
                lambda x: (x - x.mean()) / (x.std() + 1e-8)
            )
            self.df['source_rank'] = self.df.groupby('source')['reward_score'].rank(pct=True)
        return self

    def filter_quality_groups(self, min_size=50, min_avg_reward=0.5):
        before = self.df
        after = self.df.groupby('source', group_keys=False).filter(
            lambda g: len(g) >= min_size and g['reward_score'].mean() >= min_avg_reward
        )
        removed_sources = set(before['source'].unique()) - set(after['source'].unique())
        print(f"过滤前: {len(before)} 条, {before['source'].nunique()} 个来源")
        print(f"过滤后: {len(after)} 条, {after['source'].nunique()} 个来源")
        if removed_sources:
            print(f"移除来源: {removed_sources}")
        self.df = after
        return self

    def stratified_sample(self, group_col='source', frac=0.1, min_samples=5, seed=42):
        """分层抽样"""
        def sample_group(g):
            n = max(int(len(g) * frac), min_samples)
            return g.sample(min(n, len(g)), random_state=seed)

        sampled = self.df.groupby(group_col, group_keys=False).apply(sample_group)
        print(f"\n分层抽样: {len(self.df)} → {len(sampled)} ({len(sampled)/len(self.df)*100:.1f}%)")
        return sampled

    def generate_report(self):
        report = self.df.groupby('source').agg(
            count=('instruction', 'count'),
            avg_instr_len=('instr_len', 'mean'),
            avg_resp_len=('response_len', 'mean'),
            avg_reward=('reward_score', 'mean'),
            median_reward=('reward_score', 'median'),
            reward_std=('reward_score', 'std'),
            high_quality_pct=('reward_score', lambda x: (x >= 0.7).mean()),
            low_quality_pct=('reward_score', lambda x: (x < 0.3).mean()),
        ).round(3)

        report['high_pct'] = (report['high_quality_pct'] * 100).round(1)
        report['low_pct'] = (report['low_quality_pct'] * 100).round(1)

        return report.sort_values('avg_reward', ascending=False)


np.random.seed(42)
n = 1000
sft_raw = pd.DataFrame({
    'source': np.random.choice(['web_crawl', 'textbook', 'wiki', 'forum', 'paper'], n,
                               p=[0.35, 0.20, 0.20, 0.15, 0.10]),
    'instruction': [f'指令_{i}' for i in range(n)],
    'instr_len': np.random.exponential(40, n).astype(int) + 5,
    'response_len': np.random.exponential(250, n).astype(int) + 20,
    'reward_score': np.random.beta(2, 5, n),
})
sft_raw['reward_score'] = sft_raw.groupby('source')['reward_score'].transform(
    lambda x: x + np.where(x < 0.5, -0.1, 0.05) + np.random.randn(len(x)) * 0.08
).clip(0, 1)

analyzer = SFTGroupAnalyzer(sft_raw)
analyzer.add_group_features()

print("=== SFT 来源质量报告 ===")
report = analyzer.generate_report()
print(report)

sampled = analyzer.stratified_sample(frac=0.15, min_samples=10)
print(f"\n抽样后各来源分布:")
print(sampled['source'].value_counts().sort_index())
```
