---
title: isin() / between() 等便捷筛选方法
description: isin/between/contains/nunique/unique 的使用技巧、性能对比与最佳实践
---
# 便捷过滤函数


## isin()：集合成员判断

### 基础用法

```python
import pandas as pd

df = pd.DataFrame({
    'model': ['GPT-4o', 'Claude', 'Llama', 'Gemini', 'Qwen', 'DeepSeek'],
    'tier': ['frontier', 'frontier', 'open', 'frontier', 'open', 'frontier'],
    'region': ['US', 'EU', 'CN', 'US', 'CN', 'US'],
})

frontier_models = df[df['tier'] == 'frontier']
frontier_v2 = df[df['tier'].isin(['frontier'])]  # 等价但更通用
not_open = df[~df['tier'].isin(['open'])]

selected = df[df['model'].isin(['GPT-4o', 'Claude', 'Gemini'])]
print(selected[['model', 'tier']])
```

### LLM 场景：多条件集合过滤

```python
import pandas as pd
import numpy as np

np.random.seed(42)
n = 100_000

df = pd.DataFrame({
    'source': np.random.choice(['api', 'web', 'export', 'mobile'], n),
    'language': np.random.choice(['zh', 'en', 'code', 'mixed'], n),
    'category': np.random.choice(['tech', 'science', 'finance', 'medical', 'legal'], n),
    'model': np.random.choice(['GPT-4o', 'Claude', 'Llama', 'Gemini', 'Qwen', 'DeepSeek'], n),
})

source_combos = [
    ['api', 'en'],      # 英文 API 数据
    ['api', 'zh'],      # 中文 API 数据
    ['web', 'code'],     # 网页代码数据
]

m1 = (
    df['source'].isin([c[0] for c in source_combos]) &
    df['language'].isin([c[1] for c in source_combos])
)

composite_key = df['source'] + '_' + df['language']
valid_keys = [f"{s}_{l}" for s, l in source_combos]
m2 = composite_key.isin(valid_keys)

print(f"方法1 结果: {m1.sum():,}")
print(f"方法2 结果: {m2.sum():,}")

exclude_categories = ['legal']
clean = df[~df['category'].isin(exclude_categories)]

def stratified_sample(df, group_col, n_per_group=1000, random_state=42):
    """按类别等量采样"""
    
    sampled_dfs = []
    for name, group in df.groupby(group_col):
        if len(group) >= n_per_group:
            sample = group.sample(n=n_per_group, random_state=random_state)
        else:
            sample = group  # 样本不够就全部保留
        
        sampled_dfs.append(sample)
        print(f"  {name}: {len(group):,} → {len(sample):,}")
    
    return pd.concat(sampledfs, ignore_index=True)


sampled = stratified_sample(df, 'category', n_per_group=5000)
print(f"\n分层采样后: {len(sampled):,}")
print(sampled['category'].value_counts())
```

### isin() 性能优化

```python
import pandas as pd
import numpy as np
import time

n = 5_000_000
np.random.seed(42)

large_df = pd.DataFrame({
    'val': np.random.randint(0, 10000, n),
    'cat': np.random.choice(list('ABCDEFGHIJ'), n),
})

lookup_set = set(range(2000, 3000))

start = time.time()
for _ in range(5):
    r_isin = large_df[large_df['val'].isin(lookup_set)]
t_isin = time.time() - start

idx = pd.Index(large_df['val'])
start = time.time()
for _ in range(5):
    r_idx_isin = large_df[idx.isin(lookup_set)]
t_idx_isin = time.time() - start

lookup_df = pd.DataFrame({'key': list(lookup_set)})
start = time.time()
for _ in range(5):
    r_merge = large_df.merge(lookup_df, left_on='val', right_on='key', how='inner')
t_merge = time.time() - start

print(f"{'方法':<12} {'时间':>8} {'相对速度':>10}")
print(f"{'isin':<12} {t_isin:>7.3f}s {'1.0x':>10}")
print(f"{'Index.isin':<12} {t_idx_isin:>7.3f}s {t_isin/t_idx_isin:>9.1f}x")
print(f"{'merge':<12} {t_merge:>7.3f}s {t_isin/t_merge:>9.1f}x")

```

## between()：范围查询

### 数值范围

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'score': np.round(np.random.uniform(0, 100, 1000), 1),
    'tokens': np.random.randint(10, 5000, 1000),
    'price': np.round(np.random.uniform(0.5, 50, 1000), 2),
})

mid_score = df[df['score'].between(40, 60)]       # 包含两端
expensive = df[df['price'].between(10, 30)]         # 价格区间
reasonable_length = df[df['tokens'].between(100, 1000)]

manual = df[(df['score'] >= 40) & (df['score'] <= 60)]
assert len(mid_score) == len(manual)

left_open = df[df['score'].between(40, 60, inclusive='neither')]  # (40, 60)
right_closed = df[df['score'].between(40, 60, inclusive='right')]  # (40, 60]
```

### 时间范围

```python
import pandas as pd

df = pd.DataFrame({
    'date': pd.date_range('2025-01-01', periods=365, freq='D'),
    'users': np.random.randint(100, 500, 365).astype(int),
})

q1_data = df[df['date'].between('2025-01-01', '2025-03-31')]

q2_start = pd.Timestamp('2025-04-01')
q2_end = pd.Timestamp('2025-06-30')
q2_data = df[df['date'].between(q2_start, q2_end)]

from datetime import timedelta
recent_cutoff = pd.Timestamp.now() - timedelta(days=30)
recent_data = df[df['date'] >= recent_cutoff]
```

### between + 其他条件组合

```python
import pandas as pd
import numpy as np

n = 100_000
df = pd.DataFrame({
    'quality': np.round(np.random.uniform(1, 5, n), 1),
    'tokens': np.random.randint(20, 2000, n),
    'price': np.random.uniform(0.1, 15, n),
})

premium = df[
    df['quality'].between(4.0, 5.0) &
    df['price'].between(1.0, 5.0) &
    df['tokens'].between(100, 800)
]

print(f"Premium 数据: {len(premium):,} ({len(premium)/len(df)*100:.1f}%)")
```

## contains()：文本模式匹配

### 与正则结合

```python
import pandas as pd

df = pd.DataFrame({
    'text': [
        '用户询问了关于 GPT-4o 的价格',
        'ERROR: connection refused to api.openai.com',
        '<div class="error">页面加载失败</div>',
        '正常的问题文本内容',
        'https://example.com/api/v1/chat/completions',
        '',
    ]
})

has_url = df[df['text'].str.contains('http')]

has_domain = df[df['text'].str.contains(r'\.(com|org|io|ai)\b', regex=True)]

no_error = df[~df['text'].str.contains(r'ERROR|error|失败', regex=True, case=False)]

patterns = r'GPT|Claude|Llama|Gemini'
has_model = df[df['text'].str.contains(patterns, regex=True)]
```

## nunique() / unique()：唯一值分析

### 基数分析（决定是否用 category 类型）

```python
import pandas as pd
import numpy as np

n = 100_000
df = pd.DataFrame({
    'user_id': [f'u_{i%50000}' for i in range(n)],
    'prompt_hash': [f'hash_{i%10000}' for i in range(n)],
    'quality_label': np.random.choice(['low', 'medium', 'high'], n),
    'source': np.random.choice(['api', 'web', 'export'], n),
    'full_text': [f'text_{i}' for i in range(n)],
})

for col in df.columns:
    unique_count = df[col].nunique()
    total = len(df)
    ratio = unique_count / total
    
    if ratio < 0.001:
        recommendation = "🔥 极低基数 → 强烈建议用 category"
    elif ratio < 0.05:
        recommendation = "✅ 低基数 → 推荐 category"
    elif ratio < 0.5:
        recommendation = "⚠️ 中基数 → 可用 category"
    else:
        recommendation = "❌ 高基数 → 保持原类型"
    
    print(f"{col:<18s} {unique_count:>8,}/{total:<8,} "
          f"({ratio*100:>6.1f}%)  {recommendation}")
```

### unique() 获取唯一值列表

```python
all_sources = df['source'].unique()
print(all_sources)  # array(['api', 'web', 'export'])

first_appearance = df['quality_label'].unique(return_index=True)
print(first_appearance)

unique_values = df.drop_duplicates(subset=['user_id'])['user_id']
print(f"独立用户数: {len(unique_values)}")
```
