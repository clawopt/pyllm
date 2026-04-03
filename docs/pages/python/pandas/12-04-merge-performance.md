---
title: 合并与连接性能优化
description: 大数据集合并策略 / dtype 一致性 / 分类类型加速 / 内存管理
---
# 合并性能优化


## 合并操作的性能影响因素

| 因素 | 影响 | 优化建议 |
|------|------|---------|
| 键列类型不一致 | **严重**（5-50x 慢） | 合并前统一 dtype |
| 键列含 NaN | 中等 | 过滤或填充 |
| 键列是 object（字符串） | 较慢 | 改用 category 或数值编码 |
| 结果集过大 | 内存压力 | 提前筛选必要列 |
| 多对多合并 | 行数爆炸 | 先去重再合并 |

## dtype 一致性优化

```python
import pandas as pd
import numpy as np

n = 100_000

df1 = pd.DataFrame({
    'key': np.random.randint(0, n//10, n),
    'val': np.random.randn(n),
})

df2 = pd.DataFrame({
    'key': np.random.randint(0, n//10, n//10),
    'info': np.random.randn(n//10),
})

df1_bad = df1.copy()
df1_bad['key'] = df1_bad['key'].astype(str)

df1_good = df1.copy()
df2_good = df2.copy()

assert df1_good['key'].dtype == df2_good['key'].dtype
result = pd.merge(df1_good[['key', 'val']], df2_good, on='key', how='inner')
print(f"合并结果: {len(result)} 行")
```

## 分类类型加速字符串键合并

```python
import pandas as pd
import numpy as np

n = 200_000
categories = [f'model_{i}' for i in range(20)]

df_a = pd.DataFrame({
    'model': np.random.choice(categories, n),
    'score': np.random.uniform(0.6, 0.98, n),
})

df_b = pd.DataFrame({
    'model': categories,
    'price': np.random.uniform(0.14, 3.0, 20),
})

df_a['model_cat'] = pd.Categorical(df_a['model'])
df_b['model_cat'] = pd.Categorical(df_b['model'])

merged = pd.merge(
    df_a[['model_cat', 'score']],
    df_b[['model_cat', 'price']],
    on='model_cat'
)
print(f"Category 合并: {len(merged)} 行")
```

## 减少内存使用的合并技巧

```python
import pandas as pd
import numpy as np

def memory_efficient_merge(left, right, on, how='inner', needed_cols=None):
    """内存友好的合并"""
    left_cols = needed_cols[needed_cols.isin(left.columns)].tolist() if needed_cols else left.columns.tolist()
    right_cols = needed_cols[needed_cols.isin(right.columns)].tolist() + [on] if needed_cols else right.columns.tolist()

    left_trimmed = left[left_cols + ([on] if on not in left_cols else [])].copy()
    right_trimmed = right[right_cols].copy()

    for col in left_trimmed.select_dtypes(include=['float64']).columns:
        left_trimmed[col] = pd.to_numeric(left_trimmed[col], downcast='float')
    for col in right_trimmed.select_dtypes(include=['float64']).columns:
        right_trimmed[col] = pd.to_numeric(right_trimmed[col], downcast='float')

    return pd.merge(left_trimmed, right_trimmed, on=on, how=how)


np.random.seed(42)
left_df = pd.DataFrame({
    'id': range(50000),
    'feature_1': np.random.randn(50000),
    'feature_2': np.random.randn(50000),
    'feature_3': np.random.randn(50000),
})

right_df = pd.DataFrame({
    'id': range(50000),
    'label': np.random.randint(0, 10, 50000),
    'meta_1': np.random.randn(50000),
    'meta_2': np.random.randn(50000),
})

result = memory_efficient_merge(
    left_df, right_df,
    on='id',
    needed_cols=['id', 'feature_1', 'label']
)
print(f"高效合并结果: {len(result)} 行 × {len(result.columns)} 列")
print(f"列: {list(result.columns)}")
```
