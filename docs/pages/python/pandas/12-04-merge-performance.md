---
title: merge 性能优化
description: 大表合并的性能瓶颈、排序合并 vs 哈希合并、category 类型加速、分块合并策略
---
# merge 性能：当数据量变大时

两个小 DataFrame 的 `merge()` 几乎是瞬时的。但当你的表有几十万甚至上百万行时，merge 的性能就开始变得重要了——一个低效的合并操作可能需要几分钟甚至更久。

## 为什么大表合并慢

`merge()` 的默认实现是**哈希连接（hash join）**：Pandas 先把右表的连接键构建成哈希表，然后逐行扫描左表做查找。这个过程的时间复杂度大约是 O(n + m)，其中 n 和 m 是两表的行数。听起来很高效，但问题在于：

1. **构建哈希表本身需要内存**——右表越大，哈希表占用的内存越多
2. **字符串比较比整数比较慢得多**——如果连接键是字符串（比如 model 名），每次哈希计算和比较都比整数昂贵
3. **结果集可能非常大**——如果是多对多合并，结果行数可能是 n × m 级别

## 实用优化策略

```python
import pandas as pd
import numpy as np

n = 500_000
df_a = pd.DataFrame({
    'key': np.random.choice(range(1000), n),
    'val_a': np.random.randn(n),
})

df_b = pd.DataFrame({
    'key': np.random.choice(range(1000), n),
    'val_b': np.random.randn(n),
})
```

**策略一：把字符串键转成 category 或整数编码**

```python
df_a['key_cat'] = df_a['key'].astype('category')
df_b['key_cat'] = df_b['key'].astype('category')

merged = pd.merge(df_a, df_b, on='key_cat', how='inner')
```

**策略二：先过滤再合并**

如果只需要部分数据，不要先全量合并再筛选：

```python
needed_keys = df_a['key'].unique()[:100]
filtered_b = df_b[df_b['key'].isin(needed_keys)]
merged = pd.merge(df_a, filtered_b, on='key', how='inner')
```

**策略三：确保连接键已排序**

对于已经排好序的连接键，Pandas 可以使用更高效的归并排序算法：

```python
df_a = df_a.sort_values('key').reset_index(drop=True)
df_b = df_b.sort_values('key').reset_index(drop=True)
merged = pd.merge(df_a, df_b, on='key', how='inner')
```
