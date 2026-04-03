---
title: 性能对比基准
description: Pandas vs Polars vs Dask 在不同数据量下的性能测试、内存占用、加速比
---
# 性能基准

```python
import pandas as pd
import polars as pl
import numpy as np
import time

n = 5_000_000
data = {'a': np.random.randn(n), 'b': np.random.choice(list('ABCDEFGHIJ'), n)}

pd.DataFrame(data).to_parquet('bench.pq')

start = time.time()
df_pd = pd.read_parquet('bench.pq')
r1 = df_pd[df_pd['a'] > 0].groupby('b')['a'].mean().sort_values(ascending=False)
t_pandas = time.time() - start

start = time.time()
df_pl = pl.scan_parquet('bench.pq')\
    .filter(pl.col('a') > 0)\
    .group_by('b').agg(pl.col('a').mean())\
    .sort('a', descending=True)\
    .collect()
t_polars = time.time() - start

print(f"Pandas: {t_pandas:.2f}s | Polars: {t_polars:.2f}s | 加速: {t_pandas/t_polars:.1f}x")
```
