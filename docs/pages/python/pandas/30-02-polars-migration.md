---
title: Polars 迁移指南
description: API 对比表、常见代码迁移模式、性能基准测试、兼容性注意事项
---
# Polars 迁移

```python
import polars as pl

lf = pl.scan_csv('large_data.csv')

result = (
    lf.filter(pl.col('quality') >= 4.0)
    .group_by('source')
    .agg([
        pl.count().alias('count'),
        pl.col('quality').mean().alias('avg_quality'),
        pl.col('latency').median().alias('med_latency'),
    ])
    .sort('count', descending=True)
    .collect()
)

print(result)
```

**Pandas → Polars 常用对照：**
- `pd.read_csv()` → `pl.scan_csv()` 或 `pl.read_csv()`
- `df[df['x'] > 0]` → `df.filter(pl.col('x') > 0)`
- `df.groupby('x').mean()` → `df.group_by('x').agg(pl.col('*').mean())`
- `df.sort_values('x')` → `df.sort('x')`
- `.head(10)` → `.head(10)` （相同）
