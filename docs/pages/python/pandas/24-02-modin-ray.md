---
title: Modin 与 Ray：其他分布式方案
description: Modin 的 drop-in 替换策略、Ray 分布式框架、Polars 迁移路径
---
# 更多分布式方案

除了 Dask，还有几个值得了解的 Pandas 替代/扩展方案。

## Modin：零修改迁移

Modin 的设计理念是**让你一行代码都不用改就能获得加速**：

```python
import modin.pandas as pd
df = pd.read_csv('huge.csv')
result = df.groupby('model').mean()
```

只需要把 `import pandas as pd` 换成 `import modin.pandas as pd`——所有代码保持不变。Modin 在底层自动使用 Ray 或 Dask 做并行处理。适合那些**已有大量 Pandas 代码想快速并行化**的项目。

## Polars：新一代 DataFrame 库

如果你愿意学习新的 API（而不是兼容 Pandas），Polars 是目前性能最强的选择：

```python
import polars as pl

df = pl.scan_csv('huge.csv')      # 惰性加载
result = df.filter(pl.col('quality') >= 4.0)\
    .group_by('source')\
    .agg(pl.col('quality').mean())\
    .collect()
```

Polars 使用 Rust 编写，利用了 Arrow2 列式内存格式和查询优化器。在大多数基准测试中比 Pandas 快 5-20 倍。**对于新项目强烈推荐优先考虑 Polars**。
