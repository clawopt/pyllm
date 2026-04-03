---
title: 分布式处理入门：Dask 与 Pandas 的衔接
description: Dask DataFrame API、从 Pandas 到 Dask 的迁移路径、何时该升级到分布式
---
# 分布式处理：当单机 Pandas 不够用时

Pandas 是单机工具——所有数据必须装进内存。当你的数据超过可用内存（比如 50GB 的语料库在 16GB 内存的机器上），就需要考虑分布式方案了。Dask 是和 Pandas API 最接近的分布式框架。

## 什么时候需要 Dask

```
数据量判断:
│
├─ < 5 GB → Pandas 完全够用，不需要分布式
├─ 5 ~ 100 GB → Dask（单机多核 + 磁盘溢出）
├─ 100 GB ~ 10 TB → Dask 集群 或 Spark
└─ > 10 TB → Spark / Ray / Polars + 云存储
```

## 基本用法

```python
import dask.dataframe as dd

ddf = dd.read_csv('huge_corpus_*.csv',
                   dtype={'id': 'int32', 'quality': 'float32',
                          'prompt': 'string', 'response': 'string'})

result = ddf[ddf['quality'] >= 4.0]\
    .groupby('source')['quality']\
    .mean()\
    .compute()
print(result)
```

注意几个关键点：
1. `dd.read_csv()` 支持通配符 `*`——自动读取匹配的所有文件
2. 大部分操作（filter、groupby）是**惰性的**——只构建计算图不执行
3. `.compute()` 触发实际执行并返回普通 Pandas/Series 对象

**Dask 的 API 设计刻意模仿 Pandas**——大部分代码只需要把 `import pandas as pd` 换成 `import dask.dataframe as dd`，然后在最后加一个 `.compute()`。这大大降低了学习成本。
