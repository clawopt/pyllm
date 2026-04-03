---
title: 分布式处理最佳实践
description: 分块策略选择、内存管理、避免常见陷阱、何时该升级到分布式
---
# 分布式最佳实践

## 什么时候该升级

```
数据量决策树:
│
├─ < 5 GB → Pandas（不需要分布式）
├─ 5 ~ 50 GB → Pandas + dtype优化 + chunksize（通常够用）
├─ 50 ~ 500 GB → Dask / Polars（单机多核）
└─ > 500 GB → Spark / Ray 集群 / 云数据仓库
```

## Dask 实用技巧

```python
import dask.dataframe as dd

ddf = dd.read_csv('*.csv', blocksize='256MB')
```

`blocksize` 控制每个分块的大小——256MB 是一个经验值：太小则调度开销大，太大则单块可能 OOM。对于 LLM 语料这种文本密集型数据，128MB~512MB 通常是最优范围。

另一个关键点是**避免 `.compute()` 太频繁**。Dask 的优势在于惰性执行和流水线优化——每次 `compute()` 都会打断这个流程。尽量把多个操作链式组合在一起，最后只调用一次 `compute()`。
