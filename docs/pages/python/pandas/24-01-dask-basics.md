---
title: Dask 基础：Pandas 的分布式替代
description: Dask DataFrame API、与 Pandas 的兼容性、延迟执行模型、基本操作
---
# Dask：当 Pandas 不够用时

如果你的数据集超过了单机内存（比如 50GB 的语料库在 16GB 内存的机器上），Pandas 就跑不动了。Dask 是解决这个问题的最自然的选择——它的 DataFrame API 几乎和 Pandas 完全一致，但底层支持分块处理和并行计算。

## 核心概念

Dask 的核心思想是**把大 DataFrame 切成多个小 Pandas DataFrame，每个在独立进程中处理**：

```
50 GB CSV 文件
    │
    ▼ dask.read_csv() 自动分块
┌──────────┬──────────┬──────────┬──────────┐
│ Chunk 0  │ Chunk 1  │ Chunk 2  │ Chunk 3  │
│ (12.5GB) │ (12.5GB) │ (12.5GB) │ (12.5GB) │
│ Pandas   │ Pandas   │ Pandas   │ Pandas   │
└────┬─────┴────┬─────┴────┬─────┴────┬─────┘
     │         │         │         │
     └─────────┴─────────┴─────────┘
                  │
            .compute()
                  │
                  ▼
          合并结果 → 返回
```

## 基本用法

```python
import dask.dataframe as dd

ddf = dd.read_csv('huge_corpus_*.csv',
                   dtype={'id': 'int32', 'quality': 'float32',
                          'prompt': 'string', 'response': 'string'})

result = ddf[ddf['quality'] >= 4.0]\
    .groupby('source')['quality']\
    .mean()

print(result.compute())
```

注意三个关键点：
1. `read_csv` 支持通配符 `*`——自动读取所有匹配文件并合并
2. `filter` / `groupby` / `mean()` 都是**惰性的**——只构建计算图不执行
3. `.compute()` 触发实际计算——返回普通 Pandas/Series 对象
