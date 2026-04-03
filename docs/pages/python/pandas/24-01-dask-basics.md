---
title: Dask 与 Pandas 分布式处理
description: Dask DataFrame / 延迟计算 / 分区操作 / 集群模式 vs 本地模式 / 大数据集最佳实践
---
# Dask 分布式处理入门


## 为什么需要分布式处理

```
数据规模:
  < 100MB    → Pandas 单机（最快）
  100MB-10GB → Pandas + 优化技巧
  10GB-100GB → Dask + 单机多核
  100GB+     → Dask 集群 / Spark / Ray
```

## Dask 基础：DataFrame 延迟执行

```python
import pandas as pd
import numpy as np


try:
    import dask.dataframe as dd
    HAS_DASK = True
except ImportError:
    HAS_DASK = False
    print("提示: pip install dask[dataframe]")

if HAS_DASK:
    np.random.seed(42)
    n = 5_000_000

    ddf = dd.from_pandas(pd.DataFrame({
        'id': range(n),
        'model': np.random.choice(['GPT-4o', 'Claude', 'Llama', 'Qwen'], n),
        'score': np.random.uniform(60, 98, n),
        'latency_ms': np.random.exponential(500, n).astype(int) + 100,
        'cost': np.random.uniform(0.05, 15.0, n),
    }), npartitions=8)

    print(f"Dask DataFrame:")
    print(f"  分区数: {ddf.npartitions}")
    print(f"  延迟执行 (未真正计算)")
    print(f"  调用 .compute() 才会触发实际计算")
else:
    print("Dask 未安装，以下为概念性代码")
```

## 核心操作对比：Pandas vs Dask

```python
if HAS_DASK:
    filtered = ddf[ddf['score'] > 85]
    result_filtered = filtered.compute()  # 触发计算

    grouped = ddf.groupby('model').agg({
        'score': ['mean', 'std', 'count'],
        'latency_ms': ['mean', 'min', 'max'],
    }).compute()

    sorted_df = ddf.nlargest(10, 'score').compute()

    df2 = dd.from_pandas(pd.DataFrame({
        'model': ['GPT-4o']*3 + ['Claude']*2,
        'MMLU': [88.7]*3 + [89.2]*2,
    }))
    merged = ddf.merge(df2, on='model', how='left').compute()
```

## 分区（Partition）策略

```python
if HAS_DASK:
    def explain_partitions(ddf):
        """解释分区策略"""
        meta = ddf.map_partitions(
            lambda df: pd.DataFrame({'rows': len(df), 'cols': len(df.columns)})
        ).compute()

        print("=== 分区详情 ===")
        for i, row in meta.iterrows():
            bar = '█' * min(int(row['rows']/50000), 40)
            print(f"  Partition {i}: {row['rows']:>8,} 行 | {bar} | "
                  f"{row['cols']} 列")

    explain_partitions(ddf)
```

## LLM 场景：大规模日志分析

```python
if HAS_DASK:
    class DistributedLogAnalyzer:
        """分布式日志分析器"""

    @staticmethod
    def analyze_large_logs(log_path, pattern='*.jsonl',
                        date_col='timestamp',
                        partitions=16):
        """分析超大型 API 日志"""

        ddf = dd.read_json(
            log_path, lines=True,
            blocksize=256 * 1024 * 1024,  # 256MB per block
        )

        ddf = ddf.repartition(partitions=partitions)

        hourly = ddf.groupby(date_col).agg({
            'model': 'count',
            'total_tokens': 'sum',
            'error_rate': lambda s: (s == 'error').mean(),
            'avg_latency': 'mean',
        }).compute()

        return hourly.sort_index()


    print("=== 分布式处理流程 ===")
    print("1. ddf.read_json()   — 延迟读取，不一次性加载")
    print("2. .repartition()      — 按需调整分区数")
    print("3. groupby().agg()     — 分布式聚合")
    print("4. .compute()          — 触发实际计算")
    print("\n优势: 处理 TB 级别数据不 OOM")
    print("注意: 小数据集用 Pandas 反而更快")
```
