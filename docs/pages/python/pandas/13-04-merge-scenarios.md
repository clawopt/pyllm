---
title: concat 实战场景
description: 多文件批量加载合并、分块处理结果汇总、LLM 日志数据按日期合并
---
# 实战：用 concat 处理多源数据

`concat()` 在 LLM 数据处理中最典型的应用场景是**批量合并多个数据文件**。比如你每天导出一份对话日志 CSV，月底需要把整月的 30 个文件合并成一张表做分析。

## 批量读取并合并

```python
import pandas as pd
from pathlib import Path

data_dir = Path('./data/daily_logs')
all_dfs = []

for f in sorted(data_dir.glob('*.csv')):
    daily = pd.read_csv(f)
    all_dfs.append(daily)

monthly = pd.concat(all_dfs, ignore_index=True)
print(f"合并 {len(all_dfs)} 个文件，共 {len(monthly):,} 条记录")
```

这个模式的核心是：先读进一个列表里（每个元素是一个 DataFrame），最后一次性 `concat()`。比循环中逐次 concat 高效得多——因为 `concat()` 只需要调用一次，Pandas 可以在内部优化内存分配。

## 分块处理结果的合并

还记得我们在 05-04 节学过的 chunksize 模式吗？处理完所有 chunk 之后通常需要把结果合并回去：

```python
import pandas as pd

chunks = []
for chunk in pd.read_csv('huge.csv', chunksize=200_000):
    clean = chunk[chunk['quality'] >= 4.0].copy()
    chunks.append(clean)

final = pd.concat(chunks, ignore_index=True)
print(f"筛选后总计: {len(final):,} 条")
```

这是 ETL 流水线中的标准模式：读 → 分块处理 → concat 合并 → 写出。整个流程可以处理任意大小的输入文件。
