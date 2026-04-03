---
title: 并行处理入门
description: multiprocessing/Joblib 并行、分片并行处理、Pandas 并行化的注意事项
---
# 并行处理：用多核加速

如果你的机器有多个 CPU 核心（现在几乎都是），但 Pandas 默认只用一个核心——这意味着你可能有 75% 的算力在闲置。这一节介绍如何让 Pandas 操作利用多核并行。

## 基本思路：数据分片 → 各核处理 → 合并结果

```python
import pandas as pd
import numpy as np
from joblib import Parallel, delayed

n = 1_000_000
df = pd.DataFrame({'text': [f'sample {i}' for i in range(n)]})

def process_chunk(chunk):
    return chunk.assign(
        word_count=chunk['text'].str.split().str.len(),
        char_len=chunk['text'].str.len()
    )

n_cores = 4
chunks = np.array_split(df, n_cores)

results = Parallel(n_jobs=n_cores)(
    delayed(process_chunk)(chunk) for chunk in chunks
)

final = pd.concat(results, ignore_index=True)
print(f"并行处理完成: {len(final):,} 行")
```

核心模式：把 DataFrame 拆成 N 份（N = CPU 核心数），每份交给一个独立进程处理，最后 `concat()` 合并。`joblib` 是 Python 中做这种" embarrassingly parallel"任务的标准库——它自动管理进程池、负载均衡和结果收集。

**注意：只有计算密集型操作才值得并行化**。对于 I/O 密集型操作（如读文件、调 API），瓶颈在磁盘或网络而非 CPU，多核帮助有限。
