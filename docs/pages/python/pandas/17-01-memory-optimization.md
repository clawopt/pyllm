---
title: 内存优化实战
description: 内存占用分析、dtype 优化策略、分块处理减少峰值内存、大模型数据集的内存管理方案
---
# 内存优化：让单机跑完更多数据

Pandas 的默认行为是"宁可多占内存也不丢信息"——比如整数默认用 `int64`（8 字节），字符串用 `object`（每个值一个 Python 对象）。对于百万级行的数据来说，这种策略会浪费大量内存。

## 第一步：诊断

```python
import pandas as pd
import numpy as np

n = 2_000_000
df = pd.DataFrame({
    'id': range(n),
    'text': ['sample'] * n,
    'value': np.random.randn(n),
    'cat': np.random.choice(list('ABCDEFGHIJ'), n),
})

mem = df.memory_usage(deep=True)
total = mem.sum() / 1024**2
print(f"总内存: {total:.0f} MB")
print(mem.sort_values(ascending=False).head())
```

先用 `memory_usage(deep=True)` 找出内存大户。通常你会发现 `object` 类型的列占了 80% 以上的内存——这就是优化的首要目标。

## 第二步：优化

```python
df['id'] = df['id'].astype('int32')
df['value'] = pd.to_numeric(df['value'], downcast='float')
df['cat'] = df['cat'].astype('category')
df['text'] = df['text'].astype('string[pyarrow]')

mem_after = df.memory_usage(deep=True).sum() / 1024**2
print(f"优化后: {mem_after:.0f} MB (节省 {(1-mem_after/total)*100:.0f}%)")
```

这套优化流程我们在 04-04 和 07-03 节详细讨论过。核心原则是：**能用小类型就不用大类型，能转 category 就不保留 object**。
