---
title: 加载千万级对话数据
description: 大规模 CSV/JSONL 数据的加载策略、分块读取、dtype 优化、初始质量检查
---
# 案例 1：加载与探索千万级对话语料

## 项目背景

你从产品数据库导出了一份 **1000 万条**的用户对话记录 CSV 文件（约 8GB），需要为后续的 SFT 微调做数据准备。第一步是安全高效地把数据加载进来并了解它的基本状况。

## 第一步：不要直接 read_csv

```python
import pandas as pd

df = pd.read_csv('conversations_10M.csv')
```

这行代码在 16GB 内存的机器上会直接 OOM——因为 Pandas 默认用 `int64` 和 `object` 类型，1000 万行原始数据可能膨胀到 15GB+ 内存。正确的方式是：

```python
import pandas as pd

df = pd.read_csv(
    'conversations_10M.csv',
    usecols=['conversation_id', 'user_message', 'assistant_message',
             'quality_score', 'token_count', 'source', 'model'],
    dtype={
        'conversation_id': 'string[pyarrow]',
        'user_message': 'string[pyarrow]',
        'assistant_message': 'string[pyarrow]',
        'quality_score': 'float32',
        'token_count': 'int16',
        'source': 'category',
        'model': 'category',
    },
)

print(f"形状: {df.shape}")
print(f"内存: {df.memory_usage(deep=True).sum() / 1024**2:.0f} MB")
```

通过 `usecols` 只读需要的列（假设原表有 20 列但我们只需要 7 列），通过 `dtype` 指定最优类型。这两个参数的组合通常能把内存占用降低 **70-90%**。

## 第二步：快速概览

```python
print(df.info(verbose=True, memory_usage='deep'))
print(f"\n前 3 行:\n{df.head(3).to_string()}")

na = df.isna().sum()
pct = (na / len(df) * 100).round(2)
has_na = na[na > 0]
if len(has_na) > 0:
    print(f"\n缺失值:\n{pd.concat([has_na, pct], axis=1, keys=['count','pct%'])}")
```

对于 1000 万行数据，`info()` 和 `head()` 几乎是瞬间完成的——它们不需要扫描全部数据。而 `isna().sum()` 需要遍历所有行，但即使对 1000 万行也只需几秒钟。
