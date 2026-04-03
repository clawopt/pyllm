---
title: 聚合方法 agg() 详解
description: 命名聚合语法、多聚合函数、自定义聚合、agg vs transform 的核心区别
---
# agg()：灵活的聚合控制

`groupby()` 本身只负责"拆分"，真正做计算的是跟在后面的聚合函数——`mean()`、`sum()`、`count()` 等。但当你的需求是"对不同的列用不同的聚合方式"时，就需要 `agg()` 了。

## 命名聚合：现代推荐写法

```python
import pandas as pd
import numpy as np

np.random.seed(42)
n = 100_000
df = pd.DataFrame({
    'model': np.random.choice(['GPT-4o', 'Claude', 'Llama'], n),
    'latency_ms': np.random.randint(100, 2000, n),
    'tokens': np.random.randint(20, 4000, n),
    'quality': np.round(np.random.uniform(1, 5, n), 1),
})

result = df.groupby('model').agg(
    avg_latency=('latency_ms', 'mean'),
    max_latency=('latency_ms', 'max'),
    total_tokens=('tokens', 'sum'),
    avg_quality=('quality', 'mean'),
    sample_count=('quality', 'count'),
)

print(result.round(1))
```

这种**命名聚合（Named Aggregation）**语法是 Pandas 0.25+ 推荐的写法。每个元组的格式是 `(目标列, 聚合函数)`，前面的字符串是新列名。它的好处是结果 DataFrame 的列名清晰可读，不需要再做重命名操作。

## 对同一列使用多个聚合

```python
multi_agg = df.groupby('model').agg(
    latency_stats=('latency_ms', ['mean', 'std', 'min', 'max']),
    quality_stats=('quality', ['mean', 'median', 'count']),
)

print(multi_agg.round(2))
```

当传入列表作为聚合函数时，Pandas 会创建**多层列索引**——外层是原始列名，内层是聚合函数名。这在需要同时看均值、标准差、极值等统计量时特别方便。
