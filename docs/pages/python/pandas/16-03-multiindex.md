---
title: 多级索引 MultiIndex
description: 创建/操作多层索引、stack/unstack、跨层级聚合、LLM 场景下的多维数据组织
---
# 多层索引：处理高维数据

当数据有两个以上的分类维度时（比如"模型 × 任务 × 来源"），普通的单层索引就不够用了。Pandas 的 MultiIndex 允许你在行和列上同时拥有多个层级的索引。

## 创建多层索引

```python
import pandas as pd
import numpy as np

arrays = [
    ['GPT-4o']*4 + ['Claude']*4,
    ['chat', 'code', 'math', 'reasoning']*2,
]
idx = pd.MultiIndex.from_arrays(arrays, names=['model', 'task'])

df = pd.DataFrame({
    'score': np.random.uniform(80, 95, 8),
    'latency': np.random.randint(200, 1500, 8),
}, index=idx)

print(df)
```

输出是一个以 `(model, task)` 为复合索引的 DataFrame。访问时可以用 `df.loc[('GPT-4o', 'chat')]` 来定位到特定模型和任务的行。

## stack / unstack：层级转换

```python
unstacked = df['score'].unstack(level='task')
print(unstacked)
```

`unstack()` 把最内层的索引级别"展开"成列——从长格式变成宽格式。这在做交叉分析时特别有用：把 (model, task) 的长格式转成 model 为行、task 为列的宽格式表格。
