---
title: pivot_table() 与 crosstab()：透视表与交叉分析
description: 多维聚合透视、交叉表统计、margins 参数、LLM 场景下的模型×任务交叉分析
---
# 透视表：多维数据的交叉分析

当你的数据有**两个以上的分类维度**需要同时分析时——比如"不同模型在不同任务上的表现"或"不同来源的数据质量分布"——普通的 `groupby` 就不够直观了。`pivot_table()` 和 `crosstab()` 就是为此设计的。

## pivot_table()：灵活的多维聚合

```python
import pandas as pd
import numpy as np

np.random.seed(42)
n = 50_000
df = pd.DataFrame({
    'model': np.random.choice(['GPT-4o', 'Claude', 'Llama'], n),
    'task': np.random.choice(['chat', 'code', 'math', 'reasoning'], n),
    'score': np.round(np.random.uniform(0, 100, n), 1),
})

pt = df.pivot_table(
    index='model',
    columns='task',
    values='score',
    aggfunc='mean',
)

print("模型 × 任务 平均分:")
print(pt.round(1))
```

输出类似：

```
task       chat   code   math  reasoning
model                                  
GPT-4o     49.8   51.2   48.5       50.1
Claude     50.2   52.0   47.8       51.3
Llama      48.5   45.2   46.1       47.9
```

`pivot_table()` 和 `pivot()` 的关键区别在于：**`pivot()` 要求数据不重复，而 `pivot_table()` 可以通过 `aggfunc` 处理重复值**。在实际工作中几乎总是用 `pivot_table()`。

## crosstab()：快速计算频数交叉表

```python
import pandas as pd

df = pd.DataFrame({
    'quality_tier': np.random.choice(['A_优秀', 'B_良好', 'C_一般', 'D_低质'], 10_000),
    'source': np.random.choice(['api', 'web', 'export'], 10_000),
})

ct = pd.crosstab(df['source'], df['quality_tier'],
                 margins=True,
                 margins_name='合计')
print(ct)
```

`crosstab()` 专门用于计算两个分类变量的**频数交叉表**（即"有多少个 X 同时是 Y"）。`margins=True` 自动添加行/列的汇总小计。这在数据质量验收时特别有用——一眼就能看出每个来源的数据在各质量等级中的分布。
