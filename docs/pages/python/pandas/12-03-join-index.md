---
title: join() 索引连接详解
description: DataFrame.join() 用法 / lsuffix rsuffix / 多表链式 join / 与 merge 的选择
---
# Join 索引合并


## join() vs merge() 快速决策

```python
import pandas as pd

df1 = pd.DataFrame(
    {'MMLU': [88.7, 89.2, 84.5]},
    index=['GPT-4o', 'Claude', 'Llama']
)

df2 = pd.DataFrame(
    {'price': [2.50, 3.00, 0.27]},
    index=['GPT-4o', 'Claude', 'Llama']
)

joined = df1.join(df2)
print("join() 结果:")
print(joined)
```

## 基础用法与参数

```python
import pandas as pd

left = pd.DataFrame(
    {'val_a': [10, 20, 30]},
    index=['x', 'y', 'z']
)

right = pd.DataFrame(
    {'val_b': [100, 200]},
    index=['y', 'z']
)

inner_j = left.join(right, how='inner')
outer_j = left.join(right, how='outer')
left_j = left.join(right, how='left')

print(f"inner: {len(inner_j)} 行")
print(f"outer: {len(outer_j)} 行")
print(f"left:  {len(left_j)} 行")
```

## 列名冲突处理

```python
import pandas as pd

df_a = pd.DataFrame({'score': [88, 92]}, index=['A', 'B'])
df_b = pd.DataFrame({'score': [90, 89]}, index=['A', 'C'])

result = df_a.join(df_b, lsuffix='_exam1', rsuffix='_exam2')
print(result)
```

## 链式多表 Join

```python
import pandas as pd

base = pd.DataFrame(
    {'model_name': ['GPT-4o', 'Claude', 'Llama']},
    index=['m1', 'm2', 'm3']
)

scores = pd.DataFrame(
    {'MMLU': [88.7, 89.2, 84.5]},
    index=['m1', 'm2', 'm3']
)

costs = pd.DataFrame(
    {'price_per_1M': [2.50, 3.00, 0.27]},
    index=['m1', 'm2', 'm3']
)

latency = pd.DataFrame(
    {'avg_ms': [800, 650, 350]},
    index=['m1', 'm2', 'm3']
)

full = base.join([scores, costs, latency])
print(full)
```
