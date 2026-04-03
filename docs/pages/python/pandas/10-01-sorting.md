---
title: 排序：sort_values() 深度解析
description: 单列/多列排序、ascending 参数、na_position 处理、多级排序实战与性能考量
---
# 排序：让数据按你的意愿排列

排序看起来是最简单的操作——谁不会 `sort_values()` 呢？但在实际工作中，排序的正确使用涉及几个容易出错的细节：多列排序时的优先级控制、缺失值的排列位置、大数据集上的性能选择，以及排序和 TopN 选择之间的关系。

## 基础用法与方向控制

```python
import pandas as pd

df = pd.DataFrame({
    'model': ['GPT-4o', 'Claude', 'Llama', 'Gemini'],
    'score': [88.7, 89.2, 84.5, 86.8],
    'price': [2.50, 3.00, 0.27, 1.25],
})

print("升序:")
print(df.sort_values('score').head(3))

print("\n降序:")
print(df.sort_values('score', ascending=False).head(3))
```

`ascending` 参数控制排序方向。默认是 `True`（升序），设为 `False` 就是降序。这个参数在 LLM 场景中最常见的用法就是按质量分数或评测分数降序排列——你总是想看最好的结果排在前面。

## 多列排序：优先级数组

当需要按多个维度排序时（比如先按分数降序排，分数相同的再按价格升序排），`by` 和 `ascending` 都可以接受列表：

```python
result = df.sort_values(
    by=['score', 'price'],
    ascending=[False, True]
)
print(result)
```

关键点在于 `ascending` 的列表和 `by` 的列表**一一对应**——第一个元素控制第一排序列的方向，第二个元素控制第二排序列的方向，以此类推。这在"性价比排序"场景中特别有用：先看分数（越高越好），同分的情况下选更便宜的。

## 缺失值位置控制

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'name': ['Alice', 'Bob', None, 'Charlie', 'Diana'],
    'score': [95, None, 88, 92, None],
})

print("缺失值排在最后 (默认):")
print(df.sort_values('score'))

print("\n缺失值排在最前:")
print(df.sort_values('score', na_position='first'))
```

默认情况下 Pandas 把 NaN 放在排序结果的末尾（`na_position='last'`）。在某些场景下你可能希望缺失值排在最前面——比如在做数据清洗时你想先把所有有问题的行集中到顶部来批量处理。
