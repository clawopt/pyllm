---
title: merge() 高级技巧
description: left_on/right_on 不同列名、索引合并、多键合并、合并后清理重复列、性能优化
---
# 高级合并：处理真实世界的复杂情况

理想情况下两个表有同名的连接键。但现实往往不是这样——左表叫 `model_name`，右表叫 `model`；或者你需要按多个列的组合来匹配；又或者连接键在索引里而不是普通列中。这一节讲这些高级场景。

## 列名不同时：left_on + right_on

```python
import pandas as pd

left = pd.DataFrame({
    'model_name': ['GPT-4o', 'Claude', 'Llama'],
    'params_B': [1760, 175, 70],
})

right = pd.DataFrame({
    'model_id': ['GPT-4o', 'Claude', 'Llama', 'Qwen'],
    'MMLU': [88.7, 89.2, 84.5, 83.5],
})

merged = pd.merge(left, right,
                  left_on='model_name',
                  right_on='model_id',
                  how='left')

cleaned = merged.drop(columns=['model_id'])
print(cleaned)
```

`left_on` 和 `right_on` 让你分别指定左右表的连接列——它们不需要同名。合并后通常会有一个多余的重复键列（`model_id`），用 `drop()` 清理掉即可。**这是实际工作中最高频的 merge 用法之一**——因为不同数据源的命名规范几乎总是不一致的。

## 多键合并

```python
import pandas as pd
import numpy as np

conversations = pd.DataFrame({
    'user_id': ['u1', 'u1', 'u2'],
    'session_id': ['s1', 's2', 's1'],
    'prompt': ['hello', 'world', 'hi'],
})

labels = pd.DataFrame({
    'user_id': ['u1', 'u1', 'u2'],
    'session_id': ['s1', 's2', 's1'],
    'quality': [4.5, 3.8, 4.2],
})

labeled = pd.merge(conversations, labels,
                   on=['user_id', 'session_id'])
print(labeled)
```

当单个列无法唯一标识一行时（比如同一个用户有多个会话），就需要用多列组合作为连接键。`on` 参数接受列表形式的多键。
