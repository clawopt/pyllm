---
title: 迭代与分块处理
description: iterrows/itertuples 的正确用法、chunk 模式、大文件流式处理、避免逐行循环的性能陷阱
---
# 迭代与分块：处理大数据的正确姿势

Pandas 的设计哲学是"向量化优先"——能用内置方法批量处理的就不要用循环。但有些场景下你确实需要逐行或逐块操作。这一节讲清楚什么时候该用迭代、怎么用才不会踩性能陷阱。

## iterrows()：最慢但最灵活

```python
import pandas as pd

df = pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})

for idx, row in df.iterrows():
    print(f"Row {idx}: a={row['a']}, b={row['b']}")
```

`iterrows()` 返回每行的 `(索引, Series)` 元组。它是最灵活的迭代方式——你可以访问每一行的任意列。**但它也是最慢的**：每次迭代都要构造一个 Series 对象（包含完整的 dtype 信息、索引等元数据），对于百万级行的数据集来说开销巨大。

## itertuples()：更快的只读迭代

```python
for row in df.itertuples():
    print(f"a={row.a}, b={row.b}")
```

`itertuples()` 返回的是命名元组（namedtuple），比 Series 轻量得多——速度通常快 10-100 倍。如果你只需要读取值而不需要修改数据，**始终优先使用 `itertuples()` 而非 `iterrows()`**。

## 什么时候必须用循环

只有以下情况才考虑逐行迭代：
1. 需要调用外部 API（比如对每一行文本调 LLM API 做质量判断）
2. 每行逻辑完全不同且无法向量化
3. 处理嵌套结构（如 JSON 字段的复杂解析）

其他所有场景都应该先尝试向量化方案。
