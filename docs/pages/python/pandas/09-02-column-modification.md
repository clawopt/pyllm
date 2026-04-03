---
title: 列的新增、修改与删除
description: 直接赋值 / assign() 链式操作 / insert() 指定位置 / drop() 删除 / 条件列生成与映射替换
---
# 列的增删改：DataFrame 的形态变换

筛选是从 DataFrame 中"取"数据，而这一节要讲的是如何"改" DataFrame 的结构——新增一列、修改现有列、删除不需要的列。这些操作看似简单，但不同的写法在可读性、性能和副作用上有显著差异。

## 新增列的四种方式

```python
import pandas as pd

df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})

df['z'] = [7, 8, 9]
df['sum'] = df['x'] + df['y']
df['is_large'] = df['x'] > 1

print(df)
```

直接用 `df['new_col'] = ...` 是最常用也最直观的方式。它支持各种右值：Python 列表、NumPy 数组、另一个 Series、标量值，或者基于现有列的计算表达式。

当你需要在一个表达式中新增多个列并且希望链式调用时，`assign()` 更合适：

```python
result = (
    df.assign(
        z=lambda d: d['x'] + d['y'],
        product=lambda d: d['x'] * d['y'],
        ratio=lambda d: d['x'] / d['y'],
    )
)
print(result)
```

`assign()` 的两个关键特性：**不修改原 DataFrame（返回新对象）** 和 **支持链式调用**。而且它可以在同一个 `assign` 中引用前面刚创建的列——Pandas 会按顺序依次执行每个赋值。

如果你需要把新列插入到特定位置（而不是默认追加到末尾），用 `insert()`：

```python
df.insert(1, 'mid_col', ['a', 'b', 'c'])
print(df)
```

## 修改列的三种模式

修改已有列和新增列语法完全一样——区别只在于列名是否已存在：

```python
import numpy as np

df = pd.DataFrame({
    'score': [85, 92, 78],
    'grade': ['B', 'A', 'C'],
})

df['score_normalized'] = df['score'] / 100

df.loc[df['score'] >= 90, 'grade'] = 'A+'

df['grade_encoded'] = df['grade'].map({'A+': 4, 'A': 3, 'B': 2, 'C': 1})
```

三种模式对应三种常见需求：
- **计算新特征**：基于现有列派生出新列（如归一化）
- **条件更新**：满足条件的行修改为特定值（用 `loc`）
- **映射转换**：离散值的编码/解码（用 `map()`）

## 删除列

```python
df2 = df.drop(columns=['score_normalized'])
df3 = df.drop(['grade'], axis=1)
```

两种写法等价，但 `drop(columns=[...])` 更清晰。注意 `drop()` 默认返回新 DataFrame 而不修改原对象——这是 Pandas 的通用设计哲学（函数式风格）。如果你确实需要原地删除，可以用 `inplace=True`，但**不推荐这样做**（原因和前面章节提到的一样：`inplace` 容易导致链式赋值问题且不利于调试）。
