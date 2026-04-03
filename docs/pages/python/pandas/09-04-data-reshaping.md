---
title: 数据重塑：宽长格式转换
description: melt() 逆透视 / pivot() 透视 / stack() 与 unstack() / wide_to_long()，LLM 场景下的数据格式转换实战
---
# 数据重塑：宽格式与长格式的相互转换

在数据处理中你经常遇到两种表格格式。**宽格式（Wide）** 是每个变量一列、每行一个观测——适合人类阅读和 Excel 展示。**长格式（Long）** 是所有值堆在一列里、用另一列标识变量名——适合绘图和做分析。

## 宽转长：melt()

```python
import pandas as pd

wide = pd.DataFrame({
    'model': ['GPT-4o', 'Claude', 'Llama'],
    'MMLU': [88.7, 89.2, 84.5],
    'HumanEval': [92.0, 93.1, 82.4],
    'MATH': [76.6, 71.1, 50.4],
})

long = wide.melt(
    id_vars=['model'],       # 不变的主键列
    var_name='metric',        # 原来的列名变成这个新列的值
    value_name='score'        # 原来的数值变成这个新列
)

print("宽格式:")
print(wide)
print("\n长格式:")
print(long)
```

`melt()` 的作用是把"多列指标"拆成"两列：一列记录指标名称、一列记录指标值"。这在 LLM 评测场景中特别常见——你的原始数据可能是每个模型一行、每个评测指标一列（宽格式），但画图或做对比分析时需要长格式（每个模型×指标组合一行）。

## 长转宽：pivot()

```python
wide_again = long.pivot(
    index='model',
    columns='metric',
    values='score'
).reset_index()
print(wide_again)
```

`pivot()` 是 `melt()` 的逆操作——它把长格式还原回宽格式。注意 `pivot()` 要求 `(index, columns)` 的组合是唯一的，如果有重复值会报错。如果确实有重复值需要聚合，应该用 `pivot_table()` 并指定 `aggfunc`：

```python
summary = long.pivot_table(
    index='model',
    columns='metric',
    values='score',
    aggfunc='mean'
)
```

## stack() 和 unstack()：基于索引的重塑

`melt()/pivot()` 操作的是 DataFrame 的列，而 `stack()/unstack()` 操作的是索引层级：

```python
df = pd.DataFrame({
    'GPT-4o': [88.7, 92.0, 76.6],
    'Claude': [89.2, 93.1, 71.1],
}, index=['MMLU', 'HumanEval', 'MATH'])

stacked = df.stack()
unstacked = stacked.unstack()

print(f"\nStack 后形状: {stacked.shape} (Series)")
print(stacked.head())
```

`stack()` 把列"压入"行索引变成一个多层索引 Series；`unstack()` 反过来把最内层索引展开成列。当你需要在多层索引和扁平结构之间转换时这对方法非常有用。
