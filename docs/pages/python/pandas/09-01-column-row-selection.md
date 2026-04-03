---
title: 列选择与行索引
description: 单列/多列选择、iloc vs loc 的核心区别与记忆技巧、条件筛选组合、链式索引最佳实践
---
# 行列选择：DataFrame 的基本操作

在 04-02 节我们初步接触了 DataFrame 的行列选择。这一节要更系统地讲清楚 `loc`、`iloc`、`[]` 三种方式的区别和适用场景——这是 Pandas 中最容易混淆也最值得彻底搞清楚的知识点之一。

## 列选择：单列返回 Series，多列返回 DataFrame

```python
import pandas as pd

df = pd.DataFrame({
    'model': ['GPT-4o', 'Claude', 'Llama'],
    'score': [88.7, 89.2, 84.5],
    'context': [128000, 200000, 131072],
})

single = df['model']
multi = df[['model', 'score']]
single_as_df = df[['model']]

print(f"单列类型: {type(single).__name__}")
print(f"多列类型: {type(multi).__name__}")
print(f"单列转DF: {type(single_as_df).__name__}")
```

这个区别在实际中非常重要：Series 和 DataFrame 支持的方法不完全相同。如果你后续需要对结果调用 `.sort_values()` 或 `.plot()`，两者都能工作；但如果你需要调用 `.to_csv()` 或做进一步的列操作，只有 DataFrame 能做到。**所以当你不确定时，用双层括号 `[[]]` 总是更安全的选择**。

## iloc vs loc：位置 vs 标签

这是 Pandas 新手最大的困惑来源。一句话总结：

```
iloc = integer location（按整数位置）→ 类似数组下标，从 0 开始
loc  = label location（按标签值）→ 使用 Index 的实际标签值
```

```python
import pandas as pd

df = pd.DataFrame({
    'a': [10, 20, 30, 40, 50],
    'b': ['x', 'y', "z", 'w', 'v'],
}, index=['r1', 'r2', 'r3', 'r4', 'r5'])

print("iloc[0] → 第一行(位置0):")
print(df.iloc[0])

print("\nloc['r1'] → 标签为 r1 的行:")
print(df.loc['r1'])

print("\niloc[0:3] → 前三行(不包含位置3):")
print(df.iloc[0:3])

print("\nloc['r1':'r3'] → r1 到 r3(包含两端!):")
print(df.loc['r1':'r3'])
```

注意最后一个例子的关键差异：**`iloc` 的切片是左闭右开的（不含终点），而 `loc` 的切片是两端都包含的**。这是因为在 `loc` 的语义中你是在"按标签范围选取"，包含终点更符合直觉；而 `iloc` 继承了 Python 列表切片的传统行为。

## 二维选择：同时指定行和列

两种方式都支持二维索引：

```python
print(df.iloc[0, 0])     # 第1行第1列 (10)
print(df.loc['r1', 'a']) # 标签 r1 的 a 列 (10)

print(df.iloc[:3, [0, 1]])   # 前3行的前两列
print(df.loc[['r1','r3'], ['a','b']])  # 指定行和指定列
```

实际工作中最常见的用法是用 `loc` 做条件筛选 + 指定列：

```python
result = df.loc[df['a'] > 20, ['a', 'b']]
```

这等价于先布尔索引再选列，但写成一行且不会有 SettingWithCopyWarning。
