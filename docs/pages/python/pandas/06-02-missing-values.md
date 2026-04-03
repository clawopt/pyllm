---
title: 缺失值识别与统计
description: .isnull() / .isna() 的使用、缺失值分布分析、可视化缺失模式、大模型语料中的空值处理策略
---
# 缺失值：沉默的数据质量杀手

在 LLM 数据处理的实际工作中，缺失值是最常见也最容易被忽视的问题。它不会像语法错误那样抛出异常让你立刻注意到，也不会像类型错误那样导致程序崩溃——它只是静静地待在那里，让你的聚合统计产生偏差，让模型学到错误的模式，让评估结果变得不可信。

这一节我们从缺失值的来源说起，讲清楚 Pandas 中几种"缺失"表示的区别，然后给出系统化的检测和处理方法。

## 缺失值从哪里来

在 LLM 语料中，缺失值的来源比你想象的要多：

- **API 超时或失败**：调用模型 API 生成 response 时超时了，该字段被写入空值
- **数据采集中断**：爬虫抓取过程中网络波动，部分字段没抓到
- **JSON 解析异常**：原始 JSON 里某个字段不存在（比如非代码类对话没有 `code` 字段）
- **标注流程遗漏**：标注员跳过了困难的样本，标记为"无法判断"
- **ETL 管道断裂**：中间某一步骤失败但仍然写入了不完整的记录

理解来源很重要，因为不同的来源对应不同的处理策略。API 超时导致的空 response 应该直接丢弃（重新生成就行），而 ETL 管道问题导致的缺失可能需要修复管道后重新跑一遍。

## Pandas 里的"缺失"不止一种

这是最容易混淆的地方。Pandas 中有三种不同的"空"：

```python
import pandas as pd
import numpy as np

s = pd.Series([1, None, np.nan, pd.NA, '', 'None', 'missing'])
print(s)
print(f"\ndtype: {s.dtype}")
```

输出：

```
0         1
1      <NA>
2       NaN
3      <NA>
4           
5      None
6    missing
dtype: object
```

逐个解释：`None` 是 Python 原生的空对象，`np.nan` 是 NumPy 的浮点型 NaN（Not a Number），`pd.NA` 是 Pandas 的统一缺失值标量，空字符串 `''` 和 `'None'`/`'missing'` **不是缺失值而是有效的字符串数据**。

关键区别在于它们对 dtype 的影响不同。当 `None` 或 `pd.NA` 出现在整数列中时，如果使用 Nullable 类型（`Int64`），整列保持为整型；但如果使用传统的 `int64`，整列会静默降级为 `float64`——这就是为什么 04-04 节强烈推荐使用 Nullable 类型。

## 检测缺失值：isna() 和它的伙伴们

```python
import pandas as pd

df = pd.DataFrame({
    'id': [1, 2, 3, 4, 5],
    'prompt': ['什么是AI', None, '解释Python', np.nan, '如何学习LLM'],
    'response': ['AI是...', '...', 'Python是...', None, 'LLM是...'],
    'quality': [4.5, None, 3.8, 4.2, None],
})

print(df.isna())
```

输出：

```
   id  prompt  response  quality
0  False   False     False   False
1  False    True     False    True
2  False   False     False   False
3  False    True     False   False
4  False   False     True    True
```

`isna()` 返回一个和原 DataFrame 同形状的布尔矩阵，每个位置标明是否为缺失值。在实际使用中最常见的操作是对这个布尔矩阵求和来得到每列的缺失数量：

```python
print(df.isna().sum())
```

输出：

```
id          0
prompt      2
response    1
quality     2
dtype: int64
```

一行代码就告诉你每列有多少个空值。更进一步，你可以同时看数量和占比：

```python
missing = df.isna().sum()
pct = (df.isna().mean() * 100).round(2)
print(pd.concat([missing, pct], axis=1, keys=['count', 'pct(%)']))
```

输出：

```
          count  pct(%)
id            0     0.00
prompt        2    40.00
response      1    20.00
quality       2    40.00
```

除了按列统计，你还可以按行来看哪些行是完整的、哪些行有缺失：

```python
complete = df[df.notna().all(axis=1)]
incomplete = df[df.isna().any(axis=1)]

print(f"完整行: {len(complete)} / {len(df)}")
print(f"含缺失行: {len(incomplete)} / {len(df)}")
```

`notna().all(axis=1)` 的含义是：每一列都不是 NA → 这一行是完全的。`isna().any(axis=1)` 的含义是：至少有一列是 NA → 这一行有缺失。

## 一个容易忽略的问题：伪装成正常数据的缺失值

`isna()` 只能检测标准的缺失值（NaN/None/NA）。但在真实数据里，缺失值经常以其他形式出现：

```python
import pandas as pd

dirty = pd.DataFrame({
    'text': ['hello', '', 'world', 'N/A', 'null', '-', '  ', 'normal'],
    'score': ['95', 'bad', '88', '#N/A', '77', '', '82', '90'],
})

print(dirty)
print("\n标准 isna 检测:")
print(dirty.isna().sum())
```

输出显示 isna 认为所有数据都是完整的——但实际上 `text` 列有空字符串 `''`、`'N/A'`、`'null'`、`'-'`、`'  '`（纯空格），`score` 列有 `'bad'` 和 `'#N/A'` 这些非法值。这些全部逃过了 `isna()` 的检测。

解决方案是用 `na_values` 参数在读取时就统一映射，或者事后手动检测：

```python
def detect_hidden_missing(df):
    placeholders = ['', 'N/A', 'NA', 'null', 'none', 'NULL',
                     '-', '#N/A', '#REF!', 'nan', 'undefined']
    results = {}
    
    for col in df.columns:
        if df[col].dtype == 'object':
            mask = df[col].str.strip().isin(placeholders) | (df[col] == '') | (df[col].str.len() == 0)
            count = mask.sum()
            if count > 0:
                results[col] = count
    
    return results

hidden = detect_hidden_missing(dirty)
print(f"伪装的缺失值: {hidden}")
```

这种检测函数建议作为数据加载后的第一步检查——它能发现那些 `isna()` 看不到但同样会破坏后续分析的脏数据。

## 缺失机制判断：MCAR / MAR / MNAR

知道"有哪些缺失"之后，下一步是判断"为什么缺失"。统计学上把缺失机制分为三类：

- **MCAR（Missing Completely At Random）**：完全随机缺失，缺失与否和任何变量无关。比如硬盘突然坏道导致恰好丢失了几条记录。
- **MAR（Missing At Random）**：随机缺失，但缺失概率与其他已观测变量有关。比如新用户的对话更可能缺少 quality 字段（因为还没来得及评分）。
- **MNAR（Missing Not At Random）**：非随机缺失，缺失本身有含义。比如低质量对话的 response 更可能是空的（用户不满意所以没回复）。

判断缺失机制的意义在于它决定了你的处理方式：MCAR 可以放心删除或简单填充；MNAR 则需要特别小心——直接删除会引入偏差，填充也需要考虑缺失本身的含义。

一个简单的经验判断方法是看缺失列和其他列的相关性：

```python
def guess_missing_mechanism(df, target_col):
    numeric = df.select_dtypes(include=['number']).columns
    correlations = {}
    
    for col in numeric:
        if col != target_col:
            valid = df[[target_col, col]].dropna()
            if len(valid) > 10:
                is_missing = df[target_col].isna()
                corr = is_missing.corr(df[col])
                if not np.isnan(corr):
                    correlations[col] = round(corr, 3)
    
    max_corr_col = max(correlations, key=correlations.get) if correlations else None
    max_corr_val = correlations.get(max_corr_col, 0) if max_corr_col else 0
    
    if abs(max_corr_val) > 0.3:
        return f"可能 MNAR — 与 '{max_corr_col}' 相关性={max_corr_val}"
    elif abs(max_corr_val) < 0.1:
        return "可能 MCAR — 与其他变量无明显关联"
    else:
        return f"可能 MAR — 弱相关 (最强: {max_corr_col}={max_corr_val})"

mechanism = guess_missing_mechanism(df, 'response')
print(f"response 缺失机制: {mechanism}")
```

这个函数通过计算"某列是否缺失"这个布尔向量与其他数值列的相关系数来判断缺失是否随机。如果相关性很强（>0.3），说明缺失不是随机的，需要谨慎处理。

到这里我们已经覆盖了缺失值的检测和分析。下一节我们来看另一个数据质量杀手：重复值——它在 LLM 训练数据中的危害经常被严重低估。
