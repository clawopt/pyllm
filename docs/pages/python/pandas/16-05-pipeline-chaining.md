---
title: 链式调用 Pipeline
description: pipe() 方法、自定义管道函数、链式数据处理的最佳实践
---
# 链式调用：把操作串联成流水线

Pandas 的很多方法返回新的 DataFrame 而不是修改原对象（函数式设计）。这意味着你可以用 `.` 把多个操作链式地串起来——让整个数据处理流程变成一条可读的"句子"。

## 基本链式调用

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'text': ['Hello', 'ERROR: timeout', '  extra spaces  ', None, 'OK'],
    'value': [10, 20, 30, 40, 50],
})

result = (
    df.dropna()
    .assign(clean_len=lambda d: d['text'].str.strip().str.len())
    .query('clean_len > 2')
    .sort_values('value', ascending=False)
)

print(result)
```

每个 `.` 的结果都是一个新的 DataFrame，下一个方法在这个新 DataFrame 上操作。**链式调用的核心优势是可读性**：你不需要创建中间变量，整个流程从上到下一目了然。

## pipe()：在链条中插入任意函数

```python
def custom_clean(df):
    df = df.copy()
    df['word_count'] = df['text'].str.split().str.len()
    return df[df['word_count'] > 1]

result = df.pipe(custom_clean).head()
```

当你的操作无法用 Pandas 内置方法表达时（比如需要多步复杂逻辑），`pipe()` 让你在链条中插入一个自定义的 Python 函数。它接收当前 DataFrame 作为参数，返回处理后的 DataFrame 继续传递给链条上的下一个方法。
