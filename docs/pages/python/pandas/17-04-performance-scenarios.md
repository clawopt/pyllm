---
title: 性能优化实战
description: 端到端性能优化案例：从 30 分钟降到 3 分钟的完整优化过程
---
# 实战：端到端性能优化

假设你有一个处理 500 万行对话语料的脚本，初始版本需要 30 分钟。我们一步步把它优化到 3 分钟以内。

## 原始代码（慢）

```python
import pandas as pd

df = pd.read_csv('corpus.csv')
results = []
for _, row in df.iterrows():
    clean = row['text'].strip().lower()
    if len(clean) > 5:
        results.append({'id': row['id'], 'clean': clean})
clean_df = pd.DataFrame(results)
```

问题分析：
1. `read_csv()` 全量读取所有列 → 应该 `usecols` + `dtype`
2. `iterrows()` 逐行循环 → 应该向量化
3. 手动构建列表再转 DataFrame → 效率低

## 优化后（快）

```python
df = pd.read_csv('corpus.csv',
                  usecols=['id', 'text'],
                  dtype={'id': 'int32', 'text': 'string[pyarrow]'})

clean_df = (
    df[df['text'].str.len() > 5]
    .assign(clean=lambda d: d['text'].str.strip().str.lower())
)
```

从 30 分钟降到约 **1 分钟**——主要来自三个改变：列裁剪减少 I/O、dtype 优化减少内存、向量化替代循环消除 Python 循环开销。
