---
title: DataFrame：二维表格
description: 从多种数据源创建 DataFrame、RAG 场景的文档分块元数据表、DataFrame 的关键属性与操作概览
---
# DataFrame 深度解析


## DataFrame 的本质

如果 Series 是一维数组，那么 DataFrame 就是**二维表格**——你可以把它想象成：

> **多个共享同一 Index 的 Series 组成的集合**

```
DataFrame 结构示意：
┌──────────────────────────────────────────────────────┐
│                    DataFrame                          │
│                                                     │
│           col_a     col_b     col_c    col_d        │
│  index ──┬────────┬─────────┬────────┬────────      │
│   0       │ 'Alice' │   25    │  95.5   │  True      │
│   1       │  'Bob'  │   30    │  87.2   │  False     │
│   2       │ 'Carol'  │   28    │  91.8   │  True      │
│   3       │ 'Dave'   │   35    │  76.4   │  True      │
│           └────────┴─────────┴────────┴────────      │
│                                                     │
│  每列都是一个 Series，共享同一个 Index               │
└──────────────────────────────────────────────────────┘
```

### 与常见概念的类比

| 你熟悉的 | DataFrame 等价于 |
|---------|----------------|
| Excel 工作表 | 一个 sheet（有行号、列名、单元格） |
| SQL 表 | 一张表（有记录、字段、数据类型） |
| R data.frame | 完全相同的概念 |
| Python list of dicts | 带类型约束和丰富操作的版本 |
| JSON 对象数组 | 可直接转换，但多了计算能力 |

## 创建 DataFrame

### 方式一：从字典创建（最常用）

```python
import pandas as pd

df = pd.DataFrame({
    'model': ['GPT-4o', 'Claude-3.5-Sonnet', 'Llama-3.1-405B'],
    'context_window': [128000, 200000, 131072],
    'price_per_1m_input': [2.50, 3.00, 0.27],
    'supports_vision': [True, True, False],
})

print(df)

print(df.dtypes)
```

**规则**：字典的 key → 列名，value → 各列的数据。

### 方式二：从列表的字典创建（每行一个字典）

```python
rows = [
    {'model': 'GPT-4o', 'score': 88.7, 'category': 'frontier'},
    {'model': 'Claude', 'score': 89.2, 'category': 'frontier'},
    {'model': 'Llama', 'score': 84.5, 'category': 'open'},
]

df = pd.DataFrame(rows)
print(df)
```

这种格式在**读取 API 返回的 JSON 数据**时非常常见。

### 方式三：从 NumPy 数组创建

```python
import pandas as pd
import numpy as np

arr = np.random.randn(100, 4)  # 100 行 × 4 列
df = pd.DataFrame(
    arr,
    columns=['feature_1', 'feature_2', 'feature_3', 'feature_4']
)

print(df.shape)  # (100, 4)
print(df.head(3))
```

在**特征工程**中极常见——嵌入向量或特征矩阵从 NumPy 转为 DataFrame 后，就可以利用 Pandas 的标签和统计功能。

### 方式四：从文件读取（生产环境最常用）

```python
import pandas as pd

df_csv = pd.read_csv('data.csv')

df_json = pd.read_json('data.jsonl', lines=True)

df_parquet = pd.read_parquet('data.parquet')

df_excel = pd.read_excel('data.xlsx', sheet_name='Sheet1')

import sqlite3
conn = sqlite3.connect('database.db')
df_sql = pd.read_sql_query("SELECT * FROM users WHERE active=1", conn)
```

## RAG 场景实战：文档分块元数据表

这是 DataFrame 在大模型应用中最典型的使用场景之一：

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

n_chunks = 500
sources = ['docs/api-reference.md', 'docs/user-guide.md', 'blogs/llm-rag.md',
           'papers/attention-is-all-you-need.pdf', 'wiki/transformer.md']

kb_meta = pd.DataFrame({
    'chunk_id': [f'chunk_{i:05d}' for i in range(n_chunks)],
    'doc_source': np.random.choice(sources, n_chunks),
    'chunk_index': np.random.randint(0, 50, n_chunks),
    'token_count': np.random.randint(128, 1024, n_chunks),
    'embedding_model': np.random.choice(
        ['text-embedding-3-small', 'text-embedding-3-large', 'bge-large-zh-v1.5'],
        n_chunks
    ),
    'created_at': [
        datetime(2025, 1, 1) + timedelta(days=np.random.randint(0, 90))
        for _ in range(n_chunks)
    ],
    'status': np.random.choice(['active', 'archived', 'pending_update'],
                                n_chunks, p=[0.85, 0.10, 0.05),
    'category': np.random.choice(['api', 'tutorial', 'reference', 'theory'], n_chunks),
})

print(f"知识库元数据: {len(kb_meta):,} 个分块")
print(f"\n前 5 行:")
print(kb_meta.head().to_string())

print("\n=== 按来源文档统计 ===")
source_stats = kb_meta.groupby('doc_source').agg(
    chunk_count=('chunk_id', 'count'),
    avg_tokens=('token_count', 'mean'),
    max_tokens=('token_count', 'max'),
    models=('embedding_model', lambda x: ', '.join(x.unique())),
).round(1)
print(source_stats)

stale = kb_meta[kb_meta['status'] == 'pending_update']
print(f"\n⚠️ 需要更新: {len(stale)} 个分块")
if len(stale) > 0:
    print(stale[['chunk_id', 'doc_source', 'status']])

print("\n=== Embedding 模型分布 ===")
print(kb_meta['embedding_model'].value_counts())

print("\n=== 类别 × 状态 交叉表 ===)
crosstab = pd.crosstab(kb_meta['category'], kb_meta['status'])
print(crosstab)
```

输出示例：

```
知识库元数据: 500 个分块

前 5 行:
     chunk_id                   doc_source  chunk_index  token_count         embedding_model                 created_at status category
0  chunk_00000              docs/api-reference.md            23          512  text-embedding-3-small 2025-03-15 09:24:33  active       api
1  chunk_00001  papers/attention-is-all-you-need.pdf            11          768  bge-large-zh-v1.5       2025-02-20 14:55:12  active    theory
...

=== 按来源文档统计 ===
                                         chunk_count  avg_tokens  max_tokens                                              models
doc_source
blogs/llm-rag.md                                   98      456.7         992  text-embedding-3-small, text-embedding-3-large
docs/api-reference.md                               127      432.1         998  text-embedding-3-small, bge-large-zh-v1.5
...

⚠️ 需要更新: 25 个分块

=== Embedding 模型分布 ===
text-embedding-3-small    215
text-embedding-3-large    148
bge-large-zh-v1.5         137
```

## DataFrame 关键属性

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'a': [1, 2, 3],
    'b': ['x', 'y', 'z'],
    'c': [1.1, 2.2, 3.3],
    'd': [True, False, True],
})

print(df.shape)        # (3, 4)

print(df.columns)

print(df.index)

print(df.dtypes)

print(df.ndim)         # 2（始终是 2）

print(df.size)         # 12 (3行 × 4列)

print(df.T)
```

## 列操作基础

### 选择单列 → 返回 Series

```python
s = df['a']
print(type(s))  # <class 'pandas.core.series.Series'>
```

### 选择多列 → 返回 DataFrame

```python
sub = df[['a', 'c']]
print(type(sub))  # <class 'pandas.core.frame.DataFrame'>
print(sub.columns)  # Index(['a', 'c'])
```

### 新增列

```python
df['e'] = [10, 20, 30]

df['f'] = df['a'] + df['c']

df = df.assign(g=lambda x: x['a'] * 2, h=lambda x: x['c'] ** 2)

import numpy as np
df['label'] = np.where(df['a'] > 1, 'high', 'low')
```

### 删除列

```python
df_drop = df.drop(columns=['e', 'f'])

df.drop('e', axis=1, inplace=True)  # 直接修改原 df（不推荐）
```

### 重命名列

```python
df_renamed = df.rename(columns={
    'a': 'alpha',
    'b': 'beta',
    'c': 'gamma',
})
```

## 行操作基础

### 使用 loc（按标签）

```python
row = df.loc[0]

subset = df.loc[0:2]

val = df.loc[2, 'c']  # 第 2 行，'c' 列

filtered = df.loc[df['a'] > 1]
```

### 使用 iloc（按位置）

```python
first_three = df.iloc[:3]

val = df.iloc[0, 1]

scattered = df.iloc[[0, 2], [0, 2]]
```

### 布尔索引（**最重要的行选择方式**）

```python
import pandas as pd
import numpy as np

n = 1000
df = pd.DataFrame({
    'model': np.random.choice(['A', 'B', 'C'], n),
    'score': np.random.uniform(0, 100, n),
    'tokens': np.random.randint(50, 2000, n),
    'active': np.random.choice([True, False], n),
})

high_score = df[df['score'] > 80]

good_and_active = df[(df['score'] > 80) & (df['active'])]

long_or_high = df[(df['tokens'] > 1500) | (df['score'] > 90)]

not_active = df[~df['active']]

ab_models = df[df['model'].isin(['A', 'B'])]

medium_range = df[df['score'].between(40, 60)]

complex_filter = df[
    (df['model'].isin(['A', 'B'])) &
    (df['score'] > df['score'].median()) &
    (df['tokens'].between(100, 1000))
]
print(f"复杂筛选结果: {len(complex_filter)} 条")
```

### query 方法（更简洁的表达式）

```python
result = df.query("score > 80 and active == True and model in ['A', 'B']")

threshold = 75
result = df.query("score > @threshold")

result = df.query("model not in ['C'] and tokens between 100 and 1000")
```

## DataFrame 与 Series 的关系

```python
import pandas as pd

df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})

col_x = df['x']
print(type(col_x))  # Series

sub = df[['x', 'y']]
print(type(sub))    # DataFrame

row = df.iloc[0]
print(type(row))    # Series（列名变成了 index）

s1 = pd.Series([1, 2, 3], name='alpha')
s2 = pd.Series([4, 5, 6], name='beta')
from_series = pd.concat([s1, s2], axis=1)
print(from_series)
```

**记忆口诀**：
- **取一列** → Series（竖着切一刀）
- **取一行** → Series（横着切一刀）
- **取多列/多行** → DataFrame
