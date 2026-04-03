---
title: 列选择与索引
description: 单列/多列选择、基于位置的 iloc、基于标签的 loc、条件筛选组合、链式索引最佳实践
---
# 行列选择操作


## 列选择的三种方式

```python
import pandas as pd

df = pd.DataFrame({
    'model': ['GPT-4o', 'Claude', 'Llama'],
    'score': [88.7, 89.2, 84.5],
    'context': [128000, 200000, 131072],
})

single_col = df['model']
print(type(single_col))  # <class 'pandas.core.series.Series'>

multi_col = df[['model', 'score']]
print(type(multi_col))  # <class 'pandas.core.frame.DataFrame'>

single_as_df = df[['model']]  # 注意双层括号！
print(type(single_as_df))  # <class 'pandas.core.frame.DataFrame'>
```

**记忆技巧**：
- **一对方括号 `[]`** → 返回 Series（竖切）
- **两对方括号 `[[]]`** → 返回 DataFrame（保留表格结构）

## 行选择：iloc vs loc

### 核心区别

```
iloc = integer location（按位置）→ 类似数组下标，从 0 开始
loc  = label location（按标签）→ 使用 Index 的实际标签值
[]   = 混合行为（简单场景可用，复杂场景避免）
```

### iloc：基于整数位置

```python
import pandas as pd

df = pd.DataFrame({
    'a': [10, 20, 30, 40, 50],
    'b': ['x', 'y', 'z', 'w', 'v'],
}, index=['r1', 'r2', 'r3', 'r4', 'r5'])

print(df.iloc[0])        # 第一行（返回 Series）
print(df.iloc[0, 0])     # 第1行第1列的值 (10)
print(df.iloc[0, 1])     # 第1行第2列的值 ('x')

print(df.iloc[0:3])       # 前 3 行（r1, r2, r3）
print(df.iloc[:, 0])      # 第 1 列全部行
print(df.iloc[[0, 2, 4]])  # 不连续选取第 1,3,5 行

print(df.iloc[-1])        # 最后一行
print(df.iloc[-3:])        # 最后 3 行
```

### loc：基于标签

```python
import pandas as pd

df = pd.DataFrame({
    'a': [10, 20, 30, 40, 50],
    'b': ['x', 'y', 'z', 'w', 'v'],
}, index=['row_a', 'row_b', 'row_c', 'row_d', 'row_e'])

print(df.loc['row_a'])           # 标签为 'row_a' 的行
print(df.loc[['row_a', 'row_c']])  # 多个标签

print(df.loc['row_a':'row_c'])  # 包含 row_a, row_b, row_c（共 3 行）
print(df.loc['row_a'])         # 只取一行（Series）

print(df.loc['row_a', 'b'])      # row_a 行，b 列的值 ('x')
print(df.loc['row_a':'row_c', 'a'])  # row_a 到 row_c 行，只取 a 列

print(df.loc[df['a'] > 25])  # a > 25 的所有行和列
```

### loc vs iloc 对比表

| 操作 | `iloc` | `loc` | `[]` |
|------|-------|------|-----|
| **单个元素** | `df.iloc[0, 0]` ✅ | `df.loc['r1', 'col']` ✅ | `df['col'][0]` ⚠️ |
| **行切片** | `df.iloc[0:3]` ✅ | `df.loc['r1':'r3']` ✅ | 不支持 |
| **不连续行** | `df.iloc[[0,2,4]]` ✅ | `df.loc[['r1','r3','r5']]` ✅ | 不支持 |
| **布尔筛选** | ❌ 不支持 | `df.loc[df['a']>1]` ✅ | `df[df['a']>1]` ✅ |
| **赋值** | `df.iloc[0,0]=99` ✅ | `df.loc['r1','a']=99` ✅ | `df['a'][0]=99` ⚠️ |

**结论**：
- **读数据**：三种方式都可用，优先用 `loc`（更语义化）
- **写数据（赋值）**：必须用 `loc` 或 `iloc`
- **布尔筛选**：用 `[]` 最简洁，或 `query()`

## 条件组合选择实战

```python
import pandas as pd
import numpy as np

np.random.seed(42)
n = 100_000

df = pd.DataFrame({
    'id': range(n),
    'prompt': [f'问题{i%200}' for i in range(n)],
    'response': [f'回答{i}' for i in range(n)],
    'quality': np.round(np.random.uniform(1, 5, n), 1),
    'tokens': np.random.randint(20, 2000, n),
    'source': np.random.choice(['api', 'web', 'export'], n),
})

high_quality_api = df.loc[
    (df['source'] == 'api') &
    (df['quality'] >= 4.0) &
    (df['tokens'].between(100, 800))
][['id', 'prompt', 'quality', 'tokens']]

print(f"高质量API数据: {len(high_quality_api):,}")

clean = df.loc[
    df['prompt'].notna() &
    ~df['prompt'].str.contains(r'^\s*$', regex=True) &
    ~df['response'].str.contains('ERROR', case=False)
].copy()

result = df.loc[
    df['quality'] >= 3.5
].copy()

result['is_high_quality'] = result['quality'] >= 4.0
result['token_tier'] = pd.cut(
    result['tokens'],
    bins=[0, 100, 300, 500, float('inf')],
    labels=['short', 'medium', 'long', 'very_long']
)

print(result.head())
```

## 链式索引的反模式（SettingWithCopyWarning 避坑）

```python
import pandas as pd

pd.options.mode.copy_on_write = True  # Pandas 3.0 默认开启

df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})

subset = df[df['a'] > 1]
subset['new_col'] = 99

df.loc[df['a'] > 1, 'new_col'] = 99

safe_subset = df[df['a'] > 1].copy()
safe_subset['new_col'] = 99
```

## LLM 场景：按元数据字段筛选知识库文档块

```python
import pandas as pd
import numpy as np

np.random.seed(42)

kb = pd.DataFrame({
    'chunk_id': [f'ch{i:04d}' for i in range(500)],
    'doc_source': np.random.choice(
        ['docs/api.md', 'docs/guide.md', 'blogs/llm-rag.md',
         'papers/attention.pdf', 'wiki/transformer.md'], 500
    ),
    'chunk_index': np.random.randint(0, 50, 500),
    'token_count': np.random.randint(128, 1024, 500),
    'category': np.random.choice(['api', 'tutorial', 'reference', 'theory'], 500),
    'status': np.random.choice(['active', 'archived', 'pending_update'], 500,
                        p=[0.85, 0.10, 0.05]),
    'last_updated': pd.date_range('2025-01-01', periods=500, freq='h'),
})

stale = kb.loc[kb['status'] == 'pending_update']
print(f"待更新: {len(stale)} 个分块")

api_active = kb.loc[
    (kb['doc_source'] == 'docs/api.md') &
    (kb['status'] == 'active')
]
print(f"API 文档活跃分块: {len(api_active)}")

long_chunks = kb.loc[kb['token_count'] > 800]
print(f"超长分块 (>800 tokens): {len(long_chunks)}")
if len(long_chunks) > 0:
    print(long_chunks[['chunk_id', 'doc_source', 'token_count']].head())

active_by_cat = kb.loc[kb['status'] == 'active'].groupby('category').size()
print("\n各类别活跃文档数量:")
print(active_by_cat.sort_values(ascending=False))
```
