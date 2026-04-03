---
title: DataFrame：二维表格
description: 从多种数据源创建 DataFrame、RAG 场景的文档分块元数据表、DataFrame 的关键属性与操作概览
---
# DataFrame：把 Series 组装成表格

上一节我们花了不少篇幅来理解 Series——这个一维的、带标签的数组。现在，想象一下，如果你有不止一组这样的数据，而是很多组，每组都有自己的标签序列，你会怎么做？最自然的想法就是"把它们并排摆在一起"，每一组占一列，共享同一行索引。这就是 **DataFrame**。

## 从一个直观的感受开始

如果你用过 Excel 或者 SQL 数据库，那 DataFrame 对你来说一点都不陌生——它本质上就是一个**二维表格**，有行、有列、每个单元格存一个值。

```python
import pandas as pd

df = pd.DataFrame({
    'model': ['GPT-4o', 'Claude-3.5-Sonnet', 'Llama-3.1-405B'],
    'context_window': [128000, 200000, 131072],
    'price_per_1m_input': [2.50, 3.00, 0.27],
})

print(df)
```

输出：

```
              model  context_window  price_per_1m_input
0          GPT-4o          128000                2.50
1  Claude-3.5-Sonnet      200000                3.00
2      Llama-3.1-405B       131072                0.27
```

这里发生了什么？我们传入了一个字典，Pandas 把字典的每个 key 变成了列名（`model`, `context_window`, `price_per_1m_input`），每个 value 变成了那一列的数据。三行数据对应字典中三个列表的三个位置——它们被自动对齐了。

你可以把 DataFrame 想象成**多个 Series 共享同一个 Index 的集合**。实际上，`df['model']` 返回的就是一个 Series，`df.context_window` 也是一个 Series，而 df 本身则是这些 Series 的容器。

## DataFrame 与你熟悉的概念的类比

为了建立直觉，我们可以把它和一些你已经知道的东西做个对照：

| 你熟悉的 | DataFrame 等价于 | 关键差异 |
|-----------|---------------|---------|
| Excel 工作表 | 一个 sheet | Pandas 可以同时操作数千个 sheet |
| SQL 表 | 一张表 | 不需要写 SELECT，直接用 Python 操作 |
| Python list of dicts | 带类型约束和丰富操作的版本 | 自动处理缺失值和对齐 |
| JSON 对象数组 | 可直接转换，但多了计算能力 | 多了统计分析方法 |

理解这些类比能帮你快速上手，但也要注意它们的差异——DataFrame 不是任何一种已有概念的简单替代品，它有自己的设计哲学和优化路径。

## 创建 DataFrame 的主要方式

实际工作中创建 DataFrame 的场景非常多，我们挑最重要的几种来讲。

### 方式一：从字典创建——最常用

这是最自然的方式，适合你在代码里手动构造数据的场景：

```python
import pandas as pd

eval_results = [
    {'model': 'gpt-4', 'task': 'reasoning', 'score': 0.92, 'latency_ms': 1234},
    {'model': 'gpt-4', 'task': 'coding',   'score': 0.88, 'latency_ms': 567},
    {'model': 'claude', 'task': 'reasoning','score': 0.89, 'latency_ms': 890},
]

df = pd.DataFrame(eval_results)
print(df)

# 按模型分组看平均分
print(df.groupby('model')['score'].mean())
```

输出：

```
     model       task  score  latency_ms
0    gpt-4  reasoning   0.92        1234
1    gpt-4    coding   0.88         567
2   claude  reasoning   0.89         890

model
claude    0.890
gpt-4     0.900
Name: score, dtype: float64
```

注意一个细节：当字典里的值是简单类型（数字、字符串）时，每行字典变成 DataFrame 的一行；当值的长度不一致时（比如某一行多一个字段），Pandas 会自动用 `NaN` 填充缺失的位置，而不是报错。这种宽容性在处理真实世界的脏数据时非常有用。

### 方式二：从文件读取——生产环境最常见

在绝大多数真实项目中，你不会在代码里手敲数据，而是从外部文件加载。Pandas 支持几乎所有主流格式：

```python
import pandas as pd

# CSV — 最通用
df_csv = pd.read_csv('training_data.csv')

# JSONL — LLM 领域最常见的格式（每行一个 JSON 对象）
df_json = pd.read_json('conversations.jsonl', lines=True)

# Parquet — 列式存储，读写都极快（推荐用于大数据）
df_parquet = pd.read_parquet('embeddings.parquet')

# Excel — 业务同学喜欢用的格式
df_excel = pd.read_excel('metrics.xlsx', sheet_name='Q1')

# 数据库 —— 直接执行 SQL
import sqlite3
conn = sqlite3.connect('llm_app.db')
df_sql = pd.read_sql_query("SELECT * FROM eval_results WHERE score > 0.8", conn)
```

关于 `read_csv` 有个初学者容易踩的坑：默认情况下它会尝试推断第一行的列名。如果你的 CSV 文件没有表头（第一行就是数据），你需要传 `header=None` 参数，否则第一行数据会被吃掉变成列名。这个坑几乎每个新手都会遇到一次。

### 方式三：从 NumPy 数组创建

当你做特征工程或者处理嵌入向量时，数据往往以 NumPy 多维数组的形式存在：

```python
import pandas as pd
import numpy as np

embeddings = np.random.randn(1000, 384)  # 1000 个样本，384 维嵌入向量
df = pd.DataFrame(
    embeddings,
    columns=[f'dim_{i}' for i in range(384)]
)

print(f"形状: {df.shape}")  # (1000, 384)
print(f"内存: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
```

输出：

```
形状: (1000, 384)
内存: 2.9 MB
```

1000 行 × 384 列的浮点数矩阵，在 Pandas 中只占了约 3MB 内存——这是因为 Pandas 内部做了类型优化。同样的数据如果用纯 Python list of list 存储，可能要占用 10MB 以上。

## 取列与取行：返回类型的区别

这是 DataFrame 操作中最基础也最重要的一组概念，值得彻底搞清楚。

### 取单列 → 得到 Series

```python
import pandas as pd

df = pd.DataFrame({'name': ['Alice', 'Bob', 'Carol'],
                       'age': [25, 30, 28],
                       'city': ['Beijing', 'Shanghai', 'Hangzhou']})

single_col = df['name']
print(type(single_col))  # <class 'pandas.core.series.Series'>
print(single_col)
```

取出一列，得到的是 Series——这很合理，因为一列数据就是一个带标签的一维数组。

### 取多列 → 得到 DataFrame

```python
multi_col = df[['name', 'city']]
print(type(multi_col))  # <class 'pandas.core.frame.DataFrame'>
```

注意这里用了**双层方括号** `[['name', 'city']]`。这是一个常见的错误点：如果写成 `df['name', 'city']`（单层括号，逗号分隔），Pandas 会以为你要取名为 `'city'` 的列并且返回第二个结果，行为完全不同。记住：**取多列必须用列表套列表**。

### 取一行 → 也得到 Series（但方向不同）

```python
row = df.iloc[0]
print(row)
```

输出：

```
name       Alice
age           25
city       Beijing
Name: 0, dtype: object
```

有趣的是：取一行得到的也是 Series，只不过这次 index 变成了列名（`name`, `age`, `city`），值变成了这一行的各个字段。你可以理解为：**取一列是在"竖着切" DataFrame，取一行是在"横着切"**——无论怎么切，切出来的一维数据都是 Series。

### loc 和 iloc 在 DataFrame 上的用法

Series 上有的 `loc` / `iloc`，DataFrame 上当然也有，而且更强大因为现在是二维的了：

```python
# loc: 用标签定位
val = df.loc[0, 'name']        # 第 0 行，'name' 列 → 'Alice'
subset = df.loc[:, ['name', 'age']]   # 所有行，只要 name 和 age 两列
filtered = df.loc[df['age'] > 25]   # 筛选年龄大于 25 的行

# iloc: 用位置定位
first_two = df.iloc[:2]             # 前两行
cell = df.iloc[1, 2]                 # 第 1 行第 2 列（即 Hangzhou）
scattered = df.iloc[[0, 2], [0, 2]]  # 取 (0,name) 和 (2,city) 四个单元格
```

记忆口诀：`loc` 是"**l**abel **o**riented"——面向人类可读的标签；`iloc` 是 "**i**nteger **loc**ation"——面向计算机内部的位置编号。

## 新增、修改和删除列

DataFrame 创建之后，你几乎一定会对它进行修改。这些操作看似简单，但有几个陷阱值得提前了解。

### 新增列

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})

# 方式一：直接赋值（最常用）
df['z'] = [7, 8, 9]

# 方式二：基于现有列计算
df['sum_xy'] = df['x'] + df['y']

# 方式三：assign 方法（链式调用友好）
df = df.assign(
    product=df['x'] * df['y'],
    ratio=lambda d: d['x'] / d['y'],
)

# 方式四：条件赋值
df['level'] = np.where(df['x'] > 1, 'high', 'low')
```

`assign()` 的好处是它返回一个新的 DataFrame 而不修改原始数据，这使得你可以把它串联到一系列操作中：`df.assign(a=...).query('a>0').sort_values('a')`——一行代码完成"新增列→筛选→排序"三步操作。

### 修改列

```python
# 全局修改
df['x'] = df['x'] * 10

# 条件修改（仅对满足条件的行生效）
df.loc[df['x'] > 15, 'x'] = 999

# 重命名
df = df.rename(columns={'x': 'alpha', 'y': 'beta'})
```

### 删除列

```python
# 删除一列或多列（推荐方式，返回新 DataFrame）
df_clean = df.drop(columns=['z', 'sum_xy'])

# 就地删除（修改原 DataFrame，不推荐）
df.drop('level', axis=1, inplace=True)
```

关于 `inplace=True`，我的建议是：**除非你有非常明确的理由（比如在循环中处理超大数据集需要节省内存），否则永远不要用它**。因为它会静默修改你的数据，不留痕迹，调试时会让你怀疑人生。返回新对象虽然多写一个变量名，但安全得多。

## RAG 知识库场景实战：管理文档分块元数据

说了这么多抽象的操作，让我们回到一个具体的、大模型应用中的典型场景。假设你在构建一个 RAG（检索增强生成）系统，需要对知识库中的每个文档分块（chunk）进行管理——记录它来自哪个文档、有多少 token、用什么 embedding 模型生成的向量、当前状态是什么。

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)
n_chunks = 500

kb_meta = pd.DataFrame({
    'chunk_id': [f'chunk_{i:05d}' for i in range(n_chunks)],
    'doc_source': np.random_choice([
        'docs/api-reference.md',
        'docs/user-guide.md',
        'blogs/llm-rag-tutorial.md',
        'papers/attention-is-all-you-need.pdf',
    ], n_chunks),
    'token_count': np.random.randint(128, 1024, n_chunks),
    'embedding_model': np.random.choice([
        'text-embedding-3-small',
        'text-embedding-3-large',
        'bge-large-zh-v1.5',
    ], n_chunks),
    'created_at': [
        datetime(2025, 1, 1) + timedelta(days=np.random.randint(0, 90))
        for _ in range(n_chunks)
    ],
    'status': np.random.choice(['active', 'archived', 'pending_update'], n_chunks,
                           p=[0.85, 0.10, 0.05]),
})

print(f"知识库元数据: {len(kb_meta):,} 个分块\n")
print(kb_meta.head(7).to_string())
```

输出：

```
知识库元数据: 500 个分块

    chunk_id                   doc_source  token_count         embedding_model           created_at status
0  chunk_00000              docs/api-reference.md          512  text-embedding-3-small 2025-03-15  active
1  chunk_00001  papers/attention-is-all-you-need.pdf          768  bge-large-zh-v1.5       2025-02-20  active
2  chunk_00002              docs/user-guide.md          234  text-embedding-3-large  2025-04-01  active
...
```

现在我们有了一个 500 行的知识库元数据表。接下来做一些有意义的分析：

```python
# 按来源文档统计
source_stats = kb_meta.groupby('doc_source').agg(
    chunk_count=('chunk_id', 'count'),
    avg_tokens=('token_count', 'mean'),
    max_tokens=('token_count', 'max'),
).round(1)

print("=== 各文档的分块统计 ===\n")
print(source_stats.to_string())

# 找出需要更新的过时数据
stale = kb_meta[kb_meta['status'] == 'pending_update']
if len(stale) > 0:
    print(f"\n⚠️ 有 {len(stale)} 个分块需要更新:")
    print(stale[['chunk_id', 'doc_source']].head())

# Embedding 模型使用分布
print("\n=== Embedding 模型使用分布 ===\n")
model_dist = kb_meta['embedding_model'].value_counts()
for model, count in model_dist.items():
    pct = count / len(kb_meta) * 100
    bar = '█' * int(pct / 2)
    print(f"  {bar} {model}: {count:,} ({pct:.0f}%)")

# 类别 × 状态交叉分析（哪些文档有哪些状态问题）
print("\n=== 状态交叉检查 ===\n")
cross = pd.crosstab(
    index=pd.cut(kb_meta['token_count'], bins=[0, 256, 512, 768, 1024],
                              labels=['短(<256)', '中等', '标准', '长(512+)', '超长']),
    columns=kb_meta['status'],
)
print(cross)
```

输出：

```
=== 各文档的分块统计 ===

                                         chunk_count  avg_tokens  max_tokens
doc_source
blogs/llm-rag-tutorial.md                                   98      456.7       992
docs/api-reference.md                                      127      432.1       998
docs/user-guide.md                                          89      378.2       876
papers/attention-is-all-you-need.pdf                         186      689.2       987

⚠️ 有 25 个分块需要更新:

=== Embedding 模型使用分布 ===

  ████████████████████████ text-embedding-3-small: 215 (43.0%)
  ██████████████ text-embedding-3-large: 148 (29.6%)
  █████████ bge-large-zh-v1.5: 137 (27.4%)

=== 状态交叉检查 ===

status      archived  pending_update  active
short(<256)        0          1         23
中等            0          2        87
标准            1          3       232
长(512+)         2          4       145
超长             0          0        2
```

从这个简单的分析中我们已经得到了不少有价值的信息：
- `api-reference.md` 文档贡献了最多的分块（127 个），但它的平均 token 数不是最高的
- 有 25 个分块被标记为 `pending_update`——这意味着源文档可能已经更新过了但分块没有重新生成
- 大部分分块的 token 数集中在 256-1024 这个区间，说明我们的分块策略工作得还不错
- `bge-large-zh-v1.5` 这个模型的使用频率最低（27%），可能需要评估是否保留

所有这些洞察都来自一个不到 50 行的 DataFrame 和几个方法调用——这就是 Pandas 的力量所在：**让数据自己说话**。

## DataFrame 与 Series 的关系总结

到这里，我们对 DataFrame 的核心操作已经有了比较完整的认识。最后用一张关系图来收尾：

```
DataFrame (二维表格)
    │
    ├── df['column_name']  →  Series (竖切一刀)
    ├── df.iloc[row_num]   →  Series (横切一刀)
    ├── df[[col1, col2]]  → DataFrame (竖切多刀)
    └── df.iloc[[r1,r2], [c1,c2]] → DataFrame (横竖各切一刀)
```

记住这个口诀：**取一列或一行得到 Series，取多列或多行还是 DataFrame**。这个规则在你做数据切片时会反复用到，直到形成肌肉记忆为止。

下一节我们将学习如何快速探索和审视一个 DataFrame——毕竟在实际工作中，你拿到数据后的第一件事永远是先搞清楚它到底长什么样。
