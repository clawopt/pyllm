---
title: Pandas 的核心数据结构概述
description: Series 一维带标签数组与 DataFrame 二维表格的原理、创建方式与 LLM 场景示例
---
# 核心数据结构概述


## 两种核心数据结构

Pandas 的全部能力建立在两个数据结构之上：

```
┌─────────────────────────────────────────────┐
│                  Pandas                      │
│                                             │
│   ┌──────────┐      ┌──────────────────┐    │
│   │  Series  │      │    DataFrame     │    │
│   │  (一维)  │      │     (二维)       │    │
│   └──────────┘      └──────────────────┘    │
│                                             │
│   Index (索引) — 贯穿两者的"标签系统"        │
└─────────────────────────────────────────────┘
```

理解这两个结构，就理解了 Pandas 的一半。另一半是操作它们的方法——但那是后面章节的事。

## Series：带标签的一维数组

### 本质

Series 是一个**一维的、带标签的同构数组**。你可以把它理解为：

> **NumPy 数组 + 标签索引 + 缺失值支持**

```python
import pandas as pd
import numpy as np

s = pd.Series([10, 20, 30, 40, 50])
print(s)
```

注意左边那列 `0, 1, 2, 3, 4`——那就是**索引（Index）**，默认是整数位置编号。

### 带自定义标签

```python
scores = pd.Series(
    [4.5, 3.8, 4.2, 4.0],
    index=['GPT-4', 'Claude', 'Gemini', 'Llama'],
    name='quality_score'
)
print(scores)
```

这个例子展示了 Series 的三个关键属性：
- **values**：实际数据 `[4.5, 3.8, 4.2, 4.0]`
- **index**：标签 `['GPT-4', 'Claude', 'Gemini', 'Llama']`
- **name**：整个 Series 的名称 `'quality_score'`

### 从字典创建

这是大模型场景中最常见的用法之一：

```python
model_params = pd.Series({
    'GPT-4': '1.8T',
    'Claude': 'unknown',
    'Llama-3-70B': '70B',
    'Qwen2.5-72B': '72B'
})
print(model_params['GPT-4'])  # '1.8T'
print(model_params[0])         # '1.8T'（也可以用位置访问）
```

### 大模型场景：Token 级置信度序列

在分析 LLM 输出时，Series 非常适合存储逐 token 的信息：

```python
import pandas as pd

tokens = ['The', 'quick', 'brown', 'fox', 'jumps']
log_probs = [-0.12, -0.05, -0.18, -0.22, -0.08]

confidence = pd.Series(
    log_probs,
    index=tokens,
    name='log_probability'
)

print(confidence.nsmallest(2))

print(f"平均 log prob: {confidence.mean():.3f}")
```

这种模式在**模型输出质量分析**中非常常见——用 Series 存储每个 token 或每个样本的指标值，然后利用 Pandas 内置的统计方法快速洞察分布。

### Series 的关键属性

| 属性 | 说明 | 示例 |
|------|------|------|
| `.values` | 底层数据（NumPy 数组或 Arrow 数组） | `s.values` |
| `.index` | 索引对象 | `s.index` |
| `.dtype` | 数据类型 | `s.dtype` → `float64` |
| `.name` | Series 名称 | `s.name` |
| `.shape` | 形状（长度） | `s.shape` → `(5,)` |
| `.size` | 元素个数 | `s.size` → `5` |

## DataFrame：二维表格

### 本质

DataFrame 是一个**二维的、带行列标签的、可能包含异构数据的表格结构**。

类比：
- **Excel 工作表**：有行号、列名、单元格
- **SQL 表**：有行记录、列字段、数据类型
- **R data.frame**：同样的概念，不同的实现

### 创建方式

#### 方式一：从字典创建（最常用）

```python
import pandas as pd

data = {
    'model': ['GPT-4', 'Claude-3', 'Llama-3', 'Gemini-Pro'],
    'context_window': [128000, 200000, 8000, 32000],
    'price_per_1m_input': [30.0, 15.0, 0.27, 1.25],
    'supports_vision': [True, True, False, True]
}

df = pd.DataFrame(data)
print(df)
```

注意：字典的 key 自动成为**列名**，value 成为各列的数据。

#### 方式二：从列表的列表创建

```python
rows = [
    ['GPT-4', 128000, 30.0, True],
    ['Claude-3', 200000, 15.0, True],
    ['Llama-3', 8000, 0.27, False]
]

df = pd.DataFrame(
    rows,
    columns=['model', 'context_window', 'price_per_1m_input', 'supports_vision']
)
```

#### 方式三：从 NumPy 数组创建

```python
import numpy as np

arr = np.random.randn(1000, 4)  # 1000 行 × 4 列的随机数据
df = pd.DataFrame(
    arr,
    columns=['feature_a', 'feature_b', 'feature_c', 'feature_d']
)
```

这在**特征工程**场景中极常见——从 NumPy 生成的嵌入向量或特征矩阵直接转为 DataFrame。

### RAG 场景示例：文档分块元数据表

DataFrame 在 RAG（检索增强生成）系统中扮演着重要角色——管理文档分块的元数据：

```python
import pandas as pd

chunks_meta = pd.DataFrame({
    'chunk_id': ['doc001_ch01', 'doc001_ch02', 'doc002_ch01', 'doc002_ch02', 'doc003_ch01'],
    'source_doc': ['README.md', 'README.md', 'API.md', 'API.md', 'GUIDE.md'],
    'chunk_index': [0, 1, 0, 1, 0],
    'token_count': [256, 198, 342, 276, 415],
    'category': ['intro', 'usage', 'reference', 'example', 'tutorial'],
    'last_updated': ['2025-12-01', '2025-12-01', '2025-11-15', '2025-11-15', '2025-10-20']
})

print(chunks_meta)

print(chunks_meta.groupby('source_doc').size())

long_chunks = chunks_meta[chunks_meta['token_count'] > 300]
print(long_chunks[['chunk_id', 'token_count']])
```

这就是 DataFrame 的威力——**结构化数据一旦进入 DataFrame，筛选、统计、聚合都变成了一行代码的事**。

### DataFrame 的关键属性

```python
df = pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z'], 'c': [1.1, 2.2, 3.3]})

print(df.shape)        # (3, 3) —— 行数和列数
print(df.columns)      # Index(['a', 'b', 'c']) —— 列名
print(df.index)        # RangeIndex(start=0, stop=3) —— 行索引
print(df.dtypes)       # 各列的数据类型
```

## 数据查看与探索方法

拿到一个 DataFrame 后，第一件事永远是**看看它长什么样**：

```python
import pandas as pd
import numpy as np

np.random.seed(42)
n = 100000
df = pd.DataFrame({
    'epoch': np.random.randint(1, 11, n),
    'step': np.random.randint(1, 5001, n),
    'loss': np.random.uniform(0.1, 2.5, n),
    'lr': np.random.choice([1e-5, 5e-6, 1e-6], n),
    'grad_norm': np.random.uniform(0.5, 50, n),
    'throughput': np.random.randint(500, 2000, n)
})
```

### head() / tail()：快速预览

```python
df.head()    # 默认前 5 行
df.head(3)   # 前 3 行
df.tail()    # 默认后 5 行
```

**大模型场景**：当你加载了一个百万行语料文件后，先用 `head()` 确认格式是否正确，避免后续处理全部白费。

### info()：数据类型与内存概览

```python
df.info()
```

这告诉你：
- 每列有多少非空值（快速发现缺失）
- 各列是什么类型（类型是否合理）
- 总内存占用（判断是否需要优化）

### describe()：统计摘要

```python
print(df.describe())
```

对于**损失曲线分析**来说，`.describe()` 能让你一眼看到 loss 的均值、标准差、最小/最大值——快速判断训练是否正常。

## 数据类型（dtype）体系

Pandas 有丰富的类型体系，理解它对内存优化至关重要。

### 基础类型映射

| Python 类型 | Pandas / NumPy 类型 | 说明 |
|-------------|-------------------|------|
| `int` | `int64`, `int32`, `int16`, `int8` | 整数，不同位宽 |
| `float` | `float64`, `float32` | 浮点数 |
| `bool` | `bool` | 布尔值 |
| `str` | `object`（旧）/ `string`（新） | 字符串 |
| `datetime` | `datetime64[ns]` | 时间戳 |
| `timedelta` | `timedelta64[ns]` | 时间差 |

### Nullable 类型（现代方式）

这是 Pandas 1.0+ 引入的重大改进：

```python
import pandas as pd

s_old = pd.Series([1, 2, None, 4])
print(s_old.dtype)  # float64（因为 None 变成了 NaN）

s_new = pd.Series([1, 2, None, 4], dtype='Int64')
print(s_new.dtype)  # Int64（注意是大写 I！）
print(s_new[2])     # <NA>（不是 NaN）
```

**Nullable 类型对照表**：

| 传统类型 | Nullable 版本 | 缺失值表示 |
|----------|--------------|-----------|
| `int64` | `Int64` | `<NA>` |
| `float64` | `Float64` | `<NA>` |
| `bool` | `boolean` | `<NA>` |
| `object`（字符串） | `string` | `<NA>` |

### Category 类型：大幅降低内存

当一列的**基数（唯一值数量）远小于行数**时，用 `category` 类型可以显著节省内存：

```python
import pandas as pd
import numpy as np

n = 1_000_000
labels = np.random.choice(['pos', 'neg', 'neutral'], n)

df_obj = pd.DataFrame({'label': labels})
print(f"object: {df_obj.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

df_cat = pd.DataFrame({'label': labels.astype('category')})
print(f"category: {df_cat.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
```

实测结果通常能节省 **60%-80%** 内存。在大模型的**情感分类标注数据集**中，标签列用 category 是标准做法。

## Series 与 DataFrame 的关系

最后，用一个图总结二者的关系：

```
DataFrame
├── 由多个 Series 组成（每列是一个 Series）
├── 共享同一个 Index（行标签）
└── 列之间可以有不同 dtype

Series
├── 单列数据 + Index
└── 可以视为 DataFrame 的一"片"
```

```python
df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
col_a = df['a']
print(type(col_a))  # <class 'pandas.core.series.Series'>

s1 = pd.Series([1, 2, 3], name='x')
s2 = pd.Series([4, 5, 6], name='y')
df_from_series = pd.concat([s1, s2], axis=1)
```

**记忆口诀**：**DataFrame 是 Series 的容器，Series 是 DataFrame 的原子**。
