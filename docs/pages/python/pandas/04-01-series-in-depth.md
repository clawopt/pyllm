---
title: Series：一维带标签数组
description: Series 的创建方式、索引规则、大模型场景下的 token 级置信度序列等实战应用
---
# Series 深度解析


## Series 的本质

Series 是 Pandas 中最基础的数据结构。理解它，后续的 DataFrame 就水到渠成。

```
┌──────────────────────────────────────────────┐
│                   Series                     │
│                                              │
│   索引 (Index)    值 (Values)                │
│   ┌──────┐       ┌──────────┐               │
│   │  0   │  ──→  │  'Alice'  │               │
│   │  1   │  ──→  │  'Bob'    │               │
│   │  2   │  ──→  │  'Carol'  │               │
│   └──────┘       └──────────┘               │
│                                              │
│   + name 属性（可选）                         │
│   + dtype 属性（数据类型）                    │
└──────────────────────────────────────────────┘
```

**一句话定义**：Series = **一维数据** + **标签索引** + **数据类型**。

### 与 Python 内置类型的对比

| 特性 | list | numpy.ndarray | pandas.Series |
|------|------|--------------|----------------|
| 标签索引 | 只有位置 | 只有位置 | **位置 + 自定义标签** |
| 缺失值 | None | NaN（整型变 float） | **原生 NA 支持** |
| 数据类型 | 混合 | 同构 | **同构 + 可扩展** |
| 向量化运算 | 不支持 | ✅ | ✅ |
| 名称属性 | ❌ | ❌ | ✅ `s.name` |
| 描述统计 | ❌ | 部分 | ✅ `.describe()` |

## 创建 Series

### 方式一：从列表/元组创建

```python
import pandas as pd

s = pd.Series([10, 20, 30, 40, 50])
print(s)

scores = pd.Series([4.5, 3.8, 4.2], name='quality_score')
print(scores.name)  # 'quality_score'
```

### 方式二：从字典创建（**大模型场景最常用**）

```python
import pandas as pd

model_scores = pd.Series({
    'GPT-4o': 88.7,
    'Claude-3.5-Sonnet': 89.2,
    'Gemini-1.5-Pro': 86.8,
    'DeepSeek-V3': 85.3,
    'Llama-3.1-405B': 84.5,
}, name='benchmark_score')

print(model_scores)

print(model_scores.index)
```

**为什么字典创建在大模型场景中如此重要？**

因为很多 LLM 相关的数据天然就是"键值对"形式：
- 模型名 → 性能分数
- 类别 → 样本数量
- Token → 对数概率 / 嵌入向量

### 方式三：指定自定义索引

```python
import pandas as pd

tokens = ['The', 'quick', 'brown', 'fox']
log_probs = [-0.05, -0.02, -0.15, -0.25]

s = pd.Series(log_probs, index=tokens, name='log_probability')
print(s)

print(s['fox'])        # -0.25
print(s[['The', 'fox']])  # 同时取多个
```

### 方式四：从标量值广播

```python
import pandas as pd

s = pd.Series(0, index=range(100), name='zeros')
print(f"长度: {len(s)}, 所有值: {s.unique()}")

token_counts = pd.Series(0, index=['<pad>', '<unk>', '<eos>'], name='count')
token_counts['<pad>'] += 15000
token_counts['<unk>'] += 2300
print(token_counts)
```

## 索引：Series 的灵魂

### 两种访问方式

```python
import pandas as pd

s = pd.Series([100, 200, 300, 400], index=['a', 'b', 'c', 'd'])

print(s.iloc[0])    # 100（第一个元素）
print(s.iloc[1:3])  # b=200, c=300

print(s.loc['b'])   # 200
print(s.loc['a':'c'])  # a=100, b=200, c=300（注意：包含两端！）

print(s[0])       # 位置访问（当 Index 是整数时）
print(s['a'])      # 标签访问（当 Index 是字符串时）
```

**最佳实践**：

```python
s.loc['specific_label']
s.iloc[specific_position]

s[s > 200]          # 布尔筛选
s[['a', 'c']]       # 多标签选择
```

### 索引的对齐机制（Pandas 最强大的特性之一）

```python
import pandas as pd

s1 = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
s2 = pd.Series([10, 20, 30], index=['b', 'c', 'd'])

result = s1 + s2
print(result)

result = s1.add(s2, fill_value=0)
print(result)
```

这个特性在**合并不同来源的数据**时极其有用——不需要手动对齐，Pandas 自动完成。

## 大模型场景实战：Token 级分析

### 场景 A：分析模型输出的置信度分布

```python
import pandas as pd
import numpy as np

np.random.seed(42)

sentence = "The quick brown fox jumps over the lazy dog"
tokens = sentence.split()

log_probs = np.random.uniform(-3.0, -0.01, len(tokens))
log_probs[2] = -5.0  # 模拟一个不确定的词 "brown"
log_probs[5] = -4.5  # "jumps" 也不太确定

token_series = pd.Series(
    log_probs,
    index=tokens,
    name='log_probability'
)

print("=== Token 级置信度 ===")
print(token_series.round(3))

print("\n=== 统计摘要 ===")
print(token_series.describe())

print("\n=== 最不确定的 3 个 token ===")
print(token_series.nsmallest(3))

print("\n=== 最确定的 3 个 token ===")
print(token_series.nlargest(3))

threshold = -1.0
uncertain_tokens = token_series[token_series < threshold]
print(f"\n低于阈值 ({threshold}) 的 token: {len(uncertain_tokens)} 个")
for tok, lp in uncertain_tokens.items():
    print(f"  '{tok}': log_prob = {lp:.3f}")
```

输出示例：

```
=== Token 级置信度 ===
The      -0.346
quick    -0.147
brown    -5.000  ← 极不自信
fox      -0.892
jumps    -4.500  ← 很不自信
over     -0.562
the      -0.789
lazy     -0.213
dog      -0.102
Name: log_probability, dtype: float64

=== 统计摘要 ===
count    9.000000
mean    -1.394556
std      1.785123
min     -5.000000
25%      -0.789000
50%      -0.346000
75%      -0.147000
max     -0.102000

=== 最不确定的 3 个 token ===
brown   -5.0
jumps   -4.5
fox     -0.9
```

### 场景 B：构建词表到 ID 的映射

```python
import pandas as pd

vocab_list = ['<pad>', '</s>', '<unk>', 'the', 'in', 'is', 'it', 'to', 'and', 'of']

vocab_series = pd.Series(
    range(len(vocab_list)),
    index=vocab_list,
    name='token_id'
)

print("词表映射:")
print(vocab_series)

text = "the cat is in the bag"
ids = [vocab_series.get(tok, vocab_series['<unk>']) for tok in text.split()]
print(f"\n编码结果: {ids}")

reverse_map = vocab_series.index
decoded = ' '.join(reverse_map[i] for i in ids if i not in [0, 1])
print(f"解码结果: {decoded}")
```

### 场景 C：类别分布统计

```python
import pandas as pd
import numpy as np

n = 10000
labels = np.random.choice(['pos', 'neg', 'neutral'], n, p=[0.35, 0.40, 0.25])

label_series = pd.Series(labels, name='sentiment')

print("=== 分布统计 ===")
print(label_series.value_counts())
print()
print(label_series.value_counts(normalize=True).round(3))

print("\n=== 可视化准备数据 ===")
dist_df = label_series.value_counts().reset_index()
dist_df.columns = ['label', 'count']
dist_df['percentage'] = (dist_df['count'] / len(label_series) * 100).round(1)
print(dist_df)
```

## Series 的关键方法速查

### 统计方法

```python
s = pd.Series(np.random.randn(1000))

print(s.mean())            # 均值
print(s.median())          # 中位数
print(s.std())             # 标准差
print(s.var())             # 方差
print(s.min(), s.max())    # 最小/最大值
print(s.quantile([0.25, 0.5, 0.75]))  # 分位数
print(s.describe())        # 完整统计摘要
```

### 排序与排名

```python
s = pd.Series({'GPT-4o': 88.7, 'Claude': 89.2, 'Llama': 84.5})

print(s.sort_values(ascending=False))

print(s.sort_index())

print(s.rank(ascending=False))  
print(s.rank(pct=True))      
```

### 去重与唯一值

```python
s = pd.Series(['a', 'b', 'a', 'c', 'b', 'a'])

print(s.unique())              # ['a', 'b', 'c']
print(s.nunique())             # 3
print(s.value_counts())        # 各值出现次数
print(s.isin(['a', 'c']))      # 是否在集合中
print(s.duplicated())          # 是否重复
print(s.drop_duplicates())     # 去重
```

### 缺失值处理

```python
s = pd.Series([1, None, 3, None, 5], dtype='Int64')

print(s.isna())                # 哪些是 NA
print(s.notna())               # 哪些不是 NA
print(s.fillna(0))             # 用 0 填充
print(s.dropna())              # 删除 NA
print(s.interpolate())         # 线性插值
```

## Series ↔ 其他类型转换

```python
import pandas as pd
import numpy as np

s = pd.Series([1, 2, 3, 4, 5])

lst = s.tolist()        # [1, 2, 3, 4, 5]

d = s.to_dict()         # {0: 1, 1: 2, ...}

arr = s.to_numpy()       # array([1, 2, 3, 4, 5])

json_str = s.to_json()   # '{"0":1,"1":2,...}'

df = s.to_frame(name='value')  # 单列 DataFrame

s_from_arr = pd.Series(np.array([1.1, 2.2, 3.3]))
s_from_list = pd.Series([{'a': 1}, {'b': 2}])  # object 类型
```
