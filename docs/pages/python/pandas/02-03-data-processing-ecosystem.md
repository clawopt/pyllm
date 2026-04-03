---
title: Pandas 与 Python 数据处理生态链
description: NumPy → Pandas → 可视化/ML 框架的完整链路，以及与 Polars、Dask 等现代工具的关系
---
# 数据处理生态系统


## 生态全景图

Pandas 从不孤立工作。它是 Python 数据科学生态的**枢纽节点**——上游接收原始数据，下游对接各种分析工具。

```
                    ┌─────────────────┐
                    │   数据源层       │
                    │                 │
                    │ CSV · JSON      │
                    │ SQL · Parquet   │
                    │ API · HTML      │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │    Pandas        │  ← 核心工作台
                    │                 │
                    │ 清洗·转换·聚合   │
                    │ 筛选·合并·变形   │
                    └──┬──────────┬───┘
                       │          │
           ┌───────────▼──┐  ┌────▼──────────┐
           │  可视化层     │  │  ML/LLM 层     │
           │              │  │               │
           │ Matplotlib   │  │ scikit-learn  │
           │ Seaborn      │  │ PyTorch       │
           │ Plotly       │  │ HuggingFace   │
           │ (后续第5章)   │  │ JAX           │
           └──────────────┘  └───────────────┘

              同级/替代工具（本章重点）
           ┌──────────┐  ┌──────────┐  ┌──────────┐
           │  Polars  │  │  Dask    │  │  Vaex    │
           │ (并行)   │  │ (分布式)  │  │ (懒加载)  │
           └──────────┘  └──────────┘  └──────────┘
```

## NumPy：Pandas 的地基

### 关系：Pandas 建在 NumPy 之上

这是理解 Pandas 性能特征的关键：

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({'a': [1, 2, 3], 'b': [4.0, 5.0, 6.0]})

print(type(df['a'].values))

print(df['a'].values)

result = df['a'] + df['b']  # 底层是 np.add()
print(result.values)
```

### 大模型场景中的 NumPy ↔ Pandas 交互

在 LLM 开发中，NumPy 和 Pandas 经常需要互转：

```python
import numpy as np
import pandas as pd

embeddings = np.random.randn(1000, 768).astype(np.float32)
metadata = pd.DataFrame({
    'doc_id': [f'doc_{i}' for i in range(1000)],
    'category': np.random.choice(['tech', 'finance', 'medical'], 1000),
    'score': np.random.uniform(0, 1, 1000),
})

metadata['embedding'] = list(embeddings)

print(metadata[['doc_id', 'category', 'score']].head())

feature_df = pd.DataFrame({
    'prompt_length': np.random.randint(10, 500, 1000),
    'response_length': np.random.randint(20, 2000, 1000),
    'turn_count': np.random.randint(1, 10, 1000),
    'has_code': np.random.choice([0, 1], 1000),
    'sentiment_score': np.random.uniform(-1, 1, 1000),
})

X = feature_df.values  # 转为 NumPy 数组
y = np.random.choice([0, 1], 1000)  # 标签

print(f"特征矩阵形状: {X.shape}")
print(f"特征矩阵 dtype: {X.dtype}")

model_outputs = {
    'GPT-4o': np.random.randn(1000),
    'Claude': np.random.randn(1000),
    'Llama': np.random.randn(1000),
}
results = pd.DataFrame(model_outputs)
results['best_model'] = results.idxmax(axis=1)
results['confidence_gap'] = results.max(axis=1) - results.min(axis=1)

print(results.head())
```

### 性能对比：纯 Python vs NumPy vs Pandas

```python
import time

data = list(range(1_000_000))

start = time.time()
sum_python = sum(x * x + x / 2 for x in data)
t_python = time.time() - start

arr = np.array(data)
start = time.time()
sum_numpy = (arr * arr + arr / 2).sum()
t_numpy = time.time() - start

s = pd.Series(data)
start = time.time()
sum_pandas = (s * s + s / 2).sum()
t_pandas = time.time() - start

print(f"Python 循环: {t_python:.4f}s")
print(f"NumPy 向量化: {t_numpy:.4f}s ({t_python/t_numpy:.0f}x 加速)")
print(f"Pandas Series: {t_pandas:.4f}s ({t_python/t_pandas:.0f}x 加速)")
```

**结论**：Pandas 的性能几乎等同于 NumPy，因为底层调用的是同一套 C/Fortran 优化代码。差距来自 Pandas 额外的索引对齐开销。

## Polars：Pandas 的现代挑战者

### 为什么会出现 Polars

Pandas 的核心架构有一个根本性限制：**单线程执行**。即使你的 CPU 有 16 个核，Pandas 默认只使用其中 1 个。

Polars 由 Ritchie Vink 于 2019 年创建，设计目标就是解决这个限制：

| 特性 | Pandas | Polars |
|------|--------|--------|
| 执行引擎 | 单线程（依赖 NumPy） | 多线程（Rust 实现） |
| 内存格式 | 行优先（NumPy ndarray） | 列式（Apache Arrow） |
| 惰性求值 | 不支持 | 支持（`lazy` 模式） |
| 语言实现 | Python + C 扩展 | Rust（绑定到 Python） |
| API 设计风格 | 可变（in-place） | 不可变（immutable） |
| 成熟度 | 极高（15年+） | 快速成长中 |

### 性能对比实测

```python
import pandas as pd
import polars as pl
import numpy as np
import time

n = 10_000_000

np.random.seed(42)
data = {
    'id': range(n),
    'category': np.random.choice(['A', 'B', 'C', 'D', 'E'], n),
    'value_a': np.random.randn(n),
    'value_b': np.random.uniform(0, 100, n),
    'timestamp': pd.date_range('2025-01-01', periods=n, freq='s'),
}

df_pd = pd.DataFrame(data)
df_pl = pl.DataFrame(data)

def test_groupby():
    start = time.time()
    result_pd = df_pd.groupby('category').agg(
        mean_a=('value_a', 'mean'),
        sum_b=('value_b', 'sum'),
        count=('id', 'count')
    )
    t_pd = time.time() - start
    
    start = time.time()
    result_pl = df_pl.group_by('category').agg(
        pl.col('value_a').mean().alias('mean_a'),
        pl.col('value_b').sum().alias('sum_b'),
        pl.col('id').count().alias('count')
    )
    t_pl = time.time() - start
    
    return t_pd, t_pl

def test_filter():
    start = time.time()
    result_pd = df_pd[
        (df_pd['value_a'] > 0) &
        (df_pd['value_b'].between(20, 80)) &
        (~df_pd['category'].isin(['D']))
    ][['id', 'category', 'value_a']].head(10000)
    t_pd = time.time() - start
    
    start = time.time()
    result_pl = df_pl.filter(
        (pl.col('value_a') > 0) &
        (pl.col('value_b').is_between(20, 80)) &
        (~pl.col('category').is_in(['D']))
    ).select(['id', 'category', 'value_a']).head(10000)
    t_pl = time.time() - start
    
    return t_pd, t_pl

t_g_pd, t_g_pl = test_groupby()
t_f_pd, t_f_pl = test_filter()

print("=" * 50)
print("性能对比（1000 万行数据）")
print("=" * 50)
print(f"{'操作':<15} {'Pandas':>10} {'Polars':>10} {'加速比':>10}")
print("-" * 50)
print(f"{'GroupBy 聚合':<15} {t_g_pd:>8.3f}s {t_g_pl:>8.3f}s {t_g_pd/t_g_pl:>9.1f}x")
print(f"{'复杂筛选':<15} {t_f_pd:>8.3f}s {t_f_pl:>8.3f}s {t_f_pd/t_f_pl:>9.1f}x")
```

典型输出：

```
==================================================
性能对比（1000 万行数据）
==================================================
操作                Pandas     Polars      加速比
--------------------------------------------------
GroupBy 聚合         1.234s     0.089s       13.9x
复杂筛选             0.567s     0.042s       13.5x
```

### 何时选择 Polars

```python

```

### 互操作性

好消息是两者可以无缝协作：

```python
import pandas as pd
import polars as pl

df_pd = pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})
df_pl = pl.from_pandas(df_pd)
print(df_pl)

df_back = df_pl.to_pandas()
print(type(df_back))  # <class 'pandas.core.frame.DataFrame'>
```

## Dask：分布式 Pandas

### 解决什么问题

当数据量**超过单机内存**时，Pandas 和 Polars 都无能为力了。这时需要 **Dask**——它提供了与 Pandas 兼容的 API，但能将计算分布到多台机器上。

```python
import dask.dataframe as dd

ddf = dd.read_csv('huge_data_100GB.csv')

result = ddf[
    (ddf['value'] > 0) &
    (ddf['category'].isin(['A', 'B']))
].groupby('category')['value'].mean()

final_result = result.compute()  # 返回 Pandas DataFrame/Series
print(final_result)
```

### 三者定位对比

| 维度 | Pandas | Polars | Dask |
|------|--------|--------|------|
| **数据规模** | 单机内存内 (< ~50GB) | 单机内存内 (< ~50GB) | 分布式（TB-PB 级） |
| **核数利用** | 单核 | 多核 | 多机器 × 多核 |
| **延迟特性** | 即时执行 | 即时 / 惰性可选 | 惰性执行 |
| **学习成本** | 低（最成熟） | 中等 | 低（API 类似 Pandas） |
| **适用场景** | 日常数据分析 | 高性能 ETL | 大规模数据处理 |

## 完整生态链路示例

下面用一个完整的例子展示从数据采集到模型训练的全链路中各工具的角色：

```python
"""
大模型训练数据准备完整流水线
展示了 Pandas 在生态链中的枢纽作用
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import json

print("Stage 1: 数据生成")
np.random.seed(42)
n = 100_000

raw_data = pd.DataFrame({
    'prompt': [f'请解释以下概念：{np.random.choice(["机器学习", "深度学习", "神经网络", "Transformer", "注意力机制"])}' 
               for _ in range(n)],
    'response': [f'这是一个关于{topic}的详细解释...' 
                 for topic in np.random.choice(["ML", "DL", "NN", "Attention"], n)],
    'source': np.random.choice(['wiki', 'forum', 'book', 'paper'], n),
    'quality': np.random.choice([1, 2, 3, 4, 5], n, p=[0.05, 0.1, 0.2, 0.35, 0.3]),
    'domain': np.random.choice(['CS', 'Math', 'Physics', 'Bio'], n),
})
print(f"  原始数据: {len(raw_data):,} 条")

print("\nStage 2: Pandas 数据清洗")

cleaned = raw_data[
    (raw_data['quality'] >= 3) &
    (raw_data['prompt'].str.len().between(10, 500)) &
    (raw_data['response'].str.len().between(20, 2000))
].copy()

cleaned = cleaned.drop_duplicates(subset=['prompt', 'response'])
cleaned = cleaned.reset_index(drop=True)

cleaned['combined_text'] = cleaned['prompt'] + '\n' + cleaned['response']
print(f"  清洗后: {len(cleaned):,} 条")

print("\nStage 3: Tokenizer 统计")

tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')

def tokenize_row(row):
    tokens = tokenizer(
        row['combined_text'],
        truncation=False,
        return_tensors=None
    )
    return len(tokens['input_ids'])

cleaned['token_count'] = cleaned.apply(tokenize_row, axis=1)

token_stats = cleaned['token_count'].describe()
print(f"  Token 统计:")
print(f"    均值: {token_stats['mean']:.0f}")
print(f"    中位数: {token_stats['50%']:.0f}")
print(f"    最大值: {token_stats['max']}")
print(f"    P95: {cleaned['token_count'].quantile(0.95):.0f}")

cleaned = cleaned[cleaned['token_count'] <= 4096]
print(f"  长度过滤后: {len(cleaned):,} 条")

print("\nStage 4: 数据集划分")

train_df, temp_df = train_test_split(
    cleaned, test_size=0.2, random_state=42,
    stratify=cleaned['domain']
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, random_state=42,
    stratify=temp_df['domain']
)

print(f"  训练集: {len(train_df):,}")
print(f"  验证集: {len(val_df):,}")
print(f"  测试集: {len(test_df):,}")

print("\nStage 5: 导出为训练格式")

def to_messages(row):
    return [
        {'role': 'user', 'content': row['prompt']},
        {'role': 'assistant', 'content': row['response']}
    ]

train_df['messages'] = train_df.apply(to_messages, axis=1)

output_path = 'output/sft_train.jsonl'
train_df[['messages', 'domain', 'source', 'quality', 'token_count']].to_json(
    output_path,
    orient='records',
    lines=True,
    force_ascii=False
)
print(f"  已导出: {output_path}")

report = {
    'total_raw': len(raw_data),
    'after_cleaning': len(cleaned),
    'train_size': len(train_df),
    'val_size': len(val_df),
    'test_size': len(test_df),
    'avg_tokens': float(cleaned['token_count'].mean()),
    'domain_distribution': cleaned['domain'].value_counts().to_dict(),
    'source_distribution': cleaned['source'].value_counts().to_dict(),
}
with open('output/data_report.json', 'w') as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print(f"  报告已保存: output/data_report.json")
```

这个例子展示了 Pandas 在整个链路中的位置：

```
数据源 → [Pandas 清洗] → [Pandas 统计] → [HF Tokenizer] → [sklearn 划分] → [导出 JSONL]
         ↑ 核心枢纽                                          ↓ 训练输入
```

## 生态选择决策树

面对一个数据处理任务时，可以这样选择工具：

```
你需要处理数据？
│
├─ 数据能装入单机内存？
│  ├─ 是 → 需要极致性能吗？
│  │       ├─ 是 → 用 **Polars**
│  │       └─ 否 → 用 **Pandas**（默认选择）
│  │
│  └─ 否 → 需要分布式吗？
│          ├─ 是 → 用 **Dask**
│          └─ 否 → 用 **Pandas + chunksize 分块读取**
│
├─ 主要操作类型？
│  ├─ ETL / 数据管道 → **Polars** 更优
│  ├─ 探索性分析 / 交互式 → **Pandas** 更顺手
│  └─ 机器学习特征工程 → **Pandas** + sklearn 配合最佳
│
└─ 团队现状？
   ├─ 都熟悉 Pandas → 继续用 Pandas
   ├─ 有 Rust 经验 / 追求新工具 → 尝试 Polars
   └─ 有大数据集群 → 上 Dask
```

**一句话总结**：对于绝大多数大模型开发者来说，**Pandas 仍然是首选和必学工具**。Polars 和 Dask 是在你遇到特定瓶颈时的升级选项，而非替代品。
