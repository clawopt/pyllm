---
title: 数据查看与探索
description: head/tail/info/describe 等方法详解，大模型场景下快速预览百万级语料数据分布
---
# 数据探索与概览


## 为什么"先看再看"

在 Pandas 的哲学中，**拿到数据的第一件事永远是看看它长什么样**。

这不仅是好习惯——在生产环境中，这是**防止灾难性错误的防线**：

```python
```

## 快速预览

### head() / tail()

```python
import pandas as pd
import numpy as np

np.random.seed(42)
n = 1_000_000
df = pd.DataFrame({
    'prompt': [f'问题 {i}' for i in range(n)],
    'response': [f'回答 {i}' for i in range(n)],
    'quality': np.random.choice([1, 2, 3, 4, 5], n),
    'tokens': np.random.randint(20, 2000, n),
    'source': np.random.choice(['api', 'web', 'export'], n),
})

print(df.head())
print(df.tail())

print(df.head(10))    # 前 10 行
print(df.tail(3))     # 后 3 行

print(df.sample(5, random_state=42))
```

### 大模型语料的 head() 检查清单

每次加载新数据后，用 `head()` 检查这些要点：

```python
def quick_inspect(df, name="数据"):
    print(f"\n{'='*60}")
    print(f"📊 {name} 快速检查")
    print(f"{'='*60}")
    
    print(f"\n形状: {df.shape[0]:,} 行 × {df.shape[1]} 列")
    
    print(f"\n前 3 行:")
    print(df.head(3).to_string())
    
    print(f"\n列名: {list(df.columns)}")
    
    print(f"\n各列非空值数量:")
    non_null = df.notna().sum()
    total = len(df)
    for col in df.columns:
        pct = non_null[col] / total * 100
        flag = "✅" if pct > 99 else "⚠️" if pct > 90 else "❌"
        print(f"  {flag} {col}: {non_null[col]:>8,} / {total:,} ({pct:.1f}%)")

quick_inspect(df, "训练语料")
```

## info()：类型与内存全景

`info()` 是**最被低估的 Pandas 方法之一**。它一次性告诉你关于 DataFrame 的所有关键信息。

```python
df.info()
```

典型输出（百万级行）：

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1000000 entries, 0 to 999999
Data columns (total 5 columns):
---  ------      --------------   -----  
 0   prompt       1000000 non-null  object 
 1   response     998234 non-null   object 
 2   quality      1000000 non-null  int64  
 3   tokens       1000000 non-null  int64  
  4   source       1000000 non-null  object 
dtypes: int64(2), object(3)
memory usage: 38.1+ MB
```

从 `info()` 中你能读出什么？

| 信息 | 如何解读 | 发现问题时怎么办 |
|------|---------|----------------|
| **Non-Null Count** | `response` 有 **1,766 个空值** | 需要 `dropna()` 或 `fillna()` |
| **Dtype** | `object` 类型占 3 列 | 考虑转 `category` 或 `string[pyarrow]` |
| **memory usage** | **38.1 MB** | 是否需要优化？见后续第 16 章 |
| **总行数** | **1,000,000** | 数据量是否符合预期？ |

### info() 进阶用法

```python
df.info(verbose=True, show_counts=True, memory_usage='deep')

df[['prompt', 'quality', 'tokens']].info()

mem_usage = df.memory_usage(deep=True)
mem_usage['percentage'] = mem_usage / mem_usage.sum() * 100
print(mem_usage.sort_values(ascending=False))
```

## describe()：统计摘要

### 数值型列

```python
print(df.describe())
```

输出：

```
            quality        tokens
count  1,000,000.00  1,000,000.00
mean           3.01        512.34
std            1.41        289.56
min            1.00         20.00
25%            2.00        265.00
50%            3.00        510.00
75%            4.00        759.00
max            5.00       1999.00
```

每一行的含义：

| 统计量 | 含义 | LLM 场景意义 |
|--------|------|------------|
| count | 非空值数量 | 有效样本数 |
| mean | 算术均值 | 平均 token 数 / 平均质量分 |
| std | 标准差 | 数据离散程度 |
| min | 最小值 | 最短对话 / 最低分 |
| 25% | 第一四分位数 | 下四分之一数据的上界 |
| 50% (median) | 中位数 | 比 mean 更抗异常值 |
| 75% | 第三四分位数 | 上四分之一数据的下界 |
| max | 最大值 | 最长对话 / 最高分 |

### 分位数扩展

```python
percentiles = df['tokens'].describe(
    percentiles=[.01, .05, .10, .25, .50, .75, .90, .95, .99]
)
print(percentiles)

```

### 分类列

```python
print(df.describe(include=['object', 'category']))
```

输出：

```
          prompt  response     source
count   1,000,000   998,234   1,000,000
unique  1,000,000   987,432           3
top       问题 0    回答 0         api
freq             1         2     333,333
```

| 统计量 | 含义 |
|--------|------|
| unique | 唯一值数量 |
| top | 出现频率最高的值 |
| freq | 最高频值的出现次数 |

### 全部列一起描述

```python
print(df.describe(include='all'))
```

## 样本分布快速分析

### value_counts()

```python
print(df['quality'].value_counts().sort_index())

print(df['quality'].value_counts(normalize=True).round(3))

print(df['source'].value_counts())

dist = df['source'].value_counts()
pct = df['source'].value_counts(normalize=True).round(3)
pd.concat([dist, pct], axis=1, keys=['count', 'pct'])
```

### nunique() — 唯一值数量

```python
for col in df.columns:
    unique_count = df[col].nunique()
    total = len(df)
    ratio = unique_count / total
    
    if ratio < 0.05:
        category_type = "低基数（适合用 category 类型）"
    elif ratio > 0.8:
        category_type = "高基数（可能是 ID 或自由文本）"
    else:
        category_type = "中等基数"
    
    print(f"{col:12s} → {unique_count:>8,} 唯一 ({ratio:.1%}) [{category_type}]")
```

典型输出：

```
prompt       →  1,000,000 唯一 (100.0%) [高基数（可能是 ID 或自由文本）]
response     →    987,432 唯一 (98.7%) [高基数（可能是 ID 或自由文本）]
quality      →         5 唯一 (0.0%) [低基数（适合用 category 类型）]
tokens       →     1981 唯一 (0.2%) [中等基数]
source       →         3 唯一 (0.0%) [低基数（适合用 category 类型）]
```

这个分析直接指导你如何优化 dtype：
- `quality` 和 `source` → 改为 `category` 类型
- `tokens` → 保持 `int64` 或降为 `int16`
- `prompt` / `response` → 用 `string[pyarrow]`

## 缺失值与重复值检测

### isna() + sum()

```python
missing = df.isna().sum()
missing_pct = (missing / len(df) * 100).round(2)

missing_df = pd.DataFrame({
    '缺失数': missing,
    '占比(%)': missing_pct,
})
missing_df = missing_df[missing_df['缺失数'] > 0].sort_values('缺失数', ascending=False)

if len(missing_df) > 0:
    print("⚠️ 存在缺失值的列:")
    print(missing_df)
else:
    print("✅ 无缺失值")
```

### duplicated()

```python
n_duplicates = df.duplicated().sum()
print(f"完全重复的行: {n_duplicates}")

partial_dup = df.duplicated(subset=['prompt', 'response']).sum()
print(f"prompt+response 重复: {partial_dup}")
```

## 自定义探索报告模板

把以上所有检查整合为一个可复用的函数：

```python
def full_data_report(df, name="Dataset", max_unique_show=20):
    """生成完整的数据探索报告"""
    report_lines = []
    report_lines.append(f"\n{'█'*60}")
    report_lines.append(f"  📊 {name} 完整数据报告")
    report_lines.append(f"{'█'*60}")
    
    report_lines.append(f"\n📐 形状: {df.shape[0]:,} 行 × {df.shape[1]} 列")
    report_lines.append(f"💾 内存: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    report_lines.append(f"\n{'─'*40} 列详情 {'─'*40}")
    for col in df.columns:
        dtype = str(df[col].dtype)
        non_null = df[col].notna().sum()
        null_count = len(df) - non_null
        unique = df[col].nunique()
        
        if unique <= max_unique_show:
            top_values = df[col].value_counts().head(3).to_dict()
            top_str = ', '.join(f"{k}:{v}" for k,v in top_values.items())
        else:
            top_str = f"(过多, 仅显示前 {max_unique_show})"
        
        null_flag = f"[⚠️ {null_count:,} 缺失]" if null_count > 0 else "[✅]"
        
        report_lines.append(f"  {null_flag} {col:<20s} | {dtype:<18s} | "
                           f"{non_null:>8,}/{len(df):<8,} | "
                           f"唯一:{unique:>6,} | 典型值: {top_str}")
    
    report_lines.append(f"\n{'─'*40} 数值列统计 {'─'*40}")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        stats = df[numeric_cols].describe().T[['count', 'mean', 'std', 'min', '50%', 'max']]
        stats.columns = ['数量', '均值', '标准差', '最小值', '中位数', '最大值']
        report_lines.append(stats.round(2).to_string())
    else:
        report_lines.append("  (无数值列)")
    
    report_lines.append(f"\n{'─'*40} 类别列分布 {'─'*40}')
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols[:5]:
        dist = df[col].value_counts().head(10)
        report_lines.append(f"\n  【{col}】({df[col].nunique()} 种)")
        for val, cnt in dist.items():
            bar = '█' * min(int(cnt / len(df) * 100), 40)
            report_lines.append(f"    {bar} {val}: {cnt:,} ({cnt/len(df)*100:.1f}%)")
    
    report_text = '\n'.join(report_lines)
    print(report_text)
    return report_text

report = full_data_report(df, "LLM 训练语料")
```

这个函数会在终端输出一份格式化的数据报告，涵盖形状、内存、每列详情、数值统计、类别分布——**每次拿到新数据时运行一次即可全面了解数据状况**。
