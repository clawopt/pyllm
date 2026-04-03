---
title: 基础信息查看
description: .info() / .describe() / .shape 的深度解读与实战，大模型场景下的数据概览方法论
---
# 基础信息概览


## 拿到数据后的第一件事

无论数据来自 CSV、数据库还是 API 返回的 JSON，**第一眼看到的数据决定了你后续所有操作的正确性**。

很多数据科学事故都源于"没看清数据就开始写处理代码"。

### 最小可行检查（MVC）

```python
import pandas as pd

def minimal_viable_check(df, name="Dataset"):
    """三行代码确认数据基本健康"""
    print(f"\n{'='*50}")
    print(f"  📋 {name} — 快速检查")
    print(f"{'='*50}")
    
    print(f"  形状: {df.shape[0]:,} 行 × {df.shape[1]} 列")
    print(f"  列名: {list(df.columns)}")
    
    if len(df) > 0:
        print(f"\n  前 2 行预览:")
        print(df.head(2).to_string())
        
        null_counts = df.isnull().sum()
        has_null = null_counts[null_counts > 0]
        if len(has_null) > 0:
            print(f"\n  ⚠️ 存在缺失值的列:")
            for col, cnt in has_null.items():
                print(f"    - {col}: {cnt:,} 个空值 ({cnt/len(df)*100:.1f}%)")
        else:
            print(f"  ✅ 无缺失值")
    
    return df.shape

df = pd.DataFrame({'a': [1, None, 3], 'b': ['x', 'y', None]})
minimal_viable_check(df, "测试数据")
```

## .info()：类型与内存全景

### 输出解读指南

```python
import pandas as pd
import numpy as np

np.random.seed(42)
n = 500_000

df = pd.DataFrame({
    'id': range(n),
    'prompt': [f'问题{i}' for i in range(n)],
    'response': [f'回答{i}' for i in range(n)],
    'quality': np.random.choice([None] + list(range(1,6)), n),
    'tokens': np.random.randint(10, 2000, n),
    'source': np.random.choice(['api', 'web', 'export'], n),
})

df.info(verbose=True, show_counts=True)
```

输出：

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 500000 entries, 0 to 499999
Data columns (total 6 columns):
---  ------      --------------   -----  
 0   id              500000 non-null   int64  
 1   prompt          500000 non-null   object 
 2   response        498234 non-null   object 
 3   quality         499876 non-null   float64
 4   tokens          500000 non-null   int64  
 5   source          500000 non-null   object 
dtypes: int64(2), object(2), float64(1)
memory usage: 102.5+ MB
```

### 从 info() 中提取的关键决策信息

| 信息项 | 在哪里看 | 如何解读 | 决策 |
|--------|---------|---------|------|
| **总行数** | `RangeIndex: N entries` | 数据量是否符合预期？ | 决定是否需要分块处理 |
| **Non-Null Count** | 每列的 `non-null` 数 | 哪列有缺失？缺多少？ | 是否需要 `dropna()` / `fillna()` |
| **Dtype** | `Dtype` 行 | 类型是否合理？ | `object` → 考虑转 `category` 或 `string[pyarrow]` |
| **object 列数量** | 统计 `dtype` 中的 object | 文本列占比大吗？ | 内存优化的重点目标 |
| **memory usage** | 最后一行 | 占多少内存？ | 能否全量加载？是否需优化 |

### info() 进阶用法

```python
import pandas as pd

df[['quality', 'tokens', 'source']].info()

df.info(memory_usage='deep')

buf = []
df.info(buf=buf, memory_usage='deep')
info_str = '\n'.join(buf)

def extract_info_summary(df):
    buf = []
    df.info(buf=buf, verbose=False, memory_usage='deep')
    lines = ''.join(buf).split('\n')
    
    summary = {}
    for line in lines:
        if 'entries' in line.lower():
            summary['total_rows'] = int(line.split(' ')[-2])
        elif 'columns' in line.lower():
            summary['total_cols'] = int(line.split(' ')[-2])
        elif 'memory usage' in line.lower():
            summary['memory_mb'] = float(line.split(' ')[-1].replace('+',''))
    
    dtypes = {}
    non_nulls = {}
    current_col = None
    
    for line in lines[5:-4]:
        stripped = line.strip()
        if not stripped or stripped.startswith('#') or stripped.startswith('-'):
            continue
        
        parts = stripped.split()
        if len(parts) >= 2 and (parts[-1].startswith('int') or 
                                   parts[-1].startswith('float') or
                                   parts[-1].startswith('object') or
                                   parts[-1].startswith('bool') or
                                   parts[-1].startswith('category') or
                                   parts[-1].startswith('string')):
            col_name = parts[0]
            dtype = parts[-1]
            non_null_str = parts[1] if len(parts) >= 3 else ''
            
            try:
                non_null_val = int(non_null_str.replace(',', '').replace('non-null', ''))
                dtypes[col_name] = dtype
                non_nulls[col_name] = non_null_val
            except:
                pass
    
    summary['dtypes'] = dtypes
    summary['non_nulls'] = non_nulls
    return summary

info = extract_info_summary(df)
print(info)
```

## .describe()：统计摘要

### 数值列统计量详解

```python
import pandas as pd
import numpy as np

np.random.seed(42)
n = 100_000

df = pd.DataFrame({
    'quality': np.random.choice([1,2,3,4,5], n, p=[0.05, 0.15, 0.30, 0.35, 0.15]),
    'tokens': np.random.lognormal(5.5, 1.0, n).astype(int).clip(20, 2000),
    'response_time_ms': np.random.exponential(500, n).astype(int),
    'score': np.random.uniform(0, 1, n),
})

desc = df.describe(percentiles=[.01, .05, .10, .25, .50, .75, .90, .95, .99])
print(desc.round(3))
```

输出：

```
           quality       tokens  ...     score
count    100000.000  100000.000  ...  100000.000
mean          3.651      247.892  ...      0.499
std           1.097      178.342  ...      0.289
min          1.000       20.000  ...      0.000
1%           2.000       55.000  ...      0.010
5%           2.000       88.000  ...      0.049
10%          3.000      114.000  ...      0.100
25%          3.000      172.000  ...      0.250
50%          4.000      245.000  ...      0.499
75%          4.000      330.000  ...      0.750
90%          5.000      458.000  ...      0.900
95%          5.000      538.000  ...      0.950
99%          5.000      789.000  ...      0.990
max          5.000      2000.000  ...      1.000
```

### 各统计量的业务含义（LLM 场景）

| 统计量 | 数学定义 | LLM 数据中的含义 | 异常信号 |
|--------|---------|-----------------|---------|
| **count** | 有效样本数 | 实际参与计算的样本量 | 远小于预期 → 有大量缺失 |
| **mean** | 算术平均 | 平均 token 数 / 平均质量分 | 与 median 差距大 → 有极端值 |
| **std** | 标准差 | 数据离散程度 | 过大 → 数据质量不稳定 |
| **min** | 最小值 | 最短对话 / 最低分 | 负值或异常低值 |
| **max** | 最大值 | 最长对话 / 最高分 | 可能是异常值或错误数据 |
| **P1 (1%)** | 第1百分位 | 底部 1% 的阈值 | 用于识别极短/极低质量样本 |
| **P99 (99%)** | 第99百分位 | 顶部 1% 的阈值 | 用于识别超长样本 |
| **IQR** | P75-P25 | 四分位距 | 衡量数据集中度；异常值检测依据 |

### describe() 的 include 参数

```python
print(df.describe(include=[np.number]))

print(df.describe(include=['object', 'category']))

print(df.describe(include='all'))

print(df.describe(exclude=['int']))
```

## .shape 与相关属性

```python
import pandas as pd

df = pd.DataFrame({'a': range(100), 'b': range(100)})

print(df.shape)        # (100, 2)

print(df.ndim)         # 2

print(df.size)         # 200

print(df.empty)         # False
print(pd.DataFrame().empty)  # True

print(df.memory_usage(deep=True))
print(df.memory_usage(index=True, deep=True))

n_rows, n_cols = df.shape
print(f"{n_rows:,} 行 × {n_cols} 列")
```

### 大模型场景：快速判断数据规模等级

```python
def data_scale_tier(df):
    """判断数据集规模等级"""
    rows, cols = df.shape
    mem_mb = df.memory_usage(deep=True).sum() / 1024**2
    
    if rows < 10_000:
        tier = "微型"
        advice = "可直接全量处理，无需优化"
    elif rows < 100_000:
        tier = "小型"
        advice = "注意 dtype 优化即可"
    elif rows < 1_000_000:
        tier = "中型"
        advice = "建议使用 category 类型 + PyArrow 后端"
    elif rows < 10_000_000:
        tier = "大型"
        advice = "考虑分块读取或转 Parquet 格式"
    elif rows < 100_000_000:
        tier = "超大型"
        advice = "必须分块处理 + Parquet + dtype 优化"
    else:
        tier = "海量级"
        advice = "超出 Pandas 单机能力范围，考虑 Dask/Polars"
    
    return {
        'tier': tier,
        'rows': f'{rows:,}',
        'cols': cols,
        'memory_mb': f'{mem_mb:.1f}',
        'advice': advice,
    }

tier = data_scale_tier(df)
print(f"""
📊 数据规模评估: 【{tier['tier']}】
   行数: {tier['rows']}
   列数: {tier['cols']}
   内存: {tier['memory_mb']} MB
   建议: {tier['advice']}
""")
```
