---
title: 数据类型（dtype）体系
description: 基础类型、Nullable 类型、category 类型详解，以及大模型场景下的内存优化策略
---
# 类型系统详解


## 为什么 dtype 重要

数据类型决定了三件事：

1. **内存占用**：`int8`(1字节) vs `int64`(8字节) → 差 **8 倍**
2. **可支持的操作**：数值型才能做数学运算，字符串型才能用 `.str`
3. **缺失值行为**：传统 `int64` 遇 `None` 变 `float64`，`Int64` 保持整型

在大模型数据处理中，dtype 选择直接影响你能否在单机上处理千万级语料。

## 基础类型体系

### 数值类型

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'tiny_int': np.array([1, 2, 3], dtype='int8'),        # -128 ~ 127
    'small_int': np.array([1000, 2000, 3000], dtype='int16'),   # -32K ~ 32K
    'normal_int': [100000, 200000, 300000],                  # int64 (默认)
    'float_precise': [1.123456789, 2.987654321, 3.111111111],  # float64
    'float_efficient': np.array([1.1, 2.2, 3.3], dtype='float32'),
    'bool_val': [True, False, True],
})

print(df.dtypes)

print("\n内存占用:")
print(df.memory_usage(deep=True))
```

### 各整数类型的范围与适用场景

| 类型 | 字节 | 范围 | 大模型场景 |
|------|------|------|-----------|
| `int8` | 1 | -128 ~ 127 | 极小枚举（如标签编码 0-127） |
| `int16` | 2 | -32,768 ~ 32,767 | token 数量（< 32000） |
| `int32` | 4 | ±21 亿 | 行号 / ID / 计数器 |
| `int64` | 8 | ±9.2×10¹⁸ | 时间戳 / 大 ID / 累加值 |
| `uint8` | 1 | 0 ~ 255 | ASCII 码 / 颜色通道 |
| `uint16` | 2 | 0 ~ 65535 | 端口号 / Unicode 基本面 |
| `uint32` | 4 | 0 ~ 42 亿 | IPv4 地址 / 文件偏移 |
| `uint64` | 8 | 0 ~ 1.8×10¹⁹ | 超大唯一 ID |

### 浮点类型选择

| 类型 | 精度 | 适用场景 |
|------|------|---------|
| `float16` | 半精度 (~3位有效) | ML 推理压缩、GPU 存储 |
| `float32` | 单精度 (~7位有效) | ML 训练标准格式、嵌入向量 |
| `float64` | 双精度 (~15位有效) | 统计分析、财务计算、默认选项 |

## Nullable 类型：现代缺失值处理方式

### 传统方式的痛点

```python
import pandas as pd

s_old = pd.Series([1, None, 3])
print(s_old.dtype)       # float64！整型被破坏了
print(s_old[1])          # nan（不是 None，也不是 NA）
```

问题：一个 `None` 就把整个列从 `int64` 降级为 `float64`。这在需要严格区分"无数据"和"零值"的场景中是致命的。

### Nullable 类型解决方案

```python
import pandas as pd

s_int = pd.Series([1, None, 3], dtype='Int64')     # 注意大写 I
s_float = pd.Series([1.5, None, 3.5], dtype='Float64')
s_bool = pd.Series([True, None, False], dtype='boolean')
s_str = pd.Series(['hello', None, 'world'], dtype='string')

df_na = pd.DataFrame({
    'int_col': s_int,
    'float_col': s_float,
    'bool_col': s_bool,
    'str_col': s_str,
})

print(df_na.dtypes)

print(df_na)

print(type(df_na['int_col'][1]))  # <class 'pandas.core.arrays.masked._Missing'>
```

### Nullable 类型完整对照表

| 传统类型 | Nullable 版本 | 缺失值符号 | 可参与运算 |
|----------|-------------|-----------|----------|
| `int64` | **`Int64`** | `<NA>` | ✅ 运算时自动忽略 |
| `int32` | **`Int32`** | `<NA>` | ✅ |
| `float64` | **`Float64`** | `<NA>` | ✅ |
| `bool` | **`boolean`** | `<NA>` | ✅ |
| `object` (str) | **`string`** | `<NA>` | ✅ |

### PyArrow 后端下的 Nullable 类型

```python
pd.options.dtype_backend = 'pyarrow'

df_arrow = pd.DataFrame({
    'int_col': [1, None, 3],
    'text_col': ['hello', None, 'world'],
})

print(df_arrow.dtypes)

```

**推荐**：新项目直接使用 `pd.options.dtype_backend = 'pyarrow'`，它自动提供 Nullable 语义。

## category 类型：低基数列的内存神器

### 原理

`category` 类型用**整数映射**代替实际字符串存储：

```
普通 object 存储方式:
┌─────────────┬─────────────┬──────────────┐
│  "api"      │  "web"      │  "export"    │  ← 每行存完整字符串
│  "api"      │  "api"      │  "web"       │
│  ...        │  ...        │  ...         │
└─────────────┴─────────────┴──────────────┘
内存: 每个字符串独立存储，重复越多浪费越多

category 存储方式:
┌──────────┐ ┌─────────────────────────────┐
│ 整数映射: │ │ 字典 (只存一份):            │
│ 0, 1, 2  │ │ 0 → "api", 1 → "web", ...   │
│ 1, 0, 2  │ └─────────────────────────────┘
│ ...      │ 每行只需存一个小整数
└──────────┘
内存: 几乎只有 object 的 1/5 ~ 1/20
```

### 实测效果

```python
import pandas as pd
import numpy as np

n = 5_000_000
categories = ['api', 'web', 'export', 'mobile', 'desktop',
               'pos', 'neg', 'neutral', 'code', 'general']

data = {
    'label': np.random.choice(categories, n),
    'source': np.random.choice(['A', 'B', 'C', 'D'], n),
    'model': np.random.choice(['GPT-4o', 'Claude', 'Llama', 'Gemini'], n),
    'quality': np.random.choice([1, 2, 3, 4, 5], n),
    'text': ['sample conversation text data'] * n,
}

df_obj = pd.DataFrame(data)
mem_obj = df_obj.memory_usage(deep=True).sum()

df_cat = df_obj.copy()
for col in ['label', 'source', 'model', 'quality']:
    df_cat[col] = df_cat[col].astype('category')
mem_cat = df_cat.memory_usage(deep=True).sum()

print(f"{'列名':<12} {'object (MB)':>12} {'category (MB)':>14} {'节省':>8}")
print("-" * 52)
for col in data.keys():
    obj_mem = df_obj[col].memory_usage(deep=True).sum() / 1024**2
    cat_mem = df_cat[col].memory_usage(deep=True).sum() / 1024**2
    saving = (1 - cat_mem / obj_mem) * 100 if obj_mem > 0 else 0
    print(f"{col:<12} {obj_mem:>11.2f} {cat_mem:>13.2f} {saving:>7.1f}%")

print(f"\n{'总计':<12} {mem_obj/1024**2:>11.2f} {mem_cat/1024**2:>13.2f} {(1-mem_cat/mem_obj)*100:>7.1f}%")
```

典型输出：

```
列名         object (MB)  category (MB)     节省
----------------------------------------------------
label              382.34          19.01   95.0%
source              38.23           9.56   75.0%
model               47.79          14.34   70.0%
quality             38.23           9.56   75.0%
text               892.30         892.30    0.0%

总计              1398.89         944.77   32.5%
```

**关键发现**：
- 低基数列（如 `label` 有 10 种取值）→ 节省 **95%**
- 高基数列（如 `text` 每行都不同）→ **无效果**（不要对高基数列用 category）
- 总体节省约 **32%**

### 何时使用 category

```python
df['country'].astype('category')  # 200 个国家 × 100 万行 → 好
df['weekday'].astype('category')   # 7 天 × 100 万行 → 极好

df['user_id'].astype('category')  # 90 万个不同 user_id → 更占内存！


```

## dtype 转换操作

### astype()

```python
import pandas as pd

df = pd.DataFrame({
    'a': [1, 2, 3],
    'b': ['1.5', '2.7', '3.9'],
    'c': ['x', 'y', 'z'] * 50,
})

df['a_int16'] = df['a'].astype('int16')
df['b_float'] = df['b'].astype(float)

df['c_cat'] = df['c'].astype('category')

df['a_nullable'] = df['a'].astype('Int64')

df['c_arrow'] = df['c'].astype('string[pyarrow]')

print(df.dtypes)
```

### pd.to_numeric() / pd.to_datetime()

```python
import pandas as pd

df = pd.DataFrame({
    'mixed_num': ['1', '2.5', '3', 'bad_value', '5'],
    'date_str': ['2025-01-15', '2025-03-20', 'not a date', '2025-06-01'],
})

df['num_clean'] = pd.to_numeric(df['mixed_num'], errors='coerce')
print(df[['mixed_num', 'num_clean']])

df['date_clean'] = pd.to_datetime(df['date_str'], errors='coerce')
print(df[['date_str', 'date_clean']])
```

### downcast：自动降级以节省内存

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'id': range(1000),
    'score': np.random.uniform(0, 100, 1000),
    'count': np.random.randint(0, 500, 1000),
})

print("转换前:")
print(df.dtypes)
print(f"内存: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")

df['id_down'] = pd.to_numeric(df['id'], downcast='integer')
df['score_down'] = pd.to_numeric(df['score'], downcast='float')
df['count_down'] = pd.to_numeric(df['count'], downcast='integer')

print("\n转换后:")
print(df[['id_down', 'score_down', 'count_down']].dtypes)
print(f"内存: {df[['id_down', 'score_down', 'count_down']].memory_usage(deep=True).sum() / 1024:.1f} KB")
```

## 大模型场景的 dtype 优化模板

```python
def optimize_dtypes_for_llm(df):
    """针对 LLM 数据处理场景的 dtype 优化"""
    df_optimized = df.copy()
    
    for col in df_optimized.columns:
        original_type = df_optimized[col].dtype
        unique_count = df_optimized[col].nunique()
        total = len(df_optimized)
        
        if unique_count == total:
            if str(original_type) == 'object':
                df_optimized[col] = df_optimized[col].astype('string[pyarrow]')
            continue
        
        ratio = unique_count / total
        
        if pd.api.types.is_integer_dtype(original_type):
            df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='integer')
            
        elif pd.api.types.is_float_dtype(original_type):
            df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='float')
            
        elif pd.api.types.is_object_dtype(original_type) or str(original_type) == 'string':
            if ratio < 0.01 and unique_count < 1000:
                df_optimized[col] = df_optimized[col].astype('category')
            else:
                df_optimized[col] = df_optimized[col].astype('string[pyarrow]')
                
        elif unique_count <= 10:
            df_optimized[col] = df_optimized[col].astype('category')
    
    return df_optimized


import numpy as np
n = 100_000
df_test = pd.DataFrame({
    'prompt_id': range(n),
    'prompt_text': ['sample prompt text for testing purposes'] * n,
    'response_len': np.random.randint(20, 5000, n),
    'quality_score': np.round(np.random.uniform(1, 5, n), 1),
    'source': np.random.choice(['api', 'web', 'export', 'mobile'], n),
    'model': np.random.choice(['GPT-4o', 'Claude', 'Llama'], n),
    'has_code': np.random.choice([True, False], n),
})

before = df_test.memory_usage(deep=True).sum() / 1024**2
df_opt = optimize_dtypes_for_llm(df_test)
after = df_opt.memory_usage(deep=True).sum() / 1024**2

print(f"优化前: {before:.1f} MB")
print(f"优化后: {after:.1f} MB")
print(f"节省: {(1 - after/before)*100:.1f}%")

print("\n优化后各列 dtype:")
print(df_opt.dtypes)
```

这个函数会根据每列的数据特征自动选择最优 dtype——在生产环境中处理大规模语料时，这是**必做的第一步优化**。
