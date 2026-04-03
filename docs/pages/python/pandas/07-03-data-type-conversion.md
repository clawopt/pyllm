---
title: 数据类型转换
description: astype() / pd.to_datetime() / pd.to_numeric() 的使用，大模型场景下的类型修正与优化策略
---
# 数据类型转换


## 为什么类型转换很重要

```python

import pandas as pd
import numpy as np

s_int = pd.Series([1, 2, None, 4])       # float64 (因为 None)
s_str = pd.Series(['1.5', '2.7', 'bad'])    # object (无法计算)
s_mixed = pd.Series(['2025-01-15', 'bad', '2025-03-20'])  # object
```

## astype()：基础转换

### 数值类型转换

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'a': [1, 2, 3],
    'b': [1.5, 2.7, 3.9],
    'c': [100000, 200000, 300000],
    'd': [True, False, True],
})

df['a_small'] = df['a'].astype('int8')     # -128 ~ 127
df['c_short'] = df['c'].astype('int32')   # ±21 亿

df['b_f32'] = df['b'].astype('float32')

print(df.dtypes)
```

### 自动降级：downcast

```python
import pandas as pd

df = pd.DataFrame({
    'id': range(1000),
    'score': np.random.uniform(0, 100, 1000),
    'count': np.random.randint(0, 500, 1000),
})

df['count_i16'] = df['count'].astype('int16')

df['score_opt'] = pd.to_numeric(df['score'], downcast='integer')
df['count_opt'] = pd.to_numeric(df['count'], downcast='integer')
df['float_opt'] = pd.to_numeric(df['score'], downcast='float')

mem_before = df[['id', 'score', 'count']].memory_usage(deep=True).sum()
mem_after = df[['score_opt', 'count_opt', 'float_opt']].memory_usage(deep=True).sum()

print(f"转换前: {mem_before/1024:.0f} KB")
print(f"转换后: {mem_after/1024:.0f} KB ({(1-mem_after/mem_before)*100:.0f}% 节省)")
```

### 分类变量转换

```python
import pandas as pd
import numpy as np

n = 500_000
df = pd.DataFrame({
    'label': np.random.choice(['pos', 'neg', 'neutral'], n),
    'source': np.random.choice(['api', 'web', 'export', 'mobile'], n),
    'model': np.random.choice(['GPT-4o', 'Claude', 'Llama', 'Gemini', 'Qwen'], n),
})

for col in ['label', 'source', 'model']:
    before_mem = df[col].memory_usage(deep=True).sum()
    
    df[f'{col}_cat'] = df[col].astype('category')
    
    after_mem = df[f'{col}_cat'].memory_usage(deep=True).sum()
    
    print(f"{col}: {before_mem/1024:.1f} KB → {after_mem/1024:.1f} KB "
          f"({(1-after_mem/before_mem)*100:.0f}% 节省)")
```

### 字符串类型转换

```python
import pandas as pd

pd.options.dtype_backend = 'pyarrow'

df = pd.DataFrame({'text': ['hello', 'world', None]})

df['text_arrow'] = df['text'].astype('string[pyarrow]')

df['text_native'] = df['text'].astype('string')

print(df.dtypes)
```

## pd.to_datetime() / pd.to_numeric()

### 处理脏数据中的混合类型

```python
import pandas as pd

df = pd.DataFrame({
    'raw_date': ['2025-01-15', '2025/03/20', 'not a date', 
                 '15-Jan-2025', '20250601', None],
    'raw_number': ['1.5', '2.7', 'N/A', '999', '-3.14', ''],
})

df['num_clean'] = pd.to_numeric(df['raw_number'], errors='coerce')
print(df[['raw_number', 'num_clean']])

df['date_clean'] = pd.to_datetime(
    df['raw_date'],
    errors='coerce',
    format='mixed'  # Pandas 2.0+ 支持自动推断格式
)
print(df[['raw_date', 'date_clean']])
```

### 大模型时间戳处理

```python
import pandas as pd
import numpy as np

np.random.seed(42)
n = 10_000

log_df = pd.DataFrame({
    'timestamp_str': [
        f'2025-{(i%12)+1:02d}-{(i%28)+1:02d} {(i%24):02d}:{(i%60):02d}:{(i%60):02d}'
        for i in range(n)
    ],
    'unix_ts': np.random.randint(1704067200, 1735689600, n),  # 2024-01 ~ 2024-12
})

log_df['ts_from_str'] = pd.to_datetime(log_df['timestamp_str'])
log_df['ts_from_unix'] = pd.to_datetime(log_df['unix_ts'], unit='s')

log_df['year'] = log_df['ts_from_str'].dt.year
log_df['month'] = log_df['ts_from_str'].dt.month
log_df['dayofweek'] = log_df['ts_from_str'].dt.dayofweek
log_df['hour'] = log_df['ts_from_str'].dt.hour

log_df['is_workday'] = log_df['ts_from_str'].dt.dayofweek < 5

log_df['days_ago'] = (pd.Timestamp.now() - log_df['ts_from_str']).dt.days

print(log_df.head())
```

## 批量 dtype 优化函数

```python
import pandas as pd
import numpy as np


def optimize_dtypes(df, verbose=True):
    """
    自动优化 DataFrame 的 dtype 以减少内存占用
    
    策略:
    - 整数列: downcast 到最小够用类型
    - 浮点列: downcast 到 float32（除非需要高精度）
    - 低基数 object/category: 转 category
    - 高基数 object: 转 string[pyarrow]
    """
    
    result = df.copy()
    original_mem = result.memory_usage(deep=True).sum()
    savings_log = []
    
    for col in result.columns:
        original_type = str(result[col].dtype)
        new_type = original_type
        
        if pd.api.types.is_integer_dtype(result[col]):
            result[col] = pd.to_numeric(result[col], downcast='integer')
            new_type = str(result[col].dtype)
            
        elif pd.api.types.is_float_dtype(result[col]):
            if col not in ['embedding_vector']:  # 不对嵌入向量降精度
                result[col] = pd.to_numeric(result[col], downcast='float')
            new_type = str(result[col].dtype)
            
        elif pd.api.types.is_object_dtype(result[col]) or \
             str(result[col].dtype) == 'string':
            unique_ratio = result[col].nunique() / len(result)
            
            if unique_ratio < 0.01 and result[col].nunique() < 2000:
                result[col] = result[col].astype('category')
                new_type = 'category'
            else:
                try:
                    result[col] = result[col].astype('string[pyarrow]')
                    new_type = 'string[pyarrow]'
                except Exception:
                    pass
    
    optimized_mem = result.memory_usage(deep=True).total()
    
    if verbose:
        saved_mb = (original_mem - optimized_mem) / 1024**2
        pct_saved = (1 - optimized_mem / original_mem) * 100
        print(f"\n📊 Dtype 优化结果:")
        print(f"  优化前: {original_mem/1024**2:.1f} MB")
        print(f"  优化后: {optimized_mem/1024**2:.1f} MB")
        print(f"  节省:   {saved_mb:.1f} MB ({pct_saved:.1f}%)")
        
        print(f"\n  各列最终 dtype:")
        for col in result.columns:
            print(f"    {col:<25s}: {str(result[col].dtype)}")
    
    return result


np.random.seed(42)
test_data = {
    'id': range(100_000),
    'prompt': ['sample text data for testing purposes'] * 50_000 + 
              [f'unique prompt {i}' for i in range(50_000)],
    'quality': np.round(np.random.uniform(1, 5, 100_000), 1),
    'tokens': np.random.randint(20, 2000, 100_000),
    'label': np.random.choice(['pos', 'neg', 'neutral'], 100_000),
    'source': np.random.choice(['api', 'web', 'export', 'mobile'], 100_000),
}

df_test = pd.DataFrame(test_data)
df_optimized = optimize_dtypes(df_test)
```
