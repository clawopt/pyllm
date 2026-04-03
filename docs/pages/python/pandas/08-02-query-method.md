---
title: query() 方法：简洁表达式筛选
description: query() 的语法详解、变量引用、与布尔索引的对比、性能考量与大模型场景示例
---
# query() 方法详解


## 为什么需要 query()

当筛选条件变得复杂时，布尔索引会变得难以阅读：

```python
result = df[
    (df['quality'] >= 4.0) &
    (df['tokens'].between(200, 1500)) &
    (df['source'].isin(['api', 'web'])) &
    (~df['text'].str.contains('ERROR')) &
    (df['has_code'] == True)
]

result = df.query(
    "quality >= 4.0 and "
    "200 <= tokens <= 1500 and "
    "source in ['api', 'web'] and "
    "not text.str.contains('ERROR') and "
    "has_code == True"
)
```

**`query()` 的核心优势**：
1. **可读性**：更像自然语言或 SQL
2. **简洁性**：不需要反复写 `df['col']`
3. **少括号**：复杂 `&` / `|` 组合时更清晰

## 基础语法

### 比较运算符

```python
import pandas as pd
import numpy as np

np.random.seed(42)
n = 50_000

df = pd.DataFrame({
    'model': np.random.choice(['GPT-4o', 'Claude', 'Llama', 'Gemini'], n),
    'score': np.round(np.random.uniform(80, 95, n), 1),
    'price': np.round(np.random.uniform(0.1, 15, n), 2),
    'context': [128000, 200000, 131072, 1000000] * (n//4),
})

r1 = df.query("score > 88")
r2 = df.query("price < 5.0")
r3 = df.query("score >= 90 and price <= 2.0")
r4 = df.query("context > 500000")

r5 = df.query('model == "GPT-4o"')
r6 = df.query('model != "Llama"')
```

### in / not in 操作

```python
in_list = df.query("model in ['GPT-4o', 'Claude']")
not_in_list = df.query("model not in ['Llama', 'Qwen']")

selected = df.query("model in ('GPT-4o', 'Claude', 'Gemini')")
print(f"选中模型: {len(selected)} 条")
```

### 字符串方法调用

```python
import pandas as pd

df = pd.DataFrame({
    'text': [
        '什么是注意力机制',
        'ERROR: 超时',
        '<html>页面</html>',
        '正常文本内容',
    ]
})

has_html = df.query('text.str.contains("<html>", na=False)')
no_error = df.query('~text.str.contains("ERROR", na=False)')

starts_what = df.query('text.str.startswith("什么")')

matches_ai = df.query('text.str.contains("注意|PyTorch|HTML", regex=True)')

long_text = df.query('text.str.len() > 5')
```

### 布尔组合

```python
import pandas as pd

df = pd.DataFrame({
    'a': [1, 2, 3, 4, 5],
    'b': [10, 20, 30, 40, 50],
    'c': ['x', 'y', 'x', 'z', 'w'],
})

and_result = df.query("a >= 2 and b <= 40 and c != 'z'")

or_result = df.query("a == 1 or a == 5 or b == 30")

not_result = df.query("not (c == 'x' or c == 'y')")

complex_q = df.query(
    "(a > 2 or b < 25) and c in ('x', 'w')"
)
```

## 变量引用：在 query 中使用外部变量

### @ 前缀引用

```python
import pandas as pd

df = pd.DataFrame({
    'value': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
})

threshold = 5.5
min_val = 2
max_val = 8
allowed = [3, 5, 7]

above_threshold = df.query("value > @threshold")
in_range = df.query("@min_val <= value <= @max_val")
in_list = df.query("value in @allowed")
multiple_refs = df.query("@min_val <= value <= @max_val and value not in @allowed")

print(f"大于 {threshold}: {len(above_threshold)}")
print(f"在 [{min_val}, {max_val}] 且不在 {allowed} 中: {len(multiple_refs)}")
```

### 实际应用：动态参数化查询

```python
import pandas as pd
import numpy as np

np.random.seed(42)

def filter_by_config(df, min_quality=3.0, max_tokens=1000,
                     sources=None, models=None):
    """通过配置参数灵活筛选"""
    
    conditions = []
    
    if min_quality is not None:
        conditions.append(f"quality_score >= {min_quality}")
    
    if max_tokens is not None:
        conditions.append(f"token_count <= @max_tokens")
    
    if sources:
        conditions.append(f"source in @sources")
    
    if models:
        conditions.append(f"model in @models")
    
    if conditions:
        query_str = " and ".join(conditions)
        return df.query(query_str)
    else:
        return df.copy()


n = 100_000
demo_df = pd.DataFrame({
    'quality_score': np.round(np.random.uniform(1, 5, n), 1),
    'token_count': np.random.randint(20, 3000, n),
    'source': np.random.choice(['api', 'web', 'export'], n),
    'model': np.random.choice(['GPT-4o', 'claude', 'llama'], n),
})

result = filter_by_config(
    demo_df,
    min_quality=4.0,
    max_tokens=800,
    sources=['api'],
    models=['GPT-4o']
)
print(f"筛选结果: {len(result):,}")
```

## query() vs 布尔索引对比

```python
import pandas as pd
import numpy as np
import time

n = 1_000_000
np.random.seed(42)

df = pd.DataFrame({
    'a': np.random.randn(n),
    'b': np.random.randint(0, 100, n),
    'c': np.random.choice(['X', 'Y', 'Z', 'W'], n),
    'd': np.random.choice([True, False], n),
})

start = time.time()
for _ in range(10):
    r_bool = df[(df['a'] > 0) & (df['b'] > 50)]
t_bool_simple = time.time() - start

start = time.time()
for _ in range(10):
    r_query = df.query("a > 0 and b > 50")
t_query_simple = time.time() - start

start = time.time()
for _ in range(10):
    r_bool_c = df[
        (df['a'] > 0) & (df['b'].between(20, 80)) &
        (df['c'].isin(['X', 'Y'])) & (df['d'] == True)
    ]
t_bool_complex = time.time() - start

start = time.time()
for _ in range(10):
    r_query_c = df.query(
        "a > 0 and 20 <= b <= 80 and c in ('X', 'Y') and d == True"
    )
t_query_complex = time.time() - start

print(f"{'操作':<18} {'布尔索引':>10} {'query()':>10} {'query优势':>10}")
print("-" * 52)
print(f"{'简单条件':<18} {t_bool_simple:>9.3f}s {t_query_simple:>9.3f}s "
      f"{t_bool_simple/t_query_simple:>9.1f}x")
print(f"{'复杂条件':<18} {t_bool_complex:>9.3f}s {t_query_complex:>9.3f}s "
      f"{t_bool_complex/t_query_complex:>9.1f}x")
```

典型结果：

```
操作                 布尔索引   query()   query优势
----------------------------------------------------
简单条件              0.234s     0.287s       0.82x
复杂条件              0.456s     0.398s       1.15x
```

**结论**：
- **简单条件**：布尔索引略快（~20%）
- **复杂条件**：`query()` 反而更快（因为内部有优化引擎）
- **代码可读性**：`query()` **大幅领先**

## LLM 场景实战：用 query 构建交互式数据探索器

```python
import pandas as pd
import numpy as np


class DataExplorer:
    """基于 query 的交互式数据探索器"""
    
    def __init__(self, df):
        self.df = df
        self.original_shape = df.shape
    
    def explore(self, filters=None, columns=None,
                sort_by=None, ascending=False, limit=None):
        """
        参数式数据探索
        
        Args:
            filters: dict of {column: value} 或 {column: (min, max)}
            columns: list of columns to select
            sort_by: column to sort by
            ascending: sort direction
            limit: max rows to return
        """
        
        result = self.df.copy()
        
        if filters:
            conditions = []
            for col, value in filters.items():
                if isinstance(value, tuple) and len(value) == 2:
                    lo, hi = value
                    conditions.append(f"{lo} <= `{col}` <= {hi}")
                elif isinstance(value, list):
                    conditions.append(f"`{col}` in @value")
                elif isinstance(value, bool):
                    conditions.append(f"`{col}` == {int(value)}")
                else:
                    conditions.append(f"`{col}` == '{value}'")
            
            if conditions:
                result = result.query(" and ".join(conditions))
        
        if columns:
            existing = [c for c in columns if c in result.columns]
            result = result[existing]
        
        if sort_by and sort_by in result.columns:
            result = result.sort_values(sort_by, ascending=ascending)
        
        if limit and len(result) > limit:
            result = result.head(limit)
        
        print(f"\n📊 探索结果: {self.original_shape[0]:,} → {len(result):,} "
              f"(显示 {limit or len(result):,} 行)")
        
        if len(result) > 0:
            print(result.to_string())
        
        return result
    
    def quick_stats(self, group_col, agg_col):
        """快速分组统计"""
        return (
            self.df
            .groupby(group_col)[agg_col]
            .agg(['count', 'mean', 'std', 'min', 'max'])
            .round(2)
        )


np.random.seed(42)
explorer = DataExplorer(pd.DataFrame({
    'model': np.random.choice(['GPT-4o', 'Claude', 'Llama', 'Gemini', 'DeepSeek'], 10000),
    'MMLU': np.random.uniform(75, 92, 10000).round(1),
    'price_per_1m': np.random.uniform(0.1, 15, 10000).round(2),
    'context': np.random.choice([128000, 200000, 131072, 1000000], 10000),
}))

high_end = explorer.explore(
    filters={
        'MMLU': (85, 92),
        'context': (200000, float('inf')),
        'model': ['GPT-4o', 'Claude']
    },
    columns=['model', 'MMLU', 'price_per_1m', 'context'],
    sort_by='MMLU',
    ascending=False,
    limit=10
)

stats = explorer.quick_stats('model', 'MMLU')
print(stats)
```
