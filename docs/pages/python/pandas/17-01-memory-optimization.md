---
title: 内存优化实战
description: dtype 降级 / Category 转换 / PyArrow 后端 / 内存分析工具 / 大数据集策略
---
# 内存优化策略


## Pandas 性能优化全景图

```
慢 → 快
─────────────────────────────────────
iterrows() → itertuples() → apply() → 向量化 → Cython/Numba
object    → category     → int8/16   → float32/16
float64   → float32      → downcast  → PyArrow backend
```

## 第一步：诊断内存问题

```python
import pandas as pd
import numpy as np

def memory_report(df):
    """生成内存使用报告"""
    total = df.memory_usage(deep=True).sum() / 1024 / 1024

    report = []
    for col in df.columns:
        mem = df[col].memory_usage(deep=True) / 1024
        pct = mem / (total * 1024) * 100
        report.append({
            'column': col,
            'dtype': str(df[col].dtype),
            'memory_KB': round(mem, 1),
            'pct': round(pct, 1),
            'n_unique': df[col].nunique(),
            'n_null': df[col].isna().sum(),
        })

    report_df = pd.DataFrame(report).sort_values('memory_KB', ascending=False)
    print(f"总内存: {total:.2f} MB ({len(df):,} 行 × {len(df.columns)} 列)")
    print(f"\n按内存排序:")
    print(report_df.to_string(index=False))
    return report_df


np.random.seed(42)
n = 500_000
heavy_df = pd.DataFrame({
    'id': range(n),
    'model_name': np.random.choice(
        ['GPT-4o', 'Claude-3.5-Sonnet', 'Llama-3.1-70B',
         'Qwen2.5-72B', 'DeepSeek-V3'], n
    ),
    'task_type': np.random.choice(['chat', 'code', 'math', 'reasoning'] * 5, n),
    'prompt_text': [f'这是第{i}条测试数据的内容' for i in range(n)],
    'score_float64': np.random.uniform(0.6, 0.98, n),
    'latency_int64': np.random.randint(100, 5000, n).astype(np.int64),
})

report = memory_report(heavy_df)
```

## 第二步：自动优化 dtype

```python
import pandas as pd
import numpy as np

def auto_optimize_dtypes(df):
    """一键自动优化所有列的 dtype"""
    optimized = df.copy()
    savings = []

    for col in optimized.columns:
        original_mem = optimized[col].memory_usage(deep=True)

        if pd.api.types.is_object_dtype(optimized[col]):
            n_unique = optimized[col].nunique()
            ratio = n_unique / len(optimized)
            if ratio < 0.5 and n_unique < 10000:
                optimized[col] = optimized[col].astype('category')
            elif optimized[col].str.isnumeric().all():
                try:
                    optimized[col] = pd.to_numeric(optimized[col], downcast='integer')
                except (ValueError, TypeError):
                    pass

        elif pd.api.types.is_integer_dtype(optimized[col]):
            optimized[col] = pd.to_numeric(optimized[col], downcast='integer')

        elif pd.api.types.is_float_dtype(optimized[col]):
            optimized[col] = pd.to_numeric(optimized[col], downcast='float')

        new_mem = optimized[col].memory_usage(deep=True)
        saved = original_mem - new_mem
        if saved > 0:
            savings.append({
                'column': col,
                'original_dtype': str(df[col].dtype),
                'new_dtype': str(optimized[col].dtype),
                'saved_KB': round(saved / 1024, 1),
            })

    orig_total = df.memory_usage(deep=True).sum() / 1024 / 1024
    opt_total = optimized.memory_usage(deep=True).sum() / 1024 / 1024
    print(f"原始: {orig_total:.2f} MB")
    print(f"优化: {opt_total:.2f} MB")
    print(f"节省: {(1 - opt_total/orig_total)*100:.1f}%")

    if savings:
        print(f"\n优化详情:")
        for s in sorted(savings, key=lambda x: x['saved_KB'], reverse=True):
            print(f"  {s['column']:<15s}: {s['original_dtype']:>10s} → "
                  f"{s['new_dtype']:<10s} 节省 {s['saved_KB']:.1f} KB")

    return optimized


optimized_df = auto_optimize_dtypes(heavy_df.copy())
```

## PyArrow 后端（Pandas 2.0+）

```python
import pandas as pd
import numpy as np

df_arrow = pd.DataFrame({
    'text_col': pd.array(['hello world'] * 100_000, dtype='string[pyarrow]'),
    'category_col': pd.array(['A']*50000 + ['B']*50000, dtype='string[pyarrow]'),
})

df_arrow['length'] = df_arrow['text_col'].str.len()

print(f"PyArrow string 列内存:")
print(df_arrow.dtypes)
print(df_arrow.memory_usage(deep=True) / 1024)

nullable_int = pd.Series([1, 2, None, 4], dtype='Int64[pyarrow]')
print(f"\n可空整数: {nullable_int.dtype}")
```

## LLM 场景：百万级 SFT 数据集优化

```python
import pandas as pd
import numpy as np

class LargeDatasetOptimizer:
    """大数据集优化器"""

    STRATEGIES = {
        'low_card_string_to_category': {
            'condition': lambda col, df: (
                pd.api.types.is_object_dtype(col) and
                col.nunique() / len(df) < 0.3
            ),
            'transform': lambda s: s.astype('category'),
        },
        'int_downcast': {
            'condition': lambda col, df: pd.api.types.is_integer_dtype(col),
            'transform': lambda s: pd.to_numeric(s, downcast='integer'),
        },
        'float_downcast': {
            'condition': lambda col, df: pd.api.types.is_float_dtype(col),
            'transform': lambda s: pd.to_numeric(s, downcast='float'),
        },
    }

    def optimize(self, df):
        before_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        result = df.copy()
        applied = []

        for col in result.columns:
            for name, strategy in self.STRATEGIES.items():
                if strategy['condition'](result[col], result):
                    original_dtype = str(result[col].dtype)
                    result[col] = strategy['transform'](result[col])
                    after_dtype = str(result[col].dtype)
                    applied.append({
                        'col': col,
                        'strategy': name,
                        'from': original_dtype,
                        'to': after_dtype,
                    })
                    break

        after_mb = result.memory_usage(deep=True).sum() / 1024 / 1024
        saved_pct = (1 - after_mb / before_mb) * 100

        print(f"{'='*50}")
        print(f"优化报告: {before_mb:.2f} MB → {after_mb:.2f} MB "
              f"(节省 {saved_pct:.1f}%)")
        print(f"{'='*50}")
        for a in applied:
            print(f"  [{a['strategy']}] {a['col']}: {a['from']} → {a['to']}")

        return result


np.random.seed(42)
n = 1_000_000
big_sft = pd.DataFrame({
    'instruction_id': [f'inst_{i}' for i in range(n)],
    'source': np.random.choice(
        ['web_crawl', 'wiki', 'textbook', 'forum', 'paper',
         'github', 'arxiv', 'slack', 'discord', 'email'], n
    ),
    'domain': np.random.choice(['tech', 'finance', 'medical', 'legal', 'edu'], n),
    'lang': np.random.choice(['zh', 'en', 'mixed', 'code'], n),
    'quality_label': np.random.choice(['S', 'A', 'B+', 'B', 'C'], n),
    'instr_char_len': np.random.randint(10, 2000, n),
    'resp_char_len': np.random.randint(30, 8000, n),
    'reward_score': np.random.uniform(0.01, 0.99, n),
    'turn_count': np.random.randint(1, 15, n),
    'has_code': np.random.randint(0, 2, n),
    'has_math': np.random.randint(0, 2, n),
})

optimizer = LargeDatasetOptimizer()
optimized_big = optimizer.optimize(big_sft)
```
