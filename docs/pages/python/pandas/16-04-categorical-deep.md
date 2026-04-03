---
title: 分类数据（Categorical）深度应用
description: Categorical 类型优势 / 有序分类 / 内存优化 / 性能加速 / LLM 场景
---
# Categorical 深度解析


## Categorical 的核心价值

| 特性 | object (字符串) | category |
|------|-----------------|----------|
| 存储方式 | 每行存完整字符串 | 存整数索引 + 唯一值表 |
| 重复 "GPT-4o" ×100万次 | 占用 ~7MB | 占用 ~8 字节 |
| groupby 速度 | 较慢 | **快 5-10x** |
| 内存占用 (100万行) | ~80 MB | **~3 MB** |

## 基础用法

```python
import pandas as pd
import numpy as np

n = 1_000_000
df = pd.DataFrame({
    'model': np.random.choice(['GPT-4o', 'Claude', 'Llama', 'Qwen'], n),
    'task': np.random.choice(['chat', 'code', 'math', 'reasoning'], n),
    'value': np.random.randn(n),
})

df['model_cat'] = df['model'].astype('category')
df['task_cat'] = df['task'].astype('category')

print(f"原始 model 列: {df['model'].memory_usage(deep=True) / 1024:.1f} KB")
print(f"Category:        {df['model_cat'].memory_usage(deep=True) / 1024:.1f} KB")
print(f"节省:            {(1 - df['model_cat'].memory_usage() / df['model'].memory_usage()) * 100:.0f}%")
```

## 有序分类（Ordered Categorical）

```python
import pandas as pd

df = pd.DataFrame({
    'tier': ['bronze', 'gold', 'silver', 'gold', 'platinum', 'bronze'],
    'score': [70, 95, 85, 92, 98, 72],
})

tier_order = ['bronze', 'silver', 'gold', 'platinum']
df['tier_ordered'] = pd.Categorical(
    df['tier'],
    categories=tier_order,
    ordered=True
)

sorted_df = df.sort_values('tier_ordered')
print(sorted_df[['tier', 'tier_ordered', 'score']])

high_tiers = df[df['tier_ordered'] > 'silver']
print(f"\nsilver 以上: {len(high_tiers)} 条")
```

## Categorical 的性能优势

```python
import pandas as pd
import numpy as np
import time

n = 2_000_000
categories = [f'cat_{i}' for i in range(50)]

df_obj = pd.DataFrame({
    'group': np.random.choice(categories, n),
    'value': np.random.randn(n),
})
df_cat = df_obj.copy()
df_cat['group'] = df_cat['group'].astype('category')

start = time.time()
r_obj = df_obj.groupby('group')['value'].mean()
t_obj = time.time() - start

start = time.time()
r_cat = df_cat.groupby('group')['value'].mean()
t_cat = time.time() - start

print(f"object groupby: {t_obj:.4f}s")
print(f"category groupby: {t_cat:.4f}s ({t_obj/t_cat:.1f}x faster)")
assert r_obj.equals(r_cat)
```

## LLM 场景：SFT 数据集内存优化

```python
import pandas as pd
import numpy as np

class MemoryOptimizer:
    """基于 Categorical 的内存优化器"""

    OPTIMIZABLE_THRESHOLD = 0.5  # 唯一值占比低于此值则优化

    @staticmethod
    def analyze(df):
        """分析哪些列适合转为 Category"""
        results = []
        for col in df.columns:
            if not pd.api.types.is_object_dtype(df[col]):
                continue
            n_unique = df[col].nunique()
            n_total = len(df)
            ratio = n_unique / n_total
            orig_mem = df[col].memory_usage(deep=True)

            if ratio < MemoryOptimizer.OPTIMIZABLE_THRESHOLD:
                cat_mem = df[col].astype('category').memory_usage(deep=True)
                savings = (orig_mem - cat_mem) / orig_mem * 100
                results.append({
                    'column': col,
                    'unique_values': n_unique,
                    'total_rows': n_total,
                    'unique_ratio': round(ratio, 4),
                    'original_KB': round(orig_mem / 1024, 1),
                    'category_KB': round(cat_mem / 1024, 1),
                    'savings_pct': round(savings, 1),
                    'recommend': '✅ 推荐' if savings > 50 else '⚠️ 一般',
                })
        return pd.DataFrame(results)

    @staticmethod
    def optimize(df):
        """自动优化所有适合的列"""
        optimized = df.copy()
        report = MemoryOptimizer.analyze(optimized)

        for _, row in report.iterrows():
            col = row['column']
            if row['savings_pct'] > 30:
                optimized[col] = optimized[col].astype('category')

        return optimized, report


np.random.seed(42)
n = 500_000
sft_df = pd.DataFrame({
    'instruction': [f'问题 {i % 5000}' for i in range(n)],
    'source': np.random.choice(
        ['web_crawl', 'textbook', 'wiki', 'forum', 'paper',
         'synthetic', 'human_annotated', 'github', 'arxiv', 'slack'], n,
        ),
    'domain': np.random.choice(['tech', 'finance', 'medical', 'legal', 'edu'], n),
    'quality_tier': np.random.choice(['A+', 'A', 'B+', 'B', 'C'], n),
    'language': np.random.choice(['zh', 'en', 'mixed', 'code'], n),
    'response_len': np.random.randint(50, 3000, n),
})

original_size = sft_df.memory_usage(deep=True).sum() / 1024 / 1024
optimized_sft, opt_report = MemoryOptimizer.optimize(sft_df)
optimized_size = optimized_sft.memory_usage(deep=True).sum() / 1024 / 1024

print(f"=== 内存优化分析 ===")
print(f"原始大小:   {original_size:.2f} MB")
print(f"优化后大小: {optimized_size:.2f} MB")
print(f"节省:       {(1 - optimized_size/original_size)*100:.1f}%")

if len(opt_report) > 0:
    print(f"\n可优化列:")
    print(opt_report.to_string(index=False))
```
