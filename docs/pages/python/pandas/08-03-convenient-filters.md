---
title: isin() / between() 等便捷筛选方法
description: isin/between/contains/nunique/unique 的使用技巧、性能对比与最佳实践
---
# 便捷筛选工具箱

布尔索引和 `query()` 是通用的筛选框架，但 Pandas 还提供了一组"专用工具"——针对特定筛选模式优化的便捷方法。它们不是必须的（用布尔索引都能实现），但能让代码更简洁、意图更明确。

## isin()：批量等值判断

当你需要判断某列的值是否属于一个已知集合时，`isin()` 比写一堆 `==` 或 `|` 高效得多：

```python
import pandas as pd

df = pd.DataFrame({
    'model': ['GPT-4o', 'Claude', 'Llama', 'Gemini', 'Qwen', 'DeepSeek'],
    'tier': ['frontier', 'frontier', 'open', 'frontier', 'open', 'frontier'],
})

selected = df[df['model'].isin(['GPT-4o', 'Claude', 'Gemini'])]
not_open = df[~df['tier'].isin(['open'])]

print(selected)
print(not_open)
```

`~df['tier'].isin(['open'])` 的含义是"不属于 open 类别"。注意 `~` 是按位取反运算符——它把 True 变 False、False 变 True。这在排除特定值时极其常用。

## between()：范围判断的语法糖

```python
import pandas as pd
import numpy as np

np.random.seed(42)
n = 100_000
df = pd.DataFrame({
    'tokens': np.random.randint(10, 4000, n),
    'quality': np.round(np.random.uniform(1, 5, n), 1),
    'timestamp': pd.date_range('2025-01-01', periods=n, freq='min'),
})

in_range = df[df['tokens'].between(200, 1500)]
recent = df[df['timestamp'].between('2025-06-01', '2025-12-31')]

print(f"Token 在 [200, 1500]: {len(in_range):,} / {len(df):,}")
print(f"下半年数据: {len(recent):,} / {len(df):,}")
```

`between()` 默认包含两端点（即 `>=` 和 `<=`）。如果你需要开区间，可以设置 `inclusive='neither'`。它本质上就是 `(col >= low) & (col <= high)` 的语法糖，但可读性更好。

## contains()：文本模式匹配

这是 `.str` 访问器下最常用的方法之一，用于在字符串列中搜索子串或正则模式：

```python
import pandas as pd

df = pd.DataFrame({
    'prompt': [
        '什么是注意力机制',
        '如何安装PyTorch',
        '解释BERT模型',
        'GPT-4和Claude哪个好',
        'ERROR: connection timeout',
        '',
        '<html>页面未找到</html>',
    ],
})

has_error = df[df['prompt'].str.contains(r'ERROR|HTTP|<html>', regex=True, case=False)]
clean = df[
    df['prompt'].notna() &
    (df['prompt'].str.len() >= 3) &
    (~df['prompt'].str.contains(r'ERROR|<html', regex=True))
].copy()
print(f"异常数据 {len(has_error)} 条 | 清洗后 {len(clean)} 条")
```

注意 `case=False` 让匹配不区分大小写，`regex=True` 启用正则表达式支持。**`na=False` 参数很重要**——如果不加，遇到 NaN 值时 contains 返回 NaN 而非 False，导致后续布尔索引出错。

## unique() / nunique()：快速了解列的多样性

这两个方法虽然不直接用于筛选，但在决定筛选策略之前非常有用：

```python
for col in ['source', 'language', 'category']:
    n_unique = df[col].nunique()
    total = len(df)
    ratio = n_unique / total
    
    if ratio < 0.05:
        strategy = "→ 低基数，适合 category + isin"
    elif ratio > 0.8:
        strategy = "→ 高基数，适合 contains/between"
    else:
        strategy = "→ 中等基数"
    
    print(f"{col}: {n_unique:,}/{total:,} ({ratio:.1%}) {strategy}")
```

这个分析直接指导你选择哪种筛选方式：低基数列用 `isin()` 最高效，高基数列需要 `contains()` 或数值范围过滤。
