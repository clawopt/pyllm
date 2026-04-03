---
title: transform() 与 filter()：保持形状与按组过滤
description: transform 广播回原形状、filter 按条件丢弃整组、分组 Z-score 标准化、组内异常值检测
---
# transform 和 filter：agg 之外的两种分组操作

`agg()` 让每组返回一个标量值。但有时候你需要的是：**每组返回一个和原数据等长的序列**（transform），或者**根据组的某种属性决定是否保留整组**（filter）。这两种操作在 LLM 数据处理中非常实用。

## transform()：计算后广播回原形状

```python
import pandas as pd
import numpy as np

np.random.seed(42)
df = pd.DataFrame({
    'model': ['GPT-4o']*10 + ['Claude']*10 + ['Llama']*10,
    'latency_ms': np.concatenate([
        np.random.normal(800, 150, 10),
        np.random.normal(650, 120, 10),
        np.random.normal(350, 80, 10),
    ]).astype(int),
})

df['group_mean'] = df.groupby('model')['latency_ms'].transform('mean')
df['z_score'] = df.groupby('model')['latency_ms'].transform(
    lambda x: (x - x.mean()) / x.std()
)

print(df[['model', 'latency_ms', 'group_mean', 'z_score']].head(9).round(2))
```

核心区别：
- **`agg()`** → 每组返回 1 个值 → 结果行数 = 组数
- **`transform()`** → 每组返回等长序列 → 结果行数 = 原数据行数

这意味着 `transform()` 的结果可以直接赋回原 DataFrame 作为新列——这是做**组内标准化**（Z-score）、**组内排名**（percentile rank）、**组内缺失值填充**的标准方式。

## filter()：按条件保留或丢弃整组

```python
large_groups = df.groupby('model').filter(lambda g: len(g) >= 5)
print(f"样本数 >= 5 的组: {large_groups['model'].unique()}")
```

`filter()` 的判断对象是**整个组**，不是单行。上面的例子保留了所有样本数 >= 5 的模型组。这在数据清洗中常用于过滤掉样本量太少的类别——比如某个来源只有 3 条数据的组可能没有统计意义，直接丢弃。

## 实战：组内异常值检测

```python
def detect_group_outliers(df, group_col, value_col, threshold=3):
    df = df.copy()
    df['group_z'] = df.groupby(group_col)[value_col].transform(
        lambda x: (x - x.mean()) / x.std()
    )
    df['is_outlier'] = df['group_z'].abs() > threshold
    return df

result = detect_group_outliers(df, 'model', 'latency_ms', threshold=2)
outliers = result[result['is_outlier']]
print(f"检测到 {len(outliers)} 个组内异常值")
print(outliers[['model', 'latency_ms', 'group_z']].round(2))
```

这个函数先用 `transform` 计算每个值相对于其所在组的 Z-score，然后标记绝对值超过阈值的点为异常值。注意这是**组内**相对异常——一个 1200ms 的延迟在 GPT-4o 组里可能是正常的（因为 GPT-4o 本身就慢），但在 Llama 组里就是极端异常。
