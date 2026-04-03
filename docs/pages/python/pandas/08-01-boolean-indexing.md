---
title: 布尔索引：最核心的筛选方式
description: 布尔索引的原理与用法、& | ~ 运算符、多条件组合、常见陷阱与 LLM 场景实战
---
# 布尔索引：Pandas 筛选的基石

如果你只学一种 Pandas 的数据筛选方法，那就学布尔索引。它不是最简洁的（`query()` 更简洁），也不是最快的（向量化方法有时更快），但它是最通用、最灵活、最能表达复杂筛选逻辑的方式——而且理解了它之后，其他所有筛选方法都是它的语法糖。

## 核心原理：用一列 True/False 来决定保留什么

布尔索引的本质非常简单：你构造一个和 DataFrame 同长的布尔 Series（每个位置是 True 或 False），然后用它去"切"原 DataFrame——True 对应的行被保留，False 对应的被丢弃。

```python
import pandas as pd

df = pd.DataFrame({
    'model': ['GPT-4o', 'Claude', 'Llama', 'Qwen', 'DeepSeek'],
    'score': [88.7, 89.2, 84.5, 83.5, 86.0],
    'price': [15, 15, 0, 0, 1],
})

mask = df['score'] >= 85
print(mask)
print("\n筛选结果:")
print(df[mask])
```

输出：

```
0     True
1     True
2    False
3    False
4     True
Name: score, dtype: bool

      model  score  price
0   GPT-4o   88.7     15
1    Claude   89.2     15
4  DeepSeek   86.0      1
```

`df['score'] >= 85` 这个表达式返回的是一个布尔 Series，把它放进 `df[...]` 里就完成了筛选。这个机制虽然简单，但它是所有复杂筛选逻辑的基本积木块。

## 多条件组合：& 和 | 的正确用法

单个条件的筛选在实际中很少用到。真实场景几乎总是多个条件的组合——比如"质量分大于等于 4 且 token 数在合理范围内且来源不是 export"。这时候就需要用逻辑运算符把多个布尔 Series 组合起来。

```python
import pandas as pd
import numpy as np

np.random.seed(42)
n = 100_000

df = pd.DataFrame({
    'quality': np.round(np.random.uniform(1, 5, n), 1),
    'tokens': np.random.randint(10, 4000, n),
    'source': np.random.choice(['api', 'web', 'export', 'mobile'], n),
    'has_code': np.random.choice([True, False], n),
})

result = df[
    (df['quality'] >= 4.0) &
    (df['tokens'].between(100, 2000)) &
    (df['source'].isin(['api', 'web'])) &
    (~df['has_code'])
]

print(f"原始: {len(df):,} → 筛选后: {len(result):,}")
print(f"\n质量分布:\n{result['quality'].describe().round(2)}")
```

这里有几个关键细节必须注意。

**第一，每个条件必须用括号包裹**。`&` 和 `|` 的优先级比比较运算符高，所以 `df['a'] > 3 & df['b'] < 5` 实际上会被解析为 `df['a'] > (3 & df['b']) < 5`——完全不是你想要的意思。**这是新手最容易犯也最难排查的错误之一**。

**第二，用 `&` 表示"与"、`|` 表示"或"、`~` 表示"非"**。不要用 Python 的 `and`/`or`/`not`——它们在 Pandas 布尔上下文中不工作，会抛出ambiguous 错误。

**第三，`isin()` 是批量等值判断的标准写法**。`df['source'] == 'api' | df['source'] == 'web'` 虽然能工作但很啰嗦；`df['source'].isin(['api', 'web'])` 更清晰也更高效。

## 常见陷阱：链式赋值与 SettingWithCopyWarning

这是一个几乎所有 Pandas 用户都踩过的坑：

```python
import pandas as pd

df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})

subset = df[df['a'] > 1]
subset['c'] = [99, 98]
```

运行后会看到 `SettingWithCopyWarning`。原因是 `df[df['a'] > 1]` 返回的可能是一个视图（view）也可能是一个副本（copy）——Pandas 无法确定。如果它是一个视图，你对它赋值会修改原 DataFrame；如果是副本，修改不会反映到原 DataFrame 上。这种不确定性就是 warning 存在的原因。

解决方式有两种。第一种是显式复制：

```python
subset = df[df['a'] > 1].copy()
subset['c'] = [99, 98]
```

`.copy()` 创建一个独立的副本，后续操作不会影响原 DataFrame。第二种是用 `loc` 直接在原 DataFrame 上操作：

```python
df.loc[df['a'] > 1, 'c'] = [99, 98]
```

`loc` 总是在原 DataFrame 上执行操作，不存在视图/副本的歧义问题。**我的建议是：当你需要筛选后修改数据时，始终使用 `loc` 或先 `.copy()`**。
