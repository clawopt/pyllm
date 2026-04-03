---
title: query() 方法：简洁表达式筛选
description: query() 的语法详解、变量引用、与布尔索引的对比、性能考量与大模型场景示例
---
# query()：让筛选条件像 SQL 一样可读

当你的布尔索引条件超过三四个时，代码就会变成一坨括号和 `&` 符号的混合体——很难读也很难维护。`query()` 就是为此而生的：它让你用接近自然语言或 SQL 的语法来写筛选条件，不需要反复写 `df['col']`，也不需要担心运算符优先级。

## 从布尔索引到 query：一个直观的对比

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

result_bool = df[
    (df['score'] > 88) &
    (df['price'] < 5.0) &
    (df['context'] > 500000)
]

result_query = df.query("score > 88 and price < 5.0 and context > 500000")

print(f"两种方式结果一致: {len(result_bool) == len(result_query)}")
```

同样的逻辑，query 版本明显更短更清晰。核心差异在于：
- 不需要写 `df['col']`，直接用列名
- 用 `and`/`or`/`not` 而非 `&`/`|`/`~`
- 不需要为每个子条件加括号（优先级符合直觉）

## 变量引用：用 @ 符号传入外部值

实际工作中筛选条件的阈值通常来自外部变量（比如用户输入或配置文件）。在 query 中引用外部变量需要加 `@` 前缀：

```python
min_score = 90.0
max_price = 3.0
allowed_models = ['GPT-4o', 'Claude']

result = df.query(
    "score >= @min_score and "
    "price <= @max_price and "
    "model in @allowed_models"
)

print(f"匹配 {len(result)} 条: 高分 + 便宜 + 指定模型")
```

`@variable_name` 是 query 引用 Python 外部变量的唯一方式。不加 `@` 的话 Pandas 会把它当作列名去查找，找不到就报错。这个设计虽然多了一个字符但非常明确——一眼就能区分"这是列名还是外部变量"。

## 性能对比与选择建议

```python
import time

N = 2_000_000
big_df = pd.DataFrame({
    'a': np.random.randn(N),
    'b': np.random.randn(N),
    'c': np.random.choice(['x','y','z'], N),
})

start = time.time()
r1 = big_df[(big_df['a'] > 0.5) & (big_df['b'] < -0.5)]
t_bool = time.time() - start

start = time.time()
r2 = big_df.query("a > 0.5 and b < -0.5")
t_query = time.time() - start

print(f"布尔索引: {t_bool:.4f}s | query: {t_query:.4f}s | 比例: {t_query/t_bool:.2f}x")
```

对于简单条件两者性能几乎相同（query 内部会编译成和布尔索引等价的 numexpr 表达式）。但对于特别复杂的条件组合，query 有时会更快一些，因为它使用了 numexpr 库做表达式优化。不过这个差异在实际中通常可以忽略——**选择的主要依据应该是可读性而非性能**。
