---
title: 计算性能优化
description: 向量化替代循环 / eval() / numexpr 加速 / 并行处理 / Cython/Numba 入门
---
# 计算性能优化


## 向量化 vs 循环：决策树

```
需要逐行操作？
  ├─ 是 → 能用向量化函数吗？
  │        ├─ 是 → 用 .str / np.where / pd.case_when()
  │        └─ 否 → itertuples() (比 iterrows 快 10-50x)
  └─ 否 → 直接向量化（最快）
```

### 常见反模式与修正

```python
import pandas as pd
import numpy as np

n = 100_000
df = pd.DataFrame({
    'a': np.random.randn(n),
    'b': np.random.randn(n),
    'text': ['hello world'] * n,
})

def slow_conditional(df):
    result = []
    for _, row in df.iterrows():
        if row['a'] > 0 and row['b'] > 0:
            result.append('both_pos')
        elif row['a'] > 0:
            result.append('a_pos')
        elif row['b'] > 0:
            result.append('b_pos')
        else:
            result.append('both_neg')
    return result

def fast_conditional(df):
    return np.select(
        [
            (df['a'] > 0) & (df['b'] > 0),
            (df['a'] > 0),
            (df['b'] > 0),
        ],
        ['both_pos', 'a_pos', 'b_pos'],
        default='both_neg'
    )
```

## eval(): 表达式求值加速

```python
import pandas as pd
import numpy as np

n = 1_000_000
df = pd.DataFrame({
    'a': np.random.randn(n),
    'b': np.random.randn(n),
    'c': np.random.randn(n),
})

result = df.eval('''
    ab_sum = a + b
    bc_product = b * c
    complex_expr = (a**2 + b**2) ** 0.5
''')

print(result.head())
```

### eval() 的赋值模式

```python
df.eval('d = a + b', inplace=True)

df.eval("""
    e = a - c
    f = b * 2
""", inplace=True)
```

## numexpr 加速数值运算

```python
import pandas as pd
import numpy as np

pd.set_option('mode.use_inf_as_na', True)

n = 5_000_000
df = pd.DataFrame({
    'x': np.random.randn(n),
    'y': np.random.randn(n),
    'z': np.random.randn(n),
})

complex_result = df.eval('(x > 0) & (y < 0) & (abs(z) > 1)')
print(f"结果: {complex_result.sum()} 行满足条件")
```

## query() vs 布尔索引性能

```python
import pandas as pd
import numpy as np
import time

n = 2_000_000
df = pd.DataFrame({
    'model': np.random.choice(['GPT-4o', 'Claude', 'Llama', 'Qwen'], n),
    'score': np.random.uniform(60, 98, n),
    'latency': np.random.randint(100, 5000, n),
    'price': np.random.uniform(0.14, 15.0, n),
})

start = time.time()
r1 = df[
    (df['model'].isin(['GPT-4o', 'Claude'])) &
    (df['score'] >= 85) &
    (df['latency'] < 1500) &
    (df['price'] <= 3.0)
]
t1 = time.time() - start

start = time.time()
r2 = df.query(
    "model in ['GPT-4o', 'Claude'] and score >= 85 "
    "and latency < 1500 and price <= 3.0"
)
t2 = time.time() - start

assert len(r1) == len(r2)

print(f"布尔索引: {t1:.4f}s ({len(r1)} 行)")
print(f"query():  {t2:.4f}s ({len(r2)} 行)")
print(f"query 相对速度: {t1/t2:.2f}x")
```

## 并行处理：swifter / multiprocessing

```python
import pandas as pd
import numpy as np

try:
    import swifter

    n = 100_000
    df = pd.DataFrame({'text': [f'测试文本{i}' * 10 for i in range(n)]})

    def complex_func(text):
        return len(text.strip()) * 2 + text.count('测')

    result = df['text'].swifter.apply(complex_func)
    print(f"swifter 处理完成: {len(result)} 条")

except ImportError:
    print("提示: pip install swifter 可启用自动并行加速")
```
