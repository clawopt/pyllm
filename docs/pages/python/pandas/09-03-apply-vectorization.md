---
title: apply() 与向量化操作
description: apply() 的三种调用模式、向量化替代方案的性能对比、transform() 与 agg() 的区别
---
# apply()：灵活但需要谨慎使用的瑞士军刀

`apply()` 是 Pandas 中最灵活的方法之一——它让你对 Series 或 DataFrame 的每个元素/每行/每列执行任意函数。但这种灵活性是有代价的：**apply() 本质上是一个 Python 级别的循环，比原生向量化操作慢几个数量级**。这一节我们讲清楚什么时候该用 apply、什么时候应该用向量化替代。

## 三种调用模式

### 模式一：Series.apply() —— 逐元素处理

```python
import pandas as pd

df = pd.DataFrame({
    'prompt': [
        '什么是Transformer',
        '如何训练LLM',
        'BERT和GPT区别',
    ],
})

df['char_count'] = df['prompt'].apply(len)
df['first_word'] = df['prompt'].apply(lambda x: x.split()[0])

print(df)
```

Series.apply() 对每个元素执行函数，返回等长的 Series。这是最简单也最常用的模式。

### 模式二：DataFrame.apply() —— 逐列或逐行

```python
import pandas as pd

df = pd.DataFrame({
    'accuracy': [0.92, 0.88, 0.95],
    'recall': [0.85, 0.90, 0.88],
    'f1': [0.88, 0.89, 0.91],
}, index=['GPT-4o', 'Claude', 'Llama'])

col_range = df.apply(lambda col: col.max() - col.min())
print("各指标极差:")
print(col_range)

df['avg'] = df.apply(lambda row: row.mean(), axis=1)
df['best'] = df.apply(lambda row: row.idxmax(), axis=1)
print("\n按模型汇总:")
print(df)
```

注意 `axis=1` 表示按行（对每一行的所有列做操作），默认 `axis=0` 是按列。这个参数是 `apply()` 最容易搞混的地方——记住 **axis=1 就是"横向看每一行"**。

### 模式三：元素级 map()

```python
grade_map = {'A': 4, 'B': 3, 'C': 2, 'D': 1}
s = pd.Series(['A', 'B', 'A', 'C'])
encoded = s.map(grade_map)
print(encoded)
```

`map()` 是 Series 专用的、专门用于"值到值映射"的方法。当你的逻辑就是简单的字典查找时，`map()` 比 `apply(lambda x: dict[x])` 更快也更清晰。

## 性能对比：为什么能不用 apply 就不用

```python
import pandas as pd
import numpy as np
import time

n = 1_000_000
s = pd.Series(np.random.randn(n))

start = time.time()
r_apply = s.apply(lambda x: x * 2 + 1)
t_apply = time.time() - start

start = time.time()
r_vec = s * 2 + 1
t_vec = time.time() - start

print(f"apply(): {t_apply:.4f}s")
print(f"向量化:   {t_vec:.4f}s")
print(f"加速:     {t_apply/t_vec:.0f}x")
```

典型结果：向量化比 apply 快 **50-200 倍**。原因在于 `apply()` 内部是 Python 循环（每次迭代都要经过 Python 解释器），而向量化操作直接在 C 层面的 NumPy 数组上执行（利用 CPU 的 SIMD 指令并行处理）。

所以规则很简单：**如果 Pandas 或 NumPy 有原生的向量化方法来实现同样的逻辑，就不要用 apply()**。常见的向量化替代包括：
- 数学运算 → 直接用 `+ - * /`
- 条件判断 → 用 `np.where()` 或 `np.select()`
- 字符串操作 → 用 `.str` 访问器
- 类型转换 → 用 `astype()` 或 `pd.to_numeric()`

只有当你的逻辑确实无法用内置方法表达时（比如调用外部 API、复杂的自定义业务逻辑），才使用 `apply()`。
