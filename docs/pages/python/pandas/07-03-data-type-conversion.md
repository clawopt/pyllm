---
title: 数据类型转换
description: astype() / pd.to_datetime() / pd.to_numeric() 的使用，大模型场景下的类型修正与优化策略
---
# 类型转换：让每列都找到最合适的家

数据被加载到 Pandas 之后，它的 dtype 不一定是最优的——甚至不一定是正确的。`read_csv()` 会自动推断类型，但推断结果经常出错：混了一个字符串的数值列会被当成 `object`，有缺失值的整数列会变成 `float64`，日期字符串不会被自动解析为 datetime。这一节讲如何系统地修正和优化数据类型。

## astype()：基础但不够用

```python
import pandas as pd

df = pd.DataFrame({
    'a': [1, 2, 3],
    'b': [1.5, 2.7, 3.9],
    'c': [100000, 200000, 300000],
})

df['a_i8'] = df['a'].astype('int8')
df['c_i32'] = df['c'].astype('int32')
df['b_f32'] = df['b'].astype('float32')

print(df.dtypes)
```

`astype()` 是最直接的类型转换方法，但它有一个硬性限制：**源数据必须严格兼容目标类型**。如果 `b` 列里混了一个 `'bad'` 字符串，`astype(float)` 会直接抛 ValueError。

对于"可能有脏数据"的场景，你需要更宽容的工具。

## to_numeric() 和 to_datetime()：带容错的转换

```python
import pandas as pd

df = pd.DataFrame({
    'mixed': ['1', '2.5', '3', 'bad', '5'],
    'dates': ['2025-01-15', '2025-03-20', 'not-a-date', '2025-06-01'],
})

df['num_clean'] = pd.to_numeric(df['mixed'], errors='coerce')
df['date_clean'] = pd.to_datetime(df['dates'], errors='coerce')

print(df)
```

输出：

```
  mixed     dates  num_clean date_clean
0     1  2025-01-15        1.0 2025-01-15
1   2.5  2025-03-20        2.5 2025-03-20
2     3  not-a-date        3.0        NaT
3    bad  2025-06-01        NaN 2025-06-01
4     5  2025-06-01        5.0 2025-06-01
```

`errors='coerce'` 的含义是：遇到无法转换的值不报错，而是转为 NaN（数值）或 NaT（时间）。这在清洗脏数据时极其常用——先暴力转一遍，然后通过检查 NaN 来定位脏数据的分布位置。

## downcast：让 Pandas 帮你选最小够用的类型

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'id': range(1000),
    'score': np.random.uniform(0, 100, 1000),
    'count': np.random.randint(0, 500, 1000),
})

before = df.memory_usage(deep=True).sum()

df['id_opt'] = pd.to_numeric(df['id'], downcast='integer')
df['score_opt'] = pd.to_numeric(df['score'], downcast='float')
df['count_opt'] = pd.to_numeric(df['count'], downcast='integer')

after = df[['id_opt', 'score_opt', 'count_opt']].memory_usage(deep=True).sum()

print(f"转换前: {before/1024:.0f} KB")
print(f"转换后: {after/1024:.0f} KB ({(1-after/before)*100:.0f}% 节省)")
print(f"\n优化后 dtypes:")
print(df[['id_opt', 'score_opt', 'count_opt']].dtypes)
```

`downcast='integer'` 让 Pandas 自动选择能容纳当前数据的最小整数类型（int8/int16/int32/int64），`downcast='float'` 同理选择最小的浮点类型。这是我在生产环境中最常用的类型优化方式之一——一行代码，不需要手动计算取值范围。

## 分类转换：低基数列的内存神器

如果一列的取值种类远少于总行数（比如 source 列只有 "api"/"web"/"export" 三种值但有 100 万行），转成 category 类型可以大幅节省内存：

```python
import pandas as pd
import numpy as np

n = 500_000
df = pd.DataFrame({
    'label': np.random.choice(['pos', 'neg', 'neutral', 'code', 'other'], n),
    'source': np.random.choice(['api', 'web', 'export', 'mobile'], n),
    'model': np.random.choice(['GPT-4o', 'Claude', 'Llama', 'Gemini'], n),
})

for col in ['label', 'source', 'model']:
    before = df[col].memory_usage(deep=True).sum()
    df[col] = df[col].astype('category')
    after = df[col].memory_usage(deep=True).sum()
    print(f"{col}: {before/1024:.1f} KB → {after/1024:.1f} KB ({(1-after/before)*100:.0f}% 节省)")
```

但记住我们在 04-04 节强调过的规则：**只对低基数列使用 category**（唯一值数量 / 总行数 < 5%）。高基数列（如自由文本、用户 ID）转 category 反而会增加内存开销。
