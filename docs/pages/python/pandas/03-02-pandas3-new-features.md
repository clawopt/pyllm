---
title: Pandas 3.0 核心新特性一览
description: Copy-on-Write 行为变更、PyArrow 后端支持、新字符串数据类型的完整解析
---
# Pandas 3.0 新特性详解


## Pandas 3.0 的三大变革

Pandas 3.0 不是小版本迭代，而是一次**行为层面的重新定义**。三个核心变化直接影响你每天写的每一行代码：

```
┌─────────────────────────────────────────────────────┐
│              Pandas 3.0 三大变革                      │
│                                                     │
│   ┌─────────────┐  ┌──────────────┐  ┌───────────┐ │
│   │ Copy-on-Write│  │ PyArrow 后端  │  │ 新字符串类型│ │
│   │             │  │              │  │           │ │
│   │ 默认启用     │  │ 字符串默认    │  │ NA 统一   │ │
│   │ 告别 Setting │  │ 列式存储     │  │ 性能提升   │ │
│   │ WithCopy     │  │ 内存优化     │  │ 语义清晰   │ │
│   └─────────────┘  └──────────────┘  └───────────┘ │
└─────────────────────────────────────────────────────┘
```

## 变革一：Copy-on-Write（CoW）默认启用

### 它解决什么问题

Pandas 最臭名昭著的警告——`SettingWithCopyWarning`——终于要成为历史了。

```python
import pandas as pd

df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
subset = df[df['a'] > 1]  # 返回一个"视图"还是"副本"？不确定！
subset['c'] = 99

```

这个问题的根源在于 Pandas 的**索引返回语义不明确**——有时返回视图，有时返回副本，取决于操作链的上下文。这导致了无数隐蔽 Bug。

### CoW 的解决方案

```python
import pandas as pd

pd.options.mode.copy_on_write = True  # 3.0 中已默认为 True

df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
subset = df[df['a'] > 1]  # 返回一个延迟拷贝
subset['c'] = 99           # 只修改 subset，不影响 df！

print(df)

print(subset)
```

### CoW 的底层原理

**Copy-on-Write** 意味着：当你"复制"一个对象时，并不真正复制数据，而是共享同一块内存。只有当任一一方尝试**写入（修改）**时，才真正执行拷贝。

```python
import pandas as pd
import numpy as np

pd.options.mode.copy_on_write = True

df = pd.DataFrame({'data': np.array([1, 2, 3, 4, 5])})

view = df['data']        # 共享内存，零开销
print(np.shares_memory(df['data'].values, view.values))

view.iloc[0] = 999       # 写入触发真正的拷贝
print(np.shares_memory(df['data'].values, view.values))
```

### 对现有代码的影响

| 旧代码模式 | 3.0 下是否需要改 | 说明 |
|-----------|:-:|------|
| `df[col] = value` | ❌ 不需要 | 直接赋值始终安全 |
| `df.loc[row, col] = value` | ❌ 不需要 | 显式 `.loc` 赋值始终安全 |
| `subset = df[condition]; subset[x] = y` | ⚠️ **可能需要** | 如果意图是修改原 df，需加 `.copy()` |
| `df['new'] = df['old'].str.lower()` | ❌ 不需要 | 向量化操作创建新列，不受影响 |
| 链式索引 `df['a'][mask] = x` | ⚠️ **必须改** | 本就是反模式，现在会明确报错 |

### 迁移指南

```python
df.loc[df['a'] > 1, 'c'] = 99

subset = df[df['a'] > 1].copy()
subset['c'] = 99

with pd.option_context('mode.copy_on_write', False):
    df = pd.DataFrame({'a': [1, 2, 3]})
    subset = df[df['a'] > 1]
    subset['a'] = 99
    print(df)  # 在此上下文中，df 会被修改
```

### CoW 的性能影响

你可能担心"每次写都拷贝"会很慢。实际上：

```python
import pandas as pd
import numpy as np
import time

pd.options.mode.copy_on_write = True
n = 10_000_000
df = pd.DataFrame({
    'a': np.random.randn(n),
    'b': np.random.randint(0, 100, n),
})

start = time.time()
for i in range(1000):
    df.loc[i, 'a'] = i * 0.01
t_cow = time.time() - start

pd.options.mode.copy_on_write = False
df2 = df.copy()
start = time.time()
for i in range(1000):
    df2.loc[i, 'a'] = i * 0.01
t_no_cow = time.time() - start

print(f"CoW 开启: {t_cow:.4f}s")
print(f"CoW 关闭: {t_no_cow:.4f}s")
print(f"性能差异: {t_cow/t_no_cow:.2f}x")
```

## 变革二：PyArrow 后端成为字符串默认

### 从 object 到 string[pyarrow]

这是对大模型开发者**最有实际价值**的变化：

```python
import pandas as pd

df = pd.DataFrame({
    'text': ['hello world', 'foo bar baz', None, '中文文本测试'],
    'category': ['A', 'B', 'A', 'C']
})

print(df.dtypes)
```

### 为什么这对 LLM 数据处理至关重要

大模型语料本质上就是**海量文本数据**。传统 `object` 类型的字符串存储方式有严重问题：

```python
import pandas as pd
import numpy as np

n = 1_000_000
texts = ['这是一段关于大模型的对话内容，包含技术术语和业务逻辑。' * 5] * n

df_obj = pd.DataFrame({'text': texts})
mem_obj = df_obj.memory_usage(deep=True).sum()

pd.options.dtype_backend = 'pyarrow'
df_arrow = pd.DataFrame({'text': texts})
mem_arrow = df_arrow.memory_usage(deep=True).sum()

print(f"object 后端:  {mem_obj / 1024**2:.1f} MB")
print(f"PyArrow 后端: {mem_arrow / 1024**2:.1f} MB")
print(f"节省: {(1 - mem_arrow/mem_obj)*100:.1f}%")

```

**60%+ 的内存节省**——这在处理千万级 token 语料时意味着能否在单机上完成处理的关键区别。

### PyArrow 字符串的优势

| 特性 | object (NumPy) | string[pyarrow] |
|------|-----------------|-----------------|
| 存储方式 | Python 对象指针数组 | 紧凑的变长字节数组 |
| 缺失值表示 | `None`/`NaN`（混合） | 原生 `<NA>` 位掩码 |
| Null 安全 | 整数列遇 None 自动变 float | 保持原始类型不变 |
| 跨语言互操作 | 仅 Python | C++/Java/Rust/R 共享零拷贝 |
| Parquet I/O | 需转换 | **原生格式，直接读写** |
| 字符串操作性能 | 较慢 | 更快（向量化） |

### 全局配置 vs 单列控制

```python
import pandas as pd

pd.options.dtype_backend = 'pyarrow'

df = pd.read_csv('large_corpus.csv', dtype_backend='pyarrow')

df = pd.DataFrame({'text': ['hello', 'world'], 'num': [1, 2]})
df['text'] = df['text'].astype('string[pyarrow]')

df = pd.DataFrame(
    {'text': pd.array(['a', 'b', None], dtype='string[pyarrow]')}
)
```

## 变革三：统一的缺失值表示

### 旧的混乱局面

```python
import pandas as pd
import numpy as np

s_int = pd.Series([1, None, 3])      # → float64（整型被破坏）
s_float = pd.Series([1.0, None, 3.0])  # → float64
s_bool = pd.Series([True, None, False]) # → object（布尔型被破坏）
s_str = pd.Series(['a', None, 'b'])    # → object（混杂 None 和 NaN）

print(s_int)
```

### 3.0 的统一方案

```python
pd.options.dtype_backend = 'pyarrow'

df = pd.DataFrame({
    'int_col': [1, None, 3],
    'float_col': [1.5, None, 3.5],
    'bool_col': [True, None, False],
    'str_col': ['hello', None, 'world'],
})

print(df.dtypes)

print(df)

print(type(df['int_col'][1]))  # <class 'pandas.core.arrays.masked._Missing'>
```

### `<NA>` vs `NaN` vs `None`

| 表示 | 出现位置 | 类型保持 | 可参与运算 |
|------|---------|---------|-----------|
| `np.nan` | float 列 | 破坏整型 | 传播 NaN |
| `None` | object 列 | 破坏类型 | 抛错或传播 |
| `pd.NA` / `<NA>` | Nullable 列 | **保持原类型** | 传播 NA |

```python
import pandas as pd

s = pd.Series([1, pd.NA, 3], dtype='Int64')
print(s.mean())  # 2.0（自动忽略 NA，与 R 语言行为一致）

print(s + 10)
```

## 其他值得注意的变化

### `copy(deep=False)` 行为变化

```python
df = pd.DataFrame({'a': [1, 2, [3, 4]]})
shallow = df.copy(deep=False)
shallow.loc[0, 'a'][0] = 999
```

### `NDFrame.append()` 已移除

```python
df1 = pd.DataFrame({'a': [1, 2]})
df2 = pd.DataFrame({'a': [3, 4]})
result = df1.append(df2)  # AttributeError!

result = pd.concat([df1, df2], ignore_index=True)
```

### 新增便捷方法

```python
import pandas as pd

df = pd.DataFrame({'score': [85, 42, 67, 91, 55]})
df['grade'] = pd.case_when(
    (df['score'] >= 90, 'A'),
    (df['score'] >= 80, 'B'),
    (df['score'] >= 60, 'C'),
    default='D'
)
print(df)
```

## 升级检查清单

```python
def migration_checklist():
    import pandas as pd
    checks = []
    
    checks.append(('版本 >= 3.0', pd.__version__[0] == '3'))
    checks.append(('CoW 启用', pd.options.mode.copy_on_write))
    checks.append(('PyArrow 可用', hasattr(pd, 'ArrowDtype')))
    checks.append(('Int64Dtype 可用', hasattr(pd, 'Int64Dtype')))
    checks.append(('case_when 可用', hasattr(pd, 'case_when')))
    
    print("Pandas 3.0 迁移检查:")
    for name, passed in checks:
        icon = "✅" if passed else "❌"
        print(f"  {icon} {name}")

migration_checklist()
```
