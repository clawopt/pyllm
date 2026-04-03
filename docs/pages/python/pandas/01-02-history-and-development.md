---
title: Pandas 的历史与发展
description: 从 Wes McKinney 的量化金融需求到 3.0 时代的 Copy-on-Write 与 PyArrow 后端
---
# Pandas 的历史与发展


## 创始故事：一个量化分析师的烦恼

2008 年，Wes McKinney 在 AQR Capital（一家总部位于康涅狄格州的量化对冲基金）工作。他每天面对的核心问题是：

> **如何高效地处理金融时间序列数据？**

当时的 Python 数据生态远没有今天这样丰富。NumPy 虽然强大，但它只提供同构的多维数组——**没有行标签、没有列名、没有缺失值处理、没有时间索引**。对于需要按日期对齐、处理不同频率数据的金融分析来说，NumPy 就像一把只有刀刃的刀——锋利但不好用。

Wes McKinney 需要的是一个工具，能够像 R 语言中的 `data.frame` 一样方便地操作带标签的表格数据，同时又能享受 NumPy 的高性能计算能力。于是，Pandas 诞生了。

### 开源与社区成长

2009 年，Pandas 作为开源项目首次发布。随后的发展可以用几个关键节点来标记：

| 时间 | 里程碑 | 意义 |
|------|--------|------|
| **2008** | Wes McKinney 开始开发 | 诞生于量化金融的实际需求 |
| **2009** | 首次开源发布 | 进入公众视野 |
| **2012** | 发布《Python for Data Analysis》一书 | 成为该领域的权威教材 |
| **2015** | Pandas 0.16 引入 Categorical 类型 | 内存优化的重要一步 |
| **2018** | NumPy 1.16+ 支持 ExtensionArray | 为 Nullable 类型铺路 |
| **2020** | Pandas 1.0 正式发布 | 引入 `StringDtype`、`NA` 标量等 |
| **2022** | Pandas 1.5 引入 PyArrow 后端（实验性） | 列式存储的初步支持 |
| **2023** | Pandas 2.0 正式发布 | PyArrow 后端转正，Copy-on-Write（CoW）预览 |
| **2025** | Pandas 3.0 正式发布 | CoW 默认启用，PyArrow 字符串成为默认 |

## 版本演进：三个时代的分水岭

### 1.x 时代：经典 Pandas

这是大多数人认识 Pandas 的版本。核心特征：

- **NumPy 作为唯一后端**：所有数据底层都是 NumPy 数组
- **行为怪癖多**：SettingWithCopyWarning 是每个 Pandas 新手的噩梦
- **缺失值用 NaN/None 混合表示**：整数列一旦有缺失值就变成浮点数
- **字符串是 object 类型**：本质上是 Python 对象指针数组，性能差

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({'age': [25, 30, None, 40]})
print(df.dtypes)
print(df['age'][2])
```

这些设计决策在 Pandas 早期是合理的妥协——为了性能和兼容性。但随着数据规模的增长和用户群体的扩大，它们逐渐成为了瓶颈。

### 2.0 时代：架构升级

Pandas 2.0 是一次**不向后兼容的重大升级**，主要变化：

#### PyArrow 后端

```python
import pandas as pd

pd.options.mode.dtype_backend = 'pyarrow'

df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'score': [95.5, 87.2, 91.8]
})

print(df.dtypes)
```

**为什么 PyArrow 重要？**

Apache Arrow 是一种**列式内存格式**，由 Pandas 的创始人 Wes McKinney 后来主导开发。它的核心优势：

| 特性 | NumPy 后端 | PyArrow 后端 |
|------|-----------|-------------|
| 内存布局 | 行优先（C-order） | 列式存储 |
| 字符串存储 | object 数组（指针） | 紧凑的变长字符串 |
| 缺失值表示 | NaN（需提升为 float） | 原生位掩码（bitmask）null |
| 跨语言互操作性 | 仅 Python | C++ / Java / Rust / Go 共享零拷贝 |
| I/O 性能 | 标准 | Parquet / Feather 原生加速 |

对于大模型场景来说，PyArrow 后端的**字符串处理能力**尤为关键——LLM 语料本质上就是大量文本数据。

#### Copy-on-Write（CoW）预览

2.0 引入了 CoW 作为可选功能：

```python
pd.options.mode.copy_on_write = True

df = pd.DataFrame({'a': [1, 2, 3]})
df2 = df['a']  # 不再触发 SettingWithCopyWarning
df2.iloc[0] = 99
```

### 3.0 时代：新默认行为

Pandas 3.0 将以下特性设为**默认开启**：

- **Copy-on-Write 默认启用**：彻底告别 SettingWithCopyWarning
- **PyArrow 字符串作为默认 dtype**：`pd.StringDtype("pyarrow")` 取代 `object`
- **新的警告策略**：FutureWarning 更加明确地提示即将废弃的行为

```python
import pandas as pd

df = pd.DataFrame({'text': ['hello world', 'foo bar', None]})
print(df.dtypes)
```

## 大模型开发者为什么要关注版本差异

你可能觉得："我只要能读写 CSV 就行了，管它什么版本？"

但在大模型数据处理中，版本选择直接影响：

### 1. 内存占用

处理千万级 token 的语料时，字符串列的存储方式决定了一切：

```python
import pandas as pd
import numpy as np

texts = np.array(['这是一段关于大模型的对话内容...' * 10] * 1_000_000)

df_old = pd.DataFrame({'text': texts})
print(f"object 后端: {df_old.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

pd.options.mode.dtype_backend = 'pyarrow'
df_new = pd.DataFrame({'text': texts})
print(f"PyArrow 后端: {df_new.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
```

实测中，PyArrow 后端的内存占用通常比 object 后端**低 30%-60%**，取决于文本的去重程度。

### 2. 缺失值语义

大模型语料中缺失值很常见（空响应、被过滤的内容等）。3.0 的 Nullable 类型让你不再意外丢失整型信息：

```python
df = pd.DataFrame({
    'token_count': pd.array([256, None, 512, 128], dtype='Int64'),
    'label': pd.array(['pos', 'neg', None, 'pos'], dtype='string')
})
```

### 3. 写时复制避免隐蔽 Bug

在构建数据处理流水线时，CoW 能防止你无意间修改原始数据：

```python
def clean_text(df):
    subset = df[df['quality'] > 3].copy()  # 3.0 中 .copy() 不再必须
    subset['text'] = subset['text'].str.lower()
    return subset

original = pd.DataFrame({'text': ['Hello', 'World'], 'quality': [4, 2]})
cleaned = clean_text(original)

```

## 安装与版本检查

```python
import pandas as pd

print(pd.__version__)

print(pd.options.mode.copy_on_write)       # True in 3.0+
print(pd.options.dtype_backend)            # 可能是 None 或 'pyarrow'
```

安装命令：

```bash
pip install "pandas[all]"

pip install "pandas>=3.0"

pip install pyarrow
```

## 小结

Pandas 从一个量化分析师的个人工具，发展为全球超过数百万开发者使用的核心库，其演进路线清晰地反映了 Python 数据生态的发展方向：

- **1.x** 解决"有没有"的问题——让 Python 能做数据分析
- **2.x** 解决"快不快"的问题——引入 Arrow 后端和 CoW 架构
- **3.x** 解决"好不好"的问题——默认行为更加合理，减少陷阱

对于大模型开发者而言，理解这个演进过程的意义在于：**你正在使用的每一个 Pandas 操作，背后都有其历史原因和设计权衡**。知道"为什么"，才能在遇到问题时快速定位根因。
