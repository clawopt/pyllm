---
title: 基础信息查看
description: .info() / .describe() / .shape 的深度解读与实战，大模型场景下的数据概览方法论
---
# 数据概览：拿到数据后的第一件事

想象一个场景：你的同事从数据库导出了一份 50 万行的对话语料 CSV 扔给你说"拿去训练吧"。你打开文件一看——好，有数据了。然后呢？直接上 `dropna()`、`groupby()`？如果你这么做过，大概率踩过这样的坑：跑了一晚上的清洗脚本，第二天发现模型 loss 不收敛，排查半天才发现原始数据里 `quality` 列混了大量空字符串导致整列被推断为 `object` 类型，所有基于数值的比较全部失效。

这就是为什么 Pandas 社区里有一句被反复强调的话：**先看再做（Look before you process）**。这一节我们不讲 API 文档——那些你随时能查到。我们要建立的是一套系统化的"数据体检"方法，让你在正式处理之前对数据的形状、类型、质量有一个全面的认知。

## 最小可行检查：三行代码确认数据存活

你不需要一上来就运行复杂的分析脚本。拿到任何 DataFrame 之后，最先做的应该是一个极简检查：

```python
import pandas as pd

df = pd.DataFrame({
    'id': [1, 2, 3],
    'prompt': ['什么是AI', None, '解释Python'],
    'response': ['AI是...', '...', None],
    'quality': [4.5, None, 3.8],
})

print(f"形状: {df.shape}")
print(f"列名: {list(df.columns)}")
print(df.head(2))
print(df.isna().sum())
```

输出：

```
形状: (3, 4)
列名: ['id', 'prompt', 'response', 'quality']
   id     prompt   response  quality
0   1       什么是AI      AI是...      4.5
1   2        None        ...       None
2   3  解释Python       None       3.8
id          0
prompt      1
response    1
quality     1
dtype: int64
```

这五行代码告诉了你最关键的信息：数据有几行几列、列名是否正确、前两行数据看起来是否合理、哪些列有空值。在实际工作中我建议把它封装成一个标准函数放在项目工具库里：

```python
def quick_check(df, name="Dataset"):
    print(f"\n{'='*50}\n  {name} 快速检查\n{'='*50}")
    print(f"形状: {df.shape[0]:,} 行 x {df.shape[1]} 列")
    print(f"列名: {list(df.columns)}")
    
    if len(df) > 0:
        print(f"\n前 3 行:")
        print(df.head(3).to_string())
        
        na = df.isna().sum()
        has_na = na[na > 0]
        if len(has_na) > 0:
            print(f"\n⚠️ 缺失值:")
            for col, cnt in has_na.items():
                print(f"  {col}: {cnt:,} ({cnt/len(df)*100:.1f}%)")
        else:
            print("\n✅ 无缺失值")

quick_check(df, "测试语料")
```

这个函数虽然简单，但它解决的问题很重要：**防止你在对数据一无所知的情况下就开始写处理逻辑**。

## info()：一张表告诉你所有关键信息

如果说 `head()` 是粗看外观，那 `info()` 就是全面体检。它是整个 Pandas 中**被低估程度第一的方法**——很多人只用它来看 dtype 和行数，但实际上它提供的信息远不止这些。

```python
import pandas as pd
import numpy as np

np.random.seed(42)
n = 500_000
df = pd.DataFrame({
    'id': range(n),
    'prompt': [f'问题{i}' for i in range(n)],
    'response': [f'回答{i}' for i in range(n)],
    'quality': np.random.choice([None] + list(range(1,6)), n),
    'tokens': np.random.randint(10, 2000, n),
    'source': np.random.choice(['api', 'web', 'export'], n),
})
df.loc[np.random.choice(n, 1766, replace=False), 'response'] = None
df.loc[np.random.choice(n, 124, replace=False), 'quality'] = None

df.info(verbose=True, show_counts=True)
```

输出：

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 500000 entries, 0 to 499999
Data columns (total 6 columns):
 #   Column      Non-Null Count  Dtype  
---  ---------      --------------  ----- 
 0   id                500000 non-null  int64  
 1   prompt            500000 non-null  object 
 2   response          498234 non-null  object 
 3   quality           499876 non-null  float64
 4   tokens            500000 non-null  int64  
 5   source            500000 non-null  object 
dtypes: int64(2), object(3), float64(1)
memory usage: 19.1+ MB
```

这段输出里的每一行都有值得解读的信息。我们逐条拆解。

**RangeIndex 告诉你总行数**——这里是 50 万行。你需要立刻在脑子里做一个判断：这个数字是否符合预期？如果预期是 100 万但实际只有 50 万，说明数据采集或导出过程中丢了一半，需要去上游排查原因。

**Non-Null Count 是最有价值的一列**——`response` 显示 498234 non-null，意味着有 1766 行的 response 是空的；`quality` 有 124 个缺失值。这两个数字直接决定了你后续的缺失值处理策略：1766/500000 = 0.35% 的缺失率对于 SFT 数据来说很低，直接 drop 掉就行；但如果某列缺失率超过 5%，你就需要考虑填充策略而不是简单删除。

**Dtype 列暴露潜在的类型问题**——注意 `quality` 列显示的是 `float64` 而不是 `int64`。为什么？因为它包含了 `None`（我们在构造数据时故意加进去的），Pandas 的传统整数类型遇到缺失值会自动降级为浮点数。这在标签编码、ID 映射等必须保持整型的场景中是致命的（04-04 节详细讨论过 Nullable 类型的解决方案）。

**object 类型的数量暗示内存优化空间**——这里有 3 列是 object 类型。每个 object 列实际上存储的是 Python 字符串对象的指针数组，内存效率远低于原生的 string 类型。如果你的数据集有几百万行，把 object 列转成 `string[pyarrow]` 或 `category` 能节省大量内存。

最后是底部的 **memory usage: 19.1+ MB**。那个加号意味着这只是 NumPy 数组本身的内存，不包含 Python 对象内部的开销。加上 `memory_usage='deep'` 参数才能得到真实内存占用——对于包含字符串的 DataFrame 来说，真实内存可能是默认显示值的 3-5 倍。

```python
df.info(memory_usage='deep')
```

## describe()：让数据自己说话

`info()` 回答的是"数据长什么样"，而 `describe()` 回答的是"数据的分布是什么样的"。它对数值列和分类列分别给出不同类型的统计摘要。

### 数值列：从五个统计量到完整的分布画像

```python
import pandas as pd
import numpy as np

np.random.seed(42)
n = 100_000
df = pd.DataFrame({
    'quality': np.random.choice([1,2,3,4,5], n, p=[0.05, 0.15, 0.30, 0.35, 0.15]),
    'tokens': np.random.lognormal(5.5, 1.0, n).astype(int).clip(20, 2000),
    'score': np.random.uniform(0, 1, n),
})

print(df.describe())
```

输出：

```
         quality       tokens      score
count  100000.00  100000.00  100000.00
mean        3.65      274.32       0.50
std         1.10      195.43       0.29
min         1.00       20.00       0.00
25%         3.00      158.00       0.25
50%         4.00      245.00       0.50
75%         4.00      367.00       0.75
max         5.00      2000.00       1.00
```

八个统计量各有各的用途。`count` 和 info() 的 Non-Null Count 交叉验证——如果不一致说明有问题。`mean` 和 `std` 合起来描述集中趋势和离散程度：如果 std 接近甚至超过 mean，说明数据波动极大，可能存在极端值。`min` 和 `max` 给出范围——比如 tokens 的 max=2000 是因为我们手动 clip 了，如果没有 clip 而是出现了 85000 这样的值，你就知道有超长对话需要处理。

四个分位数是最容易被忽视但又最有价值的信息。**中位数（50%）和均值（mean）的差异能告诉你数据是否偏态**——如果 mean 明显大于 median 说明右偏（少数大值拉高平均），反之则左偏。IQR（75% - 25%）常用来识别异常值：超出 `[Q1-1.5*IQR, Q3+1.5*IQR]` 的点通常被视为离群点。

如果你想看到更细粒度的分位信息（比如 P99 用于决定截断阈值），可以自定义分位点：

```python
print(df['tokens'].describe(
    percentiles=[.01, .05, .10, .25, .50, .75, .90, .95, .99]
))
```

### 分类列：唯一值和高频值

`describe()` 默认只分析数值列。对于文本/分类列，需要显式指定：

```python
print(df.describe(include=['object']))
```

这里的重点看两个指标：`unique` 和 `top`/`freq`。unique 告诉你有多少种不同的取值——如果本应是分类变量（如 source）却显示了接近行数的 unique 值，说明可能混入了意外数据。`top` 和 `freq` 告诉你最高频的取值是什么以及出现频率——如果某一类占了 95% 以上，就要考虑类别不平衡的问题。

到这里，你已经掌握了用 info() + describe() + head() + isna().sum() 这组工具来全面了解一张 DataFrame 的方法。但这些都是"看"的操作——当你发现了缺失值、重复值、异常值之后，接下来就需要"动手修"了。那就是下一节的主题。
