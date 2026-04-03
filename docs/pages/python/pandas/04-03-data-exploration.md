---
title: 数据查看与探索
description: head/tail/info/describe 等方法详解，大模型场景下快速预览百万级语料数据分布
---
# 数据探索与概览

想象一下这个场景：你刚从 S3 上拉下来一份 500 万行的对话语料 CSV，准备做 SFT 微调前的数据清洗。在写任何清洗逻辑之前，你第一件事做什么？直接上 `dropna()`、`groupby()`？**绝对不行**——你连这份数据长什么样都不知道，贸然操作等于闭眼开车。

Pandas 的数据探索工具链就是你的"开车前检查清单"。这一章我们不罗列 API 文档，而是从"拿到一份数据后到底该怎么看"这个实际问题出发，把 `head()`、`info()`、`describe()`、`value_counts()` 这些方法串成一套完整的探索工作流。

## 拿到数据后的第一件事：先看再看

很多新手（甚至不少有经验的工程师）拿到 DataFrame 后的习惯是：看一眼 `shape`，然后就开始写处理逻辑。这种做法在生产环境里埋下的隐患比你想象的多得多。我见过最离谱的案例：有人拿到的 CSV 里第一行其实是元信息注释而不是列名，结果整张表所有列的类型全变成了 `object`，后续所有的数值运算全部静默失败，直到模型训练 loss 不收敛才回过头来排查——这时候已经浪费了几天的 GPU 时间。

所以 Pandas 社区里有一句老话：**先看再做（Look before you process）**。这不是建议，这是铁律。

那具体怎么看？我们用一个贴近真实场景的例子来走一遍完整的探索流程。假设你拿到了一份 LLM 训练语料：

```python
import pandas as pd
import numpy as np

np.random.seed(42)
n = 1_000_000
df = pd.DataFrame({
    'prompt': [f'问题 {i}' for i in range(n)],
    'response': [f'回答 {i}' for i in range(n)],
    'quality': np.random.choice([1, 2, 3, 4, 5], n),
    'tokens': np.random.randint(20, 2000, n),
    'source': np.random.choice(['api', 'web', 'export'], n),
})
df.loc[np.random.choice(n, 1766, replace=False), 'response'] = None
```

注意我在最后故意往 `response` 列里塞了 1766 个缺失值——这在真实语料里太常见了：爬虫抓取失败、API 超时返回空、JSON 解析异常……各种原因都会产生缺失值，而如果你不主动去发现它，这些脏数据就会悄无声息地混进你的训练集。

## head() 和 tail()：不是打印，是在找问题

`head()` 是几乎所有 Pandas 教程第一个介绍的方法，但绝大多数人只把它当成"看看前几行长啥样"的打印工具。实际上 `head()` 的真正价值在于帮你发现**结构性问题**。

```python
print(df.head())
```

输出大概长这样：

```
   prompt response  quality  tokens source
0   问题 0     回答 0        5     349     api
1   问题 1     回答 1        4    1234     web
2   问题 2     回答 2        2     890  export
3   问题 3     回答 3        5      45     api
4   问题 4     回答 4        1    1567     web
```

从这几行你能确认几件事：列名是否正确（有没有被读成 `Unnamed: 0`）、每列的数据类型看起来对不对（`quality` 显示的是数字还是字符串）、有没有明显的格式异常。但这里有个很多人忽略的技巧：**用 `head()` 检查的时候一定要同时看 `tail()`**。

```python
print(df.tail(3))
```

为什么要看尾部？因为数据采集过程中的错误往往集中在文件的末尾——写入中断、缓冲区未刷新、追加写入时的重复头部……这些问题在 `head()` 里完全看不到，但 `tail()` 一眼就能暴露。另外，如果你想随机抽样而不是只看首尾，`sample()` 才是正确选择：

```python
print(df.sample(5, random_state=42))
```

`sample()` 返回的是真正的随机行，不依赖数据的物理排列顺序，对于检查数据质量来说比 `head()/tail()` 更有代表性。不过要注意一个性能细节：`sample()` 在大数据集上需要生成随机索引，如果只是想快速瞥一眼，`head()` 的开销要小得多。

## info()：被严重低估的全景方法

如果说 `head()` 是"粗看外观"，那 `info()` 就是"全面体检"。我敢说 `info()` 是整个 Pandas 里**被低估程度排名第一的方法**——它能一次性告诉你这张表的形状、每列的类型、非空值数量、内存占用，而且执行速度极快，百万级行的 DataFrame 跑一次 `info()` 几乎是瞬间的。

```python
df.info()
```

典型输出如下：

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1000000 entries, 0 to 999999
Data columns (total 5 columns):
 #   Column    Non-Null Count    Dtype 
---  ------    --------------    ----- 
 0   prompt       1000000 non-null  object
 1   response       998234 non-null  object
 2   quality      1000000 non-null  int64  
 3   tokens       1000000 non-null  int64  
 4   source       1000000 non-null  object
dtypes: int64(2), object(3)
memory usage: 38.1+ MB
```

这段输出里藏着大量关键信息，我们逐条拆解。首先是 **Non-Null Count** 这列——看到 `response` 那一行显示的是 `998234 non-null` 了吗？总共有 100 万行，这意味着有 **1766 行的 response 是空的**。如果你没跑 `info()` 直接开始训练，这 1766 条空回复的数据就混进了 SFT 数据集，模型学到的是"遇到某些 prompt 应该返回空"，后果可想而知。

其次是 **Dtype** 列。这里显示 `object` 类型占了 3 列（prompt、response、source）。`object` 在 Pandas 里是个"万能兜底"类型——它意味着 Pandas 不知道这列到底是什么，只能用 Python 对象的指针数组来存。这不仅内存效率极低，还意味着你无法对这些列做向量化数值运算。看到 `object` 就应该立刻思考：这列真的应该是字符串吗？还是读数时出了问题？比如 `quality` 如果不小心读成了 `object`（因为原始数据里混了一个 "N/A" 字符串），后续的所有统计都会出错。

第三是底部的 **memory usage: 38.1+ MB**。那个加号很重要——默认情况下 `info()` 只计算 NumPy 数组本身的内存，不包括 Python 对象内部占用的空间（比如每个字符串对象的开销）。加上 `deep=True` 参数才能得到真实内存占用：

```python
df.info(memory_usage='deep')
```

你可能会发现实际内存比默认显示的高出好几倍——尤其是包含大量字符串的列，差异会非常惊人。这也是为什么我们在下一节（04-04）要专门讨论 dtype 优化的原因。

关于 `info()` 还有一个实用技巧：你可以对部分列单独调用，当你只关心某几个关键列的时候这很有用：

```python
df[['prompt', 'quality', 'tokens']].info()
```

另外，如果你想程序化地获取各列的内存占比来做优化决策，可以这样：

```python
mem = df.memory_usage(deep=True)
mem_pct = (mem / mem.sum() * 100).round(2)
print(mem_pct.sort_values(ascending=False))
```

## describe()：让数字自己说话

`info()` 告诉你"有哪些列、什么类型、有没有空值"，而 `describe()` 告诉你"这些数值列的分布是什么样的"。这两者是互补的关系。

### 数值列的统计摘要

```python
print(df.describe())
```

输出：

```
            quality        tokens
count  1,000,000.00  1,000,000.00
mean           3.01        512.34
std            1.41        289.56
min            1.00         20.00
25%            2.00        265.00
50%            3.00        510.00
75%            4.00        759.00
max            1999.00    1999.00
```

每一行都是一个统计量，但关键是你要知道**怎么解读它们以及什么时候该警惕**。`count` 告诉你有效样本数量——如果它小于总行数，说明有缺失值（这和 `info()` 的 Non-Null Count 是交叉验证关系）。`mean` 和 `std` 合起来描述数据的集中趋势和离散程度——如果 `std` 接近甚至超过 `mean`，说明数据波动非常大，可能存在极端值需要处理。

`min` 和 `max` 给出范围，这在 LLM 场景下特别有用。比如 `tokens` 列的 `min=20`、`max=1999`——如果你的模型上下文窗口只有 4096 token，那 max=1999 完全没问题；但如果 max 显示为 85000，你就知道有超长对话需要截断或排除。同理 `quality` 的 min=1、max=5 说明评分范围符合预期，如果突然出现 min=0 或 max=10，那就说明数据标注规则可能有变化或者混入了其他数据源。

四个分位数（25%、50%、75%）是最容易被忽视但又最有价值的信息。**中位数（50%）和均值（mean）的差异能告诉你数据是否偏态**。比如如果 mean=3.5 但 median=3.0，说明数据偏向高分端（右偏），可能存在少量极高分的样本拉高了均值。在数据质量评估中，IQR（四分位距 = 75% - 25%）常用来识别异常值：超出 `[Q1-1.5*IQR, Q3+1.5*IQR]` 范围的点通常被视为离群点。

如果你觉得默认的五位数摘要不够细，`describe()` 支持自定义分位点：

```python
print(df['tokens'].describe(percentiles=[.01, .05, .10, .25, .50, .75, .90, .95, .99]))
```

这样你就能看到 P99 的 token 数是多少——如果 P99=1800 但 max=1999，说明最长的 1% 对话集中在 1800-1999 这个区间，你可能需要决定是否截断到某个阈值以统一输入长度。

### 分类列也能 describe

`describe()` 默认只分析数值列，但你加上 `include` 参数后它对字符串/分类列同样有用：

```python
print(df.describe(include=['object']))
```

输出：

```
          prompt     response     source
count   1,000,000     998,234   1,000,000
unique  1,000,000     987,432           3
top       问题 0       回答 0         api
freq             1           2    333,333
```

这里的 `unique` 值得特别注意。`prompt` 有 100 万个唯一值——合理，因为每条对话的 prompt 都不同。但如果某列本应是分类变量（比如 `source`）却显示了接近行数的 unique 值，那就说明这列可能混入了意外数据（比如 URL、时间戳等不该出现的值被错误地归到了这一列）。`top` 和 `freq` 则告诉你最高频的取值是什么、出现了多少次——如果 `freq` 异常高（比如某一类占了 95% 以上），你就要考虑是否存在类别不平衡的问题。

## value_counts()：分布分析的瑞士军刀

如果说 `describe()` 给你的是宏观统计，那 `value_counts()` 就是让你深入每一列内部去看具体的分布情况。它是数据探索阶段**使用频率最高的方法之一**，尤其是在处理类别型数据时。

### 基础用法与归一化

```python
print(df['quality'].value_counts().sort_index())
```

输出：

```
quality
1    200102
2    199881
3    200334
4    199776
5    199907
Name: count, dtype: int64
```

`sort_index()` 让结果按值排序而非按频率排序——对于有序类别（如评分 1-5）来说这更有意义。你一眼就能看出每个分数段的样本量大致均衡（各约 20 万），这对 SFT 训练来说是好消息，说明不需要额外做类别平衡处理。

但光看绝对数量有时候不够直观，加上 `normalize=True` 就能得到比例：

```python
print(df['quality'].value_counts(normalize=True).round(3))
```

输出：

```
quality
1    0.200
2    0.200
3    0.200
4    0.200
5    0.200
Name: proportion, dtype: float64
```

在实际工作中，我习惯把计数和比例拼在一起看：

```python
dist = df['source'].value_counts()
pct = df['source'].value_counts(normalize=True).round(3)
print(pd.concat([dist, pct], axis=1, keys=['count', 'pct']))
```

输出：

```
        count    pct
source
api    333437  0.333
web    333294  0.333
export 333269  0.333
```

三个来源几乎均等——很好。如果某个来源占了 80% 以上，你就需要考虑：这个来源的数据质量是否和其他来源一致？不同来源之间是否存在系统性偏差？

### 用 nunique() 快速判断列的"身份"

`value_counts()` 能告诉你每个值的频率，但有时候你只需要知道"这一列有多少种不同的取值"。这时 `nunique()` 更高效：

```python
for col in df.columns:
    uniq = df[col].nunique()
    total = len(df)
    ratio = uniq / total
    
    if ratio < 0.05:
        hint = "→ 低基数，适合转 category"
    elif ratio > 0.8:
        hint = "→ 高基数，可能是自由文本或 ID"
    else:
        hint = "→ 中等基数"
    
    print(f"{col:12s} {uniq:>10,} 种 ({ratio:.1%}) {hint}")
```

输出：

```
prompt     1,000,000 种 (100.0%) → 高基数，可能是自由文本或 ID
response     987,432 种 (98.7%) → 高基数，可能是自由文本或 ID
quality             5 种 (0.0%) → 低基数，适合转 category
token          1,981 种 (0.2%) → 中等基数
source             3 种 (0.0%) → 低基数，适合转 category
```

这个分析直接指导 dtype 优化决策：低基数的 `quality` 和 `source` 应该转成 `category` 类型节省内存，高基数的 `prompt` 和 `response` 保持 `string[pyarrow]` 即可。这就是数据探索和下一节内容（dtype 系统）之间的自然衔接。

## 缺失值与重复值：沉默的质量杀手

数据探索的最后一步——也是最容易跳过的一步——是检查缺失值和重复值。它们不会报错，不会抛异常，但会在你最不经意的地方搞垮你的分析结果。

### 缺失值的系统化检测

```python
missing = df.isna().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({'缺失数': missing, '占比(%)': missing_pct})
missing_df = missing_df[missing_df['缺失数'] > 0].sort_values('缺失数', ascending=False)
print(missing_df)
```

输出：

```
          缺失数  占比(%)
response    1766    0.18

```

只有 `response` 列有 1766 个缺失值，占比 0.18%。这个比例算高还是低？取决于你的场景。对于 SFT 训练数据来说，0.18% 的缺失率其实很低，直接 `dropna()` 掉就行。但如果这是医疗数据或金融数据，0.18% 的缺失都可能意味着重要的病例或交易记录丢失，需要更谨慎地处理（比如用填充而非删除）。

这里有个常见的坑：**`isna()` 只能检测标准的缺失值（NaN/None/NA）**。如果你的数据里用空字符串 `""`、`"NULL"`、`"N/A"` 或者 `"-"` 来表示"无数据"，`isna()` 会漏掉它们。所以在实际项目中，你往往需要结合多种方式来检测：

```python
def detect_missing_comprehensive(df):
    results = {}
    for col in df.columns:
        na_count = df[col].isna().sum()
        if df[col].dtype == 'object':
            empty_str = (df[col] == '').sum()
            placeholder = df[col].isin(['NULL', 'N/A', '-', 'null', 'none']).sum()
            results[col] = {'NA': na_count, '空串': empty_str, '占位符': placeholder}
        else:
            results[col] = {'NA': na_count, '空串': 0, '占位符': 0}
    return pd.DataFrame(results).T

print(detect_missing_comprehensive(df[['prompt', 'response']]))
```

### 重复值检测

```python
full_dup = df.duplicated().sum()
partial_dup = df.duplicated(subset=['prompt', 'response']).sum()
print(f"完全重复行: {full_dup}")
print(f"prompt+response 重复: {partial_dup}")
```

重复值在 LLM 语料中的危害经常被低估。想象一下：你的 SFT 数据集里有 5000 条完全相同的 `(prompt, response)` 对，模型在训练时会看到同一份样本 5000 次，等效于把这个样本的学习率放大了 5000 倍，导致模型过拟合到这几个特定问答上，泛化能力大幅下降。更隐蔽的情况是**近似重复**——prompt 相同但 response 只有细微差别（比如多了个标点符号），这种重复用 `duplicated()` 检不出来，需要用语义去重（后面章节会讲到 MinHash / SimHash 等方法）。

## 把一切整合起来：自动化探索报告函数

上面介绍的每种方法单独用都不复杂，但在实际工作中你需要把它们组合成一个标准化的检查流程。下面是一个生产级的探索报告函数，每次加载新数据后跑一遍，几分钟内就能全面掌握数据状况：

```python
def explore_report(df, name="Dataset"):
    lines = []
    lines.append(f"\n{'='*60}")
    lines.append(f"  {name} 数据探索报告")
    lines.append(f"{'='*60}")
    
    lines.append(f"\n形状: {df.shape[0]:,} 行 x {df.shape[1]} 列")
    lines.append(f"内存: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    lines.append(f"\n--- 列详情 ---")
    for col in df.columns:
        dt = str(df[col].dtype)
        nn = df[col].notna().sum()
        nv = len(df) - nn
        nu = df[col].nunique()
        flag = "" if nv == 0 else f" [!{nv:,} 缺失]"
        
        if nu <= 10:
            top = df[col].value_counts().head(3).to_dict()
            top_s = ', '.join(f"{k}:{v}" for k, v in top.items())
        else:
            top_s = f"({nu:,} 种)"
        
        lines.append(f"  {flag} {col:<16s} | {dt:<14s} | {nn:>8,}/{len(df):<8,} | 唯一:{nu:>6,} | {top_s}")
    
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        lines.append(f"\n--- 数值列统计 ---")
        stats = df[num_cols].describe().T[['count','mean','std','min','50%','max']]
        lines.append(stats.round(2).to_string())
    
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols[:5]:
        dist = df[col].value_counts().head(8)
        lines.append(f"\n--- {col} 分布 ({df[col].nunique()} 种) ---")
        for val, cnt in dist.items():
            bar = '#' * min(int(cnt / len(df) * 80), 50)
            lines.append(f"  {bar} {val}: {cnt:,} ({cnt/len(df)*100:.1f}%)")
    
    dup = df.duplicated().sum()
    na_total = df.isna().sum().sum()
    lines.append(f"\n--- 质量概览 ---")
    lines.append(f"  重复行: {dup:,} | 总缺失值: {na_total:,}")
    
    text = '\n'.join(lines)
    print(text)
    return text

report = explore_report(df, "LLM 训练语料")
```

运行后会输出一份结构化的报告，涵盖形状、内存、逐列详情（类型、非空数、唯一值、高频值）、数值统计摘要、类别分布柱状图、重复/缺失汇总。**把它当成你每次拿到新数据后的标准动作**，养成习惯后能避免绝大多数因数据质量问题导致的返工。

到这里，第四章的前三节已经覆盖了 Series、DataFrame 和数据探索这三个核心基础。最后一节我们要深入 Pandas 的类型系统——理解 dtype 不仅关系到内存优化，更直接影响你能否正确地处理缺失值、进行高效的比较运算，以及在单机上跑完千万级的数据处理任务。
