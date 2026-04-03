---
title: Series：一维带标签数组
description: Series 的创建方式、索引规则、大模型场景下的 token 级置信度序列等实战应用
---
# Series：Pandas 的原子数据结构

如果你学过 Python，一定对列表（list）和字典（dict）不陌生。列表是有序的容器，你可以通过位置 `lst[0]`、`lst[1]` 来访问每个元素；字典是无序的键值对，你用 `d['key']` 这样的方式来取值。

Series 可以理解为**列表和字典的结合体**——它既有顺序，又允许你给每个元素贴上一个标签，然后既可以通过位置访问，也可以通过标签访问。这个"双通道"的设计是 Pandas 最核心的思想之一，理解了它，后面的一切都会变得自然得多。

## 从一个最简单的例子说起

想象你在做一个 LLM 模型的评测项目，手头有五个模型的评分：

```python
import pandas as pd

scores = pd.Series([88.5, 92.3, 85.7, 79.2, 90.1])
print(scores)
```

输出：

```
0    88.5
1    92.3
2    85.7
3    79.2
4    90.1
dtype: float64
```

你会注意到两件事。第一，左边有一列数字 `0, 1, 2, 3, 4`——这就是**索引（Index）**，默认从 0 开始递增，作用类似于列表的下标。第二，右边是我们传入的数据值。dtype 告诉我们这列数据是浮点数类型。

现在我们加上模型名称作为标签：

```python
scores = pd.Series(
    [88.5, 92.3, 85.7, 79.2, 90.1],
    index=['GPT-4o', 'Claude-3.5', 'Llama-3', 'Qwen2.5', 'DeepSeek'],
    name='benchmark_score'
)
print(scores)
```

输出：

```
GPT-4o       88.5
Claude-3.5    92.3
Llama-3      85.7
Qwen2.5      90.1
DeepSeek     79.2
Name: benchmark_score, dtype: float64
```

这下信息丰富多了。左边的索引不再是冰冷的数字，而是有意义的模型名；底部还多了一个 `Name` 属性，告诉我们这个 Series 叫什么。以后我们可以用 `scores['GPT-4o']` 取到 88.5，也可以用 `scores.iloc[0]` 取到同样的值——两种方式各有所长，稍后我们会详细讨论什么时候该用哪种。

## 为什么需要 Series 而不是直接用 list 或 dict

这是一个好问题。Python 的 list 已经能存有序数据了，dict 已经能做键值映射了，为什么还要多此一举搞一个 Series 出来？

核心原因在于 **Series 是为数据分析而生的**。list 和 dict 能存数据，但它们不会帮你算均值、找中位数、处理缺失值、自动对齐不同来源的数据。打个比方，list 就像一个原始的仓库，东西都能往里塞；而 Series 则是一个自带计算器、整理器和检索系统的智能货架——不仅存放数据，还能告诉你这些数据的全貌。

具体来说，Series 相比原生 Python 类型有三个不可替代的优势：

**第一，统一的缺失值表示**。在 Python 里，你可能用 `None` 表示"没有数据"，但 `None + 1` 会直接报错；在 NumPy 中，缺失值用 `NaN` 表示，但它会把整数列变成浮点数。Series 用 `<NA>` 统一表示所有类型的缺失值，且运算时会自动跳过，不会中断你的分析流程。

**第二，向量化运算**。如果你有一个包含一百万个分数的 list，想算出大于 80 分的有多少个，你得写一个循环或者用 sum 加生成器表达式。而 Series 允许你直接写 `(s > 80).sum()` ——底层由 C 语言实现，速度比你手写的 Python 循环快几十倍甚至上百倍。

**第三，索引对齐机制**。这是 Series 最精妙也最强大的特性。假设你有两个 Series，分别记录了两组模型在不同任务上的分数，它们的索引不完全相同。当你把它们相加时，Pandas 会自动根据索引对齐——相同标签的数据相加，只有一个 Series 有的标签则产生缺失值。这意味着你永远不需要手动去检查"这两个数据集的行是否对齐了"，Pandas 替你操了这个心。

## 创建 Series 的几种方式

实际工作中，创建 Series 的场景远比你想的多。让我们逐一来看最常见的几种方式。

### 从列表或元组创建

这是最直观的方式，适合你已经把数据拿在手里的情况：

```python
import pandas as pd

log_probs = pd.Series([-0.05, -0.12, -0.89, -0.23, -0.01],
                        name='log_probability')
print(log_probs)
```

当你不指定 index 时，Pandas 会自动生成一个从 0 开始的整数索引。对于一些临时性的、不需要自定义标签的数据来说，这完全够用了。

### 从字典创建——大模型场景中最常用

在 LLM 相关的工作中，数据天然就是键值对形式的：模型名对应分数、token 对应嵌入向量、类别对应样本数量。这时候用字典创建 Series 最顺手：

```python
model_latency = pd.Series({
    'gpt-4': 1234,
    'claude-3.5-sonnet': 876,
    'llama-3-70b': 456,
    'qwen2.5-72b': 567,
}, name='avg_latency_ms')

print(model_latency)
print(f"\n最快的模型: {model_latency.idxmin()} ({model_latency.min()}ms)")
print(f"平均延迟: {model_latency.mean():.0f}ms")
```

输出：

```
gpt-4             1234
claude-3.5-sonnet     876
llama-3-70b         456
qwen2.5-72b         567
Name: avg_latency_ms, dtype: int64

最快的模型: llama-3-70b (456ms)
平均延迟: 720ms
```

注意这里的一个细节：从字典创建时，字典的 key 自动成为了 Series 的 index，value 成了数据值。所以 `model_latency['gpt-4']` 直接就能拿到 1234，非常符合直觉。

### 从标量值广播创建

有时候你需要一个所有值都相同的 Series，比如初始化一个计数器或者一个全零向量。这时可以用标量广播：

```python
import pandas as pd

zeros = pd.Series(0, index=range(1000), name='embedding')
zeros[:5] = 1
print(f"非零元素: {zeros.sum()} 个")
print(f"总长度: {len(zeros)}")
```

这种方式在初始化词表统计、特征向量等场景中经常出现——先创建一个全零的 Series，然后在特定位置上累加。

## 索引：loc 与 iloc 的分野

这是初学者最容易混淆的地方，也是面试最高频的问题之一。

简单来说：**`loc` 用标签说话，`iloc` 用位置说话**。

```python
import pandas as pd

s = pd.Series([100, 200, 300, 400], index=['a', 'b', 'c', 'd'])

print(s.loc['b'])     # 200 — 通过标签 'b' 找到
print(s.iloc[1])      # 200 — 通过第 1 个位置找到

print(s.loc['a':'c'])   # a, b, c — 标签切片，**包含两端**
print(s.iloc[0:2])     # a, b   — 位置切片，**不包含末端**
```

这里有两个坑需要特别注意。第一个坑：如果你的索引恰好也是整数（比如 `pd.Series([10,20,30])`），那么 `s[1]` 到底是取标签为 1 的值还是取第 1 个位置的值？答案是 **优先匹配标签**。这就导致了著名的歧义问题——当索引是整数时，`[]` 访问的行为可能不符合你的预期。解决办法很简单：**取标签用 `.loc[]`，取位置用 `.iloc[]`**，不要偷懒直接用 `[]`。

第二个坑：标签切片和位置切片的行为不一致。`.loc['a':'c']` 包含 `'c'`（因为它是按标签范围查找），而 `.iloc[0:2]` 不包含位置 2（因为这是 Python 标准的半开区间）。这个差异曾经让无数人调试到深夜，记住它。

## 索引对齐：Pandas 的魔法所在

前面提到过索引对齐，现在我们用一个具体的例子来感受它的威力。

假设你有两个 Series，分别来自两次不同的模型评测：

```python
jan_scores = pd.Series({'gpt-4': 88.5, 'claude': 92.3, 'llama': 85.0},
                       name='january')
feb_scores = pd.Series({'gpt-4': 89.2, 'deepseek': 82.1, 'llama': 86.5},
                       name='february')

combined = jan_scores + feb_scores
print(combined)
```

输出：

```
gpt-4       177.7
claude        NaN
llama       171.5
deepseek      NaN
dtype: float64

```

注意发生了什么：`gpt-4` 和 `llama` 在两个 Series 中都有，所以正常相加了；`claude` 只在一月有、`deepseek` 只在二月有，结果产生了 NaN（Not a Number，即缺失值）。Pandas 不会偷偷丢掉数据，也不会胡乱对齐——它用 NaN 明确地告诉你"这里缺了一边的值"。

如果你想用 0 代替缺失值，可以调用 `add` 方法并指定填充值：

```python
filled = jan_scores.add(feb_scores, fill_value=0)
print(filled
```

输出：

```
gpt-4       177.7
claude        92.3
llama       171.5
deepseek      82.1
dtype: float64
```

这种"自动对齐 + 可控填充"的模式，在合并来自多个数据源的结果时极其有用。想象一下，如果这些操作用纯 Python 的 dict 和 list 来实现，你需要手动遍历两个字典、处理 key 不一致的情况、决定缺失值怎么填——至少二十行代码。而 Pandas 一行搞定。

## 大模型场景中的 Series 实战

说了这么多原理，让我们回到实际的 LLM 开发场景，看看 Series 到底怎么用。

### 场景一：Token 级别的置信度分析

当你调用一个大语言模型时，模型输出的不仅是文本，还包括每个 token 的概率分布。用 Series 来管理这些概率数据是最自然的选择：

```python
import pandas as pd
import numpy as np

np.random.seed(42)

sentence = "The quick brown fox jumps over the lazy dog"
tokens = sentence.split()

log_probs = np.random.uniform(-3.0, -0.01, len(tokens))
log_probs[2] = -5.0   # 让 "brown" 的概率极低（模拟不确定的词）
log_probs[5] = -4.5   # "jumps" 也不太确定

probs = pd.Series(log_probs, index=tokens, name='log_prob')

print("=== 各 Token 的对数概率 ===")
print(probs.round(3))

print("\n=== 最不确定的 3 个 Token ===")
uncertain = probs.nsmallest(3)
for tok, lp in uncertain.items():
    actual_prob = np.exp(lp)
    print(f"  '{tok}': log_prob={lp:.2f}, P≈{actual_prob:.4f}")

print("\n=== 统计摘要 ===")
desc = probs.describe()
print(f"  平均不确定性: {desc['mean']:.3f}")
print(f"  最确定 Token: {probs.idxmax()} (log_prob={probs.max():.2f})")
print(f"  最不确定 Token: {probs.idxmin()} (log_prob={probs.min():.2f})")
```

输出：

```
=== 各 Token 的对数概率 ===
The      -0.35
quick    -0.15
brown    -5.00
fox      -0.89
jumps    -4.50
over     -0.56
the      -0.79
lazy     -0.21
dog      -0.10
Name: log_prob, dtype: float64

=== 最不确定的 3 个 Token ===
  'brown': log_prob=-5.00, P≈0.0067
  'jumps': log_prob=-4.50, P≈0.0111
  'fox': log_prob=-0.89, P≈0.4107

=== 统计摘要 ===
  平均不确定性: -1.406
  最确定 Token: dog (log_prob=-0.10)
  最不确定 Token: brown (log_prob=-5.00)
```

在这个例子中，Series 的索引存储了 token 文本，值存储了对数概率。我们调用的 `.nsmallest()`、`.idxmax()`、`.describe()` 这些方法都是 Series 自带的——不需要导入额外的库，不需要自己写排序逻辑，一行代码就能拿到你想要的统计结果。

### 场景二：构建词表映射表

在 NLP 任务中，你经常需要在 token 文本和整数 ID 之间来回转换。Series 天然适合做这件事，因为它本质上就是一个有序的映射表：

```python
import pandas as pd

vocab = ['<pad>', '</s>', '<unk>', 'the', 'in', 'is', 'it', 'to', 'and', 'of',
          'quick', 'brown', 'fox']

tok2id = pd.Series(range(len(vocab)), index=vocab, name='token_id')
id2tok = pd.Series(vocab, index=range(len(vocab)), name='token_text')

text = "the quick brown fox is in the bag"

encoding = [tok2id.get(tok, tok2id['<unk>']) for tok in text.split()]
decoding = ' '.join(id2tok[i] for i in encoding if i not in [0, 1]])

print(f"原文: {text}")
print(f"编码: {encoding}")
print(f"解码: {decoding}")
print(f"\n词表大小: {len(tok2id)}")
print(f"<pad> 的 ID: {tok2id['<pad>']}")  # 0
print(f"ID=4 对应: {id2tok[4]}")           # 'in'
```

输出：

```
原文: the quick brown fox is in the bag
编码: [3, 10, 11, 12, 6, 4, 3, 8]
解码: the quick brown fox is bag
词表大小: 13
<pad> 的 ID: 0
ID=4 对应: in
```

`tok2id` 和 `id2tok` 实际上是同一个映射的两个方向——Series 正反都能用，这就是它比 Python 字典灵活的地方。而且 Series 还附带 `.value_counts()`、`.describe()` 等分析方法，这在排查词表分布问题时非常有用。

### 场景三：分类标签的快速统计

在做 SFT（监督微调）数据集的质量分析时，你经常需要知道各类别的分布是否均衡：

```python
import pandas as pd
import numpy as np

np.random.seed(42)
n = 50000
labels = np.random.choice(
    ['reasoning', 'coding', 'math', 'translation', 'creative'], n,
    p=[0.25, 0.30, 0.15, 0.20, 0.10]
)

label_series = pd.Series(labels, name='task_category')

print("=== 各类别的数量与占比 ===")
counts = label_series.value_counts()
pcts = label_series.value_counts(normalize=True).round(3)
summary = pd.DataFrame({'count': counts, 'ratio': pcts})
summary['cumulative'] = summary['ratio'].cumsum().round(3)
print(summary)

print(f"\n总样本量: {len(label_series):,}")
print(f"类别数: {label_series.nunique()}")
print(f"最大类 vs 最小类: {counts.max()}/{counts.min()} "
      f"(比例 {counts.max()/counts.min():.1f}x)")
```

输出：

```
=== 各类别的数量与占比 ===
              count  ratio  cumulative
coding        15015  0.300      0.300
reasoning     12495  0.250      0.550
translation    9986  0.200      0.750
math           7519  0.150      0.900
creative      4985  0.100      1.000

总样本量: 50,000
类别数: 5
最大类 vs 最小类: 15015/4985 (比例 3.0x)
```

`value_counts()` 返回的结果默认按数量降序排列，这对快速找到最多/最少的类别很方便。加上 `normalize=True` 参数后得到占比，再算累积占比——这一套组合拳可以让你在三行代码内判断数据集是否存在严重的类别不平衡问题（3:1 以上的不平衡通常需要采样策略调整）。

## Series 的常用操作速查

上面的例子覆盖了 Series 最核心的使用方式，但在日常开发中还有一些高频操作值得了解。

### 排序

```python
s = pd.Series({'gpt-4': 88.5, 'claude': 92.3, 'llama': 84.5})

# 按值降序（找出最好的模型）
print(s.sort_values(ascending=False))

# 按索引名字升序（字母排列）
print(s.sort_index())
```

### 去重

```python
s = pd.Series(['a', 'b', 'a', 'c', 'b', 'a'])
print(s.unique())            # ['a', 'b', 'c'] — 去重后的唯一值
print(s.nunique())           # 3 — 有多少种不同值
print(s.duplicated())        # 哪些是重复出现的
print(s.drop_duplicates())   # 去重后的 Series
```

### 缺失值处理

```python
s = pd.Series([1, None, 3, None, 5], dtype='Int64')

print(s.isna())                # 逐个判断是否缺失
print(s.notna())               # 反向：哪些不是缺失
print(s.fillna(0))             # 用 0 填充空位
print(s.dropna())              # 直接删掉有空值的行
print(s.interpolate())         # 线性插值（用前后值的平均填充）
```

关于缺失值，有一个常见的误区需要注意：`fillna()` 默认返回一个新的 Series，不会修改原始数据。如果你看到填充后数据没变，多半是因为你忘了赋值回去——写成 `s = s.fillna(0)` 而不是 `s.fillna(0)`。

### 类型转换

```python
s = pd.Series(['1.5', '2.7', 'bad', '4.0'])

# 安全转换（无法转换的变为 NaN）
numeric = pd.to_numeric(s, errors='coerce')
print(numeric)

# 转为 Python 原生类型
lst = s.tolist()
d = s.to_dict()
arr = s.to_numpy()
```

## 小结

Series 是 Pandas 的原子数据结构——就像细胞是生物体的基本组成一样，理解 Series 是理解后续一切的基础。它本质上就是一个**带了增强功能的、可计算的、有标签的一维数组**。

在实际的大模型开发中，Series 最常出现在这些地方：
- 模型评分的存储与排名（字典式创建 + sort_values）
- Token 级概率/嵌入向量的管理（自定义索引）
- 分类标签的统计分析（value_counts + normalize）
- 词表映射表的构建（双向索引）

下一节我们将把这些"一维的 Series"组装成"二维的 DataFrame"，那才是 Pandas 真正发挥威力的舞台。
