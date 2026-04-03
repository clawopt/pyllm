---
title: 缺失值处理
description: dropna() / fillna() / interpolate() 的使用策略，大模型场景下对话数据空响应的处理方案
---
# 缺失值处理：删还是填？

上一节我们学会了如何检测缺失值。这一节要解决的问题是：检测到了之后怎么办？答案不是简单的"全删"或"全填"——不同的场景需要完全不同的策略。

## 三条路线的选择逻辑

处理缺失值本质上是在三个选项中做选择：

- **删除（dropna）**：最干净但会丢失数据。适用于缺失比例低（<5%）且随机分布的情况
- **填充（fillna）**：保留数据量但引入噪声。适用于需要保持行数、缺失有规律可循的情况
- **保留（标记为特殊值）**：缺失本身携带信息。适用于 MNAR 场景——比如"空 response"本身就意味着用户不满意

在 LLM 语料中，我的经验法则是：**prompt 或 response 为空 → 直接删除；质量分数/元数据为空 → 填充或标记；时间戳为空 → 填充默认值或插值**。原因是 prompt/response 为空的样本对训练没有任何价值（模型不能从空输入中学到东西），而质量分数缺失可以用均值或中位数填充而不太影响整体分布。

## dropna()：什么时候该狠心删除

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'id': [1, 2, 3, 4, 5],
    'prompt': ['什么是AI', None, '解释Python', np.nan, '如何学习LLM'],
    'response': ['AI是...', '...', 'Python是...', None, 'LLM是...'],
    'quality': [4.5, None, 3.8, 4.2, None],
})

clean_all = df.dropna()
print(f"全量删除(任一列有空就删): {len(df)} → {len(clean_all)}")
```

输出：

```
全量删除(任一列有空就删): 5 → 1
```

5 行数据只剩 1 行——因为几乎每行都有至少一个空值。这种"一刀切"式的删除在实际工作中很少使用。更常见的做法是指定关键列：

```python
clean_key = df.dropna(subset=['prompt', 'response'])
print(f"只要求 prompt+response 非空: {len(df)} → {len(clean_key)}")
```

`subset` 参数的含义是：只有指定的这些列全部非空时才保留该行。对于 SFT 训练数据来说，prompt 和 response 是绝对不能为空的——它们是模型的输入和目标输出。其他字段如 quality_score 缺失了可以容忍。

另一个有用的参数是 `thresh`（阈值）——要求一行中至少有多少个非空值才保留：

```python
clean_thresh = df.dropna(thresh=3)
print(f"至少3个非空值: {len(df)} → {len(clean_thresh)}")
```

对于 LLM 对话数据，我建议封装一个标准的清洗函数：

```python
def clean_for_sft(df):
    before = len(df)
    
    df = df[df['prompt'].notna() & (df['prompt'].str.strip() != '')]
    if 'response' in df.columns:
        df = df[df['response'].notna() & (df['response'].str.strip() != '')]
    
    after = len(df)
    print(f"SFT 清洗: {before:,} → {after:,} (去除 {before-after:,}, {(before-after)/before*100:.1f}%)")
    return df.reset_index(drop=True)
```

## fillna()：填充的艺术

当你决定不删除而是填充时，下一个问题是：用什么来填？这个选择直接影响后续统计和模型训练的效果。

### 数值列的填充策略

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'value': [1.0, np.nan, 3.0, np.nan, 5.0],
    'score': [85, np.nan, 92, np.nan, 78],
})

df['filled_zero'] = df['value'].fillna(0)
df['filled_mean'] = df['value'].fillna(df['value'].mean())
df['filled_median'] = df['score'].fillna(df['score'].median())
df['filled_ffill'] = df['score'].ffill()
df['filled_bfill'] = df['score'].bfill()

print(df.round(2))
```

五种常用填充方式各有适用场景：
- **填 0**：适合"无"确实等价于零的场景（比如计数器、标志位）
- **填均值/中位数**：适合数值型特征——保持总体分布不变。中位数比均值更抗异常值
- **前向填充 ffill()**：适合时间序列或有序数据——用前一个有效值填补
- **后向填充 bfill()**：和 ffill 相反方向

### 分组填充：更精细的策略

全局均值填充有一个问题：它假设所有行的分布是一致的。但在实际数据中不同群体的均值可能差异很大。比如不同来源（api vs web）的数据质量分数分布可能完全不同——用全局均值去填 web 来源的缺失值会产生偏差：

```python
import pandas as pd
import numpy as np

np.random.seed(42)
n = 100_000
df = pd.DataFrame({
    'source': np.random.choice(['api', 'web'], n),
    'quality': np.where(
        np.random.random(n) < 0.02,
        np.nan,
        np.where(np.random.random(n) < 0.5,
                 np.random.uniform(4.0, 5.0, n),
                 np.random.uniform(2.0, 3.5, n))
    ),
})

global_mean = df['quality'].mean()
group_means = df.groupby('source')['quality'].transform('mean')

df['filled_global'] = df['quality'].fillna(global_mean)
df['filled_grouped'] = df['quality'].fillna(group_means)

print(f"全局均值: {global_mean:.2f}")
print(f"API 组均值: {group_means[df['source']=='api'].mean():.2f}")
print(f"Web 组均值: {group_means[df['source']=='web'].mean():.2f}")
```

`transform('mean')` 的作用是对每个分组计算均值，然后把均值广播回原始 DataFrame 的每一行。这样每个缺失值都被同组内的均值填充而不是被全局均值填充——结果更准确、偏差更小。

### 字符串列的填充

字符串列的缺失值通常不适合用统计量填充。常见做法是用一个明确的占位符：

```python
df['text_filled'] = df['text'].fillna('[EMPTY]')
df['category_filled'] = df['category'].fillna('unknown')
```

但如果你的列使用了 Nullable 类型（`string`），直接 `fillna()` 就能正确地填入 `<NA>` 而不会破坏类型。

到这里，第七章的前两节覆盖了缺失值的检测和处理。接下来我们要看重复值的实际清理操作、类型转换修正以及 LLM 场景下最高频也最重要的字符串清洗。
