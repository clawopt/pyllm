---
title: 重复值检测
description: .duplicated() / .drop_duplicates() 的使用、子集去重、保留策略选择，大模型语料去重实战
---
# 重复值：LLM 训练数据的头号敌人

如果让我在所有数据质量问题中选一个对 LLM 训练影响最大的，我会毫不犹豫地投给**重复数据**。原因很简单：缺失值最多让你的统计不准，但重复数据会让你的模型直接学坏。

## 为什么重复数据如此危险

想象一下你的 SFT 数据集里有 5000 条完全相同的 `(prompt, response)` 对。训练时模型会看到这 5000 条数据各一次——等效于把这个样本的学习权重放大了 5000 倍。结果就是模型会过拟合到这几个特定问答上，泛化能力大幅下降。更糟糕的是，如果你的验证集里也混入了同样的重复数据，评估指标看起来会很好（因为模型"记住"了这些答案），但实际上模型的泛化能力很差——这是一个典型的"假阳性"陷阱。

重复数据在 LLM 语料中的典型来源包括：

- 多次爬取同一批对话导致重复入库
- 不同数据源（API 日志 + Web 日志）包含相同的用户行为
- 数据增强脚本 bug 导致同一条数据被复制多次
- 标注平台导出时包含了历史版本和当前版本

## duplicated()：识别重复行

```python
import pandas as pd

df = pd.DataFrame({
    'id': [1, 2, 3, 2, 5, 6, 1],
    'text': ['hello', 'world', 'foo', 'world', 'bar', 'baz', 'hello'],
})

print(df.duplicated())
```

输出：

```
0    False
1    False
2    False
3     True     ← 第4行和第2行重复
4    False
5    False
6     True     ← 第7行和第1行重复
dtype: bool
```

`duplicated()` 的默认行为是：把第一次出现的标记为 `False`（保留），后续出现的相同行标记为 `True`（重复）。你可以通过 `keep` 参数改变这个行为：

```python
print(df.duplicated(keep='first'))   # 默认：保留第一个
print(df.duplicated(keep='last'))    # 保留最后一个
print(df.duplicated(keep=False))     # 全部标记为重复
```

`keep=False` 在你需要找出"哪些行参与了某个重复组"的场景特别有用——比如你想知道总共有多少条数据涉及了重复问题。

## subset 参数：定义"什么算重复"

这是实际工作中最重要的参数。默认情况下 `duplicated()` 比较的是**整行的所有列**——即只有当两行在每一个字段上都相同时才算重复。但在 LLM 场景中你通常不需要这么严格：

```python
df = pd.DataFrame({
    'conversation_id': [1, 2, 3, 4, 5],
    'user_id': [101, 102, 101, 103, 104],
    'prompt': ['hi', 'hello', 'hi', 'hey', 'yo'],
    'response': ['there', 'world', 'there!', 'you', 'man'],
})

print(f"全量重复: {df.duplicated().sum()} 条")
print(f"prompt+response 重复: {df.duplicated(subset=['prompt','response']).sum()} 条")
print(f"仅 prompt 重复: {df.duplicated(subset=['prompt']).sum()} 条")
```

输出：

```
全量重复: 0 条
prompt+response 重复: 0 条
仅 prompt 重复: 1 条
```

注意三种策略的含义差异：
- **全量比较**：只有完全相同的行才算重复——最严格，适合去除真正的数据录入错误
- **内容比较（prompt+response）**：同一组问答只保留一份——SFT 数据清洗的标准做法
- **Prompt 比较**：同一个问题只保留一份——更激进，会丢失多答案信息

选择哪种取决于你的目标。对于 SFT 训练来说，`subset=['prompt', 'response']` 通常是最合理的折中：它保留了不同用户问同一个问题的记录（可能需要不同的回答风格），但去掉了完全相同的对话副本。

## drop_duplicates()：删除重复并控制保留策略

识别之后就是删除。`drop_duplicates()` 和 `duplicated()` 共享相同的参数体系：

```python
import pandas as pd
import numpy as np

np.random.seed(42)
n = 50_000
df = pd.DataFrame({
    'conversation_id': range(n),
    'prompt': [f'问题{i%100}' for i in range(n)],
    'response': [f'回答{i}' for i in range(n)],
    'quality_score': np.round(np.random.uniform(1, 5, n), 1),
})

before = len(df)

dedup_first = df.drop_duplicates(subset=['prompt'], keep='first')
dedup_last = df.drop_duplicates(subset=['prompt'], keep='last')

print(f"原始: {before:,} 条")
print(f"保留首次出现: {len(dedup_first):,} 条 (去除了 {before-len(dedup_first):,})")
print(f"保留最后出现: {len(dedup_last):,} 条")
```

这里有一个关键的实战技巧：**按质量分数排序后再去重，保留质量最高的版本**：

```python
def smart_dedup(df, subset_cols, priority_col='quality_score'):
    sorted_df = df.sort_values(by=priority_col, ascending=False)
    deduped = sorted_df.drop_duplicates(subset=subset_cols, keep='first')
    return deduped.reset_index(drop=True)

result = smart_dedup(df, ['prompt'], 'quality_score')
print(f"智能去重后: {len(result):,} 条")
print(f"\n质量分布:\n{result['quality_score'].describe().round(2)}")
```

这个模式的逻辑是：先按优先级列降序排列（质量最高的排最前面），然后 `keep='first'` 自然就保留了每个 prompt 对应的质量最高的那条记录。比随机去重要合理得多。

## 性能考虑：大数据集的去重策略

当数据量达到百万级以上时，直接的字符串比较开始变慢。有两个优化方向：

第一是用哈希代替字符串比较。先把每行关键列的内容计算为一个 MD5 或 SHA256 哈希值，然后基于哈希值做去重：

```python
import hashlib

def hash_dedup(df, columns):
    def row_hash(row):
        content = '|'.join(str(row[c]) for c in columns)
        return hashlib.md5(content.encode()).hexdigest()
    
    df['_hash'] = df.apply(row_hash, axis=1)
    result = df.drop_duplicates(subset='_hash').drop(columns='_hash')
    return result
```

第二是对于千万级行的数据考虑近似去重。精确去重的时间复杂度是 O(n log n)（排序 + 比较），对于超大数据集可以考虑 MinHash / SimHash 等局部敏感哈希算法——它们能在 O(n) 时间内找到"高度相似但不完全相同"的重复项。这部分内容我们会在后续的语料清洗章节中深入展开。

到这里，第六章的前三节已经覆盖了数据概览、缺失值分析和重复值检测这三个核心的数据质量检查维度。最后一节我们把所有检查整合成一个自动化的质量报告生成器。
