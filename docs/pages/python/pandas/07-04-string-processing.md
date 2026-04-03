---
title: 字符串处理
description: .str 访问器的 contains/replace/extract/split 方法，正则表达式在大模型数据清洗中的应用，HTML 标签过滤与特殊字符处理
---
# 字符串清洗：LLM 语料处理的日常

在 LLM 数据处理中，字符串操作是最高频的操作之一——没有之一。你的 prompt 和 response 列几乎全是文本，而原始文本里充斥着 HTML 标签、多余空白、特殊字符、错误标记等各种噪声。Pandas 的 `.str` 访问器就是专门为这种场景设计的：它把 Python 字符串方法向量化到整列操作，不需要写 for 循环。

## .str 访问器基础操作

```python
import pandas as pd

df = pd.DataFrame({
    'text': [
        'Hello <b>World</b>!',
        'Email: test@example.com',
        '  多余空格  ',
        None,
    ]
})

print(df['text'].str.len())
print(df['text'].str.lower())
print(df['text'].str.strip())
```

`.str` 访问器的工作方式很简单：你对它调用的任何方法都会被应用到这一列的每一个元素上，返回结果是一个等长的 Series。如果某个元素是 NaN/None，对应位置的结果也是 NaN——不会报错。

## contains()：模式匹配与异常数据过滤

这是 `.str` 访问器中最有用的方法之一。它让你用正则表达式或简单子串来筛选包含特定模式的行：

```python
import pandas as pd

df = pd.DataFrame({
    'prompt': [
        '什么是注意力机制',
        '如何安装PyTorch',
        '解释BERT模型',
        'GPT-4和Claude哪个好',
        'ERROR: connection timeout',
        '',
        '<html>页面未找到</html>',
    ],
})

has_error = df[df['prompt'].str.contains(r'ERROR|HTTP|<html>', regex=True, case=False)]
print(f"异常数据:\n{has_error}")

clean = df[
    df['prompt'].notna() &
    (df['prompt'].str.len() >= 3) &
    (~df['prompt'].str.contains(r'ERROR|HTTP|<html', regex=True, case=False)) &
    (~df['prompt'].str.contains(r'^\s*$', regex=True))
].copy()
print(f"\n清洗后: {len(df)} → {len(clean)} 条")
```

注意 `na=False` 参数的重要性——如果不加它，当遇到 NaN 值时 `contains()` 会返回 NaN 而不是 False/True，导致后续的布尔索引出错。**这是一个极其常见但容易被忽略的 bug**。

## replace()：正则替换做深度清洗

对于 LLM 语料来说，最常见的字符串清洗任务是去除 HTML 标签和多余空白：

```python
import pandas as pd

df = pd.DataFrame({
    'text': [
        '<p>什么是<b>注意力机制</b>？</p>',
        '  请解释  \tBERT模型  ',
        'ERROR: 超时 - 请重试',
        '正常文本',
    ]
})

df['clean'] = (
    df['text']
    .str.replace(r'<[^>]+>', '', regex=True)
    .str.replace(r'\s+', ' ', regex=True)
    .str.strip()
)

print(df[['text', 'clean']])
```

输出：

```
                           text              clean
0   <p>什么是<b>注意力机制</b>？</p>  什么是注意力机制？
1     请解释  \tBERT模型          请解释 BERT模型
2       ERROR: 超时 - 请重试      ERROR: 超时 - 请重试
3                      正常文本           正常文本
```

链式调用 `.str.replace()` 的好处是每一步只做一个简单的替换动作，组合起来完成复杂的清洗逻辑。正则表达式 `r'<[^>]+>'` 匹配任意 HTML 标签（包括标签本身）并替换为空字符串；`\s+` 把连续的空白字符（空格、制表符、换行）合并为一个空格。

## extract()：从文本中提取结构化信息

有时候你需要从自由文本中提取特定字段。比如从对话记录中提取出用户提到的技术名词：

```python
import pandas as pd

df = pd.DataFrame({
    'text': [
        '我想了解 PyTorch 和 TensorFlow 的区别',
        '请解释 BERT 中的 self-attention',
        'GPT-4o 的上下文窗口是 128K 吗？',
        '推荐一本关于 NLP 的书',
    ]
})

extracted = df['text'].str.extract(
    r'\b(PyTorch|TensorFlow|BERT|GPT-\d+|NLP)\b',
    expand=False
)
df['keyword'] = extracted

print(df)
```

`extract()` 用正则表达式的捕获组从每个字符串中提取匹配的部分。`expand=False` 表示只提取第一个匹配项（返回 Series 而非 DataFrame）。这在从半结构化文本中提取关键字段时特别有用。

## split()：拆分文本为多列

如果你的某一列包含了复合信息（比如 "类别:名称" 格式），可以用 `split()` 拆成多列：

```python
import pandas as pd

df = pd.DataFrame({
    'tag': ['pos:正面', 'neg:负面', 'neutral:中性', 'code:代码'],
})

parts = df['tag'].str.split(':', expand=True)
df['category'] = parts[0]
df['label'] = parts[1]

print(df)
```

`expand=True` 让每个拆分结果变成独立的列而不是嵌套列表。这在解析固定格式日志或标注数据时非常实用。
