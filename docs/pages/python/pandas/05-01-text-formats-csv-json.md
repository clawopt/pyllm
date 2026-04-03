---
title: 文本格式：CSV 与 JSON
description: read_csv/to_csv 参数详解与常见陷阱、JSON/JSONL 读写、大文件分块读取 chunksize
---
# 文本格式：CSV 与 JSON

在所有 Pandas 的 I/O 功能中，`read_csv()` 绝对是使用频率最高的一个——没有之一。不管你是从数据库导出数据、从标注平台下载结果、还是处理开源数据集，CSV 几乎总是第一个遇到的格式。但"常用"不等于"简单用对"，实际上 `read_csv()` 有超过 50 个参数，其中至少有十几个在生产环境中直接影响你的数据处理是否正确、高效甚至能否跑通。

这一节我们从 CSV 为什么如此重要说起，逐步深入到 `read_csv()` 的关键参数、编码陷阱、大文件分块策略，然后过渡到 JSONL——这个在 LLM 训练中比 CSV 更重要的格式。

## CSV：数据交换的通用语

为什么 CSV 在 LLM 开发中无处不在？因为它是所有系统之间的**最低公分母**。产品数据库能导出 CSV，BI 工具能导出 CSV，标注平台的交付物通常是 Excel 转 CSV，监控平台的数据导出也是 CSV。你几乎不可能遇到一个不支持 CSV 导出的系统。

但这也意味着你会遇到各种质量参差不齐的 CSV 文件：有的用逗号分隔、有的用 Tab；有的带 BOM 头、有的编码是 GBK；有的第一行是列名、有的前五行都是注释……所以掌握 `read_csv()` 不是"学会调用一个函数"，而是**学会应对真实世界里各种不规范的数据文件**。

### 最基本的用法和最常见的错误

```python
import pandas as pd

df = pd.read_csv('data.csv')
print(df.head())
```

这三行代码看起来毫无问题，但它隐藏了几个你可能意识不到的风险。第一，Pandas 会**自动推断每列的 dtype**——如果某一列大部分是数字但混了一个字符串（比如 `"N/A"`），整列会被推断为 `object` 类型。第二，它会**把第一行当作列名**——如果你的 CSV 文件前面有空行或者注释行，列名就会错位。第三，它会把整张表**一次性全部加载到内存**——对于大文件来说这可能直接导致 OOM。

这些风险对应的解决方案就是接下来要介绍的几个核心参数。

### usecols + dtype：性能优化的第一道防线

这是我在生产环境中最常推荐的两个参数组合。`usecols` 告诉 Pandas "只读取这几列"，`dtype` 告诉它 "每列用什么类型存"。两者配合使用的效果非常显著：

```python
import pandas as pd

df = pd.read_csv(
    'large_corpus.csv',
    usecols=['conversation_id', 'user_message', 'assistant_message',
             'turn_count', 'timestamp', 'quality_score'],
    dtype={
        'conversation_id': 'string',
        'user_message': 'string',
        'assistant_message': 'string',
        'turn_count': 'int16',
        'quality_score': 'float32',
    },
    parse_dates=['timestamp'],
)
```

为什么要同时指定这两个？因为它们解决的是不同层面的问题。`usecols` 作用于 **I/O 层面**——不需要的列根本不会被解析，不仅省内存还省解析时间。假设你的 CSV 有 20 列但你只需要 5 列，那 15 列的文本解析工作完全被跳过了。而 `dtype` 作用于 **内存层面**——即使只读了 5 列，如果默认用 `int64` 和 `float64` 存储整数和浮点数，内存仍然可能比实际需求大好几倍。

用一个具体的 benchmark 来感受一下差异：

```python
import pandas as pd
import numpy as np
import time, os

n = 5_000_000
tmp_path = '/tmp/bench_data.csv'

np.random.seed(42)
test_df = pd.DataFrame({
    'id': range(n),
    'text': ['sample text data'] * n,
    'value': np.random.randn(n),
    'category': np.random.choice(['A', 'B', 'C', 'D'], n),
})
test_df.to_csv(tmp_path, index=False)

start = time.time()
df_all = pd.read_csv(tmp_path)
t_all = time.time() - start
mem_all = df_all.memory_usage(deep=True).sum() / 1024**2

start = time.time()
df_opt = pd.read_csv(
    tmp_path,
    usecols=['id', 'text', 'value', 'category'],
    dtype={'id': 'int32', 'text': 'string', 'value': 'float32', 'category': 'category'},
)
t_opt = time.time() - start
mem_opt = df_opt.memory_usage(deep=True).sum() / 1024**2

print(f"{'方式':<14} {'时间':>8} {'内存':>10}")
print(f"{'全量默认':<14} {t_all:>7.2f}s {mem_all:>9.1f} MB")
print(f"{'优化参数':<14} {t_opt:>7.2f}s {mem_opt:>9.1f} MB")
print(f"加速 {t_all/t_opt:.1f}x | 省内存 {(1-mem_opt/mem_all)*100:.0f}%")

os.remove(tmp_path)
```

典型输出：

```
方式             时间       内存
全量默认          3.45s   1142.3 MB
优化参数          1.12s    286.7 MB
加速 3.1x | 省内存 75%
```

3 倍的读取加速加上 75% 的内存节省，只是加了两个参数而已。这就是为什么我说 `usecols` + `dtype` 是**每次读 CSV 都应该养成的习惯**。

### 分隔符与转义：当逗号不是分隔符的时候

CSV 的名字叫"逗号分隔值"，但现实中的文件未必真的用逗号分隔。日志文件通常用 Tab（`\t`），某些导出数据用竖线（`|`），还有些文件的分隔符你根本不确定。这时候就需要 `sep` 参数：

```python
df_tsv = pd.read_csv('data.tsv', sep='\t')
df_pipe = pd.read_csv('data.psv', sep='|')
```

如果你拿到一个文件不知道用什么分隔的，可以让 Pandas 自动检测：

```python
df_auto = pd.read_csv('unknown.csv', sep=None, engine='python')
```

注意这里必须指定 `engine='python'`，因为 Python 引擎支持自动分隔符推断而 C 引擎不支持。代价是 Python 引擎比 C 引擎慢不少，所以这只适合用来探索性分析——确认了分隔符之后，正式处理时应该显式指定 `sep` 并使用 C 引擎。

还有一个更棘手的情况：字段内容本身包含分隔符。比如一个 CSV 里有一列是地址，地址里可能有逗号：

```python
from io import StringIO

csv_data = """name,age,city
"Alice, Jr.",25,"New York, NY"
"Bob, O'Brien",30,"London, UK"
"""

df = pd.read_csv(StringIO(csv_data))
print(df)
```

输出：

```
          name  age           city
0   Alice, Jr.   25    New York, NY
1  Bob, O'Brien   30      London, UK
```

Pandas 的 CSV 解析器默认会正确处理双引号包裹的字段内部的逗号——前提是你的原始文件确实用了标准的双引号转义。如果文件格式不规范（比如用单引号或没有引号包裹），你就需要额外指定 `quotechar` 和 `quoting` 参数。

### 编码问题：多语言数据的头号杀手

如果你的数据只包含纯英文文本，编码问题基本不会找上门。但在 LLM 场景下，你处理的语料大概率包含中文、日文、阿拉伯文等各种语言，这时候编码错误就是最高频的问题之一，没有之一。

```python
import pandas as pd

df_utf8 = pd.read_csv('multilingual.csv', encoding='utf-8')

df_gbk = pd.read_csv('chinese_export.csv', encoding='gbk')

df_latin = pd.read_csv('european.csv', encoding='latin-1')
```

最常见的情况是你拿到一份从旧系统导出的中文数据，用默认的 UTF-8 读直接报 `UnicodeDecodeError`。这是因为国内很多 legacy 系统（尤其是 Windows 平台上的 Excel 导出、老版本数据库工具）默认使用 GBK/GB2312 编码。解决办法很简单——加上 `encoding='gbk'` 就行。但问题是**你怎么知道该用哪种编码**？

如果不确定编码，可以用容错模式先读进来看看：

```python
try:
    df = pd.read_csv('unknown_encoding.csv')
except UnicodeDecodeError:
    df = pd.read_csv('unknown_encoding.csv', encoding_errors='replace')
```

`encoding_errors='replace'` 会把无法解码的字节替换成 `�`（Unicode 替换字符），虽然会丢失部分信息，但至少不会崩溃。另一个选项是 `encoding_errors='ignore'`，直接跳过非法字节。两种策略各有适用场景：`replace` 适合你需要保留行数结构的场景（能看到哪里出了问题），`ignore` 适合你不关心丢失少量乱码字符的场景。

写出时同样需要注意编码。如果你的 CSV 包含中文并且要在 Excel 中打开，需要用 `utf-8-sig` 编码（带 BOM 的 UTF-8），否则 Excel 可能会用 ANSI 编码打开导致中文全部变成乱码：

```python
df.to_csv('output_for_excel.csv', index=False, encoding='utf-8-sig')
```

## JSONL：LLM 训练的原生格式

如果说 CSV 是通用数据交换格式，那 **JSONL（JSON Lines）就是 LLM 训练的原生格式**。几乎所有主流训练框架（Transformers、DeepSpeed、vLLM 等）都接受 JSONL 作为 SFT（Supervised Fine-Tuning）的标准输入格式。

### 为什么是 JSONL 而不是 JSON

JSONL 的规则很简单：每行是一个独立的 JSON 对象，行与行之间没有任何嵌套关系。这种设计带来的好处是**可以逐行流式处理**——你不需要把整个文件解析成一棵巨大的 JSON 树再操作，而是读一行处理一行，内存占用恒定。

一个典型的 SFT 训练数据长这样：

```jsonl
{"messages": [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好！有什么可以帮你的？"}]}
{"messages": [{"role": "user", "content": "解释注意力机制"}, {"role": "assistant", "content": "注意力机制是一种..."}]}
{"messages": [{"role": "user", "content": "如何学习Python"}, {"role": "assistant", "content": "建议从基础语法开始..."}]}
```

每行一条对话，每条对话包含 messages 数组，数组里按顺序排列 user 和 assistant 的消息。这种结构和 OpenAI / Anthropic 的 API 格式完全一致，所以从数据采集到模型训练的整个链路可以无缝衔接。

### 用 Pandas 读写 JSONL

```python
import pandas as pd

df = pd.read_json('training_data.jsonl', lines=True)
print(df.shape)
print(df.columns)
```

`lines=True` 这个参数至关重要——不加它的话 Pandas 会尝试把整个文件当作一个 JSON 数组来解析，对于几 GB 的 JSONL 文件来说这要么极慢要么直接 OOM。加上 `lines=True` 后 Pandas 采用逐行解析模式，效率和内存占用都完全不同。

写回 JSONL 同样简单：

```python
df.to_json('output.jsonl', orient='records', lines=True, force_ascii=False)
```

三个参数各有用途：`orient='records'` 让每行变成一个字典（标准 JSONL 格式），`lines=True` 确保每行一个 JSON 对象而非一个巨大数组，`force_ascii=False` 保留中文等非 ASCII 字符原样输出而不转义成 `\uXXXX`。

### 处理嵌套的 messages 字段

在实际项目中，你经常遇到的情况不是"读进来就完美可用"，而是 JSONL 里的某个字段本身又是一个 JSON 字符串（即"双重编码"）。这在从数据库导出或经过中间系统转存的场景里特别常见：

```python
import pandas as pd
import json

data = [
    {'id': 1, 'messages': json.dumps([{'role': 'user', 'content': 'hi'},
                                       {'role': 'assistant', 'content': 'hello'}])},
    {'id': 2, 'messages': json.dumps([{'role': 'user', 'content': 'explain AI'},
                                       {'role': 'assistant', 'content': 'AI is...'}])},
]

df = pd.DataFrame(data)

df['parsed'] = df['messages'].apply(json.loads)

def extract_content(messages, role):
    return [m['content'] for m in messages if m['role'] == role]

df['prompt'] = df['parsed'].apply(lambda x: extract_content(x, 'user'))
df['response'] = df['parsed'].apply(lambda x: extract_content(x, 'assistant'))

print(df[['id', 'prompt', 'response']])
```

这里的模式是：先用 `json.loads()` 把字符串形式的 JSON 解析成真正的 Python 对象，再用一个提取函数从中取出特定 role 的 content。这是处理 LLM 对话数据时**最高频的操作模式之一**，值得把它封装成一个可复用的函数。

## 大文件分块处理：chunksize 模式

当文件大到无法一次性装入内存时——比如一份 20GB 的 CSV 或者 8GB 的 JSONL——你就需要分块读取。Pandas 的 `read_csv()` 和 `read_json()` 都支持 `chunksize` 参数，返回的是一个迭代器，每次 yield 一个指定行数的 DataFrame 子集。

### 三种分块处理模式

分块处理不是简单地"把文件切成小块"，不同的任务目标对应不同的处理策略。

**模式一：分块统计聚合**。当你只需要最终的统计结果（均值、计数、分布）而不需要保留原始行时，可以在每个 chunk 内计算局部结果然后合并：

```python
import pandas as pd

CHUNK_SIZE = 500_000
total_rows = 0
quality_sum = 0.0
source_counts = pd.Series(dtype=int)

for chunk in pd.read_csv('huge_corpus.csv', chunksize=CHUNK_SIZE,
                          usecols=['quality_score', 'source']):
    total_rows += len(chunk)
    quality_sum += chunk['quality_score'].sum()
    source_counts = source_counts.add(chunk['source'].value_counts(), fill_value=0)
    del chunk

print(f"总行数: {total_rows:,}")
print(f"平均质量分: {quality_sum/total_rows:.2f}")
print(f"\n来源分布:")
print(source_counts.sort_values(ascending=False))
```

注意这里用 `Series.add()` 来累加各 chunk 的 value_counts 结果，而不是自己维护一个 dict——这样代码更简洁而且天然处理了"某个 chunk 里没有某个来源"的情况（`fill_value=0` 保证缺失 key 不会产生 NaN）。

**模式二：分块过滤收集**。当你需要筛选出满足条件的完整行并保留下来时：

```python
import pandas as pd

filtered_chunks = []

for chunk in pd.read_csv('corpus.csv', chunksize=200_000,
                          dtype={'quality_score': 'float32'}):
    good = chunk[chunk['quality_score'] >= 4.0].copy()
    if len(good) > 0:
        filtered_chunks.append(good)
    del chunk

final = pd.concat(filtered_chunks, ignore_index=True)
print(f"筛选后: {len(final):,} 行, 平均分: {final['quality_score'].mean():.2f}")
```

这里有一个容易踩的坑：筛选后的 `.copy()`。如果不加 `.copy()`，`good` 只是 `chunk` 的一个视图（view），当 `del chunk` 释放内存后 `good` 里的数据也会被销毁。`.copy()` 创建的是独立的副本，安全地脱离了原始 chunk 的生命周期。

**模式三：分块 ETL 管道**。当你需要读 → 处理 → 写出新格式的完整流水线时：

```python
import pandas as pd

INPUT = 'raw_conversations.jsonl'
OUT_DIR = '/tmp/processed/'

for i, chunk in enumerate(pd.read_json(INPUT, lines=True, chunksize=250_000)):
    clean = chunk[
        chunk['prompt'].notna() & (chunk['prompt'].str.len() > 5) &
        chunk['response'].notna() & (chunk['response'].str.len() > 10)
    ].copy()
    
    if len(clean) == 0:
        continue
    
    clean['cleaned_prompt'] = clean['prompt'].str.replace(r'<[^>]+>', '', regex=True).str.strip()
    clean['token_est'] = clean['cleaned_prompt'].str.len() // 4
    
    clean.to_parquet(f"{OUT_DIR}/chunk_{i:04d}.parquet", index=False)
    
    if (i+1) % 10 == 0:
        print(f"已处理 {(i+1)*250_000:,} 条")
```

这种模式在实际生产中最常见：从原始 JSONL 读入 → 清洗过滤 → 写出为 Parquet 格式供下游使用。每个 chunk 处理完立即写出并释放内存，理论上可以处理任意大小的文件（只要磁盘空间够）。

## 常见陷阱速查

最后总结几个 `read_csv()` 最高频的坑，每个都曾在生产环境中造成过实际问题：

**第一行被误认为列名**。如果 CSV 没有表头（或者表头不在第一行），必须显式指定：

```python
df = pd.read_csv('no_header.csv', header=None,
                 names=['col_a', 'col_b', 'col_c'])
df = pd.read_csv('header_at_row5.csv', header=4)
```

**千位分隔符导致数值变字符串**。有些导出数据会在数字里加逗号（如 `"1,234,567"`），Pandas 默认会把它当成字符串。加上 `thousands=','` 即可：

```python
df = pd.read_csv('numbers_with_commas.csv', thousands=',')
```

**混合型缺失值标记**。不同系统用不同方式表示"无数据"：`N/A`、`NA`、`null`、空字符串、`-`……`na_values` 参数可以把它们统一映射为 NaN：

```python
df = pd.read_csv('mixed_na.csv', na_values=['N/A', 'NA', 'null', '', '#N/A', '-'])
```

到这里我们已经覆盖了 CSV 和 JSONL 这两种最基本的文本格式。下一节我们要看的是如何直接从数据库读写数据——在很多企业级场景中，你的原始数据并不是文件形式存在的，而是躺在 PostgreSQL 或 MySQL 里等待被查询。
