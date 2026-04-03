---
title: 文本格式：CSV 与 JSON
description: read_csv/to_csv 参数详解与常见陷阱、JSON/JSONL 读写、大文件分块读取 chunksize
---
# 文本格式读写：CSV 与 JSON


## CSV：最基础也最重要的格式

### 为什么 CSV 在 LLM 开发中无处不在

CSV（Comma-Separated Values）是数据交换的**通用语**——几乎所有系统都能导出 CSV，几乎所有工具都能读取 CSV。

在大模型开发中，你会在这些场景遇到 CSV：

| 场景 | 来源 | 典型大小 |
|------|------|---------|
| 用户对话日志导出 | 产品数据库 / BI 工具 | 100K - 50M 行 |
| 标注团队交付 | Excel 导出转 CSV | 1K - 500K 行 |
| API 调用记录 | 监控平台导出 | 1M - 100M 行 |
| 开源数据集 | Kaggle / HuggingFace | 10K - 10M 行 |

### read_csv() 基础用法

```python
import pandas as pd

df = pd.read_csv('data.csv')

print(f"形状: {df.shape}")
print(f"列名: {list(df.columns)}")
print(df.head(3))
```

### 必知参数详解

#### 指定列与类型（**性能关键**）

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

**为什么这很重要？**

```python
import pandas as pd
import numpy as np
import time

n = 5_000_000
tmp_path = '/tmp/benchmark.csv'

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

start = time.time()
df_select = pd.read_csv(
    tmp_path,
    usecols=['id', 'text', 'value', 'category'],
    dtype={'id': 'int32', 'text': 'string', 'value': 'float32',
           'category': 'category'},
)
t_select = time.time() - start

mem_all = df_all.memory_usage(deep=True).sum() / 1024**2
mem_select = df_select.memory_usage(deep=True).sum() / 1024**2

print(f"{'方式':<12} {'时间':>8} {'内存':>10}")
print(f"{'全量读取':<12} {t_all:>7.2f}s {mem_all:>9.1f} MB")
print(f"{'选择+指定type':<12} {t_select:>7.2f}s {mem_select:>9.1f} MB")
print(f"加速: {t_all/t_select:.1f}x, 省内存: {(1-mem_select/mem_all)*100:.0f}%")
```

典型结果：

```
方式           时间      内存
全量读取        3.45s   1142.3 MB
选择+指定type   1.12s    286.7 MB
加速: 3.1x, 省内存: 75%
```

#### 处理分隔符问题

```python
import pandas as pd

df_tsv = pd.read_csv('data.tsv', sep='\t')

df_pipe = pd.read_csv('data.psv', sep='|')

df_auto = pd.read_csv('unknown_format.csv', sep=None, engine='python')

csv_with_commas = """name,age,city
"Alice, Jr.",25,"New York, NY"
"Bob, O'Brien",30,"London, UK"
"""

from io import StringIO
df_escaped = pd.read_csv(StringIO(csv_with_commas))
print(df_escaped)
```

#### 编码问题（大模型多语言场景高频坑）

```python
import pandas as pd

df_utf8 = pd.read_csv('multilingual.csv', encoding='utf-8')

df_gbk = pd.read_csv('chinese_export.csv', encoding='gbk')

df_latin = pd.read_csv('european.csv', encoding='latin-1')

try:
    df = pd.read_csv('unknown_encoding.csv')
except UnicodeDecodeError:
    df = pd.read_csv('unknown_encoding.csv',
                     encoding_errors='replace')  # 用 � 替换无法解码的字符
    df = pd.read_csv('unknown_encoding.csv',
                     encoding_errors='ignore')   # 直接跳过非法字节
```

### to_csv() 写出控制

```python
import pandas as pd

df = pd.DataFrame({'a': [1, 2], 'b': ['x', 'y']})

df.to_csv('output.csv', index=False)

df.to_csv(
    'sft_train.csv',
    index=False,                    # 不写行索引
    encoding='utf-8',              # 明确编码
    date_format='%Y-%m-%d %H:%M:%S', # 时间格式
    float_format='%.4f',           # 浮点精度
    quoting=1,                      # 所有字符串加引号
)

df.to_csv('output_for_excel.csv', index=False, encoding='utf-8-sig')
```

## JSON / JSONL：结构化数据的自然选择

### 为什么 LLM 数据偏爱 JSONL

SFT 训练的标准输入格式是 **JSONL**（每行一个 JSON 对象）：

```jsonl
{"messages": [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好！有什么可以帮你的？"}]}
{"messages": [{"role": "user", "content": "解释注意力机制"}, {"role": "assistant", "content": "注意力机制是一种..."}]}
```

这种格式天然适合逐行流式处理。

### read_json() 基础用法

```python
import pandas as pd

df_array = pd.read_json('array_format.json')

df_lines = pd.read_json('conversations.jsonl', lines=True)

df_records = pd.read_json('records.json', orient='records')
```

### 大模型数据集的 JSONL 读写最佳实践

```python
import pandas as pd
import json

df = pd.read_json('training_data.jsonl', lines=True)

df = pd.read_json(
    'training_data.jsonl',
    lines=True,
    convert_dates=['created_at'],
)

chunk_iter = pd.read_json('huge_dataset.jsonl', lines=True, chunksize=100_000)
for i, chunk in enumerate(chunk_iter):
    processed = process_chunk(chunk)
    processed.to_json(f'output/chunk_{i:04d}.jsonl', orient='records', lines=True)
    if i % 10 == 0:
        print(f"已处理 {(i+1)*100_000:,} 行")


df.to_json('output.jsonl', orient='records', lines=True, force_ascii=False)

def write_jsonl_in_chunks(df, output_path, chunk_size=50_000):
    with open(output_path, 'w', encoding='utf-8') as f:
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i+chunk_size]
            records = chunk.to_dict(orient='records')
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
            if (i // chunk_size) % 20 == 0:
                print(f"已写入 {min(i+chunk_size, len(df)):,} 条")

write_jsonl_in_chunks(df, 'output/sft_train.jsonl')
```

### 处理嵌套 JSON 字段

LLM 数据中经常出现嵌套的 messages 字段：

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

df['parsed_messages'] = df['messages'].apply(json.loads)

def extract_roles(messages, role):
    return [m['content'] for m in messages if m['role'] == role]

df['prompt'] = df['parsed_messages'].apply(lambda x: extract_roles(x, 'user'))
df['response'] = df['parsed_messages'].apply(lambda x: extract_roles(x, 'assistant'))

print(df[['id', 'prompt', 'response']])
```

### normalize_json_meta() 处理复杂嵌套

```python
import pandas as pd

complex_json = """
{
    "model": "GPT-4o",
    "results": [
        {"question": "Q1", "answer": "A1", "score": 0.95},
        {"question": "Q2", "answer": "A2", "score": 0.88}
    ],
    "metadata": {"version": "1.0", "date": "2025-01-15"}
}
"""

from io import StringIO
df_nested = pd.read_json(StringIO(complex_json))

results_flat = pd.json_normalize(df_nested['results'])
print(results_flat)
```

## 大文件分块处理：chunksize 模式

### 为什么需要分块

当文件大小超过可用内存时，必须分块读取：

```python
import pandas as pd


CHUNK_SIZE = 100_000

results = []
for i, chunk in enumerate(pd.read_csv('very_large_file.csv', chunksize=CHUNK_SIZE)):
    print(f"处理第 {i+1} 块 ({len(chunk):,} 行)")
    
    filtered = chunk[chunk['quality'] >= 3]
    cleaned = filtered.dropna(subset=['response'])
    
    results.append(cleaned[['id', 'prompt', 'response']])
    
    del chunk, filtered, cleaned  # 释放内存

final_df = pd.concat(results, ignore_index=True)
print(f"\n最终结果: {len(final_df):,} 行")
```

### 分块处理的性能优化技巧

```python
import pandas as pd

chunk_iter = pd.read_csv(
    'huge_file.csv',
    chunksize=200_000,
    usecols=['id', 'text', 'label', 'score'],
    dtype={'id': 'string', 'text': 'string', 'label': 'category', 'score': 'float32'},
)

total_by_label = pd.Series(dtype='float64')
count_by_label = pd.Series(dtype='int64')

for chunk in chunk_iter:
    agg = chunk.groupby('label')['score'].agg(['sum', 'count'])
    total_by_label = total_by_label.add(agg['sum'], fill_value=0)
    count_by_label = count_by_label.add(agg['count'], fill_value=0)

avg_by_label = (total_by_label / count_by_label).round(2)
print(avg_by_label.sort_values(ascending=False))

with pd.ExcelWriter('output_by_category.xlsx', engine='openpyxl') as writer:
    for label in ['A', 'B', 'C', 'D']:
        chunk_iter = pd.read_csv(
            'huge_file.csv',
            chunksize=500_000,
            usecols=['id', 'text', 'label'],
        )
        category_dfs = []
        for chunk in chunk_iter:
            cat_chunk = chunk[chunk['label'] == label]
            if len(cat_chunk) > 0:
                category_dfs.append(cat_chunk)
        
        if category_dfs:
            pd.concat(category_dfs).to_excel(
                writer, sheet_name=label, index=False
            )
            print(f"{label}: {sum(len(d) for d in category_dfs):,} 行")
```

## 常见陷阱与解决方案

### 陷阱 1：第一行被误认为列名

```python
df = pd.read_csv('no_header.csv', header=None,
                 names=['col_a', 'col_b', 'col_c'])

df = pd.read_csv('header_in_row_5.csv', header=4)
```

### 陷阱 2：千位分隔符导致数值变字符串

```python
df = pd.read_csv('numbers_with_commas.csv', thousands=',')
```

### 陷阱 3：混合类型的列

```python
df = pd.read_csv('mixed_type.csv', na_values=['N/A', 'NA', 'null', '', '#N/A'])
```

### 陷阱 4：路径中的中文或空格

```python
import os

path = r'/Users/张三/data/训练数据.csv'
df = pd.read_csv(path)

from pathlib import Path
path = Path('~/数据文件/corpus.csv').expanduser()
df = pd.read_csv(path)
```
