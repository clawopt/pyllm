---
title: 字符串处理
description: .str 访问器的 contains/replace/extract/split 方法，正则表达式在大模型数据清洗中的应用，HTML 标签过滤与特殊字符处理
---
# 字符串处理技巧


## .str 访问器：Pandas 的字符串工具箱

Pandas 的 `.str` 访问器将 Python 字符串方法**向量化**到整列操作：

```python
import pandas as pd

df = pd.DataFrame({
    'text': [
        'Hello <b>World</b>!',
        'Email: test@example.com',
        'Phone: 138-1234-5678',
        '  多余空格  ',
        None,
    ]
})

print(df['text'].str.len())           # 长度
print(df['text'].str.lower())         # 小写
print(df['text'].str.upper())         # 大写
print(df['text'].str.strip())         # 去首尾空白
print(df['text'].str.replace(' ', '_'))  # 替换
```

## contains()：模式匹配

### 基础用法

```python
import pandas as pd

df = pd.DataFrame({
    'prompt': [
        '什么是注意力机制',
        '如何安装PyTorch',
        '解释BERT模型',
        'GPT-4和Claude哪个好',
        'ERROR: 超时',
        '',
    ]
})

has_ai = df[df['prompt'].str.contains('AI', case=False, na=False)]
print(has_ai)

keywords = ['注意', 'BERT', 'GPT']
matched = df[df['prompt'].str.contains('|'.join(keywords), na=False)]
print(matched)

no_error = df[~df['prompt'].str.contains('ERROR', na=False)]
print(no_error)
```

### LLM 场景：过滤异常数据

```python
import pandas as pd
import numpy as np

n = 50_000
np.random.seed(42)

prompts = [
    f'{"解释" if i%10==0 else "什么是"}{["AI","Python","深度学习","NLP","RL"][i%5]}'
    for i in range(n)
] + [
    'ERROR: connection timeout',
    'HTTP 500 Internal Server Error',
    '<html><body>页面未找到</body></html>',
    '\x00\x01\x02 二进制数据',
    '   ',  # 纯空白
]

responses = [f'这是关于{p}的回答' for p in prompts[:len(prompts)-6]] + \
            [None, '', 'bad data', 'more bad', 'empty', '']

df = pd.DataFrame({'prompt': prompts, 'response': responses})

mask = (
    df['prompt'].notna() &
    (df['prompt'].str.len() >= 3) &                    # 长度足够
    (~df['prompt'].str.contains(r'^\s*$', regex=True)) &  # 非纯空白
    (~df['prompt'].str.contains(r'ERROR|HTTP \d+|<html>', regex=True, case=False)) &  # 无错误标记
    (df['response'].notna() & (df['response'].str.len() >= 5))  # 回复有效
)

clean = df[mask].copy()
print(f"原始: {len(df):,} → 清洗后: {len(clean):,} ({len(clean)/len(df)*100:.1f}%)")
```

## replace()：替换与清理

### 基础替换

```python
import pandas as pd

s = pd.Series(['Hello World', 'foo bar baz', 'test@example.com'])

print(s.str.replace(' ', '-'))

print(s.str.replace(r'\s+', '_', regex=True))
```

### LLM 数据清洗实战：多步替换流水线

```python
import pandas as pd
import re

def clean_llm_text(series):
    """LLM 文本数据的多步清洗"""
    
    result = series.astype(str).copy()
    
    result = result.str.replace(r'<[^>]+>', '', regex=True)
    
    result = result.str.replace(r'[#*_`|]', '', regex=True)
    
    result = result.str.replace(r'\s+', ' ', regex=True).str.strip()
    
    zero_width = r'[\u200b-\u200f\u2028-\u2029\ufeff]'
    result = result.str.replace(zero_width, '', regex=True)
    
    control_chars = r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]'
    result = result.str.replace(control_chars, '', regex=True)
    
    return result


raw_texts = pd.Series([
    '<div class="msg">你好 <b>世界</b>！</div>',
    '**加粗文字** 和 `代码块`',
    '正常\t\t文本\n\n内容',
    '包含\u200b零宽字符\u200c的内容',
    'ERROR\x00二进制数据\x07响铃',
])

cleaned = clean_llm_text(raw_texts)
for raw, clean in zip(raw_texts, cleaned):
    print(f"原始: {raw}")
    print(f"清洗: {clean}")
    print()
```

## extract()：正则提取

### 提取结构化信息

```python
import pandas as pd

df = pd.DataFrame({
    'text': [
        '用户 alice@test.com 在 2025-03-15 提问了关于 GPT-4 的问题',
        '联系 bob.smith@company.co.uk 了解更多',
        '电话: +86-138-1234-5678 或发邮件至 support@llm.ai',
    ]
})

df['email'] = df['text'].str.extract(
    r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'
)
print(df[['text', 'email']])

df[['date_str', 'topic']] = df['text'].str.extract(
    r'(\d{4}-\d{2}-\d{2}).*?([A-Z][A-Za-z0-9-]+)'
)
print(df[['text', 'date_str', 'topic']])

urls = pd.Series([
    '访问 https://example.com 和 http://test.org',
    '没有链接',
    'ftp://files.example.com/path/to/file.pdf',
])

all_urls = urls.str.findall(r'https?://[^\s]+')
print(all_urls)
```

### LLM 场景：从对话中提取实体

```python
import pandas as pd

conversations = pd.DataFrame({
    'chat': [
        '用户张三(13812345678)询问了关于GPT-4o的价格问题，预算是$5000',
        '李四(zhangsan@company.com)想了解Claude的使用限制',
        '王五在2025年3月15日提交了反馈，评分4.5/5.0',
    ]
})

entities = conversations['chat'].str.extractall(
    r'(?P<name>[\u4e00-\u9fff]{2,3})'     # 中文名
    r'|(?P<phone>\d{3}-\d{4}-\d{4})'      # 电话
    r'|(?P<email>[\w.+-]+@[\w.-]+\.\w+)'    # 邮箱
    r'|(?P<price>\$\d+(?:,\d{3})*)'          # 价格
    r'|(?P<score>\d+\.?\d*)/\d+\.?\d*'       # 评分
)

print(entities)
```

## split()：拆分

```python
import pandas as pd

s = pd.Series([
    'apple,banana,cherry',
    'dog;cat;mouse',
    'one|two|three|four',
])

print(s.str.split(','))

print(s.split(r'[;,|]'))

print(s.str.split(',', n=1, expand=True))

token_series = pd.Series(['The quick brown fox', 'Hello world'])
token_df = token_series.str.split(expand=True)
token_df.columns = [f'token_{i}' for i in range(token_df.shape[1])]
print(token_df)
```

## 性能对比：向量化 vs apply

```python
import pandas as pd
import numpy as np
import time

n = 1_000_000
texts = ['sample text with some words and numbers like 123 and 456'] * n

s = pd.Series(texts)

start = time.time()
result_apply = s.apply(lambda x: len(x.strip().replace(' ', '')))
t_apply = time.time() - start

start = time.time()
result_vectorized = s.str.len()  # 不需要 strip/replace 时直接用内置方法
t_vec = time.time() - start

print(f"apply:   {t_apply:.3f}s")
print(f".str:    {t_vec:.3f}s")
print(f"加速比:  {t_apply/t_vec:.1f}x")

start = time.time()
result_replace_apply = s.apply(lambda x: x.lower().replace('sample', 'demo'))
t_r_apply = time.time() - start

start = time.time()
result_replace_str = s.str.lower().str.replace('sample', 'demo')
t_r_str = time.time() - start

print(f"\nreplace apply: {t_r_apply:.3f}s")
print(f"replace .str:  {t_r_str:.3f}s")
print(f"加速比:       {t_r_apply/t_r_str:.1f}x")
```
