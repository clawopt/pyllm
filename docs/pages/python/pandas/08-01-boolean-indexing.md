---
title: 布尔索引：基于条件的行筛选
description: 布尔索引基础语法、多条件组合（AND/OR/NOT）、链式索引 vs .loc 的最佳实践
---
# 布尔索引与条件筛选


## 布尔索引的本质

布尔索引是 Pandas 中**最常用也最强大**的行选择方式：

```python
import pandas as pd

df = pd.DataFrame({
    'model': ['GPT-4o', 'Claude', 'Llama', 'Gemini', 'Qwen'],
    'score': [88.7, 89.2, 84.5, 86.8, 83.5],
    'context': [128000, 200000, 131072, 1000000, 131072],
    'vision': [True, True, False, True, False],
})

mask = df['score'] > 85
print(mask)

high_score = df[mask]
print(high_score[['model', 'score']])
```

## 单条件筛选

### 数值条件

```python
import pandas as pd
import numpy as np

np.random.seed(42)
n = 100_000

df = pd.DataFrame({
    'prompt_id': range(n),
    'quality': np.round(np.random.uniform(1, 5, n), 1),
    'tokens': np.random.randint(20, 2000, n),
    'response_time_ms': np.random.exponential(500, n).astype(int),
    'has_code': np.random.choice([True, False], n),
})

long_responses = df[df['tokens'] > 1000]
print(f"长回复 (>1000 tokens): {len(long_responses):,}")

short_responses = df[df['tokens'] < 100]

exact_256 = df[df['tokens'] == 256]

medium_length = df[df['tokens'].between(200, 800)]
print(f"中等长度 (200-800): {len(medium_length):,}")
```

### 文本条件

```python
import pandas as pd

df = pd.DataFrame({
    'text': [
        '什么是注意力机制',
        '如何安装PyTorch',
        'ERROR: connection timeout',
        '',
        '<html>页面</html>',
        '   多余空格   ',
    ]
})

has_error = df[df['text'].str.contains('ERROR', na=False)]

starts_with = df[df['text'].str.startswith('什么')]

ends_with = df[df['text'].str.endswith('机制')]

matches_pattern = df[df['text'].str.contains(r'注意|PyTorch|HTML', regex=True)]
```

### 分类/布尔条件

```python
import pandas as pd

df = pd.DataFrame({
    'source': ['api', 'web', 'export', 'api', None, 'mobile'],
    'active': [True, True, False, True, True, False],
})

api_data = df[df['source'] == 'api']

not_export = df[df['source'] != 'export']

active_only = df[df['active'] == True]
inactive = df[~df['active']]  # 取反操作符 ~

selected_sources = ['api', 'web']
in_list = df[df['source'].isin(selected_sources)]
print(in_list)

not_in_list = df[~df['source'].isin(selected_sources)]

not_null = df[df['source'].notna()]
is_null = df[df['source'].isna()]

valid_and_active = df[df['source'].notna() & df['active']]
```

## 多条件组合

### AND 条件（同时满足）

```python
import pandas as pd
import numpy as np

np.random.seed(42)
n = 50_000

df = pd.DataFrame({
    'quality': np.round(np.random.uniform(1, 5, n), 1),
    'tokens': np.random.randint(20, 2000, n),
    'source': np.random.choice(['api', 'web', 'export'], n),
    'model': np.random.choice(['GPT-4o', 'Claude', 'Llama'], n),
    'has_code': np.random.choice([True, False], n),
})

complex_filter = df[
    (df['quality'] >= 4.0) &
    (df['tokens'].between(300, 1500)) &
    (df['source'] == 'api') &
    (df['has_code'] == True)
]

print(f"复合筛选结果: {len(complex_filter):,} ({len(complex_filter)/len(df)*100:.2f}%)")
```

**关键规则**：每个条件必须用括号包裹，运算符用 `&`（AND）、`|`（OR）、`~`（NOT）。

### OR 条件（满足任一）

```python
api_or_web = df[df['source'].isin(['api', 'web'])]

extreme_quality = df[(df['quality'] >= 4.8) | (df['quality'] <= 2.0)]

keywords = ['注意力', 'Transformer', 'PyTorch']
any_keyword = df[
    df['text'].str.contains(keywords[0], na=False) |
    df['text'].str.contains(keywords[1], na=False) |
    df['text'].str.contains(keywords[2], na=False)
]
any_keyword_v2 = df[df['text'].str.contains('|'.join(keywords), regex=True, na=False)]
```

### NOT 条件（取反）

```python
not_api = df[df['source'] != 'api']
not_api_v2 = df[~(df['source'] == 'api')]  # 等价，更推荐

neither_api_nor_export = df[~df['source'].isin(['api', 'export'])]

q1 = df['quality'].quantile(0.25)
q3 = df['quality'].quantile(0.75)
iqr = q3 - q1
normal_quality = df[
    (df['quality'] >= q1 - 1.5 * iqr) &
    (df['quality'] <= q3 + 1.5 * iqr)
]
print(f"IQR 过滤后: {len(normal_quality):,} (去除 {len(df)-len(normal_quality):,} 异常值)")
```

## LLM 场景实战：SFT 数据集构建的筛选流水线

```python
import pandas as pd
import numpy as np

np.random.seed(42)
n = 200_000

raw = pd.DataFrame({
    'conversation_id': [f'conv_{i:06d}' for i in range(n)],
    'user_message': [f'问题{i%500}' for i in range(n)],
    'assistant_message': [f'回答{i}' for i in range(n)] +
                       [None] * int(n*0.02) +
                       [''] * int(n*0.01),
    'turn_count': np.random.choice([None] + list(range(1, 10)), n),
    'quality_score': np.round(
        np.random.choice([None] + list(np.arange(1.0, 5.1, 0.5)), n), 1
    ),
    'token_count': np.random.randint(10, 4000, n),
    'source': np.random.choice([None, 'api', 'web', 'export'], n),
    'model': np.random.choice(['GPT-4o', 'Claude', 'Llama'], n),
})


def build_sft_dataset(df, config=None):
    """SFT 训练数据集的多级筛选"""
    
    if config is None:
        config = {
            'min_quality': 3.5,
            'min_tokens': 30,
            'max_tokens': 4096,
            'min_turns': 1,
            'allowed_sources': ['api', 'web'],
            'exclude_patterns': ['ERROR', 'HTTP', '<html>'],
            'max_text_ratio': 5.0,  # response/prompt 最大长度比
        }
    
    result = df.copy()
    original = len(result)
    log = []
    
    m1 = result['user_message'].notna()
    m2 = result['assistant_message'].notna()
    m3 = result['assistant_message'].str.len() > 0
    m4 = result['quality_score'].notna()
    basic = m1 & m2 & m3 & m4
    result = result[basic].copy()
    log.append(f"Stage 1 完整性: {original} → {len(result)} "
               f"(去除 {original-len(result)})")
    
    if 'quality_score' in result.columns:
        m_qual = result['quality_score'] >= config['min_quality']
        result = result[m_qual].copy()
        log.append(f"Stage 2 质量(>={config['min_quality']}): {len(df)} → {len(result)}")
    
    m_len = (
        result['token_count'].between(config['min_tokens'], config['max_tokens'])
    )
    result = result[m_len].copy()
    log.append(f"Stage 3 长度({config['min_tokens']}-{config['max_tokens']}): "
               f"{len(df)} → {len(result)}")
    
    if 'source' in result.columns and config.get('allowed_sources'):
        m_src = result['source'].isin(config['allowed_sources'])
        result = result[m_src].copy()
        log.append(f"Stage 4 来源({config['allowed_sources']}): {len(df)} → {len(result)}")
    
    if config.get('exclude_patterns'):
        pattern = '|'.join(config['exclude_patterns'])
        m_clean = ~result['user_message'].str.contains(pattern, case=False, na=False)
        m_clean &= ~result['assistant_message'].str.contains(pattern, case=False, na=False)
        result = result[m_clean].copy()
        log.append(f"Stage 5 内容清洗: {len(df)} → {len(result)}")
    
    if 'token_count' in result.columns:
        prompt_len = result['user_message'].str.len() // 4
        resp_len = result['assistant_message'].str.len() // 4
        ratio = prompt_len / resp_len.clip(lower=1)
        m_ratio = ratio < config['max_text_ratio']
        result = result[m_ratio].copy()
        log.append(f"Stage 6 比例(<{config['max_text_ratio']}x): {len(df)} → {len(result)}")
    
    final = len(result)
    print("\n" + "=" * 50)
    print("  SFT 数据集筛选报告")
    print("=" * 50)
    for entry in log:
        print(f"  {entry}")
    print("-" * 50)
    print(f"  最终: {original:,} → {final:,} "
          f"(保留率: {final/original*100:.1f}%, 去除 {original-final:,})")
    
    return result.reset_index(drop=True)


sft_data = build_sft_dataset(raw)
```
