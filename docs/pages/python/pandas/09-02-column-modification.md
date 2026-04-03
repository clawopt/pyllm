---
title: 新增/修改/删除列
description: 直接赋值 / assign() 方法 / drop() 删除、条件列生成、映射与替换
---
# 列修改与赋值


## 新增列：四种方式

### 方式一：直接赋值（最常用）

```python
import pandas as pd

df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})

df['c'] = [7, 8, 9]
df['d'] = df['a'] + df['b']
df['e'] = df['a'] * 2
df['is_large'] = df['a'] > 1

print(df)
```

### 方式二：assign()（链式友好）

```python
import pandas as pd

df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})

result = (
    df.assign(
        z=lambda d: d['x'] + d['y'],
        product=lambda d: d['x'] * d['y'],
        ratio=lambda d: d['x'] / d['y'],
        is_even=lambda d: d['x'] % 2 == 0,
    )
)

print(result)
```

**`assign()` 的优势**：
1. **不修改原 DataFrame**，返回新对象
2. **支持链式调用**
3. **可以在同一个 `assign` 中引用前面创建的列**

### 方式三：insert()（指定位置插入）

```python
import pandas as pd

df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})

df.insert(1, 'new_col', [10, 20, 30])
print(df)

df.insert(len(df.columns), 'last_col', ['X', 'Y', 'Z'])
```

## 条件列生成

### np.where()：向量化 if-else

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'score': [85, 42, 67, 91, 55, 78],
})

df['grade'] = np.where(df['score'] >= 70, 'pass', 'fail')
print(df[['score', 'grade']])

df['adjusted_score'] = np.where(
    df['score'] > 80,
    df['score'] * 1.05,  # 高分加分
    df['score'] * 0.95     # 低分扣分
)
print(df[['score', 'adjusted_score']])

df['tier'] = np.select(
    [
        (df['score'] >= 90),
        (df['score'] >= 75) & (df['score'] < 90),
        (df['score'] >= 60) & (df['score'] < 75),
        (df['score'] < 60),
    ],
    ['S', 'A', 'B', 'C', 'F'],
    default='Unknown'
)
print(df[['score', 'tier']])
```

### mask()：条件遮罩（保留原值或替换）

```python
import pandas as pd

df = pd.DataFrame({
    'value': [1, -5, 3, np.nan, 7],
})

df['cleaned'] = df['value'].mask(df['value'] < 0, other=-999)
print(df)
```

### case_when()：多条件分支（Pandas 3.0+）

```python
import pandas as pd

df = pd.DataFrame({
    'score': [88, 72, 45, 93, 61],
})

df['rating'] = pd.case_when(
    (df['score'] >= 90, 'A+'),
    (df['score'] >= 80, 'A'),
    (df['score'] >= 70, 'B'),
    (df['score'] >= 60, 'C'),
    (True, 'D')
)

print(df[['score', 'rating']])
```
```

## apply() 与向量化性能

### apply() 的正确使用场景

```python
import pandas as pd
import numpy as np
import time

n = 100_000
df = pd.DataFrame({
    'text': ['hello world'] * n,
    'value': np.random.randn(n),
})

start = time.time()
slow_result = df['text'].apply(lambda x: len(x.strip()))
t_slow = time.time() - start

start = time.time()
fast_result = df['text'].str.len()
t_fast = time.time() - start

print(f"apply: {t_slow:.3f}s")
print(f".str:  {t_fast:.3f}f ({t_slow/t_fast:.1f}x)")
```

### ✅ apply 的正确使用场景：复杂逻辑

```python
import pandas as pd

df = pd.DataFrame({
    'prompt': [
        '什么是注意力机制',
        '如何安装PyTorch',
        '解释BERT模型',
    ],
})

def analyze_complexity(text):
    """需要多步判断的复杂函数"""
    has_math = any(kw in text.lower() for kw in ['数学', '公式', '推导'])
    has_code = any(kw in text.lower() for kw in ['代码', '安装', 'API'])
    length_ok = len(text.strip()) > 10
    
    if has_code and has_math:
        return 'code+math'
    elif has_code:
        return 'code_only'
    elif has_math:
        return 'math_only'
    else:
        return 'other'

df['complexity_type'] = df['prompt'].apply(analyze_complexity)
print(df[['prompt', 'complexity_type']])
```

### ✅ apply 在 LLM 场景的核心用途：批量调用 API

```python
import pandas as pd
import numpy as np
import json
from openai import OpenAI

client = OpenAI()

df = pd.DataFrame({
    'prompt': [
        '什么是注意力机制',
        '解释 Python GIL',
        '比较 BERT 和 GPT',
    ] * 100  # 300 个样本
})

def call_llm_for_sentiment(prompt_text):
    """调用 LLM 分析情感倾向"""
    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[{'role': 'user', 'content': f'判断以下文本的情感倾向(positive/negative/neutral):\\n{prompt_text}'}],
        temperature=0.0,
        max_tokens=10,
    )
    return response.choices[0].message.content.strip()


print("开始批量调用 LLM 进行情感分析...")
start = time.time()

sentiments = df['prompt'].apply(call_llm_for_sentiment)

elapsed = time.time() - start
total_cost = sum(s.usage.total_tokens for s in sentiments) * 0.00015

df['llm_sentiment'] = sentiments
df['api_latency_ms'] = [s.usage.total_tokens for s in sentiments]

print(f"\n完成！耗时: {elapsed:.1f}s")
print(f"总 Token 数: {sum(s.usage.total_tokens for s in sentiments):,}")
print(f'估算费用: ${total_cost:.4f}')
print(f"\n情感分布:")
print(df['llm_sentiment'].value_counts())
```

## map() / replace()：映射与替换

```python
import pandas as pd

df = pd.DataFrame({
    'category': ['tech', 'finance', 'tech', 'medical', 'legal'],
    'label': ['A', 'B', 'A', 'C', 'D'],
})

category_map = {
    'tech': '技术',
    'finance': '金融',
    'medical': '医疗',
    'legal': '法律'
}
df['category_cn'] = df['category'].map(category_map)
print(df[['category', 'category_cn']])

label_map = {'A': '优秀', 'B': '良好', 'C': '及格', 'D': '待改进'}
df['label_cn'] = df['label'].replace(label_map)
print(df[['label', 'label_cn']])

df['cleaned_category'] = df['category'].str.replace(r'\s+', '-', regex=True)
```

## 删除列

```python
import pandas as pd

df = pd.DataFrame({
    'id': range(5),
    'keep_me': ['a', 'b', 'c', 'd', 'e'],
    'drop_me_1': [1, 2, 3, 4, 5],
    'drop_me_2': ['x', 'y', 'z', 'w', 'v'],
})

dropped = df.drop(columns=['drop_me_1', 'drop_me_2'])

dropped_v2 = df.drop(columns=['keep_me', 'drop_me_1', 'drop_me_2'], errors='ignore')

df_drop_inplace = df.copy()
df_drop_inplace.drop(columns=['id'], inplace=True)
```
