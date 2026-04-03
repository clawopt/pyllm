---
title: apply() 深度解析与向量化操作
description: apply() 性能对比、向量化替代方案、transform()、agg() 在 Series 上的用法
---
# apply 与向量化


## apply() 的三种调用模式

### 模式一：Series.apply() —— 逐元素处理

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'prompt': [
        '什么是Transformer',
        '如何训练LLM',
        'BERT和GPT区别',
        '解释注意力机制',
    ],
})

df['char_count'] = df['prompt'].apply(len)

df['first_word'] = df['prompt'].apply(lambda x: x.split()[0])

print(df[['prompt', 'char_count', 'first_word']])
```

**Series.apply() 的本质**：对 Series 的每个元素执行函数，返回等长的 Series 或标量。

### 模式二：DataFrame.apply() —— 逐列或逐行处理

```python
import pandas as pd
import numpy as np

np.random.seed(42)

df = pd.DataFrame({
    'accuracy': [0.92, 0.88, 0.95, 0.78, 0.91],
    'recall': [0.85, 0.90, 0.88, 0.72, 0.89],
    'f1': [0.88, 0.89, 0.91, 0.75, 0.90],
}, index=['GPT-4o', 'Claude', 'Llama', 'Qwen', 'DeepSeek'])

col_ranges = df.apply(lambda col: col.max() - col.min())
print("各指标极差:")
print(col_ranges)

df['avg_score'] = df.apply(lambda row: row.mean(), axis=1)
df['best_metric'] = df.apply(lambda row: row.idxmax(), axis=1)

print("\n按模型汇总:")
print(df)
```

### 模式三：DataFrame.applymap() / DataFrame.map()（Pandas 2.1+）

```python
import pandas as pd

df = pd.DataFrame({
    'A': [1.2345, 6.7890],
    'B': [2.3456, 7.8901],
})

rounded = df.map(lambda x: round(x, 2))
print(rounded)

formatted = df.map(lambda x: f'{x:.2f}')
print(formatted)
```

## 向量化 vs apply()：性能真相

### 基准测试框架

```python
import pandas as pd
import numpy as np
import time

n = 100_000
df = pd.DataFrame({
    'a': np.random.randn(n),
    'b': np.random.randn(n),
    'text': ['hello world python pandas'] * n,
})
```

### 测试一：数值运算

```python
def bench_numeric():
    start = time.time()
    r_apply = df.apply(lambda row: row['a'] + row['b'], axis=1)
    t_apply = time.time() - start

    start = time.time()
    r_vec = df['a'] + df['b']
    t_vec = time.time() - start

    print(f"数值加法 — apply: {t_apply:.4f}s | 向量化: {t_vec:.4f}s | 加速: {t_apply/t_vec:.1f}x")

bench_numeric()
```

### 测试二：字符串长度

```python
def bench_string_len():
    start = time.time()
    r_apply = df['text'].apply(len)
    t_apply = time.time() - start

    start = time.time()
    r_str = df['text'].str.len()
    t_str = time.time() - start

    print(f"字符串长度 — apply: {t_apply:.4f}s | .str: {t_str:.4f}s | 加速: {t_apply/t_str:.1f}x")

bench_string_len()
```

### 测试三：条件判断

```python
def bench_conditional():
    start = time.time()
    r_apply = df['a'].apply(lambda x: 'high' if x > 0 else 'low')
    t_apply = time.time() - start

    start = time.time()
    r_where = np.where(df['a'] > 0, 'high', 'low')
    t_where = time.time() - start

    start = time.time()
    r_cat = pd.Categorical(np.where(df['a'] > 0, 'high', 'low'))
    t_cat = time.time() - start

    print(f"条件分支 — apply: {t_apply:.4f}s | np.where: {t_where:.4f}s | Categorical: {t_cat:.4f}s")

bench_conditional()
```

### 性能总结表

| 操作类型 | apply() | 向量化替代 | 典型加速比 |
|---------|---------|-----------|-----------|
| 数值加减乘除 | `apply(lambda r: r.a+r.b)` | `df.a + df.b` | **50-200x** |
| 字符串长度 | `.apply(len)` | `.str.len()` | **3-15x** |
| 条件判断 | `.apply(lambda x: ...)` | `np.where()` / `pd.case_when()` | **10-100x** |
| 字符串包含 | `.apply(lambda x: 'kw' in x)` | `.str.contains('kw')` | **5-20x** |
| 正则提取 | `.apply(re.extract)` | `.str.extract()` | **3-10x** |
| 复杂多步逻辑 | `.apply(custom_func)` | 无直接替代 | **1x（必须用）** |

**核心原则**：能用内置向量化方法就绝不用 `apply()`。只有当逻辑无法用单行表达式表达时，才考虑 `apply()`。

## transform()：保持形状的分组变换

```python
import pandas as pd
import numpy as np

np.random.seed(42)

df = pd.DataFrame({
    'model': ['GPT-4o']*50 + ['Claude']*50 + ['Llama']*50,
    'latency_ms': np.concatenate([
        np.random.normal(800, 150, 50),
        np.random.normal(650, 120, 50),
        np.random.normal(350, 80, 50),
    ]),
})

df['z_score'] = df.groupby('model')['latency_ms'].transform(
    lambda x: (x - x.mean()) / x.std()
)

df['pct_rank'] = df.groupby('model')['latency_ms'].transform(
    lambda x: x.rank(pct=True)
)

df['deviation_from_mean'] = df.groupby('model')['latency_ms'].transform(
    lambda x: abs(x - x.mean())
)

print("=== 各模型延迟标准化结果 ===")
print(df.groupby('model')[['latency_ms', 'z_score', 'pct_rank']].head(3))
```

**`transform()` vs `apply()` 的关键区别**：

| 特性 | transform() | apply() |
|-----|------------|---------|
| 返回形状 | 必须与输入同长 | 可以是任意形状 |
| 结果对齐 | 自动索引对齐 | 需要手动处理 |
| 典型用途 | 标准化/填充/排名 | 聚合/汇总 |

## LLM 场景：批量 API 调用的完整模式

### 带重试和限流的 apply 模板

```python
import pandas as pd
import time
from openai import OpenAI

client = OpenAI()

def safe_llm_call(prompt, max_retries=3, base_delay=1.0):
    """带指数退避重试的 LLM 调用"""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model='gpt-4o-mini',
                messages=[{'role': 'user', 'content': prompt}],
                temperature=0.0,
                max_tokens=50,
            )
            return {
                'result': response.choices[0].message.content.strip(),
                'tokens': response.usage.total_tokens,
                'status': 'success',
                'attempts': attempt + 1,
            }
        except Exception as e:
            if attempt < max_retries - 1:
                wait = base_delay * (2 ** attempt)
                time.sleep(wait)
            else:
                return {
                    'result': str(e),
                    'tokens': 0,
                    'status': f'failed: {type(e).__name__}',
                    'attempts': max_retries,
                }

df = pd.DataFrame({
    'prompt': [
        '用一句话解释什么是 RAG',
        '什么是 LoRA 微调',
        'KV Cache 是什么',
        '什么是 RLHF',
    ] * 25,
})

results_df = df['prompt'].apply(safe_llm_call).apply(pd.Series)

final_df = pd.concat([df, results_df], axis=1)

print(final_df.head(8))
```

### 并发批处理（asyncio）

```python
import pandas as pd
import asyncio
import aiohttp
import nest_asyncio
nest_asyncio.apply()

async def batch_llm_async(prompts, batch_size=10):
    """异步并发调用 LLM API"""
    async def call_one(session, prompt):
        async with session.post(
            'https://api.openai.com/v1/chat/completions',
            json={
                'model': 'gpt-4o-mini',
                'messages': [{'role': 'user', 'content': prompt}],
                'temperature': 0.0,
                'max_tokens': 30,
            },
            headers={'Authorization': f'Bearer {API_KEY}'}
        ) as resp:
            data = await resp.json()
            return data['choices'][0]['message']['content'].strip()

    all_results = []
    async with aiohttp.ClientSession() as session:
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i+batch_size]
            tasks = [call_one(session, p) for p in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            all_results.extend(results)

    return all_results


df = pd.DataFrame({'prompt': ['问题1', '问题2', '问题3'] * 10})
results = asyncio.run(batch_llm_async(df['prompt'].tolist()))
df['llm_response'] = results
print(df.head())
```

## agg() 在 Series 上的聚合

```python
import pandas as pd
import numpy as np

np.random.seed(42)

scores = pd.Series(
    np.random.uniform(0.6, 0.98, 100),
    name='accuracy'
)

stats = scores.agg(['count', 'mean', 'std', 'min', 'median', 'max'])
print("模型准确率统计:")
print(stats.round(4))

custom_stats = scores.agg([
    ('样本数', 'count'),
    ('平均值', 'mean'),
    ('标准差', 'std'),
    ('最小值', 'min'),
    ('中位数', 'median'),
    ('最大值', 'max'),
    ('变异系数', lambda x: x.std() / x.mean()),
    ('偏度', 'skew'),
])
print("\n自定义命名统计:")
print(custom_stats.round(4))
```
