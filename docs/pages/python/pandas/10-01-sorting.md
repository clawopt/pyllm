---
title: 排序：单列/多列排序
description: sort_values() 的用法、ascending 参数、na_position 处理、多级排序实战
---
# 排序操作详解


## 基础排序

```python
import pandas as pd
import numpy as np

np.random.seed(42)

df = pd.DataFrame({
    'model': ['GPT-4o', 'Claude', 'Llama', 'Gemini', 'Qwen', 'DeepSeek'],
    'MMLU': [88.7, 89.2, 84.5, 86.8, 83.5, 85.3],
    'price': [2.50, 3.00, 0.27, 1.25, 0.27, 0.14],
    'context': [128000, 200000, 131072, 1000000, 131072, 65536],
})

sorted_asc = df.sort_values('MMLU')
print(sorted_asc[['model', 'MMLU']])

sorted_desc = df.sort_values('MMLU', ascending=False)
print(sorted_desc[['model', 'MMLU']])
```

## 多列排序

### 按优先级排序

```python
import pandas as pd

df = pd.DataFrame({
    'category': ['tech', 'finance', 'tech', 'medical'],
    'score': [92, 88, 95, 87],
    'price': [15.0, 8.0, 20.0, 12.0],
})

result = df.sort_values(
    by=['score', 'price'],
    ascending=[False, True]  # 分数降序，价格升序
)

print(result)
```

### 三级排序

```python
import pandas as pd

df = pd.DataFrame({
    'model': ['GPT-4o', 'Claude', 'Llama', 'Gemini', 'Qwen'],
    'MMLU': [88.7, 89.2, 84.5, 86.8, 83.5],
    'price': [2.50, 3.00, 0.27, 1.25, 0.27],
    'context': [128000, 200000, 131072, 1000000, 131072],
})

result = df.sort_values(
    by=['MMLU', 'price', 'context'],
    ascending=[False, True, False]  # 质量↓ 价格↑ 窗口↑
)

print(result[['model', 'MMLU', 'price', 'context']])
```

## na_position：缺失值处理

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'model': ['A', 'B', None, 'D', 'E'],
    'value': [5, 3, np.nan, 7, 2],
    'score': [None, 4.5, 9.2, None, 3.8],
})

r_first = df.sort_values('score')
print(r_first['model'])

r_last = df.sort_values('score', na_position='last')
print(r_last['model'])

r_middle = df.sort_values('score', na_position='middle')
print(r_middle['model'])

r_dropna = df.dropna(subset=['score']).sort_values('score')

print("\n缺失值位置选项:")
print(f"  first (默认): {r_first['model'].tolist()}")
print(f"  last:        {r_last['model'].tolist()}")
print(f"  middle:      {r_middle['model'].tolist()}")
```

## sort_index()：索引排序

```python
import pandas as pd

df = pd.DataFrame(
    {'val': range(5)},
    index=['e', 'a', 'c', 'b', 'd']
)

sorted_idx = df.sort_index()
print(sorted_idx)

sorted_idx_desc = df.sort_index(ascending=False)
print(sorted_idx_desc)
```

## LLM 场景：按相关性分数排序检索结果

```python
import pandas as pd
import numpy as np

np.random.seed(42)

n = 500
query_embedding = np.random.randn(768).astype(np.float32) / np.linalg.norm(np.random.randn(768))

results = pd.DataFrame({
    'chunk_id': [f'ch_{i:04d}' for i in range(n)],
    'doc_title': [f'Document {i}' for i in range(n)],
    'similarity': np.dot(
        np.random.randn(n, 768).astype(np.float32),
        query_embedding
    ),
    'token_count': np.random.randint(100, 2000, n),
    'source': np.random.choice(['api', 'wiki', 'blog'], n),
})

ranked_results = results.sort_values('similarity', ascending=False)

print("=== RAG 检索结果（按相关性排序）===")
for i, row in ranked_results.head(10).iterrows():
    bar = '█' * int(row['similarity'] * 30)
    print(f"{bar:>30s} {row['chunk_id']:<12s} "
          f"{row['similarity']:.3f} {row['doc_title'][:30]}")
```
