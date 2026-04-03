---
title: nlargest() / nsmallest()：TopN 高效选择
description: TopK 选择性能优势、多列 TopN、分组内排名、替代 sort_values + head
---
# TopN 选择操作


## 为什么不用 sort_values().head()

```python
import pandas as pd
import numpy as np
import time

n = 1_000_000
df = pd.DataFrame({
    'value': np.random.randn(n),
})

start = time.time()
r1 = df.sort_values('value', ascending=False).head(10)
t1 = time.time() - start

start = time.time()
r2 = df.nlargest(10, 'value')
t2 = time.time() - start

print(f"sort_values+head: {t1:.4f}s")
print(f"nlargest:         {t2:.4f}s")
print(f"加速比:           {t1/t2:.1f}x")
```

**原理**：`nlargest()` 使用堆选择算法（heap select），时间复杂度 O(n log k)，而 `sort_values` 是 O(n log n)。当 n >> k 时，差距显著。

## nlargest / nsmallest 基础用法

```python
import pandas as pd
import numpy as np

np.random.seed(42)

df = pd.DataFrame({
    'model': [f'Model-{i}' for i in range(20)],
    'accuracy': np.random.uniform(0.60, 0.98, 20),
    'latency_ms': np.random.randint(100, 2000, 20),
    'price_per_1M': np.random.uniform(0.14, 15.0, 20),
})

top5_acc = df.nlargest(5, 'accuracy')
print("=== 准确率 Top 5 ===")
print(top5_acc[['model', 'accuracy']])

fastest3 = df.nsmallest(3, 'latency_ms')
print("\n=== 延迟最低 Top 3 ===")
print(fastest3[['model', 'latency_ms']])
```

## 多列 TopN

```python
import pandas as pd
import numpy as np

np.random.seed(42)

df = pd.DataFrame({
    'model': ['GPT-4o', 'Claude', 'Llama', 'Qwen', 'DeepSeek',
              'Gemini', 'Mistral', 'Phi-3'],
    'MMLU': [88.7, 89.2, 84.5, 83.5, 85.3, 86.8, 81.2, 79.4],
    'price': [2.50, 3.00, 0.27, 0.27, 0.14, 1.25, 0.70, 0.14],
})

best_value = df.nlargest(5, ['MMLU', 'price'], keep='all')
print("=== 性价比 Top 5（质量优先，价格次之）===")
print(best_value)
```

**注意**：`nlargest` 的多列排序中，所有列默认都是降序。如果需要混合升降序，仍需用 `sort_values`。

## keep 参数处理并列值

```python
import pandas as pd

scores = pd.DataFrame({
    'name': list('ABCDEFGHIJ'),
    'score': [95, 92, 92, 88, 88, 88, 85, 80, 78, 75],
})

print("keep='first' (默认):")
print(scores.nlargest(5, 'score', keep='first'))

print("\nkeep='last':")
print(scores.nlargest(5, 'score', keep='last'))

print("\nkeep='all':")
print(scores.nlargest(5, 'score', keep='all'))
```

## Series 上的 TopN

```python
import pandas as pd
import numpy as np

np.random.seed(42)

s = pd.Series(
    np.random.uniform(70, 95, 30),
    index=[f'model_{i}' for i in range(30)],
    name='accuracy'
)

top3 = s.nlargest(3)
bottom3 = s.nsmallest(3)

print("准确率 Top 3:")
print(top3.round(2))

print("\n准确率 Bottom 3:")
print(bottom3.round(2))
```

## 分组内的 TopN

### 方式一：groupby + apply + nlargest

```python
import pandas as pd
import numpy as np

np.random.seed(42)

df = pd.DataFrame({
    'category': ['NLP']*20 + ['Code']*20 + ['Math']*20,
    'model': [f'Model-{i}' for i in range(60)],
    'score': np.random.uniform(60, 95, 60),
})

def top_n_per_group(group, n=3):
    return group.nlargest(n, 'score')

top_by_cat = df.groupby('category', group_keys=False).apply(top_n_per_group, n=3)

print("=== 每类 Top 3 ===")
print(top_by_cat[['category', 'model', 'score']].sort_values(['category', 'score'], ascending=[True, False]))
```

### 方式二：groupby + rank（更高效）

```python
import pandas as pd
import numpy as np

np.random.seed(42)

df = pd.DataFrame({
    'category': ['NLP']*20 + ['Code']*20 + ['Math']*20,
    'model': [f'Model-{i}' for i in range(60)],
    'score': np.random.uniform(60, 95, 60),
})

df['cat_rank'] = df.groupby('category')['score'].rank(
    method='dense', ascending=False
).astype(int)

top3_efficient = df[df['cat_rank'] <= 3].sort_values(['category', 'cat_rank'])
print(top3_efficient[['category', 'model', 'score', 'cat_rank']].head(12))
```

**性能对比**：方式二（rank + filter）比方式一（apply + nlargest）快约 **3-5x**，因为避免了 Python 层循环。

## LLM 场景：RAG 检索 Top-K

```python
import pandas as pd
import numpy as np

class RAGRetriever:
    """基于 Pandas 的 RAG 检索器"""

    def __init__(self, chunk_df):
        self.chunks = chunk_df.copy()
        self._build_index()

    def _build_index(self):
        """预建索引加速检索"""
        self.chunks['_sim_cache'] = None

    def retrieve(self, query_embedding, top_k=5, min_score=0.5,
                 source_filter=None, max_tokens=4000):

        similarities = np.dot(
            self.chunks['embedding'].tolist(),
            query_embedding
        )
        self.chunks['_sim'] = similarities

        filtered = self.chunks.copy()
        if source_filter:
            filtered = filtered[filtered['source'].isin(source_filter)]

        candidates = filtered[filtered['_sim'] >= min_score]

        if len(candidates) == 0:
            return pd.DataFrame(columns=self.chunks.columns)

        result = candidates.nlargest(top_k, '_sim').copy()
        total_tokens = result['token_count'].sum()

        if total_tokens > max_tokens:
            result = self._trim_to_token_limit(result, max_tokens)

        result.drop(columns=['_sim', '_sim_cache'], errors='ignore', inplace=True)
        result.insert(0, 'rank', range(1, len(result)+1))
        result['relevance_pct'] = (result['_sim'] * 100).round(1)

        return result.drop(columns=['_sim'], errors='ignore')

    def _trim_to_token_limit(self, df, limit):
        """逐步裁剪直到满足 token 限制"""
        sorted_df = df.sort_values('_sim', ascending=False)
        cumulative = 0
        kept_indices = []
        for idx, row in sorted_df.iterrows():
            if cumulative + row['token_count'] <= limit:
                cumulative += row['token_count']
                kept_indices.append(idx)
            else:
                break
        return sorted_df.loc[kept_indices]


np.random.seed(42)
n_chunks = 500
chunk_data = {
    'chunk_id': [f'ch_{i:04d}' for i in range(n_chunks)],
    'doc_title': [f'Doc {i % 20}' for i in range(n_chunks)],
    'content': [f'Chunk content {i}...'] * n_chunks,
    'embedding': [np.random.randn(768).astype(np.float32) for _ in range(n_chunks)],
    'token_count': np.random.randint(100, 800, n_chunks),
    'source': np.random.choice(['api_docs', 'wiki', 'blog', 'paper'], n_chunks),
}
chunks_df = pd.DataFrame(chunk_data)

query_vec = np.random.randn(768).astype(np.float32)
query_vec /= np.linalg.norm(query_vec)

retriever = RAGRetriever(chunks_df)

results = retriever.retrieve(
    query_embedding=query_vec,
    top_k=8,
    min_score=0.3,
    source_filter=['api_docs', 'wiki'],
    max_tokens=3000,
)

print(f"检索到 {len(results)} 个相关片段\n")
for _, row in results.iterrows():
    bar = '█' * int(row.get('_sim', row.get('relevance_pct', 0)/100) * 25 if '_sim' in row.columns or 'relevance_pct' in row.columns else 5)
    print(f"#{row['rank']:>2} | {bar:<28s} | {row['chunk_id']:<10s} | "
          f"{row['doc_title']} | {row['token_count']} tokens")
```
