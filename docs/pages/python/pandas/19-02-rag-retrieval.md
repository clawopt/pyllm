---
title: 知识库检索与排序
description: 相似度计算 / 多路召回 / 重排（Rerank） / 检索结果格式化 / 检索日志记录
---
# RAG 知识库检索


## 相似度基础：在 Pandas 中管理向量

```python
import pandas as pd
import numpy as np

class VectorSearch:
    """基于 Pandas 的简单向量搜索"""

    def __init__(self, kb_df, embedding_col='embedding'):
        self.kb = kb_df.copy()
        self.embedding_col = embedding_col

    def add_embeddings(self, embeddings):
        """添加 embedding 向量到知识库"""
        if len(embeddings) != len(self.kb):
            raise ValueError(f"Embedding 数量({len(embeddings)}) "
                           f"不匹配 KB 行数({len(self.kb)})")

        self.kb[self.embedding_col] = list(embeddings)
        self.kb['embedding_status'] = 'done'
        return self

    def search(self, query_vector, top_k=5, min_score=0.3,
               source_filter=None, max_tokens=4000):
        """相似度搜索"""
        if self.embedding_col not in self.kb.columns:
            raise ValueError("请先调用 add_embeddings()")

        candidates = self.kb[self.kb['is_active'] == True].copy()

        if source_filter:
            candidates = candidates[
                candidates['source_type'].isin(source_filter)
            ]

        if len(candidates) == 0:
            return pd.DataFrame(columns=self.kb.columns)

        emb_matrix = np.array(candidates[self.embedding_col].tolist())
        query_norm = query_vector / (np.linalg.norm(query_vector) + 1e-8)
        row_norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True) + 1e-8
        normalized = emb_matrix / row_norms

        similarities = normalized @ query_norm
        candidates['_similarity'] = similarities

        candidates = candidates[candidates['_similarity'] >= min_score]

        if len(candidates) == 0:
            return pd.DataFrame(columns=self.kb.columns)

        result = candidates.nlargest(top_k, '_similarity').copy()

        total_tokens = 0
        kept_indices = []
        for idx, row in result.iterrows():
            tc = row.get('token_count', 200)
            if total_tokens + tc <= max_tokens:
                total_tokens += tc
                kept_indices.append(idx)
            else:
                break

        result = result.loc[kept_indices].copy()
        result.insert(0, 'rank', range(1, len(result)+1))

        cleanup_cols = [c for c in result.columns
                       if c.startswith('_') and c != '_similarity']
        return result.drop(columns=cleanup_cols)


np.random.seed(42)
n_chunks = 100
kb_df = pd.DataFrame({
    'chunk_id': [f'ch_{i:04d}' for i in range(n_chunks)],
    'content': [f'Document chunk {i} content...' for i in range(n_chunks)],
    'source_doc': [f'Doc_{i % 10}' for i in range(n_chunks)],
    'source_type': pd.Categorical(
        np.random.choice(['api_docs', 'wiki', 'blog'], n_chunks)
    ),
    'token_count': np.random.randint(100, 800, n_chunks),
    'is_active': [True] * n_chunks,
})

embeddings = np.random.randn(n_chunks, 768).astype(np.float32)
embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

searcher = VectorSearch(kb_df)
searcher.add_embeddings(embeddings)

query_vec = np.random.randn(768).astype(np.float32)
query_vec /= np.linalg.norm(query_vec)

results = searcher.search(
    query_vec,
    top_k=5,
    min_score=0.2,
    source_filter=['api_docs', 'wiki'],
    max_tokens=2000,
)

print("=== RAG 检索结果 ===")
for _, row in results.iterrows():
    bar = '█' * int(row.get('_similarity', 0.5) * 25)
    print(f"#{row['rank']:>2d} | {bar:<28s} | "
          f"{row['chunk_id']:<10s} | {row['source_doc']} | "
          f"{row['token_count']} tokens")
```

## 检索日志记录

```python
import pandas as pd
from datetime import datetime

class RetrievalLogger:
    """RAG 检索日志记录器"""

    def __init__(self):
        self.logs = []

    def log(self, query_text, results_df, latency_ms, model_name='default'):
        """记录一次检索"""
        log_entry = {
            'timestamp': datetime.now(),
            'query': query_text[:200],
            'model_used': model_name,
            'results_count': len(results_df),
            'top_similarity': float(results_df['_similarity'].max()) if len(results_df) > 0 else 0,
            'total_tokens_retrieved': int(results_df['token_count'].sum()) if 'token_count' in results_df.columns else 0,
            'latency_ms': latency_ms,
            'retrieved_chunk_ids': results_df['chunk_id'].tolist() if len(results_df) > 0 else [],
        }
        self.logs.append(log_entry)
        return log_entry

    def to_dataframe(self):
        return pd.DataFrame(self.logs)

    def stats(self):
        df = self.to_dataframe()
        if len(df) == 0:
            return {}

        return {
            'total_queries': len(df),
            'avg_results': round(df['results_count'].mean(), 1),
            'avg_latency_ms': round(df['latency_ms'].mean(), 1),
            'avg_top_sim': round(df['top_similarity'].mean(), 3),
            'p95_latency': round(df['latency_ms'].quantile(0.95), 1),
        }


logger = RetrievalLogger()
logger.log("什么是注意力机制", results, latency_ms=85)
logger.log("如何安装 PyTorch", results.head(3), latency_ms=62)
logger.log("BERT 和 GPT 的区别", pd.DataFrame(), latency_ms=45)

stats = logger.stats()
print("=== 检索日志统计 ===")
for k, v in stats.items():
    print(f"  {k}: {v}")
```
