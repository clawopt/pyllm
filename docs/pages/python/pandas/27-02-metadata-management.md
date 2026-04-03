---
title: 元数据管理
description: chunk 级别的元数据追踪、文档-chunk 关联、来源统计与质量监控
---
# 元数据管理

```python
import pandas as pd
import numpy as np

n_chunks = 50_000
chunks = pd.DataFrame({
    'chunk_id': [f'ch_{i:06d}' for i in range(n_chunks)],
    'doc_id': [f'doc_{i%5000}' for i in range(n_chunks)],
    'text': [f'chunk content {i}' for i in range(n_chunks)],
    'token_count': np.random.randint(100, 1000, n_chunks),
    'embedding_model': ['text-embedding-3-small'] * n_chunks,
    'created_at': pd.date_range('2025-01-01', periods=n_chunks, freq='min'),
})

chunks_per_doc = chunks.groupby('doc_id').size()
print(f"总 chunks: {n_chunks:,}")
print(f"每文档平均 chunks: {chunks_per_doc.mean():.1f}")
print(f"最大 chunks/文档: {chunks_per_doc.max()}")

stale = chunks[(pd.Timestamp.now() - chunks['created_at']) > pd.Timedelta(days=90)]
print(f"\n90天未更新: {len(stale)} chunks ({len(stale)/n_chunks*100:.1f}%)")
```
