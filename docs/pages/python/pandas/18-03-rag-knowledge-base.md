---
title: RAG 知识库管理
description: 文档分块元数据管理、Embedding 向量存储、检索结果分析、知识库质量监控
---
# RAG 知识库：用 Pandas 管理文档元数据

RAG（Retrieval-Augmented Generation）系统的核心是一个知识库——通常是大量文档被切分成小块（chunks），每块配上 embedding 向量和元数据。Pandas 虽然不能直接存储向量，但非常适合**管理这些 chunks 的元数据**。

## 元数据表设计

```python
import pandas as pd
import numpy as np

n = 5000
kb = pd.DataFrame({
    'chunk_id': [f'ch_{i:05d}' for i in range(n)],
    'doc_source': np.random.choice(['api_docs', 'blog', 'paper', 'guide'], n),
    'doc_title': [f'Document {i%50}' for i in range(n)],
    'token_count': np.random.randint(100, 1000, n),
    'embedding_model': 'text-embedding-3-small',
    'created_at': pd.date_range('2025-01-01', periods=n, freq='min'),
    'last_accessed': pd.date_range('2025-01-01', periods=n, freq='min') + np.random.randint(1,30,n)*pd.Timedelta(days=1),
})

print(kb.info())
print(f"\n按来源分布:\n{kb['doc_source'].value_counts()}")
```

## 知识库质量检查

```python
def kb_health_check(df):
    issues = []
    
    short = df[df['token_count'] < 50]
    if len(short) > 0:
        issues.append(f"过短 chunk (<50 tokens): {len(short)}")
    
    stale = df[(pd.Timestamp.now() - df['last_accessed']) > pd.Timedelta(days=90)]
    if len(stale) > 0:
        issues.append(f"90天未访问: {len(stale)}")
    
    model_dist = df['embedding_model'].value_counts()
    print("Embedding 模型分布:")
    print(model_dist)
    
    if len(issues) == 0:
        print("\n✅ 知识库健康")
    else:
        print(f"\n⚠️ 发现问题:")
        for issue in issues:
            print(f"  - {issue}")

kb_health_check(kb)
```
