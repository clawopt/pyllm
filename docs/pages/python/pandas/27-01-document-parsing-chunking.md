---
title: 文档解析与分块
description: PDF/Markdown/HTML 文档解析、文本分块策略、元数据提取
---
# 案例 2：RAG 知识库构建 — 文档解析与分块

## 项目背景

你需要为公司内部知识库构建 RAG 检索系统。原始数据是 5000 份文档（PDF/Markdown/HTML），需要被切分成适合嵌入检索的文本块（chunks），并管理每个 chunk 的元数据。

## 文档元数据管理

```python
import pandas as pd
import numpy as np

n = 5_000
docs = pd.DataFrame({
    'doc_id': [f'doc_{i:05d}' for i in range(n)],
    'title': [f'文档 {i%200}' for i in range(n)],
    'source': np.random.choice(['wiki', 'blog', 'paper', 'api_docs'], n),
    'format': np.random.choice(['pdf', 'md', 'html'], n),
    'size_kb': np.random.randint(10, 5000, n),
    'last_updated': pd.date_range('2024-01-01', periods=n, freq='h'),
})

print(f"总文档: {n:,}")
print(f"\n按来源:\n{docs['source'].value_counts()}")
print(f"\n按格式:\n{docs['format'].value_counts()}")
print(f"\n大小分布:")
print(docs['size_kb'].describe().round(1))
```
