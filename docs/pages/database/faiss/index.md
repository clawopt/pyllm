# Faiss 教程

Faiss 是 Facebook 开发的高效向量相似度搜索库。

## 安装

```bash
pip install faiss-cpu
```

## 快速开始

```python
import faiss
import numpy as np

d = 768
index = faiss.IndexFlatL2(d)
index.add(np.random.rand(10000, d).astype('float32'))
D, I = index.search(np.random.rand(1, d).astype('float32'), k=5)
print(f"最近邻索引: {I}")
```

## 下一步

继续学习：
- [DuckDB教程](/pages/database/duckdb/)
- [LanceDB教程](/pages/database/lancedb/)
