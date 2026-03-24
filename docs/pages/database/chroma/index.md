# Chroma 教程

Chroma 是一个轻量级、开源的向量数据库。

## 安装

```bash
pip install chromadb
```

## 快速开始

```python
import chromadb

client = chromadb.Client()
collection = client.create_collection("documents")
collection.add(documents=["Python is great"], ids=["doc1"])
results = collection.query(query_texts=["What is Python?"], n_results=2)
print(results)
```

## 下一步

继续学习：
- [Faiss教程](/pages/database/faiss/)
- [DuckDB教程](/pages/database/duckdb/)
