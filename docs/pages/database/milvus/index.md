# Milvus 教程

Milvus 是一个高性能、可扩展的分布式向量数据库。

## 安装

```bash
pip install pymilvus
```

## 快速开始

```python
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType

connections.connect("default", host="localhost", port="19530")
fields = [FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True)]
collection = Collection("demo", CollectionSchema(fields))
```

## 下一步

继续学习：
- [Chroma教程](/pages/database/chroma/)
- [Faiss教程](/pages/database/faiss/)
