# 6.2 端到端 RAG Demo：大规模文档问答系统

> **用 Milvus 搭建 RAG——跟 pgvector 的实现方式不同，但功能等价**

---

## 这一节在讲什么？

在 pgvector 教程中，我们用 SQL 搭建了一个文档问答系统。这一节我们要用 Milvus 做同样的事情——但实现方式完全不同：没有 SQL、没有事务，取而代之的是 Python API 和 Milvus 的搜索能力。通过对比两种实现，你能更深刻地理解 Milvus 和 pgvector 的差异。

---

## Collection 设计

```python
from pymilvus import FieldSchema, CollectionSchema, DataType, MilvusClient

client = MilvusClient(uri="http://localhost:19530")

fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=64, is_partition_key=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
    FieldSchema(name="metadata", dtype=DataType.JSON),
]

schema = CollectionSchema(fields=fields, enable_dynamic_field=True)
client.create_collection(collection_name="documents", schema=schema)

# 创建索引
index_params = client.prepare_index_params()
index_params.add_index(
    field_name="embedding",
    index_type="HNSW",
    metric_type="COSINE",
    params={"M": 16, "efConstruction": 256}
)
index_params.add_index(
    field_name="category",
    index_type="INVERTED",
    index_name="idx_category"
)
client.create_index(collection_name="documents", index_params=index_params)
client.load_collection("documents")
```

---

## 完整 RAG 实现

```python
import numpy as np
from pymilvus import MilvusClient

class MilvusRAG:
    def __init__(self, milvus_uri="http://localhost:19530", collection_name="documents"):
        self.client = MilvusClient(uri=milvus_uri)
        self.collection_name = collection_name

    def ingest(self, documents, embedding_model, batch_size=1000):
        """文档入库：切分 → 编码 → 插入"""
        all_data = []
        for doc in documents:
            chunks = self._split_text(doc["content"], chunk_size=500, overlap=50)
            for i, chunk in enumerate(chunks):
                embedding = embedding_model.encode(chunk)
                all_data.append({
                    "content": chunk,
                    "source": doc["source"],
                    "category": doc.get("category", "general"),
                    "embedding": embedding.tolist(),
                    "metadata": {
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        **doc.get("metadata", {})
                    }
                })

        for i in range(0, len(all_data), batch_size):
            batch = all_data[i:i + batch_size]
            self.client.insert(collection_name=self.collection_name, data=batch)

        self.client.flush(self.collection_name)
        print(f"Ingested {len(all_data)} chunks from {len(documents)} documents")

    def search(self, query, embedding_model, top_k=5, category=None):
        """混合检索：向量搜索 + 标量过滤"""
        query_embedding = embedding_model.encode(query)

        filter_expr = None
        if category:
            filter_expr = f'category == "{category}"'

        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_embedding.tolist()],
            limit=top_k,
            filter=filter_expr,
            output_fields=["content", "source", "category", "metadata"],
            search_params={
                "metric_type": "COSINE",
                "params": {"ef": 100}
            },
            consistency_level="Session"
        )

        retrieved = []
        for hit in results[0]:
            retrieved.append({
                "content": hit["entity"]["content"],
                "source": hit["entity"]["source"],
                "category": hit["entity"]["category"],
                "metadata": hit["entity"]["metadata"],
                "distance": hit["distance"]
            })
        return retrieved

    def ask(self, query, embedding_model, llm_client, top_k=5, category=None):
        """端到端问答：检索 → 组装 Prompt → 生成"""
        retrieved = self.search(query, embedding_model, top_k, category)

        context = "\n\n".join([
            f"[来源: {r['source']} | 分类: {r['category']}]\n{r['content']}"
            for r in retrieved
        ])

        prompt = f"""基于以下参考资料回答用户的问题。如果参考资料中没有相关信息，请说明。

参考资料：
{context}

用户问题：{query}

请给出详细、准确的回答，并在回答中引用来源："""

        response = llm_client.chat(prompt)
        return {
            "answer": response,
            "sources": [{"source": r["source"], "distance": r["distance"]} for r in retrieved]
        }

    def _split_text(self, text, chunk_size=500, overlap=50):
        """简单的文本切分"""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start = end - overlap
        return chunks


# 使用示例
rag = MilvusRAG(milvus_uri="http://localhost:19530")

# 入库文档
documents = [
    {
        "source": "company_handbook.pdf",
        "content": "Our company's refund policy allows returns within 30 days...",
        "category": "policy",
        "metadata": {"department": "HR", "version": "2024"}
    },
    {
        "source": "tech_docs.md",
        "content": "The API rate limit is 1000 requests per minute...",
        "category": "tech",
        "metadata": {"team": "Platform", "last_updated": "2024-01-15"}
    }
]

# rag.ingest(documents, embedding_model)

# 问答
# result = rag.ask("What is the refund policy?", embedding_model, llm_client)
# print(result["answer"])
```

---

## 与 pgvector RAG Demo 的对比

| 维度 | pgvector 实现 | Milvus 实现 |
|------|-------------|------------|
| 数据定义 | SQL CREATE TABLE | Python FieldSchema |
| 数据插入 | psycopg2 execute | client.insert() |
| 混合查询 | WHERE + ORDER BY embedding <=> | search(filter=..., data=...) |
| 索引创建 | CREATE INDEX ... USING hnsw | client.create_index() |
| 索引加载 | 自动（PostgreSQL 管理） | client.load_collection()（必须手动） |
| 事务支持 | ✅ conn.commit() / conn.rollback() | ❌ 无事务 |
| 结果解析 | cur.fetchone() / cur.fetchall() | results[0][i]["entity"] |

核心差异在于：pgvector 用 SQL 表达一切，Milvus 用 Python API 表达一切。功能上等价，但思维方式不同——pgvector 是"数据库思维"，Milvus 是"API 思维"。

---

## 常见误区：Milvus 的 RAG 一定比 pgvector 好

Milvus 的 RAG 实现并不比 pgvector "更好"——它只是"不同"。如果你的数据量只有几十万条，pgvector 的 RAG 实现更简单（SQL 一条搞定）、更可靠（事务保证）、更易维护（运维 PostgreSQL 就行）。Milvus 的优势只在数据量超过单机承载能力时才体现出来。

---

## 小结

这一节我们用 Milvus 搭建了一个完整的 RAG 系统：MilvusRAG 类封装了文档入库、混合检索和端到端问答的功能。与 pgvector 的 RAG 实现相比，核心差异是"SQL 思维"vs"API 思维"——功能等价，表达方式不同。下一节我们聊 Milvus 在多租户场景下的应用。
