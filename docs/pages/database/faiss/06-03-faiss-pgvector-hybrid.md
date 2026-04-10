# 6.3 FAISS + pgvector 混合方案

> **FAISS 负责快，pgvector 负责稳——混合方案兼顾性能和功能**

---

## 这一节在讲什么？

FAISS 的搜索性能比 pgvector 快 5~10 倍（特别是 GPU 加速时），但 FAISS 不支持标量过滤和数据持久化。pgvector 支持 SQL、事务和持久化，但搜索性能不如 FAISS。混合方案把两者的优势结合起来——pgvector 作为主存储（持久化 + 标量过滤 + 事务），FAISS 作为搜索加速层（高性能向量搜索），搜索结果用 ID 映射回 pgvector 获取完整数据。

---

## 混合架构设计

```
FAISS + pgvector 混合架构：

  写入路径：
  应用 → pgvector（INSERT，持久化 + 事务保证）
       → FAISS（index.add()，内存索引）

  搜索路径：
  应用 → FAISS（index.search()，高性能搜索）
       → pgvector（SELECT WHERE id IN (...)，获取完整数据）

  索引更新：
  定期从 pgvector 重建 FAISS 索引（保证一致性）
```

---

## 完整实现

```python
import faiss
import numpy as np
import psycopg2
from pgvector.psycopg2 import register_vector

class HybridRAG:
    def __init__(self, d=768, pg_conn_string=None):
        self.d = d
        self.documents = []
        self.index = None

        # pgvector 连接
        self.conn = psycopg2.connect(pg_conn_string)
        register_vector(self.conn)
        self.cur = self.conn.cursor()

    def ingest(self, documents, embedding_model):
        """写入 pgvector + FAISS"""
        all_embeddings = []

        for doc in documents:
            chunks = self._split_text(doc["content"])
            for i, chunk in enumerate(chunks):
                embedding = embedding_model.encode(chunk)

                # 写入 pgvector（持久化）
                self.cur.execute("""
                    INSERT INTO documents (source, category, content, embedding, metadata)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id
                """, (doc["source"], doc.get("category", "general"),
                      chunk, embedding.tolist(),
                      {"chunk_index": i}))
                doc_id = self.cur.fetchone()[0]
                self.conn.commit()

                # 同时记录用于 FAISS
                all_embeddings.append(embedding)
                self.documents.append({"id": doc_id, "content": chunk})

        # 构建 FAISS 索引
        embeddings = np.array(all_embeddings).astype('float32')
        faiss.normalize_L2(embeddings)
        self.index = faiss.IndexHNSWFlat(self.d, 32)
        self.index.hnsw.efConstruction = 200
        self.index.add(embeddings)

    def search(self, query, embedding_model, top_k=5, category=None):
        """FAISS 搜索 → pgvector 获取数据 + 标量过滤"""
        query_emb = embedding_model.encode(query).reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_emb)

        self.index.hnsw.efSearch = 100

        # FAISS 多召回一些候选
        fetch_k = top_k * 5 if category else top_k
        distances, indices = self.index.search(query_emb, fetch_k)

        # 用 ID 从 pgvector 获取完整数据
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self.documents):
                continue
            doc_id = self.documents[idx]["id"]

            # pgvector 查询 + 标量过滤
            if category:
                self.cur.execute("""
                    SELECT content, source, category FROM documents
                    WHERE id = %s AND category = %s
                """, (doc_id, category))
            else:
                self.cur.execute("""
                    SELECT content, source, category FROM documents
                    WHERE id = %s
                """, (doc_id,))

            row = self.cur.fetchone()
            if row:
                results.append({
                    "content": row[0],
                    "source": row[1],
                    "category": row[2],
                    "distance": float(distances[0][i])
                })

            if len(results) >= top_k:
                break

        return results

    def rebuild_index(self, embedding_model):
        """从 pgvector 重建 FAISS 索引"""
        self.cur.execute("SELECT id, embedding FROM documents")
        rows = self.cur.fetchall()

        embeddings = []
        self.documents = []
        for row in rows:
            emb = np.array(row[1], dtype='float32')
            embeddings.append(emb)
            self.documents.append({"id": row[0]})

        embeddings = np.array(embeddings)
        faiss.normalize_L2(embeddings)
        self.index = faiss.IndexHNSWFlat(self.d, 32)
        self.index.hnsw.efConstruction = 200
        self.index.add(embeddings)

    def _split_text(self, text, chunk_size=500, overlap=50):
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start = end - overlap
        return chunks
```

---

## 数据同步策略

混合架构的核心挑战是数据同步——pgvector 和 FAISS 中的数据需要保持一致。推荐策略：

1. **pgvector 为主存储**——所有写入先到 pgvector，利用事务保证一致性
2. **FAISS 索引定期重建**——每隔 N 分钟从 pgvector 读取全量数据重建 FAISS 索引
3. **增量更新**——在两次重建之间，新写入的数据可以增量 `add()` 到 FAISS

---

## 小结

这一节我们实现了 FAISS + pgvector 混合方案：pgvector 负责持久化和标量过滤，FAISS 负责高性能向量搜索。数据同步策略是"pgvector 为主，FAISS 定期重建"。这个方案兼顾了 FAISS 的性能和 pgvector 的功能，是生产环境中常见的架构选择。下一节开始我们进入第 7 章，讨论 FAISS 与向量数据库的协作。
