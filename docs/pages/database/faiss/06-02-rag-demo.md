# 6.2 端到端 RAG Demo：用 FAISS 构建文档问答

> **FAISS 的 RAG 实现更轻量——但也需要你自己管理更多东西**

---

## 这一节在讲什么？

在 pgvector 和 Milvus 教程中，我们都搭建了完整的 RAG Demo。这一节我们要用 FAISS 做同样的事情，但实现方式完全不同——没有 SQL、没有 Collection、没有标量过滤，取而代之的是 numpy 数组、Index 和手动映射。通过对比三种实现，你能更深刻地理解"库"和"数据库"的差异。

---

## 完整 RAG 实现

```python
import faiss
import numpy as np
import json

class FaissRAG:
    def __init__(self, d=768, index_type="hnsw"):
        self.d = d
        self.documents = []  # 文档内容列表（ID 就是列表索引）
        self.index = None
        self.index_type = index_type

    def ingest(self, documents, embedding_model, batch_size=1000):
        """文档入库：切分 → 编码 → 构建 FAISS 索引"""
        all_embeddings = []
        for doc in documents:
            chunks = self._split_text(doc["content"], chunk_size=500, overlap=50)
            for i, chunk in enumerate(chunks):
                embedding = embedding_model.encode(chunk)
                all_embeddings.append(embedding)
                self.documents.append({
                    "content": chunk,
                    "source": doc["source"],
                    "category": doc.get("category", "general"),
                    "chunk_index": i,
                })

        embeddings = np.array(all_embeddings).astype('float32')
        faiss.normalize_L2(embeddings)  # 归一化用于余弦搜索

        # 构建索引
        if self.index_type == "flat":
            self.index = faiss.IndexFlatIP(self.d)
        elif self.index_type == "ivf":
            nlist = int(np.sqrt(len(embeddings)))
            quantizer = faiss.IndexFlatIP(self.d)
            self.index = faiss.IndexIVFFlat(quantizer, self.d, nlist)
            self.index.train(embeddings)
        elif self.index_type == "hnsw":
            self.index = faiss.IndexHNSWFlat(self.d, 32)
            self.index.hnsw.efConstruction = 200

        self.index.add(embeddings)
        print(f"Ingested {len(self.documents)} chunks")

    def search(self, query, embedding_model, top_k=5):
        """向量搜索"""
        query_embedding = embedding_model.encode(query).reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_embedding)

        if self.index_type == "hnsw":
            self.index.hnsw.efSearch = 100
        elif self.index_type == "ivf":
            self.index.nprobe = 32

        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0 and idx < len(self.documents):
                doc = self.documents[idx]
                results.append({
                    "content": doc["content"],
                    "source": doc["source"],
                    "category": doc["category"],
                    "distance": float(distances[0][i])
                })
        return results

    def ask(self, query, embedding_model, llm_client, top_k=5):
        """端到端问答"""
        retrieved = self.search(query, embedding_model, top_k)

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
        return {"answer": response, "sources": retrieved}

    def save(self, path="faiss_rag"):
        """保存索引和文档到磁盘"""
        faiss.write_index(self.index, f"{path}.faiss")
        with open(f"{path}_docs.json", "w") as f:
            json.dump(self.documents, f, ensure_ascii=False)

    def load(self, path="faiss_rag"):
        """从磁盘加载索引和文档"""
        self.index = faiss.read_index(f"{path}.faiss")
        with open(f"{path}_docs.json", "r") as f:
            self.documents = json.load(f)

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

## 与 pgvector/Milvus RAG Demo 的对比

| 维度 | pgvector | Milvus | FAISS |
|------|----------|--------|-------|
| 数据存储 | PostgreSQL 表 | Collection | Python 列表 + FAISS Index |
| 向量搜索 | SQL ORDER BY | client.search() | index.search() |
| 标量过滤 | SQL WHERE | expr 过滤 | ❌ 不支持 |
| 持久化 | 自动 | 自动 | 手动序列化 |
| 结果映射 | SQL SELECT | output_fields | documents[idx] |
| 代码量 | 中 | 中 | 少 |

FAISS 的实现最轻量，但你需要自己管理文档列表、序列化和标量过滤。

---

## 标量过滤的实现

FAISS 不支持标量过滤——如果你需要"只搜索分类为 tech 的文档"，有两种变通方法：

```python
# 方法1：搜索后过滤（简单但可能结果不够）
results = rag.search(query, embedding_model, top_k=50)  # 多召回一些
filtered = [r for r in results if r["category"] == "tech"][:5]

# 方法2：为每个分类建独立索引（精确但索引多）
indexes = {}
for category in ["tech", "science", "art"]:
    cat_docs = [d for d in documents if d["category"] == category]
    # 为每个分类建索引...
```

---

## 小结

这一节我们用 FAISS 搭建了完整的 RAG 系统：FaissRAG 类封装了文档入库、搜索、问答、保存和加载。FAISS 的实现最轻量，但需要自己管理文档映射和持久化，不支持标量过滤。下一节我们聊 FAISS + pgvector 混合方案。
