# 6.2 端到端 RAG Demo：文档问答系统

> **从 PDF 文件到智能问答——用 pgvector + Python 构建一个真正能用的 RAG 系统**

---

## 这一节在讲什么？

这一节我们要把前面学的所有知识整合起来，构建一个完整的 RAG 文档问答系统。与 Chroma 版本的 Demo 不同，pgvector 版本的核心亮点是"混合查询"——我们会在检索时同时做结构化过滤和向量搜索，用一条 SQL 完成 Chroma 需要两步才能做的事情。

---

## 表设计

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    source VARCHAR(255) NOT NULL,
    category VARCHAR(100),
    chunk_index INTEGER,
    total_chunks INTEGER,
    embedding vector(384),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- 创建索引
CREATE INDEX idx_docs_category ON documents (category);
CREATE INDEX idx_docs_metadata ON documents USING GIN (metadata);
CREATE INDEX idx_docs_embedding ON documents USING hnsw (embedding vector_cosine_ops);
```

---

## Python 完整实现

```python
import psycopg2
from psycopg2 import extras
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
import hashlib
import time

class PgVectorRAG:
    """基于 pgvector 的 RAG 文档问答系统"""

    def __init__(self, dsn="postgres://postgres:password@localhost/rag_db", dim=384):
        self.conn = psycopg2.connect(dsn)
        register_vector(self.conn)
        self.cur = self.conn.cursor()
        self.model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        self.dim = dim
        self._init_db()

    def _init_db(self):
        self.cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        self.cur.execute(f"""
            CREATE TABLE IF NOT EXISTS documents (
                id SERIAL PRIMARY KEY,
                content TEXT NOT NULL,
                source VARCHAR(255) NOT NULL,
                category VARCHAR(100),
                chunk_index INTEGER,
                total_chunks INTEGER,
                embedding vector({self.dim}),
                metadata JSONB,
                created_at TIMESTAMP DEFAULT NOW()
            );
        """)
        self.conn.commit()

    def _chunk(self, text, chunk_size=600, overlap=80):
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk.strip())
            start = end - overlap
        return chunks if chunks else [text]

    def ingest(self, text, source, category="general"):
        chunks = self._chunk(text)
        embeddings = self.model.encode(chunks, normalize_embeddings=True).tolist()
        current_ts = int(time.time())

        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            doc_id = hashlib.md5(f"{source}::chunk_{i}".encode()).hexdigest()[:16]
            self.cur.execute(
                """INSERT INTO documents (content, source, category, chunk_index, total_chunks, embedding, metadata)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT DO NOTHING""",
                (chunk, source, category, i, len(chunks), emb,
                 extras.Json({"source": source, "category": category, "chunk_index": i}))
            )
        self.conn.commit()
        print(f"✅ 入库 {len(chunks)} 个 chunk (来源: {source})")

    def search(self, query, category=None, top_k=5, max_distance=1.2):
        query_embedding = self.model.encode([query], normalize_embeddings=True).tolist()[0]

        if category:
            self.cur.execute(
                """SELECT id, content, source, category, chunk_index,
                          embedding <=> %s AS distance
                   FROM documents
                   WHERE category = %s
                   ORDER BY embedding <=> %s
                   LIMIT %s""",
                (query_embedding, category, query_embedding, top_k)
            )
        else:
            self.cur.execute(
                """SELECT id, content, source, category, chunk_index,
                          embedding <=> %s AS distance
                   FROM documents
                   ORDER BY embedding <=> %s
                   LIMIT %s""",
                (query_embedding, query_embedding, top_k)
            )

        results = []
        for row in self.cur.fetchall():
            if float(row[5]) <= max_distance:
                results.append({
                    "id": row[0], "content": row[1], "source": row[2],
                    "category": row[3], "chunk_index": row[4], "distance": float(row[5])
                })
        return results

    def ask(self, question, category=None, top_k=3):
        contexts = self.search(question, category=category, top_k=top_k)
        if not contexts:
            return {"answer": "未找到相关信息", "sources": []}

        context_text = "\n\n".join([
            f"[来源: {c['source']} 第{c['chunk_index']}段]\n{c['content']}"
            for c in contexts
        ])

        prompt = f"""基于以下参考信息回答问题。如果信息不足，请回答"我没有找到相关信息"。

参考信息：
{context_text}

问题：{question}

回答："""

        return {
            "prompt": prompt,
            "sources": [{"source": c["source"], "chunk": c["chunk_index"]} for c in contexts],
            "n_contexts": len(contexts)
        }


# ====== 使用示例 ======
rag = PgVectorRAG()

rag.ingest(
    "退款政策：购买后7天内可无条件退款。退款流程：1.在订单页面点击申请退款；2.填写退款原因；3.等待3-5个工作日审核。注意：已拆封的数码产品不支持无理由退款。",
    source="user_manual_v2.pdf",
    category="after_sales"
)

rag.ingest(
    "安装指南：1.下载安装包；2.双击运行安装程序；3.选择安装路径；4.点击安装。系统要求：Windows 10及以上，8GB内存。",
    source="install_guide.md",
    category="technical"
)

result = rag.ask("数码产品能退款吗", category="after_sales")
print(f"检索到 {result['n_contexts']} 条上下文")
print(f"来源: {result['sources']}")
```

---

## 常见误区

### 误区 1：忘记创建 HNSW 索引

没有索引时，大数据量的向量搜索会非常慢。务必在数据入库后创建 HNSW 索引。

### 误区 2：在 ingest 中逐条 INSERT 而不使用事务

每个 INSERT 都是一次 WAL 写入。把批量操作放在一个事务中，性能可以提升 5~10 倍。

---

## 本章小结

这一节我们用 pgvector 构建了一个完整的 RAG 文档问答系统。核心要点回顾：第一，表设计把 content、category、embedding、metadata 放在同一张表中，三种索引（B-tree/GIN/HNSW）协同工作；第二，ingest 流程是"切分 → encode → INSERT"，事务保证原子性；第三，search 流程是"encode → ORDER BY embedding <=> query → LIMIT"，支持 category 过滤的混合查询；第四，ask 流程是"search → 组装 prompt → 返回"，LLM 生成由应用层处理。

下一节我们将讲对话记忆与用户画像——如何用 pgvector 的 SQL 能力实现更智能的记忆管理。
