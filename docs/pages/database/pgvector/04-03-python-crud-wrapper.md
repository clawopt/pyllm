# 4.3 Python 完整 CRUD 封装

> **好的封装让你用 pgvector 像用 ORM 一样简单——但保留了 SQL 的全部灵活性**

---

## 这一节在讲什么？

前面两节我们分别讲了 pgvector 的基本 CRUD 和混合查询，但代码都是零散的 SQL 片段。这一节我们要把这些操作封装成一个可复用的 Python 类——PGVectorManager，它封装了连接管理、CRUD 操作、批量操作和混合查询，让你在 RAG 应用中用几行代码就能完成向量搜索。同时我们还会介绍 SQLAlchemy ORM 集成和 asyncpg 异步方案。

---

## PGVectorManager 完整封装

```python
import psycopg2
from psycopg2 import pool, extras
from pgvector.psycopg2 import register_vector
import json
from typing import List, Dict, Optional, Any

class PGVectorManager:
    """pgvector 完整 CRUD 封装"""

    def __init__(self, dsn: str = "postgres://postgres:password@localhost/rag_db",
                 pool_min: int = 2, pool_max: int = 10):
        self._pool = pool.ThreadedConnectionPool(
            minconn=pool_min, maxconn=pool_max, dsn=dsn
        )

    def _get_conn(self):
        conn = self._pool.getconn()
        register_vector(conn)
        return conn

    def _put_conn(self, conn):
        self._pool.putconn(conn)

    def init_db(self, dim: int = 384):
        """初始化数据库：创建扩展和表"""
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS documents (
                id SERIAL PRIMARY KEY,
                content TEXT NOT NULL,
                source VARCHAR(255),
                category VARCHAR(100),
                embedding vector({dim}),
                metadata JSONB,
                created_at TIMESTAMP DEFAULT NOW()
            );
        """)
        conn.commit()
        cur.close()
        self._put_conn(conn)

    def insert(self, content: str, embedding: List[float],
               source: str = None, category: str = None,
               metadata: Dict = None) -> int:
        """插入一条文档"""
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO documents (content, embedding, source, category, metadata)
               VALUES (%s, %s, %s, %s, %s) RETURNING id""",
            (content, embedding, source, category,
             extras.Json(metadata) if metadata else None)
        )
        doc_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        self._put_conn(conn)
        return doc_id

    def batch_insert(self, rows: List[tuple]):
        """批量插入（事务保证原子性）"""
        conn = self._get_conn()
        cur = conn.cursor()
        cur.executemany(
            """INSERT INTO documents (content, embedding, source, category, metadata)
               VALUES (%s, %s, %s, %s, %s)""",
            [(r[0], r[1], r[2] if len(r) > 2 else None,
              r[3] if len(r) > 3 else None,
              extras.Json(r[4]) if len(r) > 4 and r[4] else None)
             for r in rows]
        )
        conn.commit()
        cur.close()
        self._put_conn(conn)

    def get_by_id(self, doc_id: int) -> Optional[dict]:
        """按 ID 获取文档"""
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute(
            "SELECT id, content, source, category, metadata, created_at FROM documents WHERE id = %s",
            (doc_id,)
        )
        row = cur.fetchone()
        cur.close()
        self._put_conn(conn)
        if row:
            return {"id": row[0], "content": row[1], "source": row[2],
                    "category": row[3], "metadata": row[4], "created_at": row[5]}
        return None

    def search(self, query_embedding: List[float], top_k: int = 5,
               category: str = None, max_distance: float = None,
               where_sql: str = None, where_params: tuple = None) -> List[dict]:
        """向量搜索（支持混合查询）"""
        conn = self._get_conn()
        cur = conn.cursor()

        conditions = []
        params = []

        if category:
            conditions.append("category = %s")
            params.append(category)
        if max_distance:
            conditions.append("embedding <=> %s < %s")
            params.extend([query_embedding, max_distance])
        if where_sql:
            conditions.append(where_sql)
            if where_params:
                params.extend(where_params)

        where_clause = " AND ".join(conditions) if conditions else "TRUE"

        sql = f"""
            SELECT id, content, source, category, metadata,
                   embedding <=> %s AS distance
            FROM documents
            WHERE {where_clause}
            ORDER BY embedding <=> %s
            LIMIT %s
        """
        params = [query_embedding] + params + [query_embedding] + [top_k]

        cur.execute(sql, tuple(params))
        results = []
        for row in cur.fetchall():
            results.append({
                "id": row[0], "content": row[1], "source": row[2],
                "category": row[3], "metadata": row[4], "distance": float(row[5])
            })

        cur.close()
        self._put_conn(conn)
        return results

    def update(self, doc_id: int, **kwargs):
        """更新文档"""
        conn = self._get_conn()
        cur = conn.cursor()
        set_clauses = []
        params = []
        for key, value in kwargs.items():
            if key == "metadata" and isinstance(value, dict):
                value = extras.Json(value)
            set_clauses.append(f"{key} = %s")
            params.append(value)
        params.append(doc_id)
        cur.execute(
            f"UPDATE documents SET {', '.join(set_clauses)} WHERE id = %s",
            tuple(params)
        )
        conn.commit()
        affected = cur.rowcount
        cur.close()
        self._put_conn(conn)
        return affected

    def delete(self, doc_id: int = None, category: str = None):
        """删除文档"""
        conn = self._get_conn()
        cur = conn.cursor()
        if doc_id:
            cur.execute("DELETE FROM documents WHERE id = %s", (doc_id,))
        elif category:
            cur.execute("DELETE FROM documents WHERE category = %s", (category,))
        conn.commit()
        affected = cur.rowcount
        cur.close()
        self._put_conn(conn)
        return affected

    def count(self, category: str = None) -> int:
        """统计文档数"""
        conn = self._get_conn()
        cur = conn.cursor()
        if category:
            cur.execute("SELECT COUNT(*) FROM documents WHERE category = %s", (category,))
        else:
            cur.execute("SELECT COUNT(*) FROM documents")
        count = cur.fetchone()[0]
        cur.close()
        self._put_conn(conn)
        return count


# ====== 使用示例 ======
mgr = PGVectorManager()
mgr.init_db(dim=384)

# 插入
doc_id = mgr.insert(
    content="退款政策：购买后7天内可无条件退款",
    embedding=[0.1] * 384,
    source="manual.pdf",
    category="after_sales",
    metadata={"page": 15, "version": 2}
)

# 搜索
results = mgr.search(
    query_embedding=[0.1] * 384,
    category="after_sales",
    top_k=5,
    max_distance=1.0
)

# 自定义 WHERE 条件
results = mgr.search(
    query_embedding=[0.1] * 384,
    where_sql="(metadata ->> 'version')::int >= %s",
    where_params=(2,),
    top_k=5
)
```

---

## SQLAlchemy ORM 集成

如果你使用 SQLAlchemy，可以通过 `pgvector.sqlalchemy` 模块在 ORM 模型中使用 vector 类型：

```python
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from pgvector.sqlalchemy import Vector
import datetime

Base = declarative_base()

class Document(Base):
    __tablename__ = "documents_orm"

    id = Column(Integer, primary_key=True, autoincrement=True)
    content = Column(Text, nullable=False)
    source = Column(String(255))
    category = Column(String(100))
    embedding = Column(Vector(384))
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

engine = create_engine("postgresql://postgres:password@localhost/rag_db")
Base.metadata.create_all(engine)

Session = sessionmaker(bind=engine)
session = Session()

# 插入
doc = Document(
    content="退款政策",
    source="manual.pdf",
    category="after_sales",
    embedding=[0.1] * 384
)
session.add(doc)
session.commit()

# 向量搜索
from sqlalchemy import text
results = session.query(Document).order_by(
    Document.embedding.cosine_distance("[0.1, ...]")
).limit(5).all()
```

---

## 常见误区

### 误区 1：每次查询都创建新的 PGVectorManager

PGVectorManager 内部维护了连接池，应该作为单例使用。每次创建新实例会创建新的连接池，浪费资源。

### 误区 2：ORM 一定比原生 SQL 慢

SQLAlchemy 的查询开销通常 < 1ms，对大多数应用可以忽略。真正影响性能的是数据库查询本身。ORM 的优势是代码可读性和类型安全，不是性能。

### 误区 3：忘记关闭连接

使用连接池时，`_put_conn` 把连接归还到池中而不是关闭。但如果你直接用 `psycopg2.connect()` 创建连接，必须手动关闭，否则会泄漏连接。

---

## 本章小结

好的封装让 pgvector 的使用更简洁、更安全。核心要点回顾：第一，PGVectorManager 封装了连接池、CRUD、批量操作和混合查询，是 RAG 应用的标准模板；第二，SQLAlchemy + pgvector.sqlalchemy 提供 ORM 集成，适合偏好面向对象风格的团队；第三，连接池应该作为单例使用，避免重复创建；第四，参数化查询防止 SQL 注入，`extras.Json()` 处理 JSONB 类型。

下一章我们将深入向量索引——IVFFlat 和 HNSW 的原理、参数调优、以及如何选择适合你场景的索引类型。
