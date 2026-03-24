# PG Vector 教程

PG Vector 是 PostgreSQL 的向量扩展。

## 安装

```sql
CREATE EXTENSION vector;
```

## Python 客户端

```bash
pip install psycopg2 pgvector
```

## 基本使用

```python
import psycopg2
from pgvector.psycopg2 import register_vector

conn = psycopg2.connect("postgres://user:pass@localhost/db")
register_vector(conn)
cur = conn.cursor()
cur.execute("INSERT INTO items (content, embedding) VALUES (%s, %s)", ("Hello", [0.1, 0.2, 0.3]))
conn.commit()
```

## 下一步

继续学习：
- [Milvus教程](/pages/database/milvus/)
- [Chroma教程](/pages/database/chroma/)
