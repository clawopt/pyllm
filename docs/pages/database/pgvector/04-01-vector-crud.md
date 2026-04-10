# 4.1 向量数据的增删改查

> **CRUD 是数据库操作的地基——pgvector 的 CRUD 和普通 SQL 没什么两样，只是多了一个 vector 列**

---

## 这一节在讲什么？

在前面的章节中，我们已经用最简的 SQL 完成了向量搜索。但实际应用中，你需要处理更复杂的场景：批量插入大量向量、更新过时的 embedding、删除无效数据。这一节我们要把 pgvector 的 CRUD 操作讲完整——包括 Python 中的参数化插入、批量操作的性能优化、以及更新向量数据时的注意事项。

---

## INSERT：插入向量数据

### 单条插入

```sql
INSERT INTO documents (content, source, category, embedding)
VALUES ('退款政策：购买后7天内可无条件退款', 'manual.pdf', 'after_sales', '[0.1, 0.2, 0.3]');
```

### Python 参数化插入

使用 psycopg2 + register_vector，你可以直接传入 Python list 作为向量：

```python
import psycopg2
from pgvector.psycopg2 import register_vector

conn = psycopg2.connect("postgres://postgres:password@localhost/rag_db")
register_vector(conn)
cur = conn.cursor()

# 单条插入
cur.execute(
    """INSERT INTO documents (content, source, category, embedding)
       VALUES (%s, %s, %s, %s)""",
    ("退款政策：购买后7天内可无条件退款", "manual.pdf", "after_sales", [0.1, 0.2, 0.3])
)
conn.commit()
```

### 批量插入

批量插入是性能的关键——逐条插入 10000 条数据可能需要 30 秒，而批量插入只需要 2~3 秒：

```python
# 方法 1：executemany
docs = [
    ("文档1", "source1", "cat1", [0.1, 0.2, 0.3]),
    ("文档2", "source2", "cat2", [0.4, 0.5, 0.6]),
    # ... 更多数据
]
cur.executemany(
    """INSERT INTO documents (content, source, category, embedding)
       VALUES (%s, %s, %s, %s)""",
    docs
)
conn.commit()

# 方法 2：UNNEST 批量插入（更快，适合大数据量）
contents = ["文档1", "文档2", "文档3"]
sources = ["s1", "s2", "s3"]
categories = ["cat1", "cat2", "cat3"]
embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]

cur.execute(
    """INSERT INTO documents (content, source, category, embedding)
       SELECT * FROM unnest(%s::text[], %s::text[], %s::text[], %s::vector[])""",
    (contents, sources, categories, embeddings)
)
conn.commit()
```

### COPY 命令：最高效的批量导入

PostgreSQL 的 COPY 命令是批量导入最快的方式，比 INSERT 快 5~10 倍：

```python
import io
import numpy as np

def bulk_insert_with_copy(cur, table_name, rows):
    """使用 COPY 命令批量插入"""
    buffer = io.StringIO()
    for row in rows:
        content, source, category, embedding = row
        emb_str = '[' + ','.join(str(x) for x in embedding) + ']'
        line = f"{content}\t{source}\t{category}\t{emb_str}\n"
        buffer.write(line)

    buffer.seek(0)
    cur.copy_from(buffer, table_name, columns=('content', 'source', 'category', 'embedding'))
```

---

## SELECT：查询向量数据

### 基本向量搜索

```sql
-- 余弦距离搜索 top-5
SELECT id, content, source, embedding <=> '[0.1, 0.2, 0.3]' AS distance
FROM documents
ORDER BY embedding <=> '[0.1, 0.2, 0.3]'
LIMIT 5;
```

### 带距离阈值的搜索

```sql
-- 只返回距离小于 0.5 的结果
SELECT id, content, embedding <=> '[0.1, 0.2, 0.3]' AS distance
FROM documents
WHERE embedding <=> '[0.1, 0.2, 0.3]' < 0.5
ORDER BY embedding <=> '[0.1, 0.2, 0.3]'
LIMIT 5;
```

注意：这里的距离计算执行了两次（WHERE 一次、ORDER BY 一次）。PostgreSQL 的优化器通常会识别这种重复计算并缓存结果，但为了确保性能，你可以用子查询：

```sql
SELECT * FROM (
    SELECT id, content, embedding <=> '[0.1, 0.2, 0.3]' AS distance
    FROM documents
    ORDER BY distance
    LIMIT 20
) sub
WHERE distance < 0.5;
```

### Python 中的查询

```python
def vector_search(cur, query_embedding, top_k=5, max_distance=None):
    """向量搜索"""
    if max_distance:
        cur.execute(
            """SELECT id, content, source, embedding <=> %s AS distance
               FROM documents
               WHERE embedding <=> %s < %s
               ORDER BY distance
               LIMIT %s""",
            (query_embedding, query_embedding, max_distance, top_k)
        )
    else:
        cur.execute(
            """SELECT id, content, source, embedding <=> %s AS distance
               FROM documents
               ORDER BY distance
               LIMIT %s""",
            (query_embedding, top_k)
        )
    return cur.fetchall()
```

---

## UPDATE：更新向量数据

```sql
-- 更新文档内容和向量
UPDATE documents
SET content = '更新后的退款政策', embedding = '[0.15, 0.25, 0.35]'
WHERE id = 1;

-- 只更新向量（如 embedding 模型升级后重新编码）
UPDATE documents
SET embedding = '[0.11, 0.22, 0.33]'
WHERE source = 'manual.pdf';
```

**注意**：更新向量数据后，如果表上有 HNSW 索引，索引会自动增量更新。但如果是 IVFFlat 索引，大量更新后索引质量会下降，可能需要重建。

---

## DELETE：删除向量数据

```sql
-- 按 ID 删除
DELETE FROM documents WHERE id = 1;

-- 按条件批量删除
DELETE FROM documents WHERE created_at < '2023-01-01';

-- 删除所有数据
TRUNCATE documents;
```

---

## 常见误区

### 误区 1：批量插入时不使用事务

每次 INSERT 都是一次网络往返 + WAL 写入。把批量操作包在一个事务中，可以大幅减少 WAL 的刷盘次数：

```python
# ❌ 低效：每条插入都自动提交
for doc in docs:
    cur.execute("INSERT INTO documents ...", doc)
    conn.commit()  # 每次都刷盘

# ✅ 高效：一个事务中批量提交
for doc in docs:
    cur.execute("INSERT INTO documents ...", doc)
conn.commit()  # 只刷盘一次
```

### 误区 2：更新向量后不检查索引状态

IVFFlat 索引在大量更新后质量会下降（因为聚类中心不再匹配新的数据分布）。建议在大量更新后重建索引：

```sql
-- 重建 IVFFlat 索引
REINDEX INDEX idx_documents_embedding;
```

### 误区 3：查询时重复计算距离

`WHERE embedding <=> query < 0.5 ORDER BY embedding <=> query` 会计算两次距离。虽然优化器通常会缓存，但用子查询或列别名更保险。

---

## 本章小结

pgvector 的 CRUD 操作与普通 SQL 没有本质区别，只是多了一个 vector 类型的列。核心要点回顾：第一，Python 中使用 `register_vector(conn)` 后可以直接传入 list 作为向量；第二，批量插入用 `executemany` 或 `UNNEST`，超大数据量用 COPY 命令；第三，向量搜索就是 `ORDER BY embedding <=> query_vec LIMIT K`；第四，更新向量后 IVFFlat 索引可能需要重建；第五，批量操作务必放在一个事务中提交。

下一节我们将讲 pgvector 的杀手锏——结构化+向量混合查询。
