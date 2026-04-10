# 4.2 结构化 + 向量的混合查询（pgvector 的杀手锏）

> **一条 SQL 同时做 WHERE 过滤和向量排序——这是独立向量数据库做不到的**

---

## 这一节在讲什么？

这一节是整个 pgvector 教程最核心的一节。pgvector 相比 Chroma、Milvus 等独立向量数据库的最大优势，就是"结构化+向量混合查询"——在一条 SQL 中同时做结构化数据过滤（WHERE）和向量相似度排序（ORDER BY embedding <=> query）。这种能力让你可以写出"只搜索技术类文档中最近7天创建的、与用户问题最相似的5条记录"这样的查询，而不需要在应用层做两次查询和数据合并。

---

## 为什么混合查询是 pgvector 的核心优势

在独立向量数据库中，结构化过滤和向量搜索是两个独立的步骤。以 Chroma 为例，它的 where 过滤发生在向量搜索之前——先从 SQLite 中筛选出满足条件的文档 ID，然后只在这些文档上做 HNSW 搜索。但 Chroma 的 where 过滤不支持 SQL 的全部表达能力——没有 JOIN、没有子查询、没有窗口函数、没有 JSONB 路径查询。

pgvector 的混合查询则完全利用了 SQL 的全部表达能力：

```sql
-- pgvector 混合查询：结构化过滤 + 向量搜索 + 多表关联
SELECT d.id, d.content, d.source, d.embedding <=> %s AS distance
FROM documents d
JOIN user_permissions up ON d.category = up.category
WHERE up.user_id = 42
  AND d.created_at > NOW() - INTERVAL '7 days'
  AND d.metadata @> '{"language": "zh"}'
ORDER BY d.embedding <=> %s
LIMIT 5;
```

这条 SQL 做了四件事：
1. JOIN user_permissions 表，只搜索用户有权限的分类
2. WHERE 过滤最近7天的文档
3. JSONB @> 过滤中文文档
4. ORDER BY 向量距离排序，返回 top-5

所有这些在一个原子操作中完成，不需要应用层做任何数据合并。

---

## 混合查询的典型模式

### 模式一：分类过滤 + 向量搜索

```sql
SELECT id, content, embedding <=> '[0.1, ...]' AS distance
FROM documents
WHERE category = 'tech'
ORDER BY embedding <=> '[0.1, ...]'
LIMIT 5;
```

### 模式二：时间范围 + 向量搜索

```sql
SELECT id, content, created_at, embedding <=> '[0.1, ...]' AS distance
FROM documents
WHERE created_at > NOW() - INTERVAL '30 days'
ORDER BY embedding <=> '[0.1, ...]'
LIMIT 5;
```

### 模式三：JSONB metadata + 向量搜索

```sql
SELECT id, content, metadata, embedding <=> '[0.1, ...]' AS distance
FROM documents
WHERE metadata @> '{"category": "faq"}'
  AND (metadata ->> 'version')::int >= 2
ORDER BY embedding <=> '[0.1, ...]'
LIMIT 5;
```

### 模式四：多条件组合 + 向量搜索

```sql
SELECT id, content, embedding <=> '[0.1, ...]' AS distance
FROM documents
WHERE category IN ('tech', 'science')
  AND created_at > '2024-01-01'
  AND metadata ? 'reviewed'
ORDER BY embedding <=> '[0.1, ...]'
LIMIT 5;
```

### 模式五：权限控制 + 向量搜索（JOIN）

```sql
SELECT d.id, d.content, d.embedding <=> '[0.1, ...]' AS distance
FROM documents d
JOIN user_permissions up ON d.category = up.category
WHERE up.user_id = 42
ORDER BY d.embedding <=> '[0.1, ...]'
LIMIT 5;
```

---

## 混合查询的执行原理

当你在一条 SQL 中同时使用 WHERE 和 ORDER BY 向量距离时，PostgreSQL 的查询优化器会自动选择最优的执行策略：

```
┌──────────────────────────────────────────────────────────────────┐
│  混合查询的执行策略                                              │
│                                                                  │
│  策略 A：先过滤后排序（WHERE 选择性高时）                         │
│  1. 用 B-tree/GIN 索引执行 WHERE 条件，缩小候选集                │
│  2. 在候选集上计算向量距离并排序                                  │
│  → 适合：WHERE 能过滤掉 90%+ 的数据                              │
│                                                                  │
│  策略 B：先向量搜索后过滤（WHERE 选择性低时）                     │
│  1. 用 HNSW/IVFFlat 索引做向量搜索，取 top-K                     │
│  2. 在 top-K 结果上应用 WHERE 过滤                               │
│  → 适合：WHERE 只能过滤掉少量数据                                │
│                                                                  │
│  PostgreSQL 优化器自动选择策略！                                  │
│  用 EXPLAIN ANALYZE 查看实际选择了哪种策略                       │
└──────────────────────────────────────────────────────────────────┘
```

用 EXPLAIN ANALYZE 诊断混合查询：

```sql
EXPLAIN ANALYZE
SELECT id, content, embedding <=> '[0.1, ...]' AS distance
FROM documents
WHERE category = 'tech'
ORDER BY embedding <=> '[0.1, ...]'
LIMIT 5;
```

如果看到 `Index Scan using idx_category` 然后 `Sort`，说明用了策略 A；如果看到 `Index Scan using idx_embedding_hnsw` 然后 `Filter`，说明用了策略 B。

---

## Python 完整示例

```python
import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer

conn = psycopg2.connect("postgres://postgres:password@localhost/rag_db")
register_vector(conn)
cur = conn.cursor()

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

def hybrid_search(query_text, category=None, min_version=None, days=None, top_k=5):
    """混合查询：结构化过滤 + 向量搜索"""
    query_embedding = model.encode(query_text, normalize_embeddings=True).tolist()

    conditions = []
    params = []

    if category:
        conditions.append("category = %s")
        params.append(category)
    if min_version:
        conditions.append("(metadata ->> 'version')::int >= %s")
        params.append(min_version)
    if days:
        conditions.append("created_at > NOW() - INTERVAL '%s days'")
        params.append(str(days))

    where_clause = " AND ".join(conditions) if conditions else "TRUE"

    # 距离参数需要传两次（WHERE 和 ORDER BY 各一次）
    sql = f"""
        SELECT id, content, source, metadata,
               embedding <=> %s AS distance
        FROM documents
        WHERE {where_clause}
        ORDER BY embedding <=> %s
        LIMIT %s
    """
    params_with_emb = [query_embedding] + params + [query_embedding] + [top_k]

    cur.execute(sql, params_with_emb)
    results = cur.fetchall()

    for row in results:
        doc_id, content, source, meta, distance = row
        print(f"  [{distance:.4f}] {source} | {content[:50]}...")

    return results

# 使用
hybrid_search("退款政策", category="after_sales", min_version=2, days=30, top_k=5)
```

---

## 常见误区

### 误区 1：WHERE 过滤一定能让向量搜索更快

不一定。如果 WHERE 条件的选择性很低（比如只能过滤掉 10% 的数据），PostgreSQL 可能选择先做向量搜索再过滤，因为这样更高效。优化器的选择取决于统计信息，不是固定的。

### 误区 2：混合查询中 WHERE 和 ORDER BY 向量距离不能同时使用索引

可以。PostgreSQL 可以同时使用 B-tree 索引（加速 WHERE）和 HNSW 索引（加速向量排序）。优化器会根据成本估算选择最优的组合方式。

### 误区 3：混合查询的性能一定比纯向量搜索差

不一定。如果 WHERE 条件能大幅缩小候选集，混合查询可能比纯向量搜索更快——因为向量搜索只需要在候选集上执行，而不是全表。

---

## 本章小结

混合查询是 pgvector 的杀手锏，它让你在一条 SQL 中同时做结构化过滤和向量排序。核心要点回顾：第一，混合查询利用了 SQL 的全部表达能力——WHERE、JOIN、JSONB、子查询都可以与向量搜索组合；第二，PostgreSQL 的优化器自动选择最优执行策略——先过滤后排序或先排序后过滤；第三，五种典型模式覆盖了 RAG 系统中最常见的查询需求；第四，用 EXPLAIN ANALYZE 诊断混合查询的执行计划，确保索引被正确使用。

下一节我们将用 Python 封装一个完整的 pgvector CRUD 类，包括连接池、参数化查询和异步操作。
