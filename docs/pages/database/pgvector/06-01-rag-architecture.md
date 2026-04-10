# 6.1 RAG 架构中 pgvector 的角色

> **pgvector 在 RAG 中不是"另一个 Chroma"——它是把结构化数据和向量数据统一在一个系统中的方案**

---

## 这一节在讲什么？

前面五章我们学会了 pgvector 的安装、CRUD、混合查询和索引调优，但一直在工具层面打转。这一节我们要从架构层面理解 pgvector 在 RAG 系统中的独特定位——它和 Chroma 不是替代关系，而是互补关系。Chroma 适合快速原型，pgvector 适合需要结构化数据和事务一致性的生产系统。理解这个定位差异，是你做技术选型的关键。

---

## pgvector vs Chroma 在 RAG 中的定位差异

| 维度 | Chroma | pgvector |
|------|--------|----------|
| 核心定位 | 纯向量数据库 | 关系型数据库 + 向量能力 |
| 数据模型 | Document (id/text/embedding/metadata) | Table Row (多列 + embedding 列) |
| 结构化查询 | where (Python dict，有限语法) | WHERE (SQL，完整表达力) |
| 多表关联 | 不支持 | SQL JOIN |
| 事务 | 无 | 完整 ACID |
| 混合查询 | where 过滤 + 向量搜索（两步） | 一条 SQL 同时完成 |
| 适合规模 | < 10M 向量 | < 100M 向量（取决于内存） |
| 运维工具 | 有限 | pg_dump / pg_stat_statements / 流复制 |
| 学习曲线 | 极低 | 需要了解 PostgreSQL |

---

## pgvector 的四大独特优势

### 优势一：结构化数据 + 向量数据在同一张表

在 Chroma 中，文档的 metadata 是一个扁平的字典，查询能力有限。在 pgvector 中，你可以把核心字段（category、source、created_at）作为独立列建 B-tree 索引，灵活字段放在 JSONB 列建 GIN 索引，向量放在 embedding 列建 HNSW 索引——三种索引协同工作：

```sql
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    source VARCHAR(255),
    category VARCHAR(100),
    embedding vector(384),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- 三种索引协同
CREATE INDEX idx_category ON documents (category);                    -- B-tree
CREATE INDEX idx_metadata ON documents USING GIN (metadata);         -- GIN
CREATE INDEX idx_embedding ON documents USING hnsw (embedding vector_cosine_ops); -- HNSW
```

### 优势二：SQL JOIN 能力

你可以把文档表和用户权限表通过 JOIN 关联，实现"只搜索当前用户有权限访问的文档"：

```sql
SELECT d.id, d.content, d.embedding <=> %s AS distance
FROM documents d
JOIN user_permissions up ON d.category = up.category
WHERE up.user_id = 42
ORDER BY d.embedding <=> %s
LIMIT 5;
```

这在 Chroma 中无法实现——它没有 JOIN 能力，你必须在应用层做权限过滤。

### 优势三：事务保证

文档入库时，content 和 embedding 在同一个事务中写入，保证原子性。不存在"内容入库了但向量没入库"的不一致状态。

### 优势四：成熟的运维工具

PostgreSQL 有 30+ 年的运维工具积累——pg_dump 备份、pg_stat_statements 慢查询分析、流复制高可用、pgbouncer 连接池。这些工具直接适用于 pgvector，不需要额外的运维体系。

---

## 何时选择 pgvector vs Chroma

```
┌──────────────────────────────────────────────────────────────────┐
│  pgvector vs Chroma 选型指南                                     │
│                                                                  │
│  选 pgvector 当：                                                │
│  ✅ 已有 PostgreSQL 基础设施                                     │
│  ✅ 需要结构化数据和向量数据的强一致性                            │
│  ✅ 需要 SQL JOIN 等复杂查询能力                                  │
│  ✅ 需要事务保证                                                 │
│  ✅ 数据量在 10M~100M 之间                                       │
│  ✅ 团队熟悉 SQL                                                 │
│                                                                  │
│  选 Chroma 当：                                                  │
│  ✅ 快速原型开发，零配置启动                                      │
│  ✅ 纯向量检索，不需要复杂结构化查询                              │
│  ✅ 数据量 < 10M                                                 │
│  ✅ 团队更熟悉 Python 而非 SQL                                   │
│  ✅ 不想管理数据库运维                                           │
└──────────────────────────────────────────────────────────────────┘
```

---

## 常见误区

### 误区 1：pgvector 可以完全替代 Chroma

pgvector 和 Chroma 是互补方案，不是替代关系。快速原型用 Chroma 更快，生产系统用 pgvector 更稳。有些团队甚至在开发阶段用 Chroma，上线后迁移到 pgvector。

### 误区 2：pgvector 的向量搜索性能和 Chroma 一样

在纯向量搜索场景下，Chroma 通常比 pgvector 更快——因为 Chroma 是专门为向量搜索优化的。pgvector 的优势在于混合查询和事务一致性，而不是纯向量搜索的极致性能。

---

## 本章小结

pgvector 在 RAG 架构中的定位是"关系型数据库 + 向量能力"，而不是"另一个 Chroma"。核心要点回顾：第一，pgvector 的四大独特优势是同表存储、SQL JOIN、事务保证、成熟运维工具；第二，pgvector 适合已有 PostgreSQL 基础设施且需要结构化查询的生产系统；第三，Chroma 适合快速原型和纯向量检索场景；第四，两者是互补关系，不是替代关系。

下一节我们将用 pgvector 构建一个端到端的 RAG 文档问答系统。
