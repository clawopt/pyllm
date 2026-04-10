# 3.1 pgvector 是什么？它解决了什么问题

> **pgvector 不是一个新的数据库，而是给你已有的 PostgreSQL 装上了向量搜索的翅膀**

---

## 这一节在讲什么？

前两章我们打好了 PostgreSQL 的基础，现在终于进入正题——pgvector。这一节我们要从架构层面理解 pgvector 的定位：它不是 Chroma 或 Milvus 那样的独立向量数据库，而是 PostgreSQL 的一个扩展插件。这个定位差异决定了它的核心优势和劣势——优势是 SQL 原生、事务一致性、结构化+向量混合查询；劣势是超大规模性能不如专用向量数据库。理解这些权衡，是你决定"用 pgvector 还是用 Chroma/Milvus"的决策基础。

---

## pgvector 的定位：PostgreSQL 扩展，不是独立数据库

pgvector 的全称是 "PostgreSQL vector extension"，它通过 PostgreSQL 的扩展机制注册了新的数据类型（`vector`）、新的操作符（`<->`、`<=>`、`<#>`）和新的索引方法（IVFFlat、HNSW）。安装 pgvector 后，你可以在任何 PostgreSQL 表中添加向量列，用标准 SQL 进行向量搜索——不需要学习新的 API、不需要部署新的服务、不需要维护两套数据同步逻辑。

```
┌──────────────────────────────────────────────────────────────────┐
│  pgvector vs 独立向量数据库的架构差异                              │
│                                                                  │
│  独立向量数据库（Chroma / Milvus / Qdrant）：                    │
│  ┌──────────────┐    ┌──────────────┐                            │
│  │ 应用层       │    │ 应用层       │                            │
│  │  ↓           │    │  ↓           │                            │
│  │ PostgreSQL   │    │ Chroma/Milvus│                            │
│  │ (结构化数据) │    │ (向量数据)   │                            │
│  └──────────────┘    └──────────────┘                            │
│  → 两个系统，需要数据同步                                        │
│  → 结构化过滤和向量搜索是分离的                                   │
│                                                                  │
│  pgvector：                                                      │
│  ┌──────────────────────────────┐                                │
│  │ 应用层                       │                                │
│  │  ↓                           │                                │
│  │ PostgreSQL + pgvector        │                                │
│  │ (结构化数据 + 向量数据)       │                                │
│  └──────────────────────────────┘                                │
│  → 一个系统，无需数据同步                                        │
│  → 结构化过滤和向量搜索在同一条 SQL 中                            │
└──────────────────────────────────────────────────────────────────┘
```

---

## pgvector 的核心优势

### 优势一：SQL 原生

所有向量操作都是标准 SQL 语法，不需要学习新的查询语言：

```sql
-- 向量搜索就是 ORDER BY + 距离操作符
SELECT id, content, embedding <=> '[0.1, 0.2, ...]' AS distance
FROM documents
ORDER BY embedding <=> '[0.1, 0.2, ...]'
LIMIT 5;
```

### 优势二：结构化+向量混合查询

一条 SQL 同时做 WHERE 过滤和向量排序，这是独立向量数据库很难实现的：

```sql
SELECT id, content, embedding <=> '[0.1, ...]' AS distance
FROM documents
WHERE category = 'tech'
  AND created_at > '2024-01-01'
ORDER BY embedding <=> '[0.1, ...]'
LIMIT 5;
```

### 优势三：事务一致性

文档内容和向量数据在同一张表中，事务保证它们的原子性——不存在"内容入库了但向量没入库"的中间状态。

### 优势四：零额外运维

不需要部署和维护新的服务。PostgreSQL 的备份（pg_dump）、监控（pg_stat_statements）、高可用（流复制）等工具直接适用于 pgvector。

### 优势五：SQL JOIN 能力

你可以把文档表和用户表、权限表等通过 JOIN 关联，实现"只搜索当前用户有权限访问的文档"这类复杂逻辑：

```sql
SELECT d.id, d.content, d.embedding <=> '[0.1, ...]' AS distance
FROM documents d
JOIN user_permissions up ON d.category = up.category
WHERE up.user_id = 42
ORDER BY d.embedding <=> '[0.1, ...]'
LIMIT 5;
```

---

## pgvector 的核心劣势

### 劣势一：超大规模性能不如专用向量数据库

pgvector 的单机架构决定了它的规模上限。当向量数量超过 1 亿条时，HNSW 索引的内存消耗和查询延迟会显著增加。Milvus 的分布式架构可以水平扩展到 10 亿+ 向量。

### 劣势二：向量算法不如 FAISS 丰富

pgvector 只支持 IVFFlat 和 HNSW 两种近似索引，不支持量化索引（PQ/SQ）、不支持 DiskANN、不支持多索引组合。FAISS 提供了更丰富的算法选择。

### 劣势三：向量维度有上限

pgvector 0.7.x 版本的向量维度上限是 2000。如果你使用的 embedding 模型输出维度超过 2000（如 OpenAI 的 text-embedding-3-large 输出 3072 维），需要先降维或截断。

---

## 版本与兼容性

| pgvector 版本 | PostgreSQL 版本 | 主要特性 |
|---------------|----------------|---------|
| 0.7.x（最新） | 12~16 | HNSW 索引、半精度向量、并行查询 |
| 0.6.x | 12~16 | HNSW 索引、SP-GiST 索引 |
| 0.5.x | 11~16 | IVFFlat 索引、基础向量操作 |
| 0.4.x | 11~15 | 基础向量类型和距离计算 |

**推荐使用 pgvector 0.7.x + PostgreSQL 16**，这是目前最稳定的组合。

---

## 常见误区

### 误区 1：pgvector 可以替代 Chroma/Milvus

pgvector 和独立向量数据库不是替代关系，而是互补关系。小规模（< 10M 向量）且需要结构化查询的场景用 pgvector；大规模（> 100M 向量）或纯向量检索场景用 Milvus/Chroma。

### 误区 2：pgvector 的性能和专用向量数据库一样好

在同等硬件条件下，pgvector 的向量搜索性能通常不如 Milvus 或 Qdrant——因为 PostgreSQL 的通用查询引擎不是为向量搜索专门优化的。pgvector 的优势在于混合查询和事务一致性，而不是纯向量搜索的极致性能。

---

## 本章小结

pgvector 是 PostgreSQL 的向量搜索扩展，它的核心定位是"给已有的 PostgreSQL 加上向量能力"，而不是替代独立的向量数据库。核心要点回顾：第一，pgvector 的五大优势是 SQL 原生、混合查询、事务一致性、零额外运维、SQL JOIN 能力；第二，pgvector 的三大劣势是超大规模性能不如专用方案、向量算法不如 FAISS 丰富、维度上限 2000；第三，pgvector 和独立向量数据库是互补关系，不是替代关系；第四，推荐使用 pgvector 0.7.x + PostgreSQL 16。

下一节我们将动手安装 pgvector 并执行第一次向量搜索。
