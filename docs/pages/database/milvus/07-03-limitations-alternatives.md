# 7.3 Milvus 的局限性与替代方案

> **Milvus 很强大，但不是万能的——知道什么时候该换同样重要**

---

## 这一节在讲什么？

前面六章我们一直在夸 Milvus——分布式架构、量化索引、多向量搜索、Partition 多租户，听起来很完美。但任何技术都有边界和局限，Milvus 也不例外。这一节是本教程的最后一节，我们要诚实地聊一聊 Milvus 的局限性、什么时候该用 Milvus、什么时候该换用其他方案、以及如何让 Milvus 和其他方案共存。

---

## Milvus 的已知局限

### 局限1：无 SQL 支持

Milvus 不是关系型数据库——它不支持 SQL，不能做 JOIN、子查询、窗口函数。这意味着所有复杂查询都需要在应用层拼接。比如"找到每个分类下最相似的 3 篇文档"——在 pgvector 里是一条 SQL（窗口函数 + 向量搜索），在 Milvus 里需要先按分类查询所有分类，然后对每个分类分别做向量搜索，最后在应用层合并。

### 局限2：无事务保证

Milvus 不支持跨 Collection 的原子操作——如果你需要同时往 Milvus 插入向量数据和往 PostgreSQL 插入业务数据，这两个操作无法在同一个事务中完成。如果其中一个失败，你需要手动处理数据不一致。

### 局限3：运维复杂度高

Milvus Cluster 依赖 etcd、MinIO/S3、Pulsar/Kafka 三个外部组件——每个组件都有自己的运维需求。etcd 需要定期备份和压缩，MinIO 需要配置生命周期策略，Pulsar/Kafka 需要监控积压。当你遇到问题时，需要判断是 Milvus 本身的问题还是依赖组件的问题——排障门槛比 pgvector 高得多。

### 局限4：一致性模型较复杂

刚写入的数据可能"搜不到"——这是分布式系统的正常行为，但对习惯了 pgvector ACID 事务的开发者来说很不直观。你需要理解四种一致性级别（Strong/Bounded/Session/Eventually）并选择合适的级别，否则可能遇到"数据丢了"的假象。

### 局限5：小规模场景"杀鸡用牛刀"

当数据量只有几十万条时，Milvus 的分布式开销反而会让搜索比 pgvector 慢——因为数据需要经过消息队列、DataNode、对象存储、QueryNode 的完整链路，而 pgvector 直接在内存中搜索。Milvus 的优势只在数据量超过单机承载能力时才体现出来。

---

## 什么时候该用 Milvus

| 场景 | 为什么选 Milvus |
|------|---------------|
| 数据量 > 1000 万向量，且持续增长 | 分布式架构可以水平扩展 |
| 需要向量量化（PQ/SQ/BQ）节省内存 | pgvector 不支持量化 |
| 需要多租户物理隔离 | Partition 实现物理隔离 |
| 需要多向量搜索（多模态） | Hybrid Search + RRF |
| 需要水平扩展 | 加 QueryNode 即可扩容 |

---

## 什么时候该换

| 场景 | 推荐替代方案 | 原因 |
|------|------------|------|
| 数据量 < 1000 万 | pgvector 或 Chroma | 更简单，运维成本低 |
| 需要 SQL JOIN 和事务 | pgvector | Milvus 不支持 SQL 和事务 |
| 需要多模态内置模型 | Weaviate | Weaviate 内置多模态模型 |
| 需要极致单机性能 | Qdrant | Qdrant 的单机性能优于 Milvus |
| 需要嵌入式向量数据库 | Chroma | Milvus 不支持嵌入式 |

---

## 主流向量数据库对比

| 维度 | Milvus | pgvector | Chroma | Qdrant | Weaviate |
|------|--------|----------|--------|--------|----------|
| 架构 | 分布式 | 单机 | 嵌入式 | 单机/分布式 | 单机/分布式 |
| 最大规模 | 10 亿+ | ~1 亿 | ~1000 万 | 数亿 | 数亿 |
| 索引类型 | 8 种 | 2 种 | 1 种 | 2 种 | 1 种 |
| 向量量化 | ✅ PQ/SQ/BQ | ❌ | ❌ | ✅ PQ/SQ | ✅ PQ/BQ |
| SQL 支持 | ❌ | ✅ | ❌ | ❌ | ❌ |
| 事务 | ❌ | ✅ ACID | ❌ | ❌ | ❌ |
| 多向量搜索 | ✅ | ❌ | ❌ | ❌ | ✅ |
| 分布式 | ✅ | ❌ | ❌ | ✅ | ✅ |
| 运维复杂度 | 高 | 低 | 极低 | 中 | 中 |
| 学习曲线 | 中~高 | 低 | 低 | 中 | 中 |

---

## 混合架构：pgvector + Milvus 共存

在实际生产中，最常见的是 pgvector 和 Milvus 共存——pgvector 处理结构化数据和中等规模向量搜索，Milvus 处理超大规模纯向量搜索：

```python
class HybridVectorSearch:
    """混合架构：pgvector + Milvus"""

    def __init__(self, pg_conn, milvus_client):
        self.pg_cur = pg_conn.cursor()
        self.milvus = milvus_client

    def search(self, query_embedding, filters=None, top_k=5):
        if filters and any(f in filters for f in ['user_id', 'price_range', 'join_query']):
            # 需要结构化过滤 → pgvector
            return self._search_pgvector(query_embedding, filters, top_k)
        else:
            # 纯向量搜索 → Milvus
            return self._search_milvus(query_embedding, top_k)

    def _search_pgvector(self, query_embedding, filters, top_k):
        where_clauses = []
        params = [query_embedding, query_embedding]
        if 'user_id' in filters:
            where_clauses.append("user_id = %s")
            params.append(filters['user_id'])
        where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"
        params.append(top_k)
        self.pg_cur.execute(f"""
            SELECT id, content, embedding <=> %s AS distance
            FROM documents WHERE {where_sql}
            ORDER BY embedding <=> %s LIMIT %s
        """, params)
        return self.pg_cur.fetchall()

    def _search_milvus(self, query_embedding, top_k):
        results = self.milvus.search(
            collection_name="documents",
            data=[query_embedding],
            limit=top_k,
            output_fields=["content"]
        )
        return [(hit['id'], hit['entity']['content'], hit['distance']) for hit in results[0]]
```

混合架构的关键是**数据同步**——pgvector 为主存储（利用事务保证一致性），Milvus 为辅助索引（异步同步）。这样即使 Milvus 同步延迟，pgvector 中的数据始终是完整和一致的。

---

## 技术选型的核心原则

最后，我给你一个简单的选型原则：**简单优先，按需升级**。

- 如果你的数据量只有几十万条，用 Chroma 或 pgvector——别碰 Milvus
- 如果你的数据量到了千万级且需要 SQL，用 pgvector——别急着上分布式
- 如果你的数据量到了亿级或者需要量化/多租户/多向量，才需要 Milvus
- 如果你不想运维任何基础设施，用 Zilliz Cloud

没有银弹——只有最适合你当前阶段的选择。过早引入分布式方案，跟杀鸡用牛刀没什么区别。

---

## 教程总结

七个章节走下来，我们从 Milvus 的架构一路走到了生产部署和替代方案。让我用几句话回顾整个教程的核心脉络：

**第 1~2 章**我们理解了 Milvus 的定位和架构——存算分离的云原生设计，三层组件（协调器/工作节点/存储）各司其职，理解架构是调优和排障的前提。

**第 3~4 章**我们掌握了 Milvus 的核心操作——CRUD、search()、expr 过滤、8 种索引类型。其中量化索引（PQ/SQ/BQ）是 Milvus 相比 pgvector/Chroma 最核心的优势。

**第 5 章**我们探索了高级特性——Partition 多租户、多向量搜索 + RRF、动态字段与 JSON 过滤、一致性级别与持久化。

**第 6~7 章**我们把 Milvus 放到了 RAG 系统和生产环境中——MilvusRAG Demo、多租户隔离、集群部署、性能监控。

如果你要记住这个教程的一句话，那就是：**Milvus 的核心价值不在于向量搜索本身——而在于它让十亿级向量搜索变得可管理、可扩展、可量化**。当你需要在大规模数据面前保持毫秒级检索时，Milvus 就是最务实的选择。
