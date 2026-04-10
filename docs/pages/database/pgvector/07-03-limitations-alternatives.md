# 7.3 pgvector 的局限性与替代方案

> **没有银弹——pgvector 很好，但不是所有场景都适合它，知道什么时候该换同样重要**

---

## 这一节在讲什么？

前面六章我们一直在夸 pgvector——SQL 原生查询、事务一致性、混合查询、成熟的运维工具，听起来简直完美。但任何技术都有它的边界和局限，pgvector 也不例外。如果你不了解它的局限，就可能在错误的场景下强行使用它，最终遇到性能瓶颈或功能缺失时措手不及。这一节是本教程的最后一节，我们要诚实地聊一聊 pgvector 的局限性、什么时候该用 pgvector、什么时候该换用其他方案、以及如何让 pgvector 和其他方案共存。理解这些边界，你才能在技术选型时做出正确的判断。

---

## pgvector 的已知局限

### 局限1：单机架构，不支持原生分布式

pgvector 运行在 PostgreSQL 上，而 PostgreSQL 是一个单机数据库——它的数据存储、查询执行、索引构建都在一台机器上完成。虽然你可以通过流复制实现读写分离，但"写"操作始终只能在主库上执行，数据量增长到单机无法承载时，你没有像 Milvus 那样的分片（Sharding）机制来水平扩展。

```
pgvector 的扩展天花板：

  单机 PostgreSQL + pgvector
  ┌────────────────────────────────────────┐
  │  内存：128GB（高端服务器）              │
  │  HNSW 索引：约占数据量的 2~3 倍        │
  │  实际可承载：约 1 亿条 768 维向量       │
  │  （取决于内存和磁盘配置）               │
  └────────────────────────────────────────┘

  分布式向量数据库（如 Milvus）
  ┌──────────┐  ┌──────────┐  ┌──────────┐
  │  Node 1  │  │  Node 2  │  │  Node 3  │
  │  1亿向量  │  │  1亿向量  │  │  1亿向量  │
  └──────────┘  └──────────┘  └──────────┘
  → 理论上无限水平扩展
```

对于大多数 RAG 应用来说，1 亿条向量已经是一个非常宽裕的上限了——1000 万篇文档、每篇切 10 个 chunk、每个 chunk 一个 768 维向量，总共也就 1 亿条。但如果你做的是全网搜索引擎、电商全量商品检索这种十亿级甚至百亿级的场景，pgvector 就力不从心了。

### 局限2：HNSW 索引全内存驻留，内存消耗高

HNSW 索引的查询速度依赖于全内存访问——如果索引页面不在内存中，需要从磁盘读取，查询延迟就会从毫秒级飙升到百毫秒级。这意味着你的服务器内存必须能装下整个 HNSW 索引。

```
HNSW 索引的内存需求估算：

  数据量：100 万条 768 维向量
  原始数据大小：100万 × 768 × 4字节 ≈ 2.9GB
  HNSW 索引大小（m=16）：约 5~8GB（原始数据的 2~3 倍）
  推荐内存：8GB × 3 = 24GB（索引 + shared_buffers + OS缓存）

  数据量：1000 万条 768 维向量
  原始数据大小：约 29GB
  HNSW 索引大小：约 60~90GB
  推荐内存：128GB+

  数据量：1 亿条 768 维向量
  原始数据大小：约 290GB
  HNSW 索引大小：约 600~900GB
  推荐内存：1TB+（需要高端服务器）
```

相比之下，Milvus 和 Qdrant 支持向量量化（Quantization）——把 float32 的向量压缩成 int8 甚至二值向量，内存占用可以减少 4~32 倍，代价是精度略有下降。pgvector 目前不支持任何形式的向量量化。

### 局限3：向量维度上限 2000

pgvector 0.7.x 版本的向量维度上限是 2000——这意味着你不能用它存储维度超过 2000 的向量。目前主流的 Embedding 模型输出维度都在这个范围内（OpenAI text-embedding-3-small 是 1536 维，all-MiniLM-L6-v2 是 384 维，BGE-base-zh 是 768 维），所以大多数场景下 2000 维的上限不是问题。但如果你使用某些高维模型（比如 Cohere 的 multilingual-22-12 输出 768 维倒没问题，但一些多模态模型可能输出更高维度的向量），就需要注意这个限制。

```sql
-- pgvector 的维度上限
CREATE TABLE test (emb vector(2000));   -- ✅ 最大允许值
CREATE TABLE test (emb vector(2001));   -- ❌ 报错：dimension must be between 1 and 2000
```

### 局限4：不支持量化索引（PQ/SQ）

量化索引是大规模向量搜索的关键技术——它把高维向量压缩成低维的编码，大幅减少内存占用和距离计算的开销。主流的量化方法包括：

- **PQ（Product Quantization）**：把向量切成多个子空间，每个子空间独立量化，内存减少 4~8 倍
- **SQ（Scalar Quantization）**：把 float32 压缩成 int8，内存减少 4 倍
- **二值量化**：把向量压缩成 0/1 位，内存减少 32 倍

Milvus、Qdrant、Weaviate 都支持某种形式的量化索引，但 pgvector 目前只支持全精度的 IVFFlat 和 HNSW——所有向量都以 float32 原始精度存储和计算，没有压缩机制。这意味着在同样的内存预算下，pgvector 能承载的向量数量只有支持量化的数据库的 1/4 甚至更少。

### 局限5：IVFFlat 增量数据量大时需要重建索引

我们在第 5 章详细讨论过这个问题——IVFFlat 的索引质量依赖于建索引时的数据分布。如果建索引后新增了大量数据，这些新数据会被分配到已有的聚类中心中，但聚类中心本身不会更新，导致搜索时新数据的召回率下降。当新增数据量超过原始数据量的 20%~30% 时，你需要重建 IVFFlat 索引。

HNSW 没有这个问题——它支持增量更新，新插入的向量会自动融入图结构。但 HNSW 的代价是更大的内存占用和更慢的构建速度。

---

## 什么时候该用 pgvector

了解了局限之后，我们反过来问——在什么场景下 pgvector 是最佳选择？

### 场景1：已有 PostgreSQL 基础设施

这是 pgvector 最天然的使用场景——你的业务系统已经在用 PostgreSQL 存储用户、订单、商品等结构化数据，现在需要加一个语义搜索功能。与其引入一个全新的向量数据库（增加运维复杂度、数据同步问题、团队学习成本），不如直接在现有的 PostgreSQL 上装一个 pgvector 扩展——零额外组件、零数据迁移、SQL 原生查询。

```python
# pgvector 的"零额外组件"优势
# 不需要新数据库，不需要新连接，不需要数据同步

import psycopg2
from pgvector.psycopg2 import register_vector

# 用同一个连接同时操作结构化数据和向量数据
conn = psycopg2.connect(DATABASE_URL)
register_vector(conn)
cur = conn.cursor()

# 一条 SQL 同时做结构化过滤和向量搜索
cur.execute("""
    SELECT p.name, p.price, p.embedding <=> %s AS distance
    FROM products p
    JOIN categories c ON p.category_id = c.id
    WHERE c.name = 'electronics'
      AND p.price < 1000
      AND p.in_stock = true
    ORDER BY p.embedding <=> %s
    LIMIT 5
""", (query_embedding, query_embedding))
```

如果用独立的向量数据库，你需要维护两套数据库之间的数据同步——商品信息在 PostgreSQL，向量在 Milvus，每次商品信息更新都要同步到 Milvus，这个同步链路本身就是故障点。

### 场景2：需要结构化数据和向量数据的强一致性

在 RAG 系统中，文档入库是一个多步操作——插入文档元数据（标题、来源、分类）、插入文档内容、计算 embedding、插入向量。如果这些操作不在同一个事务中，就可能出现"有元数据没向量"或"有向量没元数据"的不一致状态。pgvector 的 ACID 事务保证了这些操作的原子性——要么全部成功，要么全部回滚。

```python
# pgvector 的事务保证
def insert_document(cur, doc, embedding):
    """文档入库：元数据和向量在同一个事务中"""
    try:
        cur.execute("""
            INSERT INTO documents (source, category, content, embedding, metadata)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
        """, (doc.source, doc.category, doc.content, embedding, doc.metadata))
        doc_id = cur.fetchone()[0]

        # 同时更新分类统计表（在同一个事务中）
        cur.execute("""
            UPDATE category_stats
            SET doc_count = doc_count + 1
            WHERE category = %s
        """, (doc.category,))

        conn.commit()
        return doc_id
    except Exception as e:
        conn.rollback()
        raise e
```

独立的向量数据库通常不支持跨表事务——你在 PostgreSQL 插入元数据成功后，如果 Milvus 插入向量失败，就需要手动回滚 PostgreSQL 的数据，这增加了系统的复杂度和出错概率。

### 场景3：数据量 < 1000 万向量

1000 万条 768 维向量的 HNSW 索引大约需要 60~90GB 内存，这在现代服务器上是完全可行的。在这个规模下，pgvector 的查询性能（5~20ms）完全可以满足大多数 RAG 应用的需求，没有必要引入更复杂的分布式方案。

### 场景4：需要复杂的 SQL 查询能力

如果你的查询不只是"找最相似的 K 个向量"，还涉及多表 JOIN、子查询、窗口函数、聚合统计等复杂操作，pgvector 的 SQL 能力就是不可替代的。比如"找到每个分类下最相似的 3 篇文档"——这在 pgvector 里是一条 SQL，在独立向量数据库里可能需要多次查询 + 应用层聚合。

```sql
-- 每个分类下最相似的 3 篇文档（窗口函数 + 向量搜索）
SELECT * FROM (
    SELECT id, content, category,
           embedding <=> '[0.1, ...]' AS distance,
           ROW_NUMBER() OVER (PARTITION BY category ORDER BY embedding <=> '[0.1, ...]') AS rank
    FROM documents
) ranked
WHERE rank <= 3
ORDER BY category, rank;
```

---

## 什么时候该换

### 场景1：数据量 > 1 亿向量

当数据量超过 1 亿条时，单机 PostgreSQL 的存储和内存都很难承载——HNSW 索引可能需要几百 GB 甚至 TB 级别的内存，这超出了单机服务器的合理配置范围。此时你需要分布式向量数据库（Milvus、Qdrant）来水平扩展。

### 场景2：需要分布式架构

如果你的系统需要多数据中心部署、跨地域复制、或者弹性扩缩容，pgvector 的单机架构就无法满足——你需要 Milvus 这种原生分布式的方案。

### 场景3：需要向量压缩（量化）

当内存预算有限但数据量很大时，向量量化是必须的——PQ 可以把内存占用减少 4~8 倍，二值量化可以减少 32 倍。pgvector 不支持任何量化方法，如果你的内存装不下全精度的 HNSW 索引，就需要考虑 Milvus 或 Qdrant。

### 场景4：需要多模态内置支持

如果你的应用涉及图像搜索、音频搜索、视频搜索等多模态场景，Weaviate 提供了内置的多模态模型集成——你可以直接上传图片，Weaviate 自动调用模型生成 embedding 并存储。pgvector 只负责向量存储和搜索，多模态的模型调用需要你自己实现。

---

## 主流向量数据库对比

| 维度 | pgvector | Milvus | Qdrant | Weaviate | Chroma |
|------|----------|--------|--------|----------|--------|
| 架构 | 单机（PG 扩展） | 分布式 | 单机/分布式 | 单机/分布式 | 嵌入式/客户端-服务器 |
| 最大规模 | ~1 亿 | 10 亿+ | 数亿 | 数亿 | ~1000 万 |
| 索引类型 | IVFFlat, HNSW | IVFFlat, HNSW, PQ, SCANN, DiskANN | HNSW, PQ | HNSW | HNSW |
| 向量量化 | ❌ | ✅ PQ/SQ/BQ | ✅ PQ/SQ | ✅ PQ/BQ | ❌ |
| 混合查询 | ✅ SQL WHERE | ✅ 标量过滤 | ✅ Payload 过滤 | ✅ GraphQL 过滤 | ✅ where 过滤 |
| 事务支持 | ✅ 完整 ACID | ❌ | ❌ | ❌ | ❌ |
| 多表 JOIN | ✅ | ❌ | ❌ | ❌ | ❌ |
| 部署复杂度 | 低 | 高 | 中 | 中 | 低 |
| 运维工具 | 成熟（PG 生态） | 专用 | 专用 | 专用 | 简单 |
| 学习曲线 | 低（会 SQL 就行） | 高 | 中 | 中 | 低 |
| 适合场景 | 结构化+向量混合 | 超大规模 | 中大规模+过滤 | 多模态 | 快速原型 |

---

## pgvector 与其他方案共存：混合架构

在实际生产中，最常见的情况不是"只用 pgvector"或"只用 Milvus"，而是两者共存——pgvector 处理结构化数据和中等规模的向量搜索，Milvus 处理超大规模的纯向量搜索。

```
混合架构示例：

  用户请求
     │
     ▼
  ┌──────────────────────────────────────────┐
  │              应用层                       │
  │                                          │
  │  if 需要结构化过滤:                       │
  │      → pgvector（WHERE + 向量搜索）       │
  │  elif 超大规模纯向量搜索:                 │
  │      → Milvus（分布式 + 量化）            │
  │  else:                                   │
  │      → pgvector（默认选择）               │
  └──────────────────────────────────────────┘
     │                    │
     ▼                    ▼
  ┌──────────┐    ┌──────────┐
  │ pgvector │    │  Milvus  │
  │          │    │          │
  │ 用户表    │    │ 全量向量  │
  │ 订单表    │    │ PQ 索引  │
  │ 热门文档  │    │          │
  │ (100万)  │    │ (1亿+)   │
  └──────────┘    └──────────┘
```

```python
class HybridVectorSearch:
    """混合架构：pgvector + Milvus"""

    def __init__(self, pg_conn, milvus_collection):
        self.pg_cur = pg_conn.cursor()
        self.milvus = milvus_collection

    def search(self, query_embedding, filters=None, top_k=5):
        """根据查询需求选择合适的搜索引擎"""
        if filters and any(f in filters for f in ['user_id', 'category', 'price_range']):
            # 需要结构化过滤 → pgvector 的混合查询更高效
            return self._search_pgvector(query_embedding, filters, top_k)
        else:
            # 纯向量搜索 → Milvus 的分布式检索更快
            return self._search_milvus(query_embedding, top_k)

    def _search_pgvector(self, query_embedding, filters, top_k):
        """pgvector 混合查询：结构化过滤 + 向量搜索"""
        where_clauses = []
        params = [query_embedding, query_embedding]

        if 'user_id' in filters:
            where_clauses.append("user_id = %s")
            params.append(filters['user_id'])
        if 'category' in filters:
            where_clauses.append("category = %s")
            params.append(filters['category'])

        where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"
        params.append(top_k)

        self.pg_cur.execute(f"""
            SELECT id, content, embedding <=> %s AS distance
            FROM documents
            WHERE {where_sql}
            ORDER BY embedding <=> %s
            LIMIT %s
        """, params)
        return self.pg_cur.fetchall()

    def _search_milvus(self, query_embedding, top_k):
        """Milvus 纯向量搜索"""
        results = self.milvus.search(
            data=[query_embedding],
            limit=top_k,
            output_fields=["id", "content"]
        )
        return [(r['id'], r['content'], r['distance']) for r in results[0]]
```

混合架构的关键是**数据同步**——你需要确保 pgvector 和 Milvus 中的数据是一致的。一个常见的策略是"pgvector 为主、Milvus 为辅"：所有数据先写入 pgvector（利用事务保证一致性），然后异步同步到 Milvus。这样即使 Milvus 的同步出现延迟或失败，pgvector 中的数据始终是完整和一致的。

```python
def insert_document_hybrid(cur, milvus, doc, embedding):
    """混合架构的数据写入：pgvector 为主，Milvus 异步同步"""

    # 第一步：写入 pgvector（主存储，事务保证）
    cur.execute("""
        INSERT INTO documents (source, category, content, embedding, metadata)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING id
    """, (doc.source, doc.category, doc.content, embedding, doc.metadata))
    doc_id = cur.fetchone()[0]

    # 第二步：异步写入 Milvus（辅助索引，允许短暂延迟）
    try:
        milvus.insert([{
            "id": doc_id,
            "content": doc.content,
            "embedding": embedding,
            "category": doc.category
        }])
    except Exception as e:
        # Milvus 写入失败不影响主流程
        # 后台任务会定期检查并补齐缺失的数据
        log.warning(f"Milvus sync failed for doc {doc_id}: {e}")

    return doc_id
```

### 常见误区：一上来就选分布式方案

很多团队在项目初期就选择了 Milvus 或 Qdrant，理由是"万一以后数据量很大呢"。但分布式向量数据库的运维复杂度远高于 pgvector——你需要管理集群节点、配置分片策略、处理节点故障、维护数据均衡。如果你的数据量只有几十万条，这些额外的运维成本完全是浪费。正确的做法是**先用 pgvector 快速上线，等数据量真的接近瓶颈时再引入分布式方案**——这就是混合架构的价值所在。

---

## 技术选型的决策框架

最后，我给你一个简单的决策框架，帮助你在 pgvector 和其他方案之间做出选择：

```
你的向量搜索需求是什么？
│
├─ 需要结构化过滤（WHERE 条件）？
│   ├─ 是 → pgvector（混合查询是杀手锏）
│   └─ 否 → 继续
│
├─ 数据量有多大？
│   ├─ < 1000 万 → pgvector（简单够用）
│   ├─ 1000 万 ~ 1 亿 → pgvector 或 Qdrant（取决于内存）
│   └─ > 1 亿 → Milvus（必须分布式）
│
├─ 已有 PostgreSQL 基础设施？
│   ├─ 是 → pgvector（零额外运维）
│   └─ 否 → 继续
│
├─ 需要快速原型验证？
│   ├─ 是 → Chroma（最快上手）
│   └─ 否 → 继续
│
├─ 需要多模态内置支持？
│   ├─ 是 → Weaviate
│   └─ 否 → pgvector
│
└─ 内存预算有限但数据量大？
    ├─ 是 → Milvus（支持向量量化）
    └─ 否 → pgvector
```

这个决策框架的核心原则是：**简单优先，按需升级**。pgvector 的最大优势不是性能——而是它让你用最简单的方式在现有的 PostgreSQL 上获得向量搜索能力。当你真的遇到 pgvector 无法解决的瓶颈时，再引入更复杂的方案也不迟。

---

## 教程总结

七个章节走下来，我们从 PostgreSQL 的基础操作一路走到了 pgvector 的生产部署和替代方案。让我用几句话回顾一下整个教程的核心脉络：

**第 1~2 章**我们打下了 PostgreSQL 的基础——SQL CRUD、索引、JSONB、事务，这些不只是"前置知识"，更是 pgvector 区别于独立向量数据库的根本所在：正是因为 PostgreSQL 有成熟的索引架构、灵活的 JSONB 类型、完整的 ACID 事务，pgvector 才能实现"结构化数据 + 向量数据在一条 SQL 中查询"这个杀手锏功能。

**第 3~4 章**我们进入了 pgvector 的核心——vector 数据类型、距离操作符、混合查询。其中混合查询是 pgvector 最重要的能力：`WHERE category = 'tech' ORDER BY embedding <=> query_vec LIMIT 5` 这一条 SQL，在独立向量数据库里需要两步操作（先向量搜索再结构化过滤），而且无法保证一致性。

**第 5 章**我们深入了向量索引——IVFFlat 和 HNSW 的原理、参数调优、以及距离操作符与索引 ops 的对应关系（这是最容易踩坑的地方）。索引不是万能的——数据量小于 1 万条时暴力搜索更快，索引选错了操作符等于白建。

**第 6 章**我们把 pgvector 放到了 RAG 系统中——文档问答、对话记忆、用户画像，这些场景都充分利用了 pgvector 的 SQL 能力：JOIN 关联用户表、WHERE 过期清理、UNION 合并多源搜索结果。

**第 7 章**我们聊了生产环境必须面对的问题——部署方案、性能调优、监控运维、局限性和替代方案。pgvector 不是银弹，但在大多数 RAG 场景下，它是最务实的选择。

最后，如果你要记住这个教程的一句话，那就是：**pgvector 的核心价值不在于向量搜索本身——而在于它让你用 SQL 的全部能力来管理向量数据**。当你需要在结构化过滤和语义搜索之间无缝切换时，当你需要事务保证数据一致性时，当你需要 JOIN 关联多张表时，pgvector 就是那个最简单、最可靠、最不需要额外运维的选择。
