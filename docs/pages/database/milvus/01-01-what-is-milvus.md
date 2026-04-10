# 1.1 Milvus 是什么？为什么需要分布式向量数据库

> **当数据量从百万到十亿——单机扛不住的时候，Milvus 登场了**

---

## 这一节在讲什么？

如果你已经学过本系列的 Chroma 和 pgvector 教程，你应该已经能用向量数据库做语义搜索了——Chroma 零配置上手快，pgvector 用 SQL 就能搜向量。但你有没有想过一个问题：当你的向量数据从 100 万条增长到 1 亿条、10 亿条的时候，会发生什么？答案是单机方案会扛不住——内存装不下、搜索变慢、写入跟不上。Milvus 就是为此而生的：它是一个分布式向量数据库，专门解决"数据太大、单机不够"的问题。这一节我们要聊清楚 Milvus 的定位、它和 Chroma/pgvector 的根本区别、以及你到底在什么场景下才需要它。

---

## 从单机到分布式：为什么需要 Milvus

让我们先用一个具体的数字来建立直觉。假设你有一个文档问答系统，每篇文档切成 10 个 chunk，每个 chunk 用 768 维的 float32 向量表示。当你的文档数量增长时，向量数据的规模是这样的：

```
文档数量与向量数据规模：

  1 万篇文档    → 10 万条向量   → 约 0.3 GB 原始数据  → Chroma 轻松搞定
  10 万篇文档   → 100 万条向量  → 约 3 GB 原始数据    → pgvector 轻松搞定
  100 万篇文档  → 1000 万条向量 → 约 30 GB 原始数据   → pgvector 还能扛
  1000 万篇文档 → 1 亿条向量    → 约 300 GB 原始数据  → pgvector 开始吃力
  1 亿篇文档    → 10 亿条向量   → 约 3 TB 原始数据    → 必须用 Milvus
```

注意这只是原始数据的大小，还没算索引。HNSW 索引的大小大约是原始数据的 2~3 倍——1 亿条 768 维向量的 HNSW 索引大约需要 600~900 GB 内存。没有多少单机服务器能装下这么多内存，而且即使装得下，单机的 CPU 算力也无法在毫秒级完成 1 亿条向量的距离计算。

这就是 Milvus 存在的意义——它把数据分片（Sharding）到多台机器上，每台机器只负责一部分数据，搜索时多台机器并行计算，最后合并结果。这样，无论数据量多大，你都可以通过增加机器来水平扩展。

```
单机 vs 分布式向量搜索：

  单机（pgvector）：
  ┌────────────────────────────────────┐
  │  1 亿条向量 → 1 台服务器            │
  │  内存：900 GB（装不下！）            │
  │  搜索：串行扫描，延迟 500ms+        │
  └────────────────────────────────────┘

  分布式（Milvus）：
  ┌──────────┐ ┌──────────┐ ┌──────────┐
  │  Node 1  │ │  Node 2  │ │  Node 3  │
  │  3300万  │ │  3300万  │ │  3400万  │
  │  300GB   │ │  300GB   │ │  300GB   │
  │  并行搜索 │ │  并行搜索 │ │  并行搜索 │
  └────┬─────┘ └────┬─────┘ └────┬─────┘
       │            │            │
       └────────────┼────────────┘
                    ▼
              合并结果，返回 Top-K
              总延迟：~10ms
```

---

## Milvus 的定位：云原生分布式向量数据库

Milvus 的官方定位是"云原生分布式向量数据库"——这几个词拆开来看：

- **云原生**：Milvus 从设计之初就考虑了容器化部署、微服务架构、存算分离，它不是一个"先做单机再硬改分布式"的系统
- **分布式**：数据可以分片到多个节点，查询可以并行执行，存储和计算都可以独立扩缩容
- **向量数据库**：Milvus 只做一件事——向量数据的存储、索引和检索。它不是通用数据库，不支持 SQL、不支持 JOIN、不支持事务

这个定位决定了 Milvus 的适用场景和边界。它不是 pgvector 的替代品——pgvector 能做的事（SQL 查询、事务一致性、混合查询），Milvus 做不了；Milvus 能做的事（十亿级向量、分布式、量化），pgvector 也做不了。它们是互补关系，不是竞争关系。

---

## Milvus vs pgvector vs Chroma：三者的根本区别

如果你已经学过 Chroma 和 pgvector 教程，下面这个对比能帮你快速理解 Milvus 的独特之处：

| 维度 | Chroma | pgvector | Milvus |
|------|--------|----------|--------|
| 架构 | 嵌入式（进程内） | 单机（PG 扩展） | 分布式（微服务） |
| 最大规模 | ~1000 万向量 | ~1 亿向量（取决于内存） | 10 亿+ 向量 |
| 索引类型 | HNSW（自动） | IVFFlat / HNSW | FLAT / IVFFlat / HNSW / IVF_PQ / SCANN / DiskANN / GPU 索引 |
| 向量量化 | ❌ | ❌ | ✅ PQ / SQ / BQ |
| 分布式 | ❌ | ❌ | ✅ 原生分片 + 多副本 |
| SQL 支持 | ❌ | ✅ 完整 SQL | ❌ |
| 事务 | ❌ | ✅ ACID | ❌（最终一致性） |
| 多向量搜索 | ❌ | ❌ | ✅ Hybrid Search + Reranker |
| 部署复杂度 | 极低（pip install） | 低（Docker） | 中~高（多组件） |
| 学习曲线 | 低 | 低 | 中~高 |

比如，下面的场景对比可以帮助你理解什么时候该选哪个：

```python
# 场景1：快速验证 RAG 原型 → Chroma
# 你只需要几行代码就能跑起来，不需要管部署、索引、调优
import chromadb
client = chromadb.Client()
collection = client.create_collection("docs")
collection.add(documents=["hello world"], ids=["1"])
results = collection.query(query_texts=["hello"], n_results=1)

# 场景2：生产系统需要 SQL 和事务 → pgvector
# 你的业务数据在 PostgreSQL 里，向量搜索和结构化查询要在同一个事务中
import psycopg2
conn = psycopg2.connect(DATABASE_URL)
cur = conn.cursor()
cur.execute("""
    SELECT name, price, embedding <=> %s AS distance
    FROM products
    WHERE category = 'electronics' AND price < 1000
    ORDER BY embedding <=> %s
    LIMIT 5
""", (query_vec, query_vec))

# 场景3：十亿级向量搜索 → Milvus
# 数据量太大，单机扛不住，需要分布式
from pymilvus import MilvusClient
client = MilvusClient(uri="http://milvus:19530")
results = client.search(
    collection_name="products",
    data=[query_vec],
    limit=5,
    output_fields=["name", "price"],
    filter='category == "electronics" and price < 1000'
)
```

---

## Milvus 的核心能力

Milvus 的核心能力可以概括为五个方面：

**1. 分布式存储与计算**：数据自动分片到多个节点，查询并行执行，存储和计算可以独立扩缩容。这意味着你的向量搜索系统可以随着数据量的增长而水平扩展，而不是被单机的内存和 CPU 限制住。

**2. 丰富的索引类型**：从暴力搜索（FLAT）到内存索引（IVFFlat、HNSW）到量化索引（IVF_PQ、IVF_SQ8）到磁盘索引（DiskANN）再到 GPU 索引，Milvus 提供了目前最全面的向量索引选择。不同的索引类型适合不同的数据量、内存预算和精度要求。

**3. 向量量化（Quantization）**：这是 Milvus 相比 pgvector 和 Chroma 最核心的优势。量化技术可以把 float32 向量压缩成 int8 甚至 1 bit，内存占用减少 4~32 倍。在同等内存预算下，Milvus 能存储和搜索的向量数量是 pgvector 的 4~32 倍。

**4. 标量过滤**：Milvus 支持在向量搜索时同时进行标量字段过滤（类似 pgvector 的 WHERE 条件），并且可以为标量字段创建独立的索引来加速过滤。

**5. 多向量搜索**：一个 Collection 可以有多个向量字段（比如文本 embedding 和图像 embedding），搜索时可以同时搜索多个向量字段并用 RRF（Reciprocal Rank Fusion）或加权方式合并结果。这是多模态 RAG 的基础能力。

---

## 谁在用 Milvus

Milvus 在工业界有大量落地案例，了解这些场景有助于你判断自己是否需要 Milvus：

- **NVIDIA**：用 Milvus 构建内部知识库的语义搜索，帮助工程师快速找到技术文档和设计方案
- **Roblox**：游戏平台的推荐系统，用 Milvus 做用户-游戏向量的相似度匹配
- **Shopee**：电商平台的商品搜索和推荐，十亿级商品向量的实时检索
- **京东**：商品去重和相似商品推荐
- **搜狐**：新闻推荐系统的语义匹配

这些场景的共同特点是：数据量大（亿级以上）、延迟要求高（毫秒级）、需要水平扩展。如果你的场景不具备这些特点，pgvector 或 Chroma 可能是更好的选择。

---

## 常见误区：一上来就选 Milvus

很多团队在项目初期就选择了 Milvus，理由是"万一以后数据量很大呢"。但 Milvus 的分布式架构带来了额外的运维复杂度——你需要管理 etcd（元数据存储）、MinIO/S3（对象存储）、Pulsar/Kafka（消息队列）以及 Milvus 自身的多个组件。如果你的数据量只有几十万条，这些额外的运维成本完全是浪费。

正确的做法是**先用最简单的方案（Chroma 或 pgvector）快速上线，等数据量真的接近瓶颈时再迁移到 Milvus**。向量数据库的迁移并不复杂——核心数据就是向量 + 标量字段，从 pgvector 导出再导入 Milvus 也就是几行代码的事。过早引入分布式方案，跟杀鸡用牛刀没什么区别。

---

## 小结

这一节我们聊了 Milvus 的定位——当向量数据量达到亿级、单机方案扛不住时，Milvus 的分布式架构、量化索引和多向量搜索能力就派上用场了。但 Milvus 不是万能的——它没有 SQL、没有事务、运维复杂，小规模场景下反而不如 pgvector 好用。下一节我们要深入 Milvus 的架构设计，理解它为什么这样设计、各个组件怎么协作，这对后续的调优和排障至关重要。
