# 6.2 性能基准与调优

> **不测量就不优化——理解 Chroma 的性能特征，才能在正确的方向上做正确的优化**

---

## 这一节在讲什么？

当你的 RAG 系统从原型走向生产，性能就变成了不可回避的问题——查询延迟能不能满足 SLA？QPS 能不能扛住峰值流量？数据量增长后性能会不会断崖式下降？这些问题都需要用数据来回答，而不是凭感觉猜测。这一节我们要建立 Chroma 的性能基准模型，讲清楚影响性能的关键因素、如何正确地测量性能、以及针对每个因素的具体调优手段。

---

## 关键性能指标

在讨论性能之前，我们需要先定义"性能"到底指什么。对于向量数据库来说，有三个核心指标：

| 指标 | 含义 | 典型目标 |
|------|------|---------|
| **QPS** | 每秒能处理的查询数 | 单机 500~2000 QPS |
| **P99 延迟** | 99% 的查询在多少时间内完成 | < 100ms（< 10K 向量）/ < 500ms（~100K 向量） |
| **Recall@K** | Top-K 结果中真正最相似的占比 | > 90%（Top-5） |

这三个指标之间存在固有的权衡——提高 QPS 可能牺牲 Recall（比如减少 HNSW 的搜索宽度），降低延迟可能牺牲 QPS（比如减少并发数）。理解这些权衡，才能做出正确的调优决策。

---

## 影响性能的六大因素

### 因素一：向量维度（d）

向量维度直接影响每次距离计算的复杂度——计算两个 d 维向量的距离需要 O(d) 次乘加运算。维度从 384 增加到 1536，单次距离计算的时间增加 4 倍。

```
单次距离计算时间 ≈ d × 2ns（CPU 浮点运算）
  384维: ~0.8μs
  768维: ~1.5μs
  1536维: ~3.0μs
  3072维: ~6.0μs

HNSW 搜索总距离计算次数 ≈ ef_search × M（M=16, ef_search=10~100）
  典型查询: 160~1600 次距离计算

总查询延迟 ≈ 距离计算次数 × 单次计算时间
  384维, ef=50: 800 × 0.8μs ≈ 0.64ms
  1536维, ef=50: 800 × 3.0μs ≈ 2.4ms
```

**调优方向**：如果维度是瓶颈，考虑使用更小的 embedding 模型（如 `all-MiniLM-L6-v2` 的 384 维而非 `text-embedding-3-large` 的 3072 维），或者对向量做降维（PCA/Matryoshka Representation）。

### 因素二：数据量（N）

数据量影响 HNSW 索引的大小和搜索深度。HNSW 的搜索复杂度是 O(log N)，所以数据量增加 10 倍，查询延迟只增加约 2 倍——这是 HNSW 相比暴力搜索（O(N)）的核心优势。

```python
import chromadb
import numpy as np
import time

def benchmark_data_scale(scales=[1000, 5000, 10000, 50000], dim=384):
    """测试不同数据量下的查询性能"""
    for n in scales:
        client = chromadb.Client()
        col = client.create_collection(name=f"scale_{n}")

        vectors = np.random.randn(n, dim).tolist()
        docs = [f"doc_{i}" for i in range(n)]
        ids = [f"id_{i}" for i in range(n)]

        batch = 5000
        for i in range(0, n, batch):
            col.add(
                documents=docs[i:i+batch],
                ids=ids[i:i+batch],
                embeddings=vectors[i:i+batch]
            )

        # 预热
        query_vec = np.random.randn(dim).tolist()
        col.query(query_embeddings=[query_vec], n_results=10)

        # 测量
        n_queries = 100
        start = time.time()
        for _ in range(n_queries):
            qv = np.random.randn(dim).tolist()
            col.query(query_embeddings=[qv], n_results=10)
        elapsed = time.time() - start

        qps = n_queries / elapsed
        avg_ms = (elapsed / n_queries) * 1000
        print(f"N={n:>6}: {avg_ms:.1f}ms/query, {qps:.0f} QPS")

        client.delete_collection(f"scale_{n}")

# benchmark_data_scale()
# 典型输出：
# N=  1000: 3.2ms/query, 312 QPS
# N=  5000: 5.1ms/query, 196 QPS
# N= 10000: 7.8ms/query, 128 QPS
# N= 50000: 18.3ms/query, 55 QPS
```

**调优方向**：数据量过大时，考虑分片（多个 Collection 按业务隔离）或定期清理过期数据。

### 因素三：n_results（K）

`n_results` 影响 HNSW 搜索需要返回的候选数量。K 越大，搜索范围越广，延迟越高。但 K 的影响是亚线性的——从 K=5 增加到 K=50，延迟通常只增加 2~3 倍。

**调优方向**：RAG 场景建议 K=3~5，Re-ranking 场景建议 K=20~50。不要设得过大。

### 因素四：Metadata 过滤

where 过滤在向量搜索之前执行，它的性能取决于候选集缩减的效果。如果 where 条件能将候选集从 100K 缩减到 1K，后续的向量搜索会快很多；但如果条件过于宽松（比如只过滤掉了 10% 的数据），过滤本身的开销可能抵消搜索加速。

**调优方向**：确保 where 条件有足够的选择性（至少缩减 50% 的候选集），避免对高基数字段做过滤。

### 因素五：持久化 I/O

持久化模式下的查询需要从磁盘加载数据。如果数据不在操作系统缓存中（冷查询），磁盘 I/O 可能成为瓶颈——特别是使用 HDD 而非 SSD 时。

**调优方向**：使用 SSD、增大操作系统的 page cache、对热数据做预加载。

### 因素六：Embedding 计算

如果使用 `query_texts` 而非 `query_embeddings`，每次查询都需要调用 Embedding Function 计算向量。本地模型的推理时间约 5~15ms（CPU），远程 API 约 50~200ms（含网络延迟）。

**调优方向**：使用本地模型 + GPU 加速，或者缓存常用查询的 embedding。

---

## 性能调优清单

按优先级排序——先做收益最大的优化：

```
┌─────────────────────────────────────────────────────────────────┐
│  Chroma 性能调优清单（按优先级排序）                              │
│                                                                 │
│  Priority 1（基础优化，收益最大）:                                │
│  ☐ 使用 SSD 存储 persist_directory                              │
│  ☐ 选择合适的 embedding 维度（384~768 足够大多数场景）            │
│  ☐ n_results 设为 3~5（RAG 场景）                                │
│  ☐ 使用本地 embedding 模型（避免远程 API 延迟）                   │
│                                                                 │
│  Priority 2（进阶优化，中等收益）:                                │
│  ☐ 用 where 过滤缩小候选集（选择性 > 50%）                       │
│  ☐ 分批 add（每批 1000~5000 条）                                 │
│  ☐ 定期清理过期数据                                              │
│  ☐ 预热热数据（启动时执行几次查询）                               │
│                                                                 │
│  Priority 3（高级优化，场景特定）:                                │
│  ☐ 使用 GPU 加速 embedding 推理                                  │
│  ☐ 缓存常用查询的 embedding                                      │
│  ☐ 使用 Server 模式 + 连接池                                     │
│  ☐ 调整 HNSW 参数（ef_construction, M）                          │
│  ☐ 向量降维（PCA / Matryoshka）                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 容量规划：磁盘空间估算

在部署 Chroma 之前，你需要估算数据量对应的磁盘空间需求：

```
单条向量存储 = dim × 4 bytes (float32)
单条文档存储 ≈ 平均文本长度 × 1 byte (UTF-8)
单条 metadata ≈ 100~500 bytes (JSON)

HNSW 索引开销 ≈ 向量数据 × 1.5~2.0

总磁盘空间 ≈ (向量数据 + 文档存储 + metadata) × 2.0

示例估算（10万条 384 维向量，平均文本 200 字符）：
  向量: 100,000 × 384 × 4 = 146 MB
  文档: 100,000 × 200 = 20 MB
  Metadata: 100,000 × 300 = 30 MB
  HNSW 索引: 146 × 1.5 = 219 MB
  总计: (146 + 20 + 30 + 219) ≈ 415 MB

示例估算（100万条 768 维向量，平均文本 500 字符）：
  向量: 1,000,000 × 768 × 4 = 2.88 GB
  文档: 1,000,000 × 500 = 500 MB
  Metadata: 1,000,000 × 300 = 300 MB
  HNSW 索引: 2.88 × 1.5 = 4.32 GB
  总计: (2.88 + 0.5 + 0.3 + 4.32) ≈ 8.0 GB
```

---

## 常见误区

### 误区 1：数据量翻倍，延迟也翻倍

HNSW 的搜索复杂度是 O(log N)，不是 O(N)。数据量从 10K 增加到 100K（10 倍），延迟通常只增加 2~3 倍。这是 HNSW 相比暴力搜索的核心优势。

### 误区 2：embedding 维度越高检索越准

维度高确实能编码更多信息，但超过一定阈值后收益递减。对于大多数 RAG 场景，384~768 维已经足够。从 384 维升级到 1536 维，存储和查询开销增加 4 倍，但检索质量可能只提升 5~10%。

### 误区 3：不测量就优化

性能优化必须基于测量数据。在优化之前，先用 benchmark 确定瓶颈在哪里——是 embedding 计算、HNSW 搜索、还是 metadata 过滤？针对瓶颈优化才能事半功倍。

---

## 本章小结

性能优化是 Chroma 从原型走向生产的关键环节。核心要点回顾：第一，三个核心性能指标是 QPS、P99 延迟和 Recall@K，它们之间存在固有权衡；第二，影响性能的六大因素按重要性排序：向量维度 > 数据量 > n_results > metadata 过滤 > 持久化 I/O > embedding 计算；第三，HNSW 的搜索复杂度是 O(log N)，数据量翻 10 倍延迟只增加 2~3 倍；第四，优先做收益最大的基础优化（SSD、合适的维度、小的 n_results、本地 embedding），再做进阶优化；第五，容量规划的关键公式是 `(向量 + 文档 + metadata) × 2.0`。

下一节我们将讲多实例与并发访问——Chroma 的线程安全性、Server 模式的并发能力、以及 Docker Compose 部署方案。
