# 4.2 HNSW 索引详解

> **HNSW 是 Milvus 中最常用的索引——理解它的参数是调优的关键**

---

## 这一节在讲什么？

HNSW（Hierarchical Navigable Small World）是 Milvus 中最常用的向量索引，也是性能最好的内存索引——它的查询延迟通常在 1~10 毫秒，召回率可以达到 95% 以上。如果你已经学过 pgvector 教程，HNSW 的原理你已经了解了——多层图结构、从顶层向下导航、O(log N) 搜索复杂度。这一节我们不讲原理（原理跟 pgvector 完全一样），而是聚焦于 Milvus HNSW 的参数调优、与 pgvector HNSW 的差异、以及内存估算。

---

## Milvus HNSW 参数详解

### M：每层最大连接数

`M` 控制的是 HNSW 图中每个节点的最大连接数——M 越大，图的连接越密集，搜索时能探索更多路径，召回率越高，但内存占用也越大。

```
M 对索引的影响：

  M = 8：  图稀疏，内存小，召回率低
  M = 16： 平衡点（默认值），适合大多数场景
  M = 32： 图密集，内存大，召回率高
  M = 64： 极致召回率，内存非常大
```

M 的选择建议：

| 数据量 | 推荐 M | 理由 |
|-------|--------|------|
| < 100 万 | 16 | 默认值，够用 |
| 100 万~1000 万 | 16~32 | 数据量大时需要更密集的图 |
| > 1000 万 | 32~64 | 大数据量需要更高的图连通性 |

### efConstruction：建索引时的搜索宽度

`efConstruction` 控制的是建索引时搜索最近邻的宽度——efConstruction 越大，建索引时搜索越充分，图的连接质量越高，但构建时间越长。

```
efConstruction 对索引质量的影响：

  efConstruction = 64：  构建快，图质量一般
  efConstruction = 128： 构建适中，图质量好
  efConstruction = 256： 构建慢，图质量很好（Milvus 默认值）
  efConstruction = 512： 构建很慢，图质量极好
```

Milvus 的默认值 256 比 pgvector 的默认值 64 更激进——这意味着 Milvus 的 HNSW 索引默认质量更高，但构建时间也更长。对于生产环境，256 是一个好的起点；如果你需要快速构建索引（比如频繁重建），可以降到 128。

### ef：搜索时的搜索宽度

`ef` 控制的是搜索时探索的图节点数量——ef 越大，搜索越充分，召回率越高，但速度越慢。这是搜索时唯一需要调整的参数。

```
ef 对搜索性能的影响（100 万条 768 维向量，M=16）：

  ef = 40：   ~2ms，召回率 ~88%
  ef = 64：   ~3ms，召回率 ~92%（Milvus 默认值）
  ef = 100：  ~4ms，召回率 ~95%
  ef = 200：  ~6ms，召回率 ~98%
  ef = 500：  ~12ms，召回率 ~99.5%
```

---

## Milvus HNSW vs pgvector HNSW

| 维度 | Milvus | pgvector |
|------|--------|----------|
| 构建参数 M | `M`（默认 16） | `m`（默认 16） |
| 构建参数 efConstruction | `efConstruction`（默认 256） | `ef_construction`（默认 64） |
| 搜索参数 ef | `ef`（默认 64） | `ef_search`（默认 40） |
| 搜索参数设置方式 | `search(params={"ef": 100})` | `SET hnsw.ef_search = 100` |
| 增量更新 | ✅ 支持 | ✅ 支持 |
| 分布式 | ✅ 多 QueryNode 并行 | ❌ 单机 |

参数名虽然不同，但含义完全一样。Milvus 的默认值更激进（efConstruction=256 vs 64，ef=64 vs 40），这意味着 Milvus 的 HNSW 索引默认质量更高，但构建和搜索也稍慢。

---

## HNSW 的内存估算

HNSW 索引全内存驻留——你必须确保 QueryNode 的内存能装下整个索引。内存估算公式：

```
每条向量的索引内存 ≈ dim × 4 + M × 2 × 8 字节

768 维、M=16：
  每条向量 ≈ 3072 + 256 = 3328 字节
  100 万条 ≈ 3.2 GB
  1000 万条 ≈ 32 GB
  1 亿条 ≈ 320 GB

768 维、M=32：
  每条向量 ≈ 3072 + 512 = 3584 字节
  100 万条 ≈ 3.4 GB
  1000 万条 ≈ 34 GB
  1 亿条 ≈ 340 GB
```

加上原始向量数据和 QueryNode 的运行时开销，建议 QueryNode 的内存至少是索引大小的 1.5 倍。

比如，下面的代码展示了如何根据数据量估算 HNSW 索引的内存需求，由于 HNSW 图结构需要额外存储连接信息，所以实际内存比原始向量数据大：

```python
def estimate_hnsw_memory(num_vectors, dim, M=16):
    """估算 HNSW 索引的内存需求"""
    vector_size = dim * 4          # 原始向量：dim × 4 字节（float32）
    graph_size = M * 2 * 8         # 图连接：M × 2 × 8 字节（每层 M 个连接，2 层，每个连接 8 字节）
    per_vector = vector_size + graph_size
    total_bytes = per_vector * num_vectors
    total_gb = total_bytes / (1024 ** 3)

    print(f"HNSW 索引内存估算：")
    print(f"  向量数量：{num_vectors:,}")
    print(f"  维度：{dim}")
    print(f"  M：{M}")
    print(f"  每条向量：{per_vector:,} 字节")
    print(f"  总内存：{total_gb:.1f} GB")
    print(f"  建议 QueryNode 内存：{total_gb * 1.5:.1f} GB（索引 × 1.5）")

estimate_hnsw_memory(10_000_000, 768, M=16)
# 输出：
# HNSW 索引内存估算：
#   向量数量：10,000,000
#   维度：768
#   M：16
#   每条向量：3,328 字节
#   总内存：31.0 GB
#   建议 QueryNode 内存：46.5 GB（索引 × 1.5）
```

---

## HNSW 索引的创建与调优实战

```python
from pymilvus import MilvusClient

client = MilvusClient(uri="http://localhost:19530")

# 创建 HNSW 索引
client.create_index(
    collection_name="documents",
    field_name="embedding",
    index_params={
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {"M": 16, "efConstruction": 256}
    }
)

# 加载索引
client.load_collection("documents")

# 搜索——调整 ef 参数
# 快速搜索（精度稍低）
results_fast = client.search(
    collection_name="documents",
    data=[[0.1] * 768],
    limit=5,
    search_params={"metric_type": "COSINE", "params": {"ef": 40}}
)

# 精确搜索（速度稍慢）
results_precise = client.search(
    collection_name="documents",
    data=[[0.1] * 768],
    limit=5,
    search_params={"metric_type": "COSINE", "params": {"ef": 200}}
)
```

### 调优方法论

1. **先用默认参数建立基线**：M=16, efConstruction=256, ef=64
2. **测量搜索延迟和召回率**：用 FLAT 索引的结果作为"标准答案"，计算 HNSW 的 Recall@K
3. **调整 ef**：如果召回率不够，增大 ef；如果延迟太高，减小 ef
4. **调整 M 和 efConstruction**：只有在调整 ef 无法满足要求时才调整构建参数——因为修改构建参数需要重建索引

---

## 常见误区：M 和 ef 设得越大越好

M 和 ef 的增大都有代价——M 增大会线性增加内存占用和构建时间，ef 增大会线性增加搜索延迟。在实际应用中，M=16~32、ef=64~200 已经能覆盖绝大多数场景的需求。把 M 设成 128 或 ef 设成 1000，只是浪费资源而不会带来有意义的召回率提升。

---

## 小结

这一节我们深入了 Milvus HNSW 索引的参数：M 控制图的连接密度（默认 16），efConstruction 控制构建质量（默认 256），ef 控制搜索精度（默认 64）。Milvus 的默认值比 pgvector 更激进，索引质量更高但构建和搜索也稍慢。HNSW 全内存驻留，内存估算公式是 `dim × 4 + M × 2 × 8` 字节/条。下一节我们聊 Milvus 最独特的索引类型——量化索引（PQ/SQ/BQ）。
