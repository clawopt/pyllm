# 4.3 量化索引：PQ / SQ / BQ——用精度换内存

> **pgvector 不支持量化——这就是 Milvus 在大规模场景下的核心优势**

---

## 这一节在讲什么？

如果你有 10 亿条 768 维向量，HNSW 索引需要约 320 GB 内存——这已经是一台高端服务器的全部内存了。如果数据量继续增长呢？Milvus 的答案是量化索引——把 float32 向量压缩成更小的表示，内存减少 4~32 倍，代价是精度略有下降。这是 pgvector 和 Chroma 都不具备的能力，也是 Milvus 在大规模场景下的核心优势。这一节我们要深入理解三种量化方法的原理、选择依据和性能特征。

---

## 为什么需要量化

让我们先用数字建立直觉：

```
10 亿条 768 维向量的存储需求：

  原始数据（float32）：10亿 × 768 × 4字节 ≈ 2.9 TB
  HNSW 索引（M=16）：  约 3.2 TB
  总内存需求：          约 6.1 TB

  → 没有多少服务器能装下 6.1 TB 内存！

  用 PQ 量化后（m=48, nbits=8）：
  原始数据：10亿 × 48 × 1字节 ≈ 48 GB
  IVF_PQ 索引：         约 100 GB
  总内存需求：           约 150 GB

  → 一台 256 GB 内存的服务器就能搞定！
```

量化把内存需求从 6.1 TB 降到了 150 GB——压缩了 40 倍。代价是召回率从 99% 降到了 90%~95%，但在大多数 RAG 场景中，95% 的召回率已经完全够用了。

---

## PQ（Product Quantization）：乘积量化

PQ 是最常用的量化方法——它把高维向量切成多个低维子空间，每个子空间独立量化。

### PQ 原理

```
PQ 量化过程（768 维向量，m=48 个子空间）：

  原始向量（768 维 float32，3072 字节）：
  [0.12, -0.34, 0.56, ..., 0.78]
  ┌───────────────────────────────────────────────┐
  │ 子空间1 │ 子空间2 │ ... │ 子空间48 │
  │ 16维    │ 16维    │     │ 16维     │
  └───────────────────────────────────────────────┘

  每个子空间用 K-Means 聚类出 256 个中心点：
  子空间1：中心点 [C1, C2, ..., C256]
  子空间2：中心点 [C1, C2, ..., C256]
  ...
  子空间48：中心点 [C1, C2, ..., C256]

  量化后（48 字节）：
  [中心点ID_1, 中心点ID_2, ..., 中心点ID_48]
  每个ID 用 1 字节（0~255）

  压缩比：3072 / 48 = 64 倍！
```

PQ 的距离计算不是逐维计算，而是用查表法——预先计算查询向量与每个子空间所有中心点的距离，然后查表求和。这比逐维计算快得多，而且内存访问模式更友好。

```python
# 创建 IVF_PQ 索引
index_params = {
    "index_type": "IVF_PQ",
    "metric_type": "COSINE",
    "params": {
        "nlist": 1024,     # IVF 的聚类区域数
        "m": 48,           # PQ 的子空间数量（768 / 48 = 16 维/子空间）
        "nbits": 8         # 每个子空间的编码位数（默认 8，即 256 个中心点）
    }
}
client.create_index(
    collection_name="documents",
    field_name="embedding",
    index_params=index_params
)

# 搜索
results = client.search(
    collection_name="documents",
    data=[[0.1] * 768],
    limit=5,
    search_params={
        "metric_type": "COSINE",
        "params": {"nprobe": 32}
    }
)
```

### PQ 的参数选择

- **m**：子空间数量。m 越大，量化越精细，但内存占用也越大。m 必须能整除向量维度。推荐值：`dim / 16`（768 维 → m=48，1536 维 → m=96）
- **nbits**：每个子空间的编码位数。默认 8（256 个中心点），目前 Milvus 只支持 8

---

## SQ8（Scalar Quantization）：标量量化

SQ8 比 PQ 更简单——它把每个 float32 维度直接压缩成 int8（1 字节），压缩比 4 倍。

```
SQ8 量化过程：

  原始向量（float32）：
  [0.12, -0.34, 0.56, ..., 0.78]  → 768 × 4 = 3072 字节

  量化后（int8）：
  [31, -87, 143, ..., 200]         → 768 × 1 = 768 字节

  压缩比：4 倍
```

SQ8 的实现方式是：找到向量每个维度的最小值和最大值，然后把 float32 线性映射到 int8 的范围（-128~127）。精度损失比 PQ 小，但压缩比也小得多。

```python
# 创建 IVF_SQ8 索引
index_params = {
    "index_type": "IVF_SQ8",
    "metric_type": "COSINE",
    "params": {"nlist": 1024}
}
client.create_index(
    collection_name="documents",
    field_name="embedding",
    index_params=index_params
)
```

---

## BQ（Binary Quantization）：二值量化

BQ 是最激进的量化——把每个 float32 维度压缩成 1 bit（0 或 1），压缩比 32 倍。

```
BQ 量化过程：

  原始向量（float32）：
  [0.12, -0.34, 0.56, ..., 0.78]  → 768 × 4 = 3072 字节

  量化后（1 bit/维度）：
  [1, 0, 1, ..., 1]                → 768 / 8 = 96 字节

  压缩比：32 倍！
```

BQ 的精度损失最大——它只保留了每个维度的正负号，丢失了所有幅度信息。适合超大规模初筛：先用 BQ 快速召回候选集，再用高精度索引或原始向量重排。

```python
# 创建带 BQ 的索引（Milvus 2.5+）
index_params = {
    "index_type": "IVF_PQ",
    "metric_type": "COSINE",
    "params": {
        "nlist": 1024,
        "m": 48,
        "nbits": 8,
        "is_pq_refine": True   # 启用 PQ 重排序
    }
}
```

---

## 三种量化方法的对比

| 维度 | PQ | SQ8 | BQ |
|------|-----|-----|-----|
| 压缩比 | 8~64 倍 | 4 倍 | 32 倍 |
| 精度损失 | 中 | 小 | 大 |
| 距离计算速度 | 快（查表法） | 中 | 极快（位运算） |
| 参数复杂度 | 高（需要选 m） | 低 | 低 |
| 推荐场景 | 内存有限、大数据量 | 内存稍紧、精度优先 | 超大规模初筛 |

比如，下面的程序展示了不同量化索引的内存对比，由于 PQ 用查表法计算距离，所以即使压缩了 64 倍，搜索速度依然很快：

```python
def estimate_quantized_memory(num_vectors, dim, method="PQ", m=48):
    """估算量化索引的内存需求"""
    if method == "PQ":
        per_vector = m * 1         # PQ：m 个子空间，每个 1 字节
    elif method == "SQ8":
        per_vector = dim * 1       # SQ8：每个维度 1 字节
    elif method == "BQ":
        per_vector = dim / 8       # BQ：每个维度 1 bit
    else:
        per_vector = dim * 4       # 原始 float32

    total_bytes = per_vector * num_vectors
    total_gb = total_bytes / (1024 ** 3)

    print(f"{method} 量化内存估算（{num_vectors/1e6:.0f}M × {dim}d）：")
    print(f"  每条向量：{per_vector:.0f} 字节")
    print(f"  总内存：{total_gb:.1f} GB")
    print()

estimate_quantized_memory(100_000_000, 768, "原始float32")
estimate_quantized_memory(100_000_000, 768, "PQ", m=48)
estimate_quantized_memory(100_000_000, 768, "SQ8")
estimate_quantized_memory(100_000_000, 768, "BQ")
```

---

## 常见误区：量化索引的召回率一定很低

量化索引的召回率取决于量化参数和搜索参数的设置。PQ 在 m=48、nprobe=32 时，召回率通常可以达到 90%~95%——对于 RAG 场景来说完全够用。如果你需要更高的召回率，可以使用"量化索引初筛 + 原始向量重排"的策略——先用量化索引快速召回 Top-100 候选，再用原始向量精确计算距离，重排得到 Top-5 结果。这样既节省了内存，又保证了召回率。

---

## 小结

这一节我们深入了 Milvus 的量化索引：PQ（乘积量化，压缩比 8~64 倍）、SQ8（标量量化，压缩比 4 倍）、BQ（二值量化，压缩比 32 倍）。量化是 Milvus 相比 pgvector/Chroma 最核心的优势——它让同等内存下能存储和搜索的向量数量提升数倍。代价是精度略有下降，但 90%~95% 的召回率在大多数 RAG 场景中已经足够。下一节我们聊索引选型和参数调优的实战方法论。
