# 4.1 索引类型全景：从暴力搜索到磁盘索引

> **Milvus 有 8 种向量索引——不是选择困难，而是不同场景的最优解不同**

---

## 这一节在讲什么？

pgvector 只有两种向量索引（IVFFlat 和 HNSW），Chroma 只有一种（自动 HNSW）。Milvus 提供了 8 种向量索引——从暴力搜索到内存索引到量化索引到磁盘索引再到 GPU 索引。这么多选择看起来让人头疼，但每种索引都有它最适合的场景。这一节我们要把 Milvus 的所有索引类型过一遍，帮你建立"什么场景用什么索引"的直觉。

---

## 为什么 Milvus 有这么多索引类型

答案很简单——不同场景的约束条件不同：

- 数据量小时，暴力搜索就够了，不需要索引
- 内存充足时，HNSW 最快、召回率最高
- 内存有限时，需要量化索引（PQ/SQ）压缩向量
- 数据量极大且内存有限时，需要磁盘索引（DiskANN）
- 有 GPU 时，GPU 索引可以提供极致吞吐

没有一种索引能在所有场景下都是最优的——这就是 Milvus 提供多种索引的原因。

---

## 索引类型全景图

| 索引类型 | 内存占用 | 构建速度 | 查询速度 | 召回率 | 适用场景 |
|---------|---------|---------|---------|-------|---------|
| FLAT | 大 | 无需构建 | 慢（暴力） | 100% | < 10 万条，基准测试 |
| IVFFlat | 中 | 快 | 快 | 中~高 | 通用场景 |
| HNSW | 很大 | 慢 | 很快 | 高 | 内存充足、低延迟 |
| IVF_PQ | 小 | 中 | 快 | 中 | 内存有限、大数据量 |
| IVF_SQ8 | 小 | 快 | 快 | 中 | 内存有限、精度可接受 |
| SCANN | 中 | 中 | 快 | 高 | Google 推荐算法 |
| DiskANN | 磁盘为主 | 慢 | 中 | 高 | 数据量极大、内存有限 |
| GPU_IVF_PQ | 小 | 快 | 极快 | 中 | GPU 服务器、极致吞吐 |

---

## FLAT：暴力搜索

FLAT 不是真正的索引——它就是暴力搜索，逐条计算查询向量与所有向量的距离。它的召回率是 100%（因为检查了每一条数据），但速度最慢。

```python
# 创建 FLAT 索引（实际上就是不建索引）
index_params = {
    "index_type": "FLAT",
    "metric_type": "COSINE",
    "params": {}
}
client.create_index(
    collection_name="documents",
    field_name="embedding",
    index_params=index_params
)
```

什么时候用 FLAT：数据量小于 10 万条时，暴力搜索的速度已经足够快（通常在 10ms 以内），不需要更复杂的索引。FLAT 也常用于基准测试——用 FLAT 的搜索结果作为"标准答案"，来评估其他索引的召回率。

---

## IVFFlat：倒排文件索引

IVFFlat 的原理我们在 pgvector 教程里已经详细讲过——它把向量空间划分为 K 个聚类区域，搜索时只扫描查询向量附近的几个区域。Milvus 的 IVFFlat 跟 pgvector 的 IVFFlat 原理相同，但参数名不同：

```python
# 创建 IVFFlat 索引
index_params = {
    "index_type": "IVFFlat",
    "metric_type": "COSINE",
    "params": {"nlist": 1024}   # 聚类区域数量，类似 pgvector 的 lists
}
client.create_index(
    collection_name="documents",
    field_name="embedding",
    index_params=index_params
)

# 搜索时指定 nprobe
results = client.search(
    collection_name="documents",
    data=[[0.1] * 768],
    limit=5,
    search_params={
        "metric_type": "COSINE",
        "params": {"nprobe": 32}  # 搜索时扫描的区域数，类似 pgvector 的 probes
    }
)
```

IVFFlat 的局限跟 pgvector 一样——需要先有数据再建索引（聚类中心依赖数据分布），增量数据量大时需要重建索引。

---

## HNSW：分层导航小世界图

HNSW 是目前最流行的向量索引——多层图结构，O(log N) 搜索复杂度，全内存驻留。Milvus 的 HNSW 跟 pgvector 的 HNSW 原理相同，参数名略有差异：

```python
# 创建 HNSW 索引
index_params = {
    "index_type": "HNSW",
    "metric_type": "COSINE",
    "params": {
        "M": 16,              # 每层最大连接数（pgvector 叫 m）
        "efConstruction": 256  # 建索引时的搜索宽度（pgvector 叫 ef_construction）
    }
}
client.create_index(
    collection_name="documents",
    field_name="embedding",
    index_params=index_params
)

# 搜索时指定 ef
results = client.search(
    collection_name="documents",
    data=[[0.1] * 768],
    limit=5,
    search_params={
        "metric_type": "COSINE",
        "params": {"ef": 100}  # 搜索宽度（pgvector 叫 ef_search）
    }
)
```

HNSW 的内存估算公式：每条向量约 `dim × 4 + M × 2 × 8` 字节。768 维、M=16 的 HNSW 索引，每条向量约 3072 + 256 = 3328 字节，100 万条约 3.2 GB。

---

## IVF_PQ / IVF_SQ8：量化索引

量化索引是 Milvus 相比 pgvector 和 Chroma 最核心的优势——它们通过压缩向量来大幅减少内存占用。下一节我们会详细讲量化原理，这里先看基本用法：

```python
# IVF_PQ 索引——乘积量化，压缩比最高
index_params = {
    "index_type": "IVF_PQ",
    "metric_type": "COSINE",
    "params": {
        "nlist": 1024,
        "m": 48,           # PQ 的子空间数量（768 / 48 = 16 维/子空间）
        "nbits": 8         # 每个子空间的编码位数（默认 8）
    }
}

# IVF_SQ8 索引——标量量化，压缩比 4 倍
index_params = {
    "index_type": "IVF_SQ8",
    "metric_type": "COSINE",
    "params": {"nlist": 1024}
}
```

---

## SCANN：Google 的 ANN 算法

SCANN（Scalable Nearest Neighbors）是 Google Research 提出的 ANN 算法，在多个基准测试中表现优异。Milvus 2.4+ 支持 SCANN 索引：

```python
index_params = {
    "index_type": "SCANN",
    "metric_type": "COSINE",
    "params": {
        "nlist": 1024,
        "with_raw_data": True   # 是否保留原始向量（用于重排序）
    }
}
```

SCANN 的搜索参数跟 IVFFlat 类似，也是 `nprobe`。SCANN 的优势是在高召回率场景下比 IVFFlat 更快——但构建速度稍慢。

---

## DiskANN：磁盘索引

DiskANN 是微软提出的磁盘向量索引——它把大部分数据放在磁盘上，只把少量元数据放在内存中，适合数据量极大但内存有限的场景：

```python
index_params = {
    "index_type": "DISKANN",
    "metric_type": "COSINE",
    "params": {}   # DiskANN 不需要额外参数
}
```

DiskANN 的搜索参数是 `search_list`——控制搜索时的候选列表大小：

```python
results = client.search(
    collection_name="documents",
    data=[[0.1] * 768],
    limit=5,
    search_params={
        "metric_type": "COSINE",
        "params": {"search_list": 100}
    }
)
```

DiskANN 的优势是内存占用极低——100 万条 768 维向量，DiskANN 只需要约 1 GB 内存（而 HNSW 需要约 3.5 GB）。代价是搜索延迟比 HNSW 高（因为需要读磁盘），但比暴力搜索快得多。

---

## GPU_IVF_PQ：GPU 加速索引

如果你有 GPU 服务器，Milvus 支持 GPU 加速的 IVF_PQ 索引——利用 GPU 的大规模并行计算能力，搜索吞吐量可以提升 5~10 倍：

```python
index_params = {
    "index_type": "GPU_IVF_PQ",
    "metric_type": "COSINE",
    "params": {
        "nlist": 1024,
        "m": 48,
        "nbits": 8
    }
}
```

GPU 索引需要 Milvus 启动时配置 GPU 资源，不是所有部署环境都支持。

---

## 索引创建流程

无论使用哪种索引，创建流程都是一样的：

```python
# 第1步：创建索引（IndexNode 在后台构建）
client.create_index(
    collection_name="documents",
    field_name="embedding",
    index_params={
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {"M": 16, "efConstruction": 256}
    }
)

# 第2步：加载索引到 QueryNode 内存（必须！）
client.load_collection("documents")

# 第3步：搜索
results = client.search(
    collection_name="documents",
    data=[[0.1] * 768],
    limit=5,
    search_params={
        "metric_type": "COSINE",
        "params": {"ef": 100}
    }
)
```

### 常见误区：建了索引但忘记 load

这是 Milvus 初学者最常犯的错误——`create_index()` 只是把索引文件构建到对象存储中，并没有加载到 QueryNode 内存。你必须调用 `load_collection()` 或 `load_partitions()` 把索引加载到内存才能搜索。如果忘记 load，搜索会报错或退化为暴力搜索。

---

## 小结

这一节我们概览了 Milvus 的 8 种向量索引：FLAT（暴力搜索）、IVFFlat（通用）、HNSW（最快最高召回率）、IVF_PQ/IVF_SQ8（量化压缩）、SCANN（Google 算法）、DiskANN（磁盘索引）、GPU_IVF_PQ（GPU 加速）。每种索引都有最适合的场景——没有银弹，只有最适合当前约束条件的选择。下一节我们深入 HNSW 索引的参数调优。
