# 4.4 索引选型与参数调优实战

> **选对索引只是第一步——调对参数才是让 Milvus 真正飞起来的关键**

---

## 这一节在讲什么？

前面三节我们学了 Milvus 的 8 种索引类型和 HNSW、量化索引的参数。但"知道参数是什么"和"知道怎么选、怎么调"是两回事。这一节我们要建立一套完整的索引选型和调优方法论——什么时候建索引、选哪种索引、参数怎么设、如何评估召回率、以及标量索引的配置。

---

## 索引选型决策树

```
你的数据量有多大？
│
├─ < 10 万条
│   → FLAT（暴力搜索足够快，不需要索引）
│
├─ 10 万 ~ 1000 万
│   │
│   ├─ 内存充足（> 索引大小 × 2）？
│   │   ├─ 是 → HNSW（M=16, efConstruction=256）
│   │   └─ 否 → IVF_SQ8 或 IVF_PQ
│   │
│   └─ 需要极致低延迟？
│       ├─ 是 → HNSW（ef=100~200）
│       └─ 否 → IVFFlat 或 SCANN
│
├─ 1000 万 ~ 1 亿
│   │
│   ├─ 内存充足（> 100GB）？
│   │   ├─ 是 → HNSW 或 SCANN
│   │   └─ 否 → IVF_PQ（m=dim/16）或 DiskANN
│   │
│   └─ 需要高召回率？
│       ├─ 是 → SCANN（with_raw_data=True）+ 重排序
│       └─ 否 → IVF_PQ
│
└─ > 1 亿
    │
    ├─ 内存充足（> 500GB）？
    │   ├─ 是 → HNSW + 分片
    │   └─ 否 → DiskANN 或 IVF_PQ + 分片
    │
    └─ 有 GPU？
        ├─ 是 → GPU_IVF_PQ
        └─ 否 → DiskANN
```

---

## 索引参数调优方法论

调优的核心原则是：**先调搜索参数，再调构建参数**。因为修改搜索参数不需要重建索引，而修改构建参数需要。

### 第1步：用默认参数建立基线

```python
# 用默认参数创建 HNSW 索引
client.create_index(
    collection_name="documents",
    field_name="embedding",
    index_params={
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {"M": 16, "efConstruction": 256}
    }
)
client.load_collection("documents")

# 用默认 ef 搜索
results = client.search(
    collection_name="documents",
    data=query_vectors,
    limit=10,
    search_params={"metric_type": "COSINE", "params": {"ef": 64}}
)
```

### 第2步：测量召回率

召回率（Recall@K）是评估索引质量的核心指标——它衡量的是"ANN 索引返回的 Top-K 结果中有多少跟暴力搜索的结果一致"。

```python
def measure_recall(client, collection_name, query_vectors, ground_truth_ids, k=10):
    """测量 HNSW 索引的召回率"""
    # 用 FLAT 索引的结果作为 ground truth
    # （假设你已经用 FLAT 索引搜索得到了 ground_truth_ids）

    # 用 HNSW 索引搜索
    results = client.search(
        collection_name=collection_name,
        data=query_vectors,
        limit=k,
        search_params={"metric_type": "COSINE", "params": {"ef": 64}},
        output_fields=[]
    )

    # 计算召回率
    total_recall = 0
    for i, hits in enumerate(results):
        ann_ids = set(hit['id'] for hit in hits)
        gt_ids = set(ground_truth_ids[i])
        recall = len(ann_ids & gt_ids) / k
        total_recall += recall

    avg_recall = total_recall / len(query_vectors)
    print(f"Recall@{k}: {avg_recall:.4f} ({avg_recall*100:.1f}%)")
    return avg_recall
```

### 第3步：调整搜索参数

如果召回率不够，先调整搜索参数（不需要重建索引）：

```python
# 逐步增大 ef，观察召回率和延迟的变化
for ef in [40, 64, 100, 200, 500]:
    results = client.search(
        collection_name="documents",
        data=query_vectors,
        limit=10,
        search_params={"metric_type": "COSINE", "params": {"ef": ef}}
    )
    recall = measure_recall(client, "documents", query_vectors, ground_truth_ids)
    print(f"ef={ef}: Recall={recall:.4f}")
```

### 第4步：调整构建参数（必要时）

只有当搜索参数调到极限仍然无法满足召回率要求时，才需要调整构建参数并重建索引：

```python
# 重建索引——增大 M 和 efConstruction
client.release_collection("documents")
client.drop_index(collection_name="documents", index_name="embedding")

client.create_index(
    collection_name="documents",
    field_name="embedding",
    index_params={
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {"M": 32, "efConstruction": 512}  # 更大的参数
    }
)
client.load_collection("documents")
```

---

## 标量索引：为过滤加速

向量索引加速距离计算，标量索引加速标量过滤。如果你经常按某个字段过滤搜索结果，应该为该字段创建标量索引：

```python
# 准备索引参数
index_params = client.prepare_index_params()

# 向量索引
index_params.add_index(
    field_name="embedding",
    index_type="HNSW",
    metric_type="COSINE",
    params={"M": 16, "efConstruction": 256}
)

# 标量索引——为高频过滤字段创建
index_params.add_index(
    field_name="category",
    index_type="INVERTED",      # 倒排索引，适合等值查询
    index_name="idx_category"
)

index_params.add_index(
    field_name="year",
    index_type="STL_SORT",      # 排序索引，适合范围查询
    index_name="idx_year"
)

# 一次性创建所有索引
client.create_index(collection_name="documents", index_params=index_params)
```

标量索引类型的选择：

| 索引类型 | 适合场景 | Milvus 版本 |
|---------|---------|------------|
| INVERTED | 等值查询、范围查询、JSON 字段 | 2.5+（推荐） |
| STL_SORT | 数值范围查询（>、<、between） | 2.x |
| MARISA-TRIE | 字符串前缀匹配 | 2.x |

Milvus 2.5+ 推荐统一使用 INVERTED 索引——它同时支持等值查询和范围查询，性能优于 STL_SORT 和 MARISA-TRIE。

---

## 常见误区：建了索引但忘记 load

这在第 3 章已经提过，但值得再强调一次——Milvus 的索引创建和加载是两个独立的步骤。`create_index()` 只是把索引文件构建到对象存储中，`load_collection()` 才是把索引加载到 QueryNode 内存中。如果你只建了索引没 load，搜索要么报错要么退化为暴力扫描。

```python
# 完整流程：建索引 → load → 搜索
client.create_index(collection_name="documents", field_name="embedding", index_params={...})
client.load_collection("documents")  # ← 这一步不能省！
results = client.search(...)
```

另一个常见误区是**修改索引后没有重新 load**——如果你 drop 了旧索引并创建了新索引，需要重新 load 才能让新索引生效。

---

## 小结

这一节我们建立了索引选型和调优的完整方法论：用决策树选择索引类型，先调搜索参数（ef/nprobe）再调构建参数（M/efConstruction/nlist），用 Recall@K 评估召回率，为高频过滤字段创建标量索引。调优的核心原则是"先搜索参数后构建参数"——因为修改搜索参数不需要重建索引。下一节开始我们进入第 5 章，探索 Milvus 的高级特性。
