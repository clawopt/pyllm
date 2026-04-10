# 2.3 距离度量：L2 / Cosine / IP

> **选错距离度量，搜索结果就全错了——这不是参数调优能救回来的**

---

## 这一节在讲什么？

在 pgvector 教程里，我们学过三种距离操作符（`<->`、`<=>`、`<#>`），它们对应三种距离度量。Milvus 也支持同样的三种距离度量——L2（欧氏距离）、Cosine（余弦距离）、IP（内积），但有一个关键区别：pgvector 在查询时选择操作符，Milvus 在创建 Collection 时指定度量类型。这意味着 Milvus 的距离度量一旦选定就不能更改——如果你选错了，只能删掉 Collection 重建。这一节我们要深入理解三种距离度量的原理、选择依据，以及 Milvus 的设计为什么跟 pgvector 不同。

---

## 三种距离度量的数学定义

### L2 距离（欧氏距离）

L2 距离是空间中两点之间的直线距离，这是最直观的距离概念——如果你把向量想象成三维空间中的点，L2 距离就是两个点之间的直线长度。

```
L2 距离公式：
  d(x, y) = √(Σ(xi - yi)²)

直觉理解：
  二维空间中，点 A(1, 2) 和点 B(4, 6) 的 L2 距离：
  d = √((1-4)² + (2-6)²) = √(9 + 16) = √25 = 5

特点：
  - 值 ≥ 0，值越小越相似
  - 受向量长度影响——长度不同的向量即使方向相同，L2 距离也可能很大
  - 适合：图像特征匹配、物理距离计算
```

### Cosine 距离（余弦距离）

Cosine 距离衡量的是两个向量方向的差异——它只关注向量指向的方向，忽略向量的长度。想象两个箭头从原点出发，Cosine 距离衡量的是它们之间的夹角。

```
余弦相似度公式：
  cos(x, y) = (x · y) / (||x|| × ||y||)

余弦距离公式：
  d(x, y) = 1 - cos(x, y)

直觉理解：
  两个方向相同的向量：cos = 1，距离 = 0（最相似）
  两个方向垂直的向量：cos = 0，距离 = 1（不相关）
  两个方向相反的向量：cos = -1，距离 = 2（最不相似）

特点：
  - 值 ∈ [0, 2]，值越小越相似
  - 不受向量长度影响——只看方向
  - 适合：文本语义搜索（embedding 的方向代表语义，长度不重要）
```

### IP 距离（内积）

IP（Inner Product）距离就是两个向量的点积。点积同时考虑了向量的长度和方向——两个向量越长、方向越一致，点积越大。

```
内积公式：
  IP(x, y) = Σ(xi × yi)

Milvus 中的内积距离：
  d(x, y) = -IP(x, y)  （取负值，使得值越小越相似）

直觉理解：
  如果向量已归一化（长度为 1），则 IP = cos(x, y)
  如果向量未归一化，IP 同时受长度和方向影响

特点：
  - 值越小越相似（Milvus 取了负值）
  - 受向量长度影响——长度越大的向量，内积可能越大
  - 适合：推荐系统（用户向量和物品向量的内积表示偏好强度）
```

---

## 距离度量的选择指南

| 场景 | 推荐度量 | 原因 |
|------|---------|------|
| 文本语义搜索 | Cosine | 文本 embedding 关注语义方向，长度不重要 |
| 图像特征匹配 | L2 或 IP | 取决于模型输出是否归一化 |
| 推荐系统 | IP | 内积表示偏好强度，未归一化的向量有额外信息 |
| 已归一化的向量 | IP 或 Cosine | 归一化后 IP = Cosine，两者等价 |
| 化学分子检索 | L2 | 分子指纹的距离用欧氏距离更直观 |

比如，下面的代码展示了如何在创建 Collection 时指定距离度量：

```python
from pymilvus import MilvusClient

client = MilvusClient(uri="http://localhost:19530")

# 文档问答系统 → 用 Cosine
client.create_collection(
    collection_name="documents",
    dimension=768,
    metric_type="COSINE"    # 指定余弦距离
)

# 推荐系统 → 用 IP
client.create_collection(
    collection_name="recommendations",
    dimension=128,
    metric_type="IP"         # 指定内积距离
)

# 图像搜索 → 用 L2
client.create_collection(
    collection_name="images",
    dimension=512,
    metric_type="L2"         # 指定欧氏距离
)
```

---

## Milvus vs pgvector：度量指定的时机

这是 Milvus 和 pgvector 在距离度量上的根本区别：

- **pgvector**：在查询时选择操作符——`<->`（L2）、`<=>`（Cosine）、`<#>`（IP）。同一个表可以用不同度量搜索
- **Milvus**：在创建 Collection 时指定 `metric_type`——之后所有搜索都使用这个度量。不能中途换

Milvus 为什么这样设计？因为距离度量决定了索引的构建方式——HNSW 索引在构建时就确定了距离计算方式，搜索时不能换。这避免了 pgvector 中"索引用 `vector_cosine_ops` 建的，但查询用了 `<->` 操作符导致索引失效"的问题。Milvus 的设计更严格，但更不容易出错。

```
pgvector 的灵活性 vs Milvus 的严格性：

  pgvector：
  CREATE INDEX idx ON docs USING hnsw (embedding vector_cosine_ops);
  SELECT * FROM docs ORDER BY embedding <-> query_vec;  -- ❌ 索引不匹配，走全表扫描！
  SELECT * FROM docs ORDER BY embedding <=> query_vec;  -- ✅ 索引匹配

  Milvus：
  create_collection(metric_type="COSINE")
  search(metric_type="COSINE")  -- ✅ 必须匹配，不会出错
  search(metric_type="L2")      -- ❌ 直接报错，不允许不匹配
```

---

## 使用 ORM API 指定距离度量

如果你使用 ORM API（而不是 MilvusClient），距离度量在创建索引时指定：

```python
from pymilvus import Collection

collection = Collection("documents")

# 创建索引时指定 metric_type
index_params = {
    "metric_type": "COSINE",
    "index_type": "HNSW",
    "params": {"M": 16, "efConstruction": 256}
}
collection.create_index(field_name="embedding", index_params=index_params)
```

注意：ORM API 中 `metric_type` 是在 `create_index()` 时指定的，而不是 `create_collection()` 时。但 MilvusClient 的快速创建方式（`create_collection(dimension=768, metric_type="COSINE")`）会自动处理这个对应关系。

---

## 常见误区：所有场景都用 Cosine

Cosine 距离在文本语义搜索中确实是最常用的，但不是所有场景都适合。最典型的错误是在推荐系统中用 Cosine——推荐系统的用户向量和物品向量通常是未归一化的，向量的长度包含了"活跃度"或"热门度"的信息。如果你用 Cosine 距离，就丢失了这些信息，搜索结果的质量会下降。

另一个常见误区是**在搜索时指定了错误的 metric_type**——如果你创建 Collection 时用了 COSINE，搜索时必须在 search 参数中也指定 `metric_type="COSINE"`。虽然 Milvus 会检查匹配性，但如果你用的是 ORM API 的低级接口，可能不会报错而是静默地使用错误的度量。

---

## 小结

这一节我们深入了三种距离度量的原理和选择：L2 适合物理距离和图像特征，Cosine 适合文本语义搜索，IP 适合推荐系统和归一化向量。Milvus 在创建 Collection 时指定度量，一旦选定不能更改——这比 pgvector 更严格，但也避免了操作符不匹配的问题。下一节开始我们进入第 3 章，学习 Milvus 的数据操作——插入、搜索、过滤、更新和删除。
