# 2.1 Index：FAISS 的核心抽象

> **FAISS 的一切都是 Index——理解 Index 就理解了 FAISS 的设计哲学**

---

## 这一节在讲什么？

在 Milvus 中，你操作的是 Collection（集合）——它包含数据、索引和 Schema。在 FAISS 中，你操作的是 Index（索引）——它同时包含数据和索引结构，是 FAISS 的核心抽象。所有的向量搜索操作都通过 Index 的方法完成。这一节我们要把 Index 的核心方法、属性和分类讲清楚，帮你建立对 FAISS API 的完整认知。

---

## Index 是什么

Index 是 FAISS 中所有向量索引的基类——它封装了"添加向量"和"搜索向量"的接口。你可以把 Index 理解为一个黑盒：你往里面塞向量，然后给它一个查询向量，它吐出最相似的 K 个向量的 ID 和距离。

```
Index 的工作模式：

  添加向量：vectors → [Index] → 内部存储+索引结构
  搜索向量：query  → [Index] → (distances, indices)
```

FAISS 提供了 20+ 种 Index 实现，每种都有不同的索引结构和性能特征。但它们的接口是一致的——你不需要为不同的 Index 学习不同的 API。

---

## Index 的核心方法

### add(vectors)：添加向量

```python
import faiss
import numpy as np

d = 768
index = faiss.IndexFlatL2(d)

# 添加 10000 条向量
vectors = np.random.rand(10000, d).astype('float32')
index.add(vectors)

print(f"索引中的向量数: {index.ntotal}")  # 10000
```

`add()` 接受一个 (n, d) 形状的 float32 numpy 数组，n 是向量数量，d 是维度。添加后，每条向量自动获得一个从 0 开始的顺序 ID。

### search(query, k)：搜索最近邻

```python
# 搜索 5 个最近邻
query = np.random.rand(1, d).astype('float32')
distances, indices = index.search(query, k=5)

# 批量搜索——一次传入多个查询向量
queries = np.random.rand(100, d).astype('float32')
distances, indices = index.search(queries, k=5)
# distances.shape = (100, 5)
# indices.shape = (100, 5)
```

`search()` 返回两个数组：
- **distances**：形状 (n_queries, k)，每个查询向量的 K 个最近邻的距离值
- **indices**：形状 (n_queries, k)，每个查询向量的 K 个最近邻的 ID

### add_with_ids(vectors, ids)：添加向量并指定自定义 ID

默认情况下，FAISS 用向量的添加顺序作为 ID（0, 1, 2, ...）。但很多场景下你需要用自定义 ID（比如数据库中的主键）。`add_with_ids()` 允许你指定任意 int64 ID：

```python
# IndexFlatL2 不支持 add_with_ids——需要用 IndexIDMap 包装
index = faiss.IndexIDMap(faiss.IndexFlatL2(d))

vectors = np.random.rand(10000, d).astype('float32')
ids = np.arange(1000, 11000)  # 自定义 ID：从 1000 开始

index.add_with_ids(vectors, ids)

# 搜索结果中的 indices 就是自定义 ID
query = np.random.rand(1, d).astype('float32')
distances, indices = index.search(query, k=5)
print(f"最近邻 ID: {indices}")  # 输出类似 [[1042 7823 5291 ...]]
```

### range_search(query, radius)：范围搜索

`range_search()` 返回距离小于某个阈值的所有向量——不是固定 K 个，而是所有满足条件的向量：

```python
query = np.random.rand(1, d).astype('float32')
# 返回 L2 距离小于 100 的所有向量
lims, distances, indices = index.range_search(query, radius=100.0)

# lims: 每个查询向量的结果范围，lims[0]:lims[1] 是第一个查询的结果
# distances: 所有满足条件的距离值
# indices: 所有满足条件的向量 ID
print(f"满足条件的向量数: {lims[1] - lims[0]}")
```

### reset()：清空索引

```python
index.reset()
print(f"清空后的向量数: {index.ntotal}")  # 0
```

---

## Index 的核心属性

```python
index = faiss.IndexFlatL2(768)

print(index.d)         # 维度：768
print(index.ntotal)    # 向量数：0（还没添加数据）
print(index.is_trained) # 是否已训练：True（Flat 索引不需要训练）
```

- **d**：向量维度，创建 Index 时指定，不可更改
- **ntotal**：索引中的向量总数，每次 add 后自动更新
- **is_trained**：索引是否已完成训练。IVF/PQ 等索引需要先 train 再 add，Flat/HNSW 不需要

---

## Index 的分类

FAISS 的 Index 可以按几个维度分类：

| 分类维度 | 类型 | 说明 |
|---------|------|------|
| 搜索方式 | Flat | 暴力搜索，100% 召回率 |
| | IVF | 倒排索引，搜索部分聚类 |
| | HNSW | 图索引，O(log N) 搜索 |
| 压缩方式 | 无压缩 | 存储原始 float32 向量 |
| | PQ | 乘积量化，压缩比 8~64 倍 |
| | SQ | 标量量化，压缩比 4 倍 |
| 是否需要训练 | 不需要 | Flat、HNSW |
| | 需要 | IVF、PQ、IVFPQ |

---

## 常见误区：Index 同时包含数据和索引结构

在 Milvus 中，数据和索引是分开的——你先创建 Collection 存数据，再创建 Index 加速搜索。在 FAISS 中，Index 同时包含数据和索引结构——`add()` 把向量添加到 Index 内部，索引结构也在 Index 内部维护。这意味着你不能像 Milvus 那样"先加数据再建索引"——FAISS 的 Index 在 `add()` 时就同时处理了数据和索引。

---

## 小结

这一节我们深入了 FAISS 的核心抽象 Index：`add()` 添加向量、`search()` 搜索最近邻、`add_with_ids()` 指定自定义 ID、`range_search()` 范围搜索。Index 同时包含数据和索引结构，这是 FAISS 与 Milvus 的一个根本区别。下一节我们聊距离度量——L2、IP 和 Cosine 在 FAISS 中的实现方式。
