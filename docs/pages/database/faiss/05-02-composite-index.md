# 5.2 Index 组合与复合索引

> **FAISS 的 Index 可以像搭积木一样组合——这是向量数据库做不到的**

---

## 这一节在讲什么？

在 Milvus 和 pgvector 中，索引类型是固定的——你选 HNSW 就是 HNSW，选 IVF_PQ 就是 IVF_PQ，不能随意组合。FAISS 的 Index 可以像搭积木一样组合——你可以先用 PCA 降维，再用 IVF 分区，再用 PQ 量化，最后用 Flat 重排序。这种灵活的组合方式是 FAISS 最独特的能力，也是它作为"向量搜索库"而非"向量数据库"的优势——数据库为了易用性牺牲了灵活性，库为了灵活性牺牲了易用性。

---

## 常用组合模式

### IndexIDMap：给任意 Index 加自定义 ID

```python
import faiss
import numpy as np

d = 768

# IndexFlatL2 不支持 add_with_ids → 用 IndexIDMap 包装
base_index = faiss.IndexFlatL2(d)
index = faiss.IndexIDMap(base_index)

vectors = np.random.rand(10000, d).astype('float32')
ids = np.arange(1000, 11000)  # 自定义 ID

index.add_with_ids(vectors, ids)
query = np.random.rand(1, d).astype('float32')
distances, indices = index.search(query, k=5)
# indices 中是自定义 ID
```

### IndexPreTransform：在索引前做预处理

`IndexPreTransform` 允许你在索引前对向量做预处理——最常用的是 PCA 降维和 OPQ 旋转：

```python
# PCA 降维 + IVF 索引
# 先把 768 维降到 128 维，再建 IVF 索引
pca_matrix = faiss.PCAMatrix(768, 128)  # 768 → 128
ivf_index = faiss.IndexIVFFlat(faiss.IndexFlatL2(128), 128, 100)

index = faiss.IndexPreTransform(pca_matrix, ivf_index)

vectors_768 = np.random.rand(100000, 768).astype('float32')
index.train(vectors_768[:50000])
index.add(vectors_768)

query_768 = np.random.rand(1, 768).astype('float32')
distances, indices = index.search(query_768, k=5)
# FAISS 自动对 query 做 PCA 降维后再搜索
```

比如，下面的程序展示了 PCA 降维 + IVF + PQ 的组合，由于 PCA 降维减少了向量维度，所以后续的 IVF 和 PQ 操作都更高效：

```python
# PCA 降维 + IVF + PQ
d_orig = 768
d_pca = 128
m = 16  # PQ 子空间数（128 / 16 = 8 维/子空间）

pca = faiss.PCAMatrix(d_orig, d_pca)
quantizer = faiss.IndexFlatL2(d_pca)
ivf_pq = faiss.IndexIVFPQ(quantizer, d_pca, 1000, m, 8)

index = faiss.IndexPreTransform(pca, ivf_pq)

vectors = np.random.rand(1000000, d_orig).astype('float32')
index.train(vectors[:50000])
index.add(vectors)

index.nprobe = 32
query = np.random.rand(1, d_orig).astype('float32')
distances, indices = index.search(query, k=5)
```

### IndexRefineFlat：粗索引初筛 + Flat 重排序

`IndexRefineFlat` 实现了两阶段搜索——先用粗索引（如 IVFPQ）快速召回候选，再用 Flat 索引精确计算距离重排：

```python
# IVF_PQ 初筛 + Flat 重排序
nlist = 1000
m = 48

quantizer = faiss.IndexFlatL2(d)
base_index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)

# IndexRefineFlat 包装——存储原始向量用于重排
index = faiss.IndexRefineFlat(base_index)

index.train(vectors[:50000])
index.add(vectors)

index.nprobe = 32
# base_index 的 nprobe 通过 index.base_index.nprobe 设置
query = np.random.rand(1, d).astype('float32')
distances, indices = index.search(query, k=5)
# 返回的距离是 Flat 精确距离，不是 PQ 近似距离
```

---

## 常见误区：组合越多越好

每个组件都有开销——PCA 降维需要额外的矩阵乘法，IndexRefineFlat 需要存储原始向量（内存翻倍），IndexIDMap 需要维护 ID 映射表。过度组合不仅不会提升性能，反而会增加延迟和内存。正确的做法是**只在有明确需求时才添加组件**——需要降维才加 PCA，需要重排才加 RefineFlat，需要自定义 ID 才加 IndexIDMap。

---

## 小结

这一节我们聊了 FAISS 的积木式索引组合：IndexIDMap 加自定义 ID，IndexPreTransform 做预处理（PCA 降维/OPQ 旋转），IndexRefineFlat 做两阶段搜索。组合是 FAISS 最独特的能力，但不要过度组合——每个组件都有开销。下一节我们聊批量搜索和距离计算。
