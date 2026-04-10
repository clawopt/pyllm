# 3.2 IVF 索引：倒排文件加速

> **IVF 是 FAISS 中最灵活的索引——train + add + search，三步搞定**

---

## 这一节在讲什么？

IVF（Inverted File）索引是 FAISS 中最重要的索引类型之一——它的原理我们在 pgvector 和 Milvus 教程中已经详细讲过：K-Means 聚类把向量空间划分为 nlist 个区域，搜索时只扫描 nprobe 个区域。FAISS 的 IVF 实现跟 pgvector/Milvus 的 IVFFlat 原理完全相同，但有一个关键区别：FAISS 的 IVF 索引需要先 `train()` 再 `add()`——这个"训练"步骤在 pgvector 和 Milvus 中是自动完成的。这一节我们要深入 FAISS IVF 的使用方法、train 概念、参数调优，以及与 pgvector/Milvus 的对比。

---

## IVF 原理回顾

```
IVF 索引的工作原理：

  训练阶段：K-Means 聚类
  ┌────────────────────────────────────────┐
  │  全量向量 → K-Means → nlist 个聚类中心  │
  └────────────────────────────────────────┘

  添加阶段：把每条向量分配到最近的聚类
  ┌──────┐ ┌──────┐ ┌──────┐
  │ 聚类1 │ │ 聚类2 │ │ 聚类3 │
  │ vec1  │ │ vec3  │ │ vec5  │
  │ vec2  │ │ vec4  │ │ vec8  │
  │ vec7  │ │ vec6  │ │ ...   │
  └──────┘ └──────┘ └──────┘

  搜索阶段：只扫描查询向量附近的 nprobe 个聚类
  query → 找到最近的 nprobe 个聚类 → 在这些聚类中搜索 Top-K
```

---

## IndexIVFFlat 的使用

```python
import faiss
import numpy as np

d = 768
n = 1000000  # 100 万条向量

vectors = np.random.rand(n, d).astype('float32')

# 第1步：创建量化器（用于聚类）
quantizer = faiss.IndexFlatL2(d)

# 第2步：创建 IVF 索引
nlist = 1000  # 聚类数量，建议 √N
index = faiss.IndexIVFFlat(quantizer, d, nlist)

# 第3步：训练——学习聚类中心
index.train(vectors)  # 必须先训练！

# 第4步：添加向量
index.add(vectors)

# 第5步：搜索
index.nprobe = 32  # 搜索时扫描的聚类数量
query = np.random.rand(1, d).astype('float32')
distances, indices = index.search(query, k=5)

print(f"索引中的向量数: {index.ntotal}")
print(f"是否已训练: {index.is_trained}")
print(f"最近邻 ID: {indices}")
```

### train() 的概念

这是 FAISS IVF 索引最独特的概念——你需要先调用 `train()` 学习聚类中心，然后才能 `add()` 向量。在 pgvector 和 Milvus 中，这个训练步骤是在 `CREATE INDEX` 时自动完成的，你不需要手动调用。但 FAISS 把这个步骤暴露给了用户，让你有更多的控制权。

```python
# ❌ 错误：不训练直接 add
index = faiss.IndexIVFFlat(quantizer, d, nlist)
index.add(vectors)  # 报错！Index not trained

# ✅ 正确：先训练再添加
index = faiss.IndexIVFFlat(quantizer, d, nlist)
index.train(vectors)  # 训练聚类中心
index.add(vectors)    # 添加向量
```

### 训练数据的要求

训练数据应该代表真实数据分布——如果你用随机数据训练，但实际数据是文本 embedding，聚类中心就不准确，搜索质量会下降。训练数据的量也有要求——FAISS 建议训练数据至少是 nlist × 39 条，即 nlist=1000 时至少需要 39000 条训练数据。

```python
# 训练数据不需要是全量数据——一个子集就够了
train_size = min(nlist * 39, n)  # 至少 nlist × 39 条
train_vectors = vectors[:train_size]
index.train(train_vectors)

# 然后用全量数据 add
index.add(vectors)
```

---

## nlist 和 nprobe 参数

| 参数 | 含义 | 建议值 | 对应 pgvector/Milvus |
|------|------|--------|---------------------|
| nlist | 聚类数量 | √N | lists |
| nprobe | 搜索时扫描的聚类数 | nlist × 5% | probes |

```python
# nprobe 的设置方式
index.nprobe = 32  # 直接设置属性

# nprobe 对性能和召回率的影响
for nprobe in [1, 10, 32, 64, 128, 256]:
    index.nprobe = nprobe
    distances, indices = index.search(query, k=10)
    recall = measure_recall(index, flat_index, test_queries, k=10)
    print(f"nprobe={nprobe}: Recall@10={recall:.4f}")
```

---

## 与 pgvector/Milvus IVFFlat 的对比

| 维度 | FAISS | pgvector | Milvus |
|------|-------|----------|--------|
| 聚类数参数 | nlist | lists | nlist |
| 搜索参数 | nprobe | probes | nprobe |
| 训练步骤 | 手动 train() | 自动（CREATE INDEX 时） | 自动（create_index 时） |
| 索引+数据 | Index 同时包含 | 索引和数据分开 | 索引和数据分开 |
| 增量添加 | ✅ add() | ✅ INSERT | ✅ insert() |
| 聚类中心更新 | ❌ 不自动更新 | ❌ 需重建索引 | ❌ 需重建索引 |

---

## 常见误区：训练数据与实际数据分布不一致

如果你用随机数据训练 IVF 索引，但实际存储的是文本 embedding，聚类中心就不准确——因为随机数据的分布是均匀的，而文本 embedding 的分布是高度聚集的。这会导致搜索时很多聚类是空的，而少数聚类包含了大部分数据，nprobe 的效果大打折扣。正确的做法是**用真实数据的一个子集来训练**。

---

## 小结

这一节我们深入了 FAISS 的 IVF 索引：必须先 `train()` 再 `add()`，训练数据要代表真实分布，nlist 和 nprobe 参数跟 pgvector/Milvus 的 lists/probes 含义相同。FAISS 把训练步骤暴露给用户，提供了更多控制权但也增加了使用复杂度。下一节我们聊 HNSW 索引。
