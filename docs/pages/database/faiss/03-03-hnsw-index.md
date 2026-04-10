# 3.3 HNSW 索引：图搜索的极速体验

> **FAISS 的 HNSW 跟 pgvector/Milvus 的 HNSW 原理一样——但默认参数更保守**

---

## 这一节在讲什么？

HNSW 的原理我们在 pgvector 和 Milvus 教程中已经详细讲过——多层图结构、从顶层向下导航、O(log N) 搜索复杂度。FAISS 的 HNSW 实现跟 pgvector/Milvus 原理完全相同，但参数名和默认值不同。这一节我们重点聊 FAISS HNSW 的参数配置、与 pgvector/Milvus 的对比，以及 FAISS HNSW 的特殊之处——不需要 train。

---

## IndexHNSWFlat 的使用

```python
import faiss
import numpy as np

d = 768
n = 1000000

vectors = np.random.rand(n, d).astype('float32')

# 创建 HNSW 索引——不需要 train！
M = 32  # 每层最大连接数
index = faiss.IndexHNSWFlat(d, M)

# 直接添加向量——HNSW 在 add 时自动构建图结构
index.add(vectors)

# 设置搜索参数
index.hnsw.efSearch = 100  # 搜索宽度

# 搜索
query = np.random.rand(1, d).astype('float32')
distances, indices = index.search(query, k=5)

print(f"最近邻 ID: {indices}")
```

HNSW 不需要 `train()`——它在 `add()` 时自动构建图结构。每添加一条向量，HNSW 就在图中找到它的最近邻并建立连接。这意味着 HNSW 的 `add()` 比 IVF 的 `add()` 慢（因为需要搜索图来决定连接），但不需要额外的训练步骤。

---

## HNSW 参数详解

### M：每层最大连接数

```python
# M 在创建 Index 时指定
index = faiss.IndexHNSWFlat(d, M=16)   # 较小的 M，内存小
index = faiss.IndexHNSWFlat(d, M=32)   # FAISS 默认值
index = faiss.IndexHNSWFlat(d, M=64)   # 较大的 M，召回率高
```

### efConstruction：构建搜索宽度

```python
# efConstruction 在 add 之前设置
index = faiss.IndexHNSWFlat(d, M=32)
index.hnsw.efConstruction = 200  # 默认 40，建议 100~400
index.add(vectors)
```

`efConstruction` 控制的是 `add()` 时搜索最近邻的宽度——值越大，图连接质量越高，但构建越慢。FAISS 的默认值 40 比 pgvector 的 64 和 Milvus 的 256 都低——这意味着 FAISS 的 HNSW 默认构建质量不如 Milvus，你需要手动调高。

### efSearch：搜索宽度

```python
# efSearch 在 search 之前设置
index.hnsw.efSearch = 100  # 默认 16，建议 50~200
distances, indices = index.search(query, k=5)
```

`efSearch` 控制搜索时探索的图节点数量——值越大，召回率越高但速度越慢。FAISS 的默认值 16 比 pgvector 的 40 和 Milvus 的 64 都低——**这是 FAISS HNSW 最大的坑**，默认参数下召回率可能只有 80% 左右，生产环境一定要调高。

---

## FAISS HNSW vs pgvector HNSW vs Milvus HNSW

| 参数 | FAISS | pgvector | Milvus |
|------|-------|----------|--------|
| 连接数 | M（默认 32） | m（默认 16） | M（默认 16） |
| 构建宽度 | efConstruction（默认 40） | ef_construction（默认 64） | efConstruction（默认 256） |
| 搜索宽度 | efSearch（默认 16） | ef_search（默认 40） | ef（默认 64） |
| 是否需要 train | ❌ | ❌ | ❌ |
| 增量添加 | ✅ | ✅ | ✅ |

FAISS 的默认参数最保守——efConstruction=40 和 efSearch=16 都比 pgvector 和 Milvus 低。这意味着如果你用 FAISS 的默认参数，HNSW 的召回率会明显低于 pgvector 和 Milvus。**生产环境务必调高 efConstruction 和 efSearch**。

---

## 常见误区：FAISS HNSW 的 efSearch 默认值太低

FAISS HNSW 的 `efSearch` 默认值是 16——这在大多数场景下太低了。比如下面的程序，由于 efSearch=16 时搜索宽度不够，所以召回率只有 80% 左右，而调到 100 后召回率可以提升到 95% 以上：

```python
# 测试不同 efSearch 的召回率
for ef in [16, 32, 64, 100, 200]:
    index.hnsw.efSearch = ef
    recall = measure_recall(index, flat_index, test_queries, k=10)
    print(f"efSearch={ef}: Recall@10={recall:.4f}")

# 典型输出：
# efSearch=16:  Recall@10=0.8120  ← 默认值，太低！
# efSearch=32:  Recall@10=0.8930
# efSearch=64:  Recall@10=0.9410
# efSearch=100: Recall@10=0.9680  ← 推荐生产值
# efSearch=200: Recall@10=0.9890
```

---

## 小结

这一节我们聊了 FAISS 的 HNSW 索引：不需要 train，直接 add 即可。参数 M/efConstruction/efSearch 跟 pgvector/Milvus 含义相同但默认值不同——FAISS 的默认值更保守，特别是 efSearch=16 在生产环境太低，务必调到 100 以上。下一节开始我们进入第 4 章，深入 FAISS 最强大的能力——量化与压缩索引。
