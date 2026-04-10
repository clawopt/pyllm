# 3.1 Flat 索引：暴力搜索的基准

> **Flat 不是"差"——它是 100% 召回率的保证，也是所有索引的基准线**

---

## 这一节在讲什么？

Flat 索引是 FAISS 中最简单的索引类型——它不做任何优化，搜索时逐条计算查询向量与所有向量的距离。听起来很"笨"，但 Flat 有两个不可替代的用途：一是数据量小时暴力搜索已经够快，二是作为基准测试的"标准答案"来评估其他索引的召回率。这一节我们快速过一遍 Flat 索引的用法和性能特征。

---

## IndexFlatL2 / IndexFlatIP

```python
import faiss
import numpy as np

d = 768
n = 100000

vectors = np.random.rand(n, d).astype('float32')

# L2 距离的 Flat 索引
index_l2 = faiss.IndexFlatL2(d)
index_l2.add(vectors)

# 内积的 Flat 索引
faiss.normalize_L2(vectors)
index_ip = faiss.IndexFlatIP(d)
index_ip.add(vectors)

# 搜索
query = np.random.rand(1, d).astype('float32')
faiss.normalize_L2(query)

distances_l2, indices_l2 = index_l2.search(query, k=5)
distances_ip, indices_ip = index_ip.search(query, k=5)
```

---

## Flat 的性能

Flat 索引的搜索复杂度是 O(N×d)——N 是向量数量，d 是维度。每次搜索需要计算查询向量与所有 N 条向量的距离。

| 向量数量 | 维度 | 搜索延迟（CPU） |
|---------|------|---------------|
| 1 万 | 768 | ~1 ms |
| 10 万 | 768 | ~10 ms |
| 100 万 | 768 | ~100 ms |
| 1000 万 | 768 | ~1000 ms |

10 万条以下，Flat 的搜索延迟在 10ms 以内，完全够用。100 万条时延迟约 100ms，对大多数 RAG 场景也可以接受。超过 100 万条就需要考虑 IVF 或 HNSW 索引了。

---

## Flat 作为基准测试

Flat 索引的召回率是 100%——因为它检查了每一条向量。这个特性让 Flat 成为评估其他索引召回率的"标准答案"：

```python
def measure_recall(index_ann, index_flat, queries, k=10):
    """测量 ANN 索引的召回率"""
    _, gt_indices = index_flat.search(queries, k)  # Ground Truth
    _, ann_indices = index_ann.search(queries, k)  # ANN 结果

    total_recall = 0
    for i in range(len(queries)):
        gt_set = set(gt_indices[i])
        ann_set = set(ann_indices[i])
        recall = len(gt_set & ann_set) / k
        total_recall += recall

    return total_recall / len(queries)

# 用法
recall = measure_recall(ivf_index, flat_index, test_queries, k=10)
print(f"Recall@10: {recall:.4f}")
```

---

## 常见误区：Flat 索引没有意义

有些同学觉得"Flat 就是暴力搜索，没有技术含量"。但 Flat 有两个不可替代的用途：第一，数据量小时 Flat 的性能已经足够好，不需要更复杂的索引；第二，Flat 是唯一能保证 100% 召回率的索引——当你需要验证其他索引的质量时，Flat 是唯一的基准。在生产环境中，很多团队会用"Flat 初筛 + 业务逻辑过滤"的方案来处理小规模数据。

---

## 小结

这一节我们快速过了 Flat 索引：IndexFlatL2/IndexFlatIP 是暴力搜索，召回率 100%，10 万条以下性能足够。Flat 的核心价值是作为基准测试的"标准答案"。下一节我们聊 IVF 索引——FAISS 中最重要的索引类型之一。
