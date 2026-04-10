# 4.3 OPQ（Optimized Product Quantization）：优化量化

> **OPQ 解决了 PQ 的"子空间不均衡"问题——用旋转矩阵让量化更精确**

---

## 这一节在讲什么？

PQ 的一个隐含假设是：把向量均匀切成 m 个子空间，每个子空间的"信息量"是差不多的。但实际数据往往不是这样的——某些维度的方差很大（信息量大），某些维度的方差很小（信息量小）。如果某个子空间恰好包含了多个高方差维度，它的量化误差就很大。OPQ（Optimized Product Quantization）通过学习一个旋转矩阵来优化子空间划分，使得每个子空间的方差更均衡，从而提升量化精度。

---

## OPQ 解决了什么问题

```
PQ 的子空间不均衡问题：

  原始向量（8 维，m=2）：
  维度:  [d1, d2, d3, d4, d5, d6, d7, d8]
  方差:  [10,  8,   0.1, 0.2, 9,   7,   0.3, 0.1]

  PQ 均匀切分：
  子空间1: [d1, d2, d3, d4] → 方差 [10, 8, 0.1, 0.2] → 极不均衡！
  子空间2: [d5, d6, d7, d8] → 方差 [9,  7, 0.3, 0.1] → 极不均衡！

  OPQ 旋转后切分：
  旋转矩阵 R 把高方差和低方差维度"打散"
  子空间1: [Rd1, Rd3, Rd5, Rd7] → 方差均衡
  子空间2: [Rd2, Rd4, Rd6, Rd8] → 方差均衡
```

OPQ 的核心思想是：在 PQ 之前学习一个旋转矩阵 R，使得 R×X 的子空间方差更均衡。这个旋转矩阵通过优化算法学习得到，目标是让量化误差最小。

---

## FAISS 中的 OPQ

```python
import faiss
import numpy as np

d = 768
n = 1000000
m = 48

vectors = np.random.rand(n, d).astype('float32')

# 方法1：使用 IndexPreTransform + IndexPQ
# OPQ 旋转 + PQ 量化
opq_matrix = faiss.OPQMatrix(d, m)
index_pq = faiss.IndexPQ(d, m, 8)
index = faiss.IndexPreTransform(opq_matrix, index_pq)

index.train(vectors[:50000])
index.add(vectors)

query = np.random.rand(1, d).astype('float32')
distances, indices = index.search(query, k=5)

# 方法2：使用 IndexIVFPQ + OPQ 旋转
nlist = 1000
quantizer = faiss.IndexFlatL2(d)
index_ivfpq = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)

# 设置 OPQ 旋转
opq = faiss.OPQMatrix(d, m)
index = faiss.IndexPreTransform(opq, index_ivfpq)

index.train(vectors[:50000])
index.add(vectors)
index.nprobe = 32
distances, indices = index.search(query, k=5)
```

---

## OPQ vs PQ 的性能对比

OPQ 的召回率通常比 PQ 高 3~5 个百分点，代价是额外的训练时间（需要学习旋转矩阵）：

| 量化方式 | 训练时间 | 召回率（典型值） | 内存占用 |
|---------|---------|---------------|---------|
| PQ | 快 | 85%~90% | 相同 |
| OPQ | 慢（2~3 倍） | 90%~95% | 相同 |

OPQ 的额外内存开销可以忽略——旋转矩阵 R 的大小是 d×d，对于 768 维来说只有 2.3 MB，相比向量数据本身微不足道。

---

## 何时使用 OPQ

- 对召回率要求高（> 90%），PQ 达不到要求时
- 数据的维度间方差差异大（某些维度信息量大，某些很小）
- 愿意多花训练时间换取更好的搜索质量

---

## 常见误区：OPQ 总是比 PQ 好

OPQ 的优势取决于数据的特性——如果数据的维度间方差本身就很均衡（比如已经做过 PCA 白化），OPQ 的收益很小。而且 OPQ 的训练时间比 PQ 长 2~3 倍——如果你的训练时间预算有限，可能 PQ + 更大的 nprobe 是更好的选择。

---

## 小结

这一节我们聊了 OPQ 量化：通过旋转矩阵优化子空间划分，使每个子空间的方差更均衡，召回率比 PQ 高 3~5 个百分点。FAISS 中用 `IndexPreTransform(OPQMatrix, IndexPQ)` 实现 OPQ。OPQ 适合对召回率要求高、愿意多花训练时间的场景。下一节我们聊 SQ——更简单但压缩比更小的量化方式。
