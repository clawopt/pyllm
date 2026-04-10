# 4.2 PQ（Product Quantization）：乘积量化详解

> **PQ 是 FAISS 最核心的量化算法——Milvus 的 IVF_PQ 底层就是它**

---

## 这一节在讲什么？

PQ（乘积量化）是 FAISS 最核心的量化算法，也是 Milvus IVF_PQ 索引的底层实现。在 Milvus 教程中我们简单介绍了 PQ 的原理——把向量切成子空间，每个子空间独立量化。这一节我们要更深入地理解 PQ 的数学原理、FAISS 中 PQ 的实现、SDC vs ADC 距离计算方式，以及参数选择。

---

## PQ 原理深入

### 编码阶段

PQ 把 d 维向量切成 m 个子空间，每个子空间用 K-Means 聚类出 2^nbits 个中心点（码本），然后用中心点的 ID 来编码原始向量：

```
PQ 编码过程（768 维，m=48，nbits=8）：

  原始向量（768 维 float32，3072 字节）：
  [0.12, -0.34, 0.56, ..., 0.78]
  ┌───────────────────────────────────────────────┐
  │ 子空间1 │ 子空间2 │ ... │ 子空间48 │
  │ 16维    │ 16维    │     │ 16维     │
  └───────────────────────────────────────────────┘
       │         │              │
       ▼         ▼              ▼
  K-Means   K-Means         K-Means
  256中心   256中心         256中心
       │         │              │
       ▼         ▼              ▼
  编码=37   编码=142  ...  编码=89

  量化后：48 字节（每个子空间 1 字节）
  压缩比：3072 / 48 = 64 倍
```

### 距离计算：SDC vs ADC

PQ 的距离计算有两种方式——SDC（Symmetric Distance Computation）和 ADC（Asymmetric Distance Computation）：

- **SDC**：查询向量也做 PQ 编码，然后用编码之间的距离近似原始距离。速度最快但精度最低
- **ADC**：查询向量保持原始精度，只对数据库向量做 PQ 编码。精度更高，是 FAISS 的默认方式

```
SDC vs ADC：

  SDC（对称距离计算）：
  query 编码 → [37, 142, ..., 89]
  DB 向量编码 → [52, 88, ..., 201]
  距离 = 查表(37→52) + 查表(142→88) + ... + 查表(89→201)

  ADC（非对称距离计算）：
  query 原始 → [0.12, -0.34, ..., 0.78]
  DB 向量编码 → [52, 88, ..., 201]
  距离 = 查表(query子空间1→中心52) + 查表(query子空间2→中心88) + ...

  ADC 更精确——因为 query 保持了原始精度
```

---

## FAISS 中的 PQ

### IndexPQ：纯 PQ 索引

```python
import faiss
import numpy as np

d = 768
n = 1000000

vectors = np.random.rand(n, d).astype('float32')

# 创建 PQ 索引
m = 48      # 子空间数量（768 / 48 = 16 维/子空间）
nbits = 8   # 每个子空间的编码位数
index = faiss.IndexPQ(d, m, nbits)

# 训练——PQ 需要训练码本
index.train(vectors[:50000])  # 用子集训练
index.add(vectors)

# 搜索
query = np.random.rand(1, d).astype('float32')
distances, indices = index.search(query, k=5)
```

### IndexIVFPQ：IVF + PQ（最常用）

IndexIVFPQ 是 FAISS 中最常用的量化索引——它先用 IVF 分区减少搜索范围，再用 PQ 压缩向量减少内存：

```python
nlist = 1000
m = 48
nbits = 8

quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)

# 训练
index.train(vectors[:50000])
index.add(vectors)

# 搜索
index.nprobe = 32
query = np.random.rand(1, d).astype('float32')
distances, indices = index.search(query, k=5)
```

---

## PQ 参数选择

| 参数 | 含义 | 建议值 | 影响 |
|------|------|--------|------|
| m | 子空间数量 | dim/16（如 768→48） | m 越大精度越高但内存越大 |
| nbits | 编码位数 | 8（默认） | 目前 FAISS 只支持 8 |

m 的选择是最关键的——m 越大，每个子空间的维度越小，量化越精细，但编码长度也越长。768 维向量建议 m=48（每个子空间 16 维），1536 维向量建议 m=96。

### 常见误区：m 设得太小

m=8 意味着每个子空间有 96 维——这么大的子空间用 256 个中心点来近似，精度损失非常大。768 维向量建议 m ≥ 32，最好 m=48 或 64。

---

## 小结

这一节我们深入了 PQ 量化：把向量切成 m 个子空间，每个子空间用 K-Means 编码，压缩比 8~64 倍。距离计算有 SDC（对称）和 ADC（非对称）两种方式，ADC 更精确是默认方式。FAISS 中最常用的是 IndexIVFPQ——IVF 分区 + PQ 压缩。下一节我们聊 OPQ——PQ 的优化版本。
