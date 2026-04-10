# 4.4 SQ（Scalar Quantization）：标量量化

> **SQ 比 PQ 简单得多——4 倍压缩，精度损失小，不需要训练（大部分模式）**

---

## 这一节在讲什么？

PQ 和 OPQ 通过子空间划分和聚类来实现高压缩比，但实现复杂、需要训练。SQ（Scalar Quantization）走了另一条路——它把每个 float32 维度直接映射到 int8，压缩比只有 4 倍，但实现简单、精度损失小、大部分模式不需要训练。这一节我们快速过一遍 FAISS 的 SQ 实现和 6 种量化模式。

---

## SQ 原理

SQ 的原理非常简单——找到每个维度的最小值和最大值，然后把 float32 线性映射到 int8 的范围（-128~127）：

```
SQ 量化过程：

  原始向量（float32）：
  [0.12, -0.34, 0.56, ..., 0.78]  → 768 × 4 = 3072 字节

  每个维度独立量化：
  维度1: 范围 [-1.0, 1.0] → 0.12 映射到 int8 值
  维度2: 范围 [-0.8, 0.8] → -0.34 映射到 int8 值
  ...

  量化后（int8）：
  [31, -87, 143, ..., 200]         → 768 × 1 = 768 字节

  压缩比：4 倍
```

---

## FAISS 的 SQ 实现

```python
import faiss
import numpy as np

d = 768
n = 1000000
vectors = np.random.rand(n, d).astype('float32')

# IndexScalarQuantizer——6 种量化模式
# QT_8bit：每个维度用 8 bit 量化（最常用）
index = faiss.IndexScalarQuantizer(d, faiss.ScalarQuantizer.QT_8bit)

# 不需要训练！——量化参数从数据统计得到
index.add(vectors)

query = np.random.rand(1, d).astype('float32')
distances, indices = index.search(query, k=5)
```

### 6 种量化模式

| 模式 | 每维度比特数 | 压缩比 | 需要训练 | 精度 |
|------|-----------|--------|---------|------|
| QT_8bit | 8 bit | 4 倍 | 否 | 好 |
| QT_6bit | 6 bit | 5.3 倍 | 否 | 中 |
| QT_4bit | 4 bit | 8 倍 | 否 | 较差 |
| QT_8bit_uniform | 8 bit | 4 倍 | 是 | 好（均匀量化） |
| QT_fp16 | 16 bit | 2 倍 | 否 | 极好 |
| QT_8bit_direct | 8 bit | 4 倍 | 否 | 取决于输入 |

最常用的是 `QT_8bit`——它不需要训练，4 倍压缩，精度损失小。`QT_fp16` 把 float32 压缩成 float16，只有 2 倍压缩但精度几乎无损。

### IndexIVFScalarQuantizer：IVF + SQ

```python
nlist = 1000
quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFScalarQuantizer(
    quantizer, d, nlist,
    faiss.ScalarQuantizer.QT_8bit
)

index.train(vectors[:50000])
index.add(vectors)
index.nprobe = 32

query = np.random.rand(1, d).astype('float32')
distances, indices = index.search(query, k=5)
```

---

## SQ vs PQ

| 维度 | SQ | PQ |
|------|-----|-----|
| 压缩比 | 4 倍（QT_8bit） | 8~64 倍 |
| 精度损失 | 小 | 中 |
| 是否需要训练 | 大部分模式不需要 | 需要 |
| 实现复杂度 | 低 | 高 |
| 距离计算速度 | 快 | 快（查表法） |
| 推荐场景 | 精度优先、压缩比要求不高 | 内存有限、需要高压缩比 |

---

## 常见误区：SQ 不需要训练

大部分 SQ 模式确实不需要训练——因为量化参数（每个维度的最小值和最大值）可以从数据统计得到。但 `QT_8bit_uniform` 模式需要训练——它使用全局统一的量化范围，需要通过训练来确定最优的范围值。如果你不确定该用哪种模式，就用 `QT_8bit`——它不需要训练，效果也不错。

---

## 小结

这一节我们聊了 SQ 标量量化：把 float32 线性映射到 int8，压缩比 4 倍，精度损失小，大部分模式不需要训练。FAISS 提供 6 种 SQ 模式，最常用的是 QT_8bit。SQ 适合精度优先、压缩比要求不高的场景；PQ 适合内存有限、需要高压缩比的场景。下一节开始我们进入第 5 章，探索 FAISS 的 GPU 加速和高级特性。
