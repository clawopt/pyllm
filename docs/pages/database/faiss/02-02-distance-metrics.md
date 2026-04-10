# 2.2 距离度量：L2 vs IP vs Cosine

> **FAISS 没有原生的 Cosine Index——但一行归一化就能搞定**

---

## 这一节在讲什么？

在 Milvus 教程中，我们学过三种距离度量（L2/Cosine/IP），并在创建 Collection 时指定。FAISS 也支持这三种度量，但实现方式不同——FAISS 有原生的 L2 和 IP Index，但没有原生的 Cosine Index。要实现余弦相似度搜索，你需要先对向量做 L2 归一化，然后用 IP Index 搜索。这一节我们要讲清楚 FAISS 中三种距离度量的实现方式和选择指南。

---

## L2 距离（欧氏距离）

FAISS 原生支持 L2 距离——Index 名字中带 `L2` 的就是 L2 距离索引：

```python
import faiss
import numpy as np

d = 768
index = faiss.IndexFlatL2(d)  # L2 距离的暴力搜索索引

vectors = np.random.rand(10000, d).astype('float32')
index.add(vectors)

query = np.random.rand(1, d).astype('float32')
distances, indices = index.search(query, k=5)
# distances 中的值是 L2 距离，≥ 0，越小越相似
```

L2 距离的值范围是 [0, +∞)——0 表示两个向量完全相同，值越大越不相似。

---

## IP 距离（内积）

FAISS 原生支持内积——Index 名字中带 `IP` 的就是内积索引：

```python
index = faiss.IndexFlatIP(d)  # 内积的暴力搜索索引

vectors = np.random.rand(10000, d).astype('float32')
index.add(vectors)

query = np.random.rand(1, d).astype('float32')
distances, indices = index.search(query, k=5)
# distances 中的值是内积，越大越相似（注意：跟 L2 相反！）
```

内积的值范围是 (-∞, +∞)——值越大越相似。注意这跟 L2 距离的"越小越相似"是相反的——FAISS 内部统一处理了这个差异，`search()` 返回的结果总是按相似度从高到低排列。

---

## Cosine 距离（余弦相似度）

FAISS **没有原生的 Cosine Index**——但你可以通过"先归一化再用 IP"来实现余弦相似度搜索。原理很简单：如果两个向量的 L2 范数都是 1（即归一化），那么它们的内积就等于余弦相似度。

```
数学推导：

  余弦相似度 = (a · b) / (||a|| × ||b||)

  如果 ||a|| = 1 且 ||b|| = 1（归一化后）：
  余弦相似度 = a · b = IP(a, b)

  所以：归一化向量 + IndexFlatIP = 余弦相似度搜索
```

```python
import faiss
import numpy as np

d = 768

# 生成随机向量
vectors = np.random.rand(10000, d).astype('float32')
query = np.random.rand(1, d).astype('float32')

# 关键步骤：L2 归一化——把每个向量的长度变为 1
faiss.normalize_L2(vectors)
faiss.normalize_L2(query)

# 用 IP Index 搜索——等价于余弦相似度搜索
index = faiss.IndexFlatIP(d)
index.add(vectors)

distances, indices = index.search(query, k=5)
# distances 中的值就是余弦相似度，范围 [0, 1]（对于非负向量），越大越相似
```

`faiss.normalize_L2()` 是 FAISS 提供的向量归一化函数——它把每个向量除以自己的 L2 范数，使得归一化后的向量长度为 1。这个操作是 in-place 的，直接修改原始数组。

---

## 距离度量的选择指南

| 场景 | 推荐度量 | FAISS 实现方式 |
|------|---------|--------------|
| 文本语义搜索 | Cosine | normalize_L2 + IndexFlatIP |
| 图像特征匹配 | L2 | IndexFlatL2 |
| 推荐系统 | IP | IndexFlatIP |
| 已归一化的向量 | IP 或 Cosine | 直接用 IndexFlatIP |

---

## 常见误区：直接用 IndexFlatIP 搜索未归一化的向量

这是 FAISS 初学者最容易犯的错误——以为 `IndexFlatIP` 就是余弦相似度搜索。实际上，`IndexFlatIP` 计算的是原始内积，只有在向量归一化后才等价于余弦相似度。如果你忘记归一化，搜索结果会偏向长度更大的向量——因为内积 = 余弦值 × 两个向量的长度，长度大的向量内积自然大，但并不代表更相似。

```python
# ❌ 错误：未归一化就用 IP——结果不是余弦相似度
vectors = np.random.rand(10000, d).astype('float32')
index = faiss.IndexFlatIP(d)
index.add(vectors)  # 向量长度不固定，IP ≠ Cosine

# ✅ 正确：先归一化再用 IP
faiss.normalize_L2(vectors)
index = faiss.IndexFlatIP(d)
index.add(vectors)  # 向量长度为 1，IP = Cosine
```

另一个常见误区是**只归一化了数据库向量，忘记归一化查询向量**——两边都需要归一化，余弦相似度才成立。

---

## 小结

这一节我们聊了 FAISS 的三种距离度量：L2 用 `IndexFlatL2`，IP 用 `IndexFlatIP`，Cosine 通过 `normalize_L2` + `IndexFlatIP` 实现。FAISS 没有原生的 Cosine Index，但归一化 + IP 的方案在数学上完全等价。关键注意事项：使用 IP 做余弦搜索时，数据库向量和查询向量都必须归一化。下一节我们聊向量 ID 与结果映射。
