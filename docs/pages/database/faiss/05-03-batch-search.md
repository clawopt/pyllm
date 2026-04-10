# 5.3 批量搜索与距离计算

> **批量搜索比逐条搜索快 10~100 倍——FAISS 的性能秘诀**

---

## 这一节在讲什么？

FAISS 的 `search()` 方法支持一次传入多个查询向量——这不是语法糖，而是性能的关键。GPU 的并行计算能力只有在批量搜索时才能充分发挥，CPU 的 SIMD 指令集也是在批量操作时效率最高。这一节我们聊批量搜索、距离矩阵计算、以及不需要创建索引的 knn 搜索。

---

## 批量搜索

```python
import faiss
import numpy as np
import time

d = 768
n = 1000000
vectors = np.random.rand(n, d).astype('float32')

index = faiss.IndexFlatL2(d)
index.add(vectors)

# 逐条搜索——慢
queries = np.random.rand(100, d).astype('float32')
start = time.time()
for q in queries:
    index.search(q.reshape(1, -1), k=5)
print(f"逐条搜索 100 个查询: {time.time() - start:.3f}s")

# 批量搜索——快
start = time.time()
distances, indices = index.search(queries, k=5)
print(f"批量搜索 100 个查询: {time.time() - start:.3f}s")
```

批量搜索的速度通常是逐条搜索的 10~100 倍——因为 FAISS 内部会对批量查询做并行化处理，充分利用 CPU 的 SIMD 指令集和 GPU 的并行计算能力。

---

## 距离矩阵计算

`faiss.pairwise_distances()` 计算两组向量之间的所有距离——不需要创建索引：

```python
# 计算两组向量之间的 L2 距离矩阵
a = np.random.rand(100, d).astype('float32')
b = np.random.rand(50, d).astype('float32')

distances = faiss.pairwise_distances(a, b)
# distances.shape = (100, 50)
# distances[i][j] = L2 距离(a[i], b[j])
```

---

## knn 搜索：不创建索引直接搜索

`faiss.knn()` 提供了不创建索引直接搜索的能力——适合一次性搜索场景：

```python
# 不创建索引，直接搜索
database = np.random.rand(100000, d).astype('float32')
query = np.random.rand(10, d).astype('float32')

distances, indices = faiss.knn(query, database, k=5)
# 等价于：
# index = faiss.IndexFlatL2(d)
# index.add(database)
# distances, indices = index.search(query, k=5)
```

---

## 常见误区：每次搜索只传一个查询向量

很多初学者习惯逐条搜索——循环调用 `search()`。这在数据量小时没问题，但在高并发场景下性能很差。正确的做法是**攒够一批查询向量再搜索**——比如把 100 个用户的搜索请求攒在一起，一次 `search()` 完成。

---

## 小结

这一节我们聊了 FAISS 的批量操作：批量搜索比逐条搜索快 10~100 倍，`pairwise_distances()` 计算距离矩阵，`knn()` 不创建索引直接搜索。核心原则是"能批量就批量"——充分利用 FAISS 的并行计算能力。下一节我们聊索引的序列化与持久化。
