# 7.1 FAISS 作为向量数据库的底层引擎

> **当你用 Milvus 创建 IVF_PQ 索引时，底层调用的就是 FAISS**

---

## 这一节在讲什么？

在 Milvus 教程中，我们用 `create_index(index_type="IVF_PQ")` 创建索引，看起来很简单。但你有没有想过，Milvus 的索引是怎么实现的？答案就是 FAISS——Milvus 的 IVFFlat、IVF_PQ、IVF_SQ8 等索引，底层都是调用 FAISS 的 C++ 实现。理解这层关系，你就能理解为什么 Milvus 的索引参数跟 FAISS 的参数名相同——因为它们就是同一套参数。

---

## Milvus 与 FAISS 的关系

```
Milvus 的索引架构：

  用户 API
  client.create_index(index_type="IVF_PQ", params={"nlist": 1024, "m": 48})
     │
     ▼
  Milvus IndexCoord → IndexNode
     │
     ▼
  FAISS C++ 库
  faiss::IndexIVFPQ(quantizer, d, nlist, m, nbits)
     │
     ▼
  索引文件 → 对象存储
```

当你调用 Milvus 的 `create_index()` 时，Milvus 的 IndexNode 会：
1. 从对象存储读取原始向量数据
2. 调用 FAISS 的 C++ API 构建索引
3. 把索引文件写回对象存储
4. QueryNode 加载索引到内存后，搜索也是调用 FAISS

---

## 参数映射

| Milvus 参数 | FAISS 参数 | 含义 |
|------------|-----------|------|
| nlist | nlist | IVF 聚类数 |
| nprobe | nprobe | 搜索时扫描的聚类数 |
| M | M | HNSW 连接数 / PQ 子空间数 |
| efConstruction | efConstruction | HNSW 构建宽度 |
| ef | efSearch | HNSW 搜索宽度 |
| m | m | PQ 子空间数 |
| nbits | nbits | PQ 编码位数 |

参数名几乎完全一致——因为 Milvus 就是 FAISS 的薄封装。

---

## 理解这层关系的价值

当你调优 Milvus 的索引参数时，实际上是在调 FAISS 的参数。如果你理解了 FAISS 的参数含义和调优方法，就能更好地调优 Milvus。比如，当你发现 Milvus 的 HNSW 搜索召回率不够时，你知道需要增大 ef——因为 FAISS 的 efSearch 控制的就是搜索宽度。

---

## 小结

这一节我们揭示了 Milvus 与 FAISS 的关系：Milvus 的 IVFFlat、IVF_PQ 等索引底层就是调用 FAISS 的 C++ 实现。理解这层关系，你就能理解为什么 Milvus 的索引参数跟 FAISS 的参数名相同——因为它们就是同一套参数。下一节我们聊 FAISS vs 向量数据库的选型决策。
