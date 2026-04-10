# 5.4 索引的序列化与持久化

> **FAISS 不自动持久化——进程退出数据就没了，你需要自己保存**

---

## 这一节在讲什么？

在 Milvus 和 pgvector 中，数据是自动持久化的——你插入数据后，即使服务器重启数据也不会丢。FAISS 不负责持久化——所有数据都在内存中，进程退出就没了。你需要自己把 Index 序列化到磁盘，下次启动时再加载。这一节我们聊 FAISS 的序列化方法、增量更新的注意事项，以及常见的持久化策略。

---

## 序列化方法

### 文件序列化

```python
import faiss
import numpy as np

d = 768
index = faiss.IndexFlatL2(d)
index.add(np.random.rand(10000, d).astype('float32'))

# 保存到文件
faiss.write_index(index, "my_index.faiss")

# 从文件加载
loaded_index = faiss.read_index("my_index.faiss")

# 验证
query = np.random.rand(1, d).astype('float32')
distances, indices = loaded_index.search(query, k=5)
print(f"加载后的向量数: {loaded_index.ntotal}")
```

### 内存序列化

```python
# 序列化为字节——适合存入数据库或对象存储
bytes_data = faiss.serialize_index(index)

# 从字节反序列化
loaded_index = faiss.deserialize_index(bytes_data)
```

内存序列化适合把索引存到 S3/OSS、Redis 或数据库的 BLOB 字段中。

---

## GPU Index 的序列化

GPU Index 不能直接序列化——你需要先转回 CPU：

```python
# GPU Index → CPU Index → 序列化
res = faiss.StandardGpuResources()
gpu_index = faiss.GpuIndexFlatL2(res, d)
gpu_index.add(np.random.rand(10000, d).astype('float32'))

# 先转回 CPU
cpu_index = faiss.index_gpu_to_cpu(gpu_index)
faiss.write_index(cpu_index, "gpu_index.faiss")

# 加载时再转到 GPU
cpu_index = faiss.read_index("gpu_index.faiss")
gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
```

---

## 增量更新与重建

FAISS 的大部分索引支持 `add()` 增量添加——你可以加载已有索引，添加新向量，再保存：

```python
# 加载已有索引
index = faiss.read_index("my_index.faiss")
print(f"原有向量数: {index.ntotal}")

# 增量添加
new_vectors = np.random.rand(1000, d).astype('float32')
index.add(new_vectors)
print(f"添加后向量数: {index.ntotal}")

# 保存更新后的索引
faiss.write_index(index, "my_index.faiss")
```

但 IVF 索引的增量添加有一个问题——聚类中心不会自动更新。如果你添加了大量新数据，聚类中心可能不再准确，需要重建索引。

---

## 常见误区：序列化后修改了原始数据，期望索引自动更新

FAISS 的索引和数据是独立的——序列化保存的是索引的内部状态（包括向量数据和索引结构），不是对原始数据的引用。如果你修改了原始 numpy 数组，索引不会自动更新——因为索引在 `add()` 时已经复制了向量数据。

---

## 小结

这一节我们聊了 FAISS 的序列化与持久化：`write_index`/`read_index` 文件序列化，`serialize_index`/`deserialize_index` 内存序列化，GPU Index 需要先转回 CPU 才能序列化。FAISS 不自动持久化，你需要自己管理保存和加载。下一节开始我们进入第 6 章，用 FAISS 搭建 RAG 系统。
