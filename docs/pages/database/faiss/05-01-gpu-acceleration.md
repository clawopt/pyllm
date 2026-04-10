# 5.1 GPU 加速：FAISS 的性能杀手锏

> **GPU 让向量搜索快 5~20 倍——但数据量小时反而更慢**

---

## 这一节在讲什么？

FAISS 是目前唯一原生支持 GPU 加速的向量搜索库——pgvector 不支持，Chroma 不支持，Milvus 只支持有限的 GPU 索引。GPU 的大规模并行计算能力可以让向量搜索的吞吐量提升 5~20 倍，但 GPU 加速不是万能的——数据量小时 CPU↔GPU 数据传输的开销可能超过计算加速。这一节我们要聊 FAISS GPU 的使用方法、CPU↔GPU 迁移、多 GPU 支持，以及性能对比。

---

## GPU Index 的使用

FAISS 的 GPU Index 跟 CPU Index 的接口完全一致——你只需要把 CPU Index 转换为 GPU Index：

```python
import faiss
import numpy as np

d = 128
n = 1000000

vectors = np.random.rand(n, d).astype('float32')

# 方法1：直接创建 GPU Index
res = faiss.StandardGpuResources()  # GPU 资源管理器
index = faiss.GpuIndexFlatL2(res, d)

index.add(vectors)

query = np.random.rand(1, d).astype('float32')
distances, indices = index.search(query, k=5)

# 方法2：CPU Index → GPU Index
cpu_index = faiss.IndexFlatL2(d)
cpu_index.add(vectors)

res = faiss.StandardGpuResources()
gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)  # 0 是 GPU 编号

distances, indices = gpu_index.search(query, k=5)

# GPU Index → CPU Index（用于序列化）
cpu_index_back = faiss.index_gpu_to_cpu(gpu_index)
faiss.write_index(cpu_index_back, "index.faiss")
```

---

## CPU ↔ GPU 迁移

FAISS 提供了两个核心函数来实现 CPU 和 GPU 之间的 Index 迁移：

- `faiss.index_cpu_to_gpu(res, gpu_id, cpu_index)`：把 CPU Index 复制到 GPU
- `faiss.index_gpu_to_cpu(gpu_index)`：把 GPU Index 复制回 CPU

为什么需要迁移？因为 FAISS 的序列化（`write_index`）只支持 CPU Index——如果你想保存 GPU Index 到磁盘，需要先转回 CPU。

---

## 多 GPU 支持

FAISS 支持多 GPU 并行搜索——两种模式：

```python
# 模式1：IndexReplicas——副本并行（数据复制到每个 GPU）
res0 = faiss.StandardGpuResources()
res1 = faiss.StandardGpuResources()

cpu_index = faiss.IndexFlatL2(d)
cpu_index.add(vectors)

# 把索引复制到两个 GPU
gpu_index = faiss.index_cpu_to_gpu_multiple(
    [res0, res1], [0, 1], cpu_index
)

# 搜索时两个 GPU 并行处理不同的查询
distances, indices = gpu_index.search(queries, k=5)

# 模式2：IndexShards——分片并行（数据分片到不同 GPU）
# 每个GPU只存一部分数据，搜索时并行搜索再合并
```

---

## GPU vs CPU 性能对比

| 场景 | CPU 延迟 | GPU 延迟 | 加速比 |
|------|---------|---------|--------|
| 100 万 × 128d，Flat，单查询 | ~5 ms | ~3 ms | 1.7x |
| 100 万 × 128d，Flat，批量 1000 查询 | ~5000 ms | ~50 ms | 100x |
| 1000 万 × 128d，IVFPQ，单查询 | ~2 ms | ~1 ms | 2x |
| 1000 万 × 128d，IVFPQ，批量 1000 查询 | ~2000 ms | ~100 ms | 20x |

GPU 的优势在**批量搜索**时最明显——因为 GPU 可以同时处理数千个查询。单查询时 GPU 的加速比不大，因为数据传输的开销占比较大。

---

## 常见误区：数据量小时 GPU 反而更慢

当你只有几千条向量时，GPU 搜索可能比 CPU 更慢——因为 CPU↔GPU 数据传输的延迟（约 1~5 ms）超过了计算加速的收益。GPU 加速只在数据量大（> 100 万条）或批量搜索（> 100 个查询同时）时才有意义。

---

## 小结

这一节我们聊了 FAISS 的 GPU 加速：`GpuIndexFlatL2`/`GpuIndexIVFPQ` 等GPU Index 跟 CPU Index 接口一致，`index_cpu_to_gpu`/`index_gpu_to_cpu` 实现迁移，多 GPU 支持 IndexReplicas 和 IndexShards。GPU 的优势在批量搜索时最明显，数据量小时反而可能更慢。下一节我们聊 FAISS 最独特的能力——积木式索引组合。
