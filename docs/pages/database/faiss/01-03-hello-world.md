# 1.3 Hello World：5 分钟跑通第一个向量搜索

> **5 行代码，从零到向量搜索——FAISS 的简洁是 Milvus 和 pgvector 都比不了的**

---

## 这一节在讲什么？

前面两节我们聊了 FAISS 的定位和安装，现在直接上手写代码。FAISS 的 Hello World 非常简洁——创建索引、添加向量、搜索结果，5 行代码就能跑通。但简洁的背后有几个容易踩的坑：向量必须是 float32、必须是 numpy 数组、维度必须一致。这一节我们逐行解析 Hello World，帮你建立对 FAISS API 的第一印象。

---

## 最简示例

```python
import faiss
import numpy as np

# 第1步：定义向量维度
d = 768

# 第2步：创建 L2 距离的暴力搜索索引
index = faiss.IndexFlatL2(d)

# 第3步：生成随机向量并添加到索引
vectors = np.random.rand(10000, d).astype('float32')
index.add(vectors)

# 第4步：搜索查询向量的 5 个最近邻
query = np.random.rand(1, d).astype('float32')
distances, indices = index.search(query, k=5)

# 第5步：解读结果
print(f"最近邻的索引: {indices}")
print(f"对应的 L2 距离: {distances}")
```

输出类似：

```
最近邻的索引: [[4823 7156  291 9438 5612]]
对应的 L2 距离: [[58.32 59.14 60.01 60.45 60.89]]
```

让我们逐行解析：

### 第1步：定义维度 d

`d = 768` 是向量的维度——768 是大多数文本 Embedding 模型（如 all-MiniLM-L6-v2、BGE-base）的输出维度。如果你用 OpenAI 的模型，维度是 1536。

### 第2步：创建索引

`faiss.IndexFlatL2(d)` 创建了一个基于 L2 距离的暴力搜索索引。`Flat` 表示不做任何索引优化——搜索时逐条计算距离，召回率 100% 但速度最慢。`L2` 表示使用欧氏距离。

### 第3步：添加向量

`index.add(vectors)` 把向量添加到索引中。这里有几个关键点：

- **向量必须是 float32 类型**——FAISS 内部用 float32 计算距离，如果你传入 float64，FAISS 会报错或产生错误结果
- **向量必须是 2D numpy 数组**——形状为 (n, d)，n 是向量数量，d 是维度
- **向量必须是连续内存**——numpy 数组默认是连续的，但如果你对数组做了切片或转置，可能不是连续的

```python
# ✅ 正确：float32 类型
vectors = np.random.rand(10000, d).astype('float32')

# ❌ 错误：float64 类型——FAISS 会报错
vectors = np.random.rand(10000, d)  # 默认 float64

# ❌ 错误：Python list——FAISS 需要 numpy 数组
vectors = [[0.1, 0.2, 0.3]] * 10000
```

### 第4步：搜索

`index.search(query, k=5)` 搜索查询向量的 5 个最近邻。返回两个数组：

- **distances**：距离值，形状为 (n_queries, k)。L2 距离的值 ≥ 0，越小越相似
- **indices**：最近邻的索引，形状为 (n_queries, k)。索引对应 add 时的向量顺序

### 第5步：解读结果

`indices[0]` 是第一个查询向量的 5 个最近邻的索引——`[4823, 7156, 291, 9438, 5612]` 表示第 4823、7156、291、9438、5612 条向量与查询向量最相似。`distances[0]` 是对应的 L2 距离值。

---

## 与 Milvus/Chroma 的对比

同样的向量搜索，不同工具的代码量对比：

```python
# FAISS：5 行代码
import faiss, numpy as np
index = faiss.IndexFlatL2(768)
index.add(np.random.rand(10000, 768).astype('float32'))
D, I = index.search(np.random.rand(1, 768).astype('float32'), k=5)

# Chroma：6 行代码
import chromadb
client = chromadb.Client()
collection = client.create_collection("demo")
collection.add(embeddings=np.random.rand(10000, 768).tolist(), ids=[str(i) for i in range(10000)])
results = collection.query(query_embeddings=np.random.rand(1, 768).tolist(), n_results=5)

# Milvus：8 行代码
from pymilvus import MilvusClient
client = MilvusClient("./demo.db")
client.create_collection("demo", dimension=768)
client.insert("demo", data=[{"id": i, "vector": np.random.rand(768).tolist()} for i in range(10000)])
results = client.search("demo", data=[np.random.rand(768).tolist()], limit=5)
```

FAISS 最简洁，但代价是你需要自己管理数据——FAISS 的 `indices` 只是整数编号，你需要自己维护一个映射表把编号映射回原始文档内容。

---

## 常见误区：向量类型不对导致搜索结果错误

这是 FAISS 初学者最常遇到的问题——numpy 默认生成 float64 数组，而 FAISS 需要 float32。如果你忘记 `.astype('float32')`，FAISS 可能不会报错（取决于版本），但搜索结果会是错的——因为 float64 的字节布局跟 float32 不同，FAISS 会把 float64 的字节重新解释为两个 float32，导致距离计算完全错误。

```python
# ❌ 危险：float64 可能不报错但结果错误
vectors = np.random.rand(10000, d)  # float64
index.add(vectors)  # 某些版本不报错！

# ✅ 正确：显式转为 float32
vectors = np.random.rand(10000, d).astype('float32')
index.add(vectors)
```

另一个常见误区是**向量维度不匹配**——如果你创建 `IndexFlatL2(768)` 但添加 512 维的向量，FAISS 会直接报错。确保所有向量的维度与索引的维度一致。

---

## 小结

这一节我们用 5 行代码跑通了 FAISS 的第一个向量搜索：`IndexFlatL2` 创建暴力搜索索引，`add()` 添加向量，`search()` 搜索最近邻。关键注意事项：向量必须是 float32、2D numpy 数组、维度必须一致。FAISS 的简洁是它的优势，但代价是你需要自己管理数据映射和持久化。下一节开始我们进入第 2 章，深入 FAISS 的核心概念——Index 抽象、距离度量和向量 ID。
