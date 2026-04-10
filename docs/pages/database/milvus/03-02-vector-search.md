# 3.2 向量搜索：search() 详解

> **search() 是你用 Milvus 做的最多的操作——理解它的每个参数是高效检索的前提**

---

## 这一节在讲什么？

向量搜索是 Milvus 的核心功能——你给它一个查询向量，它返回最相似的 K 条数据。听起来很简单，但 `search()` 方法的参数很多，每个参数都影响搜索的行为和性能。这一节我们要把 `search()` 的每个参数都讲清楚，包括搜索参数与索引类型的对应关系、批量搜索、以及搜索结果的解析。

---

## search() 的完整参数

```python
results = client.search(
    collection_name="documents",   # 搜索哪个 Collection
    data=[[0.1] * 768],           # 查询向量列表（支持批量）
    limit=5,                       # 返回 Top-K 结果
    output_fields=["content", "source"],  # 返回哪些字段
    filter='category == "tech"',   # 标量过滤表达式
    search_params={                # 搜索参数（与索引类型相关）
        "metric_type": "COSINE",
        "params": {"ef": 100}      # HNSW 的搜索参数
    }
)
```

让我们逐个解析这些参数：

### data：查询向量

`data` 参数接受一个向量列表——你可以一次传入多个查询向量，Milvus 会并行搜索，返回每个查询向量的 Top-K 结果。这比循环调用 `search()` 高效得多，因为 Milvus 内部会批量处理。

```python
# 单个查询向量
results = client.search(
    collection_name="documents",
    data=[[0.1] * 768],    # 一个查询向量
    limit=5
)
# results[0] → 第一个查询向量的结果列表

# 批量查询向量
query_vectors = [[0.1] * 768, [0.2] * 768, [0.3] * 768]
results = client.search(
    collection_name="documents",
    data=query_vectors,     # 三个查询向量
    limit=5
)
# results[0] → 第一个查询向量的结果
# results[1] → 第二个查询向量的结果
# results[2] → 第三个查询向量的结果
```

### limit：返回结果数

`limit` 指定每个查询向量返回多少条结果（Top-K）。常见的值是 5~20——RAG 场景通常取 5 条，推荐场景可能取 20~100 条。

注意：`limit` 的最大值受 Collection 配置限制，默认上限是 16384。如果你需要返回更多结果，需要在创建 Collection 时调整 `topk` 配置。

### output_fields：返回哪些字段

`output_fields` 指定搜索结果中返回哪些标量字段。默认只返回 ID 和距离，不返回其他字段。如果你需要获取文档内容、来源等信息，必须显式指定：

```python
results = client.search(
    collection_name="documents",
    data=[[0.1] * 768],
    limit=5,
    output_fields=["content", "source", "category"]
)

for hit in results[0]:
    print(f"ID: {hit['id']}, Distance: {hit['distance']:.4f}")
    print(f"  Content: {hit['entity']['content'][:50]}...")
    print(f"  Source: {hit['entity']['source']}")
    print(f"  Category: {hit['entity']['category']}")
```

### filter：标量过滤表达式

`filter` 参数用于在向量搜索的同时进行标量过滤——相当于 pgvector 的 `WHERE` 条件。下一节我们会详细讲过滤表达式，这里先看一个简单的例子：

```python
# 只搜索分类为 tech 的文档
results = client.search(
    collection_name="documents",
    data=[[0.1] * 768],
    limit=5,
    filter='category == "tech"'
)

# 多条件过滤
results = client.search(
    collection_name="documents",
    data=[[0.1] * 768],
    limit=5,
    filter='category == "tech" and year >= 2024'
)
```

### search_params：搜索参数

`search_params` 是 `search()` 中最复杂的参数——它指定了距离度量和索引相关的搜索参数。不同的索引类型有不同的搜索参数：

| 索引类型 | 搜索参数 | 说明 |
|---------|---------|------|
| FLAT | 无特殊参数 | 暴力搜索，无需调参 |
| IVFFlat | `nprobe` | 搜索时扫描的聚类区域数（默认 8） |
| HNSW | `ef` | 搜索时的搜索宽度（默认 64） |
| IVF_PQ | `nprobe` | 搜索时扫描的聚类区域数 |
| IVF_SQ8 | `nprobe` | 同 IVFFlat |
| SCANN | `nprobe` | 搜索时扫描的聚类区域数 |
| DiskANN | `search_list` | 搜索时的候选列表大小 |

```python
# HNSW 索引的搜索参数
results = client.search(
    collection_name="documents",
    data=[[0.1] * 768],
    limit=5,
    search_params={
        "metric_type": "COSINE",
        "params": {"ef": 100}    # ef 越大，召回率越高，速度越慢
    }
)

# IVFFlat 索引的搜索参数
results = client.search(
    collection_name="documents",
    data=[[0.1] * 768],
    limit=5,
    search_params={
        "metric_type": "COSINE",
        "params": {"nprobe": 16}  # nprobe 越大，召回率越高，速度越慢
    }
)
```

这些搜索参数跟 pgvector 的 `hnsw.ef_search` 和 `ivfflat.probes` 是同一个概念——它们控制搜索精度和速度的权衡。值越大，搜索越精确但越慢；值越小，搜索越快但可能漏掉一些结果。

---

## 搜索结果解析

`search()` 的返回值是一个嵌套列表——外层列表对应每个查询向量，内层列表对应每个查询向量的 Top-K 结果：

```python
results = client.search(
    collection_name="documents",
    data=[[0.1] * 768, [0.2] * 768],  # 2 个查询向量
    limit=5,
    output_fields=["content", "category"]
)

# results 的结构：
# [
#   [hit1_for_query1, hit2_for_query1, ...],  # 第 1 个查询向量的结果
#   [hit1_for_query2, hit2_for_query2, ...],  # 第 2 个查询向量的结果
# ]

# 遍历第一个查询向量的结果
for hit in results[0]:
    print(f"ID: {hit['id']}")
    print(f"Distance: {hit['distance']:.4f}")  # 距离值，越小越相似
    print(f"Content: {hit['entity']['content'][:50]}")
    print(f"Category: {hit['entity']['category']}")
    print("---")
```

每个 hit 是一个字典，包含：
- `id`：数据的主键
- `distance`：距离值（越小越相似，无论哪种度量）
- `entity`：返回的标量字段（由 `output_fields` 指定）

### 距离值的含义

距离值的含义取决于距离度量：

| 度量类型 | 距离值范围 | 完全相同 | 完全不相关 |
|---------|----------|---------|----------|
| L2 | [0, +∞) | 0 | 越大越不相似 |
| Cosine | [0, 2] | 0 | 2 |
| IP | (-∞, +∞) | 越小越相似（取了负值） | 越大越不相似 |

---

## 使用 ORM API 进行搜索

如果你使用 ORM API，搜索的写法略有不同——你需要指定 `anns_field`（搜索哪个向量字段）和 `param`（搜索参数）：

```python
from pymilvus import Collection

collection = Collection("documents")

# 确保数据已加载
collection.load()

# 搜索
results = collection.search(
    data=[[0.1] * 768],           # 查询向量
    anns_field="embedding",        # 搜索的向量字段名
    param={"metric_type": "COSINE", "params": {"ef": 100}},
    limit=5,
    expr='category == "tech"',     # 过滤表达式
    output_fields=["content", "source"]
)

# 遍历结果
for hits in results:
    for hit in hits:
        print(f"ID: {hit.id}, Distance: {hit.distance:.4f}")
        print(f"Content: {hit.entity.get('content')[:50]}")
```

ORM API 的搜索结果对象跟 MilvusClient 不同——它使用属性访问（`hit.id`、`hit.distance`）而不是字典访问（`hit['id']`、`hit['distance']`）。两种 API 的功能完全一样，只是接口风格不同。

---

## 常见误区：不 load 就搜索

Milvus 的搜索是在 QueryNode 的内存中执行的——数据和索引必须先加载到 QueryNode 内存才能搜索。如果你创建了索引但没有 `load()`，搜索会报错或退化为暴力搜索。

```python
# 创建索引
client.create_index(
    collection_name="documents",
    field_name="embedding",
    index_params={"index_type": "HNSW", "metric_type": "COSINE", "params": {"M": 16, "efConstruction": 256}}
)

# 必须先 load！
client.load_collection("documents")

# 现在可以搜索了
results = client.search(
    collection_name="documents",
    data=[[0.1] * 768],
    limit=5
)
```

MilvusClient 的 `create_collection()` 快速创建方式会自动 load，但如果你手动创建索引，就必须手动 load。这是初学者最常犯的错误——建了索引但忘记 load，然后疑惑为什么搜索还是暴力扫描。

---

## 小结

这一节我们详细解析了 `search()` 的每个参数：`data` 支持批量查询、`limit` 控制 Top-K、`output_fields` 指定返回字段、`filter` 做标量过滤、`search_params` 控制搜索精度和速度的权衡。不同索引类型有不同的搜索参数（HNSW 用 ef、IVFFlat 用 nprobe），这些参数跟 pgvector 的 ef_search/probes 是同一个概念。下一节我们要深入标量过滤——Milvus 的 expr 表达式语法和过滤性能优化。
