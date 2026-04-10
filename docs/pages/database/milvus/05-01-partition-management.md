# 5.1 Partition 分区管理

> **Partition 不是索引——它是数据隔离和搜索范围缩减的利器**

---

## 这一节在讲什么？

在第 3 章我们简单提到了 Partition——插入数据时可以指定分区，搜索时可以只搜索特定分区。但 Partition 的价值远不止于此——它是 Milvus 多租户隔离的基础，也是大规模数据管理的关键工具。这一节我们要深入 Partition 的两种模式（手动分区和 Partition Key 自动分区）、分区数量的规划、以及 Partition 与多租户的关系。

---

## Partition 的作用

Partition 把 Collection 按某个维度物理切分成多个分区，每个分区独立存储数据。搜索时可以指定只搜索某些分区，从而减少扫描的数据量。

```
没有 Partition：
  Collection "documents"（100 万条）
  ┌──────────────────────────────────────┐
  │  所有数据混在一起                      │
  │  搜索时必须扫描全部 100 万条            │
  └──────────────────────────────────────┘

有 Partition：
  Collection "documents"（100 万条）
  ┌────────────┐ ┌────────────┐ ┌────────────┐
  │ Partition  │ │ Partition  │ │ Partition  │
  │  "tech"    │ │  "science" │ │  "art"     │
  │  40 万条    │ │  35 万条    │ │  25 万条    │
  └────────────┘ └────────────┘ └────────────┘
  搜索 tech 分区 → 只扫描 40 万条，速度提升 2.5 倍
```

---

## 手动分区

手动分区需要你显式创建分区、插入时指定分区、搜索时指定分区：

```python
from pymilvus import MilvusClient

client = MilvusClient(uri="http://localhost:19530")

# 创建 Collection
client.create_collection(
    collection_name="documents",
    dimension=768,
    metric_type="COSINE"
)

# 创建分区
client.create_partition(collection_name="documents", partition_name="tech")
client.create_partition(collection_name="documents", partition_name="science")
client.create_partition(collection_name="documents", partition_name="art")

# 插入数据时指定分区
client.insert(
    collection_name="documents",
    data=tech_documents,
    partition_name="tech"
)

# 搜索时只搜索特定分区
results = client.search(
    collection_name="documents",
    data=[[0.1] * 768],
    limit=5,
    partition_names=["tech"]  # 只搜索 tech 分区
)

# 搜索多个分区
results = client.search(
    collection_name="documents",
    data=[[0.1] * 768],
    limit=5,
    partition_names=["tech", "science"]  # 搜索 tech 和 science 分区
)

# 删除分区（分区内的数据也会被删除）
client.drop_partition(collection_name="documents", partition_name="art")
```

手动分区的缺点是**应用层需要知道分区名**——你的代码需要判断每条数据应该进入哪个分区，搜索时也需要指定正确的分区名。这在分区数量多或者分区键值动态变化时很麻烦。

---

## Partition Key：自动分区路由

Milvus 2.2.9+ 提供了 Partition Key 功能——建表时指定某个字段为 Partition Key，Milvus 根据字段值的哈希自动路由到对应分区：

```python
from pymilvus import FieldSchema, CollectionSchema, DataType, MilvusClient

client = MilvusClient(uri="http://localhost:19530")

# 定义 Schema——category 字段作为 Partition Key
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=64, is_partition_key=True),
    FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
]
schema = CollectionSchema(fields=fields, num_partitions=64)  # 指定分区数量

client.create_collection(collection_name="documents", schema=schema)

# 插入时不需要指定分区——Milvus 根据 category 自动路由
client.insert(
    collection_name="documents",
    data=[
        {"category": "tech", "content": "AI news", "embedding": [0.1] * 768},
        {"category": "science", "content": "Physics paper", "embedding": [0.2] * 768},
    ]
)

# 搜索时按 category 过滤——Milvus 自动只搜索对应分区
results = client.search(
    collection_name="documents",
    data=[[0.1] * 768],
    limit=5,
    filter='category == "tech"'  # 自动路由到 tech 对应的分区
)
```

Partition Key 的核心优势是**零代码侵入**——你的应用代码不需要知道分区的存在，Milvus 自动处理路由。当你按 Partition Key 字段过滤时，Milvus 自动把搜索请求路由到对应分区，跳过不相关的分区。

### num_partitions 参数

`num_partitions` 指定自动创建的分区数量（默认 64）。Milvus 会根据 Partition Key 字段值的哈希把数据分配到不同分区——不是每个值一个分区，而是多个值共享一个分区。

比如，你有 10 个 category 值（tech/science/art/...），但 `num_partitions=64`——Milvus 会创建 64 个分区，10 个 category 值通过哈希映射到这 64 个分区中。这样设计的好处是：当 category 值增多时不需要创建新分区，坏处是一个分区可能包含多个 category 值的数据。

---

## 分区数量规划

分区数量不是越多越好——每个分区都有元数据管理开销，分区太多会增加协调器的负担。

| 场景 | 建议分区数 | 理由 |
|------|----------|------|
| 按分类分区（< 20 个分类） | 16~64 | 分类少，分区不需要太多 |
| 按时间分区（按月） | 12~36 | 一年 12 个月，保留 1~3 年 |
| 多租户（< 1000 租户） | 64~256 | 用 Partition Key + 哈希映射 |
| 多租户（> 1000 租户） | 256~512 | 更细粒度的哈希映射 |

### 常见误区：分区数 = 分类数

很多同学认为"有多少个分类就创建多少个分区"——比如有 100 个 category 就创建 100 个分区。这在分类数量少时没问题，但分类数量多时（比如有 1000 个城市名），创建 1000 个分区会导致协调器压力过大。正确做法是用 Partition Key + 哈希，把 1000 个城市映射到 64~256 个分区中。

---

## Partition 与多租户

Partition 是 Milvus 多租户隔离的推荐方案——每个租户的数据在物理上隔离在不同的分区中，搜索时只扫描自己的分区，既保证了数据隔离又提升了搜索性能。

```python
# 多租户场景：用 Partition Key 实现租户隔离
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="tenant_id", dtype=DataType.VARCHAR, max_length=64, is_partition_key=True),
    FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
]
schema = CollectionSchema(fields=fields, num_partitions=256)

# 搜索时自动路由到租户对应的分区
results = client.search(
    collection_name="documents",
    data=[[0.1] * 768],
    limit=5,
    filter='tenant_id == "tenant_123"'  # 自动路由到 tenant_123 的分区
)
```

### 常见误区：把 Partition 当成索引

Partition 减少的是扫描范围，不是距离计算加速。搜索指定分区后，Milvus 只在对应分区的数据中做向量搜索——这确实能提升速度，但提升幅度取决于分区内的数据量。如果每个分区有 100 万条数据，搜索速度跟在一个 100 万条的 Collection 中搜索差不多。Partition 的真正价值是**数据隔离**和**管理便利**（按时间分区方便清理旧数据），而不是搜索加速。

---

## 小结

这一节我们深入了 Milvus 的 Partition 管理：手动分区适合分区数量少且固定的场景，Partition Key 自动分区适合分区键值动态变化的场景。分区数量建议 64~512，太多会增加协调器负担。Partition 是多租户隔离的推荐方案，但它的价值在于数据隔离和管理便利，而不是搜索加速。下一节我们聊 Milvus 最独特的高级特性——多向量搜索与重排序。
