# 3.1 数据插入：单条、批量与 Partition

> **插入数据看似简单，但批量大小和 Partition 策略直接影响写入性能和搜索效率**

---

## 这一节在讲什么？

上一章我们定义好了 Collection 的 Schema，现在要把数据插进去了。Milvus 的 `insert()` 方法用起来很简单——传入一个字典列表就行。但在生产环境中，"能插入"和"插得快"是两回事——批量大小怎么选、Partition 怎么用、flush 的时机怎么把握，这些细节直接影响写入性能和数据的可搜索性。这一节我们从最简单的插入开始，逐步深入到批量优化和 Partition 策略。

---

## 基本插入：insert()

Milvus 的 `insert()` 方法接受一个字典列表，每个字典代表一条数据：

```python
from pymilvus import MilvusClient

client = MilvusClient(uri="http://localhost:19530")

# 单条插入
client.insert(
    collection_name="documents",
    data=[
        {
            "id": 1,
            "content": "Milvus is a distributed vector database",
            "source": "official_docs",
            "category": "tech",
            "embedding": [0.1] * 768,
            "metadata": {"author": "Zilliz", "version": "2.5"}
        }
    ]
)

# 批量插入——推荐！一次插入多条数据
documents = [
    {
        "id": i,
        "content": f"Document content {i}",
        "source": "wiki",
        "category": "tech" if i % 2 == 0 else "science",
        "embedding": [0.01 * (i % 100)] * 768,
        "metadata": {"page": i, "chapter": i // 10}
    }
    for i in range(2, 102)
]

client.insert(
    collection_name="documents",
    data=documents
)
```

`insert()` 的返回值是一个字典，包含 `insert_count`（成功插入的条数）和 `ids`（插入数据的 ID 列表）：

```python
result = client.insert(collection_name="documents", data=documents)
print(f"Inserted {result['insert_count']} records")
print(f"IDs: {result['ids'][:5]}...")  # 打印前 5 个 ID
```

---

## 批量插入的性能优化

Milvus 的写入流程是：应用 → 消息队列 → DataNode → 对象存储。每次 `insert()` 调用都会产生一次消息队列的写入和 DataNode 的消费。如果你逐条插入 10 万条数据，就会产生 10 万次消息队列交互——这非常低效。

### 批量大小的选择

```python
import numpy as np

def batch_insert(client, collection_name, all_data, batch_size=1000):
    """分批插入数据，控制每批的大小"""
    total = len(all_data)
    for i in range(0, total, batch_size):
        batch = all_data[i:i + batch_size]
        result = client.insert(collection_name=collection_name, data=batch)
        print(f"Batch {i // batch_size + 1}: inserted {result['insert_count']} records")

    # 插入完成后 flush，确保数据可搜索
    client.flush(collection_name)
    print(f"Done! Total {total} records inserted and flushed.")

# 生成 10 万条测试数据
all_data = [
    {
        "id": i,
        "content": f"Document {i}",
        "source": "test",
        "category": "tech",
        "embedding": np.random.rand(768).tolist(),
        "metadata": {"index": i}
    }
    for i in range(100000)
]

# 每批 5000 条插入
batch_insert(client, "documents", all_data, batch_size=5000)
```

批量大小建议：

| 数据规模 | 建议批量大小 | 原因 |
|---------|------------|------|
| < 1 万 | 全部一次插入 | 数据量小，一次搞定 |
| 1 万 ~ 10 万 | 1000~5000 | 平衡内存和吞吐 |
| 10 万 ~ 100 万 | 5000~10000 | 充分利用消息队列的批处理能力 |
| > 100 万 | 10000 | 配合并行写入进一步提升吞吐 |

### 常见误区：批量太大导致内存溢出

有些同学觉得批量越大越好，一次性插入 100 万条数据——但这可能导致客户端内存溢出（每条 768 维向量约 3KB，100 万条约 3GB），或者消息队列积压导致 DataNode 来不及消费。建议批量大小控制在 5000~10000 条，通过分批插入来平衡内存和吞吐。

---

## Flush：让数据可搜索

Milvus 的写入是异步的——你调用 `insert()` 后，数据先进入消息队列，DataNode 在后台消费并写入对象存储。在数据被 DataNode 处理并加载到 QueryNode 之前，这些数据是"搜不到"的。`flush()` 的作用就是强制 DataNode 立即处理缓冲区中的数据，确保数据可搜索。

```python
# 插入数据
client.insert(collection_name="documents", data=batch_data)

# flush 确保数据可搜索
client.flush(collection_name="documents")

# 现在搜索就能找到刚插入的数据了
results = client.search(
    collection_name="documents",
    data=[[0.1] * 768],
    limit=5
)
```

### Milvus Lite 自动 flush

如果你用的是 Milvus Lite（本地文件模式），`insert()` 后数据会自动 flush，不需要手动调用。但 Milvus Standalone 和 Cluster 需要手动 flush 或等待自动同步（默认间隔约 1 秒）。

### 常见误区：每次 insert 后都 flush

频繁 flush 会严重影响写入性能——每次 flush 都会触发 DataNode 的 Segment 持久化操作，产生大量小 Segment。正确的做法是：**攒够一批数据后 flush 一次**，而不是每插入几条就 flush。

```python
# 错误：每次 insert 后都 flush
for doc in documents:
    client.insert(collection_name="documents", data=[doc])
    client.flush("documents")  # ❌ 性能灾难！

# 正确：批量 insert 后 flush 一次
client.insert(collection_name="documents", data=documents)
client.flush("documents")  # ✅ 只 flush 一次
```

---

## Partition：按维度物理切分数据

Partition 是 Milvus 中一个重要的数据组织方式——它把 Collection 按某个维度（时间、分类、租户）物理切分成多个分区，搜索时可以只扫描相关分区，减少计算量。

### 手动创建 Partition

```python
# 创建分区
client.create_partition(collection_name="documents", partition_name="tech")
client.create_partition(collection_name="documents", partition_name="science")

# 插入数据时指定分区
client.insert(
    collection_name="documents",
    data=tech_documents,
    partition_name="tech"     # 数据进入 tech 分区
)

client.insert(
    collection_name="documents",
    data=science_documents,
    partition_name="science"  # 数据进入 science 分区
)

# 搜索时只搜索特定分区
results = client.search(
    collection_name="documents",
    data=[[0.1] * 768],
    limit=5,
    partition_names=["tech"]  # 只搜索 tech 分区
)
```

### Partition Key：自动分区路由

手动指定分区在数据量大时很麻烦——你需要在应用层判断每条数据应该进入哪个分区。Milvus 2.2.9+ 提供了 Partition Key 功能——建表时指定某个字段为 Partition Key，Milvus 会根据字段值自动路由到对应分区：

```python
from pymilvus import FieldSchema, CollectionSchema, DataType

fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=64, is_partition_key=True),  # 分区键
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
]
schema = CollectionSchema(fields=fields)

client.create_collection(collection_name="documents", schema=schema)

# 插入时不需要指定分区——Milvus 根据 category 自动路由
client.insert(
    collection_name="documents",
    data=[
        {"category": "tech", "embedding": [0.1] * 768},
        {"category": "science", "embedding": [0.2] * 768},
    ]
)

# 搜索时按 category 过滤——Milvus 自动只搜索对应分区
results = client.search(
    collection_name="documents",
    data=[[0.1] * 768],
    limit=5,
    filter='category == "tech"'  # 自动路由到 tech 分区
)
```

Partition Key 的好处是**零代码侵入**——你的应用代码不需要知道分区的存在，Milvus 自动处理路由。但有一个限制：Partition Key 字段的基数（不同值的数量）决定了分区数——如果 category 只有 3 个值（tech/science/art），就只有 3 个分区，粒度太粗。Milvus 会根据 `num_partitions` 参数（默认 64）自动创建指定数量的分区，然后根据哈希值把数据分配到不同分区。

### 分区数量建议

分区数不是越多越好——每个分区都有元数据管理开销，分区太多会增加协调器的负担。建议每个 Collection 的分区数在 64~512 之间。如果你有 1000 个租户需要隔离，不要创建 1000 个分区——可以用 Partition Key + 哈希，把 1000 个租户映射到 64 个分区中。

---

## 常见误区：Partition 能加速向量搜索

Partition 减少的是**扫描范围**，不是**距离计算加速**。搜索时指定分区，Milvus 只在对应分区的数据中计算距离，而不是全量扫描——这确实能提升搜索速度，但提升幅度取决于分区内的数据量。如果每个分区有 100 万条数据，搜索速度跟在一个 100 万条的 Collection 中搜索差不多。Partition 的真正价值是**数据隔离**（多租户）和**管理便利**（按时间分区方便清理旧数据），而不是搜索加速。

---

## 小结

这一节我们覆盖了 Milvus 的数据插入：基本 insert() 用法、批量插入的性能优化（建议 5000~10000 条/批）、flush 的时机（批量插入后 flush 一次，不要频繁 flush）、以及 Partition 的两种模式（手动分区和 Partition Key 自动路由）。下一节我们要深入 Milvus 最核心的操作——向量搜索 search()。
