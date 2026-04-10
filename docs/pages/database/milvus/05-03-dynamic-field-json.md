# 5.3 动态字段与 JSON 过滤

> **动态字段让 Schema 更灵活——但灵活的代价是过滤性能**

---

## 这一节在讲什么？

在第 2 章我们简单提到了动态字段——启用 `enable_dynamic_field=True` 后，你可以插入 Schema 之外的字段。这一节我们要深入动态字段的工作原理、JSON 过滤的语法、JSON 索引的用法，以及动态字段 vs 静态标量字段的性能对比。

---

## 动态字段的工作原理

当你在创建 Collection 时启用动态字段，Milvus 会在内部创建一个隐藏的 `$meta` 字段（JSON 类型），用来存储所有 Schema 之外的字段：

```python
from pymilvus import FieldSchema, CollectionSchema, DataType, MilvusClient

client = MilvusClient(uri="http://localhost:19530")

# Schema 只定义了核心字段
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
]

# 启用动态字段
schema = CollectionSchema(fields=fields, enable_dynamic_field=True)
client.create_collection(collection_name="documents", schema=schema)

# 插入时传入 Schema 之外的字段——自动存入 $meta
client.insert(
    collection_name="documents",
    data=[
        {
            "content": "AI breakthrough",
            "embedding": [0.1] * 768,
            # 以下字段不在 Schema 中
            "author": "Alice",
            "year": 2024,
            "tags": ["AI", "ML"],
            "is_published": True,
        }
    ]
)
```

在 Milvus 内部，这条数据的存储结构是：

```
id: 1 (自动生成)
content: "AI breakthrough"
embedding: [0.1, 0.1, ..., 0.1]
$meta: {"author": "Alice", "year": 2024, "tags": ["AI", "ML"], "is_published": True}
```

---

## JSON 过滤表达式

动态字段的过滤语法跟静态标量字段一样——直接用字段名：

```python
# 动态字段的等值过滤
results = client.search(
    collection_name="documents",
    data=[[0.1] * 768],
    limit=5,
    filter='author == "Alice"'
)

# 动态字段的范围过滤
results = client.search(
    collection_name="documents",
    data=[[0.1] * 768],
    limit=5,
    filter='year >= 2024'
)

# 动态字段的布尔过滤
results = client.search(
    collection_name="documents",
    data=[[0.1] * 768],
    limit=5,
    filter='is_published == true'
)

# 动态字段的 contains 过滤（数组类型）
results = client.search(
    collection_name="documents",
    data=[[0.1] * 768],
    limit=5,
    filter='tags contains "AI"'
)

# 组合过滤
results = client.search(
    collection_name="documents",
    data=[[0.1] * 768],
    limit=5,
    filter='author == "Alice" and year >= 2024 and tags contains "AI"'
)
```

### 静态 JSON 字段的过滤

如果你在 Schema 中显式定义了 JSON 字段（而不是动态字段），过滤时需要用 `[]` 语法：

```python
# Schema 中定义了 metadata 字段（JSON 类型）
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
    FieldSchema(name="metadata", dtype=DataType.JSON),  # 静态 JSON 字段
]

# 过滤时用 [] 语法
results = client.search(
    collection_name="documents",
    data=[[0.1] * 768],
    limit=5,
    filter='metadata["author"] == "Alice" and metadata["year"] >= 2024'
)
```

注意动态字段和静态 JSON 字段的过滤语法差异——动态字段直接用字段名，静态 JSON 字段需要用 `[]` 路径。这是因为动态字段的每个键在逻辑上是独立的字段，而静态 JSON 字段是一个整体，需要用路径访问其中的键。

---

## JSON 索引（Milvus 2.5+）

默认情况下，JSON 字段的过滤是全扫描的——Milvus 需要逐条解析 JSON 数据才能判断是否满足条件。Milvus 2.5+ 支持为 JSON 字段创建 INVERTED 索引，可以显著提升过滤性能：

```python
# 为 JSON 字段创建索引
index_params = client.prepare_index_params()

# 向量索引
index_params.add_index(
    field_name="embedding",
    index_type="HNSW",
    metric_type="COSINE",
    params={"M": 16, "efConstruction": 256}
)

# JSON 字段的 INVERTED 索引
index_params.add_index(
    field_name="metadata",
    index_type="INVERTED",
    index_name="idx_metadata"
)

client.create_index(collection_name="documents", index_params=index_params)
```

创建 JSON 索引后，Milvus 会自动为 JSON 字段中的所有键建立倒排索引——`metadata["author"]`、`metadata["year"]` 等都可以走索引加速。

---

## 动态字段 vs 静态标量字段

| 维度 | 静态标量字段 | 动态字段（$meta） |
|------|------------|-----------------|
| Schema | 必须预定义 | 不需要预定义 |
| 类型安全 | 严格（INT64 不能存字符串） | 灵活（任何类型都行） |
| 过滤性能 | 快（有独立索引） | 慢（需要解析 JSON） |
| JSON 索引 | 不需要 | Milvus 2.5+ 支持 INVERTED |
| 适用场景 | 高频过滤字段 | 低频、不确定的 metadata |

比如，下面的对比展示了两种方式在相同场景下的性能差异，由于静态标量字段有独立的存储和索引，过滤时不需要解析 JSON，所以性能更好：

```python
# 方案1：高频过滤字段用静态标量字段（推荐）
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=64),  # 高频过滤
    FieldSchema(name="year", dtype=DataType.INT32),                       # 高频过滤
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
    FieldSchema(name="metadata", dtype=DataType.JSON),                    # 低频字段
]
# 过滤：filter='category == "tech" and year >= 2024'  → 快！

# 方案2：所有字段都放 JSON（不推荐）
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
    FieldSchema(name="metadata", dtype=DataType.JSON),  # category/year 也在这里
]
# 过滤：filter='metadata["category"] == "tech" and metadata["year"] >= 2024'  → 慢！
```

---

## 常见误区：把所有字段都设成动态字段

有些同学觉得动态字段很方便——不需要预定义 Schema，想加什么字段就加什么。但动态字段的过滤性能显著低于静态标量字段——因为动态字段存储在 JSON 中，过滤时需要解析 JSON，而静态标量字段可以直接走索引。如果你的 Collection 有 5 个以上需要频繁过滤的字段，应该把它们定义为静态标量字段，而不是全部依赖动态字段。

---

## 小结

这一节我们深入了动态字段和 JSON 过滤：动态字段存储在隐藏的 `$meta` JSON 字段中，过滤语法跟静态字段一样；静态 JSON 字段需要用 `[]` 路径语法；Milvus 2.5+ 支持为 JSON 字段创建 INVERTED 索引加速过滤。核心原则是"高频过滤字段用静态标量字段，低频灵活字段用 JSON 或动态字段"。下一节我们聊数据一致性与持久化。
