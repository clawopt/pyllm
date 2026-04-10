# 2.1 Collection 与 Schema：Milvus 的数据模型设计

> **Collection 就是 Milvus 的"表"——但 Schema 的设计比你想的更重要**

---

## 这一节在讲什么？

在 pgvector 教程里，我们用 `CREATE TABLE` 定义表结构；在 Chroma 教程里，Collection 的 Schema 是自动推断的，你不需要手动定义。Milvus 介于两者之间——它有显式的 Schema 定义，但不是用 SQL，而是用 Python API。这一节我们要聊清楚 Milvus 的数据模型设计：Collection 是什么、Schema 怎么定义、以及 Schema 设计对性能的影响。理解这些，你才能在后续的搜索和调优中做出正确的决策。

---

## Collection：Milvus 的"表"

Collection 是 Milvus 中数据组织的基本单位——你可以把它理解成关系型数据库中的一张表。一个 Collection 包含一组同类数据，每条数据有相同的字段结构（Schema）。比如，一个文档问答系统的 Collection 可能包含这些字段：文档 ID、文档内容、来源、分类、embedding 向量、metadata。

与 Chroma 的 Collection 不同，Milvus 的 Collection 需要显式定义 Schema——你必须告诉 Milvus 这个 Collection 有哪些字段、每个字段的类型是什么、哪个字段是主键、哪个字段是向量。这看起来比 Chroma 麻烦，但显式 Schema 有一个重要的好处：**性能**。Milvus 知道每个字段的类型和位置，可以为标量字段创建独立的索引，搜索时不需要扫描整个数据集。

与 pgvector 的表相比，Milvus 的 Collection 也有不同——pgvector 的表是 PostgreSQL 的一等公民，你可以用 SQL 做任何操作（JOIN、子查询、窗口函数）；Milvus 的 Collection 是一个独立的数据容器，你只能通过 Milvus 的 API 操作它，不能做 JOIN 或子查询。

---

## Schema 定义：用 Python API 描述数据结构

Milvus 的 Schema 定义使用 `FieldSchema` 和 `CollectionSchema` 两个类。`FieldSchema` 描述单个字段，`CollectionSchema` 描述整个 Collection 的字段集合。

```python
from pymilvus import FieldSchema, CollectionSchema, DataType, MilvusClient

# 定义字段
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=64),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
    FieldSchema(name="metadata", dtype=DataType.JSON),
]

# 定义 Schema
schema = CollectionSchema(
    fields=fields,
    description="文档问答系统的文档集合",
    enable_dynamic_field=True
)

# 创建 Collection
client = MilvusClient(uri="http://localhost:19530")
client.create_collection(
    collection_name="documents",
    schema=schema
)
```

这段代码定义了一个名为 `documents` 的 Collection，包含 6 个字段。让我们逐个解析：

- **id**：主键字段，INT64 类型，自动生成（`auto_id=True`）。每条数据必须有唯一的主键
- **content**：VARCHAR 类型，最大长度 65535。存储文档内容
- **source**：VARCHAR 类型，最大长度 256。存储文档来源
- **category**：VARCHAR 类型，最大长度 64。存储文档分类，后续可以做过滤
- **embedding**：FLOAT_VECTOR 类型，维度 768。存储文档的向量表示
- **metadata**：JSON 类型。存储灵活的元数据（作者、标签、日期等）

`enable_dynamic_field=True` 表示允许插入 Schema 之外的字段——这些额外字段会自动存入一个隐藏的 `$meta` JSON 字段中。这在 metadata 字段不确定的场景下很有用，但过滤性能不如静态标量字段。

---

## MilvusClient 的快速创建方式

如果你不需要精细控制 Schema，MilvusClient 提供了一个快速创建 Collection 的方式——只需要指定维度和距离度量：

```python
from pymilvus import MilvusClient

client = MilvusClient(uri="http://localhost:19530")

# 快速创建——自动生成 id 和 vector 字段
client.create_collection(
    collection_name="quick_demo",
    dimension=768,
    metric_type="COSINE"
)
```

这行代码等价于：

```python
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=768),
]
schema = CollectionSchema(fields=fields, auto_id=True)
client.create_collection(collection_name="quick_demo", schema=schema)
```

快速创建方式适合学习和快速验证，但生产环境建议使用显式 Schema——因为你可以精确控制字段类型、长度、是否动态字段等。

---

## Schema 设计原则

Schema 的设计直接影响存储效率和查询性能。以下是几个关键原则：

**原则1：向量字段必须有，主键字段必须有**。Milvus 的 Collection 至少需要一个向量字段和一个主键字段——这是硬性要求。

**原则2：标量字段按需添加**。每个标量字段都会占用存储空间，如果你不需要按某个字段过滤，就不要加它。比如，如果你从来不会按 `source` 字段过滤搜索结果，那就不需要加这个字段——你可以把它放到 `metadata` JSON 字段里。

**原则3：VARCHAR 的 max_length 要合理**。max_length 决定了字段的最大存储长度——设太小会截断数据，设太大会浪费存储空间。对于文档内容，建议 65535（VARCHAR 的最大值）；对于分类、标签等短字符串，64~256 就够了。

**原则4：高频过滤字段用静态标量字段，低频字段用 JSON**。如果你经常按 `category` 过滤，就把 `category` 定义为 VARCHAR 字段并创建标量索引；如果你偶尔按 `author` 过滤，就把 `author` 放到 JSON 字段里。

比如，下面的 Schema 设计对比了"好的设计"和"差的设计"：

```python
# 好的设计：高频过滤字段用静态标量字段，低频字段用 JSON
fields_good = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=64),  # 高频过滤
    FieldSchema(name="year", dtype=DataType.INT32),                       # 高频过滤
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
    FieldSchema(name="metadata", dtype=DataType.JSON),                    # 低频字段
]

# 差的设计：所有字段都放在 JSON 里，过滤性能差
fields_bad = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
    FieldSchema(name="metadata", dtype=DataType.JSON),  # category/year 也放这里
]
```

由于 Milvus 对静态标量字段可以创建独立的标量索引（INVERTED / STL_SORT），过滤时可以直接走索引，速度远快于扫描 JSON 字段。所以，高频过滤字段一定要定义为静态标量字段。

---

## 常见误区：Schema 设计好了就不能改

Milvus 2.5+ 支持 `add_field()` 向已有 Collection 添加新字段——这意味着 Schema 不是一成不变的。但有几个限制：你不能删除字段、不能修改已有字段的类型、不能修改向量字段的维度。所以 Schema 设计时，向量维度一定要选对（取决于你的 Embedding 模型），其他标量字段可以后续按需添加。

```python
# 向已有 Collection 添加新字段（Milvus 2.5+）
from pymilvus import Collection

collection = Collection("documents")
collection.add_field(
    FieldSchema(name="priority", dtype=DataType.INT32, default_value=0)
)
```

另一个常见误区是**把所有字段都设成动态字段**——虽然 `enable_dynamic_field=True` 让你可以随意插入额外字段，但动态字段的过滤性能显著低于静态标量字段。如果你的 Collection 有 10 个以上需要过滤的字段，应该把它们定义为静态标量字段，而不是全部依赖动态字段。

---

## 小结

这一节我们聊了 Milvus 的数据模型设计：Collection 是数据组织的基本单位，Schema 用 Python API 显式定义字段结构。Schema 设计的核心原则是"高频过滤字段用静态标量字段，低频字段用 JSON"——这直接影响过滤性能。下一节我们要深入字段类型，了解 Milvus 支持的所有向量类型、标量类型和动态字段。
