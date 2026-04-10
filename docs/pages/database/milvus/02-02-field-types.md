# 2.2 字段类型全景：向量字段、标量字段、动态字段

> **选对字段类型是性能优化的第一步——不同类型的存储和查询性能差异巨大**

---

## 这一节在讲什么？

上一节我们学了怎么定义 Collection 和 Schema，但只是用了几个常见的字段类型。Milvus 实际上支持非常丰富的字段类型——从 float32 向量到稀疏向量，从 INT64 到 JSON，从静态字段到动态字段。这些类型不只是"名字不同"——它们直接影响数据的存储方式、索引的构建方式和查询的性能。这一节我们要把 Milvus 的所有字段类型过一遍，帮你建立完整的类型认知。

---

## 向量字段类型

向量字段是 Milvus 的核心——没有向量字段就不是向量数据库了。Milvus 支持五种向量类型，覆盖了从密集向量到稀疏向量、从全精度到半精度的全部场景：

### FLOAT_VECTOR：最常用的向量类型

`FLOAT_VECTOR` 是最常用的向量类型——每个维度用 float32（4 字节）存储。绝大多数 Embedding 模型（OpenAI、SentenceTransformers、BGE）输出的都是 float32 向量，所以这是你的默认选择。

```python
from pymilvus import FieldSchema, DataType

# 768 维 float32 向量——大多数文本 Embedding 模型的输出维度
FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768)

# 1536 维 float32 向量——OpenAI text-embedding-3-small 的输出维度
FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536)
```

存储开销计算：768 维 × 4 字节 = 3072 字节/条。100 万条向量 ≈ 2.9 GB。

### FLOAT16_VECTOR / BFLOAT16_VECTOR：半精度向量

`FLOAT16_VECTOR` 用 float16（2 字节）存储每个维度，精度比 float32 低但存储减半。`BFLOAT16_VECTOR` 用 bfloat16（2 字节）存储，是 Google 为深度学习设计的浮点格式，动态范围比 float16 大但精度更低。

```python
# 半精度向量——存储减半，精度略降
FieldSchema(name="embedding", dtype=DataType.FLOAT16_VECTOR, dim=768)
# 768 × 2 = 1536 字节/条，比 FLOAT_VECTOR 节省 50%
```

半精度向量的使用场景是：你的 Embedding 模型本身就输出 float16/bfloat16（比如某些 GPU 推理框架），或者你想在存储和精度之间做权衡。但要注意：不是所有索引类型都支持半精度向量——HNSW 支持，IVF_PQ 不一定支持（取决于版本）。

### BINARY_VECTOR：二值向量

`BINARY_VECTOR` 把每个维度压缩成 1 bit（0 或 1），存储开销极小。适合哈希类算法（如 LSH）输出的二值向量。

```python
# 二值向量——每个维度 1 bit
# dim 参数表示 bit 数，必须是 8 的倍数
FieldSchema(name="hash", dtype=DataType.BINARY_VECTOR, dim=512)
# 512 bits = 64 字节/条
```

二值向量在实际项目中用得不多，因为大多数 Embedding 模型输出的是连续值向量，不适合二值化。

### SPARSE_FLOAT_VECTOR：稀疏向量

`SPARSE_FLOAT_VECTOR` 是 Milvus 2.4+ 新增的类型——它存储的是稀疏向量，即大部分维度为 0、只有少数维度有值的向量。稀疏向量在关键词搜索（如 SPLADE、BM25 向量化）和某些推荐场景中很有用。

```python
# 稀疏向量——不需要指定 dim，维度是动态的
FieldSchema(name="sparse_embedding", dtype=DataType.SPARSE_FLOAT_VECTOR)
```

```python
# 插入稀疏向量数据——用字典表示 {维度索引: 值}
sparse_vec = {0: 0.5, 100: 0.3, 5000: 0.8, 30000: 0.1}
# 只有 4 个非零维度，其余都是 0

client.insert(
    collection_name="sparse_demo",
    data=[
        {"id": 1, "sparse_embedding": {0: 0.5, 100: 0.3, 5000: 0.8}},
        {"id": 2, "sparse_embedding": {50: 0.2, 200: 0.6, 8000: 0.4}},
    ]
)
```

稀疏向量的搜索使用 SPARSE_INVERTED_INDEX 或 SPARSE_WAND 索引，搜索逻辑跟密集向量不同——它是基于倒排索引的匹配，而不是距离计算。

### 一个 Collection 多个向量字段

Milvus 支持一个 Collection 有多个向量字段——这是多模态搜索的基础。比如，你可以同时存储文本 embedding 和图像 embedding：

```python
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="text_embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
    FieldSchema(name="image_embedding", dtype=DataType.FLOAT_VECTOR, dim=512),
    FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=256),
]
```

搜索时，你可以选择搜索哪个向量字段，或者同时搜索两个字段然后用 RRF 合并结果（这个我们在第 5 章会详细讲）。

---

## 标量字段类型

标量字段用于过滤和返回结果，不参与相似度计算。Milvus 支持以下标量类型：

| 类型 | 说明 | 示例 |
|------|------|------|
| `INT8` / `INT16` / `INT32` / `INT64` | 整数类型，范围不同 | 年龄、年份、ID |
| `FLOAT` / `DOUBLE` | 浮点类型 | 价格、评分 |
| `BOOL` | 布尔类型 | 是否上架、是否删除 |
| `VARCHAR` | 变长字符串，需指定 `max_length` | 标题、分类、来源 |
| `JSON` | JSON 对象，灵活的键值对 | metadata |
| `ARRAY` | 数组类型，需指定 `element_type` 和 `max_capacity` | 标签列表 |

```python
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=64),
    FieldSchema(name="year", dtype=DataType.INT32),
    FieldSchema(name="price", dtype=DataType.FLOAT),
    FieldSchema(name="in_stock", dtype=DataType.BOOL),
    FieldSchema(name="tags", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=10),
    FieldSchema(name="metadata", dtype=DataType.JSON),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
]
```

### VARCHAR 的 max_length

VARCHAR 必须指定 `max_length`，这个值决定了字段的最大存储长度。设太小会截断数据，设太大会浪费空间。建议值：

- 文档内容：65535（最大值）
- 标题、来源：256~512
- 分类、标签：64~128
- ID 类字符串：64

### JSON 类型

JSON 类型是 Milvus 2.x 中最灵活的标量类型——你可以存储任意结构的 JSON 数据，然后通过 JSON 路径表达式过滤：

```python
# 插入 JSON 数据
client.insert(
    collection_name="products",
    data=[
        {
            "id": 1,
            "embedding": [0.1] * 768,
            "metadata": {
                "author": "Alice",
                "tags": ["AI", "ML"],
                "rating": 4.5,
                "published": True
            }
        }
    ]
)

# 按 JSON 字段过滤
results = client.search(
    collection_name="products",
    data=[[0.1] * 768],
    limit=5,
    filter='metadata["author"] == "Alice" and metadata["rating"] > 4.0'
)
```

JSON 字段的过滤性能不如静态标量字段——因为 JSON 字段需要解析后才能过滤，而静态标量字段可以直接走索引。所以高频过滤字段不要放在 JSON 里。

### ARRAY 类型

ARRAY 类型用于存储同类型元素的列表，比如标签列表、特征列表：

```python
# 定义 ARRAY 字段
FieldSchema(name="tags", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=10)

# 插入 ARRAY 数据
client.insert(
    collection_name="products",
    data=[
        {"id": 1, "tags": ["AI", "ML", "Python"], "embedding": [0.1] * 768},
    ]
)

# 按 ARRAY 字段过滤
results = client.search(
    collection_name="products",
    data=[[0.1] * 768],
    limit=5,
    filter='array_contains(tags, "AI")'
)
```

---

## 主键字段

每个 Collection 必须有一个主键字段，用于唯一标识每条数据。主键支持两种类型：

```python
# INT64 自增主键——Milvus 自动生成 ID
FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True)

# VARCHAR 自定义主键——你自己指定 ID
FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=64, is_primary=True, auto_id=False)
```

INT64 自增主键的优点是简单——你不需要自己生成 ID，Milvus 会自动分配。VARCHAR 自定义主键的优点是灵活——你可以用文档的原始 ID（如 URL 哈希、UUID）作为主键，方便跟其他系统对接。

### 常见误区：auto_id=True 时手动指定 ID

如果你设置了 `auto_id=True`，插入数据时就不需要（也不应该）指定 id 字段——Milvus 会自动生成。如果你手动指定了 id，它会被忽略：

```python
# auto_id=True 时的正确插入方式
client.insert(
    collection_name="demo",
    data=[
        {"vector": [0.1] * 768, "text": "hello"},  # 不需要 id 字段
    ]
)

# auto_id=True 时手动指定 id 会被忽略
client.insert(
    collection_name="demo",
    data=[
        {"id": 999, "vector": [0.1] * 768, "text": "hello"},  # id=999 会被忽略
    ]
)
```

---

## 动态字段

动态字段是 Milvus 2.x 的一个实用特性——它允许你插入 Schema 之外的字段，这些额外字段会自动存入一个隐藏的 `$meta` JSON 字段中：

```python
# 创建 Collection 时启用动态字段
schema = CollectionSchema(fields=fields, enable_dynamic_field=True)

# 插入时传入 Schema 之外的字段
client.insert(
    collection_name="documents",
    data=[
        {
            "id": 1,
            "embedding": [0.1] * 768,
            "content": "hello world",
            # 以下字段不在 Schema 中，但因为启用了动态字段，会被自动存储
            "author": "Alice",
            "page_number": 42,
            "is_reviewed": True,
        }
    ]
)

# 按动态字段过滤
results = client.search(
    collection_name="documents",
    data=[[0.1] * 768],
    limit=5,
    filter='author == "Alice"'  # 动态字段也可以过滤
)
```

动态字段的代价是过滤性能——动态字段存储在 JSON 中，过滤时需要解析 JSON，比静态标量字段慢。Milvus 2.5+ 支持为 JSON 字段创建 INVERTED 索引，可以显著提升动态字段的过滤性能。

---

## 常见误区：VARCHAR 的 max_length 设得越大越好

有些同学觉得 VARCHAR 的 max_length 设大一点更安全——比如所有 VARCHAR 都设成 65535。这不会导致存储空间浪费（VARCHAR 是变长存储，实际占用空间取决于内容长度），但会影响 Milvus 的内存分配策略——过大的 max_length 可能导致 QueryNode 在加载和过滤时分配更多内存。建议根据实际数据分布设置合理的 max_length。

---

## 小结

这一节我们覆盖了 Milvus 的全部字段类型：5 种向量类型（FLOAT_VECTOR、FLOAT16_VECTOR、BFLOAT16_VECTOR、BINARY_VECTOR、SPARSE_FLOAT_VECTOR）、6 种标量类型（INT、FLOAT、BOOL、VARCHAR、JSON、ARRAY）、主键字段和动态字段。选对字段类型是性能优化的第一步——高频过滤字段用静态标量字段，低频灵活字段用 JSON 或动态字段。下一节我们聊距离度量——L2、Cosine、IP 三种度量方式的选择。
