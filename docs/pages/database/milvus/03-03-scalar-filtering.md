# 3.3 标量过滤：expr 表达式与混合查询

> **Milvus 的 expr 过滤比 pgvector 的 WHERE 更灵活——但语法完全不同**

---

## 这一节在讲什么？

在 pgvector 教程里，我们用 SQL 的 WHERE 子句做标量过滤——`WHERE category = 'tech' AND price < 1000`。Milvus 没有SQL，它用自己的一套表达式语法（expr）来做过滤。虽然功能上等价，但语法差异很大——如果你习惯了 SQL，需要适应一下。这一节我们要把 Milvus 的 expr 语法讲完整，包括比较、逻辑、集合、JSON 路径、数组操作，以及过滤的执行策略和性能优化。

---

## 过滤表达式语法全景

### 比较操作

```python
# 等值比较
filter='category == "tech"'
filter='source != "wikipedia"'

# 数值比较
filter='price < 1000'
filter='year >= 2024'
filter='rating > 4.0 and rating <= 5.0'
```

注意 Milvus 的比较操作符用的是 `==` 而不是 SQL 的 `=`，字符串用双引号而不是单引号。这是 Python 风格的表达式，不是 SQL。

### 逻辑操作

```python
# AND / OR / NOT
filter='category == "tech" and year >= 2024'
filter='category == "tech" or category == "science"'
filter='not category == "spam"'

# 复合逻辑
filter='(category == "tech" or category == "science") and year >= 2024'
```

### 集合操作

```python
# in 操作——等价于 SQL 的 IN
filter='category in ["tech", "science", "art"]'

# not in
filter='category not in ["spam", "deleted"]'
```

### JSON 路径操作

JSON 字段的过滤使用 `[]` 语法访问嵌套键：

```python
# 访问 JSON 字段的顶层键
filter='metadata["author"] == "Alice"'

# 数值比较
filter='metadata["rating"] > 4.0'

# 布尔判断
filter='metadata["published"] == true'

# contains 操作——检查 JSON 数组是否包含某个元素
filter='metadata["tags"] contains "AI"'
```

比如，下面的程序展示了 JSON 过滤的完整用法，由于 metadata 是 JSON 类型字段，所以需要用 `[]` 语法访问其中的键：

```python
from pymilvus import MilvusClient

client = MilvusClient(uri="http://localhost:19530")

# 插入带 JSON metadata 的数据
client.insert(
    collection_name="documents",
    data=[
        {
            "id": 1,
            "embedding": [0.1] * 768,
            "content": "AI breakthrough in 2024",
            "category": "tech",
            "metadata": {
                "author": "Alice",
                "tags": ["AI", "ML", "Python"],
                "rating": 4.8,
                "published": True
            }
        },
        {
            "id": 2,
            "embedding": [0.2] * 768,
            "content": "Climate change report",
            "category": "science",
            "metadata": {
                "author": "Bob",
                "tags": ["climate", "environment"],
                "rating": 4.2,
                "published": True
            }
        }
    ]
)

# 按 JSON 字段过滤搜索
results = client.search(
    collection_name="documents",
    data=[[0.1] * 768],
    limit=5,
    filter='metadata["author"] == "Alice" and metadata["rating"] > 4.0',
    output_fields=["content", "metadata"]
)
```

### 数组操作

ARRAY 字段有自己的操作符：

```python
# array_contains——数组是否包含某个元素
filter='array_contains(tags, "AI")'

# array_contains_all——数组是否包含所有指定元素
filter='array_contains_all(tags, ["AI", "ML"])'

# array_contains_any——数组是否包含任一指定元素
filter='array_contains_any(tags, ["AI", "Python"])'

# array_length——数组长度
filter='array_length(tags) > 2'
```

### 动态字段过滤

动态字段（`enable_dynamic_field=True` 时插入的额外字段）的过滤语法跟静态标量字段一样：

```python
# 动态字段直接用字段名过滤
filter='author == "Alice"'

# 动态字段也支持比较和逻辑操作
filter='page_number > 10 and is_reviewed == true'
```

---

## 过滤的执行策略

Milvus 的标量过滤有两种执行策略，理解它们的区别对性能调优很重要：

### Filter-then-Search（先过滤再搜索）

这是 Milvus 2.x 的默认策略——先根据标量条件过滤出候选集，再在候选集中做向量搜索。

```
Filter-then-Search 流程：

  全量数据（100 万条）
     │
     ▼ filter: category == "tech"
  候选集（30 万条）
     │
     ▼ vector search: Top-5
  结果（5 条）
```

这种策略的优点是简单可靠，缺点是当过滤条件的选择性很低（比如 `category == "tech"` 匹配了 90% 的数据）时，过滤步骤几乎没减少计算量；当选择性很高（比如 `user_id == 12345` 只匹配了几十条）时，ANN 索引的优势发挥不出来——因为候选集太小，暴力搜索就够了。

### Iterative Filter（迭代过滤）

Milvus 2.5+ 支持迭代过滤——搜索和过滤交替进行，先从 ANN 索引中取一批候选，过滤掉不满足条件的，再取下一批，直到凑够 Top-K 结果。

```
Iterative Filter 流程：

  ANN 索引返回候选 → 过滤 → 不够？→ 继续取候选 → 过滤 → ... → 凑够 Top-K
```

迭代过滤在高选择性过滤场景下性能更好——因为它不需要先扫描全量数据做过滤，而是从 ANN 索引中按需取候选。但 Milvus 目前对迭代过滤的支持还在逐步完善，某些复杂过滤表达式可能仍使用 Filter-then-Search。

---

## 过滤性能优化

### 为高频过滤字段创建标量索引

默认情况下，Milvus 的标量过滤是全扫描的——它需要逐条检查每条数据是否满足过滤条件。如果你经常按某个字段过滤，可以为该字段创建标量索引：

```python
# 为 category 字段创建倒排索引
index_params = client.prepare_index_params()
index_params.add_index(
    field_name="category",
    index_type="INVERTED",      # 倒排索引，适合等值查询
    index_name="idx_category"
)

# 为 year 字段创建排序索引
index_params.add_index(
    field_name="year",
    index_type="STL_SORT",      # 排序索引，适合范围查询
    index_name="idx_year"
)

# 为 source 字段创建前缀索引
index_params.add_index(
    field_name="source",
    index_type="MARISA-TRIE",   # 前缀索引，适合字符串前缀匹配
    index_name="idx_source"
)

client.create_index(collection_name="documents", index_params=index_params)
```

标量索引类型的选择：

| 索引类型 | 适合场景 | 说明 |
|---------|---------|------|
| INVERTED | 等值查询、范围查询 | Milvus 2.5+ 推荐，性能最好 |
| STL_SORT | 范围查询（>、<、between） | 对数值字段排序，加速范围过滤 |
| MARISA-TRIE | 字符串前缀匹配 | 适合 LIKE "prefix%" 类查询 |

### 常见误区：标量索引和向量索引是两回事

Milvus 的标量索引（INVERTED、STL_SORT）和向量索引（HNSW、IVFFlat）是完全独立的——标量索引加速过滤，向量索引加速距离计算。创建标量索引不会影响向量搜索的性能，反之亦然。你可以同时为一个 Collection 创建标量索引和向量索引。

---

## 与 pgvector 混合查询的对比

pgvector 的混合查询用 SQL 表达：`WHERE category = 'tech' ORDER BY embedding <=> query_vec LIMIT 5`。Milvus 的混合查询用 API 参数表达：`search(filter='category == "tech"', data=[query_vec], limit=5)`。功能上完全等价，但表达方式不同：

| 维度 | pgvector | Milvus |
|------|----------|--------|
| 过滤语法 | SQL WHERE | expr 字符串 |
| 过滤+搜索 | 一条 SQL | search() 的 filter 参数 |
| 索引加速 | GIN/B-tree 加速 WHERE | INVERTED/STL_SORT 加速过滤 |
| 执行策略 | 由查询计划器决定 | Filter-then-Search 或 Iterative Filter |

pgvector 的优势是 SQL 的表达力——你可以写非常复杂的 WHERE 条件（子查询、窗口函数、JOIN），而 Milvus 的 expr 只支持基本的比较、逻辑和集合操作。但 Milvus 的优势是分布式——过滤可以在多个 QueryNode 上并行执行。

---

## 常见误区：过滤条件太严格导致结果为空

当你同时使用向量搜索和标量过滤时，如果过滤条件太严格，可能搜不到足够的结果。比如你要求 `category == "rare_category" and year == 2024 and rating > 4.9`，可能只有 2 条数据满足条件，但你要求 `limit=5`——这时 Milvus 只会返回 2 条结果，不会报错。你需要在应用层检查返回结果的数量，如果不够就放宽过滤条件重新搜索。

---

## 小结

这一节我们覆盖了 Milvus 的标量过滤：expr 表达式支持比较、逻辑、集合、JSON 路径和数组操作；过滤的执行策略有 Filter-then-Search（默认）和 Iterative Filter（Milvus 2.5+）；为高频过滤字段创建标量索引（INVERTED/STL_SORT/MARISA-TRIE）可以显著提升过滤性能。下一节我们聊数据查询、更新和删除。
