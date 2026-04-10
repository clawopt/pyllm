# 3.4 数据查询、更新与删除

> **query()、get()、upsert()、delete()——不只是搜索，数据管理也很重要**

---

## 这一节在讲什么？

前面两节我们学了插入和搜索，这是 Milvus 最常用的两个操作。但一个完整的数据管理流程还需要查询（按条件获取数据）、更新（修改已有数据）和删除（移除数据）。Milvus 的这些操作跟 pgvector 的 SQL 有很大不同——没有 UPDATE 语句、没有 DELETE FROM WHERE，而是用 Python API 来实现。这一节我们把 query()、get()、upsert()、delete() 都过一遍，同时聊一聊 Milvus 独特的 Segment 和 Compaction 机制。

---

## query()：按条件查询数据

`query()` 用于按标量条件查询数据，不涉及向量搜索——相当于 pgvector 的 `SELECT * FROM table WHERE ...`，但没有 ORDER BY 向量距离的能力。

```python
from pymilvus import MilvusClient

client = MilvusClient(uri="http://localhost:19530")

# 基本查询——按条件获取数据
results = client.query(
    collection_name="documents",
    filter='category == "tech"',
    output_fields=["content", "source", "category"],
    limit=100
)

for doc in results:
    print(f"ID: {doc['id']}, Content: {doc['content'][:50]}...")

# 多条件查询
results = client.query(
    collection_name="documents",
    filter='category == "tech" and year >= 2024',
    output_fields=["content", "year"],
    limit=50
)

# 查询所有数据（慎用！数据量大时很慢）
results = client.query(
    collection_name="documents",
    filter="",
    output_fields=["id", "content"],
    limit=1000
)
```

### query() 的分页

Milvus 2.5+ 支持 `offset` 参数做分页：

```python
# 第一页
page1 = client.query(
    collection_name="documents",
    filter='category == "tech"',
    output_fields=["content"],
    limit=20,
    offset=0
)

# 第二页
page2 = client.query(
    collection_name="documents",
    filter='category == "tech"',
    output_fields=["content"],
    limit=20,
    offset=20
)
```

注意：`offset` 在数据量大时性能会下降——因为 Milvus 需要跳过前 offset 条数据。如果你的分页深度超过 1000，建议改用基于 ID 的游标分页。

### query() vs search() 的区别

| 操作 | 用途 | 是否需要向量 | 返回排序 |
|------|------|------------|---------|
| search() | 向量相似度搜索 | 是 | 按距离排序 |
| query() | 标量条件查询 | 否 | 不保证顺序 |

`query()` 不能按向量距离排序——如果你需要"某个分类下最相似的文档"，必须用 `search()` + `filter`，而不是 `query()`。

---

## get()：按主键精确获取

`get()` 用于按主键精确获取数据——相当于 pgvector 的 `SELECT * FROM table WHERE id = ...`：

```python
# 按主键获取单条数据
result = client.get(
    collection_name="documents",
    ids=[1],
    output_fields=["content", "source", "category"]
)
print(result)

# 批量获取
results = client.get(
    collection_name="documents",
    ids=[1, 2, 3, 4, 5],
    output_fields=["content", "embedding"]
)

for doc in results:
    print(f"ID: {doc['id']}, Content: {doc['content'][:50]}")
```

`get()` 比 `query()` 更高效——因为它直接通过主键索引定位数据，不需要扫描。如果你知道数据的 ID，优先用 `get()` 而不是 `query()`。

---

## upsert()：存在则更新，不存在则插入

`upsert()` 是 insert + update 的组合——如果主键已存在则更新，不存在则插入。这在"数据可能重复插入"的场景下很有用：

```python
# 第一次 upsert——插入
client.upsert(
    collection_name="documents",
    data=[
        {
            "id": 1,
            "content": "Updated content for doc 1",
            "source": "wiki",
            "category": "tech",
            "embedding": [0.15] * 768,
            "metadata": {"author": "Alice", "version": 2}
        }
    ]
)

# 第二次 upsert（id=1 已存在）——更新
client.upsert(
    collection_name="documents",
    data=[
        {
            "id": 1,
            "content": "Content updated again",
            "source": "wiki",
            "category": "tech",
            "embedding": [0.2] * 768,
            "metadata": {"author": "Alice", "version": 3}
        }
    ]
)
```

### 常见误区：upsert() 不是原子操作

Milvus 的 upsert 实际上是"先删除旧数据，再插入新数据"——它不是原子操作。在高并发场景下，可能出现两个 upsert 请求同时操作同一条数据的情况，导致最终状态不确定。如果你需要强一致性，应该在应用层加锁或使用 pgvector 的事务能力。

---

## delete()：按主键或表达式删除

Milvus 支持两种删除方式——按主键删除和按表达式批量删除：

```python
# 按主键删除
client.delete(
    collection_name="documents",
    ids=[1, 2, 3]
)

# 按表达式批量删除
client.delete(
    collection_name="documents",
    filter='category == "spam" or year < 2020'
)
```

### 删除是逻辑删除

Milvus 的删除是逻辑删除——数据不会立即从磁盘上消失，而是被标记为"已删除"。搜索和查询时会自动跳过已删除的数据，但存储空间不会立即释放。

要物理清理已删除的数据，需要执行 `compact()`：

```python
# 触发 Compaction——合并 Segment，清理已删除数据
client.compact(collection_name="documents")

# 等待 Compaction 完成（异步操作）
import time
while True:
    state = client.get_compaction_state(collection_name="documents")
    if state == "Completed":
        print("Compaction done!")
        break
    time.sleep(1)
```

### 常见误区：删除后空间没释放

很多同学删除了大量数据后发现存储空间没有减少——这是因为 Compaction 是异步的，而且默认的自动 Compaction 可能不会立即触发。如果你需要立即释放空间，需要手动调用 `compact()`。但频繁 Compaction 会影响性能——建议在业务低峰期执行。

---

## Segment 与 Compaction：Milvus 的数据管理机制

理解 Segment 和 Compaction 对生产环境的数据管理至关重要——它们直接影响存储效率和查询性能。

### Segment：数据的基本存储单元

Milvus 把数据组织成 Segment——每个 Segment 包含一批数据（默认大小 512MB）。当你插入数据时，DataNode 会把数据攒成 Segment 然后写入对象存储。频繁的小批量插入会产生大量小 Segment——这被称为"Segment 碎片化"。

```
Segment 碎片化问题：

  理想状态：
  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
  │  Segment 1      │ │  Segment 2      │ │  Segment 3      │
  │  500MB          │ │  500MB          │ │  500MB          │
  │  10万条数据      │ │  10万条数据      │ │  10万条数据      │
  └─────────────────┘ └─────────────────┘ └─────────────────┘
  → 搜索时只需扫描 3 个 Segment，效率高

  碎片化状态：
  ┌──────┐┌──────┐┌──────┐┌──────┐┌──────┐┌──────┐┌──────┐┌──────┐
  │Seg 1 ││Seg 2 ││Seg 3 ││Seg 4 ││Seg 5 ││Seg 6 ││Seg 7 ││Seg 8 │
  │10MB  ││5MB   ││20MB  ││8MB   ││15MB  ││3MB   ││12MB  ││7MB   │
  │2000条││1000条││4000条││1600条││3000条││600条 ││2400条││1400条│
  └──────┘└──────┘└──────┘└──────┘└──────┘└──────┘└──────┘└──────┘
  → 搜索时需要扫描 8 个 Segment，每个 Segment 都有打开/关闭开销，效率低
```

### Compaction：合并 Segment

Compaction 的作用是合并小 Segment、清理已删除数据、优化存储空间。Milvus 支持自动 Compaction（默认开启）和手动 Compaction：

```python
# 手动触发 Compaction
client.compact(collection_name="documents")
```

### 常见误区：频繁小批量 upsert 导致碎片化

upsert 操作会产生"删除旧数据 + 插入新数据"的效果——每次 upsert 都会在 Segment 中留下已删除的"空洞"。如果你频繁小批量 upsert（比如每次只更新 1~2 条数据），Segment 碎片化会非常严重。正确的做法是：**攒够一批数据再 upsert**，或者使用 Auto-Compaction 让 Milvus 在后台自动合并。

---

## 小结

这一节我们覆盖了 Milvus 的数据管理操作：query() 按条件查询（不涉及向量搜索）、get() 按主键精确获取、upsert() 存在则更新不存在则插入、delete() 按主键或表达式删除。删除是逻辑删除，需要 compact() 物理清理。频繁小批量操作会导致 Segment 碎片化，建议攒够一批再操作。下一节开始我们进入第 4 章，深入 Milvus 的索引类型和性能优化。
