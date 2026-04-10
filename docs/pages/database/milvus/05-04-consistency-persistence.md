# 5.4 数据一致性与持久化

> **刚插入的数据"搜不到"——这不是 Bug，是分布式系统的必然**

---

## 这一节在讲什么？

如果你从 pgvector 转到 Milvus，你可能会遇到一个"诡异"的现象：你刚插入了一条数据，立即搜索却搜不到。在 pgvector 里，INSERT 之后立刻 SELECT 就能看到数据——因为 pgvector 有 ACID 事务保证。但 Milvus 是分布式系统，数据写入后需要经过消息队列、DataNode 处理、对象存储持久化、QueryNode 加载等多个步骤才能被搜索到。这个过程需要时间，而你可以通过一致性级别来控制"等多久"。这一节我们要聊清楚 Milvus 的一致性模型、Flush 与 Compaction 的机制，以及如何处理"数据搜不到"的问题。

---

## 一致性级别

Milvus 提供了四种一致性级别，控制的是"写入的数据多久能被搜索到"：

```
一致性级别从强到弱：

  Strong ────────────────────────────────────── Eventually
  ↑                                              ↑
  搜索前强制同步所有写入                          不保证可见性
  最慢但最新                                     最快但可能读到旧数据
```

### Strong：强一致性

搜索前强制同步所有写入——保证你能搜到所有已插入的数据，但搜索延迟最高。

```python
# 全局设置 Strong 一致性
client = MilvusClient(uri="http://localhost:19530", consistency_level="Strong")

# 或者在搜索时指定
results = client.search(
    collection_name="documents",
    data=[[0.1] * 768],
    limit=5,
    consistency_level="Strong"  # 搜索前强制同步
)
```

### Bounded：有界一致性

允许一定延迟（默认 180 秒）——搜索结果可能不包含最近 180 秒内插入的数据。这是 Milvus 的默认一致性级别，平衡了速度和新鲜度。

```python
results = client.search(
    collection_name="documents",
    data=[[0.1] * 768],
    limit=5,
    consistency_level="Bounded"  # 允许 180 秒延迟
)
```

### Session：会话一致性

同一客户端写入后立即可见——适合单客户端场景。如果你在同一个 MilvusClient 实例中先 insert 再 search，Session 级别保证你能搜到刚插入的数据。

```python
results = client.search(
    collection_name="documents",
    data=[[0.1] * 768],
    limit=5,
    consistency_level="Session"  # 同一客户端写入立即可见
)
```

### Eventually：最终一致性

不保证可见性——搜索最快，但可能读到旧数据。适合对数据新鲜度不敏感的场景（如离线分析）。

```python
results = client.search(
    collection_name="documents",
    data=[[0.1] * 768],
    limit=5,
    consistency_level="Eventually"  # 最快，但可能搜不到最新数据
)
```

---

## 为什么 Milvus 需要一致性级别

在 pgvector 中，INSERT 之后数据立刻就能被 SELECT 到——因为 PostgreSQL 是单机数据库，写入和读取在同一个进程中完成。但 Milvus 是分布式系统，数据写入和搜索在不同的节点上执行：

```
pgvector 的数据流（同步）：
  INSERT → 写入内存 → 立即可 SELECT ✅

Milvus 的数据流（异步）：
  insert() → 消息队列 → DataNode 消费 → 对象存储 → QueryNode 加载 → 可 search()
  ↑                                                          ↑
  应用层认为写入成功了                                    数据真正可搜索
  └──────────── 这段时间内数据"搜不到" ────────────────────┘
```

一致性级别控制的就是"应用层是否等待数据真正可搜索后才返回搜索结果"——Strong 等待，Eventually 不等待。

---

## Flush：强制数据持久化

`flush()` 的作用是强制 DataNode 立即处理缓冲区中的数据，写入对象存储。flush 之后，数据可以被 QueryNode 加载并搜索到。

```python
# 插入数据
client.insert(collection_name="documents", data=batch_data)

# flush 确保数据可搜索
client.flush(collection_name="documents")

# 现在 search 一定能找到数据（即使一致性级别是 Eventually）
results = client.search(
    collection_name="documents",
    data=[[0.1] * 768],
    limit=5
)
```

### 常见误区：每次 insert 后都 flush

频繁 flush 会严重影响写入性能——每次 flush 都会触发 DataNode 的 Segment 持久化操作，产生大量小 Segment。正确的做法是攒够一批数据后 flush 一次，或者使用 Session/Bounded 一致性级别让 Milvus 自动处理同步。

---

## Compaction：合并 Segment

Compaction 的作用是合并小 Segment、清理已删除数据、优化存储空间。Milvus 支持自动 Compaction（默认开启）和手动 Compaction：

```python
# 手动触发 Compaction
client.compact(collection_name="documents")
```

Compaction 是异步操作——调用后立即返回，后台慢慢执行。你可以通过 `get_compaction_state()` 查看进度。

### 什么时候需要手动 Compaction

- 删除了大量数据后——需要清理已删除数据释放空间
- 频繁小批量插入后——需要合并小 Segment 提升搜索性能
- upsert 操作频繁——需要清理旧版本数据

---

## 常见误区：插入后立即搜索发现数据"丢了"

这是 Milvus 初学者最常遇到的问题。解决方法有三个：

1. **使用 Session 一致性级别**——同一客户端写入后立即可见
2. **插入后 flush**——强制数据持久化
3. **使用 Strong 一致性级别**——搜索前自动同步

```python
# 方法1：Session 一致性（推荐）
results = client.search(
    collection_name="documents",
    data=[[0.1] * 768],
    limit=5,
    consistency_level="Session"
)

# 方法2：手动 flush
client.insert(collection_name="documents", data=batch_data)
client.flush("documents")
results = client.search(...)

# 方法3：Strong 一致性（最慢但最安全）
results = client.search(
    collection_name="documents",
    data=[[0.1] * 768],
    limit=5,
    consistency_level="Strong"
)
```

---

## 小结

这一节我们聊了 Milvus 的数据一致性与持久化：四种一致性级别（Strong/Bounded/Session/Eventually）控制数据可见性的延迟，flush 强制数据持久化，Compaction 合并 Segment 和清理已删除数据。"插入后搜不到"不是 Bug——是分布式系统的正常行为，可以通过一致性级别或 flush 来解决。下一节开始我们进入第 6 章，用 Milvus 搭建完整的 RAG 系统。
