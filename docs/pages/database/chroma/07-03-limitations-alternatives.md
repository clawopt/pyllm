# 7.3 Chroma 的局限性及替代方案

> **知道工具的边界，才能在边界之内做到最好——也才能在需要时果断跨越边界**

---

## 这一节在讲什么？

这是本教程的最后一节，我们要诚实地面对 Chroma 的局限性。Chroma 是一个优秀的嵌入式向量数据库，但它不是银弹——它有明确的能力边界，超出这些边界就应该考虑其他方案。这一节我们要讲清楚 Chroma 的五个核心局限、每个局限的具体影响、以及当你遇到这些局限时应该转向哪个替代方案。理解这些局限性不是对 Chroma 的否定，而是对它的正确使用——在适合的场景用适合的工具。

---

## 局限一：不支持原生分布式部署

**问题**：Chroma 是单机架构，所有数据存储在一个 SQLite 数据库和一组 HNSW 索引文件中。它没有内置的分片（Sharding）、复制（Replication）或故障转移（Failover）机制。这意味着 Chroma 的容量和吞吐量受限于单机的 CPU、内存和磁盘。

**影响**：当你的数据量超过单机内存容量（通常 10M~50M 条 384 维向量），或者查询 QPS 超过单机处理能力（通常 1000~5000 QPS），Chroma 就无法继续扩展了。

**替代方案**：

| 需求 | 推荐方案 | 理由 |
|------|---------|------|
| 数据量 > 10M 向量 | Milvus | 分布式架构，支持水平扩展到 10B+ 向量 |
| 高可用（99.9%+ SLA） | Pinecone / Milvus | 内置复制和故障转移 |
| 多租户 SaaS | Milvus / Qdrant | 支持逻辑隔离和资源配额 |

**临时缓解**：在达到单机上限之前，你可以通过多 Collection 分片来缓解——按业务线或用户组把数据分散到不同的 Collection 中，每个 Collection 独立查询后合并结果。但这只是权宜之计，不是真正的分布式方案。

```python
# 临时缓解：多 Collection 分片
class ShardedCollection:
    """简单的 Collection 分片方案"""

    def __init__(self, client, base_name: str, num_shards: int = 4):
        self.shards = []
        for i in range(num_shards):
            col = client.get_or_create_collection(
                name=f"{base_name}_shard_{i}",
                metadata={"hnsw:space": "cosine"}
            )
            self.shards.append(col)

    def add(self, documents, ids, metadatas=None):
        """按 ID 哈希分片"""
        for i in range(len(ids)):
            shard_idx = hash(ids[i]) % len(self.shards)
            doc = [documents[i]]
            doc_ids = [ids[i]]
            meta = [metadatas[i]] if metadatas else None
            self.shards[shard_idx].add(documents=doc, ids=doc_ids, metadatas=meta)

    def query(self, query_texts, n_results=5, **kwargs):
        """查询所有分片，合并结果"""
        all_results = []
        for shard in self.shards:
            try:
                r = shard.query(query_texts=query_texts, n_results=n_results, **kwargs)
                all_results.append(r)
            except Exception:
                continue

        # 合并并按距离排序
        merged = {"ids": [[]], "documents": [[]], "distances": [[]], "metadatas": [[]]}
        candidates = []
        for r in all_results:
            for i in range(len(r['ids'][0])):
                candidates.append({
                    "id": r['ids'][0][i],
                    "document": r['documents'][0][i] if r['documents'] else None,
                    "distance": r['distances'][0][i] if r['distances'] else None,
                    "metadata": r['metadatas'][0][i] if r['metadatas'] else None,
                })

        candidates.sort(key=lambda x: x['distance'] if x['distance'] is not None else float('inf'))
        top_k = candidates[:n_results]

        for c in top_k:
            merged['ids'][0].append(c['id'])
            merged['documents'][0].append(c['document'])
            merged['distances'][0].append(c['distance'])
            merged['metadatas'][0].append(c['metadata'])

        return merged
```

---

## 局限二：Metadata 过滤无 B-tree 索引

**问题**：Chroma 的 metadata 过滤是对 SQLite 全表做线性扫描，没有 B-tree 索引。这意味着 where 过滤的时间复杂度是 O(N)——N 是 Collection 中的总文档数，而不是满足条件的文档数。

**影响**：当数据量超过 100K 条时，where 过滤本身可能成为查询延迟的主要来源。特别是对高基数字段（如 user_id、timestamp）做过滤时，性能问题尤为突出。

**替代方案**：

| 需求 | 推荐方案 | 理由 |
|------|---------|------|
| 高性能 metadata 过滤 | Qdrant | 支持 payload 索引（类似 B-tree） |
| 复杂结构化查询 | Milvus | 支持标量字段索引 |
| 全文搜索 + 向量搜索 | Weaviate | 内置 BM25 + 向量混合搜索 |

**临时缓解**：把高选择性的过滤条件转化为低基数的 metadata 字段。比如把 `user_id` 转化为 `user_group`（用户分组），先按组过滤缩小范围，再在应用层做精确匹配。

---

## 局限三：不支持实时更新已有文档的 Embedding

**问题**：当你用 `collection.update()` 更新文档的 document 字段时，Chroma 会自动重新计算 embedding 并更新索引。但如果你只是想更新 embedding（比如换了 embedding 模型），Chroma 没有提供单独更新 embedding 的 API——你必须同时更新 document 或使用 delete + add 的方式。

**影响**：当你需要升级 embedding 模型时，必须对整个 Collection 做全量重建——删除旧数据、用新模型重新编码、重新入库。这个过程可能需要数小时甚至数天（取决于数据量）。

**替代方案**：大多数向量数据库都面临同样的问题——embedding 模型升级需要全量重建。唯一的缓解方式是**版本化 Collection**：用新的 embedding 模型创建一个新的 Collection，数据双写一段时间后切换流量。

```python
# 版本化 Collection：平滑升级 embedding 模型
def upgrade_embedding_model(client, old_collection_name, new_ef, batch_size=1000):
    """升级 embedding 模型：创建新 Collection，迁移数据"""
    old_col = client.get_collection(old_collection_name)
    count = old_col.count()
    new_col_name = f"{old_collection_name}_v2"

    new_col = client.create_collection(
        name=new_col_name,
        embedding_function=new_ef,
        metadata={"hnsw:space": "cosine"}
    )

    # 分批迁移
    for offset in range(0, count, batch_size):
        batch = old_col.get(
            limit=batch_size,
            offset=offset,
            include=["documents", "metadatas"]
        )

        if not batch['ids']:
            break

        new_col.add(
            documents=batch['documents'],
            ids=batch['ids'],
            metadatas=batch['metadatas']
        )

        print(f"迁移进度: {min(offset + batch_size, count)}/{count}")

    print(f"✅ 迁移完成: {new_col_name} ({new_col.count()} 条)")
    return new_col
```

---

## 局限四：不支持复杂的向量运算

**问题**：Chroma 只支持向量相似度搜索（KNN），不支持向量聚合、向量过滤（如"找出与向量 A 相似但与向量 B 不相似的文档"）、向量聚类等复杂运算。

**影响**：如果你需要实现"推荐除已读之外的文章"这类逻辑，Chroma 无法直接支持——你需要在应用层先获取已读文章的 ID，然后在查询结果中排除它们。

**替代方案**：Qdrant 和 Weaviate 支持更丰富的向量运算，包括向量过滤和条件排除。

---

## 局限五：建议的最大集合大小

**问题**：Chroma 官方建议单个 Collection 的向量数不超过 1000 万（10M）。超过这个规模后，HNSW 索引的内存占用、查询延迟、写入性能都会显著下降。

**影响**：如果你的数据量超过 10M 条，Chroma 可能不是最佳选择。

**替代方案**：

| 数据规模 | 推荐方案 |
|----------|---------|
| < 1M | Chroma（完全够用） |
| 1M ~ 10M | Chroma（注意性能监控）或 Qdrant |
| 10M ~ 100M | Milvus 或 Qdrant |
| > 100M | Milvus（唯一稳定支撑十亿级向量的开源方案） |

---

## 何时该升级：一个检查清单

```
┌─────────────────────────────────────────────────────────────────┐
│  Chroma 升级检查清单                                             │
│                                                                 │
│  如果以下任一条件成立，你应该考虑迁移到其他向量数据库：            │
│                                                                 │
│  ☐ 数据量超过 1000 万向量                                       │
│  ☐ 查询 QPS 超过 5000                                          │
│  ☐ 需要多节点分布式部署                                         │
│  ☐ 需要高可用（99.9%+ SLA）                                     │
│  ☐ metadata 过滤成为性能瓶颈                                    │
│  ☐ 需要亚毫秒级查询延迟                                         │
│  ☐ 需要复杂的多模态检索                                         │
│  ☐ 需要企业级权限/审计/合规                                     │
│  ☐ 需要全文搜索 + 向量搜索的混合检索                             │
│                                                                 │
│  如果以上条件都不满足，Chroma 仍然是你的最佳选择。                │
└─────────────────────────────────────────────────────────────────┘
```

---

## 课程总结

恭喜你走到了这里！让我们回顾一下这七章的旅程：

**第1章**：我们理解了为什么需要向量数据库——语义搜索是关键词搜索的进化，Chroma 是最易上手的向量数据库选择。

**第2章**：我们掌握了 Chroma 的 CRUD 操作——add/get/update/delete 的每个参数、每个返回值、每个边界情况。

**第3章**：我们深入了 Embedding 的世界——Embedding Function 的工作机制、文档切分策略、距离度量的数学原理。

**第4章**：我们学会了高级查询——query() 的完整参数、Where 过滤语法、多阶段检索与 Re-ranking。

**第5章**：我们构建了真正的 RAG 系统——从架构设计到 PDF 问答 Demo 到对话记忆管理。

**第6章**：我们让 Chroma 走向生产——持久化、性能调优、并发访问、监控运维。

**第7章**：我们站在更高的视角——选型对比、框架集成、局限性认知。

Chroma 不是最强大的向量数据库，但它是最友好的。它让你用最少的代码、最短的时间，从零开始构建一个可工作的 RAG 系统。当你需要更强大的能力时，你在 Chroma 上学到的所有知识——向量搜索的原理、metadata 的设计、距离度量的选择、RAG 的架构——都可以无缝迁移到其他向量数据库。因为工具会变，但原理不变。
