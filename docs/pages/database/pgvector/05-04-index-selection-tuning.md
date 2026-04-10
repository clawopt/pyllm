# 5.4 索引选型与调优实战

> **选对索引只是第一步——调对参数才是让 pgvector 真正飞起来的关键**

---

## 这一节在讲什么？

前面两节我们分别学了 IVFFlat 和 HNSW 的原理和参数，但"知道参数是什么"和"知道怎么调"是两回事。这一节我们要建立一套完整的索引选型和调优方法论——什么时候该建索引、选哪种索引、参数怎么设、如何用 EXPLAIN ANALYZE 诊断问题、以及距离操作符与索引的对应关系（这是最容易踩坑的地方）。

---

## 索引选型决策树

```
你的数据量有多大？
│
├─ < 10K 条
│   → 不需要向量索引，暴力搜索足够快
│
├─ 10K ~ 1M 条
│   │
│   ├─ 内存充足（> 4GB）？
│   │   ├─ 是 → HNSW（m=16, ef_construction=64）
│   │   └─ 否 → IVFFlat（lists=√N）
│   │
│   └─ 数据频繁更新？
│       ├─ 是 → HNSW（支持增量更新）
│       └─ 否 → HNSW 或 IVFFlat 都可以
│
└─ > 1M 条
    │
    ├─ 内存充足（> 8GB）？
    │   ├─ 是 → HNSW（m=32, ef_construction=128）
    │   └─ 否 → IVFFlat（lists=√N）
    │
    └─ 需要极致查询速度？
        ├─ 是 → HNSW（ef_search=100~200）
        └─ 否 → IVFFlat（probes=lists×5%）
```

---

## 距离操作符与索引的对应关系

这是 pgvector 调优中最容易踩的坑——**索引创建时指定的 ops 必须与查询时使用的操作符一致**，否则索引不会被使用：

| 索引 ops | 对应的操作符 | 距离类型 |
|----------|------------|---------|
| `vector_l2_ops` | `<->` | L2 距离 |
| `vector_cosine_ops` | `<=>` | 余弦距离 |
| `vector_ip_ops` | `<#>` | 负内积 |

```sql
-- 创建余弦距离索引
CREATE INDEX idx_emb_cosine ON documents
USING hnsw (embedding vector_cosine_ops);

-- ✅ 使用 <=> 操作符查询——索引生效
SELECT * FROM documents ORDER BY embedding <=> '[0.1, ...]' LIMIT 5;

-- ❌ 使用 <-> 操作符查询——索引不生效！走全表扫描
SELECT * FROM documents ORDER BY embedding <-> '[0.1, ...]' LIMIT 5;
```

如果你需要同时支持多种距离度量的查询，可以创建多个索引：

```sql
-- 同时创建三种距离的索引
CREATE INDEX idx_emb_l2 ON documents USING hnsw (embedding vector_l2_ops);
CREATE INDEX idx_emb_cosine ON documents USING hnsw (embedding vector_cosine_ops);
CREATE INDEX idx_emb_ip ON documents USING hnsw (embedding vector_ip_ops);
```

但这会占用 3 倍的内存。更实际的做法是只创建最常用的距离度量索引（通常是 cosine），其他距离用暴力搜索。

---

## EXPLAIN ANALYZE 诊断

```sql
-- 检查索引是否被使用
EXPLAIN ANALYZE
SELECT id, content, embedding <=> '[0.1, ...]'::vector(384) AS distance
FROM documents
ORDER BY embedding <=> '[0.1, ...]'::vector(384)
LIMIT 5;
```

**索引生效**的输出：

```
Index Scan using idx_emb_cosine on documents
  Index Cond: (embedding <=> '[0.1, ...]'::vector(384))
  Execution Time: 2.345 ms
```

**索引未生效**（全表扫描）的输出：

```
Seq Scan on documents
  Sort Key: (embedding <=> '[0.1, ...]'::vector(384))
  Execution Time: 234.567 ms
```

如果看到 `Seq Scan`，检查以下几点：
1. 操作符是否与索引 ops 匹配
2. 数据量是否太小（PostgreSQL 可能判断全表扫描更快）
3. 索引是否已构建完成

---

## 性能基准参考

以下是在不同数据量/维度/索引配置下的典型性能参考（单机，16GB 内存，SSD）：

| 数据量 | 维度 | 索引 | 查询延迟 | QPS | 召回率@10 |
|--------|------|------|---------|-----|----------|
| 10K | 384 | 无 | ~20ms | ~50 | 100% |
| 10K | 384 | HNSW | ~1ms | ~1000 | 99% |
| 100K | 384 | HNSW | ~3ms | ~300 | 98% |
| 1M | 384 | HNSW | ~8ms | ~120 | 97% |
| 1M | 384 | IVFFlat | ~15ms | ~60 | 95% |
| 1M | 1536 | HNSW | ~25ms | ~40 | 97% |

---

## 常见误区

### 误区 1：建了索引就一定快

PostgreSQL 的优化器会根据成本估算选择执行计划。如果它判断全表扫描更快（比如数据量很小或查询返回大量行），就不会使用索引。用 EXPLAIN ANALYZE 确认。

### 误区 2：一个索引支持所有距离度量

每种距离度量需要单独的索引。用 `<=>` 查询时只有 `vector_cosine_ops` 索引生效，`vector_l2_ops` 索引不会被使用。

### 误区 3：HNSW 的 ef_search 是全局设置

`SET hnsw.ef_search = 100` 只对当前数据库会话生效，不影响其他连接。你需要在每个查询会话中设置，或者在应用代码中每次连接后设置。

---

## 本章小结

索引选型和调优是 pgvector 性能优化的核心环节。核心要点回顾：第一，数据量 < 10K 不需要索引，> 1M 优先选 HNSW；第二，索引 ops 必须与查询操作符匹配——`vector_cosine_ops` 对应 `<=>`，`vector_l2_ops` 对应 `<->`，`vector_ip_ops` 对应 `<#>`；第三，用 EXPLAIN ANALYZE 确认索引是否被使用，看到 `Seq Scan` 说明索引未生效；第四，HNSW 的推荐参数起点是 m=16、ef_construction=64、ef_search=40；第五，ef_search 只对当前会话生效，需要在应用代码中设置。

下一章我们将进入 RAG 系统实战——用 pgvector 构建一个完整的文档问答系统。
