# 2.1 索引与查询优化基础

> **没有索引的数据库就像没有目录的百科全书——每次查找都要从头翻到尾**

---

## 这一节在讲什么？

当你的数据表只有几十条记录时，任何查询都是瞬间完成的。但当数据量增长到十万、百万条时，没有索引的查询会变得非常慢——因为 PostgreSQL 必须逐行扫描整张表来找到符合条件的记录。索引就是解决这个问题的核心机制，它为数据建立了"目录"，让查询可以快速定位到目标行而不需要全表扫描。这一节我们要理解 PostgreSQL 的索引体系——B-tree 索引的工作原理、如何用 EXPLAIN ANALYZE 诊断查询性能、以及索引设计的核心原则。这些知识不仅是 PostgreSQL 调优的基础，也是后续理解 pgvector 的 IVFFlat 和 HNSW 索引的前提。

---

## 为什么需要索引

让我们用一个直观的例子来理解索引的价值。假设你有一张 100 万条记录的 documents 表，你要查找 category = 'tech' 的文档：

```sql
-- 没有 category 列的索引时：全表扫描（Sequential Scan）
SELECT * FROM documents WHERE category = 'tech';
-- PostgreSQL 必须逐行检查 100 万条记录，耗时可能数秒

-- 创建索引后
CREATE INDEX idx_documents_category ON documents (category);
-- 再次查询：索引扫描（Index Scan）
SELECT * FROM documents WHERE category = 'tech';
-- PostgreSQL 直接通过索引定位到目标行，耗时可能仅数毫秒
```

索引的加速效果取决于数据的选择性——如果 category = 'tech' 只有 100 条记录（万分之一），索引扫描比全表扫描快几千倍；但如果 category = 'tech' 有 50 万条记录（一半），索引扫描可能反而更慢（因为还需要回表取数据）。PostgreSQL 的查询优化器会自动判断使用索引还是全表扫描，你不需要手动指定。

---

## B-tree 索引：PostgreSQL 的默认索引

B-tree（B树）是 PostgreSQL 创建索引时的默认类型，也是最常见的索引类型。它适用于等值查询（`=`）、范围查询（`>`、`<`、`BETWEEN`）和排序（`ORDER BY`）。

```sql
-- 创建 B-tree 索引
CREATE INDEX idx_documents_category ON documents (category);

-- 联合索引（多列）
CREATE INDEX idx_documents_cat_source ON documents (category, source);

-- 唯一索引
CREATE UNIQUE INDEX idx_documents_source ON documents (source);
```

B-tree 索引的工作原理类似于字典的目录——它把列值按排序顺序存储在一棵平衡树中，查询时从根节点逐层向下查找，时间复杂度是 O(log N)。对于 100 万条记录，B-tree 只需要约 20 次比较就能定位到目标行。

```
┌──────────────────────────────────────────────────────────────┐
│  B-tree 索引结构示意                                         │
│                                                              │
│              [M]                         ← 根节点            │
│             /   \                                          │
│        [D]       [T]                  ← 中间节点            │
│       /   \     /   \                                      │
│    [A-C] [G-L] [N-S] [U-Z]           ← 叶子节点            │
│                                                              │
│  查找 "tech" 的过程：                                        │
│  1. 根节点：tech > M → 走右子树                              │
│  2. 中间节点：tech < T → 走左子树                            │
│  3. 叶子节点：在 [N-S] 中找到 tech                           │
│  → 只需 3 次比较，而不是扫描 100 万行                        │
└──────────────────────────────────────────────────────────────┘
```

---

## PostgreSQL 的索引类型全景

除了 B-tree，PostgreSQL 还支持多种索引类型，每种适合不同的查询场景：

| 索引类型 | 适用场景 | pgvector 关联 |
|----------|---------|--------------|
| B-tree | 等值、范围、排序 | WHERE category = 'tech' |
| Hash | 纯等值查询 | 较少使用 |
| GIN | 数组、JSONB、全文搜索 | pgvector 的 IVFFlat/HNSW 基于 GiST |
| GiST | 地理空间、自定义数据类型 | pgvector 索引的实现基础 |
| SP-GiST | 非平衡数据结构（电话号码等） | 较少使用 |
| BRIN | 大表的块级统计信息 | 适合时序数据 |

其中 GiST（Generalized Search Tree）是 pgvector 索引的实现基础——IVFFlat 和 HNSW 索引都是通过 GiST 框架注册到 PostgreSQL 中的。这意味着 pgvector 的向量索引与 PostgreSQL 的查询优化器深度集成，优化器可以自动决定何时使用向量索引、何时使用 B-tree 索引、何时两者组合。

---

## EXPLAIN ANALYZE：读懂查询计划

`EXPLAIN ANALYZE` 是 PostgreSQL 调优的最重要的工具——它不仅显示查询的执行计划，还实际执行查询并报告每个步骤的耗时：

```sql
-- 查看查询计划（不实际执行）
EXPLAIN SELECT * FROM documents WHERE category = 'tech';

-- 查看查询计划并实际执行（报告真实耗时）
EXPLAIN ANALYZE SELECT * FROM documents WHERE category = 'tech';
```

典型输出：

```
QUERY PLAN
---------------------------------------------------------------------------
Index Scan using idx_documents_category on documents  (cost=0.42..8.44 rows=1 width=100) (actual time=0.015..0.016 rows=3 loops=1)
  Index Cond: (category = 'tech'::text)
Planning Time: 0.085 ms
Execution Time: 0.032 ms
```

关键指标解读：

| 指标 | 含义 |
|------|------|
| `Index Scan` | 使用了索引扫描（好！） |
| `Seq Scan` | 全表扫描（可能需要加索引） |
| `cost=0.42..8.44` | 预估的启动成本和总成本 |
| `rows=1` | 预估返回行数 |
| `actual time=0.015..0.016` | 实际启动时间和总时间（毫秒） |
| `rows=3` | 实际返回行数 |
| `Planning Time` | 查询计划生成时间 |
| `Execution Time` | 实际执行时间 |

**判断是否需要优化**：如果看到 `Seq Scan` 且 `Execution Time` 较高（> 100ms），通常意味着需要添加索引。但也要注意——如果查询返回的行数占表的大部分（比如 30% 以上），全表扫描可能比索引扫描更快，这是正常的。

---

## 索引设计原则

### 原则一：选择性高的字段优先

选择性 = 不同值的数量 / 总行数。选择性越高，索引的过滤效果越好。比如 `email` 列的选择性接近 1（每个用户邮箱不同），而 `gender` 列的选择性只有 0.5（只有男/女两种值）。对低选择性字段建索引效果很差。

### 原则二：联合索引注意列顺序

联合索引遵循"最左前缀"原则——查询条件必须从索引的最左列开始匹配：

```sql
-- 联合索引
CREATE INDEX idx_cat_source ON documents (category, source);

-- ✅ 能使用索引
SELECT * FROM documents WHERE category = 'tech';
SELECT * FROM documents WHERE category = 'tech' AND source = 'api.md';

-- ❌ 不能使用索引（跳过了最左列 category）
SELECT * FROM documents WHERE source = 'api.md';
```

### 原则三：避免过度索引

每个索引都会增加写入开销（INSERT/UPDATE/DELETE 需要同时更新索引）和存储空间。一张表的索引数量建议不超过 5~8 个。

---

## 常见误区

### 误区 1：索引越多查询越快

索引加速了查询，但拖慢了写入。每次 INSERT/UPDATE/DELETE 都需要更新所有相关索引。过度索引会导致写入性能急剧下降。

### 误区 2：WHERE 中的所有列都需要索引

只有选择性高的列才值得建索引。对 `boolean` 类型（只有 TRUE/FALSE）建索引几乎没有价值。

### 误区 3：索引建了就一定会被使用

PostgreSQL 的优化器会根据统计信息自动选择最优的执行计划。如果它判断全表扫描更快（比如目标行数占比太高），就不会使用索引。你可以用 `EXPLAIN ANALYZE` 确认索引是否被使用。

---

## 本章小结

索引是 PostgreSQL 性能优化的核心机制。核心要点回顾：第一，B-tree 是默认索引类型，适合等值和范围查询；第二，GiST 是 pgvector 索引的实现基础，IVFFlat 和 HNSW 都通过 GiST 框架注册；第三，EXPLAIN ANALYZE 是诊断查询性能的最重要工具，关注 `Seq Scan` vs `Index Scan` 和 `Execution Time`；第四，索引设计遵循三个原则——选择性高优先、联合索引注意列顺序、避免过度索引。

下一节我们将学习 JSONB 数据类型——它是 pgvector 在 RAG 系统中存储灵活 metadata 的最佳搭档。
