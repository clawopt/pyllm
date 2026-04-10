# 1.1 PostgreSQL 是什么？为什么 RAG 工程师需要它

> **在学 pgvector 之前，你必须先理解 PostgreSQL——因为 pgvector 不是独立的数据库，而是 PostgreSQL 的一个扩展插件**

---

## 这一节在讲什么？

如果你已经对 PostgreSQL 非常熟悉，可以跳过前两章直接从第3章开始。但如果你对 PostgreSQL 的了解仅限于"听说过这个名字"，那这一节就是为你准备的。我们要回答三个问题：PostgreSQL 到底是什么？它和 MySQL、SQLite 有什么区别？以及——为什么一个做 RAG 的工程师需要学习一个关系型数据库？

答案的核心在于：pgvector 是 PostgreSQL 的扩展，不是独立的向量数据库。你不能脱离 PostgreSQL 来使用 pgvector，就像你不能脱离浏览器来使用 Chrome 扩展一样。理解 PostgreSQL 的核心能力——SQL 查询、ACID 事务、丰富的数据类型、成熟的索引体系——是理解 pgvector 为什么能做"结构化+向量混合查询"的基础。而这个混合查询能力，恰恰是 pgvector 相比 Chroma、Milvus 等独立向量数据库的最大杀手锏。

---

## 关系型数据库的核心价值

PostgreSQL 是一个关系型数据库管理系统（RDBMS），它的核心职责是**安全地存储结构化数据，并让你用 SQL 语言高效地查询和修改这些数据**。这里的"结构化"是关键词——关系型数据库要求数据有预定义的表结构（schema），每一列有明确的数据类型（整数、文本、时间戳等），每一行是一条完整的记录。

这种"先定义结构再存数据"的设计看起来很死板，但它带来了三个巨大的好处：

第一，**数据完整性保证**。你可以定义主键（PRIMARY KEY）确保每条记录唯一、定义非空约束（NOT NULL）确保关键字段不为空、定义外键（FOREIGN KEY）确保关联数据的一致性。这些约束由数据库引擎强制执行，即使你的应用代码有 bug，数据库也不会接受违反约束的数据。

第二，**SQL 查询的强大表达能力**。SQL 不只是"查数据"——它是一门完整的查询语言，支持条件过滤（WHERE）、排序（ORDER BY）、分组聚合（GROUP BY）、多表关联（JOIN）、子查询、窗口函数等。一条 SQL 可以表达非常复杂的业务逻辑，而关系型数据库的查询优化器会自动找到最高效的执行方式。

第三，**ACID 事务**。这是关系型数据库最核心的承诺——原子性（Atomicity）、一致性（Consistency）、隔离性（Isolation）、持久性（Durability）。简单来说，事务保证了一组操作要么全部成功、要么全部回滚，不会出现"转出了钱但没收到钱"这种半完成状态。对于金融、订单等关键业务，事务是不可妥协的需求。

---

## PostgreSQL vs MySQL vs SQLite

你可能会问：关系型数据库那么多，为什么 pgvector 只支持 PostgreSQL？答案在于 PostgreSQL 的三个独特能力：

### 扩展机制

PostgreSQL 拥有所有关系型数据库中最强大的扩展机制。它允许第三方开发者通过 C 语言编写扩展，向数据库注册新的数据类型、新的函数、新的索引方法。pgvector 正是利用了这个机制——它注册了一个 `vector` 数据类型、一组距离计算函数（`<->`、`<=>`、`<#>`）、以及两种向量索引方法（IVFFlat 和 HNSW）。MySQL 和 SQLite 都没有这样灵活的扩展机制，所以 pgvector 无法移植到它们上面。

### 丰富的类型系统

PostgreSQL 支持的数据类型远超 MySQL 和 SQLite。除了基本的整数、文本、时间戳，它还原生支持数组（`INTEGER[]`）、JSON/JSONB（`JSONB`）、UUID（`UUID`）、网络地址（`INET`）、范围类型（`INT4RANGE`）等。pgvector 的 `vector(N)` 类型就是在这个类型系统上扩展出来的，它和 PostgreSQL 的原生类型享有同等的地位——可以在 WHERE 条件中使用、可以在 ORDER BY 中排序、可以被索引。

### 索引架构

PostgreSQL 支持多种索引类型：B-tree（默认）、Hash、GIN（通用倒排索引）、GiST（通用搜索树）、SP-GiST、BRIN。pgvector 的 IVFFlat 和 HNSW 索引就是基于 GiST 框架实现的。这意味着 pgvector 的索引与 PostgreSQL 的查询优化器深度集成——优化器会自动决定何时使用向量索引、何时使用 B-tree 索引、何时两者组合使用。

```
┌──────────────────────────────────────────────────────────────────┐
│  PostgreSQL 的扩展架构                                            │
│                                                                  │
│  PostgreSQL 核心                                                 │
│  ├── 查询优化器（自动选择索引和执行计划）                          │
│  ├── 事务管理器（MVCC + WAL）                                    │
│  ├── 类型系统（int, text, jsonb, ...）                           │
│  └── 索引框架（B-tree, GIN, GiST, ...）                         │
│       ↑ 扩展点                                                   │
│  pgvector 扩展                                                   │
│  ├── vector(N) 数据类型                                          │
│  ├── 距离操作符（<->, <=>, <#>）                                 │
│  ├── 距离函数（l2_distance, cosine_distance, inner_product）     │
│  ├── IVFFlat 索引（基于 GiST 框架）                              │
│  └── HNSW 索引（基于 GiST 框架）                                 │
│                                                                  │
│  → pgvector 不是"外挂"，而是 PostgreSQL 的一等公民               │
│  → 向量查询和普通 SQL 查询享受同样的优化和事务保证                │
└──────────────────────────────────────────────────────────────────┘
```

---

## RAG 场景下 PostgreSQL 的独特优势

理解了 PostgreSQL 的核心能力后，我们来看看它在 RAG 系统中为什么有独特优势。

RAG 系统不只是"向量搜索"——它还需要管理用户数据、文档元信息、对话历史、访问权限等结构化数据。在独立向量数据库（如 Chroma）中，你需要同时维护两个系统：一个关系型数据库存结构化数据，一个向量库存向量数据，然后在应用层做数据同步。这种双系统架构带来了数据一致性风险和运维复杂度。

而 pgvector 让你把结构化数据和向量数据放在同一张表中，用同一条 SQL 同时查询：

```sql
-- pgvector：一条 SQL 同时做结构化过滤和向量搜索
SELECT id, content, source, embedding <=> '[0.1, 0.2, ...]' AS distance
FROM documents
WHERE category = 'tech'
  AND created_at > '2024-01-01'
  AND metadata @> '{"language": "zh"}'
ORDER BY embedding <=> '[0.1, 0.2, ...]'
LIMIT 5;
```

这条 SQL 做了三件事：第一，用 WHERE 条件过滤出 category 为 tech、创建时间在 2024 年之后、语言为中文的文档；第二，用 `<=>` 操作符计算查询向量与每条文档向量的余弦距离；第三，按距离从小到大排序，返回最相似的 5 条。整个过程是原子性的——要么全部完成，要么全部回滚，不存在"过滤了但没排序"的中间状态。

如果用 Chroma 实现同样的逻辑，你需要先在 Chroma 中用 where 过滤做向量搜索，然后在应用层用 SQL 查询补充结构化信息，最后合并两个数据源的结果。不仅代码更复杂，而且两个数据源之间的一致性也无法保证。

---

## 常见误区

### 误区 1：pgvector 可以独立使用

pgvector 是 PostgreSQL 的扩展，不是独立的数据库。你必须先安装和运行 PostgreSQL，然后在其中创建 pgvector 扩展。你不能像 Chroma 那样 `pip install chromadb` 就直接用。

### 误区 2：PostgreSQL 太重了，不适合 AI 应用

PostgreSQL 确实比 SQLite 重，但它的"重"换来的是生产级的可靠性——ACID 事务、崩溃恢复、流复制、成熟的监控工具。这些在 AI 应用的生产环境中不是"锦上添花"，而是"必需品"。而且用 Docker 启动一个 PostgreSQL 实例只需要一行命令，并不比启动 Chroma Server 复杂多少。

### 误区 3：学 pgvector 不需要学 PostgreSQL

pgvector 的所有操作都是 SQL 语句，它的所有数据都存在 PostgreSQL 的表中。如果你不理解 SQL 的基本语法、不理解 PostgreSQL 的索引和事务机制，你就无法正确使用 pgvector——就像你不会开车，再好的导航系统也帮不了你。

---

## 本章小结

PostgreSQL 是 pgvector 的运行基础，理解它的核心能力是使用 pgvector 的前提。核心要点回顾：第一，PostgreSQL 的三大核心价值是数据完整性保证、SQL 查询的强大表达力和 ACID 事务；第二，pgvector 之所以只支持 PostgreSQL，是因为 PostgreSQL 拥有最强大的扩展机制、最丰富的类型系统和最灵活的索引架构；第三，pgvector 在 RAG 中的独特优势是"结构化+向量混合查询"——一条 SQL 同时做 WHERE 过滤和向量搜索，无需维护双系统；第四，pgvector 不是独立的数据库，不能脱离 PostgreSQL 使用。

下一节我们将动手安装 PostgreSQL 和连接数据库，为后续的 pgvector 实战做好准备。
