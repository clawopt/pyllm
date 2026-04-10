# 2.3 事务与并发控制

> **事务是数据库的"原子护盾"——要么全部成功，要么全部回滚，不存在中间状态**

---

## 这一节在讲什么？

当你往数据库里插入一条文档时，实际上可能涉及多个操作：插入文档内容、更新分类统计、记录操作日志。如果中间某一步失败了怎么办？没有事务的话，你可能会遇到"文档插入了但统计没更新"的不一致状态。事务就是解决这个问题的——它把一组操作打包成一个原子单元，要么全部成功提交，要么全部回滚到操作前的状态。这一节我们要理解 PostgreSQL 的事务机制、隔离级别、MVCC 原理，以及为什么事务对 RAG 系统特别重要——因为 pgvector 让结构化数据和向量数据在同一张表中，事务保证了它们的一致性。

---

## ACID 事务基础

ACID 是事务的四个核心保证：

| 属性 | 含义 | RAG 场景举例 |
|------|------|-------------|
| **A**tomicity（原子性） | 事务中的操作要么全部成功，要么全部回滚 | 文档内容和向量必须同时入库，不能只入库一个 |
| **C**onsistency（一致性） | 事务前后数据库从一个一致状态转到另一个一致状态 | 分类统计必须与实际文档数一致 |
| **I**solation（隔离性） | 并发事务之间互不干扰 | 两个用户同时插入文档不会互相覆盖 |
| **D**urability（持久性） | 提交后的数据不会丢失（即使断电） | 已入库的文档不会因为服务器重启而消失 |

### 基本用法

```sql
-- 开始事务
BEGIN;

-- 执行多个操作
INSERT INTO documents (content, category) VALUES ('新文档', 'tech');
UPDATE doc_stats SET count = count + 1 WHERE category = 'tech';

-- 如果一切正常，提交事务
COMMIT;

-- 如果出现问题，回滚事务
-- ROLLBACK;
```

### Python 中的事务控制

psycopg2 默认在事务中执行所有操作。你可以显式控制事务的提交和回滚：

```python
import psycopg2

conn = psycopg2.connect("postgres://postgres:password@localhost/rag_db")
cur = conn.cursor()

try:
    # 开始事务（psycopg2 自动开始）
    cur.execute("INSERT INTO documents (content, category) VALUES (%s, %s)", ("文档A", "tech"))
    cur.execute("UPDATE doc_stats SET count = count + 1 WHERE category = %s", ("tech",))

    # 一切正常，提交
    conn.commit()
    print("✅ 事务提交成功")
except Exception as e:
    # 出现错误，回滚
    conn.rollback()
    print(f"❌ 事务回滚: {e}")
finally:
    cur.close()
    conn.close()
```

### 事务对 RAG 的重要性

在 pgvector 的 RAG 系统中，事务的价值尤为突出。当你入库一篇文档时，通常需要同时写入文档内容（content 列）和向量数据（embedding 列）。如果没有事务，可能出现"内容写入了但向量没写入"的不一致状态——后续查询时这条文档可以被 WHERE 条件命中，但无法参与向量排序，导致搜索结果异常。有了事务，内容和向量要么同时可见，要么同时不可见，不存在中间状态：

```python
def insert_document_with_embedding(cur, content, embedding, metadata):
    """事务保证文档内容和向量的原子性"""
    cur.execute(
        """INSERT INTO documents (content, embedding, metadata)
           VALUES (%s, %s, %s)""",
        (content, embedding, psycopg2.extras.Json(metadata))
    )
    # 如果 embedding 参数有问题（如维度不匹配），上面的 INSERT 会失败
    # 整个事务回滚，不会留下"有内容没向量"的半成品记录
```

---

## 隔离级别

当多个事务并发执行时，隔离级别决定了它们之间的可见性规则。PostgreSQL 支持四种隔离级别：

| 隔离级别 | 脏读 | 不可重复读 | 幻读 | 性能 |
|----------|------|-----------|------|------|
| READ UNCOMMITTED | 不会* | 不会* | 会 | - |
| READ COMMITTED（默认） | 不会 | 会 | 会 | 最好 |
| REPEATABLE READ | 不会 | 不会 | 不会* | 中等 |
| SERIALIZABLE | 不会 | 不会 | 不会 | 最差 |

*PostgreSQL 的 READ UNCOMMITTED 实际上等同于 READ COMMITTED；REPEATABLE READ 在 PostgreSQL 中已经防止了幻读。

**大多数 RAG 应用使用默认的 READ COMMITTED 就够了**。只有在需要严格一致性的场景（如金融交易）才需要更高级别的隔离。

```sql
-- 设置隔离级别
SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;

BEGIN;
SELECT * FROM documents WHERE category = 'tech';
-- 在这个事务中，即使其他事务修改了数据，你看到的仍然是事务开始时的快照
COMMIT;
```

---

## MVCC 机制：读写不互斥的秘密

PostgreSQL 使用 MVCC（Multi-Version Concurrency Control，多版本并发控制）来实现事务隔离。MVCC 的核心思想是：**读操作不阻塞写操作，写操作不阻塞读操作**。

```
┌─────────────────────────────────────────────────────────────────┐
│  MVCC 工作原理                                                   │
│                                                                 │
│  每行数据有两个隐藏列：                                          │
│  - xmin: 插入该行的事务 ID                                      │
│  - xmax: 删除/更新该行的事务 ID（如果为空表示行仍然有效）         │
│                                                                 │
│  事务 A（读取）：                                                │
│    只看到 xmin <= 当前事务ID 且 xmax 为空或 > 当前事务ID 的行     │
│    → 即使事务 B 正在修改数据，事务 A 仍然看到旧版本               │
│                                                                 │
│  事务 B（更新 row 1）：                                          │
│    1. 设置旧 row 1 的 xmax = 事务B的ID                          │
│    2. 插入新 row 1（xmin = 事务B的ID, xmax = 空）               │
│    → 旧版本对事务 A 仍然可见，新版本对事务 B 可见                │
│    → 读写不互斥！                                                │
│                                                                 │
│  VACUUM：                                                       │
│    当旧版本不再被任何事务需要时，VACUUM 进程回收空间              │
│    → 如果不定期 VACUUM，表会越来越膨胀                           │
└─────────────────────────────────────────────────────────────────┘
```

MVCC 对 pgvector 的直接影响是：当你更新一条文档的 embedding 时，旧版本的向量数据不会立即被删除——它仍然对正在进行中的查询可见。这意味着频繁更新向量数据会导致表膨胀（bloat），需要定期执行 VACUUM 清理：

```sql
-- 手动触发 VACUUM
VACUUM documents;

-- 更彻底的清理（回收空间给操作系统）
VACUUM FULL documents;  -- ⚠️ 会锁表，生产环境慎用

-- 自动 VACUUM（PostgreSQL 默认开启）
-- 当表中超过一定比例的死行时自动触发
-- 可以通过 ALTER TABLE 调整阈值
ALTER TABLE documents SET (autovacuum_vacuum_scale_factor = 0.1);
```

---

## 常见误区

### 误区 1：psycopg2 的 autocommit 模式

psycopg2 默认不是 autocommit——每个 SQL 语句都在事务中执行，必须手动 commit。如果你想要 autocommit（每条 SQL 自动提交），需要显式设置：

```python
conn.autocommit = True  # 每条 SQL 自动提交，不需要手动 commit
```

在 RAG 应用中，建议保持默认的事务模式，只在需要原子性的操作组中使用 BEGIN/COMMIT。

### 误区 2：VACUUM FULL 是日常维护操作

VACUUM FULL 会锁表并重写整个表，在生产环境中可能导致长时间不可用。日常维护应该用普通的 VACUUM（不锁表），只在表膨胀严重时才考虑 VACUUM FULL。

### 误区 3：事务能解决所有并发问题

事务保证了原子性和隔离性，但不保证性能。如果多个事务同时修改同一行数据，其中一个会等待另一个完成（行锁）。在高并发写入场景下，事务等待可能导致性能瓶颈。

---

## 本章小结

事务是数据库可靠性的基石，也是 pgvector 在 RAG 系统中保证数据一致性的关键机制。核心要点回顾：第一，ACID 事务保证了一组操作的原子性——要么全部成功，要么全部回滚；第二，pgvector 的文档入库需要在同一事务中写入内容和向量，避免出现"有内容没向量"的不一致状态；第三，PostgreSQL 默认使用 READ COMMITTED 隔离级别，大多数 RAG 应用足够；第四，MVCC 机制实现了读写不互斥，但频繁更新会导致表膨胀，需要定期 VACUUM；第五，psycopg2 默认在事务中执行，必须手动 commit。

下一章我们将正式进入 pgvector 的世界——安装扩展、创建 vector 列、执行第一次向量搜索！
