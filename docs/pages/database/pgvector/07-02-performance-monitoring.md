# 7.2 性能调优与监控

> **调优不是玄学——是理解参数含义、读懂监控指标、对症下药的系统工程**

---

## 这一节在讲什么？

上一节我们把 pgvector 部署到了生产环境，但"能跑"和"跑得快"是两回事。你可能会遇到这样的场景：开发环境里 10 万条数据的向量搜索只要 5 毫秒，上了生产环境同样的查询却要 200 毫秒——为什么？可能是因为 shared_buffers 太小导致频繁换页，可能是因为 work_mem 不够导致排序溢出到磁盘，可能是因为 HNSW 的 ef_search 设得太高，也可能是因为索引根本没被用到。这一节我们要系统地聊一聊 pgvector 的性能调优——从 PostgreSQL 的核心参数到 pgvector 的专用参数，从监控工具到常见性能问题的排查方法，帮你建立一套"发现问题 → 定位原因 → 解决问题"的完整方法论。

---

## PostgreSQL 核心参数调优

PostgreSQL 有上百个可配置参数，但跟 pgvector 性能密切相关的只有几个。理解这些参数的含义比记住它们的数值更重要——因为不同的数据量、不同的硬件配置、不同的查询模式，最优参数是完全不同的。

### shared_buffers：共享缓冲区

`shared_buffers` 是 PostgreSQL 最重要的参数——它决定了 PostgreSQL 能在内存中缓存多少数据页面。当查询需要读取数据时，PostgreSQL 先在 shared_buffers 里找，找到了就直接返回（内存命中），找不到就从磁盘读取（缓存未命中）。内存命中的速度是磁盘读取的 1000 倍以上，所以 shared_buffers 越大，缓存命中率越高，查询越快。

```
shared_buffers 的工作原理：

  查询请求
     │
     ▼
  ┌─────────────────────────────────┐
  │        shared_buffers           │
  │   ┌─────┐ ┌─────┐ ┌─────┐     │
  │   │Page1│ │Page2│ │Page3│ ... │  ← 命中 → 直接返回（微秒级）
  │   └─────┘ └─────┘ └─────┘     │
  └──────────────┬──────────────────┘
                 │ 未命中
                 ▼
  ┌─────────────────────────────────┐
  │        操作系统页面缓存          │  ← 命中 → 返回（微秒级）
  └──────────────┬──────────────────┘
                 │ 未命中
                 ▼
  ┌─────────────────────────────────┐
  │           磁盘 I/O              │  ← 必须读磁盘（毫秒级）
  └─────────────────────────────────┘
```

```bash
# postgresql.conf
# 推荐值：物理内存的 25%
# 16GB 内存 → 4GB
# 32GB 内存 → 8GB
# 64GB 内存 → 16GB
shared_buffers = 4GB
```

对于 pgvector 来说，shared_buffers 尤其重要——HNSW 索引是全内存访问的，如果 shared_buffers 放不下整个 HNSW 索引，PostgreSQL 就不得不频繁地从磁盘读取索引页面，查询延迟会从毫秒级飙升到百毫秒级。一个简单的估算方法是：**shared_buffers 应该至少是 HNSW 索引大小的 1.5 倍**（索引本身 + 预留空间 + 其他热数据）。

### work_mem：排序和哈希操作的内存

`work_mem` 决定了每个排序操作（ORDER BY）和哈希操作（HASH JOIN / HASH AGGREGATE）能使用的最大内存。当操作需要的内存超过 work_mem 时，PostgreSQL 会把中间结果溢出到磁盘（temp files），这会导致性能急剧下降。

```bash
# postgresql.conf
# 默认值 4MB 太小了，生产环境建议 16MB~64MB
# 但不要设太大——每个排序操作都会独立分配 work_mem
# 如果 max_connections=100 且每个连接有 2 个排序操作，总内存 = 100 × 2 × work_mem
work_mem = 32MB
```

对于 pgvector 的向量搜索来说，`ORDER BY embedding <=> query_vec LIMIT K` 这个操作需要计算所有候选向量与查询向量的距离然后排序——如果候选集很大，排序过程可能需要大量内存。如果 work_mem 不够，排序溢出到磁盘，查询就会变慢。

你可以通过以下查询检查是否有排序溢出到磁盘的情况：

```sql
-- 检查是否有排序溢出到磁盘的查询
SELECT query, calls, total_time, rows,
       shared_blks_hit, shared_blks_read,
       temp_blks_read, temp_blks_written
FROM pg_stat_statements
WHERE temp_blks_written > 0
ORDER BY temp_blks_written DESC
LIMIT 10;
```

如果 `temp_blks_written > 0`，说明有排序溢出到磁盘了，你需要增大 work_mem 或者优化查询减少排序的数据量。

### effective_cache_size：查询计划器的参考值

`effective_cache_size` 不是一个真正分配内存的参数——它是告诉 PostgreSQL 查询计划器"操作系统大概有多少内存可以用来缓存数据页面"。查询计划器会根据这个值来决定是使用索引扫描还是全表扫描——如果它认为数据大概率已经在操作系统缓存中了，就更倾向于使用索引扫描。

```bash
# postgresql.conf
# 推荐值：物理内存的 50%~75%
# 16GB 内存 → 12GB
# 32GB 内存 → 24GB
effective_cache_size = 12GB
```

这个参数对 pgvector 的索引选择有直接影响——如果 effective_cache_size 设得太小，查询计划器可能认为 HNSW 索引的随机 I/O 太贵，转而选择全表扫描（顺序 I/O），这反而更慢。

### 常见误区：把所有参数都设到最大

有些同学觉得"参数越大越好"，把 shared_buffers 设成物理内存的 80%、work_mem 设成 1GB——这反而会出问题。shared_buffers 太大会挤占操作系统的页面缓存空间（PostgreSQL 依赖操作系统缓存来加速数据读取），work_mem 太大在并发高时会导致内存耗尽（每个连接的每个排序操作都会独立分配 work_mem）。参数调优的核心是**平衡**——在 PostgreSQL 缓存和操作系统缓存之间找平衡，在单个操作的内存和并发数之间找平衡。

---

## pgvector 专用参数调优

除了 PostgreSQL 的通用参数，pgvector 还有两个专用的搜索参数——它们直接控制向量搜索的精度和速度之间的权衡。

### ivfflat.probes：IVFFlat 搜索精度

```sql
-- 设置 IVFFlat 搜索时扫描的聚类区域数量
SET ivfflat.probes = 10;       -- 当前会话生效
-- 或者
ALTER DATABASE ragdb SET ivfflat.probes = 10;  -- 数据库级别默认值
```

probes 的值越大，搜索时扫描的聚类区域越多，召回率越高但速度越慢。一个经验公式是 `probes = lists × 5%~10%`——如果你建索引时设了 `lists = 100`，那么 `probes = 5~10` 是合理的起点。

```
probes 对性能和召回率的影响（lists=100, 100万条数据）：

  probes=1    → 速度极快，召回率 ~60%（太低，不实用）
  probes=5    → 速度快，召回率 ~85%（对精度要求不高的场景可用）
  probes=10   → 速度较快，召回率 ~92%（推荐默认值）
  probes=50   → 速度中等，召回率 ~97%（精度优先场景）
  probes=100  → 速度慢，召回率 ~99%（等同于暴力搜索）
```

### hnsw.ef_search：HNSW 搜索精度

```sql
-- 设置 HNSW 搜索时的搜索宽度
SET hnsw.ef_search = 100;       -- 当前会话生效
-- 或者
ALTER DATABASE ragdb SET hnsw.ef_search = 100;  -- 数据库级别默认值
```

ef_search 的值越大，搜索时探索的图节点越多，召回率越高但速度越慢。pgvector 的默认值是 40，对于大多数场景来说偏保守。

```
ef_search 对性能和召回率的影响（m=16, 100万条数据）：

  ef_search=40   → 速度快，召回率 ~90%（默认值，精度一般）
  ef_search=100  → 速度较快，召回率 ~96%（推荐默认值）
  ef_search=200  → 速度中等，召回率 ~98%（精度优先场景）
  ef_search=500  → 速度慢，召回率 ~99.5%（极致精度）
```

### 动态调整策略

在实际生产中，你可能需要根据不同的业务场景动态调整这些参数——比如用户搜索时用较低的 probes/ef_search 保证响应速度，后台批量处理时用较高的值保证召回率：

```python
import psycopg2
from pgvector.psycopg2 import register_vector

conn = psycopg2.connect(DATABASE_URL)
register_vector(conn)
cur = conn.cursor()

def search_documents(query_embedding, top_k=5, precision="balanced"):
    """根据精度需求动态调整搜索参数"""
    if precision == "fast":
        cur.execute("SET hnsw.ef_search = 40")
    elif precision == "balanced":
        cur.execute("SET hnsw.ef_search = 100")
    elif precision == "precise":
        cur.execute("SET hnsw.ef_search = 200")

    cur.execute("""
        SELECT id, content, embedding <=> %s AS distance
        FROM documents
        ORDER BY embedding <=> %s
        LIMIT %s
    """, (query_embedding, query_embedding, top_k))
    return cur.fetchall()

# 用户实时搜索 → 快速响应，精度可以稍微牺牲
results = search_documents(query_emb, precision="fast")

# 后台推荐系统 → 精度优先，速度可以慢一点
results = search_documents(query_emb, precision="precise")
```

### 常见误区：probes/ef_search 设得越高越好

很多同学觉得"召回率越高越好"，把 probes 设成 lists、ef_search 设成 1000——这等于放弃了索引的意义，退化为暴力搜索。在实际应用中，95% 的召回率通常已经足够——因为 RAG 系统的最终质量取决于 LLM 的生成能力，而不是检索的召回率。一个 Top-5 结果里漏掉的那 1 条，对 LLM 的回答质量影响微乎其微，但搜索速度从 5ms 变成 50ms 的体验差异却是用户能直接感知的。

---

## 监控工具

调优的前提是监控——你得先知道系统现在的状态，才能判断哪里需要优化。PostgreSQL 提供了丰富的内置监控视图和扩展，配合 Prometheus + Grafana 可以实现完整的可观测性。

### pg_stat_statements：慢查询分析

`pg_stat_statements` 是 PostgreSQL 最有用的监控扩展——它记录了每条 SQL 语句的执行统计信息，包括调用次数、总耗时、平均耗时、返回行数、缓存命中/未命中等。它是定位慢查询的第一工具。

```sql
-- 启用 pg_stat_statements 扩展
CREATE EXTENSION pg_stat_statements;

-- 在 postgresql.conf 中配置
-- shared_preload_libraries = 'pg_stat_statements'
-- pg_stat_statements.track = all
-- 重启 PostgreSQL 后生效
```

```sql
-- 查询最慢的 10 条 SQL
SELECT query, calls, total_exec_time, mean_exec_time,
       rows, shared_blks_hit, shared_blks_read
FROM pg_stat_statements
ORDER BY mean_exec_time DESC
LIMIT 10;

-- 查询最耗时的 10 条 SQL（总时间）
SELECT query, calls, total_exec_time,
       round(total_exec_time / calls, 2) AS avg_time_ms
FROM pg_stat_statements
ORDER BY total_exec_time DESC
LIMIT 10;

-- 专门查看向量搜索相关的慢查询
SELECT query, calls, mean_exec_time, rows
FROM pg_stat_statements
WHERE query LIKE '%embedding%'
ORDER BY mean_exec_time DESC
LIMIT 10;
```

### pg_stat_activity：连接和锁监控

`pg_stat_activity` 显示当前所有连接的状态——谁在连接、在执行什么查询、等什么锁。当系统出现"卡住"的情况时，这个视图是排查问题的第一站。

```sql
-- 查看当前活跃的连接
SELECT pid, usename, datname, state, query, query_start,
       now() - query_start AS duration
FROM pg_stat_activity
WHERE state != 'idle'
ORDER BY duration DESC;

-- 查看等待锁的连接
SELECT pid, usename, query, wait_event_type, wait_event
FROM pg_stat_activity
WHERE wait_event_type IS NOT NULL
  AND wait_event_type != 'Client';

-- 查看连接数统计
SELECT state, count(*)
FROM pg_stat_activity
GROUP BY state
ORDER BY count DESC;
```

比如你发现有很多连接处于 `active` 状态且 duration 很长，说明有慢查询在阻塞——你可以用 `pg_terminate_backend(pid)` 终止问题查询：

```sql
-- 终止一个慢查询（谨慎使用！）
SELECT pg_terminate_backend(12345);  -- 12345 是 pid
```

### pg_stat_user_indexes：索引使用率

这个视图告诉你每个索引被使用了多少次——如果一个索引从来没被使用过，它就是在浪费存储空间和写入性能。

```sql
-- 查看索引使用率
SELECT schemaname, relname AS table_name, indexrelname AS index_name,
       idx_scan AS index_scans, idx_tup_read, idx_tup_fetch
FROM pg_stat_user_indexes
ORDER BY idx_scan ASC;

-- 找出从未被使用的索引（建了但没用上）
SELECT schemaname, relname AS table_name, indexrelname AS index_name
FROM pg_stat_user_indexes
WHERE idx_scan = 0
  AND indexrelname NOT LIKE '%pkey%';  -- 排除主键索引
```

对于 pgvector 来说，如果你发现向量索引的 `idx_scan = 0`，说明查询根本没有走索引——要么是数据量太小（PostgreSQL 选择了全表扫描），要么是操作符不匹配（索引用了 `vector_cosine_ops` 但查询用了 `<->` 操作符）。

### Prometheus + postgres_exporter + Grafana

对于生产环境，你需要一个持续运行的监控系统——不能等问题发生了再去手动查 SQL。Prometheus + postgres_exporter + Grafana 是 PostgreSQL 监控的标准组合：

```yaml
# docker-compose.yml（追加监控服务）
services:
  postgres_exporter:
    image: quay.io/prometheuscommunity/postgres-exporter
    environment:
      DATA_SOURCE_NAME: "postgresql://appuser:${PG_PASSWORD}@postgres:5432/ragdb?sslmode=disable"
    ports:
      - "9187:9187"

  prometheus:
    image: prom/prometheus
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana
    volumes:
      - grafana_data:/var/lib/grafana
    ports:
      - "3000:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: ${GRAFANA_PASSWORD}
```

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: postgres
    static_configs:
      - targets: ['postgres_exporter:9187']
```

在 Grafana 中，你可以导入 PostgreSQL Dashboard（ID: 9628），它会自动展示连接数、QPS、缓存命中率、慢查询等关键指标。对于 pgvector，你还需要关注几个自定义指标：

```sql
-- 自定义监控指标：HNSW 索引的 ef_search 设置
SHOW hnsw.ef_search;

-- 自定义监控指标：向量表的行数和索引大小
SELECT relname AS table_name,
       pg_size_pretty(pg_relation_size(relid)) AS table_size,
       pg_size_pretty(pg_indexes_size(relid)) AS index_size
FROM pg_stat_user_tables
WHERE relname IN ('documents', 'conversations', 'user_preferences');
```

---

## 常见性能问题与排查

### 问题1：索引没有被使用

这是 pgvector 生产环境中最常见的问题——你明明建了 HNSW 索引，但查询还是走全表扫描。

**排查步骤：**

```sql
-- 第一步：用 EXPLAIN ANALYZE 确认索引是否被使用
EXPLAIN ANALYZE
SELECT id, content, embedding <=> '[0.1, 0.2, ...]' AS distance
FROM documents
ORDER BY embedding <=> '[0.1, 0.2, ...]'
LIMIT 5;

-- 如果看到 "Seq Scan on documents" → 索引没被使用
-- 如果看到 "Index Scan using idx_doc_embedding" → 索引被使用了
```

**可能的原因和解决方案：**

| 原因 | 症状 | 解决方案 |
|------|------|---------|
| 操作符不匹配 | 索引用了 `vector_cosine_ops` 但查询用了 `<->` | 统一操作符：建索引和查询用同一套 ops |
| 数据量太小 | 表只有几百条数据 | 不需要索引，全表扫描更快 |
| 查询计划器判断错误 | 数据量大但计划器选择了 Seq Scan | `SET enable_seqscan = off` 强制走索引试试 |
| 索引损坏 | 极少见，但可能发生 | `REINDEX INDEX idx_doc_embedding` |

```sql
-- 临时禁用全表扫描，强制走索引（仅用于诊断！）
SET enable_seqscan = off;
EXPLAIN ANALYZE
SELECT * FROM documents ORDER BY embedding <=> '[0.1,...]' LIMIT 5;
-- 如果走索引后变快了 → 查询计划器判断有误，考虑调整 effective_cache_size
-- 如果走索引后反而变慢 → 索引确实不合适，检查操作符匹配
```

### 问题2：查询越来越慢

系统刚上线时查询很快，运行一段时间后越来越慢——这是典型的"表膨胀"或"索引膨胀"问题。

**排查步骤：**

```sql
-- 检查表和索引的膨胀率
SELECT schemaname, relname AS table_name,
       pg_size_pretty(pg_relation_size(relid)) AS total_size,
       n_dead_tup, n_live_tup,
       round(n_dead_tup::numeric / NULLIF(n_live_tup + n_dead_tup, 0) * 100, 2) AS bloat_pct
FROM pg_stat_user_tables
ORDER BY n_dead_tup DESC;

-- 检查是否需要 VACUUM
SELECT relname, last_vacuum, last_autovacuum,
       n_dead_tup
FROM pg_stat_user_tables
WHERE n_dead_tup > 10000
ORDER BY n_dead_tup DESC;
```

**解决方案：**

```sql
-- 手动 VACUUM（不锁表，清理死元组）
VACUUM documents;

-- 手动 VACUUM FULL（锁表重建整张表，彻底消除膨胀）
-- ⚠️ 生产环境慎用！会锁住整张表直到完成
VACUUM FULL documents;

-- 重建索引（不锁表，在线重建）
REINDEX INDEX CONCURRENTLY idx_doc_embedding;
```

PostgreSQL 的自动 VACUUM 机制会在后台定期清理死元组，但默认配置比较保守。对于频繁更新的向量表，你可能需要调高自动 VACUUM 的频率：

```bash
# postgresql.conf
autovacuum_vacuum_scale_factor = 0.05   # 默认 0.2，死元组达到 5% 就触发 VACUUM
autovacuum_analyze_scale_factor = 0.02  # 默认 0.1，变化达到 2% 就更新统计信息
```

### 问题3：内存不足导致查询失败

当 HNSW 索引太大、shared_buffers 放不下时，查询可能会因为内存不足而失败或变得极慢。

**排查步骤：**

```sql
-- 检查索引大小
SELECT indexrelname AS index_name,
       pg_size_pretty(pg_relation_size(indexrelid)) AS index_size
FROM pg_stat_user_indexes
WHERE indexrelname LIKE '%embedding%';

-- 检查 shared_buffers 的使用情况
SELECT pg_size_pretty(pg_relation_size('idx_doc_embedding')) AS hnsw_index_size,
       (SELECT setting FROM pg_settings WHERE name = 'shared_buffers') AS shared_buffers_pages,
       pg_size_pretty(
           (SELECT setting::bigint FROM pg_settings WHERE name = 'shared_buffers') * 8192
       ) AS shared_buffers_size;
```

**解决方案：**

```bash
# 如果 HNSW 索引 > shared_buffers 的 50%，需要增大 shared_buffers
# 或者考虑换用 IVFFlat（内存占用更小）
shared_buffers = 8GB
```

如果物理内存确实不够，你还有一个选择——把 HNSW 换成 IVFFlat。IVFFlat 的索引不需要全内存驻留，它的内存占用大约只有 HNSW 的 1/3~1/2，代价是查询速度稍慢、召回率稍低、不支持增量更新。

### 问题4：批量插入时索引构建太慢

当你需要一次性插入大量向量数据（比如几十万条），HNSW 索引的增量更新会非常慢——每插入一条数据，HNSW 都要更新图结构，这比没有索引时慢 5~10 倍。

**解决方案：先删索引、批量插入、再重建索引**

```python
import psycopg2
from pgvector.psycopg2 import register_vector

conn = psycopg2.connect(DATABASE_URL)
register_vector(conn)
cur = conn.cursor()

def bulk_insert_with_reindex(embeddings_data):
    """大批量插入的最佳实践：先删索引再重建"""

    # 第一步：删除向量索引
    print("Dropping index...")
    cur.execute("DROP INDEX IF EXISTS idx_doc_embedding")
    conn.commit()

    # 第二步：批量插入数据（没有索引时插入速度最快）
    print("Inserting data...")
    for batch in chunks(embeddings_data, 1000):
        cur.executemany(
            "INSERT INTO documents (source, content, embedding) VALUES (%s, %s, %s)",
            batch
        )
    conn.commit()

    # 第三步：重建索引
    print("Rebuilding index...")
    cur.execute("""
        CREATE INDEX idx_doc_embedding
        ON documents USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    """)
    conn.commit()
    print("Done!")

def chunks(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]
```

这个策略对于 IVFFlat 尤其重要——因为 IVFFlat 的索引质量依赖于数据分布，先插入所有数据再建索引，聚类中心才能准确反映数据的真实分布。如果你在空表上建 IVFFlat 索引然后逐条插入，索引质量会很差。

---

## 一份生产环境的参数模板

以下是一份适用于 16GB 内存、4 核 CPU 的生产环境参数模板，你可以根据自己的硬件配置按比例调整：

```bash
# postgresql.conf — pgvector 生产环境参数模板

# ── 内存相关 ──
shared_buffers = 4GB              # 物理内存的 25%
work_mem = 32MB                   # 排序/哈希操作的内存
effective_cache_size = 12GB       # 物理内存的 75%
maintenance_work_mem = 512MB      # VACUUM/CREATE INDEX 使用的内存

# ── WAL 相关 ──
wal_buffers = 64MB                # WAL 缓冲区
min_wal_size = 1GB                # WAL 文件最小大小
max_wal_size = 4GB                # WAL 文件最大大小
checkpoint_completion_target = 0.9  # 检查点分散写入

# ── 连接相关 ──
max_connections = 100             # 配合 pgbouncer 使用
superuser_reserved_connections = 3  # 超级用户保留连接

# ── 自动 VACUUM ──
autovacuum = on
autovacuum_vacuum_scale_factor = 0.05
autovacuum_analyze_scale_factor = 0.02
autovacuum_max_workers = 4

# ── 查询计划 ──
random_page_cost = 1.1            # SSD 环境下调低（默认 4.0 是为 HDD 设计的）
effective_io_concurrency = 200    # SSD 的并发 I/O 能力

# ── 日志相关 ──
log_min_duration_statement = 500  # 记录超过 500ms 的慢查询
log_checkpoints = on
log_lock_waits = on

# ── pgvector 相关（数据库级别设置）──
# ALTER DATABASE ragdb SET hnsw.ef_search = 100;
# ALTER DATABASE ragdb SET ivfflat.probes = 10;
```

其中 `random_page_cost = 1.1` 这个参数对 pgvector 特别重要——默认值 4.0 是为机械硬盘设计的，它告诉查询计划器"随机 I/O 比顺序 I/O 贵 4 倍"。但在 SSD 上，随机 I/O 和顺序 I/O 的差距很小，如果还用 4.0，查询计划器会过度回避索引扫描（因为索引扫描是随机 I/O），导致本该走索引的查询走了全表扫描。

---

## 小结

这一节我们覆盖了 pgvector 性能调优的三个层面：PostgreSQL 核心参数（shared_buffers、work_mem、effective_cache_size）决定了数据库的基础性能，pgvector 专用参数（ivfflat.probes、hnsw.ef_search）控制了向量搜索的精度和速度权衡，监控工具（pg_stat_statements、pg_stat_activity、Prometheus + Grafana）帮你发现和定位问题。调优不是一锤子买卖——你需要持续监控、持续调整、持续验证。下一节是本教程的最后一节，我们要聊一聊 pgvector 的局限性和替代方案——什么时候该用 pgvector、什么时候该换、以及如何让 pgvector 和其他方案共存。
