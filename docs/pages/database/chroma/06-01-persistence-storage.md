# 6.1 持久化与存储引擎

> **数据是 RAG 系统的命脉——理解 Chroma 的持久化机制，才能确保数据不丢失、服务不断线**

---

## 这一节在讲什么？

在前面的章节中，我们大部分时候用的是 Chroma 的内存模式——程序退出后数据就没了。这对于学习和原型开发没问题，但生产环境绝不能接受"重启一次数据全丢"。Chroma 提供了两种持久化方式：嵌入式持久化（Client 模式 + persist_directory）和独立服务模式（Chroma Server）。这一节我们要深入理解这两种方式的工作原理、数据文件结构、启动速度差异，以及如何选择适合你场景的方案。

---

## 嵌入式持久化：Client 模式

这是最简单的持久化方式——只需在创建 Client 时指定一个本地目录，Chroma 会自动把数据写入磁盘：

```python
import chromadb

client = chromadb.Client(settings=chromadb.Settings(
    is_persistent=True,
    persist_directory="./my_chroma_data"
))

# 所有操作的数据都会自动持久化到 ./my_chroma_data 目录
collection = client.get_or_create_collection(name="production_kb")
collection.add(documents=["重要数据"], ids=["important_001"])

# 程序退出后，下次启动时数据仍然存在
```

### 数据文件结构

当你指定了 `persist_directory` 后，Chroma 会在该目录下创建以下文件：

```
./my_chroma_data/
├── chroma.sqlite3              # 主数据库（SQLite）
│   ├── collections 表          # Collection 元信息
│   ├── embedding_metadata 表   # Metadata 键值对
│   ├── embeddings 表           # 向量 ID 与 Collection 的映射
│   └── segments 表             # 数据段信息
│
├── xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx/
│   ├── data_level0.bin         # HNSW 索引的第 0 层数据
│   ├── data_level1.bin         # HNSW 索引的第 1 层数据（如果有的话）
│   ├── metadata.bin            # 向量元数据
│   └── ...                     # 其他索引文件
│
└── chroma.sqlite3-wal          # WAL（Write-Ahead Log）文件
```

### WAL 机制：崩溃安全的保障

Chroma 使用 SQLite 的 WAL（Write-Ahead Logging）模式来保证数据安全。WAL 的核心思想是：在修改数据库之前，先把修改操作写入日志文件（`.sqlite3-wal`），然后再执行实际的修改。如果程序在写入过程中崩溃，下次启动时 SQLite 会根据 WAL 日志恢复到一致的状态——要么完成未完成的写入，要么回滚到崩溃前的状态。

```
┌─────────────────────────────────────────────────────────────┐
│  WAL 工作流程                                               │
│                                                             │
│  1. 应用发起写操作（add/update/delete）                      │
│     ↓                                                       │
│  2. 先将操作写入 WAL 日志（.sqlite3-wal 文件）               │
│     ↓                                                       │
│  3. 执行实际的数据库修改                                     │
│     ↓                                                       │
│  4. 确认修改完成后，WAL 日志可以被回收                       │
│                                                             │
│  崩溃恢复：                                                  │
│  - 如果步骤 2 完成但步骤 3 未完成 → 重放 WAL，完成写入       │
│  - 如果步骤 2 都未完成 → 忽略不完整的 WAL 记录              │
│  → 任何时刻崩溃都不会导致数据损坏                            │
└─────────────────────────────────────────────────────────────┘
```

**WAL 文件过大的问题**：如果长时间没有触发 checkpoint（将 WAL 日志合并到主数据库），`.sqlite3-wal` 文件可能会变得很大。解决方法是定期触发 checkpoint：

```python
# 手动触发 checkpoint（通过 SQLite 命令）
import sqlite3
conn = sqlite3.connect("./my_chroma_data/chroma.sqlite3")
conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
conn.close()
```

### 冷启动 vs 热启动

冷启动是指第一次创建 Client 并加载数据，热启动是指程序重启后再次加载已有数据。两者的性能差异主要取决于数据量：

```python
import time
import chromadb

def measure_startup(persist_dir, label):
    """测量启动时间"""
    start = time.time()
    client = chromadb.Client(settings=chromadb.Settings(
        is_persistent=True,
        persist_directory=persist_dir
    ))
    collections = client.list_collections()
    elapsed = time.time() - start
    total_docs = sum(c.count() for c in collections)
    print(f"{label}: {elapsed*1000:.0f}ms ({total_docs} 条文档)")
    return client

# 首次运行（冷启动 + 首次写入）
client = measure_startup("./perf_test", "冷启动（空库）")
col = client.get_or_create_collection(name="test")
col.add(documents=[f"文档 {i}" for i in range(10000)],
        ids=[f"d{i}" for i in range(10000)])

# 再次运行（热启动：加载已有数据）
client2 = measure_startup("./perf_test", "热启动（10K文档）")
```

典型输出：

```
冷启动（空库）: 120ms (0 条文档)
热启动（10K文档）: 450ms (10000 条文档)
```

热启动比冷启动慢是因为需要从磁盘加载 HNSW 索引和 SQLite 数据。数据量越大，启动越慢——10K 条 384 维向量的热启动约 0.5 秒，100K 条约 2~3 秒，1M 条约 10~30 秒。

---

## Chroma Server 模式：独立服务

当你需要多进程/多服务共享同一个 Chroma 实例时，嵌入式模式就不适用了——SQLite 不支持多进程并发写。这时候应该使用 Chroma Server 模式，把 Chroma 作为独立的 HTTP 服务运行：

### 启动 Server

```bash
# 最简启动
chroma run --path ./chroma_server_data

# 完整参数
chroma run \
    --path ./chroma_server_data \    # 数据持久化目录
    --host 0.0.0.0 \                # 监听地址
    --port 8000                      # 监听端口
```

### 客户端连接

```python
import chromadb

# 连接远程 Chroma Server
client = chromadb.HttpClient(
    host="localhost",
    port=8000
)

# 后续操作与嵌入式模式完全相同
collection = client.get_or_create_collection(name="server_kb")
collection.add(documents=["通过 Server 添加的数据"], ids=["server_001"])

results = collection.query(query_texts=["查询"], n_results=3)
```

### Docker 部署

```yaml
# docker-compose.yml
version: "3.8"
services:
  chroma:
    image: chromadb/chroma:latest
    ports:
      - "8000:8000"
    volumes:
      - ./chroma_data:/chroma/chroma
    environment:
      - CHROMA_SERVER_HOST=0.0.0.0
      - CHROMA_SERVER_PORT=8000
    restart: unless-stopped

  # 你的应用服务
  app:
    build: .
    ports:
      - "8080:8080"
    environment:
      - CHROMA_HOST=chroma
      - CHROMA_PORT=8000
    depends_on:
      - chroma
```

```bash
# 启动
docker-compose up -d

# 查看日志
docker-compose logs -f chroma

# 停止
docker-compose down
```

### 嵌入式模式 vs Server 模式的选择

| 维度 | 嵌入式模式 | Server 模式 |
|------|-----------|------------|
| 部署复杂度 | ⭐ 极简（import 即用） | ⭐⭐ 需要单独启动服务 |
| 多进程共享 | ❌ 不支持并发写 | ✅ 通过 HTTP API 共享 |
| 网络延迟 | 无（进程内调用） | 有（HTTP 往返 ~1-5ms） |
| 适合场景 | 单进程应用、开发测试 | 多服务共享、生产环境 |
| 资源隔离 | 与应用共享内存 | 独立进程，资源隔离 |
| 水平扩展 | ❌ 不支持 | ✅ 可加负载均衡 |

**实践建议**：开发阶段用嵌入式模式（零配置、快速迭代），生产阶段用 Server 模式（支持多服务共享、易于监控和扩展）。

---

## 备份策略

数据是 RAG 系统的核心资产，必须有可靠的备份机制。Chroma 的备份相对简单——因为所有数据都在文件系统上，直接复制目录即可：

```python
import shutil
import time
import os

def backup_chroma(persist_dir: str, backup_dir: str, max_backups: int = 5):
    """
    备份 Chroma 数据目录

    参数:
        persist_dir: Chroma 数据目录
        backup_dir: 备份存放目录
        max_backups: 保留的最大备份数量
    """
    os.makedirs(backup_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(backup_dir, f"chroma_backup_{timestamp}")

    # 停止写入后再备份（确保数据一致性）
    # 如果是 Server 模式，建议先暂停写入或使用 SQLite 的 backup API
    shutil.copytree(persist_dir, backup_path)

    print(f"✅ 备份完成: {backup_path}")

    # 清理旧备份
    backups = sorted([
        d for d in os.listdir(backup_dir)
        if d.startswith("chroma_backup_")
    ])

    while len(backups) > max_backups:
        old_backup = os.path.join(backup_dir, backups.pop(0))
        shutil.rmtree(old_backup)
        print(f"🗑️ 清理旧备份: {old_backup}")

    return backup_path


# 使用
backup_chroma("./chroma_db", "./backups", max_backups=5)
```

**备份注意事项**：

1. **一致性**：备份时确保没有正在进行的写操作。最安全的做法是在低峰期备份，或者在 Server 模式下使用 SQLite 的 `backup` API 做在线备份。

2. **WAL 文件**：备份时务必包含 `.sqlite3-wal` 文件，否则可能丢失最近的写入。

3. **恢复**：恢复时只需将备份目录复制回 `persist_directory` 的位置，然后重启 Client 即可。

---

## 常见误区

### 误区 1：嵌入式模式下多进程可以安全地共享同一个 persist_directory

SQLite 的 WAL 模式支持多进程并发读，但**写操作必须串行**。如果两个进程同时写入同一个 SQLite 数据库，可能导致数据损坏。多进程场景必须使用 Server 模式。

### 误区 2：Server 模式比嵌入式模式性能更好

不一定。Server 模式引入了 HTTP 网络开销（每次请求 ~1-5ms），对于低频查询场景，嵌入式模式反而更快。Server 模式的优势在于共享和隔离，而不是单次请求的延迟。

### 误区 3：备份只需要复制 .sqlite3 文件

不完整。HNSW 索引数据存储在单独的二进制文件中（`data_level0.bin` 等），如果只备份 `.sqlite3` 文件，恢复后向量索引会丢失，需要重建。务必备份整个 `persist_directory` 目录。

---

## 本章小结

持久化是 Chroma 从开发玩具升级为生产基础设施的第一步。核心要点回顾：第一，嵌入式持久化只需指定 `persist_directory`，Chroma 自动管理数据文件和 WAL 日志；第二，WAL 机制保证崩溃安全，但长时间运行后 WAL 文件可能过大，需要定期 checkpoint；第三，热启动比冷启动慢，因为需要从磁盘加载索引，10K 条向量约 0.5 秒，100K 条约 2~3 秒；第四，Server 模式支持多进程共享，适合生产环境，但引入了 HTTP 网络开销；第五，备份只需复制整个 persist_directory，但必须确保备份时没有正在进行的写操作。

下一节我们将深入性能基准与调优——如何测量 Chroma 的性能、影响性能的关键因素、以及具体的优化手段。
