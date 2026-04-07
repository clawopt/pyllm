# 7.3 持久化后端选择与生产部署

> MemorySaver 虽然在开发和演示中非常方便，但它有一个致命的局限性——**数据只存在于进程内存中，进程重启后全部丢失**。在生产环境中，你需要一个真正的持久化后端来存储 checkpoint 数据。LangGraph 提供了多种 checkpointer 实现，从简单的文件存储到完整的数据库方案都有覆盖。这一节我们会逐一分析各种持久化后端的特性、适用场景和配置方式。

## Checkpointer 后端全景

LangGraph 的 checkpointer 体系基于统一的抽象接口 `BaseCheckpointSaver`，所有具体实现都遵循相同的 API。目前可用的后端包括：

| 后端 | 类名 | 适用场景 | 持久性 | 性能 | 并发支持 |
|------|------|---------|--------|------|---------|
| 内存 | MemorySaver | 开发/测试/演示 | ❌ 进程重启丢失 | ⭐⭐⭐⭐⭐ | 单进程 |
| SQLite | SqliteSaver | 小型生产/单机部署 | ✅ 文件级持久 | ⭐⭐⭐⭐ | 单写多读 |
| PostgreSQL | PostgresSaver | 中大型生产环境 | ✅ 数据库级持久 | ⭐⭐⭐⭐ | 完整并发 |
| Redis | RedisSaver | 高性能/缓存优先 | ⚠️ 需要配持久化 | ⭐⭐⭐⭐⭐ | 完整并发 |

## SQLite Saver：轻量级文件持久化

SQLite 是最轻量级的持久化方案——它不需要任何额外的服务（如数据库服务器），只需要一个本地文件就能工作。适合小型应用、单机部署、或者作为开发阶段向数据库方案的过渡。

```python
from langgraph.checkpoint.sqlite import SqliteSaver

# 方式一：使用内存中的 SQLite（测试用）
sqlite_checkpointer = SqliteSaver.from_conn_string(":memory:")

# 方式二：使用文件（推荐用于小型生产）
import os
db_path = "/tmp/langgraph_checkpoints.db"
sqlite_checkpointer = SqliteSaver.from_conn_string(f"sqlite:///{db_path}")

app = your_graph.compile(checkpointer=sqlite_checkpointer)
config = {"configurable": {"thread_id": "prod-session-001"}}

result = app.invoke(initial_state, config=config)

# 重启进程后，checkpoint 仍然存在
# 新的进程可以恢复之前的执行
new_app = your_graph.compile(checkpointer=SqliteSaver.from_conn_string(f"sqlite:///{db_path}"))
recovered_result = new_app.invoke(initial_state, config=config)

# 清理
if os.path.exists(db_path):
    os.remove(db_path)
```

SQLite Saver 的优势在于**零依赖**——你不需要安装、配置或维护任何外部服务。它的缺点是**不支持高并发写入**（SQLite 的写入锁是数据库级别的），所以不适合多实例并行写入的场景。另外，SQLite 文件不能跨机器共享，这意味着如果你有多台服务器运行同一个图，它们无法共享 checkpoint。

## PostgreSQL Saver：生产级数据库方案

对于中大型生产环境，PostgreSQL 是最推荐的 checkpoint 后端。它提供了完整的 ACID 事务保证、优秀的并发性能、以及丰富的生态系统工具（备份、复制、监控等）。

```python
from langgraph.checkpoint.postgres import PostgresSaver
import asyncpg

# 方式一：直接连接字符串
postgres_uri = "postgresql://user:password@localhost:5432/langgraph_cp"

async def setup_postgres():
    postgres_checkpointer = await PostgresSaver.from_conn_string(postgres_uri)
    
    app = your_graph.compile(checkpointer=postgres_checkpointer)
    config = {"configurable": {"thread_id": "session-pg-001"}}
    
    result = await app.ainvoke(initial_state, config=config)
    
    # checkpoint 已保存到 PostgreSQL
    # 可以在任何连接了同一数据库的进程中恢复
    
    return result

# 方式二：使用已有的连接池（推荐用于长期运行的服务）
class CheckpointService:
    def __init__(self, db_uri: str):
        self.db_uri = db_uri
        self._pool: asyncpg.pool.Pool | None = None
        self._checkpointer: PostgresSaver | None = None

    async def initialize(self):
        self._pool = await asyncpg.create_pool(self.db_uri, min_size=2, max_size=10)
        self._checkpointer = PostgresSaver(async_pool=self._pool)

    @property
    def checkpointer(self) -> PostgresSaver:
        if self._checkpointer is None:
            raise RuntimeError("Checkpointer not initialized")
        return self._checkpointer

    async def close(self):
        if self._pool:
            await self._pool.close()

# 使用示例
service = CheckpointService("postgresql://user:pass@localhost:5432/lg_cp")
await service.initialize()
app = your_graph.compile(checkpointer=service.checkpointer)
```

PostgreSQL Saver 在内部使用两张表来存储 checkpoint 数据：

- **`checks` 表**：存储每个 checkpoint 的元数据和序列化后的状态数据
- **`checkpoint_blobs` 表**：如果状态数据很大，会被拆分成多个 blob 存储（避免单行过大）

这种设计有几个重要的工程含义：

第一，**checkpoint 查询是高效的**。通过 thread_id 和时间戳建立索引，即使有数百万条 checkpoint 记录，查找特定线程的最新状态也是毫秒级的。

第二，**支持 TTL（自动过期）**。你可以设置 checkpoint 的生存时间，过期的记录会被自动清理。这对于控制存储空间增长非常重要：

```python
# 设置 checkpoint 7天后自动清理
postgres_checkpointer = await PostgresSaver.from_conn_string(
    postgres_uri,
    ttl="7d"  # 7天TTL
)
```

第三，**支持多实例并发**。多个应用实例可以同时连接同一个 PostgreSQL 数据库并读写 checkpoint，每条记录的写入由数据库的事务机制保证原子性和一致性。

## Redis Saver：高性能缓存优先方案

如果你的首要需求是极致的读写性能（比如每秒需要处理数千次 checkpoint 操作），Redis 是一个很好的选择。但需要注意 Redis 默认是内存存储，如果不开启持久化配置，Redis 服务重启会丢失所有 checkpoint 数据。

```python
from langgraph.checkpoint.redis import RedisSaver
import redis

# 基本用法
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
redis_checkpointer = RedisSaver(redis_client)

app = your_graph.compile(checkpointer=redis_checkpointer)
config = {"configurable": {"thread_id": "redis-session-001"}}

result = app.invoke(initial_state, config=config)

# 高级配置：带TTL和自定义前缀
redis_checkpointer_advanced = RedisSaver(
    redis_client,
    key_prefix="lg:cp:",     # 自定义键前缀，避免与其他应用冲突
    ttl=3600                  # checkpoint 1小时后自动过期（秒）
)
```

Redis Saver 的特点：
- **极快的读写速度**：内存操作，通常在亚毫秒级别
- **丰富的数据结构支持**：天然支持 Hash、List 等结构，适合存储复杂的 checkpoint 数据
- **需要额外配置持久化**：建议开启 AOF 或 RDB 持久化以防止数据丢失
- **内存成本较高**：所有 checkpoint 都保存在内存中，大量历史数据会占用较多内存

## 生产部署 Checklist

当把 LangGraph 应用部署到生产环境时，关于 checkpointing 有几个必须检查的项目：

**1. 选择合适的后端**
- 单机 / 低流量 → SQLite（最简单）
- 多实例 / 中等流量 → PostgreSQL（最可靠）
- 高频操作 / 极低延迟要求 → Redis（最快）

**2. 配置连接池**
- 不要每次请求都创建新的数据库连接
- 使用连接池管理连接生命周期
- 设置合理的 pool_size（通常 2-10 个连接足够）

**3. 设置 TTL 自动清理**
- 根据业务需求设置合理的保留期（通常 1-7 天）
- 定期监控存储空间使用情况
- 对于 Interrupt 场景可能需要更长的 TTL

**4. 监控 checkpoint 操作**
- 记录每次 get/put 操作的延迟
- 监控 checkpoint 存储的大小增长趋势
- 设置告警阈值（如存储超过 10GB 时告警）

**5. 备份策略**
- PostgreSQL：启用 WAL 归档 + 定期全量备份
- Redis：开启 RDB 快照 + AOF 持久化
- SQLite：定期将 db 文件复制到远程存储

**6. 安全考虑**
- 数据库连接字符串不要硬编码在代码中，使用环境变量或密钥管理服务
- 如果 checkpoint 包含敏感数据（用户输入、API 密钥等），确保数据库本身有访问控制
- 考虑对敏感字段进行加密后再存储

## 从 MemorySaver 迁移到生产后端

一个常见的开发路径是：先用 MemorySaver 开发和调试，上线前切换到 PostgreSQL 或其他生产后端。好消息是这个迁移过程非常简单——因为所有 checkpointer 实现都遵循相同的接口，你只需要修改一行代码：

```python
# 开发阶段
from langgraph.checkpoint.memory import MemorySaver
checkpointer = MemorySaver()

# 生产阶段 - 切换为 PostgreSQL
from langgraph.checkpoint.postgres import PostgresSaver
checkpointer = await PostgresSaver.from_conn_string(
    "postgresql://user:pass@db:5432/lg_prod"
)

# 其余代码完全不变！
app = graph.compile(checkpointer=checkpointer)
result = app.invoke(state, config=config)
```

这种接口一致性是 LangGraph checkpointer 设计的一个核心优势——它让你可以在不改变任何业务代码的情况下灵活切换底层存储方案。
