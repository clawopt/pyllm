# 7.1 生产部署方案

> **从开发环境到生产环境，pgvector 的部署并不复杂——但细节决定成败**

---

## 这一节在讲什么？

前面六章我们一直在本地开发环境里用 pgvector 做实验——Docker 起一个容器、连上 psql 敲几条 SQL、Python 脚本跑一下检索，一切都很美好。但当你要把 pgvector 部署到生产环境时，问题就来了：连接数怎么规划？数据库挂了怎么办？数据丢了怎么办？怎么备份？怎么恢复？怎么保证高可用？这些问题跟 pgvector 本身没有直接关系，它们是 PostgreSQL 生产部署的通用问题——但既然你选择了 pgvector，就意味着你选择了 PostgreSQL 作为向量数据的存储底座，这些运维问题你一个都逃不掉。这一节我们就来系统地聊一聊 pgvector 的生产部署方案，从 Docker Compose 到云托管，从连接池到备份恢复，从单机到高可用，帮你把 pgvector 从"能跑"变成"跑得稳"。

---

## Docker Compose 部署

对于大多数中小型项目来说，Docker Compose 是最简单也最实用的部署方案——一个 `docker-compose.yml` 文件就能把 PostgreSQL + pgvector + 你的应用服务全部编排起来，一键启动、一键停止、一键升级。

```yaml
# docker-compose.yml
version: '3.8'

services:
  postgres:
    image: pgvector/pgvector:pg16
    container_name: pgvector-db
    restart: always
    environment:
      POSTGRES_USER: appuser
      POSTGRES_PASSWORD: ${PG_PASSWORD}      # 从 .env 文件读取
      POSTGRES_DB: ragdb
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data       # 数据持久化
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql  # 初始化脚本
    command: >
      postgres
      -c shared_buffers=256MB
      -c work_mem=16MB
      -c effective_cache_size=768MB
      -c max_connections=200
      -c wal_level=replica                     # 为后续主从复制预留
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U appuser -d ragdb"]
      interval: 10s
      timeout: 5s
      retries: 5

  app:
    build: .
    container_name: rag-app
    restart: always
    environment:
      DATABASE_URL: postgresql://appuser:${PG_PASSWORD}@postgres:5432/ragdb
    depends_on:
      postgres:
        condition: service_healthy
    ports:
      - "8000:8000"

volumes:
  pgdata:
    driver: local
```

这里有几个关键点需要你特别注意：

第一，**数据持久化**。`volumes: - pgdata:/var/lib/postgresql/data` 这一行绝对不能省——如果不挂载 volume，容器重建后所有数据都会丢失。这在开发环境可能无所谓，但在生产环境就是灾难。你一定要确保数据目录挂载到了宿主机的持久化存储上。

第二，**初始化脚本**。`./init.sql:/docker-entrypoint-initdb.d/init.sql` 这个挂载利用了 PostgreSQL Docker 镜像的初始化机制——容器第一次启动时会自动执行 `/docker-entrypoint-initdb.d/` 目录下的所有 `.sql` 和 `.sh` 文件。你可以把 `CREATE EXTENSION vector;` 和建表语句放在这个初始化脚本里：

```sql
-- init.sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    source TEXT NOT NULL,
    category TEXT,
    content TEXT NOT NULL,
    embedding vector(384),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_doc_embedding
ON documents USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

第三，**健康检查**。`healthcheck` 配置让 Docker 能够判断 PostgreSQL 是否真正就绪——光看容器启动是不够的，PostgreSQL 启动后还需要一小段时间完成恢复和预热。应用服务通过 `depends_on: condition: service_healthy` 确保只在数据库真正可用时才启动，避免了启动顺序问题。

第四，**密码管理**。`POSTGRES_PASSWORD: ${PG_PASSWORD}` 从环境变量读取密码，而不是硬编码在 yaml 文件里。你需要在同目录下创建一个 `.env` 文件：

```bash
# .env（不要提交到 Git！）
PG_PASSWORD=your_secure_password_here
```

```bash
# .gitignore
.env
```

### 常见误区：把密码硬编码在 docker-compose.yml 里

很多初学者为了方便，直接把密码写在 yaml 文件里然后提交到 Git。这跟把家门钥匙贴在门上没什么区别——任何人只要能看到你的代码仓库，就能拿到数据库密码。正确的做法是使用环境变量或 Docker Secrets，并且确保 `.env` 文件被 `.gitignore` 排除。

---

## 云托管方案

如果你不想自己运维 PostgreSQL，各大云厂商都提供了托管的 PostgreSQL 服务——你只需要点几下鼠标就能创建一个高可用、自动备份、自动升级的 PostgreSQL 实例，然后在上面启用 pgvector 扩展即可。

### AWS RDS PostgreSQL

AWS RDS 是目前最流行的云托管 PostgreSQL 方案。从 2023 年底开始，RDS PostgreSQL 已经原生支持 pgvector 扩展：

```sql
-- 在 RDS PostgreSQL 中启用 pgvector
CREATE EXTENSION vector;
```

RDS 的优势在于它帮你处理了大部分运维工作——自动备份、自动故障转移、自动小版本升级、安全补丁。你只需要关注业务逻辑和性能调优。但 RDS 也有一些限制：你不能修改 `shared_preload_libraries`（某些扩展无法使用），不能直接访问文件系统，参数修改范围有限。

### 阿里云 RDS PostgreSQL

阿里云 RDS PostgreSQL 同样支持 pgvector 扩展，操作方式类似：

```sql
-- 在阿里云 RDS 中启用 pgvector
CREATE EXTENSION vector;
```

阿里云的优势在于国内网络延迟更低、合规性更好、技术支持是中文的。如果你的业务主要面向国内用户，阿里云 RDS 是更实际的选择。

### 云托管方案的注意事项

| 注意事项 | 说明 |
|---------|------|
| 扩展支持 | 创建实例前确认 pgvector 版本是否满足需求，不同云厂商支持的版本可能不同 |
| 参数修改 | RDS 通常允许修改大部分参数，但 `shared_preload_libraries` 等核心参数可能受限 |
| 存储类型 | 选择 SSD（gp3 / ESSD）而非 HDD，向量索引对 IOPS 敏感 |
| 内存规格 | HNSW 索引全内存驻留，内存规格至少要是数据量的 2~3 倍 |
| 网络配置 | 确保应用服务器和数据库在同一 VPC 内，避免公网访问带来的延迟和安全风险 |
| 版本升级 | 云托管的版本升级通常需要计划内停机，提前在测试环境验证 |

### 常见误区：选了最低规格的 RDS 实例

很多团队为了省钱，选了最低规格的 RDS 实例（比如 1 vCPU / 1GB 内存），然后发现向量搜索慢得离谱。这是因为 HNSW 索引需要全内存驻留——如果你的索引有 2GB，但实例只有 1GB 内存，PostgreSQL 就不得不频繁地把索引页面换入换出，性能会断崖式下跌。对于 pgvector 生产环境，内存至少要是 HNSW 索引大小的 3 倍以上（索引本身 + shared_buffers + 操作系统缓存）。

---

## 连接池配置：pgbouncer

在开发环境里，你可能习惯每次查询都新建一个数据库连接、查完就关——这在并发量低的时候没问题，但在生产环境里，每一个新建的连接都要 fork 一个 PostgreSQL 进程、完成认证、初始化会话状态，这个过程的开销大约是 10~50 毫秒。如果你的应用有 1000 个并发请求，每个请求都新建连接，光是连接建立的开销就吃掉了大量资源。

pgbouncer 是 PostgreSQL 生态中最成熟的连接池工具——它维护一组长连接到 PostgreSQL，应用连接到 pgbouncer 时，pgbouncer 从池子里分配一个已有的连接给应用使用，用完归还而不是关闭。这样一来，应用的连接建立时间从 10~50ms 降到了微秒级。

### pgbouncer 的三种池化模式

| 模式 | 工作方式 | 适用场景 | 事务支持 |
|------|---------|---------|---------|
| Session | 一个客户端独占一个服务端连接，直到客户端断开 | 需要会话级状态（临时表、SET 参数） | 完整 |
| Transaction | 一个事务结束后连接就归还池子 | 大多数 Web 应用（推荐） | 完整 |
| Statement | 一条 SQL 执行完连接就归还 | 无事务的纯查询场景 | 不支持 |

对于 pgvector 的 RAG 应用，**Transaction 模式是最佳选择**——RAG 的典型请求模式是"开始事务 → 插入/查询 → 提交事务"，事务之间不需要保持会话状态。

### Docker Compose 中加入 pgbouncer

```yaml
# docker-compose.yml（追加 pgbouncer 服务）
services:
  pgbouncer:
    image: edoburu/pgbouncer
    container_name: pgbouncer
    restart: always
    environment:
      DATABASE_URL: postgresql://appuser:${PG_PASSWORD}@postgres:5432/ragdb
      POOL_MODE: transaction
      MAX_CLIENT_CONN: 1000
      DEFAULT_POOL_SIZE: 50
      RESERVE_POOL_SIZE: 10
      RESERVE_POOL_TIMEOUT: 3
    ports:
      - "6432:6432"
    depends_on:
      postgres:
        condition: service_healthy

  app:
    environment:
      # 应用连接 pgbouncer 而不是直接连接 PostgreSQL
      DATABASE_URL: postgresql://appuser:${PG_PASSWORD}@pgbouncer:6432/ragdb
    depends_on:
      - pgbouncer
```

### 连接数规划

连接数的规划需要考虑两个维度：应用侧的并发量和 PostgreSQL 侧的承载能力。一个经验公式是：

```
PostgreSQL max_connections = CPU 核数 × 10 + 50
pgbouncer DEFAULT_POOL_SIZE = PostgreSQL max_connections × 0.5
pgbouncer MAX_CLIENT_CONN = 应用最大并发数（可以远大于 max_connections）
```

比如你的 PostgreSQL 是 4 核机器，那么 `max_connections = 4 × 10 + 50 = 90`，pgbouncer 的 `DEFAULT_POOL_SIZE = 45`，`MAX_CLIENT_CONN` 可以设到 500 甚至 1000——因为 pgbouncer 的 Transaction 模式下，1000 个客户端连接实际上只复用 45 个 PostgreSQL 连接。

### 常见误区：pgbouncer 和 Prepared Statements 不兼容

如果你使用 psycopg2 的 `execute_values` 或 SQLAlchemy 的批量操作，它们底层会使用 Prepared Statements 来提升性能。但 pgbouncer 的 Transaction 模式**不支持跨事务的 Prepared Statements**——因为事务结束后连接就归还了，下一个事务拿到的可能是另一个连接，之前准备的语句就不存在了。

解决方法有两种：

```python
# 方法1：在 psycopg2 中禁用 Prepared Statements
import psycopg2
conn = psycopg2.connect(DATABASE_URL)
conn.prepare_threshold = None  # 禁用自动 Prepared Statement

# 方法2：使用 pgvector 的 register_vector 时确保在事务内完成
from pgvector.psycopg2 import register_vector
conn = psycopg2.connect(DATABASE_URL)
register_vector(conn)
# register_vector 内部会创建类型转换器，在 Transaction 模式下每次连接都需要重新注册
```

---

## 备份与恢复

数据是生产系统最核心的资产——代码丢了可以重写，数据丢了就是灾难。PostgreSQL 提供了多种备份方案，从简单的逻辑备份到完整的时间点恢复，你需要根据业务需求选择合适的方案。

### 逻辑备份：pg_dump

`pg_dump` 是最基础的备份工具，它把数据库导出为 SQL 脚本或自定义格式的文件。逻辑备份的优点是简单、灵活、可以只备份单个表或单个数据库；缺点是备份期间会锁表（虽然时间很短）、大数据量时备份和恢复都比较慢。

```bash
# 备份整个数据库（自定义格式，推荐）
pg_dump -h localhost -U appuser -Fc ragdb > backup_$(date +%Y%m%d).dump

# 只备份 documents 表
pg_dump -h localhost -U appuser -Fc -t documents ragdb > docs_backup.dump

# 恢复备份
pg_restore -h localhost -U appuser -d ragdb backup_20260410.dump

# 备份为纯 SQL 脚本（可读、可编辑）
pg_dump -h localhost -U appuser ragdb > backup_$(date +%Y%m%d).sql
```

对于 pgvector 的备份，`pg_dump` 会自动处理 `vector` 类型——它会先备份 `CREATE EXTENSION vector;`，然后备份表结构和数据，恢复时先创建扩展再导入数据，顺序是正确的。

### 自动化备份脚本

在生产环境中，你需要一个定时备份脚本，并且把备份文件上传到对象存储（S3 / OSS）：

```bash
#!/bin/bash
# backup_pgvector.sh

BACKUP_DIR="/backups"
S3_BUCKET="s3://my-backups/pgvector"
DB_HOST="localhost"
DB_USER="appuser"
DB_NAME="ragdb"
DATE=$(date +%Y%m%d_%H%M%S)

# 执行备份
pg_dump -h $DB_HOST -U $DB_USER -Fc $DB_NAME > "$BACKUP_DIR/ragdb_$DATE.dump"

# 检查备份是否成功
if [ $? -eq 0 ]; then
    echo "[$DATE] Backup successful"
    # 上传到 S3
    aws s3 cp "$BACKUP_DIR/ragdb_$DATE.dump" "$S3_BUCKET/ragdb_$DATE.dump"
    # 删除 7 天前的本地备份
    find $BACKUP_DIR -name "ragdb_*.dump" -mtime +7 -delete
else
    echo "[$DATE] Backup FAILED"
    # 发送告警通知
    curl -X POST "https://hooks.slack.com/services/xxx" \
         -H "Content-Type: application/json" \
         -d "{\"text\": \"⚠️ pgvector backup failed at $DATE\"}"
fi
```

```bash
# 用 crontab 设置每天凌晨 2 点自动备份
crontab -e
0 2 * * * /path/to/backup_pgvector.sh >> /var/log/pg_backup.log 2>&1
```

### WAL 归档与时间点恢复（PITR）

`pg_dump` 是逻辑备份，它只能恢复到备份时刻的数据——如果你在凌晨 2 点做了备份，下午 3 点有人误删了一张表，你只能恢复到凌晨 2 点的状态，中间 13 个小时的数据就丢了。要解决这个问题，需要 WAL（Write-Ahead Log）归档 + 时间点恢复（PITR）。

WAL 是 PostgreSQL 的事务日志——每一个数据修改操作都会先写入 WAL，然后再修改数据文件。如果你持续归档 WAL 文件，就可以把数据库恢复到任意一个时间点。

```
WAL 归档 + PITR 的工作原理：

  时间线：
  ┌──────────────────────────────────────────────────────┐
  │  凌晨2点      上午10点       下午3点       下午4点    │
  │  基础备份      正常运行      误删表         恢复      │
  │    │            │              │             │        │
  │    ▼            ▼              ▼             ▼        │
  │  base_backup  WAL_001       WAL_042      恢复到      │
  │              WAL_002        WAL_043      上午11点     │
  │              ...            ...                      │
  └──────────────────────────────────────────────────────┘

  恢复步骤：
  1. 恢复基础备份（凌晨2点的完整备份）
  2. 重放 WAL_001 ~ WAL_025（凌晨2点到上午11点的所有变更）
  3. 数据库回到上午11点的状态——误删操作从未发生
```

```bash
# 在 postgresql.conf 中启用 WAL 归档
wal_level = replica
archive_mode = on
archive_command = 'aws s3 cp %p s3://my-backups/wal/%f'

# 创建基础备份
pg_basebackup -h localhost -U appuser -D /backups/base -Ft -z -P

# 恢复到指定时间点
# 1. 停止 PostgreSQL
# 2. 清空数据目录
# 3. 恢复基础备份
tar -xzf /backups/base/base.tar.gz -C /var/lib/postgresql/data

# 4. 创建 recovery.signal 文件（PostgreSQL 12+）
touch /var/lib/postgresql/data/recovery.signal

# 5. 配置恢复目标时间
# 在 postgresql.conf 或 postgresql.auto.conf 中添加：
restore_command = 'aws s3 cp s3://my-backups/wal/%f %p'
recovery_target_time = '2026-04-10 11:00:00 UTC'
recovery_target_action = 'promote'

# 6. 启动 PostgreSQL，它会自动重放 WAL 到目标时间点
```

### 常见误区：只做 pg_dump 备份就以为万事大吉

`pg_dump` 是"快照式"备份——它只能恢复到备份时刻的数据。如果你的业务对数据丢失的容忍度是零（比如金融、医疗场景），你必须配置 WAL 归档 + PITR。一个合理的备份策略应该是：**每天一次 pg_dump 全量备份 + 持续 WAL 归档**，这样既能快速恢复到最近的备份点（用 pg_dump），也能恢复到任意时间点（用 PITR）。

---

## 高可用方案

单机数据库最大的风险就是——机器挂了，服务就停了。对于生产系统来说，可用性通常要求达到 99.9% 甚至 99.99%，这意味着全年停机时间不能超过 8.76 小时甚至 52.6 分钟。要达到这个目标，你需要高可用方案。

### 流复制（Streaming Replication）

PostgreSQL 原生支持流复制——主库（Primary）持续把 WAL 日志流式传输给从库（Standby），从库重放 WAL 日志来保持数据同步。当主库故障时，你可以把从库提升为主库，实现故障转移。

```
流复制架构：

  应用写入          应用读取（可选）
     │                  │
     ▼                  ▼
  ┌────────┐    ┌────────────┐
  │ Primary │───▶│  Standby   │
  │ (读写)  │WAL │  (只读)    │
  └────────┘流  └────────────┘
     │                  │
     │   故障转移       │
     │  ┌──────────┐   │
     └──│ Promote  │───┘
        │ 提升为主 │
        └──────────┘
```

```bash
# 在主库上配置 postgresql.conf
listen_addresses = '*'
wal_level = replica
max_wal_senders = 3
wal_keep_size = 256MB

# 在主库上配置 pg_hba.conf，允许从库连接
host replication replicator standby_ip/32 md5

# 在从库上创建基础备份并启动复制
pg_basebackup -h primary_ip -U replicator -D /var/lib/postgresql/data -Fp -Xs -P -R

# -R 参数会自动创建 standby.signal 和配置 primary_conninfo
# 从库启动后自动开始流复制
```

### Patroni 自动故障转移

手动故障转移的问题是——你需要人工判断主库是否真的挂了、手动执行提升操作、手动通知应用切换连接。这个过程可能需要几分钟甚至更长时间，而且容易出错。Patroni 是一个自动故障转移工具——它持续监控主库的健康状态，当主库不可用时自动把从库提升为主库，并通过 DCS（分布式配置存储，如 etcd 或 Consul）通知应用切换连接。

```yaml
# patroni.yml（从库配置）
scope: pgvector-cluster
name: node2

restapi:
  listen: 0.0.0.0:8008

etcd:
  hosts: etcd:2379

bootstrap:
  dcs:
    ttl: 30
    loop_wait: 10
    maximum_lag_on_failover: 1048576
    postgresql:
      use_pg_rewind: true
      parameters:
        wal_level: replica
        hot_standby: "on"
        max_wal_senders: 5
        max_replication_slots: 5

postgresql:
  listen: 0.0.0.0:5432
  data_dir: /var/lib/postgresql/data
  authentication:
    superuser:
      username: postgres
      password: ${PG_PASSWORD}
    replication:
      username: replicator
      password: ${REPL_PASSWORD}
  parameters:
    shared_buffers: "256MB"
    work_mem: "16MB"
```

```yaml
# Docker Compose 中加入 Patroni
services:
  etcd:
    image: quay.io/coreos/etcd
    environment:
      ETCD_AUTO_COMPACTION_MODE: revision
      ETCD_AUTO_COMPACTION_RETENTION: "1000"
    command: etcd -advertise-client-urls=http://etcd:2379 -listen-client-urls=http://0.0.0.0:2379

  patroni1:
    image: patroni/patroni
    hostname: node1
    environment:
      PATRONI_ETCD_HOSTS: etcd:2379
      PATRONI_SCOPE: pgvector-cluster
      PATRONI_NAME: node1
      PATRONI_POSTGRESQL_DATA_DIR: /var/lib/postgresql/data
    volumes:
      - pgdata1:/var/lib/postgresql/data

  patroni2:
    image: patroni/patroni
    hostname: node2
    environment:
      PATRONI_ETCD_HOSTS: etcd:2379
      PATRONI_SCOPE: pgvector-cluster
      PATRONI_NAME: node2
      PATRONI_POSTGRESQL_DATA_DIR: /var/lib/postgresql/data
    volumes:
      - pgdata2:/var/lib/postgresql/data

  haproxy:
    image: haproxy
    ports:
      - "5000:5000"   # 读写端口 → 主库
      - "5001:5001"   # 只读端口 → 从库
```

Patroni + HAProxy 的组合实现了全自动的故障转移——应用连接 HAProxy，HAProxy 通过 Patroni 的 REST API 感知谁是主库谁是从库，自动把写请求路由到主库、读请求路由到从库。当主库故障时，Patroni 自动提升从库，HAProxy 自动切换路由，应用完全无感知。

### 常见误区：高可用 = 数据不丢

高可用和数据持久性是两个不同的概念——高可用保证的是"服务不中断"，数据持久性保证的是"数据不丢失"。流复制是异步的（默认配置下），主库故障时从库可能还没有收到最后几个事务的 WAL 日志，这些事务就会丢失。如果你需要零数据丢失，需要使用同步复制（`synchronous_commit = on` + `synchronous_standby_names`），但这会牺牲写入性能——每个事务都要等从库确认收到 WAL 后才能提交。

---

## 部署方案选择指南

| 场景 | 推荐方案 | 理由 |
|------|---------|------|
| 个人项目 / PoC | Docker 单容器 | 最简单，够用 |
| 小团队生产 | Docker Compose + pgbouncer | 运维简单，性能够用 |
| 中型团队生产 | 云 RDS + pgbouncer | 免运维，自动备份 |
| 大型团队生产 | 云 RDS + 流复制 + Patroni | 高可用，自动故障转移 |
| 合规要求高 | 自建 + WAL 归档 + PITR | 完全可控，满足审计 |

对于大多数 RAG 应用来说，**云 RDS + pgbouncer** 是性价比最高的选择——你不需要花时间在数据库运维上，而是把精力集中在业务逻辑和检索质量上。只有当你的数据规模大到云 RDS 无法承载，或者合规要求不允许使用云服务时，才需要考虑自建方案。

---

## 小结

这一节我们覆盖了 pgvector 生产部署的完整方案：Docker Compose 是最简单的起步方式，云 RDS 是最省心的生产选择，pgbouncer 是连接池的标配，pg_dump + WAL 归档是备份的基本策略，流复制 + Patroni 是高可用的终极方案。部署方案的选择取决于你的团队规模、数据量、可用性要求和预算——没有银弹，只有最适合你当前阶段的方案。下一节我们要聊的是生产环境中的性能调优和监控——怎么让 pgvector 跑得更快、怎么发现问题、怎么排查故障。
