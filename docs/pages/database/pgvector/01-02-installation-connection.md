# 1.2 安装与连接

> **从零到一——让 PostgreSQL 在你的机器上跑起来，然后用 Python 连上它**

---

## 这一节在讲什么？

上一节我们理解了 PostgreSQL 是什么以及为什么 RAG 工程师需要它，这一节我们要动手把它装上并跑起来。PostgreSQL 的安装方式有很多种，从最简单的 Docker 一键启动到从源码编译，我们重点讲 Docker 方式——因为它最干净、最快速、而且自带 pgvector 扩展。装好之后，我们会用 psql 命令行和 Python 两种方式连接数据库，确保你能对 PostgreSQL 进行基本的交互操作。

---

## Docker 方式安装（推荐）

Docker 是安装 PostgreSQL + pgvector 最简单的方式，因为你不需要自己编译 pgvector 扩展——官方提供的 `pgvector/pgvector` 镜像已经预装好了 pgvector：

```bash
# 拉取并启动 PostgreSQL + pgvector
docker run -d \
    --name pgvector-db \
    -p 5432:5432 \
    -e POSTGRES_USER=postgres \
    -e POSTGRES_PASSWORD=mysecretpassword \
    -e POSTGRES_DB=rag_db \
    pgvector/pgvector:pg16
```

参数解释：

| 参数 | 含义 |
|------|------|
| `-d` | 后台运行（detached mode） |
| `--name pgvector-db` | 容器名称，方便后续管理 |
| `-p 5432:5432` | 将容器内的 5432 端口映射到宿主机的 5432 端口 |
| `-e POSTGRES_USER=postgres` | 数据库超级用户名 |
| `-e POSTGRES_PASSWORD=mysecretpassword` | 数据库密码（生产环境请用强密码） |
| `-e POSTGRES_DB=rag_db` | 自动创建的数据库名 |
| `pgvector/pgvector:pg16` | 镜像名，pg16 表示基于 PostgreSQL 16 |

启动后验证：

```bash
# 检查容器是否在运行
docker ps | grep pgvector

# 进入容器内的 psql 命令行
docker exec -it pgvector-db psql -U postgres -d rag_db

# 在 psql 中验证 pgvector 扩展是否可用
SELECT * FROM pg_available_extensions WHERE name = 'vector';

# 如果看到 name | default_version | installed_version | comment
#      vector | 0.7.x           |                   | vector data type and approximate nearest neighbor search
# 说明 pgvector 可用！
```

### 数据持久化

上面的 Docker 命令有一个问题——如果你删除容器，所有数据都会丢失。要让数据持久化，需要挂载一个数据卷：

```bash
docker run -d \
    --name pgvector-db \
    -p 5432:5432 \
    -e POSTGRES_USER=postgres \
    -e POSTGRES_PASSWORD=mysecretpassword \
    -e POSTGRES_DB=rag_db \
    -v pgvector_data:/var/lib/postgresql/data \
    pgvector/pgvector:pg16
```

`-v pgvector_data:/var/lib/postgresql/data` 把 PostgreSQL 的数据目录挂载到 Docker 卷 `pgvector_data`，这样即使容器被删除，数据仍然保存在卷中。

---

## 其他安装方式

### macOS：Homebrew

```bash
# 安装 PostgreSQL
brew install postgresql@16

# 启动服务
brew services start postgresql@16

# 安装 pgvector 扩展（需要编译）
git clone --branch v0.7.0 https://github.com/pgvector/pgvector.git
cd pgvector
make
make install  # 可能需要 sudo
```

### Linux：apt（Ubuntu/Debian）

```bash
# 安装 PostgreSQL
sudo apt install postgresql-16

# 安装 pgvector
sudo apt install postgresql-16-pgvector
```

### 云托管

如果你不想自己管理服务器，各大云厂商都提供 PostgreSQL 托管服务。需要注意的是，不是所有云托管服务都支持 pgvector 扩展——你需要确认该服务允许安装自定义扩展：

| 云服务 | pgvector 支持 | 说明 |
|--------|-------------|------|
| AWS RDS PostgreSQL | ✅ | 从 15.4+ 版本开始支持 |
| Azure Database for PostgreSQL | ✅ | 灵活服务器模式支持 |
| 阿里云 RDS PostgreSQL | ✅ | 需要在控制台手动安装扩展 |
| Google Cloud SQL | ✅ | 从 PostgreSQL 14+ 支持 |

---

## 用 psql 连接数据库

psql 是 PostgreSQL 自带的命令行客户端，它是调试和学习 SQL 的最佳工具：

```bash
# 连接本地数据库
psql -h localhost -p 5432 -U postgres -d rag_db

# 或者通过 Docker 进入
docker exec -it pgvector-db psql -U postgres -d rag_db
```

连接成功后你会看到 `rag_db=#` 提示符，可以输入 SQL 命令：

```sql
-- 查看当前数据库
SELECT current_database();

-- 列出所有表
\dt

-- 创建 pgvector 扩展
CREATE EXTENSION vector;

-- 验证扩展已安装
\dx

-- 退出 psql
\q
```

psql 的常用元命令（以 `\` 开头，不是 SQL）：

| 命令 | 作用 |
|------|------|
| `\l` | 列出所有数据库 |
| `\dt` | 列出当前数据库的所有表 |
| `\d table_name` | 查看表结构 |
| `\dx` | 列出已安装的扩展 |
| `\du` | 列出所有用户 |
| `\q` | 退出 psql |
| `\?` | 查看所有元命令帮助 |

---

## 用 Python 连接数据库

在 RAG 应用中，你不会手动在 psql 里敲 SQL——你需要用 Python 代码来操作数据库。最常用的 PostgreSQL Python 驱动是 `psycopg2`：

```bash
# 安装依赖
pip install psycopg2-binary pgvector
```

### 基本连接

```python
import psycopg2

# 建立连接
conn = psycopg2.connect(
    host="localhost",
    port=5432,
    database="rag_db",
    user="postgres",
    password="mysecretpassword"
)

# 创建游标
cur = conn.cursor()

# 执行 SQL
cur.execute("SELECT version();")
version = cur.fetchone()
print(f"PostgreSQL 版本: {version[0]}")

# 关闭连接
cur.close()
conn.close()
```

### 注册 pgvector 类型

为了让 psycopg2 正确处理 pgvector 的 `vector` 类型，你需要注册类型转换器：

```python
import psycopg2
from pgvector.psycopg2 import register_vector

conn = psycopg2.connect(
    host="localhost",
    port=5432,
    database="rag_db",
    user="postgres",
    password="mysecretpassword"
)

# 注册 pgvector 类型转换
register_vector(conn)

cur = conn.cursor()

# 创建扩展（如果还没创建）
cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
conn.commit()

# 验证
cur.execute("SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';")
result = cur.fetchone()
if result:
    print(f"pgvector 已安装: 版本 {result[1]}")
else:
    print("pgvector 未安装")

cur.close()
conn.close()
```

`register_vector(conn)` 的作用是告诉 psycopg2：当从数据库读到 `vector` 类型的数据时，自动把它转换成 Python 的 `numpy.ndarray`；当向数据库写入 Python 的 list 或 ndarray 时，自动把它转换成 PostgreSQL 的 `vector` 类型。没有这一步，你需要在 SQL 中手动拼接向量字面量（如 `'[0.1, 0.2, 0.3]'::vector(3)`），既麻烦又容易出错。

### 连接池

在生产环境中，你不应该为每个请求都创建新的数据库连接——连接建立的开销很大（TCP 握手 + 认证 + 会话初始化）。正确的做法是使用连接池，预先创建一批连接并复用：

```python
from psycopg2 import pool

# 创建连接池
connection_pool = pool.ThreadedConnectionPool(
    minconn=2,       # 最小连接数
    maxconn=10,      # 最大连接数
    host="localhost",
    port=5432,
    database="rag_db",
    user="postgres",
    password="mysecretpassword"
)

# 从池中获取连接
def get_connection():
    conn = connection_pool.getconn()
    return conn

# 用完后归还连接
def release_connection(conn):
    connection_pool.putconn(conn)

# 使用示例
conn = get_connection()
cur = conn.cursor()
cur.execute("SELECT count(*) FROM documents;")
print(f"文档数: {cur.fetchone()[0]}")
cur.close()
release_connection(conn)
```

---

## 常见误区

### 误区 1：忘记创建 pgvector 扩展

pgvector 安装在 PostgreSQL 中不代表它自动可用——你需要在每个要使用 vector 类型的数据库中手动执行 `CREATE EXTENSION vector;`。如果忘记这一步，后续创建带 `vector` 列的表时会报错 `type "vector" does not exist`。

### 误区 2：连接参数写错导致连接失败

PostgreSQL 默认只监听 localhost。如果你在 Docker 中运行 PostgreSQL，需要确保 `-p 5432:5432` 端口映射正确。如果连接被拒绝，先检查 Docker 容器是否在运行（`docker ps`），再检查端口是否正确（`lsof -i :5432`）。

### 误区 3：忘记 register_vector 导致向量数据读写异常

如果你用 psycopg2 读写 vector 类型的数据但没有调用 `register_vector(conn)`，可能会遇到 `can't adapt type 'numpy.ndarray'` 或 `type "vector" does not exist` 之类的错误。务必在连接建立后立即注册。

---

## 本章小结

这一节我们让 PostgreSQL + pgvector 在本地跑了起来，并用 psql 和 Python 两种方式建立了连接。核心要点回顾：第一，Docker 是安装 PostgreSQL + pgvector 最简单的方式，`pgvector/pgvector:pg16` 镜像预装了 pgvector 扩展；第二，数据持久化需要挂载 Docker 卷，否则删除容器后数据丢失；第三，psql 是调试 SQL 的最佳工具，`\dt`、`\dx`、`\d` 是最常用的元命令；第四，Python 连接使用 psycopg2，务必调用 `register_vector(conn)` 注册类型转换器；第五，生产环境使用连接池（`psycopg2.pool`）复用连接，避免频繁建立和断开连接的开销。

下一节我们将学习 SQL 的增删改查基础——这是操作 PostgreSQL 的基本功，也是后续 pgvector 操作的语法基础。
