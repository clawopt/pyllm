---
title: 数据库连接器：PostgreSQL / MySQL / MongoDB / SQLite / Redis
description: 关系型数据库与非关系型数据库的数据接入、SQL 查询到 Document 的映射、增量同步策略
---
# 数据库连接器：PostgreSQL / MySQL / MongoDB / SQLite / Redis

在企业环境中，最有价值的知识往往不在文件里，而在**数据库**中。客户信息在 CRM 系统（通常是 PostgreSQL 或 MySQL）、运营数据在数据仓库（可能是 BigQuery 或 Snowflake）、用户行为日志在 MongoDB、缓存数据在 Redis。如果 RAG 系统只能搜索文件而不能搜索数据库，那就相当于 blindfolded（蒙着眼）——明明最大的金矿就在脚下却视而不见。

这一节我们来系统学习如何用 LlamaIndex 的数据库连接器把各种数据库中的数据转化为 RAG 可用的 Document 对象。

## 关系型数据库：PostgreSQL

PostgreSQL 是企业级应用中最流行的开源关系型数据库之一。LlamaIndex 通过 `llama-index-readers-database` 包提供了对 PostgreSQL 的原生支持。

```bash
pip install llama-index-readers-database psycopg2-binary
```

### 基本用法

```python
from llama_index.readers.database import DatabaseReader

reader = DatabaseReader(
    sql_database="postgresql://user:password@localhost:5432/knowledge_db",
    sql_query="SELECT title, content, category FROM articles WHERE status = 'published'",
)

documents = reader.load_data()
print(f"从数据库加载了 {len(documents)} 条记录")

for doc in documents[:3]:
    print(f"\n标题: {doc.metadata.get('title', 'N/A')}")
    print(f"分类: {doc.metadata.get('category', 'N/A')}")
    print(f"内容: {doc.text[:150]}...")
```

`DatabaseReader` 的工作原理很简单：执行你提供的 SQL 查询，然后把每一行结果转换为一个 Document 对象。SELECT 的第一列（通常是主键或有意义的标识符）会被放入 metadata，其余列拼接为文本内容。

### 进阶用法：控制字段映射

默认的字段映射行为不一定满足你的需求。比如你可能希望某些字段作为 metadata 而非文本内容，或者想控制文本的拼接格式：

```python
from llama_index.readers.database import DatabaseReader

reader = DatabaseReader(
    sql_database="postgresql://user:password@localhost:5432/kb",
    # 使用参数化查询防止 SQL 注入
    sql_query="""
        SELECT id, title, content, author, created_at, department
        FROM knowledge_articles
        WHERE department = %s AND created_at > %s
    """,
    parameters=("技术部", "2025-01-01"),
)

documents = reader.load_data()

# 检查默认的映射结果
doc = documents[0]
print("metadata:", doc.metadata)  # 默认: 第一列(id)进入metadata
print("text:", doc.text[:200])    # 默认: 其余列拼接
```

默认情况下，`DatabaseReader` 的行为是：
- SQL 查询结果的**第一列**值存入 `metadata` 的某个 key 中
- **剩余的所有列**拼接为 Document 的 `text`

这种行为在很多场景下不够灵活。一种常见的变通方法是**在 SQL 层面就做好格式化**：

```python
reader = DatabaseReader(
    sql_database="postgresql://user:pass@localhost:5432/kb",
    sql_query="""
        SELECT
            id,
            CONCAT(
                '标题: ', title, '\n',
                '作者: ', author, '\n',
                '部门: ', department, '\n',
                '内容: ', content
            ) AS formatted_content
        FROM articles
        WHERE is_active = true
    """,
)
```

通过 SQL 的 `CONCAT` 函数，你在查询层面就控制好了文本的格式，这样 `DatabaseReader` 得到的就是一个干净的 `formatted_content` 列作为文本内容。

### 另一种方式：手动从数据库读取并构造 Document

如果 `DatabaseReader` 的默认行为不够灵活，你完全可以绕过它，直接用数据库驱动读取数据然后手动构造 Document：

```python
import psycopg2
from llama_index.core import Document

conn = psycopg2.connect("postgresql://user:pass@localhost:5432/kb")
cursor = conn.cursor(dictionary=True)

cursor.execute("""
    SELECT id, title, content, tags, author, updated_at
    FROM knowledge_base
    WHERE department IN ('产品', '技术', '售后')
""")

rows = cursor.fetchall()
documents = []

for row in rows:
    text = f"# {row['title']}\n\n{row['content']}"
    if row['tags']:
        text += f"\n\n标签: {row['tags']}"

    doc = Document(
        text=text,
        metadata={
            "id": row["id"],
            "title": row["title"],
            "author": row["author"],
            "updated_at": str(row["updated_at"]),
            "source": "postgresql://kb/knowledge_base",
        },
    )
    documents.append(doc)

cursor.close()
conn.close()

print(f"手动构建了 {len(documents)} 个 Document")
```

这种方式给了你完全的控制权——你可以自由决定哪些字段进 metadata、文本如何格式化、是否需要对内容做额外处理。代价是需要多写一些样板代码。**我的建议是：先用 DatabaseReader 快速原型验证，如果它的默认行为不够用，再切换到手动方式。**

## MySQL 连接

MySQL 的连接方式和 PostgreSQL 几乎一样，只是驱动和连接字符串格式不同：

```bash
pip install llama-index-readers-database mysqlclient
# 或者用纯 Python 实现: pip install pymysql
```

```python
from llama_index.readers.database import DatabaseReader

reader = DatabaseReader(
    sql_database="mysql://user:password@localhost:3306/company_kb",
    sql_query="""
        SELECT article_id, title, body, category
        FROM help_center_articles
        WHERE language = 'zh-CN'
    """,
)

documents = reader.load_data()
```

连接字符串格式遵循 SQLAlchemy 的约定：`mysql://user:password@host:port/database_name`。

### 注意事项

**字符集问题：** MySQL 的字符集配置不当会导致中文乱码。确保：
1. 数据库、表、字段的字符集都是 `utf8mb4`
2. 连接时指定字符集：`mysql://user:pass@host/db?charset=utf8mb4`
3. Python 端的 MySQL 客户端库也要正确设置编码

**SSL 连接：** 生产环境的 MySQL 通常要求 SSL 连接：

```python
reader = DatabaseReader(
    sql_database=(
        "mysql://user:password@db.example.com:3306/kb"
        "?ssl_ca=/path/to/ca.pem&ssl_verify_cert=true"
    ),
    sql_query="SELECT * FROM articles LIMIT 100",
)
```

## MongoDB 连接

MongoDB 是最流行的 NoSQL 文档数据库，它的数据模型（JSON-like 文档）与 LlamaIndex 的 Document 对象有着天然的亲和力——本质上它们都是"文本 + 元数据字典"的结构。

```bash
pip install llama-index-readers-database pymongo
```

```python
from llama_index.readers.database import MongoDBReader

reader = MongoDBReader(
    uri="mongodb://user:password@localhost:27017/",
    database_name="company_wiki",
    collection_name="articles",
    field_names=["title", "content", "author", "tags"],  # 要查询的字段
    query_dict={"status": "published", "department": "tech"},  # 过滤条件
)

documents = reader.load_data()
print(f"从 MongoDB 加载了 {len(documents)} 条文档")
```

MongoDB Reader 的参数设计与 SQL 数据库有明显差异，这是因为 MongoDB 的查询语言本身就是 JSON 格式的（`query_dict`），而不是 SQL 字符串。

### MongoDB 的嵌套文档处理

MongoDB 的一个强大特性是支持嵌套文档和数组。这在 RAG 场景下既可以是优势也可以是挑战：

```json
{
  "_id": "...",
  "title": "API 集成指南",
  "content": "本文介绍如何...",
  "sections": [
    {"heading": "认证", "body": "首先需要获取 API Key..."},
    {"heading": "请求格式", "body": "所有请求使用 JSON 格式..."},
    {"heading": "错误处理", "body": "错误码含义如下..."}
  ],
  "metadata": {
    "version": "2.1",
    "last_reviewed": "2025-03-15"
  }
}
```

默认情况下，MongoDBReader 会把嵌套结构展平后拼接到文本中。但更好的做法是在 `query_dict` 中使用 MongoDB 的**聚合管道（Aggregation Pipeline）**来预先处理好数据结构：

```python
reader = MongoDBReader(
    uri="mongodb://user:pass@localhost:27017/",
    database_name="wiki",
    collection_name="articles",
    field_names=["title", "content", "sections"],
    query_dict={},  # 不过滤，加载全部
    # 使用聚合管道展开 sections 数组
    pipeline=[
        {"$match": {"status": "published"}},
        {"$unwind": "$sections"},  # 把数组元素拆成独立文档
        {"$project": {
            "title": 1,
            "section_title": "$sections.heading",
            "section_body": "$sections.body",
        }},
    ],
)
documents = reader.load_data()

# 每个 section 现在是一个独立的 Document
for doc in documents:
    print(f"{doc.metadata.get('title')} - {doc.metadata.get('section_title')}")
```

通过 `$unwind` 操作，一篇有多节的文档被拆成了多个独立的 Document，每个对应一个 section。这对 RAG 系统来说是非常理想的粒度——用户问"认证怎么做"时，系统只需要返回"认证"这一节的内容，而不是整篇文章。

## SQLite 连接

SQLite 是轻量级的嵌入式数据库，特别适合本地开发和小型应用。它的优势在于**零配置**——不需要安装数据库服务器，一个 `.db` 文件就是完整的数据库。

```python
from llama_index.readers.database import DatabaseReader

reader = DatabaseReader(
    sql_database="sqlite:///./data/local_knowledge.db",  # 注意三个斜杠
    sql_query="""
        SELECT id, question, answer, category
        FROM faq
        WHERE is_verified = 1
    """,
)

documents = reader.load_data()
```

SQLite 的连接字符串格式是 `sqlite:///绝对路径/文件名.db`（注意是三个斜杠）。由于 SQLite 不需要网络连接和认证，所以连接字符串非常简洁。

SQLite 特别适合以下场景：
- **桌面应用的本地知识库**
- **开发和测试环境**（不需要搭建完整的数据库服务器）
- **边缘设备上的离线 RAG 应用**
- **数据量较小（GB 级别以下）的应用**

## Redis 连接

Redis 通常不被当作"数据库"来存储大量文本数据——它的强项是高速缓存和实时数据。但在 RAG 场景中，Redis 可以扮演两个角色：

**角色一：缓存已计算的 Embedding 向量。** 避免每次重启都重新计算 embedding。

**角色二：存储实时性强的数据。** 如在线客服的聊天记录、实时的系统状态等。

```bash
pip install llama-index-readers-database redis
```

```python
from llama_index.readers.database import RedisReader

reader = RedisReader(
    host="localhost",
    port=6379,
    password="your_redis_password",
    db=0,           # Redis 数据库编号 (0-15)
    protocol="redis",
)

documents = reader.load_data()
```

RedisReader 默认会将 Redis 中所有的 key-value 对加载为 Document。每个 key 成为 metadata 的一部分，value 成为文本内容。

### 更实用的 Redis 用法：作为 Vector Store

比起用 Redis 作为数据源（Reader），更常见的用法是把 Redis 作为**向量存储后端**（用于 Index 层，而非数据加载层）：

```python
from llama_index.vector_stores.redis import RedisVectorStore
from llama_index.core import VectorStoreIndex, StorageContext

vector_store = RedisVectorStore(
    redis_url="redis://:password@localhost:6379",
    index_name="knowledge_vectors",
    overwrite=False,  # 不要覆盖已有的索引
)

storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex(
    nodes=nodes,
    storage_context=storage_context,
)
```

Redis 的 RediSearch 模块提供了原生的向量相似度搜索能力，性能优异且支持持久化。对于已经在使用 Redis 的技术栈来说，这是一个非常自然的集成点。

## 数据库数据的增量同步

文件数据通常是一次性加载的（或者偶尔更新），但数据库中的数据是**持续变化**的——新文章不断发布、旧内容被修改、过期数据被删除。如何在 RAG 系统中保持与数据库的同步？这里有几种策略：

### 策略一：基于时间戳的增量加载

```python
import time
from llama_index.readers.database import DatabaseReader
from llama_index.core import VectorStoreIndex

LAST_SYNC_FILE = "./last_sync.txt"

def get_last_sync_time():
    try:
        with open(LAST_SYNC_FILE) as f:
            return f.read().strip()
    except FileNotFoundError:
        return "2025-01-01 00:00:00"

def save_sync_time(t):
    with open(LAST_SYNC_FILE, "w") as f:
        f.write(t)

last_sync = get_last_sync_time()

reader = DatabaseReader(
    sql_database="postgresql://user:pass@localhost:5432/kb",
    sql_query=f"""
        SELECT id, title, content, updated_at
        FROM articles
        WHERE updated_at > '{last_sync}'
        ORDER BY updated_at ASC
    """,
)

new_documents = reader.load_data()
if new_documents:
    index = VectorStoreIndex.from_documents(new_documents)
    save_sync_time(time.strftime("%Y-%m-%d %H:%M:%S"))
    print(f"同步完成，新增/更新 {len(new_documents)} 条记录")
else:
    print("没有新数据需要同步")
```

这种方式简单有效，但要求数据库表中有可靠的 `updated_at` 字段。

### 策略二：全量重建 vs 增量更新

| 策略 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| **全量重建** | 实现简单，数据一致性好 | 耗时长，资源消耗大 | 数据量小（<1000条），更新频率低 |
| **增量追加** | 快速，资源消耗小 | 可能产生重复或遗漏 | 数据量大，主要是新增操作 |
| **增量更新+删除** | 数据一致性最好 | 实现复杂，需追踪变更 | 生产环境，数据频繁变更 |

对于生产环境，推荐使用**增量更新+删除**的策略——既跟踪新增和更新的记录，也处理已删除的记录（从索引中移除对应的 Node）。

### 策略三：使用 CDC（Change Data Capture）

对于高频率变化的数据库，可以考虑使用 CDC 工具（如 Debezium）实时捕获数据库变更，然后通过消息队列触发 RAG 索引的增量更新。这是一种更加企业级的方案，但架构复杂度也更高。

## 常见误区

**误区一："数据库数据比文件数据更好处理"。** 不一定。数据库数据虽然是结构化的，但并不意味着它更适合 RAG。一条数据库记录可能包含很多冗余字段（创建人 ID、最后修改时间、审计日志等），这些噪音需要在 SQL 查询层面或后处理中清理掉。**结构化不等于干净。**

**误区二："直接 SELECT * 就行了"。** `SELECT *` 会把所有字段都拉出来，包括二进制大对象（BLOB）、加密字段、密码哈希等不该出现在 RAG 系统中的数据。**永远显式指定需要的列**，这不仅是为了安全，也是为了减少不必要的数据传输和处理开销。

**误区三："数据库连接可以一直开着"。** 数据库连接是有限资源。长时间运行的程序应该使用连接池（connection pool）并在完成后正确关闭连接。如果在 Jupyter Notebook 中做实验，别忘了重启 kernel 时之前的连接会失效。

**误区四:"Redis 适合存储大量文档"。** Redis 是内存数据库，存储成本远高于磁盘数据库。用它存储少量高频访问的热数据（如缓存的 embedding 向量）是合理的，但把整个知识库塞进 Redis 会导致内存爆炸。**选择合适的数据存储做合适的事。**
