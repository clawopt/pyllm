# 1.3 SQL 基础：增删改查

> **SQL 是与 PostgreSQL 对话的语言——掌握增删改查，你就掌握了操作数据库的钥匙**

---

## 这一节在讲什么？

这一节是 PostgreSQL 的 SQL 速成课。如果你已经熟悉 SQL，可以快速浏览后跳到下一章。如果你是 SQL 新手，这一节会带你从建表开始，逐步掌握 INSERT、SELECT、UPDATE、DELETE 四大基本操作，以及如何用 Python 执行这些操作。这些知识是后续使用 pgvector 的语法基础——因为 pgvector 的所有操作（创建 vector 列、插入向量、按距离排序）都是通过 SQL 完成的，没有独立的 Python API。

---

## CREATE TABLE：定义数据的结构

在往数据库里存数据之前，你需要先定义数据的"容器"——表（Table）。创建表时需要指定表名、每一列的名称和数据类型，以及可选的约束条件：

```sql
-- 创建文档表（后续 pgvector 教程会频繁使用这张表）
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,          -- 自增主键
    content TEXT NOT NULL,          -- 文档内容，不允许为空
    source VARCHAR(255),            -- 文档来源
    category VARCHAR(100),          -- 分类
    metadata JSONB,                 -- 灵活的元数据（JSON 格式）
    created_at TIMESTAMP DEFAULT NOW()  -- 创建时间，默认当前时间
);
```

PostgreSQL 常用数据类型速查：

| 类型 | 说明 | 示例 |
|------|------|------|
| `SERIAL` | 自增整数（主键常用） | 1, 2, 3, ... |
| `INTEGER` | 整数 | 42, -7 |
| `REAL` | 单精度浮点数 | 3.14 |
| `DOUBLE PRECISION` | 双精度浮点数 | 3.14159265358979 |
| `VARCHAR(N)` | 变长字符串（有上限） | "hello" |
| `TEXT` | 变长字符串（无上限） | "很长的文本..." |
| `BOOLEAN` | 布尔值 | TRUE, FALSE |
| `TIMESTAMP` | 时间戳 | '2024-01-15 10:30:00' |
| `JSONB` | 二进制 JSON | '{"key": "value"}' |
| `UUID` | 通用唯一标识符 | 'a0eebc99-9c0b-4ef8-bb...' |

约束（Constraint）用于保证数据完整性：

```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,              -- 主键约束：唯一且非空
    email VARCHAR(255) UNIQUE NOT NULL, -- 唯一约束 + 非空约束
    name VARCHAR(100) NOT NULL,         -- 非空约束
    age INTEGER CHECK (age >= 0),       -- 检查约束：年龄不能为负
    department_id INTEGER REFERENCES departments(id)  -- 外键约束
);
```

---

## INSERT：插入数据

### 单条插入

```sql
INSERT INTO documents (content, source, category, metadata)
VALUES (
    '退款政策：购买后7天内可无条件退款',
    'user_manual.pdf',
    'after_sales',
    '{"page": 15, "version": 2}'
);
```

### 批量插入

```sql
INSERT INTO documents (content, source, category) VALUES
    ('安装指南：下载安装包并运行', 'install_guide.md', 'technical'),
    ('定价方案：标准版每月99元', 'pricing.md', 'pricing'),
    ('API文档：RESTful接口说明', 'api_docs.md', 'technical');
```

### RETURNING：插入后获取生成的值

当你使用 `SERIAL` 自增主键时，插入后你可能想知道自动生成的 ID 是多少。`RETURNING` 子句可以让你在插入的同时返回指定列的值：

```sql
INSERT INTO documents (content, source, category)
VALUES ('新文档', 'new_doc.md', 'general')
RETURNING id, created_at;
-- 返回：id | created_at
--       5  | 2024-01-15 10:30:00
```

### Python 中的插入

```python
import psycopg2

conn = psycopg2.connect("postgres://postgres:mysecretpassword@localhost/rag_db")
cur = conn.cursor()

# 单条插入（参数化查询，防止 SQL 注入）
cur.execute(
    "INSERT INTO documents (content, source, category) VALUES (%s, %s, %s)",
    ("退款政策：购买后7天内可无条件退款", "user_manual.pdf", "after_sales")
)

# 批量插入
docs = [
    ("安装指南：下载安装包并运行", "install_guide.md", "technical"),
    ("定价方案：标准版每月99元", "pricing.md", "pricing"),
    ("API文档：RESTful接口说明", "api_docs.md", "technical"),
]
cur.executemany(
    "INSERT INTO documents (content, source, category) VALUES (%s, %s, %s)",
    docs
)

# 提交事务（psycopg2 默认开启事务，必须手动提交）
conn.commit()

# 插入并获取生成的 ID
cur.execute(
    "INSERT INTO documents (content, source) VALUES (%s, %s) RETURNING id",
    ("新文档", "new.md")
)
new_id = cur.fetchone()[0]
print(f"新插入的文档 ID: {new_id}")

conn.commit()
cur.close()
conn.close()
```

**重要**：psycopg2 默认在事务中执行所有操作。你必须调用 `conn.commit()` 才能让修改生效。如果忘记 commit，程序退出后所有修改都会回滚。

---

## SELECT：查询数据

### 基本查询

```sql
-- 查询所有列
SELECT * FROM documents;

-- 查询指定列
SELECT id, content, source FROM documents;

-- 限制返回行数
SELECT * FROM documents LIMIT 5;

-- 分页查询
SELECT * FROM documents LIMIT 10 OFFSET 20;  -- 第3页（每页10条）
```

### WHERE 条件过滤

```sql
-- 等值匹配
SELECT * FROM documents WHERE category = 'technical';

-- 多条件组合
SELECT * FROM documents
WHERE category = 'technical' AND created_at > '2024-01-01';

-- IN 匹配
SELECT * FROM documents WHERE category IN ('technical', 'pricing');

-- 模糊匹配
SELECT * FROM documents WHERE content LIKE '%退款%';

-- JSONB 查询
SELECT * FROM documents WHERE metadata @> '{"version": 2}';
```

### ORDER BY 排序

```sql
-- 按创建时间降序
SELECT * FROM documents ORDER BY created_at DESC;

-- 按来源和类别排序
SELECT * FROM documents ORDER BY source, category;
```

### 聚合与分组

```sql
-- 统计每个分类的文档数
SELECT category, COUNT(*) AS doc_count
FROM documents
GROUP BY category
ORDER BY doc_count DESC;

-- 统计总数
SELECT COUNT(*) FROM documents;
```

---

## UPDATE：更新数据

```sql
-- 更新单个字段
UPDATE documents SET category = 'after_sales' WHERE id = 1;

-- 更新多个字段
UPDATE documents
SET content = '更新后的内容', source = 'updated_doc.md'
WHERE id = 1;

-- 条件更新（影响多行）
UPDATE documents SET category = 'uncategorized' WHERE category IS NULL;

-- 更新并返回修改后的行
UPDATE documents SET category = 'tech' WHERE id = 1 RETURNING *;
```

Python 中的更新：

```python
cur.execute(
    "UPDATE documents SET category = %s WHERE id = %s",
    ("after_sales", 1)
)
conn.commit()
print(f"更新了 {cur.rowcount} 行")
```

---

## DELETE：删除数据

```sql
-- 按 ID 删除
DELETE FROM documents WHERE id = 1;

-- 按条件批量删除
DELETE FROM documents WHERE created_at < '2023-01-01';

-- 删除所有数据（保留表结构）
TRUNCATE documents;

-- 删除并返回被删除的行
DELETE FROM documents WHERE id = 1 RETURNING *;
```

Python 中的删除：

```python
cur.execute("DELETE FROM documents WHERE id = %s", (1,))
conn.commit()
print(f"删除了 {cur.rowcount} 行")
```

---

## 完整实战：Python CRUD 封装

让我们把上面的知识整合成一个可复用的 CRUD 类：

```python
import psycopg2
from psycopg2 import pool

class DocumentDB:
    """文档数据库 CRUD 封装"""

    def __init__(self, dsn="postgres://postgres:mysecretpassword@localhost/rag_db"):
        self.pool = pool.ThreadedConnectionPool(
            minconn=2, maxconn=10, dsn=dsn
        )

    def _get_conn(self):
        conn = self.pool.getconn()
        return conn

    def _put_conn(self, conn):
        self.pool.putconn(conn)

    def create_table(self):
        """创建文档表"""
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id SERIAL PRIMARY KEY,
                content TEXT NOT NULL,
                source VARCHAR(255),
                category VARCHAR(100),
                metadata JSONB,
                created_at TIMESTAMP DEFAULT NOW()
            );
        """)
        conn.commit()
        cur.close()
        self._put_conn(conn)
        print("✅ 表创建成功")

    def insert(self, content, source=None, category=None, metadata=None):
        """插入一条文档"""
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO documents (content, source, category, metadata)
               VALUES (%s, %s, %s, %s) RETURNING id""",
            (content, source, category, psycopg2.extras.Json(metadata) if metadata else None)
        )
        doc_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        self._put_conn(conn)
        return doc_id

    def get_by_id(self, doc_id):
        """按 ID 获取文档"""
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("SELECT * FROM documents WHERE id = %s", (doc_id,))
        row = cur.fetchone()
        cur.close()
        self._put_conn(conn)
        return row

    def search(self, category=None, keyword=None, limit=10):
        """搜索文档"""
        conn = self._get_conn()
        cur = conn.cursor()

        conditions = []
        params = []

        if category:
            conditions.append("category = %s")
            params.append(category)
        if keyword:
            conditions.append("content LIKE %s")
            params.append(f"%{keyword}%")

        where = " AND ".join(conditions) if conditions else "TRUE"
        params.append(limit)

        cur.execute(
            f"SELECT * FROM documents WHERE {where} ORDER BY created_at DESC LIMIT %s",
            params
        )
        rows = cur.fetchall()
        cur.close()
        self._put_conn(conn)
        return rows

    def update(self, doc_id, **kwargs):
        """更新文档"""
        conn = self._get_conn()
        cur = conn.cursor()

        set_clauses = []
        params = []
        for key, value in kwargs.items():
            set_clauses.append(f"{key} = %s")
            params.append(value)

        params.append(doc_id)
        cur.execute(
            f"UPDATE documents SET {', '.join(set_clauses)} WHERE id = %s",
            params
        )
        conn.commit()
        affected = cur.rowcount
        cur.close()
        self._put_conn(conn)
        return affected

    def delete(self, doc_id):
        """删除文档"""
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("DELETE FROM documents WHERE id = %s", (doc_id,))
        conn.commit()
        affected = cur.rowcount
        cur.close()
        self._put_conn(conn)
        return affected


# ====== 使用示例 ======
db = DocumentDB()
db.create_table()

# 插入
doc_id = db.insert("退款政策：购买后7天内可无条件退款", source="manual.pdf", category="after_sales")
print(f"插入文档 ID: {doc_id}")

# 查询
doc = db.get_by_id(doc_id)
print(f"文档内容: {doc[1]}")

# 搜索
results = db.search(category="after_sales", keyword="退款")
print(f"搜索到 {len(results)} 条文档")

# 更新
db.update(doc_id, category="refund", source="updated_manual.pdf")

# 删除
db.delete(doc_id)
```

---

## 常见误区

### 误区 1：忘记 conn.commit()

psycopg2 默认在事务中执行所有操作。如果你执行了 INSERT/UPDATE/DELETE 但没有调用 `conn.commit()`，修改只存在于当前事务中，程序退出后会自动回滚。这是新手最常犯的错误。

### 误区 2：用字符串拼接构造 SQL

永远不要用 f-string 或 `+` 拼接 SQL 语句——这会导致 SQL 注入漏洞。始终使用参数化查询（`%s` 占位符）：

```python
# ❌ 危险：SQL 注入
cur.execute(f"SELECT * FROM documents WHERE content = '{user_input}'")

# ✅ 安全：参数化查询
cur.execute("SELECT * FROM documents WHERE content = %s", (user_input,))
```

### 误区 3：SELECT * 在生产代码中使用

`SELECT *` 会返回所有列，包括你可能不需要的大字段（如长文本、向量数据）。在生产代码中，始终明确指定需要的列名：

```sql
-- ❌ 不推荐
SELECT * FROM documents;

-- ✅ 推荐
SELECT id, content, source FROM documents;
```

---

## 本章小结

SQL 的增删改查是操作 PostgreSQL 的基本功。核心要点回顾：第一，CREATE TABLE 定义数据结构，SERIAL 用于自增主键，JSONB 用于灵活的半结构化数据；第二，INSERT 插入数据，RETURNING 子句可以返回自动生成的值；第三，SELECT 查询数据，WHERE 做条件过滤，ORDER BY 排序，LIMIT/OFFSET 分页；第四，psycopg2 默认在事务中执行，必须手动 commit 才能生效；第五，永远使用参数化查询防止 SQL 注入；第六，生产代码避免 SELECT *，明确指定需要的列。

下一章我们将学习 PostgreSQL 的进阶知识——索引与查询优化、JSONB 操作、事务与并发控制，为后续的 pgvector 学习打下更坚实的基础。
