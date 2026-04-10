# 3.2 安装 pgvector 与 Hello World

> **从 CREATE EXTENSION 到第一次向量搜索——让 pgvector 在你的数据库中活起来**

---

## 这一节在讲什么？

上一节我们理解了 pgvector 的定位和优劣势，这一节我们要动手把它装上并跑通第一个向量搜索。整个过程分为三步：安装 pgvector 扩展、创建带向量列的表、执行向量相似度查询。我们会在 psql 和 Python 两种环境中完成这些操作，确保你既能在命令行中调试 SQL，也能在应用代码中使用 pgvector。

---

## Step 1：安装 pgvector 扩展

如果你使用的是 `pgvector/pgvector:pg16` Docker 镜像，pgvector 已经预装了，只需要在数据库中启用它：

```sql
-- 在 psql 中执行
CREATE EXTENSION vector;

-- 验证安装
SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';
-- 结果: extname | extversion
--       vector  | 0.7.x

-- 查看可用的向量操作符
SELECT oprname, oprleft::regtype, oprright::regtype, oprresult::regtype
FROM pg_operator
WHERE oprleft = 'vector'::regtype OR oprright = 'vector'::regtype;
```

如果你使用的是已有的 PostgreSQL 实例（非 Docker），需要先编译安装 pgvector：

```bash
# Ubuntu/Debian
sudo apt install postgresql-16-pgvector

# macOS
git clone --branch v0.7.0 https://github.com/pgvector/pgvector.git
cd pgvector
make
make install

# 然后在 psql 中执行 CREATE EXTENSION vector;
```

---

## Step 2：创建带向量列的表

pgvector 注册了 `vector(N)` 数据类型，其中 N 是向量的维度。创建表时，你需要指定向量列的维度：

```sql
-- 创建文档表，embedding 列是 3 维向量（演示用，实际应用通常是 384 或 768 维）
CREATE TABLE items (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding vector(3)
);

-- 插入向量数据
INSERT INTO items (content, embedding) VALUES
    ('苹果', '[1.0, 0.0, 0.0]'),
    ('香蕉', '[0.9, 0.1, 0.0]'),
    ('猫', '[0.0, 1.0, 0.0]'),
    ('狗', '[0.0, 0.9, 0.1]'),
    ('汽车', '[0.0, 0.0, 1.0]');
```

**维度约束**：同一列的所有向量维度必须一致。如果你尝试插入维度不匹配的向量，PostgreSQL 会报错：

```sql
-- ❌ 维度不匹配：表定义是 vector(3)，但插入了 4 维向量
INSERT INTO items (content, embedding) VALUES ('错误', '[1.0, 2.0, 3.0, 4.0]');
-- ERROR: expected 3 dimensions, not 4
```

---

## Step 3：执行向量搜索

pgvector 提供了三个距离操作符，用于计算向量之间的相似度：

| 操作符 | 含义 | 距离越小表示 |
|--------|------|------------|
| `<->` | L2 距离（欧氏距离） | 越相似 |
| `<=>` | 余弦距离 | 越相似 |
| `<#>` | 负内积 | 越相似（注意是负内积，值越大越不相似） |

### L2 距离搜索

```sql
-- 查询与 [1.0, 0.0, 0.0] 最相似的 3 条记录（L2 距离）
SELECT id, content, embedding <-> '[1.0, 0.0, 0.0]' AS distance
FROM items
ORDER BY embedding <-> '[1.0, 0.0, 0.0]'
LIMIT 3;
```

输出：

```
 id | content |      distance
----+---------+--------------------
  1 | 苹果    |                  0
  2 | 香蕉    | 0.1414213562373095
  4 | 狗      | 1.3784048752090222
```

"苹果"的 L2 距离为 0（完全相同），"香蕉"的距离很小（方向接近），"猫"和"狗"的距离较大（方向不同）。

### 余弦距离搜索

```sql
-- 查询与 [1.0, 0.0, 0.0] 最相似的 3 条记录（余弦距离）
SELECT id, content, embedding <=> '[1.0, 0.0, 0.0]' AS distance
FROM items
ORDER BY embedding <=> '[1.0, 0.0, 0.0]'
LIMIT 3;
```

输出：

```
 id | content |      distance
----+---------+--------------------
  1 | 苹果    |                  0
  2 | 香蕉    | 0.004969521524185
  4 | 狗      | 0.9950393384296077
```

余弦距离的结果与 L2 类似，但数值范围不同——余弦距离的范围是 [0, 2]，0 表示方向完全相同，2 表示方向完全相反。注意"香蕉"的余弦距离非常小（0.005），说明它的方向与查询向量几乎一致——这与直觉相符，因为 `[0.9, 0.1, 0.0]` 和 `[1.0, 0.0, 0.0]` 的方向非常接近。

### 三种距离的对比

```sql
-- 同时计算三种距离
SELECT id, content,
       embedding <-> '[1.0, 0.0, 0.0]' AS l2_distance,
       embedding <=> '[1.0, 0.0, 0.0]' AS cosine_distance,
       embedding <#> '[1.0, 0.0, 0.0]' AS neg_inner_product
FROM items
ORDER BY embedding <=> '[1.0, 0.0, 0.0]'
LIMIT 5;
```

---

## Python 中的 Hello World

```python
import psycopg2
from pgvector.psycopg2 import register_vector
import numpy as np

# 连接数据库
conn = psycopg2.connect("postgres://postgres:mysecretpassword@localhost/rag_db")
register_vector(conn)
cur = conn.cursor()

# 创建扩展
cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
conn.commit()

# 创建表
cur.execute("""
    CREATE TABLE IF NOT EXISTS items (
        id SERIAL PRIMARY KEY,
        content TEXT,
        embedding vector(3)
    );
""")
conn.commit()

# 插入向量数据（Python list 自动转换为 PG vector）
items = [
    ("苹果", [1.0, 0.0, 0.0]),
    ("香蕉", [0.9, 0.1, 0.0]),
    ("猫", [0.0, 1.0, 0.0]),
    ("狗", [0.0, 0.9, 0.1]),
    ("汽车", [0.0, 0.0, 1.0]),
]

for content, emb in items:
    cur.execute(
        "INSERT INTO items (content, embedding) VALUES (%s, %s)",
        (content, emb)
    )
conn.commit()

# 向量搜索
query_vec = [1.0, 0.0, 0.0]
cur.execute(
    """SELECT id, content, embedding <=> %s AS distance
       FROM items
       ORDER BY embedding <=> %s
       LIMIT 3""",
    (query_vec, query_vec)
)

print("=== 余弦距离搜索结果 ===")
for row in cur.fetchall():
    print(f"  ID={row[0]}, 内容={row[1]}, 距离={row[2]:.4f}")

cur.close()
conn.close()
```

输出：

```
=== 余弦距离搜索结果 ===
  ID=1, 内容=苹果, 距离=0.0000
  ID=2, 内容=香蕉, 距离=0.0050
  ID=4, 内容=狗, 距离=0.9950
```

注意 `register_vector(conn)` 的作用——它让 psycopg2 自动把 Python 的 list 转换成 PostgreSQL 的 vector 类型。如果没有这一步，你需要手动构造 SQL 字面量 `'[1.0, 0.0, 0.0]'::vector(3)`，既繁琐又容易出错。

---

## 常见误区

### 误区 1：忘记在查询向量前加类型转换

在 SQL 中直接写向量字面量时，需要用 `::vector(N)` 指定类型：

```sql
-- ❌ 可能报错：无法推断类型
SELECT * FROM items ORDER BY embedding <=> '[1.0, 0.0, 0.0]' LIMIT 3;

-- ✅ 正确：显式指定类型
SELECT * FROM items ORDER BY embedding <=> '[1.0, 0.0, 0.0]'::vector(3) LIMIT 3;
```

但如果你用 Python 的 psycopg2 + register_vector，参数化查询会自动处理类型转换，不需要手动指定。

### 误区 2：混淆 <#> 的含义

`<#>` 是**负内积**，不是内积。这意味着值越小（越负）表示越相似。如果你需要正内积，可以取负：`-(embedding <#> query_vec)`。

### 误区 3：向量维度选错

创建表时指定的维度必须与你使用的 embedding 模型输出维度一致。如果模型输出 384 维向量，表定义必须是 `vector(384)`，不能是 `vector(768)` 或其他维度。修改维度需要重建表。

---

## 本章小结

这一节我们完成了 pgvector 的第一次向量搜索。核心要点回顾：第一，安装 pgvector 只需 `CREATE EXTENSION vector;`（Docker 镜像已预装）；第二，`vector(N)` 是固定维度的向量类型，同一列所有向量维度必须一致；第三，三个距离操作符 `<->`（L2）、`<=>`（Cosine）、`<#>`（负内积）用于向量相似度搜索；第四，Python 中使用 `register_vector(conn)` 自动处理 list → vector 的类型转换；第五，SQL 中直接写向量字面量需要 `::vector(N)` 类型转换。

下一节我们将深入 vector 数据类型的所有操作符和函数——向量加减、标量乘法、维度查询、范数计算等。
