# 2.2 JSONB 与半结构化数据

> **JSONB 是 PostgreSQL 的"万能口袋"——当你不确定数据结构时，把它塞进 JSONB 就对了**

---

## 这一节在讲什么？

在 RAG 系统中，文档的元数据（metadata）往往是灵活多变的——有的文档有"页码"字段，有的有"作者"字段，有的有"标签"列表。如果为每种可能的字段都建一列，表结构会变得非常臃肿且难以维护。JSONB 就是解决这个问题的——它允许你在单个列中存储任意的 JSON 数据，同时支持高效的查询和索引。这一节我们要讲清楚 JSONB 的操作语法、GIN 索引加速、以及它在 RAG 系统中与 pgvector 配合使用的典型模式。

---

## JSONB vs JSON

PostgreSQL 支持两种 JSON 数据类型：`JSON` 和 `JSONB`。它们的核心区别在于存储方式：

| 维度 | JSON | JSONB |
|------|------|-------|
| 存储方式 | 文本（原样保存） | 二进制（解析后存储） |
| 写入速度 | 快（不需要解析） | 慢（需要解析和转换） |
| 查询速度 | 慢（每次查询都要解析） | 快（已预解析） |
| 支持索引 | 不支持 | 支持 GIN 索引 |
| 键顺序 | 保留原始顺序 | 按键名排序 |
| 空白处理 | 保留空格和换行 | 去除多余空白 |

**结论**：在几乎所有场景下，你应该使用 `JSONB` 而不是 `JSON`。JSONB 的查询性能优势和索引支持远比写入时多出的解析开销重要。

```sql
-- 创建带 JSONB 列的表
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    metadata JSONB
);

-- 插入 JSONB 数据
INSERT INTO documents (content, metadata) VALUES
    ('退款政策', '{"source": "manual.pdf", "page": 15, "tags": ["退款", "售后"]}'),
    ('安装指南', '{"source": "install.md", "version": 2, "language": "zh"}'),
    ('API文档', '{"source": "api.yaml", "category": "technical", "is_reviewed": true}');
```

---

## JSONB 查询操作符

JSONB 提供了丰富的操作符，让你可以精确地访问和查询嵌套的 JSON 数据：

### 提取操作符

```sql
-- -> 提取 JSON 对象（返回 JSONB 类型）
SELECT metadata -> 'source' FROM documents;
-- 结果: "manual.pdf" / "install.md" / "api.yaml"（带引号的 JSON 字符串）

-- ->> 提取文本（返回 TEXT 类型）
SELECT metadata ->> 'source' FROM documents;
-- 结果: manual.pdf / install.md / api.yaml（纯文本，不带引号）

-- 提取嵌套字段
SELECT metadata -> 'author' ->> 'name' FROM documents WHERE metadata -> 'author' IS NOT NULL;

-- 提取数组元素
SELECT metadata -> 'tags' -> 0 FROM documents WHERE metadata -> 'tags' IS NOT NULL;
-- 结果: "退款"（第一个标签）
```

### 包含操作符

```sql
-- @> : 左侧 JSONB 是否包含右侧（右侧是左侧的子集）
SELECT * FROM documents WHERE metadata @> '{"source": "manual.pdf"}';
-- 匹配 metadata 中 source 为 "manual.pdf" 的行

SELECT * FROM documents WHERE metadata @> '{"category": "technical", "is_reviewed": true}';
-- 匹配同时包含这两个键值对的行

-- ? : 是否包含某个键
SELECT * FROM documents WHERE metadata ? 'version';
-- 匹配 metadata 中有 version 键的行

-- ?| : 是否包含任意一个键
SELECT * FROM documents WHERE metadata ?| array['version', 'page'];
-- 匹配 metadata 中有 version 或 page 键的行

-- ?& : 是否包含所有键
SELECT * FROM documents WHERE metadata ?& array['source', 'page'];
-- 匹配 metadata 中同时有 source 和 page 键的行
```

### 修改 JSONB

```sql
-- || : 合并 JSONB（右侧覆盖左侧的同名键）
UPDATE documents SET metadata = metadata || '{"version": 3}' WHERE id = 1;

-- - : 删除键
UPDATE documents SET metadata = metadata - 'page' WHERE id = 1;

-- #- : 删除嵌套路径
UPDATE documents SET metadata = metadata #- '{author,email}' WHERE id = 1;
```

---

## GIN 索引加速 JSONB 查询

没有索引时，JSONB 查询需要对每行的 metadata 做全表扫描。GIN（Generalized Inverted Index，通用倒排索引）为 JSONB 的键和值建立了索引，让 `@>`、`?`、`?|` 等操作符的查询速度提升几个数量级：

```sql
-- 创建 GIN 索引
CREATE INDEX idx_documents_metadata ON documents USING GIN (metadata);

-- 创建带 jsonb_path_ops 的 GIN 索引（更小更快，但只支持 @> 操作符）
CREATE INDEX idx_documents_metadata_ops ON documents USING GIN (metadata jsonb_path_ops);
```

两种 GIN 索引的区别：

| 索引 | 支持的操作符 | 索引大小 | 适用场景 |
|------|------------|---------|---------|
| `USING GIN (metadata)` | `@>`、`?`、`?|`、`?&` | 较大 | 需要多种查询操作符 |
| `USING GIN (metadata jsonb_path_ops)` | 只有 `@>` | 较小（约1/3） | 只用 `@>` 做包含查询 |

**实践建议**：如果你的 JSONB 查询主要用 `@>` 操作符（这是最常见的模式），使用 `jsonb_path_ops` 更高效。如果需要 `?` 等其他操作符，使用默认的 GIN 索引。

---

## JSONB 在 RAG 中的典型用途

在 RAG 系统中，JSONB 是存储文档 metadata 的理想选择。它比 Chroma 的 metadata 字典更强大——因为你可以用 SQL 的全部表达能力来查询它：

```sql
-- RAG 文档表的典型设计
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    embedding vector(384),       -- pgvector 向量列（后续章节详解）
    metadata JSONB,              -- 灵活的元数据
    created_at TIMESTAMP DEFAULT NOW()
);

-- 插入带丰富 metadata 的文档
INSERT INTO documents (content, embedding, metadata) VALUES
    ('退款政策：购买后7天内可无条件退款', '[0.1, 0.2, ...]',
     '{"source": "user_manual.pdf", "category": "after_sales", "page": 15, "version": 2, "tags": ["退款", "售后"]}'),
    ('安装指南：下载安装包并运行', '[0.3, 0.4, ...]',
     '{"source": "install_guide.md", "category": "technical", "version": 1, "language": "zh"}');

-- JSONB + 向量混合查询（pgvector 的杀手锏！）
SELECT id, content,
       embedding <=> '[0.15, 0.25, ...]' AS distance
FROM documents
WHERE metadata @> '{"category": "after_sales"}'
  AND metadata ->> 'version' >= '2'
ORDER BY embedding <=> '[0.15, 0.25, ...]'
LIMIT 5;
```

这条 SQL 同时做了三件事：用 `@>` 过滤 category、用 `->>` 过滤 version、用 `<=>` 做向量相似度排序。这是独立向量数据库很难实现的——Chroma 的 where 过滤不支持 JSONB 嵌套查询，Milvus 的标量过滤也不支持 JSON 路径表达式。

---

## 常见误区

### 误区 1：把所有数据都塞进 JSONB

JSONB 适合存储灵活多变的半结构化数据，但不适合存储需要频繁查询和过滤的核心字段。如果你经常按 `category` 过滤，应该把它作为独立的列并建 B-tree 索引，而不是藏在 JSONB 里面。

### 误区 2：JSONB 查询不需要索引

没有 GIN 索引的 JSONB 查询是全表扫描，在大数据量下性能很差。如果你的 JSONB 查询是高频操作，务必创建 GIN 索引。

### 误区 3：JSONB 的 `->>` 返回的是数字

`->>` 返回的永远是文本（TEXT 类型）。即使 JSON 中存的是数字，`metadata ->> 'version'` 返回的是 `'2'`（字符串），不是 `2`（整数）。如果需要做数值比较，必须显式转换：

```sql
-- ❌ 错误：字符串比较，'9' > '10'（按字典序）
SELECT * FROM documents WHERE metadata ->> 'version' > '2';

-- ✅ 正确：转换为整数后比较
SELECT * FROM documents WHERE (metadata ->> 'version')::int > 2;
```

---

## 本章小结

JSONB 是 PostgreSQL 处理半结构化数据的利器，也是 pgvector 在 RAG 系统中存储灵活 metadata 的最佳搭档。核心要点回顾：第一，JSONB 比 JSON 更快且支持索引，几乎所有场景都应该用 JSONB；第二，`->` 返回 JSONB 类型，`->>` 返回 TEXT 类型，`@>` 做包含查询是最常用的模式；第三，GIN 索引加速 JSONB 查询，`jsonb_path_ops` 更小更快但只支持 `@>`；第四，JSONB + pgvector 的混合查询是 pgvector 的杀手锏——一条 SQL 同时做 JSONB 过滤和向量搜索；第五，核心过滤字段应该作为独立列建 B-tree 索引，不要全部塞进 JSONB。

下一节我们将学习事务与并发控制——理解 ACID 事务如何保证 RAG 系统中结构化数据和向量数据的原子性。
