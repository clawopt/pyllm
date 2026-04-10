# 3.3 vector 数据类型详解

> **vector 不只是一个数组——它是 PostgreSQL 的一等公民，拥有自己的操作符、函数和索引**

---

## 这一节在讲什么？

上一节我们用 `<->` 和 `<=>` 做了第一次向量搜索，但 pgvector 的 vector 类型远不止距离计算。它支持向量加减、标量乘法、维度查询、范数计算等操作，这些操作在数据预处理和调试时非常有用。这一节我们要把 vector 类型的所有操作符和函数都讲清楚，确保你在后续的 RAG 实战中能灵活运用。

---

## vector(N) 类型基础

`vector(N)` 是 pgvector 注册的自定义数据类型，N 表示向量的维度。它有以下特性：

- **固定维度**：创建表时指定维度，同一列所有向量维度必须一致
- **存储格式**：每个分量占 4 字节（float32），一个 384 维向量占 1536 字节
- **维度上限**：pgvector 0.7.x 支持最大 2000 维
- **字面量语法**：`'[1.0, 2.0, 3.0]'::vector(3)`

```sql
-- 创建不同维度的向量列
CREATE TABLE models (
    id SERIAL PRIMARY KEY,
    name TEXT,
    emb_small vector(384),    -- SentenceTransformers 小模型
    emb_large vector(1536)    -- OpenAI embedding
);

-- 插入数据
INSERT INTO models (name, emb_small, emb_large) VALUES
    ('test', array_to_vector(ARRAY[0.1, 0.2, ...]::float4[], 384), array_to_vector(ARRAY[0.1, 0.2, ...]::float4[], 1536));
```

---

## 距离操作符详解

pgvector 的三个距离操作符是向量搜索的核心，理解它们的数学含义和适用场景至关重要：

### <-> ：L2 距离（欧氏距离）

$$d_{L2}(a, b) = \sqrt{\sum_{i=1}^{d}(a_i - b_i)^2}$$

L2 距离衡量的是两个向量在空间中的直线距离。值域 $[0, +\infty)$，0 表示完全相同。

```sql
-- L2 距离搜索
SELECT content, embedding <-> '[1.0, 0.0, 0.0]' AS l2_dist
FROM items ORDER BY embedding <-> '[1.0, 0.0, 0.0]' LIMIT 3;
```

**适用场景**：需要同时考虑方向和模长的差异。但 L2 距离受高维灾难影响——维度越高，所有向量之间的 L2 距离趋于相近，区分度下降。

### <=> ：余弦距离

$$d_{cos}(a, b) = 1 - \frac{a \cdot b}{||a|| \cdot ||b||}$$

余弦距离衡量的是两个向量方向的差异，忽略模长。值域 $[0, 2]$，0 表示方向完全相同，1 表示正交，2 表示方向相反。

```sql
-- 余弦距离搜索
SELECT content, embedding <=> '[1.0, 0.0, 0.0]' AS cos_dist
FROM items ORDER BY embedding <=> '[1.0, 0.0, 0.0]' LIMIT 3;
```

**适用场景**：NLP 语义搜索的首选。大多数 embedding 模型输出归一化向量，余弦距离比 L2 更稳定。

### <#> ：负内积

$$d_{ip}(a, b) = -(a \cdot b) = -\sum_{i=1}^{d}a_i b_i$$

负内积是内积的取负值。值越小（越负）表示内积越大，即越相似。对于归一化向量，负内积 = 余弦距离 - 1。

```sql
-- 内积搜索（注意：值越小越相似）
SELECT content, embedding <#> '[1.0, 0.0, 0.0]' AS neg_ip
FROM items ORDER BY embedding <#> '[1.0, 0.0, 0.0]' LIMIT 3;
```

**适用场景**：向量已归一化且追求最快计算速度时。内积比余弦距离少一次归一化除法。

### 三种距离的等价关系

对于**归一化向量**（$||a|| = ||b|| = 1$），三种距离排序完全等价：

$$d_{cos} = 1 - IP = 1 + d_{ip}$$
$$d_{L2}^2 = 2 \cdot d_{cos} = 2(1 - IP)$$

这意味着对于归一化向量，选哪种距离度量不影响搜索结果的排序——只是数值不同。大多数 SentenceTransformers 模型输出归一化向量，所以用 `<=>` 或 `<->` 结果一致。

---

## 向量运算操作符

pgvector 支持基本的向量算术运算，这在数据预处理时很有用：

```sql
-- 向量加法
SELECT '[1.0, 2.0, 3.0]'::vector(3) + '[0.5, 0.5, 0.5]'::vector(3);
-- 结果: [1.5, 2.5, 3.5]

-- 向量减法
SELECT '[1.0, 2.0, 3.0]'::vector(3) - '[0.5, 0.5, 0.5]'::vector(3);
-- 结果: [0.5, 1.5, 2.5]

-- 标量乘法
SELECT 2.0 * '[1.0, 2.0, 3.0]'::vector(3);
-- 结果: [2.0, 4.0, 6.0]
```

### 实际应用：计算向量均值

```sql
-- 计算某个分类下所有文档向量的均值（质心）
SELECT category, avg(embedding) AS centroid
FROM documents
GROUP BY category;
```

`avg()` 函数对 vector 类型做了特殊支持，直接计算每个分量的平均值。这在"按分类聚合向量表示"的场景中非常有用。

---

## 向量函数

### vector_dims：获取维度

```sql
SELECT vector_dims(embedding) FROM items LIMIT 1;
-- 结果: 3
```

### vector_norm：获取 L2 范数

```sql
SELECT vector_norm(embedding) FROM items LIMIT 1;
-- 结果: 1.0（如果是归一化向量）
```

L2 范数就是向量的模长 $||v|| = \sqrt{\sum v_i^2}$。归一化向量的范数为 1。你可以用这个函数检查向量是否已归一化：

```sql
-- 查找未归一化的向量（范数不为 1）
SELECT id, vector_norm(embedding) AS norm
FROM items
WHERE ABS(vector_norm(embedding) - 1.0) > 0.01;
```

### 显式距离函数

除了操作符，pgvector 还提供了显式的距离函数，效果完全相同：

```sql
-- 等价于 embedding <-> query_vec
SELECT l2_distance(embedding, '[1.0, 0.0, 0.0]') FROM items;

-- 等价于 embedding <=> query_vec
SELECT cosine_distance(embedding, '[1.0, 0.0, 0.0]') FROM items;

-- 等价于 embedding <#> query_vec
SELECT inner_product(embedding, '[1.0, 0.0, 0.0]') FROM items;
```

显式函数在需要嵌套调用时更方便，比如 `WHERE cosine_distance(embedding, query_vec) < 0.5`。

---

## 向量类型转换

pgvector 提供了 vector 与 PostgreSQL 数组之间的转换函数：

```sql
-- vector → float4 数组
SELECT vector_to_array(embedding) FROM items LIMIT 1;
-- 结果: {1.0,0.0,0.0}

-- float4 数组 → vector
SELECT array_to_vector(ARRAY[1.0, 0.0, 0.0]::float4[], 3);
-- 结果: [1.0, 0.0, 0.0]
```

这在需要用 PostgreSQL 的数组函数处理向量数据时很有用，比如提取某个分量的值：

```sql
-- 获取向量的第一个分量
SELECT (vector_to_array(embedding))[1] AS first_dim FROM items LIMIT 1;
```

---

## 常见误区

### 误区 1：vector 类型可以存储任意维度的向量

`vector(N)` 的维度在创建表时就固定了。如果你需要存储不同维度的向量，必须创建不同的列或不同的表。一个常见的做法是创建多个向量列：

```sql
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT,
    emb_small vector(384),    -- 小模型向量
    emb_large vector(1536)    -- 大模型向量
);
```

### 误区 2：<#> 返回的是内积

`<#>` 返回的是**负内积**，不是内积。值越小（越负）表示越相似。如果你需要正内积，取负即可：`-(embedding <#> query_vec)`。

### 误区 3：向量运算可以跨维度

向量加减要求两个向量维度相同。不同维度的向量不能做加减运算：

```sql
-- ❌ 维度不匹配
SELECT '[1.0, 2.0]'::vector(2) + '[1.0, 2.0, 3.0]'::vector(3);
-- ERROR: cannot add vectors of different dimensions
```

---

## 本章小结

vector 是 pgvector 注册的一等数据类型，拥有完整的操作符和函数支持。核心要点回顾：第一，`vector(N)` 是固定维度类型，维度上限 2000；第二，三个距离操作符 `<->`（L2）、`<=>`（Cosine）、`<#>`（负内积）是向量搜索的核心，对于归一化向量三者排序等价；第三，向量加减和标量乘法支持基本算术运算，`avg()` 函数支持向量聚合；第四，`vector_dims()` 查维度、`vector_norm()` 查范数、`l2_distance()`/`cosine_distance()`/`inner_product()` 是显式距离函数；第五，`vector_to_array()` 和 `array_to_vector()` 支持 vector 与数组的互转。

下一章我们将进入 pgvector 的 CRUD 操作和混合查询——这是 pgvector 相比独立向量数据库的核心优势所在。
