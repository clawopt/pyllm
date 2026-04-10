# 4.2 Where 过滤器语法

> **Where 是 Chroma 的"SQL WHERE 子句"——掌握它的语法，就掌握了精确过滤的能力**

---

## 这一节在讲什么？

在上一节中我们介绍了 `query()` 方法的 `where` 参数，但只是简单展示了几个用法。实际上 Chroma 的 where 过滤器有一套完整的查询语法，支持比较操作符、逻辑组合、列表匹配和字符串搜索。这套语法虽然不如 SQL 那么强大（没有 JOIN、没有子查询、没有聚合函数），但对于向量数据库的结构化过滤来说已经足够了。

这一节我们要把 where 语法的每一个操作符都讲清楚——它支持什么、不支持什么、容易写错的地方在哪里。这些知识在日常开发中会反复用到，也是面试中"Chroma 的过滤能力有什么限制"这类问题的标准答案。

---

## 基础比较操作符

Chroma 的 where 过滤器支持六种基础比较操作符，用于对 metadata 中的标量字段做条件判断：

| 操作符 | 含义 | 支持的数据类型 | 示例 |
|--------|------|---------------|------|
| `$eq` | 等于 | str, int, float, bool | `{"category": {"$eq": "tech"}}` |
| `$ne` | 不等于 | str, int, float, bool | `{"version": {"$ne": 1}}` |
| `$gt` | 大于 | int, float | `{"price": {"$gt": 100}}` |
| `$gte` | 大于等于 | int, float | `{"version": {"$gte": 2}}` |
| `$lt` | 小于 | int, float | `{"score": {"$lt": 0.5}}` |
| `$lte` | 小于等于 | int, float | `{"created_at": {"$lte": 1704067200}}` |

### 简写形式

对于 `$eq` 操作符，Chroma 支持简写——你可以直接写值而不用显式写 `$eq`：

```python
# 完整写法
results = collection.query(
    query_texts=["查询"],
    where={"category": {"$eq": "tech"}},
    n_results=5
)

# 简写（效果完全相同）
results = collection.query(
    query_texts=["查询"],
    where={"category": "tech"},
    n_results=5
)
```

这两种写法是等价的，简写形式更简洁也更常用。但要注意，**只有 `$eq` 支持简写**，其他操作符必须显式写出。

### 数值比较的细节

`$gt`、`$gte`、`$lt`、`$lte` 只能用于 int 和 float 类型的 metadata 字段。如果你对一个 str 类型的字段使用数值比较，Chroma 会报错：

```python
# ❌ 错误：对字符串字段做数值比较
results = collection.query(
    query_texts=["查询"],
    where={"category": {"$gt": "a"}},  # category 是 str 类型！
    n_results=5
)

# ✅ 正确：对数值字段做数值比较
results = collection.query(
    query_texts=["查询"],
    where={"version": {"$gte": 2}},  # version 是 int 类型
    n_results=5
)
```

布尔值的比较只能用 `$eq` 和 `$ne`：

```python
# 查找已审核的文档
results = collection.query(
    query_texts=["查询"],
    where={"is_reviewed": True},  # 等价于 {"is_reviewed": {"$eq": True}}
    n_results=5
)

# 查找未审核的文档
results = collection.query(
    query_texts=["查询"],
    where={"is_reviewed": {"$ne": True}},
    n_results=5
)
```

---

## 逻辑组合操作符

当单个条件不够用时，你需要用逻辑操作符组合多个条件。Chroma 支持两种逻辑组合：`$and` 和 `$or`。

### $and：所有条件同时满足

```python
# 查找 category=tech 且 version>=2 的文档
results = collection.query(
    query_texts=["查询"],
    where={
        "$and": [
            {"category": "tech"},
            {"version": {"$gte": 2}}
        ]
    },
    n_results=5
)
```

`$and` 接受一个条件列表，所有条件必须同时满足。列表中的每个元素都是一个独立的 where 条件字典。

### $or：任一条件满足

```python
# 查找 category=tech 或 category=science 的文档
results = collection.query(
    query_texts=["查询"],
    where={
        "$or": [
            {"category": "tech"},
            {"category": "science"}
        ]
    },
    n_results=5
)
```

### 嵌套组合

`$and` 和 `$or` 可以嵌套使用，构建复杂的过滤逻辑：

```python
# 查找：(category=tech 且 version>=2) 或 (category=science 且 is_reviewed=True)
results = collection.query(
    query_texts=["查询"],
    where={
        "$or": [
            {
                "$and": [
                    {"category": "tech"},
                    {"version": {"$gte": 2}}
                ]
            },
            {
                "$and": [
                    {"category": "science"},
                    {"is_reviewed": True}
                ]
            }
        ]
    },
    n_results=5
)
```

嵌套组合的语法看起来有点繁琐——每个条件都是一个字典，嵌套层级多了可读性会下降。在实际开发中，建议把复杂的 where 条件封装成函数：

```python
def build_where_clause(category=None, min_version=None, language=None,
                       is_reviewed=None, days=None):
    """构建 where 过滤条件"""
    conditions = []

    if category:
        conditions.append({"category": category})
    if min_version:
        conditions.append({"version": {"$gte": min_version}})
    if language:
        conditions.append({"language": language})
    if is_reviewed is not None:
        conditions.append({"is_reviewed": is_reviewed})
    if days:
        import time
        cutoff = int(time.time()) - days * 24 * 3600
        conditions.append({"created_at": {"$gte": cutoff}})

    if len(conditions) == 0:
        return None
    elif len(conditions) == 1:
        return conditions[0]
    else:
        return {"$and": conditions}


# 使用
where = build_where_clause(
    category="tech",
    min_version=2,
    language="zh",
    is_reviewed=True,
    days=30
)

results = collection.query(
    query_texts=["查询"],
    where=where,
    n_results=5
)
```

---

## $in：列表成员检查

`$in` 操作符检查某个字段的值是否在给定的列表中，类似于 SQL 的 `WHERE category IN ('tech', 'science')`：

```python
# 查找 category 为 tech、science 或 math 的文档
results = collection.query(
    query_texts=["查询"],
    where={"category": {"$in": ["tech", "science", "math"]}},
    n_results=5
)
```

`$in` 只能用于 str 和 int 类型的字段，列表中的值类型必须与字段类型一致。`$in` 是 `$or` 的简写形式——上面的查询等价于：

```python
results = collection.query(
    query_texts=["查询"],
    where={
        "$or": [
            {"category": "tech"},
            {"category": "science"},
            {"category": "math"}
        ]
    },
    n_results=5
)
```

显然 `$in` 更简洁，当匹配值较多时优先使用。

---

## $contains：字符串子串匹配

`$contains` 检查字符串字段是否包含某个子串，类似于 SQL 的 `LIKE '%keyword%'`：

```python
# 查找 source 字段包含 "manual" 的文档
results = collection.query(
    query_texts=["查询"],
    where={"source": {"$contains": "manual"}},
    n_results=5
)
```

`$contains` 只能用于 str 类型的字段，不支持 int、float 或 bool。它的匹配是大小写敏感的——`"Manual"` 不会匹配 `"manual"`。

**$contains 的局限**：它是简单的子串匹配，不是正则表达式，也不是全文搜索。你无法用 `$contains` 实现"以某字符串开头"或"以某字符串结尾"的匹配。如果你需要更复杂的文本匹配，应该在 metadata 中预先处理好匹配字段，或者用 `where_document` 做文档原文的过滤。

---

## where_document：文档内容过滤

除了 `where`（过滤 metadata），Chroma 还提供了 `where_document`（过滤文档原文）。它支持两个操作符：

```python
# $contains：文档原文包含指定子串
results = collection.query(
    query_texts=["查询"],
    where_document={"$contains": "退款"},
    n_results=5
)

# $not_contains：文档原文不包含指定子串
results = collection.query(
    query_texts=["查询"],
    where_document={"$not_contains": "草稿"},
    n_results=5
)
```

`where` 和 `where_document` 可以同时使用，形成"metadata 过滤 + 文档内容过滤"的双重约束：

```python
# 查找 category=after_sales 且文档中包含"退款"的内容
results = collection.query(
    query_texts=["退货流程"],
    where={"category": "after_sales"},
    where_document={"$contains": "退款"},
    n_results=5
)
```

---

## 完整的 Where 语法速查表

```python
# ===== 基础比较 =====
{"field": "value"}                    # 等于（$eq 简写）
{"field": {"$eq": "value"}}           # 等于
{"field": {"$ne": "value"}}           # 不等于
{"field": {"$gt": 10}}                # 大于
{"field": {"$gte": 10}}               # 大于等于
{"field": {"$lt": 100}}               # 小于
{"field": {"$lte": 100}}              # 小于等于

# ===== 列表匹配 =====
{"field": {"$in": ["a", "b", "c"]}}   # 字段值在列表中

# ===== 字符串匹配 =====
{"field": {"$contains": "keyword"}}   # 字段包含子串

# ===== 逻辑组合 =====
{"$and": [condition1, condition2]}    # 所有条件满足
{"$or": [condition1, condition2]}     # 任一条件满足

# ===== 文档内容过滤 =====
{"$contains": "keyword"}              # 文档包含子串
{"$not_contains": "keyword"}          # 文档不包含子串
```

---

## 常见错误与排查

### 错误 1：对不存在的字段做 where 过滤

如果 where 条件中引用了一个 metadata 中不存在的字段，Chroma 不会报错——它只是不会匹配到任何文档。这可能导致你困惑"为什么查询返回空结果"：

```python
# 假设文档的 metadata 中没有 "region" 字段
results = collection.query(
    query_texts=["查询"],
    where={"region": "china"},  # 不会报错，但返回空结果
    n_results=5
)

# ✅ 排查方法：先检查 metadata 中有哪些字段
sample = collection.get(limit=1, include=["metadatas"])
if sample['metadatas'] and sample['metadatas'][0]:
    print(f"可用的 metadata 字段: {list(sample['metadatas'][0].keys())}")
```

### 错误 2：$or 条件的优先级问题

当你混合使用 `$and` 和 `$or` 时，要注意它们的优先级。Chroma 的 where 语法没有隐式的优先级规则——你必须用显式的嵌套来表达逻辑关系：

```python
# ❌ 错误理解：以为这是 (A AND B) OR C
where = {
    "$or": [
        {"$and": [{"category": "tech"}, {"version": {"$gte": 2}}]},
        {"is_reviewed": True}
    ]
}
# 实际含义：(category=tech AND version>=2) OR is_reviewed=True
# 这是正确的嵌套表达

# ❌ 错误写法：试图把 $and 和 $or 平铺在同一层级
where = {
    "$and": [{"category": "tech"}],
    "$or": [{"version": {"$gte": 2}}]
}
# 这会导致不可预期的行为！字典中同层级的 $and 和 $or 顺序不确定
```

### 错误 3：$contains 的误用

```python
# ❌ 错误：用 $contains 做精确匹配
where = {"source": {"$contains": "manual.pdf"}}
# 这会匹配 "user_manual.pdf" 和 "manual.pdf_v2" 等

# ✅ 正确：精确匹配用 $eq
where = {"source": "manual.pdf"}

# ❌ 错误：对数值字段用 $contains
where = {"version": {"$contains": "2"}}  # version 是 int，不支持 $contains

# ✅ 正确：数值比较用 $eq
where = {"version": 2}
```

### 错误 4：where_document 和 where 混淆

```python
# ❌ 错误：把 where_document 的语法用在 where 中
where = {"$contains": "退款"}  # 这不是 where 的合法语法！

# ✅ 正确：where_document 用 $contains，where 用字段名做键
where_document = {"$contains": "退款"}
where = {"category": "after_sales"}
```

---

## 性能提示

Where 过滤的性能与数据量和条件复杂度直接相关。以下是几个实用的优化建议：

1. **优先用 where 而非 where_document**：where 过滤 metadata（SQLite 列存储），where_document 过滤原文（需要扫描文本内容），前者更快。

2. **把高选择性的条件放在前面**：虽然 Chroma 内部会优化条件顺序，但在 `$and` 中把能大幅缩减候选集的条件放在前面，有助于你理解查询逻辑。

3. **避免对高基数字段做 $in 查询**：`{"field": {"$in": [...]}}` 列表过长时，SQLite 需要逐个比较，性能会下降。如果列表超过 100 个值，考虑改用应用层过滤。

4. **用 where 做粗筛，应用层做精筛**：对于 Chroma where 语法无法表达的复杂条件（如正则匹配、范围交集），先用 where 做粗筛拿到候选集，再在应用层做精确过滤。

---

## 本章小结

Where 过滤器是 Chroma 结构化过滤的核心语法，掌握它才能实现精确的混合查询。核心要点回顾：第一，六种基础比较操作符（`$eq/$ne/$gt/$gte/$lt/$lte`）覆盖了标量值的所有比较需求，其中 `$eq` 支持简写；第二，`$and` 和 `$or` 支持任意嵌套的逻辑组合，复杂条件建议封装成函数；第三，`$in` 是多值匹配的简写，`$contains` 是子串匹配但只支持 str 类型；第四，`where_document` 做文档原文过滤，性能不如 where，优先用 metadata 字段替代；第五，不存在的字段不会报错但会返回空结果，调试时先检查 metadata schema；第六，where 的性能是 O(N) 全表扫描，高选择性条件优先、高基数字段避免 `$in`。

下一节我们将讲多阶段查询与 Re-ranking——如何用"粗筛+精排"的策略在保证速度的同时提升检索精度。
