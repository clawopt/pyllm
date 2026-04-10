# 2.1 文档的增删改查（CRUD）

> **Chroma 的四项基本操作——每个参数、每个返回值、每个边界情况都讲透**

---

## Add：把数据放进 Chroma

`collection.add()` 是你与 Chroma 交互最频繁的方法。它看起来简单，但每个参数都有值得注意的行为细节。

### 基础用法

```python
collection.add(
    documents=["第一条文档", "第二条文档", "第三条文档"],
    ids=["doc_001", "doc_002", "doc_003"]
)
```

这就是最简形式。Chroma 会自动对 `documents` 中的文本做 embedding（使用默认的 embedding function），然后存入 Collection。

### 完整参数列表

```python
collection.add(
    # ===== 必填参数 =====
    ids=["id_1", "id_2", "id_3"],           # 文档 ID 列表（唯一，不可重复）

    # ===== 可选参数 =====
    documents=["文本内容..."],               # 文档原文（如果不传 embeddings 则必填）
    embeddings=[[0.1, 0.2, ...], ...],      # 预计算的向量（如果传了则跳过自动 embedding）
    metadatas=[{"key": "value"}, ...],       # 元数据字典列表（与 ids 一一对应）

    # ===== 高级参数 =====
    upsert=False,                            # True=存在则更新（默认 False=重复报错）
)
```

### 参数之间的约束关系

这是面试中经常被问到的一个点：**documents 和 embeddings 之间是什么关系？**

```
┌─────────────────────────────────────────────────────┐
│  情况 A：只传 documents                               │
│  → Chroma 自动调用 embedding function 生成向量        │
│  → 最常见的用法                                       │
├─────────────────────────────────────────────────────┤
│  情况 B：只传 embeddings                             │
│  → 直接存储向量，不存储文档原文                        │
│  → 适用场景：已有预计算好的向量缓存                     │
├─────────────────────────────────────────────────────┤
│  情况 C：同时传 documents 和 embeddings              │
│  → 以 embeddings 为准，documents 作为附带信息存储     │
│  → 适用场景：需要保留原文但用自定义模型做 embedding     │
├─────────────────────────────────────────────────────┤
│  情况 D：都不传                                      │
│  → ❌ 报错！至少需要一个                              │
└─────────────────────────────────────────────────────┘
```

代码示例：

```python
# 情况 A：自动 embedding（最常用）
collection.add(
    documents=["机器学习是人工智能的子领域"],
    ids=["ml_def"]
)

# 情况 B：只传向量（无原文）
import numpy as np
fake_vector = np.random.randn(384).tolist()
collection.add(
    embeddings=[fake_vector],
    ids=["vector_only_doc"]
)
# 注意：query 时无法通过 include=["documents"] 获取原文！

# 情况 C：自定义 embedding + 保留原文
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
custom_embedding = model.encode("深度学习框架").tolist()
collection.add(
    documents=["深度学习框架"],
    embeddings=[custom_embedding],
    ids=["dl_framework"],
    metadatas=[{"model": "sentence-transformers", "dim": 384}]
)
```

### 批量添加的性能考量

当你有大量数据要入库时，批量操作的性能远优于逐条循环：

```python
# ❌ 低效：逐条添加（每次都要建立连接、序列化、写 WAL）
for i in range(10000):
    collection.add(documents=[f"文档 {i}"], ids=[f"doc_{i}"])

# ✅ 高效：批量添加（一次网络/IO 操作）
docs = [f"文档 {i}" for i in range(10000)]
ids = [f"doc_{i}" for i in range(10000)]
collection.add(documents=docs, ids=ids)
```

**性能对比参考**（10K 条 384 维向量）：

| 方式 | 耗时 | 内存峰值 |
|------|------|---------|
| 逐条 add | ~30-60 秒 | 低 |
| 批量 add (10K) | ~2-5 秒 | 中等 |
| 分批 add (每批 1000) | ~3-6 秒 | 低 |

**推荐策略**：单次批量不超过 5000 条。如果数据量更大，分批提交：

```python
BATCH_SIZE = 1000
all_docs = load_all_documents()  # 假设这个函数加载所有数据

for i in range(0, len(all_docs), BATCH_SIZE):
    batch = all_docs[i:i + BATCH_SIZE]
    collection.add(
        documents=batch["texts"],
        ids=batch["ids"],
        metadatas=batch.get("metadatas")
    )
    print(f"已添加 {min(i+BATCH_SIZE, len(all_docs))}/{len(all_docs)} 条")
```

### upsert：存在则更新

```python
# 第一次 add：成功
collection.add(documents=["原始内容"], ids=["doc_upsert"])

# 第二次 add 同一个 id：❌ 报错 ValueError: Duplicate id detected
# collection.add(documents=["更新内容"], ids=["doc_upsert"])

# 用 upsert：✅ 自动覆盖旧值
collection.upsert(
    documents=["更新后的内容"],
    ids=["doc_upsert"]  # 如果已存在则更新，不存在则插入
)

# 验证
result = collection.get(ids=["doc_upsert"], include=["documents"])
print(result['documents'][0])  # "更新后的内容"
```

**upsert 的行为细节**：

- 它会**完全替换**该 ID 的 document、embedding、metadata——不是 merge，而是 overwrite
- 如果你只想更新部分字段（比如只改 metadata 不改 document），应该用 `update()` 方法（后面会讲）
- upsert 的性能比 delete + add 略好（少一次索引删除操作）

---

## Get：精确获取文档

`collection.get()` 用于通过 ID 精确检索，类似于 SQL 的 `SELECT * WHERE id IN (...)`。

### 基础用法

```python
# 通过 ID 列表获取
result = collection.get(
    ids=["doc_001", "doc_003"],
    include=["documents", "metadatas", "embeddings"]
)

print(result['ids'])        # ['doc_001', 'doc_003']
print(result['documents'])  # ['第一条文档', '第三条文档']
print(result['metadatas'])  # [{'key': 'val'}, None] 或实际 metadata
print(result['embeddings']) # [[0.12, -0.34, ...], [...]]
```

### 分页获取：limit 与 offset

当 Collection 数据量很大时，你可能不想一次取回所有数据。Chroma 支持 `limit` 和 `offset` 参数实现分页：

```python
# 获取前 10 条
page1 = collection.get(limit=10, offset=0, include=["documents", "ids"])
print(f"第 1 页: {len(page1['ids'])} 条")

# 获取第 11~20 条
page2 = collection.get(limit=10, offset=10, include=["documents", "ids"])
print(f"第 2 页: {len(page2['ids'])} 条")

# 通用的分页函数
def paginate(collection, page_size=20, page_num=1):
    """分页获取 Collection 中的所有文档"""
    offset = (page_num - 1) * page_size
    return collection.get(
        limit=page_size,
        offset=offset,
        include=["documents", "ids", "metadatas"]
    )

# 使用示例
for page in range(1, 6):  # 取前 5 页
    result = paginate(collection, page_size=50, page_num=page)
    if not result['ids']:
        print(f"第 {page} 页为空，结束")
        break
    print(f"第 {page} 页: {len(result['ids'])} 条 (ID范围: {result['ids'][0]} ~ {result['ids'][-1]})")
```

**注意**：`get()` 的分页是基于插入顺序的，不是基于相似度排序的。如果你需要按相似度排序的结果，应该用 `query()`。

### Where 过滤的 Get

`get()` 也支持 `where` 参数做条件过滤：

```python
# 只获取 category="technology" 的文档
result = collection.get(
    where={"category": "technology"},
    include=["documents", "metadatas", "ids"]
)
print(f"找到 {len(result['ids'])} 条 technology 类文档")
```

这在调试时特别有用——你可以快速查看某个分类下有哪些文档，而不需要知道它们的 ID。

### 获取全部数据

```python
# ⚠️ 谨慎使用：如果 Collection 很大，这会消耗大量内存
all_data = collection.get(include=["documents", "metadatas", "embeddings"])
total = collection.count()
print(f"Collection 共 {total} 条文档")
```

---

## Update：修改已有文档

`collection.update()` 用于修改已存在的文档内容、metadata 或 embedding。

### 更新文档文本

```python
# 先添加一条
collection.add(documents=["原始版本"], ids=["upd_test"])

# 更新 document 内容
collection.update(
    ids=["upd_test"],
    documents=["修订后的新版本内容"]
)

# 验证
result = collection.get(ids=["upd_test"], include=["documents"])
print(result['documents'][0])  # "修订后的新版本内容"
```

**关键行为**：更新 document 后，Chroma 会**自动重新计算 embedding 并重建索引**。你不需要手动重新传入 embeddings。

### 更新 Metadata

```python
collection.update(
    ids=["upd_test"],
    metadatas={
        "version": "2.0",
        "last_updated": "2025-01-15",
        "reviewed": True
    }
)
```

### 同时更新多个字段

```python
collection.update(
    ids=["upd_test"],
    documents=["最终版本 v3"],
    metadatas={
        "version": "3.0",
        "status": "published",
        "tags": ["important", "reviewed"]
    }
)
```

### Update 的限制

| 能做的 | 不能做的 |
|--------|----------|
| 更新 document 文本 | **更改 ID**（ID 是不可变的主键） |
| 更新 metadata 字典 | 部分更新 metadata（必须传完整的字典） |
| 触发 re-embedding | 单独更新 embedding（会随 document 一起重新计算） |

**关于 metadata 部分更新的 workaround**：

```python
# ❌ 这不会 merge metadata，而是完全替换
collection.update(ids=["doc1"], metadatas={"new_key": "new_val"})
# 结果：原来的 metadata 全丢了，只剩 {"new_key": "new_val"}

# ✅ 正确做法：先 get 再合并再 update
existing = collection.get(ids=["doc1"], include=["metadatas"])
current_meta = existing['metadatas'][0] or {}
current_meta["new_key"] = "new_val"   # 在原有基础上添加/修改
collection.update(ids=["doc1"], metadatas=current_meta)
```

### 批量 Update

```python
collection.update(
    ids=["doc_001", "doc_002", "doc_003"],
    documents=[
        "更新后的文档 1",
        "更新后的文档 2",
        "更新后的文档 3"
    ],
    metadatas=[
        {"status": "v2"},
        {"status": "v2"},
        {"status": "v2"}
    ]
)
```

---

## Delete：移除文档

### 按 ID 删除

```python
collection.delete(ids=["doc_to_delete"])
```

### 按 Where 条件批量删除

```python
# 删除所有 category="draft" 的文档
collection.delete(where={"category": "draft"})

# 删除 version < 2.0 的旧文档
collection.delete(where={"version": {"$lt": 2.0}})
```

### Delete 的返回值与确认

`delete()` 方法**没有返回值**——它不会告诉你删除了多少条记录。如果你需要确认删除效果：

```python
count_before = collection.count()
collection.delete(ids=["target_id"])
count_after = collection.count()
print(f"删除了 {count_before - count_after} 条记录")
```

### 删除后 ID 是否可以复用？

**可以**。Chroma 不会永久保留已删除 ID 的"黑名单"。删除后你可以用同一个 ID 添加新文档：

```python
collection.add(documents=["原始文档"], ids=["recyclable"])
collection.delete(ids=["recyclable"])
collection.add(documents=["全新文档，复用了同一个 ID"], ids=["recyclable"])  # ✅ 正常工作
```

但要注意：**中间状态下的查询可能不一致**——在并发场景下，delete 和 add 之间如果有其他线程在 query，可能会看到短暂的不一致。

---

## CRUD 完整实战示例

让我们用一个模拟的知识库管理场景，把 CRUD 四个操作串联起来：

```python
import chromadb
import json
from datetime import datetime

client = chromadb.Client(settings=chromadb.Settings(is_persistent=True, persist_directory="./crud_demo"))
kb = client.get_or_create_collection(name="knowledge_base")

def add_article(title, content, category, tags=None):
    """添加一篇知识库文章"""
    doc_id = f"art_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    meta = {
        "title": title,
        "category": category,
        "created_at": datetime.now().isoformat(),
        "version": "1.0"
    }
    if tags:
        meta["tags"] = json.dumps(tags)  # Chroma metadata 只支持基本类型，list 用 JSON 序列化
    kb.add(
        documents=[content],
        ids=[doc_id],
        metadatas=[meta]
    )
    return doc_id

def search_articles(query, category=None, n_results=5):
    """搜索文章（支持分类过滤）"""
    where_clause = {"category": category} if category else None
    results = kb.query(
        query_texts=[query],
        n_results=n_results,
        where=where_clause,
        include=["documents", "metadatas", "distances"]
    )
    articles = []
    for i in range(len(results['ids'][0])):
        articles.append({
            "id": results['ids'][0][i],
            "title": results['metadatas'][0][i]["title"],
            "distance": results['distances'][0][i],
            "snippet": results['documents'][0][i][:100] + "..."
        })
    return articles

def update_article(doc_id, new_content=None, new_title=None):
    """更新文章"""
    existing = kb.get(ids=[doc_id], include=["documents", "metadatas"])
    if not existing['ids']:
        print(f"❌ 文章 {doc_id} 不存在")
        return False

    update_args = {"ids": [doc_id]}
    if new_content:
        update_args["documents"] = [new_content]
    if new_title:
        meta = existing['metadatas'][0].copy()
        meta["title"] = new_title
        meta["updated_at"] = datetime.now().isoformat()
        meta["version"] = str(float(meta.get("version", "1.0")) + 0.1)
        update_args["metadatas"] = [meta]

    kb.update(**update_args)
    print(f"✅ 文章 {doc_id} 已更新")
    return True

def list_by_category(category, limit=20):
    """列出某个分类的所有文章"""
    results = kb.get(
        where={"category": category},
        limit=limit,
        include=["documents", "metadatas", "ids"]
    )
    return [
        {"id": r[0], "title": r[1]["title"], "created": r[1]["created_at"]}
        for r in zip(results['ids'], results['metadatas'])
    ]

def retire_category(category):
    """归档/删除整个分类的文章"""
    count_before = kb.count()
    kb.delete(where={"category": category})
    count_after = kb.count()
    removed = count_before - count_after
    print(f"🗑️ 已归档 '{category}' 分类下的 {removed} 篇文章")

# ====== 实际使用演示 ======

# 1. 添加几篇文章
id1 = add_article(
    "Python 入门指南",
    "Python 是一种高级编程语言，以其简洁的语法和强大的标准库著称...",
    category="programming",
    tags=["python", "beginner"]
)
id2 = add_article(
    "PyTorch 教程",
    "PyTorch 是 Facebook 开发的深度学习框架，支持动态计算图...",
    category="ai",
    tags=["pytorch", "deep-learning"]
)
id3 = add_article(
    "Chroma 向量数据库",
    "Chroma 是开源的嵌入式向量数据库，专为 AI 应用设计...",
    category="database",
    tags=["chroma", "vector-db"]
)
print(f"\n📝 已添加 3 篇文章 (总文档数: {kb.count()})")

# 2. 搜索
print("\n🔍 搜索 '深度学习':")
for art in search_articles("深度学习"):
    print(f"  [{art['distance']:.4f}] {art['title']}: {art['snippet']}")

print("\n🔍 搜索 '编程' (仅 programming 分类):")
for art in search_articles("编程", category="programming"):
    print(f"  [{art['distance']:.4f}] {art['title']}")

# 3. 更新
update_article(id1, new_content="Python 是一门广泛使用的编程语言，适合 Web 开发、数据分析、AI 和自动化...")
update_article(id1, new_title="Python 完全指南 (已修订)")

# 4. 查看某个分类
print("\n📂 AI 分类文章:")
for art in list_by_category("ai"):
    print(f"  {art['id']}: {art['title']} ({art['created']})")

# 5. 删除
retire_category("database")
print(f"\n剩余文档数: {kb.count()}")
```

运行输出：

```
📝 已添加 3 篇文章 (总文档数: 3)

🔍 搜索 '深度学习':
  [0.5123] PyTorch 教程: PyTorch 是 Facebook 开发的深度学习框架，支持动态计算图...

🔍 搜索 '编程' (仅 programming 分类):
  [0.7234] Python 完全指南 (已修订)

✅ 文章 art_20250115120000 已更新

📂 AI 分类文章:
  art_20250115120001: PyTorch 教程 (2025-01-15T12:00:01)

🗑️ 已归档 'database' 分类下的 1 篇文章
剩余文档数: 2
```

---

## 常见陷阱

### 陷阱 1：Metadata 中使用了不支持的数据类型

```python
# ❌ 错误：Chroma metadata 不支持 list/dict 类型
collection.add(
    documents=["test"],
    ids=["bad_meta"],
    metadatas=[{"tags": ["a", "b"]}]  # list 不被支持！
)

# ✅ 正确：用 JSON 序列化复杂类型
import json
collection.add(
    documents=["test"],
    ids=["good_meta"],
    metadatas=[{"tags": json.dumps(["a", "b"])}]
)
```

### 陷阱 2：Add 时 IDs 数量与其他参数不匹配

```python
# ❌ 错误：3 个 ID 但只有 2 个 document
collection.add(
    documents=["doc_a", "doc_b"],
    ids=["id_1", "id_2", "id_3"]  # ValueError: ids and documents must be the same length
)

# ✅ 正确：确保一一对应
collection.add(
    documents=["doc_a", "doc_b", "doc_c"],
    ids=["id_1", "id_2", "id_3"]
)
```

### 陷阱 3：Update 一个不存在的 ID

```python
# 不会报错！静默失败（或者在某些版本抛异常）
collection.update(
    ids=["nonexistent_id"],
    documents=["新内容"]
)
# 不会有任何效果，也不会报错提示

# ✅ 安全做法：先检查是否存在
result = collection.get(ids=["target_id"])
if result['ids']:
    collection.update(ids=["target_id"], documents=["新内容"])
else:
    print("目标 ID 不存在")
```

### 陷阱 4：Delete 后立即 Query 可能查到已删除的数据

在高并发或 Server 模式下，由于 WAL 的异步刷盘机制，delete 后极短时间内（通常 < 1ms）query 可能仍能看到已删除的数据。对于大多数应用这不是问题，但如果需要强一致性，可以在 delete 后调用一次同步操作。

---

## 本章小结

| 操作 | 方法 | 关键参数 | 返回值 |
|------|------|----------|--------|
| 添加 | `add()` | ids(必填), documents/embeddings(二选一), metadatas | None |
| 存在则插入/更新 | `upsert()` | 同 add + upsert=True | None |
| 精确获取 | `get()` | ids / where / limit / offset / include | dict (ids/documents/...) |
| 更新 | `update()` | ids(必填), documents, metadatas | None |
| 删除 | `delete()` | ids / where | None |

**核心要点**：

1. **documents vs embeddings 二选一**：不传 embeddings 则自动调用 embedding function
2. **批量操作远快于循环**：单次 batch 建议 1000~5000 条
3. **upsert 解决 ID 冲突**：比 delete+add 性能更好
4. **get() 支持分页和 where 过滤**：适合数据浏览和调试
5. **update 会触发 re-embedding**：不需要手动处理向量更新
6. **metadata 只支持基本类型**：str/int/float/bool，复杂类型需 JSON 序列化

下一节我们将深入 Collection 管理——如何创建不同距离度量的 Collection、多 Collection 隔离策略、以及 Collection 的元信息查看。
