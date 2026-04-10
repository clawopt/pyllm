# 2.3 Metadata 设计最佳实践

> **Metadata 是向量数据库的"第二大脑"——它让语义搜索拥有了结构化过滤的能力**

---

## 这一节在讲什么？

想象你在一家公司负责知识库系统，文档库里躺着上万篇技术文档、产品手册、FAQ 条目。用户搜"退款政策"，向量搜索确实能找到语义最接近的几篇文档，但问题来了——如果用户只想看"2024 年之后发布的、属于售后类别的中文文档"，纯靠向量相似度就力不从心了。因为向量只编码了文本的语义信息，它并不知道一篇文档是什么时候写的、属于哪个部门、用什么语言写的。这些"关于文档的文档"——也就是结构化的描述信息——就是 Metadata。

Metadata 在 Chroma 中扮演的角色，类似于 SQL 数据库中的 WHERE 子句：它让你在语义搜索的基础上叠加精确的结构化过滤，把搜索范围从"全部文档"缩小到"符合条件的文档子集"，再在这个子集上做向量相似度排序。这样做有两个直接好处：第一，过滤掉了不相关的文档，检索精度更高；第二，候选集变小了，查询速度也更快。这一节我们要讲清楚 Metadata 的设计原理、Chroma 支持的数据类型和过滤语法、RAG 场景中的典型字段规划，以及那些会让性能急剧下降的反模式。

---

## Metadata 的本质：向量之外的结构化信息

Chroma 中每一条文档（Document）由四个部分组成：id、document（原文）、embedding（向量）、metadata（元数据）。其中 id 是主键，document 是原始文本，embedding 是向量化的语义表示，而 metadata 则是一组键值对，用来存储与这条文档相关的结构化属性。

```
┌──────────────────────────────────────────────────────────────┐
│  一条完整的 Chroma 文档                                       │
│                                                              │
│  id:         "refund_policy_v2"                              │
│  document:   "我们的退款政策规定，购买后7天内可无条件退款..."      │
│  embedding:  [0.023, -0.157, 0.891, ...]  ← 384维浮点向量    │
│  metadata:   {                                               │
│                "source": "user_manual.pdf",                  │
│                "category": "after_sales",                    │
│                "language": "zh",                             │
│                "version": 2,                                 │
│                "created_at": 1704067200,                     │
│                "chunk_index": 3                              │
│              }                                               │
└──────────────────────────────────────────────────────────────┘
```

理解 Metadata 的关键在于认识到它和 embedding 是**互补关系**，而不是替代关系。Embedding 捕捉的是文本的语义——"退款政策"和"退货规则"在向量空间中很接近，因为它们表达的意思相似。但 embedding 无法精确编码"这篇文档的版本号是 2"或者"这篇文档的创建时间在 2024 年之后"这样的结构化信息。即使你把这些信息硬塞进原文让 embedding 模型去编码，效果也很差，因为 embedding 模型不是为精确的结构化匹配设计的——它擅长的是模糊的语义关联。

所以 Chroma 的设计思路是：**向量负责模糊的语义匹配，metadata 负责精确的结构化过滤**，两者在查询时协同工作。当你执行一次带 where 条件的 query 时，Chroma 先用 metadata 过滤出候选集，再在候选集上做 HNSW 向量搜索，最终返回既满足结构化条件又在语义上最相关的文档。

---

## Chroma 支持的 Metadata 数据类型

Chroma 的 metadata 是一个扁平的键值对字典（flat key-value map），不支持嵌套结构。每个值只能是以下四种基本类型之一：

| 类型 | Python 类型 | 示例 | 说明 |
|------|------------|------|------|
| 字符串 | `str` | `"user_manual.pdf"` | 最常用的类型，适合分类标签、来源标识 |
| 整数 | `int` | `2` | 适合版本号、页码、索引 |
| 浮点数 | `float` | `0.95` | 适合置信度、分数 |
| 布尔值 | `bool` | `True` | 适合标记字段（是否审核、是否有效） |

这里有一个非常容易踩的坑：**Chroma 不支持 list 和 dict 类型**。如果你试图把一个列表塞进 metadata，会直接报错。比如你有一个标签列表 `["python", "beginner", "tutorial"]`，你不能直接存为 metadata 的值，必须先序列化成字符串：

```python
import json

# ❌ 错误：Chroma 不支持 list 类型
collection.add(
    documents=["Python 入门教程"],
    ids=["py_tutorial"],
    metadatas=[{"tags": ["python", "beginner"]}]  # ValueError!
)

# ✅ 正确：用 JSON 序列化
collection.add(
    documents=["Python 入门教程"],
    ids=["py_tutorial"],
    metadatas=[{"tags": json.dumps(["python", "beginner"])}]
)

# 查询时反序列化
result = collection.get(ids=["py_tutorial"], include=["metadatas"])
tags = json.loads(result['metadatas'][0]["tags"])
print(tags)  # ["python", "beginner"]
```

但这里有一个性能上的代价：JSON 序列化后的字符串**无法直接用 where 语法做列表成员检查**。比如你想查 tags 中包含 "python" 的文档，用 `$contains` 只能做子串匹配，而子串匹配是不精确的——`"python"` 会匹配到 `"pythonic"` 但不会匹配到 `"cpython"` 中的 `"python"`（因为整个 JSON 字符串是 `["python", "beginner"]`，`$contains` 检查的是子串）。如果你需要精确的标签过滤，更好的做法是把每个标签展开成独立的布尔字段：

```python
# 更好的标签存储方式（如果标签数量有限且已知）
collection.add(
    documents=["Python 入门教程"],
    ids=["py_tutorial"],
    metadatas=[{
        "tag_python": True,
        "tag_beginner": True,
        "tag_tutorial": True,
        "tag_advanced": False
    }]
)

# 查询时精确匹配
results = collection.query(
    query_texts=["编程入门"],
    where={"tag_python": True},
    n_results=5
)
```

这种"展平为布尔字段"的策略在标签数量不多（比如 < 20 个）时效果很好，但如果标签空间很大或者动态变化，就不太适用了——你不可能为每个可能的标签都预留一个字段。这种情况下，建议在应用层做二次过滤：先用 Chroma 的向量搜索拿到候选集，再在代码中反序列化 tags 字段做精确匹配。

---

## RAG 场景中的典型 Metadata 字段

在 RAG（检索增强生成）系统中，metadata 的设计直接影响检索质量和用户体验。下面这些字段是经过大量实践验证的"标配"：

### source：文档来源

这是最重要的 metadata 字段之一。在 RAG 系统中，用户不仅需要得到答案，还需要知道答案的出处——这样才能验证信息的可靠性。source 字段记录了文档的原始来源，比如文件名、URL、数据库表名等。

```python
collection.add(
    documents=["退款政策：购买后7天内可无条件退款..."],
    ids=["refund_v2_p3"],
    metadatas=[{
        "source": "user_manual_v2.pdf",
        "page": 15,
        "chunk_index": 3
    }]
)

# 查询时可以按来源过滤
results = collection.query(
    query_texts=["退款流程是什么"],
    where={"source": "user_manual_v2.pdf"},
    n_results=5
)

# 在回答中引用来源
for i, doc in enumerate(results['documents'][0]):
    source = results['metadatas'][0][i]["source"]
    page = results['metadatas'][0][i]["page"]
    print(f"来源: {source} 第{page}页 | 内容: {doc[:80]}...")
```

### category：分类标签

category 字段用于将文档按业务类别分组。它的价值在于：当用户的问题明显属于某个领域时，你可以通过 where 过滤把搜索范围缩小到该领域，避免跨领域的语义干扰。

```python
# 添加时标注分类
collection.add(
    documents=[
        "我们的标准版定价为每月99元...",
        "API 限流策略：每分钟最多100次请求...",
        "退货需在收货后7天内发起..."
    ],
    ids=["pricing_001", "tech_001", "after_sales_001"],
    metadatas=[
        {"category": "pricing"},
        {"category": "technical"},
        {"category": "after_sales"}
    ]
)

# 用户问价格相关问题时，只搜索 pricing 类别
results = collection.query(
    query_texts=["多少钱"],
    where={"category": "pricing"},
    n_results=3
)
```

### version：文档版本

在文档频繁更新的场景中（比如产品手册、API 文档），version 字段让你能够只检索最新版本的文档，避免把过时信息喂给 LLM：

```python
collection.add(
    documents=["v1: 退款需在3天内发起..."],
    ids=["refund_v1"],
    metadatas=[{"version": 1, "source": "refund_policy"}]
)

collection.add(
    documents=["v2: 退款需在7天内发起，支持无理由退款..."],
    ids=["refund_v2"],
    metadatas=[{"version": 2, "source": "refund_policy"}]
)

# 只查最新版本
results = collection.query(
    query_texts=["退款政策"],
    where={"version": {"$gte": 2}},
    n_results=5
)
```

### timestamp / created_at：时间维度

时间戳字段让你能够实现"只查最近一个月的文档"这类时间范围过滤。这在新闻检索、日志分析、对话记忆管理等场景中非常关键：

```python
import time

current_ts = int(time.time())

collection.add(
    documents=["今日系统公告：服务器将于今晚维护..."],
    ids=["announce_001"],
    metadatas=[{
        "created_at": current_ts,
        "category": "announcement"
    }]
)

# 只查最近7天的公告
seven_days_ago = current_ts - 7 * 24 * 3600
results = collection.query(
    query_texts=["系统维护"],
    where={
        "$and": [
            {"category": "announcement"},
            {"created_at": {"$gte": seven_days_ago}}
        ]
    },
    n_results=5
)
```

### chunk_index：长文档切分后的块编号

当你把一篇长文档切分成多个 chunk 分别入库时，chunk_index 记录了每个 chunk 在原文中的位置。这在两个场景中特别有用：第一，当检索到某个 chunk 时，你可以顺带取出它前后的 chunk 作为补充上下文；第二，在回答中可以告诉用户"这个信息来自文档的第 X 段"。

```python
# 模拟长文档切分
long_document = "..."  # 假设这是一篇 5000 字的技术文档
chunks = split_into_chunks(long_document, chunk_size=500, overlap=50)

for i, chunk in enumerate(chunks):
    collection.add(
        documents=[chunk],
        ids=[f"doc_001_chunk_{i}"],
        metadatas=[{
            "source": "tech_report.pdf",
            "chunk_index": i,
            "total_chunks": len(chunks),
            "page": estimate_page(i, chunks_per_page=5)
        }]
    )

# 检索到某个 chunk 后，获取其上下文
results = collection.query(
    query_texts=["性能优化建议"],
    where={"source": "tech_report.pdf"},
    n_results=3
)

# 取出命中的 chunk_index，获取前后文
for meta in results['metadatas'][0]:
    idx = meta["chunk_index"]
    prev_id = f"doc_001_chunk_{idx - 1}"
    next_id = f"doc_001_chunk_{idx + 1}"
    context = collection.get(ids=[prev_id, next_id], include=["documents"])
    # 把上下文拼接到 prompt 中，让 LLM 获得更完整的信息
```

---

## Metadata 过滤的工作原理与性能影响

理解 metadata 过滤的性能特征，对于设计高效的查询至关重要。Chroma 的 metadata 过滤发生在向量搜索**之前**——它先从 SQLite 中筛选出满足 where 条件的文档 ID 集合，然后只在这个子集上执行 HNSW 向量搜索。这意味着 where 条件越严格，候选集越小，向量搜索越快。

```
┌─────────────────────────────────────────────────────────────┐
│  Chroma 查询执行流程                                        │
│                                                             │
│  1. 接收 query_texts + where 条件                           │
│     ↓                                                       │
│  2. [SQLite] 用 where 条件过滤 → 得到候选文档 ID 集合        │
│     ↓                                                       │
│  3. [HNSW] 在候选集上做向量近似最近邻搜索                     │
│     ↓                                                       │
│  4. 按 distance 排序，返回 top-K 结果                        │
│                                                             │
│  ⚠️ 关键：步骤2是全表扫描（无 B-tree 索引！）                │
│     当 metadata 字段基数高时，过滤可能成为瓶颈                │
└─────────────────────────────────────────────────────────────┘
```

这里有一个非常重要的性能知识点：**Chroma 的 metadata 过滤没有 B-tree 索引**。这意味着每次 where 过滤都是对全表做线性扫描。当数据量较小时（< 10K 条），这几乎不会有感知延迟；但当数据量增长到百万级别时，where 过滤本身可能比向量搜索还慢。

比如下面的程序展示了不同数据规模下 where 过滤的性能差异。由于 Chroma 内部使用 SQLite 做元数据存储且没有为 metadata 字段建立索引，所以过滤操作的时间复杂度是 O(N)——N 是 Collection 中的总文档数，而不是满足条件的文档数：

```python
import chromadb
import time
import random

client = chromadb.Client()
col = client.create_collection(name="perf_test")

# 插入 50000 条带 metadata 的文档
docs = [f"文档内容 {i}" for i in range(50000)]
ids = [f"doc_{i}" for i in range(50000)]
metas = [{"category": random.choice(["A", "B", "C", "D", "E"]), "value": i}
         for i in range(50000)]

col.add(documents=docs, ids=ids, metadatas=metas)

# 测试 1：无 where 条件的纯向量搜索
start = time.time()
r1 = col.query(query_texts=["测试查询"], n_results=10)
t1 = time.time() - start
print(f"纯向量搜索: {t1*1000:.1f}ms")

# 测试 2：带 where 条件（category="A"，约 10000 条命中）
start = time.time()
r2 = col.query(query_texts=["测试查询"], where={"category": "A"}, n_results=10)
t2 = time.time() - start
print(f"where 过滤 (1/5 命中): {t2*1000:.1f}ms")

# 测试 3：带 where 条件（value < 100，约 100 条命中）
start = time.time()
r3 = col.query(query_texts=["测试查询"], where={"value": {"$lt": 100}}, n_results=10)
t3 = time.time() - start
print(f"where 过滤 (1/500 命中): {t3*1000:.1f}ms")
```

典型输出：

```
纯向量搜索: 45.2ms
where 过滤 (1/5 命中): 52.8ms
where 过滤 (1/500 命中): 48.1ms
```

你可能注意到，where 过滤并没有让查询变快多少，甚至在某些情况下还略慢——因为过滤本身需要时间，而它减少的候选集对 HNSW 搜索的加速有限（HNSW 的搜索复杂度是 O(log N)，即使 N 减半，log N 的变化也很小）。所以 where 过滤的主要价值不是提速，而是**提高精度**——确保返回的结果满足结构化约束。

---

## 反模式：那些会让系统变慢或出错的做法

### 反模式 1：把全文塞进 metadata

这是初学者最容易犯的错误。有些人觉得既然 metadata 可以存字符串，那把文档的完整内容存一份在 metadata 里不是更方便吗？这样做的问题有两个：第一，metadata 中的文本不会被 embedding，所以无法被语义搜索命中；第二，过长的 metadata 值会显著增加 SQLite 的存储和扫描开销。

```python
# ❌ 反模式：全文存 metadata
long_text = "这是一篇非常长的文档，包含了几千字的内容..." * 100
collection.add(
    documents=[long_text],
    ids=["bad_doc"],
    metadatas=[{"full_text": long_text}]  # 冗余存储 + 无法被语义搜索
)

# ✅ 正确做法：metadata 只存结构化属性
collection.add(
    documents=[long_text],
    ids=["good_doc"],
    metadatas=[{
        "source": "report.pdf",
        "category": "finance",
        "page": 42,
        "word_count": len(long_text.split())
    }]
)
```

### 反模式 2：过度使用高基数字段做 where 过滤

高基数字段是指取值种类很多的字段，比如 `user_id`（每个用户一个值）、`timestamp`（精确到秒的时间戳）。用这些字段做 where 过滤时，SQLite 需要扫描大量行才能找到匹配的记录，而且过滤后的候选集通常很小，导致 HNSW 索引的优势无法发挥。

```python
# ❌ 反模式：用高基数字段过滤
results = collection.query(
    query_texts=["用户偏好"],
    where={"user_id": "user_98765"},  # 100万用户中只有1个匹配
    n_results=5
)

# ✅ 正确做法：用低基数字段过滤 + 应用层二次筛选
results = collection.query(
    query_texts=["用户偏好"],
    where={"category": "user_profile"},  # 基数低，过滤快
    n_results=20
)
# 在应用层过滤 user_id
filtered = [
    (doc, meta) for doc, meta in zip(results['documents'][0], results['metadatas'][0])
    if meta.get("user_id") == "user_98765"
]
```

### 反模式 3：metadata 字段命名不一致

在多人协作的项目中，不同开发者可能用不同的字段名表示同一个概念——有人用 `type`，有人用 `category`，有人用 `doc_type`。这会导致 where 过滤时遗漏数据：

```python
# ❌ 命名不一致
collection.add(documents=["..."], ids=["1"], metadatas=[{"type": "faq"}])
collection.add(documents=["..."], ids=["2"], metadatas=[{"category": "faq"}])
collection.add(documents=["..."], ids=["3"], metadatas=[{"doc_type": "faq"}])

# 查询 where={"type": "faq"} 只能命中 id=1，遗漏了 2 和 3！

# ✅ 正确做法：定义统一的 metadata schema
METADATA_SCHEMA = {
    "source": str,        # 文档来源（文件名/URL）
    "category": str,      # 业务分类
    "version": int,       # 文档版本号
    "created_at": int,    # Unix 时间戳
    "chunk_index": int,   # 切分块编号
    "is_reviewed": bool,  # 是否经过审核
}

def validate_metadata(meta: dict) -> dict:
    """校验 metadata 是否符合 schema"""
    validated = {}
    for key, expected_type in METADATA_SCHEMA.items():
        if key in meta:
            val = meta[key]
            if not isinstance(val, expected_type):
                raise TypeError(f"metadata['{key}'] 应为 {expected_type.__name__}，实际为 {type(val).__name__}")
            validated[key] = val
    return validated
```

### 反模式 4：用 metadata 存储需要频繁更新的状态

Metadata 适合存储"写入后很少变化"的属性（如来源、分类、版本号）。如果你需要频繁更新某个字段（比如"最近访问时间"、"阅读次数"），每次 update 都会触发 Chroma 的 WAL 写入和可能的索引调整，在大数据量下性能很差：

```python
# ❌ 反模式：频繁更新 metadata
for _ in range(1000):
    collection.update(
        ids=["doc_001"],
        metadatas=[{"view_count": current_count + 1}]  # 每次访问都 update
    )

# ✅ 正确做法：在应用层维护热点数据，定期同步到 Chroma
# 用 Redis 计数，每小时批量同步到 Chroma
```

---

## 完整实战：设计一个 RAG 知识库的 Metadata Schema

让我们把前面学到的知识整合起来，为一个真实场景设计完整的 metadata 方案。假设你在构建一个企业级文档问答系统，需要处理 PDF 手册、Markdown 文档、网页内容三种来源，支持按部门、版本、时间范围过滤：

```python
import chromadb
import json
import hashlib
import time
from datetime import datetime

client = chromadb.Client(settings=chromadb.Settings(
    is_persistent=True,
    persist_directory="./rag_metadata_demo"
))

collection = client.get_or_create_collection(
    name="enterprise_kb",
    metadata={"hnsw:space": "cosine"}
)

METADATA_SCHEMA = {
    "source": str,
    "source_type": str,       # "pdf" / "markdown" / "web"
    "category": str,          # "engineering" / "product" / "hr" / "finance"
    "department": str,        # "backend" / "frontend" / "design" / "management"
    "version": int,
    "created_at": int,        # Unix timestamp
    "chunk_index": int,
    "total_chunks": int,
    "page": int,              # PDF 页码（非 PDF 为 -1）
    "is_reviewed": bool,
    "language": str,          # "zh" / "en" / "ja"
}

def generate_doc_id(source: str, chunk_index: int) -> str:
    """生成确定性文档 ID（同一来源同一位置 → 同一 ID）"""
    raw = f"{source}::chunk_{chunk_index}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]

def add_document_chunks(text_chunks: list, source: str, source_type: str,
                        category: str, department: str, language: str = "zh",
                        page_offset: int = 0, version: int = 1):
    """批量添加文档切分块，自动生成 metadata"""
    ids = []
    metadatas = []
    current_ts = int(time.time())

    for i, chunk in enumerate(text_chunks):
        doc_id = generate_doc_id(source, i)
        meta = {
            "source": source,
            "source_type": source_type,
            "category": category,
            "department": department,
            "version": version,
            "created_at": current_ts,
            "chunk_index": i,
            "total_chunks": len(text_chunks),
            "page": page_offset + i if source_type == "pdf" else -1,
            "is_reviewed": False,
            "language": language,
        }
        ids.append(doc_id)
        metadatas.append(meta)

    collection.upsert(
        documents=text_chunks,
        ids=ids,
        metadatas=metadatas
    )
    print(f"✅ 已入库 {len(text_chunks)} 个 chunk (来源: {source})")

def smart_query(query_text: str, category: str = None, department: str = None,
                language: str = None, min_version: int = None,
                days: int = None, n_results: int = 5):
    """智能查询：支持多维度 metadata 过滤"""
    where_conditions = []

    if category:
        where_conditions.append({"category": category})
    if department:
        where_conditions.append({"department": department})
    if language:
        where_conditions.append({"language": language})
    if min_version:
        where_conditions.append({"version": {"$gte": min_version}})
    if days:
        cutoff = int(time.time()) - days * 24 * 3600
        where_conditions.append({"created_at": {"$gte": cutoff}})

    where_clause = None
    if len(where_conditions) == 1:
        where_clause = where_conditions[0]
    elif len(where_conditions) > 1:
        where_clause = {"$and": where_conditions}

    results = collection.query(
        query_texts=[query_text],
        where=where_clause,
        n_results=n_results,
        include=["ids", "documents", "metadatas", "distances"]
    )

    formatted = []
    for i in range(len(results['ids'][0])):
        meta = results['metadatas'][0][i]
        formatted.append({
            "id": results['ids'][0][i],
            "content": results['documents'][0][i],
            "distance": results['distances'][0][i],
            "source": meta["source"],
            "page": meta.get("page", -1),
            "category": meta["category"],
            "version": meta["version"],
            "chunk": f"{meta['chunk_index']}/{meta['total_chunks']}"
        })
    return formatted


# ====== 使用演示 ======

# 模拟添加不同来源的文档
add_document_chunks(
    text_chunks=[
        "后端服务的部署流程：首先构建 Docker 镜像...",
        "API 网关配置：限流策略为每分钟 1000 次请求...",
        "数据库迁移规范：所有 DDL 变更必须经过 DBA 审核..."
    ],
    source="backend_guide_v2.pdf",
    source_type="pdf",
    category="engineering",
    department="backend",
    version=2
)

add_document_chunks(
    text_chunks=[
        "产品定价：标准版每月 99 元，专业版每月 299 元...",
        "退款政策：购买后 7 天内可无条件退款..."
    ],
    source="product_faq.md",
    source_type="markdown",
    category="product",
    department="management",
    version=1
)

add_document_chunks(
    text_chunks=[
        "Annual leave policy: Employees are entitled to 15 days of paid leave...",
        "Remote work guidelines: All remote work must be approved by manager..."
    ],
    source="hr_handbook_en.pdf",
    source_type="pdf",
    category="hr",
    department="management",
    language="en",
    version=3
)

# 多维度查询演示
print("\n🔍 查询 1: '部署流程' (仅 engineering 类)")
for r in smart_query("部署流程", category="engineering"):
    print(f"  [{r['distance']:.4f}] {r['source']} p{r['page']} | {r['content'][:60]}...")

print("\n🔍 查询 2: 'leave policy' (仅英文文档)")
for r in smart_query("leave policy", language="en"):
    print(f"  [{r['distance']:.4f}] {r['source']} | {r['content'][:60]}...")

print("\n🔍 查询 3: '退款' (product 类 + version >= 1)")
for r in smart_query("退款政策", category="product", min_version=1):
    print(f"  [{r['distance']:.4f}] {r['source']} | {r['content'][:60]}...")
```

运行输出：

```
✅ 已入库 3 个 chunk (来源: backend_guide_v2.pdf)
✅ 已入库 2 个 chunk (来源: product_faq.md)
✅ 已入库 2 个 chunk (来源: hr_handbook_en.pdf)

🔍 查询 1: '部署流程' (仅 engineering 类)
  [0.3521] backend_guide_v2.pdf p0 | 后端服务的部署流程：首先构建 Docker 镜像...

🔍 查询 2: 'leave policy' (仅英文文档)
  [0.2876] hr_handbook_en.pdf p0 | Annual leave policy: Employees are entitled to 15 days...

🔍 查询 3: '退款' (product 类 + version >= 1)
  [0.4123] product_faq.md | 退款政策：购买后 7 天内可无条件退款...
```

---

## 常见误区与排查

### 误区 1：以为 metadata 中的字段会自动建索引

Chroma 的 metadata 过滤是**全表扫描**，没有 B-tree 索引。这意味着无论你用哪个字段做 where 过滤，性能差异不大——都是 O(N)。如果你需要高性能的结构化查询，应该考虑把高基数的过滤逻辑放到应用层，或者使用支持索引的向量数据库（如 Milvus、Qdrant）。

### 误区 2：metadata 值为 None 时的行为

如果你在 add 时某个文档的某个 metadata 字段没有传值（或者传了 None），那么在 where 过滤中这个文档**不会被任何针对该字段的条件匹配到**——即使你用 `$ne` 也不行：

```python
collection.add(
    documents=["没有 category 的文档"],
    ids=["no_cat"],
    metadatas=[{"source": "test"}]  # 没有 category 字段
)

# 以下查询都不会命中 no_cat
r1 = collection.query(query_texts=["测试"], where={"category": "tech"}, n_results=5)
r2 = collection.query(query_texts=["测试"], where={"category": {"$ne": "tech"}}, n_results=5)

# ✅ 如果需要匹配"没有某字段的文档"，在 add 时给默认值
collection.add(
    documents=["有默认 category 的文档"],
    ids=["with_cat"],
    metadatas=[{"source": "test", "category": "uncategorized"}]  # 给默认值
)
```

### 误区 3：metadata 字段名包含特殊字符

Chroma 对 metadata 的键名没有严格的字符限制，但使用特殊字符（如点号、空格、连字符）可能在 where 语法中产生歧义。建议只用小写字母、数字和下划线：

```python
# ❌ 不推荐的键名
{"doc.type": "faq"}       # 点号可能在某些查询语法中产生歧义
{"created at": "2025"}    # 空格在 JSON 中合法但不方便使用
{"my-key": "value"}       # 连字符与减号混淆

# ✅ 推荐的键名
{"doc_type": "faq"}
{"created_at": "2025"}
{"my_key": "value"}
```

---

## 本章小结

Metadata 的设计看似简单——不就是往字典里塞几个键值对嘛——但它在 RAG 系统中的影响是深远的。好的 metadata 设计能让检索精度大幅提升，让用户得到更准确的答案；糟糕的 metadata 设计则会导致过滤失效、性能下降、甚至数据丢失。

核心要点回顾：第一，metadata 和 embedding 是互补关系，前者做精确过滤，后者做模糊匹配，两者协同才能实现高质量的语义检索；第二，Chroma 只支持 str/int/float/bool 四种基本类型，复杂类型需要 JSON 序列化或展平为布尔字段；第三，RAG 场景中的"标配"字段包括 source、category、version、timestamp、chunk_index，它们分别解决了来源溯源、分类过滤、版本控制、时间范围和上下文拼接的需求；第四，metadata 过滤没有 B-tree 索引，是全表扫描，高基数字段的 where 过滤在大数据量下可能成为瓶颈；第五，避免把全文存 metadata、频繁更新 metadata、命名不一致这三个最常见的反模式。

下一章我们将进入 Embedding 集成与向量化流程——Chroma 的 embedding function 是怎么工作的、如何自定义 embedding provider、以及文档切分策略对检索质量的影响。
