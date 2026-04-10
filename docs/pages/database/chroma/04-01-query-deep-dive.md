# 4.1 Query 方法深度剖析

> **Query 是 Chroma 最核心的方法——理解它的每一个参数，就是理解向量搜索的全部可能性**

---

## 这一节在讲什么？

在前面的章节中，我们已经多次使用 `collection.query()` 来做语义搜索，但每次都只用了最基本的参数。实际上，`query()` 是 Chroma 中最复杂也最强大的方法——它支持文本查询和向量查询两种输入模式、可以叠加 metadata 过滤条件、可以控制返回字段和数量、甚至可以同时执行多条查询。理解 `query()` 的每个参数和行为细节，是从"会用 Chroma"到"用好 Chroma"的关键一步。

这一节我们会逐个拆解 `query()` 的所有参数，讲清楚每个参数的作用、参数之间的组合关系、以及在不同场景下应该怎么选择。这些知识不仅是日常开发的基础，也是面试中"向量数据库怎么用"这类问题的标准答案。

---

## query() 的完整参数列表

先看全貌，再逐个深入：

```python
results = collection.query(
    # ===== 输入方式（二选一）=====
    query_texts=["查询文本1", "查询文本2"],     # 文本输入，自动调用 EF 生成向量
    query_embeddings=[[0.1, 0.2, ...], ...],   # 向量输入，跳过 EF 直接搜索

    # ===== 搜索控制 =====
    n_results=10,                               # 返回结果数量（默认 10）
    where={"category": "tech"},                 # metadata 过滤条件
    where_document={"$contains": "关键词"},      # 文档内容过滤条件

    # ===== 输出控制 =====
    include=["documents", "metadatas", "distances", "embeddings"]  # 返回哪些字段
)
```

### query_texts vs query_embeddings：两种输入模式

这是 `query()` 最基础的选择——你是用文本查询还是用向量查询？

**query_texts** 是最常用的模式。你传入一段自然语言文本，Chroma 自动调用 Embedding Function 将其编码为向量，然后在 HNSW 索引中搜索最近邻。这种方式简单直观，适合大多数交互式查询场景。

**query_embeddings** 则跳过了 embedding 步骤，直接用预计算的向量做搜索。这种方式适合以下场景：你已经有了缓存的查询向量（避免重复计算）、你用的是 Chroma 不支持的 embedding 模型（需要在外部计算向量再传入）、或者你在做向量级别的分析和调试。

```python
# 模式 1：文本查询（最常用）
results = collection.query(
    query_texts=["如何退款？"],
    n_results=5
)

# 模式 2：向量查询（跳过 EF）
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
query_vector = model.encode(["如何退款？"], normalize_embeddings=True).tolist()

results = collection.query(
    query_embeddings=query_vector,
    n_results=5
)

# 两种模式的结果应该完全一致（前提是用同一个 EF）
```

**重要约束**：`query_texts` 和 `query_embeddings` 是互斥的——你不能同时传两个。如果你同时传了，Chroma 会报错。如果你一个都没传，也会报错。

### n_results：返回多少条结果

`n_results` 控制每条查询返回的最大结果数。默认值是 10。这个值的选择取决于你的下游应用：如果你在做 RAG，通常 3~5 条就足够了（太多会超出 LLM 的上下文窗口或者引入噪声）；如果你在做推荐或者需要 re-ranking，可以设大一些（比如 20~50），先粗筛再精排。

```python
# RAG 场景：3~5 条足够
results = collection.query(query_texts=["退款政策"], n_results=3)

# Re-ranking 场景：先取 50 条，再用 cross-encoder 精排
candidates = collection.query(query_texts=["退款政策"], n_results=50)
top_k = rerank(candidates, top_k=5)  # 外部 re-ranker
```

**注意**：`n_results` 不能超过 Collection 中的文档总数。如果设的值比文档总数还大，Chroma 会返回所有文档。

### where：Metadata 过滤

`where` 参数让你在向量搜索之前先按 metadata 条件过滤候选集。我们在 2.3 节已经详细讲过 metadata 的设计，这里重点讲 `where` 在 `query()` 中的行为。

```python
# 单条件过滤
results = collection.query(
    query_texts=["退款政策"],
    where={"category": "after_sales"},
    n_results=5
)

# 多条件组合过滤
results = collection.query(
    query_texts=["退款政策"],
    where={
        "$and": [
            {"category": "after_sales"},
            {"version": {"$gte": 2}},
            {"language": "zh"}
        ]
    },
    n_results=5
)
```

`where` 过滤发生在向量搜索**之前**——Chroma 先从 SQLite 中筛选出满足条件的文档 ID，然后只在这些文档上做 HNSW 搜索。这意味着如果 where 条件过于严格导致候选集为空，query 会返回空结果，即使向量空间中存在语义相关的文档。

### where_document：文档内容过滤

`where_document` 是一个容易被忽略但很有用的参数——它允许你按文档原文的内容做过滤。目前只支持 `$contains` 操作符（子串匹配）：

```python
# 只在包含"退款"二字的文档中搜索
results = collection.query(
    query_texts=["退货流程"],
    where_document={"$contains": "退款"},
    n_results=5
)

# 排除包含"草稿"的文档
results = collection.query(
    query_texts=["产品介绍"],
    where_document={"$not_contains": "草稿"},
    n_results=5
)
```

`where_document` 的性能特征需要特别注意：由于 Chroma 没有对文档原文建全文索引，`$contains` 操作是对所有文档做线性扫描。在数据量大时（> 10K 条），这个操作可能很慢。如果你的过滤条件可以转化为 metadata 字段，优先用 `where` 而不是 `where_document`。

### include：控制返回字段

`include` 参数决定返回结果中包含哪些字段。可选值有：`"ids"`、`"documents"`、`"embeddings"`、`"metadatas"`、`"distances"`。默认返回 `["metadatas", "documents", "distances"]`。

```python
# 最轻量：只要 ID 和距离（用于后续精确获取）
results = collection.query(
    query_texts=["查询"],
    n_results=10,
    include=["ids", "distances"]
)

# RAG 场景：需要文档内容和来源信息
results = collection.query(
    query_texts=["查询"],
    n_results=5,
    include=["documents", "metadatas", "distances"]
)

# 调试场景：需要看向量
results = collection.query(
    query_texts=["查询"],
    n_results=3,
    include=["ids", "documents", "embeddings", "metadatas", "distances"]
)
```

---

## 批量查询：一次传入多条查询

`query()` 支持同时传入多条查询文本或向量，Chroma 会并行处理并返回对应数量的结果列表：

```python
# 批量查询
results = collection.query(
    query_texts=["退款政策", "安装指南", "价格信息"],
    n_results=3
)

# results['ids'] 是一个嵌套列表：
# [
#   ["id_for_refund_1", "id_for_refund_2", "id_for_refund_3"],     ← 第一条查询的结果
#   ["id_for_install_1", "id_for_install_2", "id_for_install_3"],  ← 第二条查询的结果
#   ["id_for_price_1", "id_for_price_2", "id_for_price_3"]         ← 第三条查询的结果
# ]

for q_idx, query_text in enumerate(["退款政策", "安装指南", "价格信息"]):
    print(f"\n查询: '{query_text}'")
    for r_idx in range(len(results['ids'][q_idx])):
        doc = results['documents'][q_idx][r_idx]
        dist = results['distances'][q_idx][r_idx]
        print(f"  [{dist:.4f}] {doc[:60]}...")
```

批量查询的性能优势在于：如果使用远程 EF（如 OpenAI），多条查询可以在一次 API 调用中完成，减少网络往返延迟。

---

## 混合查询：语义搜索 + 结构化过滤

`query()` 最强大的用法是把语义搜索（向量相似度）和结构化过滤（where 条件）结合起来。这种"混合查询"模式是 RAG 系统中最常见的查询方式：

```python
import chromadb
from chromadb.utils import embedding_functions
import time

client = chromadb.Client(settings=chromadb.Settings(is_persistent=True, persist_directory="./hybrid_demo"))
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)

col = client.get_or_create_collection(name="hybrid_kb", embedding_function=ef)

# 添加带丰富 metadata 的文档
documents = [
    "退款政策：购买后7天内可无条件退款",
    "安装指南：请先下载安装包，双击运行安装程序",
    "定价方案：标准版每月99元，专业版每月299元",
    "API文档：所有接口均使用RESTful风格，限流1000次/分钟",
    "退货流程：在订单页面点击申请退货，填写原因后等待审核",
    "系统要求：Windows 10及以上，8GB内存，SSD硬盘",
]
metadatas = [
    {"category": "after_sales", "product": "general", "version": 2},
    {"category": "technical", "product": "desktop_app", "version": 1},
    {"category": "pricing", "product": "general", "version": 3},
    {"category": "technical", "product": "api", "version": 2},
    {"category": "after_sales", "product": "general", "version": 2},
    {"category": "technical", "product": "desktop_app", "version": 1},
]

col.add(documents=documents, ids=[f"d{i}" for i in range(len(documents))], metadatas=metadatas)

# 查询 1：纯语义搜索（无过滤）
print("=== 纯语义搜索 ===")
r = col.query(query_texts=["如何退货"], n_results=3, include=["documents", "distances", "metadatas"])
for i in range(len(r['ids'][0])):
    print(f"  [{r['distances'][0][i]:.4f}] ({r['metadatas'][0][i]['category']}) {r['documents'][0][i][:40]}...")

# 查询 2：语义搜索 + 只看 after_sales 类
print("\n=== 语义搜索 + after_sales 过滤 ===")
r = col.query(
    query_texts=["如何退货"],
    where={"category": "after_sales"},
    n_results=3,
    include=["documents", "distances", "metadatas"]
)
for i in range(len(r['ids'][0])):
    print(f"  [{r['distances'][0][i]:.4f}] ({r['metadatas'][0][i]['category']}) {r['documents'][0][i][:40]}...")

# 查询 3：语义搜索 + 版本过滤 + 产品过滤
print("\n=== 语义搜索 + 多条件过滤 ===")
r = col.query(
    query_texts=["技术文档"],
    where={
        "$and": [
            {"category": "technical"},
            {"version": {"$gte": 2}}
        ]
    },
    n_results=3,
    include=["documents", "distances", "metadatas"]
)
for i in range(len(r['ids'][0])):
    print(f"  [{r['distances'][0][i]:.4f}] ({r['metadatas'][0][i]['product']}) {r['documents'][0][i][:40]}...")
```

输出：

```
=== 纯语义搜索 ===
  [0.3245] (after_sales) 退款政策：购买后7天内可无条件退款...
  [0.4123] (after_sales) 退货流程：在订单页面点击申请退货...
  [0.6789] (technical) API文档：所有接口均使用RESTful风格...

=== 语义搜索 + after_sales 过滤 ===
  [0.3245] (after_sales) 退款政策：购买后7天内可无条件退款...
  [0.4123] (after_sales) 退货流程：在订单页面点击申请退货...

=== 语义搜索 + 多条件过滤 ===
  [0.5123] (api) API文档：所有接口均使用RESTful风格...
```

可以看到，加了 where 过滤后，搜索范围被精确缩小到了符合条件的文档子集，避免了语义相关但类别不对的"噪声"结果。

---

## query_texts vs query_embeddings 的使用时机

| 场景 | 推荐方式 | 原因 |
|------|---------|------|
| 用户输入原始文本 | `query_texts` | 最简单，自动调用 EF |
| 已有缓存的查询向量 | `query_embeddings` | 避免重复计算 embedding |
| 使用 Chroma 不支持的 EF | `query_embeddings` | 在外部计算向量后传入 |
| 需要精确控制向量 | `query_embeddings` | 比如做向量插值、向量运算 |
| 批量查询 + 远程 EF | `query_texts` | Chroma 会自动批处理 API 调用 |
| 调试向量空间 | `query_embeddings` | 可以传入自定义向量观察行为 |

---

## 常见误区

### 误区 1：where 过滤会让查询变快

不完全正确。where 过滤本身需要扫描 SQLite 全表（无 B-tree 索引），在大数据量下可能反而增加延迟。它的主要价值是**提高精度**，而不是提速。只有在候选集被大幅缩小（比如从 100K 缩减到 1K）时，HNSW 搜索的加速才能抵消过滤的开销。

### 误区 2：n_results 设得越大越好

n_results 过大会带来两个问题：第一，返回数据量大，网络传输和内存开销增加；第二，在 RAG 场景中，过多的检索结果会引入噪声，降低 LLM 生成质量。3~5 条是 RAG 的最佳范围。

### 误区 3：where 和 where_document 可以互相替代

不能。`where` 过滤的是 metadata 字段（结构化数据），`where_document` 过滤的是文档原文（非结构化数据）。两者的性能特征完全不同：`where` 扫描 SQLite 的 metadata 列，`where_document` 扫描文档原文。能用 `where` 解决的过滤需求，不要用 `where_document`。

---

## 本章小结

`query()` 是 Chroma 最核心的方法，掌握它的每个参数是从"会用"到"用好"的关键。核心要点回顾：第一，`query_texts` 和 `query_embeddings` 是互斥的两种输入模式，前者自动调用 EF，后者跳过 EF 直接搜索；第二，`n_results` 控制 top-K 数量，RAG 场景建议 3~5 条；第三，`where` 在向量搜索之前做 metadata 过滤，主要价值是提高精度而非提速；第四，`where_document` 做文档原文的子串匹配，性能较差，优先用 `where` 替代；第五，`include` 控制返回字段，按需选择可以减少数据传输；第六，批量查询支持一次传入多条查询，减少网络往返。

下一节我们将深入 Where 过滤器的完整语法——所有操作符、组合方式、以及那些容易写错的表达式。
