# 1.2 核心概念全景图

上一节我们从"为什么需要"的角度理解了向量数据库的价值。这一节我们要建立 Chroma 的完整概念模型——在写第一行代码之前，先在脑子里搞清楚 Chroma 的数据模型、核心对象和它们之间的关系。这就像在学数据库之前先理解 ER 图一样，概念清晰了之后写代码只是"翻译"的工作。

## 五个核心概念

Chroma 的数据模型围绕五个核心抽象展开。让我们逐一理解：

### Document（文档）

Document 是 Chroma 中最基本的可搜索单位。它不是一段原始文本——虽然你可以传入文本，但 Chroma 内部会把它转换成向量存储。一个 Document 由三个部分组成：

```python
{
    "id": "doc_001",              # 唯一标识符
    "document": "产品退款政策...",   # 原始文本内容（可选保留）
    "embedding": [0.12, -0.34, ...], # 向量表示（核心！）
    "metadata": {                   # 结构化的元数据
        "source": "refund_policy_v2.pdf",
        "category": "pricing",
        "author": "finance-team",
        "chunk_index": 3,
    }
}
```

**id** 是 Document 的唯一标识符。Chroma 要求每个 document 有一个 id（字符串或整数），且在同一 Collection 内不能重复。如果你不提供 id，Chroma 会自动生成一个 UUID。选择什么作为 id 取决于你的使用场景：用自增整数最简单；用业务含义明确的字符串（如 `user_manual_2024_q1`）最利于调试。

**document** 是原始文本内容。你可能会有疑问：既然已经有 embedding 了，为什么还要存原文？原因有二：第一，检索结果展示给用户时需要显示原文片段而不是一堆数字向量；第二，某些高级操作（如上下文组装）需要把原始文本拼接到 LLM 的 prompt 中。

**embedding** 是 Document 的灵魂——一个固定长度的浮点数数组。它的维度取决于你使用的 Embedding 模型：sentence-transformers 默认输出 384 维，OpenAI 的 text-embedding-3-large 输出 3072 维，OpenAI 的 text-embedding-3-small 输出 1536 维。所有存入同一个 Collection 的 Document 必须具有**相同的维度**，否则无法计算相似度。

**metadata** 是结构化的键值对，用于过滤和分类。它是标量信息（不是向量），支持 str、int、float、bool 类型。Metadata 是 Chroma 区别于简单 key-value 存储的关键特性——它可以在查询前用来缩小搜索范围（先按 metadata 过滤，再做向量相似度排序），这对性能优化至关重要。

### Collection（集合）

Collection 是逻辑上的一组相关 Document 的容器，类似于关系数据库中的 Table 或 MongoDB 中的 Collection。你在创建 Collection 时必须指定一个距离度量方式（distance metric），这个决定之后不可更改。

```python
collection = client.create_collection(
    name="product_docs",       # 集合名称
    metadata={"hnsw:space": "cosine"},  # 距离度量
)
```

Collection 有几个重要属性：
- **name**: 同一 Client 下 name 必须唯一
- **distance metric**: 创建后不可变（cosine / l2 / ip）
- **count**: 当前包含的 Document 数量
- **dimension**: 向量的维度（由第一个添加的 Document 决定）

为什么 distance metric 不可变？因为 Chroma 在创建 Collection 时会根据 metric 选择来构建内部索引结构（通常是 HNSW 图索引）。HNSW（Hierarchical Navigable Small World）是一种近似近邻搜索算法，它的索引结构与距离度量紧密耦合——为 cosine 优化的索引无法用于 l2 距离计算。所以你必须在一开始就选对。

### Query（查询）与 Result（结果）

Query 是用户发起的一次搜索请求。它可以有两种形式：
- **文本查询**：`query_texts=["什么是 Python?"]` —— Chroma 会先用配置好的 Embedding Function 把文本转成向量
- **向量查询**：`query_embeddings=[[0.1, -0.2, ...]]` —— 直接传入向量，跳过 embedding 步骤

Result 是 Query 的返回值——一组与查询最相似的 Document，按相似度降序排列。每个结果包含：
- **id**: 匹配到的 Document id
- **distance**: 与查询向量的距离值（越小越相似）
- **document**: 原始文本（如果存储了的话）
- **metadata**: 元数据字典
- **embedding**: 向量（如果请求返回的话）

### Distance Metric（距离度量）

这是初学者最容易混淆也最重要的概念之一。Chroma 支持三种距离度量：

**L2 Distance（欧几里得距离）**:
$$d_{L2}(a, b) = \sqrt{\sum_{i=1}^{d}(a_i - b_i)^2}$$

直觉：多维空间中两点之间的直线距离。简单直观，但在高维空间中会受"维度灾难"影响——随着维度增加，所有点之间的 L2 距离趋于一致，区分度下降。

**Cosine Similarity（余弦相似度）**:
$$\text{cos}(a, b) = \frac{a \cdot b}{||a|| \cdot ||b||}$$

直觉：两个向量指向方向的相似程度，不受长度影响。值域 [-1, 1]，1 表示完全相同方向，-1 表示完全相反方向。Chroma 内部实际用的是 cosine **distance** = $1 - \text{cos}(a,b)$，所以结果范围是 [0, 2]，0 表示完全相同。

**IP（Inner Product / 点积）**:
$$\text{ip}(a, b) = a \cdot b = \sum_{i=1}^{d} a_i \times b_i$$

直觉：L2 去掉开方后就是 IP。当且仅当输入向量已经归一化（模长为 1）时，IP 等价于 Cosine Similarity（此时 $a \cdot b = ||a||||b|| \cdot \text{cos}$）。速度最快但前提条件严格。

### Metadata Filtering（元数据过滤）

这是 Chroma 实现高性能的关键机制。想象你有 100 万份文档，用户问的是"定价问题"。如果你不做任何过滤就直接做向量搜索，需要计算 100 万次余弦相似度——即使 Chroma 很快，这也需要好几秒。但如果你的 metadata 里已经标记了每份文档的 category（pricing/technical/hr/legal），你可以先用一条 `where={"category": "pricing"}` 把候选集缩小到 maybe 5 万份，再做向量搜索——速度提升 20 倍，而精度几乎不受影响。

```
查询: "定价问题"
      │
      ▼ where category="pricing" (100万 → 5万) ← Metadata 过滤
      │
      ▼ 向量 Top-50 搜索 (5万 → 50) ← 相似度排序
      │
      ▼ 返回 Top-10 结果               ← 最终答案
```

## Chroma 的架构特点

最后补充一些 Chroma 作为产品的架构特点，这些会在后续章节中反复用到：

**嵌入式 SQLite + DuckDB 引擎**: Chroma 底层用 SQLite 做元数据和 ID 的持久化存储（轻量级、无需额外安装），同时集成了 DuckDB 作为其 SQL 查询引擎来执行 metadata 过滤和聚合操作。这种组合让它既有 NoSQL 的灵活性又有 SQL 的查询能力。

**HNSW 图索引**: 这是 Chroma 实现高效向量搜索的核心算法。HNSW（Hierarchical Navigable Small World）构建了一个多层级的 proximity graph，搜索时从顶层节点开始逐层向下探索，平均时间复杂度为 O(log N) 而暴力搜索的 O(N)。这也是为什么 Chroma 能在毫秒级处理百万级向量的原因。

**WAL 日志保证持久化**: Write-Ahead Logging 是一种预写式日志机制。每次写入操作先记录到 WAL 文件中（追加写入，很快），然后再异步地应用到主数据库文件上。这意味着即使程序在写入过程中崩溃，已提交的操作也不会丢失——重启后 Chroma 会重放 WAL 来恢复状态。

现在概念框架已经建立清楚了。下一节我们就要动手安装 Chroma 并运行第一个完整的示例程序。
