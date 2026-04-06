---
title: 多源数据融合：同时从数据库 + 文件 + API 加载数据
description: 异构数据源的统一加载策略、数据去重与冲突解决、元数据标准化、混合检索的最佳实践
---
# 多源数据融合：同时从数据库 + 文件 + API 加载数据

前面的章节我们分别学习了如何从文件、数据库、API、云服务等单一数据源加载数据。但在真实的企业环境中，知识几乎总是**分散在多个地方**的——产品手册是 PDF 存在 S3 上、FAQ 是 Markdown 文件在 Git 仓库里、最新政策写在 Notion Database 中、客户工单在 Jira 里、技术方案讨论在 GitHub Issues 上。

如果你只能搜索其中某一个来源，那得到的信息注定是不完整的。用户问"退款流程是什么"，答案可能一半在公司政策 PDF 里，另一半在 Notion 的操作指南中。**多源数据融合（Multi-Source Data Fusion）的目标就是把散落在各处的知识汇聚到一个统一的 RAG 索引中，让用户的一次查询就能触及所有相关知识。**

这一节我们会学习如何在实际项目中实现多源数据融合，包括架构设计、数据去重、冲突解决、元数据标准化等关键问题。

## 为什么需要多源融合？

先看一个具体的场景来感受单源搜索的局限性：

> 一家电商公司的客服团队需要 RAG 系统来辅助回答客户问题。他们的知识分布在：
> - **Confluence**（内部 Wiki）：产品功能介绍、技术架构说明
> - **Notion**：最新的运营政策和促销规则（更新频率高）
> - **PostgreSQL 数据库**：商品信息（名称、价格、规格、库存）
> - **Jira**：已知问题和解决方案的历史记录
> - **SharePoint**：合同模板和法务文档
> - **Slack**：临时的决策记录和非正式经验分享

如果客服问"某款商品的保修期是多长"，答案可能在 Confluence 的产品文档里，也可能在 Notion 的最新政策更新中，还可能在数据库的商品信息表里。只有同时搜索所有来源，才能给出完整且准确的回答。

## 基础的多源加载模式

最直接的方式就是分别从各个数据源加载 Document，然后合并到一个列表中：

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.readers.database import DatabaseReader
from llama_index.readers.notion import NotionPageReader
from llama_index.readers.jira import JiraReader
from llama_index.readers.slack import SlackReader

all_documents = []

# ===== 来源 1：本地文件（产品文档）=====
file_reader = SimpleDirectoryReader(
    input_dir="./docs/product_manuals",
    required_exts=[".pdf", ".md"],
    file_metadata=lambda fname: {"source_type": "local_file"},
)
all_documents.extend(file_reader.load_data())

# ===== 来源 2：PostgreSQL（商品信息）=====
db_reader = DatabaseReader(
    sql_database="postgresql://user:pass@localhost:5432/ecommerce",
    sql_query="""
        SELECT id, name, category, description, warranty_months,
               price, specifications
        FROM products
        WHERE is_active = true
    """,
)
db_docs = db_reader.load_data()
for doc in db_docs:
    doc.metadata["source_type"] = "database_postgres"
all_documents.extend(db_docs)

# ===== 来源 3：Notion（运营政策）=====
notion_reader = NotionPageReader(
    integration_token=os.getenv("NOTION_TOKEN"),
)
notion_docs = notion_reader.load_data(database_id="policy-db-uuid")
for doc in notion_docs:
    doc.metadata["source_type"] = "notion_policy"
all_documents.extend(notion_docs)

# ===== 来源 4：Jira（问题记录）=====
jira_reader = JiraReader(
    server_url="https://support.atlassian.net",
    email="bot@company.com",
    api_token=os.getenv("JIRA_TOKEN"),
)
jira_docs = jira_reader.load_data(
    jql_query='project = SUPPORT AND resolution = Fixed',
    max_results=200,
)
for doc in jira_docs:
    doc.metadata["source_type"] = "jira_issues"
all_documents.extend(jira_docs)

# ===== 来源 5：Slack（团队讨论）=====
slack_reader = SlackReader(
    slack_token=os.getenv("SLACK_TOKEN"),
    channel_ids=["C01SUPPORT"],
    earliest_timestamp=str(int(time.time()) - 86400 * 90),  # 最近90天
)
slack_docs = slack_reader.load_data()
for doc in slack_docs:
    doc.metadata["source_type"] = "slack_messages"
all_documents.extend(slack_docs)

# ===== 统一建索引 =====
print(f"总计从 {5} 个数据源加载了 {len(all_documents)} 个文档")
index = VectorStoreIndex.from_documents(all_documents)
query_engine = index.as_query_engine(similarity_top_k=5)

response = query_engine.query("智能音箱 S1 的保修期是多久？")
print(response.response)
print("\n--- 来源分析 ---")
for node in response.source_nodes:
    print(f"  [{node.metadata.get('source_type')}] "
          f"分数: {node.score:.3f}")
```

这段代码展示了多源融合的基本模式。注意几个关键点：

**第一，`source_type` 元数据标记。** 我们在每个来源的 Document上都打上了 `source_type` 标签。这看似简单，却极其重要——它让后续的检索结果分析成为可能。当你发现某些查询总是返回低质量结果时，可以通过 `source_type` 定位是哪个数据源出了问题。

**第二，顺序无关性。** `VectorStoreIndex.from_documents()` 不关心 Document 来自哪里、以什么顺序传入。它会统一地对所有 Document 做分块、嵌入和索引。这意味着你可以随时添加新的数据源，而不需要修改现有的索引逻辑。

**第三，不同来源的 Document 可能质量差异很大。** Slack 消息可能是碎片化的对话片段（"那个保修的事我看了一下好像是两年"），而 PDF 文档则是正式的、结构化的内容。它们被平等地送入同一个向量空间，这在某些情况下可能导致质量问题——后面会讲如何应对。

## 数据去重：同一知识出现多次怎么办？

多源融合的一个典型问题是**数据重复**。同一份文档可能同时在本地文件系统和 Notion 中存在，同一个问题的解答可能既在 Jira 的 Issue 中又在 Confluence 的Wiki 页面里。如果不做去重，检索结果中会出现大量重复内容，浪费 LLM 的上下文窗口并降低答案质量。

### 基于内容的去重

最直观的去重方式是基于文本内容的相似度判断：

```python
from hashlib import md5
from difflib import SequenceMatcher

def content_hash(text: str) -> str:
    """生成文本的哈希值（用于精确去重）"""
    return md5(text.encode()).hexdigest()

def is_similar(text1: str, text2: str, threshold: float = 0.85) -> bool:
    """判断两段文本是否相似（用于模糊去重）"""
    return SequenceMatcher(None, text1, text2).ratio() >= threshold


def deduplicate_documents(
    documents: list,
    mode: str = "exact",  # "exact" 或 "fuzzy"
    threshold: float = 0.85,
) -> list:
    """对文档列表进行去重"""
    seen = set()
    unique = []

    for doc in documents:
        if mode == "exact":
            h = content_hash(doc.text)
            if h not in seen:
                seen.add(h)
                unique.append(doc)
        elif mode == "fuzzy":
            is_dup = False
            for existing in unique:
                if is_similar(doc.text, existing.text, threshold):
                    is_dup = True
                    break
            if not is_dup:
                unique.append(doc)

    return unique


# 使用
original_count = len(all_documents)
unique_docs = deduplicate_documents(all_documents, mode="fuzzy")
removed = original_count - len(unique_docs)
print(f"去重: {original_count} → {len(unique_docs)} (移除 {removed} 个重复)")
```

**精确去重（mode="exact"）** 适用于完全相同的文档在不同来源中出现的情况（比如同一份 PDF 既在本地又在 S3 上）。它速度快（O(n) 复杂度）且不会误删。

**模糊去重（mode="fuzzy"）** 适用于内容相似但表述不完全一致的情况（比如同一份文档在 Confluence 和 Notion 中可能有轻微的编辑差异）。代价是速度慢（O(n²) 复杂度，因为要两两比较）且可能误删真正不同的内容。阈值的选择很重要——太高则漏掉重复，太低则误删不同内容。

### 基于 ID 的去重

如果你的数据源都有可靠的全局唯一标识符（如文档 ID、URL、主键等），这是最高效的去重方式：

```python
def deduplicate_by_id(documents: list, id_field: str = "id") -> list:
    seen_ids = set()
    unique = []

    for doc in documents:
        doc_id = doc.metadata.get(id_field)
        if doc_id and doc_id not in seen_ids:
            seen_ids.add(doc_id)
            unique.append(doc)
        elif not doc_id:
            unique.append(doc)  # 没有 ID 的保留

    return unique
```

这种方式的前提是你的各个数据源都能提供可靠的 ID。如果做不到，就退回到基于内容的去重。

## 冲突解决：同一事实有不同说法怎么办？

比重复更棘手的问题是**冲突**——关于同一个事实，不同数据源给出了不同的说法。比如：
- 产品手册 PDF 写着"保修期 24 个月"
- Notion 的最新政策写着"保修期调整为 12 个月（2025年1月1日起生效）"
- Slack 讨论中有人说"好像改成了18个月"

这时候该怎么办？

### 策略一：信任层级（Trust Hierarchy）

为每个数据源分配一个可信度等级，冲突时优先采信高等级来源：

```python
SOURCE_TRUST_LEVELS = {
    "notion_policy": 5,       # 最高：官方政策文档
    "local_file_pdf": 4,      # 高：正式发布的文档
    "database_postgres": 4,   # 高：结构化主数据
    "confluence": 3,          # 中：团队 Wiki
    "jira_issues": 2,         # 低：历史记录
    "slack_messages": 1,      # 最低：非正式讨论
}


def resolve_conflict(source_nodes):
    """根据信任层级解决冲突"""
    if len(source_nodes) <= 1:
        return source_nodes

    best = max(
        source_nodes,
        key=lambda n: SOURCE_TRUST_LEVELS.get(
            n.metadata.get("source_type", "unknown"), 0
        ),
    )
    return [best]
```

### 策略二：时间优先（Recency Wins）

对于时效性强的信息（如价格、政策、状态），采用"最新的为准"原则：

```python
from datetime import datetime

def resolve_by_recency(source_nodes):
    """根据时间戳解决冲突，取最新的"""
    def get_time(node):
        ts_str = node.metadata.get("updated_at")
                     or node.metadata.get("created_at")
                     or node.metadata.get("timestamp")
        if ts_str:
            try:
                return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            except ValueError:
                pass
        return datetime.min

    latest = max(source_nodes, key=get_time)
    return [latest]
```

### 策略三：交给 LLM 判断

最灵活但也最昂贵的方式是把冲突信息都提供给 LLM，让它来判断：

```python
conflicting_info = "\n".join([
    f"[{n.metadata.get('source_type')} - {n.metadata.get('updated_at', '未知时间')}]\n{n.text}"
    for n in source_nodes
])

prompt = f"""以下是关于同一主题的不同来源的信息，其中可能存在矛盾。
请综合判断最准确的答案，并说明你的理由。

{conflicting_info}
"""
```

实际项目中，**策略一和策略二的组合使用最为常见**——先用信任层级筛选，再用时间戳在同层级内排序。策略三只在关键决策场景中使用。

## 元数据标准化

不同数据源返回的 metadata 字段名和格式各不相同。Notion 返回的是 `"Title"`（首字母大写），Jira 返回的是 `"summary"`（小写），数据库返回的是 `"title"`（全小写）。如果不做标准化，后续的过滤和查询就会非常混乱。

```python
def standardize_metadata(doc, source_type: str):
    """将不同来源的元数据标准化为统一的 schema"""

    source = doc.metadata.get("source", "")
    created = (
        doc.metadata.get("created_at")
        or doc.metadata.get("creation_date")
        or doc.metadata.get("timestamp")
        or ""
    )
    updated = (
        doc.metadata.get("updated_at")
        or doc.metadata.get("last_modified")
        or doc.metadata.get("modification_date")
        or created
    )
    title = (
        doc.metadata.get("title")
        or doc.metadata.get("Title")
        or doc.metadata.get("summary")
        or doc.metadata.get("name")
        or doc.metadata.get("file_name")
        or "Untitled"
    )
    author = (
        doc.metadata.get("author")
        or doc.metadata.get("owner")
        or doc.metadata.get("assignee")
        or doc.metadata.get("created_by")
        or "Unknown"
    )

    doc.metadata.update({
        "_std_source_type": source_type,
        "_std_source_url": source,
        "_std_title": title,
        "_std_author": author,
        "_std_created_at": created,
        `_std_updated_at`: updated,
        "_std_original_metadata": doc.metadata.copy(),  # 保留原始 metadata
    })
    return doc


# 批量标准化
standardized_docs = [
    standardize_metadata(doc, doc.metadata.get("source_type", "unknown"))
    for doc in all_documents
]
```

以 `_std_` 前缀命名的标准化字段保证了无论原始数据来自哪里，你都可以用统一的字段名来做过滤和分析。同时 `_std_original_metadata` 保留了完整的原始 metadata，以防需要回溯。

## 按来源加权检索

不同来源的数据质量不同，也许你希望来自官方文档的结果排名更高，而 Slack 消息的排名适当降低。这可以通过在 Node 层面添加权重来实现：

```python
from llama_index.core import Document
from llama_index.core.schema import TextNode

SOURCE_WEIGHTS = {
    "notion_policy": 1.5,       # 官方政策：提升权重
    "local_file_pdf": 1.3,      # 正式文档：略微提升
    "database_postgres": 1.2,   # 主数据：略微提升
    "confluence": 1.0,          # Wiki：正常权重
    "jira_issues": 0.9,         # Issue：略微降低
    "slack_messages": 0.7,      # Slack：降低权重
}

def apply_source_weights(documents):
    """为每个 Document 添加来源权重"""
    weighted_nodes = []
    for doc in documents:
        source_type = doc.metadata.get("source_type", "unknown")
        weight = SOURCE_WEIGHTS.get(source_type, 1.0)

        node = TextNode(
            text=doc.text,
            metadata={
                **doc.metadata,
                "_source_weight": weight,
            },
        )
        weighted_nodes.append(node)

    return weighted_nodes


weighted_nodes = apply_source_weights(standardized_docs)
index = VectorStoreIndex(nodes=weighted_nodes)
```

权重可以在后处理器中使用——比如在检索结果返回前，将相关性分数乘以来源权重来调整排名。不过需要注意，**权重的调整应该在充分评估的基础上进行**，而不是凭感觉设定。第八章会讲的评估体系可以帮助你找到最优的权重配置。

## 架构最佳实践

对于一个生产级别的多源 RAG 系统，推荐的架构如下：

```
┌──────────────────────────────────────────────┐
│              数据采集调度器 (Scheduler)          │
│                                              │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐        │
│  │定时触发  │ │事件触发  │ │手动触发  │        │
│  └────┬────┘ └────┬────┘ └────┬────┘        │
│       └──────────┼──────────┘                 │
│                  ▼                            │
│  ┌───────────────────────────────────────┐    │
│  │        Source Connector Layer          │    │
│  │                                       │    │
│  │  FileReader │ DBReader │ APIReader   │    │
│  │  CloudReader│ CustomReader│ ...       │    │
│  └───────────────────┬───────────────────┘    │
│                      ▼                         │
│  ┌───────────────────────────────────────┐    │
│  │      Data Processing Pipeline         │    │
│  │                                       │    │
│  │  1. 元数据标准化                       │    │
│  │  2. 内容清洗                           │    │
│  │  3. 去重                               │    │
│  │  4. 冲突检测与标记                     │    │
│  │  5. 来源加权                           │    │
│  └───────────────────┬───────────────────┘    │
│                      ▼                         │
│  ┌───────────────────────────────────────┐    │
│  │     Unified Index (VectorStore)        │    │
│  └───────────────────┬───────────────────┘    │
│                      ▼                         │
│           Query Engine → Response              │
└──────────────────────────────────────────────┘
```

这个架构的核心思想是**关注点分离**：数据采集、数据处理、索引构建、查询服务各自独立，通过标准化的 Document 格式作为中间协议进行通信。这样的好处是：
- 新增数据源只需在 Connector Layer 添加一个新的 Reader
- 数据处理逻辑的修改不影响数据采集
- 索引可以独立重建而不需要重新采集数据
- 每一层都可以独立扩展和替换

## 常见误区

**误区一："把所有数据源的数据全部加载到一个索引就好"。** 这是最简单的做法但不一定是最好的。如果某些数据源的查询模式与其他数据源截然不同（比如结构化的商品数据 vs 非结构化的讨论帖），考虑为它们建立独立的索引并使用路由查询（第五章会讲 Router Query Engine），效果往往更好。

**误区二:"去重越彻底越好"。** 过度的去重（尤其是模糊去重阈值设得太低）可能会误删有价值的内容。"同一件事的两种不同表述"有时候恰好包含了互补的信息。**宁可保留少量重复，也不要丢失独特信息。**

**误区三:"所有数据源同等重要"。** 几乎不可能。官方文档的可信度天然高于闲聊消息，结构化数据的准确性天然高于自由文本。承认这种差异并通过权重或分层来体现它，会让你的 RAG 系统输出更可靠的结果。

**误区四:"多源融合是一次性的工作"。** 数据源会变化（新系统上线、旧系统退役）、数据分布会漂移（某些来源的质量随时间下降）、业务需求会演进（新的查询类型出现）。**多源融合是一个持续运营的过程，需要定期的审查和调整。**
