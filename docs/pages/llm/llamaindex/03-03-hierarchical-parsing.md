---
title: 层级化文档解析：标题感知与结构保留
description: HTMLHierarchicalSplitter / MarkdownNodeParser、文档树构建、层级索引与检索
---
# 层级化文档解析：标题感知与结构保留

前两节我们讨论了文档解析的核心挑战和基础的 Node Parser 工具。那些工具大多采用"扁平化"的切分策略——不管原文档有什么样的层级结构，最终产出都是一堆平铺的文本块。但对于很多类型的文档来说，**层级结构本身就是最重要的信息之一**。

想想看：一本技术手册的目录本身就是一张知识地图——"第三章 2.4 节讲的是 API 认证"，这个信息比章节的具体内容更能帮助你定位所需的知识。一篇学术论文的"摘要→引言→方法→实验→结论"结构告诉你应该按什么顺序阅读。一份法律合同的"第一条 定义 → 第二条 权利义务 → 第三条 违约责任"结构让你快速找到关心的条款。

如果我们在解析过程中丢弃了这些结构信息，就相当于把一本书的目录撕掉了——你还能读到每一页的内容，但你失去了在这本书中导航的能力。

## 为什么层级结构如此重要？

让我们用一个具体的例子来说明层级结构的价值。假设你有以下 Markdown 文档：

```markdown
# 企业级 RAG 系统架构指南

## 1. 系统概述

本指南介绍如何设计和实现一个企业级的 RAG（检索增强生成）系统。
涵盖数据接入、索引策略、检索优化、评估体系等核心模块。

### 1.1 设计原则
- 数据驱动决策
- 渐进式优化
- 可观测性优先

### 1.2 技术选型
推荐使用 LlamaIndex 作为 RAG 框架...
推荐使用 PostgreSQL + pgvector 作为向量数据库...

## 2. 数据接入层

### 2.1 文件连接器
PDF、Word、Markdown、Excel 等格式的统一接入...

### 2.2 数据库连接器
PostgreSQL、MySQL、MongoDB 等数据库的数据读取...

### 2.3 API 连接器
Notion、GitHub、Slack 等 SaaS 平台的集成...

## 3. 索引策略

### 3.1 VectorStoreIndex
基于向量相似度的索引，适用于大多数语义搜索场景...

### 3.2 KeywordTableIndex
基于关键词匹配的索引，适用于精确查找...

## 4. 高级检索

### 4.1 混合检索
结合向量搜索和关键词搜索的优势...

### 4.2 重排序（Reranking）
使用交叉编码器（Cross-Encoder）对初步检索结果重新排序...
```

现在用户问了这样一个问题：**"你们的系统怎么处理 Notion 数据？"**

### 扁平切分的结果

如果用普通的 `SentenceSplitter(chunk_size=256)` 来处理，Notion 相关的内容可能被切成这样的 chunks：

```
Chunk A:
...API 连接器
Notion、GitHub、Slack 等 SaaS 平台的集成...

Chunk B:
SaaS 平台的集成方式详见第四章...
```

当用户搜索"Notion 数据"时，系统可能返回 Chunk A 和 Chunk B。但这两个 chunk 都没有告诉 LLM：**这段内容是在讲"数据接入层"中的"API 连接器"**。LLM 只看到了一些零散的文字片段，缺乏结构化的上下文。

### 层级保留的结果

如果我们保留了文档的层级结构，每个 chunk 会携带自己的"坐标"：

```
Chunk A (path: "2. 数据接入层 > 2.3 API 连接器"):
Notion、GitHub、Slack 等 SaaS 平台的集成...

Chunk B (path: "1. 系统概述 > 1.2 技术选型"):
推荐使用 LlamaIndex 作为 RAG 框架...
```

现在当用户问 Notion 相关问题时：
1. 系统能精确地定位到 "2.3 API 连接器" 这一节
2. LLM 知道这个答案来自"数据接入层"的上下文
3. 如果用户进一步追问"那数据库呢？"，系统能知道去"2.2 数据库连接器"找答案
4. 如果用户问"整体架构是什么？"，系统可以返回"1. 系统概述"级别的更高层次摘要

这就是**层级结构带来的导航能力**——它让 RAG 系统不仅能"找到相关信息"，还能"理解信息在整个知识体系中的位置"。

## HTMLHierarchicalSplitter：HTML 的层级解析

对于 HTML 格式的文档（网页、在线文档等），LlamaIndex 提供了 `HTMLHierarchicalSplitter`，它能利用 HTML 标签的天然层级结构来进行智能切分：

```bash
pip install llama-index-readers-web
```

```python
from llama_index.readers.web import HTMLHierarchicalSplitter

splitter = HTMLHierarchicalSplitter(
    header_tags=["h1", "h2", "h3"],  # 作为层级依据的标签
    include_metadata=True,             # 是否在 metadata 中包含路径信息
)

documents = SimpleDirectoryReader("./docs/html_pages").load_data()
nodes = splitter.get_nodes_from_documents(documents)

for node in nodes[:5]:
    print(f"[Level {node.metadata.get('header_level', '?')}] "
          f"{node.metadata.get('header', 'N/A')}")
    print(f"  内容: {node.text[:100]}...")
    print()
```

`HTMLHierarchicalSplitter` 的工作原理是利用 HTML 的标题标签（`<h1>` 到 `<h6>`）来构建文档树：

```html
<!-- 输入 HTML -->
<html>
<body>
  <h1>RAG 系统架构指南</h1>
  <p>本指南介绍...</p>

  <h2>1. 系统概述</h2>
  <p>涵盖数据接入、索引策略...</p>

  <h3>1.1 设计原则</h3>
  <ul><li>数据驱动决策</li></ul>

  <h3>1.2 技术选型</h3>
  <p>推荐使用 LlamaIndex...</p>

  <h2>2. 数据接入层</h2>
  ...
</body>
</html>
```

```
输出节点树:

Root
├── [H1] RAG 系统架构指南
│   └── "本指南介绍..."
├── [H2] 1. 系统概述
│   ├── "涵盖数据接入..."
│   ├── [H3] 1.1 设计原则
│   │   └── "- 数据驱动决策"
│   └── [H3] 1.2 技术选型
│       └── "推荐使用 LlamaIndex..."
├── [H2] 2. 数据接入层
│   └── ...
```

每个输出的 Node 都带有丰富的 metadata：

```python
node.metadata
# {
#   'header_level': 2,                    # 当前标题级别
#   'header': '1. 系统概述',               # 当前标题文本
#   'header_path': ['RAG 系统架构指南', '1. 系统概述'],  # 从根到当前节点的路径
#   'tag': 'h2',                          # 对应的 HTML 标签
# }
```

### 实际应用示例

```python
from llama_index.readers.web import HTMLHierarchicalSplitter
from llama_index.core import VectorStoreIndex

splitter = HTMLHierarchicalSplitter(
    header_tags=["h1", "h2", "h3"],
)

reader = SimpleDirectoryReader("./docs", required_exts=[".html"])
documents = reader.load_data()

nodes = splitter.get_nodes_from_documents(documents)

index = VectorStoreIndex(nodes)
query_engine = index.as_query_engine(similarity_top_k=3)

response = query_engine.query("Notion 数据怎么接入？")
print(response.response)

# 检查来源的层级信息
for node in response.source_nodes:
    path = " > ".join(node.metadata.get("header_path", []))
    print(f"\n[{path}]")
    print(f"  {node.text[:120]}...")
```

输出类似：

```
根据文档的 2.3 API 连接器 章节，Notion 数据的接入方式如下：
首先需要获取 Notion Integration Token，然后使用 NotionPageReader
加载数据...

[企业级 RAG 系统架构指南 > 2. 数据接入层 > 2.3 API 连接器]
  Notion、GitHub、Slack 等 SaaS 平台的集成方式包括 OAuth 认证、
  API Key 认证等方式...
```

注意 LLM 的回复中自动引用了层级路径信息（"2.3 API 连接器"），这是因为层级信息被注入到了发送给 LLM 的上下文中。

## MarkdownNodeParser：Markdown 的层级解析

Markdown 文档虽然没有 HTML 那样显式的标签嵌套，但通过 `#`、`##`、`###` 等标记同样表达了清晰的层级结构。`MarkdownNodeParser` 专门用于解析这种结构：

```python
from llama_index.core.node_parser import MarkdownNodeParser

parser = MarkdownNodeParser(
    max_chunks_per_page=1000,  # 安全限制
)

documents = SimpleDirectoryReader("./docs", required_exts=[".md"]).load_data()
nodes = parser.get_nodes_from_documents(documents)

for node in nodes[:5]:
    if "header" in node.metadata:
        print(f"[{'#' * node.metadata.get('header_level', 0)} "
              f"{node.metadata['header']}]")
        print(f"  {node.text[:100]}...")
```

MarkdownNodeParser 的输出结构与 HTMLHierarchicalSplitter 类似——每个 Node 都携带 `header_level`、`header`、`header_path` 等 metadata。

### Markdown 特有的处理

Markdown 有一些 HTML 没有的结构元素，MarkdownNodeParser 会特别处理：

**代码块保留：** Markdown 中的代码块（``` ``` 围起来的内容）会被当作整体保留，不会被在中间切断：

```markdown
## 安装依赖

```bash
pip install llama-index
pip install llama-index-readers-file
```

## 配置环境
```

上面的代码块会被完整地保留在一个 Node 中，即使在 `chunk_size` 较小时也不会被拆散。

**表格识别：** Markdown 表格会被检测并尝试保留结构：

```markdown
| 索引类型 | 适用场景 | 检索方式 |
|----------|---------|----------|
| VectorStore | 语义搜索 | 向量相似度 |
| KeywordTable | 精确查找 | 关键词匹配 |
```

**列表分组：** 同一级别的列表项会被尽量保持在同一个 Node 中：

```markdown
### 设计原则
1. 数据驱动决策
2. 渐进式优化
3. 可观测性优先
```

## 层级关系的利用：父子节点检索

层级解析不只是为了在 metadata 中存个路径——它还能改变检索的行为方式。LlamaIndex 支持**利用父子关系来增强检索质量**：

```python
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import IndexNode

parser = MarkdownNodeParser()
documents = SimpleDirectoryReader("./docs").load_data()
nodes = parser.get_nodes_from_documents(documents)

# 构建层级索引 — 子节点指向父节点
for i, node in enumerate(nodes):
    if i > 0:  # 第一个节点没有父节点
        # 创建一个指向父节点的索引关系
        node.relationships[NodeRelationship.PARENT] = RelatedNodeInfo(
            node_id=nodes[i - 1].node_id
        )

index = VectorStoreIndex(nodes)
query_engine = index.as_query_engine(
    similarity_top_k=3,
    # 启用父节点检索：如果子节点被命中，同时返回其父节点
    include_parent=True,
)
```

这种模式的原理是：**当一个细粒度的子节点（如某个具体的小节内容）被检索到时，它的父节点（包含更广泛的上下文）也被一并返回给 LLM**。这让 LLM 既能获得精准的信息片段，又能获得理解该片段所需的宏观背景。

```
普通检索:
  Query "Notion 接入"
    → Node C (2.3 API 连接器的具体内容)
    → 只给了 LLM 一个孤立的技术细节

父子检索:
  Query "Notion 接入"
    → Node C (2.3 API 连接器的具体内容)  ← 子节点
    → Node B (2. 数据接入层的概述)          ← 父节点
    → Node A (1. 系统概述)                 ← 祖父节点
    → LLM 同时获得了细节和全局视角
```

## 从层级解析到自动生成摘要

层级结构的另一个强大应用是**自动生成多级摘要**。既然我们已经知道了文档的层级结构，就可以在每个层级上生成摘要，形成一个"摘要金字塔"：

```
全文摘要 (Level 0):
  "本文介绍了企业级 RAG 系统的设计和实现，涵盖数据接入、
   索引策略、检索优化等方面..."

第1章摘要 (Level 1 - H1):
  "本章概述了 RAG 系统的整体架构、设计原则和技术选型..."

  1.1节摘要 (Level 2 - H2):
    "本节介绍了三大设计原则：数据驱动决策、渐进式优化..."

  1.2节摘要 (Level 2 - H2):
    "本节推荐了 LlamaIndex + pgvector 的技术组合..."

第2章摘要 (Level 1 - H1):
  "本章详细讲解了三类数据连接器的实现方式..."
```

这种多级摘要在用户提出概括性问题时特别有价值——系统可以根据问题的抽象程度选择合适层级的摘要来回答。

```python
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core import SummaryIndex

parser = MarkdownNodeParser()
nodes = parser.get_nodes_from_documents(documents)

# 按 header_level 分组
from collections import defaultdict
level_groups = defaultdict(list)
for node in nodes:
    level = node.metadata.get("header_level", 0)
    level_groups[level].append(node)

# 为每个层级创建摘要索引
summary_indexes = {}
for level, level_nodes in level_groups.items():
    if level > 0:  # 不为根级别创建摘要
        summary_indexes[level] = SummaryIndex(nodes=level_nodes)
```

## 常见误区

**误区一："层级解析只对长文档有用"。** 即使是只有几百字的文档，如果有明确的标题结构（如 FAQ 页面、API 文档的单个端点说明），层级解析也能带来好处——它能让系统区分"问题描述"和"问题答案"，避免把两者混在一起送入 LLM。

**误区二:"层级解析会让 chunk 变大"。** 不一定。层级解析的核心价值在于 metadata 中的路径信息，而不是把更多文本塞进 chunk。实际上，由于有了结构信息做上下文，你甚至可以使用更小的 chunk_size——因为不再需要在每个 chunk 中重复包含上下文信息了。

**误区三:"所有文档都适合用层级解析"。** 不适合的情况包括：纯文本日志（没有标题结构）、自由格式的对话记录（非层级组织）、高度非结构化的笔记（思维导图式的碎片化内容）。对于这些文档，普通的 SentenceSplitter 反而更合适。**根据文档的实际结构特征来选择解析策略。**
