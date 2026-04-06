---
title: 核心概念预览：Document / Node / Index / Query Engine / Response Synthesizer
description: LlamaIndex 五大核心概念的通俗解释与相互关系，建立整体认知框架
---
# 核心概念预览：Document / Node / Index / Query Engine / Response Synthesizer

在前面的几节里，我们已经用过 `SimpleDirectoryReader`、`VectorStoreIndex`、`query_engine` 这些组件，但你可能对它们之间的关系还有些模糊。这一节的目标就是把这些散落的珠子串起来，让你建立起对 LlamaIndex 核心概念体系的完整认知。

LlamaIndex 的架构围绕着五个核心概念展开，它们构成了一个从"原始数据"到"最终答案"的完整管线：

```
原始数据文件
    ↓
┌─────────────┐
│  Document   │  ← 数据加载阶段的产物
└──────┬──────┘
       ↓ (Node Parser)
┌─────────────┐
│    Node     │  ← 文档解析后的最小单元
└──────┬──────┘
       ↓ (Index Builder)
┌─────────────┐
│    Index    │  ← 数据的组织与索引结构
└──────┬──────┘
       ↓ (Query Engine Creator)
┌─────────────┐
│Query Engine │  ← 查询的入口与协调者
└──────┬──────┘
       ↓ (Response Synthesizer)
┌─────────────┐
│  Response   │  ← 最终答案 + 引用来源
└─────────────┘
```

下面我们逐一深入每个概念。

## Document：数据的容器

`Document` 是 LlamaIndex 中最基础的数据结构，代表一份从外部加载的原始文档。你可以把它想象成一个"带标签的文件夹"——里面装着文档的实际内容，外面贴着各种元数据标签。

```python
from llama_index.core import Document

doc = Document(
    text="这是一份产品说明书的内容...",
    metadata={
        "file_name": "product_manual.pdf",
        "file_path": "/data/product_manual.pdf",
        "category": "产品文档",
        "created_at": "2025-01-15",
        "author": "产品团队",
    },
    excluded_llm_metadata_keys=["file_path"],  # 这些 key 不会发给 LLM
    excluded_embed_metadata_keys=["created_at"],  # 这些 key 不会参与 embedding
)
```

Document 对象有三个核心属性需要理解：

**`text`（文本内容）：** 这是文档的主体内容，通常是纯文本格式。即使原始文件是 PDF 或 Word，加载器也会将其转换为纯文本。需要注意的是，转换过程中可能会丢失一些格式信息（如表格结构、字体样式等），这也是为什么后面会有专门的章节讲解高级文档解析。

**`metadata`（元数据）：** 这是一个字典，存放关于文档的描述性信息。元数据有两个重要作用：
1. **过滤依据**——你可以在查询时根据元数据过滤结果（如"只搜索作者为张三的文档"）
2. **上下文增强**——元数据可以被注入到发送给 LLM 的 Prompt 中，帮助 LLM 更好地理解文档背景

这里有一个容易被忽略的设计细节：`excluded_llm_metadata_keys` 和 `excluded_embed_metadata_keys`。这两个字段允许你控制哪些元数据参与哪些流程。比如文件路径信息对 LLM 回答问题没有帮助（排除在 LLM 之外），但对调试很有用（保留在 metadata 中）；创建时间信息对 embedding 没有意义（排除在 embedding 之外），但可能用于排序（保留在 metadata 中）。这种细粒度的控制在生产环境中非常有价值。

**`id_`（唯一标识）：** 每个文档都有一个全局唯一的 ID，通常由 UUID 生成。这个 ID 用于在索引和检索过程中跟踪文档的身份。一般情况下你不需要手动指定它，但如果你需要在不同会话间保持文档身份的一致性（比如做增量更新），就可以手动设置：

```python
import uuid
doc = Document(
    text="...",
    id_=str(uuid.uuid5(uuid.NAMESPACE_URL, "https://example.com/doc/001"))
)
```

### Document 从哪里来？

在实际应用中，你很少会手动创建 Document 对象——它们通常由各种 **Loader（加载器）** 自动产生：

```python
from llama_index.core import SimpleDirectoryReader

reader = SimpleDirectoryReader(
    input_dir="./data",
    required_exts=[".pdf", ".md"],
    recursive=True,
)
documents = reader.load_data()

print(f"共加载 {len(documents)} 份文档")
for doc in documents[:3]:
    print(f"  ID: {doc.id_[:8]}...")
    print(f"  文件: {doc.metadata['file_name']}")
    print(f"  长度: {len(doc.text)} 字符")
    print()
```

`SimpleDirectoryReader` 是最常用的加载器，它会自动识别文件类型并调用相应的解析器。第二章会详细介绍各种 Connector（连接器），包括文件、数据库、API、云服务等数十种数据源。

## Node：索引的最小单元

如果说 Document 是一本完整的书，那么 Node 就是书中的一个**章节或段落**。在 LlamaIndex 中，原始文档很少直接被索引——它们首先会被切分成更小的单元，这个单元就是 Node。

```python
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

document = Document(text="很长的一篇文档内容..." * 100)

splitter = SentenceSplitter(
    chunk_size=256,
    chunk_overlap=50,
)
nodes = splitter.get_nodes_from_documents([document])

print(f"生成了 {len(nodes)} 个节点")
for i, node in enumerate(nodes[:3]):
    print(f"\n--- Node #{i+1} ---")
    print(f"ID: {node.node_id[:8]}...")
    print(f"所属文档: {node.ref_doc_id[:8]}...")
    print(f"内容长度: {len(node.text)} 字符")
    print(f"内容预览: {node.text[:80]}...")
```

Node 继承自 Document，因此它拥有 `text`、`metadata`、`id_`（在 Node 中叫 `node_id`）这三个基础属性。但它增加了几个重要的新属性：

**`ref_doc_id`（引用文档 ID）：** 这个字段记录了该 Node 来自哪个 Document。这个关系非常重要——当检索返回一个 Node 时，你可以通过 `ref_doc_id` 追溯到原始文档，从而告诉用户"这个答案来自 product_manual.pdf 第 3 节"。这就是 LlamaIndex 能够自动提供引用来源的技术基础。

**`start_char_idx` 和 `end_char_idx`（位置索引）：** 记录了该 Node 的内容在原始 Document 文本中的起止字符位置。这对于"高亮显示原文出处"这类功能至关重要——你知道答案出自原文的哪个位置，就能在前端做精准的高亮标记。

**`relationships`（关系图）：** 这是一个更高级的概念。Node 之间可以建立各种关系：

```python
from llama_index.core.schema import NodeRelationship, RelatedNodeInfo

node.relationships[NodeRelationship.NEXT] = RelatedNodeInfo(
    node_id="next_node_id"
)
node.relationships[NodeRelationship.PARENT] = RelatedNodeInfo(
    node_id="parent_document_id"
)
```

常见的关系类型包括：
- `PREVIOUS` / `NEXT` — 前后相邻的 Node（用于顺序浏览）
- `PARENT` — 所属的父级 Document
- `SOURCE` — 原始数据源（如 URL、文件路径）
- `CHILD` — 子节点（用于层级化文档）

这些关系使得 LlamaIndex 不只是一个扁平的向量搜索引擎，而是能够理解文档结构的**知识图谱**。在第五章讲高级检索时，你会看到如何利用这些关系实现"找到当前段落的下一节"这类高级查询。

### 为什么不直接用 Document？

你可能会问：**为什么不直接把整个 Document 作为索引单元，非要切成 Node？** 原因有三个：

**原因一：粒度控制。** 一个 Document 可能长达数万字（比如一份完整的产品手册），如果把它作为一个整体来索引，检索时会返回整篇文档——即使用户只问了其中一个很小的问题。这不仅浪费 LLM 的输入 token，还会引入大量无关信息干扰答案质量。切成 Node 后，每次只返回最相关的几个段落，精准且高效。

**原因二：嵌入质量。** 嵌入模型（如 `text-embedding-3-small`）对长文本的编码能力是有限的。一般来说，**256-512 个 token 的文本块能获得最佳的嵌入质量**——太短缺乏语义完整性，太长则主题发散导致向量表征模糊。Node 的分块机制正是为了保证每个单元都在这个最佳范围内。

**原因三：灵活性。** 不同 Document 可以有不同的分块策略。法律合同可能按"条"来切分，技术文档可能按"节"来切分，聊天记录可能按"消息"来切分。Node 抽象让这种差异化处理成为可能。

## Index：数据的组织结构

Index 是 LlamaIndex 中最核心的概念——它是**数据从"一堆文本"变成"可查询的知识库"的关键转化步骤**。你可以把 Index 想象成一个图书馆的目录系统：没有它，你只能在书架上盲目翻找；有了它，你能快速定位到任何一本书的具体位置。

LlamaIndex 提供了多种 Index 类型，每种对应不同的数据组织策略。目前你只需要认识最重要的一种：

### VectorStoreIndex（向量存储索引）

这是最常用的索引类型，也是我们在前面几节的例子中一直在使用的。它的原理是：

```
Documents → Nodes → Embedding Vectors → Vector Store
                                    ↓
                          ┌─────────────────────┐
                          │   Vector Database   │
                          │                     │
                          │  Node_1 → [0.1, ...]│
                          │  Node_2 → [-0.3,...]│
                          │  Node_3 → [0.7, ...]│
                          │  ...                │
                          └─────────────────────┘
```

每个 Node 的文本内容通过嵌入模型转换为一个高维向量（通常是 768 维或 1536 维），这些向量被存储在一个向量数据库中。查询时，用户的问题也被转换为向量，然后在向量空间中找出距离最近的 K 个 Node。

```python
from llama_index.core import VectorStoreIndex

index = VectorStoreIndex.from_documents(documents)
# 或者从已有的 nodes 创建
index = VectorStoreIndex(nodes=nodes)

# 也可以增量添加文档
for new_doc in new_documents:
    index.insert(new_doc)
```

`VectorStoreIndex` 在内部做了很多事情：
1. 调用 Node Parser 将 Document 切分为 Nodes（如果传入的是 Documents）
2. 调用 Embedding Model 为每个 Node 生成向量
3. 将向量存入 Vector Store（默认使用简单的内存存储，也可以换成 Chroma、Pinecone、Qdrant 等）
4. 构建辅助数据结构加速检索

### 其他索引类型预览

虽然 `VectorStoreIndex` 覆盖了 80% 的使用场景，但了解其他索引类型有助于你建立完整的认知地图：

- **ListIndex**：将所有 Node 存储在一个有序列表中，查询时依次遍历。适用于需要阅读全部内容的场景（如"总结这篇文档"）。
- **TreeIndex**：将 Node 组织成一棵树，每个节点是其子节点的摘要。适用于层级化文档（如书籍、法规）。
- **KeywordTableIndex**：为每个 Node 提取关键词，建立关键词到 Node 的映射表。适用于精确关键词匹配。
- **SummaryIndex**：为整个文档集生成一个全局摘要。适用于"这篇文章主要讲了什么"这类概括性查询。
- **GraphIndex**：将实体和关系提取为图谱结构。适用于"张三和李四是什么关系"这类关系型查询。

第四章会对每种索引类型做深入的原理讲解和实践指导。

## Query Engine：查询的入口

如果说 Index 是图书馆的目录，那 **Query Engine 就是图书馆的前台接待员**——你告诉它你想找什么，它会去各个索引中查找相关信息，整理后交给你一个完整的答案。

```python
query_engine = index.as_query_engine(
    similarity_top_k=5,           # 检索最相关的 5 个节点
    response_mode="compact",      # 响应合成模式
    streaming=False,              # 是否流式输出
)

response = query_engine.query("产品的保修期是多长？")
```

`as_query_engine()` 是 Index 对象提供的方法，它会根据 Index 的类型自动创建最适合的 Query Engine。对于 `VectorStoreIndex`，创建的是一个 `RetrieverQueryEngine`，其内部工作流程如下：

```
用户问题 "产品的保修期是多长？"
       │
       ▼
┌──────────────┐
│  Retriever   │  从 Index 中检索 top-k 个相关 Node
│  (检索器)    │  返回: [Node_A(score:0.92), Node_B(score:0.87), ...]
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Synthesize  │  将检索结果 + 问题 合成为最终答案
│  (合成器)    │  返回: Response(response_text, source_nodes=[...])
└──────┬───────┘
       │
       ▼
   Response 对象
```

Query Engine 内部由两个子组件协作完成：**Retriever（检索器）** 和 **Response Synthesizer（响应合成器）**。这两个组件都可以独立配置和替换，这也是 LlamaIndex 高度可定制的体现。

### Retriever 的配置选项

```python
query_engine = index.as_query_engine(
    similarity_top_k=5,                    # 返回几个结果
    filters=MetadataFilter(key="category", # 元数据过滤
                           value="产品文档",
                           operator=FilterOperator.EQ),
    node_postprocessors=[               # 后处理器流水线
        SimilarityPostProcessor(similarity_cutoff=0.7),  # 过滤低分结果
        CohereRerank(top_n=3),                      # 用 Cohere 重排序
    ],
)
```

这些选项让你能够精细控制检索行为：返回多少个结果、是否只搜索特定类型的文档、是否需要对结果做进一步的重排序或过滤。第六章会专门讲解 Query Engine 的高级配置。

### Response Synthesizer：答案的制造工厂

Response Synthesizer 负责**将检索到的 Node 和用户问题转化为最终的答案文本**。这听起来简单，但实际上有多种策略可以选择：

**Refine 模式（默认）：** 先用第一个 Node 生成初始答案，然后依次用后续 Node 来"精炼"（refine）这个答案。就像写论文时的反复修改——初稿 → 加入第二份参考资料修改 → 加入第三份资料再修改……

**Compact 模式：** 把所有检索到的 Node 拼接起来，如果总长度超过 LLM 的上下文限制，就自动截断或压缩，然后一次性让 LLM 生成答案。类似 LangChain 的做法。

**Tree Summarize 模式：** 先对每个 Node 生成摘要，然后把这些摘要两两合并，递归地进行直到得到最终答案。适合 Node 数量很多的场景。

**Simple Summarize 模式：** 最简单的方式——把所有内容一股脑塞进 Prompt，让 LLM 一次性回答。

```python
from llama_index.core import get_response_synthesizer
from llama_index.core.response_synthesizers import ResponseMode

synthesizer = get_response_synthesizer(
    response_mode=ResponseMode.REFINE,  # 选择合成模式
    use_async=True,                     # 异步执行（提高吞吐量）
)

query_engine = index.as_query_engine(
    response_synthesizer=synthesizer,
    similarity_top_k=10,  # Refine 模式可以承受更多的检索结果
)
```

第七章会深入探讨每种模式的适用场景和性能特征。

## 五个概念的协作关系

现在让我们把这五个概念放在一起，看它们如何协作完成一次完整的查询：

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# Stage 1: Document — 加载原始数据
documents = SimpleDirectoryReader("./data").load_data()
# 结果: [Document(text=..., metadata={...}), Document(...), ...]

# Stage 2: Node — 文档被切分为更小的单元
# （这一步在 from_documents 内部自动完成）
index = VectorStoreIndex.from_documents(documents)
# 内部: documents → SentenceSplitter → [Node, Node, Node, ...]

# Stage 3: Index — Nodes 被组织成可查询的结构
# （同上，在 from_documents 内部完成）
# 内部: nodes → Embedding Model → Vector Store

# Stage 4: Query Engine — 创建查询入口
query_engine = index.as_query_engine(similarity_top_k=3)
# 内部: 创建 Retriever + ResponseSynthesizer

# Stage 5: Response — 执行查询并获得结果
response = query_engine.query("退款政策是什么？")
# 内部: Retriever 检索 → Synthesizer 合成 → Response 对象

# 使用 Response
print(response.response)        # 答案文本
for node in response.source_nodes:  # 引用来源
    print(f"[{node.score:.3f}] {node.text[:60]}")
```

这五个概念形成了一个清晰的流水线，每个概念都有明确的职责边界：

| 概念 | 职责 | 输入 | 输出 |
|------|------|------|------|
| **Document** | 承载原始数据 | 文件/数据库/API | 带元数据的文本对象 |
| **Node** | 细粒度切分 | Document 列表 | 带关系的文本块 |
| **Index** | 组织与索引 | Node 列表 | 可查询的数据结构 |
| **Query Engine** | 协调查询 | Index + 配置 | 检索 + 合成的协调者 |
| **Response** | 呈现结果 | 问题 + 检索结果 | 答案 + 来源 |

## 常见误区

**误区一："Document 和 Node 差不多，随便用哪个"。** 不一样。Document 代表原始文档的完整性，Node 代表索引的原子性。混淆两者会导致元数据丢失、引用溯源失败、分块策略失效等一系列问题。**始终记住：Document 是给人看的，Node 是给机器检索的。**

**误区二："一种 Index 打天下"。** 虽然 VectorStoreIndex 覆盖了大多数场景，但某些查询模式用它效率很低甚至完全不适用。比如"列出文档中提到的所有人名"这种提取型查询，用 KeywordTableIndex 或 GraphIndex 会好得多。**选择正确的索引类型是 RAG 系统性能的第一要素。**

**误区三："Query Engine 只能问问题"。** `query()` 方法确实是最常用的接口，但 Query Engine 还支持 `aquery()`（异步查询）、`chat()`（多轮对话，维护上下文历史）等模式。对于交互式应用，`chat()` 模式往往更适合——它能记住之前的对话内容，避免用户重复提供背景信息。

**误区四："Response 只是一个字符串"。** `Response` 对象包含的信息远不止答案文本——还有 `source_nodes`（来源节点及分数）、`metadata`（响应元数据）。忽略这些信息等于丢弃了 LlamaIndex 最有价值的调试和信任建设能力。

到这里，第一章的全部五个小节就结束了。你应该已经建立了对 LlamaIndex 的整体认知：它是什么、解决什么问题、怎么安装、怎么写出第一个应用、和 LangChain 有什么区别、核心概念有哪些。从下一章开始，我们将深入数据连接器的世界——看看 LlamaIndex 如何优雅地接入各种异构数据源。
