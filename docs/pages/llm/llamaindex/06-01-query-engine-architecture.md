---
title: Query Engine 架构与工作原理
description: Query Engine 的内部组成、Retriever + Synthesizer 协作模式、请求生命周期、与 LangChain Chain 的对比
---
# Query Engine 架构与工作原理

从第一章的第一个 RAG 示例开始，`query_engine` 就一直是我们与 LlamaIndex 系统交互的主要接口——我们调用 `query_engine.query("问题")`，然后得到答案。但这个看似简单的调用背后，隐藏着一个精心设计的多层架构。

在前面五章中，我们分别学习了数据如何被加载（第二章）、如何被解析（第三章）、如何被索引（第四章）、以及如何被高级检索技术处理（第五章）。而 **Query Engine 就是所有这些能力的"指挥中心"**——它协调 Retriever（检索器）和 Response Synthesizer（响应合成器）来完成从"用户问题"到"最终答案"的完整流程。

这一节我们将打开 Query Engine 的黑盒，深入理解它的内部架构和工作原理。

## Query Engine 在整体架构中的位置

回顾一下我们在第一章建立的认知框架：

```
Document → Node → Index → [Query Engine] → Response
                              ↑
                        本章的重点
```

Query Engine 位于 Index 和 Response 之间，是**连接"数据层"和"答案层"的桥梁**。更准确地说：

```
┌─────────────────────────────────────────────────────┐
│                   用户问题 (Query)                     │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│              Query Engine (查询引擎)                 │
│                                                     │
│   ┌─────────────────────────────────────────────┐   │
│   │           Retriever (检索器)                  │   │
│   │   职责: 从 Index 中找到相关内容               │   │
│   └────────────────────┬────────────────────────┘   │
│                      │ 相关节点列表                    │
│                      ▼                               │
│   ┌─────────────────────────────────────────────┐   │
│   │     Response Synthesizer (响应合成器)          │   │
│   │   职责: 将节点 + 问题合成为最终答案            │   │
│   └────────────────────┬────────────────────────┘   │
│                      │                               │
└──────────────────────┼──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│           Response (最终答案)                         │
│                                                     │
│   response.response    → 答案文本                    │
│   response.source_nodes → 来源节点及分数             │
│   response.metadata    → 响应元数据                  │
└─────────────────────────────────────────────────────┘
```

这个"Retriever + Synthesizer"的两段式设计是 LlamaIndex Query Engine 最核心的架构决策。理解了这一点，你就理解了 Query Engine 80% 的工作方式。

## Retriever：检索器

Retriever 的职责非常明确：**给定一个查询，从索引中返回最相关的 Node 列表**。

```python
from llama_index.core import VectorStoreIndex

index = VectorStoreIndex.from_documents(documents)

retriever = index.as_retriever(
    similarity_top_k=5,      # 返回 top-5 个结果
)

nodes = retriever.retrieve("产品的保修期是多长？")

for node in nodes:
    print(f"[score={node.score:.3f}] {node.text[:80]}...")
```

### Retriever vs query() 的区别

你可能会问：`retriever.retrieve()` 和 `query_engine.query()` 有什么区别？区别在于：

| 操作 | retriever.retrieve() | query_engine.query() |
|------|---------------------|----------------------|
| 输入 | 查询字符串 | 查询字符串 |
| 输出 | `List[NodeWithScore]` | `Response` 对象 |
| 是否调用 LLM | ❌ 不调用 | ✅ 调用（用于合成答案） |
| 用途 | 调试、分析检索质量 | 生产使用、获取完整答案 |

`retrieve()` 只做检索不做合成——它返回的是原始的检索结果（Node 列表 + 相似度分数）。这对于调试非常有用：你可以检查检索阶段是否找到了正确的内容，而不需要等待 LLM 生成完整的答案。

### 不同类型的 Retriever

不同的 Index 类型会产生不同类型的 Retriever：

```python
# VectorStoreIndex → 向量检索器
vector_index = VectorStoreIndex.from_documents(docs)
vector_retriever = vector_index.as_retriever(similarity_top_k=5)

# ListIndex → 全文遍历检索器
list_index = ListIndex.from_documents(docs)
list_retriever = list_index.as_retriever()

# KeywordTableIndex → 关键词检索器
keyword_index = KeywordTableIndex.from_documents(docs)
keyword_retriever = keyword_index.as_retriever()
```

第五章学过的混合检索也是一种特殊的 Retriever：

```python
hybrid_retriever = QueryFusionRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    mode="reciprocal_rank",
)
```

## Response Synthesizer：响应合成器

如果 Retriever 负责"找到相关内容"，那 Response Synthesizer 就负责**把这些内容和用户的问题组合成一段连贯、准确的答案**。

```python
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.response_synthesizers import ResponseMode

synthesizer = get_response_synthesizer(
    response_mode=ResponseMode.REFINE,
    use_async=True,
)

response = synthesizer.synthesize(
    query="产品的保修期是多长？",
    nodes=retrieved_nodes,  # 来自 Retriever 的结果
)
```

### 四种合成模式

Response Synthesizer 支持四种主要的合成模式（第七章会详细讲解每种模式的内部机制）：

**1. REFINE（精炼模式）— 默认推荐**

```
Step 1: 用第 1 个 Node 生成初始答案
  "根据文档，产品提供 24 个月保修..."

Step 2: 用第 2 个 Node 精炼答案
  "结合补充信息，保修范围包括..."

Step 3: 用第 3 个 Node 继续精炼
  "另外需要说明的是..."

... (对所有 Node 依次执行)

Final: 完整的综合答案
```

**适用场景：** 需要综合多个信息源的场景；Node 数量适中（<20）；希望答案有层次感。

**2. SIMPLE_SUMMARIZE（简单汇总）**

```
将所有 Node 文本拼接:
[Node 1 内容]
[Node 2 内容]
[Node 3 内容]
...

一次性发给 LLM:
"根据以上信息回答: {query}"
```

**适用场景：** Node 数量少（<5）；追求速度；对答案深度要求不高。

**3. COMPACT_ACCUMULATE（压缩累积）**

```
先尝试把所有 Node 塞进上下文窗口
如果塞不下 → 自动压缩/截断 → 再发给 LLM
```

**适用场景：** Node 数量不确定（可能多可能少）；需要一个"自适应"方案。

**4. TREE_SUMMARIZE（树状汇总）**

```
先对每个 Node 生成摘要
然后两两合并摘要
递归直到得到最终答案
```

**适用场景：** Node 数量很多（>20）；需要对大量信息做结构化总结。

## 从组件到 Query Engine：组装过程

现在我们来看 `index.as_query_engine()` 到底做了什么：

```python
query_engine = index.as_query_engine(
    similarity_top_k=5,
    response_mode="refine",
    streaming=False,
)
```

这行代码等价于以下手动组装过程：

```python
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizer import get_response_synthesizer
from llama_index.core.response_synthesizers import ResponseMode

# Step 1: 创建 Retriever
retriever = index.as_retriever(similarity_top_k=5)

# Step 2: 创建 Response Synthesizer
synthesizer = get_response_synthesizer(
    response_mode=ResponseMode.REFINE,
    streaming=False,
)

# Step 3: 组装为 Query Engine
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=synthesizer,
)
```

`as_query_engine()` 只是一个便捷方法，它帮你自动完成了上述三步。但当你需要更精细的控制时（比如使用自定义的 Retriever 或 Synthesizer），手动组装的方式就更有用了。

## 一次完整的查询请求生命周期

让我们追踪一个查询从进入到返回的完整生命周期：

```
用户调用: query_engine.query("S1 的保修期？")

═══ Phase 1: 查询预处理 ═══
│
│  1.1 接收查询字符串
│  1.2 如果配置了 transform_queries (HyDE/MultiQuery 等)
│       → 执行查询转换
│       → 可能产生多个子查询
│
═══ Phase 2: 检索 (Retrieval) ═══
│
│  2.1 将(子)查询发送给 Retriever
│  2.2 Retriever 在 Index 中搜索
│  2.3 返回候选 Node 列表 (带相似度分数)
│
═══ Phase 3: 后处理 (Postprocessing) ═══
│
│  3.1 如果配置了 node_postprocessors
│       → 依次执行: 过滤 / rerank / 去重 / 增强 / 重排
│  3.2 得到最终的 Node 列表
│
═══ Phase 4: 响应合成 (Synthesis) ═══
│
│  4.1 将查询 + 最终 Node 列表发送给 Synthesizer
│  4.2 Synthesizer 根据 response_mode 执行合成逻辑
│  4.3 可能涉及多次 LLM 调用 (取决于模式和 Node 数量)
│
═══ Phase 5: 响应构建 ═══
│
│  5.1 构建 Response 对象
│       - .response: 最终答案文本
│       - .source_nodes: 使用的 Node 及其元数据/分数
│       - .metadata: 响应级别的元信息
│  5.2 返回给调用者
```

每一 phase 都是可以独立定制和替换的——这就是 LlamaIndex Query Engine 设计的灵活性所在。

## Query Engine 与 LangChain Chain 的深层对比

既然我们已经深入了解了 Query Engine 的架构，现在可以做一个更有深度的与 LangChain Chain 的对比：

| 维度 | LlamaIndex Query Engine | LangChain Chain |
|------|------------------------|-----------------|
| **抽象层级** | 面向 RAG 场景的高级抽象 | 通用的管道抽象 |
| **默认行为** | 开箱即用的 RAG 最佳实践 | 需要自己组装每一步 |
| **Retriever 概念** | 一等公民，多种类型可选 | 存在但通常较简单 |
| **Synthesizer** | 内置 4+ 种合成策略 | 通常只有一次 LLM 调用 |
| **来源追踪** | 自动内置（source_nodes） | 需要自行实现 |
| **流式输出** | 一等公民支持 | 支持（通过 Streaming） |
| **可观测性** | 内置 callbacks 系统 | 通过 LangSmith |

一个具体的代码对比最能说明差异。同样的"从 PDF 构建 RAG 并查询"任务：

**LangChain 方式（约 40 行）：**
```python
loader = DirectoryLoader("./data", glob="**/*.pdf")
docs = loader.load()
splits = RecursiveCharacterTextSplitter(...).split_documents(docs)
vectorstore = Chroma.from_documents(splits, embeddings)
retriever = vectorstore.as_retriever()

prompt = ChatPromptTemplate.from_messages([...])
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
result = chain.invoke("问题")
# result 只是字符串，没有来源信息
```

**LlamaIndex 方式（约 10 行）：**
```python
documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(similarity_top_k=5)
response = query_engine.query("问题")
print(response.response)        # 答案
for n in response.source_nodes: # 来源
    print(f"[{n.score:.3f}] {n.text[:80]}")
```

LlamaIndex 的代码量不到 LangChain 的 1/4，而且**自动获得了来源追踪能力**——这在 LangChain 中需要额外的大量代码来实现。

当然，这种简洁性的代价是**灵活性的降低**。如果你需要完全自定义 RAG 管道的每一个细节（比如在检索和合成之间插入自定义的业务逻辑），LangChain 的 Chain 表达式可能更合适。但对于绝大多数 RAG 场景来说，LlamaIndex Query Engine 提供的默认行为已经足够好且大大减少了样板代码。

## 常见误区

**误区一:"Query Engine 和 Retriever 是一回事"。** 不对。Retriever 只负责"找东西"，Query Engine 负责"找东西 + 给出答案"。当你只需要检查检索质量时用 Retriever；当需要完整答案时用 Query Engine。

**误区二:"as_query_engine() 的参数就是全部可配置项"。** 不够。`as_query_engine()` 只暴露了最常用的参数。如果你需要配置 Retriever 的详细参数（如混合检索的权重）或 Synthesizer 的详细参数（如 refine 的 prompt 模板），应该手动组装 `RetrieverQueryEngine`。

**误区三:"一个 Index 只能有一个 Query Engine"。** 完全可以从同一个 Index 创建多个配置不同的 Query Engine——比如一个用 REFINE 模式做深度问答，另一个用 SIMPLE_SUMMARIZE 模式做快速概览。它们共享同一个底层数据但服务于不同的查询需求。
