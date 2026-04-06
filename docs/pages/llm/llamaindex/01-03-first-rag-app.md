---
title: 第一个 RAG 应用：5 行代码跑通问答系统
description: 用最少代码实现完整的 RAG 流程，理解数据加载→索引构建→查询的全链路
---
# 第一个 RAG 应用：5 行代码跑通问答系统

前面两节讲了"为什么需要 LlamaIndex"以及"如何搭建环境"，现在终于到了最激动人心的时刻——让我们用代码来感受一下 LlamaIndex 到底有多简洁。我会带你从零开始，用尽可能少的代码跑通一个完整的 RAG 问答系统，然后逐步拆解每一行代码背后的原理。

## 最简 RAG：真的只要 5 行

假设你的项目目录下有一个 `data/` 文件夹，里面放了一些 `.txt` 文件作为知识库（比如公司产品文档）。那么，一个能工作的 RAG 系统只需要这些代码：

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
response = query_engine.query("公司的退款政策是什么？")
print(response)
```

是的，你没数错——**真正核心的逻辑就是 5 行代码**。我们来逐行拆解每一步发生了什么：

**第 1 行：加载数据**
```python
documents = SimpleDirectoryReader("./data").load_data()
```
`SimpleDirectoryReader` 会扫描指定目录下的所有文件（默认支持 txt、md、pdf、html、csv 等常见格式），把它们加载为 `Document` 对象列表。每个 `Document` 对象包含三个核心属性：
- `text` — 文档的纯文本内容
- `metadata` — 元数据字典（如文件名、文件路径、创建时间等）
- `id_` — 全局唯一的文档标识符

**第 2 行：构建索引**
```python
index = VectorStoreIndex.from_documents(documents)
```
这一行做了很多事。首先，文档会被自动切分成较小的文本块（chunks）；然后每个块会被转换成向量表示（通过嵌入模型）；最后这些向量会被存入一个向量存储中（默认使用内存存储）。整个过程对用户是完全透明的——你不需要关心分块策略是什么、用的什么嵌入模型、向量存在哪里。

**第 3 行：创建查询引擎**
```python
query_engine = index.as_query_engine()
```
索引本身只是数据的组织形式，要回答问题还需要一个"查询引擎"。查询引擎负责接收用户的问题、从索引中检索相关内容、然后把检索结果交给 LLM 生成最终答案。

**第 4-5 行：提问并获取答案**
```python
response = query_engine.query("公司的退款政策是什么？")
print(response)
```

当你调用 `query()` 时，背后发生了一整套 RAG 流程：
1. 用户问题被转换为向量（Query Embedding）
2. 在向量存储中做相似度搜索，找到最相关的 K 个文本块
3. 把问题和检索到的文本块组合成一个 Prompt
4. 发送给 LLM 生成答案
5. 返回生成的答案

## 让它变得更实用：带来源引用的版本

上面的  行版本虽然能工作，但在实际使用中还有一个关键需求——**用户想知道答案来自哪里**。LlamaIndex 的响应对象天然支持这一点：

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(similarity_top_k=3)
response = query_engine.query("公司的退款政策是什么？")

print("=" * 50)
print("答案:", response.response)
print("=" * 50)
print("\n引用来源:")
for node in response.source_nodes:
    print(f"  [相关度 {node.score:.3f}] {node.metadata.get('file_name', '未知')}")
    print(f"  内容片段: {node.text[:100]}...")
    print()
```

运行后会输出类似这样的结果：

```
==================================================
答案: 根据公司的退款政策，客户可在购买后30天内申请全额退款...
==================================================

引用来源:
  [相关度 0.897] refund_policy.txt
  内容片段: 公司为客户提供30天无理由退款服务。若产品存在质量问题...

  [相关度 0.823] terms_of_service.txt
  内容片段: 退款流程：1. 登录账户进入订单管理页面；2. 选择需要退款的订单...
```

这里的 `response` 对象是一个 `Response` 类实例，它包含丰富的信息：

| 属性 | 类型 | 说明 |
|------|------|------|
| `response` | `str` | LLM 生成的最终答案文本 |
| `source_nodes` | `list[NodeWithScore]` | 检索到的节点列表（含相关性分数） |
| `metadata` | `dict` | 响应级别的元数据 |

每个 `source_nodes` 中的元素都是一个 `NodeWithScore` 对象，包含了原始文本、元数据和与查询的相关性分数。这种设计让调试和展示变得非常方便——你既可以给终端用户展示引用来源，也可以在后台分析每次查询的检索质量。

## 理解 RAG 的完整数据流

光看代码还不够，我们需要理解数据在整个 RAG 管道中是如何流动的。让我用一个具体的例子来说明：

假设你的 `data/` 目录下有两个文件：

**data/product_guide.txt:**
```
智能音箱 S1 产品指南

一、产品概述
智能音箱 S1 是本公司最新推出的智能语音助手设备，支持语音控制智能家居，
内置高品质扬声器，音质清晰饱满。

二、退款政策
购买后30天内可无理由申请退款。退款将在3-5个工作日内原路退回。
```

**data/faq.txt:**
```
常见问题解答

Q: 如何申请退款？
A: 请登录官网账户，进入"我的订单"，点击对应订单的"申请退款"按钮，
填写退款原因后提交审核。
```

当用户问"怎么退款"时，完整的数据流是这样的：

```
用户问题: "怎么退款"
       │
       ▼
┌──────────────────┐
│ 1. Query Embed   │  将问题转为向量 [0.12, -0.34, 0.56, ...]
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────┐
│ 2. Vector Similarity Search  │  在向量空间中找最近的邻居
│                              │
│   ┌────────────────────┐     │
│   │ Chunk #12 (score:0.89) │ ← "购买后30天内可无理由申请退款..."
│   │ Chunk #23 (score:0.85) │ ← "请登录官网账户...点击申请退款按钮..."
│   │ Chunk #05 (score:0.72) │ ← "智能音箱S1是本公司最新推出..." (不太相关)
│   └────────────────────┘     │
└────────┬─────────────────────┘
         │
         ▼
┌──────────────────────────────────────────┐
│ 3. Prompt Assembly                       │
│                                          │
│ System: 你是一个有帮助的助手。              │
│   请根据以下上下文信息回答用户的问题。      │
│   如果你不知道答案，就说不知道。            │
│                                          │
│ Context:                                 │
│   [1] 购买后30天内可无理由申请退款...       │
│   [2] 请登录官网账户...点击申请退款按钮...  │
│                                          │
│ User Question: 怎么退款                   │
└────────┬─────────────────────────────────┘
         │
         ▼
┌──────────────────┐
│ 4. LLM Generate  │  GPT-4o-mini 生成答案
└────────┬─────────┘
         │
         ▼
  最终答案 + 来源引用
```

这个流程图揭示了 RAG 系统的核心价值：**LLM 不再需要"记住"所有知识，它只需要学会如何在正确的时候找到正确的信息并加以综合。** 这大大减少了幻觉（hallucination）的可能性，因为答案是有据可查的。

## 自定义分块策略

前面的例子使用了默认的分块策略，但在实际项目中，你可能需要根据文档类型调整分块参数。比如对于技术文档，你希望每个 chunk 包含一个完整的章节；对于法律合同，你希望按条款来切分。

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

documents = SimpleDirectoryReader("./data").load_data()

splitter = SentenceSplitter(
    chunk_size=512,       # 每个 chunk 最大 512 个字符
    chunk_overlap=100,    # 相邻 chunk 之间重叠 100 个字符
)
nodes = splitter.get_nodes_from_documents(documents)

index = VectorStoreIndex(nodes)  # 注意：这里传的是 nodes 而非 documents
query_engine = index.as_query_engine(similarity_top_k=5)
response = query_engine.query("产品的保修期是多长？")
print(response.response)
```

这里有几点值得深入说明：

**为什么需要 chunk_overlap？** 想象一下，如果没有 overlap，一段跨边界的句子可能会被切断："我们的产品提供两年保修服务。如果在保修期内出现质量问题..." 这句话的前半句在 chunk A 末尾，后半句在 chunk B 开头。当用户问"保修期多久"时，chunk A 只包含"两年保修服务。"但没有上下文说明这是什么产品的保修；chunk B 有上下文但不包含"两年"这个关键信息。有了 overlap，两个 chunk 都会包含完整的上下文，检索质量就会大幅提升。

**chunk_size 设多少合适？** 这是一个没有标准答案但非常重要的问题，取决于多个因素：
- **文档类型**：技术文档 512-1024 较好，法律文书 1024-2048 更合适，社交媒体短文 256 就够了
- **嵌入模型维度**：像 `text-embedding-3-small` 支持 8192 token 的上下文，较大的 chunk 也能有效编码
- **查询类型**：如果是简单的事实查询（"谁创建了这家公司？"），小 chunk 更精确；如果是需要综合分析的复杂问题（"总结这家公司的商业模式"），大 chunk 效果更好
- **经验法则**：从 512 开始，根据检索质量调整。太小会丢失上下文，太大会引入噪音

## 使用流式输出

对于长答案或实时交互场景，流式输出能让用户体验更好——用户不需要等到整个答案生成完毕才能看到第一行文字：

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(streaming=True)

streaming_response = query_engine.query("详细介绍一下这款产品的功能特点")

for text in streaming_response.response_gen:
    print(text, end="", flush=True)
```

`streaming=True` 参数让查询引擎以流式模式工作。`response_gen` 是一个生成器，每次 yield 一小段文本。这在 Web 应用中特别有用——你可以用 Server-Sent Events (SSE) 或 WebSocket 把这些增量文本实时推送到前端。

## 常见误区与最佳实践

**误区一："5 行代码就够了，不需要理解内部机制"。** 这 5 行代码确实能跑通 demo，但生产环境的 RAG 系统远比这复杂。你需要关注分块策略、检索质量调优、响应合成模式选择、评估指标等问题。这 5 行代码的价值在于帮你建立直觉，而不是替代深入学习。

**误区二："VectorStoreIndex 是唯一的索引类型"。** `VectorStoreIndex` 只是最常用的一种索引（基于向量相似度），LlamaIndex 还提供了 `ListIndex`（顺序遍历）、`TreeIndex`（层级摘要）、`KeywordTableIndex`（关键词匹配）、`SummaryIndex`（全局摘要）、`GraphIndex`（知识图谱）等多种索引类型。不同场景下选择合适的索引类型能显著提升效果，我们会在第四章详细讲解。

**误区三："默认参数就是最优的"。** LlamaIndex 的默认参数是为通用场景设计的"合理默认值"，但它们几乎永远不会是你特定场景下的最优值。比如默认的 `similarity_top_k=3` 对于简单查询可能太多（引入噪音），对于复杂问题又太少（遗漏关键信息）。**一定要根据你的数据和查询模式来调优这些参数。**

**误区四："RAG 系统建好了就不用管了"。** RAG 系统的质量会随着数据变化而漂移。新文档加入后，旧的 embedding 可能不再是最优的；用户的查询模式也会随时间演变。建立持续评估机制（第八章会讲）是保证长期质量的关键。

**性能提示：** 当文档数量较多时，`VectorStoreIndex.from_documents()` 可能会比较耗时（因为它要对每个 chunk 计算 embedding）。对于首次构建，这是正常的。LlamaIndex 支持将索引持久化到磁盘，下次启动时直接加载，不需要重新计算：

```python
# 首次构建后保存
index.storage_context.persist(persist_dir="./index_storage")

# 下次直接加载
from llama_index.core import StorageContext, load_index_from_storage
storage_context = StorageContext.from_defaults(persist_dir="./index_storage")
index = load_index_from_storage(storage_context)
query_engine = index.as_query_engine()
```

至此，你已经成功运行了第一个 RAG 应用，并且理解了它的基本工作原理。下一节，我们会拿同样的任务分别用 LlamaIndex 和 LangChain 来实现，看看两者在代码风格和设计理念上的差异到底在哪里。
