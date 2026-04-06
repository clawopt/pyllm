---
title: 与 LangChain RAG 的对比：同一问题的两种解法
description: 同一个 RAG 任务用 LlamaIndex 和 LangChain 分别实现，深度对比两种框架的设计哲学差异
---
# 与 LangChain RAG 的对比：同一问题的两种解法

既然你已经用 LlamaIndex 写出了一个能工作的 RAG 应用，那自然会产生一个问题：**如果用 LangChain 来完成一模一样的任务，代码会长什么样？两者的区别到底在哪里？** 这一节我们就来做一次正面对比——用两种框架分别实现同一个 RAG 问答系统，然后从代码结构、设计理念、扩展性等多个维度进行深度比较。

## 任务定义

为了公平对比，我们先明确任务需求：

> 从 `./data` 目录加载 Markdown 格式的产品文档，构建一个 RAG 问答系统。用户提问时，系统应从文档中检索相关信息，然后用 GPT-4o-mini 生成带有引用来源的答案。

这是一个足够简单但又涵盖了 RAG 核心流程的任务，非常适合用来对比两个框架的差异。

## LangChain 实现

先用 LangChain 来实现这个任务。LangChain 的 RAG 实现通常采用 Retrieval Chain 的模式：

```python
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


loader = DirectoryLoader("./data", glob="**/*.md")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=100,
)
chunks = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(
    chunks,
    embeddings=embeddings,
    persist_directory="./chroma_db"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有帮助的助手。请根据以下上下文信息回答用户的问题。"
              "如果你不知道答案，就说不知道。\n\n上下文:\n{context}"),
    ("human", "{question}")
])

def format_docs(docs):
    return "\n\n".join(
        f"[文档 {i+1}] {doc.page_content}"
        for i, doc in enumerate(docs)
    )

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

result = chain.invoke("公司的退款政策是什么？")
print(result)
```

这段代码大约 40 行，涉及了 7 个不同的类/函数导入。让我们仔细分析一下它的结构和特点。

### LangChain 代码的结构分析

LangChain 的 RAG 实现采用了经典的 **LCEL（LangChain Expression Language）管道模式**。整个流程被表达为一个链式（chain）表达式：

```
{"context": retriever | format_docs, "question": RunnablePassthrough()}
    → prompt
    → llm
    → StrOutputParser()
```

每一层都是一个 `Runnable` 对象，数据像水流一样从一层传递到下一层。这种设计的优点是**高度灵活**——你可以在任意位置插入新的处理步骤、替换某个组件、或者把链拆开重新组合。但代价是**你需要自己组装每一个零件**：

- 你需要自己选 Document Loader
- 你需要自己选 Text Splitter 并配置参数
- 你需要自己选 Vector Store
- 你需要自己把 Retriever 的输出格式化成 Prompt 能接受的格式
- 你需要自己写 Prompt Template
- 你需要自己组装 Chain

这就是 LangChain 的核心理念：**给你一堆乐高积木，你自己拼出想要的形状。** 它提供了极大的自由度，但也要求你对每个组件都有足够的了解才能拼出一个稳定可用的系统。

## LlamaIndex 实现

现在来看同一个任务的 LlamaIndex 实现：

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.1)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(similarity_top_k=3)
response = query_engine.query("公司的退款政策是什么？")

print(response.response)
print("\n--- 引用来源 ---")
for node in response.source_nodes:
    print(f"[{node.score:.3f}] {node.metadata.get('file_name')}: "
          f"{node.text[:80]}...")
```

这段代码只有约 15 行，而且逻辑更加线性——没有链式组装，没有手动格式化，没有 Prompt 模板编写。**LlamaIndex 把 RAG 最佳实践封装成了默认行为**。

## 逐维度的深度对比

### 1. 代码量与上手难度

| 维度 | LangChain | LlamaIndex |
|------|-----------|------------|
| 核心代码行数 | ~40 行 | ~15 行 |
| 导入模块数 | 7 个 | 4 个 |
| 需要手动组装的组件 | 6 个 | 0 个（全自动） |
| 新手出错概率 | 高（容易漏掉某个环节） | 低（默认配置即可工作） |

LlamaIndex 的代码量之所以少得多，是因为它把 LangChain 中需要手动组装的步骤全部内置到了 `VectorStoreIndex` 和 `query_engine` 中。这不是偷懒，而是**将 RAG 领域的最佳实践固化成了框架约定**。

### 2. 数据流的透明度

**LangChain** 的数据流是显式的——你在 chain 表达式中清楚地看到每一步的输入输出：

```python
chain = (
    {"context": retriever | format_docs, ...}
    | prompt          # 输入: dict(context, question) → 输出: ChatPromptValue
    | llm             # 输入: ChatPromptValue → 输出: AIMessage
    | StrOutputParser()  # 输入: AIMessage → 输出: str
)
```

这种显式性的好处是**易于调试**——你可以在任意位置打断点查看中间状态。缺点是**认知负担重**——你需要理解每种数据类型的转换规则。

**LlamaIndex** 的数据流是隐式的——你调用 `query()`，然后得到结果。内部的检索、Prompt 组装、LLM 调用、响应合成全部隐藏在 `QueryEngine` 内部。好处是**使用简单**，坏处是**初学者可能不清楚内部到底发生了什么**（这也是本教程存在的意义——帮你看透黑盒）。

### 3. 检索结果的来源追踪

这是两者在设计理念上差异最大的地方之一。

**LangChain** 中，Retriever 返回的是 `Document` 对象列表，其中包含 `page_content` 和 `metadata`。但问题是，**Document 对象在经过 Chain 传递后，其来源信息很容易丢失**——除非你专门在 Prompt 中要求 LLM 输出来源编号，否则最终得到的只是一个纯文本字符串：

```python
result = chain.invoke("退款政策是什么？")
# result 只是字符串，没有来源信息！
```

如果你想要来源信息，需要额外的工作——要么修改 Prompt 让 LLM 在回复中标注来源编号，要么在 Chain 外部再做一次检索来关联来源。

**LlamaIndex** 则从设计之初就把**来源追踪作为一等公民**。`Response` 对象自带 `source_nodes` 属性，每个 node 都保留了完整的元数据和相关性分数：

```python
response = query_engine.query("退款政策是什么？")
# response.source_nodes 自动包含所有检索到的节点及其来源
for node in response.source_nodes:
    print(node.score, node.metadata["file_name"], node.text[:80])
```

这种设计不是事后添加的功能，而是**贯穿整个框架的数据模型**——从 Document 到 Node 到 Index 到 Response，元数据一路传递，永不丢失。

### 4. 响应合成策略

**LangChain** 的 RAG Chain 本质上只有一种响应合成方式：**把所有检索到的内容塞进 Prompt，让 LLM 一次性生成答案**。这种方式简单粗暴，但当检索内容很多时（比如 `k=10` 或更多），LLM 的输入窗口可能不够用，或者过多无关信息反而干扰了答案质量。

**LlamaIndex** 内置了多种响应合成策略（Response Mode），你可以根据场景选择：

```python
from llama_index.core.response_synthesizers import (
    ResponseMode,
    get_response_synthesizer
)

# 方式一：Refine（迭代精炼）— 默认模式
# 先用第一个 chunk 生成初步答案，然后用后续 chunk 逐步精炼
synthesizer = get_response_synthesizer(
    response_mode=ResponseMode.REFINE
)

# 方式二：Compact and Refine（压缩后精炼）
# 先把所有 chunk 压缩到 token 限制内，再精炼
synthesizer = get_response_synthesizer(
    response_mode=ResponseMode.COMPACT_ACCUMULATE
)

# 方式三：Tree Summarize（树状汇总）
# 先对每个 chunk 生成摘要，再递归合并
synthesizer = get_response_synthesizer(
    response_mode=ResponseMode.TREE_SUMMARIZE
)

# 方式四：Simple Summarize（简单拼接）
# 类似 LangChain 的做法，把所有内容拼在一起
synthesizer = get_response_synthesizer(
    response_mode=ResponseMode.SIMPLE_SUMMARIZE
)

query_engine = index.as_query_engine(
    response_synthesizer=synthesizer,
    similarity_top_k=10  # 可以用更大的 k 了
)
```

不同的响应合成模式适用于不同场景，我们会在第七章深入讲解。这里想强调的是：**LlamaIndex 认为响应合成不是一个 trivial 的问题，它值得被认真对待并提供多种策略。**

### 5. 索引类型的丰富度

**LangChain** 的 RAG 方案本质上只有一种索引：**向量索引（Vector Store）**。虽然你可以搭配不同的 Vector Store 后端（Chroma、Pinecone、FAISS 等），但索引的逻辑都是一样的——把文本转向量，做相似度搜索。

**LlamaIndex** 提供了 6 种原生索引类型，每种针对不同的数据特征和查询模式：

| 索引类型 | 适用场景 | 检索方式 |
|----------|----------|----------|
| `VectorStoreIndex` | 通用语义搜索 | 向量相似度 |
| `ListIndex` | 需要遍历全部内容的场景 | 顺序扫描 |
| `TreeIndex` | 层级化文档（如书籍） | 树形导航 |
| `KeywordTableIndex` | 结构化查询（如"查找所有提到 X 的段落"） | 关键词匹配 |
| `SummaryIndex` | 需要全局概览的场景 | 全文摘要 |
| `GraphIndex` | 实体关系查询（如"A 公司的 CEO 是谁？"） | 图遍历 |

更重要的是，**这些索引可以组合使用**。比如你可以用一个 `VectorStoreIndex` 做初步筛选，再用 `KeywordTableIndex` 在结果中做精确匹配。这种组合能力是 LlamaIndex 区别于 LangChain RAG 的核心竞争力之一。

### 6. 扩展性与定制能力

这一点上 **LangChain 实际上更强**。由于 LangChain 的组件化程度更高，你可以轻松地：

```python
# 在 Chain 中插入自定义处理逻辑
from langchain_core.runnables import RunnableLambda

def my_custom_processor(input_dict):
    input_dict["context"] = enhance_context(input_dict["context"])
    return input_dict

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | RunnableLambda(my_custom_processor)  # 自定义步骤
    | prompt
    | llm
    | StrOutputParser()
)
```

LlamaIndex 也支持定制，但通常是通过**回调（callbacks）、后处理器（postprocessors）、或自定义组件**来实现，灵活性略逊于 LCEL 的链式表达式。不过在实际的 RAG 场景中，LlamaIndex 提供的定制能力已经覆盖了 95% 的需求。

## 一个更复杂的对比案例

为了让对比更有说服力，我们来看一个稍微复杂一点的需求：**除了语义搜索外，还希望支持关键词过滤**（比如"只搜索 PDF 文件中提到的内容"）。

**LangChain 实现：** 需要使用 SelfQueryRetriever 或自定义 Retriever 逻辑，代码量会增加约 20-30 行。

**LlamaIndex 实现：**

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.vector_stores import MetadataFilter, FilterOperator

documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine(
    similarity_top_k=5,
    filters=MetadataFilter(
        key="file_type",
        value="pdf",
        operator=FilterOperator.EQ,
    )
)

response = query_engine.query("退货流程是怎样的？")
```

只需增加一个 `filters` 参数，LlamaIndex 就能在检索阶段自动应用元数据过滤。这种"声明式"的过滤语法比 LangChain 中需要自行编写过滤逻辑要简洁得多。

## 如何选择？

说了这么多差异，最终的选择其实并不难：

**选 LlamaIndex 当：**
- 你的项目**核心是 RAG**——围绕数据检索展开
- 你希望**快速获得高质量的检索结果**，不想花大量时间调参
- 数据来源**多样**（文件、数据库、API 都有）
- 需要**细粒度的来源追踪和引用**
- 团队成员**不全是 AI 工程师**，需要低门槛的数据接入方案

**选 LangChain 当：**
- 你的项目**核心是 Agent 编排**——RAG 只是其中一个工具
- 你需要**极高的灵活性和可控性**
- 项目涉及**多种外部服务的集成**（不只是 RAG）
- 团队**熟悉 LCEL** 且喜欢链式编程的风格

**两者结合使用：** 这是实践中最常见的做法。用 LlamaIndex 构建 RAG 检索层（利用其强大的数据连接和索引能力），然后将检索结果作为 Tool 传给 LangChain Agent 进行编排和推理。LlamaIndex 甚至提供了专门的集成接口来支持这种用法：

```python
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.tools import tool
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

@tool
def knowledge_base_search(query: str) -> str:
    """从公司知识库中搜索相关信息"""
    response = query_engine.query(query)
    return response.response

agent = create_tool_calling_agent(llm, [knowledge_base_search], prompt)
executor = AgentExecutor(agent=agent, tools=[knowledge_base_search], verbose=True)
```

这样你就同时获得了两者的优势：LlamaIndex 的专业 RAG 能力 + LangChain 的强大 Agent 编排能力。

## 总结

如果要用一句话概括两者的关系：**LangChain 给了你一套万能乐高，LlamaIndex 给你了一套专为 RAG 设计的专业工具箱。** 乐高的乐趣在于自由创造，专业工具箱的优势在于开箱即用且效果可靠。在实际项目中，最好的策略往往是**了解两者，然后根据场景选择或组合使用**。接下来的章节，我们将深入探索 LlamaIndex 的各个核心组件，你会发现它在 RAG 领域的设计确实称得上"专业"二字。
