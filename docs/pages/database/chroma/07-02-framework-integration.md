# 7.2 与 LLM Framework 的集成

> **Chroma 是引擎，LLM Framework 是方向盘——两者结合才能开动 RAG 这辆车**

---

## 这一节在讲什么？

在前面的章节中，我们一直是直接使用 Chroma 的 Python API 来构建 RAG 系统——手动切分文档、手动检索、手动组装 prompt、手动调用 LLM。这种方式灵活但繁琐，特别是当你需要添加查询改写、Re-ranking、对话记忆等高级功能时，代码量会急剧膨胀。LLM Framework（如 LangChain、LlamaIndex、Haystack）就是为了解决这个问题而生的——它们提供了预构建的 RAG 组件和编排能力，让你用更少的代码实现更完整的功能。这一节我们要讲清楚 Chroma 如何与三大主流 LLM Framework 集成，各自的优劣势，以及什么时候该用框架、什么时候该用原生 API。

---

## LangChain + Chroma

LangChain 是目前最流行的 LLM 应用开发框架，它的 Chroma 集成是最成熟的。

### 基础用法：一行入库

```python
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 配置 embedding
embeddings = SentenceTransformerEmbeddings(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)

# 方式 1：从文本直接创建向量库
vectorstore = Chroma.from_texts(
    texts=["退款政策：购买后7天内可无条件退款", "安装指南：下载安装包并运行"],
    embedding=embeddings,
    collection_name="langchain_demo",
    persist_directory="./langchain_chroma"
)

# 方式 2：从文档加载器创建
loader = TextLoader("./data/document.txt")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=80
)
chunks = splitter.split_documents(docs)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="langchain_docs",
    persist_directory="./langchain_chroma"
)

# 检索
results = vectorstore.similarity_search(
    query="如何退款",
    k=3
)
for doc in results:
    print(f"  {doc.page_content[:60]}... | {doc.metadata}")
```

### 与 RetrievalQA Chain 集成

LangChain 的 `RetrievalQA` Chain 把"检索 → 组装 prompt → LLM 生成"封装成一个调用：

```python
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# 初始化
embeddings = SentenceTransformerEmbeddings(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)

vectorstore = Chroma(
    collection_name="langchain_demo",
    embedding_function=embeddings,
    persist_directory="./langchain_chroma"
)

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)

# 创建 QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # "stuff" = 把所有检索结果塞进一个 prompt
    retriever=vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    ),
    return_source_documents=True
)

# 一行完成 RAG
result = qa_chain.invoke({"query": "退款政策是什么？"})
print(f"回答: {result['result']}")
print(f"来源: {[doc.metadata for doc in result['source_documents']]}")
```

### ConversationalRetrievalChain：带对话记忆的 RAG

```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

conv_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    memory=memory
)

# 多轮对话
result1 = conv_chain.invoke({"question": "退款政策是什么？"})
print(f"回答1: {result1['answer']}")

result2 = conv_chain.invoke({"question": "它有时间限制吗？"})  # "它"指代"退款政策"
print(f"回答2: {result2['answer']}")
```

---

## LlamaIndex + Chroma

LlamaIndex 专注于"数据索引"这一层，它的 Chroma 集成更注重细粒度的节点管理和索引策略。

### 基础用法

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# 初始化 Chroma
chroma_client = chromadb.Client(settings=chromadb.Settings(
    is_persistent=True,
    persist_directory="./llamaindex_chroma"
))
chroma_collection = chroma_client.get_or_create_collection("llamaindex_demo")

# 创建 LlamaIndex 的 VectorStore 包装
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# 加载文档
documents = SimpleDirectoryReader("./data").load_data()

# 创建索引（自动切分 + embedding + 入库）
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context
)

# 查询
query_engine = index.as_query_engine(similarity_top_k=3)
response = query_engine.query("退款政策是什么？")
print(f"回答: {response}")
print(f"来源: {[n.metadata for n in response.source_nodes]}")
```

### LlamaIndex 的优势：节点级别的细粒度控制

LlamaIndex 允许你对每个节点（chunk）做精细的配置——不同的切分策略、不同的 embedding 模型、不同的 metadata：

```python
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import SummaryExtractor

# 自定义切分器
splitter = SentenceSplitter(
    chunk_size=600,
    chunk_overlap=80
)

# 自定义 metadata 提取器
extractors = [
    SummaryExtractor()  # 自动为每个 chunk 生成摘要
]

# 构建索引
nodes = splitter.get_nodes_from_documents(documents)
index = VectorStoreIndex(
    nodes,
    storage_context=storage_context
)
```

---

## Haystack + Chroma

Haystack 是 deepset 开发的 RAG 框架，它的设计哲学是"管道（Pipeline）"——每个处理步骤是一个独立的组件，通过管道串联起来。

### 基础用法

```python
from haystack import Pipeline
from haystack.components.converters import PyPDFToDocument
from haystack.components.writers import DocumentWriter
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever

# 初始化 Chroma Document Store
document_store = ChromaDocumentStore(
    collection_name="haystack_demo",
    persist_path="./haystack_chroma"
)

# 构建入库 Pipeline
indexing_pipeline = Pipeline()
indexing_pipeline.add_component("converter", PyPDFToDocument())
indexing_pipeline.add_component(
    "embedder",
    SentenceTransformersDocumentEmbedder(model="paraphrase-multilingual-MiniLM-L12-v2")
)
indexing_pipeline.add_component(
    "writer",
    DocumentWriter(document_store=document_store, policy=DuplicatePolicy.OVERWRITE)
)

indexing_pipeline.connect("converter", "embedder")
indexing_pipeline.connect("embedder", "writer")

# 运行入库
indexing_pipeline.run({"converter": {"sources": ["./data/document.pdf"]}})

# 构建查询 Pipeline
query_pipeline = Pipeline()
query_pipeline.add_component(
    "text_embedder",
    SentenceTransformersTextEmbedder(model="paraphrase-multilingual-MiniLM-L12-v2")
)
query_pipeline.add_component(
    "retriever",
    ChromaEmbeddingRetriever(document_store=document_store)
)

query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")

# 查询
result = query_pipeline.run({"text_embedder": {"text": "退款政策是什么？"}})
for doc in result["retriever"]["documents"]:
    print(f"  [{doc.score:.4f}] {doc.content[:60]}...")
```

---

## 三大框架对比

| 维度 | LangChain | LlamaIndex | Haystack |
|------|-----------|------------|----------|
| **设计哲学** | 链式编排 | 数据索引优先 | 管道组件化 |
| **Chroma 集成成熟度** | ⭐⭐⭐ 最成熟 | ⭐⭐ 良好 | ⭐⭐ 良好 |
| **上手难度** | ⭐⭐ 中等 | ⭐⭐ 中等 | ⭐⭐⭐ 较高 |
| **灵活性** | ⭐⭐⭐ 最高 | ⭐⭐ 中等 | ⭐⭐ 中等 |
| **RAG 开箱即用** | ✅ RetrievalQA | ✅ as_query_engine | ✅ Pipeline |
| **对话记忆** | ✅ 内置 | ✅ 内置 | ✅ 需组装 |
| **适合场景** | 快速原型、灵活定制 | 文档密集型应用 | 生产级 Pipeline |
| **社区规模** | 最大 | 大 | 中等 |

---

## 什么时候用框架，什么时候用原生 API

```
┌─────────────────────────────────────────────────────────────────┐
│  框架 vs 原生 API 的选择指南                                     │
│                                                                 │
│  用框架（LangChain/LlamaIndex/Haystack）当：                     │
│  ✅ 需要快速搭建端到端 RAG 系统                                  │
│  ✅ 需要对话记忆、查询改写等高级功能                              │
│  ✅ 团队中有人已经熟悉某个框架                                   │
│  ✅ 需要与多种 LLM/工具集成（不只是 Chroma）                     │
│                                                                 │
│  用原生 Chroma API 当：                                          │
│  ✅ 只需要向量检索能力，不需要完整的 RAG 编排                     │
│  ✅ 需要极致的性能控制（框架有抽象开销）                          │
│  ✅ 需要深度定制检索逻辑（框架的抽象可能不够灵活）                │
│  ✅ 项目简单，不想引入框架的依赖和复杂度                          │
│  ✅ 想要理解底层原理（框架隐藏了太多细节）                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 常见误区

### 误区 1：用了框架就不需要理解 Chroma 的原理

框架封装了 Chroma 的 API，但没有封装 Chroma 的原理。如果你不理解 distance metric、metadata 过滤、embedding 维度这些底层概念，用框架也会踩同样的坑——只是踩坑的方式从"API 调用错误"变成了"配置参数错误"。

### 误区 2：LangChain 是唯一选择

LangChain 虽然最流行，但不是唯一选择。LlamaIndex 在文档索引方面更专业，Haystack 在 Pipeline 编排方面更灵活。根据你的具体需求选择最合适的框架。

### 误区 3：框架一定比原生 API 慢

框架的抽象开销通常很小（< 1ms），对大多数应用来说可以忽略。真正影响性能的是 embedding 计算、HNSW 搜索和 LLM 推理，这些无论用不用框架都一样。

---

## 本章小结

LLM Framework 与 Chroma 的集成让 RAG 开发更高效。核心要点回顾：第一，LangChain 的 Chroma 集成最成熟，`Chroma.from_texts()` 一行入库，`RetrievalQA` 一行完成 RAG；第二，LlamaIndex 注重节点级别的细粒度控制，适合文档密集型应用；第三，Haystack 的管道设计适合构建生产级 RAG Pipeline；第四，框架适合快速搭建和需要高级功能的场景，原生 API 适合需要极致控制和深度定制的场景；第五，用框架不代表不需要理解 Chroma 的底层原理——框架封装的是 API，不是知识。

下一节也是本教程的最后一节，我们将讨论 Chroma 的局限性及替代方案——什么时候该继续用 Chroma，什么时候该考虑迁移。
