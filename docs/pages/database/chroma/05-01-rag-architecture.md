# 5.1 RAG 架构中的角色定位

> **Chroma 在 RAG 中不是主角，但它是不可或缺的"记忆中枢"——没有它，LLM 就是一个没有图书馆的学者**

---

## 这一节在讲什么？

前面四章我们学了 Chroma 的基本操作、CRUD、Embedding、查询和过滤——这些都是工具层面的知识。但工具只有在具体场景中才能发挥价值，而 Chroma 最核心的应用场景就是 RAG（Retrieval-Augmented Generation，检索增强生成）。这一节我们要从架构层面理解 Chroma 在 RAG 系统中扮演什么角色、它与其他组件如何协作、以及它的能力边界在哪里。这些理解不仅帮助你正确使用 Chroma，也是面试中"RAG 架构怎么设计"这类系统级问题的回答基础。

---

## RAG 的全链路拆解

RAG 的核心思想很简单：LLM 的知识是有限的（只包含训练数据中的信息），但用户的问题可能涉及任何领域的私有数据。与其把所有知识都塞进 LLM 的参数里（微调成本高、更新慢），不如在每次提问时，先从外部知识库中检索相关信息，再把检索到的信息作为上下文喂给 LLM，让它基于这些信息生成回答。

完整的 RAG 链路可以分为五个阶段：

```
┌─────────────────────────────────────────────────────────────────────┐
│  RAG 全链路                                                         │
│                                                                     │
│  ① 用户提问                                                         │
│     "我们公司的退款政策是什么？"                                      │
│     ↓                                                               │
│  ② Query Understanding（查询理解）                                   │
│     - 改写/扩展用户的原始问题                                         │
│     - 提取关键实体和意图                                              │
│     - 生成适合检索的 query                                           │
│     ↓                                                               │
│  ③ Retrieval（检索）← Chroma 在这里！                                │
│     - Query → Embedding Function → 查询向量                          │
│     - 查询向量 → HNSW 向量搜索 → Top-K 候选文档                       │
│     - 可选：Where 过滤 → 缩小候选范围                                 │
│     - 可选：Re-ranking → 精排 Top-K                                  │
│     ↓                                                               │
│  ④ Context Assembly（上下文组装）                                    │
│     - 将检索到的文档拼接成 prompt                                     │
│     - 添加系统指令（如"基于以下信息回答问题"）                         │
│     - 控制总长度不超过 LLM 上下文窗口                                  │
│     ↓                                                               │
│  ⑤ Generation（生成）                                               │
│     - LLM 基于 prompt + 检索到的上下文生成回答                        │
│     - 可选：引用来源、置信度评估                                      │
│     ↓                                                               │
│  回答："根据公司政策，购买后7天内可无条件退款..."                      │
└─────────────────────────────────────────────────────────────────────┘
```

Chroma 负责的是第③阶段——检索（Retrieval）。它是整个 RAG 系统的"记忆中枢"，存储着所有可检索的知识，并在用户提问时快速返回最相关的文档片段。

---

## Chroma 在 RAG 中的具体职责

### 职责一：知识存储

RAG 系统需要一个地方来存储结构化的知识——产品文档、FAQ、技术手册、用户历史等。Chroma 的 Collection 就是这些知识的容器。每条知识以 Document 的形式存储，包含原文（document）、向量（embedding）、元数据（metadata）和唯一标识（id）。

```python
# 知识入库：把文档切分后存入 Chroma
def ingest_knowledge(collection, documents, source, category):
    """将文档入库到 Chroma"""
    chunks = recursive_chunk(documents, chunk_size=600, overlap=80)
    for i, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk],
            ids=[f"{source}_chunk_{i}"],
            metadatas=[{
                "source": source,
                "category": category,
                "chunk_index": i,
                "total_chunks": len(chunks)
            }]
        )
```

### 职责二：语义检索

当用户提问时，Chroma 负责从知识库中快速找到最相关的文档片段。这个过程包括：将用户的查询文本编码为向量、在 HNSW 索引中搜索最近邻、返回按相似度排序的 top-K 结果。

```python
# 语义检索：从 Chroma 中找到最相关的文档
def retrieve(collection, query, category=None, n_results=5):
    """从 Chroma 检索相关文档"""
    where = {"category": category} if category else None
    results = collection.query(
        query_texts=[query],
        where=where,
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    return results
```

### 职责三：结构化过滤

通过 metadata 的 where 过滤，Chroma 可以在语义搜索的基础上叠加结构化约束——比如只搜索某个类别的文档、只搜索最新版本、只搜索特定语言的内容。这种"语义+结构"的混合查询模式是 RAG 系统中提高检索精度的关键手段。

---

## Chroma 不负责什么

理解 Chroma 的能力边界同样重要。以下是 Chroma 在 RAG 架构中**不负责**的部分：

| 不负责的部分 | 由谁负责 | 原因 |
|-------------|---------|------|
| 查询改写/扩展 | LLM 或专用模型 | 需要理解用户意图，不是向量搜索的职责 |
| 文档切分 | 应用层（ingest pipeline） | 切分策略取决于业务需求 |
| Prompt 组装 | 应用层 | 需要考虑 LLM 的上下文窗口和指令格式 |
| 文本生成 | LLM | Chroma 是数据库，不是生成模型 |
| Re-ranking | Cross-Encoder 模型 | 需要更精细的语义匹配 |
| 对话历史管理 | 应用层 + Chroma Memory Collection | 需要区分短期和长期记忆 |

Chroma 的定位是**专注做好检索这一件事**。它不试图成为端到端的 RAG 框架——那是 LangChain、LlamaIndex 等工具的职责。Chroma 的优势在于它的 API 简洁、零配置启动、Python 原生集成，使得你可以快速搭建一个可工作的检索层，然后根据需要逐步添加查询改写、Re-ranking、对话管理等高级功能。

---

## 检索质量如何影响生成质量

RAG 系统有一个根本性的特征：**检索是生成的上限**。如果检索阶段没有找到正确的文档，LLM 再强也无法生成正确的回答——它要么编造答案（幻觉），要么坦白说"我不知道"。反过来，如果检索到了正确的文档，即使 LLM 能力一般，也能基于文档内容生成合理的回答。

```
┌─────────────────────────────────────────────────────────────┐
│  检索质量与生成质量的关系                                     │
│                                                             │
│  检索正确 + LLM 强 → ✅ 高质量回答                           │
│  检索正确 + LLM 弱 → ⚠️ 回答可能不够流畅，但信息正确          │
│  检索错误 + LLM 强 → ❌ 流畅但可能包含幻觉（更危险！）        │
│  检索错误 + LLM 弱 → ❌ 回答错误或"我不知道"                 │
│                                                             │
│  结论：检索质量是 RAG 系统的地基                              │
│  → 优化 RAG 的第一步永远是优化检索，而不是换更大的 LLM        │
└─────────────────────────────────────────────────────────────┘
```

这意味着在 RAG 系统的优化中，你应该把大部分精力放在检索层——选择合适的 embedding 模型、设计合理的 metadata schema、调整切分参数、引入 re-ranking。这些优化对最终效果的影响，往往比从 GPT-3.5 升级到 GPT-4 更大。

---

## 检索与生成的接口契约

Chroma（检索层）和 LLM（生成层）之间的接口契约是：**检索到的文档必须包含足够的信息来回答用户的问题**。这个契约对 Chroma 的使用方式有几个直接影响：

1. **切分粒度要合适**：每个 chunk 必须包含足够的上下文让 LLM 理解其含义。如果切得太碎，LLM 看到的只是孤立的片段，无法生成连贯的回答。

2. **Metadata 要完整**：检索结果中的 metadata（特别是 source 和 chunk_index）应该传递给 LLM，让它能在回答中引用来源——"根据《用户手册》第 15 页的内容..."比一个没有出处的回答更可信。

3. **Top-K 数量要适中**：太少可能遗漏关键信息，太多会引入噪声并超出 LLM 的上下文窗口。3~5 条是大多数场景的最佳范围。

4. **距离阈值要设定**：不是所有检索结果都值得传给 LLM。如果最相似的文档距离也很远（比如 cosine > 1.5），说明知识库中可能没有相关信息，此时应该让 LLM 回答"我没有找到相关信息"而不是强行生成。

```python
def retrieve_with_threshold(collection, query, max_distance=1.0, n_results=5):
    """带距离阈值的检索：过滤掉距离过远的结果"""
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )

    filtered = []
    for i in range(len(results['ids'][0])):
        if results['distances'][0][i] <= max_distance:
            filtered.append({
                "document": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "distance": results['distances'][0][i]
            })

    return filtered
```

---

## 完整的 RAG 查询流程示例

让我们把前面学的所有知识整合起来，实现一个完整的 RAG 查询流程：

```python
import chromadb
from chromadb.utils import embedding_functions

class SimpleRAG:
    """最简 RAG 实现：检索 → 组装 → 生成"""

    def __init__(self, collection_name="rag_kb", persist_dir="./rag_db"):
        self.client = chromadb.Client(settings=chromadb.Settings(
            is_persistent=True,
            persist_directory=persist_dir
        ))

        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="paraphrase-multilingual-MiniLM-L12-v2"
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=ef
        )

    def ingest(self, text: str, source: str, category: str = "general"):
        """入库文档"""
        chunks = self._chunk(text)
        ids = [f"{source}_chunk_{i}" for i in range(len(chunks))]
        metas = [{
            "source": source,
            "category": category,
            "chunk_index": i,
            "total_chunks": len(chunks)
        } for i in range(len(chunks))]

        self.collection.upsert(documents=chunks, ids=ids, metadatas=metas)
        print(f"✅ 入库 {len(chunks)} 个 chunk (来源: {source})")

    def query(self, question: str, category: str = None, n_results: int = 3,
              max_distance: float = 1.2):
        """RAG 查询：检索 → 组装 prompt → 返回上下文"""
        where = {"category": category} if category else None

        results = self.collection.query(
            query_texts=[question],
            where=where,
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )

        # 过滤距离过远的结果
        context_parts = []
        sources = []
        for i in range(len(results['ids'][0])):
            dist = results['distances'][0][i]
            if dist <= max_distance:
                doc = results['documents'][0][i]
                meta = results['metadatas'][0][i]
                context_parts.append(doc)
                sources.append(f"{meta['source']} (chunk {meta['chunk_index']})")

        if not context_parts:
            return {
                "answer": None,
                "context": "",
                "sources": [],
                "message": "未找到相关信息"
            }

        context = "\n\n".join(context_parts)

        # 组装 prompt（这里只返回上下文，实际生成由 LLM 完成）
        prompt = f"""基于以下参考信息回答用户的问题。如果参考信息中没有相关内容，请回答"我没有找到相关信息"。

参考信息：
{context}

用户问题：{question}

回答："""

        return {
            "prompt": prompt,
            "context": context,
            "sources": sources,
            "n_retrieved": len(results['ids'][0]),
            "n_filtered": len(context_parts)
        }

    def _chunk(self, text, chunk_size=600, overlap=80):
        """递归字符级切分"""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk.strip())
            start = end - overlap
        return chunks if chunks else [text]


# ====== 使用示例 ======
rag = SimpleRAG()

# 入库知识
rag.ingest(
    "退款政策：购买后7天内可无条件退款。退款流程：1.在订单页面点击申请退款；2.填写退款原因；3.等待3-5个工作日审核。注意：已拆封的数码产品不支持无理由退款。",
    source="user_manual_v2",
    category="after_sales"
)

rag.ingest(
    "安装指南：1.下载安装包；2.双击运行安装程序；3.选择安装路径；4.点击安装并等待完成。系统要求：Windows 10及以上，8GB内存。",
    source="install_guide",
    category="technical"
)

# RAG 查询
result = rag.query("数码产品能退款吗", category="after_sales")
print(f"\n检索到 {result['n_filtered']} 条相关文档")
print(f"来源: {result['sources']}")
print(f"\n--- 组装的 Prompt ---\n{result['prompt'][:300]}...")
```

输出：

```
✅ 入库 2 个 chunk (来源: user_manual_v2)
✅ 入库 1 个 chunk (来源: install_guide)

检索到 1 条相关文档
来源: ['user_manual_v2 (chunk 0)']

--- 组装的 Prompt ---
基于以下参考信息回答用户的问题。如果参考信息中没有相关内容，请回答"我没有找到相关信息"。

参考信息：
退款政策：购买后7天内可无条件退款。退款流程：1.在订单页面点击申请退款；2.填写退款原因；3.等待3-5个工作日审核。注意：已拆封的数码产品不支持无理由退款。

用户问题：数码产品能退款吗...
```

---

## 常见误区

### 误区 1：把 Chroma 当成端到端的 RAG 解决方案

Chroma 只负责检索，不负责查询理解、prompt 组装和文本生成。如果你需要完整的 RAG 能力，应该把 Chroma 与 LLM 结合使用，或者使用 LangChain/LlamaIndex 等框架来编排完整的 RAG 流程。

### 误区 2：检索结果越多越好

过多的检索结果会引入噪声，降低 LLM 的生成质量。3~5 条高质量的相关文档比 20 条半相关的文档效果好得多。设定距离阈值（max_distance）过滤掉不相关的结果，是提高 RAG 质量的有效手段。

### 误区 3：忽略检索失败的信号

当所有检索结果的距离都很远时（比如 cosine > 1.5），说明知识库中没有与用户问题相关的信息。此时应该让 LLM 回答"我没有找到相关信息"，而不是强行基于不相关的文档生成回答——后者更容易产生幻觉。

---

## 本章小结

Chroma 在 RAG 架构中扮演"记忆中枢"的角色，负责知识存储、语义检索和结构化过滤三个核心职责。核心要点回顾：第一，RAG 的五阶段链路是"提问→查询理解→检索→上下文组装→生成"，Chroma 负责第③阶段；第二，检索是生成的上限——优化 RAG 的第一步永远是优化检索，而不是换更大的 LLM；第三，检索与生成之间的接口契约是"检索到的文档必须包含足够的信息来回答问题"，这要求切分粒度合适、metadata 完整、top-K 适中；第四，设定距离阈值过滤不相关结果，避免 LLM 基于错误信息产生幻觉；第五，Chroma 不负责查询改写、prompt 组装和文本生成，这些由应用层或 LLM 框架处理。

下一节我们将实现一个端到端的 PDF 文档问答 Demo——从 PDF 加载到切分入库到检索生成，完整走通 RAG 的全流程。
