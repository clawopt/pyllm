---
title: 检索器：相似度搜索与高级检索策略
description: 基础相似度搜索、MMR 多样性搜索、上下文压缩检索、metadata 过滤、Retriever 统一接口
---
# 检索器：相似度搜索与高级检索策略

前三节我们完成了 RAG 的数据准备阶段——文档加载、分块、向量化、存储。现在进入在线查询阶段的核心环节：**如何从向量库中精准地找到与用户问题最相关的那几个文档块？**

这一节我们将学习从基础到高级的各种检索策略。

## 基础：相似度搜索

最直接的检索方式——把用户的问题向量化，然后在向量空间中找距离最近的 top-k 个文档：

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# 基础相似度搜索
results = vectorstore.similarity_search("Python 装饰器的用法", k=3)

for i, doc in enumerate(results, 1):
    print(f"\n--- 结果 {i} ---")
    print(f"来源: {doc.metadata.get('source', '?')}")
    print(doc.page_content[:120] + "...")
```

### 带分数的搜索

如果你需要知道每个结果的相关性分数（用于过滤低质量结果）：

```python
results_with_scores = vectorstore.similarity_search_with_score(
    "Python GIL 机制",
    k=3
)

for doc, score in results_with_scores:
    similarity = 1 - score   # Chroma 返回的是距离，越小越近
    print(f"[{similarity:.2f}] {doc.page_content[:60]}...")
```

输出示例：

```
[0.89] Python 的 GIL（全局解释器锁）使得同一时刻只有一个线程...
[0.72] 在多线程编程中，由于 GIL 的存在，Python 无法利用多核...
[0.65] Go 语言的 goroutine 可以充分利用多核 CPU 进行并发计算...
```

有了分数后就可以设置阈值过滤低质量结果：

```python
MIN_SCORE = 0.6

filtered = [
    (doc, score) for doc, score in results_with_scores
    if (1 - score) >= MIN_SCORE
]

if not filtered:
    print("⚠️ 未找到足够相关的文档")
else:
    for doc, _ in filtered:
        print(doc.page_content)
```

## Retriever：统一的检索接口

直接调用 `vectorstore.similarity_search()` 是可行的，但在 LangChain 的标准 Chain 架构中，推荐使用 **Retriever** 接口。每个 Vector Store 都可以通过 `.as_retriever()` 创建一个 Retriever：

```python
retriever = vectorstore.as_retriever(
    search_type="similarity",      # 搜索类型
    k=3                            # 返回数量
)

# 用法和直接调用类似，但可以接入 Chain 管道
results = retriever.invoke("什么是 RAG？")
for doc in results:
    print(doc.page_content[:80])
```

`.as_retriever()` 支持的参数：

| 参数 | 说明 | 示例值 |
|------|------|--------|
| `search_type` | 搜索策略 | `"similarity"`, `"mmr"` |
| `k` | 返回结果数 | `3`, `5` |
| `search_kwargs` | 额外参数 | `{"score_threshold": 0.5}` |

## 高级策略一：MMR — 增加结果多样性

默认的相似度搜索返回的是**与查询最相似**的结果，但这些结果之间可能高度重复。**MMR（Maximal Marginal Relevance）** 在相关性和多样性之间做平衡：

```python
retriever_mmr = vectorstore.as_retriever(
    search_type="mmr",
    k=3,
    search_kwargs={"lambda_mult": 0.7}   # 0=纯多样性, 1=纯相关性
)
```

`lambda_mult` 控制平衡点：
- **接近 1**（如 0.9）：优先保证每条结果都与查询高度相关，允许重复
- **接近 0**（如 0.3）：优先保证各条结果之间差异大，可能牺牲部分相关性
- **0.7**：常用起点值

适用场景：
- 当知识库中有大量重复或高度相似的内容时
- 当你需要覆盖多个不同方面的信息时（如"总结一下这篇文档的所有要点"）

## 高级策略二：Metadata 过滤

很多时候你需要在特定范围内搜索——比如只在某个分类、某个来源或某段时间的文档中查找：

```python
# 创建带丰富 metadata 的向量库
from langchain_core.documents import Document

docs = [
    Document(page_content="Python 的 GIL 限制多线程性能", metadata={"category": "language", "topic": "concurrency"}),
    Document(page_content="Java 的线程模型基于 OS 原生线程", metadata={"category": "language", "topic": "concurrency"}),
    Document(page_content="Docker 容器化部署最佳实践", metadata={"category": "devops", "topic": "deployment"}),
    Document(page_content="Kubernetes 集群管理入门", metadata={"category": "devops", "topic": "deployment"}),
]

vectorstore = Chroma.from_documents(docs, embeddings)

# 只在 language 类别中搜索
retriever_lang = vectorstore.as_retriever(
    search_kwargs={
        "filter": {"category": "language"}
    }
)

results = retriever_lang.invoke("并发编程")
# 只会返回 language 类别的结果，不会出现 devops 的内容
```

这在企业知识库中特别有用——用户选择了"HR 政策"分类，就只在这个分类内搜索。

## 高级策略三：上下文压缩检索

基础检索返回的文本块可能包含大量无关信息。**Contextual Compression Retriever** 先用廉价的方法做粗筛，再用 LLM 对结果进行精炼和压缩：

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# 基础检索器（先用便宜的方式粗筛）
base_retriever = vectorstore.as_retriever(search_type="mmr", k=5)

# 压缩器（用 LLM 从每个块中提取最相关的片段）
compressor = LLMChainExtractor.from_llm(ChatOpenAI(model="gpt-4o-mini"))

# 组合成压缩检索器
compressed_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

results = compressed_retriever.invoke("GIL 如何影响多线程性能？")
```

压缩后的结果通常比原始块更短更聚焦，减少了送入提示词的噪音。代价是每次检索多一次 LLM 调用，增加了延迟和成本。

## 四种检索策略对比

| 策略 | 速度 | 精准度 | 成本 | 适用场景 |
|------|------|--------|------|---------|
| **Similarity** | ⚡ 最快 | ★★★ | 低 | 大多数场景的默认选择 |
| **MMR** | ⚡ 快 | ★★☆ | 低 | 有重复内容 / 需要覆盖面广 |
| **Metadata Filter** | ⚡ 快 | ★★★ | 低 | 已知分类/来源范围 |
| **Contextual Compression** | 🐢 较慢 | ★★★★ | 中 | 需要高精度 / token 预算紧张 |

## 检索质量调优指南

在实际项目中，检索质量直接影响 RAG 的最终效果。以下是几个关键的调优方向：

**方向一：调整 k 值**

k 值太小 → 可能遗漏关键信息；k 值太大 → 引入过多噪音。

```python
# 尝试不同的 k 值，观察答案质量的变化
for k in [2, 3, 5]:
    retriever = vectorstore.as_retriever(k=k)
    docs = retriever.invoke(user_question)
    total_chars = sum(len(d.page_content) for d in docs)
    print(f"k={k}: {len(docs)} 个块, ~{total_chars} 字符")
```

经验规则：
- 简单事实问答 → k=2~3
- 需要多角度分析 → k=5
- 复杂推理任务 → k=5~10

**方向二：优化分块策略**

如果检索结果总是不相关，大概率是分块出了问题：
- 切得太碎 → 一个完整概念被拆散在多个块里 → 增大 chunk_size 或 overlap
- 切得太粗糙 → 单个块包含太多主题 → 减小 chunk_size
- 在语义边界处切断 → 关键信息被截断 → 使用 MarkdownHeaderTextSplitter 或增大 overlap

**方向三：混合检索**

纯向量搜索有时不如关键词匹配精确（比如搜专有名词、代码标识符）。**混合检索（Hybrid Search）** 同时使用向量相似度和关键词匹配：

```python
# 某些向量库原生支持混合搜索
# 以 Chroma 为例，可以通过 query 方式实现近似的关键词匹配
results = vectorstore.search(
    query_texts=["Python GIL threading"],
    n_results=3,
    where={"$or": [
        {"page_content": {"$contains": "GIL"}},
        {"page_content": {"$contains": "threading"}}
    ]}
)
```

到这里，RAG 的全部技术组件都已经介绍完毕——从文档加载、分块、向量化存储，到各种检索策略。下一节我们将把它们全部串联起来，搭建一个完整的、可运行的 RAG 问答系统。
