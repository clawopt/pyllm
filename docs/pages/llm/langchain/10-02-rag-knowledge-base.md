---
title: 利用 RAG 加载产品知识库
description: 构建客服专用的 RAG 管线、知识库文档设计、检索质量优化策略、端到端问答链实现
---
# 利用 RAG 加载产品知识库

上一节我们完成了需求分析和技术选型，确定了系统的整体架构。现在开始动手写第一块积木——**RAG 问答链**。这是整个智能客服系统的核心：用户问产品相关问题，系统从知识库中检索相关信息，然后基于这些信息生成准确回复。

## 客服场景的 RAG 特殊性

我们在第 4 章已经详细学过 RAG 的原理和实现。但客服场景的 RAG 有一些特殊的挑战，值得单独讨论：

**第一，答案必须精确**。通用 RAG 场景（比如"总结这篇文章"）允许一定程度的概括和发挥，但客服场景不行——"免费版支持 5 个成员"不能回答成"大约五六个"，"专业版月费 99 元"不能说成"大概一百块"。**精确性是客服 RAG 的生命线**。

**第二，知识库结构化程度高**。客服的知识来源通常是产品手册、FAQ 文档、定价表等——这些文档本身就有清晰的结构（标题、列表、表格）。如果分块策略不合理，很容易把一个完整的条目切成两半，导致检索时只能拿到一半信息。

**第三，需要处理"我不知道"的情况**。当用户的问题在知识库中找不到相关内容时，系统应该诚实地说"这个问题我暂时无法回答"，而不是硬着头皮编造一个答案。这需要在 prompt 层面做约束。

**第四，多语言/多版本考虑**。如果你的产品面向全球市场，同一份知识库可能有中文版、英文版、日文版；或者同一个功能在不同版本的产品中有不同的说明。RAG 需要在 metadata 中记录这些信息，在检索时做过滤。

## 设计产品知识库

在写代码之前，我们先来设计知识库的内容。对于 CloudDesk 这个虚拟产品，我们需要准备以下几类文档：

### 知识库目录结构

```
knowledge_base/
├── pricing.md          # 定价方案（最常被问到的）
├── features.md         # 功能介绍与对比
├── faq.md              # 常见问题汇总
├── policies.md         # 退款政策、服务条款等
└── getting-started.md  # 新手入门指南
```

让我们看看其中最关键的 `pricing.md` 应该怎么组织——它的质量直接影响定价相关问题的回答准确度：

```markdown
# CloudDesk 定价方案

## 免费版 (Free)

- **团队成员**: 最多 5 人
- **存储空间**: 2 GB
- **项目数量**: 最多 3 个
- **API 调用**: 每月 1,000 次
- **技术支持**: 社区论坛
- **价格**: 永久免费

## 专业版 (Professional) - ¥99/月

- **团队成员**: 无限制
- **存储空间**: 100 GB
- **项目数量**: 无限制
- **API 调用**: 每月 50,000 次
- **技术支持**: 邮件支持（24 小时内响应）
- **额外功能**: 高级权限管理、审计日志、SSO 单点登录
- **支持升级**: 可随时从免费版升级

## 企业版 (Enterprise) - 定制报价

- 包含专业版所有功能
- **存储空间**: 1 TB 起
- **专属客户成功经理**
- **SLA 服务等级协议**: 99.9% 可用性保证
- **定制集成**: 支持 API 定制开发
- **技术支持**: 7×24 专属热线 + 在线客服
- **联系销售**: sales@cloud desk.example.com

## 常见定价问题

### 可以随时升降级吗？
可以。升级立即生效，按剩余天数比例计费。降级在下个计费周期生效。

### 支持哪些支付方式？
支持支付宝、微信支付、银行转账（企业版）。不支持现金。

### 有学生优惠吗？
有。使用 .edu 邮箱注册可享受专业版 5 折优惠。
```

注意这个文档的设计细节：**每个定价层级用清晰的标题分隔**，关键数字用粗体标注，常见问题放在同文件底部。这种结构化的写法不是偶然的——它直接影响后面文本分割的效果。

### 为什么文档质量比算法更重要

这是 RAG 领域的一条铁律：**Garbage In, Garbage Out**。如果你的知识库文档写得乱七八糟，再先进的检索算法也救不回来。

好的客服知识库文档应该满足：
- **一条信息只在一个地方出现**（避免矛盾）
- **数值信息明确具体**（不用"大概""左右"等模糊词）
- **保持更新**（产品改了价，文档要同步改）
- **用用户会用的语言写**（不要用内部术语）

## 构建 RAG 管线

现在我们来写代码，把上面的知识库文档加载、分块、向量化、构建成可检索的 RAG 管线。

### 第一步：加载文档

```python
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

def load_knowledge_base(kb_dir: str = "./knowledge_base"):
    loader = DirectoryLoader(
        path=kb_dir,
        glob="**/*.md",
        loader_cls=TextLoader,
        show_progress=True,
    )
    documents = loader.load()
    print(f"已加载 {len(documents)} 个文档")
    return documents

docs = load_knowledge_base()
# 输出: 已加载 5 个文档
```

`DirectoryLoader` 会自动扫描指定目录下所有 `.md` 文件并逐个加载。每个文件变成一个 `Document` 对象，`metadata` 中自动包含 `source`（文件路径）信息。

### 第二步：智能分块

客服文档的分块策略需要特别注意。我们使用 `RecursiveCharacterTextSplitter` 并针对 Markdown 文档优化分隔符顺序：

```python
def split_documents(documents, chunk_size=800, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"分割为 {len(chunks)} 个文本块")
    return chunks

chunks = split_documents(docs)
# 输出: 分割为 28 个文本块
```

这里的 `separators` 参数定义了分块时的"切割优先级"：

| 优先级 | 分隔符 | 效果 |
|--------|--------|------|
| 1（最高） | `\n\n` | 先按段落切 |
| 2 | `\n` | 段落太长则按行切 |
| 3 | `。！？` | 句子级别切分 |
| 4 | 空格 | 词级别 |
| 5（最低） | 空字符串 | 强制切断 |

这个顺序确保了：**尽量保持段落完整 → 保持句子完整 → 最后才强制截断**。对于定价表这类结构化内容，`\n\n` 分隔符能保证每个定价层级的描述通常落在一个完整的块中。

来看看实际分块效果：

```python
for i, chunk in enumerate(chunks[:3]):
    print(f"\n--- Chunk {i} (来源: {chunk.metadata['source']}) ---")
    print(chunk.page_content[:200] + "...")
```

输出结果：

```
--- Chunk 0 (来源: pricing.md) ---
# CloudDesk 定价方案

## 免费版 (Free)

- **团队成员**: 最多 5 人
- **存储空间**: 2 GB
- **项目数量**: 最多 3 个
...

--- Chunk 1 (来源: pricing.md) ---
- **API 调用**: 每月 1,000 次
- **技术支持**: 社区论坛
- **价格**: 永久免费

## 专业版 (Professional) - ¥99/月
...
```

可以看到，免费版的完整信息被合理地分配到了相邻的两个块中（因为有 overlap），不会丢失关键信息。

### 第三步：创建向量索引

```python
def create_vectorstore(chunks, persist_dir="./chroma_cs_db"):
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(
        collection_name="customer_service_kb",
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )
    vectorstore.add_documents(chunks)
    print(f"向量索引创建完成，共 {len(chunks)} 条记录")
    return vectorstore

vectorstore = create_vectorstore(chunks)
# 输出: 向量索引创建完成，共 28 条记录
```

Chroma 会自动把每个文本块通过 `OpenAIEmbeddings` 转化为向量并存入本地数据库。下次启动时可以直接从持久化目录加载，无需重新计算嵌入。

```python
def load_or_create_vectorstore(chunks=None, persist_dir="./chroma_cs_db"):
    import os
    embeddings = OpenAIEmbeddings()
    if os.path.exists(persist_dir) and chunks is None:
        return Chroma(
            collection_name="customer_service_kb",
            embedding_function=embeddings,
            persist_directory=persist_dir,
        )
    vectorstore = Chroma(
        collection_name="customer_service_kb",
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )
    if chunks:
        vectorstore.add_documents(chunks)
    return vectorstore
```

这个 `load_or_create` 函数实现了**懒加载逻辑**：如果本地已有索引就直接加载，没有才重新构建。在实际项目中非常有用——你不想每次重启服务都等几十秒重建索引。

## 组装问答链

有了 retriever 之后，我们可以用 LCEL 语法把它组装成一个完整的问答链。这里的关键是 **prompt 的设计**——它直接决定了回答的质量。

### 设计客服专用 Prompt

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

SYSTEM_TEMPLATE = """你是 CloudDesk 的智能客服助手。你的职责是基于以下已知信息回答用户关于产品的问题。

规则：
1. 只使用【参考信息】中的内容来回答问题。如果参考信息中没有相关内容，请诚实地说"抱歉，这个问题我暂时无法解答，建议您联系人工客服获取帮助"。绝对不要编造信息。
2. 回答要简洁明了，优先给出用户最关心的核心信息。
3. 如果涉及价格、数量等具体数字，请确保与参考信息完全一致。
4. 使用友好、专业的语气。

【参考信息】
{context}

"""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_TEMPLATE),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}"),
])
```

注意这个 prompt 的几个精心设计的要点：

- **规则 1** 是最重要的——强制模型"不知道就说不知道"，从根本上杜绝幻觉
- **规则 3** 强调数字一致性——客服场景对精度要求极高
- `MessagesPlaceholder("chat_history")` 为后续接入记忆组件预留了位置

### 用 LCEL 组装完整链路

```python
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

def format_docs(docs):
    return "\n\n".join(f"[{i+1}] {d.page_content}" for i, d in enumerate(docs))

rag_chain = (
    RunnablePassthrough.assign(
        context=retriever | format_docs,
    )
    | qa_prompt
    | get_llm()
    | StrOutputParser()
)
```

这条链的数据流如下：

```
输入 {"question": "...", "chat_history": [...]}
    │
    ▼
RunnablePassthrough.assign
    │  把 question 和 chat_history 原样传递
    │  同时新增 context 字段 = retriever检索 → format_docs格式化
    ▼
{"question": "...", "chat_history": [...], "context": "[1] ... [2] ..."}
    │
    ▼
ChatPromptTemplate → 组装成完整消息
    │
    ▼
ChatOpenAI → LLM 生成回答
    │
    ▼
StrOutputParser → 提取纯文本
    │
    ▼
最终输出: "根据我们的定价方案..."
```

### 测试问答效果

```python
result = rag_chain.invoke({
    "question": "免费版支持几个人？",
    "chat_history": [],
})

print(result)
```

输出：

```
根据 CloudDesk 的定价方案，**免费版最多支持 5 名团队成员**。

此外，免费版还包含以下限制：
- 存储空间：2 GB
- 项目数量：最多 3 个
- API 调用：每月 1,000 次

如果您团队人数超过 5 人或需要更多资源，可以考虑升级到**专业版（¥99/月）**，支持无限成员和更大的存储空间。需要了解升级方式吗？
```

再看一个知识库中没有的问题：

```python
result = rag_chain.invoke({
    "question": "你们的产品支持离线使用吗？",
    "chat_history": [],
})
print(result)
```

输出：

```
抱歉，这个问题我暂时无法解答，建议您联系人工客服获取帮助。

如果您还有其他关于 CloudDesk 功能、定价或使用方法的问题，我很乐意为您解答！
```

这就是 prompt 中规则 1 发挥的作用——模型在参考信息中找不到相关内容时，选择了诚实拒绝而不是胡乱猜测。**在客服场景中，"不知道"远好于"瞎说"**。

## 检索质量优化

基础 RAG 链跑通之后，我们需要关注检索质量。客服场景中，最常见的检索问题有以下几类：

### 问题一：同义词/口语化表达

用户可能问："你们便宜的那个套餐多少钱？"——但知识库里写的全是"免费版""专业版"。语义上它们高度相关，但字面上差异很大。OpenAI Embeddings 对这种情况的处理通常不错，但如果发现效果不好，可以考虑：

```python
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 5, "score_threshold": 0.3},
)
```

设置 `score_threshold` 过滤掉相似度过低的检索结果，让 prompt 中的"不知道"规则更容易触发。

### 问题二：跨文档的信息分散

比如用户问"专业版和免费版有什么区别？"，理想情况下检索结果应该同时包含两个版本的描述。可以通过增大 `k` 值来缓解：

```python
retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
```

但 `k` 太大会引入噪声。更好的做法是在 prompt 中引导模型综合多个参考片段：

```python
SYSTEM_TEMPLATE += """
5. 如果多个参考片段都包含了有用信息，请综合所有片段给出完整回答，不要只依赖单一来源。
"""
```

### 问题三：元数据过滤

如果你的知识库很大（几百篇文档），无约束的全量搜索可能返回不相关的结果。利用 metadata 过滤可以大幅提升精准度：

```python
for chunk in chunks:
    chunk.metadata["category"] = "pricing" if "pricing" in chunk.metadata["source"] else "general"

retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 5,
        "filter": {"category": "pricing"},
    }
)
```

这样当系统识别到用户问的是定价问题时，只在定价相关文档中搜索，排除其他干扰。

## 将 RAG 管线封装为模块

为了方便后续与其他模块集成，我们把整个 RAG 管线封装成一个独立的模块：

```python
class CustomerServiceRAG:
    def __init__(self, kb_dir: str = "./knowledge_base", persist_dir: str = "./chroma_cs_db"):
        self.kb_dir = kb_dir
        self.persist_dir = persist_dir
        self.vectorstore = None
        self.retriever = None
        self.chain = None

    def initialize(self):
        docs = load_knowledge_base(self.kb_dir)
        chunks = split_documents(docs)
        self.vectorstore = load_or_create_vectorstore(chunks, self.persist_dir)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
        self.chain = self._build_chain()
        return self

    def _build_chain(self):
        return (
            RunnablePassthrough.assign(context=self.retriever | format_docs)
            | qa_prompt
            | get_llm()
            | StrOutputParser()
        )

    def query(self, question: str, chat_history: list = None) -> str:
        return self.chain.invoke({
            "question": question,
            "chat_history": chat_history or [],
        })

    def add_documents(self, new_docs_path: str):
        loader = DirectoryLoader(path=new_docs_path, glob="**/*.md", loader_cls=TextLoader)
        new_docs = loader.load()
        new_chunks = split_documents(new_docs)
        self.vectorstore.add_documents(new_chunks)
        print(f"已新增 {len(new_chunks)} 条知识")
```

这个封装提供了清晰的接口：`initialize()` 做一次性初始化，`query()` 执行查询，`add_documents()` 支持热更新知识库。下一节我们会把这个 RAG 模块挂载到意图路由系统中。
