---
title: 实战：从零实现一个"公司内部文档问答机器人"
description: 完整 RAG 问答系统：索引构建、检索问答、来源展示、CLI 交互
---
# 实战：从零实现一个"公司内部文档问答机器人"

前面五节我们系统地学习了 RAG 的每一个技术环节。现在到了最激动人心的部分——**把它们全部串联起来，搭建一个完整的"公司内部文档问答机器人"**。

这个系统将具备以下能力：
- 自动加载指定目录下的所有文档（PDF / Markdown / 文本）
- 将文档分块、向量化并存入向量数据库
- 接收用户问题 → 检索相关文档 → 基于参考资料生成答案
- 标注信息来源，支持追溯

## 第一步：准备示例数据

在项目目录下创建 `data/company-kb/` 并放入几个示例文件：

```markdown
<!-- data/company-kb/employee-handbook.md -->
# 员工手册 v2.3

## 第一章 入职指南

### 1.1 工作时间

公司实行弹性工作制：
- 核心工作时间：10:00 - 16:00
- 弹性时段：可自行安排，每天至少工作 8 小时
- 午休时间：12:00 - 13:30

### 1.2 考勤制度

- 迟到：每月允许 3 次 15 分钟内的迟到
- 早退：需提前向直属主管申请
- 年假：入职满一年后享受 5 天带薪年假
- 病假：每年 10 天带薪病假，需提供医疗证明

## 第二章 报销政策

### 2.1 差旅费

| 项目 | 标准 |
|------|------|
| 交通费 | 实报实销（经济舱） |
| 住宿费 | 一线城市 ≤500 元/晚，其他城市 ≤350 元/晚 |
| 餐饮补贴 | 100 元/天 |
| 市内交通 | 50 元/天 |

**注意**：单笔差旅费超过 **2000 元** 需要 **部门总监审批**，
超过 **5000 元** 需要 **CEO 审批**。

### 2.2 办公用品

- 笔记本电脑：入职领取，每三年可申请更换
- 显示器：可申请第二台显示器（需主管审批）
- 键盘鼠标：按需申领，无数量限制
```

```markdown
<!-- data/company-kb/product-faq.md -->
# 产品常见问题 FAQ

## Q1: 我们的产品支持哪些部署方式？

A1: 支持三种部署方式：
1. **SaaS 云端版** — 开箱即用，按月付费，无需运维
2. **私有化部署** — 部署在客户自有服务器上，数据完全自控
3. **混合云模式** — 敏感数据放本地，其他走云端

## Q2: 数据安全如何保障？

A2: 多层安全保障机制：
- 传输层：TLS 1.3 加密
- 存储层：AES-256 加密 + 密钥轮换
- 访问层：RBAC 角色权限 + IP 白名单
- 审计层：全量操作日志，保留 180 天

## Q3: API 速率限制是多少？

A3:
| 套餐 | 请求数/分钟 | 并发连接数 |
|------|------------|-----------|
| 免费版 | 20 | 5 |
| 专业版 | 200 | 20 |
| 企业版 | 1000 | 100 |

## Q4: 如何联系技术支持？

A4:
- 在线工单：控制台 → 提交工单（响应 < 2 小时）
- 技术热线：400-XXX-XXXX（工作日 9:00-18:00）
- VIP 客户：专属技术群，7×24 小时响应
```

## 第二步：索引构建脚本

```python
"""
rag_index.py — 构建公司知识库向量索引
用法: python rag_index.py
"""
import os
import shutil
from langchain_community.document_loaders import (
    DirectoryLoader, TextLoader, PyPDFLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

KNOWLEDGE_BASE = "data/company-kb/"
CHROMA_DIR = "./chroma_company_db"


def build_index():
    """加载文档 → 分块 → 向量化 → 存储"""

    # === Step 1: 加载文档 ===
    print("📂 正在加载知识库文档...")

    all_docs = []

    # 加载 Markdown 和文本文件
    if os.path.exists(KNOWLEDGEGE_BASE):
        md_loader = DirectoryLoader(
            path=KNOWLEDGE_BASE,
            glob="**/*.md",
            loader_cls=TextLoader,
            encoding="utf-8",
            show_progress=True
        )
        all_docs.extend(md_loader.load())

        txt_loader = DirectoryLoader(
            path=KNOWLEDGE_BASE,
            glob="**/*.txt",
            loader_cls=TextLoader,
            encoding="utf-8"
        )
        all_docs.extend(txt_loader.load())

        pdf_loader = DirectoryLoader(
            path=KNOWLEDGE_BASE,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        )
        all_docs.extend(pdf_loader.load())

    if not all_docs:
        print("⚠️ 未找到任何文档！请检查 data/company-kb/ 目录")
        return None

    print(f"\n✅ 共加载 {len(all_docs)} 个文档")
    total_chars = sum(len(d.page_content) for d in all_docs)
    print(f"   📊 总字符数: {total_chars:,}")

    # === Step 2: 分块 ===
    print("\n✂️  正在分块...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(all_docs)

    print(f"✅ 切分为 {len(chunks)} 个文本块")

    # === Step 3: 向量化并存储 ===
    print("\n🔢 正在生成向量嵌入...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)
        print("   🗑️  清除了旧索引")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )

    print(f"\n🎉 索引构建完成！")
    print(f"   📁 向量库位置: {CHROMA_DIR}")
    print(f"   📦 文档块总数: {len(chunks)}")

    return vectorstore


if __name__ == "__main__":
    build_index()
```

运行：

```bash
python rag_index.py
```

输出：

```
📂 正在加载知识库文档...
✅ 共加载 2 个文档
   📊 总字符数: 2,847

✂️  正在分块...
✅ 切分为 12 个文本块

🔢 正在生成向量嵌入...
   🗑️ 清除了旧索引

🎉 索引构建完成！
   📁 向量库位置: ./chroma_company_db
   📦 文档块总数: 12
```

## 第三步：问答系统核心

```python
"""
company_qa.py — 公司内部文档问答机器人
需要先运行 rag_index.py 构建索引
"""
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

CHROMA_DIR = "./chroma_company_db"


def format_docs(docs):
    """格式化检索结果为可读文本"""
    parts = []
    for i, doc in enumerate(docs, 1):
        src = doc.metadata.get("source", "?")
        src = os.path.basename(src) if isinstance(src, str) else "?"
        content = doc.page_content.replace("\n", " ")[:150]
        parts.append(f"[资料 {i} | {src}]\n{content}")
    return "\n\n".join(parts)


def create_qa_chain():
    """创建 RAG 问答 Chain"""

    # 加载已有索引
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-small")
    )

    # 检索器：返回 top-3 个最相关的文档块
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        k=3
    )

    # LLM
    chat = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # RAG 提示词模板
    prompt = ChatPromptTemplate.from_template("""
你是一个公司内部知识助手。请严格基于以下参考资料回答用户的问题。

规则：
- 只基于参考资料回答，不要编造或使用外部知识
- 如果资料中没有相关信息，明确说"我在知识库中找不到关于此问题的信息"
- 回答要简洁专业
- 引用具体的参考来源

参考资料：
{context}

用户问题：{question}
""")

    # 组装 RAG Chain
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | chat
        | StrOutputParser()
    )

    return chain


def main():
    chain = create_qa_chain()

    print("=" * 56)
    print("   🏢 公司内部文档问答机器人")
    print("=" * 56)
    print("   知识范围：员工手册、产品 FAQ 等")
    print("   输入 '退出' 结束对话\n")

    while True:
        question = input("❓ 你: ").strip()

        if not question:
            continue
        if question.lower() in ["退出", "exit", "quit"]:
            print("👋 再见！")
            break

        print("\n🤖 正在查询知识库...\n")
        answer = chain.invoke(question)
        print(f"💡 {answer}\n")


if __name__ == "__main__":
    main()
```

## 第四步：运行效果演示

```bash
python company_qa.py
```

交互效果：

```
========================================================
   🏢 公司内部文档问答机器人
========================================================
   知识范围：员工手册、产品 FAQ 等
   输入 '退出' 结束对话

❓ you: 我出差去北京，住宿费标准是多少？

🤖 正在查询知识库...

💡 根据员工手册的报销政策：

**住宿费标准**：
- **一线城市**（含北京、上海、广州、深圳）：≤ **500 元/晚**
- **其他城市**：≤ **350 元/晚**

💡 [资料来源: employee-handbook.md]

---

❓ you: 如果差旅费超过 5000 怎么办？

🤖 正在查询知识库...

💡 根据报销政策规定：

差旅费的审批门槛如下：
- 单笔超过 **2,000 元** → 需 **部门总监审批**
- 单笔超过 **5,000 元** → 需 **CEO 审批**

如果你的差旅费预计超过 5,000 元，建议提前准备详细的预算说明和业务理由。

💡 [资料来源: employee-handbook.md]

---

❓ you: API 速率限制是多少？

🤖 正在查询知识库...

💡 根据 FAQ 文档，API 速率限制如下：

| 套餐 | 请求数/分钟 | 并发连接数 |
|------|------------|-----------|
| 免费版 | 20 | 5 |
| 专业版 | 200 | 20 |
| 企业版 | 1,000 | 100 |

💡 [资料来源: product-faq.md]

---

❓ you: 公司的 CEO 叫什么名字？

🤖 正在查询知识库...

💡 我在公司知识库中找不到关于此问题的信息。
当前知识库包含员工手册和产品 FAQ，不包含高管个人信息。
```

注意最后一个问题——当知识库中没有相关信息时，模型诚实地说了"找不到"，而不是像普通 LLM 那样瞎编一个名字。**这就是 RAG 相对于纯 LLM 问答的核心优势之一**。

## 第五步：完整项目结构

```
company-qa-bot/
├── .env                          # OPENAI_API_KEY
├── .gitignore
├── requirements.txt
├── data/
│   └── company-kb/               # 知识库目录
│       ├── employee-handbook.md
│       └── product-faq.md
├── rag_index.py                  # 索引构建脚本
├── company_qa.py                 # 问答主程序
└── chroma_company_db/             # 向量数据（自动生成）
```

`requirements.txt`：

```
langchain>=0.3
langchain-openai>=0.2
langchain-community>=0.3
langchain-chroma>=0.3
langchain-text-splitters>=0.3
langchain-core>=0.3
chromadb>=0.5
pypdf>=4.0
python-dotenv>=1.0
pydantic>=2.0
```

## 扩展方向

完成基础版后，有几个值得探索的方向：

**方向一：添加记忆能力**

```python
from langchain.runnables.history import RunnableWithMessageHistory
from langchain.chat_history import InMemoryChatMessageHistory

chain_with_memory = RunnableWithMessageHistory(
    runnable=base_chain,
    get_session_history=lambda sid: InMemoryChatMessageHistory(),
    input_messages_key="question"
)
```

这样用户可以追问："刚才说的那个审批流程具体怎么操作？"——系统能回忆之前的上下文。

**方向二：多格式知识库支持**

目前只支持 Markdown 和文本。扩展 PDF 支持：

```python
# 在 rag_index.py 中加入 PDF 加载
if has_pypdf:
    from langchain_community.document_loaders import PyPDFLoader
    pdf_loader = DirectoryLoader(path=KB_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader)
    all_docs.extend(pdf_loader.load())
```

**方向三：接入真实搜索补充**

对于知识库中没有但可以通过搜索引擎找到的信息：

```python
@tool
def search_web(query):
    """搜索互联网获取实时信息"""
    ...

# 把 web search 也作为工具加入 RAG 流程
```

到这里，第四章的全部 6 个小节就结束了。我们从 RAG 的基本原理出发，学习了文档加载（100+ 数据源）、文本分割策略（按字符/递归/语义）、向量存储与嵌入模型选择、高级检索策略（MMR/压缩/过滤），最终搭建了一个完整的公司内部文档问答机器人。这套知识是后续所有高级 RAG 应用和 Agent 系统的基础。
