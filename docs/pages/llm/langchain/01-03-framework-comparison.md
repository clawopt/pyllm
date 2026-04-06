---
title: LangChain 与同类框架的对比
description: LlamaIndex / Semantic Kernel / CrewAI 等框架的定位差异、选型决策指南、各框架的适用场景分析
---
# LangChain 与同类框架：怎么选？

上一节我们提到 LangChain 不是唯一的 LLM 编排框架。这一节我们把几个主要竞品摆在一起，从实际开发的角度做一次客观的横向对比，帮你建立"什么时候该用什么工具"的判断力。

## 主流框架速览

目前 LLM 应用开发领域有几个影响力较大的框架：

| 框架 | 创立者 | 定位 | GitHub Stars |
|------|--------|------|-------------|
| **LangChain** | Harrison Chase (2022.10) | 通用 LLM 编排 | ~95k |
| **LlamaIndex** | Jerry Liu (2022.10) | RAG / 数据连接 | ~38k |
| **Semantic Kernel** | 微软 (2023.05) | 企业级多模型编排 | ~22k |
| **CrewAI** | João Moura (2023.06) | 轻量级 Agent | ~32k |
| **AutoGen** | 微软 (2023.09) | 多 Agent 协作 | ~45k |

注意这些框架并不是完全互斥的——很多项目会组合使用它们（比如用 LlamaIndex 做 RAG、用 LangChain 做流程编排）。理解它们的定位差异比纠结"哪个更好"更有意义。

## LangChain vs LlamaIndex：通用 vs 专用

这是最容易混淆的两个框架，因为它们的功能重叠度很高。核心区别在于**设计哲学的不同**：

```
LangChain 的思维方式：
  "我有一个复杂的业务流程 —— 需要调用 LLM、搜索数据库、
   调用 API、做分支判断…… 我需要一个框架来编排这一切"

LlamaIndex 的思维方式：
  "我有大量私有数据（文档/PDF/数据库）—— 需要让 LLM 能基于
   这些数据来回答问题 —— 我需要一个框架来管理数据索引和检索"
```

用一个具体的例子来说明区别：

```python
# === 用 LangChain 构建一个客服机器人 ===

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o-mini")
prompt = ChatPromptTemplate.from_template("""
你是一个客服助手。请根据以下用户问题和知识库信息回答。
如果知识库中没有相关信息，请诚实地说不知道。

问题：{question}
知识库信息：{context}
""")

chain = prompt | llm | StrOutputParser()
answer = chain.invoke({
    "question": "你们的产品支持退款吗？",
    "context": "我们的产品支持 7 天无理由退款..."
})

# LangChain 的强项：
# - 定义清晰的多步流程（Chain）
# - 灵活地接入各种 Tool（搜索、数据库、API）
# - 记忆管理（Memory）
# - Agent 自主决策（ReAct 循环）
```

```python
# === 用 LlamaIndex 构建一个文档问答系统 ===

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# 加载文档并自动切分 + 向量化 + 存储
documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)

# 查询时自动完成：检索相关片段 → 注入 prompt → 生成回答
query_engine = index.as_query_engine(similarity_top_k=3)
response = query_engine.query("你们的退货政策是什么？")

# LlamaIndex 的强项：
# - 一行代码加载 100+ 种数据源（PDF、Notion、GitHub 等）
# - 自动文本切分（递归字符分割、语义分割等策略）
# - 内置向量存储和检索管道
# - RAG 特有的优化（重排序、查询变换等）
```

### 选型建议

| 你的需求 | 推荐框架 | 原因 |
|---------|---------|------|
| 复杂的多步工作流（客服、数据分析 Agent） | **LangChain** | Chain/Agent/Memory 更成熟 |
| 文档问答 / 知识库检索为主 | **LlamaIndex** | 数据加载和 RAG 是它的主场 |
| 两者都需要 | **LangChain + LlamaIndex** | 可以通过 LlamaIndex Tool 接入 LangChain |

## LangChain vs Semantic Kernel / AutoGen：开源 vs 企业

微软同时维护着两个框架——Semantic Kernel 和 AutoGen——它们的设计目标不同：

**Semantic Kernel** 的卖点是**企业级集成**：

```python
# Semantic Kernel 的独特之处：同一套代码对接多个 LLM 提供商
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.connectors.ai.azure_chat_completion import AzureChatCompletion

kernel = sk.Kernel()
# 同一个 kernel 可以切换不同的后端 —— 对企业多云策略很友好
kernel.add_service(OpenAIChatCompletion("openai"))
kernel.add_service(AzureChatCompletion("azure"))
```

如果你的公司同时使用 OpenAI、Azure OpenAI、甚至本地部署的模型，Semantic Kernel 的统一抽象层能大幅减少适配工作量。

**AutoGen** 的卖点是**多 Agent 协作**：

```python
# AutoGen 的核心：多个 Agent 之间可以对话和协作
import autogen

user = autogen.UserProxyAgent(name="User")
coder = autogen.AssistantAgent(
    name="Coder",
    llm_config={"config_list": [{"model": "gpt-4o"}]}
)
reviewer = autogen.AssistantAgent(
    name="Reviewer",
    llm_config={"config_list": [{"model": "gpt-4o"}]}
)

# 启动一个多人协作：用户提需求 → Coder 写代码 → Reviewer 审查 → 迭代修改
groupchat = autogen.GroupChat(agents=[user, coder, reviewer])
result = await groupchat.a_run("帮我写一个快速排序算法", summary_method="last_msg")
```

AutoGen 在"多个 AI 角色协作完成复杂任务"这个场景下非常强大——比如让一个 Agent 写代码、另一个 Agent review、第三个 Agent 做测试。

## LangChain vs CrewAI：重量级 vs 轻量级

CrewAI 是一个相对年轻的框架（2023 年才出现），它的设计哲学是**极简主义**：

```python
# CrewAI 的风格：角色驱动，极少的样板代码
from crewai import Agent, Task, Crew

researcher = Agent(
    role="研究员",
    goal="搜索并整理最新信息",
    backstory="你是一个专业的研究员",
    llm="gpt-4o-mini"
)

writer = Agent(
    role="写手",
    goal="基于研究结果撰写报告",
    backstory="你是一个技术写作专家",
    llm="gpt-4o"
)

task = Task(
    description="研究 2025 年大模型领域的最新进展",
    expected_output="一份包含 5 个要点的报告",
    agent=researcher
)

crew = Crew(agents=[researcher, writer], tasks=[task])
result = crew.kickoff()  # 一行启动
```

CrewAI 的优势是**上手极快**——定义几个 Agent 和 Task 就能跑起来。但代价是定制性不如 LangChain：当你需要精细控制每一步的行为（比如自定义中间件、复杂的路由逻辑），CrewAI 的抽象层反而成了限制。

## 一个实用的选型决策树

```
你需要构建什么？
│
├─ 文档/知识库问答为主
│   └─→ LlamaIndex（RAG 是它的核心能力）
│
├─ 复杂的业务流程（多步骤、多工具、多分支）
│   ├── 不需要多 Agent 协作 → LangChain
│   └─ 需要 Agent 团队协作 → LangChain 或 AutoGen
│
├─ 快速原型 / 个人项目
│   └─→ CrewAI 或直接调 API（最简单）
│
├─ 企业环境 / 多云策略
│   └─→ Semantic Kernel（统一抽象层）
│
└─ 学习和理解 LLM 应用开发
    └─→ LangChain（生态最大，学习资源最多）
```

## 本教程的选择与说明

本教程以 **LangChain 为主线**，原因有三：

第一，**生态成熟度最高**。LangChain 拥有最多的集成（200+ 工具/数据源）、最完善的文档、最大的社区——你在开发过程中遇到的绝大多数问题都能找到现成方案。

第二，**覆盖面最广**。从简单的单轮对话到复杂的多 Agent 系统，从 RAG 到记忆到流式输出，LangChain 都有一等一的解决方案——学完一套框架就能应对大部分场景。

第三，**就业市场需求最大**。在招聘市场上，"熟悉 LangChain"几乎已经成为 LLM 应用开发岗位的标准要求之一。掌握它之后，切换到其他框架的成本很低（概念相通），但反过来不一定成立。

下一章我们就正式进入实战：搭建开发环境，安装依赖，写出你的第一个 LangChain 应用。
