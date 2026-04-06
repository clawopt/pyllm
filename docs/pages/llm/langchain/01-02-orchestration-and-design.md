---
title: 什么是编排框架？LangChain 的设计哲学
description: 编排（Orchestration）的核心概念、LangChain 的模块化架构、设计理念与核心抽象
---
# 什么是编排框架？LangChain 的设计哲学

上一节我们列出了大模型的五大痛点。这一节要回答的问题是：**LangChain 是怎么解决这些问题的？** 在回答之前，我们需要先理解一个更基础的概念——"编排"。

## 从"直接调用"到"编排"

当你第一次使用大模型 API 时，代码大概长这样：

```python
from openai import OpenAI

client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "你好"}]
)
print(response.choices[0].message.content)
```

这叫**直接调用（Direct Call）**——你手动构造请求、发送给模型、拿到响应。对于单轮问答来说，这完全够用。但如果你想做一个稍微复杂一点的应用——比如一个能搜索最新信息再回答问题的助手——事情就开始变麻烦了：

```python
# 你需要自己管理这样的流程：
# 1. 用户提问 → 2. 判断是否需要搜索 → 3. 调用搜索工具 
# → 4. 把搜索结果塞进 prompt → 5. 调用 LLM 生成回答 → 6. 返回结果

# 每一步都要你自己写 if/else、错误处理、状态传递……
```

当流程变得更复杂——比如需要多步推理、调用多个工具、在工具之间做分支判断——纯手写的代码就会变成一堆难以维护的 `if/else` 嵌套和状态变量。这就是**编排框架**要解决的问题：**把"决定下一步做什么"这个逻辑，从你的业务代码中抽出来，交给一个统一的框架来管理**。

### LangChain 的核心理念：链式与组件

LangChain 的名字本身就揭示了它的两个核心概念：

**Chain（链）**：把多个步骤串联成一个流水线。就像工厂的装配线——原料进入第一道工序，加工后传到第二道、第三道……最终产出成品。每道工序可以是一个 LLM 调用、一次工具调用、或者一段数据变换逻辑。

```python
# 一个最简单的 Chain：用户问题 → 翻译成英文 → 翻译回中文
chain = (
    prompt_1 | llm_translate_to_en   # 第一步：翻译成英文
    | prompt_2 | llm_translate_back     # 第二步：翻译回中文
)
result = chain.invoke("你好世界")
```

注意这里的 `|` 操作符——它不是 Unix 的管道，而是 LangChain 的**管道符号（Pipe Operator）**，表示"把上一步的输出作为下一步的输入"。这是 LangChain 最具标志性的语法特征。

**Component（组件）**：构成链条的基本单元。LangChain 定义了几类标准组件：

| 组件类型 | 职责 | 举例 |
|---------|------|------|
| **ChatModel** | 与大模型对话 | ChatOpenAI, ChatAnthropic |
| **PromptTemplate** | 提示词模板 | 支持变量注入、Few-shot 示例 |
| **OutputParser** | 解析模型输出 | JSON 提取器、列表解析器 |
| **Tool / ToolNode** | 外部工具封装 | 搜索 API、数据库查询、代码执行 |
| **Retriever** | 信息检索 | 向量相似度搜索 |
| **Memory** | 对话记忆 | 缓冲区记忆、向量存储记忆 |

你可以把这些组件像搭积木一样组合成各种不同的 Chain——这就是 LangChain 的**组合式设计（Compositional Design）**哲学。

## 设计哲学：声明式而非命令式

理解 LangChain 的设计哲学，关键在于区分两种编程范式：

**命令式（Imperative）**：你告诉计算机**怎么做**——先做 A，再做 B，如果 C 就做 D。
**声明式（Declarative）**：你告诉计算机**你要什么**——具体执行步骤由框架来安排。

传统方式是命令式的：

```python
# 命令式：你需要手动控制每一步
question = "北京今天天气怎么样"
search_result = search_api(question)        # 手动调搜索
context = f"已知信息：{search_result}"      # 手动拼上下文
answer = llm.invoke(context)                # 手动调 LLM
parsed = json.loads(answer)                 # 手动解析输出
```

LangChain 的方式是声明式的：

```python
# 声明式：你定义"做什么"，框架控制"怎么做"
chain = (
    {"question": RunnablePassthrough()}       # 透传用户问题
    | search_tool                               # 自动调用搜索
    | prompt_template                           # 自动组装提示词
    | llm                                       # 自动调用模型
    | JsonOutputParser()                       # 自动解析输出
)
answer = chain.invoke("北京今天天气怎么样")
```

两者的区别看起来只是语法糖，但在复杂场景下差异巨大。声明式的好处是：

1. **流程可视化**：`|` 链条本身就是一张流程图，一眼就能看懂数据如何流动
2. **易于修改**：想在中间加一步？插入一个新组件就行，不用重构整个逻辑
3. **可测试性强**：每个组件可以独立 mock 和验证
4. **框架优化空间**：LangChain 可以在底层做批处理、缓存、并行化等优化，而你的业务代码不需要改动

## LangChain v0.1 到 v1.0：架构演进

如果你在网上看到一些比较老的 LangChain 教程，可能会发现它用的 API 和现在很不一样。这是因为 LangChain 经历了一次重大的架构升级：

| 版本 | 核心概念 | 特点 |
|------|---------|------|
| **v0.1~0.2** | Chain、Agent、Tool、Memory | 功能丰富但 API 混乱，很多类功能重叠 |
| **v0.3** | 引入 LCEL（LangChain Expression Language） | 新增 `\|` 管道符，开始走向声明式 |
| **v1.0** (2024年底发布) | 基于 LCEL 重构全部 API | 统一为 LangGraph 架构，Agent 能力大幅增强 |

v1.0 的最大变化是把所有东西统一到了 **LCEL（LangChain Expression Language）** 这个表达式语言之上。不管你是构建简单的单步对话还是复杂的多 Agent 协作系统，底层都是同一个 `Runnable` 接口 + `|` 管道操作符。

```python
# v1.0 中，一切都是 Runnable
from langchain_core.runnables import RunnablePassthrough

chain = (
    RunnablePassthrough.assign(                  # 这是一个 Runnable
        question=lambda x: x["input"]
    )
    | chat_model                                 # 这也是一个 Runnable
    | StrOutputParser()                          # 这还是 Runnable
)

# 所有组件都实现同一套接口 —— 这是 v1.0 的核心设计
```

## LangChain 不是唯一的选择

最后需要客观地说：LangChain 不是唯一的 LLM 编排框架。市场上还有几个重要的竞争者：

| 框架 | 特点 | 适用场景 |
|------|------|---------|
| **LangChain** | 生态最大、文档最全、社区活跃 | 通用 LLM 应用开发 |
| **LlamaIndex** | 专注 RAG（检索增强生成） | 文档问答、知识库检索 |
| **Semantic Kernel (微软)** | 企业级、多模型支持 | 大型企业应用 |
| **CrewAI** | 轻量级 Agent 框架 | 快速原型开发 |

它们各有侧重，没有绝对的优劣。本教程选择 LangChain 作为主线，是因为它的生态最为成熟、社区资源最丰富、学习曲线相对平缓——对于从零开始学习 LLM 应用开发的工程师来说，是最友好的起点。

下一节我们就要动手了：搭建环境、安装依赖、跑通你的第一个 LangChain 应用。
