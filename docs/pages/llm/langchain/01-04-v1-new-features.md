---
title: LangChain v1.0 新特性概览
description: LCEL 表达式语言、Agent 架构升级、LangGraph 统一架构、中间件系统、从 v0.x 迁移的注意事项
---
# LangChain v1.0：一次全面的架构升级

如果你在网上看到一些 2023 年或更早的 LangChain 教程，可能会发现代码风格和现在很不一样。这是因为 LangChain 在 2024 年底发布了 v1.0 版本——这不是一个小版本号迭代，而是一次**底层的架构重写**。这一节帮你理解 v1.0 带来了哪些关键变化，以及为什么这些变化很重要。

## v0.x 的痛点：API 碎片化

在 v0.x 时代，LangChain 的 API 被广泛诟病的问题主要有三个：

**第一是概念碎片化**。同样叫 "Chain" 的东西有好几种不同的实现方式（`LLMChain`、`SequentialChain`、`TransformChain`……），同样叫 "Memory" 的也有好几种（`ConversationBufferMemory`、`ConversationSummaryMemory`、`VectorStoreRetrieverMemory`……）。新手面对这么多类名会感到困惑："我到底该用哪个？"

**第二是命令式风格主导**。大部分代码还是 `chain = SomeChain([...])` 这种构造器模式，虽然能用，但不够直观。你无法一眼看出数据在各个组件之间是怎么流动的。

**第三是 Agent 能力有限**。v0.x 的 Agent 基于 `AgentExecutor`，本质上是一个 while 循环 + if/else 分支。对于简单的"思考-行动-观察"循环够用了，但一旦你需要多个 Agent 协作、需要长期规划、需要反思和自我纠正，这套机制就开始显得力不从心。

## v1.0 的核心变化：LCEL 成为一等公民

v1.0 把所有东西统一到了 **LCEL（LangChain Expression Language）** 这套表达式语言之上。不管你构建什么——单轮对话、多步 Chain、复杂 Agent —— 底层都是同一个编程模型。

### Runnable：一切皆可运行

v1.0 引入了 `Runnable` 作为统一的组件抽象：

```python
from langchain_core.runnables import (
    RunnablePassthrough,   # 透传输入（不做任何处理）
    RunnableLambda,        # 自定义任意函数
    RunnableParallel,     # 并行执行多个分支
    RunnableBranch,       # 条件分支路由
)

# 每个组件都实现同一套接口：
# - invoke() / ainvoke()      同步/异步调用
# - stream() / astream()      流式输出
# - bind()                  绑定运行时参数
# - pipe()                  链式组合

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

# 一个最简单的 chain —— 用管道符串联
chain = RunnablePassthrough() | llm
result = chain.invoke("你好")
```

注意这里没有 `Chain` 类、没有 `LLMChain`、没有 `SequentialChain`——只有 `Runnable` + `|`。这就是 v1.0 的核心理念：**把所有组件统一到一套接口上，用管道符来声明它们之间的连接关系**。

### 管道操作符 `|`：可视化你的流程

`|` 操作符是 v1.0 最具标志性的语法特征。它的作用是把左侧 Runnable 的输出作为右侧 Runnable 的输入：

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("将以下内容翻译成英文：{input}")
parser = StrOutputParser()

# 数据流向一目了然：
# 用户输入 → prompt 注入 → LLM 翻译 → 解析输出
chain = prompt | llm | parser

result = chain.invoke({"input": "你好世界"})
print(result)  # "Hello World"
```

你可以把它理解为 Unix 管道的 LLM 版本——但比 Unix 管道更强的是，LangChain 的 `|` 不只是传递原始字符串，它会自动处理不同组件之间的格式适配（比如把 dict 变成字符串、把字符串解析成结构化对象）。

### 并行与分支

v1.0 的另一个重要能力是**并行执行**和**条件路由**：

```python
from langchain_core.runnables import RunnableParallel, RunnableBranch

# 并行：同时调用两个模型做对比
parallel = RunnableParallel(
    gpt4o=ChatOpenAI(model="gpt-4o"),
    claude=ChatOpenAI(model="claude-sonnet-4-20250514")
)

# 分支：根据问题类型走不同处理路径
router = (
    RunnableBranch(
        (lambda x: "计算" in x["input"], math_chain),   # 数学题走计算链
        (lambda x: "翻译" in x["input"], trans_chain)  # 翻译题走翻译链
    )
    | default_chain                                       # 其他走默认链
)
```

这在 v0.x 时代需要写大量 `if/else` 才能实现的逻辑，现在一行声明式代码就搞定了。

## Agent 架构升级：从循环到图

v1.0 中 Agent 的底层架构被替换为 **LangGraph**——一个基于有向图的执行引擎。这带来了几个重要的能力提升：

```python
from langgraph.prebuilt import create_react_agent

# v1.0 推荐的 Agent 创建方式
agent = create_react_agent(
    llm,
    tools=[search_tool, calculator_tool],
    # 内部使用 LangGraph 编排 ReAct 循环
)
```

LangGraph 的核心思想是把 Agent 的行为建模为一个**状态图（State Graph）**：

```
┌──────────┐    ┌──────────┐    ┌──────────┐
│  开始    │───→│  思考    │───→│  行动    │
│          │    └────┬─────┘    └────┬─────┘
└──────────┘         │                │
                    ▼                ▼
              ┌──────────┐    ┌──────────┐
              │  观察    │───→│  结束？  │
              └──────────┘    └────┬─────┘
                                  │
                        ┌────▼────┐
                        │  是→结束  │
                        │  否→回到  │
                        │  思考     │
                        └──────────┘
```

这就是经典的 **ReAct（Reasoning + Acting）** 循环：思考 → 行动 → 观察结果 → 决定是否继续。LangGraph 把这个循环建模为图中的节点和边，让框架能精确控制每一步的状态转换。

## 中间件（Middleware）：横切关注点

v1.0 正式引入了**中间件（Middleware）**的概念——它让你能在不修改业务逻辑的情况下，横切地插入通用处理逻辑：

```python
from langchain_community.callbacks import get_openai_callback

# 中间件的典型用途：
chain = (
    prompt
    | llm.with_config({
        "callbacks": [get_openai_callback()],   # 记录每次调用的 token 和延迟
        "tags": ["my-app", "production"],            # 用于追踪和过滤
    })
)

# 其他常见的中间件用途：
# - 日志记录：打印每次 invoke 的完整信息
# - 限流控制：防止调用频率过高触发 API 限流
# - 重试机制：网络错误时自动重试
# - 内容审核：在发送给 LLM 之前检查敏感内容
```

中间件的设计借鉴了 Web 框架（如 Express/Koa）的中间件模式——它是一种**面向切面（AOP）** 的编程范式，让你能把日志、监控、安全检查等"横切关注点"从业务逻辑中解耦出来。

## 从 v0.x 迁移：需要注意什么

如果你正在维护一个基于 v0.x 的项目，迁移到 v1.0 时有几个关键点需要注意：

| 变化项 | v0.x 写法 | v1.0 写法 |
|--------|-----------|-----------|
| 创建 Chain | `LLMChain([prompt, llm])` | `prompt \| llm` |
| 创建 Agent | `AgentExecutor.from_tools_and_functions(llm, tools)` | `create_react_agent(llm, tools)` |
| 输出解析 | `OutputFixingParser` | `StrOutputParser()` / `JsonOutputParser()` |
| 自定义步骤 | 继承 `BaseChain` | 继承 `Runnable` 或使用 `RunnableLambda` |

最大的思维模式转变是：**从"构造参数式"到"管道组合式"**。刚开始可能不太习惯，但一旦你适应了 `\|` 链式语法，就会发现它比旧方式更直观、更灵活、更容易调试。

## 总结：v1.0 带来了什么

| 维度 | v0.x | v1.0 |
|------|------|------|
| 核心抽象 | 多种 Chain/Memory/Agent 类 | 统一为 Runnable 接口 |
| 组合方式 | 构造器参数 | `\|` 管道操作符 |
| Agent 能力 | 简单 ReAct 循环 | LangGraph 图编排 |
| 可观测性 | 回调函数 | 中间件（Middleware） |
| 并行/分支 | 手动管理 | RunnableParallel / RunnableBranch |
| 学习曲线 | 较陡（概念多且重叠） | 更平缓（统一模型） |

好了，理论部分到此结束。下一章我们就要动手了——安装依赖、配置环境、跑通你的第一个 LangChain 应用。
