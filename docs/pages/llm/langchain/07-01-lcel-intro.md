---
title: 为什么需要 LCEL？声明式 vs 命令式
description: LCEL 解决的问题、Runnable 统一接口、声明式链式组合的优势
---
# 为什么需要 LCEL？声明式 vs 命令式

在前面的章节中，我们一直在使用 `|` 管道操作符来组装 LangChain 应用：

```python
chain = prompt | chat | StrOutputParser()
result = chain.invoke({"topic": "RAG"})
```

这种写法简洁、直观，你可能已经习以为常了。但这个 `|` 并不是 Python 的原生语法——它是 **LCEL（LangChain Expression Language）** 的核心表达方式。这一章我们将系统地学习 LCEL：它是什么、为什么需要它、以及如何用它构建更复杂的应用。

## 从命令式到声明式

要理解 LCEL 的价值，先看看在没有它的时候，我们是怎么写代码的。

### 命令式风格（旧方式）

```python
# 旧式命令式代码 — 手动管理每一步的输入输出
from langchain_openai import ChatOpenAI

chat = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Step 1: 格式化提示词
prompt_text = "解释{topic}"
formatted_prompt = prompt_text.format(topic="装饰器")

# Step 2: 调用模型
response = chat.invoke(formatted_prompt)

# Step 3: 提取文本
answer = response.content

print(answer)
```

这段代码能工作，但有几个问题：

**问题一：步骤之间是割裂的**。每一步的输出需要手动传给下一步——如果中间环节增加或减少一步，你需要修改所有后续的数据传递逻辑。

**问题二：难以复用和组合**。如果你想把这个"格式化→调用→解析"的流程复用到另一个场景（比如换一个 prompt 或加一个检索步骤），你需要复制粘贴然后逐行修改。

**问题三：无法统一处理**。不同的组件有不同的调用方式——有的接收字符串，有的接收消息列表，有的接收字典。手动处理这些差异非常繁琐。

### 声明式风格（LCEL）

```python
# LCEL 声明式 — 只描述"做什么"，不关心"怎么做"
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

chat = ChatOpenAI(model="gpt-4o-mini", temperature=0)
prompt = ChatPromptTemplate.from_template("解释{topic}")
parser = StrOutputParser()

# 用管道串联 — 声明数据流，自动处理中间细节
chain = prompt | chat | parser

result = chain.invoke({"topic": "装饰器"})
```

同样的功能，代码量更少，而且更重要的是——**你只声明了组件之间的连接关系，不需要手动管理数据传递的细节**。

## LCEL 的核心理念

LCEL 基于 **Runnable** 这个统一的抽象接口。在 LangChain v1.0 中，几乎所有组件都实现了 Runnable 接口——包括 LLM、PromptTemplate、OutputParser、Retriever、甚至一个普通的 Python 函数。

### Runnable 接口的通用方法

任何 Runnable 对象都支持以下方法：

```python
from langchain_core.runnables import Runnable

class MyComponent(Runnable):
    def invoke(self, input):
        return f"处理结果: {input}"

comp = MyComponent()

# 同步调用
result = comp.invoke("hello")          # "处理结果: hello"

# 批量调用
results = comp.batch(["a", "b", "c"])   # ["处理结果: a", ...]

# 流式输出
for chunk in comp.stream("test"):
    print(chunk, end="")
```

这意味着不管底层是什么——LLM、模板、解析器还是自定义函数——**它们的调用方式完全一致**。这就是 LCEL 能用同一个 `|` 操作符把它们串起来的根本原因。

### 管道操作符 `|` 的含义

`|` 在 LCEL 中的语义是：**把左侧组件的输出作为右侧组件的输入**。

```python
chain = A | B | C

# 等价于：
# temp = A.invoke(input)
# temp = B.invoke(temp)
# result = C.invoke(temp)
```

但 `|` 比手动调用强大得多：
- 它自动处理类型转换（比如把 dict 自动匹配到 template 的变量）
- 它支持并行分支（后面会讲）
- 它可以被整个包装成一个新的 Runnable（可以继续参与管道）
- 它自带错误传播和调试信息

## LCEL 能解决什么具体问题

### 问题一：动态路由

假设你想根据用户问题的类型选择不同的处理路径：

```python
# 命令式 — 大量 if-else
def process(question: str):
    if is_math_question(question):
        result = calculator_chain.invoke(question)
    elif needs_search(question):
        result = search_chain.invoke(question)
    else:
        result = general_chain.invoke(question)
    return result
```

```python
# LCEL 声明式 — 用 RunnableBranch 自动路由
branch = RunnableBranch(
    (lambda x: is_math(x["question"]), math_chain),
    (lambda x: needs_search(x["question"]), search_chain),
    general_chain
)

chain = branch
# 不需要手写 if-else，路由逻辑被声明式地定义
```

### 问题二：并行执行

同时做多个独立操作，然后把结果合并：

```python
# LCEL 声明式 — RunnableParallel 自动并行
parallel = RunnableParallel(
    summary=summary_chain,
    keywords=keyword_chain,
    sentiment=sentiment_chain
)

result = parallel.invoke({"text": long_article})
# 三个 chain 同时执行，结果合并到一个字典中
```

这在命令式风格中需要手动管理线程或 asyncio，而在 LCEL 中只需要声明"哪些部分应该并行"。

### 问题三：中间件 / 横切关注点

想在 Chain 的每个环节添加日志、限流、重试等横切逻辑：

```python
# LCEL — 用 pipe() 添加中间件
logged_chain = (
    prompt 
    | log_input("📥 Prompt")       # 中间件：记录输入
    | chat
    | log_output("🤖 LLM 回复")     # 中间件：记录输出
    | parser
    | measure_time("⏱️ 耗时")      # 中间件：计时
)
```

每个中间件本身也是一个 Runnable，可以随时插入或移除，不影响主流程代码。

## LCEL 不是什么

理解 LCEL 的边界同样重要：

**LCEL 不是一门新语言**。它不是要替代 Python，而是一套基于 Python 的**组件编排协议**。你写的仍然是标准 Python 代码，只是利用了 LangChain 提供的 `|` 操作符和 Runnable 接口来做组件组合。

**LCEL 不强制你必须用它**。对于简单的应用，直接调用 `.invoke()` 完全没问题。但当你的应用开始变复杂——多个组件需要组合、条件分支、并行执行、添加中间件时——LCEL 的优势就会越来越明显。

**LCEL 不限制你的架构选择**。你可以只用 LCEL 组合一个小 Chain，也可以用它搭建包含几十个组件的复杂 Agent 系统。它的设计是增量式的——从简单到复杂都可以优雅地表达。

接下来的一节，我们将深入 LCEL 的核心原语 **Runnable 接口**，理解它提供的每一个方法及其使用场景。
