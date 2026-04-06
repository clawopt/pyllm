---
title: LCEL 的核心原语：Runnable 接口
description: Runnable 的 invoke/batch/stream 方法、RunnableLambda、RunnablePassthrough、RunnableConfig
---
# LCEL 的核心原语：Runnable 接口

上一节我们从宏观角度理解了 LCEL 的价值。这一节我们将深入它的核心——**Runnable 接口**。理解了 Runnable，你就掌握了 LCEL 的一半功力。

## Runnable 是什么

在 LangChain v1.0 中，**Runnable** 是所有可运行组件的统一基类。你可以把它理解为 LangChain 世界里的"接口标准"——只要一个类实现了 Runnable，它就能：

1. 用 `|` 管道操作符和其他组件串联
2. 统一地调用 `invoke()` / `batch()` / `stream()` 
3. 被 `RunnableWithMessageHistory` 包装以获得记忆能力
4. 参与 Agent 的工具系统

几乎所有你在 LangChain 中用到的组件都是（或可以包装成）Runnable：

```python
from langchain_openai import ChatOpenAI        # ✅ Runnable
from langchain_core.prompts import ChatPromptTemplate   # ✅ Runnable
from langchain_core.output_parsers import StrOutputParser # ✅ Runnable
from langchain_community.vectorstores import Chroma       # ✅ 可转为 Retriever(Runnable)
```

## Runnable 的核心方法

### invoke()：同步调用

最基础的调用方式——传入输入，等待输出返回：

```python
from langchain_openai import ChatOpenAI

chat = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 单次调用
response = chat.invoke("说一句话")
print(response.content)

# 带结构化输入的调用
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("用{style}的风格解释{topic}")
chain = prompt | chat

result = chain.invoke({"style": "幽默", "topic": "量子计算"})
print(result.content)
```

`invoke()` 是阻塞式的——它会等到 LLM 完成全部生成后才返回。对于简单的问答这没问题，但对于长文本生成，用户可能要等好几秒。

### stream()：流式输出

`stream()` 让模型一边生成一边推送结果，而不是等全部完成才一次性返回：

```python
# 流式输出 — 逐 token 返回
for chunk in chat.stream("用三句话介绍 Python"):
    print(chunk.content, end="", flush=True)
print()
```

运行效果：
```
Python 是一门由 Guido van Rossum 于 1991 年创建的高级编程语言...
```

文字会像打字一样逐个出现。流式输出对用户体验至关重要——没有用户愿意盯着空白屏幕等 10 秒钟。

`stream()` 在 Chain 中同样适用：

```python
chain = prompt | chat | StrOutputParser()

for chunk in chain.stream({"topic": "RAG", "style": "简洁"}):
    print(chunk, end="", flush=True)
```

注意这里 `StrOutputParser` 放在 stream 链中也能工作——它会对每个 chunk 做 parse 操作。

### batch()：批量调用

当你需要同时处理多个独立请求时，`batch()` 比循环调用 `invoke()` 高效得多：

```python
questions = [
    {"topic": "装饰器", "style": "简洁"},
    {"topic": "GIL", "style": "详细"},
    {"topic": "异步编程", "style": "幽默"}
]

results = chain.batch(questions)

for i, r in enumerate(results):
    print(f"\nQ{i+1}: {r[:60]}...")
```

`batch()` 内部会自动并发执行这些调用（如果底层支持的话），比串行快很多。

## RunnablePassthrough：原样传递数据

这是 LCEL 中使用频率最高的组件之一。它的作用是**原封不动地把输入传递到输出**，不做任何修改：

```python
from langchain_core.runnables import RunnablePassthrough

passthrough = RunnablePassthrough()

result = passthrough.invoke({"name": "张三", "age": 25})
print(result)
# {'name': '张三', 'age': 25}
```

看起来好像什么都没做？但在管道组合中它有独特的用途——**让同一个输入同时流向多个分支**：

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

chat = ChatOpenAI(model="gpt-4o-mini")

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个助手"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

# RunnablePassthrough 让 question 同时作为 prompt 的变量和最终输出的一部分
chain = (
    {
        "history": lambda x: [],          # 历史消息（空列表）
        "question": RunnablePassthrough()  # 用户问题原样传递
    }
    | prompt
    | chat
)

result = chain.invoke("什么是 RAG？")
# question="什么是 RAG？" 被同时传给了 prompt 和后续流程
```

这在构建带记忆的 Chain 时特别有用——用户的原始问题需要出现在两个地方：提示词模板和最终的输出上下文中。

## RunnableLambda：包装任意函数为 Runnable

有时候你需要把一段自定义逻辑插入到 Chain 中——比如数据转换、格式化、条件判断等。`RunnableLambda` 可以把任意 Python 函数变成 Runnable：

```python
from langchain_core.runnables import RunnableLambda

# 把普通函数变成 Runnable
uppercase = RunnableLambda(lambda x: x.upper())
length_counter = RunnableLambda(lambda x: f"长度: {len(x)}")

print(uppercase.invoke("hello"))      # "HELLO"
print(length_counter.invoke("hello"))   # "长度: 5"
```

### 实际使用场景

```python
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

chat = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 场景一：在 Chain 中做数据预处理
chain = (
    RunnableLambda(lambda x: x["text"].lower())   # 先转小写
    | chat                                          # 再发给模型
)

result = chain.invoke({"text": "Hello WORLD"})
# 模型收到的是 "hello world"

# 场景二：在 Chain 中做后处理
chain = (
    chat
    | RunnableLambda(lambda x: x.content.split("\n")[0])  # 只取第一行
    | RunnableLambda(lambda x: f"摘要: {x}")           # 加前缀
)

# 场景三：动态选择模型参数
def smart_model_router(input_dict):
    """根据问题复杂度选择不同模型"""
    if len(input_dict["question"]) > 50:
        return ChatOpenAI(model="gpt-4o")      # 复杂问题用强模型
    else:
        return ChatOpenAI(model="gpt-4o-mini")  # 简单问题用快模型

smart_chain = RunnableLambda(smart_model_router) | ...
```

## RunnableConfig：访问配置信息

有时候你的函数需要访问调用时的配置（如 session_id）：

```python
from langchain_core.runnables import RunnableConfig

def get_session_id(config: RunnableConfig) -> str:
    """从配置中提取 session_id"""
    return config.configurable.get("session_id", "default")

config_aware = RunnableLambda(get_session_id)

result = config_aware.invoke(
    "dummy",
    config=RunnableConfig(configurable={"session_id": "user_123"})
)
# result = "user_123"
```

这在构建需要感知会话信息的中间件时非常有用。

## 组合模式总结

到目前为止我们见过的 LCEL 组合方式：

| 模式 | 语法 | 含义 |
|------|------|------|
| **线性管道** | `A \| B \| C` | A→B→C 顺序执行 |
| **并行** | `RunnableParallel(a=A, b=B)` | A 和 B 并行执行 |
| **原样传递** | `RunnablePassthrough()` | 输入直接透传 |
| **函数包装** | `RunnableLambda(fn)` | 任意函数变 Runnable |
| **分支路由** | `RunnableBranch(...)` | 条件选择执行路径 |

下一节我们将深入学习**并行化（RunnableParallel）**和**条件分支（RunnableBranch）**这两个高级组合模式的详细用法和实战示例。
