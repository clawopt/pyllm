---
title: 模型类型：LLM vs. 聊天模型
description: 理解 LangChain 中两种模型接口的区别、适用场景和迁移指南
---
# 模型类型：LLM vs. 聊天模型

在第二章的"第一个应用"中，我们直接使用了 `ChatOpenAI` 来调用 GPT-4o mini，一切看起来都很自然。但如果你翻阅 LangChain 的早期文档或一些老教程，你会发现还有另一个类叫 `OpenAI`（注意没有 `Chat` 前缀），它对应的是另一种模型接口。这两个东西到底有什么区别？什么时候该用哪个？这一节我们就来彻底搞清楚这个问题。

## LLM 类：文本补全接口

LangChain 中的 `LLM` 类（对应 `langchain_openai.OpenAI`）封装的是 OpenAI 的 **Completions API**，也就是最早的 **文本补全（Text Completion）** 接口。它的核心逻辑非常简单：你给它一段文字，它会基于这段文字**继续往下写**。

```python
from langchain_openai import OpenAI

llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0)

response = llm.invoke("Python 是一门")
print(repr(response))
# ' 编程语言，由 Guido van Rossum 于 1991 年首次发布。\n'
```

看到发生了什么吗？我们只给了 "Python 是一门"，模型自动把后面的话补全了——这就是"补全"的含义。这种接口的工作方式就像你在写代码时按 Tab 键触发自动补全一样：**模型看到的全部内容都是"上文"，它的任务是预测接下来最可能出现的文字**。

这个接口的设计哲学来自 GPT 系列模型的本质——GPT 的全称就是 Generative Pre-trained Transformer（生成式预训练变换器），它的训练目标就是在给定前文的情况下预测下一个 token。所以 Completions API 可以说是对 GPT 模型能力最直接的暴露。

但在实际使用中，纯文本补全有一个明显的局限：**你很难精确控制模型的"角色"或"行为模式"**。比如你想让模型扮演一个 Python 助教，用 Completions API 你只能把角色设定混在输入文本里：

```python
response = llm.invoke(
    "你是一个 Python 助教。请用简洁的语言回答问题。\n\n"
    "用户：什么是装饰器？\n\n助手："
)
print(response)
```

这种方式能工作，但它很脆弱——你需要手动拼接字符串、管理格式、处理边界情况，而且不同模型对这种"提示词注入"方式的响应一致性也不高。

## ChatModel 类：对话消息接口

`ChatModel` 类（对应 `langchain_openai.ChatOpenAI`）封装的是 OpenAI 的 **Chat Completions API**，这是目前主流的、也是 LangChain v1.0 推荐使用的接口。与 Completions API 不同，Chat API 的输入不是一段纯文本，而是一个 **消息列表（message list）**，每条消息都有一个明确的角色标签：

```python
from langchain_openai import ChatOpenAI

chat = ChatOpenAI(model="gpt-4o-mini", temperature=0)

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

messages = [
    SystemMessage(content="你是一个 Python 助教，回答要简洁明了"),
    HumanMessage(content="什么是装饰器？")
]

response = chat.invoke(messages)
print(response.content)
# 装饰器是 Python 中的一种语法糖，用于在不修改函数代码的前提下...
```

这里的关键区别在于：**ChatModel 把"谁在说话"这个信息显式地编码到了数据结构中**。`SystemMessage` 代表系统指令（设定角色和行为规则）、`HumanMessage` 代表用户的输入、`AIMessage` 代表模型之前的回复（用于多轮对话）。这种结构化的方式比纯文本拼接要可靠得多，因为模型可以清楚地知道每段话是谁说的、应该怎么理解。

让我们通过一个多轮对话的例子来感受一下 ChatModel 的优势：

```python
messages = [
    SystemMessage(content="你是一个 Python 助教"),
    HumanMessage(content="什么是装饰器？"),
]

response1 = chat.invoke(messages)
print(f"第一轮回复: {response1.content}")

# 把模型的回复追加到消息历史中，继续第二轮对话
messages.append(AIMessage(content=response1.content))
messages.append(HumanMessage(content="能给我一个具体例子吗？"))

response2 = chat.invoke(messages)
print(f"\n第二轮回复: {response2.content}")
```

输出大概是这样的：

```
第一轮回复: 装饰器是一种特殊的函数，它可以修改其他函数的行为...

第二轮回复: 当然！这是一个简单的计时装饰器例子：
import time
def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} 耗时 {time.time()-start:.2f}s")
        return result
    return wrapper
```

注意看我们是怎样构建多轮对话的：**每轮对话结束后，把 AI 的回复作为 `AIMessage` 追加到消息列表中，再把新的用户问题作为 `HumanMessage` 追加进去**。这样模型就能"记住"之前的上下文，给出连贯的回答。这是 ChatModel 相对于 LLM 的一个核心优势——**多轮对话的状态管理变得结构化和可预测**。

## 核心区别对比

为了让你更清晰地把握两者的差异，我们把关键区别整理成一张表：

| 特性 | LLM (`OpenAI`) | ChatModel (`ChatOpenAI`) |
|------|----------------|-------------------------|
| 底层 API | Completions API (Legacy) | Chat Completions API (Current) |
| 输入格式 | 纯文本字符串 | 结构化消息列表 `[Message]` |
| 角色控制 | 需要在文本中手动嵌入 | 通过 `SystemMessage` 显式声明 |
| 多轮对话 | 手动拼接历史文本 | 追加 `AIMessage` / `HumanMessage` |
| 函数调用支持 | 不支持 | 原生支持 tool/function calling |
| 流式输出 | 支持 `.stream()` | 支持 `.stream()` |
| LangChain 推荐度 | 已标记为 Legacy | ✅ 首选 |

从这张表可以看出，**ChatModel 在几乎所有维度上都优于 LLM**。这也是为什么 LangChain 在 v0.1 之后逐步将重心转向 ChatModel，并在 v1.0 中把 ChatModel 作为默认推荐的原因。

## 什么时候还需要用 LLM？

虽然 ChatModel 是首选，但 LLM 并非完全没有用武之地。以下场景你可能仍然会碰到 LLM：

**第一，兼容旧代码和旧模型**。如果你维护的项目使用了早期的 LangChain 版本，或者你调用的模型提供商只提供了 Completions 接口而没有 Chat 接口（某些开源模型或定制化部署可能如此），那么 LLM 类仍然是唯一选择。

**第二，纯文本生成任务**。如果你的任务不需要多轮对话、不需要角色设定、只是简单地让模型"续写"一段文字（比如自动生成代码注释、文章摘要续写等），LLM 的简单接口反而更直观：

```python
llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.7)

comments = llm.invoke("""
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
""")

print(comments)
# '\n    对数组进行快速排序。首先检查基准条件...
```

**第三，理解历史代码**。你在阅读别人写的 LangChain 项目时，一定会遇到 `OpenAI` 而不是 `ChatOpenAI` 的情况。了解 LLM 接口的存在和用法，能帮你快速理解这些代码的意图。

## 共同的 Runnable 接口

尽管 LLM 和 ChatModel 的内部实现不同，但它们在 LangChain v1.0 中共享同一个对外接口——**Runnable 接口**。这意味着无论你用的是哪种模型类，调用方式都是统一的：

```python
from langchain_openai import OpenAI, ChatOpenAI

llm = OpenAI(model="gpt-3.5-turbo-instruct")
chat = ChatOpenAI(model="gpt-4o-mini")

# 两种模型都支持相同的调用方式
result_llm = llm.invoke("说一句话")
result_chat = chat.invoke([HumanMessage(content="说一句话")])

# 都支持流式输出
for chunk in llm.stream("讲个笑话"):
    print(chunk, end="")

for chunk in chat.stream([HumanMessage(content="讲个笑话")]):
    print(chunk.content, end="")

# 都支持批量调用
results = llm.batch(["你好", "谢谢", "再见"])
results_chat = chat.batch([
    [HumanMessage(content="你好")],
    [HumanMessage(content="谢谢")],
    [HumanMessage(content="再见")]
])
```

这种统一性是 LangChain v1.0 的设计精髓之一——**不管底层是什么模型，你用同样的方法去调用它**。当你后续学习 Chain（链）、Agent（智能体）等高级概念时，你会越来越 appreciate 这种统一接口带来的便利：你可以随时替换底层的模型实现，而不需要修改上层的业务逻辑。

## 迁移指南：从 LLM 到 ChatModel

如果你手头有使用 `OpenAI`（LLM）的老代码需要迁移到 `ChatOpenAI`（ChatModel），核心改动其实很少。最常见的迁移模式是这样的：

```python
# === 旧代码 (LLM) ===
from langchain_openai import OpenAI
llm = OpenAI(model="gpt-3.5-turbo-instant")
response = llm.invoke("解释一下什么是 RAG")

# === 新代码 (ChatModel) ===
from langchain_openai import ChatOpenAI
chat = ChatOpenAI(model="gpt-4o-mini")
response = chat.invoke([HumanMessage(content="解释一下什么是 RAG")])
```

主要变化只有三处：
1. 导入类名从 `OpenAI` 改为 `ChatOpenAI`
2. `invoke()` 的参数从纯字符串改为 `[HumanMessage(...)]` 包裹
3. 取值方式不变——仍然是 `response.content`

如果你的旧代码还使用了 PromptTemplate（第二章介绍过），迁移会更加无缝，因为 `ChatPromptTemplate` 本身就产出的是 Message 格式，天然适配 ChatModel：

```python
# 旧代码
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate

llm = OpenAI()
template = PromptTemplate.from_template("解释{topic}")
chain = template | llm
result = chain.invoke({"topic": "RAG"})

# 新代码 — 只需换两个类名
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

chat = ChatOpenAI()
template = ChatPromptTemplate.from_template("解释{topic}")
chain = template | chat
result = chain.invoke({"topic": "RAG"})
```

看到了吗？**当使用管道操作符 `|` 组装 Chain 时，模板和模型之间的对接完全是自动的**——你只需要把类名替换掉，其余代码一行不用改。这就是 LangChain 统一接口设计的威力。

## 常见误区

在实际开发中，有几个关于模型选择的常见错误值得特别注意：

**误区一：以为 `ChatOpenAI` 只能做聊天**。名字里的 "Chat" 容易让人误解为这类模型只能用于对话场景。实际上 ChatModel 可以完成任何 LLM 能做的任务——文本分类、摘要、翻译、代码生成等等。"Chat" 指的是接口格式（消息列表），而不是应用场景。

```python
chat = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ChatModel 完全可以做非聊天任务
classification = chat.invoke([HumanMessage(content="""
请判断以下文本的情感倾向，只返回"正面"、"负面"或"中性"：
"这家餐厅的服务态度很好，菜品也很新鲜"
""")])
print(classification.content)  # 正面
```

**误区二：混淆 `response` 和 `response.content`**。无论是 LLM 还是 ChatModel，`invoke()` 返回的对象都不是纯字符串。LLM 返回的是 `LLMResult` 包装对象，ChatModel 返回的是 `AIMessage` 对象。你需要通过 `.content` 属性来提取实际的文本内容。初学者经常忘记这一点，直接 `print(response)` 然后纳闷为什么输出了一堆元信息。

```python
response = chat.invoke([HumanMessage(content="你好")])
print(type(response))       # <class 'langchain_core.messages.ai.AIMessage'>
print(response)             # AIMessage(content='你好！有什么我可以帮你的？', ...)
print(response.content)     # 你好！有什么我可以帮你的？
```

**误区三：忽略 `temperature` 参数的影响**。`temperature` 控制输出的随机性：0 表示确定性输出（相同输入永远得到相同结果），值越高输出越随机/有创意。对于需要稳定可重复结果的任务（如分类、抽取），建议设为 0 或接近 0；对于创作类任务（如写诗、头脑风暴），可以设到 0.7~1.0。

```python
chat_deterministic = ChatOpenAI(model="gpt-4o-mini", temperature=0)
chat_creative = ChatOpenAI(model="gpt-4o-mini", temperature=0.9)

r1 = chat_deterministic.invoke([HumanMessage(content="用一个词形容夏天")])
r2 = chat_deterministic.invoke([HumanMessage(content="用一个词形容夏天")])
print(r1.content == r2.content)  # True — temperature=0 时结果可复现

r3 = chat_creative.invoke([HumanMessage(content="用一个词形容夏天")])
r4 = chat_creative.invoke([HumanMessage(content="用一个词形容夏天")])
print(r3.content == r4.content)  # False — 高温度时每次都不同
```

到这里，我们已经清楚了 LangChain 中两种模型接口的本质区别和各自的使用场景。接下来的一节，我们将深入探讨如何用 **提示词模板（Prompt Template）** 来精细化管理发送给模型的输入——这是构建可靠 LangChain 应用的另一个基础能力。
