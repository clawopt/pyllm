---
title: 常用记忆类型实战
description: RunnableWithMessageHistory 集成、各种 Memory 的代码示例与对比
---
# 常用记忆类型实战

上一节我们从概念层面介绍了 LangChain 的各类记忆组件。这一节我们动手写代码，把 Memory 真正集成到 Chain 中，构建出有记忆能力的对话应用。同时我们会重点介绍 LangChain v1.0 推荐的新方式——`RunnableWithMessageHistory`。

## RunnableWithMessageHistory：v1.0 推荐方式

在 LangChain v1.0 的 LCEL 架构中，给 Chain 添加记忆能力不再使用旧式的 `LLMChain(memory=...)` 模式，而是通过 **`RunnableWithMessageHistory`** 包装器来实现。这是当前官方推荐的做法。

### 基本用法

先看一个最简单的例子：

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain.chat_history import InMemoryChatMessageHistory

load_dotenv()

# 1. 创建基础 Chain（没有记忆）
chat = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有帮助的助手。请简洁地回答问题。"),
    MessagesPlaceholder(variable_name="chat_history"),   # 历史消息的插槽
    ("human", "{question}")
])

base_chain = prompt | chat

# 2. 用 RunnableWithMessageHistory 包装
from langchain.runnables.history import RunnableWithMessageHistory

# 存储每个会话的历史消息
store = {}   # key=session_id, value=InMemoryChatMessageHistory

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

with_memory = RunnableWithMessageHistory(
    runnable=base_chain,
    get_session_history=get_session_history,
    input_messages_key="question",
    history_messages_key="chat_history"
)

# 3. 使用 — 通过 session_id 区分不同会话
r1 = with_memory.invoke(
    {"question": "我叫张三"},
    config={"configurable": {"session_id": "user_001"}}
)
print(f"AI: {r1.content}")
# AI: 你好，张三！很高兴认识你。

r2 = with_memory.invoke(
    {"question": "我叫什么名字？"},
    config={"configurable": {"session_id": "user_001"}}
)
print(f"AI: {r2.content}")
# AI: 根据之前的对话，您叫张三。
```

这次模型正确地记住了"张三"！让我们拆解这段代码中的关键部分：

**MessagesPlaceholder（第 10 行）**：在提示词模板中插入一个动态插槽，它的值会在运行时由 `RunnableWithMessageHistory` 自动填充为该会话的历史消息列表。这就是为什么模板里需要 `chat_history` 这个变量名——它和后面 `history_messages_key="chat_history"` 对应。

**get_session_history 函数（第 23-27 行）**：这是一个你自定义的函数，职责是根据 `session_id` 返回对应会话的消息历史对象。上面的例子用了内存字典 (`store={})` 来存储历史，但你可以轻松替换为 Redis、数据库等持久化后端——只需要修改这个函数的实现即可。

**config 参数（第 35 行）**：每次调用时通过 `config["configurable"]["session_id"]` 指定当前属于哪个会话。不同的 session_id 意味着完全独立的对话上下文：

```python
# 用户 A 的对话
r_a1 = with_memory.invoke({"question": "我叫张三"}, config={"configurable": {"session_id": "user_A"}})
r_a2 = with_memory.invoke({"question": "我叫什么？"}, config={"configurable": {"session_id": "user_A"}})
print(r_a2.content)  # 您叫张三

# 用户 B 的对话 — 完全独立
r_b1 = with_memory.invoke({"question": "我叫什么？"}, config={"configurable": {"session_id": "user_B"}})
print(r_b1.content)  # 我不知道您的名字（因为 user_B 从未自报家门）
```

### 完整的对话循环

把上面的代码包装成一个可交互的对话程序：

```python
"""
memory_demo.py — 有记忆的对话助手
"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chat_history import InMemoryChatMessageHistory
from langchain.runnables.history import RunnableWithMessageHistory

load_dotenv()

chat = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个简洁有用的 Python 助教。回答要准确且简短。"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

base_chain = prompt | chat

store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

chain_with_memory = RunnableWithMessageHistory(
    runnable=base_chain,
    get_session_history=get_session_history,
    input_messages_key="question",
    history_messages_key="chat_history"
)

def main():
    print("=" * 50)
    print("🤖 Python 助教（输入 '退出' 结束）")
    print("=" * 50)

    while True:
        question = input("\n你: ").strip()
        if not question:
            continue
        if question.lower() in ["退出", "exit", "quit"]:
            print("再见！")
            break

        response = chain_with_memory.invoke(
            {"question": question},
            config={"configurable": {"session_id": "demo_user"}}
        )
        print(f"\n助教: {response.content}")

if __name__ == "__main__":
    main()
```

运行效果：

```
==================================================
🤖 Python 助教（输入 '退出' 结束）
==================================================

你: 什么是装饰器？

助教: 装饰器是一种语法糖，用于在不修改原函数代码的前提下扩展其功能。
       常见用法包括计时、日志记录和权限验证。

你: 给个例子

助教: 这是一个计时装饰器：
       ```python
       import time
       def timer(func):
           def wrapper(*args, **kwargs):
               start = time.time()
               result = func(*args, **kwargs)
               print(f"{func.__name__} 耗时 {time.time()-start:.4f}s")
               return result
           return wrapper
       ```

你: 那异步编程呢？

助教: Python 的异步编程基于 asyncio 库，核心是 async/await 语法和事件循环机制。
       它可以在单线程内实现并发 I/O 操作。

你: 刚才那个装饰器能用在异步函数上吗？

助教: 可以！需要做一些调整：
       ```python
       import asyncio
       import time
       
       async def async_timer(func):
           async def wrapper(*args, **kwargs):
               start = time.time()
               result = await func(*args, **kwargs)
               print(f"{func.__name__} 耗时 {time.time()-start:.4f}s")
               return result
           return wrapper
       ```
       主要区别是 wrapper 和内部调用都需要加 await。
```

注意最后的问题——用户说"刚才那个装饰器"，模型正确地理解成了第一轮讨论的计时装饰器，并给出了适配异步函数的版本。这就是 Memory 在发挥作用：**4 轮对话的完整上下文都被保留着，模型能够跨轮次地理解指代关系**。

## 不同 Memory 类型的实际效果对比

让我们用同一个对话场景来对比不同 Memory 类型的工作效果。假设我们进行了以下 6 轮对话：

```python
conversation_rounds = [
    ("你好", "你好！有什么可以帮你的？"),
    ("我叫李明", "你好李明！很高兴认识你"),
    ("我是一名后端工程师", "后端开发是个很有挑战性的领域"),
    ("我在学LangChain", "很好的选择，LangChain是LLM开发的利器"),
    ("帮我写个RAG示例", "好的，这是一个基本的RAG实现..."),
    ("刚才说的RAG怎么加记忆", "..."),  # 关键测试：能否回溯到第5轮
]
```

### BufferMemory 效果

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(return_messages=True)
for u, a in conversation_rounds:
    memory.save_context({"input": u}, {"output": a})

history = memory.load_memory_variables({})['history']
print(f"消息条数: {len(history)}")
print(f"最后一条: {history[-1].content[:60]}")
```

输出：
- 保留了全部 12 条消息（6 轮 × 2 条/轮）
- 第 6 轮问"刚才说的RAG"，模型能看到第 5 轮的 RAG 示例内容 → 能正确回答
- 但 token 消耗最大

### WindowMemory (k=3) 效果

```python
from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(k=3, return_messages=True)
for u, a in conversation_rounds:
    memory.save_context({"input": u}, {"output": a})

history = memory.load_memory_variables({})['history']
print(f"消息条数: {len(history)}")
# 只保留了最近 3 轮 = 6 条消息
```

输出：
- 只保留第 4~6 轮（"学LangChain"、"写RAG"、"RAG怎么加记忆"）
- 第 6 轮问"刚才说的RAG" → 第 5 轮还在窗口内 → **能正确回答**
- 但如果问"我叫什么名字？" → 第 2 轮已被丢弃 → **无法回答**

### SummaryMemory 效果

```python
from langchain.memory import ConversationSummaryMemory
from langchain_openai import ChatOpenAI

chat = ChatOpenAI(model="gpt-4o-mini", temperature=0)
memory = ConversationSummaryMemory(llm=chat)
for u, a in conversation_rounds:
    memory.save_context({"input": u}, {"output": a})

summary = memory.load_memory_variables({})['history']
print(summary)
```

输出类似：
```
用户自我介绍叫李明，是一名正在学习 LangChain 的后端工程师。
用户请求了一个 RAG 示例代码，然后询问如何在 RAG 中添加记忆功能。
```

- 全部 6 轮压缩成了 ~40 字的摘要
- 第 6 轮问"刚才说的RAG" → 摘要中提到了 RAG → 大概能回答但缺少具体代码细节
- Token 消耗最低

### 如何选择

从上面三种效果的对比可以看出选择逻辑：

| 你的需求 | 推荐类型 |
|---------|---------|
| 对话不超过 10 轮，需要精确回溯 | `BufferMemory` |
| 对话中等长度，只需近期精确回溯 | `BufferWindowMemory(k=5)` |
| 对话很长，只需大概了解之前聊过什么 | `SummaryMemory` |
| 生产环境，兼顾长期+短期 | `ConversationSummaryBufferMemory` |

## 自定义 Memory 后端

到目前为止，我们的所有例子都用的是 **InMemoryChatMessageHistory**——即把对话历史保存在进程内存中的字典里。这意味着程序重启后所有记忆都会丢失。在生产环境中，你需要一个持久化的存储后端。

### 基于 Redis 的持久化记忆

```python
import redis
from langchain_community.chat_message_histories import RedisChatMessageHistory

def get_redis_session_history(session_id: str):
    return RedisChatMessageHistory(
        session_id=session_id,
        url="redis://localhost:6379/0",
        ttl=3600   # 1 小时后自动过期
    )

with_memory = RunnableWithMessageHistory(
    runnable=base_chain,
    get_session_history=get_redis_session_history,
    input_messages_key="question",
    history_messages_key="chat_history"
)
```

改动只有一处——把 `get_session_history` 函数的返回值从 `InMemoryChatMessageHistory` 换成了 `RedisChatMessageHistory`。Chain 的其余代码一行不用改。这就是 LangChain 统一接口设计的威力。

Redis 方案的优势：
- **持久化**：程序重启不丢失数据
- **分布式**：多个服务实例可以共享同一份记忆
- **自动过期**：`ttl` 参数设置后，长时间不活跃的会话自动清理
- **高性能**：Redis 的读写延迟在亚毫秒级

### 基于 SQLite 的本地持久化

如果你不想依赖 Redis 服务，SQLite 是一个零配置的轻量替代方案：

```python
from langchain_community.chat_message_histories import SQLChatMessageHistory

def get_sqlite_session_history(session_id: str):
    return SQLChatMessageHistory(
        session_id=session_id,
        connection_string="sqlite:///chat_history.db"
    )
```

数据会被保存到当前目录下的 `chat_history.db` 文件中。适合单机部署的小型应用。

## 记忆与 RAG 结合

在实际项目中，Memory 经常和 RAG 一起使用——既有对话记忆又有知识库检索。组合方式很直观：

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# RAG 检索器
vectorstore = Chroma(persist_directory="./chroma_db",
                     embedding_function=OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_type="similarity", k=3)

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是基于以下知识库回答问题的助手。如果没有相关信息就说不知道。"),
    ("human", "{context}\n\n{input}")   # 注意这里用的是 LCEL 标准格式
])

rag_chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt
    | ChatOpenAI(model="gpt-4o-mini", temperature=0)
)

# 用 RunnableWithMessageHistory 包装 RAG Chain
chain_with_rag_and_memory = RunnableWithMessageHistory(
    runnable=rag_chain,
    get_session_history=get_session_history,
    input_messages_key="input"
)
```

这样你的对话助手就同时拥有了两种"记忆"：
- **短期记忆（Memory）**：记住这轮对话中用户说过什么
- **长期知识（RAG）**：从知识库中检索相关文档

两者的协作方式是：每次用户提问时，系统同时做两件事——去向量库里搜相关文档 + 从 Memory 里加载历史对话——然后把两者一起组装进提示词发给 LLM。

## 常见误区与调试技巧

**误区一：混淆 session_id 和用户 ID**。`session_id` 代表一次"对话会话"，不是"用户身份"。同一个用户可能开启多个会话（比如同时在网页版和 APP 上聊天），每个会话应该有不同的 session_id。如果你想实现"跨设备的统一记忆"，需要在应用层自己管理 session_id 到 user_id 的映射。

**误区二：Memory 会自动处理超长对话**。不会。如果你的对话历史超过了模型的上下文窗口限制，Memory 不会自动截断——它会老老实实地把全部历史塞进去，然后 API 报错。你必须主动选择带有限制机制的 Memory 类型（如 Window / TokenBuffer / Summary），或者在 `get_session_history` 中自己实现截断逻辑。

**调试技巧：打印实际发送给模型的内容**

当你发现模型的回答不符合预期时，最有效的排查方法是查看它实际收到了什么输入：

```python
# 查看某个会话的完整历史
history_store = store.get("demo_user")
if history_store:
    for msg in history_store.messages:
        role = type(msg).__name__.replace("Message", "")
        print(f"[{role}] {msg.content[:100]}")
```

这能帮你快速定位问题是出在 Memory（历史没存对）、Prompt（提示词写得不好）还是 Model（模型本身的理解偏差）。

下一节我们将综合运用本章所学，构建一个功能完整的、带持久化记忆的智能对话助手。
