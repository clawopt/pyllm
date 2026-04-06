---
title: ChatEngine：多轮对话引擎
description: ChatMode 与 QueryMode 的区别、记忆管理、上下文窗口优化、对话式 RAG 的最佳实践
---
# ChatEngine：多轮对话引擎

到目前为止，我们使用的 `query()` 方法都是**无状态的**——每次调用都是独立的，Query Engine 不记得你之前问过什么问题。这在"一问一答"的场景下完全够用，但在真实的交互场景中，用户往往是多轮对话的形式：

```
用户: S1 这款产品怎么样？
助手: S1 是我们最新推出的智能音箱，支持语音控制智能家居...

用户: 那它的价格呢？          ← "它"指的是 S1
助手: S1 的售价为 299 元...

用户: 和 S2 比呢？            ← 比较对象是 S1 和 S2
助手: 相比 S2（售价 399 元），S1 在价格上更有优势...
```

注意到了吗？第二轮中的"它"和第三轮中的"比较"都依赖于前面的对话上下文。如果每次查询都是独立的，系统就无法理解这些指代关系。

LlamaIndex 通过 **ChatEngine** 来解决这个问题——它在 `query_engine` 的基础上增加了**对话历史管理**能力。

## ChatEngine vs QueryEngine：核心区别

| 特性 | QueryEngine | ChatEngine |
|------|------------|------------|
| **状态** | 无状态（每次独立） | 有状态（维护对话历史） |
| **核心方法** | `query(question)` | `chat(message)` |
| **上下文** | 只有当前查询 | 当前查询 + 历史对话 |
| **适用场景** | 单次问答 | 多轮对话、交互式应用 |
| **返回类型** | `Response` | `ChatResponse` |

## 基础用法

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)

chat_engine = index.as_chat_engine(
    chat_mode="context",       # 最常用的模式
    similarity_top_k=5,
    verbose=True,
)

# 第一轮
response1 = chat_engine.chat("S1 产品的主要功能有哪些？")
print(f"助手: {response1.response}")

# 第二轮（自动携带第一轮的上下文）
response2 = chat_engine.chat("那它的价格是多少？")
print(f"助手: {response2.response}")

# 第三轮
response3 = chat_engine.chat("和竞品比有什么优势？")
print(f"助手: {response3.response}")
```

### 内部工作机制

当你在第二轮问"那它的价格是多少？"时，ChatEngine 实际发送给 LLM 的 Prompt 大致如下：

```
System: 你是一个有帮助的基于文档的问答助手。请根据以下上下文回答问题。

=== 对话历史 ===
User: S1 产品的主要功能有哪些？
Assistant: S1 是一款智能音箱产品，主要功能包括：
1. 语音控制智能家居设备...
2. 高品质音频播放...
3. 支持多种无线连接协议...

=== 相关文档上下文 ===
[检索到的与当前查询相关的 Node 内容]

=== 当前问题 ===
User: 那它的价格是多少？
```

LLM 现在能看到完整的对话历史了——它知道"它"指的是 S1（从第一轮的上下文中推断），因此能给出准确且连贯的回答。

## Chat Mode 选择

`chat_mode` 参数决定了 ChatEngine 如何处理对话历史和检索策略：

### mode="context"（默认推荐）

这是最常用的模式。它的行为是：
1. 将**所有历史对话**作为上下文追加到当前查询中
2. 用**组合后的查询**（历史 + 当前问题）去检索相关内容
3. 将检索结果 + 全部历史 + 当前问题一起发给 LLM 生成答案

```python
chat_engine = index.as_chat_engine(
    chat_mode="context",
    memory_type="token_limit",   # 自动管理上下文长度
    system_prompt=(
        "你是公司的智能客服助手。请基于知识库信息友好地回答客户问题。"
        "如果知识库中没有相关信息，请诚实告知并建议联系人工客服。"
    ),
)
```

**优点：** LLM 能看到完整的对话历史，理解能力强。
**缺点：** 对话越长，消耗的 token 越多；历史中可能包含与当前问题无关的内容（噪音）。

### mode="condense_question"

这种模式下，ChatEngine 会先用 LLM 把当前问题结合历史"压缩"成一个独立的、自包含的问题，然后用这个压缩后的问题去做检索：

```
历史: User: S1 功能？ Assistant: S1 是智能音箱...
当前: User: 价格多少？

→ LLM 压缩 → "智能音箱 S1 产品的价格是多少？"

→ 用这个自包含的问题去检索（不需要带历史）
```

**优点：** 检索更精准（查询更完整），token 消耗更低（只把压缩后的问题而非全部历史传给 Retriever）。
**缺点：** 多了一次 LLM 调用（用于压缩），且压缩过程可能丢失一些细微的语义。

### mode="react"

最复杂的模式，让 Agent 自主决定每一步该做什么（检索、回答、或追问澄清）：

```python
chat_engine = index.as_chat_engine(
    chat_mode="react",
    verbose=True,  # 打印 Agent 的推理过程
)
```

**优点：** 最灵活，Agent 可以在需要时主动搜索、在信息不足时反问用户。
**缺点：** 最慢（可能需要多轮 Agent 推理），成本最高，行为较难预测。

**选择建议：**
- 大多数场景 → `mode="context"`
- 需要控制 token 成本 → `mode="condense_question"`
- 复杂的多步推理需求 → `mode="react"`

## 记忆管理

对话历史不可能无限增长——LLM 的上下文窗口有限制（GPT-4o-mini 是 128K tokens，但实际使用中超过 4-8K tokens 后效果就开始下降）。ChatEngine 提供了几种自动管理对话长度的策略：

### token_limit 记忆（默认）

```python
from llama_index.core.memory import ChatMemoryBuffer

chat_engine = index.as_chat_engine(
    chat_mode="context",
    memory=ChatMemoryBuffer(token_limit=3000),  # 最多保留 3000 token 的历史
)
```

当对话历史超过 3000 tokens 时，最早的对话会被自动裁剪掉，只保留最近的对话。这是一种简单有效的 FIFO（先进先出）策略。

### 自定义记忆策略

如果默认的 token_limit 不满足你的需求，可以实现自定义的记忆管理：

```python
from llama_index.core.memory import BaseMemory
from typing import List, Dict, Optional


class SlidingWindowMemory(BaseMemory):
    """滑动窗口记忆 — 保留最近 N 轮对话"""

    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.messages: List[Dict] = []

    def get(self, **kwargs) -> List[Dict]:
        return self.messages[-self.window_size * 2:]  # 每轮有 user+assistant 两条

    def put(self, message: Dict[str, str], **kwargs):
        self.messages.append(message)

    def set(self, messages: List[Dict], **kwargs):
        self.messages = messages

    def clear(self):
        self.messages.clear()


chat_engine = index.as_chat_engine(
    chat_mode="context",
    memory=SlidingWindowMemory(window_size=8),  # 保留最近 8 轮
)
```

### 关键词触发的选择性记忆

更高级的策略是根据当前问题的内容动态决定需要保留多少历史：

```python
class SmartContextMemory(BaseMemory):
    """智能上下文记忆 — 根据查询相关性筛选历史"""

    def __init__(self, max_tokens: int = 2000):
        self.max_tokens = max_tokens
        self.all_messages: List[Dict] = []

    def get(self, last_message: Optional[str] = None, **kwargs) -> List[Dict]:
        if not last_message or not self.all_messages:
            return self.all_messages

        relevant = []
        total_tokens = 0

        for msg in reversed(self.all_messages):  # 从最近的消息往前遍历
            msg_tokens = len(msg["content"]) // 3  # 粗略估算 token 数
            if total_tokens + msg_tokens > self.max_tokens:
                break
            relevant.insert(0, msg)  # 保持时间顺序
            total_tokens += msg_tokens

        return relevant

    def put(self, message: Dict[str, str], **kwargs):
        self.all_messages.append(message)

    def set(self, messages: List[Dict], **kwargs):
        self.all_messages = messages

    def clear(self):
        self.all_messages.clear()
```

## 流式输出与 ChatEngine

ChatEngine 同样支持流式输出，这对于实时聊天界面非常重要：

```python
chat_engine = index.as_chat_engine(
    chat_mode="context",
    streaming=True,
)

streaming_response = chat_engine.chat("介绍一下这款产品")

for chunk in streaming_response.response_gen:
    print(chunk, end="", flush=True)
```

在 Web 应用中，你可以通过 Server-Sent Events (SSE) 或 WebSocket 将这些增量文本推送到前端，实现类似 ChatGPT 的逐字显示效果。

## 多用户会话隔离

在生产环境中，一个 ChatEngine 实例通常同时服务多个用户。每个用户的对话历史必须是**相互隔离**的：

```python
from llama_index.core.chat_engine import CondensePlusContextChatEngine

class SessionManager:
    """管理多个用户的独立会话"""

    def __init__(self, index):
        self.index = index
        self.sessions: Dict[str, dict] = {}

    def get_or_create_session(self, session_id: str):
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "chat_engine": self.index.as_chat_engine(
                    chat_mode="context",
                    memory=ChatMemoryBuffer(token_limit=3000),
                ),
                "created_at": datetime.now(),
            }
        return self.sessions[session_id]

    def chat(self, session_id: str, message: str) -> str:
        session = self.get_or_create_session(session_id)
        response = session["chat_engine"].chat(message)
        return response.response

    def clear_session(self, session_id: str):
        if session_id in self.sessions:
            self.sessions[session_id]["chat_engine"].memory.clear()


# 使用
manager = SessionManager(index)

# 用户 A 的对话
resp_a1 = manager.chat("session_abc123", "S1 怎么样？")
resp_a2 = manager.chat("session_abc123", "价格呢？")  # 能看到 A 的上一轮

# 用户 B 的对话（完全独立）
resp_b1 = manager.chat("session_xyz789", "S1 怎么样？")
# B 看不到 A 的任何对话历史
```

每个 `session_id` 对应一个独立的 ChatEngine 实例和独立的记忆缓冲区，确保了用户之间的数据隔离。

## 常见误区

**误区一:"ChatEngine 就是 QueryEngine 加了个历史记录"。** 不完全是。ChatEngine 还改变了**检索策略**——它会将历史对话作为上下文来影响检索结果（mode="context"）或重写查询（mode="condense_question"）。这意味着同样的"价格多少？"这个问题，在有历史和无历史的上下文中可能会检索到完全不同的文档。

**误区二:"对话历史越多越好"。** 不是的。过长的历史会带来三个问题：(1) token 成本线性增长；(2) 早期不相关的对话占据宝贵的上下文空间；(3) LLM 可能被早期信息"带偏"，对当前问题的关注不足。**设置合理的 token 上限（如 2000-4000）并在必要时主动总结或清理历史。**

**误区三:"所有对话都应该用 ChatEngine"。** 如果你的应用本质上是单次问答（如搜索框式的 FAQ 系统），QueryEngine 反而更合适——没有记忆管理的开销，响应更快，架构更简单。**只在真正需要多轮交互的场景中使用 ChatEngine。**
