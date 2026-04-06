---
title: 实战：构建一个有记忆的智能对话助手
description: 综合运用 Memory 组件，搭建带持久化、多会话管理的生产级对话系统
---
# 实战：构建一个有记忆的智能对话助手

前面三节我们从"为什么需要记忆"出发，学习了 LangChain 的各类 Memory 组件，并在代码中实践了 `RunnableWithMessageHistory` 的基本用法。现在到了本章的高潮部分——我们将综合所有知识，从零搭建一个**功能完整的智能对话助手**。

这个助手将具备以下特性：
- **多会话管理**：不同用户/场景使用独立的对话上下文
- **持久化记忆**：程序重启后不丢失历史（基于 SQLite）
- **可配置角色**：启动时可设定 AI 的人设和风格
- **Token 控制机制**：自动防止历史过长导致 API 报错
- **会话列表与切换**：可以查看所有会话、在会话间切换

## 第一步：项目结构设计

```
chat-assistant/
├── .env                          # API Key
├── .gitignore
├── requirements.txt
├── assistant.py                  # 主程序入口
├── config.py                     # 配置管理
└── chat_history.db               # SQLite 数据库（自动生成）
```

## 第二步：配置模块

```python
# config.py — 集中管理所有配置

import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # 模型配置
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))

    # 记忆配置
    MAX_HISTORY_TOKENS = int(os.getenv("MAX_HISTORY_TOKENS", "2000"))
    SESSION_TTL = int(os.getenv("SESSION_TTL", "86400"))   # 24小时过期

    # 默认人设
    DEFAULT_SYSTEM_PROMPT = (
        "你是一个有帮助、友好且专业的 AI 助手。"
        "回答要简洁准确，不确定的信息要明确说明。"
    )

    # SQLite 路径
    DB_PATH = os.getenv("DB_PATH", "chat_history.db")
```

把所有可调参数集中到一个地方是好的工程实践——后续调整模型、改人设、换 token 限制都只需要改这一个文件。

## 第三步：核心助手类

这是整个项目的核心——一个封装了所有功能的 `ChatAssistant` 类：

```python
# assistant.py — 智能对话助手

import os
from config import Config
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableConfig
from langchain.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory


class ChatAssistant:
    def __init__(
        self,
        system_prompt: str | None = None,
        model_name: str | None = None,
        temperature: float | None = None
    ):
        self.model_name = model_name or Config.MODEL_NAME
        self.temperature = temperature if temperature is not None else Config.TEMPERATURE
        self.system_prompt = system_prompt or Config.DEFAULT_SYSTEM_PROMPT

        # 初始化 LLM
        self.chat = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature
        )

        # 构建提示词模板
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])

        # 构建基础 Chain
        self.base_chain = self.prompt | self.chat

        # 包装为带记忆的 Chain
        self.chain = RunnableWithMessageHistory(
            runnable=self.base_chain,
            get_session_history=self._get_session_history,
            input_messages_key="question",
            history_messages_key="chat_history"
        )

    def _get_session_history(self, session_id: str) -> SQLChatMessageHistory:
        """获取或创建指定会话的历史记录"""
        return SQLChatMessageHistory(
            session_id=session_id,
            connection_string=f"sqlite:///{Config.DB_PATH}"
        )

    def chat(self, question: str, session_id: str = "default") -> str:
        """发送消息并获取回复"""
        response = self.chain.invoke(
            {"question": question},
            config=RunnableConfig(configurable={"session_id": session_id})
        )
        return response.content

    def get_session_messages(self, session_id: str = "default") -> list:
        """获取某个会话的所有历史消息"""
        history = self._get_session_history(session_id)
        return history.messages

    def list_sessions(self) -> list[str]:
        """列出所有存在的会话 ID"""
        import sqlite3
        conn = sqlite3.connect(Config.DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT session_id FROM message_store")
        sessions = [row[0] for row in cursor.fetchall()]
        conn.close()
        return sessions

    def clear_session(self, session_id: str):
        """清除指定会话的历史"""
        import sqlite3
        conn = sqlite3.connect(Config.DB_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM message_store WHERE session_id = ?", (session_id,))
        conn.commit()
        conn.close()
```

让我们逐段理解这个类的设计：

### 构造函数（`__init__`）

构造函数做了三件事：
1. **读取配置**——支持通过参数覆盖默认值，让调用者可以灵活地定制模型和人设
2. **创建 LLM + Prompt + Chain**——和之前学过的标准流程一致
3. **用 `RunnableWithMessageHistory` 包装**——自动注入对话历史到提示词中

关键设计点：`MessagesPlaceholder(variable_name="chat_history")` 这个插槽会在每次调用时被 `RunnableWithMessageHistory` 自动填充为当前会话的历史消息。你不需要手动管理消息列表。

### 会话持久化（`_get_session_history`）

```python
def _get_session_history(self, session_id: str):
    return SQLChatMessageHistory(
        session_id=session_id,
        connection_string=f"sqlite:///{Config.DB_PATH}"
    )
```

这里用了 `SQLChatMessageHistory` 作为后端存储。每次调用这个函数时，它会返回一个对应 `session_id` 的历史记录对象——如果该 session 已存在就加载已有数据，如果不存在就创建一个新的空记录。数据全部保存在 SQLite 文件中，程序重启后依然存在。

### 核心聊天方法（`chat`）

```python
def chat(self, question: str, session_id: str = "default") -> str:
    response = self.chain.invoke(
        {"question": question},
        config=RunnableConfig(configurable={"session_id": session_id})
    )
    return response.content
```

看起来只有几行代码，但背后发生了很多事情：
1. 用户的问题被包装成 `{"question": "..."}`
2. `RunnableWithMessageHistory` 自动调用 `_get_session_history(session_id)` 加载历史
3. 历史消息被填入 `MessagesPlaceholder` 占位符
4. 组装好的完整提示词发送给 LLM
5. LLM 的回复被自动追加到该 session 的历史中
6. 返回回复文本给调用者

整个过程对调用者完全透明——你只需要传问题和 session_id 就行。

## 第四步：交互式主程序

有了 `ChatAssistant` 类之后，主程序就非常简洁了：

```python
# assistant.py（续）— 主程序入口

def print_banner():
    print("=" * 56)
    print("   🤖 智能对话助手 (输入命令进行操作)")
    print("=" * 56)
    print("  /session <name>   切换/创建会话")
    print("  /sessions         查看所有会话")
    print("  /history          查看当前会话历史")
    print("  /clear            清除当前会话")
    print("  /help             显示帮助")
    print("  /exit 或 quit     退出程序")
    print("=" * 56)


def main():
    assistant = ChatAssistant()
    current_session = "default"

    print_banner()
    print(f"\n📌 当前会话: {current_session}")
    print(f"📌 模型: {assistant.model_name} (temperature={assistant.temperature})\n")

    while True:
        try:
            user_input = input("❓ 你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 再见！")
            break

        if not user_input:
            continue

        # 处理斜杠命令
        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if cmd in ["/exit", "/quit"]:
                print("👋 再见！")
                break
            elif cmd == "/help":
                print_banner()
            elif cmd == "/sessions":
                sessions = assistant.list_sessions()
                if sessions:
                    print(f"\n📋 共 {len(sessions)} 个会话:")
                    for s in sessions:
                        marker = " ← 当前" if s == current_session else ""
                        print(f"   • {s}{marker}")
                else:
                    print("\n📋 暂无会话记录")
            elif cmd == "/session":
                if arg:
                    current_session = arg.strip()
                    print(f"✅ 已切换到会话: {current_session}")
                else:
                    print(f"📌 当前会话: {current_session}")
            elif cmd == "/history":
                messages = assistant.get_session_messages(current_session)
                if messages:
                    print(f"\n📜 {current_session} 的对话历史 ({len(messages)} 条):")
                    for msg in messages:
                        role = "🧑" if msg.type == "human" else "🤖"
                        content = msg.content.replace("\n", " ")[:80]
                        print(f"  {role} {content}")
                else:
                    print("\n📜 该会话暂无历史记录")
            elif cmd == "/clear":
                assistant.clear_session(current_session)
                print(f"🗑️  已清除会话: {current_session}")
            else:
                print(f'❓ 未知命令: {cmd}，输入 /help 查看帮助')
            continue

        # 普通对话
        print("\n🤖 助手正在思考...\n")
        try:
            reply = assistant.chat(user_input, session_id=current_session)
            print(f"💬 {reply}\n")
        except Exception as e:
            print(f"⚠️  出错了: {e}\n")


if __name__ == "__main__":
    main()
```

## 第五步：运行测试

```bash
pip install langchain langchain-openai langchain-community langchain-core python-dotenv sqlalchemy
python assistant.py
```

交互效果演示：

```
========================================================
   🤖 智能对话助手 (输入命令进行操作)
========================================================
  /session <name>   切换/创建会话
  /sessions         查看所有会话
  /history          查看当前会话历史
  /clear            清除当前会话
  /help             显示帮助
  /exit 或 quit     退出程序
========================================================

📌 当前会话: default
📌 模型: gpt-4o-mini (temperature=0)

❓ 你: 我叫小明

🤖 助手正在思考...

💬 你好，小明！很高兴认识你。有什么我可以帮助你的吗？

❓ 你: 我在学习 Python 的装饰器

🤖 助手正在思考...

💬 装饰器是 Python 中一个非常强大且优雅的特性！它本质上是一个函数，
用来修改其他函数的行为，而不需要改变原函数的源代码。
核心语法是在函数定义前加上 @decorator_name。

❓ 你: 给我写个计时装饰器的例子

🤖 助手正在思考...

💬 好的，这是一个经典的计时装饰器：

```python
import time

def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"{func.__name__} 执行耗时: {elapsed:.4f} 秒")
        return result
    return wrapper

@timer
def slow_operation():
    time.sleep(1)
    return "完成"

slow_operation()
# 输出: slow_operation 执行耗时: 1.xxxx 秒
```

❓ 你: 刚才那个装饰器能用在异步函数上吗？

🤖 助手正在思考...

💬 可以的！只需要做几点调整：
1. 装饰器和内部函数都要加 async
2. 函数调用需要 await
3. 使用 time.perf_counter() 替代 time.time() 更精确

示例：
```python
import asyncio
import time

async def async_timer(func):
    async def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = await func(*args, **kwargs)
        print(f"{func.__name__} 耗时 {time.perf_counter()-start:.4f}s")
        return result
    return wrapper
```

❓ you: /sessions

📋 共 1 个会话:
   • default ← 当前

❓ you: /session work

✅ 已切换到会话: work

📌 当前会话: work

❓ you: 我叫什么名字？

🤖 助手正在思考...

💬 我不知道您的名字。您可以告诉我您叫什么，我会记住的！

❓ you: /session default

✅ 已切换到会话: default

📌 当前会话: default

❓ you: 我叫什么名字？

🤖 助手正在思考...

💬 根据我们的对话记录，您叫小明。
```

注意最后几个操作的精妙之处：
- 切换到 `"work"` 会话后问"我叫什么" → 回答不知道（因为 work 会话没有自报家门的记录）
- 切回 `"default"` 会话后再问同样的问题 → 正确回答"小明"

这就是多会话 Memory 的威力——**每个会话拥有独立的、持久的对话上下文**。

## 第六步：定制化用法

`ChatAssistant` 类的设计让它很容易被定制和扩展：

### 定制人设

```python
# 创建一个严格的代码审查助手
code_reviewer = ChatAssistant(
    system_prompt=(
        "你是一位资深代码审查工程师。你的职责是审查用户提交的代码，"
        "指出潜在的问题、性能瓶颈和安全风险。"
        "回答要专业、直接，不要客套话。"
    ),
    temperature=0
)
```

### 在其他项目中复用

```python
from assistant import ChatAssistant

# 在 Web 应用中使用
assistant = ChatAssistant(system_prompt="你是客服助手")

@app.route("/chat", methods=["POST"])
def chat_endpoint():
    data = request.json
    session_id = data.get("session_id", request.remote_addr)
    reply = assistant.chat(data["message"], session_id=session_id)
    return {"reply": reply}
```

### 与 RAG 结合

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

vectorstore = Chroma(persist_directory="./chroma_db",
                     embedding_function=OpenAIEmbeddings())
retriever = vectorstore.as_retriever(k=3)

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", "基于以下参考资料回答问题。资料中没有就说不知道。"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "参考资料:\n{context}\n\n问题:{question}")
])

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | ChatOpenAI(model="gpt-4o-mini", temperature=0)
)

rag_assistant = RunnableWithMessageHistory(
    runnable=rag_chain,
    get_session_history=lambda sid: SQLChatMessageHistory(
        session_id=sid, connection_string="sqlite:///chat_history.db"
    ),
    input_messages_key="question",
    history_messages_key="chat_history"
)
```

## 性能与扩展建议

当你的对话助手投入实际使用后，有几个方面值得关注：

**Memory 大小监控**。定期检查各会话的消息数量和预估 token 数：

```python
def estimate_tokens(text: str) -> int:
    """粗略估算 token 数（中文约 1.5 字符/token，英文约 4 字符/token）"""
    return max(len(text) // 3, len(text.split()) * 13 // 10)

for sid in assistant.list_sessions():
    msgs = assistant.get_session_messages(sid)
    total_chars = sum(len(m.content) for m in msgs)
    est_tokens = estimate_tokens(total_chars)
    print(f"{sid}: {len(msgs)} 条消息, ~{est_tokens} tokens")
```

如果发现某些会话的 token 数接近模型的上下文窗口上限，需要考虑实现自动摘要策略。

**冷启动优化**。首次连接 SQLite 时会有一次初始化开销（~50ms）。对于延迟敏感的场景，可以在程序启动时预热：

```python
def warm_up(assistant: ChatAssistant):
    _ = assistant.get_session_messages("_warmup_")
    assistant.clear_session("_warmup_")
```

到这里，第五章的全部内容就结束了。我们从 LLM 无状态的本质问题出发，系统地学习了 LangChain 的 Memory 组件体系，掌握了 Buffer / Window / TokenBuffer / Summary 等多种记忆类型的原理和使用方法，最终构建了一个具备多会话管理、持久化存储、命令行交互等完整功能的智能对话助手。这些能力是构建任何对话式 AI 应用的基础——无论是客服机器人、编程辅导还是个人助理，都离不开可靠的记忆管理。接下来的一章，我们将学习 LangChain 中另一个强大的概念：Chain（链），它能让多个组件像流水线一样协同工作，完成远比单次问答复杂的任务。
