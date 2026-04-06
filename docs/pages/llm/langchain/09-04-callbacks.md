---
title: 回调机制：深入 LangChain 内部运行流程
description: CallbackHandler / StdOutCallbackHandler / ConsoleCallbackHandler / 自定义回调、start/end/llm_start/chain_end 事件
---
# 回调机制：深入 LangChain 内部运行流程

上一节我们用 `RunnableLambda` 实现了中间件——它工作在 Chain 的**输入/输出边界**。但如果你想深入到 Chain **内部**的每一个环节（比如 LLM 开始生成之前、每个 tool call 的参数和结果），就需要用到更强大的 **Callbacks（回调）** 机制。

## Callbacks 是什么

在 LangChain 中，**Callback（回调）** 是一种事件监听器——当 Chain 执行过程中的特定"事件"发生时，回调函数会被自动调用。你可以把它理解为 Chain 内部的"钩子（Hook）"：

```
Chain 开始执行
    ↓
[on_chain_start] ← 链路开始
    ↓
prompt.invoke() 执行
    [on_prompt_end]   ← 提示词处理完成
    ↓
chat.invoke() 执行
    [on_llm_start]    ← LLM 调用开始
    [on_llm_new_token] ← 每生成一个新 token 触发
    [on_llm_end]      ← LLM 调用结束
    ↓
parser.invoke() 执行
    [on_chain_end]    ← 链路结束
```

LangChain 定义了丰富的事件类型，让你能在几乎任何执行节点上插入自定义逻辑。

## 使用内置回调：StdOutCallbackHandler

最简单的入门方式是使用 LangChain 内置的 `StdOutCallbackHandler`——它会把所有事件信息打印到控制台：

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.callbacks import StdOutCallbackHandler

chat = ChatOpenAI(model="gpt-4o-mini", temperature=0)
prompt = ChatPromptTemplate.from_template("回答: {question}")
chain = prompt | chat | StrOutputParser()

# 创建回调处理器
handler = StdOutCallbackHandler()

# 在 invoke 时传入
result = chain.invoke(
    {"question": "什么是 RAG？"},
    config={"callbacks": [handler]}
)
```

输出大概是这样的：

```
\n\n Entered new chain.run() chain...
> Entering new chain...
> Created new chain with inputs: {'question': '什么是 RAG？'}
> Beginning invocation of prompt with input: {'question': '什么是 RAG？'}
> Prompt after formatting:
Human: 回答: 什么是 RAG？
> Entering new llm chain...
> Starting LLM run for chat with input:
Human: 回答: 什么是 RAG？
> Got stream response, now returning final answer.
> Finished chain.
```

你看到了吗？`StdOutCallbackHandler` 暴露了 Chain 内部执行的完整时间线：
1. Chain 开始 → 输入被接收
2. Prompt 格式化完成 → 显示最终发送给 LLM 的消息
3. LLM 调用开始/结束
4. Chain 结束 → 返回结果

对于调试 "我的提示词到底发出去长什么样？" 或 "LLM 收到的到底是什么？" 这类问题，这个回调非常实用。

## 使用 ConsoleCallbackHandler 获得更详细的信息

如果你需要看到**每一个 token 级别的细节**（比如流式输出的每个 chunk），可以用 `ConsoleCallbackHandler`：

```python
from langchain_core.tracers import ConsoleCallbackHandler

handler = ConsoleCallbackHandler()

result = await chain.ainvoke(
    {"question": "说一句话"},
    config={"callbacks": [handler]}
)
```

`ConsoleCallbackHandler` 比 `StdOutCallbackHandler` 更详细，特别适合：
- 查看 LLM 生成的完整 token 序列
- 调试流式输出的具体内容
- 分析 Agent 的工具调用过程（每一步的 input/output）
- 性能分析（哪个环节耗时最长）

## 自定义 CallbackHandler

内置的 Handler 固然方便，但它们只能打印到控制台。实际项目中你可能想把事件信息写入数据库、发送到监控系统、或触发告警。这时就需要**自定义 CallbackHandler**：

```python
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

class MetricsCollector(BaseCallbackHandler):
    """自定义回调：收集性能指标"""

    def __init__(self):
        self.metrics = {
            "start_time": None,
            "end_time": None,
            "token_count": 0,
            "tool_calls": []
        }

    def on_chain_start(self, inputs, **kwargs):
        """Chain 开始时触发"""
        from datetime import datetime
        self.metrics["start_time"] = datetime.now()
        print(f"🚀 Chain 开始 @ {self.metrics['start_time']}")

    def on_chain_end(self, outputs, **kwargs):
        """Chain 结束时触发"""
        from datetime import datetime
        self.metrics["end_time"] = datetime.now()
        duration = (self.metrics["end_time"] - self.metrics["start_time"]).total_seconds()
        
        print(f"\n✅ Chain 完成")
        print(f"   ⏱️ 总耗时: {duration:.2f}s")
        print(f"   📊 Token 数: {self.metrics['token_count']}")
        print(f"   🔧 工具调用: {len(self.metrics['tool_calls'])} 次")
        
        # 这里可以写入数据库或发送到监控系统
        # save_to_metrics_db(self.metrics)

    def on_llm_end(self, response: LLMResult, **kwargs):
        """每次 LLM 调用结束时触发"""
        if response.llm_output and hasattr(response.llm_output, 'token_usage'):
            usage = response.llm_output.token_usage
            self.metrics["token_count"] += usage.get("total_tokens", 0)

    def on_tool_end(self, tool_output, **kwargs):
        """每次工具调用结束时触发"""
        tool_name = kwargs.get("name", "unknown_tool")
        self.metrics["tool_calls"].append({
            "tool": tool_name,
            "output_len": len(str(tool_output))
        })


# 使用
handler = MetricsCollector()

result = chain.invoke(
    {"question": "北京天气怎么样？175*23=?"},
    config={"callbacks": [handler]}
)
```

输出：

```
🚀 Chain 开始 @ 2025-04-06 15:30:12

✅ Chain 完成
   ⏱️ 总耗时: 3.45s
   📊 Token 数: 287
   🔧 工具调用: 1 次
```

### BaseCallbackHandler 的核心方法

| 方法 | 触发时机 | 常见用途 |
|------|---------|---------|
| `on_chain_start` | Chain 开始执行 | 记录开始时间、初始化状态 |
| `on_chain_end` | Chain 执行完毕 | 计算总耗时、保存指标 |
| `on_llm_start` | LLM 调用开始 | 记录 prompt token 数 |
| `on_llm_end` | LLM 调用结束 | 记录 completion token 数 |
| `on_llm_new_token` | 流式输出每个新 token | 实时推送给用户 |
| `on_tool_start` | 工具调用开始 | 记录工具名和参数 |
| `on_tool_end` | 工具调用结束 | 记录工具返回值 |
| `on_text` | 文本类组件输出 | 处理纯文本输出 |
| `on_error` | 任何步骤出错 | 错误日志和告警 |

## 回调与中间件的关系

你可能会有疑问：**Callbacks 和上一节的 Middleware 有什么区别？**

| 维度 | Middleware（中间件） | Callbacks（回调） |
|------|-------------------|---------------|
| **位置** | Chain 的输入/输出边界 | Chain **内部**的任意节点 |
| **粒度** | 整个 Chain 为一个黑盒 | 每个 Runnable 单独可见 |
| **数据访问** | 只能看到输入和最终输出 | 能看到中间过程的每一步 |
| **适用场景** | 日志、审核、限流、重试 | 调试、监控、性能分析、实时推送 |

简单来说：
- **Middleware = 门卫**：站在门口检查进出的人
- **Callbacks = 监控摄像头**：装在屋里每个角落的摄像机

两者经常配合使用——Middleware 做粗粒度的控制和过滤，Callbacks 做细粒度的观察和记录。

## 实战：带完整追踪的对话系统

把前面学的内容组合起来，构建一个带完整追踪能力的生产级对话系统：

```python
"""
traced_chat.py — 带回调追踪的对话服务
"""
import os
from dotenv import load_dotenv
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain.callbacks import StdOutCallbackHandler
from langchain.chat_history import InMemoryChatMessageHistory
from langchain.runnables.history import RunnableWithMessageHistory

load_dotenv()

chat = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是 Python 助教。回答简洁。"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

base_chain = prompt | chat | StrOutputParser()

store = {}

def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# 带记忆 + 带追踪的 Chain
chain_with_all = RunnableWithMessageHistory(
    runnable=base_chain,
    get_session_history=get_session_history,
    input_messages_key="question",
    history_messages_key="history"
)


def main():
    handler = StdOutCallbackHandler()

    session_id = "demo_user"
    
    print("=" * 50)
    print("💬 带追踪的对话系统")
    print("=" * 50)

    questions = [
        "你好",
        "什么是装饰器？",
        "给我写个例子",
        "刚才那个例子能改一下吗？"
    ]

    for q in questions:
        print(f"\n{'='*40}")
        print(f"👤 你: {q}")
        print(f"🤖 ", end="", flush=True)

        result = chain_with_all.invoke(
            {"question": q},
            config={
                "configurable": {"session_id": session_id},
                "callbacks": [handler]
            }
        )
        print(result)
    
    print(f"\n{'='*50}")

if __name__ == "__main__":
    main()
```

运行后你会看到完整的内部执行过程——包括每轮对话的历史消息加载、prompt 组装、LLM 调用的详细信息等。这对于排查"为什么第 3 轮的回答不符合预期"这类问题极其有用。

到这里，第九章的前四个小节就结束了。我们学习了流式输出、异步编程、中间件模式和回调机制——这四者构成了 LangChain 应用的**运行时基础设施**。最后一节我们将综合运用这些知识，构建一个带对话审核功能的完整应用。
