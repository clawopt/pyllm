# 1.5 开发工具链与调试体系

## 写好 LangGraph 程序的"左膀右臂"

前面四节我们学会了 LangGraph 的核心概念和基本用法。但会写代码只是第一步——在真实项目中你还需要一整套开发工具来保证代码质量、快速定位问题、以及理解程序的运行行为。这一节要介绍的是 LangGraph 生态中的几个关键工具：**LangSmith 可视化、Checkpointer 调试、Streaming 实时观察，以及常见的调试技巧**。

## LangSmith：让图的执行过程可视化

LangSmith 是 LangChain 官方的 LLM 应用监控平台，它与 LangGraph 的集成深度是所有监控工具中最高的。如果你用过 Chrome DevTools 调试过前端页面，那 LangSmith 对 LangGraph 来说就是类似的东西——只不过它调试的不是 DOM 而是 State 的变化和 Node 的跳转。

### 基本设置

```python
import os
os.environ["LANGCHAIN_TRACING"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-langsmith-api-key"  # 从 https://smith.langchain.com 获取


# 在你的 Graph 编译后，所有执行都会自动上传到 LangSmith
from langgraph.graph import StateGraph, START, END

# ... 定义你的 graph ...

app = graph.compile()

# 这一次 invoke 的完整轨迹（每个节点的输入输出、State 变化）都会被记录
result = app.invoke({"question": "什么是 LangGraph?"})
```

只需要设置两个环境变量（`LANGCHAIN_TRACING=true` 和 `LANGCHAIN_API_KEY`），你的 LangGraph 程序就会自动把每次执行的详细信息发送到 LangSmith。不需要修改任何业务代码。

### 你能在 LangSmith 上看到什么

打开 https://smith.langchain.com ，你会看到：

1. **Trace 列表**：每次 `invoke()` 或 `stream()` 调用都记录为一条 Trace。你可以看到调用时间、耗时、最终状态。

2. **Trace 详情页**：点击某条 Trace 进入详情后，你会看到一个**可视化的图结构**：
   - 每个节点显示为一个方框
   - 方框之间有箭头表示 Edge
   - 每个方框内展示该节点的输入 State 和输出 State
   - 条件边会标注走的是哪个分支以及为什么
   - 如果某个节点调用了 LLM，可以看到完整的 prompt 和 response

3. **State Timeline**：一个时间线视图，展示 State 中每个字段随时间的变化——这对调试"为什么 State 里的值不对"这类问题极其有用。

4. **Token 用量和成本统计**：每条 Trace 都会统计消耗的 token 数量和 API 成本。

### 一个真实的调试案例

假设你的 Agent 出了 bug——它总是返回空答案。没有 LangSmith 你只能加 print 语句盲猜；有了 LangSmith 你可以这样做：

```
问题现象: app.invoke() 返回 {"answer": ""}

LangSmith 调试步骤:

1. 打开最新的 Trace → 查看 Graph 视图
2. 发现执行路径是: START → researcher → answerer → END ✅ 路径正确
3. 点击 [researcher] 节点:
   输入 State: {"question": "Python GIL 是什么?", ...}
   输出 State: {"research_result": "", ...}  ← 问题在这！
   research_result 是空的！
4. 点击 researcher 内部的 LLM 调用:
   Prompt: "你是研究助手。请简要回答..."
   Response: (empty / error)  ← LLM 调用失败了？
5. 检查 API Key 和网络连接...

结论: 不是图逻辑的问题，是 researcher 节点内部的 LLM 调用出了异常。
修复: 检查 OPENAI_API_KEY 是否正确配置。
```

整个排查过程可能只要 3 分钟。而没有可视化工具的话，同样的排查可能需要反复加 log → 重启 → 观察 → 再加 log → 再重启，花上半小时。

## Streaming：实时观察 State 变化

除了事后去 LangSmith 上看，你还可以在程序运行时实时观察 State 的变化。LangGraph 的 `stream()` 模式会在每次 Node 执行完毕、State 更新后产生一个事件：

```python
import json

# stream_mode="values": 每次 State 更新都 yield 完整的新 State
print("=== Streaming 模式 ===\n")

for chunk in app.stream(
    {"question": "帮我分析 Python 和 Go 的并发模型"},
    stream_mode="values",
    subgraphs="deep",
):
    # chunk 就是更新后的完整 State
    step = chunk.get("current_step", "?")
    messages_count = len(chunk.get("messages", []))
    
    print(f"[{step}] messages={messages_count}")
    
    if "answer" in chunk and chunk["answer"]:
        print(f"\n📝 回答预览: {chunk['answer'][:100]}...")
```

输出示例：

```
=== Streaming 模式 ===

[classify] messages=0
[research_python] messages=2
[research_go] messages=4
[compare] messages=6
[generate_report] messages=8

📝 回答预览: Python 使用 GIL（全局解释器锁）实现伪并发...
```

`stream_mode` 有几种不同的模式值得了解：

- **`values`**（最常用）：每次 State 变化都 yield 完整的 State 字典。适合需要观察整体进展的场景。
- **`updates`**：只 yield 发生变化的字段（增量）。适合 State 很大时减少传输量。
- **`messages`**：只 yield messages 字段的变化。适合 Chat 类应用。

## Checkpoint 与断点续跑

这是 LangGraph 区别于其他 workflow 框架的杀手级特性之一。当你的图包含长时间运行的步骤时（比如等待人工审核、调用外部 API 需要几分钟），Checkpoint 能让你在任何崩溃或中断后从断点恢复：

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END


class LongRunningState(TypedDict):
    task_id: str
    phase: str              # init / processing / reviewing / done
    data_collected: dict
    human_approval: str     # pending / approved / rejected
    result: str


def long_task(state):
    """模拟一个耗时的任务"""
    import time
    time.sleep(2)  # 模拟耗时操作
    return {"phase": "processing", "data_collected": {"temp": 42}}


def wait_for_human(state):
    """等待人类审批 —— 这里会产生 INTERRUPT"""
    return {"phase": "reviewing"}


def finalize(state):
    """完成"""
    return {"phase": "done", "result": "任务完成"}


graph = StateGraph(LongRunningState)
graph.add_node("long_task", long_task)
graph.add_node("wait_for_human", wait_for_human)
graph.add_node("finalize", finalize)

graph.add_edge(START, "long_task")
graph.add_edge("long_task", "wait_for_human")
graph.add_conditional_edges(
    "wait_for_human",
    lambda s: "finalize" if s.get("human_approval") == "approved" else "wait_for_human",
    {"finalize": "finalize", "wait_for_human": "wait_for_human"},
)
graph.add_edge("finalize", END)

# 关键：使用 MemorySaver 作为 checkpointer
checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)


# 第一次运行（假设在第3步被中断）
config = {"configurable": {"thread_id": "task-001"}}
try:
    result = app.invoke(
        {"task_id": "task-001", "phase": "init", "human_approval": "pending", "data_collected": {}, "result": ""},
        config=config,
    )
except KeyboardInterrupt:
    print("\n⚠️ 用户中断了执行！")

# 从断点恢复 —— 不需要重新执行已完成的步骤
print("\n🔄 从断点恢复...")
result = app.invoke(
    {"task_id": "task-001", "phase": "init", "human_approval": "pending", "data_collected": {}, "result": ""},
    config=config,
)
```

第一次运行时，`MemorySaver` 会在每个节点执行前后自动保存 State 快照到内存中。当执行被中断（用户按了 Ctrl+C、或者程序崩溃）后，第二次 `invoke()` 时 `MemorySaver` 会检测到该 thread_id 已有历史记录，自动从最近的快照恢复而不是从头开始。

在生产环境中你应该用 SQLite 或者 Postgres 后端存储替代 `MemorySaver`：

```python
from langgraph.checkpoint.sqlite import SqliteSaver

# 数据持久化到文件，重启不丢失
checkpointer = SqliteSaver.from_conn_string(":memory:")  # 或 "./checkpoints.db"
# 生产环境:
# checkpointer = SqliteSaver.from_conn_string("postgresql://user:pass@localhost:5432/langgraph")

app = graph.compile(checkpointer=checkpointer)
```

## 常见调试模式与 Checklist

基于大量实际开发经验，我整理了一个 LangGraph 调试的通用 checklist：

```
┌─────────────────────────────────────────────────────┐
│            LangGraph 调试 Checklist                  │
│                                                     │
│ □ 图结构验证                                        │
│   ├── 所有节点是否都有入口边？                        │
│   ├── 是否存在孤儿节点（无入口或无出口）？             │
│   ├── 是否所有路径都能到达 END？                       │
│   └── 循环是否有明确的退出条件？                     │
│                                                     │
│ □ State 设计检查                                      │
│   ├── State 字段是否都是必要的（无冗余临时变量）？       │
│   ├── 路由函数是否只做轻量判断？                     │
│   ├── path_map 是否覆盖了路由函数的所有返回值？           │
│   └── 是否有字段需要特殊合并策略（list/dict）？         │
│                                                     │
│ □ Node 函数检查                                     │
│   ├── 每个 Node 是否只做一件事？                       │
│   ├── Node 内部是否有应该抽出到独立 Node 的逻辑？      │
│   ├── 异步 Node 是否正确使用了 async/await？               │
│   └── Node 返回值是否符合 State 的 TypedDict 定义？        │
│                                                     │
│ □ Edge 检查                                         │
│   ├── 是否有两条边指向同一个目标且条件互斥？          │
│   ├── 条件边的优先级是否正确（更严格的条件先判断）？   │
│   └── 是否有死循环风险（无退出条件的环）？                │
│                                                     │
│ □ 运行时问题                                         │
│   ├── invoke() 返回的 State 是否缺少预期字段？            │
│   ├── 某个节点是否被意外跳过？                          │
│   ├── 条件边是否走了非预期的分支？                      │
│   └── 循环次数是否符合预期？                           │
└─────────────────────────────────────────────────────┘
```

当你遇到问题时，按照这个清单逐项排查，90% 的问题都能定位到原因。

## 性能优化初步

最后简单提一下性能方面的注意点——虽然详细的优化我们会到后面章节展开，但有几个原则现在就可以遵循：

**第一，减少不必要的 Node 执行。** 如果你的图中有些 Node 只在某些特定条件下才需要执行，确保它们不会在不需要的时候被调用。条件边就是干这个的。

**第二，LLM 调用是最昂贵的操作。** 如果一个图中有 5 个 Node 都调用了 LLM，而其中 3 个的结果实际上没被后续 Node 使用，那就是浪费。考虑用缓存或者延迟计算来优化。

**第三，State 的大小影响 checkpoint 的开销。** 如果你的 State 包含了巨大的列表或字典（比如把整份文档内容塞进了 State），每次 checkpoint 序列化/反序列化的代价会很高。大块数据应该放在外部存储中，State 里只存引用（ID 或 URL）。

## 总结

