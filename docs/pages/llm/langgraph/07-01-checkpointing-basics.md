# 7.1 Checkpointing 基础：让图的状态可持久化

> 在前面的所有章节中，我们一直在使用 `MemorySaver` 作为 checkpointer——它把状态保存在内存中，进程结束后数据就丢失了。这在开发和演示时足够方便，但在生产环境中是远远不够的。想象一下：一个需要运行数分钟的复杂工作流在执行到第 80% 时因为服务器重启而中断，所有中间状态全部丢失，用户必须从头开始。这显然是不可接受的。LangGraph 的 **Checkpointing（检查点）机制**就是为了解决这个问题而设计的——它能在图的执行过程中定期保存完整的状态快照，使得断点续跑、时间旅行调试、甚至人机协作的 Interrupt 恢复都成为可能。

## 为什么需要 Checkpointing

先理解一下没有 checkpointing 时的问题。当你调用 `app.invoke(initial_state)` 时，整个执行过程发生在内存中：

```
时间线:
t=0s   invoke() 开始
t=1s   节点A 执行完毕，状态更新
t=5s   节点B 执行完毕（调用了LLM）
t=10s  节点C 执行中...
       💥 服务器崩溃 / 进程被杀死 / 网络超时
       
结果: 所有状态丢失！节点A和B的产出全部白费
```

有了 checkpointing 之后：

```
时间线:
t=0s   invoke() 开始，保存初始状态快照 S0
t=1s   节点A 执行完毕，保存状态快照 S1
t=5s   节点B 执行完毕，保存状态快照 S2
t=10s  节点C 执行中...
       💥 服务器崩溃

恢复:
       从最近的快照 S2 加载状态
       直接从节点C开始继续执行 ✅
```

除了**断点续跑（Crash Recovery）**之外，checkpointing 还支持以下关键能力：

- **时间旅行调试（Time Travel Debugging）**：回退到任意历史状态重新执行后续步骤
- **Interrupt 恢复**：第4章讨论的人机协作依赖 checkpointing 来暂停和恢复执行
- **并发安全**：多个请求共享同一个 checkpointer 时，各自有独立的状态隔离
- **审计追踪**：完整的执行历史可用于合规审查和问题排查

## MemorySaver：内存级 Checkpointing

`MemorySaver` 是最简单的 checkpointer 实现，它把所有状态快照保存在 Python 字典（内存）中。适合开发测试、单实例部署、以及不需要跨进程持久化的场景。

```python
from typing import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

class TaskState(TypedDict):
    task_id: str
    description: step_count: Annotated[int, operator.add]
    current_step: str
    results: list[str]
    status: str
    log: list[str]

def step1_research(state: TaskState) -> dict:
    return {
        "current_step": "research",
        "results": state["results"] + [f"研究: {state['description']}"],
        "step_count": 1,
        "log": [f"[步骤1] 完成研究"]
    }

def step2_analyze(state: TaskState) -> dict:
    return {
        "current_step": "analyze",
        "results": state["results"] + [f"分析: {state['results'][-1]}"],
        "step_count": 1,
        "log": [f"[步骤2] 完成分析"]
    }

def step3_report(state: TaskState) -> dict:
    return {
        "status": "completed",
        "log": [f"[步骤3] 报告生成完毕"]
    }

task_graph = StateGraph(TaskState)
task_graph.add_node("research", step1_research)
task_graph.add_node("analyze", step2_analyze)
task_graph.add_node("report", step3_report)

task_graph.add_edge(START, "research")
task_graph.add_edge("research", "analyze")
task_graph.add_edge("analyze", "report")
task_graph.add_edge("report", END)

# 使用 MemorySaver
memory_checkpointer = MemorySaver()
app = task_graph.compile(checkpointer=memory_checkpointer)

config = {"configurable": {"thread_id": "task-001"}}

print("=" * 60)
print("第一次执行")
print("=" * 60)
result1 = app.invoke({
    "task_id": "T-001",
    "description": "Python GIL 的影响",
    "step_count": 0,
    "current_step": "",
    "results": [],
    "status": "",
    "log": []
}, config=config)

for entry in result1["log"]:
    print(entry)
print(f"\n总步骤数: {result1['step_count']}")

# 验证状态已被保存
saved_state = memory_checkpointer.get(config)
print(f"\nCheckpointer 中保存了状态: {saved_state is not None}")
if saved_state:
    print(f"最后一步: {saved_state.metadata.get('step', 'unknown')}")
```

这段代码展示了 MemorySaver 的基本用法。关键点是：

1. 创建 `MemorySaver()` 实例
2. 在 `compile()` 时传入 `checkpointer=memory_checkpointer`
3. 每次调用 `invoke()` 时传入包含 `thread_id` 的 config
4. LangGraph 会在每个节点执行后自动保存状态快照到 MemorySaver 中

`thread_id` 是 checkpointing 的核心概念——它是**一次独立执行的唯一标识符**。不同的 thread_id 对应完全独立的执行上下文，互不干扰。你可以把它理解为数据库中的主键或会话 ID。

## Checkpoint 的生命周期

理解 checkpoint 的生命周期对于正确使用它至关重要。一次完整的图执行会创建多个 checkpoint，每个 checkpoint 对应图执行过程中的一个"时间切片"。

```python
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
app = some_graph.compile(checkpointer=checkpointer)
config = {"configurable": {"thread_id": "demo-session"}}

# 第一次调用：创建 S0 (初始状态) → S1 → S2 → ... → Sn (最终状态)
result = app.invoke(initial_state, config=config)

# 此时 checkpointer 中包含了从 S0 到 Sn 的所有快照
checkpoint = checkpointer.get(config)
print(f"当前 checkpoint 序列号: {checkpoint.metadata.get('seq', '?')}")
print(f"最终状态中的值: {checkpoint.channel_values.get('status', 'N/A')}")

# 可以获取历史 checkpoint 列表
checkpoint_list = checkpointer.list(config)
print(f"\n总共保存了 {len(checkpoint_list)} 个 checkpoint")
for cp in checkpoint_list:
    print(f"  序列 {cp.metadata.get('seq')}: "
          f"parent={cp.parent_config}, "
          f"step={cp.metadata.get('step', '?')}")

# 第二次调用相同的 thread_id：会基于最新的 checkpoint 继续
# （对于已经完成的图，会返回缓存的结果）
result2 = app.invoke(new_input_state, config=config)
```

每次 `invoke()` 调用都会产生一系列 checkpoint：
- **S0**：invoke 开始前，保存初始输入状态
- **S1**：第一个节点执行后，保存第一个节点的输出合并后的状态
- **S2**：第二个节点执行后，保存第二个节点的输出合并后的状态
- ...
- **Sn**：最后一个节点执行后（或 END），保存最终状态

这些 checkpoint 形成了一个**链表结构**——每个 checkpoint 都记录了它的父 checkpoint（即上一步的 checkpoint），这样就能沿着链表回溯任意历史状态。

## 用 Checkpoint 做断点续跑

这是 checkpoint 最实用的功能之一。当图中途失败时，你不需要重头再来，而是可以从最近成功的 checkpoint 恢复并继续执行。

```python
class UnstableTaskState(TypedDict):
    data: str
    processed_parts: Annotated[list[str], operator.add]
    attempt: int
    max_attempts: int
    unstable_step_completed: bool
    final_result: str
    log: list[str]

def stable_step(state: UnstableTaskState) -> dict:
    return {
        "processed_parts": [f"稳定处理: {state['data']}"],
        "attempt": 1,
        "log": ["[稳定步骤] 完成"]
    }

def unstable_step(state: UnstableTaskState) -> dict:
    import random
    if random.random() < 0.7:
        raise RuntimeError("模拟的不稳定操作失败了!")

    return {
        "unstable_step_completed": True,
        "processed_parts": [f"不稳定处理完成"],
        "attempt": 1,
        "log": ["[不稳定步骤] 幸运地成功了!"]
    }

def finalize(state: UnstableTaskState) -> dict:
    return {
        "final_result": f"处理完成: {', '.join(state['processed_parts'])}",
        "log": ["[完成] 最终结果已生成"]
    }

unstable_graph = StateGraph(UnstableTaskState)
unstable_graph.add_node("stable", stable_step)
unstable_graph.add_node("unstable", unstable_step)
unstable_graph.add_node("finalize", finalize)

unstable_graph.add_edge(START, "stable")
unstable_graph.add_edge("stable", "unstable")
unstable_graph.add_edge("unstable", "finalize")
unstable_graph.add_edge("finalize", END)

checkpointer = MemorySaver()
app = unstable_graph.compile(checkpointer=checkpointer)
config = {"configurable": {"thread_id": "unstable-demo"}}

print("=== 尝试第一次执行 ===")
try:
    result = app.invoke({
        "data": "test-data-12345",
        "processed_parts": [], "attempt": 0, "max_attempts": 3,
        "unstable_step_completed": False, "final_result": "", "log": []
    }, config=config)
except RuntimeError as e:
    print(f"❌ 执行中断: {e}")

# 检查保存到了哪里
checkpoint = checkpointer.get(config)
if checkpoint:
    print(f"\n✅ 断点已保存!")
    print(f"   最后成功完成的步骤: {checkpoint.metadata.get('source', 'unknown')}")
    saved_data = checkpoint.channel_values
    print(f"   已处理的部件: {saved_data.get('processed_parts', [])}")

print("\n=== 重试（从断点恢复）===")
try:
    result = app.invoke({
        "data": "test-data-12345",
        "processed_parts": [], "attempt": 0, "max_attempts": 3,
        "unstable_step_completed": False, "final_result": "", "log": []
    }, config=config)
    print(f"✅ 最终结果: {result['final_result']}")
    for entry in result["log"]:
        print(f"  {entry}")
except RuntimeError as e:
    print(f"❌ 再次失败: {e}，可以继续重试...")
```

这个例子展示了一个关键行为：**当第二次用相同的 thread_id 调用 `invoke()` 时，LangGraph 会检测到之前有未完成的执行，并自动从最近的 checkpoint 恢复状态**。这意味着不稳定步骤之前的所有工作都不需要重复做——`stable` 步骤的产出已经被保存在 checkpoint 中了。

注意这里有一个微妙但重要的细节：第二次 `invoke()` 我们传入了全新的初始状态（`processed_parts: []`），但由于使用了相同的 thread_id，LangGraph 会优先使用 checkpoint 中保存的状态而不是我们传入的初始状态。这就是为什么 checkpointing 能实现真正的断点续跑。

## Checkpoint 的内部结构

虽然你在大多数情况下不需要直接操作 checkpoint 的内部结构，但了解它能帮助你更好地调试和理解其行为。每个 checkpoint 大致包含以下信息：

```python
class CheckpointData:
    # 元数据（不包含业务数据）
    metadata: {
        "thread_id": "task-001",           # 所属线程ID
        "parent_ts": "2024-01-01T12:00:01", # 父checkpoint的时间戳
        "source": "unstable",              # 产生此checkpoint的节点名
        "step": 2,                         # 当前步数
        "writes": {"processed_parts", "attempt"},  # 本步写入的字段列表
    }
    
    # 通道值（实际的业务状态数据）
    channel_values: {
        "data": "test-data-12345",
        "processed_parts": ["稳定处理: test-data-12345"],
        "attempt": 1,
        # ... 其他字段
    }
    
    # 待处理的挂起操作（用于Interrupt）
    pending_sends: []  # 如果有Interrupt，这里会记录等待的操作
    
    # 配置版本
    config_version: 1  # 用于检测配置变更
```

`channel_values` 是最重要的部分——它存储了那个时间点上图的完整状态快照。当你调用 `checkpointer.get(config)` 时，返回的就是最新 checkpoint 的数据。LangSmith 的 trace 视图中看到的每一步的状态快照，本质上就是每个 checkpoint 的 `channel_values`。
