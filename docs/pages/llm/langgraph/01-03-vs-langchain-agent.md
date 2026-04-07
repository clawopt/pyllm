# 1.3 LangGraph vs LangChain Agent：深度对比与选型指南

## 同根同源，但走了不同的路

LangGraph 和 LangChain 的 `create_react_agent()` 都来自同一个团队（LangChain Inc.），它们共享底层的 LLM 调用、Tool 定义、Message 格式等基础设施。但它们解决的是不同层面的问题——如果你把 LangChain 比作一辆"标准版轿车"，那 LangGraph 就是同一品牌推出的"越野改装版"：底盘一样，但悬挂、四驱、车身结构都完全不同了。

## 架构层面的根本差异

让我们从最底层的设计哲学开始对比：

```
┌─────────────────────────────┐    ┌─────────────────────────────┐
│   LangChain ReAct Agent     │    │      LangGraph              │
│                             │    │                             │
│  ┌───────────────────┐      │    │  ┌───────────────────┐      │
│  │ AgentExecutor    │      │    │  │ StateGraph        │      │
│  │                   │      │    │  │                   │      │
│  │ while loop:       │      │    │  │ State (TypedDict) │      │
│  │   1. LLM 思考      │      │    │  │   ↓               │      │
│  │   2. 选工具/结束   │      │    │  │ [Node A] →[Node B]  │      │
│  │   3. 调用工具      │      │    │  │     ↓     ↘        │      │
│  │   4. 观察结果     │      │    │  │ [Node C] → END     │      │
│  │   5. goto 1        │      │    │  │                   │      │
│  │                   │      │    │  └───────────────────┘      │
│  └───────────────────┘      │    │                             │
│                             │    │  核心区别:                    │
│  状态: 隐式维护在          │    │  - 状态是显式的、类型化的      │
│    messages 列表里         │    │  - 控制流是图结构(非线性)    │
│                             │    │  - 支持持久化和时间旅行       │
│  控制流: 固定的线性循环     │    │  - 原生支持人机中断点        │
│                             │    │                             │
│  持久化: ❌ 不支持           │    │  持久化: ✅ CheckpointerSaver  │
│  人机协作: ❌ 仅初始输入     │    │  人机协作: ✅ Interrupt 节点    │
│  可视化: LangSmith trace  │    │  可视化: 完整图谱 + 状态快照   │
└─────────────────────────────┘    └─────────────────────────────┘
```

### 状态管理：隐式 vs 显式

这是最本质的区别。`create_react_agent()` 把所有状态塞在一个 `messages: List[BaseMessage]` 列表里——Agent 的"记忆"、中间结果、工具调用记录全部混在一起。这就像你把所有的变量都声明为全局可变字典一样——能用但不安全。

```python
# === LangChain ReAct: 状态是隐式的 ===
from langchain.agents import create_react_agent, Tool
from langchain_openai import ChatOpenAI

tools = [
    Tool(name="search", func=lambda q: f"搜索结果: {q}"),
    Tool(name="calculator", func=lambda e: str(eval(e))),
]

agent = create_react_agent(llm=ChatOpenAI(model="gpt-4o"), tools=tools)

result = agent.invoke({"input": "3+5*2等于多少？用计算器验证"})
# result 中只有一个 "output" 字段
# 你无法知道中间调用了哪个工具、返回了什么、LLM 的思考过程是什么
# 如果程序崩溃重启，这些信息全部丢失

# === LangGraph: 状态是显式的、类型化的 ===
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END

class MathState(TypedDict):
    question: str
    current_answer: str
    tool_calls: list
    calculation_steps: list
    verified: bool

def think(state):
    """LLM 决策节点"""
    # 返回 {"current_answer": "..."} —— 明确更新哪个字段
    ...

def use_calculator(state):
    """工具调用节点"""
    # tool_calls 和 calculation_steps 都被显式追踪
    return {
        "tool_calls": state["tool_calls"] + [{"tool": "calc", "expr": ...}],
        "calculation_steps": state["calculation_steps"] + [...],
    }

def verify(state):
    """验证节点"""
    ...
```

在 LangGraph 版本中，每个字段都有明确的类型和语义。`question` 是用户输入（只读）、`current_answer` 是当前答案（由 LLM 生成）、`tool_calls` 记录了工具使用历史（用于调试和审计）、`verified` 标记验证状态。这种**类型安全的显式状态**让代码更不容易出 bug，也让后续的状态检查和恢复变得可能。

### 控制流：线性循环 vs 图结构

`create_react_agent()` 内部就是一个 `while True` 循环——每次迭代都是相同的三个步骤（思考→行动→观察），唯一退出条件是 LLM 输出了 "Final Answer"。这意味着它无法表达以下模式：

```
ReAct 能表达的:
  A → B → C → D → E (线性)
  A → B → C → B → C → D (循环，但只能从头重复)

ReAct 不能表达的:
  A → B ─→ D
  ↓     ↑
  C ──┘ (有环/分支汇聚)

  A → {B, C} → D (并行/汇合)
  
  A → B → [human_review] → C 或 D (条件中断)
```

而 LangGraph 的图结构天然支持以上所有模式。下面用一个具体例子来展示这种差异：

**场景：客服工单处理流程**

```python
# 用 LangGraph 表达一个真实的工单处理流程:
#
#   用户提交工单
#       ↓
#   [自动分类] ← LLM 判断工单类型
#       ↓
#   ├── 账单类 → [查询账单系统] → [自动退款] → END
#   ├── 技术类 → [检索知识库]
#       ├── 能解决 → [自动回复] → [满意度确认] → END
#       ├── 不能解决 → [转人工] → [人工处理] → [用户确认] → END
#   └── 投诉类 → [升级处理] → [主管审核] → [执行] → END
#
# 这个流程有分支、有并行可能性、有人工介入点、有多种终止路径
# 用 create_react_agent() 几乎不可能正确实现

from typing import TypedDict
from langgraph.graph import StateGraph, START, END


class TicketState(TypedDict):
    ticket_id: str
    user_query: str
    category: str            # billing / technical / complaint
    knowledge_result: str
    action_taken: str
    human_input: str
    resolution: str
    status: str


def classify_ticket(state): ...
def query_billing(state): ...
def search_knowledge(state): ...
def auto_reply(state): ...
def escalate_to_human(state): ...
def handle_human_response(state): ...
def escalate_to_manager(state): ...
def confirm_resolution(state): ...

graph = StateGraph(TicketState)
# ... 添加节点和边 ...
```

### 持久化：无状态 vs 有状态快照

这是一个生产环境中的生死攸关问题。假设你的 Agent 正在处理一个需要运行 10 分钟的任务：

```python
# === 场景：Agent 在第 7 分钟时崩溃了 ===

# create_react_agent() 的情况:
# 重启之后：
# - 之前 6 分钟的所有工作全部丢失
# - 必须从头开始
# - 用户要重新提交请求
# - 如果这个任务涉及外部副作用（发了邮件、调了 API），
#   可能产生重复操作

# LangGraph 的情况:
from langgraph.checkpoint.memory import MemorySaver

app = workflow.compile(checkpointer=MemorySaver())

# 第一次运行到第 7 分钟崩溃:
config = {"thread_id": "ticket-12345", "input_state": {...}}
result = app.invoke(config)  # 运行到第 4 步后崩溃

# 从第 4 步的断点恢复:
result = app.invoke(config)  # 自动从第 4 步继续！
# State 中保留了前 4 步的所有数据
# 不会重复任何已完成的操作
```

这个能力在长运行任务中是不可替代的。想象一下一个数据分析 Agent 需要从 50 个数据源拉取数据、清洗、分析、生成报告——整个过程可能要跑 30 分钟。如果没有持久化，任何一次网络抖动或 OOM 都会让 29 分钟的工作白费。

## 功能对照表

| 能力 | create_react_agent | LangGraph | 实际影响 |
|------|-------------------|-----------|---------|
| **多步推理** | ✅ 原生支持 | ✅ 原生支持 | 两者都能做 |
| **工具调用** | ✅ 原生支持 | ✅ 原生支持（且更灵活） | LG 可以在不同节点用不同工具集 |
| **状态类型安全** | ❌ messages 列表 | ✅ TypedDict | LG 减少 50%+ 的状态相关 bug |
| **条件分支** | ❌ 只能线性循环 | ✅ 条件边 + 多路径 | LG 支持 if/else/switch/case |
| **并行执行** | ❌ 串行 | ✅ Map/Send 子图 | LG 可同时调用多个工具 |
| **循环控制** | ⚠️ 隐式（max_iterations） | ✅ 图天然支持循环 | LG 可精确控制循环条件和出口 |
| **人机中断** | ❌ 不支持 | ✅ Interrupt 节点 | LG 可在任意步骤暂停等人类 |
| **持久化/恢复** | ❌ 不支持 | ✅ CheckpointerSaver | LG 支持断点续跑 |
| **时间旅行调试** | ❌ 不支持 | ✅ 可回放历史状态 | LG 可查看每步的完整 state |
| **子图复用** | ❌ 不支持 | ✅ 编译子图 | LG 支持模块化设计 |
| **可视化** | Trace 日志 | ✅ 完整图谱视图 | LG 在 LangSmith 中展示图结构 |

## 什么时候该用哪个？

说了这么多 LangGraph 的好，那是不是应该全面迁移过去呢？也不是。选型建议如下：

**继续用 `create_react_agent()` 的场景：**
- 单轮问答（"帮我查一下天气"）
- 简单的工具链调用（搜个东西 → 总结一下）
- 快速原型验证想法（MVP 阶段）
- 对话式 ChatBot（不需要复杂状态的）

**该升级到 LangGraph 的场景：**
- 任务需要多个步骤且有明确顺序（审批流、数据处理管道）
- 需要在执行过程中保留和传递状态（"第1步的结果传给第3步用"）
- 需要人机协作（"Agent 先草拟 → 人类修改 → 再提交"）
- 任务可能需要长时间运行（超过几秒或需要跨会话）
- 需要处理错误和异常分支（"API 失败了走备用方案"）
- 需要可视化和调试复杂的决策逻辑

一条简单的判断原则：**如果你的 Agent 代码里出现了嵌套的 if/else 或者你开始手动维护一个 state 字典，那就该上 LangGraph 了。**

## 迁移路径

好消息是从 LangChain Agent 迁移到 LangGraph 并不需要重写一切。LangGraph 与 LangChain 的生态完全兼容：

```python
# 你的 LangChain 工具定义可以原封不动地搬过来
from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

@tool
def search(query: str) -> str:
    """搜索互联网"""
    return DuckDuckGoSearchRun(query).run()

@tool
def calculator(expression: str) -> float:
    """计算数学表达式"""
    return eval(expression)

# LangGraph 的 Node 里直接用这些 tools
def research_node(state):
    llm_with_tools = llm.bind_tools([search, calculator])
    response = llm_with_tools.invoke(state["question"])
    return {"research_data": response.content}
```

而且 LangChain 的很多预构建组件（如 `create_react_agent` 本身）底层已经在使用 LangGraph 了——所以你可能已经在间接使用 LangGraph 了。

## 总结

