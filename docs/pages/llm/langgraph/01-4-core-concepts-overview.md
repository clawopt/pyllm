# 1.4 核心概念全景：Graph / Node / Edge / State

## 拆开黑盒：LangGraph 的五个核心概念

前面三节我们通过代码示例直观地感受了 LangGraph 的用法——定义 State、写 Node 函数、组装 Graph、加边、compile、invoke。现在是时候把这些零散的认识串成一张完整的知识图谱了。LangGraph 虽然框架本身不大，但它背后的概念模型是经过精心设计的，每一个概念都有其存在的理由和最佳使用方式。理解这些概念不仅帮你写出能跑通的代码，更能在遇到奇怪的行为时快速定位问题。

## 概念一：State（状态）—— 图的"共享内存"

State 是整个 LangGraph 编程模型中最重要的概念，没有之一。你可以把它理解为图中所有节点共享的一块"黑板"或者一个"全局变量字典"——每个节点都可以从上面读取数据，也可以往上面写入新的值。

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END


# State 是用 Python 的 TypedDict 定义的
class MyState(TypedDict):
    # 每个 key-value 对就是状态中的一个字段
    messages: list          # 对话消息历史
    current_step: str       # 当前执行到哪一步
    error_count: int        # 错误计数
    result: str            # 最终结果
    user_id: str            # 用户标识（只读字段）
```

关于 State 有几个关键的设计决策值得深入理解：

**第一，State 是累积合并的，不是整体替换的。** 这意味着如果初始 State 是 `{a: 1, b: 2, c: 3}`，节点 A 返回了 `{a: 10}`，那么合并后的新 State 是 `{a: 10, b: 2, c: 3}`。`b` 和 `c` 保持不变，只有 `a` 被更新了。这个行为背后的设计哲学是：**每个节点只关心自己要修改的字段，不需要知道其他字段的存在**。这带来了很好的模块性——你可以在不破坏现有节点的情况下给 State 增加新字段。

但这里有一个常见的陷阱：如果你想"清空"某个列表字段怎么办？返回 `{"messages": []}` 并不会清空它——因为合并操作会把旧值和新值混在一起。正确的做法是使用 `operator` 或 `Annotated` 类型来明确指定更新策略：

```python
from operator import add
from typing import Annotated

class CounterState(TypedDict):
    count: Annotated[int, operator.add]   # 使用 add 运算符：新值 = 旧值 + 返回值
    history: list[str]

def increment(state: CounterState) -> dict:
    return {"count": 1}  # 实际效果: state.count += 1

def reset_counter(state: CounterState) -> dict:
    return {"count": -state["count"]}  # 实际效果: state.count = 0
```

**第二，State 应该尽量保持扁平。** 有些开发者喜欢在 State 里嵌套复杂的对象：

```python
# ❌ 不推荐：嵌套过深
class BadState(TypedDict):
    research: dict       # { "query": "...", "results": [...], "meta": {...} }
    user: dict           # { "id": "...", "profile": {"name": ..., "dept": ...} }

# ✅ 推荐：扁平化
class GoodState(TypedDict):
    research_query: str
    research_results: list[dict]
    research_meta_created_at: str
    user_id: str
    user_name: str
    user_dept: str
```

扁平化的好处是：条件边的路由函数可以方便地直接访问任何字段（`state["user_dept"]`），而不需要写成 `state["user"]["profile"]["dept"]` 这种容易出错的长链式访问。而且扁平化的 State 在序列化（用于 Checkpoint）和反序列化时也更不容易出问题。

**第三，State 的设计应该围绕"这个任务需要记住什么"来展开。** 不要把临时变量塞进 State——如果一个值只在当前节点内部使用、后续节点不需要知道，那就应该作为函数的局部变量而不是 State 字段。好的 State 设计原则是：**只放那些需要在多个节点之间传递的信息**。

## 概念二：Node（节点）—— 图中的"处理单元"

Node 是图的基本执行单元。每个 Node 就是一个 Python 函数，它接收完整的 State 作为输入，返回一个包含需要更新的字段的字典。这是 LangGraph 编程中最常写的代码：

```python
def my_node(state: MyState) -> dict:
    """
    Node 函数的标准签名:
    - 输入: 完整的 State 字典
    - 输出: 要更新的字段字典（会被 merge 到 State 上）
    """
    
    # 1. 从 State 中读取需要的数据
    input_data = state["some_field"]
    
    # 2. 执行业务逻辑（调用 LLM、查询数据库、调 API 等）
    result = do_something(input_data)
    
    # 3. 返回需要更新的字段
    return {
        "result_field": result,
        "current_step": "my_node",
    }
```

Node 函数有几种不同的"口味"，了解它们有助于你在不同场景下做选择：

**普通 Node（最常见）**：同步执行一些逻辑，返回 State 更新。90% 的 Node 都属于这一类。

**异步 Node（async def）**：如果你的 Node 需要做 I/O 操作（HTTP 请求、数据库查询），可以定义为 async 函数。LangGraph 会自动 await 它：

```python
import httpx

async def fetch_data_node(state):
    """异步 Node：从 API 获取数据"""
    response = await httpx.get(f"https://api.example.com/data/{state['id']}")
    return {"api_data": response.json()}
```

**LLM Node**：专门调用 LLM 的 Node 是一类特殊的 Node，通常占整个图的 60%-80% 的执行时间：

```python
llm = ChatOpenAI(model="gpt-4o")

def analyst_node(state):
    """LLM Node：让 AI 分析数据并给出结论"""
    response = llm.invoke(
        f"你是数据分析专家。请分析以下数据并给出结论:\n\n{state['raw_data']}"
    )
    return {
        "analysis": response.content,
        "current_step": "analysis_done",
    }
```

关于 Node 的设计有一个重要的经验法则：**单一职责原则**。一个好的 Node 应该只做一件事并且把它做好。如果你发现一个 Node 函数超过了 30 行代码，或者在函数体内出现了 `if xxx: ... else: ...` 的复杂分支逻辑，那大概率应该把它拆分成两个或更多的 Node，然后用条件边来控制走哪个。

## 概念三：Edge（边）—— 节点之间的"导航规则"

Edge 定义了从一个节点执行完后，下一步该去哪里。LangGraph 中有两种 Edge：**普通边（Edge）** 和 **条件边（Conditional Edge）**。

### 普通边：无条件转移

```python
graph.add_edge("node_a", "node_b")
```

这意味着每次 `node_a` 执行完毕后，一定会接着执行 `node_b`。没有例外、没有条件判断。普通边表达的是一种**确定性的顺序关系**——"A 之后一定是 B"。

### 条件边：根据 State 决定走向

```python
def routing_function(state) -> str:
    """路由函数：接收 State，返回目标节点的名称字符串"""
    if state["status"] == "success":
        return "celebrate"
    elif state["status"] == "failed":
        return "retry"
    else:
        return "escalate"

graph.add_conditional_edges(
    "decision_point",           # 从哪个节点出发
    routing_function,              # 路由函数
    {                             # 路径映射表
        "celebrate": "celebrate_node",
        "retry": "retry_node",
        "escalate": "escalate_node",
    },
)
```

条件边是 LangGraph 控制流能力的核心。它让你能够实现：
- **分支**：根据条件走不同的路径（if/else）
- **循环**：通过一条指回前面节点的边实现重复执行
- **提前终止**：满足某个条件时直接跳到 END
- **动态路由**：由 LLM 根据上下文决定下一步（Agent 最常用的模式）

关于条件边有几个重要的工程细节：

**路由函数必须返回 path_map 中存在的 key。** 如果你的路由函数返回了 `"retry"` 但 path_map 里只有 `"retry_node"`（没有 `"retry"` 这个 key），LangGraph 会抛出 `ValueError: 'retry' is not a valid node name`。这是一个极其常见的错误——尤其是在重构时改了节点名字但忘了更新 path_map。

**路由函数应该尽量简单。** 复杂的逻辑应该放在专门的 Node 里去做，路由函数只做最终的判断。原因前面提过：路由函数会在每次状态更新后被调用，如果它很慢会拖慢整个图。

**可以有多层条件边。** 一个节点的出口边可以指向另一个条件边分发的入口，形成链式的条件判断树：

```
node_a → [条件边1] → node_b → [条件边2] → {c, d, e}
                                     → {f, g}
```

## 概念四：START 与 END —— 图的入口和出口

`START` 和 `END` 是 LangGraph 中的两个特殊"伪节点"。它们不是真正的 Node（你不能往 START 里添加逻辑），而是图的边界标记：

- **START**：图的唯一入口。所有执行都从这里开始。你必须至少有一条边从 START 出发。
- **END**：图的终止标记。当执行流到达 END 时，`invoke()` 返回最终的 State。一个图可以有多个节点连接到 END（多条路径都能结束）。

```python
# 正确：每条路径都必须最终到达 END
START → A → B → C → END
START → A → D → END     # 另一条结束路径

# 错误：存在永远无法到达 END 的路径（死循环风险）
START → A → B → A → B → A → ...  # 没有 → END 的边！
```

LangGraph 在编译阶段就会检测这种潜在的无穷循环问题并发出警告（但不一定报错——因为有些循环是有意设计的且有退出条件的）。如果你的图确实需要循环，确保循环中有明确的退出路径（通过条件边在达到某条件时跳转到 END）。

## 概念五：Compiled Graph —— 从蓝图到可运行实例

当你用 `StateGraph()` 创建图对象、添加节点和边之后，你得到的是一张"图的蓝图"——它描述了结构和逻辑，但还不能运行。`compile()` 方法就是把这个蓝图编译成一个可执行的 Application：

```python
# 定义图（蓝图阶段）
graph = StateGraph(MyState)
graph.add_node(...)
graph.add_edge(...)
graph.add_conditional_edges(...)

# 编译为应用（可运行阶段）
app = graph.compile()

# 此时 app 可以:
# 1. 同步调用
result = app.invoke(initial_state)

# 2. 流式调用 (stream_mode="values")
for chunk in app.stream(initial_state):
    print(chunk)  # 每次 State 更新都会产生一个 chunk

# 3. 异步调用
result = await ainvoke(initial_state)

# 4. 带持久化调用
app_with_checkpointer = graph.compile(checkpointer=MemorySaver())
result = app_with_checkpointer.invoke(initial_state, config={"thread_id": "abc-123"})
```

`compile()` 内部做了很多重要的事情：验证图的连通性（是否有孤儿节点）、优化执行计划、初始化内部调度器等。对于大多数用户来说 `compile()` 是透明的——你只需要记得在定义完图之后一定要调用它。

## 五个概念的协作关系图

```
                    ┌──────────────┐
                    │   用户调用    │
                    │  app.invoke() │
                    └──────┬───────┘
                           ▼
              ┌────────────────────────┐
              │    Compiled Graph      │
              │                        │
              │  ┌──────┐  compile()  │
              │  │State │────────────→│ invoke()
              │  └──┬───┘             │
              │     ▼                  │
              │  ┌─────────────┐       │
              │  │   START     │       │
              │  └──────┬──────┘       │
              │         ▼               │
              │  [ Node A ]──┬───────┐  │
              │         │   │       │  │
              │         ▼   ▼       │  │
              │  [Edge: always]  │  │
              │         ▼          │  │
              │  [ Node B ]──┬───────┤  │
              │         │   │       │  │
              │    ┌────┘   └───┐   │  │
              │    ▼          ▼   │  │
              │ [Cond Edge]  [Edge]│  │
              │    ├→[C]   └→[END]│  │
              │    └→[D] ───→[END]│  │
              │                  │  │
              └──────────────────┘  │
                                 │
              返回最终 State ─────────┘
```

这张图展示了五个概念如何在一次 `invoke()` 调用中协作：State 作为共享数据贯穿始终；Node 是执行步骤的基本单元；Edge 决定了步骤之间的转移规则；START/END 标记了边界；而 Compiled Graph 把这一切编排成一次完整的执行过程。

## 总结

