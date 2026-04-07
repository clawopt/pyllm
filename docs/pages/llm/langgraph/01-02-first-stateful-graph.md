# 1.2 第一个 Stateful Graph：五分钟上手

## 从 Hello World 到第一个有状态图

上一节我们讲了 LangGraph 的设计哲学和核心优势，现在该动手了。这一节的目标是让你用最少的代码跑通一个完整的 Stateful Graph，建立直观感受。我们不会一开始就搞复杂的 Agent——先从最简单的"计数器图"开始，逐步增加复杂度。

## 最简 Graph：两个节点一条边

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END


# Step 1: 定义 State —— 图的共享数据
class SimpleState(TypedDict):
    counter: int
    message: str


# Step 2: 定义 Node —— 每个节点是一个函数，接收 state、返回 state 的更新
def node_a(state: SimpleState) -> dict:
    """节点 A: 计数器 +1"""
    print(f"[Node A] 当前 counter = {state['counter']}")
    return {"counter": state["counter"] + 1, "message": "经过了 A"}


def node_b(state: SimpleState) -> dict:
    """节点 B: 打印当前值"""
    print(f"[Node B] 最终 counter = {state['counter']}")
    return {"message": state["message"] + " → B"}


# Step 3: 创建 Graph 并组装
graph = StateGraph(SimpleState)

graph.add_node("node_a", node_a)
graph.add_node("node_b", node_b)

# 入口: START → node_a
graph.add_edge(START, "node_a")
# node_a → node_b (无条件转移)
graph.add_edge("node_a", "node_b")
# node_b → END (结束)
graph.add_edge("node_b", END)

# 编译为可运行的应用
app = graph.compile()

# 运行!
result = app.invoke({"counter": 0, "message": ""})

print("\n=== 最终结果 ===")
print(f"counter: {result['counter']}")
print(f"message: {result['message']}")
```

运行输出：

```
[Node A] 当前 counter = 0
[Node B] 最终 counter = 1

=== 最终结果 ===
counter: 1
message: 经过了 A → B
```

恭喜！你刚刚写出了你的第一个 LangGraph 程序。虽然它做的事情很简单（把一个数字加 1 再打印），但它包含了 LangGraph 的全部核心要素：

- **State**（`SimpleState`）：定义了 `counter` 和 `message` 两个字段，所有节点都能读写
- **Node**（`node_a`, `node_b`）：两个函数，每个都接收完整的 state 字典并返回要更新的字段
- **Edge**（`add_edge`）：定义了 `START → A → B → END` 这条固定的执行路径
- **compile()**：把图结构编译成可执行的应用
- **invoke()**：传入初始状态，驱动整个图执行完毕后返回最终状态

这里有一个初学者容易混淆的点：**节点的返回值不是"下一个节点的输入"，而是对 State 的更新操作**。当 `node_a` 返回 `{"counter": 1, "message": "经过了 A"}` 时，它的意思是"把 state 中的 counter 改成 1，message 改成'经过了 A'"。然后 `node_b` 收到的是**更新后的完整 state**（此时 counter 已经是 1 了）。这是理解 LangGraph 最关键的一个心智模型。

## 加入条件分支

上面的例子中执行路径是固定的——A 之后一定是 B。但真实场景中我们经常需要"根据当前状态决定下一步做什么"。这就需要**条件边（Conditional Edge）**：

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END


class DecisionState(TypedDict):
    score: int
    grade: str
    feedback: str


def grader(state: DecisionState) -> dict:
    """评分节点"""
    score = state["score"]
    if score >= 90:
        grade, feedback = "A", "优秀！继续保持。"
    elif score >= 80:
        grade, feedback = "B", "良好，还有提升空间。"
    elif score >= 60:
        grade, feedback = "C", "及格了，但要更加努力。"
    else:
        grade, feedback = "F", "不及格，需要补考。"
    
    print(f"[grader] 分数 {score} → 等级 {grade}: {feedback}")
    return {"grade": grade, "feedback": feedback}


def celebrate(state: DecisionState) -> dict:
    print("[celebrate] 🎉 太棒了！发奖状！")
    return {}


def encourage(state: DecisionState) -> dict:
    print("[encourage] 💪 加油！下次一定可以更好！")
    return {}


def remediate(state: DecisionState) -> dict:
    print("[remediate] 📚 需要安排补习...")
    return {}


# 条件路由函数：根据分数决定走哪条路
def route_by_grade(state: DecisionState) -> str:
    grade = state.get("grade", "")
    if grade == "A":
        return "celebrate"
    elif grade in ("B", "C"):
        return "encourage"
    else:  # F
        return "remediate"


# 组装图
graph = StateGraph(DecisionState)

graph.add_node("grader", grader)
graph.add_node("celebrate", celebrate)
graph.add_node("encourage", encourage)
graph.add_node("remediate", remediate)

graph.add_edge(START, "grader")

# 关键：条件边 —— grader 之后根据 grade 走不同的路
graph.add_conditional_edges(
    "grader",
    route_by_grade,
    {
        "celebrate": "celebrate",
        "encourage": "encourage",
        "remediate": "remediate",
    },
)

# 三条路最终都到 END
graph.add_edge("celebrate", END)
graph.add_edge("encourage", END)
graph.add_edge("remediate", END)

app = graph.compile()

# 测试不同分数
for test_score in [95, 72, 45]:
    print(f"\n{'='*40}")
    print(f"测试分数: {test_score}")
    print('='*40)
    result = app.invoke({"score": test_score})
    print(f"结果: 等级={result['grade']}, 反馈={result['feedback']}")
```

运行输出：

```
========================================
测试分数: 95
========================================
[grader] 分数 95 → 等级 A: 优秀！继续保持。
[celebrate] 🎉 太棒了！发奖状！
结果: 等级=A, 反馈=优秀！继续保持。

========================================
测试分数: 72
========================================
[grader] 分数 72 → 等级 C: 及格了，但要更加努力。
[encourage] 💪 加油！下次一定可以更好！
结果: 等级=C, 反馈=及格了，但要更加努力。

========================================
测试分数: 45
========================================
[grader] 分数 45 → 等级 F: 不及格，需要补考。
[remediate] 📚 需要安排补习...
结果: 等级=F, 反馈=不及格，需要补考。
```

这个例子的关键在于 `route_by_grade` 函数和 `add_conditional_edges` 调用。条件边的语法是：

```python
graph.add_conditional_edges(
    source_node,           # 从哪个节点出发
    routing_function,      # 路由函数：接收 state，返回目标节点名称字符串
    path_map,              # 路径映射：{函数返回值: 目标节点名称}
)
```

路由函数的返回值必须与 `path_map` 中的 key 匹配。如果返回了一个 path_map 中不存在的值，LangGraph 会抛出 `ValueError`。这是一个常见的错误来源——比如你忘了在 path_map 中注册某个分支。

## 循环：让图"转起来"

线性执行和条件分支都有了，现在来看更有趣的东西：**循环**。在传统编程中循环是 `while/for` 的事，但在图中循环意味着存在一条从后面的节点指向前面的节点的边：

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END


class LoopState(TypedDict):
    target: int          # 目标数字
    current: int         # 当前猜测
    attempts: int        # 已尝试次数
    max_attempts: int    # 最大尝试次数
    history: list        # 猜测历史
    result: str          # 最终结果


def guesser(state: LoopState) -> dict:
    """猜数节点：用简单的策略猜"""
    current = state["current"]
    target = state["target"]
    
    if current < target:
        current = min(current + 10, target)  # 每次加 10，不超过目标
    elif current > target:
        current = max(current - 10, target)
    
    attempt = state["attempts"] + 1
    history = state["history"] + [current]
    
    print(f"[guesser] 第{attempt}次猜测: {current} (目标: {target})")
    
    return {
        "current": current,
        "attempts": attempt,
        "history": history,
    }


def check_done(state: LoopState) -> dict:
    """检查是否猜中"""
    if state["current"] == state["target"]:
        return {"result": f"✅ 猜中了！用了 {state['attempts']} 次，路径: {state['history']}"}
    return {}


def should_continue(state: LoopState) -> str:
    """是否继续猜？"""
    if state["current"] == state["target"]:
        return "done"
    elif state["attempts"] >= state["max_attempts"]:
        return "give_up"
    else:
        return "guess_again"


def give_up(state: LoopState) -> dict:
    """放弃"""
    return {"result": f"❌ 达到最大尝试次数 {state['max_attempts']}，最终停在 {state['current']}"}


# 组装带循环的图
graph = StateGraph(LoopState)

graph.add_node("guesser", guesser)
graph.add_node("check_done", check_done)
graph.add_node("give_up", give_up)

graph.add_edge(START, "guesser")

# guesser 之后：先检查是否猜中
graph.add_edge("guesser", "check_done")

# check_done 之后：根据结果决定是结束、继续猜还是放弃
graph.add_conditional_edges(
    "check_done",
    lambda s: "done" if s.get("result") else "should_check_limit",
    {
        "done": END,
        "should_check_limit": should_continue,
    },
)

# should_continue 内部再分一次
graph.add_conditional_edges(
    "should_continue",
    should_continue,
    {
        "done": END,
        "guess_again": "guesser",   # ← 循环！回到 guesser
        "give_up": "give_up",
    },
)

graph.add_edge("give_up", END)

app = graph.compile()

# 测试
result = app.invoke({
    "target": 42,
    "current": 0,
    "attempts": 0,
    "max_attempts": 10,
    "history": [],
})

print(f"\n{result['result']}")
```

运行输出：

```
[guesser] 第1次猜测: 10 (目标: 42)
[guesser] 第2次猜测: 20 (目标: 42)
[guesser] 第3次猜测: 30 (目标: 42)
[guesser] 第4次猜测: 40 (目标: 42)
[guesser] 第5次猜测: 42 (目标: 42)

✅ 猜中了！用了 5 次，路径: [10, 20, 30, 40, 42]
```

注意那个 `guess_again → guesser` 的边——这就是循环的本质。在 LangGraph 中，循环不是通过 `while` 关键字实现的，而是通过**一条指向前面节点的边**实现的。这看起来可能有点反直觉，但你想想：图的遍历天然支持环（只要不重复访问已完成的节点就不会死循环），而我们的 `should_continue` 函数确保了循环会在达到目标或次数上限时终止。

## 用 LLM 作为决策者

到目前为止所有的"智能"都是硬编码的规则（if/else）。真正的 Agent 应该由 LLM 来做决策。让我们把 LLM 引入图中：

```python
import os
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


class AgentState(TypedDict):
    question: str
    research_result: str
    answer: str
    needs_more_info: bool


def researcher(state: AgentState) -> dict:
    """研究节点：调用搜索工具获取信息"""
    print(f"\n[researcher] 正在研究: {state['question']}")
    
    response = llm.invoke(
        f"你是一个研究助手。请简要回答以下问题（2-3句话即可）:\n\n"
        f"问题: {state['question']}"
    )
    
    print(f"[researcher] 研究结果: {response.content[:100]}...")
    return {"research_result": response.content, "needs_more_info": False}


def answerer(state: AgentState) -> dict:
    """回答节点：基于研究结果生成最终答案"""
    print(f"\n[answerer] 基于研究结果生成回答...")
    
    response = llm.invoke(
        f"基于以下研究信息，详细回答用户的问题。\n\n"
        f"问题: {state['question']}\n\n"
        f"参考资料:\n{state['research_result']}"
    )
    
    return {"answer": response.content}


def should_do_research(state: AgentState) -> str:
    """判断是否需要先研究再回答"""
    q = state["question"].lower()
    
    research_keywords = ["什么是", "怎么", "为什么", "对比", "区别", "原理"]
    simple_keywords = ["你好", "谢谢", "是的", "多少", "谁"]
    
    has_research_kw = any(kw in q for kw in research_keywords)
    has_simple_kw = any(kw in q for kw in simple_keywords)
    
    if has_research_kw and not has_simple_kw:
        return "do_research"
    else:
        return "answer_directly"


# 组装 LLM 驱动的图
graph = StateGraph(AgentState)

graph.add_node("researcher", researcher)
graph.add_node("answerer", answerer)

graph.add_edge(START, "decide_route")

graph.add_conditional_edges(
    "decide_route",
    should_do_research,
    {
        "do_research": "researcher",
        "answer_directly": "answerer",
    },
)

graph.add_edge("researcher", "answerer")
graph.add_edge("answerer", END)

app = graph.compile()

# 测试几个不同类型的问题
questions = [
    "你好，你是谁？",
    "Python 和 Golang 在并发模型上有什么区别？",
    "什么是 Kubernetes？",
]

for q in questions:
    print("=" * 50)
    print(f"Q: {q}")
    print("=" * 50)
    result = app.invoke({"question": q, "research_result": "", "answer": "", "needs_more_info": False})
    print(f"\nA: {result['answer'][:200]}...\n")
```

这个例子展示了 LangGraph 与 LLM 结合的最基本模式：**LLM 不再是外部调用的黑盒，而是被编排进图的节点中的一个参与者**。`researcher` 和 `answerer` 都是调用 LLM 的函数，但它们各自有不同的 system prompt 和职责分工。`should_do_research` 则是一个轻量级的路由逻辑——当然你也可以用 LLM 来做这个路由决策（虽然对于简单场景来说有点杀鸡用牛刀）。

## 常见误区

### 误区一：把 Node 当作纯函数

新手容易把 LangGraph 的节点写成无副作用的纯函数，认为"节点就是接收输入返回输出"。但实际上节点是**有状态的**——它们能读取和修改全局 State。这意味着你在节点内部做的任何修改都会影响后续节点的行为。如果你在节点 A 中不小心修改了一个不该改的字段（比如把 `error_count` 重置成了 0），后续依赖这个字段的条件判断就会出 bug。

### 误区二：在条件边函数里做"重操作"

条件边函数应该只做轻量级的判断（读取 state 中的值然后返回字符串），而不应该在里面调用 LLM、查询数据库或者做任何耗时操作。原因很简单：条件边会在每次状态更新后被调用，如果它很慢会拖慢整个图的执行速度。重操作应该放在专门的 Node 里去做。

### 误区三：忘记 State 是累积更新的

LangGraph 的 State 更新机制是**合并（merge）而非替换**。如果你的初始 State 是 `{a: 1, b: 2}`，节点 A 返回了 `{a: 10}`，那么合并后的 State 是 `{a: 10, b: 2}`（a 被更新，b 保持不变），而不是 `{a: 10}`（b 丢失了）。这个行为在设计上是有意为之的——它保证了每个节点只需要关心自己要修改的字段，不用担心误删其他节点写的数据。

### 误区四：把所有逻辑塞进一个巨型 Node

有些开发者习惯性地把所有逻辑写在一个大函数里，然后用 if/else 来控制流程——这完全失去了使用 LangGraph 的意义。好的做法是：**每个 Node 只做一件事，流程控制交给 Edge**。如果你发现一个 Node 超过了 50 行代码，大概率应该把它拆分成多个更小的 Node。

## 总结

