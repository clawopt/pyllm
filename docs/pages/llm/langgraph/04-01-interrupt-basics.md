# 4.1 Interrupt 基础：让人在图的执行过程中介入

> 前面所有章节我们讨论的都是"全自动"的图——一旦开始执行，就会一直运行到 END，中间没有任何停顿。但在很多实际场景中，你需要在图的执行过程中插入人工审核、人工确认或者人工决策的环节。比如代码审查流程中，AI 分析完代码后需要人工确认是否批准；或者贷款审批流程中，风控系统标记了高风险申请后需要人工复核。LangGraph 提供的 `Interrupt` 机制就是用来实现这种人机协作的核心能力——它能让图在任意节点暂停执行，等待人类输入后再继续。

## 从一个直观例子开始：代码审查的人工确认

先通过一个完整的例子来看看 Interrupt 是如何工作的。假设我们有一个代码审查系统，AI 会自动分析代码并给出审查意见，但最终是否批准合并需要人工确认。

```python
from typing import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

class CodeReviewState(TypedDict):
    repo_url: str
    file_path: str
    code_content: str
    analysis_result: dict
    review_score: int
    issues_found: list[str]
    human_decision: str
    human_comment: str
    final_status: str
    review_log: Annotated[list[str], operator.add]

def fetch_code(state: CodeReviewState) -> dict:
    url = state["repo_url"]
    path = state["file_path"]
    code = f"""# 模拟从 {url}{path} 获取的代码

def process_user_data(user_id: int) -> dict:
    import json
    data = fetch_from_db(user_id)
    print(f"Debug: {data}")  # TODO: remove debug print
    return json.dumps(data)

def fetch_from_db(user_id: int):
    # TODO: implement database connection
    pass
"""
    return {
        "code_content": code,
        "review_log": [f"[抓取] 从 {url}{path} 获取代码"]
    }

def analyze_code(state: CodeReviewState) -> dict:
    code = state["code_content"]
    issues = []

    if "TODO" in code or "FIXME" in code:
        issues.append("存在 TODO/FIXME 标记")
    if "print(" in code and "debug" in code.lower():
        issues.append("包含调试打印语句")
    if "pass" in code and "implement" in code.lower():
        issues.append("函数未实现")
    if "import json" in code and not "json.loads" in code and not "json.dumps" in code:
        issues.append("导入了 json 但未使用")

    score = 100 - len(issues) * 15
    analysis = {
        "line_count": len(code.split('\n')),
        "function_count": code.count("def "),
        "issues": issues,
        "score": max(0, score)
    }

    return {
        "analysis_result": analysis,
        "review_score": max(0, score),
        "issues_found": issues,
        "review_log": [f"[分析] 发现 {len(issues)} 个问题，评分: {max(0, score)}"]
    }

def human_review(state: CodeReviewState) -> dict:
    from langgraph.types import interrupt

    score = state["review_score"]
    issues = state["issues_found"]

    prompt = f"""代码审查结果：
评分: {score}/100
发现的问题:
{chr(10).join(f'  - {issue}' for issue in issues) if issues else '  无'}

请决定是否批准此代码合并：
- 输入 'approve' 批准
- 输入 'reject' 拒绝
- 输入 'request_changes' 要求修改

你也可以添加评论（可选），格式: 决策 [空格] 评论
例如: approve 看起来不错，可以合并
"""

    user_input = interrupt(prompt)

    if not user_input or not user_input.strip():
        return {
            "human_decision": "pending",
            "human_comment": "等待人工审核",
            "review_log": ["[人工审核] 等待用户输入"]
        }

    parts = user_input.strip().split(maxsplit=1)
    decision = parts[0].lower()
    comment = parts[1] if len(parts) > 1 else ""

    valid_decisions = {"approve", "reject", "request_changes"}
    if decision not in valid_decisions:
        return {
            "human_decision": "pending",
            "human_comment": f"无效决策: {decision}，请输入 approve/reject/request_changes",
            "review_log": [f"[人工审核] 无效输入: {decision}"]
        }

    return {
        "human_decision": decision,
        "human_comment": comment,
        "review_log": [f"[人工审核] 决策: {decision}, 评论: {comment}"]
    }

def route_by_decision(state: CodeReviewState) -> str:
    decision = state.get("human_decision", "pending")
    if decision == "approve":
        return "approve"
    elif decision == "reject":
        return "reject"
    elif decision == "request_changes":
        return "request_changes"
    else:
        return "pending"

def approve_merge(state: CodeReviewState) -> dict:
    return {
        "final_status": "merged",
        "review_log": ["[最终] 代码已合并到主分支"]
    }

def reject_merge(state: CodeReviewState) -> dict:
    return {
        "final_status": "rejected",
        "review_log": ["[最终] 代码合并被拒绝"]
    }

def request_changes(state: CodeReviewState) -> dict:
    return {
        "final_status": "changes_requested",
        "review_log": ["[最终] 已要求作者修改代码"]
    }

graph = StateGraph(CodeReviewState)
graph.add_node("fetch", fetch_code)
graph.add_node("analyze", analyze_code)
graph.add_node("human_review", human_review)
graph.add_node("approve", approve_merge)
graph.add_node("reject", reject_merge)
graph.add_node("request_changes", request_changes)

graph.add_edge(START, "fetch")
graph.add_edge("fetch", "analyze")
graph.add_edge("analyze", "human_review")
graph.add_conditional_edges("human_review", route_by_decision, {
    "approve": "approve",
    "reject": "reject",
    "request_changes": "request_changes",
    "pending": "human_review"  # 决策无效，重新等待输入
})
graph.add_edge("approve", END)
graph.add_edge("reject", END)
graph.add_edge("request_changes", END)

checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "code-review-001"}}

print("="*60)
print("步骤1: 开始代码审查流程")
print("="*60)

result = app.invoke({
    "repo_url": "https://github.com/example/repo",
    "file_path": "/src/processor.py",
    "code_content": "",
    "analysis_result": {},
    "review_score": 0,
    "issues_found": [],
    "human_decision": "",
    "human_comment": "",
    "final_status": "",
    "review_log": []
}, config=config)

print(f"\n当前状态:")
print(f"  评分: {result['review_score']}")
print(f"  问题: {result['issues_found']}")
print(f"  状态: {result['final_status'] or '等待人工审核'}")
for log in result["review_log"]:
    print(f"  {log}")
```

这段程序描述了一个带人工审核的代码审查流程。关键在于 `human_review` 节点中的 `interrupt(prompt)` 调用——当执行流到达这个节点时，LangGraph 会暂停图的执行，把 `prompt` 返回给调用者，等待人类提供输入。注意这里有一个重要的细节：**Interrupt 节点必须配合 checkpointer 使用**，因为暂停的执行状态需要被持久化保存，否则人类输入后无法恢复执行。

## Interrupt 的执行模型

理解 Interrupt 的执行模型对于正确使用它至关重要。当一个节点调用了 `interrupt()` 时，会发生以下事情：

1. **节点函数返回**——`interrupt()` 实际上会抛出一个特殊的异常，被 LangGraph 的运行时捕获
2. **执行暂停**——图的执行流在当前节点暂停，不会继续执行后续的节点
3. **状态保存**——当前的状态快照被保存到 checkpointer 中（这就是为什么必须配置 checkpointer 的原因）
4. **返回 prompt**——`interrupt(prompt)` 的返回值（即 prompt 字符串）被作为 `invoke()` 的返回值传给调用者
5. **等待人类输入**——此时 `invoke()` 调用已经返回，程序控制权回到调用者手中
6. **恢复执行**——当人类提供了输入后，再次调用 `app.invoke()` 时，LangGraph 会从 checkpointer 中恢复之前的状态，把人类的输入作为 `interrupt()` 的返回值，然后从暂停的地方继续执行

```python
# 第一次调用: 执行到 interrupt 暂停
result1 = app.invoke(initial_state, config=config)
# result1 会包含 interrupt() 返回的 prompt
print(f"需要人工输入: {result1}")

# 人类输入: "approve 看起来不错"
human_input = "approve 看起来不错"

# 第二次调用: 传入人类输入，继续执行
result2 = app.invoke(
    Command(resume=human_input),  # 传入人类输入
    config=config
)
# result2 是完整的执行结果
print(f"最终状态: {result2['final_status']}")
```

这里有一个关键点：**第二次调用时必须使用相同的 config（特别是 thread_id）**，这样 LangGraph 才能找到之前保存的状态快照。如果 thread_id 不同，LangGraph 会认为这是一个全新的执行，而不是恢复之前的暂停执行。

## Interrupt 节点的状态设计

Interrupt 节点的设计有几个需要注意的细节。第一，`interrupt()` 可以接受任意类型的参数，不一定是字符串——你可以传入一个字典、一个对象、或者一个复杂的数据结构，只要它能够被序列化（因为要保存到 checkpointer 中）。

```python
def complex_interrupt_node(state: SomeState) -> dict:
    from langgraph.types import interrupt

    prompt_data = {
        "title": "请审核以下内容",
        "content": state["content_to_review"],
        "options": ["approve", "reject", "escalate"],
        "metadata": {
            "priority": state["priority"],
            "deadline": state["deadline"]
        }
    }

    user_response = interrupt(prompt_data)

    return {
        "human_decision": user_response.get("decision"),
        "human_comment": user_response.get("comment"),
        "review_timestamp": datetime.now().isoformat()
    }
```

第二，Interrupt 节点可以多次调用 `interrupt()`——虽然不常见，但在某些需要多轮交互的场景下是有用的。比如先让人类选择一个大类，再根据大类选择具体的子类。

```python
def multi_round_interrupt(state: SomeState) -> dict:
    from langgraph.types import interrupt

    category = interrupt("请选择类别: [technical/business/other]")
    category = category.strip().lower()

    if category == "technical":
        subcategory = interrupt("请选择子类别: [bug/feature/performance]")
        subcategory = subcategory.strip().lower()
    else:
        subcategory = "general"

    return {
        "category": category,
        "subcategory": subcategory
    }
```

第三，Interrupt 节点应该在状态中记录"等待输入"的状态，这样即使人类输入了无效值，图也能正确地回到等待状态而不是继续执行错误的路径。

```python
def safe_interrupt_node(state: SomeState) -> dict:
    from langgraph.types import interrupt

    valid_options = ["approve", "reject", "escalate"]

    user_input = interrupt(f"请选择: {valid_options}")

    if user_input not in valid_options:
        return {
            "status": "waiting_for_input",
            "last_error": f"无效输入: {user_input}"
        }

    return {
        "status": "completed",
        "decision": user_input
    }

def route_by_status(state: SomeState) -> str:
    if state["status"] == "waiting_for_input":
        return "retry_interrupt"
    return "continue"
```

## 恢复执行时的状态合并

当人类提供了输入并恢复执行时，LangGraph 会把人类的输入合并到状态中，然后继续执行。这里有一个容易混淆的点：**人类输入是通过 `Command(resume=...)` 传入的，而不是通过普通的 invoke 参数**。

```python
# ❌ 错误: 人类输入不应该放在 invoke 的 state 参数中
result = app.invoke(
    {"human_decision": "approve"},  # 这样是错误的!
    config=config
)

# ✅ 正确: 人类输入通过 Command.resume 传入
from langgraph.types import Command
result = app.invoke(
    Command(resume="approve"),  # 正确的方式
    config=config
)
```

当执行恢复时，`interrupt()` 的返回值就是 `Command.resume` 的参数值。这个值会被节点函数使用，通常会写入到状态的某个字段中（比如 `human_decision`）。然后图的执行会继续，从 Interrupt 节点之后的节点开始执行。

## 常见使用场景

Interrupt 机制在很多场景下都非常有用。下面列出几个最常见的使用场景：

**场景一：人工审核/审批**。这是最直接的使用场景——AI 做初步分析和判断，但最终决策需要人工确认。比如代码审查、贷款审批、内容审核等。在这种场景下，Interrupt 节点通常放在 AI 分析之后，根据人工决策走不同的后续路径（批准/拒绝/要求修改）。

**场景二：人工纠错/补充**。AI 生成的结果可能不够准确或完整，需要人类进行纠错或补充。比如 AI 生成的报告可能有一些事实错误，需要人类修正；或者 AI 提取的信息可能有遗漏，需要人类补充。在这种场景下，Interrupt 节点通常放在 AI 生成之后，人类可以修改 AI 的输出，然后继续执行。

**场景三：人工引导/选择**。AI 可能需要人类提供一些方向性的指导或选择。比如 AI 正在生成一份报告，需要人类选择报告的风格（正式/非正式）、受众（技术人员/非技术人员）、详细程度（简略/详细）等。在这种场景下，Interrupt 节点通常放在 AI 开始生成之前，人类的选择会影响后续的生成逻辑。

**场景四：人工干预/调试**。在开发或调试阶段，你可能需要在图的执行过程中插入一些断点，观察当前的状态，然后决定是继续执行还是修改某些参数。在这种场景下，Interrupt 节点可以临时添加到图的任何位置，用于调试目的。

## 常见陷阱与注意事项

使用 Interrupt 时有几个常见的陷阱需要避免。第一个陷阱是**忘记配置 checkpointer**。Interrupt 节点必须配合 checkpointer 使用，否则会抛出异常。这是因为暂停的执行状态需要被持久化保存，否则人类输入后无法恢复执行。

```python
# ❌ 错误: 没有 checkpointer
app = graph.compile()  # 缺少 checkpointer
result = app.invoke(state)  # 会抛出异常!

# ✅ 正确: 配置 checkpointer
from langgraph.checkpoint.memory import MemorySaver
checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)
result = app.invoke(state, config=config)
```

第二个陷阱是**恢复执行时使用了不同的 thread_id**。每次恢复执行时必须使用和暂停时相同的 thread_id，否则 LangGraph 会认为这是一个全新的执行，而不是恢复之前的暂停执行。

```python
# 第一次调用: 暂停执行
result1 = app.invoke(state, config={"configurable": {"thread_id": "session-1"}})

# ❌ 错误: 使用了不同的 thread_id
result2 = app.invoke(
    Command(resume="approve"),
    config={"configurable": {"thread_id": "session-2"}}  # 错误!
)

# ✅ 正确: 使用相同的 thread_id
result2 = app.invoke(
    Command(resume="approve"),
    config={"configurable": {"thread_id": "session-1"}}  # 正确
)
```

第三个陷阱是**Interrupt 节点中做了耗时操作**。Interrupt 节点的目的是暂停执行等待人类输入，不应该在其中做耗时的操作（比如 LLM 调用、数据库查询）。如果需要在等待人类输入之前做一些准备工作，应该把这些操作放在 Interrupt 节点之前的一个独立节点中。

```python
# ❌ 错误: 在 Interrupt 节点中做耗时操作
def bad_interrupt_node(state):
    analysis = llm.invoke(state["content"])  # 耗时操作!
    user_input = interrupt(f"分析结果: {analysis}")
    return {"decision": user_input}

# ✅ 正确: 耗时操作放在独立节点中
def prepare_node(state):
    analysis = llm.invoke(state["content"])
    return {"analysis": analysis}

def good_interrupt_node(state):
    user_input = interrupt(f"分析结果: {state['analysis']}")
    return {"decision": user_input}
```

第四个陷阱是**没有处理无效的人类输入**。人类可能会输入任何内容，包括无效的选项、空字符串、或者完全无关的内容。Interrupt 节点应该对输入进行验证，如果输入无效，应该返回到等待状态而不是继续执行错误的路径。

```python
# ❌ 错误: 没有验证输入
def unsafe_interrupt(state):
    decision = interrupt("请选择: [approve/reject]")
    return {"decision": decision}  # 可能是任何值!

# ✅ 正确: 验证输入
def safe_interrupt(state):
    decision = interrupt("请选择: [approve/reject]")
    valid_options = {"approve", "reject"}
    if decision not in valid_options:
        return {"status": "waiting", "error": f"无效输入: {decision}"}
    return {"status": "completed", "decision": decision}
```

总的来说，Interrupt 是 LangGraph 中实现人机协作的核心机制。它让图能够在任意节点暂停执行，等待人类输入后再继续，这为构建需要人工审核、人工决策或人工纠错的系统提供了强大的能力。理解 Interrupt 的执行模型、正确配置 checkpointer、合理设计 Interrupt 节点的状态，是成功使用这个机制的关键。
