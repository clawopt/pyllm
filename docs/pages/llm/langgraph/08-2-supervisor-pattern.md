# 8.2 Supervisor 模式深度解析

> 在上一节中我们介绍了 Supervisor 模式的基本概念——一个中央的 Supervisor Agent 负责协调多个 Worker Agent。但实际应用中，Supervisor 模式有很多变体和高级用法。这一节我们会深入探讨 Supervisor 的不同实现方式、如何处理 Worker 之间的依赖关系、如何实现动态的 Worker 注册与注销，以及如何优化 Supervisor 的决策逻辑。

## Supervisor 的三种实现方式

LangGraph 中实现 Supervisor 有三种主要方式，各有其适用场景：

### 方式一：条件边路由（最常用）

这是我们在上一节中使用的实现方式——Supervisor 通过条件边根据当前状态决定下一步该调用哪个 Worker。

```python
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END

class SupervisorState(TypedDict):
    user_query: str
    research_result: str
    code_result: str
    review_result: str
    final_answer: str
    current_step: str
    supervisor_log: list[str]

def supervisor_router(state: SupervisorState) -> Literal["researcher", "coder", "reviewer", "finalize"]:
    if not state.get("research_result"):
        return "researcher"
    if not state.get("code_result"):
        return "coder"
    if not state.get("review_result"):
        return "reviewer"
    return "finalize"

# Worker 节点定义...
def researcher_node(state: SupervisorState) -> dict:
    return {
        "research_result": f"关于 '{state['user_query'][:20]}...' 的研究结果",
        "supervisor_log": ["[Supervisor] 分配任务给研究员"]
    }

def coder_node(state: SupervisorState) -> dict:
    return {
        "code_result": f"基于研究的代码实现",
        "supervisor_log": ["[Supervisor] 分配任务给程序员"]
    }

def reviewer_node(state: SupervisorState) -> dict:
    return {
        "review_result": f"代码审查结果",
        "supervisor_log": ["[Supervisor] 分配任务给审查员"]
    }

def finalize_node(state: SupervisorState) -> dict:
    return {
        "final_answer": f"综合所有结果: {state['research_result']}",
        "supervisor_log": ["[Supervisor] 任务完成"]
    }

supervisor_graph = StateGraph(SupervisorState)
supervisor_graph.add_node("researcher", researcher_node)
supervisor_graph.add_node("coder", coder_node)
supervisor_graph.add_node("reviewer", reviewer_node)
supervisor_graph.add_node("finalize", finalize_node)

supervisor_graph.add_edge(START, "researcher")
supervisor_graph.add_edge("researcher", "coder")
supervisor_graph.add_edge("coder", "reviewer")
supervisor_graph.add_edge("reviewer", "finalize")
supervisor_graph.add_edge("finalize", END)

app = supervisor_graph.compile()
```

这种方式的特点是**流程固定、顺序清晰**——Worker 的调用顺序在图的拓扑结构中就确定了，Supervisor 的路由函数只是检查每个步骤是否完成。适合那些有明确先后顺序的任务。

### 方式二：LLM 驱动的动态路由

当任务之间的依赖关系不固定，或者需要根据任务内容动态决定下一步时，可以让 LLM 来做路由决策。

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

supervisor_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个任务调度员。根据当前的任务状态，决定下一步应该调用哪个 Worker。

可用的 Worker:
- researcher: 研究员，负责信息收集和分析
- coder: 程序员，负责代码实现
- reviewer: 审查员，负责质量检查
- writer: 写手，负责文档撰写

当前状态:
- 已完成的工作: {completed_work}
- 待处理的任务: {pending_tasks}
- 用户需求: {user_query}

请只返回一个 Worker 名称，不要返回其他内容。"""),
    ("user", "{user_query}\n已完成: {completed_work}\n待处理: {pending_tasks}")
])

supervisor_chain = supervisor_prompt | llm

def llm_supervisor(state: SupervisorState) -> dict:
    completed = []
    if state.get("research_result"):
        completed.append("research")
    if state.get("code_result"):
        completed.append("coding")
    if state.get("review_result"):
        completed.append("review")
    if state.get("doc_result"):
        completed.append("documentation")

    pending = []
    if not state.get("research_result"):
        pending.append("research")
    if state.get("research_result") and not state.get("code_result"):
        pending.append("coding")
    if state.get("code_result") and not state.get("review_result"):
        pending.append("review")
    if state.get("review_result") and not state.get("doc_result"):
        pending.append("documentation")

    decision = supervisor_chain.invoke({
        "user_query": state["user_query"],
        "completed_work": ", ".join(completed) if completed else "无",
        "pending_tasks": ", ".join(pending) if pending else "无"
    }).content.strip().lower()

    return {
        "current_step": decision,
        "supervisor_log": [f"[Supervisor LLM] 决策调用: {decision}"]
    }

def llm_supervisor_router(state: SupervisorState) -> str:
    decision = state.get("current_step", "")
    if decision == "researcher":
        return "researcher"
    elif decision == "coder":
        return "coder"
    elif decision == "reviewer":
        return "reviewer"
    elif decision == "writer":
        return "writer"
    return "finalize"

# Worker 节点...
def doc_writer_node(state: SupervisorState) -> dict:
    return {
        "doc_result": "文档已生成",
        "supervisor_log": ["[Supervisor LLM] 分配任务给写手"]
    }

llm_supervisor_graph = StateGraph(SupervisorState)
llm_supervisor_graph.add_node("researcher", researcher_node)
llm_supervisor_graph.add_node("coder", coder_node)
llm_supervisor_graph.add_node("reviewer", reviewer_node)
llm_supervisor_graph.add_node("writer", doc_writer_node)
llm_supervisor_graph.add_node("llm_supervisor", llm_supervisor)
llm_supervisor_graph.add_node("finalize", finalize_node)

llm_supervisor_graph.add_edge(START, "llm_supervisor")
llm_supervisor_graph.add_conditional_edges("llm_supervisor", llm_supervisor_router, {
    "researcher": "researcher",
    "coder": "coder",
    "reviewer": "reviewer",
    "writer": "writer",
    "finalize": "finalize"
})
llm_supervisor_graph.add_edge("researcher", "llm_supervisor")
llm_supervisor_graph.add_edge("coder", "llm_supervisor")
llm_supervisor_graph.add_edge("reviewer", "llm_supervisor")
llm_supervisor_graph.add_edge("writer", "llm_supervisor")
llm_supervisor_graph.add_edge("finalize", END)

app = llm_supervisor_graph.compile()
```

这种 LLM 驱动的 Supervisor 更加灵活——它能根据任务的具体内容动态调整执行顺序，甚至可以跳过某些步骤（如果 LLM 判断某步不需要）。但代价是每次路由决策都需要一次 LLM 调用，增加了延迟和成本。

### 方式三：并行分发 + 结果聚合

当多个 Worker 可以并行工作时，Supervisor 可以同时分发任务给多个 Worker，然后收集所有结果进行聚合。

```python
class ParallelSupervisorState(TypedDict):
    user_query: str
    researcher_result: str
    coder_result: str
    writer_result: str
    all_results: dict
    final_answer: str

def parallel_dispatcher(state: ParallelSupervisorState) -> dict:
    return {
        "supervisor_log": ["[Supervisor] 并行分发任务给所有 Worker"]
    }

def aggregator(state: ParallelSupervisorState) -> dict:
    all_results = {
        "researcher": state.get("researcher_result", ""),
        "coder": state.get("coder_result", ""),
        "writer": state.get("writer_result", "")
    }
    return {
        "all_results": all_results,
        "supervisor_log": ["[Supervisor] 聚合所有 Worker 结果"]
    }

parallel_supervisor_graph = StateGraph(ParallelSupervisorState)
parallel_supervisor_graph.add_node("dispatcher", parallel_dispatcher)
parallel_supervisor_graph.add_node("researcher", researcher_node)
parallel_supervisor_graph.add_node("coder", coder_node)
parallel_supervisor_graph.add_node("writer", doc_writer_node)
parallel_supervisor_graph.add_node("aggregator", aggregator)
parallel_supervisor_graph.add_node("finalize", finalize_node)

parallel_supervisor_graph.add_edge(START, "dispatcher")
parallel_supervisor_graph.add_edge("dispatcher", "researcher")  # 并行
parallel_supervisor_graph.add_edge("dispatcher", "coder")       # 并行
parallel_supervisor_graph.add_edge("dispatcher", "writer")     # 并行
parallel_supervisor_graph.add_edge("researcher", "aggregator")
parallel_supervisor_graph.add_edge("coder", "aggregator")
parallel_supervisor_graph.add_edge("writer", "aggregator")
parallel_supervisor_graph.add_edge("aggregator", "finalize")
parallel_supervisor_graph.add_edge("finalize", END)

app = parallel_supervisor_graph.compile()
```

这种并行模式的执行流程是：`START → dispatcher → [researcher, coder, writer] → aggregator → finalize → END`。注意从 `dispatcher` 同时引出了三条边到三个 Worker 节点，实现了扇出（fan-out）；三个 Worker 的结果都汇聚到 `aggregator` 节点，实现了扇入（fan-in）。

## Worker 之间的依赖关系处理

在实际应用中，Worker 之间往往不是完全独立的——某些 Worker 的输出是其他 Worker 的输入。Supervisor 需要正确处理这些依赖关系。

### 场景一：线性依赖链

```
用户请求 → [研究员] → [程序员] → [测试员] → [部署员] → 完成
```

这种情况下，每个 Worker 的输出是下一个 Worker 的输入。Supervisor 的路由逻辑很简单：按固定顺序依次调用。

### 场景二：条件依赖

```
用户请求 → [研究员]
              ↓
         [分类器] → {技术类} → [程序员]
                      → {业务类} → [分析师]
                      → {其他类} → [通用客服]
```

这里研究员的输出需要先经过一个分类器，根据分类结果再路由到不同的 Worker。

```python
def classifier_node(state: SupervisorState) -> dict:
    research = state["research_result"]
    if "技术" in research or "代码" in research or "bug" in research:
        category = "technical"
    elif "业务" in research or "流程" in research:
        category = "business"
    else:
        category = "other"
    return {
        "task_category": category,
        "supervisor_log": [f"[Supervisor] 任务分类为: {category}"]
    }

def route_by_category(state: SupervisorState) -> str:
    category = state.get("task_category", "")
    if category == "technical":
        return "coder"
    elif category == "business":
        return "analyst"
    else:
        return "general_support"
```

### 场景三：部分并行 + 依赖

```
用户请求 → [研究员]
              ↓
         [分类器] → {技术类} → [程序员] → [审查员]
                      → {业务类} → [分析师]
                      → {混合类} → [程序员] + [分析师] → [审查员]
```

对于混合类任务，可能需要程序员和分析师并行工作，然后一起交给审查员。

```python
def route_mixed(state: SupervisorState) -> str:
    if not state.get("code_result") or not state.get("analysis_result"):
        return "parallel_execute"
    return "reviewer"

def parallel_execute_node(state: SupervisorState) -> dict:
    return {
        "supervisor_log": ["[Supervisor] 并行调用程序员和分析师"]
    }

parallel_dep_graph = StateGraph(SupervisorState)
parallel_dep_graph.add_node("classifier", classifier_node)
parallel_dep_graph.add_node("coder", coder_node)
parallel_dep_graph.add_node("analyst", lambda s: {"analysis_result": "业务分析完成"})
parallel_dep_graph.add_node("parallel_execute", parallel_execute_node)
parallel_dep_graph.add_node("reviewer", reviewer_node)
parallel_dep_graph.add_node("finalize", finalize_node)

parallel_dep_graph.add_edge(START, "researcher")
parallel_dep_graph.add_edge("researcher", "classifier")
parallel_dep_graph.add_conditional_edges("classifier", route_by_category, {
    "coder": "coder",
    "analyst": "analyst",
    "general_support": "finalize"
})
parallel_dep_graph.add_conditional_edges("coder", route_mixed, {
    "parallel_execute": "parallel_execute",
    "reviewer": "reviewer"
})
parallel_dep_graph.add_conditional_edges("analyst", route_mixed, {
    "parallel_execute": "parallel_execute",
    "reviewer": "reviewer"
})
parallel_dep_graph.add_edge("parallel_execute", "coder")
parallel_dep_graph.add_edge("parallel_execute", "analyst")
parallel_dep_graph.add_edge("coder", "reviewer")
parallel_dep_graph.add_edge("analyst", "reviewer")
parallel_dep_graph.add_edge("reviewer", "finalize")
parallel_dep_graph.add_edge("finalize", END)
parallel_dep_graph.add_edge("general_support", END)
```

## 动态 Worker 注册与注销

在某些场景下，Worker 的集合不是固定的——可能需要根据运行时的条件动态注册或注销 Worker。比如一个插件化的系统，不同的插件提供不同的 Worker 能力。

```python
from typing import Callable, Dict, Any

class WorkerRegistry:
    def __init__(self):
        self._workers: Dict[str, Callable] = {}

    def register(self, name: str, worker_func: Callable) -> None:
        self._workers[name] = worker_func
        print(f"[Registry] Worker '{name}' 已注册")

    def unregister(self, name: str) -> None:
        if name in self._workers:
            del self._workers[name]
            print(f"[Registry] Worker '{name}' 已注销")

    def get(self, name: str) -> Callable | None:
        return self._workers.get(name)

    def list_all(self) -> list[str]:
        return list(self._workers.keys())

# 全局 Worker 注册表
worker_registry = WorkerRegistry()

# 注册基础 Worker
worker_registry.register("researcher", researcher_node)
worker_registry.register("coder", coder_node)
worker_registry.register("reviewer", reviewer_node)

# 动态注册新 Worker
def plugin_worker_node(state: SupervisorState) -> dict:
    return {
        "plugin_result": "插件处理完成",
        "supervisor_log": ["[Supervisor] 调用插件 Worker"]
    }

worker_registry.register("plugin_worker", plugin_worker_node)

def dynamic_supervisor_router(state: SupervisorState) -> str:
    available_workers = worker_registry.list_all()
    print(f"[Supervisor] 可用的 Worker: {available_workers}")
    
    # 根据状态决定调用哪个 Worker
    if not state.get("research_result") and "researcher" in available_workers:
        return "researcher"
    if not state.get("code_result") and "coder" in available_workers:
        return "coder"
    if not state.get("review_result") and "reviewer" in available_workers:
        return "reviewer"
    if not state.get("plugin_result") and "plugin_worker" in available_workers:
        return "plugin_worker"
    return "finalize"
```

这种动态注册机制让系统具有了很好的扩展性——新增一个 Worker 只需要调用 `worker_registry.register()`，不需要修改 Supervisor 的代码。
