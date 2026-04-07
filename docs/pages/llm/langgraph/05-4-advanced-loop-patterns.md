# 5.4 高级循环模式：嵌套循环与并行迭代

> 前面几节我们讨论的都是单层循环——一个节点通过条件边回到自身或前序节点，形成简单的闭环。但在实际应用中，你可能需要更复杂的循环结构：比如两层嵌套的循环（外层控制整体迭代、内层处理子任务）、或者多个独立的循环并行执行。这一节我们会探讨这些高级循环模式的实现方法，以及如何用 LangGraph 的图结构来表达它们。

## 嵌套循环：外层控制、内层执行

嵌套循环是指一个循环结构内部还包含另一个循环结构。在编程中这很常见——比如遍历一个列表，对每个元素再执行多次尝试；或者在优化算法中，外层控制迭代次数、内层执行具体的搜索步骤。在 LangGraph 中，嵌套循环可以通过子图来实现——外层循环在父图中，内层循环封装为子图。

```python
from typing import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, START, END

class InnerLoopState(TypedDict):
    item: str
    attempts: Annotated[int, operator.add]
    max_attempts: int
    success: bool
    result: str
    inner_log: Annotated[list[str], operator.add]

def inner_process(state: InnerLoopState) -> dict:
    item = state["item"]
    attempt = state["attempts"] + 1

    import random
    random.seed(hash(item + str(attempt)) % 10000)
    success = random.random() > 0.6

    if success:
        return {
            "success": True,
            "result": f"{item} 处理成功 (第{attempt}次尝试)",
            "inner_log": [f"  [内层] {item}: ✅ 第{attempt}次成功"]
        }
    else:
        return {
            "success": False,
            "inner_log": [f"  [内层] {item}: ❌ 第{attempt}次失败"]
        }

def inner_should_continue(state: InnerLoopState) -> str:
    if state["success"]:
        return "success"
    if state["attempts"] >= state["max_attempts"]:
        return "exhausted"
    return "retry"

inner_graph = StateGraph(InnerLoopState)
inner_graph.add_node("process", inner_process)
inner_graph.add_node("success", lambda s: {"result": s["result"]})
inner_graph.add_node("exhausted", lambda s: {"result": f"{s['item']} 处理失败"})

inner_graph.add_edge(START, "process")
inner_graph.add_conditional_edges("process", inner_should_continue, {
    "retry": "process",
    "success": "success",
    "exhausted": "exhausted"
})
inner_graph.add_edge("success", END)
inner_graph.add_edge("exhausted", END)

compiled_inner = inner_graph.compile()

class OuterLoopState(TypedDict):
    items: list[str]
    current_index: int
    results: list[str]
    outer_log: Annotated[list[str], operator.add]

def outer_get_next(state: OuterLoopState) -> dict:
    idx = state["current_index"]
    items = state["items"]

    if idx >= len(items):
        return {"outer_log": ["[外层] 所有项目已处理完毕"]}

    item = items[idx]
    return {"outer_log": [f"[外层] 处理项目 {idx+1}/{len(items)}: {item}"]}

def outer_should_continue(state: OuterLoopState) -> str:
    if state["current_index"] >= len(state["items"]):
        return "done"
    return "process_next"

def run_inner_loop(state: OuterLoopState) -> dict:
    idx = state["current_index"]
    item = state["items"][idx]

    inner_result = compiled_inner.invoke({
        "item": item,
        "attempts": 0,
        "max_attempts": 5,
        "success": False,
        "result": "",
        "inner_log": []
    })

    results = list(state["results"])
    results.append(inner_result["result"])

    return {
        "results": results,
        "outer_log": inner_result["inner_log"]
    }

def advance_index(state: OuterLoopState) -> dict:
    return {"current_index": state["current_index"] + 1}

outer_graph = StateGraph(OuterLoopState)
outer_graph.add_node("get_next", outer_get_next)
outer_graph.add_node("run_inner", run_inner_loop)
outer_graph.add_node("advance", advance_index)
outer_graph.add_node("done", lambda s: {"outer_log": ["[外层] 完成"]})

outer_graph.add_edge(START, "get_next")
outer_graph.add_conditional_edges("get_next", outer_should_continue, {
    "process_next": "run_inner",
    "done": "done"
})
outer_graph.add_edge("run_inner", "advance")
outer_graph.add_edge("advance", "get_next")
outer_graph.add_edge("done", END)

app = outer_graph.compile()

result = app.invoke({
    "items": ["item-A", "item-B", "item-C"],
    "current_index": 0,
    "results": [],
    "outer_log": []
})

print(f"\n{'='*60}")
print(f"嵌套循环执行结果")
print(f"{'='*60}")
for entry in result["outer_log"]:
    print(entry)

print(f"\n最终结果:")
for i, res in enumerate(result["results"], 1):
    print(f"  {i}. {res}")
```

这个嵌套循环的例子展示了如何用子图来实现内层循环。外层循环（`outer_graph`）遍历一个项目列表，对每个项目调用内层循环（`inner_graph`）来处理。内层循环本身也是一个完整的图，有自己的状态、节点和循环逻辑。执行流程是这样的：

```
外层 START → get_next → (判断是否还有项目)
                    ↓ 有项目
                 run_inner (调用内层子图)
                    ↓ 内层循环执行完毕
                 advance (索引+1)
                    ↓
                 get_next (回到外层循环开始)
                    ↓ 所有项目处理完毕
                 done → END
```

内层子图的执行是独立的——它有自己的状态空间，执行完成后把结果返回给外层图。这种封装的好处是内层循环的逻辑可以独立测试和复用，外层图不需要关心内层循环的内部实现细节。

## 并行循环：多个独立循环同时执行

有些场景下你需要多个独立的循环同时执行，比如同时处理多个数据集、同时运行多个优化算法、同时监控多个指标。在 LangGraph 中，这可以通过从同一个节点引出多条边到不同的循环子图来实现。

```python
from typing import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, START, END

class ParallelLoopState(TypedDict):
    dataset_a: list[int]
    dataset_b: list[int]
    dataset_c: list[int]
    results_a: list[int]
    results_b: list[int]
    results_c: list[int]
    combined_results: dict
    parallel_log: Annotated[list[str], operator.add]

def process_dataset_a(state: ParallelLoopState) -> dict:
    data = state["dataset_a"]
    results = [x * 2 for x in data]
    return {
        "results_a": results,
        "parallel_log": [f"[循环A] 处理了 {len(data)} 个元素"]
    }

def process_dataset_b(state: ParallelLoopState) -> dict:
    data = state["dataset_b"]
    results = [x + 10 for x in data]
    return {
        "results_b": results,
        "parallel_log": [f"[循环B] 处理了 {len(data)} 个元素"]
    }

def process_dataset_c(state: ParallelLoopState) -> dict:
    data = state["dataset_c"]
    results = [x ** 2 for x in data]
    return {
        "results_c": results,
        "parallel_log": [f"[循环C] 处理了 {len(data)} 个元素"]
    }

def merge_parallel_results(state: ParallelLoopState) -> dict:
    combined = {
        "dataset_a": state["results_a"],
        "dataset_b": state["results_b"],
        "dataset_c": state["results_c"],
        "summary": {
            "total_elements": len(state["dataset_a"]) + len(state["dataset_b"]) + len(state["dataset_c"]),
            "total_results": len(state["results_a"]) + len(state["results_b"]) + len(state["results_c"])
        }
    }
    return {
        "combined_results": combined,
        "parallel_log": [f"[合并] 所有循环已完成"]
    }

parallel_graph = StateGraph(ParallelLoopState)
parallel_graph.add_node("process_a", process_dataset_a)
parallel_graph.add_node("process_b", process_dataset_b)
parallel_graph.add_node("process_c", process_dataset_c)
parallel_graph.add_node("merge", merge_parallel_results)

parallel_graph.add_edge(START, "process_a")
parallel_graph.add_edge(START, "process_b")  # 并行启动
parallel_graph.add_edge(START, "process_c")  # 并行启动
parallel_graph.add_edge("process_a", "merge")
parallel_graph.add_edge("process_b", "merge")
parallel_graph.add_edge("process_c", "merge")
parallel_graph.add_edge("merge", END)

app = parallel_graph.compile()

result = app.invoke({
    "dataset_a": [1, 2, 3, 4, 5],
    "dataset_b": [10, 20, 30],
    "dataset_c": [2, 3, 4, 5],
    "results_a": [], "results_b": [], "results_c": [],
    "combined_results": {},
    "parallel_log": []
})

print(f"\n{'='*60}")
print(f"并行循环执行结果")
print(f"{'='*60}")
for entry in result["parallel_log"]:
    print(entry)

print(f"\n合并结果:")
print(f"  数据集A (x2): {result['results_a']}")
print(f"  数据集B (+10): {result['results_b']}")
print(f"  数据集C (x²): {result['results_c']}")
print(f"\n汇总: {result['combined_results']['summary']}")
```

这个并行循环的例子展示了如何从 START 节点同时引出三条边到三个不同的处理节点，实现扇出效果。三个处理节点各自独立地处理自己的数据集，然后全部汇聚到 `merge` 节点进行结果合并。

需要注意的是，在 LangGraph 默认的单线程执行模式下，这三个"并行"循环实际上是串行执行的（一个接一个）。图的结构表达了"它们之间没有依赖关系、可以并行"的语义，但实际的并行执行需要借助其他机制（如 Send API + 异步执行框架）。不过，即使不真正并行，这种结构化表达仍然有价值——它清晰地说明了三个循环的独立性，便于后续优化为真正的并行执行。

## 动态循环：根据状态决定循环次数

有些场景下循环的次数不是固定的，而是根据执行过程中的状态动态决定的。比如在优化算法中，可能设置了最大迭代次数，但如果连续 N 次迭代都没有明显改进，就提前结束；或者在搜索算法中，一旦找到满足条件的解就立即终止，不再继续迭代。

```python
from typing import TypedDict, Annotated
import operator
import random
from langgraph.graph import StateGraph, START, END

class DynamicLoopState(TypedDict):
    target_value: int
    current_value: int
    iteration_count: Annotated[int, operator.add]
    max_iterations: int
    stagnation_count: Annotated[int, operator.add]
    max_stagnation: int
    found_solution: bool
    best_value: int
    dynamic_log: Annotated[list[str], operator.add]

def dynamic_step(state: DynamicLoopState) -> dict:
    target = state["target_value"]
    current = state["current_value"]
    iters = state["iteration_count"] + 1

    random.seed(iters * 1000)
    delta = random.randint(-5, 5)
    new_value = current + delta

    improved = abs(new_value - target) < abs(current - target)

    log_msg = (
        f"[迭代 {iters}] 当前: {current}, 目标: {target}, "
        f"调整: {delta:+d}, 新值: {new_value}, "
        f"改进: {'是' if improved else '否'}"
    )

    return {
        "current_value": new_value,
        "dynamic_log": [log_msg]
    }

def dynamic_should_continue(state: DynamicLoopState) -> str:
    current = state["current_value"]
    target = state["target_value"]
    iters = state["iteration_count"]
    max_iters = state["max_iterations"]
    stagnation = state["stagnation_count"]
    max_stag = state["max_stagnation"]

    if abs(current - target) <= 1:
        return "found"

    if iters >= max_iters:
        return "max_iterations"

    if stagnation >= max_stag:
        return "stagnated"

    return "continue"

def check_stagnation(state: DynamicLoopState) -> dict:
    target = state["target_value"]
    current = state["current_value"]
    best = state["best_value"]

    current_dist = abs(current - target)
    best_dist = abs(best - target) if best != 0 else float('inf')

    if current_dist < best_dist:
        return {
            "best_value": current,
            "stagnation_count": 0,
            "dynamic_log": [f"[检查] 发现新的最佳值: {current}"]
        }
    else:
        return {
            "stagnation_count": 1,
            "dynamic_log": [f"[检查] 未改善 (停滞计数+1)"]
        }

dynamic_graph = StateGraph(DynamicLoopState)
dynamic_graph.add_node("step", dynamic_step)
dynamic_graph.add_node("check_stag", check_stagnation)
dynamic_graph.add_node("found", lambda s: {
    "found_solution": True,
    "dynamic_log": [f"[完成] ✅ 找到解: {s['current_value']}"]
})
dynamic_graph.add_node("max_iters", lambda s: {
    "found_solution": False,
    "dynamic_log": [f"[完成] ⚠️ 达到最大迭代次数"]
})
dynamic_graph.add_node("stagnated", lambda s: {
    "found_solution": False,
    "dynamic_log": [f"[完成] ⚠️ 改善停滞 ({s['stagnation_count']}次无改善)"]
})

dynamic_graph.add_edge(START, "step")
dynamic_graph.add_edge("step", "check_stag")
dynamic_graph.add_conditional_edges("check_stag", dynamic_should_continue, {
    "continue": "step",
    "found": "found",
    "max_iterations": "max_iters",
    "stagnated": "stagnated"
})
dynamic_graph.add_edge("found", END)
dynamic_graph.add_edge("max_iters", END)
dynamic_graph.add_edge("stagnated", END)

app = dynamic_graph.compile()

result = app.invoke({
    "target_value": 100,
    "current_value": 50,
    "iteration_count": 0,
    "max_iterations": 20,
    "stagnation_count": 0,
    "max_stagnation": 5,
    "found_solution": False,
    "best_value": 0,
    "dynamic_log": []
})

print(f"\n{'='*60}")
print(f"动态循环执行结果")
print(f"{'='*60}")
for entry in result["dynamic_log"]:
    print(entry)

print(f"\n最终状态:")
print(f"  是否找到解: {result['found_solution']}")
print(f"  最终值: {result['current_value']}")
print(f"  最佳值: {result['best_value']}")
print(f"  迭代次数: {result['iteration_count']}")
```

这个动态循环的例子展示了三种不同的退出条件：

1. **找到解**：当前值足够接近目标值（误差 <= 1），立即退出
2. **达到最大迭代次数**：即使还没找到解，迭代次数达到上限后也退出
3. **改善停滞**：连续多次迭代都没有改善，说明当前策略可能无效，提前退出

`check_stag` 节点负责检查当前迭代是否带来了改善——如果有改善就更新最佳值并重置停滞计数器；如果没有改善就增加停滞计数器。`dynamic_should_continue` 条件函数综合这三个条件来决定是继续迭代还是退出。

这种多条件退出的设计在实际应用中非常常见——它能避免无效的长时间循环，同时保证在合理次数内尽可能找到好的解。

## 循环中的错误处理与恢复

循环结构中的错误处理比线性流程更复杂，因为错误可能发生在任意一次迭代中，而且你需要决定是终止整个循环、跳过当前迭代继续下一次、还是尝试恢复当前迭代。

```python
from typing import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, START, END

class ErrorHandlingLoopState(TypedDict):
    items: list[str]
    processed_items: list[str]
    failed_items: list[str]
    error_messages: list[str]
    current_index: int
    loop_log: Annotated[list[str], operator.add]

def process_item_with_error_handling(state: ErrorHandlingLoopState) -> dict:
    idx = state["current_index"]
    items = state["items"]

    if idx >= len(items):
        return {"loop_log": ["[循环] 所有项目已处理"]}

    item = items[idx]

    try:
        if "error" in item.lower():
            raise ValueError(f"项目 '{item}' 包含错误标记")

        processed = f"已处理: {item}"
        processed_list = list(state["processed_items"])
        processed_list.append(processed)

        return {
            "processed_items": processed_list,
            "loop_log": [f"[处理] ✅ {item}"]
        }

    except Exception as e:
        failed_list = list(state["failed_items"])
        failed_list.append(item)

        error_msgs = list(state["error_messages"])
        error_msgs.append(str(e))

        return {
            "failed_items": failed_list,
            "error_messages": error_msgs,
            "loop_log": [f"[处理] ❌ {item} - {str(e)}"]
        }

def should_continue_loop(state: ErrorHandlingLoopState) -> str:
    if state["current_index"] >= len(state["items"]):
        return "done"
    return "continue"

def advance_index(state: ErrorHandlingLoopState) -> dict:
    return {"current_index": state["current_index"] + 1}

def summarize_results(state: ErrorHandlingLoopState) -> dict:
    total = len(state["items"])
    processed = len(state["processed_items"])
    failed = len(state["failed_items"])

    summary = (
        f"总计: {total} | 成功: {processed} | 失败: {failed}"
    )

    return {"loop_log": [f"[汇总] {summary}"]}

error_handling_graph = StateGraph(ErrorHandlingLoopState)
error_handling_graph.add_node("process", process_item_with_error_handling)
error_handling_graph.add_node("advance", advance_index)
error_handling_graph.add_node("summarize", summarize_results)

error_handling_graph.add_edge(START, "process")
error_handling_graph.add_conditional_edges("process", should_continue_loop, {
    "continue": "advance",
    "done": "summarize"
})
error_handling_graph.add_edge("advance", "process")
error_handling_graph.add_edge("summarize", END)

app = error_handling_graph.compile()

result = app.invoke({
    "items": ["item-1", "item-2-error", "item-3", "item-4-error", "item-5"],
    "processed_items": [],
    "failed_items": [],
    "error_messages": [],
    "current_index": 0,
    "loop_log": []
})

print(f"\n{'='*60}")
print(f"带错误处理的循环执行结果")
print(f"{'='*60}")
for entry in result["loop_log"]:
    print(entry)

print(f"\n成功处理的项目: {result['processed_items']}")
print(f"失败的项目: {result['failed_items']}")
print(f"错误信息: {result['error_messages']}")
```

这个错误处理循环的例子展示了"跳过错误继续"的策略。`process_item_with_error_handling` 节点用 try-except 包裹了处理逻辑，如果某个项目处理失败，不会抛出异常中断整个循环，而是把失败的项目和错误信息记录下来，然后继续处理下一个项目。这种策略在很多场景下是合理的——比如批量处理数据时，个别记录的失败不应该影响其他记录的处理。

当然，根据业务需求的不同，你也可以选择其他策略：

- **终止策略**：遇到第一个错误就立即终止整个循环
- **重试策略**：失败的项目重试 N 次后才跳过
- **降级策略**：失败的项目使用降级逻辑处理（如返回默认值）
- **隔离策略**：把失败的项目放到单独的队列，稍后人工处理

## 总结：高级循环模式的适用场景

本节讨论的几种高级循环模式各有其适用场景：

- **嵌套循环**：适用于需要两层或多层迭代控制的场景，如外层遍历数据集、内层对每个元素进行多次尝试
- **并行循环**：适用于多个独立的任务需要同时执行的场景，如同时处理多个数据源、同时运行多个算法
- **动态循环**：适用于循环次数或退出条件需要在执行过程中动态决定的场景，如优化算法、搜索算法
- **错误处理循环**：适用于需要容错能力的批量处理场景，如数据导入、批量 API 调用

选择哪种模式取决于你的具体需求——如果任务之间有依赖关系，用嵌套循环；如果任务之间完全独立，用并行循环；如果需要根据执行情况动态调整，用动态循环；如果需要容错能力，用错误处理循环。理解这些模式的差异和适用场景，能帮助你在面对复杂需求时快速做出正确的架构决策。
