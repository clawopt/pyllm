# 7.2 时间旅行调试与状态回溯

> Checkpointing 不仅能让我们在崩溃后恢复执行，还开启了一种全新的调试范式——**时间旅行（Time Travel）**。想象一下：你发现图的最终输出不对，但你不知道是在哪一步出了问题。有了完整的 checkpoint 历史，你可以"回到过去"，查看任意历史时刻的完整状态，甚至从某个历史点重新开始后续的执行。这就像给图的状态机装上了一台时光机，让调试过程从"猜测发生了什么"变成了"亲眼看到发生了什么"。

## 时间旅行的核心概念

时间旅行调试的核心能力是**回溯到任意历史 checkpoint 并查看/修改状态**。LangGraph 通过 checkpointer 的 API 提供了这个能力：

```python
from typing import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

class DebugState(TypedDict):
    input_value: int
    doubled: int
    tripled: int
    squared: int
    final_result: int
    step_log: Annotated[list[str], operator.add]

def double_it(state: DebugState) -> dict:
    v = state["input_value"]
    return {
        "doubled": v * 2,
        "step_log": [f"[double] {v} → {v * 2}"]
    }

def triple_it(state: DebugState) -> dict:
    v = state["doubled"]
    return {
        "tripled": v * 3,
        "step_log": [f"[triple] {v} → {v * 3}"]
    }

def square_it(state: DebugState) -> dict:
    v = state["tripled"]
    return {
        "squared": v ** 2,
        "final_result": v ** 2,
        "step_log": [f"[square] {v} → {v ** 2}"]
    }

debug_graph = StateGraph(DebugState)
debug_graph.add_node("double", double_it)
debug_graph.add_node("triple", triple_it)
debug_graph.add_node("square", square_it)

debug_graph.add_edge(START, "double")
debug_graph.add_edge("double", "triple")
debug_graph.add_edge("triple", "square")
debug_graph.add_edge("square", END)

checkpointer = MemorySaver()
app = debug_graph.compile(checkpointer=checkpointer)
config = {"configurable": {"thread_id": "debug-session"}}

print("=== 执行完整流程 ===")
result = app.invoke({
    "input_value": 5,
    "doubled": 0, "tripled": 0, "squared": 0,
    "final_result": 0, "step_log": []
}, config=config)

print(f"最终结果: {result['final_result']}")
for entry in result["step_log"]:
    print(f"  {entry}")

# === 时间旅行：查看所有历史 checkpoint ===
print("\n=== 历史Checkpoint列表 ===")
checkpoints = list(checkpointer.list(config))
for i, cp in enumerate(checkpoints):
    vals = cp.channel_values
    parent_ts = cp.parent_config.get("ts", "")[:19] if cp.parent_config else "root"
    source = cp.metadata.get("source", "?")
    step_num = cp.metadata.get("step", i)
    
    print(f"\n  [{i}] Step={step_num} | 来源: {source}")
    print(f"      input_value={vals.get('input_value')}, "
          f"doubled={vals.get('doubled')}, "
          f"tripled={vals.get('tripled')}, "
          f"squared={vals.get('squared')}")

# === 回溯到特定checkpoint并查看当时的状态 ===
print("\n=== 回溯到第1个checkpoint（double之后）===")
if len(checkpoints) > 1:
    target_cp = checkpoints[1]
    state_at_that_time = target_cp.channel_values
    
    print(f"此时 input_value: {state_at_that_time['input_value']}")
    print(f"此时 doubled: {state_at_that_time['doubled']}")
    print(f"此时 tripled: {state_at_that_time['tripled']} (尚未计算)")
    print(f"日志: {state_at_that_time['step_log']}")
```

这个例子展示了如何获取和浏览完整的 checkpoint 历史。每个 checkpoint 都是一个独立的时间切片——它包含了在那个时刻图中所有状态的完整快照。通过遍历 `checkpointer.list(config)` 返回的列表，你可以看到每一步之后状态是如何演变的。

## 从历史点重新分支执行

比单纯查看历史更强大的能力是**从某个历史 checkpoint 创建一个新的分支（fork）并继续执行**。这在调试时特别有用——你发现某一步的输入有问题，想看看如果修正了输入后续步骤会怎么变化。

```python
print("\n=== 分支实验：从步骤1后改变输入值 ===")
if len(checkpoints) > 1:
    # 获取步骤1后的checkpoint作为新分支的起点
    fork_from_cp = checkpoints[1]
    
    # 创建新的thread_id来创建一个分支
    fork_config = {"configurable": {"thread_id": "debug-session-fork-1"}}
    
    # 用修改后的状态从该点继续执行
    modified_state = dict(fork_from_cp.channel_values)
    modified_state["doubled"] = 100  # 人为修改：假设 doubled 应该是 100 而不是 10
    
    forked_result = app.invoke(modified_state, config=fork_config)
    
    print(f"分支结果: {forked_result['final_result']}")
    print(f"(原始结果是 {(5*2*3)**2} = 900)")
    print(f"(分支结果是 (100*3)**2 = {forked_result['final_result']})")
    for entry in forked_result["step_log"]:
        print(f"  {entry}")
```

注意这里的关键操作：我们用了一个**新的 thread_id** (`debug-session-fork-1`) 来创建分支。这确保了原始执行的 checkpoint 历史不会被覆盖——分支执行会产生自己独立的 checkpoint 链。如果你用同一个 thread_id，那就会直接覆盖原来的执行结果。

## 在 LangSmith 中可视化时间旅行

虽然上面的代码展示了如何通过 API 访问历史 checkpoint，但实际开发中更常用的方式是通过 LangSmith 的 Web 界面来进行可视化时间旅行。当你配置了 `LANGCHAIN_TRACING=true` 并设置了 LangSmith API key 后，每次图的执行都会被自动记录到 LangSmith 平台。

在 LangSmith 的 trace 视图中：

1. 你可以看到整个图的拓扑结构和执行顺序
2. 点击任意节点可以看到该节点的输入和输出
3. 每个节点对应一个 checkpoint——你可以看到那个时刻的完整状态
4. 对于包含循环的图，你可以展开每一次迭代查看状态变化
5. 如果有 Interrupt 节点，你可以清楚地看到暂停点和恢复点

这种可视化能力对于理解复杂图的执行行为极其有价值——特别是当你的图有条件路由、循环、子图嵌套等复杂结构时，纯代码层面的追踪会非常困难。

## 实际调试场景示例

让我们用一个更贴近实际的例子来展示时间旅行调试的完整工作流。

```python
from typing import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

class DataPipelineState(TypedDict):
    raw_data: str
    cleaned: str
    tokenized: list[str]
    entity_count: int
    sentiment_score: float
    category: str
    confidence: float
    pipeline_log: Annotated[list[str], operator.add]

def clean_data(state: DataPipelineState) -> dict:
    text = state["raw_data"].strip().lower()
    text = " ".join(text.split())
    return {
        "cleaned": text,
        "pipeline_log": [f"[清洗] 原始{len(state['raw_text']) if 'raw_text' in state else '?'}字符 → {len(text)}字符"]
    }

def tokenize(state: DataPipelineState) -> dict:
    words = state["cleaned"].split()
    return {
        "tokenized": words,
        "pipeline_log": [f"[分词] {len(words)} 个词"]
    }

def extract_entities(state: DataPipelineState) -> dict:
    tokens = state["tokenized"]
    entities = [t for t in tokens if t[0].isupper() and len(t) > 1]
    return {
        "entity_count": len(entities),
        "pipeline_log": [f"[实体提取] 发现 {len(entities)} 个实体: {entities[:5]}"]
    }

def analyze_sentiment(state: DataPipelineState) -> dict:
    text = state["cleaned"]
    pos_words = {"good", "great", "excellent", "love", "happy"}
    neg_words = {"bad", "terrible", "hate", "awful", "sad"}
    pos = sum(1 for w in pos_words if w in text)
    neg = sum(1 for w in neg_words if w in text)
    score = (pos - neg) / max(len(text.split()), 1) * 10
    score = max(-10, min(10, score))
    return {
        "sentiment_score": round(score, 2),
        "pipeline_log": [f"[情感分析] 得分: {score:.1f}"]
    }

def categorize(state: DataPipelineState) -> dict:
    score = state["sentiment_score"]
    ents = state["entity_count"]
    if score > 3:
        cat = "positive"
        conf = min(0.95, 0.7 + abs(score) * 0.03)
    elif score < -3:
        cat = "negative"
        conf = min(0.95, 0.7 + abs(score) * 0.03)
    else:
        cat = "neutral"
        conf = 0.6
    return {
        "category": cat,
        "confidence": round(conf, 2),
        "pipeline_log": [f"[分类] {cat} (置信度: {conf:.2f})"]
    }

pipe_graph = StateGraph(DataPipelineState)
pipe_graph.add_node("clean", clean_data)
pipe_graph.add_node("tokenize", tokenize)
pipe_graph.add_node("entities", extract_entities)
pipe_graph.add_node("sentiment", analyze_sentiment)
pipe_graph.add_node("categorize", categorize)

pipe_graph.add_edge(START, "clean")
pipe_graph.add_edge("clean", "tokenize")
pipe_graph.add_edge("tokenize", "entities")
pipe_graph.add_edge("entities", "sentiment")
pipe_graph.add_edge("sentiment", "categorize")
pipe_graph.add_edge("categorize", END)

checkpointer = MemorySaver()
app = pipe_graph.compile(checkpointer=checkpointer)
config = {"configurable": {"thread_id": "pipe-debug"}}

print("=== 执行数据管道 ===\n")
result = app.invoke({
    "raw_data": "  I LOVE this GREAT product! It is AWESOME and makes me HAPPY!  ",
    "cleaned": "", "tokenized": [], "entity_count": 0,
    "sentiment_score": 0.0, "category": "", "confidence": 0.0,
    "pipeline_log": []
}, config=config)

for entry in result["pipeline_log"]:
    print(entry)

print(f"\n最终分类: {result['category']} (置信度: {result['confidence']})")

# 模拟调试场景：用户报告分类结果不正确
print("\n=== 调试场景：为什么分类是 neutral？ ===")
cps = list(checkpointer.list(config))

# 查看 sentiment 步骤后的状态
for cp in cps:
    if cp.metadata.get("source") == "sentiment":
        print(f"\nsentiment 节点后的状态:")
        vals = cp.channel_values
        print(f"  sentiment_score: {vals.get('sentiment_score')}")
        print(f"  entity_count: {vals.get('entity_count')}")
        print(f"  当时的日志:")
        for log in vals.get("pipeline_log", []):
            print(f"    {log}")

# 发现问题：sentiment_score 的计算可能有问题
# 让我们从 sentiment 之后的状态分支，手动修正分数
if len(cps) >= 4:
    sentiment_cp = cps[3]
    debug_state = dict(sentiment_cp.channel_values)
    
    print(f"\n=== 发现问题: sentiment_score = {debug_state['sentiment_score']} ===")
    print("   原因分析: 正面词数量很多但得分却不高")
    print("   可能原因: 归一化分母过大稀释了分数")
    
    # 修正：使用不同的归一化策略
    debug_state["sentiment_score"] = 8.5  # 手动修正
    
    fork_config = {"configurable": {"thread_id": "pipe-debug-fix"}}
    fixed_result = app.invoke(debug_state, config=fork_config)
    
    print(f"\n修正后的分类: {fixed_result['category']} (置信度: {fixed_result['confidence']})")
```

这个调试场景模拟了一个真实的工作流：运行管道 → 发现最终结果不符合预期 → 使用时间旅行回溯到可疑的中间步骤 → 分析当时的完整状态数据 → 定位可能的问题原因 → 创建分支验证修复方案。

## 时间旅行的性能考量

时间旅行功能虽然强大，但也需要考虑其性能影响。

**存储开销**：每个 checkpoint 都是一份完整的状态快照。如果你的状态很大（比如包含大段文本或列表），或者图有很多节点（意味着有很多 checkpoint），存储开销会线性增长。建议：
- 只在需要持久化的场景启用 checkpointing（开发阶段可以用 MemorySaver）
- 定期清理过期的 checkpoint（TTL 机制）
- 大块数据用外部引用代替内联存储

**查询开销**：`checkpointer.list()` 需要返回所有历史 checkpoint。对于长时间运行的图（数百次迭代），这个列表可能会很长。大多数 checkpointer 实现支持分页查询或限制返回数量。

**分支开销**：每次创建分支（使用新的 thread_id）都会复制一份完整的状态历史。频繁创建大量分支会显著增加内存和存储的使用量。
