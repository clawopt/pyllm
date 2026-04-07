# 8.3 Map-Reduce 并行模式

> Supervisor 模式中的 Worker 是按顺序或条件路由依次调用的，但很多场景下多个 Worker 可以完全独立地并行工作。比如分析一份财报时，可以同时让财务分析师、风险评估师、行业对比专家各自从不同角度进行分析，然后把所有报告汇总成最终结论。这种"分发到多个 Worker 并行执行 → 收集所有结果统一处理"的模式就是 **Map-Reduce（映射-归约）**模式——Map 阶段把任务分发给 Worker，Reduce 阶段汇聚和整合结果。

## Map-Reduce 的基本结构

Map-Reduce 模式在 LangGraph 中的拓扑结构非常清晰：

```
                    [Supervisor/Dispatcher]
                           │
              ┌──────────┼──────────┐
              ↓          ↓          ↓
        [Worker A]   [Worker B]   [Worker C]
        (财务分析)   (风险分析)   (行业对比)
              │          │          │
              └──────────┼──────────┘
                           ↓
                   [Reducer/Aggregator]
                           ↓
                      最终输出
```

关键特征是：
1. **Dispatcher 节点**：准备任务数据并启动并行分支
2. **Worker 节点**：彼此独立、无依赖关系、可同时执行
3. **Reducer 节点**：收集所有 Worker 的输出并合并为最终结果

## 完整实现：多角度文档分析系统

让我们用一个完整的多角度文档分析系统来展示 Map-Reduce 的实现：

```python
from typing import TypedDict, Annotated, Literal
import operator
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class DocumentAnalysisState(TypedDict):
    document_content: str
    analysis_type: str
    
    financial_analysis: str
    risk_analysis: str
    compliance_analysis: str
    technical_analysis: str
    
    aggregated_report: str
    final_verdict: str
    pipeline_log: Annotated[list[str], operator.add]

# === Map 阶段的 Worker Agents ===

def financial_analyst(state: DocumentAnalysisState) -> dict:
    response = llm.invoke([
        SystemMessage(content="你是一个资深的财务分析师。请从财务角度分析以下文档：关注收入结构、成本构成、盈利能力、现金流状况等。"),
        HumanMessage(content=state["document_content"])
    ])
    return {
        "financial_analysis": response.content,
        "pipeline_log": ["[财务分析师] 分析完成"]
    }

def risk_analyst(state: DocumentAnalysisState) -> dict:
    response = llm.invoke([
        SystemMessage(content="你是一个专业的风险评估师。请识别文档中提到的各类风险：市场风险、运营风险、合规风险、技术风险等，并评估其严重程度和发生概率。"),
        HumanMessage(content=state["document_content"])
    ])
    return {
        "risk_analysis": response.content,
        "pipeline_log": ["[风险评估师] 分析完成"]
    }

def compliance_analyst(state: DocumentAnalysisState) -> dict:
    response = llm.invoke([
        SystemMessage(content="你是一个合规审查专家。请检查文档是否符合相关法律法规要求、行业标准、内部政策等。列出发现的问题和建议的改进措施。"),
        HumanMessage(content=state["document_content"])
    ])
    return {
        "compliance_analysis": response.content,
        "pipeline_log": ["[合规审查员] 分析完成"]
    }

def technical_analyst(state: DocumentAnalysisState) -> dict:
    response = llm.invoke([
        SystemMessage(content="你是一个技术评审专家。请从技术可行性、架构合理性、性能指标、安全漏洞等方面评估文档中涉及的技术方案。"),
        HumanMessage(content=state["document_content"])
    ])
    return {
        "technical_analysis": response.content,
        "pipeline_log": ["[技术评审员] 分析完成"]
    }

# === Reduce 阶段：聚合器 ===

def aggregate_analyses(state: DocumentAnalysisState) -> dict:
    analyses = {
        "financial": state.get("financial_analysis", ""),
        "risk": state.get("risk_analysis", ""),
        "compliance": state.get("compliance_analysis", ""),
        "technical": state.get("technical_analysis", "")
    }
    
    summary_parts = []
    for name, content in analyses.items():
        if content:
            lines = content.split('\n')
            summary = '\n'.join(lines[:5]) + ('\n...' if len(lines) > 5 else '')
            summary_parts.append(f"=== {name.upper()} 分析摘要 ===\n{summary}")
    
    combined_summary = "\n\n".join(summary_parts)
    
    return {
        "aggregated_report": combined_summary,
        "pipeline_log": [
            f"[聚合器] 收集了 {sum(1 for v in analyses.values() if v)} 份分析报告",
            f"[聚合器] 各报告长度: {', '.join(f'{k}={len(v)}字' for k, v in analyses.items() if v)}"
        ]
    }

def final_judge(state: DocumentAnalysisState) -> dict:
    report = state["aggregated_report"]
    
    verdict_response = llm.invoke([
        SystemMessage(content="你是一个高级决策顾问。基于四份专业分析报告，给出最终的综合性判断和建议。包括：总体评价、主要发现、核心建议、优先级排序。"),
        HumanMessage(content=f"综合分析报告:\n{report}")
    ])
    
    return {
        "final_verdict": verdict_response.content,
        "pipeline_log": ["[终审官] 综合判断完成"]
    }

# === 构建图 ===
map_reduce_graph = StateGraph(DocumentAnalysisState)
map_reduce_graph.add_node("dispatch", lambda s: {"pipeline_log": ["[分发器] 开始并行分析"]})
map_reduce_graph.add_node("financial", financial_analyst)
map_reduce_graph.add_node("risk", risk_analyst)
map_reduce_graph.add_node("compliance", compliance_analyst)
map_reduce_graph.add_node("technical", technical_analyst)
map_reduce_graph.add_node("aggregate", aggregate_analyses)
map_reduce_graph.add_node("final_judge", final_judge)

map_reduce_graph.add_edge(START, "dispatch")
map_reduce_graph.add_edge("dispatch", "financial")     # 扇出
map_reduce_graph.add_edge("dispatch", "risk")         # 扇出
map_reduce_graph.add_edge("dispatch", "compliance")    # 扇出
map_reduce_graph.add_edge("dispatch", "technical")    # 扇出
map_reduce_graph.add_edge("financial", "aggregate")  # 扇入
map_reduce_graph.add_edge("risk", "aggregate")
map_reduce_graph.add_edge("compliance", "aggregate")
map_reduce_graph.add_edge("technical", "aggregate")
map_reduce_graph.add_edge("aggregate", "final_judge")
map_reduce_graph.add_edge("final_judge", END)

app = map_reduce_graph.compile()

sample_doc = """
2024年度财务报告

一、收入情况
本年度公司实现总收入 12.5 亿元，同比增长 23%。
其中主营业务收入 10.2 亿元，新业务收入 2.3 亿元。

二、成本结构
总成本 8.7 亿元，毛利率 30.4%。
研发投入 1.8 亿元，占收入比例 14.4%。

三、现金流
经营活动净现金流 3.2 亿元，投资活动净流出 1.5 亿元。
"""

print("=" * 60)
print("Map-Reduce 多角度文档分析")
print("=" * 60)

result = app.invoke({
    "document_content": sample_doc,
    "analysis_type": "annual_report",
    "financial_analysis": "", "risk_analysis": "",
    "compliance_analysis": "", "technical_analysis": "",
    "aggregated_report": "", "final_verdict": "",
    "pipeline_log": []
})

for entry in result["pipeline_log"]:
    print(entry)

print(f"\n{'='*60}")
print(f"综合判断预览:\n{result['final_verdict'][:400]}...")
```

这个 Map-Reduce 系统展示了完整的并行分析流程：
1. **Dispatch** 节点准备文档内容
2. 四个 **Worker** 节点（财务/风险/合规/技术）同时接收文档并独立分析
3. **Aggregate** 节点收集四份报告并生成摘要
4. **Final Judge** 节点基于汇总信息做出最终判断

注意图拓扑的关键特征：`dispatch` 通过四条边同时扇出到四个 Worker，四个 Worker 的输出全部汇聚到 `aggregate` 节点。这就是经典的 fan-out / fan-in 结构。

## 动态 Worker 数量的 Map-Reduce

有些场景下参与分析的 Worker 数量不是固定的，而是根据输入动态决定的。比如用户可以选择启用哪些分析维度。

```python
class DynamicMapReduceState(TypedDict):
    document: str
    selected_analyses: list[str]
    worker_results: dict[str, str]
    final_output: str
    log: list[str]

def dynamic_dispatch(state: DynamicMapReduceState) -> dict:
    selected = state.get("selected_analyses", ["financial", "risk"])
    return {
        "log": [f"[分发器] 启用 {len(selected)} 个分析维度: {selected}"]
    }

def dynamic_aggregate(state: DynamicMapReduceState) -> dict:
    results = state.get("worker_results", {})
    summary = "\n\n".join(
        f"=== {name} ===\n{content[:200]}..."
        for name, content in results.items()
    )
    return {
        "final_output": summary,
        "log": [f"[聚合器] 汇总 {len(results)} 个维度的分析结果"]
    }

AVAILABLE_WORKERS = {
    "financial": financial_analyst,
    "risk": risk_analyst,
    "compliance": compliance_analyst,
    "technical": technical_analyst,
    "market": lambda s: {"market_analysis": "市场分析完成"},
    "legal": lambda s: {"legal_analysis": "法律分析完成"},
}

dynamic_mr_graph = StateGraph(DynamicMapReduceState)
dynamic_mr_graph.add_node("dispatch", dynamic_dispatch)
dynamic_mr_graph.add_node("aggregate", dynamic_aggregate)
dynamic_mr_graph.add_node("finalize", lambda s: {"log": ["[完成"]})

dynamic_mr_graph.add_edge(START, "dispatch")

for worker_name in AVAILABLE_WORKERS:
    dynamic_mr_graph.add_node(worker_name, AVAILABLE_WORKERS[worker_name])
    dynamic_mr_graph.add_edge("dispatch", worker_name)
    dynamic_mr_graph.add_edge(worker_name, "aggregate")

dynamic_mr_graph.add_edge("aggregate", "finalize")
dynamic_mr_graph.add_edge("finalize", END)

app = dynamic_mr_graph.compile()

result = app.invoke({
    "document": sample_doc,
    "selected_analyses": ["financial", "risk", "technical"],
    "worker_results": {},
    "final_output": "",
    "log": []
})

for entry in result["log"]:
    print(entry)
```

这个动态版本的 Map-Reduce 通过 `selected_analyses` 列表来决定调用哪些 Worker。图中预先注册了所有可能的 Worker（通过遍历 `AVAILABLE_WORKERS` 字典），但只有被选中的 Worker 会被实际执行——未选中的 Worker 虽然在图中存在，但由于它们的状态字段不会被后续节点依赖，所以不影响最终结果。

## Map-Reduce 与 Checkpointing 的配合

当使用 Map-Reduce 模式时，checkpointing 的行为值得特别注意。由于多个 Worker 是从同一个 Dispatcher 节点扇出的，LangGraph 在默认的单线程模式下会**串行**执行这些 Worker（一个接一个），每个 Worker 完成后都会创建一个 checkpoint。这意味着：

- 如果有 N 个 Worker，会有 N+1 个 checkpoint（含 Dispatcher 后的一个）
- 如果某个 Worker 失败，前面已完成的 Worker 的结果已经保存在 checkpoint 中了
- 恢复执行时只需要重试失败的 Worker，不需要重新执行所有 Worker

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
app_with_cp = map_reduce_graph.compile(checkpointer=checkpointer)
config = {"configurable": {"thread_id": "mr-session"}}

try:
    result = app_with_cp.invoke({...}, config=config)
except Exception as e:
    print(f"某Worker失败: {e}")
    
    # 恢复时只需继续未完成的 Worker
    recovered = app_with_cp.invoke({...}, config=config)
```

如果未来需要真正的并行执行（让多个 Worker 同时运行以减少总延迟），可以使用 LangGraph 的 Send API 配合异步执行框架来实现，图的拓扑结构不需要任何改变。
