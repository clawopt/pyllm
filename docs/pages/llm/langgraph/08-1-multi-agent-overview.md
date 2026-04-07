# 8.1 多Agent架构概览：从单Agent到Agent团队

> 前面所有章节讨论的都是**单个图/单个 Agent**的工作方式——无论这个图有多复杂、有多少节点、是否包含循环和子图，本质上都是一个执行单元在独立工作。但很多真实世界的问题超出了单个 Agent 的能力范围：一个 Agent 可能擅长写代码但不擅长做研究，另一个 Agent 擅长分析数据但不懂业务逻辑，第三个 Agent 能很好地与用户沟通但缺乏技术深度。如果我们能让这些各有所长的 Agent 协同工作，就能构建出远比任何单一 Agent 都强大的系统。这就是 LangGraph **多Agent协作（Multi-Agent Collaboration）**要解决的核心问题。

## 为什么需要多Agent

在深入具体模式之前，先理解为什么单一 Agent 有其固有的局限性。

**能力边界**：每个 Agent 都是基于特定的 prompt、工具集和状态设计来构建的，这决定了它的能力边界。一个代码审查 Agent 可能对 Python 语法了如指掌，但面对一个关于市场营销的问题就束手无策。试图让一个 Agent 做所有事情的结果往往是——它什么都懂一点，但什么都不精。

**上下文窗口限制**：LLM 的上下文窗口是有限的（即使是 GPT-4 也只有 128K tokens）。当任务变得复杂时，单个 Agent 需要在上下文中塞入越来越多的信息——系统提示词、历史对话、工具定义、中间结果——很快就会触及上限。

**关注点分散**：单一 Agent 需要同时关注多个维度：理解用户意图、选择合适的工具、管理对话状态、保证输出质量。这种多任务并行处理会显著降低每个维度的表现质量。

**可维护性**：一个"全能" Agent 的代码会变得越来越复杂、越来越难调试。修改一处可能影响多处功能，新增功能可能破坏已有行为。

多Agent 架构通过**分工合作**来解决这些问题——每个 Agent 专注于自己擅长的领域，通过明确的通信协议协同完成复杂任务。这就像一支专业团队：每个人都是某个领域的专家，通过良好的配合完成个人无法独自完成的复杂项目。

## LangGraph 中实现多Agent的三种核心模式

LangGraph 支持多种多Agent 编排模式，其中三种最常用的是：

### 模式一：Supervisor（主管）模式

一个中央的 "Supervisor" Agent 负责接收用户请求、将任务分解为子任务、分发给各个 Worker Agent、收集结果并整合为最终回复。Worker Agents 各自专注于特定领域，不需要知道全局情况。

```
用户请求
    ↓
[Supervisor Agent] ← 分析请求，决定分发策略
    ↓ ↓ ↓
[Worker A] [Worker B] [Worker C]
(代码)   (搜索)   (写作)
    ↓ ↓ ↓
[Supervisor Agent] ← 收集结果，整合回复
    ↓
最终回答
```

### 模式二：Map-Reduce（并行分发汇聚）模式

Supervisor 把同一个任务同时分发给多个 Worker 并行处理（Map），然后收集所有 Worker 的结果进行汇总（Reduce）。适合需要多角度分析或并行处理的场景。

```
用户请求: "分析这份财报"
    ↓
[Supervisor] → 同时发送给:
    ├──→ [财务分析师Agent] → 利润分析报告
    ├──→ [风险评估Agent]   → 风险评估报告
    └──→ [行业对比Agent]   → 竞品对比报告
    ↓
[Reducer] ← 汇总三份报告 → 综合分析结论
```

### 模式三：Hand-off（接力）模式

Agent 之间按照预定的顺序传递任务，像接力赛一样。第一个 Agent 完成自己的部分后把结果传给下一个 Agent，直到最后一个 Agent 完成最终输出。

```
用户请求
    ↓
[Agent 1: 需求分析师] → 明确需求文档
    ↓ (hand-off)
[Agent 2: 技术架构师] → 技术方案设计
    ↓ (hand-off)
[Agent 3: 开发工程师] → 代码实现
    ↓ (hand-off)
[Agent 4: 测试工程师] → 测试报告
    ↓
最终交付物
```

## 从概念到代码：最简单的多Agent示例

让我们用最简单的 Supervisor 模式来建立一个直观的理解：

```python
from typing import TypedDict, Annotated, Literal
import operator
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class TeamState(TypedDict):
    user_query: str
    researcher_analysis: str
    writer_draft: str
    reviewer_feedback: str
    final_output: str
    team_log: Annotated[list[str], operator.add]

# === Worker Agent 1: 研究员 ===
def researcher(state: TeamState) -> dict:
    response = llm.invoke([
        SystemMessage(content="你是一个专业的研究员。请对用户的查询进行深入研究，提供详细的事实性分析和数据支持。"),
        HumanMessage(content=state["user_query"])
    ])
    return {
        "researcher_analysis": response.content,
        "team_log": ["[研究员] 分析完成"]
    }

# === Worker Agent 2: 写手 ===
def writer(state: TeamState) -> dict:
    research = state["researcher_analysis"]
    response = llm.invoke([
        SystemMessage(content="你是一个专业的技术写手。基于研究员的分析结果，撰写一篇清晰、结构化的文章。"),
        HumanMessage(content=f"研究分析:\n{research}\n\n请基于以上内容撰写文章:")
    ])
    return {
        "writer_draft": response.content,
        "team_log": ["[写手] 初稿完成"]
    }

# === Worker Agent 3: 审核员 ===
def reviewer(state: TeamState) -> dict:
    draft = state["writer_draft"]
    response = llm.invoke([
        SystemMessage(content="你是一个严格的审核员。审阅以下文章草稿，指出问题和改进建议。"),
        HumanMessage(content=f"文章草稿:\n{draft}")
    ])
    return {
        "reviewer_feedback": response.content,
        "team_log": ["[审核员] 审核完成"]
    }

# === Supervisor: 编排协调 ===
def supervisor_route(state: TeamState) -> Literal["researcher", "writer", "reviewer", "finalize"]:
    if not state.get("researcher_analysis"):
        return "researcher"
    if not state.get("writer_draft"):
        return "writer"
    if not state.get("reviewer_feedback"):
        return "reviewer"
    return "finalize"

def finalize(state: TeamState) -> dict:
    feedback = state["reviewer_feedback"]
    final_response = llm.invoke([
        SystemMessage(content="你是一个最终编辑。基于原始查询、初稿和审核反馈，生成最终的优化版本。"),
        HumanMessage(content=(
            f"原始查询: {state['user_query']}\n\n"
            f"初稿摘要: {state['writer_draft'][:200]}...\n\n"
            f"审核反馈: {feedback}\n\n"
            f"请输出最终版本:"
        ))
    ])
    return {
        "final_output": final_response.content,
        "team_log": ["[主管] 最终版本已生成"]
    }

team_graph = StateGraph(TeamState)
team_graph.add_node("researcher", researcher)
team_graph.add_node("writer", writer)
team_graph.add_node("reviewer", reviewer)
team_graph.add_node("finalize", finalize)

team_graph.add_edge(START, "researcher")
team_graph.add_conditional_edges("researcher", supervisor_route, {
    "researcher": "researcher",
    "writer": "writer",
    "reviewer": "reviewer",
    "finalize": "finalize"
})
team_graph.add_conditional_edges("writer", supervisor_route, {
    "researcher": "researcher",
    "writer": "writer",
    "reviewer": "reviewer",
    "finalize": "finalize"
})
team_graph.add_conditional_edges("reviewer", supervisor_route, {
    "researcher": "researcher",
    "writer": "writer",
    "reviewer": "reviewer",
    "finalize": "finalize"
})
team_graph.add_edge("finalize", END)

app = team_graph.compile()

print("=" * 60)
print("多Agent 团队协作演示")
print("=" * 60)

result = app.invoke({
    "user_query": "Python 和 Go 在并发编程方面的差异是什么？",
    "researcher_analysis": "",
    "writer_draft": "",
    "reviewer_feedback": "",
    "final_output": "",
    "team_log": []
})

for entry in result["team_log"]:
    print(entry)

print(f"\n{'='*60}")
print(f"最终输出预览:\n{result['final_output'][:300]}...")
```

这段程序描述了一个三 Agent 团队的完整协作过程。`supervisor_route` 函数扮演着调度员的角色——它检查当前状态中哪些 Worker 的产出还缺失，然后把控制权路由给对应的 Worker。三个 Worker（研究员、写手、审核员）各自用 LLM 完成自己的专业任务，最后由 `finalize` 节点综合所有信息生成最终输出。

注意这里的关键设计：
1. **每个 Worker 是一个独立的节点函数**，有自己的 system prompt 定义角色
2. **Supervisor 通过条件边来编排流程**，根据当前状态决定下一步该谁上场
3. **共享的 `TeamState` 让所有 Agent 能访问彼此的产出**
4. **`team_log` 字段记录完整的协作过程**，便于追踪和调试

## 多Agent vs 单Agent：何时选择

不是所有场景都需要多Agent。以下是一些决策参考：

**使用单Agent的场景**：
- 任务简单明确，一个 LLM 调用就能解决
- 响应延迟要求极低（多Agent会增加总延迟）
- 成本敏感（每次 Agent 切换都涉及额外的 LLM 调用）
- 处于原型验证阶段（先跑通再优化）

**使用多Agent的场景**：
- 任务复杂，涉及多个不同领域的专业知识
- 需要多角度分析或并行处理
- 对输出质量有极高要求（如生产级内容生成）
- 不同步骤可能需要不同的工具集或权限配置
- 需要清晰的职责划分以便团队分工开发

作为一条经验法则：**如果你的 Agent 需要"戴着三顶以上的帽子"（同时承担研究者、写手、审核者等角色），那就应该考虑拆分为多Agent**。
