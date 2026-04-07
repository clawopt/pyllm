# 8.4 Hand-off 接力模式与状态传递

> Supervisor 模式有一个中央协调者，Map-Reduce 模式有分发器和聚合器，但还有一种更轻量的多Agent协作方式——**Hand-off（接力）模式**。在 Hand-off 模式中，Agent 之间没有中央管理者，它们像接力赛一样按预定的顺序依次处理任务，每个 Agent 完成自己的部分后把结果"交接"给下一个 Agent。这种模式特别适合那些有明确流水线特征的任务——比如需求分析 → 架构设计 → 代码实现 → 测试验证这样的软件开发生命周期。

## Hand-off 的基本结构

Hand-off 的拓扑结构是纯粹的线性链：

```
用户请求
    ↓
[Agent 1: 需求分析师] → 输出1
                        ↓ (hand-off)
              [Agent 2: 架构师] → 输出2
                                ↓ (hand-off)
                          [Agent 3: 开发者] → 输出3
                                            ↓ (hand-off)
                                      [Agent 4: 测试员] → 最终输出
```

每个 Agent 只关心两件事：接收上一个 Agent 的产出、完成自己的工作后传给下一个 Agent。它不需要知道全局情况，也不需要和其他 Agent 并行通信。

## 完整实现：软件开发流水线

让我们用一个完整的软件开发流水线来展示 Hand-off 模式的实现：

```python
from typing import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class DevPipelineState(TypedDict):
    user_request: str
    
    requirements_doc: str
    architecture_design: str
    implementation_code: str
    test_report: str
    delivery_package: str
    
    current_phase: str
    pipeline_log: Annotated[list[str], operator.add]

# === Phase 1: 需求分析 ===
def requirements_analyst(state: DevPipelineState) -> dict:
    response = llm.invoke([
        SystemMessage(content="你是一个资深的需求分析师。请将用户的模糊需求转化为清晰的结构化需求文档。包含功能需求、非功能需求、约束条件、验收标准等。"),
        HumanMessage(content=state["user_request"])
    ])
    
    return {
        "requirements_doc": response.content,
        "current_phase": "requirements",
        "pipeline_log": [
            "━━━ 阶段1: 需求分析 ━━━━",
            f"[需求分析师] 已生成需求文档 ({len(response.content)} 字符)"
        ]
    }

# === Phase 2: 架构设计 ===
def architect(state: DevPipelineState) -> dict:
    req = state["requirements_doc"]
    response = llm.invoke([
        SystemMessage(content="你是一个系统架构师。基于需求文档，设计技术架构方案。包括系统分层、模块划分、技术选型、接口定义、数据模型等。使用 Markdown 格式输出。"),
        HumanMessage(content=f"需求文档:\n{req}")
    ])
    
    return {
        "architecture_design": response.content,
        "current_phase": "architecture",
        "pipeline_log": [
            "━━━ 阶段2: 架构设计 ━━━━",
            f"[架构师] 已完成技术方案设计 ({len(response.content)} 字符)"
        ]
    }

# === Phase 3: 代码实现 ===
def developer(state: DevPipelineState) -> dict:
    arch = state["architecture_design"]
    response = llm.invoke([
        SystemMessage(content="你是一个高级开发工程师。根据架构设计方案，编写高质量的实现代码。代码要有清晰的注释、错误处理和类型标注。只输出核心代码，不需要完整的文件结构。"),
        HumanMessage(content=f"架构方案:\n{arch}")
    ])
    
    return {
        "implementation_code": response.content,
        "current_phase": "implementation",
        "pipeline_log": [
            "━━━ 阶段3: 代码实现 ━━━━",
            f"[开发者] 已实现核心代码 ({len(response.content)} 字符)"
        ]
    }

# === Phase 4: 测试验证 ===
def tester(state: DevPipelineState) -> dict:
    code = state["implementation_code"]
    response = llm.invoke([
        SystemMessage(content="你是一个QA测试工程师。对以下代码进行审查测试。检查潜在bug、边界条件、异常处理、性能问题等。给出测试用例和问题清单。"),
        HumanMessage(content=f"待测代码:\n{code}")
    ])
    
    return {
        "test_report": response.content,
        "current_phase": "testing",
        "pipeline_log": [
            "━━━ 阶段4: 测试验证 ━━━━",
            f"[测试师] 已完成测试报告 ({len(response.content)} 字符)"
        ]
    }

# === Phase 5: 最终交付 ===
def delivery_manager(state: DevPipelineState) -> dict:
    all_phases = {
        "requirements": state.get("requirements_doc", ""),
        "architecture": state.get("architecture_design", ""),
        "implementation": state.get("implementation_code", ""),
        "testing": state.get("test_report", "")
    }
    
    response = llm.invoke([
        SystemMessage(content="你是一个项目经理/交付经理。整合所有阶段的产出物，生成一份完整的项目交付包摘要。包括各阶段成果概述、质量评估、风险提示和后续建议。"),
        HumanMessage(content=(
            f"原始需求: {state['user_request']}\n\n"
            f"需求文档: {all_phases['requirements'][:100]}...\n\n"
            f"架构设计: {all_phases['architecture'][:100]}...\n\n"
            f"实现代码: {all_phases['implementation'][:100]}...\n\n"
            f"测试报告: {all_phases['testing'][:100]}..."
        ))
    ])
    
    return {
        "delivery_package": response.content,
        "current_phase": "delivered",
        "pipeline_log": [
            "━━━ 阶段5: 项目交付 ━━━━",
            "[交付经理] 已打包所有阶段产出"
        ]
    }

# === 构建图 ===
pipeline_graph = StateGraph(DevPipelineState)
pipeline_graph.add_node("phase1_requirements", requirements_analyst)
pipeline_graph.add_node("phase2_architecture", architect)
pipeline_graph.add_node("phase3_implementation", developer)
pipeline_graph.add_node("phase4_testing", tester)
pipeline_graph.add_node("phase5_delivery", delivery_manager)

pipeline_graph.add_edge(START, "phase1_requirements")
pipeline_graph.add_edge("phase1_requirements", "phase2_architecture")   # hand-off 1→2
pipeline_graph.add_edge("phase2_architecture", "phase3_implementation") # hand-off 2→3
pipeline_graph.add_edge("phase3_implementation", "phase4_testing")     # hand-off 3→4
pipeline_graph.add_edge("phase4_testing", "phase5_delivery")         # hand-off 4→5
pipeline_graph.add_edge("phase5_delivery", END)

app = pipeline_graph.compile()

print("=" * 60)
print("Hand-off 软件开发流水线")
print("=" * 60)

result = app.invoke({
    "user_request": "开发一个支持多用户的在线协作文档编辑器",
    "requirements_doc": "",
    "architecture_design": "",
    "implementation_code": "",
    "test_report": "",
    "delivery_package": "",
    "current_phase": "",
    "pipeline_log": []
})

for entry in result["pipeline_log"]:
    print(entry)

print(f"\n{'='*60}")
print(f"最终交付包预览:\n{result['delivery_package'][:400]}...")
```

这个 Hand-off 流水线展示了五个 Agent 的接力过程：
1. **需求分析师**：把模糊的用户需求转化为结构化文档
2. **架构师**：基于需求文档设计技术方案
3. **开发者**：基于架构方案编写代码
4. **测试师**：审查代码并生成测试报告
5. **交付经理**：整合所有产出物形成交付包

注意每两个节点之间都是简单的 `add_edge`（无条件边），这就是 Hand-off 模式的拓扑特征——**顺序固定、无分支、无循环**。

## 带 Loop-back 的 Hand-off

纯线性的 Hand-off 有时不够灵活——如果测试发现了严重问题，可能需要回到开发者那里修复 bug 然后再重新测试。这时可以引入一个**带回退的 Hand-off** 变体：

```python
def test_reviewer(state: DevPipelineState) -> dict:
    code = state["implementation_code"]
    response = llm.invoke([
        SystemMessage(content="审查代码并判断是否通过。如果发现严重问题返回 'needs_fix'，否则返回 'passed'。"),
        HumanMessage(content=code)
    ])
    
    verdict = response.content.lower()
    if "needs_fix" in verdict or "不通过" in verdict or "failed" in verdict:
        return {
            "test_report": response.content,
            "current_phase": "testing_failed",
            "pipeline_log": ["[测试师] ❌ 未通过，需要回退修复"]
        }
    
    return {
        "test_report": response.content,
        "current_phase": "testing_passed",
        "pipeline_log": ["[测试师] ✅ 通过测试"]
    }

def route_after_test(state: DevPipelineState) -> str:
    phase = state.get("current_phase", "")
    if phase == "testing_passed":
        return "delivery"
    if phase == "testing_failed":
        return "fix_loop"
    return "delivery"

def fix_and_resubmit(state: DevPipelineState) -> dict:
    code = state["implementation_code"]
    test_feedback = state["test_report"]
    
    response = llm.invoke([
        SystemMessage(content="你是开发者。根据测试反馈修复代码中的问题。只修改有问题的部分，保持其他部分不变。"),
        HumanMessage(content=f"当前代码:\n{code}\n\n测试反馈:\n{test_feedback}")
    ])
    
    return {
        "implementation_code": response.content,
        "current_phase": "fix_done",
        "pipeline_log": [
            "━━━ 修复循环 ━━━━",
            "[开发者] 已根据测试反馈修复代码"
        ]
    }

def route_after_fix(state: DevPipelineState) -> str:
    if state.get("current_phase") == "fix_done":
        return "re_test"
    return "delivery"

handoff_with_loop = StateGraph(DevPipelineState)
handoff_with_loop.add_node("p1", requirements_analyst)
handoff_with_loop.add_node("p2", architect)
handoff_with_loop.add_node("p3", developer)
handoff_with_loop.add_node("p4", test_reviewer)
handoff_with_loop.add_node("p4_fix", fix_and_resubmit)
handoff_with_loop.add_node("p5", delivery_manager)

handoff_with_loop.add_edge(START, "p1")
handoff_with_loop.add_edge("p1", "p2")
handoff_with_loop.add_edge("p2", "p3")
handoff_with_loop.add_edge("p3", "p4")
handoff_with_loop.add_conditional_edges("p4", route_after_test, {
    "delivery": "p5",
    "testing_failed": "p4_fix"
})
handoff_with_loop.add_conditional_edges("p4_fix", route_after_fix, {
    "re_test": "p4",
    "delivery": "p5"
})
handoff_with_loop.add_edge("p5", END)

app_loop = handoff_with_loop.compile()

# 设置最大循环次数防止无限循环
```

这个变体在测试失败时不是直接结束或跳到交付，而是回到开发者那里进行修复，然后再重新测试。这增加了一个小的循环结构（`p4 → p4_fix → p4`），让 Hand-off 流水线有了基本的自愈能力。

## Hand-off vs Supervisor vs Map-Reduce：如何选择

三种多Agent模式各有其最佳适用场景：

| 维度 | Hand-off | Supervisor | Map-Reduce |
|------|----------|-----------|-----------|
| **拓扑结构** | 线性链式 | 树状（中心辐射） | 扇入-扇出 |
| **执行顺序** | 固定顺序 | 动态路由 | 并行执行 |
| **Agent 数量** | 通常 3-7 个 | 可动态增减 | 通常 3-10 个 |
| **Agent 依赖** | 强依赖（串行） | 弱/中依赖 | 无依赖 |
| **总延迟** | 所有步骤延迟之和 | 步骤延迟 + 路由决策 | 最慢 Worker 延迟 |
| **复杂度** | 低 | 中高 | 中 |
| **适用场景** | 流水线、审批链 | 复杂任务编排 | 多角度分析 |

选择建议：
- 任务有**明确的先后顺序**且每步依赖上一步的输出 → **Hand-off**
- 需要**灵活决策**和动态调整执行路径 → **Supervisor**
- 需要**多角度并行分析**且各角度独立 → **Map-Reduce**
- 以上特点都有 → **组合使用**（如 Hand-off + Map-Reduce）
