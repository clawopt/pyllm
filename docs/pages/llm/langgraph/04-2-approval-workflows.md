# 4.2 审批流与多级人工审核

> 在上一节中我们了解了 Interrupt 的基本用法——在一个节点中暂停执行，等待人类输入后恢复。但实际业务中的审批流程往往比这复杂得多：可能需要多级审批（主管→经理→总监）、可能需要并行审批（多个部门同时审核）、可能需要条件性审批（金额小于一定阈值只需要一级审批，超过则需要多级）、还可能在任意一级被驳回并退回修改。这一节我们会用 LangGraph 的 Interrupt 机制来构建这些复杂的审批流模式。

## 单级审批模式

先从最简单的单级审批开始，然后逐步增加复杂度。单级审批的流程是：提交申请 → AI 初审 → 人工审批 → 批准/拒绝。虽然简单，但它包含了所有审批流程的核心要素。

```python
from typing import TypedDict, Annotated
import operator
from datetime import datetime
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

class ApprovalState(TypedDict):
    request_id: str
    requester: str
    request_type: str
    amount: float
    description: str
    ai_assessment: str
    risk_level: str
    approver_name: str
    approval_decision: str
    approval_comment: str
    final_status: str
    audit_trail: Annotated[list[str], operator.add]

def validate_request(state: ApprovalState) -> dict:
    req_id = state["request_id"]
    requester = state["requester"]
    amount = state["amount"]
    req_type = state["request_type"]

    errors = []
    if not req_id or not req_id.startswith("REQ-"):
        errors.append("请求ID格式无效")
    if not requester:
        errors.append("请求人不能为空")
    if amount <= 0:
        errors.append("金额必须大于0")

    if errors:
        return {
            "ai_assessment": f"校验失败: {'; '.join(errors)}",
            "risk_level": "invalid",
            "audit_trail": [f"[校验] ❌ {'; '.join(errors)}"]
        }

    return {
        "ai_assessment": "基本信息校验通过",
        "risk_level": "pending",
        "audit_trail": [
            f"[校验] ✅ 请求ID: {req_id}, 请求人: {requester}, 类型: {req_type}, 金额: ¥{amount:.2f}"
        ]
    }

def route_after_validation(state: ApprovalState) -> str:
    if state["risk_level"] == "invalid":
        return "auto_reject"
    return "ai_risk_check"

def ai_risk_assess(state: ApprovalState) -> dict:
    amount = state["amount"]
    req_type = state["request_type"]

    risk_factors = []
    if amount > 50000:
        risk_factors.append(f"大额申请 (¥{amount:.0f})")
    if req_type in ["entertainment", "travel"]:
        risk_factors.append("敏感费用类别")
    if amount > 100000:
        risk_factors.append("超大额需特别关注")

    if len(risk_factors) >= 2:
        level = "high"
    elif len(risk_factors) == 1:
        level = "medium"
    else:
        level = "low"

    assessment = (f"AI风险评估结果: {level}风险"
                  + (f" | 风险因素: {', '.join(risk_factors)}" if risk_factors else "")
                  + f" | 建议审批路径: {'多级' if level != 'low' else '单级'}")

    return {
        "ai_assessment": assessment,
        "risk_level": level,
        "audit_trail": [f"[风控] {assessment}"]
    }

def human_approve(state: ApprovalState) -> dict:
    prompt_data = {
        "title": "待审批申请",
        "summary": (
            f"请求ID: {state['request_id']}\n"
            f"申请人: {state['requester']}\n"
            f"类型: {state['request_type']}\n"
            f"金额: ¥{state['amount']:.2f}\n"
            f"描述: {state['description']}\n"
            f"\nAI评估: {state['ai_assessment']}"
        ),
        "options": ["approve", "reject"],
        "instruction": "输入 'approve' 批准 或 'reject' 拒绝，可附加评论"
    }

    user_input = interrupt(prompt_data)

    if not user_input or not user_input.strip():
        return {
            "approval_decision": "pending",
            "approval_comment": "",
            "audit_trail": ["[审批] 等待审批人操作"]
        }

    parts = user_input.strip().split(maxsplit=1)
    decision = parts[0].lower()
    comment = parts[1] if len(parts) > 1 else ""

    if decision == "approve":
        status_msg = "✅ 已批准"
    elif decision == "reject":
        status_msg = "❌ 已拒绝"
    else:
        return {
            "approval_decision": "pending",
            "approval_comment": f"无效指令: {decision}",
            "audit_trail": [f"[审批] ⚠️ 无效指令: {decision}"]
        }

    return {
        "approver_name": "当前审批人",
        "approval_decision": decision,
        "approval_comment": comment,
        "final_status": "approved" if decision == "approve" else "rejected",
        "audit_trail": [f"[审批] {status_msg} | 评论: {comment or '无'}"]
    }

def auto_reject(state: ApprovalState) -> dict:
    return {
        "final_status": "rejected",
        "audit_trail": ["[自动拒绝] 申请信息不完整或无效"]
    }
```

现在把节点组装成图：

```python
graph = StateGraph(ApprovalState)
graph.add_node("validate", validate_request)
graph.add_node("ai_risk", ai_risk_assess)
graph.add_node("human_approve", human_approve)
graph.add_node("auto_reject", auto_reject)

graph.add_edge(START, "validate")
graph.add_conditional_edges("validate", route_after_validation, {
    "invalid": "auto_reject",
    "ai_risk_check": "ai_risk"
})
graph.add_edge("ai_risk", "human_approve")
graph.add_conditional_edges("human_approve", lambda s: s.get("approval_decision", "pending"), {
    "approve": END,
    "reject": END,
    "pending": "human_approve"  # 无效输入时重新等待
})
graph.add_edge("auto_reject", END)

checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "approval-001"}}

print("=" * 60)
print("步骤1: 提交审批申请")
print("=" * 60)

result = app.invoke({
    "request_id": "REQ-2024-0888",
    "requester": "张三",
    "request_type": "equipment_purchase",
    "amount": 15000.00,
    "description": "采购新的开发用显示器",
    "ai_assessment": "",
    "risk_level": "",
    "approver_name": "",
    "approval_decision": "",
    "approval_comment": "",
    "final_status": "",
    "audit_trail": []
}, config=config)

print(f"\n当前状态:")
print(f"  AI评估: {result['ai_assessment']}")
print(f"  风险等级: {result['risk_level']}")
print(f"  审批决策: {result['approval_decision'] or '(等待人工审批)'}")
for entry in result["audit_trail"]:
    print(f"  {entry}")

if result["approval_decision"] == "" or result["approval_decision"] == "pending":
    print("\n" + "=" * 60)
    print("步骤2: 审批人进行审批操作")
    print("=" * 60)

    result2 = app.invoke(
        Command(resume="approve 显示器是必要的办公设备"),
        config=config
    )

    print(f"\n最终状态:")
    print(f"  决策: {result2['approval_decision']}")
    print(f"  评论: {result2['approval_comment']}")
    print(f"  最终状态: {result2['final_status']}")
    for entry in result2["audit_trail"]:
        print(f"  {entry}")
```

这个单级审批流程展示了几个重要的设计要点：第一，在进入人工审批之前有自动化的前置处理（校验和风控），只有通过了这些自动化检查才会进入人工审批环节；第二，AI 风控的结果会展示给审批人作为参考信息，帮助审批人做出更明智的决策；第三，对人类输入做了验证——如果输入了无效的指令（不是 approve 也不是 reject），会重新回到 `human_approve` 节点等待正确的输入。

## 多级审批模式

在实际企业环境中，很多审批需要经过多个层级的确认——比如普通员工的报销申请需要主管批准，超过一定金额的需要经理再批一次，更大额的可能还需要总监甚至副总裁的批准。这种多级审批可以用链式的 Interrupt 节点来实现。

```python
class MultiLevelApprovalState(TypedDict):
    request_id: str
    requester: str
    amount: float
    description: string
    current_level: int
    max_levels: int
    level1_decision: str
    level1_comment: str
    level2_decision: str
    level2_comment: string
    level3_decision: str
    level3_comment: string
    final_status: string
    audit_trail: Annotated[list[string], operator.add]

LEVEL_CONFIG = {
    1: {"title": "主管审批", "threshold": 5000, "name": "李主管"},
    2: {"title": "经理审批", "threshold": 20000, "name": "王经理"},
    3: {"title": "总监审批", "threshold": 100000, "name": "赵总监"},
}

def determine_required_levels(state: MultiLevelApprovalState) -> dict:
    amount = state["amount"]

    if amount > LEVEL_CONFIG[3]["threshold"]:
        required = 3
    elif amount > LEVEL_CONFIG[2]["threshold"]:
        required = 2
    else:
        required = 1

    return {
        "max_levels": required,
        "current_level": 1,
        "audit_trail": [f"[规则] 金额¥{amount:.0f} 需要{required}级审批"]
    }

def level_approval_node(level_num):
    def node_fn(state: MultiLevelApprovalState) -> dict:
        config = LEVEL_CONFIG[level_num]
        current = state["current_level"]

        prev_decisions = []
        for lv in range(1, level_num):
            dec = state.get(f"level{lv}_decision", "")
            cmt = state.get(f"level{lv}_comment", "")
            prev_decisions.append(f"L{lv}: {dec} ({cmt})" if cmt else f"L{lv}: {dec}")

        prompt_info = (
            f"{'='*50}\n"
            f"{config['title']} ({config['name']})\n"
            f"{'='*50}\n"
            f"请求ID: {state['request_id']}\n"
            f"申请人: {state['requester']}\n"
            f"金额: ¥{state['amount']:.2f}\n"
            f"描述: {state['description']}\n"
            f"当前级别: {current}/{state['max_levels']}级\n"
        )
        if prev_decisions:
            prompt_info += f"前序审批: {' → '.join(prev_decisions)}\n"

        prompt_info += (
            f"\n请输入 'approve' 批准 或 'reject' 驳回\n"
            f"(驳回将终止整个审批流程)"
        )

        user_input = interrupt(prompt_info)

        if not user_input or not user_input.strip():
            return {"audit_trail": [f"[L{level_num}] 等待审批..."]}

        parts = user_input.strip().split(maxsplit=1)
        decision = parts[0].lower()
        comment = parts[1] if len(parts) > 1 else ""

        updates = {
            f"level{level_num}_decision": decision,
            f"level{level_num}_comment": comment,
            "audit_trail": []
        }

        if decision == "approve":
            symbol = "✅"
            updates["current_level"] = current + 1
            updates["audit_trail"].append(
                f"[L{level_num}] {symbol} {config['name']} 已批准 | {comment or '无评论'}"
            )
        elif decision == "reject":
            symbol = "❌"
            updates["final_status"] = "rejected"
            updates["audit_trail"].append(
                f"[L{level_num}] {symbol} {config['name']} 已驳回 | {comment or '无评论'}"
            )
        else:
            updates["audit_trail"].append(f"[L{level_num}] ⚠️ 无效指令: {decision}")

        return updates

    return node_fn

def route_after_level(state: MultiLevelApprovalState) -> str:
    level = state["current_level"]
    max_level = state["max_levels"]

    last_decision_key = f"level{level - 1}_decision"
    last_decision = state.get(last_decision_key, "")

    if last_decision == "reject":
        return "end_rejected"

    if level > max_level:
        return "end_approved"

    next_level_map = {1: "level1", 2: "level2", 3: "level3"}
    return next_level_map.get(level, "end_approved")

def finalize_approved(state: MultiLevelApprovalState) -> dict:
    return {
        "final_status": "approved",
        "audit_trail": [f"[完成] 🎉 全部{state['max_levels']}级审批通过!"]
    }

def finalize_rejected(state: MultiLevelApprovalState) -> dict:
    return {
        "final_status": "rejected",
        "audit_trail": ["[完成] 审批被驳回"]
    }

graph = StateGraph(MultiLevelApprovalState)
graph.add_node("determine_levels", determine_required_levels)
graph.add_node("level1", level_approval_node(1))
graph.add_node("level2", level_approval_node(2))
graph.add_node("level3", level_approval_node(3))
graph.add_node("approved", finalize_approved)
graph.add_node("rejected", finalize_rejected)

graph.add_edge(START, "determine_levels")
graph.add_edge("determine_levels", "level1")
graph.add_conditional_edges("level1", route_after_level, {
    "level2": "level2",
    "end_approved": "approved",
    "end_rejected": "rejected"
})
graph.add_conditional_edges("level2", route_after_level, {
    "level3": "level3",
    "end_approved": "approved",
    "end_rejected": "rejected"
})
graph.add_conditional_edges("level3", route_after_level, {
    "end_approved": "approved",
    "end_rejected": "rejected"
})
graph.add_edge("approved", END)
graph.add_edge("rejected", END)

app = graph.compile(checkpointer=MemorySaver())
config = {"configurable": {"thread_id": "multi-approval-001"}}

initial_state = {
    "request_id": "REQ-2024-0999",
    "requester": "王五",
    "amount": 35000.00,
    "description": "采购服务器用于项目部署",
    "current_level": 0,
    "max_levels": 0,
    "level1_decision": "", "level1_comment": "",
    "level2_decision": "", "level2_comment": "",
    "level3_decision": "", "level3_comment": "",
    "final_status": "",
    "audit_trail": []
}

print("=== 开始多级审批流程 ===\n")
result = app.invoke(initial_state, config=config)

for log in result["audit_trail"]:
    print(log)

# 模拟 L1 审批
print("\n--- L1 主管审批 ---")
r2 = app.invoke(Command(resume="approve 项目需要服务器"), config=config)
for log in r2["audit_trail"]:
    print(log)

# 模拟 L2 经理审批
print("\n--- L2 经理审批 ---")
r3 = app.invoke(Command(resume="approve 预算范围内"), config=config)
for log in r3["audit_trail"]:
    print(log)

print(f"\n最终状态: {r3['final_status']}")
```

这段程序描述了一个三级审批系统的工作原理。核心设计思路是用一个工厂函数 `level_approval_node(level_num)` 来生成每一级的审批节点函数，这样避免了为每个级别重复编写相似的代码。每级审批节点都会显示当前级别的信息、前面各级的审批结果、以及本次审批的选项。路由函数 `route_after_level` 根据当前级别和最大级别数来决定下一步：如果上一级驳回了就直接结束；如果还没到最大级别就进入下一级；如果已经通过所有级别就结束并标记为 approved。

注意这里的一个重要细节：**每级审批的状态存储在不同的字段中**（`level1_decision`/`level2_decision`/`level3_decision`），而不是用一个列表来存储。这样做的好处是每个字段都有明确的语义，便于后续查询和审计；缺点是当审批层级变化时需要调整状态定义。如果你的审批层级是固定的（比如公司规定就是三级），那这种方式完全可行；如果层级经常变动，可以考虑用一个更通用的数据结构。

## 并行审批模式

有些场景下需要多个审批人同时（或独立地）审批同一个申请，比如跨部门的项目需要技术负责人和财务负责人同时审批。这种并行审批模式可以通过从同一个节点引出多条到不同审批节点的边来实现：

```python
class ParallelApprovalState(TypedDict):
    request_id: str
    amount: float
    description: str
    tech_reviewer: str
    tech_decision: str
    tech_comment: str
    finance_reviewer: str
    finance_decision: str
    finance_comment: str
    final_status: str
    audit_trail: Annotated[list[str], operator.add]

def tech_review(state: ParallelApprovalState) -> dict:
    prompt = (
        f"【技术审批】\n"
        f"请求ID: {state['request_id']}\n"
        f"描述: {state['description']}\n\n"
        f"请从技术角度审批:\n"
        f"- approve: 技术可行\n"
        f"- reject: 技术不可行\n"
        f"(可附评论)"
    )
    user_input = interrupt(prompt)

    if not user_input or not user_input.strip():
        return {"tech_decision": "pending", "audit_trail": ["[技术] 等待审批"]}

    parts = user_input.strip().split(maxsplit=1)
    dec = parts[0].lower()
    cmt = parts[1] if len(parts) > 1 else ""

    status = "✅ 通过" if dec == "approve" else ("❌ 驳回" if dec == "reject" else "⚠️ 无效")
    return {
        "tech_decision": dec,
        "tech_comment": cmt,
        "audit_trail": [f"[技术] {status} | {cmt or '无'}"]
    }

def finance_review(state: ParallelApprovalState) -> dict:
    prompt = (
        f"【财务审批】\n"
        f"请求ID: {state['request_id']}\n"
        f"金额: ¥{state['amount']:.2f}\n"
        f"描述: {state['description']}\n\n"
        f"请从财务角度审批:\n"
        f"- approve: 预算充足\n"
        f"- reject: 预算不足\n"
        f"(可附评论)"
    )
    user_input = interrupt(prompt)

    if not user_input or not user_input.strip():
        return {"finance_decision": "pending", "audit_trail": ["[财务] 等待审批"]}

    parts = user_input.strip().split(maxsplit=1)
    dec = parts[0].lower()
    cmt = parts[1] if len(parts) > 1 else ""

    status = "✅ 通过" if dec == "approve" else ("❌ 驳回" if dec == "reject" else "⚠️ 无效")
    return {
        "finance_decision": dec,
        "finance_comment": cmt,
        "audit_trail": [f"[财务] {status} | {cmt or '无'}"]
    }

def merge_parallel_results(state: ParallelApprovalState) -> dict:
    tech = state["tech_decision"]
    fin = state["finance_decision"]

    if tech == "approve" and fin == "approve":
        final = "approved"
        msg = "🎉 双方均通过"
    elif tech in ("reject", "") or fin in ("reject", ""):
        final = "rejected"
        reasons = []
        if tech == "reject":
            reasons.append(f"技术方驳回: {state['tech_comment']}")
        if fin == "reject":
            reasons.append(f"财务方驳回: {state['finance_comment']}")
        msg = f"❌ 未通过 | {'; '.join(reasons)}"
    else:
        final = "pending"
        msg = "⏳ 等待另一方审批..."

    return {
        "final_status": final,
        "audit_trail": [f"[汇总] {msg}"]
    }

def route_parallel_result(state: ParallelApprovalState) -> str:
    if state["final_status"] == "approved":
        return "done"
    if state["final_status"] == "rejected":
        return "done"
    return "wait_more"

graph = StateGraph(ParallelApprovalState)
graph.add_node("tech_review", tech_review)
graph.add_node("finance_review", finance_review)
graph.add_node("merge", merge_parallel_results)
graph.add_node("done", lambda s: {"audit_trail": ["[完成] 审批流程结束"]})

graph.add_edge(START, "tech_review")
graph.add_edge(START, "finance_review")  # 并行启动两个审批
graph.add_edge("tech_review", "merge")
graph.add_edge("finance_review", "merge")
graph.add_conditional_edges("merge", route_parallel_result, {
    "done": "done",
    "wait_more": "merge"  # 一方还在等待，继续等
})
graph.add_edge("done", END)

app = graph.compile(checkpointer=MemorySaver())

config = {"configurable": {"thread_id": "parallel-001"}}
init_state = {
    "request_id": "REQ-PARALLEL-001",
    "amount": 80000.00,
    "description": "购买GPU服务器用于模型训练",
    "tech_reviewer": "技术总监",
    "tech_decision": "", "tech_comment": "",
    "finance_reviewer": "财务经理",
    "finance_decision": "", "finance_comment": "",
    "final_status": "",
    "audit_trail": []
}
```

这里有一个关键的结构特征：START 节点同时引出两条边到 `tech_review` 和 `finance_review`，实现了**扇出（fan-out）**效果——两个审批节点可以并行执行（虽然在默认的单线程执行模式下它们实际上是串行的，但图的结构表达了并行的语义）。然后两个审批节点都汇聚到 `merge` 节点做结果合并，这就是**扇入（fan-in）**。

需要注意的是，由于 LangGraph 默认是单线程执行的，这里的"并行"更多是结构上的并行而非真正的并发执行。如果你需要真正的并行执行（比如两个审批人同时在不同的终端上操作），需要使用 LangGraph 的 Send API 或者在外层管理多个独立的图实例。但即使不真正并行，这种结构化表达方式仍然有价值——它清晰地说明了这两个审批之间没有依赖关系，顺序不重要。

## 审批流的常见误区

在使用 Interrupt 构建审批流时，有几个常见的误区值得注意。

第一个误区是**在 Interrupt 节点之前缺少必要的信息准备**。审批人在做决策时需要看到足够的信息——申请的基本内容、AI 的分析结果、相关的历史记录等。如果直接跳到一个空的 Interrupt 节点让审批人凭空做决定，体验会很差且容易出错。应该在 Interrupt 节点之前安排专门的数据准备节点，收集和整理所有相关信息。

第二个误区是**没有考虑审批超时的情况**。如果审批人长时间不操作怎么办？在实际系统中应该设置审批超时机制——比如 24 小时内没有审批则自动提醒，48 小时内没有审批则自动升级或拒绝。LangGraph 本身不提供内置的超时机制，你需要在外层实现定时检查逻辑，或者结合任务调度框架（如 Celery beat）来实现。

第三个误区是**没有保留完整的审计轨迹**。审批流程通常有合规要求，需要记录谁在什么时间做了什么决策、基于什么信息、留下了什么评论。确保你的状态设计中包含足够的审计字段（如 `audit_trail`），并且在每个关键节点都追加相应的日志信息。

第四个误区是**忽略了驳回后的处理逻辑**。审批被驳回后不应该只是简单地标记为 rejected 就完事了——可能需要通知申请人、可能需要记录驳回原因以便后续改进、可能需要触发某些清理操作。在图中应该为驳回分支也设计完整的后续处理逻辑。
