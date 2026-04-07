# 7.4 Checkpoint 与 Interrupt 的协同工作

> 在第4章中我们学习了 Interrupt（人机协作暂停）机制，在前面几节中又深入了解了 Checkpointing（状态持久化）。这两个特性之间存在一个深层的依赖关系：**Interrupt 必须配合 Checkpointing 才能正常工作**。原因很简单——当图执行到 Interrupt 节点并暂停时，当前的全部状态必须被持久化保存下来，否则当人类提供输入后恢复执行时，之前的状态已经丢失了，图不知道该从哪里继续。这一节我们会详细探讨这两个机制是如何协同工作的，以及在实际应用中需要注意的边界情况。

## Interrupt 为什么需要 Checkpointing

让我们先从"为什么"开始理解这个依赖关系。

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command

class ApprovalState(TypedDict):
    request_data: str
    ai_analysis: str
    human_decision: str
    final_status: str
```

假设有一个包含 Interrupt 节点的图，但没有配置 checkpointer：

```python
# ❌ 错误：有 Interrupt 但没有 checkpointer
def review_node(state: ApprovalState) -> dict:
    user_input = interrupt("请审批此请求: [approve/reject]")
    return {"human_decision": user_input}

graph = StateGraph(ApprovalState)
graph.add_node("review", review_node)
graph.add_edge(START, "review")
graph.add_edge("review", END)

# 没有 checkpointer！
app_no_cp = graph.compile()  # ← 这里没有 checkpointer 参数

try:
    result = app_no_cp.invoke({
        "request_data": "test",
        "ai_analysis": "AI分析结果",
        "human_decision": "",
        "final_status": ""
    })
except Exception as e:
    print(f"❌ 错误: {type(e).__name__}: {e}")
    # 会抛出类似这样的错误：
    # "Graph encountered an 'interrupt' but no checkpointer was configured.
    #  Please pass a checkpointer to compile() to use interrupts."
```

LangGraph 在编译阶段和运行时都会检查：如果图中包含了 Interrupt 节点但没配置 checkpointer，会直接报错拒绝执行。这是一个**设计上的强制约束**，而不是可选的建议——它防止了你在开发阶段用 MemorySaver 测试通过后，到生产环境忘记配 checkpointer 导致的严重 bug。

## 协同工作的完整生命周期

现在来看看有 checkpointer 时，Interrupt 的完整生命周期：

```
时间线 (thread_id = "approval-001"):

t=0s   用户调用 app.invoke(initial_state, config=config)
       │
t=0.1s LangGraph 创建 checkpoint S0 (初始状态)
       │
t=0.2s 节点A (ai_analyze) 执行 → 更新状态
       │
t=0.3s LangGraph 创建 checkpoint S1 (节点A后的状态)
       │
t=0.4s 节点B (human_review) 开始执行
       │   内部调用 interrupt("请审批...")
       │
t=0.4s ⚡️ interrupt 触发!
       │   - 保存当前状态为 checkpoint S2
       │   - 标记 S2 为 "pending_interrupt" 状态
       │   - 返回 prompt 数据给调用者
       │
       ──── invoke() 返回给用户 ────
       │     返回值 = interrupt() 的参数 (prompt数据)
       │     图的执行在此处暂停!
       │
       ... 时间流逝 (可能几分钟、几小时、几天) ...
       │
       │   人类看到 prompt，做出决策
       │
t=Ns   用户调用 app.invoke(Command(resume="approve"), config=config)
       │   使用相同的 config (相同的 thread_id!)
       │
t=N.1s LangGraph 从 checkpointer 加载最新的 checkpoint
       │   发现是 S2 (pending_interrupt 状态)
       │
t=N.2s 将 resume 值 ("approve") 作为 interrupt() 的返回值
       │   继续执行 human_review 节点的剩余逻辑
       │
t=N.3s human_review 节点完成 → 更新状态
       │
t=N.4s LangGraph 创建 checkpoint S3 (人类决策后的状态)
       │
t=N.5s 后续节点继续执行...
       │
t=NMs  最终完成，返回最终结果
```

这个生命周期的关键特征是：**两次 `invoke()` 调用之间可以间隔任意长的时间**——因为中间的状态被安全地保存在 checkpointer 中了。无论是 1 秒钟还是 1 周，只要 checkpointer 的数据还在，就能正确恢复。

## 多次 Interrupt 与多级暂停

有些场景下，一次完整的执行过程中可能会有多次 Interrupt——比如先让用户确认基本信息，再让专家审核技术细节，最后让经理做最终批准。每次 Interrupt 都会创建一个新的 checkpoint，形成一条完整的中断-恢复链路。

```python
from typing import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

class MultiApprovalState(TypedDict):
    request_id: str
    amount: float
    tier1_approval: str
    tier1_comment: str
    tier2_approval: str
    tier2_comment: str
    final_status: str
    log: Annotated[list[str], operator.add]

def auto_validate(state: MultiApprovalState) -> dict:
    amount = state["amount"]
    if amount <= 0:
        return {
            "final_status": "rejected_invalid",
            "log": ["[自动校验] ❌ 金额无效"]
        }
    return {"log": [f"[自动校验] ✅ 金额 ¥{amount:.2f} 有效"]}

def tier1_approve(state: MultiApprovalState) -> dict:
    prompt = (
        f"一级审批\n"
        f"请求ID: {state['request_id']}\n"
        f"金额: ¥{state['amount']:.2f}\n\n"
        f"输入 approve 或 reject [可附加评论]:"
    )
    user_input = interrupt(prompt)

    if not user_input or not user_input.strip():
        return {"log": ["[一级审批] 等待输入..."]}

    parts = user_input.strip().split(maxsplit=1)
    decision = parts[0].lower()
    comment = parts[1] if len(parts) > 1 else ""

    if decision not in ("approve", "reject"):
        return {"log": [f"[一级审批] ⚠️ 无效指令: {decision}"]}

    symbol = "✅" if decision == "approve" else "❌"
    return {
        "tier1_approval": decision,
        "tier1_comment": comment,
        "log": [f"[一级审批] {symbol} {decision} {comment}"]
    }

def route_after_tier1(state: MultiApprovalState) -> str:
    if state["tier1_approval"] == "reject":
        return "end_rejected"
    if state["amount"] > 10000:
        return "tier2_approve"
    return "end_approved"

def tier2_approve(state: MultiApprovalState) -> dict:
    prompt = (
        f"二级审批 (大额)\n"
        f"请求ID: {state['request_id']}\n"
        f"金额: ¥{state['amount']:.2f}\n"
        f"一级审批: {state['tier1_approval']} ({state['tier1_comment']})\n\n"
        f"输入 approve 或 reject [可附加评论]:"
    )
    user_input = interrupt(prompt)

    parts = user_input.strip().split(maxsplit=1)
    decision = parts[0].lower()
    comment = parts[1] if len(parts) > 1 else ""

    symbol = "✅" if decision == "approve" else "❌"
    return {
        "tier2_approval": decision,
        "tier2_comment": comment,
        "log": [f"[二级审批] {symbol} {decision} {comment}"]
    }

def end_approved(state: MultiApprovalState) -> dict:
    return {"final_status": "approved", "log": ["[完成] 🎉 审批通过"]}
def end_rejected(state: MultiApprovalState) -> dict:
    reason = state.get("tier1_comment") or state.get("tier2_comment") or "未说明"
    return {"final_status": "rejected", "log": [f"[完成] 审批被拒绝: {reason}"]}

multi_graph = StateGraph(MultiApprovalState)
multi_graph.add_node("validate", auto_validate)
multi_graph.add_node("tier1", tier1_approve)
multi_graph.add_node("tier2", tier2_approve)
multi_graph.add_node("approved", end_approved)
multi_graph.add_node("rejected", end_rejected)

multi_graph.add_edge(START, "validate")
multi_graph.add_edge("validate", "tier1")
multi_graph.add_conditional_edges("tier1", route_after_tier1, {
    "tier2_approve": "tier2",
    "end_approved": "approved",
    "end_rejected": "rejected"
})
multi_graph.add_conditional_edges("tier2",
    lambda s: "approved" if s["tier2_approval"] == "approve" else "rejected",
    {"approved": "approved", "rejected": "rejected"}
)
multi_graph.add_edge("approved", END)
multi_graph.add_edge("rejected", END)

checkpointer = MemorySaver()
app = multi_graph.compile(checkpointer=checkpointer)
config = {"configurable": {"thread_id": "multi-approval-demo"}}

print("=" * 60)
print("步骤1: 启动审批流程")
print("=" * 60)

result1 = app.invoke({
    "request_id": "REQ-2024-042",
    "amount": 25000.00,
    "tier1_approval": "", "tier1_comment": "",
    "tier2_approval": "", "tier2_comment": "",
    "final_status": "", "log": []
}, config=config)

for entry in result1["log"]:
    print(entry)

print("\n" + "=" * 60)
print("步骤2: 一级审批人操作")
print("=" * 60)

result2 = app.invoke(
    Command(resume="approve 金额在预算范围内"),
    config=config
)

for entry in result2["log"]:
    print(entry)

if result2.get("final_status"):
    print(f"\n最终状态: {result2['final_status']}")
else:
    print("\n需要二级审批...")

    print("\n" + "=" * 60)
    print("步骤3: 二级审批人操作")
    print("=" * 60)

    result3 = app.invoke(
        Command(resume="approve 大额但风险可控"),
        config=config
    )

    for entry in result3["log"]:
        print(entry)
    print(f"\n最终状态: {result3['final_status']}")
```

这个多级审批流程展示了两次 Interrupt 的协同工作：
1. 第一次 invoke → 执行到 `tier1` 节点的 interrupt → 暂停，保存 checkpoint
2. 第二次 invoke（resume 一级审批）→ 继续执行 → 如果金额大进入 `tier2` 节点的 interrupt → 再次暂停
3. 第三次 invoke（resume 二级审批）→ 继续执行 → 完成

每次暂停都对应一个独立的 checkpoint，checkpointer 中保存了完整的链式历史。你可以随时查看任意时刻的状态快照。
