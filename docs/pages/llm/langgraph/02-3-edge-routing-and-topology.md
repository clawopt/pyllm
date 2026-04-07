# 2.3 Edge 编程：条件路由与图拓扑设计

## 边是图的"神经系统"——控制信息流动的规则

如果 Node 是图中的"器官"，那 Edge 就是连接这些器官的"神经"。Edge 决定了执行流经哪个节点、以什么顺序、在什么条件下跳转或终止。写好 Node 只是让你的图能"跑起来"，写好 Edge 才能让你的图"跑得对"。这一节我们要深入探讨 Edge 的各种模式和设计策略。

## Edge 的两种基本类型

LangGraph 中有两种 Edge，它们的使用场景和语法都不同：

```python
# 类型一：普通边（无条件转移）
graph.add_edge("node_a", "node_b")
# 含义: node_a 执行完后 → 100% 转移到 node_b

# 类型二：条件边（有条件转移）
graph.add_conditional_edges(
    "decision_node",
    routing_function,           # 函数: 接收 State → 返回字符串(目标节点名)
    path_map,                   # 字典:   {函数返回值: 目标节点名}
)
# 含义: decision_node 执行完后 → 根据路由函数的返回值选择路径
```

普通边表达的是**确定性关系**（"A 之后一定是 B"），而条件边表达的是**动态决策**（"A 之后看情况，可能去 B 也可能去 C"）。一个实际的图通常会同时使用两者——大部分路径用普通边串联，关键分支点用条件边分叉。

## 条件边的完整生命周期

让我们追踪一个条件边从定义到执行的完整过程：

```python
from typing import TypedDict


class TicketState(TypedDict):
    ticket_content: str
    category: str          # billing / technical / complaint
    priority: int         # P1/P2/P3
    auto_resolve: bool     # 能否自动解决
    resolution: str
    status: str


def categorize(state: TicketState) -> dict:
    """Node: 分类工单"""
    content = state["ticket_content"].lower()
    
    if "退款" in content or "账单" in content:
        return {"category": "billing", "auto_resolve": True}
    elif "bug" in content or "报错" in content or "无法":
        return {"category": "technical", "auto_resolve": False}
    elif "投诉" in content or "不满意":
        return {"category": "complaint", "auto_resolve": False}
    else:
        return {"category": "general", "auto_resolve": True}


def should_auto_resolve(state: TicketState) -> str:
    """条件路由: 根据分类和优先级决定是否尝试自动解决"""
    
    if state["priority"] == 1 and state["auto_resolve"]:
        return "auto_fix"
    elif not state["auto_resolve"]:
        return "escalate_to_human"
    else:
        return "try_auto_fix"


def try_auto_fix(state: TicketState) -> dict:
    """Node: 尝试自动修复"""
    # ... 自动修复逻辑 ...
    success = attempt_fix(state["ticket_content"])
    return {
        "resolution": f"自动修复结果: {success}",
        "status": "resolved" if success else "fix_failed",
    }


def escalate(state: TicketState) -> dict:
    """Node: 升级处理"""
    return {
        "resolution": "已升级给资深工程师处理",
        "status": "escalated",
    }


def confirm_with_user(state: TicketState) -> dict:
    """Node: 等待用户确认"""
    return {
        "status": "awaiting_confirmation",
    }


def close_ticket(state: TicketState) -> dict:
    """Node: 关闭工单"""
    return {"status": "closed"}


# 组装图
graph = StateGraph(TicketState)

graph.add_node("categorize", categorize)
graph.add_node("should_auto_resolve", should_auto_resolve)
graph.add_node("try_auto_fix", try_auto_fix)
graph.add_node("escalate", escalate)
graph.add_node("confirm_with_user", confirm_with_user)
graph.add_node("close_ticket", close_ticket)

graph.add_edge(START, "categorize")

graph.add_conditional_edges(
    "categorize",
    should_auto_resolve,
    {
        "auto_fix": "try_auto_fix",
        "escalate_to_human": "escalate",
        "general": "try_auto_fix",  # general 类别也先尝试自动修复
    },
)

graph.add_edge("try_auto_fix", "confirm_with_user")      # 修复后需确认
graph.add_edge("try_auto_fix", "close_ticket")         # 修复成功直接关闭
graph.add_edge("escalate", "confirm_with_user")            # 升级后需确认
graph.add_edge("confirm_with_user", "close_ticket")             # 确认后关闭
graph.add_edge("close_ticket", END)

app = graph.compile()

# 测试不同类型的工单
test_cases = [
    ("我要退款，上个月多扣了50块", "billing"),
    ("系统报错 500 Internal Server Error", "technical"),
    ("你们的服务太差了！我要投诉！", "complaint"),
]

for content, expected_cat in test_cases:
    result = app.invoke({
        "ticket_content": content,
        "status": "new",
        "priority": 2 if "紧急" in content else 3,
        "auto_resolve": False,
        "resolution": "",
    })
    print(f"\n[{expected_cat}] {content[:30]} → {result['status']} ({result.get('resolution', '')})")
```

这个例子展示了条件边的几个重要特性：

**路由函数可以访问完整的 State。** `should_auto_resolve` 不仅看了 `category` 还看了 `priority` 和 `auto_resolve`——这意味着路由决策可以是多维度的组合判断。

**path_map 不需要覆盖所有可能的返回值。** 注意上面的 `general` 类别的工单被映射到了 `"try_auto_fix"` 而不是独立的 `"general"` 处理路径。这是有意为之——通用类别的工单也应该先尝试自动修复（因为大多数情况下能修好），只有修复失败才走升级路径。这是一种**"乐观默认 + 保守回退"**的策略。

**同一个目标节点可以被多条边指向。** `confirm_with_user` 有三条入边分别来自 `try_auto_fix`、`escalate` 和 `general` 分支——这说明无论从哪条路过来，最终都需要用户确认后才关闭工单。这种"多入单出"的模式在实际中非常常见。

## 动态路由：让 LLM 做路由决策

到目前为止我们所有的路由函数都是硬编码的 `if/else` 规则。但在 Agent 场景中，很多时候你希望 **LLM 根据上下文动态决定下一步做什么**——这就是 LangGraph 的动态路由能力：

```python
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI


class ResearchState(TypedDict):
    question: str
    research_plan: list[dict]
    current_step_idx: int
    findings: dict
    final_report: str
    needs_clarification: bool
    status: str


router_llm = ChatOpenAI(model="gpt-4o", temperature=0)


def researcher(state: ResearchState) -> dict:
    """研究 Node: 执行一步研究"""
    
    plan = state["research_plan"]
    step_idx = state["current_step_idx"]
    
    if step_idx >= len(plan):
        return {"status": "all_researched"}
    
    step = plan[step_idx]
    
    print(f"[研究步骤 {step_idx+1}/{len(plan)}]: {step['description']}")
    
    # 调用工具获取信息
    tool_result = execute_tool(step["tool"], step["query"])
    
    findings = state["findings"]
    findings[step["key"]] = tool_result
    
    return {
        "current_step_idx": step_idx + 1,
        "findings": findings,
        "needs_clarification": False,
    }


def route_next_step(state: ResearchState) -> str:
    """
    动态路由: LLM 决定下一步是继续研究还是综合报告
    
    这是 LangGraph 最强大的特性之一 ——
    路由函数可以让 LLM 参与路径决策！
    """
    
    current_idx = state["current_step_idx"]
    total_steps = len(state["research_plan"])
    findings = state["findings"]
    
    # 如果所有研究步骤已完成
    if current_idx >= total_steps:
        return "synthesize"
    
    # 让 LLM 判断是否需要更多信息
    prompt = f"""你是一个研究协调者。当前研究进度:

已完成的步骤: {current_idx}/{total_steps}
总计划步骤数: {total_steps}

已收集的信息:
{json.dumps(findings, indent=2, ensure_ascii=False)}

下一步应该:
- 如果已经收集到足够的信息来回答原始问题 → 输出 "synthesize"
- 如果某些关键信息还缺失 → 输出 "continue_research" 并指出缺失什么
- 如果发现方向完全错误 → 输出 "pivot" (改变研究方向)

只输出一个词: continue_research 或 synthesize 或 pivot"""

    response = router_llm.invoke(prompt)
    decision = response.content.strip().lower()
    
    # 标准化 LLM 的输出
    for keyword in ["continue", "go on", "more info", "继续", "research"]:
        if keyword in decision:
            return "continue_research"
    
    for keyword in ["synthesize", "write", "report", "conclude", "answer", "done", "finish"]:
        if keyword in decision:
            return "synthesize"
    
    # 默认：继续研究
    return "continue_research"


def synthesize(state: ResearchState) -> dict:
    """综合 Node: 把研究结果写成最终报告"""
    
    report = router_llm.invoke(
        f"基于以下研究发现，撰写关于 '{state['question']}' 的报告:\n\n"
        f"{json.dumps(state['findings'], indent=2)}"
    )
    
    return {
        "final_report": report.content,
        "status": "done",
    }
```

动态路由让 Agent 具备了真正的**自主决策能力**——它不再是按照预定的线性流程机械地执行，而是可以根据中间结果灵活调整策略。这比硬编码的条件边更强大，但也更难调试——因为 LLM 的路由决定不那么可预测。

使用动态路由时的注意事项：

**LLM 路由的稳定性问题。** LLM 可能返回 path_map 中不存在的值（比如输出了 "keep_going" 但你只注册了 "continue_research"）。解决方案：（1）在路由函数中做标准化映射；（2）使用结构化 output parser 强制 LLM 返回合法值；（3）设置 fallback 默认值。

**成本考虑。** 每次条件边调用都会产生一次 LLM API 调用。如果你的图有 10 个 Node 且其中 5 个使用了条件边，每次 invoke 可能会产生 5+ 次额外的 LLM 调用。在高并发场景下这可能成为性能瓶颈。可以考虑对路由结果做短期缓存（相同的 State 在短时间内不需要重新路由）。

## 图拓扑模式实战

最后让我们看几种常见的图拓扑结构和它们的适用场景：

### 模式一：线性流水线

```
START → [Step1] → [Step2] → [Step3] → [Step4] → END

适用: ETL 数据管道、固定步骤的审批流
特点: 简单可靠，无分支，易于理解和调试
```

### 模式二：扇出汇聚

```
                → [并行A] ─┐
START → [分发] ─┼→ [汇聚] → END
                → [并行B] ┘┘

适用: 多源数据聚合、并行调研后汇总
特点: 提高效率，但汇聚逻辑需要注意顺序
```

### 模式三：带循环的自省

```
START → [Try] → {成功?}─→ YES → END
                 │ NO
                 ↓
              [Fix] → {足够好?}─→ YES → END
                 │ NO
                 ↓
           [Escalate] → [Human] → END

适用: 自我修复循环、最大重试限制
特点: 有明确的退出条件，防止无限循环
```

### 模式四：人机协作环

```
START → [Agent Action] → {需要确认?}─→ NO → [修正后重试] → (回到 Action)
                              │ YES
                              ▼
                        [Human Confirm] → END

适用: 需要人工审核的关键决策点
特点: Interrupt 机制支持任意节点暂停
```

## 总结

