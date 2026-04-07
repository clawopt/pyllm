# 3.2 多层路由与嵌套判断

> 实际业务中的决策逻辑很少是简单的一层 if-else——更常见的情况是"先判断 A，如果 A 成立再判断 B，B 成立还要看 C"，这种多层嵌套的决策结构在代码中用 if-elif-else 很自然地表达，但在图的语境下需要用多个条件边的串联来实现。这一节我们会探讨如何用 LangGraph 的条件边来构建多层路由系统，包括链式路由、分阶段决策、以及如何避免"路由爆炸"这种常见的架构恶化问题。

## 链式路由：把大决策拆成多步小决策

当你的路由逻辑需要考虑多个维度的因素时，最自然的做法不是写一个巨大的路由函数在里面堆砌嵌套的 if-elif，而是把决策过程拆分成多个连续的路由步骤，每一步只关注一个维度。这就像面试时的多轮筛选：第一轮看学历够不够，第二轮看技术能力，第三轮看文化匹配度——每一轮的判断标准不同，但只有全部通过才能拿到 offer。

```python
from typing import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, START, END

class LoanApprovalState(TypedDict):
    applicant_id: str
    income: float
    credit_score: int
    employment_years: float
    loan_amount: float
    debt_ratio: float
    stage1_result: str
    stage2_result: str
    final_decision: str
    interest_rate: float
    audit_log: Annotated[list[str], operator.add]

def stage1_basic_check(state: LoanApprovalState) -> dict:
    income = state["income"]
    score = state["credit_score"]
    years = state["employment_years"]
    log = [f"[第一阶段] 收入={income}, 评分={score}, 工龄={years}"]

    if income < 3000:
        return {"stage1_result": "reject_income", "audit_log": log + ["→ 收入不达标"]}
    if score < 400:
        return {"stage1_result": "reject_credit", "audit_log": log + ["→ 评分过低"]}
    if years < 0.5:
        return {"stage1_result": "reject_employment", "audit_log": log + ["→ 工龄不足"]}

    return {"stage1_result": "pass", "audit_log": log + ["✅ 基本条件通过"]}

def route_stage1(state: LoanApprovalState) -> str:
    return state["stage1_result"]

def stage2_risk_assessment(state: LoanApprovalState) -> dict:
    amount = state["loan_amount"]
    income = state["income"]
    debt = state.get("debt_ratio", 0)
    monthly_payment = amount / 36
    dti = (monthly_payment + debt * income) / income

    log = [f"[第二阶段] 贷款额={amount}, DTI={dti:.1%}"]
    if dti > 0.55:
        return {
            "stage2_result": "reject_dti",
            "debt_ratio": dti,
            "audit_log": log + [f"→ 债务收入比过高 ({dti:.1%})"]
        }
    if amount > income * 10:
        return {
            "stage2_result": "reject_amount",
            "debt_ratio": dti,
            "audit_log": log + [f"→ 贷款金额超限 (>{income*10})"]
        }

    risk_premium = 0
    if state["credit_score"] < 600:
        risk_premium += 2.0
    if dti > 0.4:
        risk_premium += 1.5
    base_rate = 4.5 + risk_premium

    return {
        "stage2_result": "approve",
        "debt_ratio": dti,
        "interest_rate": base_rate,
        "audit_log": log + [f"✅ 风控通过, 利率={base_rate}%"]
    }

def route_stage2(state: LoanApprovalState) -> str:
    return state["stage2_result"]

def reject_handler(msg: str):
    def handler(state: LoanApprovalState) -> dict:
        return {"final_decision": "rejected", "audit_log": [msg]}
    return handler

def approve_handler(state: LoanApprovalState) -> dict:
    rate = state["interest_rate"]
    return {
        "final_decision": "approved",
        "interest_rate": rate,
        "audit_log": [f"🎉 贷款批准! 最终利率: {rate}%"]
    }

graph = StateGraph(LoanApprovalState)
graph.add_node("stage1_check", stage1_basic_check)
graph.add_node("stage2_risk", stage2_risk_assessment)
graph.add_node("reject_income", reject_handler("❌ 拒绝: 月收入低于最低要求 (3000元)"))
graph.add_node("reject_credit", reject_handler("❌ 拒绝: 信用评分不足 (低于400)"))
graph.add_node("reject_employment", reject_handler("❌ 拒绝: 工作年限不足 (少于6个月)"))
graph.add_node("reject_dti", reject_handler("❌ 拒绝: 债务收入比超过55%"))
graph.add_node("reject_amount", reject_handler("❌ 拒绝: 贷款金额超过收入10倍"))
graph.add_node("approve", approve_handler)

graph.add_edge(START, "stage1_check")
graph.add_conditional_edges("stage1_check", route_stage1, {
    "pass": "stage2_risk",
    "reject_income": "reject_income",
    "reject_credit": "reject_credit",
    "reject_employment": "reject_employment"
})
graph.add_conditional_edges("stage2_risk", route_stage2, {
    "approve": "approve",
    "reject_dti": "reject_dti",
    "reject_amount": "reject_amount"
})
graph.add_edge("reject_income", END)
graph.add_edge("reject_credit", END)
graph.add_edge("reject_employment", END)
graph.add_edge("reject_dti", END)
graph.add_edge("reject_amount", END)
graph.add_edge("approve", END)

app = graph.compile()

case1 = app.invoke({
    "applicant_id": "U-1001", "income": 8000, "credit_score": 720,
    "employment_years": 3, "loan_amount": 200000, "debt_ratio": 0.1,
    "stage1_result": "", "stage2_result": "", "final_decision": "",
    "interest_rate": 0.0, "audit_log": []
})
print(f"\n申请人 {case1['applicant_id']}: {case1['final_decision']}")
for entry in case1["audit_log"]:
    print(f"  {entry}")

case2 = app.invoke({
    "applicant_id": "U-1002", "income": 2500, "credit_score": 650,
    "employment_years": 2, "loan_amount": 50000, "debt_ratio": 0.05,
    "stage1_result": "", "stage2_result": "", "final_decision": "",
    "interest_rate": 0.0, "audit_log": []
})
print(f"\n申请人 {case2['applicant_id']}: {case2['final_decision']}")
for entry in case2["audit_log"]:
    print(f"  {entry}")
```

这个贷款审批流程展示了经典的**两阶段链式路由**模式。第一阶段做基本资格审查（收入底线、信用评分下限、工作年限），任何一项不达标就立即拒绝并走对应的拒绝处理节点；只有三项都通过了才进入第二阶段的风控评估（债务收入比检查、贷款金额上限检查）。第二阶段的三个结果（批准/DTI拒绝/金额拒绝）分别走向不同的终端节点。

注意这里的一个设计细节：每种拒绝原因都有自己独立的处理节点（`reject_income`、`reject_credit` 等），而不是共用一个通用的拒绝节点。这样做的好处是每个拒绝节点可以携带特定的拒绝消息和后续处理逻辑（比如低收入拒绝可能触发推荐其他金融产品的动作，而低信用评分拒绝可能触发信用修复建议）。当然如果你的业务不需要区分拒绝类型，也可以合并为一个节点来减少图中的节点总数。

## 分阶段决策中的状态传递

在多层路由中，一个需要特别注意的问题是状态在各阶段之间的传递。因为每一层的路由函数只能看到当前的全局状态，你需要确保前一阶段的计算结果被正确地写入到状态中，后一阶段才能读到。

在上面的贷款例子中，`stage1_basic_check` 把结果写入了 `stage1_result` 字段，然后 `route_stage1` 读取这个字段来做路由。同样地，`stage2_risk_assessment` 把 DTI 计算结果和利率都写入了状态，供后面的 `approve_handler` 使用。这种显式的中间状态存储虽然看起来有些冗余（毕竟这些数据也可以重新计算），但它有两个重要的好处：一是**可追溯性**——你可以在 checkpoint 中看到每个阶段的原始输出，方便调试和审计；二是**避免重复计算**——特别是对于那些涉及外部 API 调用或复杂计算的值，算一次存下来比每次都重新算要高效得多。

```python
class MultiStageState(TypedDict):
    raw_input: str
    stage1_output: str
    stage2_output: str
    stage3_output: str
    final_result: str
    decision_path: Annotated[list[str], operator.add]

def stage1_normalize(state: MultiStageState) -> dict:
    text = state["raw_input"].strip().lower()
    normalized = " ".join(text.split())
    return {
        "stage1_output": normalized,
        "decision_path": ["[S1] 文本规范化完成"]
    }

def stage1_route(state: MultiStageState) -> str:
    text = state["stage1_output"]
    if not text:
        return "empty"
    return "continue"

def stage2_classify(state: MultiStageState) -> dict:
    text = state["stage1_output"]
    if any(w in text for w in ["error", "exception", "bug", "失败", "错误"]):
        category = "issue_report"
    elif any(w in text for w in ["how", "what", "why", "如何", "什么", "为什么"]):
        category = "question"
    elif any(w in text for w in ["please", "can you", "请", "能否", "帮忙"]):
        category = "request"
    else:
        category = "other"
    return {
        "stage2_output": category,
        "decision_path": [f"[S2] 分类为: {category}"]
    }

def stage2_route(state: MultiStageState) -> str:
    cat = state["stage2_output"]
    mapping = {"issue_report": "handle_issue", "question": "handle_question",
               "request": "handle_request", "other": "handle_other"}
    return mapping.get(cat, "handle_other")

def handle_issue(state: MultiStageState) -> dict:
    return {
        "stage3_output": "已创建工单，分配给技术支持团队",
        "final_result": "工单已创建",
        "decision_path": ["[S3] 问题报告 → 工单创建"]
    }
def handle_question(state: MultiStageState) -> dict:
    return {
        "stage3_output": "已查询知识库并生成回答",
        "final_result": "知识库回答",
        "decision_path": ["[S3] 问题 → 知识库检索"]
    }
def handle_request(state: MultiStageState) -> dict:
    return {
        "stage3_output": "请求已转发给对应服务台",
        "final_result": "请求已转交",
        "decision_path": ["[S3] 请求 → 服务台分发"]
    }
def handle_other(state: MultiStageState) -> dict:
    return {
        "stage3_output": "转入人工客服队列",
        "final_result": "等待人工处理",
        "decision_path": ["[S3] 其他 → 人工队列"]
    }
def handle_empty(state: MultiStageState) -> dict:
    return {
        "final_result": "输入为空，忽略",
        "decision_path": ["[S1] 空输入 → 忽略"]
    }

graph = StateGraph(MultiStageState)
graph.add_node("normalize", stage1_normalize)
graph.add_node("classify", stage2_classify)
graph.add_node("handle_issue", handle_issue)
graph.add_node("handle_question", handle_question)
graph.add_node("handle_request", handle_request)
graph.add_node("handle_other", handle_other)
graph.add_node("handle_empty", handle_empty)

graph.add_edge(START, "normalize")
graph.add_conditional_edges("normalize", stage1_route, {
    "continue": "classify",
    "empty": "handle_empty"
})
graph.add_conditional_edges("classify", stage2_route, {
    "handle_issue": "handle_issue",
    "handle_question": "handle_question",
    "handle_request": "handle_request",
    "handle_other": "handle_other"
})
graph.add_edge("handle_issue", END)
graph.add_edge("handle_question", END)
graph.add_edge("handle_request", END)
graph.add_edge("handle_other", END)
graph.add_edge("handle_empty", END)

app = graph.compile()

inputs = [
    "我的页面报错了，显示 500 Internal Server Error",
    "Python 的装饰器是什么？能举个例子吗？",
    "请帮我重置密码",
    "",
    "今天天气不错"
]

for inp in inputs:
    result = app.invoke({
        "raw_input": inp, "stage1_output": "", "stage2_output": "",
        "stage3_output": "", "final_result": "", "decision_path": []
    })
    print(f"\n输入: '{inp}'")
    print(f"  → {result['final_result']}")
    for step in result["decision_path"]:
        print(f"    {step}")
```

这个客服消息分类器展示了三层决策链路：第一层做文本规范化并检查是否为空输入；第二层做意图分类（问题报告/提问/请求/其他）；第三层根据分类结果分发到对应的处理逻辑。`decision_path` 字段记录了完整的决策路径，这在生产环境中对于追踪"这条消息为什么被分到了这个处理队列"非常有价值。

## 避免"路由爆炸"

当你面对一个有很多维度、每个维度又有很多取值的决策空间时，如果不加控制地使用条件边，很容易出现"路由爆炸"的问题——比如 3 个维度各有 4 种取值，理论上就有 4×4×4=64 种组合，如果你试图为每种组合创建一个独立的分支，图的结构会变得极其复杂且难以维护。

解决这个问题的核心思路是**优先级排序 + 短路求值**——不要试图穷举所有组合，而是按照优先级从高到低依次检查各个维度，一旦某个维度的条件触发了明确的决策就直接返回，不再检查后续维度。这其实就是前面链式路由模式的本质：每一层只关注最重要的那个维度，把复杂的联合决策拆解成一系列简单的单一维度决策。

```python
class SmartRouterState(TypedDict):
    user_tier: str
    request_type: str
    data_sensitivity: str
    time_of_day: str
    region: str
    routing_target: str
    reason: str
    trace: Annotated[list[str], operator.add]

def smart_router(state: SmartRouterState) -> str:
    tier = state.get("user_tier", "normal")
    req_type = state.get("request_type", "general")
    sensitivity = state.get("data_sensitivity", "public")
    region = state.get("region", "domestic")

    trace = []

    if sensitivity == "top_secret":
        trace.append("规则1: 绝密数据 → 安全审核通道")
        return "security_review"
    if tier == "enterprise":
        if req_type == "api_abuse":
            trace.append("规则2: 企业用户API滥用 → 高优通道")
            return "priority_handling"
        trace.append("规则3: 企业用户 → VIP通道")
        return "vip_channel"
    if tier == "trial":
        if req_type in ["bulk_export", "full_access"]:
            trace.append("规则4: 试用用户越权操作 → 拒绝")
            return "access_denied"
        trace.append("规则5: 试用用户 → 标准通道(受限)")
        return "limited_channel"
    if region == "international" and sensitivity == "confidential":
        trace.append("规则6: 跨境敏感数据 → 合规审查")
        return "compliance_check"
    if req_type == "emergency":
        trace.append("规则7: 紧急请求 → 快速响应")
        return "emergency_channel"

    trace.append("默认: 标准通道")
    return "standard_channel"

def update_routing(state: SmartRouterState) -> dict:
    target = smart_router(state)
    return {"routing_target": target, "trace": [f"路由目标: {target}"]}

graph = StateGraph(SmartRouterState)
graph.add_node("route", update_routing)
graph.add_edge(START, "route")
graph.add_edge("route", END)

app = graph.compile()

test_cases = [
    {"user_tier": "enterprise", "request_type": "general_query",
     "data_sensitivity": "public", "time_of_day": "morning", "region": "domestic"},
    {"user_tier": "trial", "request_type": "bulk_export",
     "data_sensitivity": "public", "time_of_day": "afternoon", "region": "domestic"},
    {"user_tier": "normal", "request_type": "emergency",
     "data_sensitivity": "public", "time_of_day": "night", "region": "international"},
    {"user_tier": "normal", "request_type": "general_query",
     "data_sensitivity": "top_secret", "time_of_day": "morning", "region": "domestic"},
]

for case in test_cases:
    result = app.invoke({
        **case, "routing_target": "", "reason": "", "trace": []
    })
    print(f"层级={case['user_tier']:12} | 类型={case['request_type']:15} | "
          f"敏感度={case['data_sensitivity']:12} → {result['routing_target']}")
    for t in result["trace"]:
        print(f"      {t}")
```

这个智能路由器的关键在于它没有尝试穷举所有 3×5×4×4×2=240 种可能的输入组合，而是定义了一套有明确优先级的规则链：绝密数据无论其他条件如何一律走安全审核通道（最高优先级）；企业用户的 API 滥用行为走高优通道；试用用户的越权操作直接拒绝……一共只有 7 条规则加上一条默认规则就能覆盖所有场景。这种基于优先级的短路求值策略不仅让路由函数保持简洁，也让后续的规则变更变得非常容易——新增一条规则只需要在合适的位置插入一个新的 if 分支即可。

## 条件边与循环的结合

多层路由还有一个常见的应用场景是在循环体内部做条件退出判断。想象一下一个自动修复流程：不断尝试修复代码问题，每次修复后重新检查，直到所有问题都被修复或者达到最大尝试次数。这里的"重新检查后决定继续还是退出"就是一个典型的循环+条件边的组合：

```python
class AutoFixState(TypedDict):
    source_code: str
    issues_found: list[str]
    fix_attempts: Annotated[int, operator.add]
    max_attempts: int
    all_fixed: bool
    fix_summary: str
    action_log: Annotated[list[str], operator.add]

def detect_issues(state: AutoFixState) -> dict:
    code = state["source_code"]
    issues = []
    if "TODO" in code or "FIXME" in code:
        issues.append("存在 TODO/FIXME 标记")
    if "print(" in code and "debug" in code.lower():
        issues.append("包含调试打印语句")
    if "except:" in code and "Exception" not in code:
        issues.append("使用了裸 except 子句")
    if len(code.split('\n')) > 100 and "def " not in code:
        issues.append("文件过长且无函数划分")
    return {
        "issues_found": issues,
        "all_fixed": len(issues) == 0,
        "action_log": [f"[检测] 发现 {len(issues)} 个问题: {issues}"]
    }

def should_continue_fixing(state: AutoFixState) -> str:
    if state["all_fixed"]:
        return "done"
    if state["fix_attempts"] >= state["max_attempts"]:
        return "exhausted"
    return "fix"

def apply_fix(state: AutoFixState) -> dict:
    code = state["source_code"]
    issues = state["issues_found"]
    attempt = state["fix_attempts"] + 1

    fixed_code = code
    applied = []
    for issue in issues:
        if "TODO" in issue or "FIXME" in issue:
            fixed_code = fixed_code.replace("TODO", "# RESOLVED").replace("FIXME", "# RESOLVED")
            applied.append("标记TODO/FIXME为已解决")
        elif "调试打印" in issue:
            lines = fixed_code.split('\n')
            fixed_code = '\n'.join(l for l in lines if 'debug' not in l.lower() or 'print(' not in l)
            applied.append("移除调试打印语句")
        elif "裸 except" in issue:
            fixed_code = fixed_code.replace("except:", "except Exception:")
            applied.append("补全异常类型")
        else:
            applied.append(f"(模拟修复): {issue}")

    remaining = len(state["issues_found"]) - len(applied)
    return {
        "source_code": fixed_code,
        "fix_attempts": 1,
        "all_fixed": remaining == 0,
        "action_log": [
            f"[修复 第{attempt}次] 应用了{len(applied)}个修复: {applied}",
            f"         剩余约 {max(0, remaining)} 个问题"
        ]
    }

def done_handler(state: AutoFixState) -> dict:
    attempts = state["fix_attempts"]
    msg = f"✅ 所有问题已修复 (共{attempts}轮)" if attempts > 0 else "✅ 未发现问题"
    return {"fix_summary": msg, "action_log": [msg]}

def exhausted_handler(state: AutoFixState) -> dict:
    remaining = state["issues_found"]
    attempts = state["fix_attempts"]
    msg = f"⚠️ 达到最大尝试次数({attempts}次)，仍有{len(remaining)}个未解决问题"
    return {"fix_summary": msg, "action_log": [msg]}

fix_graph = StateGraph(AutoFixState)
fix_graph.add_node("detect", detect_issues)
fix_graph.add_node("apply_fix", apply_fix)
fix_graph.add_node("done", done_handler)
fix_graph.add_node("exhausted", exhausted_handler)

fix_graph.add_edge(START, "detect")
fix_graph.add_conditional_edges("detect", should_continue_fixing, {
    "done": "done",
    "exhausted": "exhausted",
    "fix": "apply_fix"
})
fix_graph.add_edge("apply_fix", "detect")  # 循环回到检测
fix_graph.add_edge("done", END)
fix_graph.add_edge("exhausted", END)

app = fix_graph.compile()

result = app.invoke({
    "source_code": """def process_data():
    print(debug_info)
    # TODO: add error handling
    try:
        result = fetch()
    except:
        pass
""",
    "issues_found": [], "fix_attempts": 0, "max_attempts": 5,
    "all_fixed": False, "fix_summary": "", "action_log": []
})

print(result["fix_summary"])
for entry in result["action_log"]:
    print(f"  {entry}")
```

这段程序描述了一个自动修复循环的工作原理。执行路径是：`START → detect → (条件判断) → apply_fix → detect → (条件判断) → apply_fix → detect → ... → done/exhausted → END`。关键的拓扑特征是 `apply_fix` 通过普通边回到了 `detect`，形成了一个循环结构；而 `detect` 通过条件边在三条路径之间选择：如果所有问题都修好了就走 `"done"`，如果达到最大次数了就走 `"exhausted"`，否则继续走 `"apply_fix"` 再修一轮。由于 `fix_attempts` 使用了 `Annotated[int, operator.add]`，每经过一次 `apply_fix` 自动加 1，计数逻辑完全由框架管理。
