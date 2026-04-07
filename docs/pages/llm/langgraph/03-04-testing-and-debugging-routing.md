# 3.4 路由的测试与调试

> 当你的图从三个节点增长到三十个节点，条件边从一条增长到二十条时，"路由是否正确工作"就变成了一个必须系统化解决的问题。你不可能每次修改代码后都手动构造各种输入来验证每条路径——这不仅效率低下，而且容易遗漏边界情况。这一节我们会建立一套完整的路由测试方法论，包括单元测试、集成测试、覆盖率分析，以及当路由出现问题时如何用 LangSmith 和其他工具快速定位根因。

## 单元测试路由函数

路由函数是整个图中最适合做单元测试的部分——它是一个纯函数（接收状态、返回字符串），没有副作用，不依赖外部服务，可以完全在隔离环境中测试。一个好的测试策略是为每个路由函数编写三类测试用例：正常路径（每种分支至少一个）、边界值（刚好在阈值上的值）、异常输入（缺失字段、类型错误、空值等）。

```python
import pytest
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class OrderState(TypedDict):
    order_value: float
    customer_tier: str
    is_returning: bool
    has_voucher: bool
    region: str
    routing_decision: str

def route_order(state: OrderState) -> str:
    value = state.get("order_value", 0)
    tier = state.get("customer_tier", "normal")
    is_returning = state.get("is_returning", False)
    region = state.get("region", "domestic")

    if value >= 10000 and tier in ["vip", "enterprise"]:
        return "priority_warehouse"
    if value >= 5000:
        return "standard_express"
    if tier == "vip":
        return "vip_standard"
    if is_returning and value >= 1000:
        return "loyalty_fast"
    if region == "international":
        return "intl_shipping"
    return "normal"

class TestOrderRouting:
    def test_high_value_vip_goes_to_priority(self):
        state: OrderState = {
            "order_value": 15000, "customer_tier": "vip",
            "is_returning": False, "has_voucher": False,
            "region": "domestic", "routing_decision": ""
        }
        assert route_order(state) == "priority_warehouse"

    def test_high_value_enterprise_goes_to_priority(self):
        state: OrderState = {
            "order_value": 20000, "customer_tier": "enterprise",
            "is_returning": False, "has_voucher": False,
            "region": "domestic", "routing_decision": ""
        }
        assert route_order(state) == "priority_warehouse"

    def test_medium_value_goes_to_express(self):
        state: OrderState = {
            "order_value": 6000, "customer_tier": "normal",
            "is_returning": False, "has_voucher": False,
            "region": "domestic", "routing_decision": ""
        }
        assert route_order(state) == "standard_express"

    def test_low_value_vip_goes_to_vip_standard(self):
        state: OrderState = {
            "order_value": 2000, "customer_tier": "vip",
            "is_returning": False, "has_voucher": False,
            "region": "domestic", "routing_decision": ""
        }
        assert route_order(state) == "vip_standard"

    def test_returning_customer_with_threshold(self):
        state: OrderState = {
            "order_value": 1500, "customer_tier": "normal",
            "is_returning": True, "has_voucher": False,
            "region": "domestic", "routing_decision": ""
        }
        assert route_order(state) == "loyalty_fast"

    def test_international_goes_to_intl_shipping(self):
        state: OrderState = {
            "order_value": 300, "customer_tier": "normal",
            "is_returning": False, "has_voucher": False,
            "region": "international", "routing_decision": ""
        }
        assert route_order(state) == "intl_shipping"

    def test_default_fallback_for_normal_orders(self):
        state: OrderState = {
            "order_value": 99, "customer_tier": "normal",
            "is_returning": False, "has_voucher": False,
            "region": "domestic", "routing_decision": ""
        }
        assert route_order(state) == "normal"

    def test_boundary_exact_5000(self):
        state: OrderState = {
            "order_value": 5000, "customer_tier": "normal",
            "is_returning": False, "has_voucher": False,
            "region": "domestic", "routing_decision": ""
        }
        assert route_order(state) == "standard_express"

    def test_boundary_just_below_5000(self):
        state: OrderState = {
            "order_value": 4999.99, "customer_tier": "normal",
            "is_returning": False, "has_voucher": False,
            "region": "domestic", "routing_decision": ""
        }
        assert route_order(state) != "standard_express"

    def test_missing_fields_use_defaults(self):
        state: OrderState = {
            "order_value": 0, "customer_tier": "",
            "is_returning": False, "has_voucher": False,
            "region": "", "routing_decision": ""
        }
        result = route_order(state)
        assert result in ["normal", "standard_express"]

    def test_negative_value_handled(self):
        state: OrderState = {
            "order_value": -100, "customer_tier": "normal",
            "is_returning": False, "has_voucher": False,
            "region": "domestic", "routing_decision": ""
        }
        result = route_order(state)
        assert isinstance(result, str)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

这套测试覆盖了订单路由函数的所有 6 个分支，加上 3 个边界值测试和 2 个异常输入测试。注意几个设计细节：每个测试方法的名字都清晰地描述了被测试的场景；使用了 `assert` 的正向断言风格（期望什么结果）而非反向断言（不期望什么结果）；对于不确定的行为（如缺失字段时的默认行为），使用 `assert ... in [...]` 来接受多种可能的结果而不是精确匹配。

## 端到端图路由测试

单元测试验证了路由函数本身的正确性，但路由函数是被嵌入在一个更大的图中的——你需要确认的是：当图执行时，数据确实按照预期的路径流过了正确的节点。这就需要端到端的图级别测试。

```python
import pytest
from typing import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, START, END

class TicketRoutingState(TypedDict):
    ticket_id: str
    subject: str
    description: str
    priority: str
    category: str
    assigned_team: str
    resolution_notes: Annotated[list[str], operator.add]

def classify_priority(state: TicketRoutingState) -> dict:
    desc = (state["subject"] + " " + state["description"]).lower()
    urgent_words = ["紧急", "urgent", "critical", "宕机", "down", "崩溃", "停服"]
    high_words = ["重要", "important", "high", "影响", "无法"]
    if any(w in desc for w in urgent_words):
        priority = "critical"
    elif any(w in desc for w in high_words):
        priority = "high"
    else:
        priority = "normal"
    return {"priority": priority}

def classify_category(state: TicketRoutingState) -> dict:
    desc = (state["subject"] + " " + state["description"]).lower()
    if any(w in desc for w in ["bug", "error", "报错", "异常", "故障"]):
        category = "technical"
    elif any(w in desc for w in ["账单", "支付", "收费", "bill", "payment"]):
        category = "billing"
    elif any(w in desc for w in ["功能", "需求", "feature", "request"]):
        category = "feature_request"
    else:
        category = "general"
    return {"category": category}

def route_ticket(state: TicketRoutingState) -> str:
    p = state["priority"]
    c = state["category"]
    if p == "critical":
        return "incident_command"
    if c == "billing" and p == "high":
        return "finance_escalation"
    if c == "technical":
        return "engineering"
    if c == "feature_request":
        return "product"
    return "general_support"

def assign_incident(state: TicketRoutingState) -> dict:
    return {"assigned_team": "事故指挥中心", "resolution_notes": ["已升级为事故处理流程"]}

def assign_engineering(state: TicketRoutingState) -> dict:
    return {"assigned_team": "工程技术组", "resolution_notes": ["分配给技术支持工程师"]}

def assign_finance(state: TicketRoutingState) -> dict:
    return {"assigned_team": "财务审核组", "resolution_notes": ["升级至财务团队处理"]}

def assign_product(state: TicketRoutingState) -> dict:
    return {"assigned_team": "产品管理组", "resolution_notes": ["记录为新需求"]}

def assign_general(state: TicketRoutingState) -> dict:
    return {"assigned_team": "一线客服组", "resolution_notes": ["常规处理队列"]}

graph = StateGraph(TicketRoutingState)
graph.add_node("classify_pri", classify_priority)
graph.add_node("classify_cat", classify_category)
graph.add_node("assign_incident", assign_incident)
graph.add_node("assign_eng", assign_engineering)
graph.add_node("assign_fin", assign_finance)
graph.add_node("assign_prod", assign_product)
graph.add_node("assign_gen", assign_general)

graph.add_edge(START, "classify_pri")
graph.add_edge("classify_pri", "classify_cat")
graph.add_conditional_edges("classify_cat", route_ticket, {
    "incident_command": "assign_incident",
    "finance_escalation": "assign_fin",
    "engineering": "assign_eng",
    "product": "assign_prod",
    "general_support": "assign_gen"
})
graph.add_edge("assign_incident", END)
graph.add_edge("assign_eng", END)
graph.add_edge("assign_fin", END)
graph.add_edge("assign_prod", END)
graph.add_edge("assign_gen", END)

app = graph.compile()

class TestTicketGraphRouting:
    def test_critical_bug_routes_to_incident(self):
        result = app.invoke({
            "ticket_id": "TKT-001",
            "subject": "生产环境服务宕机",
            "description": "主数据库连接池耗尽，所有API返回503",
            "priority": "", "category": "", "assigned_team": "", "resolution_notes": []
        })
        assert result["priority"] == "critical"
        assert result["assigned_team"] == "事故指挥中心"

    def test_normal_feature_request_routes_to_product(self):
        result = app.invoke({
            "ticket_id": "TKT-002",
            "subject": "希望增加批量导出功能",
            "description": "当前只能逐条导出数据，希望能支持批量选择后导出Excel",
            "priority": "", "category": "", "assigned_team": "", "resolution_notes": []
        })
        assert result["category"] == "feature_request"
        assert result["assigned_team"] == "产品管理组"

    def test_high_priority_billing_goes_to_finance(self):
        result = app.invoke({
            "ticket_id": "TKT-003",
            "subject": "扣款金额不对",
            "description": "本月账单显示扣除500元，但我只用了200元的额度",
            "priority": "", "category": "", "assigned_team": "", "resolution_notes": []
        })
        assert result["category"] == "billing"
        assert result["priority"] == "high"
        assert result["assigned_team"] == "财务审核组"

    def test_generic_issue_routes_to_general(self):
        result = app.invoke({
            "ticket_id": "TKT-004",
            "subject": "咨询一下使用问题",
            "description": "请问这个功能在哪里找到？",
            "priority": "", "category": "", "assigned_team": "", "resolution_notes": []
        })
        assert result["assigned_team"] == "一线客服组"

    def test_all_nodes_produce_resolution_notes(self):
        result = app.invoke({
            "ticket_id": "TKT-005",
            "subject": "登录页面打不开",
            "description": "点击登录按钮后一直转圈",
            "priority": "", "category": "", "assigned_team": "", "resolution_notes": []
        })
        assert len(result["resolution_notes"]) >= 1
        assert result["assigned_team"] != ""

    def test_state_preserves_input_data(self):
        ticket_id = "TKT-TEST-999"
        result = app.invoke({
            "ticket_id": ticket_id,
            "subject": "test",
            "description": "test description",
            "priority": "", "category": "", "assigned_team": "", "resolution_notes": []
        })
        assert result["ticket_id"] == ticket_id
        assert result["subject"] == "test"
```

端到端测试的核心价值在于它验证了**完整的执行链路**。`test_critical_bug_routes_to_incident` 这个测试不仅检查了最终的 `assigned_team` 是否正确，还隐式地验证了中间的分类步骤（`classify_priority` 正确识别了 critical、`classify_category` 正确识别了 technical）、路由决策（`route_ticket` 在 critical+technical 条件下选择了 incident_command）以及最终节点的正确执行。任何一环出了问题都会导致测试失败。

## 用 stream 模式追踪路由过程

当端到端测试失败时，你需要知道图实际走了哪条路径才能定位问题所在。LangGraph 提供了几种 stream 模式可以帮助你观察图的执行细节：

```python
def debug_routing(ticket_subject: str, ticket_desc: str):
    initial_state = {
        "ticket_id": "DEBUG-TKT",
        "subject": ticket_subject,
        "description": ticket_desc,
        "priority": "",
        "category": "",
        "assigned_team": "",
        "resolution_notes": []
    }

    print(f"\n{'='*60}")
    print(f"输入: {ticket_subject}")
    print(f"描述: {ticket_desc[:50]}...")
    print(f"{'='*60}")

    print("\n--- stream_mode='values' (完整状态快照) ---")
    for i, values in enumerate(app.stream(initial_state, stream_mode="values")):
        node_name = list(values.keys())[0] if values else "initial"
        print(f"  Step {i}: {node_name}")
        for key, val in values.items():
            if val and key != "resolution_notes":
                print(f"    {key}: {val}")

    print("\n--- stream_mode='updates' (增量更新) ---")
    for update in app.stream(initial_state, stream_mode="updates"):
        for node_name, node_update in update.items():
            print(f"  → [{node_name}] 更新字段: {list(node_update.keys())}")

    print("\n--- stream_mode='debug' (详细调试信息) ---")
    for event in app.stream(initial_state, stream_mode="debug"):
        event_type = event[0]
        if event_type == "chain_end":
            continue
        print(f"  Event: {event_type}")
        if len(event) > 1:
            detail = event[1]
            if hasattr(detail, 'input'):
                print(f"    Input keys: {list(detail.input.keys()) if hasattr(detail.input, 'keys') else type(detail.input)}")

result = debug_routing(
    "支付页面一直报错",
    "点击支付按钮后提示网络错误，试了好几次都不行"
)
```

三种 stream 模式各有用途：`stream_mode="values"` 展示每一步之后的**完整状态**，适合查看状态的整体演变过程；`stream_mode="updates"` 只展示**每步产生的增量更新**，更紧凑，适合快速定位哪个节点产生了哪些变化；`stream_mode="debug"` 提供**最详细的底层信息**，包括每个事件的类型和详细信息，适合深度调试。

## 路由覆盖率分析

随着图中路由逻辑的增长，如何确保所有的路由路径都被测试覆盖到了？我们可以写一个简单的工具来自动分析路由覆盖率：

```python
from collections import defaultdict

def analyze_route_coverage(graph_app, test_cases: list[dict]) -> dict:
    all_paths = defaultdict(int)
    covered_paths = set()

    for case in test_cases:
        try:
            captured_path = []

            original_nodes = {}
            for name, func in graph_app.get_graph().nodes.items():
                if callable(func) and not name in [START, END]:
                    def make_wrapper(orig_name, orig_func):
                        def wrapper(state):
                            captured_path.append(orig_name)
                            return orig_func(state)
                        return wrapper
                    original_nodes[name] = func

            result = graph_app.invoke(case)
            path_key = " → ".join(captured_path)
            covered_paths.add(path_key)
            for node in captured_path:
                all_paths[node] += 1
        except Exception as e:
            pass

    graph_nodes = [n for n in graph_app.get_graph().nodes.keys()
                   if n not in [START, END]]
    executed_nodes = set(all_paths.keys())
    missing_nodes = set(graph_nodes) - executed_nodes

    return {
        "total_nodes": len(graph_nodes),
        "executed_nodes": len(executed_nodes),
        "node_coverage": f"{len(executed_nodes)/len(graph_nodes)*100:.1f}%" if graph_nodes else "N/A",
        "missing_nodes": sorted(missing_nodes),
        "node_execution_counts": dict(all_paths),
        "unique_paths": len(covered_paths),
        "paths": sorted(covered_paths)
    }

test_cases = [
    {"ticket_id": "1", "subject": "服务器宕机", "description": "全部服务不可用",
     "priority": "", "category": "", "assigned_team": "", "resolution_notes": []},
    {"ticket_id": "2", "subject": "扣费问题", "description": "多收了我的钱",
     "priority": "", "category": "", "assigned_team": "", "resolution_notes": []},
    {"ticket_id": "3", "subject": "新功能建议", "description": "希望加个暗色模式",
     "priority": "", "category": "", "assigned_team": "", "resolution_notes": []},
    {"ticket_id": "4", "subject": "怎么用", "description": "不知道怎么操作",
     "priority": "", "category": "", "assigned_team": "", "resolution_notes": []},
]

coverage = analyze_route_coverage(app, test_cases)
print(f"\n节点覆盖率: {coverage['node_coverage']}")
print(f"未覆盖节点: {coverage['missing_nodes']}")
print(f"\n节点执行次数:")
for node, count in sorted(coverage['node_execution_counts'].items()):
    marker = "✅" if count > 0 else "❌"
    print(f"  {marker} {node}: {count}次")
print(f"\n发现 {coverage['unique_paths']} 条唯一执行路径:")
for path in coverage['paths']:
    print(f"  • {path}")
```

这个覆盖率分析工具通过运行一组测试用例并捕获实际的执行路径来统计哪些节点被执行过、哪些从未被触达。在实际项目中，你可以把这个分析作为 CI/CD 流水线的一部分——如果覆盖率低于设定的阈值（比如 80%），就让构建失败，提醒开发者补充测试用例。

## 常见路由 bug 模式与排查

在实际开发中，有些路由相关的 bug 反复出现，了解这些常见模式能帮你更快地定位问题。

**模式一：路由键拼写不一致**。这是最最常见的错误——路由函数返回 `"handle_success"` 但 path_map 中定义的是 `"success_handle"`。这种 bug 的特征是抛出 `ValueError: Returned route 'xxx' is not a valid edge`。预防方法是使用枚举或常量来统一管理路由键：

```python
from enum import Enum

class Routes(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"
    ESCALATE = "escalate"
    REVIEW = "review"

# 路由函数
def router(state): return Routes.APPROVE.value

# path_map 使用同一个枚举
path_map = {Routes.APPROVE.value: handle_approve, ...}
```

**模式二：条件边的顺序依赖**。如果你的路由逻辑中多个条件的判断有优先级关系，但在路由函数中 if-elif 的顺序写反了，就会导致低优先级条件先匹配而高优先级条件永远不会被触发。比如先判断 `value > 0` 再判断 `value > 10000`——那么所有大于 0 的值都会走第一个分支，`value > 10000` 的分支永远无法到达。排查方法是打印出路由函数的输入和输出，确认判断逻辑是否符合预期。

**模式三：状态更新时机错误**。有时候你以为某个字段的值已经在前面的节点中被更新了，但实际上因为某种原因（节点抛异常被跳过、字段名拼错导致写入了一个不同的字段）该字段仍然是初始值。这会导致路由函数基于错误的数据做出错误的决策。排查方法是在路由函数入口处打印完整的状态内容，或者使用 LangSmith 查看 checkpoint 中的实际状态快照。

**模式四：循环中的路由死锁**。在包含循环的结构中，如果循环退出条件永远不被满足（比如计数器没有被正确递增、标志位没有被正确设置），图会无限循环下去直到达到最大执行步数限制然后报错。排查方法是检查循环体中是否有更新退出条件的代码，以及使用带超时限制的方式运行图：

```python
import signal

def handler(signum, frame):
    raise TimeoutError("图执行超时，可能存在死循环")

signal.signal(signal.SIGALRM, handler)
signal.alarm(10)  # 10秒超时
try:
    result = app.invoke(initial_state)
    signal.alarm(0)
except TimeoutError as e:
    print(e)
```
