# 9.2 核心模块实现：状态定义与图拓扑

> 在上一节中我们完成了项目的需求分析和架构设计。这一节要把设计落地——定义完整的状态类型（State）、实现所有节点函数、组装成可执行的图，并验证核心流程的正确性。这是整个项目中最关键的一步，因为状态设计和图拓扑一旦确定下来，后续的 API 层和前端层都只是"套壳"，核心逻辑全部在这里。

## 状态设计：分层的状态模型

根据第2章学到的状态设计最佳实践，我们采用**三层状态模型**来组织工单系统的状态：

```python
from typing import TypedDict, Annotated, Literal, Optional
import operator
from datetime import datetime

class TicketInputState(TypedDict):
    """第一层：输入/输出层 — 与外部系统交互的数据"""
    ticket_id: str
    channel: Literal["web", "mobile", "api", "email"]
    raw_message: str
    sender_info: dict  # {user_id, name, tier, session_id}
    attachments: list[dict]
    final_reply: str

class TaskContextState(TypedDict):
    """第二层：任务上下文层 — 工单处理过程中的中间数据"""
    intent_category: str       # technical / billing / complaint / suggestion / other
    intent_confidence: float     # 分类置信度 0-1
    urgency_level: Literal["critical", "high", "medium", "low"]
    priority_score: int          # 综合优先级分值 0-100
    
    kb_match_result: Optional[dict]   # 知识库匹配结果
    auto_resolved: bool            # 是否已自动解决
    
    assigned_agent_id: Optional[str] # 被分配的客服ID
    assigned_team: Optional[str]    # 被分配的团队
    
    resolution_steps: list[dict]      # 处理步骤记录
    current_step: int               # 当前步骤索引
    
    execution_log: Annotated[list[str], operator.add]  # 处理日志
    
    human_decision: Optional[str]   # 人工决策内容
    decision_type: Optional[str]     # approve / reject / escalate / request_info
    interrupt_point: Optional[str]    # 当前暂停点

class ConversationState(TypedDict):
    """第三层：对话上下文层 — 对话历史和长期数据"""
    conversation_history: list[dict]  # 完整对话历史
    user_satisfaction: Optional[int]  # 满意度评分 1-5
    follow_up_status: Literal["pending", "sent", "confirmed", "closed"]
    follow_up_sent_at: Optional[str]

class TicketState(TicketInputState, TaskContextState, ConversationState):
    """合并后的完整工单状态"""
    pass
```

这个三层状态设计的意图是：
- **Layer 1 (I/O)**：只在与外部世界交互时读写（接收消息、发送回复）
- **Layer 2 (Task Context)**：图执行过程中的所有中间数据，每一步都往这里写
- **Layer 3 (Conversation)**：跨会话的持久化数据（对话历史、满意度等）

## 核心节点函数实现

### 节点1：消息接收与预处理

```python
def receive_ticket(state: TicketState) -> dict:
    raw = state["raw_message"]
    channel = state.get("channel", "web")
    sender = state.get("sender_info", {})
    
    cleaned = raw.strip()
    if len(cleaned) < 2:
        return {
            "auto_resolved": True,
            "final_reply": "消息过短，无法处理。请提供更详细的问题描述。",
            "execution_log": [f"[接收] ❌ 消息过短 ({len(cleaned)} 字符)"],
            "current_step": -1,
            "intent_category": "invalid"
        }
    
    entity_patterns = {
        r"(?:订单号?|order[_\s]?[\w]+)": "order_id",
        r"(?:退款|退货|return)": "refund_intent",
        r"(?:密码|password|登录|login)": "auth_issue",
        r"(?:支付|付款|payment|扣费)": "payment_issue",
        r"(?:API|接口|报错|error|500|404)": "api_error",
        r"(?:投诉|举报|不满|差评)": "complaint",
        r"(?:建议|意见|反馈|feature)": "suggestion"
    }
    
    detected_entities = []
    for pattern, label in entity_patterns.items():
        matches = re.findall(pattern, cleaned, re.IGNORECASE)
        if matches:
            detected_entities.append({"type": label, "value": matches[0]})
    
    log_msg = f"[接收] 来自{channel}渠道 | 发送者:{sender.get('name','未知')} | 消息:{cleaned[:50]}..."
    if detected_entities:
        log_msg += f" | 实体:{', '.join(f'{e[\"type\"]}:{e[\"value\"]}' for e in detected_entities)}"
    
    return {
        "execution_log": [log_msg],
        "conversation_history": [{"role": "user", "content": raw, "timestamp": datetime.now().isoformat()}],
        "current_step": 0
    }

import re
```

### 节点2：智能分类

```python
def classify_intent(state: TicketState) -> dict:
    message = state["raw_message"].lower()
    sender_tier = state.get("sender_info", {}).get("tier", "normal")
    
    rule_based_category = None
    confidence = 0.8
    
    if any(w in message for w in ["订单", "order", "物流", "发货", "配送"]):
        rule_based_category = "order_inquiry"
        confidence = 0.95
    elif any(w in message for w in ["退款", "退货", "钱", "扣款", "charge"]):
        rule_based_category = "billing"
        confidence = 0.92
    elif any(w in message for w in ["密码", "登录", "账号", "无法访问"]):
        rule_based_category = "account_issue"
        confidence = 0.90
    elif any(w in message for w in ["投诉", "不满", "差评", "垃圾", "骗"]):
        rule_based_category = "complaint"
        confidence = 0.88
    elif any(w in message for w in ["建议", "功能", "希望", "能不能"]):
        rule_based_category = "suggestion"
        confidence = 0.85
    else:
        rule_based_category = "other"
        confidence = 0.3
    
    urgency = "medium"
    if any(w in message for w in ["紧急", "urgent", "马上", "立即", "崩溃", "挂了"]):
        urgency = "critical"
        confidence += 0.05
    elif any(w in message for w in ["尽快", "今天内", "加急"]):
        urgency = "high"
        confidence += 0.03
    
    priority_map = {"critical": 95, "high": 75, "medium": 50, "low": 25}
    priority_score = priority_map.get(urgency, 50)
    
    if sender_tier == "vip":
        priority_score += 15
    elif sender_tier == "enterprise":
        priority_score += 25
    
    priority_score = min(100, priority_score)
    
    log_entry = (
        f"[分类] 类别={rule_based_category} | "
        f"置信度={confidence:.0%} | "
        f"紧急度={urgency} | 优先分={priority_score}"
    )
    
    return {
        "intent_category": rule_based_category,
        "intent_confidence": confidence,
        "urgency_level": urgency,
        "priority_score": priority_score,
        "current_step": 1,
        "execution_log": [log_entry]
    }
```

### 节点3：知识库自动应答

```python
def try_auto_resolve(state: TicketState) -> dict:
    category = state["intent_category"]
    message = state["raw_message"]
    
    knowledge_base = {
        "order_inquiry": [
            ("查询订单", "您可以在「我的订单」页面输入订单号查询最新状态和物流信息。",
            ("修改地址", "订单未发货前可在订单详情页修改收货地址；已发货请联系客服。"),
            ("取消订单", "未发货订单可在「我的订单」中点击取消；已发货订单需联系客服处理。"),
            ("退换货", "支持7天无理由退换货。请在「我的订单」→「申请售后」提交申请。"),
        ],
        "billing": [
            ("查询账单", "账单查询路径：个人中心 → 我的账单 → 选择月份查看详情。"),
            ("开发票", "企业用户可在「财务管理 → 开票管理」中申请开票。"),
            ("退款进度", "退款一般3-5个工作日原路退回。"),
        ],
        "account_issue": [
            ("重置密码", "重置路径：登录页 → 忘记密码 → 验证手机/邮箱 → 设置新密码。"),
            ("账号冻结", "如非本人操作导致冻结，请通过「安全中心」申诉解冻。"),
        ],
        "other": []
    }
    
    faq_entries = knowledge_base.get(category, [])
    
    if not faq_entries:
        return {
            "auto_resolved": False,
            "kb_match_result": None,
            "execution_log": [f"[知识库] '{category}' 无匹配条目"]
        }
    
    best_match = None
    best_score = 0
    
    for title, answer in faq_entries:
        score = sum(1 for keyword in answer.split() if keyword in message.lower())
        if score > best_score:
            best_score = score
            best_match = (title, answer)
    
    if best_score >= 2:
        title, answer = best_match
        reply = f"📋 **{title}**\n\n{answer}\n\n以上信息是否解决了您的问题？如果还有其他疑问，请继续描述。"
        
        return {
            "auto_resolved": True,
            "kb_match_result": {"matched_title": title, "confidence": best_score},
            "final_reply": reply,
            "execution_log": [f"[知识库] ✅ 匹配到: {title} (关键词匹配度: {best_score})"]
        }
    
    return {
        "auto_resolved": False,
        "kb_match_result": {"matched_title": best_match[0] if best_match else None, "confidence": best_score} if best_match else None,
        "execution_log": [f"[知识库] ⚠️ 匹配度不足 (最高: {best_score})"]
    }
```

### 路由逻辑：分类后决定下一步

```python
def route_after_classify(state: TicketState) -> str:
    auto_resolved = state.get("auto_resolved", False)
    urgency = state.get("urgency_level", "medium")
    
    if auto_resolved:
        return "send_reply"
    
    if urgency == "critical":
        return "escalate_and_notify"
    
    if urgency == "high":
        return "assign_to_expert"
    
    return "assign_to_normal"

def route_after_resolve(state: TicketState) -> str:
    if state.get("auto_resolved"):
        return "send_reply"
    return "assign_to_normal"

def route_after_assign(state: TicketState) -> str:
    if state.get("human_decision"):
        decision = state.get("decision_type", "")
        if decision == "approve":
            return "execute_resolution"
        elif decision == "escalate":
            return "escalate_higher"
        elif decision == "request_more_info":
            return "collect_more_info"
        return "collect_more_info"
    
    if state.get("resolution_steps") and len(state["resolution_steps"]) > 0:
        last_step = state["resolution_steps"][-1]
        if last_step.get("status") == "completed":
            return "follow_up_check"
        return "execute_resolution"
    
    return "execute_resolution"

def route_after_execute(state: TicketState) -> str:
    steps = state.get("resolution_steps", [])
    if not steps or steps[-1].get("status") != "completed":
        return "follow_up_check"
    return "send_reply_with_followup"
```

### 其他关键节点

```python
def assign_to_normal(state: TicketState) -> dict:
    agents = ["客服-Alice", "客服-Bob", "客服-Carol"]
    import random
    agent = random.choice(agents)
    return {
        "assigned_agent_id": agent,
        "assigned_team": "一线客服组",
        "resolution_steps": [{
            "step": 1, "action": "分配给客服", "agent": agent,
            "status": "assigned", "timestamp": datetime.now().isoformat()
        }],
        "current_step": 2,
        "execution_log": [f"[分配] 已分配给 {agent} (一线客服组)"]
    }

def assign_to_expert(state: TicketState) -> dict:
    agents = ["技术专家-Tom", "高级专家-Diana"]
    agent = agents[0]
    return {
        "assigned_agent_id": agent,
        "assigned_team": "技术专家组",
        "resolution_steps": [{
            "step": 1, "action": "升级分配给技术专家", "agent": agent,
            "status": "assigned", "timestamp": datetime.now().isoformat()
        }],
        "current_step": 2,
        "execution_log": [f"[升级] 已升级分配给 {agent}"]
    }

def escalate_and_notify(state: TicketState) -> dict:
    managers = ["经理-Erica", "总监-Frank"]
    manager = managers[0]
    return {
        "assigned_agent_id": manager,
        "assigned_team": "管理层",
        "resolution_steps": [{
            "step": 1, "action": "加急通知管理层", "target": manager,
            "status": "notified", "timestamp": datetime.now().isoformat()
        }],
        "current_step": 2,
        "execution_log": [f"[加急] ⚠️ 已通知管理层: {manager}"]
    }

def collect_more_info(state: TicketState) -> dict:
    from langgraph.types import interrupt
    prompt = (
        f"需要更多信息来处理此工单:\n\n"
        f"原始问题: {state['raw_message'][:200]}...\n\n"
        f"当前分类: {state['intent_category']}\n"
        f"当前负责人: {state.get('assigned_agent_id', '未分配')}\n\n"
        f"请补充信息或确认操作:\n"
        f"- 输入 'continue' 继续当前处理\n"
        f"- 输入 'escalate' 升级处理\n"
        f"- 输入 'resolve' 直接关闭"
    )
    user_input = interrupt(prompt)
    
    parts = user_input.strip().split(maxsplit=1)
    decision = parts[0].lower()
    comment = parts[1] if len(parts) > 1 else ""
    
    valid = {"continue", "escalate", "resolve"}
    if decision not in valid:
        return {"execution_log": [f"[收集信息] ⚠️ 无效指令: {decision}"]}
    
    return {
        "human_decision": decision,
        "decision_type": decision,
        "execution_log": [f"[收集信息] 用户选择: {decision} {comment}"]
    }

def execute_resolution(state: TicketState) -> dict:
    agent = state.get("assigned_agent_id", "system")
    category = state["intent_category"]
    message = state["raw_message"]
    
    resolution_templates = {
        "order_inquiry": f"关于您的订单问题，我们已经核实了相关信息并做了处理。",
        "billing": f"关于您的账单/支付问题，财务部门已完成核查。",
        "account_issue": f"您的账户相关问题已经处理完成。",
        "complaint": f"非常抱歉给您带来不便。我们已记录您的反馈并转交相关部门处理。",
        "suggestion": f"感谢您的宝贵建议！我们会认真评估并考虑纳入产品改进计划。",
        "other": f"您的问题已收到，我们的团队正在处理中。"
    }
    
    template = resolution_templates.get(category, resolution_templates["other"])
    full_reply = f"{template}\n\n参考编号: {state['ticket_id']}。如有其他问题，随时联系我们！"
    
    step_record = {
        "step": len(state.get("resolution_steps", [])) + 1,
        "action": "执行解决方案",
        "agent": agent,
        "status": "completed",
        "result_summary": template[:100],
        "timestamp": datetime.now().isoformat()
    }
    
    steps = list(state.get("resolution_steps", []))
    steps.append(step_record)
    
    return {
        "final_reply": full_reply,
        "resolution_steps": steps,
        "current_step": len(steps) + 1,
        "execution_log": [f"[执行] {agent} 已完成解决方案"]
    }

def send_reply(state: TicketState) -> dict:
    return {
        "follow_up_status": "pending",
        "execution_log": [f"[回复] 回复已发送"]
    }

def send_reply_with_followup(state: TicketState) -> dict:
    return {
        "follow_up_status": "sent",
        "follow_up_sent_at": datetime.now().isoformat(),
        "execution_log": [f"[回复+回访] 回复已发送，待确认"]
    }

def follow_up_check(state: TicketState) -> dict:
    return {
        "follow_up_status": "confirmed",
        "execution_log": [f"[回访] 用户已确认，工单关闭"]
    }
```

## 组装主图

```python
from langgraph.graph import StateGraph, START, END

ticket_graph = StateGraph(TicketState)

# 注册所有节点
ticket_graph.add_node("receive", receive_ticket)
ticket_graph.add_node("classify", classify_intent)
ticket_graph.add_node("try_kb", try_auto_resolve)
ticket_graph.add_node("assign_normal", assign_to_normal)
ticket_graph.add_node("assign_expert", assign_to_expert)
ticket_graph.add_node("escalate", escalate_and_notify)
ticket_graph.add_node("collect_info", collect_more_info)
ticket_graph.add_node("execute", execute_resolution)
ticket_graph.add_node("send_reply", send_reply)
ticket_graph.add_node("send_reply_followup", send_reply_with_followup)
ticket_graph.add_node("follow_up", follow_up_check)

# 定义边的连接关系
ticket_graph.add_edge(START, "receive")
ticket_graph.add_edge("receive", "classify")

ticket_graph.add_conditional_edges("classify", route_after_classify, {
    "send_reply": "try_kb",
    "escalate_and_notify": "escalate",
    "assign_to_expert": "assign_expert",
    "assign_to_normal": "assign_normal"
})

ticket_graph.add_edge("try_kb", send_reply")

ticket_graph.add_conditional_edges("assign_normal", route_after_assign, {
    "execute": "execute",
    "collect_info": "collect_info"
})
ticket_graph.add_edge("assign_expert", "execute")

ticket_graph.add_edge("escalate", "collect_info")

ticket_graph.add_conditional_edges("collect_info", lambda s: 
    "execute" if (s.get("decision_type") == "continue") 
    else "escalate_higher" if (s.get("decision_type") == "escalate")
    else "resolve" if (s.get("decision_type") == "resolve")
    else "collect_info",
    {
        "execute": "execute",
        "escalate_higher": "escalate",
        "resolve": "send_reply"
    })

ticket_graph.add_conditional_edges("execute", route_after_execute, {
    "follow_up_check": "follow_up",
    "send_reply_followup": "send_reply_followup"
})

ticket_graph.add_edge("send_reply", END)
ticket_graph.add_edge("follow_up_check", END)
ticket_graph.add_edge("send_reply_followup", END)

# 编译图
app = ticket_graph.compile()

print("=" * 60)
print("智能客服工单系统 - 核心图编译成功")
print("=" * 60)

# 测试运行
test_tickets = [
    ("TKT-001", "web", "我的订单什么时候能送到？", {"name": "张三", "tier": "normal"}),
    ("TKT-002", "mobile", "我要退款，商品质量有问题", {"name": "李四", "tier": "vip"}),
    ("TKT-003", "web", "系统崩溃了！赶紧处理！", {"name": "王五", "tier": "enterprise"}),
    ("TKT-004", "api", "API返回500错误怎么解决？", {"name": "赵六", "tier": "normal"}),
    ("TKT-005", "web", "希望你们能增加暗色模式", {"name": "孙七", "tier": "normal"}),
]

for tid, channel, msg, sender in test_tickets:
    result = app.invoke({
        "ticket_id": tid,
        "channel": channel,
        "raw_message": msg,
        "sender_info": sender,
        "attachments": [],
        "final_reply": "",
        "intent_category": "", "intent_confidence": 0.0,
        "urgency_level": "medium", "priority_score": 0,
        "kb_match_result": None, "auto_resolved": False,
        "assigned_agent_id": "", "assigned_team": "",
        "resolution_steps": [], "current_step": 0,
        "execution_log": [], "human_decision": "", "decision_type": "",
        "interrupt_point": None,
        "conversation_history": [], "user_satisfaction": None,
        "follow_up_status": "pending", "follow_up_sent_at": None
    })
    
    print(f"\n{'─'*50}")
    print(f"工单 {tid}: [{channel}] {msg[:30]}...")
    print(f"  分类: {result['intent_category']} | "
          f"紧急度: {result['urgency_level']} | "
          f"自动解决: {'是' if result['auto_resolved'] else '否'}")
    for entry in result["execution_log"]:
        print(f"  {entry}")
    if result.get("final_reply"):
        print(f"  回复预览: {result['final_reply'][:60]}...")
```
