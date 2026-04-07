# 9.3 API 服务层与前端界面

> 在上一节中我们实现了智能客服工单系统的核心图逻辑——所有节点函数、状态定义、路由规则和图拓扑。但一个完整的应用不能只有后端逻辑，还需要 API 层来接收外部请求、前端界面让用户能直观地与系统交互。这一节我们会用 FastAPI 构建 RESTful API 服务层，并实现一个简洁的 Web 前端界面。

## FastAPI 服务层设计

```python
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
import uuid
import json

app_fastapi = FastAPI(
    title="智能客服工单系统 API",
    version="1.0.0",
    description="基于 LangGraph 的智能客服工单处理系统"
)

# 全局存储（生产环境应替换为数据库）
ticket_store: dict[str, dict] = {}
checkpointer_store: dict = {}  # 模拟 checkpoint 存储

class CreateTicketRequest(BaseModel):
    channel: str = Field(..., description="渠道: web/mobile/api/email")
    message: str = Field(..., description="用户消息内容")
    user_id: str = Field(..., description="用户ID")
    user_name: str = Field("匿名用户", description="用户名")
    user_tier: str = Field("normal", description="用户等级")

class TicketResponse(BaseModel):
    ticket_id: str
    status: str
    category: str
    urgency: str
    priority: int
    auto_resolved: bool
    reply: Optional[str]
    assigned_agent: Optional[str]
    requires_human: bool
    current_step: Optional[int]

class HumanActionRequest(BaseModel):
    ticket_id: str
    action: str = Field(..., description: continue/escalate/resolve/approve/reject)
    comment: Optional[str] = None

@app_fastapi.post("/api/tickets", response_model=TicketResponse, status_code=201)
async def create_ticket(req: CreateTicketRequest):
    ticket_id = f"TKT-{uuid.uuid4().hex[:8].upper()}"
    
    initial_state = {
        "ticket_id": ticket_id,
        "channel": req.channel,
        "raw_message": req.message,
        "sender_info": {"user_id": req.user_id, "name": req.user_name, "tier": req.user_tier},
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
    }
    
    result = app.invoke(initial_state)
    
    ticket_store[ticket_id] = {
        "created_at": datetime.now().isoformat(),
        "state": result,
        "config": {"thread_id": f"ticket-{ticket_id}"}
    }
    
    return TicketResponse(
        ticket_id=ticket_id,
        status="completed" if result.get("auto_resolved") else "processing",
        category=result.get("intent_category"),
        urgency=result.get("urgency_level"),
        priority=result.get("priority_score", 0),
        auto_resolved=result.get("auto_resolved", False),
        reply=result.get("final_reply") if result.get("auto_resolved") else None,
        assigned_agent=result.get("assigned_agent_id"),
        requires_human=True if not result.get("auto_resolved") and 
                       result.get("interrupt_point") else False,
        current_step=result.get("current_step")
    )

@app_fastapi.post("/api/tickets/{ticket_id}/action", response_model=TicketResponse)
async def submit_action(req: HumanActionRequest):
    ticket_id = ticket_id
    stored = ticket_store.get(ticket_id)
    
    if not stored:
        raise HTTPException(status_code=404, detail=f"工单 {ticket_id} 不存在")
    
    config = stored["config"]
    state = stored["state"]
    
    if req.action == "continue":
        resume_value = f"{req.action} {req.comment or ''}"
    elif req.action in ("approve", "reject", "escalate", "resolve"):
        resume_value = f"{req.action} {req.comment or ''}"
    else:
        raise HTTPException(status_code=400, detail=f"无效操作: {req.action}")
    
    updated = app.invoke(Command(resume=resume_value), config=config)
    stored["state"] = updated
    
    return TicketResponse(
        ticket_id=ticket_id,
        status="completed" if updated.get("follow_up_status") == "confirmed" else "processing",
        category=updated.get("intent_category"),
        urgency=updated.get("urgency_level"),
        priority=updated.get("priority_score", 0),
        auto_resolved=updated.get("auto_resolved", False),
        reply=updated.get("final_reply"),
        assigned_agent=updated.get("assigned_agent_id"),
        requires_human=False,
        current_step=updated.get("current_step")
    )

@app_fastapi.get("/api/tickets/{ticket_id}", response_model=TicketResponse)
async def get_ticket(ticket_id: str):
    stored = ticket_store.get(ticket_id)
    if not stored:
        raise HTTPException(status_code=404, detail=f"工单 {ticket_id} 不存在")
    
    state = stored["state"]
    return TicketResponse(
        ticket_id=ticket_id,
        status="completed" if state.get("follow_up_status") == "confirmed"
                else ("waiting_input" if state.get("interrupt_point") else "processing"),
        category=state.get("intent_category"),
        urgency=state.get("urgency_level"),
        priority=state.get("priority_score", 0),
        auto_resolved=state.get("auto_resolved", False),
        reply=state.get("final_reply"),
        assigned_agent=state.get("assigned_agent_id"),
        requires_human=True if state.get("interrupt_point") else False,
        current_step=state.get("current_step")
    )

@app_fastapi.get("/api/tickets", response_model=list)
async def list_tickets():
    return [
        {
            "ticket_id": tid,
            "status": data["state"].get("follow_up_status", "processing"),
            "category": data["state"].get("intent_category"),
            "urgency": data["state"].get("urgency_level"),
            "created_at": data["created_at"]
        }
        for tid, data in ticket_store.items()
    ]

@app_fastapi.get("/api/dashboard/stats")
async def dashboard_stats():
    total = len(ticket_store)
    by_status = {}
    by_category = {}
    by_urgency = {}
    avg_steps = []
    
    for tid, data in ticket_store.items():
        s = data["state"]
        by_status[s.get("follow_up_status", "unknown")] = by_status.get(s.get("follow_up_status", "unknown"), 0) + 1
        by_category[s.get("intent_category", "unknown")] = by_category.get(s.get("intent_category", "unknown"), 0) + 1
        by_urgency[s.get("urgency_level", "unknown")] = by_urgency.get(s.get("urgency_level", "unknown"), 0) + 1
        if s.get("current_step"):
            avg_steps.append(s["current_step"])
    
    return {
        "total_tickets": total,
        "by_status": by_status,
        "by_category": by_category,
        "by_urgency": by_urgency,
        "avg_processing_steps": sum(avg_steps) / len(avg_steps) if avg_steps else 0,
        "active_tickets": sum(1 for s in [d["state"] for d in ticket_list.values() 
                        if d.get("follow_up_status") in ("pending", "waiting_input"))
    }

from langgraph.types import Command
```

## 前端界面

下面是一个精简版的前端页面，展示核心的交互功能：

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>智能客服工单系统</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacFont, 'Segoe UI', sans-serif;
       background: #f5f7fa; color: #333; padding: 20px; }
.container { max-width: 900px; margin: 0 auto; }
.header { text-align: center; padding: 20px 0; border-bottom: 2px solid #4a90e2; }
.stats-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 20px 0; }
.stat-card { background: white; padding: 16px; border-radius: 8px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,.1); }
.stat-num { font-size: 24px; font-weight: bold; color: #4a90e2; }
.stat-label { font-size: 12px; color: #888; margin-top: 4px; }
.ticket-list { background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,.1); }
.ticket-item { padding: 16px 20px; border-bottom: 1px solid #eee; display: flex; align-items: center; cursor: pointer; transition: background .2s; }
.ticket-item:hover { background: #f8f9fa; }
.ticket-id { font-family: monospace; font-size: 13px; min-width: 100px; }
.ticket-meta { flex: 1; margin-left: 12px; }
.ticket-category { display: inline-block; padding: 2px 8px; border-radius: 4px; 
               font-size: 12px; font-weight: 600; }
.critical { background: #fee2e2; color: #c00; }
.high { background: #fff3cd; color: #856404; }
.medium { background: #ffeaa7; color: #856404; }
.low { background: #e8f5e9; color: #388e3c; }
.btn { padding: 10px 20px; border: none; border-radius: 6px; cursor: pointer; font-size: 14px; font-weight: 500; }
.btn-primary { background: #4a90e2; color: white; }
.btn-outline { background: white; color: #4a90e2; border: 1px solid #4a90e2; }
.form-area { width: 100%; min-height: 120px; padding: 12px; border: 1px solid #ddd; border-radius: 8px; resize: vertical; font-family: inherit; font-size: 14px; }
</style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>🎧 智能客服工单系统</h1>
        <p>基于 LangGraph 的多级路由 · 自动分类 · 知识库匹配 · 人机协作</p>
    </div>

    <!-- 统计仪表盘 -->
    <div id="stats-panel"></div>
    <div style="margin-top:20px;">
        <textarea id="msg-input" class="form-area" placeholder="描述您的问题..."></textarea>
        <div style="margin-top:12px; display:flex; gap:10px;">
            <button class="btn btn-primary" onclick="submitTicket()">提交工单</button>
            <button class="btn btn-outline" onclick="loadTickets()">刷新列表</button>
        </div>
    </div>

    <div id="tickets-panel" class="ticket-list"></div>
</div>

<script>
const API_BASE = "";

async function loadStats() {
    const resp = await fetch(`${API_BASE}/api/dashboard/stats`);
    const data = await resp.json();
    
    document.getElementById('stats-panel').innerHTML = `
        <div class="stats-grid">
            <div class="stat-card"><div class="stat-num">${data.total_tickets}</div><div class="stat-label">总工单</div></div>
            <div class="stat-card"><div class="stat-num">${data.by_status.get('confirmed', 0)}</div><div class="stat-label">已完成</div></div>
            <div class="stat-card"><div class="stat-num">${data.by_status.get('pending', 0) + data.by_status.get('waiting_input', 0)}</div><div class="stat-label">待处理</div></div>
            <div class="stat-card"><div class="stat-num">${data.by_urgency.get('critical', 0)}</div><div class="stat-label">紧急</div></div>
            <div class="stat-card"><div class="stat-num">${data.avg_processing_steps.toFixed(1)}</div><div class="stat-label">平均步骤</div></div>
        </div>
    `;
}

async function submitTicket() {
    const msg = document.getElementById('msg-input').value.trim();
    if (!msg) return alert("请输入问题内容");
    
    const resp = await fetch(`${API_BASE}/api/tickets`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            channel: "web",
            message: msg,
            user_id: "u-" + Date.now().toString(36),
            user_name: "访客用户",
            user_tier: "normal"
        })
    });
    
    const ticket = await resp.json();
    showTicketDetail(ticket);
    loadTickets();
    loadStats();
}

async function loadTickets() {
    const resp = await fetch(`${API_BASE}/api/tickets`);
    const tickets = await resp.json();
    
    document.getElementById('tickets-panel').innerHTML = tickets.length === 0 
        ? '<p style="text-align:center;color:#999;padding:40px;">暂无工单</p>'
        : tickets.map(t => `
            <div class="ticket-item" onclick="showTicketDetail(${JSON.stringify(t.ticket_id).replace(/"/g,'')})">
                <span class="ticket-id">${t.ticket_id}</span>
                <div class="ticket-meta">
                    <span class="ticket-category ${t.urgency}">${t.category || '...'}</span>
                    ${t.auto_resolved ? '<span style="color:#16a34a;font-weight:600;">✅ 自动解决</span>' : 
                     t.requires_human ? '<span style="color:#e67e22;font-weight:600;">⏳ 待人工</span>' :
                     '<span>处理中</span>'}
                    <span style="margin-left:auto;color:#888;font-size:12px;">${t.assigned_agent || '-'}</span>
                </div>
            </div>
        `;
}

let currentTicketId = null;

async function showTicketDetail(ticket) {
    if (typeof ticket === 'string') ticket = JSON.parse(ticket);
    currentTicketId = ticket.ticket_id;
    
    const resp = await fetch(`${API_BASE}/api/tickets/${ticket.ticket_id}`);
    const detail = await resp.json();
    
    const panel = document.getElementById('tickets-panel');
    panel.innerHTML = `
        <div style="padding:20px;background:white;border-radius:8px;margin-top:12px;box-shadow:0 2px 8px rgba(0,0,0,.1);">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;">
                <h3 style="margin:0">📋 工单 ${detail.ticket_id}</h3>
                <button class="btn btn-outline" onclick="loadTickets()" style="font-size:12px;padding:6px 12px;">← 返回列表</button>
            </div>
            
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px;">
                <div style="background:#f8f9fa;padding:12px;border-radius:6px;">
                    <strong>分类</strong><br>${detail.category || '-'}
                </div>
                <div style="background:#f8f9fa;padding:12px;border-radius:6px;">
                    <strong>紧急度</strong><br>${detail.urgency || '-'} | 优先分: ${detail.priority || '-'}
                </div>
            </div>
            
            <div style="background:#fef9e7;padding:16px;border-radius:6px;margin-bottom:16px;border-left:3px solid #f59e0b;">
                <strong>💬 原始消息</strong><br>
                <pre style="white-space:pre-wrap;font-size:13px;margin:8px 0;">${detail.reply || '(等待处理...)'}</pre>
            </div>
            
            ${detail.requires_human ? `
            <div style="margin-top:16px;">
                <strong>🔔 需要您操作</strong>
                <div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:12px;">
                    <button class="btn btn-primary" onclick="sendAction('continue')">继续处理</button>
                    <button class="btn btn-outline" onclick="sendAction('escalate')">升级处理</button>
                    <button class="btn btn-outline" onclick="sendAction('resolve')">直接关闭</button>
                </div>
                <input id="action-comment" placeholder="可选备注..." style="flex:1;padding:8px 12px;border:1px solid #ddd;border-radius:6px;">
            </div>` : ''}
            
            <div style="margin-top:12px;font-size:12px;color:#888;text-align:center;">
                步骤详情: ${detail.current_step || 0} | 
                负责人: ${detail.assigned_agent || '-'}
            </div>
        </div>
    `;
}

async function sendAction(action) {
    const comment = document.getElementById('action-comment').value;
    const resp = await fetch(`${API_BASE}/api/tickets/${currentTicketId}/action`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ticket_id: currentTicketId, action, comment})
    });
    
    const detail = await resp.json();
    showTicketDetail(detail);
    loadStats();
}
</script>
</body>
</html>
```

这个前端实现了以下关键功能：
1. **统计仪表盘**：实时显示总工单数、已完成数、待处理数、紧急工单数、平均处理步骤
2. **工单创建**：文本框输入 → 提交按钮 → 后端自动处理 → 展示结果或等待人工操作
3. **工单列表**：展示所有工单，点击可查看详情
4. **工单详情页**：显示完整的分类、消息、回复；如果需要人工操作则显示操作按钮
5. **操作按钮**：继续/升级/关闭，附带可选备注

## API 测试流程

启动服务后可以通过以下方式测试完整流程：

```bash
# 启动 API 服务
uvicorn main:app --host 0.0.0.0 --port 8000

# 测试1：提交普通咨询（预期：知识库自动回答）
curl -X POST http://localhost:8000/api/tickets \
  -H "Content-Type: application/json" \
  -d '{"channel":"web","message":"怎么重置密码？","user_id":"u-001","user_name":"测试用户","user_tier":"normal"}'

# 测试2：提交紧急问题（预期：升级路由+需要人工）
curl -X POST http://localhost:8000/api/tickets \
  - H "Content-Type: application/json" \
  -d '{"channel":"web","message":"系统崩溃了！赶紧处理！","user_id":"u-002","user_name":"VIP用户","user_tier":"vip"}'

# 测试3：查看工单列表
curl http://localhost:8000/api/tickets

# 测试4：对需人工操作的工单执行操作
curl -X POST http://localhost:8000/api/tickets/TKT-XXXX/action \
  - H "Content-Type: application/json" \
  -d '{"ticket_id":"TKT-XXXX","action":"resolve","comment":"已解决"}'
```
