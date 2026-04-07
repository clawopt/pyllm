# 4.4 构建交互式应用与 Interrupt 集成

> 前面几节我们一直在用命令行的方式模拟人工输入——通过 `Command(resume="some input")` 来恢复被 Interrupt 暂停的执行。但在真实的应用中，人类用户不会在终端里敲命令，而是通过 Web 界面、聊天窗口、移动 App 等方式与应用交互。这一节我们会探讨如何把 LangGraph 的 Interrupt 机制与各种前端交互模式集成起来，构建真正可用的交互式应用。

## 架构概览：Interrupt 如何连接到前端

要理解如何将 Interrupt 与前端集成，首先需要理清楚数据流的全貌。当图中的某个节点调用 `interrupt()` 时，整个执行流程会暂停，LangGraph 需要把"等待输入"这个状态通知到前端；当用户在前端界面上完成了操作（比如点击了"批准"按钮、填写了表单、发送了消息），前端需要把这个输入传回给 LangGraph 以恢复执行。这中间需要一个中间层来协调两端——通常是一个 API 服务端。

```
┌─────────────┐    HTTP/SSE     ┌──────────────┐    LangGraph API   ┌─────────────┐
│             │ ◄──────────────► │              │ ◄────────────────► │             │
│  前端界面    │                 │  API 服务端   │                   │  LangGraph  │
│ (Web/Chat)  │  1.提交请求      │  (FastAPI)    │  2.invoke()       │  (带        │
│             │  4.展示结果/     │               │  3.返回interrupt   │  checkpointer)│
│             │    等待输入      │              │                    │             │
└─────────────┘                 └──────────────┘                    └─────────────┘
```

整个交互流程分为四个阶段：

**阶段一：用户发起请求**。用户在前端界面中触发一个操作（比如提交代码审查请求），前端通过 HTTP POST 把请求发送给 API 服务端。

**阶段二：服务端调用 LangGraph**。API 服务端收到请求后，构造初始状态并调用 `app.invoke(initial_state, config=config)`。如果图中包含 Interrupt 节点且执行到了该节点，`invoke()` 会返回 interrupt 的 prompt 数据而不是完整的最终结果。

**阶段三：服务端通知前端等待输入**。服务端检测到返回结果不是最终状态（比如检查是否有 `human_decision` 字段为空），就把 interrupt 的 prompt 信息通过 HTTP 响应或 SSE 推送给前端，前端据此渲染出等待用户输入的界面元素（如确认对话框、表单、操作按钮等）。

**阶段四：用户输入后恢复执行**。用户在前端完成操作后，前端再次发送请求（这次包含了用户的输入），服务端调用 `app.invoke(Command(resume=user_input), config=config)` 恢复执行，拿到最终结果后返回给前端展示。

下面我们用一个完整的 FastAPI 示例来演示这个过程。

## FastAPI 后端集成示例

```python
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

app_fastapi = FastAPI(title="审批系统 API")

checkpointer = MemorySaver()

class ReviewRequest(BaseModel):
    code_content: str = Field(..., description="待审查的代码")
    reviewer_id: str = Field(..., description="审查人ID")

class ReviewResponse(BaseModel):
    thread_id: str
    status: str
    message: str
    ai_analysis: Optional[dict] = None
    prompt_for_human: Optional[str] = None

class ApprovalInput(BaseModel):
    thread_id: str
    decision: str = Field(..., description="approve/reject/request_changes")
    comment: Optional[str] = Field(None, description="可选评论")

class CodeReviewState(TypedDict):
    code_content: str
    reviewer_id: str
    issues: list[str]
    score: int
    human_decision: str
    human_comment: str
    final_status: str

def analyze_code_node(state: CodeReviewState) -> dict:
    code = state["code_content"]
    issues = []
    if "TODO" in code or "FIXME" in code:
        issues.append("存在 TODO/FIXME 标记")
    if "print(" in code and "debug" in code.lower():
        issues.append("调试打印语句未移除")
    if len(code.split('\n')) > 100 and "def " not in code:
        issues.append("文件过长且缺少函数划分")

    score = max(0, 100 - len(issues) * 20)
    return {
        "issues": issues,
        "score": score,
        "ai_analysis": {"issues": issues, "score": score}
    }

def human_review_node(state: CodeReviewState) -> dict:
    issues_str = "\n".join(f"- {i}" for i in state["issues"]) or "- 无问题"
    prompt = (
        f"代码审查结果 (评分: {state['score']}/100):\n{issues_str}\n\n"
        f"请决定:\n"
        f"- approve: 批准合并\n"
        f"- reject: 拒绝\n"
        f"- request_changes: 要求修改"
    )
    user_input = interrupt(prompt)

    if not user_input or not user_input.strip():
        return {"human_decision": "pending"}

    parts = user_input.strip().split(maxsplit=1)
    dec = parts[0].lower()
    cmt = parts[1] if len(parts) > 1 else ""

    valid = {"approve", "reject", "request_changes"}
    if dec not in valid:
        return {"human_decision": "pending", "human_comment": f"无效: {dec}"}

    status_map = {"approve": "merged", "reject": "rejected",
                  "request_changes": "changes_requested"}
    return {
        "human_decision": dec,
        "human_comment": cmt,
        "final_status": status_map[dec]
    }

review_graph = StateGraph(CodeReviewState)
review_graph.add_node("analyze", analyze_code_node)
review_graph.add_node("human_review", human_review_node)
review_graph.add_edge(START, "analyze")
review_graph.add_edge("analyze", "human_review")
review_graph.add_conditional_edges("human_review",
    lambda s: END if s["human_decision"] and s["human_decision"] != "pending"
                else "human_review",
    {END: END, "human_review": "human_review"}
)

review_app = review_graph.compile(checkpointer=checkpointer)

@app_fastapi.post("/api/review/start", response_model=ReviewResponse)
async def start_review(req: ReviewRequest):
    import uuid
    thread_id = f"review-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}

    try:
        result = review_app.invoke({
            "code_content": req.code_content,
            "reviewer_id": req.reviewer_id,
            "issues": [], "score": 0,
            "human_decision": "", "human_comment": "",
            "final_status": ""
        }, config=config)

        if result.get("final_status"):
            return ReviewResponse(
                thread_id=thread_id,
                status="completed",
                message=f"自动完成: {result['final_status']}",
                ai_analysis=result.get("ai_analysis")
            )

        return ReviewResponse(
            thread_id=thread_id,
            status="waiting_for_input",
            message="等待审查人决策",
            prompt_for_human=result.get("_interrupt_data", "请进行代码审查")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app_fastapi.post("/api/review/approve", response_model=ReviewResponse)
async def submit_approval(inp: ApprovalInput):
    config = {"configurable": {"thread_id": inp.thread_id}}

    try:
        resume_value = inp.decision
        if inp.comment:
            resume_value = f"{inp.decision} {inp.comment}"

        result = review_app.invoke(
            Command(resume=resume_value),
            config=config
        )

        if result.get("final_status"):
            return ReviewResponse(
                thread_id=inp.thread_id,
                status="completed",
                message=f"审批完成: {result['final_status']}",
                ai_analysis=result.get("ai_analysis")
            )

        return ReviewResponse(
            thread_id=inp.thread_id,
            status="waiting_for_input",
            message="仍需进一步操作",
            prompt_for_human="请重新输入有效指令"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app_fastapi.get("/api/review/status/{thread_id}")
async def get_review_status(thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    try:
        checkpoint = checkpointer.get(config)
        if not checkpoint:
            return JSONResponse(content={"error": "未找到该会话"}, status_code=404)

        metadata = checkpoint.metadata
        return {
            "thread_id": thread_id,
            "status": metadata.get("status", "unknown"),
            "step": metadata.get("step", "unknown")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

这个 FastAPI 服务展示了后端集成的核心模式。`start_review` 端点接收代码审查请求，创建唯一的 `thread_id`，然后调用 `review_app.invoke()` 启动图的执行。如果执行过程中遇到了 Interrupt，返回的响应中会包含 `status="waiting_for_input"` 和 `prompt_for_human` 字段，前端根据这些信息来渲染等待用户操作的界面。

`submit_approval` 端点处理用户的审批操作——它接收 `thread_id` 和用户的决策（加上可选评论），然后用 `Command(resume=...)` 的形式恢复图的执行。注意这里我们把 `decision` 和 `comment` 合并成一个字符串传入，因为我们的 `human_review_node` 函数是用空格分隔来解析这两部分的。

还有一个 `get_review_status` 端点用于查询当前审批的状态——这在实际应用中非常有用，因为用户可能刷新页面或切换设备后需要知道之前的审批进展到哪里了。

## 前端集成模式

有了上面的后端 API，前端就可以用多种方式与之集成。最简单的方式是传统的请求-响应模式——先发请求启动流程，如果返回"等待输入"，就显示操作界面让用户操作，然后再发请求提交操作。

```html
<!-- 审批页面的简化 HTML 结构 -->
<div id="review-app">
    <div id="step1-submit">
        <h3>提交代码审查</h3>
        <textarea id="code-input" placeholder="粘贴代码..."></textarea>
        <button onclick="submitCode()">开始审查</button>
    </div>

    <div id="step2-waiting" style="display:none;">
        <div class="spinner">AI 分析中...</div>
    </div>

    <div id="step3-review" style="display:none;">
        <h3>代码审查</h3>
        <div id="analysis-result"></div>
        <div class="actions">
            <button onclick="makeDecision('approve')">✅ 批准</button>
            <button onclick="makeDecision('reject')">❌ 拒绝</button>
            <button onclick="makeDecision('request_changes')">📝 要求修改</button>
        </div>
        <input type="text" id="comment-input" placeholder="评论（可选）">
    </div>

    <div id="step4-result" style="display:none;">
        <h3 id="result-title"></h3>
        <p id="result-message"></p>
        <button onclick="resetForm()">新的审查</button>
    </div>
</div>

<script>
let currentThreadId = null;

async function submitCode() {
    const code = document.getElementById('code-input').value;
    if (!code.trim()) return;

    document.getElementById('step1-submit').style.display = 'none';
    document.getElementById('step2-waiting').style.display = 'block';

    try {
        const resp = await fetch('/api/review/start', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                code_content: code,
                reviewer_id: 'user-' + Date.now()
            })
        });
        const data = await resp.json();

        currentThreadId = data.thread_id;

        if (data.status === 'completed') {
            showResult(data.message);
        } else if (data.status === 'waiting_for_input') {
            showReviewPanel(data);
        }
    } catch (err) {
        alert('错误: ' + err.message);
        resetForm();
    }
}

function showReviewPanel(data) {
    document.getElementById('step2-waiting').style.display = 'none';
    document.getElementById('step3-review').style.display = 'block';

    let analysisHtml = '<pre>' + JSON.stringify(
        data.ai_analysis || data.prompt_for_human, null, 2
    ) + '</pre>';
    document.getElementById('analysis-result').innerHTML = analysisHtml;
}

async function makeDecision(decision) {
    const comment = document.getElementById('comment-input').value;

    try {
        const resp = await fetch('/api/review/approve', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                thread_id: currentThreadId,
                decision: decision,
                comment: comment || undefined
            })
        });
        const data = await resp.json();
        showResult(data.message || data.final_status);
    } catch (err) {
        alert('错误: ' + err.message);
    }
}

function showResult(message) {
    document.getElementById('step3-review').style.display = 'none';
    document.getElementById('step4-result').style.display = 'block';
    document.getElementById('result-title').textContent = '审查完成';
    document.getElementById('result-message').textContent = message;
}

function resetForm() {
    location.reload();
}
</script>
```

这个简化的前端页面展示了四步交互流程：第一步是用户粘贴代码并点击"开始审查"；第二步显示加载动画表示 AI 正在分析；第三步展示 AI 的分析结果和三个操作按钮供用户选择；第四步展示最终的审批结果。虽然简单，但它完整地覆盖了"提交→处理→等待→操作→完成"的完整生命周期。

## SSE 实时推送模式

对于更复杂的场景（特别是需要实时展示 LLM 流式输出的情况），使用 Server-Sent Events (SSE) 会比传统的请求-响应模式体验更好。SSE 允许服务端主动向前端推送事件，这样当 LLM 在生成内容时可以实时展示每个 token，而不需要等到全部生成完毕才一次性返回。

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import json

@app_fastapi.post("/api/review/stream")
async def stream_review(req: ReviewRequest):
    import uuid
    thread_id = f"stream-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}

    async def event_generator():
        try:
            for event in review_app.stream({
                "code_content": req.code_content,
                "reviewer_id": req.reviewer_id,
                "issues": [], "score": 0,
                "human_decision": "", "human_comment": "",
                "final_status": ""
            }, config=config, stream_mode="updates"):

                for node_name, update in event.items():
                    payload = {
                        "type": "node_update",
                        "node": node_name,
                        "data": {k: str(v)[:200] for k, v in update.items()}
                    }
                    yield f"data: {json.dumps(payload)}\n\n"

            yield f"data: {json.dumps({'type': 'interrupt_detected'})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

前端使用 EventSource 来监听 SSE 事件：

```javascript
const evtSource = new EventSource('/api/review/stream');

evtSource.onmessage = function(event) {
    const data = JSON.parse(event.data);

    switch(data.type) {
        case 'node_update':
            updateNodeStatus(data.node, data.data);
            break;
        case 'interrupt_detected':
            showUserActionPanel();
            evtSource.close();
            break;
        case 'error':
            showError(data.message);
            evtSource.close();
            break;
    }
};
```

## 多用户并发场景下的注意事项

当你的应用需要支持多个用户同时使用 Interrupt 功能时，有几个关键的设计考量。

第一，**每个用户必须有独立的 thread_id**。这是最重要的原则——thread_id 是 LangGraph 区分不同执行实例的唯一标识符，如果两个用户共享同一个 thread_id，他们的操作会互相干扰。通常的做法是用用户 ID 加上业务 ID（如请求 ID）来生成 thread_id，例如 `"review-user123-req456"`。

第二，**使用持久化的 checkpointer 替代 MemorySaver**。MemorySaver 只是把状态保存在内存中，进程重启后所有状态都会丢失。在生产环境中应该使用 PostgresSaver 或 SqliteSaver 等基于数据库的 checkpointer，这样即使服务重启，用户之前暂停的审批流程也能恢复。

第三，**考虑权限控制**。不是所有人都应该能够恢复任意一个被暂停的执行——只有对应的审批人才有权限对特定的审批做决策。你需要在 API 层面加入权限验证逻辑，确保 `submit_approval` 端点只接受来自合法审批人的请求。

第四，**设置合理的超时和清理机制**。如果一个 Interrupt 已经暂停了很久（比如超过 7 天）都没有人来恢复它，这个悬挂的状态会一直占用存储空间。应该有一个定时任务来扫描和清理过期的中断状态，或者至少标记它们为"已过期"。
