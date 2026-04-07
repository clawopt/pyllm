# 5. 部署与扩展

## 从原型到产品：让研究助手服务真实用户

经过前面四个小节的开发，DeepResearch 已经拥有了一个功能完整、能力强大的核心引擎——从查询解析到多步推理，从工具调用到假设验证，从报告生成到质量校验。但到目前为止，它还只是一个"能在本地跑起来"的 Python 脚本。要让它变成一个真正可用的产品，我们还需要完成最后一步：**部署与扩展**。

这一节的任务是把 DeepResearch 打包成一个完整的 Web 服务，包括 REST API 层、用户交互界面、容器化部署配置，以及面向未来的扩展架构设计。我们会借鉴第九章（客服工单系统）的很多工程化实践，但同时也会针对研究助手的特殊性做一些定制化的设计——比如长时间运行任务的异步处理机制、流式进度推送、以及结果缓存策略。

### API 服务层设计

研究助手和客服工单系统在 API 设计上有一个根本性的区别：**工单系统的每个请求通常在几秒内就能完成响应**（分类、路由、自动回复都是轻量操作），而**一次深度研究任务可能需要 5-10 分钟甚至更久**。这意味着我们不能用简单的同步请求-响应模式来处理研究任务——HTTP 连接会在这么长的时间内超时，用户体验也会非常差。

解决方案是采用**异步任务模式**：用户提交研究请求后立即获得一个任务 ID 和一个查询端点，然后通过轮询或 SSE（Server-Sent Events）来获取实时进度和最终结果。

```python
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import uuid
import asyncio
import json

app = FastAPI(
    title="DeepResearch API",
    description="自主深度研究助手 API",
    version="2.0.0"
)

# ========== 数据模型 ==========

class ResearchRequest(BaseModel):
    query: str = Field(..., description="研究查询", min_length=3, max_length=1000)
    max_rounds: int = Field(3, ge=1, le=8, description="最大搜索轮次")
    max_sources: int = Field(20, ge=5, le=50, description="最大信息源数量")
    depth_level: str = Field("medium", description="研究深度: simple/medium/complex")
    output_format: str = Field("markdown", description="输出格式: markdown/html/pdf")
    language: str = Field("zh", description="语言: zh/en/ja")
    required_source_types: Optional[List[str]] = None
    excluded_domains: Optional[List[str]] = None

class ResearchResponse(BaseModel):
    task_id: str
    status: str
    message: str
    created_at: str
    estimated_duration_seconds: int

class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    progress_percent: float
    current_phase: str
    current_round: int
    total_rounds: int
    facts_collected: int
    sources_collected: int
    log_entries: List[str]
    result: Optional[Dict] = None
    error: Optional[str] = None
    started_at: str
    updated_at: str
    completed_at: Optional[str] = None

# ========== 任务管理器 ==========

class TaskManager:
    def __init__(self):
        self.tasks: Dict[str, Dict] = {}
        self._lock = asyncio.Lock()
    
    async def create_task(self, request: ResearchRequest) -> str:
        task_id = f"research_{uuid.uuid4().hex[:12]}"
        
        estimated_duration = request.max_rounds * 120
        
        self.tasks[task_id] = {
            "task_id": task_id,
            "status": "pending",
            "progress": 0.0,
            "phase": "排队中",
            "current_round": 0,
            "total_rounds": request.max_rounds,
            "facts_collected": 0,
            "sources_collected": 0,
            "log_entries": [],
            "result": None,
            "error": None,
            "request": request.model_dump(),
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "completed_at": None,
            "estimated_duration": estimated_duration,
            "subscribers": []
        }
        
        return task_id
    
    async def update_progress(self, task_id: str, **updates):
        if task_id in self.tasks:
            task = self.tasks[task_id]
            for key, value in updates.items():
                task[key] = value
            task["updated_at"] = datetime.now().isoformat()
            
            for queue in task.get("subscribers", []):
                if not queue.full():
                    await queue.put(task)
    
    async def get_task(self, task_id: str) -> Optional[Dict]:
        return self.tasks.get(task_id)
    
    async def subscribe(self, task_id: str) -> asyncio.Queue:
        queue = asyncio.Queue(maxsize=50)
        if task_id in self.tasks:
            self.tasks[task_id]["subscribers"].append(queue)
        return queue
    
    async def unsubscribe(self, task_id: str, queue: asyncio.Queue):
        if task_id in self.tasks:
            subs = self.tasks[task_id].get("subscribers", [])
            if queue in subs:
                subs.remove(queue)

task_manager = TaskManager()

# ========== API 端点 ==========

@app.post("/api/research", response_model=ResearchResponse, tags=["研究"])
async def create_research_task(request: ResearchRequest):
    """创建一个新的深度研究任务"""
    task_id = await task_manager.create_task(request)
    
    background_tasks = BackgroundTasks()
    background_tasks.add_task(execute_research_task, task_id, request)
    
    return ResearchResponse(
        task_id=task_id,
        status="pending",
        message=f"研究任务已创建，预计需要 {task_manager.tasks[task_id]['estimated_duration']} 秒",
        created_at=task_manager.tasks[task_id]["created_at"],
        estimated_duration_seconds=task_manager.tasks[task_id]["estimated_duration"]
    )

@app.get("/api/research/{task_id}", response_model=TaskStatusResponse, tags=["研究"])
async def get_research_status(task_id: str):
    """查询研究任务的当前状态和结果"""
    task = await task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    return TaskStatusResponse(**{k: v for k, v in task.items() 
                                if k in TaskStatusResponse.__fields__})

@app.get("/api/research/{task_id}/stream", tags=["研究"])
async def stream_research_progress(task_id: str):
    """SSE 流式推送研究进度"""
    task = await task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    queue = await task_manager.subscribe(task_id)
    
    async def event_generator():
        try:
            initial_data = {k: v for k, v in task.items() 
                          if k in ["status", "progress", "phase", "current_round", 
                                   "total_rounds", "facts_collected", "sources_collected",
                                   "log_entries"]}
            yield f"data: {json.dumps(initial_data, ensure_ascii=False)}\n\n"
            
            while True:
                try:
                    update = await asyncio.wait_for(queue.get(), timeout=30.0)
                    data = {k: v for k, v in update.items() 
                           if k in ["status", "progress", "phase", "current_round", 
                                    "total_rounds", "facts_collected", "sources_collected",
                                    "log_entries", "result", "error"]}
                    yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                    
                    if update.get("status") in ("completed", "failed"):
                        yield f"event: done\ndata: {json.dumps({'status': update['status']})}\n\n"
                        break
                        
                except asyncio.TimeoutError:
                    yield f": heartbeat\n\n"
                    
        finally:
            await task_manager.unsubscribe(task_id, queue)
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@app.delete("/api/research/{task_id}", tags=["研究"])
async def cancel_research_task(task_id: str):
    """取消正在执行的研究任务"""
    task = await task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    if task["status"] in ("completed", "failed"):
        raise HTTPException(status_code=400, detail="任务已完成或已失败，无法取消")
    
    await task_manager.update_progress(task_id, status="cancelled", error="用户主动取消")
    
    return {"message": f"任务 {task_id} 已取消"}

@app.get("/api/research/{task_id}/timeline", tags=["研究"])
async def get_research_timeline(task_id: str):
    """获取研究的详细时间线日志"""
    task = await task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    log_entries = task.get("log_entries", [])
    parsed_timeline = []
    
    for entry in log_entries:
        entry_type = "info"
        if "[Reactor" in entry or "[Round" in entry:
            entry_type = "action"
        elif "[Hypothesis" in entry:
            entry_type = "hypothesis"
        elif "[DONE]" in entry:
            entry_type = "milestone"
        elif "评估完成" in entry:
            entry_type = "decision"
        
        parsed_timeline.append({
            "type": entry_type,
            "content": entry
        })
    
    return {"task_id": task_id, "timeline": parsed_timeline}

@app.get("/api/research/{task_id}/facts", tags=["研究"])
async def get_research_facts(task_id: str, min_confidence: float = 0.0):
    """获取研究中提取的所有事实"""
    task = await task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    result = task.get("result")
    if not result:
        raise HTTPException(status_code=400, detail="研究尚未完成")
    
    facts = result.get("extracted_facts", [])
    filtered = [f for f in facts if f.get("confidence", 0) >= min_confidence]
    
    filtered.sort(key=lambda x: x.get("confidence", 0), reverse=True)
    
    return {
        "task_id": task_id,
        "total_facts": len(facts),
        "filtered_count": len(filtered),
        "facts": filtered
    }

@app.get("/api/dashboard", tags=["系统"])
async def get_dashboard_stats():
    """获取系统仪表盘统计"""
    tasks = list(task_manager.tasks.values())
    
    total = len(tasks)
    pending = sum(1 for t in tasks if t["status"] == "pending")
    running = sum(1 for t in tasks if t["status"] == "running")
    completed = sum(1 for t in tasks if t["status"] == "completed")
    failed = sum(1 for t in tasks if t["status"] == "failed")
    
    avg_duration = 0
    completed_tasks = [t for t in tasks if t.get("completed_at") and t.get("created_at")]
    if completed_tasks:
        durations = [
            (datetime.fromisoformat(t["completed_at"]) - datetime.fromisoformat(t["created_at"])).total_seconds()
            for t in completed_tasks
        ]
        avg_duration = sum(durations) / len(durations)
    
    total_facts_all = sum(len(t.get("result", {}).get("extracted_facts", [])) 
                         for t in tasks if t.get("result"))
    total_sources_all = sum(len(t.get("result", {}).get("collected_sources", [])) 
                           for t in tasks if t.get("result"))
    
    return {
        "tasks": {
            "total": total,
            "pending": pending,
            "running": running,
            "completed": completed,
            "failed": failed
        },
        "performance": {
            "avg_completion_time_seconds": round(avg_duration, 1),
            "total_facts_generated": total_facts_all,
            "total_sources_collected": total_sources_all,
            "avg_facts_per_task": round(total_facts_all / max(completed, 1), 1),
            "avg_sources_per_task": round(total_sources_all / max(completed, 1), 1)
        },
        "system_info": {
            "active_tools": len(tool_registry.list_all()),
            "version": "2.0.0"
        }
    }
```

这套 API 的设计有几个值得深入讨论的要点：

**异步任务模式的核心**：`create_research_task` 端点在接收请求后立即返回 `task_id`，真正的执行通过 `BackgroundTasks` 放到后台进行。用户拿到 `task_id` 后有两种方式跟踪进度：

1. **轮询模式**（`GET /api/research/{task_id}`）：客户端每隔几秒请求一次，简单可靠但有一定延迟
2. **SSE 推送模式**（`GET /api/research/{task_id}/stream`）：服务器主动推送每次状态变更，实时性更好，适合 Web 前端使用

SSE 实现中的几个细节：
- 使用 `asyncio.Queue` 作为发布-订阅的消息通道，每个订阅者有自己的队列
- 30 秒超时的心跳机制（`heartbeat`）防止连接被代理或负载均衡器断开
- 当任务完成时发送特殊的 `event: done` 事件通知前端关闭连接
- `X-Accel-Buffering: no` 头告诉 Nginx 不要缓冲 SSE 响应

**细粒度的数据访问**：除了整体状态外，我们还提供了 `/facts` 端点来单独获取提取的事实列表（支持按置信度过滤），以及 `/timeline` 端点来获取结构化的时间线日志。这些端点为前端的交互式回顾功能提供了数据支撑。

### 后台任务执行器

有了 API 层之后，我们需要实现真正的研究任务执行逻辑——把之前定义的 LangGraph 图包装成一个可以在后台运行的异步函数。

```python
import traceback

async def execute_research_task(task_id: str, request: ResearchRequest):
    try:
        await task_manager.update_progress(
            task_id, status="running", phase="初始化", progress=5.0
        )
        
        initial_state: ResearchState = {
            "input_config": request.model_dump(),
            "task_context": {
                "plan_id": task_id,
                "started_at": datetime.now().isoformat(),
                "current_round": 0,
                "current_focus_index": 0,
                "total_searches_performed": 0,
                "total_sources_collected": 0,
                "total_facts_extracted": 0,
                "token_usage": {"input": 0, "output": 0},
                "estimated_cost_usd": 0.0,
                "rounds_without_new_info": 0
            },
            "topic": "",
            "sub_questions": [],
            "search_strategies": [],
            "collected_sources": [],
            "extracted_facts": [],
            "current_search_results": [],
            "current_facts_batch": [],
            "coverage_map": {},
            "sufficiency_verdict": "",
            "stuck_reason": "",
            "final_report": "",
            "report_metadata": {},
            "status": "planning",
            "research_log": [],
            "active_hypotheses": [],
            "current_conflicts": []
        }
        
        class ProgressCallback:
            def __init__(self, tid):
                self.task_id = tid
            
            async def on_node_complete(self, node_name: str, state: dict):
                tc = state.get("task_context", {})
                current_round = tc.get("current_round", 0)
                max_rounds = state.get("input_config", {}).get("max_rounds", 3)
                
                phase_names = {
                    "parse_query": "解析查询",
                    "create_plan": "制定计划",
                    "reactor_loop": "搜索与分析",
                    "generate_hypotheses": "生成假设",
                    "verify_hypotheses": "验证假设",
                    "evaluate_findings": "评估发现",
                    "adjust_strategy": "调整策略",
                    "generate_report": "生成报告"
                }
                
                phase = phase_names.get(node_name, node_name)
                
                facts_count = len(state.get("extracted_facts", []))
                sources_count = len(state.get("collected_sources", []))
                
                base_progress = (current_round / max(max_rounds, 1)) * 80
                node_progress = {
                    "parse_query": 5, "create_plan": 10,
                    "reactor_loop": 15, "generate_hypotheses": 5,
                    "verify_hypotheses": 5, "evaluate_findings": 10,
                    "adjust_strategy": 5, "generate_report": 20
                }.get(node_name, 5)
                
                total_progress = min(95, base_progress + node_progress)
                
                logs = state.get("research_log", [])
                recent_logs = logs[-3:] if logs else []
                
                await task_manager.update_progress(
                    self.task_id,
                    status="running",
                    phase=phase,
                    current_round=current_round,
                    total_rounds=max_rounds,
                    progress=total_progress,
                    facts_collected=facts_count,
                    sources_collected=sources_count,
                    log_entries=recent_logs
                )
        
        callback = ProgressCallback(task_id)
        
        final_state = await enhanced_research_graph.ainvoke(
            initial_state,
            config={"configurable": {"thread_id": task_id}}
        )
        
        report_text, metadata, qa_result, final_logs = await generate_final_report_with_qa(final_state)
        
        await task_manager.update_progress(
            task_id,
            status="completed",
            progress=100.0,
            phase="完成",
            result={
                "report_markdown": report_text,
                "report_metadata": metadata,
                "quality_assessment": qa_result,
                "extracted_facts": final_state["extracted_facts"],
                "collected_sources": final_state["collected_sources"],
                "active_hypotheses": final_state.get("active_hypotheses", []),
                "coverage_map": final_state["coverage_map"],
                "research_log": final_state["research_log"],
                "task_context": final_state["task_context"]
            },
            log_entries=final_logs,
            completed_at=datetime.now().isoformat()
        )
        
    except Exception as e:
        error_detail = f"{type(e).__name__}: {str(e)}"
        error_traceback = traceback.format_exc()[-2000:]
        
        print(f"[ERROR] 任务 {task_id} 执行失败: {error_detail}")
        print(error_traceback)
        
        await task_manager.update_progress(
            task_id,
            status="failed",
            progress=-1,
            phase="错误",
            error=error_detail,
            log_entries=[f"[ERROR] {error_detail}"]
        )
```

`execute_research_task` 是整个后台执行的入口函数。它的关键设计点是 **ProgressCallback**——这是一个自定义的回调类，在每个节点完成后被调用，负责计算当前的进度百分比、确定当前阶段名称、收集最新的日志条目，然后通过 `task_manager.update_progress()` 推送给所有订阅者。

进度计算的策略是：基础进度由当前轮次占总轮次的比例决定（占 80%），再加上当前节点的固定权重（解析 5%、规划 10%、搜索分析 15% 等）。最终报告生成占最后的 20%。这样用户看到的进度条会是一个相对平滑的增长过程。

错误处理也很完善——任何未捕获的异常都会被捕获，记录详细的错误信息和堆栈追踪，然后把任务标记为失败并通知所有等待者。

### 前端界面实现

接下来是用户直接交互的部分——一个现代化的单页面 Web 应用。这个界面需要展示研究提交表单、实时进度面板、报告查看器和交互式探索工具。

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DeepResearch - 自主深度研究助手</title>
    <style>
        :root {
            --primary: #6366f1;
            --primary-dark: #4f46e5;
            --bg: #0f172a;
            --surface: #1e293b;
            --surface-light: #334155;
            --text: #e2e8f0;
            --text-muted: #94a3b8;
            --success: #22c55e;
            --warning: #f59e0b;
            --error: #ef4444;
            --border: #334155;
        }
        
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.6;
        }
        
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        
        header {
            text-align: center;
            padding: 40px 0;
            border-bottom: 1px solid var(--border);
            margin-bottom: 30px;
        }
        
        header h1 { font-size: 2.5rem; background: linear-gradient(135deg, #6366f1, #a78bfa); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        header p { color: var(--text-muted); margin-top: 8px; }
        
        .card {
            background: var(--surface);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
            border: 1px solid var(--border);
        }
        
        .form-group { margin-bottom: 16px; }
        .form-group label { display: block; font-size: 0.9rem; color: var(--text-muted); margin-bottom: 6px; }
        .form-group textarea, .form-group input, .form-group select {
            width: 100%;
            padding: 12px 16px;
            background: var(--bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            color: var(--text);
            font-size: 1rem;
            transition: border-color 0.2s;
        }
        .form-group textarea:focus, .form-group input:focus { outline: none; border-color: var(--primary); }
        .form-group textarea { min-height: 100px; resize: vertical; }
        
        .btn {
            padding: 12px 28px;
            border: none;
            border-radius: 8px;
            font-size: 1rem;
            cursor: pointer;
            transition: all 0.2s;
            font-weight: 500;
        }
        .btn-primary { background: var(--primary); color: white; }
        .btn-primary:hover { background: var(--primary-dark); transform: translateY(-1px); }
        .btn-primary:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
        
        .progress-container { margin: 30px 0; }
        .progress-bar-bg {
            height: 12px;
            background: var(--surface-light);
            border-radius: 6px;
            overflow: hidden;
            position: relative;
        }
        .progress-bar-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--primary), #a78bfa);
            border-radius: 6px;
            transition: width 0.5s ease;
            position: relative;
        }
        .progress-bar-fill::after {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
            animation: shimmer 2s infinite;
        }
        @keyframes shimmer { 0% { transform: translateX(-100%); } 100% { transform: translateX(100%); } }
        
        .status-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 16px; margin: 20px 0; }
        .stat-card {
            background: var(--bg);
            padding: 16px;
            border-radius: 8px;
            text-align: center;
        }
        .stat-value { font-size: 1.8rem; font-weight: 700; color: var(--primary); }
        .stat-label { font-size: 0.85rem; color: var(--text-muted); margin-top: 4px; }
        
        .log-container {
            background: var(--bg);
            border-radius: 8px;
            padding: 16px;
            max-height: 300px;
            overflow-y: auto;
            font-family: 'Monaco', 'Consolas', monospace;
            font-size: 0.85rem;
        }
        .log-entry { padding: 4px 0; border-bottom: 1px solid var(--border); }
        .log-entry:last-child { border-bottom: none; }
        .log-action { color: var(--primary); }
        .log-hypothesis { color: var(--warning); }
        .log-milestone { color: var(--success); font-weight: 600; }
        .log-error { color: var(--error); }
        
        .report-viewer {
            background: white;
            color: #1e293b;
            padding: 40px;
            border-radius: 8px;
            line-height: 1.8;
            max-height: 70vh;
            overflow-y: auto;
        }
        .report-viewer h1 { font-size: 2rem; margin-bottom: 16px; color: #0f172a; }
        .report-viewer h2 { font-size: 1.5rem; margin: 32px 0 16px; color: #1e293b; border-bottom: 2px solid #e2e8f0; padding-bottom: 8px; }
        .report-viewer h3 { font-size: 1.2rem; margin: 24px 0 12px; color: #334155; }
        .report-viewer blockquote { border-left: 4px solid #6366f1; padding: 12px 20px; margin: 16px 0; background: #f8fafc; border-radius: 0 8px 8px 0; }
        .report-viewer code { background: #f1f5f9; padding: 2px 6px; border-radius: 4px; font-size: 0.9em; }
        .report-viewer table { width: 100%; border-collapse: collapse; margin: 16px 0; }
        .report-viewer th, .report-viewer td { padding: 10px 14px; text-align: left; border: 1px solid #e2e8f0; }
        .report-viewer th { background: #f8fafc; font-weight: 600; }
        
        .tabs { display: flex; gap: 4px; margin-bottom: 20px; border-bottom: 2px solid var(--border); }
        .tab {
            padding: 10px 20px;
            cursor: pointer;
            border: none;
            background: none;
            color: var(--text-muted);
            font-size: 0.95rem;
            border-bottom: 2px solid transparent;
            margin-bottom: -2px;
            transition: all 0.2s;
        }
        .tab.active { color: var(--primary); border-bottom-color: var(--primary); }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        
        .quality-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 600;
        }
        .grade-A { background: #dcfce7; color: #166534; }
        .grade-B { background: #fef9c3; color: #854d0e; }
        .grade-C { background: #fee2e2; color: #991b1b; }
        .grade-D { background: #fecaca; color: #991b1b; }
        
        .hidden { display: none !important; }
        .fade-in { animation: fadeIn 0.3s ease; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🔬 DeepResearch</h1>
            <p>基于 LangGraph 的自主深度研究助手 —— 提交问题，获得专业级研究报告</p>
        </header>
        
        <div id="submitSection" class="card">
            <h2 style="margin-bottom: 20px;">📝 发起研究</h2>
            <div class="form-group">
                <label>研究主题</label>
                <textarea id="queryInput" placeholder="例如：2025年AI Agent的技术路线图和主要玩家分析"></textarea>
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px;">
                <div class="form-group">
                    <label>研究深度</label>
                    <select id="depthSelect">
                        <option value="simple">快速概览 (2轮)</option>
                        <option value="medium" selected>标准研究 (3轮)</option>
                        <option value="complex">深度调研 (5轮)</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>输出格式</label>
                    <select id="formatSelect">
                        <option value="markdown" selected>Markdown</option>
                        <option value="html">HTML</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>语言</label>
                    <select id="languageSelect">
                        <option value="zh" selected>中文</option>
                        <option value="en">English</option>
                    </select>
                </div>
            </div>
            <button class="btn btn-primary" onclick="startResearch()" id="submitBtn">
                🚀 开始研究
            </button>
        </div>
        
        <div id="progressSection" class="card hidden">
            <h2 style="margin-bottom: 16px;">⏳ 研究进行中</h2>
            
            <div class="progress-container">
                <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                    <span id="phaseText" style="color: var(--primary); font-weight: 500;">准备中...</span>
                    <span id="percentText">0%</span>
                </div>
                <div class="progress-bar-bg">
                    <div class="progress-bar-fill" id="progressBar" style="width: 0%"></div>
                </div>
            </div>
            
            <div class="status-grid">
                <div class="stat-card">
                    <div class="stat-value" id="roundValue">0/3</div>
                    <div class="stat-label">当前轮次</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="sourcesValue">0</div>
                    <div class="stat-label">信息来源</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="factsValue">0</div>
                    <div class="stat-label">提取事实</div>
                </div>
            </div>
            
            <h3 style="margin: 16px 0 8px; font-size: 0.95rem; color: var(--text-muted);">📋 实时日志</h3>
            <div class="log-container" id="logContainer">
                <div class="log-entry" style="color: var(--text-muted);">等待开始...</div>
            </div>
            
            <button class="btn btn-primary" style="margin-top: 16px; background: var(--error);" onclick="cancelTask()" id="cancelBtn">
                取消任务
            </button>
        </div>
        
        <div id="resultSection" class="card hidden">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
                <h2>📄 研究报告</h2>
                <div id="qualityBadge"></div>
            </div>
            
            <div class="tabs">
                <button class="tab active" onclick="switchTab('report')">📊 报告</button>
                <button class="tab" onclick="switchTab('facts')">🔍 事实列表</button>
                <button class="tab" onclick="switchTab('timeline')">📅 时间线</button>
                <button class="tab" onclick="switchTab('stats')">📈 统计</button>
            </div>
            
            <div id="tabReport" class="tab-content active">
                <div class="report-viewer" id="reportContent"></div>
            </div>
            
            <div id="tabFacts" class="tab-content">
                <table style="width: 100%; border-collapse: collapse;">
                    <thead>
                        <tr style="background: var(--surface-light);">
                            <th style="padding: 10px; text-align: left;">主体</th>
                            <th style="padding: 10px; text-align: left;">谓语</th>
                            <th style="padding: 10px; text-align: left;">客体</th>
                            <th style="padding: 10px; text-align: center;">置信度</th>
                            <th style="padding: 10px; text-align: left;">来源</th>
                        </tr>
                    </thead>
                    <tbody id="factsTableBody"></tbody>
                </table>
            </div>
            
            <div id="tabTimeline" class="tab-content">
                <div class="log-container" id="timelineContainer"></div>
            </div>
            
            <div id="tabStats" class="tab-content">
                <div class="status-grid" id="statsGrid"></div>
            </div>
            
            <button class="btn btn-primary" style="margin-top: 20px;" onclick="newResearch()">
                ✨ 发起新研究
            </button>
        </div>
    </div>

    <script>
        const API_BASE = window.location.hostname === 'localhost' ? '' : '';
        let currentTaskId = null;
        let eventSource = null;

        async function startResearch() {
            const query = document.getElementById('queryInput').value.trim();
            if (!query) { alert('请输入研究主题'); return; }
            
            const depthMap = { simple: 2, medium: 3, complex: 5 };
            const maxRounds = depthMap[document.getElementById('depthSelect').value];
            
            document.getElementById('submitBtn').disabled = true;
            
            try {
                const resp = await fetch(`${API_BASE}/api/research`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        query: query,
                        max_rounds: maxRounds,
                        output_format: document.getElementById('formatSelect').value,
                        language: document.getElementById('languageSelect').value
                    })
                });
                
                const data = await resp.json();
                currentTaskId = data.task_id;
                
                document.getElementById('submitSection').classList.add('hidden');
                document.getElementById('progressSection').classList.remove('hidden');
                document.getElementById('progressSection').classList.add('fade-in');
                
                connectSSE(currentTaskId);
                
            } catch (err) {
                alert('创建任务失败: ' + err.message);
                document.getElementById('submitBtn').disabled = false;
            }
        }

        function connectSSE(taskId) {
            eventSource = new EventSource(`${API_BASE}/api/research/${taskId}/stream`);
            
            eventSource.onmessage = (event) => {
                const data = JSON.parse(event.data);
                updateProgressUI(data);
            };
            
            eventSource.addEventListener('done', (event) => {
                const data = JSON.parse(event.data);
                setTimeout(() => loadResult(taskId), 1500);
            });
            
            eventSource.onerror = () => {
                console.log('SSE 断开，切换到轮询模式');
                eventSource.close();
                startPolling(taskId);
            };
        }

        let pollInterval = null;
        function startPolling(taskId) {
            pollInterval = setInterval(async () => {
                try {
                    const resp = await fetch(`${API_BASE}/api/research/${taskId}`);
                    const data = await resp.json();
                    updateProgressUI(data);
                    
                    if (data.status === 'completed' || data.status === 'failed') {
                        clearInterval(pollInterval);
                        if (data.status === 'completed') loadResult(taskId);
                    }
                } catch (e) { console.error('轮询失败:', e); }
            }, 3000);
        }

        function updateProgressUI(data) {
            const pct = Math.min(100, Math.max(0, data.progress || 0));
            document.getElementById('progressBar').style.width = pct + '%';
            document.getElementById('percentText') = Math.round(pct) + '%';
            document.getElementById('phaseText').textContent = data.phase || '处理中...';
            document.getElementById('roundValue').textContent = 
                `${data.current_round || 0}/${data.total_rounds || 3}`;
            document.getElementById('sourcesValue').textContent = data.sources_collected || 0;
            document.getElementById('factsValue').textContent = data.facts_collected || 0;
            
            if (data.log_entries && data.log_entries.length > 0) {
                const container = document.getElementById('logContainer');
                container.innerHTML = data.log_entries.map(entry => {
                    let cls = 'log-entry';
                    if (entry.includes('[Reactor') || entry.includes('[Round')) cls += ' log-action';
                    else if (entry.includes('[Hypothesis')) cls += ' log-hypothesis';
                    else if (entry.includes('[DONE]')) cls += ' log-milestone';
                    else if (entry.includes('[ERROR]')) cls += ' log-error';
                    return `<div class="${cls}">${escapeHtml(entry)}</div>`;
                }).join('');
                container.scrollTop = container.scrollHeight;
            }
            
            if (data.error) {
                document.getElementById('logContainer').innerHTML += 
                    `<div class="log-entry log-error">❌ ${escapeHtml(data.error)}</div>`;
            }
        }

        async function loadResult(taskId) {
            const resp = await fetch(`${API_BASE}/api/research/${taskId}`);
            const data = await resp.json();
            
            if (data.status !== 'completed' || !data.result) {
                alert('研究未能完成: ' + (data.error || '未知错误'));
                return;
            }
            
            document.getElementById('progressSection').classList.add('hidden');
            document.getElementById('resultSection').classList.remove('hidden');
            document.getElementById('resultSection').classList.add('fade-in');
            
            const result = data.result;
            
            // 报告内容
            const reportHtml = marked.parse(result.report_markdown || '');
            document.getElementById('reportContent').innerHTML = reportHtml;
            
            // 质量徽章
            const qa = result.quality_assessment || {};
            const grade = qa.grade || 'N/A';
            const score = `${qa.overall_score || 0}/${qa.max_score || 100}`;
            document.getElementById('qualityBadge').innerHTML = 
                `<span class="quality-badge grade-${grade}">质量评级: ${grade} (${score})</span>`;
            
            // 事实表格
            const facts = (result.extracted_facts || []).slice(0, 50);
            document.getElementById('factsTableBody').innerHTML = facts.map(f => `
                <tr style="border-bottom: 1px solid var(--border);">
                    <td style="padding: 8px;">${escapeHtml(f.subject || '')}</td>
                    <td style="padding: 8px;">${escapeHtml(f.predicate || '')}</td>
                    <td style="padding: 8px;">${escapeHtml((f.object_value || '').substring(0, 100))}</td>
                    <td style="padding: 8px; text-align: center;">
                        <span style="color: ${f.confidence >= 0.8 ? 'var(--success)' : f.confidence >= 0.6 ? 'var(--warning)' : 'var(--error)'}">
                            ${(f.confidence * 100).toFixed(0)}%
                        </span>
                    </td>
                    <td style="padding: 8px; font-size: 0.8rem; color: var(--text-muted);">
                        ${(f.source_urls || []).length} 个来源
                    </td>
                </tr>
            `).join('');
            
            // 时间线
            const timeline = (result.research_log || []).map(entry => {
                let cls = 'log-entry';
                if (entry.includes('[Reactor') || entry.includes('[Round')) cls += ' log-action';
                else if (entry.includes('[Hypothesis')) cls += ' log-hypothesis';
                else if (entry.includes('[DONE]')) cls += ' log-milestone';
                return `<div class="${cls}">${escapeHtml(entry)}</div>`;
            }).join('');
            document.getElementById('timelineContainer').innerHTML = timeline || '<div class="log-entry">无记录</div>';
            
            // 统计
            const ctx = result.task_context || {};
            document.getElementById('statsGrid').innerHTML = `
                <div class="stat-card"><div class="stat-value">${ctx.current_round || 0}</div><div class="stat-label">总轮次</div></div>
                <div class="stat-card"><div class="stat-value">${result.collected_sources?.length || 0}</div><div class="stat-label">总来源数</div></div>
                <div class="stat-card"><div class="stat-value">${result.extracted_facts?.length || 0}</div><div class="stat-label">总事实数</div></div>
                <div class="stat-card"><div class="stat-value">${(result.active_hypotheses || []).length}</div><div class="stat-label">研究假设</div></div>
                <div class="stat-card"><div class="stat-value">$${(ctx.estimated_cost_usd || 0).toFixed(2)}</div><div class="stat-label">预估成本</div></div>
                <div class="stat-card"><div class="stat-value">${Object.keys(result.coverage_map || {}).length}</div><div class="stat-label">覆盖维度</div></div>
            `;
            
            if (eventSource) { eventSource.close(); }
            if (pollInterval) { clearInterval(pollInterval); }
        }

        function switchTab(tabName) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            event.target.classList.add('active');
            document.getElementById('tab' + tabName.charAt(0).toUpperCase() + tabName.slice(1)).classList.add('active');
        }

        async function cancelTask() {
            if (!currentTaskId) return;
            if (!confirm('确定要取消当前研究任务吗？')) return;
            
            await fetch(`${API_BASE}/api/research/${currentTaskId}`, { method: 'DELETE' });
            if (eventSource) eventSource.close();
            if (pollInterval) clearInterval(pollInterval);
            newResearch();
        }

        function newResearch() {
            currentTaskId = null;
            document.getElementById('submitBtn').disabled = false;
            document.getElementById('submitSection').classList.remove('hidden');
            document.getElementById('progressSection').classList.add('hidden');
            document.getElementById('resultSection').classList.add('hidden');
            document.getElementById('queryInput').value = '';
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
    </script>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
</body>
</html>
```

这是一个功能完整的单页应用，大约 450 行代码。它包含以下几个核心部分：

**提交表单区域**：用户输入研究主题、选择研究深度（快速/标准/深度对应不同轮次）、输出格式和语言。点击"开始研究"后通过 POST 请求创建异步任务。

**实时进度面板**：包含动态进度条（带渐变色和微光动画效果）、四格统计卡片（当前轮次/来源数/事实数）、实时日志滚动区。进度更新优先使用 SSE 推送，如果 SSE 断开会自动降级为 3 秒间隔的轮询模式。

**结果展示区域**：四个标签页——**报告视图**（用 marked.js 将 Markdown 渲染为 HTML）、**事实表格**（结构化展示所有提取的事实及置信度）、**时间线**（完整的研究过程回放）、**统计概览**（六格数据卡片）。顶部还有质量评级徽章（A/B/C/D 不同颜色）。

整个界面采用了暗色主题（深蓝背景 + 靛蓝主色调），视觉风格现代且专业。所有的交互都有平滑的过渡动画，给用户一种"高科技"的感觉。

### Docker 部署配置

最后，让我们把整个应用打包成 Docker 容器，以便在任何环境中一致地运行。

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
```

```yaml
version: '3.8'

services:
  deep-research-api:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - TAVILY_API_KEY=${TAVILY_API_KEY}
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      redis:
        condition: service_started
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.5'
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/dashboard"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    command: redis-server --maxmemory 256mb --maxmemory-policy allkeys-lru
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"

volumes:
  redis_data:
```

Docker Compose 配置中有两个服务：**deep-research-api**（主应用）和 **redis**（用于任务状态共享和缓存）。注意几个生产环境的最佳实践：

- **资源限制**：`deploy.resources.limits` 限制了容器的内存上限为 2GB、CPU 为 1.5 核，防止单个任务耗尽宿主机资源
- **Redis 内存策略**：`maxmemory-policy allkeys-lru` 确保当内存不足时淘汰最少使用的键
- **健康检查**：定期探测 API 是否正常响应
- **workers 数量**：设置为 2 而不是默认的 1，因为研究任务是 CPU 密集型的（大量 LLM 调用），多 worker 可以更好地利用多核

### 扩展方向展望

到这里，DeepResearch 项目已经从一个概念发展成了一个完整可部署的产品。但在实际应用中，还有很多可以继续扩展的方向：

**水平扩展与队列系统**：当前版本用内存字典存储任务状态，只能单机运行。如果要支持高并发和多实例部署，需要引入 Redis 作为共享状态存储，以及 Celery 或 ARQ 这样的分布式任务队列来管理工作节点。

**结果缓存层**：相似的研究主题可能产生高度重叠的结果。引入语义相似度匹配的缓存机制，对于查询相似度超过阈值的新请求可以直接返回缓存的增量更新结果，大幅降低延迟和成本。

**用户个性化**：记录用户的历史研究领域和偏好，逐步构建个人知识图谱。当用户再次提出相关问题时，可以利用已有的知识积累提供更有针对性的研究结果。

**多模态研究能力**：除了文本搜索和网页抓取，还可以接入图像理解（分析图表和数据可视化）、视频转录（从技术演讲中提取信息）、PDF 解析（读取学术论文全文）等更多模态的信息源。

**协作式研究**：支持多人同时对同一主题进行研究，各自贡献不同的视角和信息源，然后合并为一份综合报告。这类似于学术界同行评审的模式，能显著提高研究报告的质量。

### 总结：两个项目的对比与启示

作为 LangGraph 教程的压轴项目，DeepResearch 与第九章的客服工单系统形成了鲜明的对比：

| 维度 | 工单系统 | 研究助手 |
|------|---------|---------|
| 输入特征 | 结构化（消息+元数据） | 开放式（自然语言查询） |
| 执行路径 | 预定义的条件分支 | LLM 驱动的动态决策 |
| 循环模式 | 单次线性流程 | 多轮迭代 Reactor 循环 |
| 核心挑战 | 状态一致性 | 信息充分性判断 |
| 响应时间 | 秒级 | 分钟级 |
| 产出物 | 结构化工单数据 | 长篇研究报告 |
| LangGraph 优势体现 | 条件路由 + 中断 | 状态累积 + 循环控制 |

这两个项目展示了 LangGraph 在两类截然不同的场景中的应用潜力：
- **工单系统**代表了**规则密集型**的工作流——业务逻辑清晰、分支条件明确、需要人机协作中断
- **研究助手**代表了**智能驱动型**的工作流——每一步都需要 LLM 参与决策、路径高度动态、需要复杂的推理和验证

无论哪一类场景，LangGraph 通过其显式的 State 管理、灵活的条件边、可控的循环结构和完善的持久化机制，都提供了比传统 ReAct Agent 更强的表达能力和工程可靠性。这正是它在生产级 AI 应用中被越来越多的团队选择的原因。
