---
title: 将 LangChain 应用封装为 API 服务（FastAPI）
description: 从脚本到服务的架构转变、FastAPI 项目结构设计、同步/异步接口、流式 SSE 端点、错误处理与中间件
---
# 将 LangChain 应用封装为 API 服务（FastAPI）

到目前为止，我们构建的所有应用——客服系统、代码分析助手、数据分析 Agent——都是以 **CLI 脚本**或 **Jupyter Notebook** 的形式运行的。这种模式适合开发和调试，但要真正让用户使用（尤其是非技术用户），我们需要把它封装成一个**Web API 服务**。

这一节的目标是：把前面构建的 LangChain 应用变成一个**生产级的 HTTP API**，让任何客户端（网页前端、移动 App、第三方系统）都能通过标准协议调用。

## 为什么需要 API 封装

先理解从"脚本"到"服务"的转变意味着什么：

| 维度 | CLI 脚本 | API 服务 |
|------|---------|---------|
| **交互方式** | 命令行输入/输出 | HTTP 请求/响应 |
| **并发能力** | 单用户 | 多用户同时访问 |
| **生命周期** | 运行一次就退出 | 持续运行，处理无数请求 |
| **部署方式** | 本地执行 | 服务器上 7×24 运行 |
| **可观测性** | print 输出 | 结构化日志 + 监控指标 |
| **错误处理** | try-except 打印 | HTTP 状态码 + 标准化错误格式 |

核心挑战在于：**LLM 调用是慢的（2-10 秒）、昂贵的（每次几分钱到几毛钱）、不可靠的（可能超时或报错）**。API 服务需要优雅地处理这些特性。

## FastAPI 项目结构

一个生产级的 FastAPI + LangChain 项目应该有清晰的结构：

```
api_service/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI 应用入口
│   ├── config.py            # 配置管理
│   ├── dependencies.py      # 依赖注入（数据库连接等）
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── chat.py          # 对话接口
│   │   ├── health.py        # 健康检查
│   │   └── admin.py         # 管理接口
│   ├── services/
│   │   ├── __init__.py
│   │   ├── cs_bot.py        # 客服机器人业务逻辑
│   │   ├── code_agent.py    # 代码分析 Agent
│   │   └── data_analyst.py  # 数据分析 Agent
│   ├── models/
│   │   ├── __init__.py
│   │   ├── requests.py      # 请求模型 (Pydantic)
│   │   └── responses.py     # 响应模型
│   ├── middleware/
│   │   ├── __init__.py
│   │   ├── rate_limiter.py  # 限流中间件
│   │   ├── logger.py        # 日志中间件
│   │   └── error_handler.py # 全局异常处理
│   └── utils/
│       ├── __init__.py
│       ├── cache.py         # 缓存工具
│       └── metrics.py       # 指标收集
├── tests/
│   ├── test_chat.py
│   └── test_health.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env                     # 环境变量（不提交到 Git）
```

## 核心实现：FastAPI 应用入口

### main.py — 应用骨架

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
import logging

from app.config import Settings
from app.routers import chat, health, admin
from app.middleware.logger import LoggingMiddleware
from app.middleware.error_handler import register_error_handlers

logger = logging.getLogger("api_service")
settings = Settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 服务启动中...")
    logger.info(f"环境: {settings.env}")
    logger.info(f"LLM 模型: {settings.llm_model}")
    yield
    logger.info("👋 服务关闭")

app = FastAPI(
    title="LangChain API Service",
    version="1.0.0",
    description="基于 LangChain 的智能助手 API",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(LoggingMiddleware)

register_error_handlers(app)

app.include_router(health.router, prefix="/health", tags=["健康检查"])
app.include_router(chat.router, prefix="/v1/chat", tags=["对话"])
app.include_router(admin.router, prefix="/v1/admin", tags=["管理"])

@app.get("/")
async def root():
    return {
        "service": "LangChain API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }
```

### config.py — 配置管理

```python
from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    env: str = "development"
    debug: bool = False

    openai_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 2048

    cors_origins: List[str] = ["http://localhost:3000"]

    rate_limit_per_minute: int = 30
    request_timeout_seconds: int = 60
    max_concurrent_requests: int = 100

    redis_url: str = ""
    chroma_persist_dir: str = "./data/chroma_db"

    langchain_tracing_v2: bool = False
    langchain_project: str = "langchain-api-prod"

    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
```

### 同步聊天接口

```python
# app/routers/chat.py
from fastapi import APIRouter, Depends, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional
import time

from app.models.requests import ChatRequest
from app.models.responses import ChatResponse, ErrorResponse
from app.services.cs_bot import CustomerServiceBot

router = APIRouter()
bot_instance = None

def get_bot() -> CustomerServiceBot:
    global bot_instance
    if bot_instance is None:
        bot_instance = CustomerServiceBot()
        bot_instance.initialize()
    return bot_instance

@router.post("", response_model=ChatResponse, summary="发送消息并获取回复")
async def chat(request: ChatRequest):
    """标准聊天接口：发送消息，等待完整回复后返回。

    适用于：需要完整回答的场景（如后台任务、批量处理）

    - **question**: 用户的消息内容（必填，1-5000 字符）
    - **session_id**: 会话 ID（可选，用于多轮对话上下文）
    """
    start_time = time.time()

    try:
        bot = get_bot()
        result = bot.process_message(
            user_input=request.question,
            session_id=request.session_id,
        )

        latency_ms = (time.time() - start_time) * 1000

        return ChatResponse(
            response=result["response"],
            session_id=result["session_id"],
            intent=result.get("intent", "unknown"),
            handoff=result.get("handoff", False),
            latency_ms=round(latency_ms, 1),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### 流式聊天接口（SSE）

这是 LLM API 最关键的接口——**流式输出**能大幅降低用户的感知延迟：

```python
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse
import json

@router.post("/stream", summary="流式聊天（SSE）")
async def chat_stream(request: ChatRequest):
    """流式聊天接口：以 Server-Sent Events 格式逐 token 返回回复。

    适用于：实时对话场景（如 Web 聊天窗口、移动端对话）

    客户端示例:
    ```javascript
    const eventSource = new EventSource('/v1/chat/stream?question=你好');
    eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);
        // data.token: 当前 token 文本
        // data.done: 是否结束
    };
    ```
    """

    async def event_generator():
        bot = get_bot()
        full_response = ""

        yield {"event": "start", "data": json.dumps({"session_id": request.session_id or "new"})}

        try:
            chain = bot.chain
            async for chunk in chain.astream({
                "question": request.question,
                "chat_history": [],
            }):
                content = chunk.content
                if content:
                    full_response += content
                    yield {
                        "event": "token",
                        "data": json.dumps({"token": content})
                    }

            yield {
                "event": "done",
                "data": json.dumps({
                    "full_response": full_response,
                    "session_id": request.session_id or "new",
                })
            }

        except Exception as e:
            yield {
                "event": "error",
                "data": json.dumps({"error": str(e)})
            }

    return EventSourceResponse(
        event_generator(),
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
```

### 请求/响应模型

```python
# app/models/requests.py
from pydantic import BaseModel, Field, field_validator
import re

class ChatRequest(BaseModel):
    question: str = Field(
        ...,
        description="用户输入的问题或消息",
        min_length=1,
        max_length=5000,
    )
    session_id: Optional[str] = Field(
        default=None,
        description="会话标识符，用于保持多轮对话上下文",
        pattern=r"^[a-zA-Z0-9_-]{1,64}$",
    )
    stream: bool = Field(
        default=False,
        description="是否使用流式输出",
    )

    @field_validator('question')
    @classmethod
    def sanitize_question(cls, v: str) -> str:
        v = v.strip()
        v = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', v)
        return v


class BatchChatRequest(BaseModel):
    questions: list[str] = Field(..., min_length=1, max_length=20)
```

```python
# app/models/responses.py
from pydantic import BaseModel, Field
from typing import Optional, Any
from datetime import datetime

class ChatResponse(BaseModel):
    response: str = Field(description="AI 生成的回复内容")
    session_id: str = Field(description="会话 ID（可用于后续请求关联上下文）")
    intent: Optional[str] = Field(default=None, description="识别到的意图类型")
    handoff: bool = Field(default=False, description="是否触发了人工接管")
    latency_ms: float = Field(description="服务器处理耗时（毫秒）")
    model: Optional[str] = Field(default=None, description="使用的 LLM 模型名称")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

class ErrorResponse(BaseModel):
    error: str = Field(description="错误信息")
    error_code: str = Field(description="错误代码")
    detail: Optional[Any] = Field(default=None, description="额外详情")
    request_id: Optional[str] = Field(default=None)
```

### 批量接口

```python
@router.post("/batch", summary="批量查询")
async def chat_batch(request: BatchChatRequest):
    """批量聊天接口：一次请求处理多个问题。
    注意：每个问题独立处理，不共享上下文。
    最大支持 20 个问题并行处理。
    """
    import asyncio

    bot = get_bot()
    semaphore = asyncio.Semaphore(5)

    async def process_one(question: str) -> dict:
        async with semaphore:
            result = bot.process_message(user_input=question)
            return {"question": question, "response": result["response"]}

    tasks = [process_one(q) for q in request.questions]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    responses = []
    errors = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            errors.append({"index": i, "error": str(result)})
        else:
            responses.append(result)

    return {
        "responses": responses,
        "errors": errors if errors else None,
        "total": len(request.questions),
        "success_count": len(responses),
        "error_count": len(errors),
    }
```

## 中间件：日志、限流与错误处理

### 日志中间件

```python
# app/middleware/logger.py
import time
import logging
import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("api_service.request")

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id

        start = time.time()
        method = request.method
        path = url.path = request.url.path

        logger.info(f"[{request_id}] → {method} {path}")

        response: Response = await call_next(request)

        duration_ms = (time.time() - start) * 1000
        status_code = response.status_code

        log_level = logging.WARNING if status_code >= 400 else logging.INFO
        logger.log(log_level,
                   f"[{request_id}] ← {status_code} {method} {path} "
                   f"{duration_ms:.0f}ms")

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = f"{duration_ms:.1f}ms"

        return response
```

### 限流中间件

```python
# app/middleware/rate_limiter.py
import time
from collections import defaultdict
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

class InMemoryRateLimiter(BaseHTTPMiddleware):
    def __init__(self, app, max_requests: int = 30, window_seconds: int = 60):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()

        window_start = now - self.window_seconds
        self.requests[client_ip] = [
            t for t in self.requests[client_ip] if t > window_start
        ]

        if len(self.requests[client_ip]) >= self.max_requests:
            retry_after = int(self.window_seconds - (now - self.requests[client_ip][0]))
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "Rate limit exceeded",
                    "retry_after": retry_after,
                    "limit": self.max_requests,
                },
            )

        self.requests[client_ip].append(now)
        return await call_next(request)
```

### 全局异常处理

```python
# app/middleware/error_handler.py
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import traceback

def register_error_handlers(app: FastAPI):

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request, exc):
        return JSONResponse(
            status_code=422,
            content={
                "error": "请求参数验证失败",
                "error_code": "VALIDATION_ERROR",
                "detail": exc.errors(),
            },
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        trace = traceback.format_exc()
        print(trace)

        if isinstance(exc, HTTPException):
            return JSONResponse(
                status_code=exc.status_code,
                content={"error": exc.detail},
            )

        return JSONResponse(
            status_code=500,
            content={
                "error": "内部服务器错误",
                "error_code": "INTERNAL_ERROR",
            },
        )
```

### 健康检查端点

```python
# app/routers/health.py
from fastapi import APIRouter
from datetime import datetime
import os

router = APIRouter()

@router.get("")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "environment": os.getenv("ENV", "unknown"),
    }

@router.get("/ready")
async def readiness_check():
    ready_checks = {}

    try:
        from app.services.cs_bot import CustomerServiceBot
        bot = CustomerServiceBot()
        bot.initialize()
        ready_checks["llm_connection"] = True
    except Exception:
        ready_checks["llm_connection"] = False

    all_ready = all(ready_checks.values())
    return {
        "ready": all_ready,
        "checks": ready_checks,
    }
```

## 启动与测试

```bash
cd api_service
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2
```

测试接口：

```bash
# 标准聊天
curl -X POST http://localhost:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "免费版支持几个人？"}'

# 流式聊天
curl -N http://localhost:8000/v1/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "免费版和专业版有什么区别？"}'

# 健康检查
curl http://localhost:8000/health
curl http://localhost:8000/health/ready
```

API 文档会自动生成在 `http://localhost:8000/docs`（Swagger UI），可以直接在浏览器中交互式测试所有端点。

## 常见误区

**误区一：用同步接口处理 LLM 调用**。LLM 调动通常需要 2-10 秒，如果用同步的 `/v1/chat` 接口且没有设置合理的超时，高并发时线程池会被耗尽。生产环境应该**优先使用流式接口**或在同步接口中设置较短超时。

**误区二：每次请求都重新初始化 Bot**。`CustomerServiceBot().initialize()` 包含加载知识库、创建向量索引等耗时操作。应该在应用启动时初始化一次（通过 lifespan 或模块级变量），后续请求复用实例。

**误区三：忽略请求大小限制**。恶意用户可能发送超长文本（几十万字）导致 Token 用量爆炸和账单飙升。必须在 Pydantic 模型和中间件两层都做长度校验。
