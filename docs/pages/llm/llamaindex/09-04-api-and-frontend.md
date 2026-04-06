# 9.4 API 服务与前后端集成

## 把 RAG 引擎变成可调用的服务

前三节我们完成了需求分析、架构设计和核心模块实现。RAG 引擎本身已经可以工作了——你可以在 Python 里创建 `EnterpriseRAGEngine` 实例、调用 `query()` 方法并得到结果。但一个不能被外部访问的引擎就像一台没有接口的服务器，再强大也无法服务于用户。这一节的目标是：**用 FastAPI 把核心引擎包装成一套完整的 RESTful API 服务，并提供前端集成的示例代码**。

## FastAPI 应用入口与配置

```python
from fastapi import FastAPI, Request, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from contextlib import asynccontextmanager
import logging
import time

from app.config import Settings
from app.dependencies import get_current_user, get_rag_engine
from app.core.rag_engine import EnterpriseRAGEngine, RAGConfig

logger = logging.getLogger(__name__)

rag_engine: EnterpriseRAGEngine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global rag_engine

    logger.info("🚀 企业知识库问答系统启动中...")

    settings = Settings()
    config = RAGConfig(
        embedding_model=settings.embedding_model,
        llm_model=settings.llm_model,
        chunk_size=settings.chunk_size,
        use_reranker=settings.use_reranker,
    )

    rag_engine = EnterpriseRAGEngine(config=config)

    # 预热：检查必要的外部依赖是否可用
    try:
        from app.utils.health_check import health_check
        health = health_check()
        if not health["healthy"]:
            logger.warning(f"健康检查发现问题: {health['issues']}")
        else:
            logger.info("✅ 所有外部依赖正常")
    except Exception as e:
        logger.warning(f"健康检查跳过: {e}")

    logger.info("✅ 系统就绪")
    yield

    logger.info("👋 系统正在关闭...")
    # 清理资源（如有）


app = FastAPI(
    title="企业知识库问答系统 API",
    description="基于 LlamaIndex 的企业级 RAG 问答服务",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

# 注册中间件
setup_middlewares(app)


@app.get("/api/health")
async def health_check_endpoint():
    """健康检查端点——用于负载均衡器和监控探针"""
    return {
        "status": "healthy" if rag_engine else "degraded",
        "engine_ready": rag_engine is not None,
        "timestamp": time.time(),
    }


@app.get("/api/v1/info")
async def system_info():
    """返回系统基本信息和配置概览"""
    if not rag_engine:
        raise HTTPException(status_code=503, detail="引擎尚未初始化")

    stats = rag_engine.get_stats()
    return {
        "name": "Enterprise KB QA System",
        "version": "1.0.0",
        "engine_status": stats.get("status", "unknown"),
        "config": stats.get("config", {}),
    }
```

`lifespan` 是 FastAPI 推荐的应用生命周期管理方式。相比旧的 `on_event("startup")` / `on_event("shutdown")` 装饰器，`lifespan` 上下文管理器提供了更好的类型支持和资源清理保证。在这个例子中，我们在启动时初始化了全局的 `RAGEngine` 实例并做了健康检查；在关闭时可以释放连接池、保存未完成的任务状态等。

注意 `rag_engine` 作为模块级全局变量的设计选择。在单进程模式下这完全没问题——FastAPI 的 async 处理方式本身就支持并发请求共享同一个引擎实例（因为 LlamaIndex 的底层调用大多是 I/O 等待型而非 CPU 密集型的）。但如果未来需要多 worker 部署（Gunicorn + uvicorn），每个 worker 进程会有自己独立的 `rag_engine` 实例，这是安全的但会多占几份内存。如果需要跨 worker 共享引擎状态（比如缓存），就需要引入 Redis 等外部存储。

## 认证与授权中间件

```python
import jwt
import hashlib
import secrets
from datetime import datetime, timedelta
from fastapi import Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional

security = HTTPBearer(auto_error=False)
JWT_SECRET = None  # 从环境变量或配置中心获取
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 8


class AuthError(HTTPException):
    def __init__(self, detail: str = "认证失败"):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            headers={"WWW-Authenticate": "Bearer"},
        )


class PermissionDeniedError(HTTPException):
    def __init__(self, detail: str = "权限不足"):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=detail,
        )


def create_token(user_id: str, username: str, role: str) -> str:
    """生成 JWT Token"""
    payload = {
        "sub": user_id,
        "username": username,
        "role": role,
        "iat": datetime.utcnow(),
        "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS),
        "jti": secrets.token_hex(16),  # JWT ID，用于吊销追踪
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return token


def decode_token(token: str) -> dict:
    """解码并验证 JWT Token"""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise AuthError("Token 已过期，请重新登录")
    except jwt.InvalidTokenError:
        raise AuthError("无效的认证凭证")


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> dict:
    """
    从请求中提取并验证当前用户信息

    这是所有需要认证的端点的依赖注入函数。
    返回包含 user_id, username, role, department 等信息的字典。
    """
    if credentials is None:
        raise AuthError("请提供认证凭证")

    payload = decode_token(credentials.credentials)

    # 可选：从数据库加载最新的用户信息（处理角色变更等）
    user_info = await _load_user_profile(payload["sub"])

    return {
        "user_id": payload["sub"],
        "username": payload.get("username"),
        "role": user_info.get("role", payload.get("role", "regular_user")),
        "department": user_info.get("department", ""),
        "permissions": user_info.get("permissions", []),
        "token_jti": payload.get("jti"),
    }


async def _load_user_profile(user_id: str) -> dict:
    """从数据库加载用户完整信息"""
    try:
        from app.services.user_service import UserService
        service = UserService()
        return await service.get_by_external_id(user_id)
    except Exception as e:
        logger.warning(f"无法加载用户 {user_id} 信息: {e}")
        return {}


async def auth_middleware(request: Request, call_next):
    """认证中间件——对受保护的路径进行鉴权"""

    public_paths = {
        "/api/health",
        "/api/docs",
        "/api/redoc",
        "/api/openapi.json",
        "/api/v1/auth/login",
    }

    if request.url.path in public_paths or request.url.path.startswith("/static"):
        return await call_next(request)

    try:
        auth_header = request.headers.get("authorization", "")
        if not auth_header.startswith("Bearer "):
            raise AuthError()

        token = auth_header[7:]
        payload = decode_token(token)
        request.state.user = payload

    except (AuthError, Exception) as e:
        if request.url.path.startswith("/api/"):
            raise AuthError(str(e))
        else:
            pass  # 前端页面请求不强制认证（由前端路由守卫处理）

    response = await call_next(request)
    return response
```

JWT（JSON Web Token）是无状态的认证方案，非常适合微服务和 API 场景。Token 中包含了用户 ID、角色等基本信息，服务端不需要每次都查数据库来验证身份——只需要验证签名是否有效即可。这里有一个重要的安全细节：**Token 过期时间设为 8 小时而不是常见的 30 天**。原因是企业系统的安全要求更高，短过期时间可以减少 Token 泄露后的风险窗口。如果用户觉得频繁登录太麻烦，可以使用 refresh token 机制——access token 短期（8h），refresh token 长期（7天），access token 过期后用 refresh token 自动续期。

## 核心问答 API

```python
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List


class ChatRequest(BaseModel):
    """聊天请求模型"""
    message: str = Field(..., min_length=1, max_length=2000, description="用户问题")
    session_id: Optional[str] = Field(None, description="会话ID（用于多轮对话）")
    stream: bool = Field(False, description="是否使用流式输出")
    context: Optional[dict] = Field(None, description="额外上下文信息")

    @field_validator('message')
    @classmethod
    def validate_message(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 1:
            raise ValueError('问题不能为空')
        dangerous_patterns = [
            '忽略', 'Ignore previous',
            'system prompt', 'System prompt',
            '<script>', 'javascript:',
        ]
        lower_v = v.lower()
        for pattern in dangerous_patterns:
            if pattern.lower() in lower_v:
                raise ValueError('输入内容包含不安全的模式')
        return v


class CitationItem(BaseModel):
    """引用条目"""
    id: str
    title: str
    source_type: str
    section: Optional[str] = None
    snippet: Optional[str] = None
    relevance_score: float


class SourceItem(BaseModel):
    """来源条目"""
    doc_id: str
    title: str
    score: float
    snippet: str


class ChatResponse(BaseModel):
    """聊天响应模型"""
    answer: str
    session_id: str
    citations: List[CitationItem] = []
    sources: List[SourceItem] = []
    confidence: float
    latency_ms: float
    query_type: str = ""
    suggested_questions: List[str] = []


class ChatMessage(BaseModel):
    """聊天消息记录"""
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str
    timestamp: float = 0


# 内存中的会话存储（生产环境应替换为 Redis）
_chat_sessions: dict[str, list[ChatMessage]] = {}
SESSION_TIMEOUT_SECONDS = 1800  # 30分钟


@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    核心问答接口

    接收用户问题，经过完整的 RAG 流水线处理后返回回答。

    - 支持流式输出（SSE）：设置 stream=true
    - 支持多轮对话：传入 session_id
    - 自动权限过滤：根据用户角色过滤检索结果
    """
    start_time = time.perf_counter()

    if not rag_engine:
        raise HTTPException(
            status_code=503,
            detail="知识库引擎尚未就绪，请联系管理员",
        )

    # 会话管理
    session_id = request.session_id or _create_session_id()
    chat_history = _get_or_create_session(session_id)

    if request.stream:
        return StreamingResponse(
            _stream_response(request.message, session_id, chat_history, current_user),
            media_type="text/event-stream",
            headers={
                "X-Session-ID": session_id,
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    result = await rag_engine.query(
        question=request.message,
        user_context={
            "department": current_user.get("department", ""),
            "role": current_user.get("role", ""),
        },
        chat_history=[
            {"question": m.content, "answer": ""}
            for m in chat_history[-6:] if m.role == "user"
        ],
        allowed_doc_ids=current_user.get("allowed_doc_ids"),
        stream=False,
    )

    # 更新会话历史
    chat_history.append(ChatMessage(role="user", content=request.message))
    chat_history.append(ChatMessage(role="assistant", content=result.answer))

    return ChatResponse(
        answer=result.answer,
        session_id=session_id,
        citations=result.citations,
        sources=result.sources,
        confidence=result.confidence,
        latency_ms=result.latency_ms,
        query_type=result.retrieval_details.get("query_type", ""),
        suggested_questions=_generate_suggestions(request.message),
    )


async def _stream_response(
    question: str,
    session_id: str,
    chat_history: list[ChatMessage],
    current_user: dict,
):
    """流式响应生成器（Server-Sent Events 格式）"""
    if not rag_engine:
        yield f"data: {json.dumps({'error': '引擎未就绪'}, ensure_ascii=False)}\n\n"
        return

    try:
        result = await rag_engine.query(
            question=question,
            user_context={"department": current_user.get("department", "")},
            chat_history=[{"question": m.content} for m in chat_history[-4:] if m.role == "user"],
            allowed_doc_ids=current_user.get("allowed_doc_ids"),
            stream=True,
        )

        # SSE 格式的流式输出
        yield f"data: {json.dumps({'type': 'start', 'session_id': session_id}, ensure_ascii=False)}\n\n"

        for chunk in result.response.gen:  # 假设流式响应有 gen 属性
            chunk_data = {
                "type": "chunk",
                "content": chunk.delta if hasattr(chunk, 'delta') else str(chunk),
                "finish_reason": None,
            }
            yield f"data: {json.dumps(chunk_data, ensure_ascii=False)}\n\n"

        yield f"data: {json.dumps({'type': 'done', 'citations': result.citations, 'confidence': result.confidence}, ensure_ascii=False)}\n\n"

    except Exception as e:
        error_data = {"type": "error", "message": str(e)}
        yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"


def _create_session_id() -> str:
    """生成新的会话 ID"""
    import uuid
    return f"sess_{uuid.uuid4().hex[:12]}"


def _get_or_create_session(session_id: str) -> list[ChatMessage]:
    """获取或创建会话历史"""
    global _chat_sessions
    if session_id not in _chat_sessions:
        _chat_sessions[session_id] = []
    return _chat_sessions[session_id]


def _generate_suggestions(question: str) -> list[str]:
    """基于当前问题生成推荐相关问题"""
    suggestions_map = {
        "年假": ["年假如何计算？", "年假未休完怎么处理？", "新员工年假政策"],
        "报销": ["差旅报销流程", "报销需要哪些材料？", "报销审批周期多久？"],
        "工资": ["工资发放时间", "工资条哪里查看？", "个税怎么扣？"],
        "请假": ["病假怎么请？", "事假扣工资吗？", "产假多少天？"],
        "VPN": ["VPN 怎么连接？", "VPN 连不上怎么办？", "远程办公 VPN 设置"],
        "API": ["API 文档在哪里？", "API Key 如何申请？", "API 限频规则"],
    }

    for keyword, suggs in suggestions_map.items():
        if keyword in question:
            return suggs

    return []
```

这个问答 API 有几个值得深入讨论的设计点：

### Pydantic 模型的双重作用

`ChatRequest` 和 `ChatResponse` 不只是数据容器——它们同时承担了三个职责：

1. **请求/响应的自动校验**：`min_length=1, max_length=2000` 确保问题不会太短或太长；`field_validator` 做更复杂的内容安全检查（检测 prompt injection 尝试）。如果客户端发送了不符合规范的数据，FastAPI 会自动返回 422 Unprocessable Entity 错误，而不会让这些数据进入到业务逻辑层。

2. **OpenAPI 文档的自动生成**：FastAPI 会根据 Pydantic 模型自动生成 API 文档的字段说明、类型信息和示例值。你在 `/api/docs` 页面看到的交互式文档就是从这里来的。

3. **前端类型提示的基础**：如果你的前端也是 TypeScript 项目，可以用工具（如 `openapi-typescript-codegen`）直接从这些模型生成前端的类型定义，确保前后端的接口契约一致。

### 流式输出的 SSE 实现

`_stream_response` 函数是一个 Python async generator，它以 Server-Sent Events (SSE) 格式逐步输出回答片段。SSE 的协议非常简单——每条消息就是 `data: <JSON>\n\n`。我们定义了三种消息类型：
- `start`：通知前端流开始，附上 session_id
- `chunk`：携带实际的文本片段（来自 LLM 的增量输出）
- `done`：流结束，附带引用列表和置信度

前端接收端的 JavaScript 代码大致如下：

```javascript
// 前端 SSE 消费示例
async function streamChat(question, sessionId, token) {
  const response = await fetch('/api/v1/chat', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${token}`,
    },
    body: JSON.stringify({
      message: question,
      session_id: sessionId,
      stream: true,
    }),
  });

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let fullAnswer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    const text = decoder.decode(value);
    const lines = text.split('\n').filter(line => line.startsWith('data: '));

    for (const line of lines) {
      const data = JSON.parse(line.slice(6));
      if (data.type === 'start') {
        sessionId = data.session_id;
      } else if (data.type === 'chunk') {
        fullAnswer += data.content;
        appendToChatUI(data.content);  // 追加到界面
      } else if (data.type === 'done') {
        showCitations(data.citations);
        showConfidence(data.confidence);
      } else if (data.type === 'error') {
        showError(data.message);
      }
    }
  }
}
```

### 会话管理的内存实现与生产化注意事项

上面的 `_chat_sessions` 用 Python 字典存储会话历史，这在开发阶段够用了但在生产环境中存在严重问题：（1）服务重启后所有会话丢失；（2）多实例部署时各实例的会话不共享；（3）内存会随用户增长无限膨胀。生产环境应该用 Redis 替代：

```python
import json
import redis.asyncio as aioredis

redis_client: aioredis.Redis = None

SESSION_PREFIX = "kb:session:"
SESSION_MAX_MESSAGES = 20


async def redis_get_session(session_id: str) -> list[dict]:
    key = SESSION_PREFIX + session_id
    data = await redis_client.get(key)
    if data:
        return json.loads(data)
    return []


async def redis_save_session(session_id: str, messages: list[dict]):
    key = SESSION_PREFIX + session_id
    truncated = messages[-SESSION_MAX_MESSAGES:]
    await redis_client.setex(
        key,
        SESSION_TIMEOUT_SECONDS,
        json.dumps(truncated, ensure_ascii=False),
    )
```

Redis 版本还自然地解决了 TTL 过期问题——`setex` 直接设置了 30 分钟的超时，到期后键自动删除，不需要额外的清理逻辑。

## 数据源管理 API

```python
from pydantic import BaseModel, Field
from typing import Optional, Literal
from enum import Enum


class DataSourceCreateRequest(BaseModel):
    name: str = Field(..., min_length=2, max_length=100)
    source_type: Literal[
        "file_directory", "database_postgres", "database_mysql"
    ] = Field(..., description="数据源类型")
    config: dict = Field(..., description="数据源连接配置")
    classification: str = Field("internal", description="数据密级")


class DataSourceResponse(BaseModel):
    id: int
    name: str
    source_type: str
    classification: str
    document_count: int
    last_sync_at: Optional[str]
    sync_status: str
    created_at: str


@app.post(
    "/api/v1/data-sources",
    response_model=DataSourceResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_data_source(
    request: DataSourceCreateRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    创建新数据源

    权限要求：knowledge_manager 或 admin
    """
    if current_user["role"] not in ("admin", "knowledge_manager"):
        raise PermissionDeniedError("只有管理员可以管理数据源")

    from app.services.data_source_service import DataSourceService
    service = DataSourceService()

    ds = await service.create(
        name=request.name,
        source_type=request.source_type,
        config=request.config,
        classification=request.classification,
        created_by=current_user["user_id"],
    )

    return DataSourceResponse(**ds.to_dict())


@app.get("/api/v1/data-sources")
async def list_data_sources(
    skip: int = 0,
    limit: int = 20,
    current_user: dict = Depends(get_current_user),
):
    """列出当前用户可见的数据源"""
    from app.services.data_source_service import DataSourceService
    service = DataSourceService()

    sources = await service.list_for_user(
        user_role=current_user["role"],
        user_id=current_user["user_id"],
        skip=skip,
        limit=limit,
    )
    return {"items": [DataSourceResponse(**s.to_dict()) for s in sources], "total": len(sources)}


@app.post("/api/v1/data-sources/{source_id}/sync")
async def trigger_sync(
    source_id: int,
    force_rebuild: bool = False,
    current_user: dict = Depends(get_current_user),
):
    """
    触发指定数据源的同步和索引构建

    这是一个异步操作——立即返回任务ID，
    通过 GET /api/v1/tasks/{task_id} 查询进度。
    """
    if current_user["role"] not in ("admin", "knowledge_manager"):
        raise PermissionDeniedError()

    from app.celery_app import build_index_task
    task = build_index_task.delay(source_id=source_id, force_rebuild=force_rebuild)

    return {
        "task_id": task.id,
        "status": "pending",
        "message": f"同步任务已提交，可通过 /api/v1/tasks/{task.id} 查看进度",
    }


@app.get("/api/v1/tasks/{task_id}")
async def get_task_status(task_id: str, current_user: dict = Depends(get_current_user)):
    """查询异步任务状态"""
    from celery.result import AsyncResult
    from app.celery_app import celery_app

    result = AsyncResult(task_id, app=celery_app)

    status_map = {
        "PENDING": "等待中",
        "STARTED": "执行中",
        "SUCCESS": "完成",
        "FAILURE": "失败",
        "RETRY": "重试中",
        "REVOKED": "已取消",
    }

    response = {
        "task_id": task_id,
        "status": status_map.get(result.state, result.state),
    }

    if result.ready():
        if result.successful():
            response["result"] = result.result
        else:
            response["error"] = str(result.result) if result.result else "未知错误"

    return response
```

数据源管理 API 遵循 RESTful 设计风格：POST 创建、GET 列表、触发操作也用 POST（因为同步不是幂等的）。`trigger_sync` 端点返回的是 Celery 任务 ID 而不是直接等待同步完成——这是因为全量索引重建可能需要几分钟甚至更长时间，HTTP 请求不可能等那么久。前端拿到 task_id 后可以轮询 `/tasks/{task_id}` 或者通过 WebSocket 接收完成通知。

## 分析统计 API

```python
from datetime import datetime, timedelta
from typing import Optional


class AnalyticsQueryParams(BaseModel):
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    department: Optional[str] = None
    group_by: Optional[str] = Field("day", pattern="^(hour|day|week|month)$")


@app.get("/api/v1/analytics/overview")
async def analytics_overview(
    days: int = 7,
    current_user: dict = Depends(get_current_user),
):
    """查询分析概览——仪表盘首页数据"""
    if current_user["role"] not in ("admin", "knowledge_manager"):
        raise PermissionDeniedError()

    from app.services.analytics_service import AnalyticsService
    service = AnalyticsService()

    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)

    overview = await service.get_overview(start_date, end_date)
    return overview


@app.get("/api/v1/analytics/top-queries")
async def top_queries(
    limit: int = 20,
    days: int = 7,
    current_user: dict = Depends(get_current_user),
):
    """热门问题排行"""
    if current_user["role"] not in ("admin", "knowledge_manager"):
        raise PermissionDeniedError()

    from app.services.analytics_service import AnalyticsService
    service = AnalyticsService()
    return await service.get_top_queries(limit=limit, days=days)


@app.get("/api/v1/analytics/gap-analysis")
async def gap_analysis(
    days: int = 7,
    min_confidence: float = 0.3,
    current_user: dict = Depends(get_current_user),
):
    """
    差距分析——找出用户问了但系统答不好的问题

    这些问题是优化知识库内容的优先方向：
    低置信度 + 高频出现 = 最需要补充的内容
    """
    if current_user["role"] not in ("admin", "knowledge_manager"):
        raise PermissionDeniedError()

    from app.services.analytics_service import AnalyticsService
    service = AnalyticsService()
    gaps = await service.find_gaps(days=days, min_confidence=min_confidence)
    return {
        "gaps": gaps,
        "summary": {
            "total_analyzed": len(gaps),
            "priority_count": sum(1 for g in gaps if g["frequency"] >= 5 and g["avg_confidence"] < 0.4),
        },
    }
```

差距分析（Gap Analysis）是企业知识库运营中最有价值的洞察之一。它的思路很简单：找出那些**高频出现且低置信度**的问题——这意味着很多用户都在问这个问题，但系统每次都回答得不好。这些问题指向的就是知识库中缺失或者描述不清的内容。产品经理看到这个报告后就知道该优先补充哪方面的文档了。

## 前端集成示例

下面给出一个精简版的前端 Vue 3 组件，展示如何与后端 API 对接：

```vue
<template>
  <div class="kb-chat-container">
    <!-- 侧边栏 -->
    <aside class="sidebar">
      <div class="logo">📚 企业知识库</div>
      <nav class="nav-menu">
        <div class="nav-item active">
          <span>💬 智能问答</span>
        </div>
        <div class="nav-item" v-if="isAdmin">
          <span>📊 数据分析</span>
        </div>
        <div class="nav-item" v-if="isAdmin">
          <span>⚙️ 数据源管理</span>
        </div>
      </nav>
      <div class="user-info">
        <div class="avatar">{{ userName.charAt(0) }}</div>
        <div class="user-detail">
          <div class="name">{{ userName }}</div>
          <div class="role">{{ roleLabel }}</div>
        </div>
      </div>
    </aside>

    <!-- 主聊天区域 -->
    <main class="chat-main">
      <header class="chat-header">
        <h2>智能问答助手</h2>
        <button class="new-chat-btn" @click="startNewSession">+ 新对话</button>
      </header>

      <div class="messages-container" ref="messagesContainer">
        <div v-if="messages.length === 0" class="welcome-screen">
          <div class="welcome-icon">🔍</div>
          <h3>有什么我可以帮您的？</h3>
          <p>试试问这些问题：</p>
          <div class="suggestion-chips">
            <button
              v-for="s in welcomeSuggestions"
              :key="s"
              class="chip"
              @click="sendSuggestion(s)"
            >{{ s }}</button>
          </div>
        </div>

        <div
          v-for="(msg, idx) in messages"
          :key="idx"
          :class="['message-row', msg.role]"
        >
          <div class="avatar">
            {{ msg.role === 'user' ? '👤' : '🤖' }}
          </div>
          <div class="message-content">
            <div class="text" v-html="formatText(msg.content)"></div>

            <!-- 引用卡片 -->
            <div v-if="msg.citations && msg.citations.length > 0" class="citations">
              <div class="citation-label">📎 参考来源</div>
              <div
                v-for="cite in msg.citations"
                :key="cite.id"
                class="citation-card"
              >
                <div class="cite-title">{{ cite.title }}</div>
                <div class="cite-snippet">{{ cite.snippet }}</div>
                <div class="cite-meta">
                  <span class="badge">{{ cite.source_type }}</span>
                  <span class="score">相关度 {{ (cite.relevance_score * 100).toFixed(0) }}%</span>
                </div>
              </div>
            </div>

            <!-- 操作按钮 -->
            <div v-if="msg.role === 'assistant'" class="message-actions">
              <button @click="copyMessage(msg)" class="action-btn">📋 复制</button>
              <button @click="likeMessage(msg)" class="action-btn">👍</button>
              <button @click="dislikeMessage(msg)" class="action-btn">👎</button>
            </div>
          </div>

          <div class="meta" v-if="msg.latency">
            ⏱️ {{ msg.latency.toFixed(1) }}s · 置信度 {{ (msg.confidence * 100).toFixed(0) }}%
          </div>
        </div>

        <!-- 加载指示器 -->
        <div v-if="isLoading" class="typing-indicator">
          <span></span><span></span><span></span>
          <span class="loading-text">正在思考...</span>
        </div>
      </div>

      <!-- 输入区 -->
      <div class="input-area">
        <textarea
          v-model="inputText"
          placeholder="输入您的问题..."
          rows="1"
          @keydown.enter.exact="handleSend"
          @input="autoResize"
          :disabled="isLoading"
          ref="inputRef"
        ></textarea>
        <button
          class="send-btn"
          @click="handleSend"
          :disabled="!inputText.trim() || isLoading"
        >
          发送
        </button>
      </div>
    </main>
  </div>
</template>

<script setup>
import { ref, nextTick, onMounted, computed } from 'vue'
import axios from 'axios'

const api = axios.create({ baseURL: '/api/v1' })
const messages = ref([])
const inputText = ref('')
const isLoading = ref(false)
const sessionId = ref(null)
const messagesContainer = ref(null)
const inputRef = ref(null)

const userName = computed(() => localStorage.getItem('username') || '用户')
const isAdmin = computed(() => localStorage.getItem('role') === 'admin')
const roleLabel = computed(() => {
  const map = { admin: '管理员', knowledge_manager: '知识管理员', regular_user: '员工' }
  return map[localStorage.getItem('role')] || '普通用户'
})

const welcomeSuggestions = [
  '公司年假政策是怎样的？',
  '如何申请差旅报销？',
  'Pro 版和企业版有什么区别？',
  'API 返回 403 错误怎么办？',
]

function formatText(text) {
  return text
    .replace(/\n/g, '<br>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
}

function autoResize() {
  const el = inputRef.value
  el.style.height = 'auto'
  el.style.height = Math.min(el.scrollHeight, 120) + 'px'
}

async function handleSend() {
  const text = inputText.value.trim()
  if (!text || isLoading.value) return

  messages.value.push({ role: 'user', content: text })
  inputText.value = ''
  isLoading.value = true

  scrollToBottom()

  try {
    const response = await api.post('/chat', {
      message: text,
      session_id: sessionId.value,
      stream: false,
    }, {
      headers: { Authorization: `Bearer ${getToken()}` },
    })

    const data = response.data
    sessionId.value = data.session_id

    messages.value.push({
      role: 'assistant',
      content: data.answer,
      citations: data.citations,
      confidence: data.confidence,
      latency: data.latency_ms / 1000,
    })

    if (data.suggested_questions?.length > 0) {
      // 可以展示推荐问题
    }
  } catch (err) {
    const errorMsg = err.response?.data?.detail || '抱歉，出了点问题，请稍后重试'
    messages.value.push({ role: 'assistant', content: `❌ ${errorMsg}` })
  } finally {
    isLoading.value = false
    scrollToBottom()
  }
}

async function sendSuggestion(text) {
  inputText.value = text
  handleSend()
}

function copyMessage(msg) {
  navigator.clipboard.writeText(msg.content)
}

function likeMessage(msg) {
  api.post('/feedback', {
    session_id: sessionId.value,
    message_content: msg.content,
    feedback: 1,
  })
}

function dislikeMessage(msg) {
  api.post('/feedback', {
    session_id: sessionId.value,
    message_content: msg.content,
    feedback: -1,
  })
}

function startNewSession() {
  messages.value = []
  sessionId.value = null
}

function getToken() {
  return localStorage.getItem('token') || ''
}

function scrollToBottom() {
  nextTick(() => {
    if (messagesContainer.value) {
      messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight
    }
  })
}
</script>
```

这个 Vue 组件虽然简化了很多样式和边界情况处理，但它展示了前后端集成的几个关键交互模式：

1. **消息流的渲染**：用户消息右对齐、AI 消息左对齐，引用信息折叠在 AI 回复下方
2. **自动滚动**：每次新消息到达后自动滚到底部（`scrollToBottom` + `nextTick`）
3. **反馈收集**：每个 AI 回复都有 👍👎 按钮，点击后异步提交到后端用于质量评估
4. **欢迎页建议**：首次打开时显示几个热门问题作为引导，降低用户的冷启动门槛
5. **输入框自适应高度**：随着输入文字增多自动变高（最大 120px），避免固定高度的输入框在小屏幕上体验不佳

## 总结

