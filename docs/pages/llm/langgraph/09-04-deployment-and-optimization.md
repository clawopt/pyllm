# 4. 部署与优化

## 从开发到生产：让工单系统真正跑起来

经过前面三个小节的努力，我们已经完成了一个功能完整的智能客服工单系统——从状态设计、节点实现、图组装，到 API 服务层和前端界面。但到目前为止，所有的代码都还只是在本地环境中运行，用内存字典存储数据，用单线程处理请求。当我们要把这个系统真正部署到生产环境、面对真实用户的时候，就需要考虑一系列工程化的问题：如何容器化打包？如何编排多个服务？如何在高并发下保持性能稳定？如何做灰度发布和回滚？这一节我们就来系统地解决这些问题。

### Docker 容器化：把应用装进标准化的"集装箱"

Docker 是现代应用部署的事实标准，它通过容器技术把应用及其所有依赖打包成一个独立的运行单元，确保在任何环境下都能一致地运行。对于我们的工单系统来说，容器化不仅能解决"在我机器上能跑"的经典问题，还能为后续的横向扩展和自动化运维打下基础。

首先我们需要为 LangGraph 应用创建一个 `Dockerfile`。这里的关键点在于：我们的应用依赖 Python 环境、需要安装一系列 pip 包（langgraph、langchain-openai、fastapi、uvicorn 等），而且还需要在容器启动时正确地暴露端口并运行服务。

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

这个 Dockerfile 看起来很简单，但有几个值得注意的细节。我们选择了 `python:3.11-slim` 作为基础镜像而不是完整的 `python:3.11`，这是因为 slim 版本去掉了许多不必要的系统包，镜像体积更小（大约 150MB vs 1GB），安全攻击面也更小。`--no-cache-dir` 参数告诉 pip 在安装完包之后不保留下载缓存，进一步减小镜像体积。`WORKDIR /app` 设置了工作目录，这样后续的所有 COPY 和 RUN 命令都会在这个目录下执行。最后，`CMD` 指定了容器启动时默认执行的命令——用 uvicorn 启动 FastAPI 应用，监听所有网络接口的 8000 端口。

不过，在实际的生产环境中，我们通常不会直接用 `docker build` 和 `docker run` 来管理单个容器，而是使用 **Docker Compose** 来编排多个相互协作的服务。因为我们的工单系统不仅仅是一个 Python 应用——它还需要数据库来持久化工单数据，可能还需要 Redis 来做缓存，甚至还需要向量数据库（如 Qdrant）来做知识库检索。这些服务之间的依赖关系、网络配置、数据卷挂载等，都需要一个统一的声明式配置文件来管理。

下面是一个完整的 `docker-compose.yml` 配置，它定义了我们工单系统的完整运行环境：

```yaml
version: '3.8'

services:
  api:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - DATABASE_URL=postgresql://ticketuser:ticketpass@db:5432/ticketdb
      - REDIS_URL=redis://redis:6379/0
      - QDRANT_URL=http://qdrant:6333
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_started
      qdrant:
        condition: service_started
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: ticketuser
      POSTGRES_PASSWORD: ticketpass
      POSTGRES_DB: ticketdb
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql:ro
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ticketuser -d ticketdb"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"

  qdrant:
    image: qdrant/qdrant:v1.7.4
    volumes:
      - qdrant_data:/qdrant/storage
    ports:
      - "6333:6333"

volumes:
  postgres_data:
  redis_data:
  qdrant_data:
```

这个 docker-compose 文件定义了四个服务：**api**（我们的主应用）、**db**（PostgreSQL 数据库）、**redis**（缓存）、**qdrant**（向量数据库）。每个服务都有其特定的配置，而且它们之间通过 Docker 内部网络互相通信——注意看 `environment` 中的数据库连接字符串，用的是 `db:5432` 而不是 `localhost:5432`，因为在 Docker 网络中，服务名就是主机名。

有几个关键的设计决策值得深入讨论：

**第一，健康检查（healthcheck）的重要性。** 我们在 `api` 和 `db` 服务中都配置了健康检查。对于 PostgreSQL 来说，`pg_isready` 命令可以检测数据库是否已经准备好接受连接；对于 API 服务来说，我们假设有一个 `/api/health` 端点返回健康状态。`depends_on` 中的 `condition: service_healthy` 确保了 API 服务会在数据库完全就绪后才启动，避免了"连接被拒绝"的错误。这是一个在生产环境中非常常见的坑——很多人只用了 `depends_on` 但没有配 healthcheck，结果应用启动时数据库还没 ready，导致启动失败。

**第二，数据持久化（volumes）。** 数据库的数据、Redis 的 AOF 文件、Qdrant 的向量索引，都通过 named volumes 进行持久化。这意味着即使容器被删除重建，数据也不会丢失。`postgres_data`、`redis_data`、`qdrant_data` 这些 volume 会在首次创建时自动生成，并绑定到宿主机的 Docker 管理目录下。

**第三，环境变量注入。** 敏感信息如 `OPENAI_API_KEY` 和数据库密码没有硬编码在 compose 文件中，而是通过 `${OPENAI_API_KEY}` 从 `.env` 文件或环境变量读取。这是安全生产的基本要求——永远不要把密钥提交到代码仓库中。

有了这个 compose 文件之后，部署就变得非常简单了：只需要执行 `docker-compose up -d`（`-d` 表示后台运行），Docker 就会按照依赖顺序依次拉取镜像、构建应用、启动服务。如果要更新版本，执行 `docker-compose build && docker-compose up -d` 即可。

### 性能优化：从"能用"到"好用"的关键一步

当我们把系统部署到生产环境后，第一个要面对的问题就是性能。在开发阶段，我们可能一天只处理几十个测试请求，响应时间慢一点也无所谓；但在生产环境中，如果每天有 500+ 工单涌入，高峰期每秒可能有数十个并发请求，任何性能瓶颈都会直接影响用户体验和业务指标。

让我们从几个维度来分析工单系统的性能优化空间。

#### 1. LLM 调用的异步化与并行化

在我们的系统中，LLM 调用是最耗时的操作——一次 OpenAI GPT-4 的调用可能需要 3-10 秒，而分类意图、自动回复、人工分配建议等多个节点都涉及 LLM 调用。如果这些调用是串行执行的，那么一个工单的处理时间可能是所有 LLM 调用时间的总和；但如果能把其中一些无关的调用并行化，总时间就能大幅缩短。

LangGraph 本身是支持异步执行的，我们可以利用 Python 的 `asyncio` 和 LangChain 的异步接口来实现这一点：

```python
import asyncio
from langchain_openai import ChatOpenAI

async_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

async def classify_intent_async(state: TicketState) -> TicketState:
    messages = state["messages"][-1].content if state["messages"] else ""
    
    parallel_tasks = [
        async_llm.ainvoke(f"分类以下消息的类别(技术/账务/投诉/咨询): {messages}"),
        async_llm.ainvoke(f"判断以下消息的紧急程度(高/中/低): {messages}"),
        async_llm.ainvoke(f"评估以下消息的处理优先级(1-10): {messages}")
    ]
    
    results = await asyncio.gather(*parallel_tasks)
    
    return {
        "category": results[0].content.strip(),
        "urgency": results[1].content.strip(),
        "priority": int(results[2].content.strip())
    }
```

这段代码展示了如何将三个原本串行的 LLM 调用改为并行执行。`asyncio.gather()` 会同时发起三个异步请求，然后等待它们全部完成后返回结果。如果每个调用平均需要 2 秒，串行执行需要 6 秒，而并行执行只需要约 2 秒（受限于最慢的那个调用），性能提升了 3 倍。

不过需要注意的是，并行化不是万能药。首先，并行调用量大会增加对 LLM API 的并发压力，可能触发速率限制（rate limit）；其次，某些调用之间如果有数据依赖关系（比如第二个调用的输入依赖于第一个调用的输出），就无法并行化；第三，过多的并行调用会增加代码复杂度和调试难度。所以在实际应用中，我们需要仔细分析调用之间的依赖关系，只对真正独立的调用进行并行化。

#### 2. 缓存策略：减少重复计算

在工单系统中，很多操作的结果是可以缓存的。比如，同一个用户短时间内提交的相似问题，其意图分类结果很可能是一样的；知识库的检索结果在一定时间内也不会变化；一些统计性的仪表盘数据更是可以缓存较长时间。

我们可以利用 Redis 来实现多层次的缓存：

```python
import json
import hashlib
from typing import Optional
import redis.asyncio as aioredis

redis_client = aioredis.from_url("redis://localhost:6379/0")

async def cached_classify(messages: str, ttl: int = 300) -> Optional[dict]:
    cache_key = f"classify:{hashlib.md5(messages.encode()).hexdigest()}"
    
    cached = await redis_client.get(cache_key)
    if cached:
        return json.loads(cached)
    
    result = await do_actual_classification(messages)
    await redis_client.setex(cache_key, ttl, json.dumps(result))
    
    return result

async def get_dashboard_stats() -> dict:
    cache_key = "dashboard:stats"
    cached = await redis_client.get(cache_key)
    if cached:
        return json.loads(cached)
    
    stats = compute_stats_from_db()
    await redis_client.setex(cache_key, 60, json.dumps(stats))
    
    return stats
```

这里的缓存设计遵循了几个原则：**第一，选择合适的 TTL（Time To Live）**。意图分类结果的缓存时间是 300 秒（5 分钟），因为用户的意图短期内不太会变；而仪表盘统计数据只缓存 60 秒，因为它需要相对实时地反映最新情况。**第二，使用哈希值作为缓存键**。对于用户消息这种非结构化输入，直接用它作为 key 不太合适（太长且有特殊字符），所以我们先用 MD5 哈希一下。**第三，缓存穿透保护**。虽然上面的简化示例中没有展示，但在生产环境中应该考虑缓存未命中时的防护措施，比如使用布隆过滤器或者设置空值的短期缓存。

#### 3. 连接池与资源管理

另一个容易被忽视的性能问题是连接管理。每次 HTTP 请求进来，如果我们都新建一个数据库连接、一个新的 Redis 客户端实例，那么在高并发场景下连接的创建和销毁开销会非常大，甚至可能耗尽数据库的最大连接数限制。

正确的做法是在应用启动时就初始化好连接池，然后在整个应用生命周期内复用这些连接：

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
import redis.asyncio as aioredis

engine = None
async_session_factory = None
redis_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, async_session_factory, redis_client
    
    engine = create_async_engine(
        "postgresql+asyncpg://ticketuser:ticketpass@db:5432/ticketdb",
        pool_size=20,
        max_overflow=10,
        pool_pre_ping=True
    )
    async_session_factory = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    redis_client = aioredis.from_url(
        "redis://redis:6379/0",
        max_connections=20,
        decode_responses=True
    )
    
    yield
    
    await engine.dispose()
    await redis_client.close()

app = FastAPI(lifespan=lifespan)

async def get_db():
    async with async_session_factory() as session:
        try:
            yield session
        finally:
            await session.close()
```

这段代码使用了 FastAPI 的 `lifespan` 上下文管理器来管理资源的生命周期。在应用启动时（`yield` 之前），我们创建了 SQLAlchemy 的异步引擎（带连接池）和 Redis 异步客户端；在应用关闭时（`yield` 之后），我们优雅地关闭这些连接。SQLAlchemy 引擎的 `pool_size=20` 表示维护 20 个常驻连接，`max_overflow=10` 表示在高峰期可以额外借用 10 个连接（总共最多 30 个），`pool_pre_ping=True` 会在每次从池中取出连接前先 ping 一下确认连接有效，避免使用到已经被数据库端断开的"僵尸连接"。

#### 4. 批量处理与队列削峰

在某些场景下，用户可能会批量导入大量工单，或者某个突发事件导致工单量暴增。这时候如果还是一个个同步处理，很容易压垮系统。解决方案是引入消息队列（如 RabbitMQ 或 Kafka）作为缓冲层，把同步处理改为异步消费：

```python
from fastapi import BackgroundTasks
import asyncio

pending_queue = asyncio.Queue(maxsize=1000)
processing_semaphore = asyncio.Semaphore(5)

async def process_ticket_worker():
    while True:
        ticket_data = await pending_queue.get()
        async with processing_semaphore:
            try:
                result = await graph.ainvoke(ticket_data, config={"configurable": {"thread_id": ticket_data["id"]}})
                await save_result_to_db(result)
            except Exception as e:
                log_error(e, ticket_data)
            finally:
                pending_queue.task_done()

@app.post("/api/tickets/batch")
async def batch_create_tickets(request: BatchCreateRequest, background_tasks: BackgroundTasks):
    for ticket in request.tickets:
        if pending_queue.full():
            raise HTTPException(status_code=503, detail="队列已满，请稍后重试")
        await pending_queue.put(ticket.model_dump())
    
    background_tasks.add_task(process_ticket_worker)
    
    return {"message": f"已接收 {len(request.tickets)} 个工单，正在排队处理"}
```

这里我们用了 Python 内置的 `asyncio.Queue` 作为简单的内存队列（在实际生产中应替换为 RabbitMQ 或 Redis Queue），并用 `Semaphore` 控制同时处理的工单数量不超过 5 个。当批量导入的工单数量超过队列容量时，直接返回 503 错误告知客户端稍后重试，而不是让系统默默崩溃。`BackgroundTasks` 让 FastAPI 在返回响应后继续在后台处理任务，不会阻塞 HTTP 响应。

### 生产级配置：那些开发阶段容易忽略的细节

除了性能优化，从开发走向生产还有大量"琐碎"但至关重要的配置项需要处理。这些配置单独看起来都不复杂，但遗漏任何一个都可能导致生产事故。

#### 日志系统：出问题时怎么排查

在生产环境中，完善的日志系统是我们定位问题的第一手段。一个好的日志应该包含：时间戳、日志级别、请求 ID（用于追踪一个请求的完整链路）、模块名、具体的日志消息，以及相关的结构化数据。

```python
import logging
import sys
from logging.handlers import RotatingFileHandler
import uuid

def setup_logging():
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(request_id)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    file_handler = RotatingFileHandler(
        'logs/ticket_system.log',
        maxBytes=50*1024*1024,
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

class RequestIDMiddleware:
    def __init__(self, app):
        self.app = app
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            request_id = str(uuid.uuid4())[:8]
            scope["state"]["request_id"] = request_id
        
        await self.app(scope, receive, send)
```

这个日志系统做了几件事：**格式统一化**——所有日志都按照相同的模板输出，方便后续用 ELK（Elasticsearch + Logstash + Kibana）或类似工具收集分析；**日志轮转**——`RotatingFileHandler` 会在单个日志文件达到 50MB 时自动切换到新文件，最多保留 5 个历史文件，防止磁盘被日志写满；**请求追踪**——通过中间件为每个 HTTP 请求生成唯一的 `request_id`，这样当一个请求经过多个服务和组件时，我们可以通过这个 ID 把所有相关日志串联起来，快速定位问题。

#### 错误处理与降级策略

在生产环境中，错误不是"会不会发生"的问题，而是"什么时候发生"的问题。外部服务可能宕机、网络可能抖动、LLM API 可能超限，我们需要为每一种可能的故障设计相应的应对策略。

```python
from fastapi import Request
from fastapi.responses import JSONResponse
import time

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.last_failure_time = 0
        self.state = "closed"
    
    async def call(self, func, *args, **kwargs):
        if self.state == "open":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "half-open"
            else:
                raise Exception("Circuit breaker is OPEN - service unavailable")
        
        try:
            result = await func(*args, **kwargs)
            if self.state == "half-open":
                self.state = "closed"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
            raise

llm_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    request_id = request.state.get("request_id", "unknown")
    logging.error(f"[{request_id}] 未处理的异常: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "内部服务器错误",
            "request_id": request_id,
            "detail": "请联系管理员并提供此请求 ID 以便排查"
        }
    )

async def safe_llm_call(prompt: str, fallback_response: str = "暂时无法处理"):
    try:
        return await llm_breaker.call(async_llm.ainvoke, prompt)
    except Exception as e:
        logging.warning(f"LLM 调用失败，启用降级方案: {e}")
        return fallback_response
```

这里实现了两个重要的生产级机制：**熔断器（Circuit Breaker）** 和 **全局异常处理器**。熔断器的原理我们在第五章详细讲过，这里把它封装成了一个可复用的类，专门用来保护 LLM API 调用——当连续 5 次调用失败后，熔断器进入"开启"状态，接下来的 60 秒内不再尝试调用 LLM，而是直接返回降级响应，避免雪崩效应。全局异常处理器则捕获所有未被其他 handler 处理的异常，返回统一的错误格式给客户端，同时在服务端记录详细的错误日志（包括堆栈信息 `exc_info=True`）。

#### 安全加固

最后但同样重要的是安全性。作为一个面向互联网的服务，我们的工单系统必须防范各种常见的安全威胁：

```python
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Depends, HTTPException, status
from slowapi import Limiter
from slowapi.util import get_remote_address
import jwt

security = HTTPBearer()
limiter = Limiter(key_func=get_remote_address)
JWT_SECRET = "your-secret-key-change-in-production"
JWT_ALGORITHM = "HS256"

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token 已过期")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="无效的 Token")

@app.post("/api/tickets")
@limiter.limit("10/minute")
async def create_ticket_protected(
    request: Request,
    body: CreateTicketRequest,
    user: dict = Depends(verify_token)
):
    user_role = user.get("role", "user")
    if user_role not in ["admin", "agent"]:
        raise HTTPException(status_code=403, detail="权限不足")
    
    return await create_ticket(body)
```

这段代码展示了三层安全措施：**身份认证**（JWT Token 验证）、**授权控制**（基于角色的访问控制，只有 admin 和 agent 角色才能创建工单）、**速率限制**（每个 IP 地址每分钟最多 10 次请求，防止恶意刷接口）。这三层防线配合起来，可以抵御大部分常见的攻击手段。

### CI/CD 流水线：自动化部署的基础设施

手动部署虽然在小型项目中可行，但随着团队规模扩大和发布频率提高，手动操作的出错风险也会急剧上升。CI/CD（持续集成/持续部署）流水线可以帮我们把构建、测试、发布的流程自动化，确保每次代码变更都能以可重复、可追溯的方式部署到生产环境。

下面是一个基于 GitHub Actions 的 CI/CD 配置示例：

```yaml
name: Deploy Ticket System

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install pytest pytest-cov
      
      - name: Run tests
        run: pytest tests/ -v --cov=ticket_system --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v4

  deploy-staging:
    needs: test
    if: github.ref == 'refs/heads/main' && github.event_name == 'push'
    runs-on: ubuntu-latest
    environment: staging
    steps:
      - uses: actions/checkout@v4
      
      - name: Build and push Docker image
        run: |
          docker build -t ticket-system:${{ github.sha }} .
          docker tag ticket-system:${{ github.sha }} registry.example.com/ticket-system:staging
          docker push registry.example.com/ticket-system:staging
      
      - name: Deploy to staging
        run: |
          ssh deployer@staging-server "cd /opt/ticket-system && docker-compose pull && docker-compose up -d"

  deploy-production:
    needs: deploy-staging
    if: success()
    runs-on: ubuntu-latest
    environment: production
    steps:
      - name: Approval
        uses: trstringer/manual-approval@v1
        with:
          secret: ${{ secrets.GITHUB_TOKEN }}
          approvers: dev-lead,ops-manager
      
      - name: Deploy to production
        run: |
          ssh deployer@prod-server "cd /opt/ticket-system && docker-compose pull && docker-compose up -d"
      
      - name: Health check
        run: |
          sleep 30
          curl -f https://ticket-api.example.com/api/health || exit 1
      
      - name: Notify team
        if: always()
        run: |
          curl -X POST "$WEBHOOK_URL" -d '{
            "text": "部署${{ job.status }}: commit ${{ github.sha }}'
          }'
```

这个 CI/CD 流水线定义了三个阶段的任务：

**第一阶段（test）**：当代码推送到任意分支或创建 PR 时触发，运行单元测试并收集代码覆盖率报告。这是质量把关的第一道门——如果测试不通过，后面的部署流程就不会执行。

**第二阶段（deploy-staging）**：只有当代码合并到 main 分支并且测试全部通过后才会执行。这一步会构建 Docker 镜像、推送到私有镜像仓库、然后 SSH 到预发布服务器执行 `docker-compose pull && docker-compose up -d` 完成部署。预发布环境是生产环境的镜像，用于最终验证。

**第三阶段（deploy-production）**：预发布部署成功后，需要人工审批（`manual-approval` action 会发送通知给指定的审批人，只有他们点击"批准"后才会继续）。审批通过后部署到生产服务器，然后等待 30 秒让服务启动完成，再执行健康检查确认部署成功。无论成功还是失败，最后都会通过 Webhook 通知团队。

这种"测试→预发布→审批→生产→验证→通知"的流水线模式，是目前业界比较成熟的实践。它平衡了自动化效率和人工把控的需求，既能保证发布速度，又能最大限度地降低线上事故的风险。

### 性能基准测试：用数据说话

在完成了上述所有优化之后，我们应该用实际的基准测试来量化优化的效果。Locust 是一个优秀的 Python 负载测试工具，它可以模拟成百上千个并发用户访问我们的 API，然后收集响应时间、吞吐量、错误率等关键指标。

```python
from locust import HttpUser, task, between

class TicketSystemUser(HttpUser):
    wait_time = between(1, 3)
    
    def on_start(self):
        response = self.client.post("/api/auth/login", json={
            "username": "test_user",
            "password": "test_pass"
        })
        self.token = response.json()["token"]
        self.headers = {"Authorization": f"Bearer {self.token}"}
    
    @task(5)
    def create_ticket(self):
        self.client.post("/api/tickets", json={
            "message": "我的账户余额显示不对，请帮忙查询",
            "user_id": "user_12345",
            "channel": "web"
        }, headers=self.headers)
    
    @task(3)
    def view_ticket(self):
        self.client.get("/api/tickets/ticket_001", headers=self.headers)
    
    @task(2)
    def dashboard(self):
        self.client.get("/api/dashboard/stats", headers=self.headers)
    
    @task(1)
    def submit_action(self):
        self.client.post("/api/tickets/ticket_001/action", json={
            "action": "resolve",
            "comment": "已处理完毕",
            "agent_id": "agent_001"
        }, headers=self.headers)
```

这个 Locust 测试脚本模拟了四种用户行为：创建工单（权重最高，占 5/11）、查看工单详情（3/11）、查看仪表盘（2/11）、提交操作（1/11）。`wait_time = between(1, 3)` 表示每个用户在两次操作之间随机等待 1-3 秒，模拟真实用户的行为间隔。

运行 `locust -f locustfile.py --host=http://localhost:8000 -u 100 -r 10 -t 60s` 就可以启动测试：模拟 100 个虚拟用户，每秒新增 10 个用户，持续 60 秒。测试结束后，Locust 会生成一份详细的报告，包含：

- **平均响应时间**（应该 < 500ms 对于简单操作）
- **P95/P99 响应时间**（P99 应该 < 2s）
- **每秒请求数（RPS）**
- **错误率**（应该 < 1%）

如果测试结果不达标，就需要回到前面的优化环节，针对性地进行调整——是数据库查询太慢就加索引，是 LLM 调用太多就加缓存，是连接数不够就调整连接池大小。性能优化本质上就是一个"测量→分析→优化→再测量"的迭代过程。

### 总结：部署优化的核心要点回顾

到这里，我们已经把智能客服工单系统从一个"能在本地跑起来的 demo"升级为一个"可以在生产环境稳定运行的系统"。让我们回顾一下这一节覆盖的核心内容：

**容器化层面**：通过 Dockerfile 标准化了应用的打包方式，通过 Docker Compose 编排了 API、数据库、Redis、向量数据库四个服务的协作关系，配置了健康检查、数据持久化、环境变量注入等生产必需项。

**性能优化层面**：从 LLM 调用的异步并行化（3倍提速）、Redis 多层次缓存（减少重复计算）、连接池复用（避免频繁建连开销）、到消息队列削峰（应对突发流量），全方位地提升了系统的吞吐能力和响应速度。

**生产配置层面**：建立了统一的日志系统（含请求追踪 ID 和日志轮转）、实现了熔断器和降级策略（防止单点故障扩散）、添加了 JWT 认证 + RBAC 授权 + 速率限制的三层安全防线。

**DevOps 层面**：搭建了 GitHub Actions CI/CD 流水线，实现了"测试→预发布→审批→生产→验证→通知"的自动化发布流程，配合 Locust 基准测试确保每次变更都有数据支撑。

这些优化并不是一次性就能做完的事情——随着业务的发展，新的性能瓶颈会出现，新的安全威胁会产生，新的部署需求会浮现。但只要掌握了这套方法论，我们就有能力在面对新问题时快速找到合适的解决方案。
