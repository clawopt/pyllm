# 9.2 系统架构设计

## 架构设计的原则与权衡

在上一节中，我们明确了系统的功能需求和非功能需求。现在要把这些需求转化成具体的架构方案。但在此之前，我想先聊聊架构设计中最重要的东西——不是画漂亮的图，不是选最酷的技术，而是**做出正确的权衡（Trade-off）**。

每一个架构决策都是一种取舍。你选择了微服务就放弃了简单性；选择了强一致性就牺牲了可用性；选择了实时同步就增加了系统复杂度。好的架构师不是选择"最好"的方案（因为不存在最好的方案），而是为当前的业务阶段和约束条件找到"最合适"的方案。对于我们的企业知识库项目来说，核心的几个权衡是：

1. **单体 vs 微服务**：初期用户量不大（500 并发），团队规模小（3-5 人），选单体应用更合理。但模块边界要清晰，为未来拆分预留接口。
2. **同步 vs 异步处理**：问答请求本身是同步的（用户等回答），但数据同步、索引构建、日志记录这些后台任务应该异步化。
3. **精度 vs 延迟**：用更强的模型、更多的检索后处理能提升质量，但会增加延迟。需要在 P99 < 8s 的约束下找到最佳平衡点。
4. **通用 vs 定制**：LlamaIndex 提供了很多开箱即用的组件，但企业场景往往需要深度定制。我们在通用框架上做有针对性的扩展。

带着这些权衡思路，让我们来看整体架构。

## 整体架构图

```
                              ┌─────────────────┐
                              │   用户浏览器     │
                              │  (Vue3 Frontend)│
                              └────────┬────────┘
                                       │ HTTPS / WebSocket
                                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│                          API Gateway (Nginx)                         │
│    ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │
│    │ SSL 终结  │  │ 静态资源  │  │ 路由转发  │  │ Rate Limiting    │   │
│    │          │  │  缓存     │  │          │  │ (令牌桶/滑动窗口) │   │
│    └──────────┘  └──────────┘  └──────────┘  └──────────────────┘   │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        Application Layer                             │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    FastAPI Application                       │   │
│  │                                                              │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐           │   │
│  │  │ Auth    │ │ Chat    │ │ Admin   │ │Analytics│           │   │
│  │  │ Middleware│ │ Router │ │ Router │ │ Router  │           │   │
│  │  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘           │   │
│  │       │           │           │           │                 │   │
│  │  ┌────▼───────────▼───────────▼───────────▼─────────┐      │   │
│  │  │              Service Layer                       │      │   │
│  │  │  ┌──────────┐ ┌──────────┐ ┌────────────────┐   │      │   │
│  │  │  │ AuthService│ │ RAGService│ │PermissionSvc  │   │      │   │
│  │  │  └──────────┘ └─────┬────┘ └────────────────┘   │      │   │
│  │  │                    │                              │      │   │
│  │  │  ┌─────────────────▼────────────────────────┐   │      │   │
│  │  │  │           Core RAG Engine                │   │      │   │
│  │  │  │                                         │   │      │   │
│  │  │  │  Query → Classify → Transform → Retrieve │   │      │   │
│  │  │  │         → Postprocess → Rerank → Synthesize│  │      │   │
│  │  │  │              → Citation → Response        │   │      │   │
│  │  │  └──────────────────────────────────────────┘   │      │   │
│  │  └────────────────────────────────────────────────┘      │   │
│  └───────────────────────────────────────────────────────────┘   │
│                                                                     │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                     │
          ▼                    ▼                     ▼
┌─────────────────┐  ┌─────────────────┐  ┌──────────────────┐
│  Data Layer     │  │  Cache Layer    │  │ Message Queue     │
│                 │  │                 │  │                  │
│ ┌─────────────┐ │  │ ┌─────────────┐ │  │ ┌──────────────┐ │
│ │ Qdrant      │ │  │ │ Redis       │ │  │ │ Celery +     │ │
│ │ (Vector DB) │ │  │ │ (Query Cache│ │  │ │ RabbitMQ/    │ │
│ │             │ │  │ │ Session     │ │  │ │ Redis Broker)│ │
│ │ Collections:│ │  │ │ Rate Limit) │ │  │ │              │ │
│ │ · kb_main   │ │  │ └─────────────┘ │  │ │ Tasks:       │ │
│ │ · kb_public │ │  └─────────────────┘  │ │ · IndexBuild │ │
│ │ · kb_confid │ │                      │ │ · DataSync   │ │
│ └─────────────┘ │                      │ │ · EvalRun    │ │
│ ┌─────────────┐ │                      │ └──────────────┘ │
│ │ PostgreSQL  │ │                      └──────────────────┘
│ │ (Metadata)  │ │
│ │ Tables:     │ │
│ │ · users     │ │
│ │ · documents │ │
│ │ · data_sources│
│ │ · query_log │ │
│ │ · audit_log │ │
│ └─────────────┘ │
└─────────────────┘

          ┌────────────────────┐
          │  External Services  │
          │                    │
          │ ┌────────────────┐ │
          │ │ OpenAI API     │ │
          │ │ (GPT-4o /      │ │
          │ │  text-embed)   │ │
          │ └────────────────┘ │
          │ ┌────────────────┐ │
          │ │ SSO/IAM System│ │
          │ │ (SAML/OAuth)  │ │
          │ └────────────────┘ │
          └────────────────────┘
```

这个架构图从上到下分为五层：客户端层、网关层、应用层、数据层和外部服务层。让我们逐层拆解每一层的职责和关键设计点。

## 网关层：Nginx 作为统一入口

Nginx 在这个架构中承担了多个角色：

```nginx
# nginx.conf 核心配置片段
upstream fastapi_backend {
    server app:8000;
    keepalive 32;
}

server {
    listen 443 ssl http2;
    server_name kb.yourcompany.com;

    # SSL 配置（使用 Let's Encrypt 或企业证书）
    ssl_certificate     /etc/nginx/ssl/fullchain.pem;
    ssl_certificate_key /etc/nginx/ssl/privkey.pem;
    ssl_protocols       TLSv1.2 TLSv1.3;
    ssl_ciphers         HIGH:!aNULL:!MD5;

    # 静态资源缓存
    location /static/ {
        alias /app/frontend/dist/;
        expires 30d;
        add_header Cache-Control "public, immutable";
    }

    # API 代理到 FastAPI
    location /api/ {
        proxy_pass http://fastapi_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # SSE 流式响应支持
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 300s;

        # WebSocket 支持（用于实时推送）
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    # 速率限制定义
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=30r/m;
    limit_req_zone $binary_remote_addr zone=chat_limit:10m rate=10r/m;

    location /api/v1/chat/ {
        limit_req zone=chat_limit burst=20 nodelay;
        proxy_pass http://fastapi_backend;
        # ... 同上 proxy 设置
    }
}
```

这里有几个关键的设计考量：

**速率限制的分区分策略**：普通 API 接口限制每分钟 30 次请求（`rate=30r/m`），而问答接口单独限制为每分钟 10 次（`rate=10r/m`）。这是因为每次问答调用都会产生 LLM API 成本——一个恶意用户或者 bug 导致的死循环可能在一小时内消耗掉几千美元。`burst=20 nodelay` 允许短时间的突发流量（比如用户快速连续问了两个相关问题），但超过 burst 后就会触发 429 响应。

**SSE 代理的特殊配置**：当使用流式输出时，FastAPI 会以 Server-Sent Events 格式逐步返回答案片段。默认情况下 Nginx 会缓冲输出直到攒够一定大小的数据块再发送，这会导致前端迟迟收不到第一个字。所以必须设置 `proxy_buffering off` 和 `proxy_cache off` 来禁用缓冲。同时 `proxy_read_timeout 300s` 要设置得足够长——LLM 生成长回答可能需要几十秒。

**WebSocket 支持**：虽然 REST + SSE 已经能满足大部分需求，但如果将来需要做实时通知（如"您的文档已同步完成"或"系统维护公告"），WebSocket 就派上用场了。提前在 Nginx 层面配好 Upgrade 头的透传可以避免后续改造时的麻烦。

## 应用层：FastAPI 服务内部结构

应用层是整个系统的核心，它承载了所有的业务逻辑。我们采用分层设计来组织代码：

```
Application Layer Internal Structure

┌─────────────────────────────────────────────────────────────┐
│                     Request Lifecycle                        │
│                                                             │
│  HTTP Request                                               │
│       │                                                     │
│       ▼                                                     │
│  ┌─────────────┐                                            │
│  │ Middleware  │ ← Auth / CORS / Request ID / Timing        │
│  │ Chain       │                                            │
│  └──────┬──────┘                                            │
│         │                                                    │
│         ▼                                                    │
│  ┌─────────────┐                                            │
│  │    Router   │ ← URL routing → handler function            │
│  │             │                                            │
│  └──────┬──────┘                                            │
│         │                                                    │
│         ▼                                                    │
│  ┌─────────────┐    ┌─────────────┐                         │
│  │ Pydantic    │ →→ │ Service     │ → Business Logic        │
│  │ Validation  │    │ Layer       │                         │
│  └─────────────┘    └──────┬──────┘                         │
│                            │                                 │
│                            ▼                                 │
│                   ┌─────────────────┐                       │
│                   │  Core RAG Engine │ ← Framework-agnostic  │
│                   │                 │   pure logic           │
│                   └────────┬────────┘                       │
│                            │                                 │
│              ┌─────────────┼─────────────┐                  │
│              ▼             ▼             ▼                  │
│        ┌──────────┐ ┌──────────┐ ┌──────────┐              │
│        │ VectorDB │ │ PostgreSQL│ │ LLM API  │              │
│        │ (Qdrant) │ │ (CRUD)   │ │(OpenAI)  │              │
│        └──────────┘ └──────────┘ └──────────┘              │
│                                                            │
│  HTTP Response                                              │
└─────────────────────────────────────────────────────────────┘
```

### 中间件链设计

中间件是横切关注点的标准处理方式。我们的中间件链按执行顺序排列如下：

```python
from fastapi import Request, Response
from fastapi.middleware.cors import CORSMiddleware
import time
import uuid
import logging

logger = logging.getLogger(__name__)


async def request_id_middleware(request: Request, call_next):
    """为每个请求生成唯一 ID，方便日志追踪"""
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id

    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


async def timing_middleware(request: Request, call_next):
    """记录每个请求的处理耗时"""
    start_time = time.perf_counter()

    response = await call_next(request)

    process_time = (time.perf_counter() - start_time) * 1000
    logger.info(
        f"{request.state.request_id} | {request.method} {request.url.path} "
        f"| {response.status_code} | {process_time:.1f}ms"
    )

    response.headers["X-Process-Time"] = f"{process_time:.1f}"
    return response


def setup_middlewares(app):
    """注册所有中间件（注意：后注册的先执行）"""

    # 1. CORS — 最外层
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["https://kb.yourcompany.com"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 2. 请求 ID 生成
    app.middleware("http")(request_id_middleware)

    # 3. 计时
    app.middleware("http")(timing_middleware)

    # 4. 认证中间件（排除公开路径）
    from app.api.auth import auth_middleware
    app.middleware("http")(auth_middleware)
```

中间件的注册顺序至关重要。FastAPI/Starlette 的中间件采用"洋葱模型"——请求从外层向内层穿入，响应从内层向外层穿出。所以最后注册的 `auth_middleware` 实际上是第一个执行的（在 CORS 之后），这意味着未认证的请求会在到达路由之前就被拦截。`request_id_middleware` 在认证之前执行是因为即使认证失败我们也希望有一个 request_id 来关联错误日志。

### 数据库 Schema 设计

PostgreSQL 不仅存储业务数据，还承担了向量元数据的补充存储角色。下面是核心表结构：

```python
from sqlalchemy import Column, Integer, String, Text, DateTime, JSON, Boolean, ForeignKey, Enum as SAEnum
from sqlalchemy.orm import relationship, declarative_base
from datetime import datetime
import enum

Base = declarative_base()


class DataSourceType(enum.Enum):
    FILE_DIRECTORY = "file_directory"
    DATABASE_POSTGRES = "database_postgres"
    DATABASE_MYSQL = "database_mysql"
    API_REST = "api_rest"


class DataClassification(enum.Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    external_id = Column(String(128), unique=True, nullable=False, comment="SSO系统中的用户ID")
    username = Column(String(100), nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    department = Column(String(100))
    role = Column(String(50), default="regular_user")  # admin/knowledge_manager/regular_user/guest
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)


class DataSource(Base):
    __tablename__ = "data_sources"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(200), nullable=False, comment="数据源名称")
    source_type = Column(SAEnum(DataSourceType), nullable=False)
    config = Column(JSON, nullable=False, comment="连接配置（密码等敏感信息加密存储）")
    classification = Column(SAEnum(DataClassification), default=DataClassification.INTERNAL)
    allowed_roles = Column(JSON, default=list, comment="可访问此数据源的角色列表")
    sync_schedule = Column(String(50), comment="Cron 表达式，如 '0 */6 * * *'")
    last_sync_at = Column(DateTime)
    sync_status = Column(String(20), default="idle")  # idle/syncing/success/failed
    error_message = Column(Text)
    document_count = Column(Integer, default=0)
    created_by = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)

    creator = relationship("User", backref="data_sources")


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    source_id = Column(Integer, ForeignKey("data_sources.id"), nullable=False)
    doc_id = Column(String(128), unique=True, nullable=False, comment="LlamaIndex 内部文档ID")
    title = Column(String(500))
    file_path = Column(String(1000))
    file_type = Column(String(20))  # pdf/docx/md/html/xlsx
    classification = Column(SAEnum(DataClassification), default=DataClassification.INTERNAL)
    metadata = Column(JSON, comment="自定义元数据（标签、作者、版本等）")
    node_count = Column(Integer, default=0, comment="分块后的节点数量")
    qdrant_collection = Column(String(100), comment="所属的Qdrant集合")
    status = Column(String(20), default="active")  # active/deleted/indexing
    indexed_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)

    data_source = relationship("DataSource", backref="documents")


class QueryLog(Base):
    __tablename__ = "query_log"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    session_id = Column(String(128), index=True)
    query_text = Column(Text, nullable=False)
    response_text = Column(Text)
    sources_used = Column(JSON, comment="引用的文档ID列表")
    latency_ms = Column(Float)
    model_used = Column(String(50))
    tokens_used = Column(Integer)
    feedback = Column(Integer, comment="用户反馈: -1差评/0无/1好评")
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    user = relationship("User", backref="query_logs")


class AuditLog(Base):
    __tablename__ = "audit_log"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    action = Column(String(50), nullable=False)  # CREATE/UPDATE/DELETE/QUERY/LOGIN
    resource_type = Column(String(50))  # data_source/document/user/config
    resource_id = Column(String(100))
    details = Column(JSON)
    ip_address = Column(String(45))
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
```

关于这个 Schema 设计有几个值得讨论的点：

**DataSource.config 使用 JSON 类型**：不同类型的数据源需要完全不同的配置参数——文件目录需要 path 和 file_patterns，数据库需要 host/port/db_name/credentials，API 需要 endpoint_url/auth_method/api_key。如果为每种类型建一张表会产生大量空列且难以扩展。JSONB 类型让每种数据源只存自己需要的字段，同时 PostgreSQL 的 JSONB 还支持 GIN 索引来做高效查询。

**Document.doc_id 与 LlamaIndex 文档 ID 对应**：这是关系数据库和向量数据库之间的桥梁。当我们需要在检索时做权限过滤（"用户 A 只能看到 confidential 级别以下的文档"），就需要先从 PostgreSQL 查出该用户有权访问的所有 doc_id 列表，然后把这个列表传给 Qdrant 做过滤查询。这种"先查权限再查向量"的模式比把权限信息直接存进 Qdrant 更灵活——因为权限变更只需要更新 PostgreSQL 而不需要重建向量索引。

**QueryLog 的分区策略**：对于一个活跃的系统来说，query_log 表会增长得非常快——假设每天 5000 次查询，一年就是 180 万条记录。所以在生产环境中应该对这张表按月做分区（Partition），历史数据可以归档到冷存储中。`created_at` 上建立索引是为了支持时间范围查询（"上周的热门问题有哪些"）。

## RAG 引擎核心流程

整个系统中最重要的部分就是 RAG 引擎——它决定了问答的质量和性能。下面是引擎内部的完整数据流转过程：

```
RAG Engine Internal Pipeline

用户问题: "公司年假政策是怎么规定的？"
                    │
                    ▼
         ┌──────────────────┐
         │  1. Query Preprocess │
         │  · 去除空白符        │
         │  · 长度截断(<2000字) │
         │  · 敏感词检测       │
         └────────┬─────────┘
                  │
                  ▼
         ┌──────────────────┐
         │  2. Query Classify  │
         │  · 问题类型识别      │
         │    (事实/流程/对比/) │
         │  · 意图澄清(如有歧义) │
         └────────┬─────────┘
                  │
                  ▼
         ┌──────────────────┐
         │  3. Context Enrich  │
         │  · 多轮对话上下文    │
         │  · 用户画像(部门/角色)│
         │  · 权限范围获取      │
         └────────┬─────────┘
                  │
                  ▼
    ┌─────────────────────────┐
    │   4. Query Transform     │
    │   · HyDE (可选)          │
    │   · Multi-Query (可选)   │
    │   · Decompose (复杂问题) │
    └───────────┬─────────────┘
                │
        ┌───────┴───────┐
        ▼               ▼
  ┌──────────┐   ┌──────────┐
  │ Vector   │   │ BM25     │
  │ Search   │   │ Search   │
  │ (语义相似)│   │ (关键词)  │
  └────┬─────┘   └────┬─────┘
       │              │
       └──────┬───────┘
              ▼
    ┌──────────────────┐
    │  5. Fusion (RRF)  │
    │  合并+去重+排序    │
    │  取 top_k=15      │
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │  6. Permission    │
    │  Filter           │
    │  过滤无权访问节点   │
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │  7. Rerank        │
    │  Cross-Encoder    │
    │  精排 top_k=5     │
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │  8. Post-process  │
    │  · 相似度阈值过滤  │
    │  · Lost in Middle │
    │  重排序           │
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │  9. Synthesize    │
    │  LLM 生成回答     │
    │  (REFINE模式)     │
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │ 10. Citation Build│
    │  提取引用来源     │
    │  格式化展示       │
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │ 11. Response      │
    │  组装最终返回对象  │
    └──────────────────┘
```

这个 11 步流水线看起来很长，但实际上大部分步骤的处理时间是毫秒级的——真正耗时的只有第 4 步（HyDE 需要 LLM 调用）、第 9 步（Synthesis 是主要的 LLM 调用）以及可能的第 7 步（Reranker 如果用远程 API）。下面用一个具体的例子来走一遍完整的流程：

假设用户问："我们公司的 Pro 版和企业版有什么区别？"

**Step 1 Preprocess**：输入被清理为干净的中文文本，长度检查通过（远小于 2000 字上限），敏感词检测未命中。

**Step 2 Classify**：分类器判断这是一个"对比性问题"（comparative），这类问题的特点是需要同时检索多个产品/方案的信息然后进行横向比较。分类结果会影响后续的检索策略——对比性问题通常需要更大的 top_k 来确保覆盖所有相关实体。

**Step 3 Context Enrich**：系统发现这是多轮对话的第 3 轮，前两轮分别问了"Pro 版的价格"和"企业版的功能"。这些上下文会被追加到当前问题后面变成："[前文背景：用户之前询问了 Pro 版价格和企业版功能] 我们公司的 Pro 版和企业版有什么区别？"。同时获取用户的角色信息——假设他是销售部门的 regular_user，有权访问 internal 及以下级别的文档。

**Step 4 Transform**：对于对比类问题，系统自动启用 DecomposeQueryTransform 将其分解为子问题：
- 子问题 1："Pro 版的核心功能和定价是什么？"
- 子问题 2："企业版的核心功能和定价是什么？"
- 子问题 3："Pro 版和企业版的差异对比？"

**Step 5 Hybrid Retrieval**：三个子问题各自同时发起向量检索和 BM25 关键词检索。向量搜索在 Qdrant 中用 embedding 找语义相似的文档片段；BM25 搜索在内存索引中找包含"Pro 版""企业版""区别""差异"等关键词的片段。两组结果通过 RRF（Reciprocal Rank Fusion）算法融合。

**Step 6 Permission Filter**：假设检索到了 15 个候选节点，其中 2 个来自 restricted 级别的文档（可能是内部定价策略文件）。由于用户只是 regular_user，这 2 个节点被过滤掉，剩下 13 个。

**Step 7 Rerank**：剩下的 13 个节点送入 Cohere Rerank 或本地 bge-reranker 模型进行精排。Reranker 会逐一对（query, node）打分，重新排序后取 top 5。

**Step 8 Post-process**：检查 top 5 中是否有相似度过低的节点（< 0.5 则丢弃），然后按照"Lost in the Middle"原则重排序——把最相关的节点放在首尾位置而非中间。

**Step 9 Synthesize**：将过滤后的 5 个节点和原始问题一起送给 GPT-4o，使用 REFINE 模式迭代生成回答。REFINE 模式会先基于第一个节点生成初始回答，然后依次用后续节点来精炼和补充，确保不遗漏任何重要信息。

**Step 10 Citation Build**：从最终回答中提取关键陈述，将其映射回对应的源文档节点，生成格式化的引用列表。

**Step 11 Response**：组装最终的响应对象，包含回答文本、引用列表、置信度分数、相关推荐问题等，返回给调用方。

## 权限控制架构

权限控制是这个系统区别于 demo 的关键特征之一。我们的权限模型采用 RBAC + 数据分级两层机制：

```
Permission Control Flow

User Request (user_id, query)
        │
        ▼
┌───────────────────┐
│ 1. Authenticate   │  ← JWT Token 验证
│   (SSO/JWT)       │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│ 2. Load Profile   │  ← 从 DB 加载用户角色/部门
│   & Roles         │
└────────┬──────────┘
         │
         ▼
┌───────────────────────────────────────┐
│ 3. Compute Access Scope               │
│                                       │
│   用户角色: regular_user              │
│   所属部门: sales                     │
│                                       │
│   → 可访问级别: public + internal     │
│   → 可访问数据源: [DS-001, DS-003,    │
│     DS-005, DS-008] (allowed_roles    │
│     包含 regular_user 的)             │
│                                       │
│   → allowed_doc_ids: [doc_001,       │
│     doc_003, ..., doc_042] (共 1200  │
│     个文档)                           │
└────────┬──────────────────────────────┘
         │
         ▼
┌───────────────────┐
│ 4. Retrieve with   │  ← Qdrant filter:
│   Access Filter    │    where doc_id IN [...]
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│ 5. Double Check   │  ← 合成后的回答再次
│   (Output Guard)  │    检查是否泄露受限信息
└───────────────────┘
```

权限控制的实现分为两个层面：**检索前过滤**和**输出后校验**。检索前过滤是在 Qdrant 查询时通过 `filter` 参数限定只返回允许访问的文档，这是第一道防线也是最高效的方式——不让不允许的数据进入处理管道。输出后校验则是一道保险措施：即使前面的过滤出了漏洞（比如某个文档的分类标签设错了），LLM 在生成回答后还会再做一次内容扫描，确保不会在回答中出现明显的敏感信息（如具体定价、内部策略细节等）。

## 缓存策略设计

在企业知识库场景中，缓存的价值比一般的 Web 应用要大得多。原因很简单：**很多人会问同样的问题**。新产品发布那天，可能有几百人都在问"新版本有什么变化"；出了个安全漏洞，所有人都在问"怎么修复"；年底了，大家都在问"年假怎么算"。如果没有缓存，每个相同的问题都要走一遍完整的 RAG 流水线（包括 LLM 调用），既慢又贵。

```
Cache Strategy Architecture

Query: "年假政策是怎样的？"
        │
        ▼
┌───────────────────┐
│ L1: Exact Match   │  ← Redis, key=hash(query+user_role)
│ Cache (TTL=5min)  │    命中率 ~30%
└────────┬──────────┘
         │ miss
         ▼
┌───────────────────┐
│ L2: Semantic      │  ← Qdrant, 查询缓存集合
│ Similarity Cache  │    cosine_sim > 0.95 视为同一问题
│ (TTL=30min)       │    命中率 ~25% (累计 ~55%)
└────────┬──────────┘
         │ miss
         ▼
┌───────────────────┐
│ Full RAG Pipeline │  ← 完整11步流水线
│ (LLM call needed) │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│ Write-back to     │  ← 结果写入 L1 + L2
│ L1 & L2 Cache     │
└───────────────────┘
```

两级缓存的配合逻辑是这样的：L1 用精确匹配（问题文本完全一样 + 相同的角色权限范围），命中率约 30%，TTL 只有 5 分钟——因为知识库的内容可能在任何时刻被更新，太长的 TTL 会导致用户看到过期的回答。L2 用语义相似度匹配（embedding 余弦相似度 > 0.95 就认为是同一个问题的不同表述方式），命中率约 25%，TTL 可以稍长到 30 分钟。两者叠加后整体缓存命中率可以达到 55% 左右，意味着超过一半的查询可以直接从缓存返回，节省了大量 LLM API 调用。

但缓存也带来了一个微妙的问题：**如何避免缓存污染**？假设有人问了一个带有误导性的问题（"听说公司要取消年终奖了是真的吗？"），系统基于当时的知识库给出了否定的回答并缓存了这个结果。后来公司确实调整了奖金政策，知识库更新了，但缓存还没过期——其他用户问类似问题时仍然得到旧的否定回答。解决方案是：在数据同步完成后主动失效相关缓存（cache invalidation），而不是完全依赖 TTL 过期。

## 异步任务架构

除了实时的问答请求之外，系统还有大量不需要立即完成的后台任务：数据同步、索引构建、评估运行、报告生成等等。这些任务通过 Celery + Redis/RabbitMQ 来管理：

```
Async Task Architecture

                    ┌─────────────┐
                    │  FastAPI     │
                    │  (Producer)  │
                    └──────┬──────┘
                           │ send_task()
                           ▼
                    ┌─────────────┐
                    │  Redis /    │
                    │  RabbitMQ   │  ← 消息Broker
                    │  (Broker)   │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
       ┌──────────┐ ┌──────────┐ ┌──────────┐
       │ Worker   │ │ Worker   │ │ Worker   │
       · Index    │ │ · Sync   │ │ · Eval   │
       · Embed    │ │ · Parse  │ │ · Report │
       └──────────┘ └──────────┘ └──────────┘
              │            │            │
              ▼            ▼            ▼
       ┌────────────────────────────────┐
       │  Task Result Backend (Redis)   │
       │  · 任务状态追踪                 │
       │  · 结果临时存储                 │
       │  · 重试计数                     │
       └────────────────────────────────┘
```

Celery 任务的定义大致如下：

```python
from celery import Celery
from typing import Optional
import logging

logger = logging.getLogger(__name__)

celery_app = Celery(
    "enterprise_kb",
    broker="redis://redis:6379/1",
    backend="redis://redis:6379/2",
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Shanghai",
    task_track_started=True,
    task_time_limit=3600,      # 单个任务最大执行时间 1 小时
    worker_prefetch_multiplier=1,  # 公平调度，不要预取太多任务
    task_acks_late=True,        # 任务完成后再确认（防止worker崩溃丢失任务）
)


@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
def build_index_task(self, source_id: int, force_rebuild: bool = False):
    """
    异步构建/重建指定数据源的索引

    Args:
        source_id: 数据源ID
        force_rebuild: 是否强制全量重建（否则尝试增量）
    """
    from app.core.rag_engine import RAGEngine
    from app.data.sync.change_detector import ChangeDetector

    try:
        logger.info(f"[Task-{self.request.id}] 开始构建索引, source_id={source_id}")

        engine = RAGEngine()

        if not force_rebuild:
            detector = ChangeDetector(source_id)
            changed_docs = detector.detect_changes()

            if not changed_docs:
                logger.info(f"[Task-{self.request.id}] 无变更，跳过索引构建")
                return {"status": "skipped", "reason": "no_changes"}

            logger.info(f"[Task-{self.request.id}] 检测到 {len(changed_docs)} 个文档变更")
            result = engine.incremental_index(source_id, changed_docs)
        else:
            result = engine.full_rebuild_index(source_id)

        logger.info(f"[Task-{self.request.id}] 索引构建完成: {result}")
        return {"status": "success", **result}

    except Exception as exc:
        logger.error(f"[Task-{self.request.id}] 索引构建失败: {exc}", exc_info=True)
        raise self.retry(exc=exc, countdown=60 * (self.request.retries + 1))


@celery_app.task(bind=True, max_retries=2)
def run_evaluation_task(self, eval_config: dict):
    """
    运行评估任务
    """
    try:
        from app.utils.evaluation_runner import EvaluationRunner

        runner = EvaluationRunner(eval_config)
        results = runner.run()

        logger.info(f"[Task-{self.request.id}] 评估完成")
        return {"status": "success", "results": results}

    except Exception as exc:
        logger.error(f"[Task-{self.request.id}] 评估失败: {exc}", exc_info=True)
        raise self.retry(exc=exc)
```

注意 `build_index_task` 中的 `force_rebuild` 参数和增量检测逻辑。全量重建一个大型数据源的索引可能需要几分钟甚至十几分钟（取决于文档数量和 embedding 模型的速度），而且会消耗大量 API 额度。所以正常情况下应该优先走增量路径——只处理新增、修改和删除的文档。只有在检测到增量状态不一致（比如上次同步中断了）的时候才需要强制全量重建。

`task_acks_late=True` 是一个重要的可靠性设置。默认情况下 Celery 在 worker 收到任务时就发送 ACK（确认消息），这意味着如果 worker 在处理过程中崩溃了，这个任务就永远丢失了。设置为 late ack 后，只有任务成功完成（或达到最大重试次数）后才确认，broker 会保留消息直到确认为止。代价是如果 worker 崩溃了，正在处理的任务会被另一个 worker 重新执行（幂等性很重要！）。

## 数据同步架构

数据同步是保证知识库时效性的基础。我们的同步架构支持三种触发模式：

```
Data Sync Architecture Triggers

┌─────────────────────────────────────────────────────┐
│                  Sync Orchestrator                   │
│                                                     │
│  ┌──────────────┐  ┌──────────────┐                │
│  │ Scheduled    │  │ Event-driven │                │
│  │ (定时触发)    │  │ (事件触发)    │                │
│  │              │  │              │                │
│  │ Cron:        │  │ · Webhook    │                │
│  │ · 每6小时全量 │  │ · 文件监听   │                │
│  │ · 每小时增量  │  │ · DB CDC     │                │
│  └──────┬───────┘  └──────┬───────┘                │
│         │                  │                        │
│         └────────┬─────────┘                        │
│                  ▼                                  │
│         ┌──────────────────┐                        │
│         │  Sync Executor    │                        │
│         │                  │                        │
│         │  1. Fetch Source  │                        │
│         │  2. Diff & Classify│                        │
│         │  3. Parse Docs    │                        │
│         │  4. Chunk Nodes   │                        │
│         │  5. Embed Vectors │                        │
│         │  6. Update Store   │                        │
│         └────────┬─────────┘                        │
│                  │                                  │
│         ┌────────▼─────────┐                        │
│         │  Status Tracker   │                        │
│         │  · Progress %     │                        │
│         │  · Error Log      │                        │
│         │  · Doc Count Δ    │                        │
│         └──────────────────┘                        │
└─────────────────────────────────────────────────────┘
```

三种触发模式的适用场景各不相同：

**定时触发（Scheduled）**：适用于大多数稳定的数据源。比如 Confluence 页面通常不会有频繁变动，每 6 小时同步一次就够了；FAQ 数据库可能有客服团队日常更新，每小时同步一次更合理。定时触发的优点是实现简单、可预测性强；缺点是有延迟——新内容最多要等一个同步周期才能被检索到。

**Webhook 触发（Event-driven）**：适用于支持 webhook 回调的外部系统。比如 GitHub 仓库有 push 事件时自动触发同步、Notion 页面更新时收到通知。这种方式延迟最低（秒级），但实现复杂度高——需要处理 webhook 的鉴权、幂等性、失败重试等问题。

**文件监听（Watchdog）**：适用于本地或 NAS 上的文件目录。使用 Python 的 `watchdog` 库监控目录变化（创建/修改/删除文件），变动的文件名会被放入队列等待处理。适合开发测试环境使用，生产环境中建议还是走定时或事件驱动。

无论哪种触发方式，同步的核心执行逻辑是一致的：拉取数据 → 差异比对 → 解析文档 → 分块切分 → 向量化 → 写入存储。下一节我们会深入到每个核心模块的具体代码实现。

## 总结

