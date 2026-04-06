# 9.5 部署运维与性能优化

## 从"能跑"到"跑好"：生产化的最后一公里

代码写完了、API 调通了、前端也对接好了——现在系统在开发环境上运行得很好，你用 Postman 发了几个测试请求都得到了满意的回答。但这时候如果你跟运维同事说"帮我部署上线吧"，他可能会问一堆让你措手不及的问题：Dockerfile 写了吗？资源需要多少？怎么保证高可用？监控怎么做？出了问题怎么排查？

这一节就是回答这些问题的。我们会从容器化部署开始，逐步覆盖反向代理配置、Docker Compose 编排、监控告警搭建、性能调优策略以及故障排查方法论。目标是让这个系统能够真正在生产环境中稳定运行。

## Docker 容器化

```dockerfile
# 多阶段构建：构建阶段 + 运行阶段
# ============================================
# Stage 1: Builder
FROM python:3.11-slim as builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# Stage 2: Runtime
FROM python:3.11-slim as runtime

LABEL maintainer="your-team@company.com"
LABEL description="Enterprise KB QA System"

RUN apt-get update && apt-get install -y --no-install-recommends \
    # PDF 解析依赖
    libmupdf-dev \
    tesseract-ocr \
    tesseract-ocr-chi-sim \
    tesseract-ocr-chi-tra \
    # 网络工具（调试用）
    curl \
    iputils-ping \
    dnsutils \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

WORKDIR /app

# 从 builder 阶段复制安装好的包
COPY --from=builder /install /usr/local

# 复制应用代码
COPY --chown=appuser:appuser . .

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

关于这个 Dockerfile 有几个重要的工程实践值得解释：

**多阶段构建（Multi-stage Build）**：第一阶段 (`builder`) 安装所有编译依赖和 Python 包；第二阶段 (`runtime`) 只复制编译好的包和运行时需要的库文件。最终镜像的大小通常只有完整构建的一半左右——因为 `build-essential`、`gcc` 这些编译工具不会被包含进最终的运行镜像。

**非 root 用户运行**：`USER appuser` 确保应用以非特权用户身份运行。这是一个基本的安全最佳实践——如果攻击者设法利用了应用中的漏洞执行任意代码，他们也只能以低权限用户的身份操作，无法修改系统文件或安装后门程序。

**HEALTHCHECK 指令**：Docker 和 Kubernetes 都依赖健康检查来判断容器是否正常运行。这里我们每 30 秒检查一次 `/api/health` 端点，超时 5 秒，启动后有 10 秒的宽限期（让服务有时间完成初始化），连续失败 3 次则标记为 unhealthy。Kubernetes 的 liveness probe 和 readiness probe 可以直接使用这个检查机制。

**Tesseract OCR 安装**：如果你的知识库中有扫描版 PDF 或图片格式的文档，Tesseract 是必不可少的 OCR 引擎。`tesseract-ocr-chi-sim` 和 `tesseract-ocr-chi-tra` 分别是简体中文和繁体中文的语言包。注意这会让镜像增大约 100MB——如果不需要 OCR 功能可以去掉这两行。

## Docker Compose 编排

```yaml
version: "3.8"

services:
  # ============================
  # 应用服务
  # ============================
  app:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: kb-app
    restart: unless-stopped
    ports:
      - "8000:8000"
    environment:
      - APP_ENV=production
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - COHERE_API_KEY=${COHERE_API_KEY}
      - DATABASE_URL=postgresql://kb_user:${DB_PASSWORD}@postgres:5432/kb_db
      - REDIS_URL=redis://redis:6379/0
      - QDRANT_URL=http://qdrant:6333
      - JWT_SECRET=${JWT_SECRET}
      - LOG_LEVEL=INFO
      - RAG_CHUNK_SIZE=512
      - RAG_SIMILARITY_TOP_K=15
      - RAG_RERANK_TOP_K=5
      - WORKERS=1
    volumes:
      - ./data:/app/data:ro              # 只读挂载文档数据目录
      - ./logs:/app/logs                 # 日志持久化
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_started
      qdrant:
        condition: service_started
    networks:
      - kb-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/health"]
      interval: 30s
      timeout: 5s
      retries: 3
    deploy:
      resources:
        limits:
          cpus: "4.0"
          memory: 4G
        reservations:
          cpus: "1.0"
          memory: 1G

  # ============================
  # PostgreSQL
  # ============================
  postgres:
    image: postgres:16-alpine
    container_name: kb-postgres
    restart: unless-stopped
    environment:
      POSTGRES_USER: kb_user
      POSTGRES_PASSWORD: ${DB_PASSWORD}
      POSTGRES_DB: kb_db
    volumes:
      - pg_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql:ro
    ports:
      - "5432:5432"
    networks:
      - kb-network
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U kb_user -d kb_db"]
      interval: 10s
      timeout: 5s
      retries: 5

  # ============================
  # Redis
  # ============================
  redis:
    image: redis:7-alpine
    container_name: kb-redis
    restart: unless-stopped
    command: >
      redis-server
      --maxmemory 512mb
      --maxmemory-policy allkeys-lru
      --appendonly yes
      --appendfsync everysec
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
    networks:
      - kb-network

  # ============================
  # Qdrant 向量数据库
  # ============================
  qdrant:
    image: qdrant/qdrant:v1.8.4
    container_name: kb-qdrant
    restart: unless-stopped
    volumes:
      - qdrant_data:/qdrant/storage
    ports:
      - "6333:6333"   # REST API
      - "6334:6334"   # gRPC API (可选)
    environment:
      QDRANT__SERVICE__GRPC_ENABLED: "false"
      QDRANT__STORAGE__SNAPSHOT_PATH: "/qdrant/storage/snapshots"
    networks:
      - kb-network

  # ============================
  # Celery Worker (后台任务)
  # ============================
  celery-worker:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: kb-worker
    restart: unless-stopped
    command: celery -A app.celery_app worker --loglevel=info --concurrency=2
    environment:
      - APP_ENV=production
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - DATABASE_URL=postgresql://kb_user:${DB_PASSWORD}@postgres:5432/kb_db
      - REDIS_URL=redis://redis:6379/1
      - QDRANT_URL=http://qdrant:6333
    depends_on:
      - redis
      - postgres
      - qdrant
    networks:
      - kb-network
    deploy:
      resources:
        limits:
          memory: 2G

  # ============================
  # Celery Beat (定时任务调度)
  # ============================
  celery-beat:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: kb-beat
    restart: unless-stopped
    command: celery -A app.celery_app beat --loglevel=info
    environment:
      - REDIS_URL=redis://redis:6379/1
    depends_on:
      - redis
    networks:
      - kb-network

  # ============================
  # Nginx 反向代理
  # ============================
  nginx:
    image: nginx:1.25-alpine
    container_name: kb-nginx
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/conf.d/default.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
      - ./frontend/dist:/usr/share/nginx/html:ro
    depends_on:
      app:
        condition: service_healthy
    networks:
      - kb-network


volumes:
  pg_data:
  redis_data:
  qdrant_data:

networks:
  kb-network:
    driver: bridge
```

这个 Docker Compose 文件定义了一个完整的 7 容器编排方案。让我们逐一分析关键设计决策：

### 资源限制的设定依据

`deploy.resources.limits` 中给 app 服务设置了 4 CPU + 4GB 内存的上限。这个数字不是随便写的，而是基于以下估算：

- **FastAPI + uvicorn worker**：基础内存约 200MB
- **LlamaIndex 引擎 + 索引缓存**：1500 个文档 × 平均 500 字/node × 约 50 nodes/doc ≈ 200MB 的文本数据，加上 embedding 缓存约 500MB
- **BM25 倒排索引**：约 100MB
- **并发请求缓冲**：假设同时处理 20 个请求，每个请求的中间数据约 20MB → 400MB
- **Python 运行时开销**：约 300-500MB

总计约 1.5-2GB 正常使用量，设置 4GB 上限留有足够余量应对突发情况。CPU 方面，RAG 流水线中真正的 CPU 密集型操作不多（主要是 BM25 排序和一些字符串处理），大部分时间在等 I/O（LLM API、向量数据库查询），所以 4 核对于单 worker 来说绰绰有余。

### Redis 内存策略

`--maxmemory 512mb --maxmemory-policy allkeys-lru` 这两个参数组合的含义是：当 Redis 使用内存超过 512MB 时，自动淘汰最近最少使用的 key。这对我们的场景很合适——缓存的数据本来就是"有了更好没有也能工作"，淘汰一些旧缓存总比让 Redis OOM 崩溃要好。`--appendonly yes` 开启 AOF 持久化，即使 Redis 重启也不会丢失所有缓存数据（虽然缓存丢失只是性能问题而非数据丢失问题）。

### Celery Worker 并发度

`--concurrency=2` 表示每个 worker 同时最多处理 2 个任务。为什么只设 2 而不是更多？因为 Celery worker 的主要任务是索引构建和评估运行，这两种任务都是 I/O 密集型的（大量调用 OpenAI API），但也会消耗不少内存（加载文档、生成 embedding）。并发太高可能导致内存溢出，特别是在 embedding 大批量文档的时候。如果任务队列积压严重，更好的做法是增加 worker 数量而不是单个 worker 的并发度。

## Nginx 配置

```nginx
# nginx/conf.d/default.conf — 生产环境 Nginx 配置

upstream backend {
    server app:8000;
    keepalive 32;              # 长连接池大小
    keepalive_timeout 60s;     # 长连接空闲超时
}

# 速率限制定义
limit_req_zone $binary_remote_addr zone=general_limit:10m rate=30r/m;
limit_req_zone $binary_remote_addr zone=chat_limit:10m rate=10r/m;
limit_conn_zone $binary_remote_addr zone=conn_limit:10m;

server {
    listen 80;
    server_name _;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl http2;
    server_name kb.yourcompany.com;

    ssl_certificate     /etc/nginx/ssl/fullchain.pem;
    ssl_certificate_key /etc/nginx/ssl/privkey.pem;
    ssl_protocols       TLSv1.2 TLSv1.3;
    ssl_ciphers         ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_cache   shared:SSL:10m;
    ssl_session_timeout 1d;

    # 安全头
    add_header X-Frame-Options DENY always;
    add_header X-Content-Type-Options nosniff always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy strict-origin-when-cross-origin always;
    add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; connect-src 'self' https://api.openai.com; img-src 'self' data:;" always;

    # 静态资源（前端）
    location / {
        root /usr/share/nginx/html;
        try_files $uri $uri/ /index.html;

        # 静态资源长期缓存
        location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg|woff|woff2)$ {
            expires 30d;
            add_header Cache-Control "public, immutable";
            access_log off;
        }
    }

    # API 代理
    location /api/ {
        limit_req zone=general_limit burst=40 nodelay;
        limit_conn conn_limit 10;

        proxy_pass http://backend;
        proxy_http_version 1.1;

        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header Connection "";

        # 超时设置（LLM 调用可能较慢）
        proxy_connect_timeout 10s;
        proxy_send_timeout 120s;
        proxy_read_timeout 300s;

        # SSE 流式响应支持
        proxy_buffering off;
        proxy_cache off;
        chunked_transfer_encoding on;
    }

    # 问答接口单独限流
    location /api/v1/chat {
        limit_req zone=chat_limit burst=15 nodelay;

        proxy_pass http://backend;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header Connection "";

        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 300s;
    }

    # 健康检查端点不限流
    location = /api/health {
        proxy_pass http://backend;
        access_log off;
    }

    # 日志格式
    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                    '$status $body_bytes_sent "$http_referer" '
                    '"$http_user_agent" '
                    'rt=$request_time uct="$upstream_connect_time" '
                    'uht="$upstream_header_time" urt="$upstream_response_time"';

    access_log /var/log/nginx/access.log main;
    error_log  /var/log/nginx/error.log warn;
}
```

Nginx 配置中有几个与普通 Web 应用不同的特殊考量：

**超时时间的激进设置**：`proxy_read_timeout 300s`（5 分钟）看起来非常长——一般的 Web API 设置 30 秒就足够了。但对于 RAG 应用来说，一个复杂的问答请求可能经历以下步骤：HyDE 生成（~3s）+ 双路检索（~1s）+ Rerank（~2s）+ LLM 生成回答（最长可达 30-60s 如果回答很长）+ 后处理（<1s）。极端情况下整个流程可能接近 2 分钟。如果把 read_timeout 设为 30 秒，很多正常的长回答都会被 Nginx 主动断开连接，用户看到的就是"网络错误"。当然，5 分钟也不是无限——如果真的超过 5 分钟还没返回，那大概率是哪里卡死了，断开也是合理的。

**安全头的完整性**：`Content-Security-Policy`（CSP）是最重要的一条。它限制了页面可以加载的资源来源，能有效防止 XSS 攻击。注意其中的 `connect-src 'self' https://api.openai.com`——因为我们的前端可能会直接调用 OpenAI API（比如做实时 streaming 时），所以需要把 openai.com 加入白名单。如果你的架构是通过后端代理所有外部调用（更安全的做法），就不需要这一条。

## 监控告警体系

```
Monitoring Architecture

┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Prometheus  │←───│ Exporters   │    │   Alerts    │
│ (TSDB)      │    │             │    │             │
│             │    │ · node_exp  │    │ ┌─────────┐ │
│             │    │ · pgsql_exp │    │ │P99 > 8s │ │
│             │    │ · redis_exp │    │ │Error%>5%│ │
│             │    │ · custom    │    │ │Mem>90%  │ │
└──────┬──────┘    └─────────────┘    └─────┬───────┘
       │                                    │
       ▼                                    ▼
┌─────────────┐                     ┌──────────────┐
│  Grafana    │                     │ Alertmanager │
│ (Dashboard) │                     │ (通知路由)    │
│             │                     │              │
│ · 查询延迟   │                    │ → 钉钉/Slack │
│ · QPS 趋势   │                    │ → 邮件       │
│ · 错误率     │                    │ → 电话(紧急)  │
│ · 资源占用   │                    └──────────────┘
└─────────────┘
```

下面是自定义的 Prometheus Exporter 实现，用于暴露 RAG 特有的业务指标：

```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time
import functools

# 定义指标
QUERY_TOTAL = Counter(
    'kb_query_total',
    'Total number of queries',
    ['query_type', 'result_status']  # factual/success, comparative/error
)

QUERY_LATENCY = Histogram(
    'kb_query_latency_seconds',
    'Query latency in seconds',
    ['query_type'],
    buckets=[0.5, 1, 2, 3, 5, 8, 10, 15, 20, 30],
)

CONFIDENCE_SCORE = Histogram(
    'kb_confidence_score',
    'Answer confidence score distribution',
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

RETRIEVAL_NODES = Histogram(
    'kb_retrieval_nodes_count',
    'Number of nodes retrieved at each stage',
    ['stage'],  # raw/filtered/reranked/final
    buckets=[1, 3, 5, 10, 15, 20, 30, 50],
)

TOKENS_USED = Counter(
    'kb_llm_tokens_total',
    'Total LLM tokens consumed',
    ['model', 'operation'],  # gpt-4o/query, text-embedding-large/index
)

CACHE_HIT_RATE = Gauge(
    'kb_cache_hit_rate',
    'Cache hit rate (0-1)',
)

ACTIVE_SESSIONS = Gauge(
    'kb_active_sessions',
    'Number of active chat sessions',
)

INDEX_DOCUMENT_COUNT = Gauge(
    'kb_index_document_count',
    'Number of documents in the index',
)


def track_query(func):
    """装饰器：自动记录查询指标"""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        query_type = kwargs.get('query_type', 'unknown')
        start = time.time()

        try:
            result = await func(*args, **kwargs)
            latency = time.time() - start

            QUERY_LATENCY.labels(query_type=query_type).observe(latency)
            QUERY_TOTAL.labels(query_type=query_type, result_status='success').inc()
            CONFIDENCE_SCORE.observe(result.confidence if hasattr(result, 'confidence') else 0)

            if hasattr(result, 'retrieval_details'):
                rd = result.retrieval_details
                RETRIEVAL_NODES.labels(stage='raw').observe(rd.get('raw_retrieved', 0))
                RETRIEVAL_NODES.labels(stage='filtered').observe(rd.get('after_permission_filter', 0))
                RETRIEVAL_NODES.labels(stage='reranked').observe(rd.get('after_rerank', 0))
                RETRIEVAL_NODES.labels(stage='final').observe(rd.get('after_postprocess', 0))

            return result

        except Exception as e:
            latency = time.time() - start
            QUERY_LATENCY.labels(query_type=query_type).observe(latency)
            QUERY_TOTAL.labels(query_type=query_type, result_status='error').inc()
            raise

    return wrapper
```

Prometheus 的四种核心指标类型在这个系统中都用到了：

- **Counter**（计数器）：`QUERY_TOTAL` 和 `TOKENS_USED`——只增不减的累计值，用于统计总量和速率
- **Histogram**（直方图）：`QUERY_LATENCY` 和 `CONFIDENCE_SCORE`——记录值的分布情况，可以计算 P50/P95/P99 分位数
- **Gauge**（仪表盘）：`CACHE_HIT_RATE` 和 `ACTIVE_SESSIONS`——可以上下波动的瞬时值
- `track_query` 装饰器展示了如何无侵入地将指标收集融入业务代码——只需要在被监控的函数上加一个 `@track_query` 装饰器，所有指标就会自动记录

对应的 Grafana 告警规则示例：

```yaml
groups:
  - name: kb-alerts
    rules:
      - alert: HighQueryLatencyP99
        expr: histogram_quantile(0.99, sum(rate(kb_query_latency_seconds_bucket[5m])) by (le)) > 8
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "RAG 查询 P99 延迟超过 8 秒"
          description: "当前 P99={{ $value }}秒，阈值=8秒"

      - alert: HighErrorRate
        expr: |
          sum(rate(kb_query_total{result_status="error"}[5m]))
          /
          sum(rate(kb_query_total[5m])) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "RAG 查询错误率超过 5%"
          description: "当前错误率={{ $value | humanizePercentage }}"

      - alert: LowConfidenceAnswers
        expr: |
          histogram_quantile(0.5, sum(rate(kb_confidence_score_bucket[10m])) by (le)) < 0.5
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "回答置信度中位数低于 0.5"
          description: "可能是检索质量下降或知识库内容不足"

      - alert: HighMemoryUsage
        expr: container_memory_usage_bytes{name="kb-app"} / container_spec_memory_limit_bytes{name="kb-app"} > 0.9
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "App 容器内存使用超过 90%"
          description: "当前使用率={{ $value | humanizePercentage }}"
```

## 性能调优实战

部署上线后第一件事不是庆祝而是压测和调优。下面是一套完整的性能优化方法论：

### 第一步：建立基线

先用压力测试工具获取系统的基准性能数据：

```python
import asyncio
import aiohttp
import time
import statistics
from dataclasses import dataclass


@dataclass
class LoadTestResult:
    total_requests: int
    successful: int
    failed: int
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    requests_per_second: float
    errors: list[str]


async def load_test(
    base_url: str,
    token: str,
    questions: list[str],
    concurrency: int = 10,
    duration_seconds: int = 60,
) -> LoadTestResult:
    """
    RAG 系统负载测试

    Args:
        base_url: API 基础地址
        token: JWT Token
        questions: 测试问题列表
        concurrency: 并发数
        duration_seconds: 测试持续时间
    """
    latencies = []
    errors = []
    completed = 0
    failed = 0
    semaphore = asyncio.Semaphore(concurrency)
    end_time = time.time() + duration_seconds
    question_idx = 0

    async def single_request(session, question):
        nonlocal completed, failed, question_idx
        try:
            start = time.perf_counter()
            async with session.post(
                f"{base_url}/api/v1/chat",
                json={"message": question, "stream": False},
                headers={"Authorization": f"Bearer {token}"},
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                elapsed = (time.perf_counter() - start) * 1000
                if resp.status == 200:
                    latencies.append(elapsed)
                    completed += 1
                else:
                    body = await resp.text()
                    errors.append(f"{resp.status}: {body[:100]}")
                    failed += 1
        except Exception as e:
            errors.append(str(e))
            failed += 1

    async def bounded_request(session):
        nonlocal question_idx
        while time.time() < end_time:
            question = questions[question_idx % len(questions)]
            question_idx += 1
            async with semaphore:
                await single_request(session, question)

    connector = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [bounded_request(session) for _ in range(concurrency)]
        await asyncio.gather(*tasks)

    sorted_latencies = sorted(latencies)
    total = completed + failed
    actual_duration = min(duration_seconds, time.time() - (end_time - duration_seconds))

    return LoadTestResult(
        total_requests=total,
        successful=completed,
        failed=failed,
        avg_latency_ms=statistics.mean(latencies) if latencies else 0,
        p50_latency_ms=sorted_latencies[int(len(sorted_latencies) * 0.50)] if sorted_latencies else 0,
        p95_latency_ms=sorted_latencies[int(len(sorted_latencies) * 0.95)] if sorted_latencies else 0,
        p99_latency_ms=sorted_latencies[int(len(sorted_latencies) * 0.99)] if sorted_latencies else 0,
        requests_per_second=total / actual_duration if actual_duration > 0 else 0,
        errors=errors[:20],
    )


# 使用示例
TEST_QUESTIONS = [
    "公司年假政策是怎样的？",
    "如何申请差旅报销？",
    "Pro 版和企业版有什么区别？",
    "API 返回 403 错误怎么办？",
    "什么是 RBAC 权限模型？",
    "新员工入职流程是什么？",
    "VPN 怎么连接？",
    "工资什么时候发放？",
]

async def main():
    result = await load_test(
        base_url="http://localhost:8000",
        token="your-test-token",
        questions=TEST_QUESTIONS,
        concurrency=10,
        duration_seconds=60,
    )

    print(f"=== 负载测试结果 ===")
    print(f"总请求数: {result.total_requests}")
    print(f"成功: {result.successful}, 失败: {result.failed}")
    print(f"QPS: {result.requests_per_second:.1f}")
    print(f"平均延迟: {result.avg_latency_ms:.0f}ms")
    print(f"P50: {result.p50_latency_ms:.0f}ms")
    print(f"P95: {result.p95_latency_ms:.0f}ms")
    print(f"P99: {result.p99_latency_ms:.0f}ms")

    if result.errors:
        print(f"\n错误样本:")
        for err in result.errors[:5]:
            print(f"  - {err}")


if __name__ == "__main__":
    asyncio.run(main())
```

### 第二步：定位瓶颈

拿到基线数据后，如果 P99 延迟超标（> 8s），需要逐段分析瓶颈在哪。RAG 流水线的主要耗时分布通常是：

```
典型耗时分布（总 P99 = 6.5s）

├── Query Preprocess + Classify     ~  50ms   (0.8%)
├── HyDE Transform (可选)           ~3000ms   (46%)  ← 最大瓶颈
├── Vector Search (Qdrant)          ~ 200ms   (3%)
├── BM25 Search                     ~  30ms   (0.5%)
├── RRF Fusion                      ~  10ms   (0.2%)
├── Permission Filter               ~   5ms   (0.1%)
├── Reranking (Cohere)             ~ 500ms   (8%)
├── Postprocessing                  ~  10ms   (0.2%)
├── LLM Synthesis (GPT-4o)         ~2500ms   (38%)  ← 第二大瓶颈
└── Citation Building               ~  20ms   (0.3%)
```

从这个分布可以看出，**LLM 相关的操作（HyDE + Synthesis）占据了 84% 的总耗时**。这是 RAG 应用的本质特征——你不可能绕过 LLM 调用，但可以通过以下方式优化：

### 第三步：针对性优化

**优化一：减少 LLM 调用次数**

```python
# 原始方案：每次查询都调两次 LLM（HyDE + Synthesis）
# 优化方案：对高频问题跳过 HyDE，直接用原始 query 做向量搜索

class SmartQueryEngine:
    """智能查询引擎——根据问题类型动态调整策略"""

    def __init__(self, rag_engine):
        self.rag_engine = rag_engine
        self.cache = LRUCache(maxsize=1000)

    async def query(self, question, **kwargs):
        query_type = self.rag_engine._query_classifier.classify(question)

        # 事实性问题通常语义明确，不需要 HyDE
        if query_type.question_type in (
            QuestionType.FACTUAL,
            QuestionType.DEFINITIONAL,
        ):
            kwargs["use_hyde"] = False

        # 对比性问题需要更大的 top_k
        if query_type.question_type == QuestionType.COMPARATIVE:
            kwargs["similarity_top_k"] = 25

        # 排查类问题优先关键词匹配
        if query_type.question_type == QuestionType.TROUBLESHOOTING:
            kwargs["keyword_boost"] = 2.0

        return await self.rag_engine.query(question, **kwargs)
```

这个优化可以将约 40% 的查询（事实性和定义性问题）省掉 HyDE 那次额外的 LLM 调用，平均节省 ~2-3 秒。

**优化二：并行化独立操作**

```python
import asyncio

async def parallel_retrieve(self, query: str, allowed_doc_ids: list[str]):
    """并行执行向量和关键词检索"""
    vector_task = asyncio.create_task(
        self._vector_retriever.aretrieve(query)
    )
    bm25_task = asyncio.create_task(
        self._bm25_retriever.aretrieve(query)
    )

    vector_results, bm25_results = await asyncio.gather(
        vector_task, bm25_task
    )
    return self._rrf_fusion(vector_results, bm25_results)
```

原本串行的两个检索操作（向量 ~200ms + BM25 ~30ms = 230ms）变成并行后取 max(200, 30) = 200ms，节省 30ms。虽然绝对值不大，但在高并发下积少成多。

**优化三：Embedding 缓存**

```python
import hashlib
import json

class EmbeddingCache:
    """基于文本哈希的 Embedding 结果缓存"""

    def __init__(self, redis_client, ttl_seconds=86400):  # 默认24小时
        self.redis = redis_client
        self.ttl = ttl_seconds
        self.prefix = "emb:cache:"

    def _cache_key(self, text: str) -> str:
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return f"{self.prefix}{text_hash}"

    async def get_or_compute(self, texts: list[str], embed_fn) -> list[list[float]]:
        """批量获取或计算 embedding"""
        results = [None] * len(texts)
        cache_miss_indices = []

        for i, text in enumerate(texts):
            cached = await self.redis.get(self._cache_key(text))
            if cached:
                results[i] = json.loads(cached)
            else:
                cache_miss_indices.append((i, text))

        if cache_miss_indices:
            miss_texts = [t for _, t in cache_miss_indices]
            embeddings = await embed_fn(miss_texts)

            for (idx, _), emb in zip(cache_miss_indices, embeddings):
                results[idx] = emb
                await self.redis.setex(
                    self._cache_key(miss_texts[cache_miss_indices.index((idx, miss_texts[0]))][1]),
                    self.ttl,
                    json.dumps(emb),
                )

        return results
```

Embedding 缓存的效果取决于问题的重复率。在企业场景中，重复率通常在 30%-55%（前面讨论过），这意味着超过一半的 embedding 计算可以被缓存命中替代。每次 embedding 调用的成本约为 0.00002 美元（text-embedding-3-large），听起来微不足道，但在每天 5000 次查询的场景下一年就能省下约 36 美元——更重要的是它减少了约 100-200ms 的 API 延迟。

**优化四：流式输出降低感知延迟**

这不是真正的性能优化（总耗时不变），但它极大地改善了用户体验：

```javascript
// 用户感知到的延迟对比：
//
// 非流式：等待 6.5s → 一次性显示全部回答
// 用户感受："好慢啊，系统是不是卡了？"
//
// 流式：等待 1.5s → 开始显示第一个字 → 逐步显示剩余内容
// 用户感受："哦，它在思考呢，已经出结果了"
//
// 感知延迟从 6.5s 降低到了 1.5s！
```

### 第四步：验证优化效果

每次优化后都要重新跑负载测试来验证效果。建议建立一个简单的回归表：

```
优化轮次 | P50(ms) | P95(ms) | P99(ms) | QPS  | 主要改动
─────────┼─────────┼─────────┼─────────┼──────┼────────────
Baseline |  1800   |  4500   |  6500   | 12   | 初始状态
Round 1  |  1600   |  4000   |  5800   | 14   | 跳过事实类HyDE
Round 2  |  1550   |  3800   |  5500   | 15   | 检索并行化
Round 3  |  1400   |  3500   |  5000   | 18   | Embedding缓存
Round 4  |  1300   |  3200   |  4500   | 20   | Redis查询缓存
Target   |  2000   |  5000   |  8000   | 50   | 目标值
```

## 故障排查手册

最后，整理一份常见故障的快速排查指南：

```
┌─────────────────────────────────────────────────────────────┐
│                   故障快速排查树                             │
│                                                             │
│  用户报告"系统有问题"                                        │
│       │                                                     │
│       ├── 能访问页面吗？                                     │
│       │   ├── NO → 检查 Nginx/DNS/网络                       │
│       │   └── YES → 继续                                    │
│       │                                                     │
│       ├── 登录是否正常？                                      │
│       │   ├── NO → 检查 SSO/JWT/Token 过期                   │
│       │   └── YES → 继续                                    │
│       │                                                     │
│       ├── 提问后是否有响应？                                  │
│       │   ├── 超时无响应 →                                  │
│       │   │   ├── 检查 QPS 是否超限 (429?)                   │
│       │   │   ├── 检查 LLM API 是否可用 (OpenAI 状态页)       │
│       │   │   └── 检查 Qdrant 连接                           │
│       │   │                                                 │
│       │   ├── 返回错误信息 →                                 │
│       │   │   ├── 500 → 查看 app logs (Exception stack)      │
│       │   │   ├── 503 → 引擎未初始化 / 外部依赖不可用          │
│       │   │   └── 422 → 请求参数校验失败                      │
│       │   │                                                 │
│       │   └回答但质量差 →                                    │
│       │       ├── 回答不相关 → 检索质量问题                   │
│       │       │   ├── top_k 太小?                            │
│       │       │   ├── embedding 模型不匹配?                   │
│       │       │   └── 数据源未同步?                           │
│       │       ├── 回答有幻觉 →                               │
│       │       │   ├── system prompt 约束不足?                 │
│       │       │   └── REFINE 模式 vs COMPACT 切换             │
│       │       └── 回答太短/空 →                              │
│       │           ├── 相似度阈值过高过滤掉了所有节点?          │
│       │           └── 权限过滤把所有节点都过滤掉了?            │
│       │                                                     │
│       └── 关键日志命令                                       │
│           docker logs kb-app --tail 100                      │
│           docker logs kb-app 2>&1 | grep ERROR               │
│           curl -s http://localhost:8000/api/health           │
└─────────────────────────────────────────────────────────────┘
```

## 总结

