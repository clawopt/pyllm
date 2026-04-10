# 生产级架构设计

> **白板时间**：你在本地用 vLLM + LangChain 跑通了一个 RAG demo，效果很好。老板说："把它部署到生产环境，要支持 100 并发用户、99.9% 可用性、完整的监控告警。" 这不是简单地把 `python -m vllm` 放到后台——你需要一个经过设计的架构。这一节我们给出一个完整的生产级架构方案。

## 一、整体架构

### 1.1 分层架构图

```
                         ┌─────────────────────────────────┐
                         │        Client Layer             │
                         │  (Web App / Mobile / CLI)       │
                         └──────────────┬──────────────────┘
                                        │ HTTP/WS
                    ┌───────────────────▼───────────────────┐
                    │           API Gateway              │
                    │  (Nginx / Envoy / Kong)            │
                    │  ┌──────────────────────────────┐    │
                    │  │ • TLS 终结                   │    │
                    │  │ • Rate Limiting (per user/IP)│    │
                    │  │ • Auth (API Key / OAuth2)      │    │
                    │  │ • Request Logging            │    │
                    │  │ • Circuit Breaker            │    │
                    │  └──────────────────────────────┘    │
                    └──────────────┬───────────────────────┘
                                   │
                    ┌──────────────▼───────────────────────┐
                    │         Service Mesh / Sidecar         │
                    │  (可选: 认证/限流/日志/缓存)          │
                    └──────────────┬───────────────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                  │                        │
     ┌────────▼────────┐  ┌────────▼────────┐  ┌──────▼──────┐
     │   vLLM-A :8000   │  │   vLLM-B :8001   │  │ vLLM-C:8002 │
     │                 │  │                 │  │             │
     │ TP=2, AWQ INT4  │  │ TP=2, FP16       │  │ TP=1, FP16  │
     │ medical-lora    │  │ code-lora        │  │ creative-lora│
     └────────┬────────┘  └────────┬────────┘  └──────┬──────┘
              │                      │                │
              └──────────────────────┬┴────────────────┘
                                     │
              ┌──────────────────────▼──────────────────────┐
              │           Data Layer                       │
              │  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
              │  │ Redis    │  │ ChromaDB  │  │ PostgreSQL│  │
              │  │(语义缓存)│  │(向量索引)│  │(日志/审计)│  │
              │  └──────────┘  └──────────┘  └──────────┘  │
              └──────────────────────────────────────────────┘
              
              ┌──────────────────────────────────────────────┐
              │           Observability Layer               │
              │  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
              │  │Prometheus│  │Grafana   │  │Loki/ELK  │  │
              │  │(指标采集)│  │(可视化)  │  │(日志聚合)│  │
              │  └──────────┘  └──────────┘  └──────────┘  │
              └──────────────────────────────────────────────┘
```

## 二、Nginx 反向代理配置

### 2.1 生产级 Nginx 配置

```nginx
# /etc/nginx/conf.d/vllm.conf

upstream vllm_backend {
    server 127.0.0.1:8000;  # vLLM 实例 A
    server 127.0.0.1:8001;  # vLLM 实例 B (可选)
    
    # 健康检查
    keepalive 32;
    keepalive_timeout 60s;
    keepalive_requests 100;
}

# 限流区域定义
limit_req_zone $binary_remote_addr zone=general:10r/s;
limit_req_zone $binary_remote_addr zone=streaming:30r/s;

server {
    listen 443 ssl http2;
    server_name llm.yourcompany.com;

    # SSL 配置
    ssl_certificate     /etc/nginx/certs/fullchain.pem;
    ssl_certificate_key /etc/nginx/certs/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    # === SSE 流式输出关键配置 ===
    proxy_buffering off;          # ⭐ 必须！否则流式输出会卡住
    proxy_cache off;               # 禁用响应缓存
    chunked_transfer_encoding on;

    # 安全头
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;

    # === API 端点 ===
    location /v1/ {
        limit_req zone=general;
        
        proxy_pass http://vllm_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # 超时配置
        proxy_connect_timeout 10s;
        proxy_send_timeout 300s;       # 长生成任务可能需要几分钟
        proxy_read_timeout 300s;
        
        # WebSocket 支持（如需要）
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    # === 流式端点特殊处理 ===
    location ~* /v1/chat/completions$ {
        limit_req zone=streaming;
        
        proxy_pass http://vllm_backend;
        proxy_set_header Host $host;
        proxy_buffering off;            # 再次强调！
        proxy_cache off;
        
        proxy_connect_timeout 10s;
        proxy_send_timeout 600s;       # 流式连接保持更长时间
        proxy_read_timeout 600s;
    }

    # === 健康检查端点（无需认证）===
    location /health {
        proxy_pass http://vllm_backend;
        access_log off;
    }
    
    location /version {
        proxy_pass http://vllm_backend;
        access_log off;
    }

    # === 默认拒绝其他请求 ===
    location / {
        return 404 '{"error": "Not Found"}';
        default_type application/json;
    }

    # === 访问日志 ===
    access_log /var/log/nginx/vllm_access.log combined if=$loggable;
    error_log  /var/log/nginx/vllm_error.log warn;
}
```

## 三、语义缓存层

### 3.1 Redis 缓存实现

```python
import json
import hashlib
import time
import asyncio
import aiohttp
from dataclasses import dataclass
from typing import Optional


@dataclass
class CacheEntry:
    """缓存条目"""
    query: str
    response: str
    model: str
    created_at: float
    ttl: int = 3600  # 默认 1 小时


class SemanticCache:
    """基于语义相似度的智能缓存"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379",
                 similarity_threshold: float = 0.95,
                 max_cache_size: int = 10000):
        self.redis_url = redis_url
        self.threshold = similarity_threshold
        self.max_size = max_cache_size
    
    async def _get_embedding(self, text: str, embed_client) -> list:
        """获取文本的 embedding 向量"""
        resp = await embed_client.embeddings.create(
            model="BAAI/bge-m3",
            input=text,
        )
        return resp.data[0].embedding
    
    def _query_hash(self, query: str) -> str:
        """生成查询的 hash"""
        return hashlib.sha256(query.encode()).hexdigest()[:16]
    
    async def get(self, query: str, embed_client,
                ttl: Optional[int] = None) -> Optional[str]:
        """查询缓存"""
        
        import aioredis
        
        redis = aioredis.from_url(self.redis_url)
        
        cache_key = f"cache:query:{self._query_hash(query)}"
        
        # 检查精确匹配
        cached = await redis.get(cache_key)
        if cached:
            entry = json.loads(cached)
            
            # 检查 TTL
            if time.time() - entry["created_at"] < (ttl or entry["ttl"]):
                return entry["response"]
        
        # 检查语义相似命中（简化版：实际需要向量搜索）
        # 生产环境建议使用 Redis with RediSearch 或 Milvus
        return None
    
    async def set(self, query: str, response: str, 
                 model: str, embed_client, ttl: int = 3600):
        """写入缓存"""
        
        import aioredis
        
        redis = aioredis.from_url(self.redis_url)
        cache_key = f"cache:query:{self._query_hash(query)}"
        
        entry = CacheEntry(
            query=query[:200],  # 截断长查询
            response=response,
            model=model,
            created_at=time.time(),
            ttl=ttl,
        )
        
        await redis.setex(
            cache_key,
            ttl,
            json.dumps(entry.__dict__)
        )
```

## 四、熔断器与降级策略

### 4.1 Circuit Breaker 实现

```python
import time
import asyncio
from enum import Enum
from dataclasses import dataclass, field
from typing import Callable


class CircuitState(Enum):
    CLOSED = "closed"    # 正常工作
    OPEN = "open"        # 熔断中，快速失败
    HALF_OPEN = "half_open"  # 半开，试探恢复


@dataclass
class CircuitBreaker:
    """熔断器"""
    
    name: str
    failure_threshold: int = 5       # 连续失败 N 次触发熔断
    recovery_timeout: float = 30.0     # 熔断持续时间（秒）
    half_open_max_tries: int = 3       # 半开状态允许的试探次数
    
    _state: CircuitState = CircuitState.CLOSED
    _failure_count: int = 0
    _last_failure_time: float = 0
    _success_count: int = 0
    
    def can_execute(self) -> bool:
        """检查是否可以执行请求"""
        now = time.time()
        
        if self._state == CircuitState.CLOSED:
            return True
        
        elif self._state == CircuitState.OPEN:
            if now - self._last_failure_time >= self.recovery_timeout:
                self._state = CircuitState.HALF_OPEN
                self._success_count = 0
                return True
            return False
        
        else:  # HALF_OPEN
            return self._success_count < self.half_open_max_tries
    
    def record_success(self):
        """记录成功调用"""
        self._failure_count = 0
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.half_open_max_tries:
                self._state = CircuitState.CLOSED
    
    def record_failure(self):
        """记录失败调用"""
        self._failure_count += 1
        self._last_failure_time = time.time()
        
        if self._state == CircuitState.CLOSED and \
           self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN
        elif self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN


async def call_with_circuit_breaker(
    circuit: CircuitBreaker,
    func: Callable,
    fallback_func: Callable = None,
    *args, **kwargs
):
    """带熔断保护的调用"""
    
    if not circuit.can_execute():
        if fallback_func:
            return await fallback_func(*args, **kwargs)
        raise Exception(f"Circuit '{circuit.name}' is OPEN")
    
    try:
        result = await func(*args, **kwargs)
        circuit.record_success()
        return result
    except Exception as e:
        circuit.record_failure()
        raise
```

## 五、监控告警体系

### 5.1 Prometheus + Grafana Stack

```yaml
# prometheus.yml (scrape config)
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'vllm'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
    scrape_interval: 5s
```

**关键监控指标**：

| 指标 | Prometheus 名称 | 告警阈值 | 说明 |
|------|-----------------|---------|------|
| GPU Cache 使用率 | `vllm:gpu_cache_usage_perc` | > 90% | 即将 OOM |
| 排队请求数 | `vllm:num_requests_waiting` | > 50 | 吞吐不足 |
| 运行请求数 | `vllm:num_requests_running` | > max_seqs × 0.8 | 达到上限 |
| P99 TTFT | `vllm:time_to_first_token_seconds` (P99) | > 2s | 用户体验差 |
| P99 TPOT | `vllm:time_per_output_token_seconds` (P99) | > 100ms | 卡顿 |
| 错误率 | `vllm:request_failures_total` / total | > 1% | 服务异常 |
| QPS | 自计算 (requests/sec) | < 预期值 | 容量不足 |

## 六、完整 Docker Compose 生产部署

```yaml
# docker-compose.production.yml
version: '3.8'

services:
  # ===== vLLM 主服务 =====
  vllm-primary:
    image: vllm/vllm-openai:latest
    ports:
      - "8000:8000"
    volumes:
      - ./models:/root/.cache/huggingface/hub
      - ./adapters:/models/adapters
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu]
              device_ids: ['0', '1']
    command: >
      --model meta-llama/Meta-Llama-3.1-8B-Instruct-AWQ
      --quantization awq
      --tensor-parallel-size 2
      --enable-lora
      --lora-modules
        domain-expert=/models/adapters/domain-v1
      --max-loras 8
      --max-model-len 8192
      --gpu-memory-utilization 0.92
      --enable-prefix-caching
      --port 8000
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 15s
      timeout: 5s
      retries: 3
    networks:
      - vllm-network

  # ===== Nginx 反向代理 =====
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/conf.d/default.conf:ro
      - ./nginx/certs:/etc/nginx/certs:ro
    depends_on:
      - vllm-primary
    networks:
      - vllm-network

  # ===== Redis (语义缓存) =====
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    command: redis-server --maxmemory 512mb --maxmemory-policy allkeys-lru
    volumes:
      - redis-data:/data
    networks:
      - vllm-network

networks:
  vllm-network:
    driver: bridge
```

---

## 七、总结

本节完成了生产级架构的设计：

| 层次 | 组件 | 核心功能 |
|------|------|---------|
| **接入层** | Nginx | TLS、限流、SSE buffer 关闭、安全头 |
| **网关层** | API Gateway (Envoy/Kong) | Auth、Rate Limit、日志、Circuit Breaker |
| **服务层** | vLLM 多实例 | TP+LoRA、负载均衡、Prefix Cache |
| **数据层** | Redis + ChromaDB + PG | 语义缓存、向量存储、日志 |
| **可观测** | Prometheus + Grafana + Loki | 指标采集、可视化、日志聚合 |

**核心要点回顾**：

1. **`proxy_buffering off` 是 Nginx 配置的第一铁律**——没有它 SSE 流式输出无法正常工作
2. **语义缓存是降低成本和延迟的秘密武器**——相似问题直接返回缓存结果
3. **熔断器保护系统不因单点故障而雪崩**——自动降级到备用方案或返回友好错误
4. **多实例 + 负载均衡**是实现高可用的基础——不要把所有鸡蛋放一个篮子
5. **监控 → 告警 → 自动修复** 是运维的黄金闭环

至此，**Chapter 9（LangChain/LlamaIndex 集成）全部完成**！最后一章：**Chapter 10：生产部署与运维**。
