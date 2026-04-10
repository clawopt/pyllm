# 07-4 生产级集成架构

## 从原型到生产：架构演进

当你把一个基于 Ollama 的 LLM 应用从"本地跑通"推进到"团队使用"再到"面向用户"的过程中，架构需要经历几次重要的升级：

```
┌─────────────────────────────────────────────────────────────┐
│              LLM 应用架构成熟度模型                           │
│                                                             │
│  Level 0: 原型阶段 (个人开发)                               │
│  ┌──────────┐                                              │
│  │ Python   │ → 直接 HTTP 调用 localhost:11434              │
│  │ Script   │    模型在开发者机器上运行                       │
│  └──────────┘                                              │
│                                                             │
│  Level 1: 团队共享 (内网部署)                                │
│  ┌─────┐    ┌──────┐    ┌──────────┐                      │
│  │ 用户 │ → │ Nginx │ → │ Ollama   │                      │
│  └─────┘    │ 反代  │    │ Server   │                      │
│             └──────┘    │ (GPU)    │                      │
│                         └──────────┘                      │
│                                                             │
│  Level 2: 生产级 (高可用)                                   │
│  ┌─────┐  ┌──────┐  ┌─────┬─────┐  ┌──────┐           │
│  │ 用户 │→│ LB  │→│Node1│Node2│→│ 共享  │          │
│  └─────┘  └──────┘  └─────┴─────┘  │ 存储  │          │
│                          ↑         └──────┘           │
│                    ┌────────┐                         │
│                    │ 监控   │                         │
│                    │ 告警   │                         │
│                    └────────┘                         │
│                                                             │
│  Level 3: 企业级 (混合 + 多租户)                             │
│  ┌──────────────────────────────────────────────┐         │
│  │ API Gateway (认证/限流/路由)                  │         │
│  │  ├── /v1/chat → Ollama Cluster (本地)       │         │
│  │  ├── /v1/embed → Embedding Service          │         │
│  │  └── /v1/chat-pro → Cloud Fallback (OpenAI)  │         │
│  ├──────────────────────────────────────────────┤         │
│  │ Cache Layer (Redis: 语义缓存 + 结果缓存)      │         │
│  ├──────────────────────────────────────────────┤         │
│  │ Observability (OTel Tracing + Prometheus)     │         │
│  └──────────────────────────────────────────────┘         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 架构设计：Ollama 作为 LLM Service Layer

### 核心原则

```
生产环境中的 Ollama 应该被定位为 "LLM Service Layer"（LLM 服务层），
而不是一个独立的工具。这意味着：

1. 它不应该直接暴露给最终用户（应该有 API Gateway 层）
2. 它的状态应该是可替换的（可以随时切换到云端 API）
3. 它的调用应该被记录和监控
4. 它的输出应该经过安全过滤
```

### 完整参考架构

```yaml
# docker-compose.production.yml
# 生产级 Ollama 部署模板

version: '3.8'

services:
  # ========== Ollama 主服务 ==========
  ollama-primary:
    image: ollama/ollama:latest
    container_name: ollama-primary
    restart: unless-stopped
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
      - ./models:/models:ro        # 预加载模型目录
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu]
              device_ids: ['0']
    environment:
      - OLLAMA_HOST=0.0.0.0
      - OLLAMA_KEEP_ALIVE=10m
      - OLLAMA_NUM_PARALLEL=4
      - OLLAMA_MAX_LOADED_MODELS=3
      - OLLAMA_REQUEST_TIMEOUT=120
      - OLLAMA_DEBUG=false
    networks:
      - ollama-net
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:11434/api/tags"]
      interval: 30s
      timeout: 5s
      retries: 3

  # ========== Nginx 反向代理 ==========
  nginx:
    image: nginx:alpine
    container_name: ollama-proxy
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/conf.d:/etc/nginx/conf.d:ro
      - ./certs:/etc/nginx/certs:ro
    depends_on:
      - ollama-primary
    networks:
      - ollama-net

  # ========== Redis 缓存层 ==========
  redis:
    image: redis:7-alpine
    container_name: ollama-cache
    restart: unless-stopped
    command: redis-server --maxmemory 512mb --maxmemory-policy allkeys-lru
    volumes:
      - redis_data:/data
    networks:
      - ollama-net

networks:
  ollama-net:
    driver: bridge

volumes:
  ollama_data:
  redis_data:
```

### Nginx 配置（含基本安全和限流）

```nginx
# nginx/conf.d/ollama.conf

upstream ollama_backend {
    server ollama-primary:11434;
}

# 限流配置
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=30r/s;
limit_conn_zone conn_limit_zone:10m;

server {
    listen 80;
    server_name ai.yourcompany.com;

    # 重定向到 HTTPS
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl http2;
    server_name ai.yourcompany.com;

    ssl_certificate     /etc/nginx/certs/fullchain.pem;
    ssl_certificate_key /etc/nginx/certs/privkey.pem;

    # 安全头
    add_header X-Frame-Options DENY always;
    add_header X-Content-Type-Options nosniff always;
    add_header X-XSS-Protection "1; mode=block" always;

    # 只暴露必要的端点
    location /api/chat {
        limit_req zone=api_limit burst=50 nodelay;
        limit_conn conn_limit_zone 10;

        proxy_pass http://ollama_backend/api/chat;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # 超时设置
        proxy_connect_timeout 10s;
        proxy_send_timeout 120s;
        proxy_read_timeout 180s;

        # 请求体大小限制
        client_max_body_size 20m;
    }

    location /api/generate {
        limit_req zone=api_limit burst=30 nodelay;
        proxy_pass http://ollama_backend/api/generate;
        # ... 同上配置
    }

    location /api/embeddings {
        limit_req zone=api_limit burst=100 nodelay;
        proxy_pass http://ollama_backend/api/embeddings;
        # ... 同上配置
    }

    # 禁止访问其他所有路径
    location / {
        return 404;
    }
}
```

## 缓存策略：语义缓存

对于 RAG 场景，相同或高度相似的问题会被反复提问。实现语义缓存可以大幅减少对 Ollama 的调用次数：

```python
#!/usr/bin/env python3
"""语义缓存层：相似问题命中缓存，避免重复调用"""

import hashlib
import json
import time
import numpy as np
import requests
from typing import Optional


class SemanticCache:
    """基于向量相似度的语义缓存"""
    
    def __init__(self, 
                 redis_url="redis://localhost:6379",
                 embedding_model="nomic-embed-text",
                 similarity_threshold=0.92,
                 ttl_seconds=3600):
        
        import redis
        self.redis = redis.from_url(redis_url)
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.ttl = ttl_seconds
    
    def _embed(self, text: str) -> np.ndarray:
        """生成查询向量"""
        resp = requests.post(
            "http://localhost:11434/api/embeddings",
            json={"model": self.embedding_model, "prompt": text},
            timeout=30
        )
        return np.array(resp.json()["embedding"])
    
    def _query_key(self, text: str) -> str:
        """生成缓存查询键（基于文本哈希）"""
        return f"scache:q:{hashlib.md5(text.encode()).hexdigest()}"
    
    def _cache_key(self, text: str) -> str:
        """生成缓存存储键"""
        return f"scache:c:{hashlib.sha256(text.encode()).hexdigest()}"
    
    def get(self, query_text: str) -> Optional[str]:
        """
        尝试从缓存获取答案
        
        Returns:
            缓存的答案文本，如果未命中则返回 None
        """
        
        start = time.time()
        query_vec = self._embed(query_text)
        
        # Step 1: 精确匹配（完全相同的查询）
        exact_key = self._query_key(query_text)
        cached = self.redis.get(exact_key)
        
        if cached:
            data = json.loads(cached)
            print(f"[Cache] 精确命中! ({(time.time()-start)*1000:.0f}ms)")
            return data["answer"]
        
        # Step 2: 语义近似匹配
        # 在实际生产中，这里可以用 FAISS 或 Redis Vector Search
        # 这里简化为检查最近 N 个缓存条目
        
        cache_prefix = "scache:c:"
        cursor = 0
        best_match = None
        best_score = 0
        
        while True:
            cursor, keys = self.redis.scan(cursor, match=f"{cache_prefix}*", count=20)
            
            if not keys:
                break
            
            for key in keys:
                cached_data = self.redis.get(key)
                if not cached_data:
                    continue
                
                data = json.loads(cached_data)
                cached_vec = np.array(data["query_embedding"])
                
                # 余弦相似度
                score = np.dot(query_vec, cached_vec) / (
                    np.linalg.norm(query_vec) * np.linalg.norm(cached_vec)
                )
                
                if score > best_score and score > self.similarity_threshold:
                    best_score = score
                    best_match = data
            
            if cursor == 0:
                break
        
        if best_match:
            elapsed = (time.time() - start) * 1000
            print(f"[Cache] 语义命中! (sim={best_score:.3f}, {elapsed:.0f}ms)")
            # 将精确查询也加入缓存，下次可以精确命中
            self.redis.setex(
                self._query_key(query_text),
                self.ttl,
                json.dumps({"answer": best_match["answer"]})
            )
            return best_match["answer"]
        
        print(f"[Cache] 未命中 ({(time.time()-start)*1000:.0f}ms)")
        return None
    
    def set(self, query_text: str, answer: str):
        """将问答结果存入缓存"""
        
        query_vec = self._embed(query_text)
        
        cache_data = {
            "answer": answer,
            "query_embedding": query_vec.tolist(),
            "timestamp": time.time()
        }
        
        pipe = self.redis.pipeline()
        pipe.setex(
            self._cache_key(query_text),
            self.ttl,
            json.dumps(cache_data)
        )
        pipe.setex(
            self._query_key(query_text),
            self.ttl,
            json.dumps({"answer": answer})
        )
        pipe.execute()


# 使用示例
if __name__ == "__main__":
    cache = SemanticCache(similarity_threshold=0.90)
    
    question = "什么是 Docker 容器？"
    
    # 第一次查询：未命中，调用 Ollama
    result = cache.get(question)
    if result is None:
        print("调用 Ollama 生成回答...")
        resp = requests.post("http://localhost:11434/api/chat", json={
            "model": "qwen2.5:7b",
            "messages": [{"role": "user", "content": question}],
            "stream": False
        })
        answer = resp.json()["message"]["content"]
        cache.set(question, answer)
        result = answer
    
    print(f"\n回答: {result[:200]}...")
    
    # 第二次查询（相同问题）：精确命中
    result2 = cache.get(question)
    print(f"\n第二次查询结果: {result2[:100]}..." if result2 else "miss")
    
    # 第三次查询（相似问题）：语义命中
    similar_q = "Docker 容器是什么意思？"
    result3 = cache.get(similar_q)
    print(f"\n相似查询结果: {result3[:100]}..." if result3 else "miss")
```

## 降级方案：Ollama 不可用时 Fallback 到云端

```python
#!/usr/bin/env python3
"""带降级的 LLM 服务客户端"""

import requests
import time
from typing import Optional


class ResilientLLMClient:
    """带自动降级的 LLM 客户端"""
    
    def __init__(self,
                 primary_url="http://localhost:11434/v1",
                 fallback_url="https://api.openai.com/v1",
                 fallback_api_key="sk-your-openai-key",
                 model="qwen2.5:7b",
                 fallback_model="gpt-4o-mini"):
        
        self.primary_url = primary_url
        self.fallback_url = fallback_url
        self.fallback_api_key = fallback_api_key
        self.model = model
        self.fallback_model = fallback_model
    
    def chat(self, messages: list, **kwargs) -> dict:
        """带降级的对话接口"""
        
        # 尝试主服务 (Ollama)
        try:
            result = self._call_api(self.primary_url, None, 
                                     self.model, messages, **kwargs)
            result["_source"] = "ollama-local"
            return result
        except Exception as e:
            print(f"⚠️ Ollama 不可用: {e}")
            print("🔄 降级到云端 API...")
        
        # 降级到备用服务 (OpenAI)
        try:
            result = self._call_api(self.fallback_url, self.fallback_api_key,
                                     self.fallback_model, messages, **kwargs)
            result["_source"] = "openai-cloud-fallback"
            return result
        except Exception as e2:
            raise RuntimeError(
                f"主服务和备用服务均不可用。主错误: {e}, 备用错误: {e2}"
            )
    
    def _call_api(self, base_url, api_key, model, messages, **kwargs):
        """统一的 API 调用方法"""
        
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 2048),
        }
        
        start = time.time()
        resp = requests.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=kwargs.get("timeout", 60)
        )
        resp.raise_for_status()
        
        data = resp.json()
        data["_latency_ms"] = round((time.time() - start) * 1000)
        
        return data


# 监控与告警
def health_check_with_alert():
    """健康检查 + 告警通知"""
    
    try:
        resp = requests.get(
            "http://localhost:11434/api/tags",
            timeout=5
        )
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            print(f"✅ Ollama 健康 | 已加载模型: {len(models)} 个")
            return True
    except:
        pass
    
    print("❌ Ollama 不响应!")
    # 发送告警（Webhook / 邮件 / Slack）
    # send_alert("Ollama service is down!")
    return False


if __name__ == "__main__":
    client = ResilientLLMClient()
    
    response = client.chat([
        {"role": "user", "content": "解释微服务架构"}
    ])
    
    print(f"\n来源: {response['_source']}")
    print(f"延迟: {response.get('_latency_ms', '?')}ms")
    print(f"回答: {response['choices'][0]['message']['content'][:200]}...")
```

## 日志与追踪

```python
#!/usr/bin/env python3
"""请求链路追踪中间件"""

import time
import uuid
import json
import functools
from typing import Callable, Any


class LLMObservability:
    """LLM 调用可观测性中间件"""
    
    def __init__(self, log_file="./llm_traces.jsonl"):
        self.log_file = log_file
    
    def _log_call(self, request: dict, response: dict, 
                  latency_ms: float, error=None):
        """记录每次 LLM 调用的完整信息"""
        
        entry = {
            "trace_id": str(uuid.uuid4())[:8],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "request": {
                "model": request.get("model"),
                "messages_count": len(request.get("messages", [])),
                "prompt_tokens_estimate": sum(
                    len(m.get("content", "")) // 4 
                    for m in request.get("messages", [])
                ),
            },
            "response": {
                "output_tokens": len(response.get("choices", [{}])[0]
                                      .get("message", {})
                                      .get("content", "")) // 4 if response else 0,
                "source": response.get("_source", "unknown"),
            },
            "performance": {
                "latency_ms": round(latency_ms, 1),
                "tokens_per_second": round(
                    (response.get("choices", [{}])[0]
                     .get("message", {}).get("content", "").count(" ") or 1) 
                    / (latency_ms / 1000), 1
                ) if latency_ms > 0 else 0,
            },
            "error": str(error) if error else None
        }
        
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# 装饰器用法
observability = LLMObservability()

def traced_llm_call(func: Callable) -> Callable:
    """为 LLM 调用添加追踪的装饰器"""
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        try:
            result = func(*args, **kwargs)
            observability._log_call(
                kwargs, result, (time.time() - start) * 1000
            )
            return result
        except Exception as e:
            observability._log_call(
                kwargs, {}, (time.time() - start) * 1000, error=e
            )
            raise
    
    return wrapper


@traced_llm_call
def call_ollama(messages, model="qwen2.5:7b"):
    """被追踪的 Ollama 调用函数"""
    resp = requests.post("http://localhost:11434/v1/chat/completions", json={
        "model": model,
        "messages": messages
    }, timeout=60)
    return resp.json()


if __name__ == "__main__":
    call_ollama([{"role": "user", "content": "Hello"}])
    # 自动生成追踪日志到 llm_traces.jsonl
```

## 本章小结

这一节讨论了将 Ollama 集成提升到生产级别所需的关键架构决策：

1. **四层成熟度模型**：从个人脚本到企业级多租户平台
2. **Ollama 定位为 LLM Service Layer**——不应直接暴露给终端用户
3. **完整的 Docker Compose 生产模板**包含 Ollama + Nginx 反代 + Redis 缓存
4. **Nginx 配置**包含限流、安全头、超时控制和路径过滤
5. **语义缓存**通过向量相似度匹配减少重复调用，命中率可达 40%+
6. **降级方案**在 Ollama 不可用时自动切换到 OpenAI 云端 API
7. **可观测性中间件**记录每次调用的 trace_id、延迟、token 消耗和错误信息

至此，第七章"LangChain/LlamaIndex 集成"全部完成。下一章我们将进入性能优化领域。
