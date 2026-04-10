# 08-3 并发与吞吐优化

## Ollama 的并发模型

理解 Ollama 如何处理并发请求是优化吞吐的前提。Ollama 的并发架构可以用一句话概括：

**"Ollama 默认串行处理请求——同一时间只处理一个推理任务。`OLLAMA_NUM_PARALLEL` 控制的是内部 batch size（序列打包），而不是真正的并发连接数。"**

```
┌─────────────────────────────────────────────────────────────┐
│              Ollama 并发处理模型                                │
│                                                             │
│  默认行为 (OLLAMA_NUM_PARALLEL=1):                         │
│                                                             │
│  Request A ──→ ┌──────┐                                    │
│                │ 处理A │ ← 正在生成 (占用全部算力)            │
│  Request B ──→ 🔄 等待...                                  │
│  Request C ──→ 🔄 等待...                                  │
│                └──────┘                                    │
│                  ↓ A 完成                                   │
│  Request B ──→ ┌──────┐                                    │
│                │ 处理B │                                    │
│  Request C ──→ 🔄 等待...                                  │
│                └──────┘                                    │
│                                                             │
│  特点: 公平但慢，每个请求的延迟 = 自身处理时间 + 排队等待    │
│                                                             │
│  ──────────────────────────────────────────────────────── │
│                                                             │
│  多连接 + 反向代理方案:                                      │
│                                                             │
│  Client 1 ──→ ┌─────────┐   ┌──────────┐                 │
│  Client 2 ──→ │ Nginx/LB │ → │ Ollama #1 │  (端口11434)   │
│  Client 3 ──→ │ (负载均衡) │   ├──────────┤                 │
│  Client 4 ──→ │           │ → │ Ollama #2 │  (端口11435)  │
│                └─────────┘   └──────────┘                 │
│                                                             │
│  特点: 高吞吐，需要多实例/进程                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 方案一：单实例 + 队列管理

如果你只有一个 Ollama 实例但需要服务多个客户端，合理的做法是在前面加一个请求队列：

```python
#!/usr/bin/env python3
"""带队列管理的 Ollama API 服务"""

import asyncio
import time
import threading
from queue import Queue, Empty
import requests
from typing import Optional


class QueuedOllamaService:
    """带请求队列的 Ollama 服务包装"""
    
    def __init__(self,
                 ollama_url="http://localhost:11434",
                 max_concurrent=1,
                 queue_timeout=60,
                 request_timeout=120):
        
        self.ollama_url = ollama_url
        self.max_concurrent = max_concurrent
        self.queue_timeout = queue_timeout
        self.request_timeout = request_timeout
        
        self.request_queue = Queue()
        self.active_count = 0
        self.lock = threading.Lock()
        self.stats = {"total": 0, "completed": 0, "rejected": 0, "timeout": 0}
        
        # 启动工作线程
        for i in range(max_concurrent):
            t = threading.Thread(target=self._worker, daemon=True, name=f"worker-{i}")
            t.start()
    
    def _worker(self):
        """工作线程：从队列取请求并处理"""
        
        while True:
            try:
                item = self.request_queue.get(timeout=1)
            except Empty:
                continue
            
            request_data, future = item
            start_time = time.time()
            
            try:
                result = self._call_ollama(request_data)
                elapsed = time.time() - start_time
                future.set_result({"result": result, "time": elapsed})
                
                with self.lock:
                    self.stats["completed"] += 1
                    
            except Exception as e:
                future.set_exception(e)
            
            finally:
                self.request_queue.task_done()
                with self.lock:
                    self.active_count -= 1
    
    def _call_ollama(self, data):
        """实际调用 Ollama"""
        resp = requests.post(
            f"{self.ollama_url}/api/chat",
            json=data,
            timeout=self.request_timeout
        )
        resp.raise_for_status()
        return resp.json()
    
    async def chat(self, messages, model="qwen2.5:7b", **options):
        """
        异步接口：提交请求到队列并等待结果
        """
        
        import concurrent.futures
        future = concurrent.futures.Future()
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": options or {}
        }
        
        with self.lock:
            self.stats["total"] += 1
            if self.active_count < self.max_concurrent:
                self.active_count += 1
                self.request_queue.put((payload, future))
            else:
                # 检查队列是否已满（简单实现）
                if self.request_queue.qsize() > 100:
                    self.stats["rejected"] += 1
                    raise Exception("Queue full")
                self.request_queue.put((payload, future))
        
        # 等待结果（带超时）
        try:
            result = future.result(timeout=self.queue_timeout)
            return result["result"]
        except concurrent.futures.TimeoutError:
            self.stats["timeout"] += 1
            raise TimeoutError(f"Request timed out after {self.queue_timeout}s")
    
    def get_stats(self):
        """获取统计信息"""
        with self.lock:
            return {**self.stats, "queue_size": self.request_queue.qsize(),
                    "active": self.active_count}


# 使用示例
if __name__ == "__main__":
    service = QueuedOllamaService(max_concurrent=2)
    
    async def simulate_load():
        """模拟高并发负载"""
        
        import asyncio
        
        questions = [
            "什么是 Python?",
            "Docker 和 VM 的区别?",
            "解释量子计算",
            "如何部署 K8s?",
            "RAG 是什么?",
            "写一个快速排序",
            "微服务的优缺点?",
            "什么是向量数据库?",
        ]
        
        tasks = [service.chat([{"role": "user", "content": q}]) 
                 for q in questions]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                print(f"Q{i}: ❌ {r}")
            else:
                content = r.get("message", {}).get("content", "")[:100]
                print(f"Q{i}: ✅ ({'%.1f'%r.get('time', 0)}s) {content}")
    
        print(f"\n📊 统计: {service.get_stats()}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(simulate_load())
```

## 方案二：多实例部署

当单实例无法满足吞吐需求时，可以在同一台机器上启动多个 Ollama 进程：

```bash
# === 多实例部署脚本 ===

# 实例 1: 主力对话模型 (端口 11434)
OLLAMA_HOST=0.0.0.0:11434 \
OLLAMA_KEEP_ALIVE=-1 \
OLLAMA_NUM_PARALLEL=4 \
ollama serve &

sleep 2

# 实例 2: Embedding 服务 (端口 11435)
OLLAMA_HOST=0.0.0.0:11435 \
OLLAMA_KEEP_ALIVE=-1 \
ollama serve &

sleep 2

# 实例 3: 代码模型 (端口 11436)
OLLAMA_HOST=0.0.0.0:11436 \
OLLAMA_KEEP_ALIVE=10m \
ollama serve &
```

配合负载均衡器使用：

```python
#!/usr/bin/env python3
"""多实例 Ollama 负载均衡客户端"""

import requests
import random
import time
from typing import List


class OllamaCluster:
    """Ollama 多实例集群客户端"""
    
    def __init__(self, instances: List[str]):
        """
        Args:
            instances: 实例 URL 列表
                       ["http://localhost:11434", 
                        "http://localhost:11435"]
        """
        self.instances = instances
        self.instance_health = {url: True for url in instances}
    
    def _get_healthy_instance(self) -> str:
        """获取一个健康的实例（轮询策略）"""
        
        for _ in range(len(self.instances)):
            url = self.instances[0]
            self.instances = self.instances[1:] + [url[:1]]  # 轮询
            
            if self.instance_health.get(url, False):
                return url
        
        raise Exception("No healthy instances available")
    
    def _health_check(self):
        """后台健康检查"""
        for url in self.instances:
            try:
                resp = requests.get(f"{url}/api/tags", timeout=3)
                self.instance_health[url] = (resp.status_code == 200)
            except:
                self.instance_health[url] = False
    
    def chat(self, messages, model="qwen2.5:7b"):
        """发送请求到健康的实例"""
        
        instance = self._get_healthy_instance()
        
        start = time.time()
        resp = requests.post(
            f"{instance}/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": False
            },
            timeout=120
        )
        latency = time.time() - start
        
        data = resp.json()
        data["_instance"] = instance
        data["_latency"] = round(latency, 2)
        
        return data


if __name__ == "__main__":
    cluster = OllamaCluster([
        "http://localhost:11434",
        "http://localhost:11435",
    ])
    
    # 后台健康检查
    import threading
    health_thread = threading.Thread(
        target=lambda: [cluster._health_check() or time.sleep(10) for _ in range(100)],
        daemon=True
    )
    health_thread.start()
    
    # 测试
    for i in range(6):
        result = cluster.chat([{"role": "user", "content": f"问题{i+1}: 你好"}])
        inst = result["_instance"].split(":")[-1]
        print(f"Q{i+1} → Instance :{inst} ({result['_latency']}s)")
```

## 吞吐量参考数据

基于实测和社区数据，不同硬件配置下的典型吞吐量：

| 配置 | 模型 | 延迟 P50 | QPS | tokens/s |
|------|------|---------|-----|----------|
| M2 MacBook 8GB | qwen2.5:7b q4 | ~800ms | 1.2 | ~15 |
| M2 MacBook 16GB | qwen2.5:7b q4 | ~500ms | 2.0 | ~25 |
| M2 Pro 32GB | qwen2.5:14b q4 | ~900ms | 1.5 | ~20 |
| RTX 3090 24GB | llama3.1:70b q4 | ~200ms | 5.0 | ~80 |
| RTX 4090 24GB | qwen2.5:32b q4 | ~350ms | 3.5 | ~55 |
| 2× RTX 4090 48GB | qwen2.5:72b q4 | ~600ms | 2.0 | ~35 |

**关键结论**：
- **单个 Ollama 实例的 QPS 通常在 1-5 之间**（取决于模型大小和硬件）
- 如果需要更高的 QPS，必须用多实例 + 负载均衡
- 对于大多数内部工具场景，1-2 QPS 已经足够
- 对于面向用户的产品，需要至少 5-10 QPS 才能接受

## 连接池与 Keep-Alive

HTTP 连接本身也有开销——每次请求都建立新的 TCP 连接会浪费 10-50ms。使用连接池可以减少这部分开销：

```python
#!/usr/bin/env python3
"""HTTP 连接池 + Keep-Alive 优化"""

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time


class OptimizedOllamaClient:
    """带连接池优化的 Ollama 客户端"""
    
    def __init__(self, base_url="http://localhost:11434"):
        self.base_url = base_url
        
        # 创建 Session（复用 TCP 连接）
        self.session = requests.Session()
        
        # 重试策略：失败自动重试 3 次
        retry_strategy = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[502, 503, 504],
        )
        
        # 连接池配置
        adapter = HTTPAdapter(
            pool_connections=10,       # 连接池大小
            pool_maxsize=10,          # 最大连接数
            max_retries=retry_strategy,
            pool_block=False           # 不等待空闲连接
        )
        
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
    
    def chat(self, messages, model="qwen2.5:7b"):
        """通过连接池发送请求"""
        
        start = time.time()
        resp = self.session.post(
            f"{self.base_url}/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": False
            },
            timeout=60
        )
        
        latency = (time.time() - start) * 1000
        return resp.json(), latency
    
    def close(self):
        """关闭连接池"""
        self.session.close()


# 性能对比测试
def benchmark_connection_pool():
    
    print("\n连接池 vs 无连接池对比:\n")
    
    # 无连接池（每次新建连接）
    times_no_pool = []
    for i in range(20):
        start = time.time()
        requests.post("http://localhost:11434/api/chat", json={
            "model": "qwen2.5:1.5b",
            "messages": [{"role": "user", "content": f"hi{i}"}],
            "options": {"num_predict": 10}
        }, timeout=30)
        times_no_pool.append((time.time() - start) * 1000)
    
    avg_no_pool = sum(times_no_pool) / len(times_no_pool)
    
    # 有连接池
    client = OptimizedOllamaClient()
    times_with_pool = []
    
    for i in range(20):
        _, latency = client.chat([{"role": "user", "content": f"hi{i}"}])
        times_with_pool.append(latency)
    
    avg_with_pool = sum(times_with_pool) / len(times_with_pool)
    client.close()
    
    print(f"无连接池: 平均延迟 {avg_no_pool:.1f}ms (20次)")
    print(f"有连接池: 平均延迟 {avg_with_pool:.1f}ms (20次)")
    print(f"提升: {(1 - avg_with_pool/avg_no_pool)*100:.1f}%")
    

if __name__ == "__main__":
    benchmark_connection_pool()
```

## 本章小结

这一节从系统层面探讨了并发与吞吐优化：

1. **Ollama 默认串行处理请求**——`OLLAMA_NUM_PARALLEL` 控制 batch 大小而非连接数
2. **请求队列模式**在单实例上实现了公平的多客户端服务
3. **多实例部署 + 负载均衡**是突破 QPS 天花板的主要手段
4. **连接池（Session + HTTPAdapter）**可以节省 10-30% 的连接开销
5. **典型生产环境 QPS**: 单实例 1-5，多实例可达 10+
6. **选择建议**: 个人工具不需要优化；团队内网用队列；对外产品用多实例

下一节我们将讨论模型预加载和热管理。
