# 6.3 多实例与并发访问

> **从单机到服务——让 Chroma 支撑多进程、多服务的并发访问**

---

## 这一节在讲什么？

当你的 RAG 系统从单进程脚本升级为多服务架构时，并发访问就成了必须解决的问题。比如你的 API 服务有 4 个 worker 进程，它们都需要访问同一个 Chroma 数据库——这时候嵌入式模式就不行了，因为 SQLite 不支持多进程并发写。这一节我们要讲清楚 Chroma 的并发模型、Server 模式的部署方式、以及如何通过连接池和负载均衡支撑更高的并发量。

---

## Chroma 的线程安全性

### 嵌入式模式：线程安全但进程不安全

Chroma 的 Client 对象是**线程安全**的——你可以在同一个进程内用多线程共享同一个 Client 实例，不需要加锁：

```python
import chromadb
from concurrent.futures import ThreadPoolExecutor

client = chromadb.Client(settings=chromadb.Settings(
    is_persistent=True,
    persist_directory="./thread_safe_demo"
))

collection = client.get_or_create_collection(name="concurrent_test")

# 预先添加一些数据
collection.add(
    documents=[f"文档 {i}" for i in range(1000)],
    ids=[f"d{i}" for i in range(1000)]
)

def query_task(query_id):
    """多线程查询任务"""
    results = collection.query(
        query_texts=[f"查询 {query_id}"],
        n_results=5
    )
    return query_id, len(results['ids'][0])

# 多线程并发查询——安全！
with ThreadPoolExecutor(max_workers=8) as executor:
    futures = [executor.submit(query_task, i) for i in range(50)]
    for f in futures:
        qid, count = f.result()
        # print(f"查询 {qid}: 返回 {count} 条结果")

print("✅ 多线程查询完成，无错误")
```

但是，**多进程共享同一个 persist_directory 是不安全的**——SQLite 的 WAL 模式虽然支持多进程并发读，但写操作必须串行。如果两个进程同时写入，可能导致数据库锁冲突甚至数据损坏：

```python
# ❌ 不安全：多进程共享同一个 persist_directory
# 进程 A
client_a = chromadb.Client(settings=chromadb.Settings(persist_directory="./shared_db"))
collection_a = client_a.get_or_create_collection(name="shared")
collection_a.add(documents=["进程A的数据"], ids=["a_1"])  # 可能与进程B冲突

# 进程 B（同时运行）
client_b = chromadb.Client(settings=chromadb.Settings(persist_directory="./shared_db"))
collection_b = client_b.get_or_create_collection(name="shared")
collection_b.add(documents=["进程B的数据"], ids=["b_1"])  # 可能与进程A冲突
```

### Server 模式：支持多客户端并发

Server 模式是解决多进程并发访问的标准方案。Chroma Server 内部使用异步 I/O 处理并发请求，多个客户端可以通过 HTTP API 同时读写，Server 内部负责串行化写操作：

```python
# Server 端（单独启动）
# chroma run --path ./server_data --host 0.0.0.0 --port 8000

# 客户端 A（可以是不同的进程/机器）
import chromadb
client_a = chromadb.HttpClient(host="localhost", port=8000)
col_a = client_a.get_or_create_collection(name="shared_kb")
col_a.add(documents=["客户端A的数据"], ids=["a_1"])  # ✅ 安全

# 客户端 B（同时运行，不同的进程/机器）
client_b = chromadb.HttpClient(host="localhost", port=8000)
col_b = client_b.get_collection(name="shared_kb")
col_b.add(documents=["客户端B的数据"], ids=["b_1"])  # ✅ 安全
```

---

## Docker Compose 部署方案

生产环境中，Chroma Server 通常与你的应用服务一起部署在 Docker Compose 中：

```yaml
# docker-compose.yml
version: "3.8"

services:
  chroma:
    image: chromadb/chroma:latest
    ports:
      - "8000:8000"
    volumes:
      - chroma_data:/chroma/chroma
    environment:
      - CHROMA_SERVER_HOST=0.0.0.0
      - CHROMA_SERVER_PORT=8000
      - ANONYMIZED_TELEMETRY=FALSE
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/heartbeat"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 4G

  api:
    build: .
    ports:
      - "8080:8080"
    environment:
      - CHROMA_HOST=chroma
      - CHROMA_PORT=8000
    depends_on:
      chroma:
        condition: service_healthy
    restart: unless-stopped

volumes:
  chroma_data:
```

应用服务中连接 Chroma Server 的代码：

```python
import chromadb
import os

def get_chroma_client():
    """获取 Chroma Server 客户端"""
    host = os.getenv("CHROMA_HOST", "localhost")
    port = int(os.getenv("CHROMA_PORT", "8000"))

    return chromadb.HttpClient(
        host=host,
        port=port
    )

# 在 FastAPI 应用中使用
from fastapi import FastAPI

app = FastAPI()
chroma_client = get_chroma_client()

@app.get("/search")
def search(q: str, n: int = 5):
    collection = chroma_client.get_or_create_collection(name="kb")
    results = collection.query(query_texts=[q], n_results=n)
    return {
        "query": q,
        "results": results['documents'][0]
    }
```

---

## 连接池与并发优化

当你的 API 服务有大量并发请求时，每次都创建新的 HttpClient 会有性能开销。建议使用连接池复用 Client 实例：

```python
import chromadb
from threading import Lock

class ChromaClientPool:
    """Chroma 客户端连接池"""

    _instance = None
    _lock = Lock()

    def __init__(self, host: str = "localhost", port: int = 8000, pool_size: int = 4):
        self.host = host
        self.port = port
        self.pool_size = pool_size
        self._clients = []
        self._current = 0
        self._client_lock = Lock()

        for _ in range(pool_size):
            self._clients.append(
                chromadb.HttpClient(host=host, port=port)
            )

    @classmethod
    def get_instance(cls, host="localhost", port=8000, pool_size=4):
        """获取单例连接池"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(host, port, pool_size)
        return cls._instance

    def get_client(self) -> chromadb.HttpClient:
        """轮询获取客户端"""
        with self._client_lock:
            client = self._clients[self._current]
            self._current = (self._current + 1) % self.pool_size
            return client


# 使用
pool = ChromaClientPool.get_instance(host="chroma", port=8000, pool_size=4)

def handle_request(query: str):
    client = pool.get_client()
    collection = client.get_or_create_collection(name="kb")
    results = collection.query(query_texts=[query], n_results=5)
    return results
```

---

## 常见误区

### 误区 1：嵌入式模式完全不能多线程使用

错误。嵌入式模式的 Client 是线程安全的，可以在多线程中共享。只是不能在多进程中共享同一个 persist_directory。

### 误区 2：Server 模式下写操作可以真正并行

Chroma Server 内部仍然会串行化写操作（因为底层 SQLite 是单写者模型）。但读操作可以并行——多个客户端可以同时查询，互不阻塞。

### 误区 3：HttpClient 每次请求都要创建

HttpClient 底层使用 HTTP 连接，创建开销不大但也不是零。建议复用 Client 实例，或者使用连接池。

---

## 本章小结

并发访问是 Chroma 从单机走向服务化的关键问题。核心要点回顾：第一，嵌入式模式的 Client 是线程安全的，但多进程不能共享同一个 persist_directory；第二，Server 模式通过 HTTP API 支持多客户端并发，写操作内部串行化、读操作可以并行；第三，Docker Compose 是部署 Chroma Server + 应用服务的推荐方式，配合 healthcheck 确保服务可用；第四，高并发场景建议使用连接池复用 HttpClient 实例，减少连接创建开销。

下一节我们将讲监控与运维——如何确保 Chroma 在生产环境中健康运行。
