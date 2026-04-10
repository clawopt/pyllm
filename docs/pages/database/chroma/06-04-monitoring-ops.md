# 6.4 监控与运维

> **看不见的系统就是不可靠的系统——监控是生产环境的第一道防线**

---

## 这一节在讲什么？

当 Chroma 在生产环境中运行时，你需要知道它是否健康、性能是否正常、数据量是否在预期范围内。没有监控的系统就像蒙着眼睛开车——出了问题你都不知道。这一节我们要讲 Chroma 的健康检查、数据统计、Prometheus 集成、备份策略和常见故障排查，确保你的 Chroma 服务在生产环境中可观测、可诊断、可恢复。

---

## 健康检查

### 基础心跳检测

Chroma Server 提供了 heartbeat API，用于检测服务是否存活：

```python
import chromadb

client = chromadb.HttpClient(host="localhost", port=8000)

# 心跳检测
heartbeat = client.heartbeat()
print(f"Chroma Server 心跳: {heartbeat}")
# 返回一个时间戳，表示服务器当前时间
```

在 Docker Compose 中，可以用 heartbeat 做健康检查：

```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/heartbeat"]
  interval: 30s
  timeout: 10s
  retries: 3
```

### 完整的健康检查函数

```python
import chromadb
import time

def health_check(client: chromadb.HttpClient) -> dict:
    """
    Chroma 健康检查

    返回:
        dict 包含状态、延迟、Collection 统计等信息
    """
    status = {
        "healthy": True,
        "timestamp": time.time(),
        "issues": []
    }

    # 1. 心跳检测
    try:
        start = time.time()
        heartbeat = client.heartbeat()
        latency = (time.time() - start) * 1000
        status["heartbeat_latency_ms"] = round(latency, 2)
        if latency > 1000:
            status["issues"].append(f"心跳延迟过高: {latency:.0f}ms")
    except Exception as e:
        status["healthy"] = False
        status["issues"].append(f"心跳失败: {e}")
        return status

    # 2. Collection 统计
    try:
        collections = client.list_collections()
        status["total_collections"] = len(collections)
        status["collections"] = []
        for col in collections:
            count = col.count()
            col_info = {
                "name": col.name,
                "count": count,
                "metadata": col.metadata
            }
            status["collections"].append(col_info)
            if count > 1000000:
                status["issues"].append(f"Collection '{col.name}' 数据量过大: {count}")
    except Exception as e:
        status["issues"].append(f"Collection 统计失败: {e}")

    return status


# 使用
client = chromadb.HttpClient(host="localhost", port=8000)
health = health_check(client)
print(f"健康状态: {'✅ 正常' if health['healthy'] else '❌ 异常'}")
if health['issues']:
    for issue in health['issues']:
        print(f"  ⚠️ {issue}")
```

---

## Prometheus 集成

Chroma Server 内置了 Prometheus metrics endpoint，默认在 `/api/v1/metrics` 路径下暴露指标。你可以用 Prometheus 采集这些指标，配合 Grafana 做可视化监控：

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'chroma'
    static_configs:
      - targets: ['chroma:8000']
    metrics_path: '/api/v1/metrics'
    scrape_interval: 15s
```

### 关键监控指标

```python
# 自定义监控指标（如果 Chroma 内置指标不够用）
import time
from collections import defaultdict

class ChromaMonitor:
    """Chroma 自定义监控"""

    def __init__(self):
        self.metrics = {
            "query_count": 0,
            "query_latency_sum": 0.0,
            "query_latency_max": 0.0,
            "add_count": 0,
            "add_latency_sum": 0.0,
            "error_count": 0,
            "error_types": defaultdict(int),
        }

    def record_query(self, latency_ms: float):
        """记录查询指标"""
        self.metrics["query_count"] += 1
        self.metrics["query_latency_sum"] += latency_ms
        self.metrics["query_latency_max"] = max(
            self.metrics["query_latency_max"], latency_ms
        )

    def record_add(self, latency_ms: float, count: int = 1):
        """记录添加指标"""
        self.metrics["add_count"] += count
        self.metrics["add_latency_sum"] += latency_ms

    def record_error(self, error_type: str):
        """记录错误"""
        self.metrics["error_count"] += 1
        self.metrics["error_types"][error_type] += 1

    def get_summary(self) -> dict:
        """获取监控摘要"""
        avg_query_latency = (
            self.metrics["query_latency_sum"] / self.metrics["query_count"]
            if self.metrics["query_count"] > 0 else 0
        )
        return {
            "total_queries": self.metrics["query_count"],
            "avg_query_latency_ms": round(avg_query_latency, 2),
            "max_query_latency_ms": round(self.metrics["query_latency_max"], 2),
            "total_adds": self.metrics["add_count"],
            "total_errors": self.metrics["error_count"],
            "error_breakdown": dict(self.metrics["error_types"]),
        }


# 在应用中集成
monitor = ChromaMonitor()

def monitored_query(collection, query_text, **kwargs):
    """带监控的查询"""
    start = time.time()
    try:
        results = collection.query(query_texts=[query_text], **kwargs)
        latency = (time.time() - start) * 1000
        monitor.record_query(latency)
        return results
    except Exception as e:
        monitor.record_error(type(e).__name__)
        raise
```

---

## 告警规则

基于 Prometheus 指标，你可以设置告警规则：

```yaml
# alert_rules.yml
groups:
  - name: chroma_alerts
    rules:
      - alert: ChromaHighLatency
        expr: chroma_query_latency_p99 > 500
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Chroma 查询延迟过高"
          description: "P99 延迟超过 500ms，当前值: {{ $value }}ms"

      - alert: ChromaHighErrorRate
        expr: rate(chroma_errors_total[5m]) > 0.1
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Chroma 错误率过高"
          description: "5分钟内错误率超过 10%"

      - alert: ChromaDataGrowth
        expr: chroma_collection_count > 500000
        for: 1h
        labels:
          severity: warning
        annotations:
          summary: "Chroma 数据量增长过快"
          description: "Collection 文档数超过 50 万，建议评估容量规划"
```

---

## 常见故障排查

### 故障 1：chroma.sqlite3-wal 文件过大

**症状**：磁盘空间告警，`.sqlite3-wal` 文件占用了大量空间。

**原因**：长时间运行后 WAL 日志没有触发 checkpoint，导致日志文件持续增长。

**解决**：

```bash
# 手动触发 checkpoint
sqlite3 ./chroma_data/chroma.sqlite3 "PRAGMA wal_checkpoint(TRUNCATE);"

# 或者重启 Chroma Server（重启时会自动 checkpoint）
docker-compose restart chroma
```

### 故障 2：连接超时

**症状**：HttpClient 连接 Chroma Server 时超时。

**排查步骤**：

```python
# 1. 检查 Server 是否在运行
import requests
try:
    resp = requests.get("http://localhost:8000/api/v1/heartbeat", timeout=5)
    print(f"Server 状态: {resp.status_code}")
except requests.exceptions.ConnectionError:
    print("❌ 无法连接到 Chroma Server")

# 2. 检查端口是否被占用
# lsof -i :8000

# 3. 检查防火墙规则
# iptables -L -n | grep 8000
```

### 故障 3：查询结果为空

**症状**：query 返回空结果，但数据明明已经 add 了。

**排查清单**：

```python
def debug_empty_results(collection, query_text):
    """排查查询结果为空的问题"""
    # 1. 检查 Collection 是否有数据
    count = collection.count()
    print(f"1. Collection 文档数: {count}")
    if count == 0:
        print("   ❌ Collection 为空！请检查 add 操作是否成功")
        return

    # 2. 检查 distance metric 是否一致
    print(f"2. Collection metadata: {collection.metadata}")
    # 确认 hnsw:space 与你预期的一致

    # 3. 不带 where 条件查询
    results = collection.query(
        query_texts=[query_text],
        n_results=min(5, count),
        include=["documents", "distances", "metadatas"]
    )
    print(f"3. 无过滤查询结果数: {len(results['ids'][0])}")
    if results['ids'][0]:
        for i in range(len(results['ids'][0])):
            print(f"   [{results['distances'][0][i]:.4f}] {results['documents'][0][i][:60]}...")

    # 4. 检查 embedding 维度是否匹配
    sample = collection.get(limit=1, include=["embeddings"])
    if sample['embeddings'] and sample['embeddings'][0]:
        dim = len(sample['embeddings'][0][0])
        print(f"4. 向量维度: {dim}")
    else:
        print("4. ⚠️ 无法获取 embedding，可能 EF 配置有误")
```

### 故障 4：内存使用持续增长

**症状**：Chroma 进程的内存使用量持续增长，最终被 OOM Killer 杀掉。

**原因**：HNSW 索引在内存中维护，数据量增长时内存使用自然增长。如果数据量超过物理内存，操作系统会使用 swap，导致性能急剧下降。

**解决**：

1. 监控内存使用，设置告警阈值
2. 定期清理过期数据，控制 Collection 大小
3. 如果数据量超过单机内存，考虑分片到多个 Collection 或迁移到 Milvus/Qdrant

---

## 本章小结

监控与运维是 Chroma 在生产环境中稳定运行的保障。核心要点回顾：第一，使用 heartbeat API 做基础健康检查，配合 Docker healthcheck 实现自动重启；第二，Chroma Server 内置 Prometheus metrics endpoint，可以与 Grafana 集成做可视化监控；第三，关键监控指标包括查询延迟、错误率、数据量增长趋势；第四，WAL 文件过大、连接超时、查询结果为空、内存增长是四个最常见的故障，各有对应的排查方法；第五，定期备份数据、设置告警规则、控制数据量增长是运维的基本功。

下一章我们将进入进阶主题——Chroma 与其他向量数据库的选型对比、与 LLM Framework 的集成、以及 Chroma 的局限性和替代方案。
