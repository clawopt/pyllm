# 监控与可观测性

## 白板导读

当 Ollama 从开发测试走向生产环境后，**"模型能跑"不再是唯一标准**——你需要回答的问题是：系统现在健康吗？哪个模型响应最慢？GPU 资源是否吃紧？昨天凌晨 3 点为什么延迟飙升？这些问题的答案，全部依赖一套完整的**可观测性（Observability）体系**。

可观测性由三大支柱构成：**Metrics（指标）**、**Logs（日志）**、**Traces（链路追踪）**。Ollama 本身提供了 Prometheus 兼容的 `/metrics` 端点，可以导出推理耗时、Token 生成速度、活跃连接数等核心指标。但生产级监控远不止拉取几个数字那么简单——你还需要结构化日志聚合、智能告警规则、Grafana 可视化大屏，以及分布式链路追踪来串联从 Nginx 反向代理到 Ollama 推理引擎再到向量数据库的完整请求路径。本节将手把手搭建这套体系。

---

## 10.3.1 Ollama 指标体系深度解析

### Ollama 的 /metrics 端点

Ollama 内置了 Prometheus 格式的指标暴露端点，启动服务后直接访问即可看到原始数据：

```bash
# 查看原生指标
curl http://localhost:11434/metrics
```

返回的是标准的 Prometheus 文本格式，每一行以 `# HELP` 开头是指标说明，`# TYPE` 是类型声明，后续是具体数值：

```
# HELP ollama:prompt_eval_count_total Total number of prompt tokens evaluated
# TYPE ollama:prompt_eval_count_total counter
ollama:prompt_eval_count_total{model="qwen2.5:7b"} 15234

# HELP ollama:prompt_eval_duration_seconds Total time spent evaluating prompts
# TYPE ollama:prompt_eval_duration_seconds counter
ollama:prompt_eval_duration_seconds{model="qwen2.5:7b"} 12.567

# HELP ollama:eval_count_total Total number of response tokens generated
# TYPE ollama:eval_count_total counter
ollama:eval_count_total{model="qwen2.5:7b"} 8901

# HELP ollama:eval_duration_seconds Total time spent generating responses
# TYPE ollama:eval_duration_seconds counter
ollama:eval_duration_seconds{model="qwen2.5:7b"} 45.231
```

### 核心指标分类与业务含义

理解每个指标的业务含义是构建有效监控的前提。我们将 Ollama 的指标分为 **五大类别**：

| 类别 | 指标名称 | 类型 | 含义 |
|------|---------|------|------|
| **吞吐量** | `prompt_eval_count_total` | Counter | 累计处理的 Prompt Token 数 |
| **吞吐量** | `eval_count_total` | Counter | 累计生成的 Response Token 数 |
| **延迟** | `prompt_eval_duration_seconds` | Counter | Prompt 处理累计耗时（秒） |
| **延迟** | `eval_duration_seconds` | Counter | Response 生成累计耗时（秒） |
| **速率** | （派生）Tokens/sec | Gauge | `eval_count / eval_duration`，生成速度 |
| **资源** | `ollama:requests_in_progress` | Gauge | 当前正在处理的请求数 |

> **关键洞察**：Ollama 原生只提供 Counter（累加器）类型的原始计数和耗时，**不直接提供 P50/P99 分位数或速率**。这些高级指标需要通过 Prometheus 的查询语言 PromQL 在采集端计算得出。

### 用 Python 解析 /metrics 并计算派生指标

下面是一个完整的 MetricsParser 类，它定期拉取 `/metrics`、解析 Prometheus 格式、计算派生指标（如 Tokens/sec、平均延迟），并支持输出为 JSON 或直接推送到 Pushgateway：

```python
import re
import time
import json
import requests
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional
from datetime import datetime
from collections import defaultdict


@dataclass
class ModelMetrics:
    model: str = ""
    prompt_tokens: int = 0
    eval_tokens: int = 0
    prompt_duration: float = 0.0
    eval_duration: float = 0.0
    tokens_per_second: float = 0.0
    avg_prompt_latency_ms: float = 0.0
    avg_eval_latency_ms: float = 0.0
    timestamp: str = ""


class OllamaMetricsCollector:
    """Ollama Prometheus 指标采集器"""

    METRIC_PATTERN = re.compile(
        r'^ollama:(\w+)(?:\{([^}]*)\})?\s+([\d.eE+-]+)$',
        re.MULTILINE
    )

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip("/")
        self._prev_snapshot: Dict[str, dict] = {}
        self._session = requests.Session()

    def fetch_raw_metrics(self) -> str:
        resp = self._session.get(f"{self.base_url}/metrics", timeout=10)
        resp.raise_for_status()
        return resp.text

    def parse_metrics(self, raw_text: str) -> Dict[str, ModelMetrics]:
        models = defaultdict(lambda: ModelMetrics(timestamp=datetime.now().isoformat()))

        for match in self.METRIC_PATTERN.finditer(raw_text):
            metric_name = match.group(1)
            labels_str = match.group(2) or ""
            value = float(match.group(3))

            model_label = self._extract_label(labels_str, "model") or "_global"
            m = models[model_label]
            m.model = model_label if model_label != "_global" else "all"

            if metric_name == "prompt_eval_count_total":
                m.prompt_tokens = int(value)
            elif metric_name == "prompt_eval_duration_seconds":
                m.prompt_duration = value
            elif metric_name == "eval_count_total":
                m.eval_tokens = int(value)
            elif metric_name == "eval_duration_seconds":
                m.eval_duration = value

        for name, m in models.items():
            if m.eval_duration > 0:
                m.tokens_per_second = round(m.eval_tokens / m.eval_duration, 2)
            if m.prompt_tokens > 0:
                m.avg_prompt_latency_ms = round(
                    (m.prompt_duration / m.prompt_tokens) * 1000, 2
                )
            if m.eval_tokens > 0:
                m.avg_eval_latency_ms = round(
                    (m.eval_duration / m.eval_tokens) * 1000, 2
                )

        return dict(models)

    @staticmethod
    def _extract_label(labels_str: str, key: str) -> Optional[str]:
        for item in labels_str.split(","):
            item = item.strip()
            if item.startswith(f'{key}="'):
                return item[len(f'{key}="'):-1]
        return None

    def collect(self) -> Dict[str, ModelMetrics]:
        raw = self.fetch_raw_metrics()
        return self.parse_metrics(raw)

    def to_json(self) -> str:
        metrics = self.collect()
        return json.dumps(
            [asdict(m) for m in metrics.values()],
            indent=2, ensure_ascii=False
        )


if __name__ == "__main__":
    collector = OllamaMetricsCollector()
    print(collector.to_json())
```

运行效果示例：

```json
[
  {
    "model": "qwen2.5:7b",
    "prompt_tokens": 15234,
    "eval_tokens": 8901,
    "prompt_duration": 12.567,
    "eval_duration": 45.231,
    "tokens_per_second": 196.82,
    "avg_prompt_latency_ms": 0.83,
    "avg_eval_latency_ms": 5.08,
    "timestamp": "2026-01-15T14:30:00"
  }
]
```

### 关键指标的解读阈值

光有数字不够，你需要知道什么值代表"正常"、什么值意味着"告警"：

| 指标 | 🟢 健康 | 🟡 注意 | 🔴 告警 | 说明 |
|------|--------|--------|--------|------|
| **Tokens/sec** | >50 tok/s | 20-50 tok/s | <20 tok/s | 受量化级别和硬件影响 |
| **Prompt 延迟 (TTFT)** | <500ms | 500ms-2s | >2s | 首个 Token 时间 |
| **Eval 延迟/Token** | <10ms/token | 10-30ms/token | >30ms/token | 每个 Token 生成间隔 |
| **并发请求数** | <4 | 4-8 | >8 | Ollama 默认串行处理 |
| **VRAM 使用率** | <80% | 80%-95% | >95% | 接近爆显存时性能骤降 |

> **面试考点**：为什么 Ollama 的 Token 生成速度在不同请求间波动很大？因为 **KV Cache 的命中情况不同**——如果新请求的 System Prompt 与上一个相同，KV Cache 可以复用，Prompt Eval 阶段几乎瞬间完成；如果完全不同，则需要重新计算所有 Prompt Token 的 KV 状态，延迟显著增加。这就是为什么在生产环境中尽量固定 SYSTEM 模板能带来稳定性能的原因。

---

## 10.3.2 Prometheus + Grafana 全栈搭建

### 架构总览

```
┌─────────────┐     scrape     ┌──────────────┐    query    ┌─────────────┐
│   Ollama    │ ────────────▶  │  Prometheus   │ ──────────▶ │   Grafana   │
│ :11434/metrics│  (每15秒)     │  :9090       │             │  :3000      │
└─────────────┘               └──────────────┘             └─────────────┘
                                     │
                              ┌──────▼──────┐
                              │ Alertmanager│
                              │ :9093       │
                              └─────────────┘
```

### Prometheus 配置文件

创建 `prometheus.yml` 配置 Ollama 作为采集目标：

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    monitor: 'ollama-monitor'

scrape_configs:
  - job_name: 'ollama'
    static_configs:
      - targets: ['host.docker.internal:11434']
    metrics_path: '/metrics'
    scrape_timeout: 10s

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['host.docker.internal:9100']
```

### Docker Compose 一键启动完整监控栈

```yaml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:v2.51.0
    container_name: ollama-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.retention.time=30d'
      - '--web.enable-lifecycle'

  grafana:
    image: grafana/grafana:11.0.0
    container_name: ollama-grafana
    ports:
      - "3000:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin123
      GF_USERS_ALLOW_SIGN_UP: "false"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/provisioning:/etc/grafana/provisioning
    depends_on:
      - prometheus

  alertmanager:
    image: prom/alertmanager:v0.27.0
    container_name: ollama-alertmanager
    ports:
      - "9093:9093"
    volumes:
      - ./alertmanager.yml:/etc/alertmanager/alertmanager.yml

volumes:
  prometheus_data:
  grafana_data:
```

### Grafana Dashboard 核心面板配置

在 Grafana 中创建 Dashboard 时，以下是**最关键的 5 个面板**及其 PromQL 查询语句：

#### 面板 1：Token 生成速度趋势

```promql
rate(ollama:eval_count_total[5m]) /
rate(ollama:eval_duration_seconds[5m])
```

这个查询利用了 Prometheus 的 `rate()` 函数计算每秒增量，再相除得到实时的 Tokens/sec 速率。建议设置面板类型为 **Time Series**，单位设为 **tok/s**。

#### 面板 2：请求延迟 P50/P95/P99 分布

由于 Ollama 不原生提供直方图指标，我们需要通过 `eval_duration_seconds / eval_count_total` 来估算单次请求的平均延迟。对于更精确的分位数，推荐使用下面的**自定义导出方式**（见下一节）。

简化版平均延迟查询：

```promql
rate(ollama:eval_duration_seconds[5m]) /
rate(ollama:eval_count_total[5m])
```

#### 面板 3：各模型调用次数排行

```promql
topk(10, sum by (model) (increase(ollama:eval_count_total[1h])))
```

使用 `topk()` 取 Top 10 最常被调用的模型，配合 **Stat** 或 **Bar Gauge** 面板类型展示。

#### 面板 4：系统资源使用率（需 Node Exporter）

```promql
# GPU 内存使用率（需要 nvidia-smi exporter）
nvidia_gpu_memory_used_bytes / nvidia_gpu_memory_total_bytes * 100

# CPU 使用率
100 - (avg by(instance) (irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)

# 磁盘使用率
node_filesystem_avail_bytes{mountpoint="/"} /
node_filesystem_size_bytes{mountpoint="/"} * 100
```

#### 面板 5：错误率监控

```promql
# Ollama HTTP 错误率（通过反向代理日志或自定义指标）
sum(rate(http_requests_total{status=~"5.."}[5m])) /
sum(rate(http_requests_total[5m])) * 100
```

### Grafana 自动化 Provisioning 配置

为了避免每次重启容器都要手动导入 Dashboard，使用 Grafana 的 provisioning 功能自动加载：

```yaml
# grafana/provisioning/datasources/datasource.yml
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
```

```json
// grafana/provisioning/dashboards/dashboard.json
{
  "annotations": {
    "list": []
  },
  "editable": true,
  "gnetId": null,
  "graphTooltip": 0,
  "id": null,
  "panels": [
    {
      "title": "Token Generation Speed",
      "type": "timeseries",
      "gridPos": { "h": 8, "w": 12, "x": 0, "y": 0 },
      "targets": [
        {
          "expr": "rate(ollama:eval_count_total[5m]) / rate(ollama:eval_duration_seconds[5m])",
          "legendFormat": "{{model}}",
          "refId": "A"
        }
      ],
      "fieldConfig": {
        "defaults": {
          "unit": "short",
          "color": { "mode": "palette-classic" }
        }
      }
    },
    {
      "title": "Active Requests",
      "type": "gauge",
      "gridPos": { "h": 6, "w": 6, "x": 12, "y": 0 },
      "targets": [
        {
          "expr": "ollama:requests_in_progress",
          "refId": "A"
        }
      ],
      "fieldConfig": {
        "defaults": {
          "max": 10,
          "min": 0,
          "thresholds": {
            "mode": "absolute",
            "steps": [
              { "value": 0, "color": "green" },
              { "value": 5, "color": "yellow" },
              { "value": 8, "color": "red" }
            ]
          }
        }
      }
    }
  ],
  "schemaVersion": 38,
  "tags": ["ollama"],
  "templating": { "list": [] },
  "time": { "from": "now-1h", "to": "now" },
  "timepicker": {},
  "timezone": "browser",
  "title": "Ollama Operations Dashboard",
  "uid": "ollama-main",
  "version": 1
}
```

---

## 10.3.3 自定义指标导出器

Ollama 原生指标缺少一些生产环境关键信息：**P50/P99 延迟分位数**、**按 API 端点分类的统计**、**错误码分布**等。我们可以编写一个轻量级的 Sidecar 导出器来补充这些能力。

### OllamaCustomExporter 设计

核心思路是在 Ollama 和客户端之间插入一个代理层，记录每个请求的详细计时信息，然后以 Prometheus 格式暴露 `/custom-metrics` 端点：

```python
"""
Ollama Custom Metrics Exporter
补充 Ollama 原生指标缺失的分位数、错误率、按端点分类等高级指标。
同时作为反向代理转发所有请求到 Ollama。
"""

import time
import json
import threading
from flask import Flask, request, Response, jsonify
from prometheus_client import (
    Counter, Histogram, Gauge, generate_latest,
    CONTENT_TYPE_LATEST
)
import requests as http_requests


app = Flask(__name__)

REQUEST_COUNT = Counter(
    'ollama_custom_requests_total',
    'Total requests proxied to Ollama',
    ['method', 'endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'ollama_custom_request_duration_seconds',
    'Request latency in seconds',
    ['method', 'endpoint'],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0]
)

TOKEN_THROUGHPUT = Histogram(
    'ollama_custom_tokens_per_second',
    'Token generation throughput distribution',
    ['model'],
    buckets=[5, 10, 20, 30, 50, 80, 100, 150, 200, 300]
)

ACTIVE_REQUESTS = Gauge(
    'ollama_custom_active_requests',
    'Currently active requests being processed'
)

MODEL_LOAD_DURATION = Histogram(
    'ollama_custom_model_load_seconds',
    'Model loading duration',
    ['model'],
    buckets=[1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0]
)

ERROR_COUNTER = Counter(
    'ollama_custom_errors_total',
    'Total errors by error type',
    ['error_type', 'model']
)


OLLAMA_BASE = "http://localhost:11434"


@app.route('/metrics')
def metrics():
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


def _proxy_request(path: str, method: str = 'POST') -> Response:
    start_time = time.time()
    ACTIVE_REQUESTS.inc()

    try:
        target_url = f"{OLLAMA_BASE}{path}"

        if method == 'POST':
            body = request.get_json(silent=True) or {}
            resp = http_requests.post(target_url, json=body, stream=True, timeout=300)
        else:
            resp = http_requests.get(target_url, timeout=60)

        status = str(resp.status_code)
        endpoint = path.replace('/', '_').strip('_') or 'root'

        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status).inc()

        latency = time.time() - start_time
        REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(latency)

        if status.startswith('5'):
            ERROR_COUNTER.labels(error_type='http_5xx', model='_unknown').inc()

        excluded_headers = {'content-encoding', 'transfer-encoding', 'connection'}
        headers = [(k, v) for k, v in resp.raw.headers.items()
                   if k.lower() not in excluded_headers]

        return Response(resp.content, status=resp.status_code, headers=headers)

    except http_requests.exceptions.Timeout:
        REQUEST_COUNT.labels(method=method, endpoint=path, status='504').inc()
        ERROR_COUNTER.labels(error_type='timeout', model='_unknown').inc()
        return jsonify({"error": "Ollama timeout"}), 504
    except Exception as e:
        REQUEST_COUNT.labels(method=method, endpoint=path, status='500').inc()
        ERROR_COUNTER.labels(error_type=str(type(e).__name__), model='_unknown').inc()
        return jsonify({"error": str(e)}), 500
    finally:
        ACTIVE_REQUESTS.dec()


@app.route('/api/chat', methods=['POST'])
def proxy_chat():
    return _proxy_request('/api/chat')


@app.route('/api/generate', methods=['POST'])
def proxy_generate():
    return _proxy_request('/api/generate')


@app.route('/api/embeddings', methods=['POST'])
def proxy_embeddings():
    return _proxy_request('/api/embeddings')


@app.route('/api/show', methods=['POST'])
def proxy_show():
    return _proxy_request('/api/show')


@app.route('/health')
def health():
    try:
        resp = http_requests.get(f"{OLLAMA_BASE}/", timeout=5)
        return jsonify({
            "status": "healthy",
            "ollama_status": "up" if resp.status_code == 200 else "degraded"
        }), 200
    except Exception:
        return jsonify({"status": "unhealthy", "ollama_status": "down"}), 503


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9091, threaded=True)
```

现在你可以通过 `http://localhost:9091/metrics` 访问更丰富的指标，PromQL 查询也变得更强大了：

```promql
# P99 延迟（Histogram 自带分位数）
histogram_quantile(0.99, rate(ollama_custom_request_duration_seconds_bucket[5m]))

# 按 API 端点的 QPS
sum by (endpoint) (rate(ollama_custom_requests_total[5m]))

# 错误率
sum by (error_type) (rate(ollama_custom_errors_total[1h]))
```

---

## 10.3.4 结构化日志与聚合

### 为什么需要结构化日志？

Ollama 默认将日志输出到 stdout/stderr，格式是非结构的纯文本。在生产环境中这有三个致命问题：
1. **无法机器解析**——grep 能搜但无法做统计分析
2. **无上下文关联**——不知道哪条日志对应哪个请求
3. **无法集中管理**——多节点部署时日志散落在各台机器

解决方案是引入**结构化 JSON 日志** + **日志聚合系统**。

### 日志架构选型

| 方案 | 适用场景 | 复杂度 | 特点 |
|------|---------|--------|------|
| **Loki + Grafana** | 中小规模首选 ⭐ | 低 | 与 Grafana 天然集成，标签索引，成本低 |
| **ELK Stack** | 企业级复杂分析 | 高 | 功能最强，但资源消耗大（ES 内存密集） |
| **Meilisearch** | 轻量全文检索 | 低 | 部署简单，适合日志搜索场景 |
| **Vector/Datadog** | 云原生环境 | 中 | 统一日志收集管道 |

对于大多数 Ollama 部署场景，**Loki + Grafana** 是性价比最高的选择——你已经有了 Grafana，加一个 Loki 容器就能统一查看指标和日志。

### OllamaLogShipper：结构化日志收集器

```python
"""
Ollama 结构化日志 Shipper
拦截并格式化 Ollama 的访问日志，输出为 JSON 格式，
可选推送到 Loki / 文件 / stdout。
"""

import sys
import json
import logging
import threading
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
from collections import deque
import hashlib


@dataclass
class LogEntry:
    timestamp: str
    level: str
    source: str
    model: str = ""
    api_endpoint: str = ""
    request_id: str = ""
    client_ip: str = ""
    prompt_length: int = 0
    response_length: int = 0
    latency_ms: float = 0.0
    tokens_generated: int = 0
    error: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class OllamaLogShipper:
    """结构化日志收集与分发器"""

    def __init__(
        self,
        output: str = "stdout",
        loki_url: Optional[str] = None,
        log_file: Optional[str] = None,
        batch_size: int = 50,
        flush_interval: float = 5.0
    ):
        self.output = output
        self.loki_url = loki_url
        self.log_file = log_file
        self.batch_size = batch_size
        self.flush_interval = flush_interval

        self._buffer: deque = deque(maxlen=10000)
        self._lock = threading.Lock()
        self._running = False
        self._flush_thread: Optional[threading.Thread] = None

        if output == "file" and log_file:
            self._fh = open(log_file, 'a', encoding='utf-8')
        else:
            self._fh = None

    def emit(self, entry: LogEntry):
        with self._lock:
            self._buffer.append(entry)

    def emit_from_dict(self, data: dict):
        entry = LogEntry(
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
            level=data.get("level", "INFO"),
            source=data.get("source", "ollama"),
            model=data.get("model", ""),
            api_endpoint=data.get("api_endpoint", ""),
            request_id=data.get("request_id", self._gen_request_id()),
            client_ip=data.get("client_ip", ""),
            prompt_length=data.get("prompt_length", 0),
            response_length=data.get("response_length", 0),
            latency_ms=data.get("latency_ms", 0.0),
            tokens_generated=data.get("tokens_generated", 0),
            error=data.get("error", ""),
            extra=data.get("extra", {})
        )
        self.emit(entry)

    @staticmethod
    def _gen_request_id() -> str:
        return hashlib.md5(
            f"{threading.get_ident()}-{time.time_ns()}".encode()
        ).hexdigest()[:12]

    def start(self):
        self._running = True
        self._flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._flush_thread.start()

    def stop(self):
        self._running = False
        if self._flush_thread:
            self._flush_thread.join(timeout=10)
        self._flush_now()
        if self._fh:
            self._fh.close()

    def _flush_loop(self):
        while self._running:
            time.sleep(self.flush_interval)
            self._flush_now()

    def _flush_now(self):
        with self._lock:
            if not self._buffer:
                return
            batch = list(self._buffer)
            self._buffer.clear()

        lines = [json.dumps(e.to_dict(), ensure_ascii=False) for e in batch]

        if self.output == "stdout":
            for line in lines:
                print(line)
        elif self.output == "file" and self._fh:
            for line in lines:
                self._fh.write(line + "\n")
            self._fh.flush()
        elif self.output == "loki" and self.loki_url:
            self._push_to_loki(lines)

    def _push_to_loki(self, lines: List[str]):
        payload = {
            "streams": [{
                "stream": {"job": "ollama", "source": "api"},
                "values": [
                    [str(int(time.time() * 1e9)), line]
                    for line in lines
                ]
            }]
        }
        try:
            resp = requests.post(
                f"{self.loki_url}/loki/api/v1/push",
                json=payload,
                timeout=5
            )
            resp.raise_for_status()
        except Exception:
            pass


shipper = OllamaLogShipper(output="stdout")
shipper.start()


def log_api_call(
    model: str,
    endpoint: str,
    prompt_text: str,
    response_text: str,
    latency_ms: float,
    error: str = "",
    client_ip: str = ""
):
    shipper.emit(LogEntry(
        timestamp=datetime.now(timezone.utc).isoformat(),
        level="ERROR" if error else "INFO",
        source="ollama-api",
        model=model,
        api_endpoint=endpoint,
        client_ip=client_ip,
        prompt_length=len(prompt_text.encode('utf-8')),
        response_length=len(response_text.encode('utf-8')),
        latency_ms=latency_ms,
        tokens_generated=len(response_text.split()) // 2,
        error=error
    ))


if __name__ == "__main__":
    log_api_call(
        model="qwen2.5:7b",
        endpoint="/api/chat",
        prompt_text="解释量子计算的基本原理",
        response_text="量子计算是一种利用量子力学原理...",
        latency_ms=1250.5,
        client_ip="192.168.1.100"
    )
    shipper.stop()
```

输出的 JSON 日志示例：

```json
{
  "timestamp": "2026-01-15T14:32:15.123Z",
  "level": "INFO",
  "source": "ollama-api",
  "model": "qwen2.5:7b",
  "api_endpoint": "/api/chat",
  "request_id": "a3f8c2d1e4b5",
  "client_ip": "192.168.1.100",
  "prompt_length": 28,
  "response_length": 512,
  "latency_ms": 1250.5,
  "tokens_generated": 86,
  "error": "",
  "extra": {}
}
```

### Loki + Grafana LogQL 查询示例

当日志进入 Loki 后，可以用 **LogQL**（Loki Query Language）进行强大的日志分析：

```logql
# 查看最近1小时所有错误日志
{job="ollama"} |= "ERROR" | json

# 找出延迟超过5秒的慢请求
{job="ollama"} | json | latency_ms > 5000

# 按模型统计请求量
sum by (model) (count_over_time({job="ollama"} [1h]))

# 某个 IP 的异常行为检测
{job="ollama"} | json | client_ip="192.168.1.100"

# Token 生成效率最低的请求 TOP 10
{job="ollama"} | json | line_format "{{.latency_ms}},{{.tokens_generated}}"
```

---

## 10.3.5 告警规则与通知

### Alertmanager 配置

告警是监控系统的"发声器官"——没有告警，再漂亮的 Dashboard 也只是摆设。Alertmanager 负责：

1. **去重**——同一问题不重复通知
2. **分组**——相关问题合并为一条消息
3. **静默**——维护期间抑制告警
4. **路由**——根据严重程度发送到不同渠道

```yaml
# alertmanager.yml
global:
  resolve_timeout: 5m

route:
  group_by: ['alertname', 'severity']
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 12h
  receiver: 'default-receiver'
  routes:
    - match:
        severity: critical
      receiver: 'pagerduty-webhook'
      continue: true
    - match:
        severity: warning
      receiver: 'slack-notifications'

receivers:
  - name: 'default-receiver'
    webhook_configs:
      - url: 'https://hooks.slack.com/services/TXXX/BXXX/XXXX'
        send_resolved: true

  - name: 'pagerduty-webhook'
    webhook_configs:
      - url: 'https://events.pagerduty.com/v2/enqueue'
        send_resolved: true

  - name: 'slack-notifications'
    slack_configs:
      - channel: '#ollama-alerts'
        send_resolved: true
        title: '[{{ .Status }}] {{ .Labels.alertname }}'
        text: '{{ range .Alerts }}*{{ .Labels.severity }}*: {{ .Annotations.summary }}{{ end }}'

inhibit_rules:
  - source_match:
      severity: 'critical'
    target_match:
      severity: 'warning'
    equal: ['alertname', 'instance']
```

### 核心 Prometheus 告警规则

创建 `alert_rules.yml` 定义 Ollama 相关的所有告警规则：

```yaml
groups:
  - name: ollama_alerts
    rules:

      # ========== 服务可用性 ==========
      - alert: OllamaDown
        expr: up{job="ollama"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Ollama 服务不可达"
          description: "实例 {{ $labels.instance }} 已宕机超过 1 分钟"

      # ========== 性能退化 ==========
      - alert: OllamaHighLatency
        expr: |
          histogram_quantile(0.95,
            rate(ollama_custom_request_duration_seconds_bucket[5m])
          ) > 30
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Ollama P95 延迟过高"
          description: "当前 P95 延迟为 {{ $value }}s，阈值 30s"

      - alert: OllamaLowThroughput
        expr: |
          rate(ollama:eval_count_total[5m]) /
          rate(ollama:eval_duration_seconds[5m]) < 10
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Token 生成速度过低"
          description: "当前速度 {{ $value }} tok/s，可能存在资源争抢"

      # ========== 资源预警 ==========
      - alert: OllamaHighConcurrency
        expr: ollama:requests_in_progress > 6
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "Ollama 并发过高"
          description: "当前并发 {{ $value }}，接近串行处理上限"

      - alert: GPUMemoryExhaustion
        expr: |
          nvidia_gpu_memory_used_bytes /
          nvidia_gpu_memory_total_bytes > 0.92
        for: 3m
        labels:
          severity: critical
        annotations:
          summary: "GPU 显存即将耗尽"
          description: "显存使用率 {{ $humanize $value }}%，可能导致 OOM"

      - alert: GPUTemperatureHigh
        expr: nvidia_gpu_temperature_celsius > 85
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "GPU 温度过高"
          description: "GPU 温度 {{ $value }}°C，有降频风险"

      # ========== 存储空间 ==========
      - alert: DiskSpaceLow
        expr: |
          node_filesystem_avail_bytes{mountpoint="/"} /
          node_filesystem_size_bytes{mountpoint="/"} < 0.1
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "磁盘空间不足"
          description: "根分区剩余空间不足 10%"

      # ========== 错误率 ==========
      - alert: OllamaHighErrorRate
        expr: |
          sum(rate(ollama_custom_errors_total[5m])) /
          sum(rate(ollama_custom_requests_total[5m])) > 0.05
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Ollama 错误率过高"
          description: "错误率 {{ $humanize $value }}，阈值 5%"
```

### 告警分级策略

| 级别 | 条件 | 通知渠道 | 响应时间要求 |
|------|------|---------|------------|
| **P0-Critical** | 服务宕机、显存耗尽、磁盘满 | PagerDuty + 电话 + Slack | 5 分钟内响应 |
| **P1-Warning** | 延迟超标、温度过高、错误率升高 | Slack + 邮件 | 30 分钟内处理 |
| **P2-Info** | 吞吐量下降、模型切换频繁 | 仅 Slack 记录 | 工作时间内关注 |

---

## 10.3.6 分布式链路追踪

### 为什么需要链路追踪？

当一个用户请求经过 **Nginx → Ollama Exporter → Ollama → 向量数据库** 这条链路时，单纯看各组件的日志很难定位瓶颈到底在哪一环。**分布式链路追踪（Distributed Tracing）** 为每个请求分配唯一的 Trace ID，沿途经过的每个组件都记录 Span（时间片段），最终形成一张完整的请求时间线图。

### OpenTelemetry 集成方案

OpenTelemetry（简称 OTel）是 CNCF 主导的可观测性标准，支持自动插桩和手动埋点。我们为 Ollama 请求链路添加 OTel 追踪：

```python
"""
Ollama OpenTelemetry Tracing 集成
为每个 LLM 请求创建完整的 Trace，覆盖：
  Client → Proxy → Ollama Inference → (Optional) Vector DB
"""

import time
import json
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from contextlib import contextmanager
from typing import Optional, Generator, Any, Dict


resource = Resource.create({
    SERVICE_NAME: "ollama-service"
})

provider = TracerProvider(resource=resource)
processor = BatchSpanProcessor(OTLPSpanExporter(endpoint="localhost:4317"))
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)
tracer = trace.get_tracer(__name__)


@contextmanager
def trace_ollama_request(
    model: str,
    endpoint: str,
    metadata: Optional[Dict[str, Any]] = None
) -> Generator[None, None, None]:
    """
    上下文管理器：自动包裹一次 Ollama 调用的完整追踪。
    
    比如：
        with trace_ollama_request("qwen2.5:7b", "/api/chat"):
            result = ollama.chat(...)
    """
    span_name = f"{endpoint} [{model}]"
    with tracer.start_as_current_span(span_name) as span:
        span.set_attribute("llm.model", model)
        span.set_attribute("llm.endpoint", endpoint)
        span.set_attribute("service.name", "ollama")

        if metadata:
            for k, v in metadata.items():
                span.set_attribute(f"llm.{k}", str(v))

        start = time.time()
        try:
            yield
        except Exception as e:
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise
        finally:
            span.set_attribute("llm.duration_ms", (time.time() - start) * 1000)


class TracedOllamaClient:
    """带 OTel 追踪的 Ollama 客户端封装"""

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.tracer = tracer

    def chat(self, model: str, messages: list, **kwargs):
        import requests

        metadata = {
            "num_messages": len(messages),
            "total_chars": sum(len(m.get("content", "")) for m in messages),
            "stream": kwargs.get("stream", False)
        }

        with trace_ollama_request(model, "/api/chat", metadata):
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json={"model": model, "messages": messages, **kwargs},
                timeout=kwargs.get("timeout", 120),
                stream=metadata["stream"]
            )

            if metadata["stream"]:
                full_content = ""
                token_count = 0
                for line in resp.iter_lines():
                    if line:
                        chunk = json.loads(line)
                        content = chunk.get("message", {}).get("content", "")
                        if content:
                            full_content += content
                            token_count += 1

                span = trace.get_current_span()
                span.set_attribute("llm.response_tokens", token_count)
                span.set_attribute("llm.response_length", len(full_content))

                return {"content": full_content, "token_count": token_count}
            else:
                data = resp.json()
                message = data.get("message", {})
                span = trace.get_current_span()
                span.set_attribute("llm.finish_reason", message.get("reason", ""))
                return data

    def embed(self, model: str, texts: list):
        import requests

        with trace_ollama_request(model, "/api/embeddings", {"num_texts": len(texts)}):
            resp = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": model, "prompt": texts[0]},
                timeout=60
            )
            return resp.json()


if __name__ == "__main__":
    client = TracedOllamaClient()

    result = client.chat(
        model="qwen2.5:7b",
        messages=[
            {"role": "system", "content": "你是技术助手"},
            {"role": "user", "content": "用Python写一个快速排序"}
        ]
    )
    print(f"生成内容长度: {len(result['content'])}")
```

### Jaeger UI 查看 Trace

启动 Jaeger UI 来可视化追踪数据：

```bash
docker run -d --name jaeger \
  -p 16686:16686 \
  -p 4317:4317 \
  jaegertracing/all-in-one:1.57
```

打开 `http://localhost:16686` 即可看到类似这样的 Trace 视图：

```
Trace: a3f8c2d1e4b5 (Duration: 1.25s)
├── POST /api/chat [qwen2.5:7b] ────────── 1.25s
│   ├── llm.model = qwen2.5:7b
│   ├── llm.num_messages = 2
│   ├── llm.total_chars = 48
│   ├── llm.response_tokens = 156
│   └── llm.response_length = 2048
```

### Trace 数据的价值

| 场景 | 如何用 Trace 定位 |
|------|------------------|
| **某个请求特别慢** | 查看 Trace 总时长，找到耗时最长的 Span |
| **模型 A 比 B 慢** | 对比同问题的两个 Trace 的 Duration |
| **Prompt 过长导致超时** | 看 `total_chars` 属性与延迟的相关性 |
| **间歇性失败** | 搜索 Status=Error 的 Trace，找共性 |

---

## 10.3.7 健康检查与自愈机制

### 多层次健康检查

生产环境的健康检查不能只看"进程是否存活"，而要从多个维度验证服务真正可用：

```python
"""
Ollama Health Check & Self-Healing Framework
多层次健康检查 + 自动恢复策略
"""

import subprocess
import time
import socket
import psutil
import requests
from dataclasses import dataclass
from enum import Enum
from typing import List, Callable, Optional
from datetime import datetime


class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class CheckResult:
    name: str
    status: HealthStatus
    message: str
    latency_ms: float = 0.0
    details: dict = None


class OllamaHealthChecker:
    """多层次 Ollama 健康检查器"""

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.checks: List[Callable] = [
            self._check_process_alive,
            self._check_port_open,
            self._check_http_response,
            self._check_model_list,
            self._check_inference_capability,
            self._check_resource_usage,
        ]

    def run_all(self) -> List[CheckResult]:
        results = []
        for check_fn in self.checks:
            try:
                result = check_fn()
                results.append(result)
            except Exception as e:
                results.append(CheckResult(
                    name=check_fn.__name__,
                    status=HealthStatus.UNHEALTHY,
                    message=f"检查异常: {e}"
                ))
        return results

    def get_overall_status(self, results: List[CheckResult]) -> HealthStatus:
        if any(r.status == HealthStatus.UNHEALTHY for r in results):
            return HealthStatus.UNHEALTHY
        if any(r.status == HealthStatus.DEGRADED for r in results):
            return HealthStatus.DEGRADED
        return HealthStatus.HEALTHY

    def _check_process_alive(self) -> CheckResult:
        start = time.time()
        found = any(
            'ollama' in proc.name().lower()
            for proc in psutil.process_iter(['name'])
        )
        return CheckResult(
            name="process_alive",
            status=HealthStatus.HEALTHY if found else HealthStatus.UNHEALTHY,
            message="Ollama 进程运行中" if found else "未找到 Ollama 进程",
            latency_ms=(time.time() - start) * 1000
        )

    def _check_port_open(self) -> CheckResult:
        start = time.time()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex(('localhost', 11434))
        sock.close()
        ok = result == 0
        return CheckResult(
            name="port_open",
            status=HealthStatus.HEALTHY if ok else HealthStatus.UNHEALTHY,
            message="端口 11434 可达" if ok else "端口 11434 无法连接",
            latency_ms=(time.time() - start) * 1000
        )

    def _check_http_response(self) -> CheckResult:
        start = time.time()
        try:
            resp = requests.get(f"{self.base_url}/", timeout=5)
            latency = (time.time() - start) * 1000
            if resp.status_code == 200:
                return CheckResult(
                    name="http_response",
                    status=HealthStatus.HEALTHY,
                    message=f"HTTP 200 OK ({latency:.0f}ms)",
                    latency_ms=latency
                )
            else:
                return CheckResult(
                    name="http_response",
                    status=HealthStatus.DEGRADED,
                    message=f"HTTP {resp.status_code}",
                    latency_ms=latency
                )
        except Exception as e:
            return CheckResult(
                name="http_response",
                status=HealthStatus.UNHEALTHY,
                message=f"HTTP 不可达: {e}",
                latency_ms=(time.time() - start) * 1000
            )

    def _check_model_list(self) -> CheckResult:
        start = time.time()
        try:
            resp = requests.post(f"{self.base_url}/api/tags", timeout=10)
            data = resp.json()
            models = data.get("models", [])
            latency = (time.time() - start) * 1000
            return CheckResult(
                name="model_list",
                status=HealthStatus.HEALTHY if len(models) > 0 else HealthStatus.DEGRADED,
                message=f"已加载 {len(models)} 个模型",
                latency_ms=latency,
                details={"models": [m['name'] for m in models]}
            )
        except Exception as e:
            return CheckResult(
                name="model_list",
                status=HealthStatus.UNHEALTHY,
                message=f"模型列表获取失败: {e}"
            )

    def _check_inference_capability(self) -> CheckResult:
        start = time.time()
        try:
            resp = requests.post(f"{self.base_url}/api/generate", json={
                "model": "qwen2.5:7b",
                "prompt": "Hi",
                "options": {"num_predict": 5}
            }, timeout=30)
            latency = (time.time() - start) * 1000
            if resp.status_code == 200:
                return CheckResult(
                    name="inference_test",
                    status=HealthStatus.HEALTHY,
                    message=f"推理正常 ({latency:.0f}ms)",
                    latency_ms=latency
                )
            else:
                return CheckResult(
                    name="inference_test",
                    status=HealthStatus.UNHEALTHY,
                    message=f"推理失败: HTTP {resp.status_code}",
                    latency_ms=latency
                )
        except Exception as e:
            return CheckResult(
                name="inference_test",
                status=HealthStatus.UNHEALTHY,
                message=f"推理异常: {e}",
                latency_ms=(time.time() - start) * 1000
            )

    def _check_resource_usage(self) -> CheckResult:
        mem = psutil.virtual_memory()
        gpu_mem = self._get_gpu_memory()
        checks = []

        if mem.percent > 90:
            checks.append(f"内存使用 {mem.percent}%")
        if gpu_mem and gpu_mem > 95:
            checks.append(f"显存使用 {gpu_mem}%")

        if checks:
            return CheckResult(
                name="resource_usage",
                status=HealthStatus.DEGRADED,
                message="; ".join(checks),
                details={"memory_percent": mem.percent, "gpu_memory_percent": gpu_mem}
            )
        return CheckResult(
            name="resource_usage",
            status=HealthStatus.HEALTHY,
            message=f"内存 {mem.percent}%, 显存 {gpu_mem or 'N/A'}%",
            details={"memory_percent": mem.percent, "gpu_memory_percent": gpu_mem}
        )

    @staticmethod
    def _get_gpu_memory() -> Optional[float]:
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used,memory.total',
                 '--format=csv,nounits,noheader'],
                capture_output=True, text=True, timeout=5
            )
            used, total = map(float, result.stdout.strip().split(', '))
            return (used / total) * 100
        except Exception:
            return None


class OllamaSelfHealer:
    """Ollama 自愈管理器"""

    def __init__(self, checker: OllamaHealthChecker):
        self.checker = checker
        self.recovery_actions = [
            ("restart_service", self._restart_ollama),
            ("clear_cache", self._clear_gpu_cache),
            ("notify_ops", self._notify_operations),
        ]

    def check_and_heal(self) -> dict:
        results = self.checker.run_all()
        overall = self.checker.get_overall_status(results)

        report = {
            "timestamp": datetime.now().isoformat(),
            "status": overall.value,
            "checks": [{"name": r.name, "status": r.status.value,
                        "message": r.message} for r in results],
            "actions_taken": []
        }

        if overall == HealthStatus.UNHEALTHY:
            for action_name, action_fn in self.recovery_actions:
                try:
                    action_result = action_fn(results)
                    report["actions_taken"].append({
                        "action": action_name,
                        "result": action_result
                    })
                    time.sleep(5)

                    recheck = self.checker.run_all()
                    if self.checker.get_overall_status(recheck) != HealthStatus.UNHEALTHY:
                        break
                except Exception as e:
                    report["actions_taken"].append({
                        "action": action_name,
                        "result": f"失败: {e}"
                    })

        return report

    @staticmethod
    def _restart_ollama(results: list) -> str:
        subprocess.run(["brew", "services", "restart", "ollama"], timeout=30)
        time.sleep(10)
        return "已执行 Ollama 重启"

    @staticmethod
    def _clear_gpu_cache(results: list) -> str:
        requests.post("http://localhost:114/api/gc", timeout=10)
        return "已触发 GPU 缓存清理"

    @staticmethod
    def _notify_operations(results: list) -> str:
        failed = [r.name for r in results if r.status == HealthStatus.UNHEALTHY]
        msg = f"[ALERT] Ollama 自愈失败，仍不健康的检查项: {', '.join(failed)}"
        print(msg)
        return f"已通知运维团队: {msg}"


if __name__ == "__main__":
    checker = OllamaHealthChecker()
    healer = OllamaSelfHealer(checker)

    report = healer.check_and_heal()
    print(json.dumps(report, indent=2, ensure_ascii=False))
```

输出报告示例：

```json
{
  "timestamp": "2026-01-15T15:00:00",
  "status": "healthy",
  "checks": [
    {"name": "process_alive", "status": "healthy", "message": "Ollama 进程运行中"},
    {"name": "port_open", "status": "healthy", "message": "端口 11434 可达"},
    {"name": "http_response", "status": "healthy", "message": "HTTP 200 OK (12ms)"},
    {"name": "model_list", "status": "healthy", "message": "已加载 5 个模型"},
    {"name": "inference_test", "status": "healthy", "message": "推理正常 (856ms)"},
    {"name": "resource_usage", "status": "healthy", "message": "内存 72%, 显存 78%"}
  ],
  "actions_taken": []
}
```

---

## 10.3.8 完整监控栈一键部署脚本

最后，把 Prometheus + Grafana + Loki + Alertmanager + Exporter 整合成一个完整的 `docker-compose.monitoring.yml`：

```yaml
version: '3.8'

services:
  # ===== 指标采集 =====
  prometheus:
    image: prom/prometheus:v2.51.0
    container_name: ollama-prom
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - ./alert_rules.yml:/etc/prometheus/alert_rules.yml
      - prom_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.retention.time=30d'
      - '--web.enable-lifecycle'

  # ===== 自定义导出器 =====
  ollama-exporter:
    build:
      context: .
      dockerfile: Dockerfile.exporter
    container_name: ollama-exporter
    ports:
      - "9091:9091"
    depends_on:
      - ollama
    restart: unless-stopped

  # ===== 可视化 =====
  grafana:
    image: grafana/grafana:11.0.0
    container_name: ollama-grafana
    ports:
      - "3000:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: ${GRAFANA_PASSWORD:-admin_change_me}
      GF_USERS_ALLOW_SIGN_UP: "false"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/provisioning:/etc/grafana/provisioning
      - ./grafana/dashboards:/var/lib/grafana/dashboards
    depends_on:
      - prometheus

  # ===== 日志聚合 =====
  loki:
    image: grafana/loki:3.0.0
    container_name: ollama-loki
    ports:
      - "3100:3100"
    volumes:
      - ./loki-config.yml:/etc/loki/local-config.yaml
      - loki_data:/loki
    command: -config.file=/etc/loki/local-config.yaml

  # ===== 日志收集 =====
  promtail:
    image: grafana/promtail:3.0.0
    container_name: ollama-promtail
    volumes:
      - ./promtail-config.yml:/etc/promtail/config.yml
      - /var/log:/var/log:ro
    depends_on:
      - loki

  # ===== 告警 =====
  alertmanager:
    image: prom/alertmanager:v0.27.0
    container_name: ollama-alertmanager
    ports:
      - "9093:9093"
    volumes:
      - ./alertmanager.yml:/etc/alertmanager/alertmanager.yml
    depends_on:
      - prometheus

volumes:
  prom_data:
  grafana_data:
  loki_data:
```

启动命令：

```bash
docker compose -f docker-compose.monitoring.yml up -d
```

启动后的访问入口：

| 组件 | 地址 | 用途 |
|------|------|------|
| **Grafana** | `http://localhost:3000` | 可视化大盘 (admin/admin_change_me) |
| **Prometheus** | `http://localhost:9090` | 指标查询与告警状态 |
| **Alertmanager** | `http://localhost:9093` | 告警管理与静默 |
| **Loki** | `http://localhost:3100` | 日志查询 API |
| **Jaeger** | `http://localhost:16686` | 链路追踪视图 |

---

## 要点回顾

| 维度 | 关键要点 |
|------|---------|
| **指标体系** | Ollama 原生提供 Counter 类型指标（token count/duration），需通过 PromQL 计算 Tokens/sec、P50/P99 等派生指标 |
| **Prometheus** | 每 15 秒 scrape `/metrics`，配合 Histogram 导出器获得分位数能力 |
| **Grafana** | 5 个核心面板：Token 速率趋势、延迟分布、模型调用量排行、资源利用率、错误率 |
| **结构化日志** | JSON 格式 + Loki 聚合 + LogQL 查询，替代不可解析的纯文本日志 |
| **告警规则** | 三级分层：Critical（宕机/显存满）→ Warning（延迟高/温度高）→ Info（吞吐降） |
| **链路追踪** | OpenTelemetry 为每个请求创建 Trace ID，Jaeger UI 可视化完整调用链路 |
| **健康检查** | 六维检查：进程存活 → 端口可达 → HTTP 响应 → 模型列表 → 推理验证 → 资源水位 |
| **自愈机制** | 检测到 UNHEALTHY 后依次尝试：重启服务 → 清理缓存 → 通知运维 |

> **一句话总结**：生产级可观测性不是"装个 Grafana 就完事"，而是 **Metrics（Prometheus 采集）→ Logs（Loki 聚合）→ Traces（OTel 追踪）→ Alerts（Alertmanager 通知）→ Dashboards（Grafana 展示）→ Self-Healing（自动恢复）** 形成的闭环体系。只有这套闭环跑通了，你才敢放心地说"这个 Ollama 服务已经 ready for production"。
