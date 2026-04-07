# 5. 监控与分析

## 让系统"可观测"：从黑盒到透明

在上一节中，我们把工单系统部署到了生产环境，做了容器化、性能优化、安全加固等一系列工程化改造。但部署上线只是开始——系统真正运行起来之后，我们怎么知道它是否健康？用户提交的工单有没有被正确处理？LLM 的分类准确率如何？哪些环节是性能瓶颈？如果出了问题，我们能在多长时间内发现并定位原因？这些问题的答案，都依赖于一个完善的**监控与分析体系**。

很多人对监控的理解还停留在"看看服务器 CPU 是不是满了"这个层面，但对于一个基于 LLM 的智能系统来说，传统的服务器监控远远不够。我们需要关注的是业务层面的指标——工单处理成功率、平均处理时长、自动解决率、人工介入率——以及 AI 特有的指标——模型调用延迟、Token 消耗量、分类准确率、知识库命中率。这一节我们就来构建这样一套全方位的监控体系。

### 监控体系的三个层次：基础设施 → 应用 → 业务

一个成熟的监控系统通常分为三个层次，每一层关注的对象和目标都不相同：

**第一层：基础设施监控。** 这是最基础的层面，关注的是服务器的资源使用情况——CPU 利用率、内存占用、磁盘空间、网络 I/O 等。如果这一层出了问题（比如磁盘写满了导致数据库崩溃），上层的一切都会受影响。常用的工具包括 Prometheus + Node Exporter（收集服务器指标）、cAdvisor（容器资源监控）等。

**第二层：应用性能监控（APM）。** 这一层关注的是应用本身的运行状况——HTTP 请求的响应时间和错误率、数据库查询耗时、外部 API 调用的成功率、并发连接数等。对于我们的 FastAPI 服务来说，需要采集每个端点的 QPS、P50/P95/P99 延迟分布、4xx/5xx 错误率等指标。

**第三层：业务监控。** 这是最高层也是最关键的层面，关注的是系统是否在达成业务目标——今天处理了多少工单？自动解决了多少？平均等待时间是多少？用户满意度如何？LLM 分类准确率怎么样？这些指标直接反映了系统的价值，也是运营团队最关心的数据。

下面我们从底层往上，逐层构建这套监控体系。

### 第一层：Prometheus + Grafana 基础设施监控

Prometheus 是目前云原生领域最流行的监控系统，它采用拉取模式（pull model）定期从目标服务抓取指标数据，然后存储在自带的时序数据库中，支持强大的 PromQL 查询语言。Grafana 则是一个可视化平台，可以连接 Prometheus 作为数据源，绘制出漂亮的仪表盘。

首先，我们需要让我们的 FastAPI 应用暴露 Prometheus 格式的指标。这可以通过 `prometheus-fastapi-instrumentator` 库来实现：

```python
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
import time
import random

app = FastAPI()

Instrumentator().instrument(app).expose(app)

metrics_counter = {
    "tickets_created": 0,
    "tickets_resolved": 0,
    "tickets_escalated": 0,
    "llm_calls_total": 0,
    "llm_call_errors": 0,
}

@app.middleware("http")
async def add_metrics_middleware(request: Request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    endpoint = request.url.path
    
    print(f"[METRICS] endpoint={endpoint} status={response.status_code} duration={process_time:.3f}s")
    
    return response

@app.post("/api/tickets")
async def create_ticket(request: CreateTicketRequest):
    metrics_counter["tickets_created"] += 1
    return await do_create_ticket(request)

@app.get("/api/metrics/custom")
async def get_custom_metrics():
    return {
        "tickets_created_today": metrics_counter["tickets_created"],
        "tickets_resolved_today": metrics_counter["tickets_resolved"],
        "llm_calls_total": metrics_counter["llm_calls_total"],
        "llm_error_rate": (
            metrics_counter["llm_call_errors"] / max(metrics_counter["llm_calls_total"], 1) * 100
        )
    }
```

`Instrumentator` 会自动为 FastAPI 应用添加一组标准的 HTTP 指标（请求数、响应时间、活跃请求数等），并通过 `/metrics` 端点暴露给 Prometheus 抓取。除此之外，我们还通过中间件记录了每个请求的处理时间，并在关键的业务操作处手动埋点（比如创建工单时递增 `tickets_created` 计数器）。

接下来需要在 `docker-compose.yml` 中添加 Prometheus 和 Grafana 服务：

```yaml
services:
  prometheus:
    image: prom/prometheus:v2.48.0
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana:10.2.0
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin123
      GF_USERS_ALLOW_SIGN_UP: "false"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/provisioning:/etc/grafana/provisioning:ro
    ports:
      - "3001:3000"
    depends_on:
      - prometheus

volumes:
  prometheus_data:
  grafana_data:
```

Prometheus 的配置文件 `prometheus.yml` 定义了要抓取的目标：

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'ticket-system-api'
    static_configs:
      - targets: ['api:8000']
    metrics_path: '/metrics'

  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
```

配置很简单：每 15 秒从 `api:8000/metrics` 抓取一次指标数据。启动整个 stack 后，访问 `http://localhost:9090` 就能看到 Prometheus 的 Web UI，输入 PromQL 查询就能看到各种指标的实时值。

比如，查询过去 5 分钟内 HTTP 请求的错误率：

```graphql
sum(rate(http_requests_total{status=~"5.."}[5m])) 
/ sum(rate(http_requests_total[5m])) * 100
```

或者查询 P95 响应时间：

```graphql
histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le))
```

不过，原始的 Prometheus UI 虽然功能强大但不够直观，所以我们用 Grafana 来搭建可视化的仪表盘。Grafana 支持预配置的数据源和仪表盘（provisioning），我们可以提前把常用的面板定义好：

```json
{
  "dashboard": {
    "title": "工单系统总览",
    "panels": [
      {
        "title": "QPS",
        "type": "stat",
        "targets": [{
          "expr": "sum(rate(http_requests_total[5m]))",
          "legendFormat": "{{method}} {{path}}"
        }]
      },
      {
        "title": "平均响应时间",
        "type": "gauge",
        "targets": [{
          "expr": "sum(rate(http_request_duration_seconds_sum[5m])) / sum(rate(http_request_duration_seconds_count[5m]))",
          "legendFormat": "avg latency"
        }],
        "fieldConfig": {
          "defaults": {
            "thresholds": {
              "steps": [
                {"value": 0, "color": "green"},
                {"value": 0.5, "color": "yellow"},
                {"value": 2, "color": "red"}
              ]
            }
          }
        }
      },
      {
        "title": "错误率趋势",
        "type": "timeseries",
        "targets": [{
          "expr": "sum(rate(http_requests_total{status=~\"4..|5..\"}[5m])) / sum(rate(http_requests_total[5m])) * 100",
          "legendFormat": "error rate %"
        }]
      },
      {
        "title": "今日工单统计",
        "type": "stat",
        "gridPos": {"h": 4, "w": 12},
        "panels": [
          {"title": "新建", "targets": [{"expr": "tickets_created_total"}]},
          {"title": "已解决", "targets": [{"expr": "tickets_resolved_total"}]},
          {"title": "已升级", "targets": [{"expr": "tickets_escalated_total"}]}
        ]
      }
    ],
    "refresh": "30s"
  }
}
```

这个仪表盘包含了四个核心面板：实时 QPS 统计、平均响应时间仪表盘（绿色<0.5秒、黄色0.5-2秒、红色>2秒）、错误率趋势折线图、以及今日工单数量概览。当所有服务都启动后，访问 `http://localhost:3001`（注意端口映射到了 3001 避免与本地其他服务冲突），用 `admin/admin123` 登录后就能看到这个精心设计的监控面板了。

### 第二层：APM — 应用性能深度洞察

基础设施监控告诉我们"机器是不是健康"，但它无法回答"应用内部哪里慢"的问题。比如，我们知道某个 API 端点的 P99 延迟达到了 3 秒，但这 3 秒到底花在了哪里？是数据库查询慢？还是 LLM 调用慢？还是某个复杂的计算逻辑慢？这就需要 **APM（Application Performance Monitoring）** 工具来深入到应用的内部。

对于 Python/FastAPI 应用来说，有几个选择：商业方案如 Datadog、New Relic 功能强大但价格不菲；开源方案如 **Sentry**（专注于错误追踪）、**Jaeger/OpenTelemetry**（分布式追踪）、**Py-Spy**（性能剖析器）。这里我们重点介绍 OpenTelemetry 方案，因为它已经成为云原生可观测性的事实标准。

OpenTelemetry（简称 OTel）提供了一套统一的 API 来收集三种类型的遥测数据：**Metrics**（指标）、**Logs**（日志）、**Traces**（链路追踪）。对于我们的工单系统，最重要的是 Traces——它可以完整地记录一个请求从进入系统到返回响应所经过的所有环节及其耗时。

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

trace.set_tracer_provider(TracerProvider())
tracer = trace.get tracer(__name__)

processor = BatchSpanProcessor(ConsoleSpanExporter())
trace.get_tracer_provider().add_span_processor(processor)

FastAPIInstrumentor().instrument_app(app)
RequestsInstrumentor().instrument()
SQLAlchemyInstrumentor().instrument(engine=sync_engine)

@tracer.start_as_current_span("classify_intent_node")
def classify_intent_with_tracing(state: TicketState) -> TicketState:
    span = trace.get_current_span()
    span.set_attribute("ticket.id", state.get("ticket_id"))
    span.set_attribute("message.length", len(state["messages"][-1].content))
    
    result = classify_intent(state)
    
    span.set_attribute("classification.category", result["category"])
    span.set_attribute("classification.urgency", result["urgency"])
    
    return result
```

这段代码做了几件事：初始化 OpenTelemetry 的 Trace Provider 和 Span Processor；自动对 FastAPI、HTTP 请求、SQLAlchemy 进行埋点（这样所有的 HTTP 请求、外部 API 调用、数据库操作都会被自动追踪）；然后在关键的节点函数 `classify_intent_with_tracing` 上手动添加了一个自定义 span，记录了工单 ID、消息长度、分类结果等属性信息。

当一个工单请求进来时，OTel 会生成一条完整的调用链路（Trace），类似这样的结构：

```
POST /api/tickets [1200ms]
├── receive_ticket_node [50ms]
│   ├── regex extraction [20ms]
│   └── entity normalization [30ms]
├── classify_intent_node [800ms]  ← 瓶颈在这里！
│   ├── LLM call: category [350ms]
│   ├── LLM call: urgency [320ms]
│   └── LLM call: priority [280ms]
├── route_decision_node [30ms]
└── try_auto_resolve_node [320ms]
    ├── knowledge_base_search [80ms]
    └── LLM call: generate_reply [240ms]
```

从这个链路图可以一眼看出，`classify_intent_node` 占用了总时间的 67%（800ms/1200ms），其中三个 LLM 调用占了绝大部分。这就是性能瓶颈所在！有了这个信息，我们就可以有针对性地进行优化——比如把三个 LLM 调用改成并行（上一节讲过的 asyncio.gather），或者引入缓存机制减少重复调用。

在实际的生产环境中，我们不会用 ConsoleSpanExporter（只是打印到控制台），而是会对接 Jaeger 或 Zipkin 这样的分布式追踪后端，配合它们的 Web UI 可以可视化和搜索所有的调用链路。

### 第三层：业务监控 — 关注真正的价值

基础设施和应用层面的监控虽然重要，但对业务决策者来说，他们更关心的是系统能不能帮助公司降低成本、提升效率、改善用户体验。这就需要一套面向业务的监控体系来回答这些问题。

#### 核心业务指标的定义与采集

对于智能客服工单系统，以下是最核心的业务指标：

**1. 工单吞吐量（Throughput）**
- 今日新建工单数
- 今日已关闭工单数
- 当前待处理工单数（积压量）
- 小时级工单到达率曲线

**2. 处理效率（Efficiency）**
- 平均处理时长（MTTR - Mean Time To Resolution）
- 自动解决率（Auto-Resolution Rate）：不需要人工干预就解决的工单占比
- 首次接触解决率（FCR - First Contact Resolution）
- 人工介入率（Human Intervention Rate）

**3. 质量指标（Quality）**
- 用户满意度评分（CSAT）
- 工单重开率（Reopen Rate）
- 分类准确率（Classification Accuracy）
- 知识库命中率（KB Hit Rate）

**4. 成本指标（Cost）**
- 单工单 Token 消耗量
- 单工单 LLM API 成本
- 日均 API 调用费用

让我们来实现这些指标的采集和计算逻辑：

```python
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List
import json

class BusinessMetricsCollector:
    def __init__(self):
        self.hourly_stats: Dict[str, Dict] = defaultdict(lambda: {
            "created": 0,
            "resolved": 0,
            "auto_resolved": 0,
            "escalated": 0,
            "human_intervened": 0,
            "total_tokens": 0,
            "total_llm_cost": 0.0,
            "resolution_times": [],
            "satisfaction_scores": []
        })
        
        self.category_distribution: Dict[str, int] = defaultdict(int)
        self.urgency_distribution: Dict[str, int] = defaultdict(int)
        
    def record_ticket_created(self, ticket_id: str, category: str, urgency: str):
        hour_key = datetime.now().strftime("%Y-%m-%d %H:00")
        self.hourly_stats[hour_key]["created"] += 1
        self.category_distribution[category] += 1
        self.urgency_distribution[urgency] += 1
        
    def record_ticket_resolved(self, ticket_id: str, was_auto: bool, resolution_time_seconds: float):
        hour_key = datetime.now().strftime("%Y-%m-%d %H:00")
        self.hourly_stats[hour_key]["resolved"] += 1
        if was_auto:
            self.hourly_stats[hour_key]["auto_resolved"] += 1
        else:
            self.hourly_stats[hour_key]["human_intervened"] += 1
        self.hourly_stats[hour_key]["resolution_times"].append(resolution_time_seconds)
        
    def record_ticket_escalated(self, ticket_id: str):
        hour_key = datetime.now().strftime("%Y-%m-%d %H:00")
        self.hourly_stats[hour_key]["escalated"] += 1
        
    def record_llm_usage(self, tokens_used: int, estimated_cost: float):
        hour_key = datetime.now().strftime("%Y-%m-%d %H:00")
        self.hourly_stats[hour_key]["total_tokens"] += tokens_used
        self.hourly_stats[hour_key]["total_llm_cost"] += estimated_cost
        
    def record_satisfaction(self, score: int):
        hour_key = datetime.now().strftime("%Y-%m-%d %H:00")
        self.hourly_stats[hour_key]["satisfaction_scores"].append(score)
        
    def get_dashboard_data(self) -> dict:
        now = datetime.now()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        
        today_stats = {
            "created": 0,
            "resolved": 0,
            "auto_resolved": 0,
            "escalated": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "resolution_times": [],
            "scores": []
        }
        
        for hour_key, stats in self.hourly_stats.items():
            hour_dt = datetime.strptime(hour_key, "%Y-%m-%d %H:00")
            if today_start <= hour_dt <= now:
                today_stats["created"] += stats["created"]
                today_stats["resolved"] += stats["resolved"]
                today_stats["auto_resolved"] += stats["auto_resolved"]
                today_stats["escalated"] += stats["escalated"]
                today_stats["total_tokens"] += stats["total_tokens"]
                today_stats["total_cost"] += stats["total_llm_cost"]
                today_stats["resolution_times"].extend(stats["resolution_times"])
                today_stats["scores"].extend(stats["satisfaction_scores"])
        
        avg_resolution_time = (
            sum(today_stats["resolution_times"]) / len(today_stats["resolution_times"])
            if today_stats["resolution_times"] else 0
        )
        
        auto_resolution_rate = (
            today_stats["auto_resolved"] / max(today_stats["resolved"], 1) * 100
        )
        
        avg_satisfaction = (
            sum(today_stats["scores"]) / len(today_stats["scores"])
            if today_stats["scores"] else 0
        )
        
        cost_per_ticket = (
            today_stats["total_cost"] / max(today_stats["created"], 1)
        )
        
        hourly_throughput = []
        for hour in range(24):
            hour_key = today_start.replace(hour=hour).strftime("%Y-%m-%d %H:00")
            count = self.hourly_stats.get(hour_key, {}).get("created", 0)
            hourly_throughput.append({"hour": f"{hour:02d}:00", "count": count})
        
        return {
            "summary": {
                "today_created": today_stats["created"],
                "today_resolved": today_stats["resolved"],
                "pending": today_stats["created"] - today_stats["resolved"],
                "auto_resolve_rate": round(auto_resolution_rate, 1),
                "avg_resolution_minutes": round(avg_resolution_time / 60, 1),
                "avg_csat": round(avg_satisfaction, 2),
                "total_tokens": today_stats["total_tokens"],
                "total_cost_usd": round(today_stats["total_cost"], 2),
                "cost_per_ticket": round(cost_per_ticket, 3)
            },
            "category_breakdown": dict(self.category_distribution),
            "urgency_breakdown": dict(self.urgency_distribution),
            "hourly_throughput": hourly_throughput
        }

collector = BusinessMetricsCollector()
```

这个 `BusinessMetricsCollector` 类实现了完整的业务指标采集逻辑。它的设计思路是这样的：用一个以小时为键的字典 `hourly_stats` 来存储每个小时的统计数据，每次发生业务事件（创建工单、解决工单、升级工单等）时调用对应的 `record_*` 方法来更新计数器。当需要展示仪表盘数据时，`get_dashboard_data()` 方法会把今天的所有小时数据聚合起来，计算出各项汇总指标。

特别值得注意的是几个派生指标的计算方式：
- **自动解决率** = auto_resolved / resolved × 100%，这个指标直接衡量了系统的智能化程度——越高说明越多的工单能由 AI 自动处理，节省人力成本。
- **平均处理时长** = 所有已解决工单的处理时间之和 / 数量，单位转换成了分钟以便阅读。
- **单工单成本** = 总 LLM API 费用 / 创建工单数，这个指标可以帮助我们评估系统的经济性。

为了让这些数据能被前端消费，我们在 FastAPI 中添加对应的端点：

```python
from fastapi import Query

@app.get("/api/analytics/dashboard")
async def get_analytics_dashboard(date: str = Query(None)):
    if date:
        pass
    return collector.get_dashboard_data()

@app.get("/api/analytics/hourly-trend")
async def get_hourly_trend(days: int = Query(7, ge=1, le=30)):
    trend_data = []
    for i in range(days):
        target_date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
        day_data = {"date": target_date}
        for hour in range(24):
            hour_key = f"{target_date} {hour:02d}:00"
            stats = collector.hourly_stats.get(hour_key, {})
            day_data[f"h{hour}"] = stats.get("created", 0)
        trend_data.append(day_data)
    return list(reversed(trend_data))

@app.get("/api/analytics/performance-report")
async def get_performance_report():
    all_rt = []
    for stats in collector.hourly_stats.values():
        all_rt.extend(stats["resolution_times"])
    
    if not all_rt:
        return {"message": "暂无足够数据"}
    
    all_rt_sorted = sorted(all_rt)
    n = len(all_rt_sorted)
    
    return {
        "total_resolved": n,
        "avg_minutes": round(sum(all_rt) / n / 60, 1),
        "p50_minutes": round(all_rt_sorted[n // 2] / 60, 1),
        "p90_minutes": round(all_rt_sorted[int(n * 0.9)] / 60, 1),
        "p99_minutes": round(all_rt_sorted[int(n * 0.99)] / 60, 1),
        "max_minutes": round(max(all_rt) / 60, 1),
        "min_minutes": round(min(all_rt) / 60, 1)
    }
```

这三个端点分别提供了不同粒度的分析数据：`dashboard` 返回实时仪表盘数据，`hourly-trend` 返回最近 N 天的小时级趋势（用于绘制热力图或面积图），`performance-report` 返回处理时长的详细分位数报告（P50/P90/P99 对于理解性能分布非常重要——平均值可能被极端值拉偏，而分位数更能反映真实体验）。

### 告警系统：问题要在用户发现之前发现

监控的目的不仅是看数据，更重要的是在异常发生时及时通知相关人员。一个好的告警系统应该满足几个条件：**准确性高**（不要频繁误报导致告警疲劳）、**及时性好**（关键问题要在几分钟内通知）、**分级清晰**（不同严重程度的通知给不同的人、走不同的渠道）。

我们来设计一个基于规则的告警引擎：

```python
import smtplib
from email.mime.text import MIMEText
from dataclasses import dataclass
from enum import Enum
from typing import List, Callable, Optional
import time

class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

@dataclass
class AlertRule:
    name: str
    level: AlertLevel
    check_func: Callable[[], bool]
    message_template: str
    cooldown_seconds: int = 300
    _last_triggered: float = 0
    
    def should_alert(self) -> tuple[bool, str]:
        if self.check_func():
            now = time.time()
            if now - self._last_triggered >= self.cooldown_seconds:
                self._last_triggered = now
                return True, self.message_template
        return False, ""

class AlertManager:
    def __init__(self):
        self.rules: List[AlertRule] = []
        self.alert_history: List[dict] = []
        self.max_history = 1000
        
    def add_rule(self, rule: AlertRule):
        self.rules.append(rule)
        
    def check_all_rules(self) -> List[dict]:
        alerts = []
        for rule in self.rules:
            should_fire, message = rule.should_alert()
            if should_fire:
                alert_record = {
                    "timestamp": datetime.now().isoformat(),
                    "rule": rule.name,
                    "level": rule.level.value,
                    "message": message
                }
                alerts.append(alert_record)
                self.alert_history.append(alert_record)
                if len(self.alert_history) > self.max_history:
                    self.alert_history.pop(0)
                    
                self._send_notification(alert_record)
                
        return alerts
    
    def _send_notification(self, alert: dict):
        level = alert["level"]
        msg = f"[{level.upper()}] {alert['rule']}: {alert['message']}"
        
        if level == "critical":
            send_sms(oncall_phone, msg)
            send_email(ops_team, msg)
            send_slack("#alerts-critical", msg)
        elif level == "warning":
            send_slack("#alerts-warning", msg)
            send_email(dev_team, msg)
        else:
            logger.info(msg)

alert_manager = AlertManager()

def get_current_error_rate() -> float:
    recent_errors = sum(
        1 for entry in collector.alert_history[-100:]
        if "error" in entry.get("message", "").lower()
    )
    return recent_errors / 100

alert_manager.add_rule(AlertRule(
    name="high_error_rate",
    level=AlertLevel.CRITICAL,
    check_func=lambda: get_current_error_rate() > 0.05,
    message_template="错误率超过 5%，当前值: {rate:.1%}",
    cooldown_seconds=300
))

alert_manager.add_rule(AlertRule(
    name="large_backlog",
    level=AlertLevel.WARNING,
    check_func=lambda: collector.get_dashboard_data()["summary"]["pending"] > 50,
    message_template="工单积压超过 50 条，当前积压: {count} 条",
    cooldown_seconds=600
))

alert_manager.add_rule(AlertRule(
    name="low_auto_resolve_rate",
    level=AlertLevel.WARNING,
    check_func=lambda: collector.get_dashboard_data()["summary"]["auto_resolve_rate"] < 30,
    message_template="自动解决率低于 30%，当前值: {rate:.1f}%",
    cooldown_seconds=1800
))

alert_manager.add_rule(AlertRule(
    name="llm_api_latency_spike",
    level=AlertLevel.CRITICAL,
    check_func=lambda: get_avg_llm_latency() > 10,
    message_template="LLM API 平均延迟超过 10s，当前值: {latency:.1f}s",
    cooldown_seconds=120
))
```

这个告警系统的设计包含几个核心组件：

**AlertRule** 类封装了一条告警规则——它有一个检查函数（`check_func`）来判断条件是否满足，一个冷却时间（`cooldown_seconds`）防止同一条规则在短时间内反复触发（避免告警轰炸），以及一个消息模板。`should_alert()` 方法会在条件满足且冷却期已过时返回 `True` 和格式化后的消息。

**AlertManager** 类负责管理所有规则，定期执行检查（可以放在后台定时任务中每 30 秒跑一次 `check_all_rules()`），并根据告警级别分发通知。这里实现了一个简单的分级通知策略：CRITICAL 级别的告警同时发送短信、邮件和 Slack（确保值班人员第一时间收到）；WARNING 级别只发 Slack 和邮件；INFO 级别只记日志。

示例中定义了四条典型的告警规则：
1. **错误率过高（CRITICAL）**：5 分钟内错误率 > 5% 说明可能有系统性故障
2. **工单积压过多（WARNING）**：待处理工单 > 50 条说明处理能力跟不上
3. **自动解决率过低（WARNING）**：< 30% 可能意味着分类模型或知识库有问题
4. **LLM 延迟过高（CRITICAL）**：> 10 秒说明上游 API 可能有故障

### 数据分析与持续改进：让系统越来越聪明

监控不仅是为了发现问题，更是为了驱动改进。通过对历史数据的分析，我们可以发现系统的薄弱环节、识别优化机会、验证改进效果。这一节我们来看看如何利用监控数据进行深度的业务分析和模型迭代。

#### 分析维度一：分类准确率的持续跟踪

意图分类是整个工单系统的入口——如果分类错了，后面的路由、自动回复、人工分配全都会出错。所以我们需要持续跟踪分类准确率，并且在准确率下降时及时发现。

```python
def track_classification_accuracy():
    accuracy_records = []
    
    for ticket_id, ticket in all_tickets.items():
        if ticket.get("agent_corrected_category"):
            predicted = ticket["category"]
            actual = ticket["agent_corrected_category"]
            
            accuracy_records.append({
                "ticket_id": ticket_id,
                "predicted": predicted,
                "actual": actual,
                "is_correct": predicted == actual,
                "timestamp": ticket["resolved_at"],
                "messages_preview": ticket["original_message"][:100]
            })
    
    total = len(accuracy_records)
    correct = sum(1 for r in accuracy_records if r["is_correct"])
    
    by_category = {}
    for r in accuracy_records:
        cat = r["actual"]
        if cat not in by_category:
            by_category[cat] = {"total": 0, "correct": 0, "errors": []}
        by_category[cat]["total"] += 1
        if r["is_correct"]:
            by_category[cat]["correct"] += 1
        else:
            by_category[cat]["errors"].append(r)
    
    report = {
        "overall_accuracy": correct / total * 100 if total > 0 else 0,
        "sample_size": total,
        "by_category": {
            cat: {
                "accuracy": data["correct"] / data["total"] * 100,
                "sample_size": data["total"],
                "common_errors": [
                    {"predicted": e["predicted"], "example": e["messages_preview"]}
                    for e in data["errors"][:5]
                ]
            }
            for cat, data in by_category.items()
        },
        "improvement_opportunities": generate_improvement_suggestions(by_category)
    }
    
    return report

def generate_improvement_suggestions(by_category):
    suggestions = []
    for cat, data in by_category.items():
        acc = data["correct"] / data["total"] * 100
        if acc < 70:
            suggestions.append({
                "category": cat,
                "issue": f"分类准确率仅 {acc:.1f}%，低于阈值",
                "suggestion": f"增加 '{cat}' 类别的训练样本，重点关注常见误分类模式"
            })
        error_patterns = analyze_error_patterns(data["errors"])
        if error_patterns:
            suggestions.append({
                "category": cat,
                "issue": "存在系统性误分类模式",
                "suggestion": f"误分类主要来自: {error_patterns}"
            })
    return suggestions
```

这段代码实现了一个分类准确率追踪系统。核心思路是：每当人工客服修正了系统自动分配的分类时，就把这条记录标记为"预测值 vs 实际值"，然后定期计算总体准确率和各类别的准确率。对于准确率较低的类别（< 70%），还会自动生成改进建议——比如增加训练样本、分析误分类模式等。

这种"人工反馈闭环"是提升 AI 系统质量的关键机制。很多团队犯的一个错误是上线后就不管模型的准确率了，几个月后发现效果已经大幅退化才去修复。而有了这套追踪系统，我们可以在准确率下降的第一时间发现问题、定位原因、采取行动。

#### 分析维度二：成本优化分析

LLM API 的费用是这类系统的主要运营成本之一，所以我们需要精细地分析 Token 消耗情况，找到优化的空间：

```python
def analyze_cost_optimization():
    cost_analysis = {
        "by_endpoint": {},
        "by_model": {},
        "optimization_recommendations": []
    }
    
    total_tokens = 0
    total_cost = 0
    
    for endpoint, calls in llm_call_log.items():
        endpoint_tokens = sum(c["tokens"] for c in calls)
        endpoint_cost = sum(c["cost"] for c in calls)
        total_tokens += endpoint_tokens
        total_cost += endpoint_cost
        
        cost_analysis["by_endpoint"][endpoint] = {
            "call_count": len(calls),
            "total_tokens": endpoint_tokens,
            "total_cost": round(endpoint_cost, 4),
            "avg_tokens_per_call": round(endpoint_tokens / len(calls), 0),
            "avg_cost_per_call": round(endpoint_cost / len(calls), 4)
        }
        
        if endpoint_cost > total_cost * 0.3:
            cost_analysis["optimization_recommendations"].append({
                "area": f"{endpoint} 接口",
                "current_cost": f"${endpoint_cost:.2f}",
                "percentage": f"{endpoint_cost/total_cost*100:.1f}%",
                "recommendations": [
                    "考虑增加缓存命中率",
                    "尝试切换到更便宜的模型（如 gpt-4o-mini 替代 gpt-4o）",
                    "精简 prompt 减少输入 token 数",
                    "批量合并多个小请求"
                ]
            })
    
    cost_analysis["total_daily"] = {
        "tokens": total_tokens,
        "cost_usd": round(total_cost, 2),
        "estimated_monthly": round(total_cost * 30, 2)
    }
    
    potential_savings = calculate_potential_savings(cost_analysis)
    cost_analysis["potential_savings"] = potential_savings
    
    return cost_analysis

def calculate_potential_savings(cost_analysis):
    savings = 0
    strategies = []
    
    for endpoint, data in cost_analysis["by_endpoint"].items():
        if "gpt-4o" in endpoint and data["avg_tokens_per_call"] < 500:
            saving = data["total_cost"] * 0.75
            savings += saving
            strategies.append(f"{endpoint}: 切换到 gpt-4o-mini 可节省 ~${saving:.2f}/天")
            
        if data["call_count"] > 1000:
            cacheable_ratio = estimate_cacheability(endpoint)
            if cacheable_ratio > 0.3:
                saving = data["total_cost"] * cacheable_ratio * 0.9
                savings += saving
                strategies.append(f"{endpoint}: 增加缓存（预估命中率 {cacheable_ratio*100:.0f}%）可节省 ~${saving:.2f}/天")
    
    return {
        "total_potential_daily_usd": round(savings, 2),
        "total_potential_monthly_usd": round(savings * 30, 2),
        "strategies": strategies
    }
```

这个成本分析工具会按接口和模型两个维度拆解 Token 消耗和费用，找出"大头"在哪里（比如某个接口占了总费用的 30% 以上），然后给出具体的优化建议——换便宜模型、加缓存、精简 prompt 等。最后还会计算潜在的节省金额，帮助管理层做投入产出比的决策。

### 总结：构建可观测性文化的三个要点

经过这一节的深入探讨，我们已经为智能客服工单系统建立了一套从基础设施到业务层面的完整监控体系。最后，我想强调三个在实践中容易被忽略但又至关重要的要点：

**第一，监控不是为了"看着好看"，而是为了"快速行动"。** 很多团队花了大量精力搭漂亮的 Grafana 大屏，但当告警真的触发时却没有人知道该做什么。正确的做法是：为每条告警规则配套一份**运维手册（Runbook）**，明确写出"当这个告警触发时，第一步查什么、第二步做什么、什么情况下可以自动恢复、什么时候需要人工介入"。没有 Runbook 的告警只是一条噪音。

**第二，指标不是越多越好，而是要对齐业务目标。** 初学者容易陷入"什么都想监控"的误区，最终搞了几百个面板几千个指标，反而找不到关键信息。建议遵循 **ONE Metric 原则**——先问自己"如果只能看一个指标来判断系统健康与否，我会选哪个？"找到那个核心指标后，再围绕它逐步补充辅助指标。对于我们的工单系统来说，那个 ONE Metric 可能是"自动解决率"——它综合反映了 AI 能力、知识库质量、流程设计的水平。

**第三，监控数据要驱动持续的改进闭环。** 最有价值的监控不是发现了多少问题，而是因为监控数据的驱动，系统变得越来越好。每周花 30 分钟回顾上周的核心指标趋势，每月做一次深度的分析报告，每季度根据数据调整一次优化方向——这种节奏能让监控投资产生最大的回报。

到这里，第九章"智能客服工单系统"的全部内容就结束了。我们从需求分析出发，经历了状态设计与节点实现、API 与前端开发、部署与优化、监控与分析五个阶段，完成了一个从零到生产的完整项目实战。在这个过程中，LangGraph 的状态管理、条件路由、人机协作、循环迭代、子图组合、持久化检查点等核心概念都得到了充分的实践。下一章，我们将迎来第二个项目——自主研究助手 Agent，它将带我们探索 LangGraph 在另一个完全不同的场景中的应用。
