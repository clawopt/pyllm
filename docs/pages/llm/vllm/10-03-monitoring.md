# 监控、日志与告警

> **白板时间**：你的 vLLM 服务已经在生产环境跑了。凌晨 3 点老板打电话问："服务正常吗？" 你怎么回答？靠猜？不——你需要一套完整的可观测性（Observability）体系：**监控**看现在是否健康，**日志**看发生了什么，**告警**看什么时候出了问题。这三者构成了生产运维的"三只眼"。

## 一、vLLM 内置 Metrics 端点

### 1.1 内置指标一览

vLLM 在 `/metrics` 端点暴露 Prometheus 格式的指标：

```
GET /metrics
Content-Type: text/plain; version=0.0.4; charset=utf-8

# ===== 核心指标 =====

# 请求计数
vllm:requests_total{model="...", status="success|fail"}    # 总请求数（按状态）
vllm:request_success_total                                 # 成功请求数
vllm:request_failure_total                                  # 失败请求数

# 运行状态
vllm:num_requests_running{model="..."}                    # 当前正在运行的请求数
vllm:num_requests_waiting{model="..."}                     # 排队中的请求数
vllm:num_requests_swapped{model="..."}                    # 被 swap out 的请求数

# GPU 资源
vllm:gpu_cache_usage_perc{gpu_id="0", model="..."}      # GPU KV Cache 使用率 (0-100+)
vllm:gpu_cache_usage_blocks{gpu_id="0", model="..."}   # 已用 Block 数 / 总 Block 数

# 性能延迟 (Histogram)
vllm:e2e_request_latency_seconds_bucket{model="..."}     # 端到端延迟分布
vllm:time_to_first_token_seconds_bucket{model="..."}        # TTFT 分布 ⭐
vllm:time_per_output_token_seconds_bucket{model="..."}     # TPOT 分布 ⭐
vllm:generation_output_tokens_total{model="..."}         # 总生成 token 数

# 预处理/生成 吞吐
vllm:prompt_tokens_total{model="..."}                   # 处理的 prompt token 总数
vllm:generation_throughput_tokens{model="..."}             # tokens/sec 吞吐量
```

### 1.2 指标解读指南

```python
def metrics_guide():
    """vLLM Metrics 完整解读"""
    
    guide = """
    ══════════════════════════════════════════════════════
              vLLM 监控指标解读指南
    ══════════════════════════════════════════════════════
    
    ┌─────────────────────┬──────────┬─────────────────────────────┐
    │ 指标                 │ 正常值    │ 告警阈值 / 含义            │
    ├─────────────────────┼──────────┼─────────────────────────────┤
    │                     │          │                             │
    │ num_running        │ 2-20     │ > max_seqs × 0.8 → 达到上限       │
    │ num_waiting         │ 0-5      │ > 15 → 吞吐不足，需扩容       │
    │ num_swapped         │ 0        │ > 0 → 有请求被换出到 CPU        │
    │                     │          │                             │
    │ gpu_cache_usage_%  │ 60-90%   │ > 95% → 即将 OOM ⚠️           │
    │                     │          │                             │
    │ TTFT P50           │ < 500ms  │ > 1s → 用户感知慢               │
    │ TTFT P99           │ < 1.5s   │ > 3s → SLA 违背               │
    │ TPOT P50           │ < 40ms   │ > 80ms → 打字卡顿             │
    │ TPOT P99           │ < 100ms  │ > 200ms → 严重卡顿            │
    │                     │          │                             │
    │ throughput(tok/s)  │ 稳定     │ 突降 > 30% → 异常               │
    │ success_rate       │ > 99%    │ < 98% → 有系统性问题           │
    │ failure_rate       │ < 1%     │ > 3% → 需要立即排查           │
    └─────────────────────┴──────────┴─────────────────────────────┘
    
    📊 三大黄金指标:
    
    ┌──────────────────────────────────────────────────────┐
    │ 1️⃣ TTFT P99 (Time to First Token, P99)                │
    │    定义: 99% 的请求在多长时间内收到首个 token        │
    │    影响: 用户感知速度的第一印象                          │
    │    目标: < 1.5s (交互场景) / < 3s (批处理)            │
    │                                                       │
    │ 2️⃣ TPOT P99 (Time Per Output Token, P99)             │
    │    定义: 99% 的输出 token 间隔时间的 P99 值          │
    │    影响: "打字"流畅感                                   │
    │    目标: < 100ms                                        │
    │                                                       │
    │ 3️⃣ GPU Cache 使用率                                    │
    │    定义: KV Cache 占用 GPU 显存的比例                  │
    │    影响: 能否接纳更多并发请求                            │
    │    目标: 70-90% (太高=OOM风险, 太低=浪费资源)        │
    └──────────────────────────────────────────────────────────────┘
    """
    print(guide)

metrics_guide()
```

## 二、Prometheus + Grafana 配置

### 2.1 Prometheus Scrape 配置

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'vllm-primary'
    scrape_interval: 5s          # vLLM 指标变化快，需要更频繁采集
    static_configs:
      - targets: ['vllm-svc.llm-system.svc:8000']
    metrics_path: '/metrics'
    sample_limit: 50000
    
  - job_name: 'vllm-secondary'
    static_configs:
      - targets: ['vllm-svc-backup.llm-system.svc:8001']
    metrics_path: '/metrics'
```

### 2.2 Grafana Dashboard JSON

```json
{
  "dashboard": {
    "title": "vLLM Production Dashboard",
    "tags": ["vllm", "production", "monitoring"],
    "panels": [
      {
        "title": "QPS & Throughput",
        "type": "timeseries",
        "gridPos": {"h": 8, "w": 12},
        "targets": [
          {
            "expr": "rate(vllm:request_success_total[5m])",
            "legendFormat": "{{value}} req/s"
          },
          {
            "expr": "rate(vllm:generation_output_tokens_total[5m])",
            "legendFormat": "{{value}} tok/s"
          }
        ]
      },
      {
        "title": "TTFT Distribution",
        "type": "heatmap",
        "gridPos": {"h": 8, "w": 12},
        "targets": [
          {
            "expr": "histogram_quantile(0.95, vllm:time_to_first_token_seconds_bucket)",
            "legendFormat": "P95 {{value}}ms"
          }
        ]
      },
      {
        "title": "TPOT Distribution",
        "type": "heatmap",
        "gridPos": {"h": 8, "w": 12},
        "targets": [
          {
            "expr": "histogram_quantile(0.95, vllm:time_per_output_token_seconds_bucket)",
            "legendFormat": "P95 {{value}}ms"
          }
        ]
      },
      {
        "title": "GPU Cache Usage",
        "type": "gauge",
        "gridPos": {"h": 8, "w": 12},
        "targets": [
          {
            "expr": "avg(vllm:gpu_cache_usage_perc)",
            "legendFormat": "{{value}}%"
          }
        ]
      },
      {
        "title": "Request Queue Depth",
        "type": "graph",
        "gridPos": {"h": 8, "w": 12},
        "targets": [
          {
            "expr": "vllm:num_requests_waiting",
            "legendFormat": "排队中"
          },
          {
            "expr": "vllm:num_requests_running",
            "legendFormat": "运行中"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "stat",
        "gridPos": {"h": 8, "w": 12},
        "targets": [
          {
            "expr": "rate(vllm:request_failure_total[5m])",
            "legendFormat": "失败/min"
          }
        ]
      }
    ]
  }
}
```

## 三、结构化日志

### 3.1 日志格式设计

vLLM 支持通过 `--disable-log-requests` 和 `VLLM_LOGGING_LEVEL` 控制日志行为：

```python
def log_format_design():
    """vLLM 生产级日志格式规范"""
    
    spec = """
    ══════════════════════════════════════════════════════
              vLLM 结构化日志规范
    ══════════════════════════════════════════════════════
    
    日志级别选择:
    ├── WARNING  → 仅错误和警告 (推荐生产默认)
    ├── INFO     → 关键事件 (启动/停止/模型加载)
    ├── DEBUG    → 详细调试信息 (仅开发/排障)
    └── ERROR    → 仅严重错误
    
    推荐配置:
    ├── 生产环境: VLLM_LOGGING_LEVEL=WARNING
    ├── 排查阶段: VLLM_LOGGING_LEVEL=DEBUG
    └── 性能测试: VLLM_LOGGING_LEVEL=INFO + --disable-log-requests
    
    日志字段 (每条日志):
    {
      "timestamp": "2024-12-15T03:24:56.789Z",
      "level": "info",
      "message": "Request completed",
      "request_id": "req-abc123",
      "model": "Qwen/Qwen2.5-7B-Instruct",
      "lora": null,
      
      "timing": {
        "ttft_ms": 245.3,
        "tpot_ms": 32.1,
        "e2e_latency_ms": 1523.7,
        "queue_time_ms": 12.1,
        "total_tokens": 45
      },
      
      "usage": {
        "prompt_tokens": 18,
        "completion_tokens": 27,
        "total_tokens": 45
      },
      
      "client_ip": "10.0.0.123",
      "status": "success",
      "finish_reason": "stop"
    }
    
    日志聚合方案:
    ├── Fluentd → Elasticsearch → Kibana (轻量)
    ├── Vector → Loki → Grafana Explore (云原生)
    └── Docker json-file driver → Loki (最简单)
    """
    print(spec)

log_format_design()
```

### 3.2 Fluent Bitd + Loki 收集

```dockerfile
# Dockerfile.fluent-bitd
FROM fluent/fluent-bit:latest

USER root

# 安装 plugins
fluent-gem install fluent-plugin-loki
fluent-gem install fluent-plugin-prometheus-remote-write

COPY fluent.conf /fluent/etc/fluent.conf

EXPOSE 20224
ENTRYPOINT ["fluent-bit", "-c", "/fluent/etc/fluent.conf"]
```

```xml
<!-- fluent.conf -->
<source>
  @type forward
  port 24224
</source>

<source>
  @type tail
  path /var/log/vllm/*.log
  pos_file /var/log/vllm/fluent.pos
  tag kubernetes.*
  format json
</source>

<filter>
  @type record_transformer
  <record>
    log_level ${$.["level"]}
  </record>
</filter>

<match>
  @type loki
  url http://loki:3100/loki/api/v1/push
  extra {"app": "vllm-inference"}}
</match>
```

## 四、告警规则

### 4.1 关键告警规则

```yaml
# alerts/vllm-alerts.yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  labels:
    severity: warning
    team: llm-ops
spec:
  groups:
    - name: vllm-critical
      rules:
      - alert: VLLMDOWN
        expr: up{job="vllm-exporter"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "vLLM 服务不可达"
          description: "vLLM inference service is down and not responding to health checks."
        
      - alert: HighLatencyTTFT
        expr: histogram_quantile(0.99, rate(vllm:time_to_first_token_seconds_bucket[5m])) > 2000
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "TTFT P99 过高"
          description: "99% of requests take > 2s to return first token."
        
      - alert: HighLatencyTPOT
        expr: histogram_quantile(0.99, rate(vllm:time_per_output_token_seconds_bucket[5m]) > 150
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "TPOT P99 过高"
          description: "Output token interval P99 > 150ms."
        
      - alert: GPUCacheExhausted
        expr: avg(vllm:gpu_cache_usage_perc) > 95
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "GPU Cache 将耗尽"
          description: "Average GPU cache usage exceeds 95%, imminent OOM risk."
        
      - alert: QueueBacklog
        expr: avg(vllm:num_requests_waiting) > 30
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "请求队列积压"
          description: "Average waiting requests exceed 30, indicating capacity shortage."
        
      - alert: ErrorRateSpike
        expr |>
          (
            rate(vllm:request_failure_total[5m])
            /
            (rate(vllm:request_success_total[5m]) + rate(vllm:request_failure_total[5m]))
          ) > 0.05
        |
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "错误率飙升"
          description: "Error rate exceeded 5% in the last 5 minutes."

receivers:
  - name: slack-webhook
    url: https://hooks.slack.com/services/TXXXXXXXXX/BXXXXXXXXXXX
    resolver: webhoo
```

## 五、OpenTelemetry 分布式追踪

### 5.1 配置 vLLM 输出 Traces

```bash
# 启动时启用 OpenTelemetry
export OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4318
export OTEL_SERVICE_NAME=vllm-inference

python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --enable-opentelemetry \
    --port 8000
```

### 5.2 Jaeger UI 查看调用链路

```
用户请求链路 (Jaeger Trace View):

Client App
    │ POST /v1/chat/completions [t=0ms]
    ▼
Nginx (reverse proxy)
    │ proxy_pass to :8000 [+2ms]
    ▼
vLLM Server
    │ receive request [t=3ms]
    │ tokenize prompt [t=15ms]
    │ scheduler.admit() [t=18ms]
    │ model.forward() generate loop [t=200ms]
    │ return response [t=205ms]
    ▼
Client receives first chunk [t=210ms] ← TTFT = ~210ms ✅
```

---

## 六、总结

本节完成了完整的可观测性体系：

| 层 | 工具 | 功能 |
|-----|------|------|
| **Metrics** | Prometheus (`/metrics`) | 内置 20+ 指标，PromQL 查询 |
| **可视化** | Grafana Dashboard | TTFT/TPOT/GPU/Queue 四大面板 |
| **日志** | Fluent Bitd → Loki | 结构化 JSON 格式，标签路由 |
| **告警** | AlertManager → Slack/Email/PagerDuty | 6 条关键告警规则 |
| **追踪** | OpenTelemetry → Jaeger | 端到端完整调用链路 |

**核心要点回顾**：

1. **三大黄金指标是监控的核心**：TTFT P99（用户体验）、TPOT P99（流畅度）、GPU Cache%（容量）
2. **`proxy_buffering off` 是 Nginx 第一铁律**——没有它 SSE 流式无法工作
3. **日志要结构化不要裸文本**——JSON 格式 + 字段索引才能做高效查询
4. **告警要分级**：Critical（立即响应）> Warning（关注）> Info（记录）
5. **OpenTelemetry 让你能看到每个请求的完整生命周期**——从 Nginx 到 vLLM 到模型推理

下一节我们将学习 **高可用与弹性伸缩**。
