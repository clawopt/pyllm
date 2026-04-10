# 9.2 模型监控与可观测性

上一节我们把模型包装成了可访问的 API 服务，它能够接收请求、生成响应、返回结果。但服务上线之后故事才刚刚开始——你怎么知道它是否在正常工作？用户反馈"回答质量下降了"是主观感受还是客观事实？模型有没有出现退化（drift）？GPU 利用率是多少？平均延迟和 P99 延迟分别是多少？这些问题的答案决定了你的服务是"能跑"还是"跑得好"。这一节我们将构建一个完整的**可观测性（Observability）体系**——从日志到指标，从追踪到告警，让你的 LLM 服务在生产环境中完全透明可控。

## 可观测性的三大支柱

业界标准的可观测性框架由三个核心组成部分构成：

```
                    ┌─────────────────────┐
                    │   OBSERVABILITY      │
                    │  (可观测性)           │
                    ├──────────┬──────────┤
                    │          │          │
              ┌─────┴──┐  ┌───┴─────┐  ┌──┴──────┐
              │  LOGS  │  │ METRICS │  │ TRACES  │
              │ (日志)  │  │ (指标)  │  │ (追踪)  │
              └────────┘  └─────────┘  └─────────┘
              
发生了什么?    发生了多少?     花了多长时间?
(What happened) (How much/often) (How long)
```

**Logs（日志）**：记录离散的事件——"用户 A 在 14:32:05 发送了一个 prompt"、"第 42 步的 loss 是 2.345"、"GPU 0 的温度达到了 82°C"。日志适合排查具体问题（"为什么这个请求返回了错误？"），但不适合做趋势分析。

**Metrics（指标）**：数值化的聚合数据——"过去 5 分钟的平均延迟是 234ms"、"P99 延迟是 1.2 秒"、"当前 QPS 是 15.3"、"GPU 利用率 87%"。指标适合做监控面板、告警和容量规划。

**Traces（追踪）**：请求的完整生命周期——从接收 HTTP 请求 → tokenize → 前向传播 → sampling → decode → 返回响应，每一步的耗时都记录下来。追踪适合分析性能瓶颈（"哪一步最慢？"）。

对于 LLM 推理服务来说，这三者缺一不可。下面我们逐一实现。

## 结构化日志：用 JSON 替代 print()

很多 Python 开发者习惯用 `print()` 来输出日志。这在开发阶段没问题，但在生产环境中有几个致命缺陷：print 输出混在标准输出中难以过滤；没有时间戳和级别信息；无法被日志收集系统（如 ELK/Loki）自动解析。

```python
import logging
import json
import time
import sys


class JSONFormatter(logging.Formatter):
    """结构化 JSON 日志格式化器"""
    
    def format(self, record):
        log_obj = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.%fZ", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
        }
        
        if hasattr(record, 'extra_data'):
            log_obj.update(record.extra_data)
        
        return json.dumps(log_obj, ensure_ascii=False)


def setup_logger(name="llm_service", level=logging.INFO):
    """配置结构化日志"""
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)
    
    return logger


logger = setup_logger()


def generate_with_logging(service, request):
    """带完整日志记录的推理函数"""
    
    req_id = f"{time.time_ns()}"[:12]
    
    logger.info("request_started", extra={
        "extra_data": {
            "request_id": req_id,
            "prompt_length": len(request.prompt),
            "max_tokens": request.max_new_tokens,
            "temperature": request.temperature,
        }
    })
    
    t_start = time.perf_counter()
    
    try:
        response = service.generate(request)
        
        latency_ms = (time.perf_counter() - t_start) * 1000
        
        logger.info("request_completed", extra={
            "extra_data": {
                "request_id": req_id,
                "latency_ms": round(latency_ms, 2),
                "tokens_generated": response.tokens_generated,
                "tokens_per_sec": round(response.tokens_generated / (latency_ms/1000), 1),
                "response_length": len(response.text),
            }
        })
        
        return response
        
    except Exception as e:
        latency_ms = (time.perf_counter() - t_start) * 1000
        
        logger.error("request_failed", extra={
            "extra_data": {
                "request_id": req_id,
                "latency_ms": round(latency_ms, 2),
                "error_type": type(e).__name__,
                "error_message": str(e)[:200],
            }
        })
        
        raise


# 日志输出示例:
# {"timestamp":"2024-03-15T14:32:05.123456Z","level":"INFO",
#  "logger":"llm_service","message":"request_started",
#  "module":"serve","line":42,"extra_data":
#  {"request_id":"837291028371","prompt_length":25,"max_tokens":256}}
#
# {"timestamp":"2024-03-15T14:32:06.567890Z","level":"INFO",
#  "logger":"llm_service","message":"request_completed",
#  "module":"serve","line":72,"extra_data":
#  {"request_id":"837291028371","latency_ms":1443.21,"tokens_generated":86,...}}
```

这种 JSON 格式的日志可以直接被 Loki（Grafana 的日志系统）、CloudWatch Logs 或 ELK Stack 收集和查询。比如你想找出所有耗时超过 5 秒的请求：

```sql
-- Loki 查询示例
{job="llm-service"} |= "request_completed"
| json
| __error__ != ""
| extra_data.latency_ms > 5000
```

## 核心指标定义与采集

对于 LLM 推理服务来说，以下指标是最关键的：

### 1. 延迟指标（Latency Metrics）

```python
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
import time

registry = CollectorRegistry()

REQUEST_LATENCY = Histogram(
    "llm_request_duration_seconds",
    "Request latency in seconds",
    ["endpoint", "model_name"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
    registry=registry,
)

PREFILL_LATENCY = Histogram(
    "llm_prefill_duration_seconds",
    "Prefill phase duration",
    registry=registry,
)

DECODE_LATENCY = Histogram(
    "llm_decode_duration_seconds",
    "Decode phase duration per token",
    registry=registry,
)

TOKENS_GENERATED = Counter(
    "llm_tokens_generated_total",
    "Total tokens generated",
    ["model_name"],
    registry=registry,
)

GPU_UTILIZATION = Gauge(
    "gpu_utilization_percent",
    "GPU utilization percentage",
    ["device"],
    registry=registry,
)

GPU_MEMORY_USED = Gauge(
    "gpu_memory_used_bytes",
    "GPU memory used in bytes",
    ["device"],
    registry=registry,
)


def record_metrics(request, response, prefill_time, decode_time):
    """记录一次请求的所有指标"""
    
    total_latency = response.latency_ms / 1000
    
    REQUEST_LATENCY.labels(
        endpoint="/v1/chat/completions",
        model_name=response.model_name,
    ).observe(total_latency)
    
    PREFILL_LATENCY.observe(prefill_time)
    DECODE_LATENCY.observe(decode_time / max(response.tokens_generated, 1))
    
    TOKENS_GENERATED.labels(model_name=response.model_name).inc(response.tokens_generated)
    
    # GPU 指标
    for i in range(torch.cuda.device_count()):
        util = torch.cuda.utilization(i) * 100
        mem_used = torch.cuda.memory_allocated(i)
        
        GPU_UTILIZATION.labels(device=f"cuda:{i}").set(util)
        GPU_MEMORY_USED.labels(device=f"cuda:{i}").set(mem_used)
```

### 2. 质量/健康指标（Quality & Health Metrics）

除了性能指标，你还需要监控模型输出的质量——这比延迟指标更难定义但同样重要：

```python
OUTPUT_LENGTH = Histogram(
    "llm_output_length_tokens",
    "Generated output length in tokens",
    registry=registry,
)

PERPLEXITY_SCORE = Histogram(
    "llm_perplexity_score",
    "Perplexity of generated text (sampled)",
    registry=registry,
)

EMPTY_RESPONSE_COUNT = Counter(
    "llm_empty_responses_total",
    "Count of empty or very short responses",
    registry=registry,
)

ERROR_COUNT = Counter(
    "llm_errors_total",
    "Count of errors by type",
    ["error_type"],
    registry=registry,
)


def record_quality_metrics(response):
    """记录质量相关指标"""
    
    OUTPUT_LENGTH.observe(len(response.text))
    
    if len(response.text.strip()) < 10:
        EMPTY_RESPONSE_COUNT.inc()
    
    # 简单启发式：检查是否有重复内容（LLM 幻觉的一种表现）
    words = response.text.split()
    if len(words) > 20:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.3:
            logger.warning("low_diversity_detected", extra={
                "extra_data": {
                    "unique_ratio": round(unique_ratio, 3),
                    "total_words": len(words),
                    "preview": response.text[:100],
                }
            })
```

### Prometheus 集成与 Grafana Dashboard

有了指标定义后，需要暴露一个 `/metrics` 端点供 Prometheus 抓取：

```python
from fastapi import Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST


@app.get("/metrics")
async def metrics():
    return Response(
        generate_latest(registry),
        media_type=CONTENT_TYPE_LATEST,
    )
```

Prometheus 每 15 秒（默认配置）抓取一次 `/metrics`，然后你可以用 Grafana 构建可视化仪表板。以下是推荐的 Dashboard 布局：

```
┌─────────────────────────────────────────────────────────┐
│                  LLM Service Dashboard               │
├──────────────┬──────────────┬──────────────┬──────────┤
│  Request Rate │  Latency p50  │  Latency p99  │ Errors/s │
│    15.3 rps  │    234 ms     │   1,203 ms    │   0.02   │
├──────────────┼──────────────┼──────────────┼──────────┤
│ Tokens/sec  │  Avg Output  │  GPU Util %  │  Mem GB  │
│   1,247 t/s │   86 tokens  │     87%       │   18.2   │
├──────────────┴──────────────┴──────────────┴──────────┤
│                    Time Series (last 1h)           │
│  ┌──────────────────────────────────────────────┐    │
│  │  ▁▃▂ Request Rate (rps)                     │    │
│  │  ▔▃▂ Latency (ms)                           │    │
│  └──────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────┘
```

## 分布式追踪：OpenTelemetry 实现

当服务变慢时，你需要知道时间花在哪里了——是在 tokenize？在前向传播？还是在 sampling？分布式追踪（Distributed Tracing）就是用来回答这个问题的。

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.exporter.otlp.proto.grpc_trace_exporter import (
    OTLPSpanExporter,
)
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource


def setup_tracing(service_name="llm-service"):
    """初始化 OpenTelemetry tracing"""
    
    resource = Resource.create({
        "service.name": service_name,
        "service.version": "1.0.0",
    })
    
    exporter = OTLPSpanExporter(endpoint="localhost:4317")
    
    provider = TracerProvider(
        resource=resource,
        span_processors=[BatchSpanProcessor(exporter)],
    )
    
    trace.set_tracer_provider(provider)
    return provider.get_tracer(__name__)


tracer = setup_tracing()


@tracer.start_as_current_span("generate")
def traced_generate(service, request):
    """带追踪的推理函数"""
    
    with tracer.start_as_span("tokenize") as span:
        messages = [{"role": "user", "content": request.prompt}]
        text = service.tokenizer.apply_chat_template(messages, tokenize=False)
        inputs = service.tokenizer(text, return_tensors="pt").to(service.device)
        span.set_attribute("input.length", str(inputs['input_ids'].shape[1]))
    
    with tracer.start_as_span("prefill") as span:
        t_start = time.perf_counter()
        with tracer.start_as_span("forward_pass"):
            with torch.no_grad():
                outputs = service.model(**inputs)
        prefill_time = time.perf_counter() - t_start
        span.set_attribute("prefill.ms", str(prefill_time * 1000))
    
    with tracer.start_as_span("decode") as span:
        t_start = time.perf_counter()
        output_ids = service.model.generate(
            **inputs,
            max_new_tokens=request.max_new_tokens,
            do_sample=request.do_sample,
        )
        decode_time = time.perf_counter() - t_start
        gen_count = output_ids.shape[1] - inputs['input_ids'].shape[1]
        span.set_attribute("decode.ms", str(decode_time * 1000))
        span.set_attribute("tokens.generated", str(gen_count))
        span.set_attribute("decode.ms_per_token", 
                          str(decode_time/max(gen_count,1)*1000))
    
    with tracer.start_as_span("decode_text") as span:
        generated_text = service.tokenizer.decode(
            output_ids[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True,
        )
        span.set_attribute("output.length", str(len(generated_text)))
    
    return generated_text
```

追踪数据发送到 OpenTelemetry Collector 后，可以在 Jaeger 或 Grafana Tempo 中看到类似这样的调用链视图：

```
POST /v1/chat/completions [1.23s]
├── generate [1.22s]
│   ├── tokenize [3.2ms]
│   ├── prefill [445.6ms]
│   │   └── forward_pass [442.1ms]
│   │       ├── embedding [2.1ms]
│   │       ├── block_0 [28.3ms]
│   │       │   ├── attention [18.2ms]  ← 最慢！
│   │       │   └── ffn [9.8ms]
│   │       ├── block_1 [26.7ms]
│   │       │   ...
│   │       └── lm_head [0.3ms]
│   └── decode [772.3ms]
│       ├── step_1 [6.1ms]
│       ├── step_2 [6.0ms]
│       ├── ...
│       └── step_86 [5.9ms]
└── decode_text [0.2ms]
```

从这个视图中你能立刻发现：Prefill 占了 36% 的时间，其中 Attention 层是 Prefill 的主要瓶颈；Decode 占了 63%，每步约 6ms。如果你要优化性能，这个图会告诉你应该优先优化哪里。

## 告警规则定义

有了指标之后，下一步是设置告警——让系统在出问题时主动通知你，而不是等你手动去查看 Dashboard。

```yaml
# prometheus_rules.yml — 告警规则示例
groups:
  - name: llm_service_alerts
    rules:
      # 高延迟告警
      - alert: HighLatencyP99
        expr: histogram_quantile(0.99, rate(llm_request_duration_seconds_bucket[5m])) > 5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "P99 latency > 5 seconds"
          description: "P99 latency is {{ $value }}s over the last 5 minutes"

      # 错误率告警
      - alert: HighErrorRate
        expr: rate(llm_errors_total[5m]) / rate(llm_request_duration_seconds_count[5m]) > 0.01
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Error rate > 1%"
          description: "Error rate is {{ $value | humanizePercentage }}"

      # GPU 显存告警
      - alert: HighGPUMemory
        expr: gpu_memory_used_bytes / (1024*1024*1024) > 70
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "GPU memory > 70GB"
          description: "GPU {{ $labels.device }} using {{ $value }}GB"

      # 空回复率告警（可能意味着模型问题）
      - alert: EmptyResponseRate
        expr: rate(llm_empty_responses_total[10m]) / rate(llm_request_duration_seconds_count[10m]) > 0.05
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Empty response rate > 5%"
          description: "{{ $value | humanizePercentage }} responses are empty"
```

## 日志 + 指标 + 追踪 的协同效应

这三个组件不是独立工作的——它们互相补充形成一个完整的诊断闭环：

```
告警触发 (Alert fired!)
    │
    ▼ "P99 latency > 5s"
    │
    ├──→ 查看 Metrics Dashboard
    │   发现: 延迟 spike 发生在 14:30~14:35
    │   GPU利用率正常 (85%)
    │   错误率没有上升
    │
    ├──→ 查询对应时间的 Traces
    │   发现: 多个请求的 decode 阶段异常慢 (>10ms/token)
    │   正常应该是 ~6ms/token
    │
    └──→ 过滤同时间段内的 Logs
        发现: "CUDA OOM: out of memory" 出现多次
        + "fallback to CPU offloading"
        
结论: 某个请求导致 KV Cache 暴涨，
      触发了频繁的 CPU-GPU 数据交换
      → 解决: 增加 max_batch_size 限制或启用 PagedAttention
```

这个诊断流程展示了可观测性的真正价值：不是等用户投诉后才去查问题，而是系统主动告诉你"有问题"，然后给你提供足够的信息让你快速定位根因。
