---
title: 调试工具与可观测性
description: Callbacks 事件系统、LlamaIndex Trace 工具、日志最佳实践、性能瓶颈定位、常见调试案例
---
# 调试工具与可观测性（Debugging & Observability）

当你的 RAG 系统出现问题时——无论是答案不准确、响应太慢还是直接报错——你需要一套系统化的调试方法来快速定位问题所在。这一节我们学习 LlamaIndex 提供的调试和可观测性工具。

## Callbacks：事件驱动的可观测性

LlamaIndex 内置了一个强大的 **Callback（回调）系统**，它能让你"监听" RAG 管道中发生的每一个事件，包括：

```
┌─────────────────────────────────────────────────────┐
│                  RAG 查询执行过程                    │
│                                                     │
│  event_start(query)                              │
│      ↓                                            │
│  event_end(query)                                │
│                                                     │
│  event_start(retrieve)                            │
│      ↓                                            │
│  event_end(retrieve)                              │
│       → 返回 N 个节点                               │
│                                                     │
│  event_start(node_postprocessor)                   │
│      ↓                                            │
│  event_end(node_postprocessor)                    │
│       → 返回 M 个节点（过滤后）                     │
│                                                     │
│  event_start(synthesize)                          │
│      ↓                                            │
│  event_end(synthesize)                            │
│       → 最终 Response 对象                           │
│                                                     │
│  event_end(query)                                  │
└─────────────────────────────────────────────────────┘
```

每个事件都携带丰富的时间戳、元数据和耗时信息。

### 基础用法：CBEventHandler

```python
from llama_index.core.callbacks import CallbackManager, CBEventHandler
import time

handler = CBEventHandler(
    event_starts_to_ignore=[],  # 不忽略任何事件
    event_ends_to_ignore=[],
)

callback_manager = CallbackManager([handler])

index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(callback_manager=callback_manager)

start = time.perf_counter()
response = query_engine.query("产品的保修政策是什么？")
elapsed = (time.perf_counter() - start) * 1000

print(f"\n总耗时: {elapsed:.0f}ms")

# 查看 handler 收集的事件
for event_pair in handler.event_pairs:
    id_ = event_pair.id
    start = event_pair.start
    end = event_pair.end
    duration_ms = (end.time - start.time) * 1000 if end else 0

    print(f"[{id_}] {id_.split('.')[-1]} "
          f"({duration_ms:.0f}ms)")
```

输出示例：
```
[query.start] query (0.2ms)
[query.end] query (3520.3ms)     ← 总查询耗时
[retrieve.start] retrieve (0.1ms)
[retrieve.end] retrieve (120.5ms)   ← 检索花了 120ms
[node_postprocessor.start] node_postprocessor (0.0ms)
[node_postprocessor.end] node_postprocessor (85.3ms)  ← 后处理 85ms
[synthesize.start] synthesize (0.0ms)
[synthesize.end] synthesize (3315.2ms)  ← 合成占了绝大部分时间！
[query.end] query (3520.3ms)
```

从这个输出我们可以立刻看出：**这次查询中，合成阶段（3315ms）占据了总时间（3520ms）的 94%**！如果我们要优化性能，显然应该从合成环节入手而不是去调检索参数。

### 自定义 EventHandler

你可以通过继承 `CBEventHandler` 来自定义要关注的信息：

```python
class DebugEventHandler(CBHandler):
    def on_event_start(self, event, **kwargs):
        print(f"▶ START: {event.id_}")

    def on_event_end(self, event, **kwargs):
        duration = (event.time_sent - event.time) * 1000
        payload_size = kwargs.get("payload_size", "?")
        print(f"✅ END: {event.id_} ({duration:.0f}ms, size={payload_size})")


handler = DebugEventHandler()
callback_manager = CallbackManager([handler])
query_engine = index.as_query_engine(callback_manager=callback_manager)
response = query_engine.query("问题")
```

## LlamaIndex Debug 模式

除了 Callbacks，LlamaIndex 还提供了一个更高级的调试功能：`set_global_handler` + `start_trace`：

```python
import llama_index.core

llama_index.core.set_global_handler("simple")  # 启用简单追踪模式

documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

# 开启追踪
query_engine.query("问题", verbose=True)
```

`verbose=True` 会在控制台打印详细的执行过程，包括每次 LLM 调用的输入输出 Token 数。

## 日志最佳实践

### 结构化日志格式

```python
import logging
import json
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("rag_debug.log"),
        logging.StreamHandler(),
    ],
)


class StructuredLogger:
    """结构化日志记录器"""

    @staticmethod
    def log_query(query, response, latency_ms, metadata=None):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query[:100],
            "response_length": len(response.response),
            "source_count": len(response.source_nodes),
            "latency_ms": latency_ms,
            "top_score": max(
                (n.score for n in response.source_nodes), default=0
            ),
            "metadata": metadata or {},
        }
        logging.info(json.dumps(log_entry, ensure_ascii=False))
```

### 关键日志点

你应该在以下位置记录日志：

```python
# 1. 接收到查询时
logger.info(f"收到查询: {query}")

# 2. 检索完成后
logger.info(f"检索完成: {len(nodes)} 个候选, "
             f"top_score={nodes[0].score:.3f}")

# 3. 后处理完成后
logger.info(f"后处理完成: {len(filtered)} 个节点 "
             f"(原始: {len(raw_nodes)})")

# 4. 合成开始前
logger.info(f"开始合成, 使用 {mode} 模式, "
             f"{len(final_nodes)} 个节点")

# 5. 合成完成后
logger.info(f"合成完成: {len(response.response)} 字符, "
             f"耗时 {synth_elapsed}ms")

# 6. 用户反馈不满意时
if user_feedback == "bad":
    logger.warning(f"负面反馈! 查询: {query}, "
                  f"答案: {response.response[:100]}")
```

这些日志对于后续的问题排查极其有价值——特别是当你需要回溯"上周三那个奇怪的错误是怎么产生的"这种问题时。

## 性能瓶颈定位

当系统变慢时，Callbacks 能帮你精确定位瓶颈：

```python
class PerformanceProfiler(CBEventHandler):
    """性能分析器"""

    def __init__(self):
        self.phases = {}
        self.current_phase = None

    def on_event_start(self, event, **kwargs):
        self.current_phase = event.id_
        self.phases[event.id_] = {
            "start_time": event.time,
            "payload_size": kwargs.get("payload_size", 0),
        }
        print(f"🚀 开始: {event.id_}")

    def on_event_end(self, event, **kwargs):
        if self.current_phase == event.id_:
            phase = self.phases.pop(event.id_)
            duration = (event.time - phase["start_time"]) * 1000
            size = phase["payload_size"]
            print(f"✅ 完成: {event.id_} "
                  f"({duration:.0f}ms, size={size})")

            if duration > 1000:  # 超过 1 秒就值得警惕
                print(f"⚠️ 慢警告: {event.id_} 耗时 {duration:.0f}ms!")


# 使用
profiler = PerformanceProfiler()
callback_manager = CallbackManager([profiler])
query_engine = index.as_query_engine(callback_manager=callback_manager)
response = query_engine.query("复杂问题")
```

### 常见性能瓶颈及对应策略

| 瓶颈症状 | 可能原因 | 定位方法 | 解决方案 |
|-----------|---------|---------|---------|
| 总耗时 > 5s 且主要在合成阶段 | Synthesizer 模式不当或 Node 过多 | Callbacks 看 synthesize 耗时 | 换轻量模式；减少 top_k；压缩上下文 |
| 总耗时 > 5s 且主要在检索阶段 | 向量数据库慢或网络延迟高 | Callbacks 看 retrieve 耗时 | 换向量存储后端；启用缓存；检查网络 |
| 总耗时 > 5s 且各阶段都慢 | LLM API 本身慢 | Callbacks 各阶段累加确认 | 换更快的模型；启用异步并发 |
| 偶发性高 P99 延迟 | 单次查询正常但并发时排队 | 观察并发时的资源竞争 | 增加 Worker；做限流；水平扩展 |
| 内存占用持续增长 | 内存泄漏 | 监控进程内存 | 检查是否有未释放的大对象；重启服务 |

## 常见调试案例

### 案例一："答案明显错误"

**现象：** 用户问"S1 价格"，回答说是"299 元"，但实际价格已经改成了"399 元"。

**调试步骤：**
1. 检查该文档是否在最新一次索引构建之后更新过
2. 用 `retrieve("S1 价格")` 直接查看检索结果，确认新价格文档是否在 Top-K 中
3. 如果检索到了但合成时仍给出旧价格 → 检查 Prompt 是否允许"使用最新信息"

### 案例二："偶尔返回空答案"

**现象：** 同一个问题有时有答案，有时返回"我不知道"。

**调试步骤：**
1. 检查是否启用了 `similarity_cutoff` 且阈值过高
2. 用 Callbacks 查看 retrieve 返回的分数分布——是否有边缘 case 分数低于阈值
3. 检查是否有后处理器把所有结果都过滤掉了
4. 查询是否触发了某种异常降级逻辑

### 案例三："答案中有幻觉信息"

**现象：** 回答中包含了文档中不存在的产品功能。

**调试步骤：**
1. 用忠实度评估工具（第八章 8.3 节的方法）检测幻觉
2. 如果确认存在幻觉 → 检查 Prompt 约束是否足够强
3. 考虑启用事后校验（Post-hoc Validation）
4. 尝换为更强约束力更高的 LLM（如 GPT-4o 替代 mini 版本）

### 案例四："第一次快，第二次慢"

**现象：** 同一个查询，首次执行很快（<1s），第二次就很慢（>5s）。

**调试步骤：**
1. 第二次命中缓存了吗？（如果有缓存的话，可能是缓存失效后的重建）
2. 向量数据库的连接池是否耗尽？
3. LLM API 是否触发了速率限制？（检查 429 Too Many Requests）
4. 是否有垃圾回收（GC）导致的暂停？

## 可观测性的长期建设

调试不应该是一次性的救火行为，而应该是**持续的基础设施**：

```python
class ObservabilityDashboard:
    """RAG 系统可观测性仪表板"""

    def __init__(self):
        self.metrics = {
            "total_queries": 0,
            "total_errors": 0,
            "avg_latency_ms": 0,
            "p99_latency_ms": 0,
            "faithfulness_avg": 0,
            "hit_rate_avg: 0,
            "negative_feedback_count": 0,
        }
        self.recent_errors = []

    def record_query(self, query, response, latency_ms, success=True, error=None):
        self.metrics["total_queries"] += 1
        self.metrics["total_latency_ms"] += latency_ms

        if not success:
            self.metrics["total_errors"] += 1
            self.recent_errors.append({
                "time": datetime.now().isoformat(),
                "query": query[:80],
                "error": error,
            })
            self.metrics["negative_feedback_count"] += 1

        # 更新滑动平均延迟
        alpha = 0.1
        self.metrics["avg_latency_ms"] = (
            (1 - alpha) * self.metrics["avg_latency_ms"]
            + alpha * latency_ms
        )
        return self

    def get_health_status(self):
        error_rate = self.metrics["total_errors"] / max(
            self.metrics["total_queries"], 1
        )
        if error_rate > 0.1:  # 错误率超过 10%
            return "UNHEALTHY"
        if self.metrics["avg_latency_ms"] > 5000:
            return "DEGRADED"
        return "HEALTHY"


dashboard = ObservabilityDashboard()

# 在每个查询后记录
record_query(dashboard, question, response, latency=3200)
print(dashboard.get_health_status())
```

将这样的仪表板接入监控系统（如 Prometheus + Grafana），你就能实时监控 RAG 系统的健康状态，在问题影响用户之前就收到告警。
