# 5.2 重试与容错模式

> 在上一节中我们讨论的自省循环主要关注"质量改进"——通过多轮迭代让输出变得更好。但循环结构的另一个重要用途是"错误恢复"——当某个操作因为临时性故障（网络超时、API 限流、服务不可用等）而失败时，通过自动重试来从故障中恢复。这一节我们会深入探讨 LangGraph 中各种重试与容错模式的实现方法，包括简单重试、指数退避重试、断路器模式以及级联失败的处理策略。

## 简单重试：固定次数的重试机制

最基础的重试模式是：操作失败后立即重试，最多重试 N 次，全部失败后走降级或报错路径。这种模式实现简单，适用于那些失败概率较低且重试成本不高的场景。

```python
from typing import TypedDict, Annotated
import operator
import random
import time
from langgraph.graph import StateGraph, START, END

class RetryState(TypedDict):
    target_url: str
    payload: dict
    response_data: dict
    error_message: str
    attempt: Annotated[int, operator.add]
    max_attempts: int
    success: bool
    final_status: str
    retry_log: Annotated[list[str], operator.add]

def simulate_api_call(url: str, payload: dict) -> tuple[dict | None, str | None]:
    """模拟一个有概率失败的 API 调用"""
    random.seed(hash(url + str(payload.get("id", ""))) % 10000)
    rand = random.random()

    if rand < 0.25:
        return None, "503 Service Unavailable: 服务暂时过载"
    if rand < 0.35:
        return None, "429 Too Many Requests: 请求频率超限"
    if rand < 0.40:
        return None, "ConnectionError: 连接超时"

    return {
        "status": "success",
        "data": f"来自 {url} 的响应数据",
        "request_id": f"req-{random.randint(10000, 99999)}",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }, None

def call_api(state: RetryState) -> dict:
    url = state["target_url"]
    payload = state["payload"]
    attempt_num = state["attempt"] + 1

    response, error = simulate_api_call(url, payload)

    log_prefix = f"[尝试 {attempt_num}/{state['max_attempts']}]"

    if error:
        return {
            "error_message": error,
            "success": False,
            "attempt": 1,
            "retry_log": [f"{log_prefix} ❌ 失败: {error}"]
        }

    return {
        "response_data": response,
        "error_message": "",
        "success": True,
        "attempt": 1,
        "retry_log": [f"{log_prefix} ✅ 成功 | ID: {response['request_id']}"]
    }

def should_retry(state: RetryState) -> str:
    if state["success"]:
        return "success"
    if state["attempt"] >= state["max_attempts"]:
        return "exhausted"
    return "retry"

def handle_success(state: RetryState) -> dict:
    attempts = state["attempt"]
    data = state["response_data"]
    return {
        "final_status": f"✅ 成功 (第{attempts}次尝试)",
        "retry_log": [f"[完成] 数据已获取: {data['data'][:50]}"]
    }

def handle_exhaustion(state: RetryState) -> dict:
    attempts = state["attempt"]
    last_error = state["error_message"]
    return {
        "final_status": f"❌ 失败 ({attempts}次尝试均未成功)",
        "retry_log": [
            f"[放弃] 最后错误: {last_error}",
            "[建议] 请稍后重试或联系技术支持"
        ]
    }

retry_graph = StateGraph(RetryState)
retry_graph.add_node("call_api", call_api)
retry_graph.add_node("handle_success", handle_success)
retry_graph.add_node("handle_exhaustion", handle_exhaustion)

retry_graph.add_edge(START, "call_api")
retry_graph.add_conditional_edges("call_api", should_retry, {
    "retry": "call_api",
    "success": "handle_success",
    "exhausted": "handle_exhaustion"
})
retry_graph.add_edge("handle_success", END)
retry_graph.add_edge("handle_exhaustion", END)

app = retry_graph.compile()

print("=" * 60)
print("简单重试模式演示")
print("=" * 60)

result = app.invoke({
    "target_url": "https://api.example.com/v2/data",
    "payload": {"id": "test-001", "action": "query"},
    "response_data": {},
    "error_message": "",
    "attempt": 0,
    "max_attempts": 5,
    "success": False,
    "final_status": "",
    "retry_log": []
})

print(f"\n最终状态: {result['final_status']}")
for entry in result["retry_log"]:
    print(f"  {entry}")
```

这段程序描述了简单重试的工作原理。`call_api` 节点模拟了一个可能失败的 API 调用（25% 概率 503、10% 概率 429、5% 概率超时），每次调用后 `should_retry` 条件函数检查是否成功、是否达到最大尝试次数，然后决定下一步是再试一次（回到 `call_api`）、成功结束还是失败退出。由于 `attempt` 字段使用了 `Annotated[int, operator.add]`，每经过一次 `call_api` 节点计数器自动加 1。

## 指数退避重试：避免雪崩效应

简单重试有一个问题：如果下游服务已经因为负载过高而开始返回错误，连续快速的重试只会让它更加不堪重负——这就是所谓的"重试风暴"或"雪崩效应"。指数退避（Exponential Backoff）策略通过在每次重试之间增加等待时间来缓解这个问题：第一次重试等 1 秒，第二次等 2 秒，第三次等 4 秒，以此类推。这样既给了下游服务恢复的时间，也避免了短时间内大量重试请求的冲击。

```python
import time
import asyncio

class BackoffRetryState(TypedDict):
    api_endpoint: str
    request_params: dict
    response: dict
    last_error: str
    attempt: Annotated[int, operator.add]
    max_attempts: int
    base_delay: float
    max_delay: float
    success: bool
    status: str
    log: Annotated[list[str], operator.add]

def call_with_backoff(state: BackoffRetryState) -> dict:
    endpoint = state["api_endpoint"]
    params = state["request_params"]
    attempt = state["attempt"] + 1
    base_delay = state["base_delay"]
    max_delay = state["max_delay"]

    if attempt > 1:
        delay = min(base_delay * (2 ** (attempt - 2)), max_delay)
        time.sleep(delay)

    response, error = simulate_api_call(endpoint, params)

    log_parts = [f"[第{attempt}次] 调用 {endpoint}"]

    if attempt > 1:
        delay_actual = min(base_delay * (2 ** (attempt - 2)), max_delay)
        log_parts.append(f"(退避 {delay_actual:.1f}s)")

    if error:
        log_parts.append(f"→ ❌ {error}")
        return {
            "last_error": error,
            "success": False,
            "attempt": 1,
            "log": [" ".join(log_parts)]
        }

    log_parts.append(f"→ ✅ 成功")
    return {
        "response": response,
        "last_error": "",
        "success": True,
        "attempt": 1,
        "log": [" ".join(log_parts)]
    }

backoff_graph = StateGraph(BackoffRetryState)
backoff_graph.add_node("call_with_backoff", call_with_backoff)
backoff_graph.add_node("success_handler", lambda s: {
    "status": f"✅ 完成 (第{s['attempt']}次)", "log": ["[完成]"]
})
backoff_graph.add_node("fail_handler", lambda s: {
    "status": f"❌ 放弃 (已尝试{s['attempt']}次)",
    "log": [f"[放弃] 错误: {s['last_error']}"]
})

backoff_graph.add_edge(START, "call_with_backoff")
backoff_graph.add_conditional_edges("call_with_backoff",
    lambda s: "success" if s["success"] else ("exhausted" if s["attempt"] >= s["max_attempts"] else "retry"),
    {"retry": "call_with_backoff", "success": "success_handler", "exhausted": "fail_handler"}
)
backoff_graph.add_edge("success_handler", END)
backoff_graph.add_edge("fail_handler", END)

app = backoff_graph.compile()

start = time.time()
result = app.invoke({
    "api_endpoint": "https://api.heavy-load.example.com/data",
    "request_params": {"limit": 100},
    "response": {}, "last_error": "", "attempt": 0,
    "max_attempts": 6, "base_delay": 0.5, "max_delay": 8.0,
    "success": False, "status": "", "log": []
})
elapsed = time.time() - start

print(f"\n最终状态: {result['status']}")
print(f"总耗时: {elapsed:.2f}s")
for entry in result["log"]:
    print(f"  {entry}")
```

注意 `call_with_backoff` 函数中的关键逻辑：只有当 `attempt > 1` 时才计算并执行退避等待（第一次调用不需要等待）。退避时间使用公式 `min(base_delay * 2^(attempt-2), max_delay)` 计算——这意味着第二次尝试前等 0.5s（即 `0.5 * 2^0`）、第三次前等 1s（`0.5 * 2^1`）、第四次前等 2s、第五次前等 4s、第六次前等 8s（被 max_delay=8.0 封顶）。这种指数增长的等待时间既能给下游足够的恢复空间，又不会让用户等太久（因为有 max_delay 上限）。

## 断路器模式：防止级联失败

比重试更高级的容错模式是断路器（Circuit Breaker）模式。它的核心思想是：如果某个服务的失败频率超过了阈值，就暂时"熔断"对它的所有调用，直接返回错误或降级结果，而不是继续重试。经过一段冷却期后再尝试恢复调用，如果成功了就关闭断路器恢复正常，如果失败了就继续保持熔断状态。这就像电路中的保险丝——当电流过大时自动跳闸保护整个电路不被烧毁。

```python
from typing import TypedDict, Annotated
import operator
import time
from langgraph.graph import StateGraph, START, END

class CircuitBreakerState(TypedDict):
    service_name: str
    request_payload: dict
    failure_count: int
    success_count: int
    threshold: int
    cooldown_seconds: float
    circuit_state: str
    last_failure_time: float
    response_data: dict
    fallback_data: dict
    final_result: str
    log: Annotated[list[str], operator.add]

def check_circuit(state: CircuitBreakerState) -> str:
    circuit = state["circuit_state"]
    now = time.time()

    if circuit == "closed":
        return "try_call"
    elif circuit == "open":
        elapsed = now - state["last_failure_time"]
        if elapsed >= state["cooldown_seconds"]:
            return "half_open"
        return "use_fallback"
    else:
        return "try_call"

def make_call(state: CircuitBreakerState) -> dict:
    service = state["service_name"]
    response, error = simulate_api_call(service, state["request_payload"])

    if error:
        new_failures = state["failure_count"] + 1
        threshold = state["threshold"]

        if new_failures >= threshold:
            new_circuit = "open"
            log_msg = (
                f"[调用] ❌ {service} 失败\n"
                f"       失败数: {new_failures}/{threshold}\n"
                f"       → 断路器打开! 冷却 {state['cooldown_seconds']}s"
            )
        else:
            new_circuit = "closed"
            log_msg = (
                f"[调用] ❌ {service} 失败\n"
                f"       失败数: {new_failures}/{threshold}"
            )

        return {
            "failure_count": new_failures,
            "circuit_state": new_circuit,
            "last_failure_time": time.time(),
            "log": [log_msg]
        }

    new_successes = state["success_count"] + 1
    return {
        "response_data": response,
        "success_count": new_successes,
        "failure_count": 0,
        "circuit_state": "closed",
        "log": [f"[调用] ✅ {service} 成功 (连续{new_successes}次成功)"]
    }

def provide_fallback(state: CircuitBreakerState) -> dict:
    service = state["service_name"]
    fallback = {
        "status": "fallback",
        "source": "cached/degraded_response",
        "message": f"{service} 当前不可用，返回降级数据",
        "timestamp": time.strftime("%H:%M:%S"),
        "data": "这是降级后的默认数据"
    }
    return {
        "fallback_data": fallback,
        "final_result": f"⚠️ 使用降级响应 ({service} 断路中)",
        "log": [f"[降级] 返回缓存/降级数据 (原因: {service} 不可用)"]
    }

def probe_service(state: CircuitBreakerState) -> dict:
    service = state["service_name"]
    response, error = simulate_api_call(service, state["request_payload"])

    if error:
        return {
            "circuit_state": "open",
            "last_failure_time": time.time(),
            "log": [f"[探测] ❌ {service} 仍未恢复，保持断路"]
        }

    return {
        "response_data": response,
        "circuit_state": "closed",
        "failure_count": 0,
        "log": [f"[探测] ✅ {service} 已恢复! 断路器关闭"]
    }

def finalize_success(state: CircuitBreakerState) -> dict:
    return {
        "final_result": f"✅ 调用成功 ({state['service_name']})",
        "log": ["[完成] 正常响应"]
    }

cb_graph = StateGraph(CircuitBreakerState)
cb_graph.add_node("check_circuit", lambda s: {})
cb_graph.add_node("make_call", make_call)
cb_graph.add_node("fallback", provide_fallback)
cb_graph.add_node("probe", probe_service)
cb_graph.add_node("success", finalize_success)

cb_graph.add_edge(START, "check_circuit")
cb_graph.add_conditional_edges("check_circuit", check_circuit, {
    "try_call": "make_call",
    "half_open": "probe",
    "use_fallback": "fallback"
})
cb_graph.add_conditional_edges("make_call",
    lambda s: "success" if s.get("response_data") else "check_circuit",
    {"success": "success", "check_circuit": "check_circuit"}
)
cb_graph.add_conditional_edges("probe",
    lambda s: "success" if s.get("response_data") else "fallback",
    {"success": "success", "fallback": "fallback"}
)
cb_graph.add_edge("fallback", END)
cb_graph.add_edge("success", END)

app = cb_graph.compile()

result = app.invoke({
    "service_name": "payment-service",
    "request_payload": {"action": "charge", "amount": 99.9},
    "failure_count": 0, "success_count": 0,
    "threshold": 3, "cooldown_seconds": 5.0,
    "circuit_state": "closed",
    "last_failure_time": 0,
    "response_data": {}, "fallback_data": {},
    "final_result": "", "log": []
})

print(f"\n最终结果: {result['final_result']}")
for entry in result["log"]:
    print(entry)
```

这个断路器实现展示了三种状态的转换：

- **Closed（闭合）**：正常状态，允许所有调用通过。连续失败次数达到阈值时转换为 Open。
- **Open（打开）**：熔断状态，所有调用直接走降级路径，不再尝试调用实际服务。冷却期结束后转换为 Half-Open 进行探测。
- **Half-Open（半开）**：探测状态，允许一个探测调用通过以检测服务是否恢复。如果探测成功则转为 Closed，失败则重新转为 Open 并重新计时冷却期。

三种状态之间的转换关系可以用下面的状态图表示：

```
         连续失败 ≥ threshold
   [Closed] ──────────────────► [Open]
       ▲                           │
       │                           │ 冷却期结束
       │      探测成功             ▼
       └────────────────── [Half-Open]
                               │
                          探测失败
                               │
                               ▼
                            [Open]
```

## 多依赖场景下的容错设计

在实际系统中，你的图往往需要调用多个外部服务，每个服务都可能失败。这时候就需要考虑更复杂的容错策略——比如某个非关键服务失败了要不要影响整体流程？多个服务都失败了怎么处理？如何区分关键依赖和非关键依赖？

```python
class MultiServiceState(TypedDict):
    primary_service_resp: dict
    secondary_service_resp: dict
    cache_service_resp: dict
    primary_ok: bool
    secondary_ok: bool
    cache_ok: bool
    combined_result: dict
    status: str
    log: Annotated[list[str], operator.add]

def call_primary(state: MultiServiceState) -> dict:
    resp, err = simulate_api_call("primary-api", {"type": "main"})
    ok = resp is not None
    return {
        "primary_service_resp": resp or {},
        "primary_ok": ok,
        "log": [f"[主服务] {'✅' if ok else '❌'} {'成功' if ok else err}"]
    }

def call_secondary(state: MultiServiceState) -> dict:
    resp, err = simulate_api_call("secondary-api", {"type": "aux"})
    ok = resp is not None
    return {
        "secondary_service_resp": resp or {},
        "secondary_ok": ok,
        "log": [f"[辅助服务] {'✅' if ok else '❌'} {'成功' if ok else err}"]
    }

def call_cache(state: MultiServiceState) -> dict:
    resp, err = simulate_api_call("cache-service", {"type": "lookup"})
    ok = resp is not None
    return {
        "cache_service_resp": resp or {},
        "cache_ok": ok,
        "log": [f"[缓存服务] {'✅' if ok else '❌'} {'成功' if ok else err}"]
    }

def merge_results(state: MultiServiceState) -> dict:
    p_ok = state["primary_ok"]
    s_ok = state["secondary_ok"]
    c_ok = state["cache_ok"]

    combined = {}
    status_msgs = []

    if p_ok:
        combined.update(state["primary_service_resp"])
        status_msgs.append("主服务数据已整合")
    else:
        status_msgs.append("⚠️ 主服务不可用")

    if s_ok:
        combined["auxiliary"] = state["secondary_service_resp"]
        status_msgs.append("辅助服务数据已补充")
    else:
        status_msgs.append("ℹ️ 辅助服务缺失（非关键）")

    if c_ok and not p_ok:
        combined["from_cache"] = state["cache_service_resp"]
        status_msgs.append("使用缓存数据作为替代")

    if p_ok or c_ok:
        final_status = "✅ 部分或完全成功"
    else:
        final_status = "❌ 所有数据源均不可用"

    return {
        "combined_result": combined,
        "status": final_status,
        "log": [f"[汇总] {'; '.join(status_msgs)}"]
    }

ms_graph = StateGraph(MultiServiceState)
ms_graph.add_node("primary", call_primary)
ms_graph.add_node("secondary", call_secondary)
ms_graph.add_node("cache", call_cache)
ms_graph.add_node("merge", merge_results)

ms_graph.add_edge(START, "primary")
ms_graph.add_edge(START, "secondary")  # 并行调用
ms_graph.add_edge(START, "cache")
ms_graph.add_edge("primary", "merge")
ms_graph.add_edge("secondary", "merge")
ms_graph.add_edge("cache", "merge")
ms_graph.add_edge("merge", END)

app = ms_graph.compile()

result = app.invoke({
    "primary_service_resp": {}, "secondary_service_resp": {}, "cache_service_resp": {},
    "primary_ok": False, "secondary_ok": False, "cache_ok": False,
    "combined_result": {}, "status": "", "log": []
})

print(f"\n{result['status']}")
for entry in result["log"]:
    print(f"  {entry}")
```

这个多服务调用的例子展示了几种不同的容错策略组合：主服务（primary）是关键依赖，它的失败会显著影响结果质量；辅助服务（secondary）是非关键的，失败了只是缺少一些补充信息但不影响核心功能；缓存服务（cache）作为后备方案——只有在主服务失败时才会使用缓存数据。`merge_results` 节点根据各服务的可用情况智能地合并数据，尽可能多地利用可用的信息源来构建完整的结果。

## 容错设计的最佳实践总结

在 LangGraph 中设计和实现容错机制时，有几个经验法则值得遵循：

**第一，始终设置最大重试次数和最大超时时间**。没有任何重试应该是无限次的，也没有任何等待应该是无限长的。这两个硬性约束能确保即使出现异常情况，图的执行也会在有限的时间内终止。

**第二，区分临时性错误和永久性错误**。对于临时性错误（网络超时、503 服务不可用、429 限流），重试是有意义的；但对于永久性错误（401 认证失败、404 资源不存在、400 参数错误），重试是没有意义的，应该立即失败并走错误处理路径。在路由函数中加入错误类型的判断可以避免无意义的重试。

**第三，为每个外部依赖定义 SLA（服务水平协议）**。明确每个外部服务的超时时间、重试策略、降级方案，并在图中体现这些约定。不要让图的容错行为依赖于隐式的假设。

**第四，记录详细的故障信息用于事后分析**。每次失败都应该记录完整的上下文——什么时间、调用了哪个服务、传了什么参数、收到了什么错误、当时的状态是什么。这些信息对于排查生产问题和持续优化容错策略至关重要。

**第五，在开发阶段故意注入故障来测试容错机制**。不要等到真正出问题了才发现容错代码有 bug。可以在测试环境中模拟各种故障场景（服务宕机、网络延迟、返回异常格式等），验证你的容错机制能否正确处理每种情况。
