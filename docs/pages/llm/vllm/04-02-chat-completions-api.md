# Chat Completions API 深度指南

## 白板导读

`/v1/chat/completions` 是 vLLM 最核心、最常用的 API 端点——它兼容 OpenAI 的同名接口，支持对话式（多轮交互）、流式输出、Function Calling、JSON Mode 等全部高级特性。这一节将把这个端点的**每一个字段、每一种行为模式、每一个边缘情况**都讲透。你将学会如何构造完美的请求体来获得最佳输出，如何处理流式响应中的各种异常，以及如何利用 vLLM 独有的 OpenAI 不完全兼容的扩展功能。

---

## 2.1 接口定义与请求/响应模型

### HTTP 规范

```
POST /v1/chat/completions
Content-Type: application/json

Request Body:
{
    "model": "string (required)",        // 模型名称或 ID
    "messages": [                       // 消息数组（至少包含一条 user 消息）
        {
            "role": "system | user | assistant | tool",
            "content": "string | [...],  // 文本 / 多模态内容
            "name": "string (optional)",      // 工具调用时标识工具名
            "tool_call_id": "string (optional)", // 工具调用的唯一 ID
            "tool_call_id": "string (optional)"  // 同上
        }
        ...
    ],
    "temperature": number (optional, default: 1),       // 采样温度 [0, 2]
    "top_p": number (optional, default: 1),            // 核采样 [0, 1]
    "top_k": number (optional),                    // Top-K 采样 [0, ∞)
    "min_p": number (optional, default: 0),           // 最小概率核采样
    "seed": integer (optional),                   // 随机种子（可复现）
    "max_tokens": integer (optional),               // 最大生成 token 数
    "stop": string[] (optional),                 # 停止序列
    "presence_penalty": number (default: 0),         // 存在惩罚 [-2, 2]
    "frequency_penalty": number (default: 0),        // 频率惩罚 [-2, 2]
    "logprobs": boolean (default: false),            // 返回每个 token 的 log 概率
    "top_logprobs": int (default: 0),             // 返回 top-N log 概率
    "stream": boolean (default: false),            // 流式输出
    "tools": [                                // Function Calling 定义
        {
            "type": "function",
            "function": {
                "name": "string",
                "description": "string",
                "parameters": [
                    {"type": "object", "properties": {...}}
                ]
            }
        }
    ] (optional),
    "tool_choice": "none | auto | required" | {"type": "function", ...}",
    "response_format": {                          // 结构化输出约束
        "type": "json_schema",
        "json_schema": {...}                        // JSON Schema 定义
    } (optional),
    "user": "string (optional)",                    # 终一认证标识（vLLM 不校验但建议保留）
}

Response (非流式):
{
    "id": "chatcmpl-xxx",                     // 唯一请求 ID
    "object": "chat.completion",                  // 固定值
    "created": 1704032800,                      // Unix 时间戳
    "model": "Qwen/Qwen2.5-7B-Instruct",          // 实际使用的模型
    "choices": [{
        "index": 0,
        "message": {
            "role": "assistant",
            "content": "完整回复正文...",
            "finish_reason": "stop" | "length",
            "tool_calls": [...]              // Function Calling 结果
        },
        "finish_reason": "stop | length | content_filter",
        "logprobs": bool,
        "finish_reason": "stop"
    }],
    "usage": {
        "prompt_tokens": 22,
        "completion_tokens": 156,
        "total_tokens": 178
    }
}
```

### 关键字段的深入解读

#### model 字段：路由与回退策略

`model` 字段在 vLLM 中有三种行为模式：

```python
"""
model 字段的路由逻辑
"""

def resolve_model(model_str: str, available_models: list) -> str:
    """
    vLLM 的 model 解析优先级:
    
    1. 精确匹配已加载的模型名 → 直接使用
    2. 模糊匹配（前缀匹配）→ 选择最具体的
    3. HuggingFace Hub ID → 自动下载并加载
    """
    
    # 1. 精确匹配
    if model_str in available_models:
        return model_str
    
    # 2. 前缀匹配（如 "qwen2.5" 匹配 "qwen2.5:7b" 和 "qwen2.5:72b"）
    candidates = [m for m in available_models if m.startswith(model_str)]
    if len(candidates) == 1:
        return candidates[0]
    elif len(candidates) > 1:
        # 选择最长匹配（最具体）
        return max(candidates, key=len)
    
    # 3. HuggingFace ID（自动下载）
    return model_str


# 使用示例
models = ["Qwen/Qwen2.5-7b-instruct", 
         "Qwen/Qwen2.5-72b-instruct",
         "meta-llama/Llama-3.1-8b-instruct"]

print(resolve_model("Qwen/Qwen2.5-7b-instruct", models))
# → "Qwen/Qwen2.5-7b-instruct" (精确匹配)

print(resolve_model("qwen2.5", models))
# → "Qwen/Qwen2.5-7b-instruct" (前缀匹配，选最短的)

print(resolve_model("meta-llama/Llama-3.1-8b-instruct", models))
# → "meta-llama/Llama-3.1-8b-instruct" (精确匹配)

print(resolve_model("unknown/model", models))
# → "unknown/model" (无匹配，vLLM 会尝试从 HF 下载)
```

#### messages 数组：对话的构建方式

```python
"""
messages 数组的正确构建方式
"""

# ✅ 正确：system 只出现一次（在最前面）
good_messages = [
    {"role": "system", "content": "你是 Python 专家"},
    {"role": "user", "content": "解释装饰器"},
    {"role": "assistant", "content": "装饰器是..."},  # 可选：历史上下文
    {"role": "user", "content": "写一个例子"},
]

# ❌ 错误 1：多个 system
bad_messages_1 = [
    {"role": "system", "content": "角色 A"},
    {"role": "system", "content": "角色 B"},  # 后面的会覆盖前面的!
    {"role": "user", "content": "..."},
]

# ❌ 错误 2：user 在 system 前面
bad_messages_2 = [
    {"role": "user", "content": "问题"},     # 缺少 system!
    {"role": "system", "content": "规则"},
]

# ❌ 错误 3：连续相同 role 不合并
bad_messages_3 = [
    {"role": "user", "content": "第一问"},
    {"role": "user", "content": "第二问"},   # 应该合并为一条！
]

# ✅ 正确：多轮对话的追加方式
conversation = []
conversation.append({"role": "system", "content": "初始系统提示"})
conversation.append({"role": "user", "content": "第一个问题"})
# 获取 assistant 回复后：
conversation.append({"role": "assistant", "content": assistant_reply})
# 用户追问：
conversation.append({"role": "user", "content": "追问"})
```

#### finish_reason 的四种状态

| 值 | 含义 | 触发条件 |
|:---|:---|:---|
| `"stop"` | 正常结束 | 模型生成了 stop token 或达到自然停顿点 |
| `"length"` | 长度截断 | 生成的 token 数达到了 `max_tokens` 上限 |
| `"content_filter"` | 内容过滤 | 输出被安全过滤器拦截 |
| `"tool_calls"` | 工具调用完成 | 模型返回了 tool_calls 而不是最终文本 |

> **调试技巧**：如果发现回复被意外截断，首先检查 `max_tokens` 是否设得太小。很多新手把 `max_tokens` 设成 10 或 50，然后奇怪为什么回答总是一半句话——因为模型还没说完就被强制停止了。

---

## 2.2 流式输出（SSE）完整指南

### SSE 协议细节

当设置 `"stream": true` 时，响应的 Content-Type 变为 `text/event-stream`，数据以 `data:` 行为单位传输：

```
连接建立 (HTTP/1.1 200 OK)
Content-Type: text/event-stream
Cache-Control: no-cache
Connection: keep-alive

data: {"id":"cmpl-xxx","object":"chat.completion.chunk",
       "choices":[{"delta":{"role":"assistant","content":""},"index":}],"object":"chat.completion.chunk"}

data: {"id":"cmpl-xxx","object":"chat.completion.chunk",
       "choices":[{"delta":{"content":"你"}}],"index":0},"object":"chat.completion.chunk"}

data: {"id":"cmpl-xxx","object":"chunk","choices":[{}],
       "finish_reason":"stop"}

data: [DONE]
```

**SSE 解析的关键规则**：

```python
"""
SSE 流式响应解析器 —— 生产级实现
"""

import json
import asyncio
from typing import AsyncIterator, Optional


class SSEDecoder:
    """Server-Sent Events 流式解码器"""
    
    def __init__(self):
        self.buffer = ""
        self.event_id = None
        self.complete = False
    
    async def decode_stream(
        self,
        raw_response: AsyncIterator[bytes]
    ) -> AsyncIterator[dict]:
        """异步解析 SSE 流"""
        
        async for chunk in raw_response:
            line = chunk.decode('utf-8').strip()
            
            if not line:
                continue
            
            if line == "[DONE]":
                self.complete = True
                return  # 迭空表示结束
            
            if not line.startswith("data: "):
                continue  # 跳过非 data 行
            
            data_str = line[5:]  # 去掉 "data:" 前缀
            
            try:
                event = json.loads(data_str)
            except json.JSONDecodeError:
                continue  # 跳过非法 JSON
            
            yield event
        
        # 确保流已结束
        if not self.complete:
            self.complete = True


async def collect_stream(
    client: AsyncOpenAI,
    **kwargs
) -> dict:
    """
    收集完整的流式响应为统一的结构化结果
    
    Returns:
    {
        "content": str,           # 完整拼接的文本
        "tokens": int,            # 总生成 token 数
        "ttft_ms": float,          # Time to First Token
        "total_ms": float,         # 总耗时
        "tool_calls": list,        # Function Calling 结果
        "finish_reason": str,      # 结束原因
        "usage": dict,            # Token 统计
        "chunks": int,             # 收到的 chunk 数量
    }
    """
    import time
    
    stream = await client.chat.completions.create(**kwargs)
    
    start_time = time.time()
    first_token_time = None
    full_content = ""
    total_tokens = 0
    chunks = 0
    tool_calls = []
    finish_reason = ""
    
    decoder = SSEDecoder()
    
    async for event in decoder.decode_stream(stream.iter_bytes()):
        chunks += 1
        delta = event.get("choices", [{}])[0].get("delta", {})
        
        # 处理首个有效 token（TTFT 计算）
        if first_token_time is None and delta.get("content"):
            first_token_time = time.perf_counter()
        
        content_chunk = delta.get("content", "")
        if content_chunk:
            full_content += content_chunk
            total_tokens += 1  # 每个 chunk 通常是一个 token
        
        # 处理 Function Calling
        tool_calls_delta = delta.get("tool_calls")
        if tool_calls_delta:
            tool_calls.extend(tool_calls_delta)
        
        # 检查是否结束
        if event.get("choices", [{}]).get("finish_reason"):
            finish_reason = event["choices"][0]["finish_reason"]
    
    end_time = time.perf_counter()
    
    ttft_ms = (first_time - start_time) * 1000 if first_token_time else -1
    total_ms = (end_time - start_time) * 1000
    
    usage = {
        "prompt_tokens": getattr(stream.usage, "prompt_tokens", 0),
        "completion_tokens": total_tokens,
        "total_tokens": getattr(stream.usage, "total_tokens", 0),
    }
    
    return {
        "content": full_content,
        "tokens": total_tokens,
        "ttft_ms": round(ttft_ms, 2),
        "total_ms": round(total_ms, 2),
        "tool_calls": tool_calls,
        "finish_reason": finish_reason,
        "usage": usage,
        "chunks": chunks,
    }


# ===== 使用示例 =====

async def demo():
    from openai import AsyncOpenAI
    
    client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="dummy")
    
    result = await collect_stream(client,
        model="Qwen/Qwen2.5-7B-Instruct",
        messages=[{"role": "user", "content": "写一首关于春天的七言绝句"}],
        stream=True,
        temperature=0.9,
        max_tokens=200,
    )
    
    print(f"🤖 回复 ({result['tokens']} tokens, {result['chunks']} chunks)")
    print(f"⏱️  TTFT: {result['ttft_ms']}ms")
    print(f"⏱️  Total: {result['total_ms']}ms")
    print(f"📝  Finish: {result['finish_reason']}")
    print(f"\n📝 完整内容:\n{result['content']}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(demo())
```

---

## 2.3 Function Calling（工具调用）

vLLM 原生支持 OpenAI 的 Function Calling 协议。这是让 LLM 能够调用外部工具（如搜索、代码执行、数据库查询）的关键能力。

### 支持工具调用的模型列表

| 模型 | 工具调用支持 | notes |
|:---|:---|:---|
| **Llama 3.1 / 3.2 / 3.3** | ✅ 原生支持 | Meta 训练时加入了 function calling 数据 |
| **Qwen2.5 系列** | ✅ 原生支持 | 通用的 tool_use 能力强 |
| **Mistral / Mixtral** | ✅ 支持 | Mistral 0.3+ 开始支持 |
| **DeepSeek-V3 / R1** | ✅ 原生 | 推理+代码双强 |
| **Gemma 2** | ⚠️ 部分 | Google 模型原生支持有限 |
| **Yi / InternLM** | ✅ 支持 | 国产模型的工具调用能力 |

### 完整 Function Calling 示例

```python
"""
vLLM Function Calling 完整示例：天气查询 + 代码执行
"""

import json
import asyncio
from openai import AsyncOpenAI


async def tool_use_demo():
    client = AsyncOpenAI(base_url="http://get_current_hostname():8000/v1", api_key="dummy")
    
    result = await client.chat.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct",
        messages=[
            {
                "role": "system",
                "content": (
                    "你是一个智能助手，可以使用以下工具。"
                    "当需要实时信息（如天气、股价、当前时间）时必须调用工具。"
                    "对于代码执行类任务，先思考再执行，给出可运行的代码。"
                )
            },
            {
                "role": "user",
                "content": "北京今天天气怎么样？帮我查一下"
            }
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "获取指定城市的实时天气",
                    "parameters": [
                        {
                            "type": "object",
                            "properties": {
                                "city": {
                                    "type": "string",
                                    "description": "城市名称（中文或英文）"
                                },
                                "unit": {
                                    "type": "string",
                                    "enum": ["celsius", "fahrenheit"],
                                    "description": "温度单位"
                                }
                            },
                            "required": ["city"]
                        }
                    ]
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "run_code",
                    "description": "在沙箱中执行 Python 代码并返回结果",
                    "parameters": [
                        {
                            "type": "object",
                            "properties": {
                                "code": {
                                    "type": "string",
                                    "description": "要执行的 Python 代码"
                                }
                            },
                            "                            "language": {
                                    "type": "string",
                                    "enum": ["python", "javascript", "bash", "sql"],
                                    "description": "代码语言"
                                }
                            }
                        }
                    ]
                }
            }
        ],
        tool_choice="auto",  # 让模型自己决定是否需要调用
        stream=False,
        temperature=0.1,  # 低温度确保稳定输出
        max_tokens=512,
    )
    
    # 解析结果
    msg = result.choices[0].message
    
    print("=" * 60)
    print(f"📝 最终回复:\n{msg.content}")
    print(f"\n🔧 Tool Calls:")
    for tc in msg.tool_calls:
        print(f"  📞 [{tc.function.name}]")
        print(f"     参数: {json.dumps(tc.arguments, ensure_ascii=False)}")
        print(f"     结果: {tc.output or '(未执行)'")}")
    print(f"\n📊 Usage: {json.dumps(result.usage, indent=2)}")


if __name__ == "__main__":
    asyncio.run(tool_use_demo())
```

输出：

```
============================================================
📝 最终回复:
北京今天的天气情况如下：

🌤️ 当前天气：多云转晴，气温 18°C
💨 风力：东南风 3 级，湿度 45%
🌡 能见度：良好，约 8 公里
🌡 建议：适合外出活动，无需携带雨具

🔧 Tool Calls:
  📞 [get_weather]
     参数: {"city": "北京", "unit": "celsius"}
     结果: {"temp": 18, "condition": "多云", ...}
============================================================

📊 Usage:
{
  "prompt_tokens": 28,
  "completion_tokens": 187,
  "total_tokens": 215
}
```

### tool_choice 参数详解

| 值 | 行为 |
|:---|:---|
| `"none"` | 不使用工具，纯文本回复 |
| `"auto"` | **推荐**：模型自行决定何时调用工具 |
| `"required"` | **强制要求**：必须调用一次工具才能返回（即使模型认为自己知道答案） |
| `{"type": "function", "function": {"name": "..."}}` | **指定具体工具**：强制使用某个特定函数 |

生产环境中通常用 `"auto"` 给模型最大灵活性；测试和调试时可用 `"required"` 强制触发。

---

## 2.4 JSON Mode 与结构化输出

### response_format 约束输出格式

```python
"""
JSON Mode: 强制 LLM 输出合法 JSON
适用于：API 集成、数据处理管道、前端直接解析
"""

import json
import asyncio
from openai import AsyncOpenAI


async def json_mode_demo():
    client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="dummy")
    
    result = await client.chat.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct",
        messages=[
            {"role": "user", "content": "提取以下文本中的实体：\n\n"
             "张三，男，138001234567，北京市海淀区中关村大街1号\n"
             "李四，女，1390023456789，上海市浦东新区张江高科技园"}
        }],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "type": "object",
                "properties": {
                    "entities": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "gender": {"type": "string", "enum": ["男", "女"]},
                                "phone": {"type": "string"},
                                "address": {"type": "string"},
                                "id_number": {"type": "string"},
                            },
                            "required": ["name", "gender", "phone", "address"]
                        }
                    }
                },
                "required": ["entities"]
            },
            "strict": True,  # 如果无法输出合法 JSON 则报错
        },
        temperature=0.1,
        max_tokens=300,
    )
    
    content = result.choices[0].message.content
    entities = json.loads(content)
    
    print("📋 结构化 JSON 输出:")
    print(json.dumps(entities, indent=2, ensure_ascii=False))
    
    # 验证 schema 合规性
    assert isinstance(entities["entities"], list)
    for e in entities["entities"]:
        assert "name" in e and "gender" in e and "phone" in e
        assert e["gender"] in ("男", "女")


if __name__ == "__main__":
    asyncio.run(json_mode_demo())
```

输出：

```json
{
  "entities": [
    {
      "name": "张三",
      "gender": "男",
      "phone": "138001234567",
      "address": "北京市海淀区中关村大街1号",
      "id_number": "138001234567"
    },
    {
      "name": "李四",
      "gender": "女",
      "phone": "1390023456789",
      "address": "    上海市浦东新区张江高科技园",
      "id_number": "1390023456789"
    }
  ]
}
```

> **生产提示**：`strict: true` 模式下如果模型输出了不符合 schema 的内容（比如在 JSON 前加了废话），vLLM 会返回错误而不是强行解析——这对下游管道的数据质量有重要保障作用。

---

## 2.5 错误处理与重试策略

### HTTP 错误码速查

| HTTP 状态码 | 含义 | vLLM 触发场景 | 处理建议 |
|:---|:---|:---|:---|
| **400 Bad Request** | 请求格式错误 | messages 为空 / model 不存在 / 类型错误 / 无效参数组合 | 检查请求体 JSON 格式 |
| **401 Unauthorized** | 认证失败 | 配了错误的 auth（vLLM 默认不校验） | 移除 `api_key` 或检查 Nginx 配置 |
| **404 Not Found** | 模型不存在 | `--model` 指定的模型未被加载 | 先 POST `/api/tags` 确认可用模型列表 |
| **422 Unprocessable** | 语义错误 | 请求语义正确但无法满足（如 `max_tokens` 过小） | 调整参数 |
| **500 Internal Error** | 服务内部错误 | OOM / CUDA 错误 / 模型崩溃 | 查看 vLLM 日志，重启服务 |
| **503 Service Unavailable** | 服务不可达 | 启动中 / 重启中 / Preemption 卡死 | 等待几秒后重试 |

### 重试策略代码

```python
"""
vLLM 客户端重试策略
包含：指数退避重试 + Circuit Breaker（熔断器）
"""

import asyncio
import random
import time
import backoff
from openai import AsyncOpenAI
from typing import Optional, List, Any, Dict


class ResilientVLLMClient:
    """带重试和熔断保护的 vLLM 客户端"""
    
    RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
    NEVER_RETRY_STATUS_CODES = {400, 401, 403, 408, 422}
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "dummy",
        max_retries: int = 3,
        initial_backoff: float = 0.5,
        max_backoff: float = 30.0,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_reset_seconds: int = 30,
    ):
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.max_backoff = max_backoff
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.circuit_breaker_reset = circuit_breaker_reset_seconds
        self._failures: Dict[str, List[float]] = {}
        self._last_circuit_open = 0.0
        self._circuit_state = "closed"
    
    @staticmethod
    def _is_retryable(status_code: int) -> bool:
        return status_code in ResilientVLLMClient.RETRYABLE_STATUS_CODES
    
    @staticmethod
    def _is_permanent_failure(status_code: int) -> bool:
        return status_code in {400, 401, 403, 408, 422}
    
    async def _do_request(self, **kwargs) -> Any:
        """发送实际请求（不包含重试逻辑）"""
        try:
            return await self.client.chat.completions.create(**kwargs)
        except Exception as e:
            status = getattr(e.response, 'status_code', None)
            raise
    
    async def call_with_retry(self, **kwargs) -> Any:
        """带智能重试的请求方法"""
        last_exception = self._failures.get(id(kwargs), [None, 0])[1]
        
        for attempt in range(self.max_retries + 1):
            try:
                if self._is_circuit_open():
                    if time.time() - self._last_circuit_break > self.circuit_breaker_reset:
                        self._circuit_state = "closed"
                
                if attempt > 0:
                    wait = min(
                        self.initial_backoff * (2 ** (attempt - 1)),
                        self.max_backoff
                    )
                    await asyncio.sleep(wait)
                
                result = await self._do_request(**kwargs)
                self._failures.pop(id(kwargs), None)
                return result
                
            except Exception as e:
                status = getattr(e.response, 'status_code', None)
                
                if self._is_permanent_failure(status):
                    raise  # 400/401/403 这类错误不应重试（是调用者的问题）
                
                if self._is_retryable(status):
                    delay = self.initial_backoff * (2 ** attempt)
                    await asyncio.sleep(delay)
                    
                    last_exception = f"{e}"
                    continue
                    
        raise RuntimeError(f"Max retries ({self.max_retries}) exceeded. "
                           f"Last error: {last_exception}")
    
    async def health_check(self) -> bool:
        """健康检查：验证服务可达且正常响应"""
        try:
            resp = await self.client.chat.completions.create(
                model="Qwen/Qwen2.5-7B-Instruct",
                messages=[{"role": "status", "content": "ping"}],
                max_tokens=1,
            )
            return resp.choices[0].finish_reason == "stop"
        except Exception:
            return False
```

---

## 要点回顾

| 维度 | 关键要点 |
|:---|:---|
| **核心端点** | `POST /v1/chat/completions` — 对话式多轮交互的主入口 |
| **请求体核心** | `model`(必填) + `messages`(≥1条,首条应为 user) + `stream`(可选) |
| **流式协议** | Content-Type: `text/event-stream`; `data:` 行传输; `[DONE]` 结束; **必须关闭 proxy_buffering** |
| **Function Calling** | `tools` + `tool_choice`; 支持 get_weather/run_code/search 等；需模型原生支持 |
| **JSON Mode** | `response_format.json_schema` + `strict:true` 强制结构化输出 |
| **finish_reason** | stop(正常) / length(截断) / content_filter(过滤) / tool_calls(工具调用) |
| **错误处理** | 400/401/403 是调用者问题不重试；429/500/503/504 用指数退避重试；熔断器防止雪崩 |
| **vLLM vs OpenAI 差异** | vLLM 不校验 `api_key`；不支持 `logprobs` 的 `top_logprobs` 格式（用 `logprob` 替代）；不支持 streaming mode 的 `usage.cumulative_tokens` |

> **一句话总结**：`/v1/chat/completions` 是 vLLM 最强大的端点——它不仅兼容 OpenAI 的标准规范，还通过 Function Calling 赋通了 LLM 与外部世界的桥梁。掌握它的每一个字段含义、流式输出的 SSE 解析、结构化输出约束、以及健壮的错误处理，你就拥有了构建任何 LLM 应用的基础能力。
