# 3.2 流式输出与 Token 统计

> **流式输出让你实时看到 AI 的"打字过程"，Token 统计让你精确控制 API 费用——这两个能力在生产环境中缺一不可。**

---

## 这一节在讲什么？

上一节我们学了 Chat Completions 的基本参数，这一节深入两个生产环境必备的能力：流式输出和 Token 统计。流式输出不只是"好看"——在长文本生成场景下，它能让用户更快看到结果，提升体验；Token 统计不只是"记账"——通过缓存命中统计，你能精确计算实际费用，优化缓存策略。这一节我们把这两个能力讲透。

---

## 流式输出的工作原理

非流式输出的工作方式是：用户发送请求 → 模型生成完整回答 → 一次性返回。如果模型需要生成 3000 token 的回答，用户可能要等 10-20 秒才能看到任何内容。

流式输出的工作方式是：用户发送请求 → 模型逐 token 生成 → 每个 token 生成后立即返回。用户几乎可以实时看到 AI 的"打字过程"。

```
非流式输出：

  用户发送 ────────────────────────────────── 等待 15 秒 ──── 完整回答
  [         用户看到空白，等待中...          ] [  突然出现一大段文字  ]

流式输出：

  用户发送 ── token1 ── token2 ── token3 ── ... ── tokenN ── [DONE]
  [  用户实时看到文字逐字出现，像打字机一样  ]
```

---

## 流式输出的代码实现

### Python

```python
stream = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "解释什么是递归，给出 3 个例子"}],
    stream=True
)

for chunk in stream:
    delta = chunk.choices[0].delta
    if delta.content:
        print(delta.content, end="", flush=True)
    if chunk.choices[0].finish_reason == "stop":
        print("\n[生成完成]")
```

### Node.js

```javascript
const stream = await client.chat.completions.create({
    model: "deepseek-chat",
    messages: [{ role: "user", content: "解释什么是递归" }],
    stream: true
});

for await (const chunk of stream) {
    const delta = chunk.choices[0].delta;
    if (delta.content) {
        process.stdout.write(delta.content);
    }
}
```

### cURL

```bash
curl https://api.deepseek.com/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-xxxxxxxx" \
  -d '{
    "model": "deepseek-chat",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

---

## 流式响应格式

流式响应使用 Server-Sent Events（SSE）格式，每个 chunk 是一个 JSON 对象：

```
data: {"id":"chatcmpl-xxx","choices":[{"index":0,"delta":{"content":"递"},"finish_reason":null}]}

data: {"id":"chatcmpl-xxx","choices":[{"index":0,"delta":{"content":"归"},"finish_reason":null}]}

data: {"id":"chatcmpl-xxx","choices":[{"index":0,"delta":{"content":"是"},"finish_reason":null}]}

...

data: {"id":"chatcmpl-xxx","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

关键点：
- 每个 chunk 的 `delta.content` 包含新生成的 token
- `finish_reason: null` 表示还在生成中
- `finish_reason: "stop"` 表示生成完成
- 最后一个事件是 `data: [DONE]`

---

## Token 统计

### 非流式输出的 Token 统计

非流式输出的 `usage` 字段直接包含完整的 token 统计：

```python
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "Hello!"}]
)

usage = response.usage
print(f"Input tokens: {usage.prompt_tokens}")
print(f"Output tokens: {usage.completion_tokens}")
print(f"Total tokens: {usage.total_tokens}")
print(f"Cache hit tokens: {usage.prompt_cache_hit_tokens}")
print(f"Cache miss tokens: {usage.prompt_cache_miss_tokens}")
```

### 流式输出的 Token 统计

默认情况下，流式输出的 `usage` 字段为 null——因为 token 统计需要等全部生成完才能计算。如果你需要在流式输出中获取 token 统计，设置 `stream_options`：

```python
stream = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True,
    stream_options={"include_usage": True}
)

for chunk in stream:
    if chunk.usage:
        print(f"Token 统计: input={chunk.usage.prompt_tokens}, "
              f"output={chunk.usage.completion_tokens}")
```

设置 `stream_options={"include_usage": True}` 后，最后一个 chunk 会包含完整的 token 统计。

---

## 缓存命中与成本计算

DeepSeek API 支持前缀缓存——如果你的请求前缀跟之前的请求相同，缓存命中的 token 价格降低 75%。

### 缓存命中的原理

```
第一次请求：
  messages = [
    {"role": "system", "content": "你是一个编程助手..."},  ← 100 token
    {"role": "user", "content": "写一个快速排序"}          ← 20 token
  ]
  → 全部按正常价格计费（120 token × $0.27/M）

第二次请求（同一会话的第二轮）：
  messages = [
    {"role": "system", "content": "你是一个编程助手..."},  ← 100 token（缓存命中！）
    {"role": "user", "content": "写一个快速排序"},          ← 20 token（缓存命中！）
    {"role": "assistant", "content": "好的，这是..."},      ← 200 token（缓存命中！）
    {"role": "user", "content": "能优化一下吗？"}           ← 15 token（新内容）
  ]
  → 前 320 token 缓存命中（320 × $0.07/M）
  → 后 15 token 按正常价格（15 × $0.27/M）
```

### 成本计算公式

```python
def calculate_cost(usage, model="deepseek-chat"):
    if model == "deepseek-chat":
        input_price = 0.27 / 1_000_000
        output_price = 1.10 / 1_000_000
        cache_price = 0.07 / 1_000_000
    elif model == "deepseek-reasoner":
        input_price = 0.55 / 1_000_000
        output_price = 2.19 / 1_000_000
        cache_price = 0.14 / 1_000_000

    cache_hit = usage.prompt_cache_hit_tokens
    cache_miss = usage.prompt_cache_miss_tokens
    output = usage.completion_tokens

    input_cost = cache_hit * cache_price + cache_miss * input_price
    output_cost = output * output_price

    return input_cost + output_cost
```

### 提高缓存命中率的策略

```
缓存命中的关键：保持请求前缀一致

  ✅ 好的做法：
  - 使用固定的 system prompt
  - 多轮对话中复用历史消息
  - 相同类型的请求使用相同的 prompt 模板

  ❌ 常见错误：
  - 每次请求都改 system prompt
  - 在消息列表开头插入新消息（改变前缀）
  - 用随机内容作为 system prompt
```

---

## 常见误区

**误区一：流式输出比非流式输出贵**

不会。流式输出和非流式输出的 token 消耗完全相同——只是返回方式不同，不影响计费。

**误区二：流式输出时拿不到 token 统计**

默认情况下流式输出的 usage 为 null，但设置 `stream_options={"include_usage": True}` 后，最后一个 chunk 会包含完整的 token 统计。

**误区三：缓存命中是自动的，不需要做任何事**

缓存命中需要请求前缀一致。如果你每次请求都改 system prompt，缓存就不会命中。保持 system prompt 固定是提高缓存命中率最简单的方法。

**误区四：缓存命中只对 system prompt 有效**

不是。缓存对整个请求前缀有效——包括 system prompt、历史消息、用户消息。只要前缀跟之前的请求一致，就能命中缓存。多轮对话中，历史消息部分通常都能命中缓存。

---

## 小结

这一节我们深入了流式输出和 Token 统计：流式输出用 `stream=True` 启用，逐 token 返回结果；Token 统计通过 `usage` 字段获取，包括 input/output/cache 命中 token 数；缓存命中的 token 价格降低 75%，保持请求前缀一致是提高缓存命中率的关键。下一节我们学习 JSON Output 和结构化输出。
