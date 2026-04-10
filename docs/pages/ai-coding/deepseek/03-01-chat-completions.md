# 3.1 Chat Completions 接口

> **Chat Completions 是 DeepSeek API 的核心接口——掌握它的每个参数，你就能精确控制 AI 的行为。**

---

## 这一节在讲什么？

上一章我们用 5 行代码完成了第一次 API 调用，但那只用了最基础的参数。Chat Completions 接口有很多参数可以控制 AI 的行为——temperature 控制随机性、max_tokens 控制输出长度、frequency_penalty 控制重复度。理解这些参数的含义和最佳设置，是从"能用"到"用好"的关键。这一节我们把 Chat Completions 接口的每个参数都讲清楚。

---

## 接口地址与请求格式

```
POST https://api.deepseek.com/chat/completions
Content-Type: application/json
Authorization: Bearer sk-xxxxxxxx
```

请求体是一个 JSON 对象，包含以下参数：

```json
{
  "model": "deepseek-chat",
  "messages": [
    {"role": "system", "content": "你是一个编程助手"},
    {"role": "user", "content": "写一个快速排序"}
  ],
  "temperature": 0.3,
  "top_p": 0.9,
  "max_tokens": 4096,
  "frequency_penalty": 0,
  "presence_penalty": 0,
  "stop": null,
  "stream": false
}
```

---

## 参数详解

### model：模型选择

| 值 | 模型 | 说明 |
|---|------|------|
| `deepseek-chat` | V3.2 | 通用模型，快速、便宜 |
| `deepseek-reasoner` | R1 | 推理模型，深度思考 |

DeepSeek 会自动使用指定系列的最新版本——你不需要指定具体版本号。

### messages：对话消息列表

messages 是一个消息数组，每条消息包含 `role` 和 `content`：

| role | 说明 | 示例 |
|------|------|------|
| `system` | 系统提示词，定义 AI 的角色和行为规范 | "你是一个专业的 Python 开发者" |
| `user` | 用户消息 | "写一个快速排序" |
| `assistant` | AI 的回复（用于多轮对话） | "好的，这是一个快速排序的实现..." |
| `tool` | 工具调用的结果（Function Calling 时使用） | '{"temperature": 25}' |

**多轮对话示例**：

```python
messages = [
    {"role": "system", "content": "你是一个编程助手，用中文回答"},
    {"role": "user", "content": "什么是快速排序？"},
    {"role": "assistant", "content": "快速排序是一种分治算法..."},
    {"role": "user", "content": "能写一个 Python 实现吗？"}  # 第二轮对话
]
```

多轮对话中，之前的 assistant 消息会作为上下文传给模型——这样模型就能"记住"之前讨论的内容。

### temperature：采样温度

temperature 控制模型输出的随机性，范围 0-2：

```
temperature 的效果：

  0.0 → 完全确定性，每次输出相同（适合代码生成）
  0.1-0.3 → 低随机性，输出稳定（适合代码、技术文档）
  0.5-0.7 → 中等随机性，有一定创造性（适合内容创作）
  0.8-1.0 → 高随机性，输出多样（适合创意写作、头脑风暴）
  1.0-2.0 → 非常随机，可能产生不连贯的内容（不推荐）
```

**推荐设置**：

```python
# 代码生成：低 temperature，确保输出稳定
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "写一个 REST API"}],
    temperature=0.1
)

# 创意写作：高 temperature，增加多样性
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "写一首关于编程的诗"}],
    temperature=0.8
)
```

### top_p：核采样参数

top_p 控制模型从概率最高的 token 中采样的范围，范围 0-1：

```
top_p 的效果：

  0.1 → 只从概率最高的 10% token 中采样（非常保守）
  0.9 → 从概率最高的 90% token 中采样（推荐默认值）
  1.0 → 从所有 token 中采样（不限制）
```

**注意**：`temperature` 和 `top_p` 不要同时调整——建议只调其中一个。一般用 `temperature` 控制随机性就够了，`top_p` 保持默认 0.9。

### max_tokens：最大生成 token 数

max_tokens 限制模型单次生成的最大 token 数：

| 模型 | 默认值 | 最大值 |
|------|--------|--------|
| deepseek-chat | 4096 | 8192（Beta） |
| deepseek-reasoner | 32K | 64K |

**注意**：max_tokens 限制的是**生成**的 token 数，不包括输入的 token 数。如果模型生成的内容超过了 max_tokens，输出会被截断。

```python
# 代码生成：4096 通常够用
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "写一个 REST API"}],
    max_tokens=4096
)

# 长文生成：需要更大的 max_tokens
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "写一篇 3000 字的技术博客"}],
    max_tokens=8192  # Beta 功能
)
```

### frequency_penalty：频率惩罚

frequency_penalty 控制模型对"已经出现过的 token"的惩罚力度，范围 -2 到 2：

- **正值**（如 0.5）：降低重复使用相同 token 的概率，减少重复内容
- **零值**（默认）：不惩罚
- **负值**（如 -0.5）：增加重复使用相同 token 的概率（不推荐）

```python
# 减少代码中的重复模式
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "写一个 CRUD API"}],
    frequency_penalty=0.3
)
```

### presence_penalty：存在惩罚

presence_penalty 控制模型对"已经出现过的 token"的惩罚力度，但跟 frequency_penalty 不同——它只关注 token 是否出现过，不关注出现频率：

- **正值**（如 0.5）：增加讨论新话题的概率
- **零值**（默认）：不惩罚

```python
# 鼓励模型讨论更多不同的话题
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "介绍 Python 的 10 个应用领域"}],
    presence_penalty=0.5
)
```

### stop：停止序列

stop 指定一个或多个字符串，模型遇到这些字符串时停止生成：

```python
# 遇到 "```" 时停止（避免生成多余的代码块）
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "写一个函数"}],
    stop=["```"]
)
```

### stream：流式输出

```python
# 流式输出
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True
)
```

---

## 响应格式

```json
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "deepseek-chat",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "这是一个快速排序的实现..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 50,
    "completion_tokens": 200,
    "total_tokens": 250,
    "prompt_cache_hit_tokens": 30,
    "prompt_cache_miss_tokens": 20
  }
}
```

关键字段：
- `choices[0].message.content`：AI 的回答
- `choices[0].finish_reason`：结束原因（`stop` 正常结束，`length` 达到 max_tokens 被截断）
- `usage`：token 使用统计
- `usage.prompt_cache_hit_tokens`：缓存命中的 token 数（这些 token 按 75% 折扣计费）

---

## 常见误区

**误区一：max_tokens 设太小导致输出被截断**

这是最常见的错误。代码生成通常需要 2000-4000 token，如果你设了 `max_tokens=1000`，AI 的代码可能写到一半就被截断了。建议代码生成至少设 4096，长文生成设 8192。

**误区二：temperature 和 top_p 同时调整**

不建议。这两个参数都会影响输出的随机性，同时调整可能导致不可预测的行为。建议只调 temperature，top_p 保持默认 0.9。

**误区三：frequency_penalty 和 presence_penalty 设太大**

设太大（如 1.5+）会导致输出不自然——模型会刻意避免使用常见的词汇和表达方式。建议值在 0-0.5 之间。

**误区四：finish_reason 为 "length" 时以为 API 出错了**

不是错误。`finish_reason: "length"` 只是表示输出达到了 max_tokens 的限制被截断了——你需要增大 max_tokens 或者缩短输入。

---

## 小结

这一节我们详细讲解了 Chat Completions 接口的每个参数：model 选择模型、messages 定义对话、temperature 控制随机性（代码用 0.1-0.3）、max_tokens 控制输出长度（代码至少 4096）、frequency_penalty/presence_penalty 控制重复、stop 指定停止序列、stream 启用流式输出。掌握这些参数，你就能精确控制 DeepSeek 的输出行为。下一节我们深入流式输出和 Token 统计。
