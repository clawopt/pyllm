# 2.2 Hello World：第一次 API 调用

> **从安装 SDK 到拿到第一个 AI 回复，只需要 5 行代码——因为 DeepSeek 完全兼容 OpenAI 格式，迁移零成本。**

---

## 这一节在讲什么？

上一节我们拿到了 API Key，这一节开始写代码。DeepSeek 的 API 完全兼容 OpenAI 格式——这意味着你不需要学习新的 SDK、新的接口、新的参数，只需要把 `base_url` 从 OpenAI 改成 DeepSeek，代码就能跑。这一节我们从安装 SDK 开始，完成第一次 API 调用、流式输出，以及从 OpenAI 迁移的方法。

---

## 安装 OpenAI SDK

DeepSeek 不需要专用 SDK——直接用 OpenAI 的官方 Python SDK 就行：

```bash
pip install openai
```

如果你用的是 Node.js：

```bash
npm install openai
```

---

## 第一次 API 调用

```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-xxxxxxxx",  # 你的 DeepSeek API Key
    base_url="https://api.deepseek.com"  # 关键：改 base_url
)

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "你是一个专业的编程助手"},
        {"role": "user", "content": "用 Python 写一个快速排序"}
    ]
)

print(response.choices[0].message.content)
```

运行这段代码，你会看到 DeepSeek 生成的快速排序实现：

```python
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

# 测试
print(quicksort([3, 6, 8, 10, 1, 2, 1]))
```

就这么简单——5 行核心代码，你已经在用 DeepSeek 生成代码了。

---

## 流式输出

如果你想让 AI 的回答实时显示（而不是等全部生成完才显示），使用流式输出：

```python
stream = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "解释什么是递归"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

流式输出的效果就像 ChatGPT 的"打字机"效果——AI 一个字一个字地输出，你可以实时看到生成过程。这在长文本生成时特别有用——不需要等几十秒才能看到结果。

---

## 使用 R1 推理模型

如果你想用 DeepSeek-R1 的推理能力，只需要改模型名：

```python
response = client.chat.completions.create(
    model="deepseek-reasoner",  # 改成 R1
    messages=[{"role": "user", "content": "证明根号2是无理数"}]
)

# R1 的思考过程
reasoning = response.choices[0].message.reasoning_content
print("=== 思考过程 ===")
print(reasoning)

# R1 的最终回答
answer = response.choices[0].message.content
print("\n=== 最终回答 ===")
print(answer)
```

R1 的响应包含两个字段：
- `reasoning_content`：模型的思考过程（可能很长）
- `content`：模型的最终回答

---

## 从 OpenAI 迁移

如果你已经在用 OpenAI 的 API，迁移到 DeepSeek 只需要改两行：

```python
# 之前用 OpenAI
client = OpenAI(
    api_key="sk-...",  # OpenAI API Key
    base_url="https://api.openai.com/v1"  # OpenAI 的 base_url
)

# 现在用 DeepSeek
client = OpenAI(
    api_key="sk-...",  # DeepSeek API Key
    base_url="https://api.deepseek.com"  # 改成 DeepSeek 的 base_url
)

# 其余代码完全不变！
response = client.chat.completions.create(
    model="deepseek-chat",  # 模型名不同
    messages=[{"role": "user", "content": "Hello!"}]
)
```

迁移清单：

| 需要改的 | OpenAI | DeepSeek |
|---------|--------|----------|
| base_url | `https://api.openai.com/v1` | `https://api.deepseek.com` |
| api_key | OpenAI Key | DeepSeek Key |
| model | `gpt-4o` | `deepseek-chat` 或 `deepseek-reasoner` |

其他参数（messages、temperature、max_tokens 等）完全兼容，不需要修改。

---

## Node.js 示例

如果你用的是 Node.js：

```javascript
import OpenAI from "openai";

const client = new OpenAI({
    apiKey: "sk-xxxxxxxx",
    baseURL: "https://api.deepseek.com"
});

const response = await client.chat.completions.create({
    model: "deepseek-chat",
    messages: [{ role: "user", content: "用 JavaScript 写一个防抖函数" }]
});

console.log(response.choices[0].message.content);
```

---

## cURL 示例

如果你想在命令行中测试：

```bash
curl https://api.deepseek.com/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-xxxxxxxx" \
  -d '{
    "model": "deepseek-chat",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

---

## 常见误区

**误区一：以为需要安装 DeepSeek 专用 SDK**

不需要。DeepSeek 的 API 完全兼容 OpenAI 格式，直接用 `pip install openai` 安装 OpenAI SDK 即可。只需要改 `base_url` 和 `api_key`，代码零修改。

**误区二：base_url 末尾要加 /v1**

不需要。DeepSeek 的 base_url 是 `https://api.deepseek.com`，不需要加 `/v1`。OpenAI SDK 会自动拼接正确的路径。如果你加了 `/v1` 变成 `https://api.deepseek.com/v1`，会导致 404 错误。

**误区三：R1 的思考过程不需要，可以忽略**

思考过程是 R1 的核心价值——它让你能看到 AI 是怎么推理的。在调试场景下，思考过程比最终回答更有价值——它告诉你 AI 为什么得出这个结论，推理链中哪一步可能有问题。

**误区四：流式输出时拿不到 usage 统计**

默认情况下，流式输出的 `usage` 字段可能为 null。如果你需要统计 token 用量，设置 `stream_options={"include_usage": True}`：

```python
stream = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True,
    stream_options={"include_usage": True}
)

for chunk in stream:
    if chunk.usage:
        print(f"Token 用量: {chunk.usage}")
```

---

## 小结

这一节我们完成了第一次 DeepSeek API 调用：安装 OpenAI SDK，改 `base_url` 为 `https://api.deepseek.com`，5 行代码就能生成代码。流式输出用 `stream=True`，R1 模型用 `model="deepseek-reasoner"`。从 OpenAI 迁移只需要改 base_url、api_key 和 model 三个地方。下一节我们看看 DeepSeek 的网页版和 App 怎么用。
