# 3.4 FIM 补全与对话前缀续写

> **Chat Completions 是"从头写"，FIM 是"填中间"——两种模式覆盖了代码生成的所有场景。**

---

## 这一节在讲什么？

Chat Completions 接口适合"从头生成"——你给一个需求，AI 生成完整代码。但实际开发中，你经常需要"填中间"——比如在已有代码的某个位置插入一段逻辑，或者续写被截断的输出。DeepSeek 提供了两种特殊能力来解决这些问题：**FIM（Fill-In-the-Middle）补全**和**对话前缀续写**。这一节我们讲解这两种能力的使用方法和适用场景。

---

## FIM 补全

FIM（Fill-In-the-Middle）补全让你提供代码的前缀和后缀，模型补全中间的内容。这在代码编辑器场景中非常有用——用户在代码中间打字，编辑器需要补全光标位置的内容。

### 接口地址

FIM 使用 Beta 端点：

```
POST https://api.deepseek.com/beta/completions
```

### 代码示例

```python
client = OpenAI(
    api_key="sk-xxxxxxxx",
    base_url="https://api.deepseek.com/beta"  # 注意：用 beta 端点
)

response = client.completions.create(
    model="deepseek-chat",
    prompt="<｜fim▁begin｜>def quicksort(arr):\n<｜fim▁hole｜>\n    return arr<｜fim▁end｜>",
    max_tokens=256,
    stop=["<｜fim▁end｜>"]
)

print(response.choices[0].text)
```

FIM 的 prompt 格式使用特殊标记：
- `<｜fim▁begin｜>`：前缀开始
- `<｜fim▁hole｜>`：需要补全的位置
- `<｜fim▁end｜>`：后缀开始

上面的例子中，模型会补全 `quicksort` 函数的中间部分——分区和递归逻辑。

### FIM 的适用场景

```
✅ 适合 FIM 的场景：
- 代码编辑器的自动补全（VS Code 插件、Jupyter 插件）
- 在已有代码中插入新逻辑
- 根据函数签名和注释生成函数体
- 根据前后文补全代码片段

❌ 不适合 FIM 的场景：
- 从零生成完整代码（用 Chat Completions）
- 多轮对话（用 Chat Completions）
- Function Calling（用 Chat Completions）
```

---

## 对话前缀续写

对话前缀续写让你指定 assistant 消息的前缀，模型按照前缀续写。这在以下场景很有用：

- 强制模型以特定格式开始输出（如 ` ```python `）
- 续写被 max_tokens 截断的输出
- 引导模型的输出方向

### 使用方法

在 messages 的最后添加一条 `role: "assistant"` 消息，并设置 `prefix: True`：

```python
client = OpenAI(
    api_key="sk-xxxxxxxx",
    base_url="https://api.deepseek.com/beta"  # 注意：用 beta 端点
)

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "user", "content": "写一个 Python 的快速排序"},
        {"role": "assistant", "content": "```python\n", "prefix": True}
    ],
    max_tokens=4096
)

print(response.choices[0].message.content)
```

模型会从 ` ```python\n ` 后面续写，直接输出代码内容，不需要再输出代码块标记。

### 续写被截断的输出

当模型的输出因为 max_tokens 限制被截断时，你可以用对话前缀续写来继续：

```python
# 第一次请求：输出被截断
response1 = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "写一个完整的 REST API"}],
    max_tokens=2048
)
partial_output = response1.choices[0].message.content

# 第二次请求：续写
response2 = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "user", "content": "写一个完整的 REST API"},
        {"role": "assistant", "content": partial_output, "prefix": True}
    ],
    max_tokens=2048
)
continued_output = response2.choices[0].message.content

# 拼接完整输出
full_output = partial_output + continued_output
```

---

## 常见误区

**误区一：FIM 和 Chat Completions 可以混用**

不能。FIM 使用 `/beta/completions` 端点，Chat Completions 使用 `/chat/completions` 端点。它们的接口格式不同——FIM 用 `prompt` 参数，Chat Completions 用 `messages` 参数。

**误区二：对话前缀续写不需要 Beta 端点**

需要。对话前缀续写的 `prefix: True` 参数只在 Beta 端点（`https://api.deepseek.com/beta`）下有效。如果你用正式端点，`prefix` 参数会被忽略。

**误区三：FIM 只能补全代码**

不是。FIM 可以补全任何文本——代码、文章、配置文件等。只要你能提供前缀和后缀，FIM 就能补全中间的内容。但它在代码补全场景下效果最好，因为代码的结构性最强。

**误区四：Beta 功能随时可能变化**

是的。FIM 和对话前缀续写都是 Beta 功能，接口格式可能在未来版本中变化。在生产环境中使用时，建议关注 DeepSeek 的更新日志，及时调整代码。

---

## 小结

这一节我们学习了 DeepSeek 的两种特殊能力：FIM 补全通过前缀+后缀的方式补全中间内容，适合代码编辑器的自动补全；对话前缀续写通过 `prefix: True` 让模型从指定前缀续写，适合强制格式输出和续写被截断的内容。两者都使用 Beta 端点，接口可能在未来版本中变化。下一章我们深入 DeepSeek 的推理模式与思考过程。
