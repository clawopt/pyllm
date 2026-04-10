# 4.1 DeepSeek-R1 的思考模式

> **R1 不只是"回答问题"——它先"想"再"说"，思考过程完全可见，让你能审查 AI 的推理链。**

---

## 这一节在讲什么？

DeepSeek-R1 最大的特点不是"回答得准"，而是"思考过程可见"。当你问 R1 一个复杂问题时，它不会直接给出答案——而是先在内部进行一段推理（思考过程），然后再生成最终回答。这个思考过程通过 `reasoning_content` 字段返回，你可以看到 AI 是怎么一步步推导出答案的。这一节我们深入 R1 的思考模式：它怎么工作、怎么获取思考过程、思考过程的格式和内容。

---

## 思考模式的工作原理

R1 的思考模式借鉴了人类解决复杂问题的方式——先想后说。当你问一个简单问题（如"1+1等于几"），R1 会快速回答，不需要深度思考。但当你问一个复杂问题（如"证明根号2是无理数"），R1 会先在内部进行推理，然后再给出最终答案。

```
R1 的工作流程：

  用户问题 → 模型内部推理（<think)> 标签内） → 生成最终回答

  简单问题：
  "1+1等于几？" → [几乎不思考] → "2"

  复杂问题：
  "证明根号2是无理数" → [思考 5-10 秒] → 完整的证明过程
```

思考过程的长度取决于问题的复杂度——简单问题可能只有几十个 token，复杂问题可能消耗数万个 token。

---

## 获取思考过程

在 API 调用中，R1 的响应包含两个字段：

```python
response = client.chat.completions.create(
    model="deepseek-reasoner",
    messages=[{"role": "user", "content": "证明根号2是无理数"}]
)

# 思考过程
reasoning = response.choices[0].message.reasoning_content
print("=== 思考过程 ===")
print(reasoning)

# 最终回答
answer = response.choices[0].message.content
print("\n=== 最终回答 ===")
print(answer)
```

`reasoning_content` 包含完整的思考链——你可以看到模型是怎么一步步推导出答案的。这在调试和审查场景下非常有价值。

---

## 思考过程的格式

思考过程是一段纯文本，通常包含以下内容：

```
让我分析一下这个问题...

首先，我需要理解题目要求。题目要求证明根号2是无理数。
无理数的定义是：不能表示为两个整数之比的实数。

我考虑用反证法。假设根号2是有理数，即根号2 = p/q，
其中 p 和 q 是互质的正整数。

两边平方，得到 2 = p²/q²，即 p² = 2q²。
这意味着 p² 是偶数，所以 p 也是偶数。
设 p = 2k，代入得 4k² = 2q²，即 q² = 2k²。
这意味着 q² 也是偶数，所以 q 也是偶数。

但 p 和 q 都是偶数，与它们互质的假设矛盾。
因此，根号2是无理数。证毕。

让我检查一下推理过程是否有漏洞...
[检查过程]

推理没有问题，可以给出最终回答了。
```

思考过程的价值在于：
- **可审查**：你可以检查 AI 的推理链是否正确
- **可调试**：如果 AI 的回答有误，你可以从思考过程中找到出错的步骤
- **可学习**：你可以从 AI 的思考过程中学习解题方法

---

## 流式输出中的思考过程

R1 的流式输出也会逐步返回思考过程：

```python
stream = client.chat.completions.create(
    model="deepseek-reasoner",
    messages=[{"role": "user", "content": "证明根号2是无理数"}],
    stream=True
)

reasoning = ""
answer = ""

for chunk in stream:
    delta = chunk.choices[0].delta
    if hasattr(delta, "reasoning_content") and delta.reasoning_content:
        reasoning += delta.reasoning_content
        print(delta.reasoning_content, end="", flush=True)
    if delta.content:
        answer += delta.content
        print(delta.content, end="", flush=True)
```

流式输出时，思考过程和最终回答是分阶段返回的——先返回思考过程，再返回最终回答。

---

## 思考模式的触发方式

### 方式一：使用 deepseek-reasoner 模型

```python
response = client.chat.completions.create(
    model="deepseek-reasoner",  # 自动启用思考模式
    messages=[{"role": "user", "content": "..."}]
)
```

### 方式二：通过 thinking 参数控制（V3.1+）

```python
# 启用思考模式
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "..."}],
    extra_body={"thinking": {"type": "enabled"}}
)

# 禁用思考模式
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "..."}],
    extra_body={"thinking": {"type": "disabled"}}
)
```

---

## 常见误区

**误区一：思考过程不收费**

思考过程消耗的 token 也计入费用——按 output 价格计费。复杂问题的思考过程可能消耗 10K-30K token，这会显著增加 API 费用。使用 R1 时要注意监控 token 消耗。

**误区二：思考过程总是正确的**

不一定。R1 的思考过程可能包含错误的推理步骤——虽然最终回答可能是正确的（因为模型在思考过程中会自我纠正），但中间步骤可能有误。审查思考过程时，不要假设每一步都是对的。

**误区三：思考过程可以编辑后重新提交**

目前 API 不支持编辑思考过程后重新提交。思考过程是模型内部生成的，你只能读取，不能修改。

**误区四：R1 的所有回答都有思考过程**

大部分回答都有，但极简单的问题（如"1+1等于几"）可能没有思考过程，或者思考过程非常短。`reasoning_content` 可能为空字符串。

---

## 小结

这一节我们深入了 R1 的思考模式：模型先在 `<think)>` 标签内进行内部推理，再生成最终回答；思考过程通过 `reasoning_content` 字段返回；流式输出中思考过程和最终回答分阶段返回。思考过程让你能审查 AI 的推理链，但注意思考 token 也按 output 价格计费。下一节我们学习什么时候用推理模式、什么时候用非推理模式。
