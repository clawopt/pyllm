# 3.3 JSON Output 与结构化输出

> **让 AI 输出"自由散文"容易，让它输出"结构化表格"难——JSON Output 就是解决这个问题的。**

---

## 这一节在讲什么？

当你用 DeepSeek 做代码审查、数据提取、分类任务时，你需要 AI 输出结构化的 JSON 格式——而不是一段自由格式的文本。DeepSeek 的 JSON Output 功能强制模型输出合法的 JSON 对象，让你的代码可以可靠地解析 AI 的回答。这一节我们讲解 JSON Output 的启用方式、使用要求、代码示例和常见陷阱。

---

## 启用 JSON Output

通过 `response_format` 参数启用 JSON Output：

```python
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "你是一个代码审查助手。请以 JSON 格式输出审查结果，包含 summary 和 issues 两个字段。"},
        {"role": "user", "content": "审查这段代码：def add(a, b): return a + b"}
    ],
    response_format={"type": "json_object"}
)

import json
result = json.loads(response.choices[0].message.content)
print(result)
```

输出示例：

```json
{
  "summary": "代码功能正确但缺少类型注解和输入验证",
  "issues": [
    {
      "type": "style",
      "description": "缺少类型注解",
      "suggestion": "def add(a: int, b: int) -> int: return a + b"
    },
    {
      "type": "safety",
      "description": "没有输入验证",
      "suggestion": "添加类型检查，确保 a 和 b 是数字"
    }
  ]
}
```

---

## 使用要求

JSON Output 有一个**必须遵守的规则**：你必须在 system 或 user 消息中明确要求模型输出 JSON 格式。如果你只设置了 `response_format` 但没有在提示词中提到 JSON，模型可能输出非 JSON 内容。

```python
# ❌ 错误：设置了 JSON Output 但提示词中没有要求 JSON
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "审查这段代码"}],
    response_format={"type": "json_object"}
)
# 模型可能输出纯文本，不是 JSON

# ✅ 正确：提示词中明确要求 JSON 格式
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "请以 JSON 格式输出，包含 summary 和 issues 字段"},
        {"role": "user", "content": "审查这段代码"}
    ],
    response_format={"type": "json_object"}
)
```

---

## 实用示例

### 代码审查结构化输出

```python
def review_code(code: str) -> dict:
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role": "system",
                "content": """你是一个代码审查助手。请以 JSON 格式输出审查结果：
{
  "score": 1-10 的代码质量评分,
  "summary": "总体评价",
  "issues": [
    {"severity": "high/medium/low", "description": "问题描述", "line": 行号, "suggestion": "修复建议"}
  ],
  "positive": ["做得好的地方"]
}"""
            },
            {"role": "user", "content": f"审查以下代码：\n{code}"}
        ],
        response_format={"type": "json_object"},
        temperature=0.1
    )
    return json.loads(response.choices[0].message.content)
```

### 数据提取

```python
def extract_info(text: str) -> dict:
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role": "system",
                "content": """从文本中提取信息，以 JSON 格式输出：
{
  "entities": [{"name": "实体名", "type": "人名/地名/组织/日期"}],
  "key_facts": ["关键事实1", "关键事实2"],
  "sentiment": "positive/neutral/negative"
}"""
            },
            {"role": "user", "content": text}
        ],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)
```

---

## 常见误区

**误区一：设置了 JSON Output 就不需要在提示词中要求 JSON**

这是最常见的错误。DeepSeek 要求你必须在提示词中明确要求 JSON 格式——否则模型可能忽略 `response_format` 设置，输出纯文本。

**误区二：JSON Output 能保证特定的 JSON Schema**

JSON Output 只保证输出是合法的 JSON，不保证输出符合特定的 Schema。如果你需要特定的字段和结构，必须在提示词中详细描述——包括字段名、类型、示例值。

**误区三：JSON Output 会降低回答质量**

不会。JSON Output 只是改变了输出的格式，不影响内容质量。实际上，结构化输出通常比自由文本更精确——因为模型需要按照指定的结构组织信息，减少了"跑题"的可能。

**误区四：JSON Output 只能用于 deepseek-chat**

`deepseek-reasoner` 也支持 JSON Output，但 R1 的思考过程不受 JSON 格式约束——只有最终回答是 JSON 格式。

---

## 小结

这一节我们学习了 JSON Output：通过 `response_format={"type": "json_object"}` 启用，强制模型输出合法 JSON。关键要求是必须在提示词中明确要求 JSON 格式。JSON Output 适合代码审查、数据提取、分类任务等需要结构化输出的场景。下一节我们学习 FIM 补全和对话前缀续写。
