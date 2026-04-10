# 5.2 构建 AI Agent

> **Agent = LLM + 工具 + 循环——让 AI 自主决定调用什么工具、执行多少轮，直到完成任务。**

---

## 这一节在讲什么？

上一节我们学了 Function Calling 的单次调用流程——模型请求调用工具，你执行后返回结果，模型生成最终回答。但实际场景中，一个任务可能需要多轮工具调用——AI 先查天气，再查航班，再订票。AI Agent 就是一个能自主进行多轮工具调用的循环——AI 自己决定调用什么工具、执行多少轮，直到完成任务。这一节我们用 DeepSeek 构建一个完整的 AI Agent。

---

## Agent 循环

Agent 的核心是一个循环：

```
Agent 循环：

  while 任务未完成:
    1. 发送 messages 给 LLM
    2. LLM 返回响应
    3. 如果 LLM 请求调用工具：
       a. 执行工具
       b. 把结果加入 messages
       c. 继续循环
    4. 如果 LLM 直接给出最终回答：
       a. 返回回答
       b. 退出循环
```

---

## 完整的 Agent 实现

```python
from openai import OpenAI
import json

client = OpenAI(api_key="sk-xxxxxxxx", base_url="https://api.deepseek.com")

# 工具定义
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "搜索互联网获取信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "执行数学计算",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "数学表达式"}
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取文件内容",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "文件路径"}
                },
                "required": ["path"]
            }
        }
    }
]

# 工具实现
def search_web(query: str) -> str:
    return f"搜索结果：关于'{query}'的最新信息..."

def calculate(expression: str) -> str:
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"计算错误：{e}"

def read_file(path: str) -> str:
    try:
        with open(path, "r") as f:
            return f.read()[:2000]
    except FileNotFoundError:
        return f"文件不存在：{path}"

tool_map = {
    "search_web": search_web,
    "calculate": calculate,
    "read_file": read_file
}

# Agent 主循环
def run_agent(user_message: str, max_rounds: int = 10) -> str:
    messages = [
        {"role": "system", "content": "你是一个AI助手，可以使用工具帮助用户完成任务。"},
        {"role": "user", "content": user_message}
    ]

    for round_num in range(max_rounds):
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=tools,
            temperature=0.1
        )

        message = response.choices[0].message
        messages.append(message)

        if not message.tool_calls:
            return message.content

        for tool_call in message.tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)

            print(f"[Round {round_num + 1}] 调用工具: {func_name}({func_args})")

            result = tool_map[func_name](**func_args)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result)
            })

    return "Agent 达到最大轮数限制，任务未完成。"

# 使用 Agent
answer = run_agent("计算 (15 + 27) * 3 的结果")
print(answer)
```

---

## 错误处理

Agent 的工具调用可能失败——你需要处理这些错误：

```python
for tool_call in message.tool_calls:
    func_name = tool_call.function.name
    func_args = json.loads(tool_call.function.arguments)

    try:
        result = tool_map[func_name](**func_args)
    except Exception as e:
        result = f"工具执行失败：{e}"

    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": str(result)
    })
```

当工具执行失败时，把错误信息返回给模型——模型可能会换一种方式完成任务，或者告诉用户遇到了问题。

---

## 常见误区

**误区一：Agent 不需要限制工具调用轮数**

需要。如果模型陷入循环（反复调用同一个工具），可能导致 API 费用失控。建议设置最大轮数（如 10 轮），超过后强制退出。

**误区二：Agent 只能用 deepseek-chat**

`deepseek-reasoner` 也支持 Function Calling，但 V3 的工具调用更可靠、更快速。建议 Agent 场景优先用 V3。

**误区三：Agent 的工具调用总是正确的**

不一定。模型可能传入错误的参数、调用不相关的工具、甚至"幻觉"出不存在的工具名。你需要在代码中验证模型的工具调用请求——检查函数名是否存在、参数是否合法。

**误区四：Agent 能完成任何任务**

不能。Agent 的能力受限于你提供的工具——没有数据库查询工具，Agent 就无法查数据库；没有文件写入工具，Agent 就无法修改文件。Agent 的能力边界 = LLM 的推理能力 + 你提供的工具集。

---

## 小结

这一节我们用 DeepSeek 构建了一个完整的 AI Agent：Agent 循环（LLM → 工具调用 → 执行 → 结果返回 → 继续）、工具注册表模式、错误处理、最大轮数限制。Agent 的核心是"自主决策"——AI 自己决定调用什么工具、执行多少轮。下一节我们学习 DeepSeek + RAG 实战。
