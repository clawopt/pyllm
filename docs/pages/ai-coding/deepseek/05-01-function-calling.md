# 5.1 Function Calling 详解

> **Function Calling 让 AI 从"只会说话"升级为"会调用工具"——这是构建 AI Agent 的基础能力。**

---

## 这一节在讲什么？

前面四章我们用的都是"对话模式"——你问问题，AI 回答。但 AI 的能力不止于此——通过 Function Calling，AI 可以调用你定义的函数，获取实时数据、执行操作、与外部系统交互。这是构建 AI Agent 的基础能力。这一节我们详细讲解 Function Calling 的工作原理、工具定义格式、调用流程和代码示例。

---

## Function Calling 的工作原理

Function Calling 的工作流程是一个"三方协作"的过程：

```
Function Calling 的完整流程：

  1. 你定义工具（函数签名 + 参数描述）
  2. 你发送 messages + tools 给 API
  3. 模型决定是否需要调用工具
     → 如果需要：返回 tool_calls（函数名 + 参数）
     → 如果不需要：直接返回文本回答
  4. 你的代码执行工具，获取结果
  5. 你把工具结果以 role: "tool" 追加到 messages
  6. 你再次请求 API
  7. 模型根据工具结果生成最终回答
```

关键理解：**模型不会直接执行工具**——它只是"建议"调用哪个工具、传什么参数。真正的执行权在你的代码里。这种设计保证了安全性——你可以审查模型的工具调用请求，决定是否执行。

---

## 定义工具

工具用 JSON Schema 描述函数签名和参数：

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，如'北京'、'上海'"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "温度单位"
                    }
                },
                "required": ["city"]
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
                    "expression": {
                        "type": "string",
                        "description": "数学表达式，如'2+3*4'"
                    }
                },
                "required": ["expression"]
            }
        }
    }
]
```

工具定义的要点：
- `name`：函数名，只允许字母、数字、下划线
- `description`：函数描述，越详细越好——模型根据描述决定何时调用
- `parameters`：参数的 JSON Schema，包含类型、描述、枚举值
- `required`：必填参数列表

---

## 完整的 Function Calling 示例

```python
from openai import OpenAI
import json

client = OpenAI(api_key="sk-xxxxxxxx", base_url="https://api.deepseek.com")

# 1. 定义工具
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称"}
                },
                "required": ["city"]
            }
        }
    }
]

# 2. 实际的函数实现
def get_weather(city: str) -> str:
    weather_data = {
        "北京": "晴天，25°C",
        "上海": "多云，28°C",
        "深圳": "阵雨，30°C"
    }
    return weather_data.get(city, f"{city}的天气信息暂无")

# 3. 工具注册表
tool_map = {
    "get_weather": get_weather
}

# 4. 发送请求
messages = [{"role": "user", "content": "北京今天天气怎么样？"}]

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=messages,
    tools=tools
)

# 5. 处理工具调用
message = response.choices[0].message

if message.tool_calls:
    # 模型请求调用工具
    messages.append(message)  # 把模型的回复加入历史

    for tool_call in message.tool_calls:
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)

        # 执行工具
        result = tool_map[function_name](**function_args)

        # 把工具结果加入消息
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result
        })

    # 6. 再次请求，获取最终回答
    final_response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        tools=tools
    )

    print(final_response.choices[0].message.content)
else:
    # 模型直接回答，不需要调用工具
    print(message.content)
```

输出：

```
北京今天天气是晴天，气温25°C，适合户外活动。
```

---

## 并行工具调用

模型可能在一次响应中请求调用多个工具：

```python
# 用户问："北京和上海的天气怎么样？"
# 模型可能同时请求调用两次 get_weather

for tool_call in message.tool_calls:
    # tool_call 1: get_weather(city="北京")
    # tool_call 2: get_weather(city="上海")
    ...
```

你需要执行所有工具调用，把所有结果都加入 messages，然后再请求 API。

---

## 常见误区

**误区一：模型会直接执行工具**

不会。模型只是返回"建议调用哪个工具、传什么参数"——真正的执行权在你的代码里。你需要自己实现函数、执行调用、返回结果。这种设计保证了安全性。

**误区二：Function Calling 只能用 deepseek-chat**

`deepseek-reasoner` 也支持 Function Calling（R1-0528 新增），但 V3 的工具调用更可靠。建议 Function Calling 场景优先用 V3。

**误区三：工具的 description 不重要**

非常重要。模型根据 description 决定何时调用工具——如果描述不清楚，模型可能在不该调用时调用，或者该调用时不调用。写清楚工具的功能、适用场景、参数含义。

**误区四：一次请求可以传入任意多个工具**

DeepSeek 限制一次请求最多传入 128 个工具定义。如果你有更多工具，需要根据用户问题筛选相关的工具传入。

---

## 小结

这一节我们学习了 Function Calling 的完整流程：定义工具（JSON Schema）、发送请求（messages + tools）、处理工具调用（执行函数、返回结果）、获取最终回答。模型不会直接执行工具——它只是"建议"调用，执行权在你的代码里。下一节我们基于 Function Calling 构建 AI Agent。
