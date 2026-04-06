---
title: ReAct 模式：构建你的第一个 Agent
description: create_react_agent 用法、思考-行动-观察循环详解、调试与优化技巧
---
# ReAct 模式：构建你的第一个 Agent

前两节我们理解了 Agent 的概念和工具的定义方法。现在到了最关键的一步——**把工具交给 Agent，让它真正跑起来**。这一节我们将使用 LangChain 最经典的 ReAct 模式，从零构建一个能自主决策的智能体。

## 创建 Agent 的标准方式

LangChain v1.0 推荐使用 `create_react_agent()` 函数来创建 Agent。它的接口非常简洁：

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_react_agent

# 1. 定义 LLM（Agent 的大脑）
chat = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 2. 定义工具（Agent 的手和脚）
@tool
def search(query: str) -> str:
    """搜索互联网获取实时信息"""
    if "天气" in query:
        return "北京今天晴 28°C，上海多云 32°C"
    elif "新闻" in query:
        return "2025年4月：OpenAI 发布 GPT-5 预览版"
    else:
        return f"关于 '{query}' 的搜索结果"

@tool
def calculator(expression: str) -> str:
    """计算数学表达式"""
    try:
        result = eval(expression, {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"计算错误: {e}"

tools = [search, calculator]

# 3. 创建 Agent
agent = create_react_agent(
    model=chat,
    tools=tools,
    prompt="你是一个有帮助的助手。使用可用的工具来回答问题。"
)
```

三步就完成了一个 Agent 的创建。现在让我们用它来处理几个问题：

```python
# 问题一：需要搜索
result1 = agent.invoke({"input": "北京今天的天气怎么样？"})
print(result1["output"])
# 根据搜索结果，北京今天是晴天，气温约 28°C。

# 问题二：需要计算
result2 = agent.invoke({"input": "175 乘以 23 再加上 456 等于多少？"})
print(result2["output"])
# 让我来计算一下：
# 175 × 23 = 4025
# 4025 + 456 = 4481
# 所以答案是 **4481**

# 问题三：需要多步推理 + 工具调用
result3 = agent.invoke({
    "input": "如果北京今天的最高温是28度，上海是32度，两地的温差是多少？"
})
print(result3["output"])
```

第三个问题的执行过程大概是这样的（Agent 内部的 ReAct 循环）：

```
> Entering new AgentExecutor chain...

Thought: 我需要分别查询北京和上海的天气来获取温度信息。
Action: search[{"query": "北京今天天气"}]

Observation: 北京今天晴 28°C，上海多云 32°C

Thought: 我已经获取到北京温度为28°C，上海温度为32°C。
现在可以计算温差了。

Action: calculator[{"expression": "32 - 28"}]

Observation: 4

Thought: 计算结果是4度。我有足够的信息来回答用户的问题了。

Final Answer: 根据天气数据，北京今天最高温约28°C，
上海约32°C，两地温差约为 **4°C**（上海更热）。

> Chain finished.
```

看到了吗？Agent 自动完成了以下操作：
1. **判断**需要先查天气 → 调用 search 工具
2. **分析**返回结果 → 提取出两个城市的温度
3. **决定**需要做减法运算 → 调用 calculator 工具
4. **综合**所有结果 → 组织最终答案

整个过程没有人工干预——Agent 自己规划、自己执行、自己做最终决策。

## 深入理解 ReAct 循环

让我们把上面的执行过程拆解得更清楚一些。`create_react_agent` 内部的工作机制如下：

### Step 1：Prompt 组装

Agent 的提示词模板大致长这样（简化版）：

```
你是一个有帮助的助手。使用可用的工具来回答问题。

你有以下工具可用：
search: 搜索互联网获取实时信息
calculator: 计算数学表达式

使用以下格式：

Question: 用户的问题
Thought: 你应该做什么
Action: 工具名[参数]
Observation: 工具返回的结果
... (这个 Thought/Action/Observation 可以重复多次)
Thought: 我现在知道最终答案了
Final Answer: 最终答案

开始！

Question: {input}
{agent_scratchpad}
```

关键部分是 `{agent_scratchpad}` —— 这是一个动态区域，每次 ReAct 循环都会在这里追加新的 Thought/Action/Observation 内容。

### Step 2：LLM 决策

LLM 收到完整的 Prompt 后，生成第一轮输出：

```
Thought: 我需要查询北京的天气...
Action: search[{"query": "北京今天天气"}]
```

### Step 3：工具执行

执行引擎解析出 Action 中的工具名和参数，调用对应的函数：

```python
tool_name = "search"
tool_args = {"query": "北京今天天气"}
observation = tools_map[tool_name].invoke(tool_args)
```

### Step 4：结果回填

把 Observation 追加到 scratchpad 中，然后连同之前的内容一起再次发给 LLM：

```
Question: 两地温差是多少？

Thought: 我需要查天气
Action: search[{"query": "北京今天天气"}]
Observation: 北京今天晴 28°C，上海多云 32°C   ← 新增

Thought: 温度拿到了，现在计算温差      ← LLM 的新一轮推理
Action: calculator[{"expression": "32-28"}]
Observation: 4                           ← 新增

Thought: 答案出来了                    ← LLM 判断任务完成
Final Answer: 温差约 4 度
```

### Step 5：终止判断

当 LLM 输出以 "Final Answer:" 开头的内容时，执行引擎识别到终止信号，停止循环并返回结果。

## 完整的可运行示例

下面是一个完整的多工具 Agent 示例，包含更多实用功能：

```python
"""
my_first_agent.py — 第一个 ReAct Agent
"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_react_agent

load_dotenv()

# ========== 定义工具 ==========

@tool
def search_web(query: str) -> str:
    """
    搜索互联网获取最新信息。
    
    适用场景：新闻、天气、价格、体育比分等时效性信息。
    不适用：数学计算、常识问答、编程问题。
    """
    data = {
        "北京天气": "晴，28°C，空气质量良",
        "上海天气": "多云，32°C，湿度65%",
        "Python版本": "Python 最新稳定版 3.12.7 (2024年10月发布)",
        "GPT-5消息": "OpenAI 于2025年3月发布 GPT-5 预览版",
    }
    for key, value in data.items():
        if key.lower() in query.lower() or any(k in query for k in key.split()):
            return value
    return f"未找到关于'{query}'的最新信息"

@tool
def calculate(expr: str) -> str:
    """
    计算数学表达式。
    
    支持: 加(+)、减(-)、乘(*)、除(/)、幂(**)、括号()
    示例: "175 * 23 + 456", "(100 + 200) * 3 / 50"
    """
    allowed_names = {
        "abs": abs, "max": max, "min": min,
        "round": round, "pow": pow, "sqrt": __import__("math").sqrt
    }
    try:
        result = eval(expr, {"__builtins__": {}}, allowed_names)
        return f"{expr} = {result}"
    except ZeroDivisionError:
        return "错误：除数不能为零"
    except Exception as e:
        return f"计算错误: {e}"

@tool
def get_current_datetime() -> str:
    """获取当前的日期和时间（含星期几）"""
    from datetime import datetime
    now = datetime.now()
    weekdays = ["一", "二", "三", "四", "五", "六", "日"]
    return now.strftime(f"%Y年%m月%d日 %H:%M:%S 星期{weekdays[now.weekday()]}")

# ========== 创建 Agent ==========

chat = ChatOpenAI(model="gpt-4o-mini", temperature=0)

tools = [search_web, calculate, get_current_datetime]

agent = create_react_agent(
    model=chat,
    tools=tools,
    prompt="你是一个精准的助手。仔细分析用户需求，选择最合适的工具完成任务。"
)

# ========== 测试 ==========

questions = [
    "现在几点了？",
    "帮我算一下 (175 * 23 + 456) / 10",
    "北京和上海今天的温差多大？",
    "Python 最新版是什么时候发布的？",
]

for q in questions:
    print(f"\n{'='*50}")
    print(f"❓ 问: {q}")
    print("-" * 50)
    result = agent.invoke({"input": q})
    print(f"💡 答: {result['output']}")
```

运行效果：

```
==================================================
❓ 问: 现在几点了？
--------------------------------------------------
💡 答: 现在的时间是 2025年04月06日 14:32:15 星期日

==================================================
❓ 问: 帮我算一下 (175 * 23 + 456) / 10
--------------------------------------------------
💡 答: 经过计算：(175 * 23 + 456) / 10 = 448.1

==================================================
❓ 问: 北京和上海今天的温差多大？
--------------------------------------------------
💡 答: 根据搜索到的天气信息：
- 北京：晴，28°C
- 上海：多云，32°C
两地温差约为 **4°C**（上海比北京热）

==================================================
❓ 问: Python 最新版是什么时候发布的？
--------------------------------------------------
💡 答: Python 最新稳定版是 **3.12.7**，
于 **2024年10月** 发布。
```

## 查看 Agent 的详细执行过程

默认情况下，`agent.invoke()` 只返回最终答案。如果你想看 Agent 内部的完整思维过程（对调试非常有帮助），可以在创建时开启 verbose 模式或使用特殊配置：

```python
from langchain.agents import AgentExecutor

# 用 AgentExecutor 包装，获得更多控制能力
agent_executor = AgentExecutor(
    agent=agent["agent"],       # create_react_agent 返回的是一个字典
    tools=tools,
    verbose=True,               # 打印详细的执行过程
    max_iterations=10,          # 最大循环次数限制
    handle_parsing_errors=True  # 自动处理解析错误
)

result = agent_executor.invoke({"input": "北京和上海温差多少？"})
```

开启 `verbose=True` 后，控制台会打印出每一轮的 Thought / Action / Observation，让你能清楚地看到 Agent 是如何一步步推理的。这在调试时极其有用——如果 Agent 给出了错误答案，你可以通过查看日志找出它在哪一步做出了错误决策。

## 常见问题排查

**问题一：Agent 不调用工具直接瞎答**

这是最常见的 Agent 失败模式。原因通常是：
1. **工具描述不够清晰**——LLM 不知道该在什么时候用它
2. **模型太弱**——小模型可能无法正确理解工具调用的指令格式
3. **Prompt 太简短**——没有足够强调"必须使用工具"

解决方案：
- 重写工具描述，明确说明适用场景
- 升级到更强的模型（如 gpt-4o）
- 在 system prompt 中强调："你必须使用提供的工具来获取信息"

**问题二：Agent 陷入死循环**

有时候 Agent 会在两个工具之间反复跳转，无法停下来：

```
Thought: 我需要搜索 A
Action: search[A]
Observation: ...
Thought: 信息不够，我还需要搜索 B  
Action: search[B]
Observation: ...
Thought: 还是觉得不够，再搜一次 A
Action: search[A]     ← 重复了！
```

解决方法：
- 设置 `max_iterations` 限制最大循环次数（如 5~10 次）
- 改进工具描述，让 Agent 更好地判断何时信息已充足
- 使用更强的模型（gpt-4o 比 gpt-4o-mini 更少出现死循环）

**问题三：工具参数传错类型**

比如工具期望 `int` 类型但 LLM 传了字符串 `"123"`。解决方法是在工具内部做好类型转换：

```python
@tool
def get_user_age(user_id: str) -> str:
    """查询用户的年龄"""
    try:
        uid = int(user_id)   # 兜底转换
    except ValueError:
        return f"错误: user_id 必须是数字，收到的是 '{user_id}'"
    # ... 继续正常逻辑
```

下一节我们将综合运用本章所学，构建一个功能丰富的实战 Agent——它将整合搜索、计算、文件操作等多种能力，展示 Agent 在真实场景中的工作方式。
