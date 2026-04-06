---
title: Tool（工具）：给 AI 装上"手和脚"
description: "tool装饰器、内置工具、自定义工具、Tool的设计原则与最佳实践"
---
# Tool（工具）：给 AI 装上"手和脚"

上一节我们理解了 Agent 的核心概念——一个能自主思考和决策的智能体。但光有"大脑"（LLM）是不够的，Agent 还需要"手和脚"来与外部世界交互。**工具（Tool）** 就是 Agent 与外部世界交互的接口。

这一节我们将系统地学习如何定义和使用 LangChain 工具——从最简单的自定义函数到功能丰富的内置工具库。

## 什么是 Tool

在 LangChain 中，一个 **Tool** 本质上就是一个被包装过的 Python 函数，它包含三个关键信息：

1. **函数本身**：实际执行操作的代码
2. **名称（name）**：Agent 用来引用这个工具的标识符
3. **描述（description）**：告诉 Agent 这个工具是做什么的、什么时候该用它

```python
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """查询指定城市的当前天气"""
    # 模拟天气数据
    weather_data = {
        "北京": "晴，28°C",
        "上海": "多云，32°C",
        "广州": "雷阵雨，30°C"
    }
    return weather_data.get(city, f"{city}的天气数据暂不可用")

# 查看 Tool 的元信息
print(get_weather.name)         # "get_weather"
print(get_weather.description)  # "查询指定城市的当前天气"
print(get_weather.args)         # 参数 schema (JSON Schema)
```

注意 `@tool` 装饰器——它是 LangChain 定义工具的标准方式。它会自动从函数签名和 docstring 中提取名称、描述和参数信息。**描述文本尤其重要**，因为 LLM 就是靠阅读这些文本来决定何时调用哪个工具的。

## 基础用法：创建和使用工具

### 最简单的工具

```python
from langchain_core.tools import tool

@tool
def add(a: int, b: int) -> int:
    """计算两个整数的和"""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """计算两个整数的乘积"""
    return a * b

# 把工具收集到一个列表中
tools = [add, multiply]

# 直接调用（就像普通函数一样）
result = add.invoke({"a": 3, "b": 5})
print(result)  # 8
```

### 带复杂参数的工具

实际场景中的工具通常有更丰富的参数：

```python
@tool
def search_database(
    table: str,
    filters: dict = None,
    limit: int = 10,
    sort_by: str = "created_at"
) -> str:
    """
    在数据库中搜索记录
    
    Args:
        table: 要查询的表名（users / orders / products）
        filters: 过滤条件字典，如 {"status": "active", "age__gt": 18}
        limit: 返回的最大记录数，默认 10
        sort_by: 排序字段，默认按创建时间降序
    """
    # 实际项目中这里会连接真实数据库
    results = [
        {"id": i, "table": table, **(filters or {})}
        for i in range(min(limit, 3))
    ]
    import json
    return json.dumps(results, ensure_ascii=False, indent=2)

# 测试
print(search_database.invoke({
    "table": "users",
    "filters": {"status": "active"},
    "limit": 5
}))
```

输出：

```
[
  {
    "id": 0,
    "table": "users",
    "status": "active"
  },
  {
    "id": 1,
    "table": "users",
    "status": "active"
  },
  {
    "id": 2,
    "table": "users",
    "status": "active"
  }
]
```

## 工具设计的关键原则

写好一个 Tool 不只是写好一个函数——你还需要考虑 Agent 如何理解和使用它。以下是几个最重要的原则：

### 原则一：描述要精准且有用

描述是 Agent 决策的唯一依据。好的描述应该回答三个问题：

```python
@tool
def search_web(query: str) -> str:
    """
    搜索互联网获取实时信息。
    
    适用场景：
    - 需要最新的新闻、价格、天气等时效性信息
    - 用户询问训练数据中可能不包含的内容
    - 需要验证某个事实的准确性
    
    不适用场景：
    - 纯数学计算（请使用 calculator 工具）
    - 编程问题（代码知识已在训练数据中）
    - 简单常识问答
    """
```

对比一下糟糕的描述：

```python
# ❌ 糟糕 — 太模糊，Agent 不知道什么时候该用
@tool
def search_web(query: str) -> str:
    """搜索网络"""

# ✅ 好 — 明确说明了适用和不适用场景
@tool  
def search_web(query: str) -> str:
    """搜索引擎，用于查找实时信息如新闻、天气、股票价格等"""
```

### 原则二：参数要有类型注解和说明

```python
# ❌ 糳糕 — 参数含义不清楚
@tool
def query(data):
    ...

# ✅ 好 — 参数名、类型、含义一目了然
@tool
def query_stock_price(
    symbol: str,
    period: str = "1d"
) -> str:
    """
    查询股票价格
    
    Args:
        symbol: 股票代码，如 "AAPL"、"000001.SZ"、"TSLA"
        period: 时间周期，可选 "1d"(日), "1w"(周), "1m"(月)
    """
```

### 原则三：返回值要结构化

工具的返回值会被送回 LLM 进行理解和推理。如果返回格式混乱或冗长，会增加 LLM 处理的难度：

```python
# ❌ 糟糕 — 返回一大段非结构化文字
@tool
def get_user_info(user_id: int) -> str:
    user = db.find(user_id)
    return f"用户{user.name}，邮箱{user.email}，注册于{user.created_at}..."

# ✅ 好 — 返回结构化的 JSON 或清晰的键值对
@tool
def get_user_info(user_id: int) -> str:
    user = db.find(user_id)
    import json
    return json.dumps({
        "name": user.name,
        "email": user.email,
        "created_at": str(user.created_at),
        "is_active": user.is_active
    }, ensure_ascii=False)
```

## LangChain 内置工具

LangChain 和社区提供了大量开箱即用的内置工具，覆盖常见的操作需求：

### 搜索工具

```python
# Tavily 搜索（推荐，专为 AI 设计）
from langchain_community.tools.tavily_search import TavilySearchResults

search = TavilySearchResults(max_results=3)
result = search.invoke("Python 最新版本")
# 返回格式化好的搜索结果列表
```

> **Tavily 注册**：需要去 tavily.com 免费注册获取 API Key，每月有 1000 次免费额度，足够学习和开发使用。

### 数学计算工具

```python
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
result = wiki.invoke("快速排序算法")
# 返回维基百科关于快速排序的文章摘要
```

### 文件操作工具

```python
from langchain_community.tools.file_management.read import ReadFileTool
from langchain_community.tools.file_management.write import WriteFileTool
from langchain_community.tools.file_management.list_dir import ListDirectoryTool

read_tool = ReadFileTool(base_path="./workspace")
write_tool = WriteFileTool(base_path="./workspace")
list_tool = ListDirectoryTool(base_path="./workspace")

# 列出目录内容
files = list_tool.invoke({"path": "."})

# 写入文件
write_tool.invoke({"file_path": "hello.txt", "content": "Hello World!"})

# 读取文件
content = read_tool.invoke({"file_path": "hello.txt"})
```

### Python 代码执行工具

```python
from langchain_experimental.tools import PythonREPLTool

python_repl = PythonREPLTool()

# 执行任意 Python 代码并获取输出
result = python_repl.invoke("""
import math
area = math.pi * 5 ** 2
f"半径为5的圆面积: {area:.2f}"
""")
# 输出: '半径为5的圆面积: 78.54'
```

> **安全警告**：`PythonREPLTool` 会执行任意 Python 代码，在生产环境中需要配合沙箱环境使用，避免恶意代码造成损害。

## 组合多个工具

在实际应用中，Agent 通常同时拥有多个工具可供选择：

```python
from langchain_core.tools import tool

@tool
def search_internet(query: str) -> str:
    """搜索互联网获取实时信息（新闻、天气、价格等）"""
    # 模拟搜索结果
    if "天气" in query or "weather" in query.lower():
        return "北京今天晴，28°C；上海多云，32°C"
    elif "Python" in query:
        return "Python 最新稳定版为 3.12.7，发布于2024年10月"
    else:
        return f"关于'{query}'的搜索结果：未找到精确匹配"

@tool
def calculate(expression: str) -> str:
    """计算数学表达式，支持加减乘除、括号、幂运算等"""
    try:
        result = eval(expression, {"__builtins__": {}})
        return f"计算结果: {expression} = {result}"
    except Exception as e:
        return f"计算错误: {e}"

@tool
def get_current_time() -> str:
    """获取当前的日期和时间"""
    from datetime import datetime
    now = datetime.now()
    return now.strftime("%Y年%m%d日 %H:%M:%S 星期%w")

# 收集所有工具
all_tools = [search_internet, calculate, get_current_time]

# 打印工具概览
for t in all_tools:
    print(f"🔧 {t.name}: {t.description}")
```

输出：

```
🔧 search_internet: 搜索互联网获取实时信息（新闻、天气、价格等）
🔧 calculate: 计算数学表达式，支持加减乘除、括号、幂运算等
🔧 get_current_time: 获取当前的日期和时间
```

当 Agent 同时拥有这三个工具时，面对不同的问题会自动选择最合适的那个：

| 用户提问 | Agent 选择 | 原因 |
|---------|-----------|------|
| "现在几点了？" | `get_current_time` | 问题明确要求时间信息 |
| "175 * 23 + 456 等于多少？" | `calculate` | 这是一个数学表达式 |
| "最新款的 iPhone 价格是多少？" | `search_internet` | 需要实时信息 |

这种自动选择的机制正是 Agent 相对于 Chain 的核心优势——**不需要 if-else 分支来判断该走哪条路，Agent 自己根据语义理解来做决策**。

## 常见误区

**误区一：把所有逻辑都塞进工具里**。有些开发者倾向于在工具内部做大量业务逻辑判断，然后只返回最终结果。这其实限制了 Agent 的灵活性——更好的做法是让工具专注于单一职责，把组合和决策交给 Agent。

```python
# ❌ 工具做了太多事情
@tool
def handle_user_request(request: str) -> str:
    """处理用户的任何请求"""
    if "天气" in request:
        return get_weather(...)
    elif "计算" in request:
        return calculate(...)
    # ... 各种 if-else

# ✅ 每个工具专注一件事，让 Agent 来选
@tool
def get_weather(city: str) -> str: ...
@tool
def calculate(expr: str) -> str: ...
```

**误区二：描述写得像给人看的文档**。工具的描述不是写给人类开发者看的 API 文档，而是写给 LLM "看"的决策依据。应该用简洁明确的语言说明**做什么**和**什么时候用**，而不是详细解释实现原理。

**误区三：忽略错误处理**。工具可能会收到 LLM 传来的不合理参数（比如负数、空字符串、超长输入）。好的工具应该在内部做好校验和容错，返回有意义的错误信息而不是直接抛异常导致整个 Agent 循环中断：

```python
@tool
def divide(a: float, b: float) -> str:
    """计算 a 除以 b 的结果"""
    if b == 0:
        return "错误：除数不能为零"
    if abs(b) < 1e-10:
        return "警告：除数过小，可能导致精度丢失"
    return str(a / b)
```

到这里，我们已经学会了如何定义高质量的工具。下一节我们将把这些工具交给 Agent，学习 ReAct 模式的完整工作流程——看 Agent 如何利用工具自主地完成多步骤任务。
