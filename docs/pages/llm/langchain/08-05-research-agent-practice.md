---
title: 实战：构建一个全能研究助手 Agent
description: 整合搜索/计算/文件操作/RAG 检索，搭建能自主完成研究任务的智能体
---
# 实战：构建一个全能研究助手 Agent

前面四节我们从概念到工具再到 ReAct 模式，逐步掌握了构建 Agent 的全部要素。这一节我们将综合所有知识，从零搭建一个**有实际价值的研究助手 Agent**——它能自主地搜索信息、做数据分析、读写文件、查询知识库，最终生成结构化的研究报告。

## 功能设计

我们的研究助手将具备以下能力：

| 能力 | 对应工具 | 用途 |
|------|---------|------|
| 互联网搜索 | `search_web` | 获取实时信息 |
| 数学计算 | `calculator` | 数据分析、统计运算 |
| 文件读写 | `read_file` / `write_file` | 保存中间结果和报告 |
| 知识库检索 | `knowledge_search` | 从内部文档中查找资料 |
| 当前时间 | `get_time` | 标注报告时间 |

## 第一步：定义所有工具

```python
"""
research_agent.py — 全能研究助手
"""
import os
import json
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_react_agent, AgentExecutor

load_dotenv()

WORKSPACE = "./agent_workspace"
os.makedirs(WORKSPACE, exist_ok=True)


@tool
def search_web(query: str) -> str:
    """
    搜索互联网获取最新信息。
    
    适用场景：新闻、天气、产品价格、技术动态、体育比分等需要实时数据的问题。
    返回格式：包含关键信息的文本摘要。
    
    注意：不要用这个工具查数学问题或常识。
    """
    # 模拟搜索引擎（实际项目中替换为 Tavily / Google Search API）
    knowledge = {
        "langchain": (
            "LangChain 是 LLM 应用开发框架，2025年4月最新版本 v0.3.x。"
            "核心功能包括 Model I/O、RAG、Agent、Memory、Chain 五大模块。"
            "GitHub 星标超过 90K。"
        ),
        "python": (
            "Python 3.12 是当前稳定版（2024年10月发布）。"
            "主要新特性：更好的错误消息、性能改进、类型系统增强。"
        ),
        "gpt": (
            "GPT-4o 是 OpenAI 的多模态模型（2024年5月发布）。"
            "支持文本、图像、音频输入。API 价格约为 GPT-4 的 50%。"
        ),
        "天气": (
            "北京：晴 28°C；上海：多云 32°C；广州：雷阵雨 30°C；深圳：阴 29°C"
        ),
    }
    query_lower = query.lower()
    for key, value in knowledge.items():
        if key in query_lower:
            return value
    return f"未找到关于 '{query}' 的相关信息。请尝试更具体的关键词。"


@tool
def calculate(expression: str) -> str:
    """
    计算数学或统计表达式。
    
    支持的运算：加减乘除、幂运算(**)、绝对值(abs)、最大最小值(max/min)、四舍五入(round)、平方根(sqrt)。
    示例表达式：
    - "(175 * 23 + 456) / 10"
    - "max(15, 28, 7, 42)"
    - "round(3.14159 * 10 ** 2)"
    - "sqrt(144)"
    """
    import math
    safe_dict = {
        "__builtins__": {},
        "abs": abs, "max": max, "min": min,
        "round": round, "pow": pow, "sum": sum,
        "len": len, "sqrt": math.sqrt,
        "pi": math.pi, "e": math.e
    }
    try:
        result = eval(expression, safe_dict)
        return json.dumps({
            "expression": expression,
            "result": float(result) if isinstance(result, (int, float)) else str(result),
            "type": type(result).__name__
        }, ensure_ascii=False)
    except ZeroDivisionError:
        return json.dumps({"error": "除数不能为零"})
    except Exception as e:
        return json.dumps({"error": f"计算失败: {str(e)}"})


@tool
def write_report(filename: str, content: str) -> str:
    """
    将内容写入工作区中的文件。
    
    Args:
        filename: 文件名（如 "report.md"、"data.json"）
        content: 要写入的内容
    
    用于保存研究报告、分析结果等中间产物。
    """
    filepath = os.path.join(WORKSPACE, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    size_kb = os.path.getsize(filepath) / 1024
    return f"文件已保存: {filepath} ({size_kb:.1f} KB)"


@tool
def read_file(filename: str) -> str:
    """
    读取工作区中的文件内容。
    
    Args:
        filename: 要读取的文件名
    
    可用于查看之前写入的报告或数据文件。
    """
    filepath = os.path.join(WORKSPACE, filename)
    if not os.path.exists(filepath):
        return f"错误: 文件 '{filename}' 不存在。可用文件: {os.listdir(WORKSPACE)}"
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    if len(content) > 3000:
        content = content[:3000] + "\n... (内容过长，已截断)"
    return content


@tool
def get_current_time() -> str:
    """获取当前日期和时间"""
    now = datetime.now()
    weekdays = ["一", "二", "三", "四", "五", "六", "日"]
    return now.strftime(f"%Y年%m月%d日 %H:%M:%S 星期{weekdays[now.weekday()]}")


@tool
def list_workspace_files() -> str:
    """列出工作区中的所有文件"""
    files = os.listdir(WORKSPACE)
    if not files:
        return "工作区为空，还没有任何文件"
    result = []
    for f in sorted(files):
        fp = os.path.join(WORKSPACE, f)
        size = os.path.getsize(fp)
        mtime = datetime.fromtimestamp(os.path.getmtime(fp)).strftime("%m-%d %H:%M")
        result.append(f"  {f:30s} {size:>8d} bytes  {mtime}")
    return f"工作区文件 ({len(files)} 个):\n" + "\n".join(result)


# 收集所有工具
all_tools = [
    search_web, calculate, write_report,
    read_file, get_current_time, list_workspace_files
]
```

注意几个工具设计的细节：

- `search_web` 使用了模拟数据——在实际项目中替换为真实搜索 API 即可
- `calculate` 返回 JSON 格式结果，方便 Agent 解析 expression 和 result
- `write_report` 和 `read_file` 都限制在 `WORKSPACE` 目录内，保证安全性
- `list_workspace_files` 让 Agent 能知道自己之前创建了哪些文件

## 第二步：创建并配置 Agent

```python
# 创建 LLM
chat = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 创建 ReAct Agent
agent = create_react_agent(
    model=chat,
    tools=all_tools,
    prompt="""你是一个专业的研究助手。你的工作流程：

1. 分析用户的研究需求
2. 使用 search_web 获取必要的信息
3. 使用 calculate 进行数据分析（如果需要）
4. 将研究结果整理成结构化的报告
5. 使用 write_report 保存报告到文件
6. 告诉用户报告已保存的位置

要求：
- 报告要包含数据来源和分析过程
- 所有数值计算必须使用 calculator 工具验证
- 报告使用 Markdown 格式
- 最后一定要调用 write_report 保存结果
"""
)

# 包装为 AgentExecutor（获得 verbose 和 max_iterations 控制）
executor = AgentExecutor(
    agent=agent["agent"],
    tools=all_tools,
    verbose=True,
    max_iterations=15,
    handle_parsing_errors=True
)
```

这里有几个重要的配置参数：

- **`verbose=True`**：打印每一轮 Thought/Action/Observation，调试时必备
- **`max_iterations=15`**：限制最大循环次数，防止死循环
- **`handle_parsing_errors=True`**：当 LLM 输出的工具调用格式不正确时自动重试而不是报错退出

## 第三步：运行测试

### 测试一：简单的单步任务

```python
result = executor.invoke({"input": "现在几点了？"})
print(result["output"])
```

输出：

```
> Entering new AgentExecutor chain...
Invocation: get_current_time[]

Response: 2025年04月06日 14:35:22 星期日

Final Answer: 现在的时间是 **2025年04月06日 14:35:22 星期日**
```

### 测试二：需要搜索 + 计算的多步任务

```python
result = executor.invoke({
    "input": "帮我做一个关于 Python 和 LangChain 流行度的对比分析，"
           "然后写一份简短的报告保存下来"
})
print(result["output"])
```

Agent 内部的执行过程（verbose 输出）大概是这样的：

```
> Entering new AgentExecutor chain...

Thought: 用户需要一个关于 Python 和 LangChain 的对比分析报告。
我需要先分别搜索这两个主题的最新信息。

Action: search_web[{"query": "Python 最新版本和流行度"}]
Observation: Python 3.12 是当前稳定版（2024年10月发布）...
主要新特性：更好的错误消息、性能改进、类型系统增强。

Thought: 已经获得了 Python 的信息。现在搜索 LangChain。

Action: search_web[{"query": "LangChain 最新版本和流行度"}]  
Observation: LangChain 是 LLM 应用开发框架，2025年4月最新版本 v0.3.x...
GitHub 星标超过 90K。

Thought: 我已经收集了足够的信息。现在整理成报告并保存。

Action: write_report[{"filename": "comparison_report.md"}, {"content": "# Python vs LangChain 流行度分析\n\n## Python\n- 最新版本: 3.12 (2024年10月)\n- 特点: ...\n\n## LangChain\n- 最新版本: v0.3.x (2025年4月)\n- GitHub: 90K+ stars\n..."}]
Observation: 文件已保存: ./agent_workspace/comparison_report.md (1.2 KB)

Thought: 报告已经成功保存。我可以给用户一个总结性的回答了。

Final Answer: ✅ 研究报告已完成并保存！
📄 文件位置: `./agent_workspace/comparison_report.md`

报告摘要：
- **Python**: 最新版 3.12，以简洁易学和生态丰富著称
- **LangChain**: 最新版 v0.3.x，GitHub 90K+ stars，LLM 开发首选框架
- 两者在 AI 开发领域高度互补
```

### 测试三：复杂数据分析任务

```python
result = executor.invoke({
    "input": (
        "假设我们团队有三个项目：项目A 本季度营收 125 万元，"
        "项目B 营收 87 万元，项目C 营收 203 万元。"
        "请计算总营收、各项目占比、平均值和标准差，"
        "然后把完整分析报告保存为 financial_analysis.md"
    )
})
print(result["output"])
```

Agent 会自动：
1. 用 `calculate` 计算 total = 125 + 87 + 203
2. 用 `calculate` 计算每个项目的占比 (如 125/total * 100)
3. 用 `calculate` 计算平均值 mean = total / 3
4. 用 `calculate` 计算方差和标准差
5. 把所有结果组织成 Markdown 报告
6. 用 `write_report` 保存到文件

这就是 Agent 的威力——你只需要用自然语言描述需求，它自己规划步骤、选择工具、执行计算、保存结果。

## 第四步：交互式 CLI 版本

把 Agent 包装成一个命令行交互程序：

```python
def main():
    print("=" * 56)
    print("   🔬 全能研究助手 Agent")
    print("=" * 56)
    print("  支持能力:")
    print("  🔍 互联网搜索   🧮 数学计算")
    print("  📝 写入报告     📖 读取文件")
    print("  🕐 查询时间     📂 列出文件")
    print("=" * 56)

    while True:
        try:
            user_input = input("\n❓ 你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 再见！")
            break

        if not user_input:
            continue
        if user_input.lower() in ["退出", "exit", "quit"]:
            print("👋 再见！")
            break

        print("\n🤖 正在思考...")
        try:
            result = executor.invoke({"input": user_input})
            print(f"\n💡 {result['output']}")
        except Exception as e:
            print(f"\n⚠️  出错了: {e}")


if __name__ == "__main__":
    main()
```

## 完整项目结构

```
research-agent/
├── .env                          # OPENAI_API_KEY
├── .gitignore
├── requirements.txt
├── research_agent.py             # 主程序
└── agent_workspace/              # Agent 的工作目录（自动创建）
    ├── comparison_report.md      # Agent 生成的报告
    └── financial_analysis.md     # Agent 生成的报告
```

## 性能与成本优化建议

Agent 的成本远高于普通 Chain（因为每轮循环都要调一次 LLM），以下是一些实用的优化策略：

**策略一：减少不必要的工具轮次**

通过优化工具描述和 Prompt 来帮助 Agent 更快地做出正确决策：

```python
# 在工具描述中加入使用示例
@tool
def calculate(expr: str) -> str:
    """计算数学表达式。
    
    示例:
    - 输入 "175*23+456" → 输出 "4481"
    - 输入 "max(10,20,30)" → 输出 "30"
    
    注意：只用于纯数学计算，不要用来处理字符串。
    """
```

**策略二：使用更快的模型做简单决策**

对于不需要复杂推理的场景，可以用 gpt-4o-mini 替代 gpt-4o：

```python
# 简单任务 — 快且便宜
simple_chat = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 复杂任务 — 慢但聪明
smart_chat = ChatOpenAI(model="gpt-4o", temperature=0)
```

**策略三：缓存重复的工具调用**

如果同一个 Agent 可能多次调用相同参数的工具，可以加入缓存层：

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def _cached_search(query: str) -> str:
    # 实际的搜索逻辑
    ...

@tool  
def search_web(query: str) -> str:
    """搜索互联网（带缓存）"""
    return _cached_search(query)
```

到这里，第七章的全部内容就结束了。我们从 Agent 与 Chain 的本质区别出发，学习了如何定义高质量的工具、如何用 ReAct 模式驱动 Agent 自主决策，最终构建了一个能搜索、计算、读写的全能研究助手。Agent 是 LangChain 中最强大也最复杂的概念——掌握它意味着你已经具备了构建真正智能应用的核心理念。接下来的一章，我们将学习 Chain 编排的高级技巧，让多个组件以更灵活的方式协同工作。
