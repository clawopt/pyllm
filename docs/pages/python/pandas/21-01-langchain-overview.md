---
title: LangChain Agent 与 Pandas 集成概述
description: Pandas 在 LangChain 生态中的角色 / Tool 定义 / DataFrame 作为 Agent 记忆 / 数据分析 Agent 模式
---
# LangChain Agent 集成概览


## Pandas × LangChain：为什么需要集成

LangChain 是 LLM 应用开发框架，但它**不擅长数据操作**。当 Agent 需要：
- 从结构化数据中查找信息
- 对查询结果做统计分析
- 管理多轮对话的历史数据

→ **Pandas 就是 LangChain 的"数据大脑"**

```
┌──────────────────────────────────────────────┐
│              LangChain Agent                 │
│                                               │
│  User Query → LLM Reasoning → Tool Call     │
│                           ↓                  │
│              ┌──────────────┐               │
│              │ Pandas Tool   │ ← 数据查询/   │
│              │ (DataFrame)    │   统计/聚合    │
│              └──────────────┘               │
│                           ↓                  │
│              LLM Response → User             │
└──────────────────────────────────────────────┘
```

## 核心概念：PandasDataFrameTool

```python
from langchain.agents import Agent, Tool
from langchain_experimental.tools import PythonAstREPLTool
import pandas as pd

df = pd.DataFrame({
    "model": ["GPT-4o", "Claude", "Llama", "Qwen", "DeepSeek"],
    "MMLU": [88.7, 89.2, 84.5, 83.5, 86.8],
    "price_per_1M": [12.50, 18.00, 0.87, 1.47, 0.42],
})

try:
    from langchain_experimental.tools.python.tool import PythonAstREPLTool
    tool = PythonAstREPLTool(locals={"df": df})
except ImportError:
    print("需要: pip install langchain-experimental")
```

## 方式二：自定义 Pandas Tool（推荐）

```python
import pandas as pd
import json
from typing import Optional
from langchain_core.tools import BaseTool, Field


class PandasQueryTool(BaseTool):
    """用自然语言查询 Pandas DataFrame"""

    name: str = "pandas_query"
    description: str = (
        "对 Pandas DataFrame 执行数据分析查询。"
        "输入应为自然语言描述的分析需求，如："
        "'找出分数最高的模型'、'计算各厂商的平均分'、'按价格排序'"
    )
    df: pd.DataFrame = Field(description="要分析的 DataFrame")

    def _run(self, query: str) -> str:
        try:
            query_lower = query.lower().strip()

            if "最贵" in query or "价格最高" in query or "most expensive" in query:
                result = self.df.nlargest(3, 'price_per_1M')[['model', 'price_per_1M']]
                return result.to_markdown()

            elif "最便宜" in query or "cheapest" in query:
                result = self.df.nsmallest(3, 'price_per_1M')[['model', 'price_per_1M']]
                return result.to_markdown()

            elif "最高分" in query or "best" in query or "top" in query:
                score_col = [c for c in self.df.columns if 'score' in c.lower() or c == 'MMLU']
                if score_col:
                    result = self.df.nlargest(3, score_col[0])
                    return result.to_markdown()

            elif "平均" in query or "average" in query or "mean" in query:
                numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
                if numeric_cols:
                    result = self.df[numeric_cols].mean()
                    return f"平均值:\n{result.round(2).to_string()}"

            elif "统计" in query or "describe" in query or "summary" in query:
                desc = self.df.describe()
                return f"统计摘要:\n{desc.round(2).to_string()}"

            elif "全部" in query or "all" in query or "show" in query:
                return f"完整数据:\n{self.df.to_markdown()}"

            else:
                return f"无法理解查询: '{query}'。支持的操作包括: 最贵/最便宜/最高分/平均/统计/全部"

        except Exception as e:
            return f"执行出错: {str(e)}"


tool = PandasQueryTool(df=df)
print(tool._run("哪个模型最贵？"))
print("\n" + tool._run("MMLU 最高分的前三名"))
print("\n" + tool._run("给我数据的统计摘要"))
```

## Agent 配置与运行

```python
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Agent

eval_df = pd.DataFrame({
    "model": ["GPT-4o", "Claude-3.5-Sonnet", "Llama-3.1-70B",
              "Qwen2.5-72B", "DeepSeek-V3"],
    "MMLU": [88.7, 89.2, 84.5, 83.5, 86.8],
    "HumanEval": [92.0, 93.1, 82.4, 78.6, 87.2],
    "price": [12.50, 18.00, 0.87, 1.47, 0.42],
})

tools = [PandasQueryTool(df=eval_df)]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
agent = Agent(tools=tools, llm=agent)

def ask_agent(question):
    """向 Agent 提问"""
    try:
        response = agent.invoke({"input": question})
        return response["output"]
    except Exception as e:
        return f"Agent 错误: {e}"


print("=== LangChain + Pandas Agent ===\n")
questions = [
    "哪些模型的 MMLU 分数超过 85？",
    "性价比最高的模型是哪个？",
    "画一个各模型 MMLU 分数的对比",
]

for q in questions:
    print(f"Q: {q}")
    print(f"A: {ask_agent(q)}\n")
```

## Pandas 在 Agent 架构中的典型角色

| 角色 | 说明 | 示例 |
|------|------|------|
| **知识库** | Agent 查询的结构化数据源 | 产品目录、API 文档 |
| **记忆体** | 存储对话历史和用户偏好 | 用户画像、历史交互 |
| **日志表** | 记录每次 Tool Call | 审计追踪、调试分析 |
| **状态管理** | 维护 Agent 运行时状态 | 任务进度、上下文窗口 |

## 设计原则

```python
class AgentDataLayer:
    """Agent 数据层设计原则"""

    PRINCIPLES = {
        1: "DataFrame 应该是只读的快照，避免并发修改",
        2: "每个 Tool 调用应该有明确的输入输出契约",
        3: "返回值应该是可序列化的（JSON/字符串），不要返回原始 DataFrame",
        4: "大数据集应先做聚合再传给 LLM（避免 token 超限）",
        5: "敏感列在传入 Tool 前必须脱敏或移除",
    }

    @staticmethod
    def prepare_for_agent(df, max_rows=100, sensitive_cols=None):
        """将 DataFrame 准备为 Agent 友好格式"""
        safe = df.head(max_rows).copy()

        if sensitive_cols:
            for col in sensitive_cols:
                if col in safe.columns:
                    safe[col] = '***'

        return safe
```
