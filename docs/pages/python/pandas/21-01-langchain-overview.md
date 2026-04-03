---
title: LangChain + Pandas 集成概览
description: LangChain 生态与 Pandas 的关系、DataFrame Agent 的原理、适用场景与架构设计
---
# LangChain 与 Pandas：让 LLM 成为你的数据分析师

LangChain 是目前最流行的 LLM 应用开发框架之一，它提供了 `create_pandas_dataframe_agent` 这个专门为 Pandas 设计的 Agent 组件。这一节从架构层面讲清楚它的工作原理和正确使用方式。

## 为什么需要 Agent 而不是直接调用 API

你可能想：既然我可以用 OpenAI API 直接发 prompt 让 LLM 分析数据，为什么还需要 LangChain 的 Agent？核心区别在于**推理循环（ReAct 模式）**：

```
直接调 API:
  你 → [问题 + 数据] → LLM → 回答
  问题：数据太大放不进 context window；LLM 可能"幻觉"出不存在的列名

Agent 模式:
  你 → [问题] → Agent
    Agent → Thought: 我需要先看看有哪些列
    Agent → Action: df.columns
    Agent → Observation: ['id', 'model', 'score', ...]
    Agent → Thought: 现在我需要按 model 分组计算均值
    Agent → Action: df.groupby('model')['score'].mean()
    Agent → Observation: GPT-4o=88.7, Claude=89.2, ...
    Agent → Final Answer: Claude 平均分最高...
```

Agent 不是一次性给出答案，而是**多步推理**——先探索数据结构，再逐步构建查询。这大大降低了错误率。

## 基本架构

```python
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
import pandas as pd

df = pd.DataFrame({
    'model': ['GPT-4o', 'Claude', 'Llama'],
    'MMLU': [88.7, 89.2, 84.5],
})

agent = create_pandas_dataframe_agent(
    llm=ChatOpenAI(model='gpt-4o-mini', temperature=0),
    df=df,
    verbose=True,
    allow_dangerous_code=True,
)
```

三个关键参数：
- **`temperature=0`**：数据分析需要确定性输出
- **`verbose=True`**：打印思考过程（调试时必备，生产环境可关闭）
- **`allow_dangerous_code=True`**：允许执行任意 Pandas 代码（安全警告见下节）
