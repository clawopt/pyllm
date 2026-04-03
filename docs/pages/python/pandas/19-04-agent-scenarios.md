---
title: Agent 实战场景
description: 构建 LLM 数据分析助手、自动报告生成、交互式数据问答系统
---
# 实战：构建数据分析 Agent

这一节把 PandasAI 和 LangChain Agent 的知识整合起来，构建一个**可实际部署的 LLM 数据分析助手**。

## 端到端实现

```python
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
import pandas as pd

class DataAnalysisAgent:
    def __init__(self, df):
        self.df = df
        self.agent = create_pandas_dataframe_agent(
            llm=ChatOpenAI(model='gpt-4o-mini', temperature=0),
            df=df,
            verbose=False,
            max_iterations=5,
        )
    
    def ask(self, question):
        try:
            response = self.agent.invoke(question)
            return response['output']
        except Exception as e:
            return f"抱歉，处理时出错: {str(e)[:100]}"
    
    def batch_analyze(self, questions):
        results = {}
        for q in questions:
            results[q] = self.ask(q)
            print(f"✓ {q}")
        return results


df = pd.DataFrame({
    'model': ['GPT-4o']*100 + ['Claude']*100,
    'score': np.random.uniform(80, 95, 200),
})

agent = DataAnalysisAgent(df)
answers = agent.batch_analyze([
    "各模型的平均分是多少？",
    "哪个模型分数超过90的样本最多？",
    "两个模型分数的分布有什么差异？",
])

for q, a in answers.items():
    print(f"\nQ: {q}")
    print(f"A: {a[:200]}")
```

这个 Agent 类封装了 LangChain 的 DataFrame Agent，提供了 `ask()` 单次查询和 `batch_analyze()` 批量查询两种接口。`max_iterations=5` 防止 Agent 在错误路径上无限循环。**这是你在内部工具或演示项目中可以直接使用的模板**。
