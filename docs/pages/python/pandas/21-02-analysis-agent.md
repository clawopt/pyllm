---
title: 构建数据分析 Agent
description: 完整的 Agent 实现、Tool 自定义、错误处理与重试、多 DataFrame 支持
---
# 构建可用的数据分析 Agent

上一节我们了解了 Agent 的基本架构。这一节把它变成一个**真正可用的工具**——支持自定义分析函数、优雅的错误处理和多表联合查询。

## 完整实现

```python
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
import pandas as pd

class PandasAnalysisAgent:
    def __init__(self, df, model='gpt-4o-mini'):
        self.df = df
        self.agent = create_pandas_dataframe_agent(
            llm=ChatOpenAI(model=model, temperature=0),
            df=df,
            verbose=False,
            max_iterations=5,
            handle_parsing_errors=True,
        )
    
    def ask(self, question):
        try:
            result = self.agent.invoke(question)
            return result['output']
        except Exception as e:
            return f"处理失败: {str(e)[:150]}"
    
    def analyze(self):
        questions = [
            "数据有多少行？每列的类型是什么？",
            "数值列的统计摘要是什么？",
            "有没有缺失值或异常值？",
        ]
        results = {}
        for q in questions:
            results[q] = self.ask(q)
        return results


df = pd.DataFrame({
    'model': ['GPT-4o']*50 + ['Claude']*50,
    'score': [88 + i*0.1 for i in range(100)],
})

agent = PandasAnalysisAgent(df)
for q, a in agent.analyze().items():
    print(f"Q: {q}\nA: {a[:200]}\n")
```

`handle_parsing_errors=True` 让 Agent 在代码解析失败时自动重试而不是直接报错——这在生产环境中非常重要，因为 LLM 生成的代码偶尔会有语法错误。`max_iterations=5` 防止无限循环。
