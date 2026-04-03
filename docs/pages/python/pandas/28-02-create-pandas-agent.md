---
title: 创建 Pandas Agent
description: Agent 初始化、Tool 定义、错误处理策略、多 DataFrame 支持
---
# 创建 Agent

```python
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
import pandas as pd

df = pd.DataFrame({
    'model': ['GPT-4o', 'Claude', 'Llama'] * 30,
    'score': [88 + i*0.3 for i in range(90)],
    'latency': [800 + i*10 for i in range(90)],
})

agent = create_pandas_dataframe_agent(
    llm=ChatOpenAI(model='gpt-4o-mini', temperature=0),
    df=df,
    verbose=False,
    max_iterations=5,
    handle_parsing_errors=True,
)

response = agent.invoke("哪个模型平均分最高？延迟 P95 是多少？")
print(response['output'])
```
