---
title: LangChain + Pandas Agent
description: 用 LangChain 构建 DataFrame 分析 Agent、Tool 定义、ReAct 模式、错误处理
---
# LangChain Pandas Agent：更可控的 LLM 数据分析

PandasAI 是一个"黑盒"——你不知道 LLM 生成了什么代码。LangChain 的 `create_pandas_dataframe_agent` 提供了更透明的方案：你可以自定义工具（Tools）、控制推理过程（ReAct）、甚至限制可用的操作范围。

## 基本用法

```python
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'model': ['GPT-4o', 'Claude', 'Llama', 'Qwen'],
    'MMLU': [88.7, 89.2, 84.5, 83.5],
    'price': [15, 15, 0.27, 0.14],
})

agent = create_pandas_dataframe_agent(
    llm=ChatOpenAI(model='gpt-4o', temperature=0),
    df=df,
    verbose=True,
    allow_dangerous_code=True,
)

response = agent.invoke("哪个模型每分性价比最高？计算公式：MMLU / price")
print(response['output'])
```

和 PandasAI 的关键区别在于 **`verbose=True`**——它会打印出 LLM 的完整思考过程（Thought）和实际执行的代码（Action），让你能**审计**它做了什么。这在调试和分析结果可信度时极其重要。

## 生产环境注意事项

1. **设置 temperature=0**——数据分析需要确定性输出，不要随机性
2. **使用 gpt-4o-mini 降低成本**——简单的查询不需要最强模型
3. **限制数据量**——不要把百万行数据直接喂给 agent，先聚合再传入
4. **添加人工确认步骤**——对修改性操作（删除/写入）要求二次确认
