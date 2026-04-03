---
title: PandasAI 核心用法
description: chat() 查询 / plot() 可视化 / charts() 图表 / generate_code() 代码生成 / 多轮对话
---
# PandasAI 核心用法


## 基础查询：chat()

```python
import pandas as pd
from pandasai import Agent

df = pd.DataFrame({
    "model": ["GPT-4o", "Claude", "Llama", "Qwen", "DeepSeek", "Gemini"],
    "vendor": ["OpenAI", "Anthropic", "Meta", "Alibaba", "DeepSeek", "Google"],
    "MMLU": [88.7, 89.2, 84.5, 83.5, 86.8, 87.3],
    "HumanEval": [92.0, 93.1, 82.4, 78.6, 87.2, 85.1],
    "price_per_1M": [12.50, 18.00, 0.87, 1.47, 0.42, 6.25],
})

agent = Agent(df)

result = agent.chat("列出所有模型及其 MMLU 分数")
print(result)

result = agent.chat("哪些模型的 MMLU 超过 87？")
print(result)

result = agent.chat("按价格从低到高排列，显示模型名和价格")
print(result)
```

## 数据描述与统计

```python
summary = agent.chat("总结这个数据集的关键信息")
print(summary)

stats = agent.chat("各 vendor 的平均 MMLU 和 HumanEval 是多少？")
print(stats)
```

## 代码生成：generate_code()

```python
code = agent.generate_code(
    "找出 MMLU 最高的前3个模型，并计算它们的性价比(分数/价格)"
)
print(code)

```

## 可视化：plot() / charts()

```python
agent.plot("画一个各模型 MMLU 分数的柱状图")

agent.charts(
    "画一个散点图，x轴是价格，y轴是MMLU分数，"
    "每个点代表一个模型，用不同颜色区分厂商"
)

agent.plot("画一个分组柱状图，对比每个模型的 MMLU 和 HumanEval")
```

## 多轮对话

```python
agent = Agent(df)

response1 = agent.chat("哪个模型最贵？")

response2 = agent.chat("那最便宜的呢？")

response3 = agent.chat("它们的分数差多少？")
```

## 配置选项

```python
from pandasai import Agent
from pandasai.llm import OpenAI

agent = Agent(
    df,
    config={
        "llm": OpenAI(model="gpt-4o-mini"),
        "verbose": True,           # 打印详细日志
        "save_charts": True,       # 保存图表到文件
        "max_retries": 3,
        "enable_cache": True,      # 缓存相同问题的回答
    }
)
```
