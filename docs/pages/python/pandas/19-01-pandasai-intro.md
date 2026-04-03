---
title: PandasAI：用自然语言操作 DataFrame
description: PandasAI 库的安装与使用、LLM 驱动的数据分析、限制与最佳实践
---
# PandasAI：让 LLM 帮你写 Pandas 代码

PandasAI 是一个开源库，它让你可以用**自然语言提问**来操作 DataFrame——你问"哪个模型的平均质量分最高"，它自动生成对应的 Pandas 代码并执行。

## 快速开始

```python
from pandasai import Agent
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'model': ['GPT-4o', 'Claude', 'Llama'],
    'MMLU': [88.7, 89.2, 84.5],
    'price': [15, 15, 0],
})

agent = Agent(df, config={'verbose': True})
result = agent.chat("哪个模型性价比最高？")
print(result)
```

PandasAI 在底层做了几件事：
1. 把你的问题 + DataFrame 的结构信息（列名、dtype、前几行数据）组装成 prompt
2. 发送给 LLM（默认 GPT-4o）
3. LLM 返回生成的 Python/Pandas 代码
4. `agent` 在沙箱环境中执行这段代码并返回结果

## 适用场景与局限

**适合的场景**：
- 探索性分析阶段快速了解数据概况
- 不熟悉 Pandas 语法的团队成员做简单查询
- 快速验证某个假设（"这个模型在代码任务上真的比其他好吗？"）

**不适合的场景**：
- 生产环境中的数据处理流水线（不可靠、有延迟、有成本）
- 复杂的多步骤 ETL 操作
- 对性能和正确性要求极高的场景

**我的建议是把它当作"数据分析助手"而非"生产工具"**——用它来探索和发现，最终确认的分析逻辑还是要写成确定的 Pandas 代码。
