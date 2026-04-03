---
title: PandasAI 简介与环境搭建
description: 什么是 PandasAI / 安装配置 / 核心概念 / 与传统 Pandas 的区别 / 适用场景
---
# PandasAI 简介与环境搭建


## PandasAI 是什么

**PandasAI** 是一个让 Pandas DataFrame 支持**自然语言查询**的库。你可以用人类语言提问，它自动将问题翻译成 Pandas/SQL 代码并执行。

```
传统方式:
  df[df['score'] > 85].groupby('model')['latency'].mean()

PandasAI 方式:
  df.chat("哪些模型的分数大于85，它们的平均延迟是多少？")
```

## 安装与基础配置

```bash
pip install pandasai
```

```python
import pandas as pd
from pandasai import Agent

df = pd.DataFrame({
    "model": ["GPT-4o", "Claude", "Llama", "Qwen", "DeepSeek"],
    "MMLU": [88.7, 89.2, 84.5, 83.5, 86.8],
    "price_per_1M": [12.50, 18.00, 0.87, 1.47, 0.42],
})

agent = Agent(df)

response = agent.chat("哪个模型性价比最高？")
print(response)
```

## 支持的 LLM 后端

```python
from pandasai.llm import OpenAI, AzureOpenAI, GoogleGemini

llm_openai = OpenAI(model="gpt-4o-mini")

llm_gemini = GoogleGemini(model="gemini-2.0-flash")

agent = Agent(df, config={"llm": llm_openai})
```

## 核心能力一览

| 能力 | 示例查询 | 翻译为 |
|------|---------|--------|
| 过滤 | "分数大于90的模型" | `df[df['score'] > 90]` |
| 排序 | "按价格从低到高排序" | `df.sort_values('price')` |
| 聚合 | "各厂商的平均分" | `df.groupby('vendor').mean()` |
| 统计 | "有多少模型超过85分" | `(df['score'] > 85).sum()` |
| 描述 | "描述这个数据集" | `df.describe()` + 文字总结 |
| 可视化 | "画一个分数分布图" | 自动生成 plot |

## 与传统 Pandas 的定位关系

```
┌─────────────────────────────────────┐
│           数据分析师工作流           │
│                                     │
│  ┌──────────┐   ┌──────────────┐  │
│  │ PandasAI │ → │ 生成代码预览  │  │
│  │ (自然语言)│   │  (可审核/修改) │  │
│  └──────────┘   └──────┬───────┘  │
│                         │          │
│                  ┌──────▼───────┐  │
│                  │  Pandas 执行  │  │
│                  │  (精确控制)    │  │
│                  └──────────────┘  │
└─────────────────────────────────────┘
```

**PandasAI 不是替代 Pandas，而是补充**：
- **探索阶段**：用自然语言快速了解数据
- **验证阶段**：审查生成的代码，确保正确性
- **生产阶段**：用原生 Pandas 代码精确执行
```
