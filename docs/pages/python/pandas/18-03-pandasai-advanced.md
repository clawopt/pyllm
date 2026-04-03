---
title: PandasAI 高级功能与最佳实践
description: 多 DataFrame 查询 / 自定义技能 / 成本控制 / 安全性 / 与原生 Pandas 协作
---
# PandasAI 高级功能


## 多 DataFrame 联合查询

```python
import pandas as pd
from pandasai import Agent

models = pd.DataFrame({
    "model": ["GPT-4o", "Claude", "Llama", "Qwen"],
    "MMLU": [88.7, 89.2, 84.5, 83.5],
})

pricing = pd.DataFrame({
    "model": ["GPT-4o", "Claude", "Llama", "Qwen", "DeepSeek"],
    "input_price": [2.50, 3.00, 0.27, 0.27, 0.14],
    "output_price": [10.0, 15.0, 0.60, 1.20, 0.28],
})

agent = Agent([models, pricing])

result = agent.chat(
    "结合两个表，找出 MMLU>85 且总价格低于 $5 的模型"
)
print(result)

result2 = agent.chat(
    "哪个模型的性价比(MMLU/总价格)最高？计算过程是什么？"
)
print(result2)
```

## 自定义技能（Skills）

```python
from pandasai import Agent
from pandasai.skills import skill

@skill
def calculate_llm_value_score(mmlu_score, price_per_million):
    """
    计算 LLM 性价比分数。
    Args:
        mmlu_score (Series): MMLU 基准测试分数
        price_per_million (Series): 每百万 token 的价格(美元)
    Returns:
        Series: 性价比分数
    """
    return mmlu_score / price_per_million


df = pd.DataFrame({
    "model": ["GPT-4o", "Claude", "Llama", "Qwen"],
    "MMLU": [88.7, 89.2, 84.5, 83.5],
    "price": [12.50, 18.00, 0.87, 1.47],
})

agent = Agent(df)
agent.chat("用性价比公式计算每个模型的 value_score 并排序")
```

## 成本控制策略

```python
import pandas as pd
from pandasai import Agent
from pandasai.llm import OpenAI

class CostControlledAgent:
    """带成本控制的 PandasAI Agent"""

    def __init__(self, df, budget_usd=1.00):
        self.df = df
        self.budget = budget_usd
        self.spent = 0.0
        self.query_count = 0
        self.cache = {}

        self.agent = Agent(
            df,
            config={
                "llm": OpenAI(model="gpt-4o-mini"),
                "enable_cache": True,
            }
        )

    def ask(self, question):
        if question in self.cache:
            return self.cache[question]

        self.query_count += 1
        result = self.agent.chat(question)

        estimated_cost = 0.0005  # gpt-4o-mini 单次约 $0.0005
        self.spent += estimated_cost

        self.cache[question] = result

        print(f"[查询 #{self.query_count}] 预估花费: ${estimated_cost:.4f} | "
              f"累计: ${self.spent:.4f} / ${self.budget:.2f}")

        if self.spent > self.budget:
            print("⚠️ 接近预算上限，建议切换到原生 Pandas")

        return result


df = pd.DataFrame({
    "model": ["GPT-4o", "Claude", "Llama"],
    "score": [88.7, 89.2, 84.5],
})

controlled = CostControlledAgent(df, budget_usd=0.50)
controlled.ask("哪个模型分最高？")
controlled.ask("平均分是多少？")
controlled.ask("按分数排序")
```

## 安全性注意事项

```python
import pandas as pd
from pandasai import Agent


def sanitize_for_ai(df):
    """为 AI 查询准备安全的数据"""
    safe = df.copy()

    sensitive_patterns = ['api_key', 'token', 'password', 'secret']
    for col in safe.columns:
        if any(p in col.lower() for p in sensitive_patterns):
            safe[col] = '***REDACTED***'

    if len(safe) > 1000:
        safe = safe.head(1000).copy()
        print(f"⚠️ 数据已截断至 1000 行以保护隐私")

    return safe


raw_df = pd.DataFrame({
    "user_id": range(100),
    "prompt": [f"问题{i}" for i in range(100)],
    "api_key": ["sk-xxx"] * 100,
})

safe_df = sanitize_for_ai(raw_df)
agent = Agent(safe_df)
```
