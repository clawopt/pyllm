---
title: PandasAI 在 LLM 开发中的实战
description: 模型评估数据分析 / API 日志查询 / SFT 数据集探索 / 与原生 Pandas 协作工作流
---
# PandasAI 实战案例


## 场景一：模型评估数据快速探索

```python
import pandas as pd
import numpy as np

class EvalDataExplorer:
    """基于 PandasAI 的评估数据探索器"""

    def __init__(self, eval_df):
        self.df = eval_df.copy()
        self.agent = None

    def setup_agent(self):
        from pandasai import Agent
        self.agent = Agent(self.df)
        return self

    def explore(self):
        if not self.agent:
            self.setup_agent()

        questions = [
            "这个数据集有哪些列？每列的含义是什么？",
            "数据集中有多少个模型？每个模型有多少条记录？",
            "各模型的 MMLU 平均分是多少？排序看看",
            "哪个模型在 HumanEval 上表现最好？最差的是谁？",
            "有没有哪个模型在所有基准测试中都排前三？",
            "画一个热力图展示各模型在各任务上的得分",
            "性价比（分数/价格）最高的模型是哪个？",
        ]

        print("=" * 55)
        print("PandasAI 评估数据探索")
        print("=" * 55)

        for i, q in enumerate(questions, 1):
            print(f"\n[Q{i}] {q}")
            try:
                answer = self.agent.chat(q)
                print(f"    → {answer}")
            except Exception as e:
                print(f"    → 错误: {e}")

        return self


np.random.seed(42)
eval_data = pd.DataFrame({
    'model': np.random.choice(
        ['GPT-4o', 'Claude-3.5-Sonnet', 'Llama-3.1-70B',
         'Qwen2.5-72B', 'DeepSeek-V3'], 100
    ),
    'benchmark': np.random.choice(
        ['MMLU', 'HumanEval', 'MATH', 'GPQA', 'BBH'], 100
    ),
    'score': np.random.uniform(65, 95, 100).round(1),
    'latency_ms': np.random.randint(200, 3000, 100),
    'price_per_1M': np.random.uniform(0.14, 18.0, 100).round(2),
})

explorer = EvalDataExplorer(eval_data)
explorer.explore()
```

## 场景二：PandasAI + 原生 Pandas 协作模式

```python
import pandas as pd
import numpy as np

class HybridAnalyzer:
    """PandasAI 探索 + Pandas 精确执行"""

    def __init__(self, df):
        self.original = df
        self.working = df.copy()

    def phase1_explore(self, questions):
        """阶段一：用自然语言快速了解数据"""
        from pandasai import Agent
        agent = Agent(self.working.head(200))

        insights = []
        for q in questions:
            try:
                result = agent.chat(q)
                insights.append({'question': q, 'ai_answer': str(result)})
                code = agent.generate_code(q)
                insights[-1]['generated_code'] = code
            except Exception as e:
                insights.append({'question': q, 'error': str(e)})

        return insights

    def phase2_execute(self, verified_code):
        """阶段二：用审查后的代码精确执行"""
        local_vars = {'df': self.working, 'pd': pd, 'np': np}
        exec(verified_code, local_vars)
        return local_vars.get('result', self.working)


df = pd.DataFrame({
    'model': ['GPT-4o', 'Claude', 'Llama', 'Qwen', 'DeepSeek'] * 20,
    'score': np.random.uniform(75, 95, 100),
    'cost': np.random.uniform(0.5, 15.0, 100),
})

analyzer = HybridAnalyzer(df)

insights = analyzer.phase1_explore([
    "哪些模型的 score > 88 且 cost < $5？",
])

for ins in insights:
    print(f"\nQ: {ins['question']}")
    print(f"A: {ins.get('ai_answer', ins.get('error', '?'))}")
    if 'generated_code' in ins:
        print(f"Code:\n{ins['generated_code']}")

print("\n" + "="*50)
print("协作流程总结:")
print("  1. 用 PandasAI 自然语言探索 → 快速理解")
print("  2. 审查生成的代码 → 确保逻辑正确")
print("  3. 用原生 Pandas 执行 → 生产级精确性")
```
