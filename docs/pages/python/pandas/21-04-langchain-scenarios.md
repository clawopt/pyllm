---
title: Agent 实战场景
description: 自动化报告生成、交互式数据问答、集成到 Web 应用中
---
# Agent 实战：从原型到产品

这一节把前面学的 Agent 技术整合成一个**可以直接部署的数据分析工具**。

## 场景：自动生成周报

```python
import pandas as pd
import numpy as np

np.random.seed(42)
weekly = pd.DataFrame({
    'date': pd.date_range('2025-01-06', periods=7),
    'api_calls': [1200+int(i*50+np.random.randn()*100) for i in range(7)],
    'avg_latency': [800 + np.random.randn()*100 for _ in range(7)],
    'error_rate': [0.02 + np.random.rand()*0.02 for _ in range(7)],
})

agent = PandasAnalysisAgent(weekly)

report = agent.ask("""
请分析这周的 API 运营数据，生成一份包含以下内容的中文报告：
1. 核心指标概览（调用量/延迟/错误率）
2. 与上周相比的变化趋势（如果有增长或下降，说明百分比）
3. 需要关注的异常点（如延迟突然升高或错误率飙升）
4. 下周建议
""")

print(report)
```

这个场景展示了 Agent 的真正价值——它不只是执行查询，而是**理解业务上下文并生成可读的分析报告**。对于需要每周写运营汇报的团队来说，这种自动化能节省大量时间。
