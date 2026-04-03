---
title: merge 实战场景
description: 模型评估数据关联、SFT 数据与标注结果合并、RAG 检索日志关联用户反馈
---
# 实战：LLM 场景中的 merge 应用

前面四节我们学了 merge 的基础、高级用法、join 和性能优化。这一节用三个真实的 LLM 工程场景把它们串起来。

## 场景一：模型评估报告

```python
import pandas as pd
import numpy as np

np.random.seed(42)
models = pd.DataFrame({
    'model': ['GPT-4o', 'Claude', 'Llama', 'Qwen'],
    'params_B': [1760, 175, 70, 72],
    'context': [128000, 200000, 131072, 131072],
})

evals = pd.DataFrame({
    'model': ['GPT-4o', 'Claude', 'Llama', 'Gemini'],
    'MMLU': [88.7, 89.2, 84.5, 86.8],
    'HumanEval': [92.0, 93.1, 82.4, 78.6],
    'latency_p95_ms': [1200, 980, 450, 520],
    'price_per_1M': [15, 15, 0.27, 0.14],
})

report = pd.merge(models, evals, on='model', how='left')
report['cost_per_point'] = (
    report['price_per_1M'] / report['MMLU']
).round(3)
report = report.sort_values('MMLU', ascending=False)

print("=== 模型综合评估 ===")
print(report.round(1).to_string())
```

这个场景展示了 `merge` 最经典的用法：以模型信息表为主（左表），左连接评测分数表（右表），然后基于合并后的完整数据计算派生指标（如"每分成本"）。

## 场景二：SFT 数据关联人工标签

```python
conversations = pd.DataFrame({
    'conv_id': ['c001', 'c002', 'c003', 'c004'],
    'prompt': ['什么是AI', '解释Python', '如何学习', '推荐书籍'],
    'response': ['AI是...', 'Python是...', '建议从...', '推荐...'],
})

labels = pd.DataFrame({
    'conv_id': ['c001', 'c002', 'c003'],
    'human_score': [4.5, 3.8, 4.2],
    'annotator': ['alice', 'bob', 'alice'],
})

labeled_data = pd.merge(conversations, labels, on='conv_id', how='left')
unlabeled = labeled_data[labeled_data['human_score'].isna()]

print(f"已标注: {len(labeled_data) - len(unlabeled)}")
print(f"未标注: {len(unlabeled)}")
```

`how='left'` 确保即使没有标签的对话也不会丢失——它们只是 `human_score` 为 NaN。这在数据质量管理中是标准做法。
