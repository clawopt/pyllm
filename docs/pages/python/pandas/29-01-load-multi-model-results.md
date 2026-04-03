---
title: 加载多模型评测结果
description: 多源数据合并、结果标准化、缺失值处理、数据对齐
---
# 案例 4：模型评估与对比分析 — 数据加载

```python
import pandas as pd
import numpy as np

np.random.seed(42)
models = ['GPT-4o', 'Claude', 'Llama', 'Qwen', 'DeepSeek']
benchmarks = ['MMLU', 'HumanEval', 'MATH', 'GPQA', 'BBH']

rows = []
for m in models:
    base = {'GPT-4o':88,'Claude':89,'Llama':84,'Qwen':83,'DeepSeek':87}[m]
    for b in benchmarks:
        noise = np.random.randn() * (6 if b in ('MATH','GPQA') else 3)
        rows.append({'model': m, 'benchmark': b, 'score': round(base + noise, 1)})

df = pd.DataFrame(rows)
print(f"数据: {len(df)} 条 ({len(models)} 模型 × {len(benchmarks)} 基准)")
print(f"\n预览:\n{df.head(10)}")
```
