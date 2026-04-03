---
title: merge() 基础：四种连接类型
description: inner/left/right/full outer 连接的原理与选择、on/how 参数、多键合并、LLM 场景实战
---
# 数据合并：把分散的数据拼在一起

在 LLM 开发中，你的数据几乎总是来自多个来源——模型评估分数在一个表里，API 调用成本在另一个表里，人工标注质量又在第三个表里。`merge()` 就是 Pandas 中把这些数据"拼"在一起的核心工具。它的设计灵感直接来自 SQL 的 `JOIN` 操作。

## 四种连接类型：一张图讲清楚

```
左表 (df_a)              右表 (df_b)
┌──────┬──────┐          ┌──────┬──────┐
│  key │ val_a│          │  key │ val_b│
├──────┼──────┤          ├──────┼──────┤
│   A  │   1  │          │   A  │  10  │
│   B  │   2  │          │   B  │  20  │
│   C  │   3  │          │   D  │  40  │
│   D  │   4  │          │   E  │  50  │
└──────┴──────┘          └──────┴──────┘

inner (交集):     A, B        ← 只保留两边都有的 key
left outer:       A, B, C, D  ← 保留左边所有，右边没有的填 NaN
right outer:      A, B, D, E  ← 保留右边所有，左边没有的填 NaN
full outer:       A,B,C,D,E   ← 保留所有 key，缺失处填 NaN
```

## 基本用法

```python
import pandas as pd

models = pd.DataFrame({
    'model': ['GPT-4o', 'Claude', 'Llama', 'Qwen'],
    'params_B': [1760, 175, 70, 72],
})

scores = pd.DataFrame({
    'model': ['GPT-4o', 'Claude', 'Llama', 'Gemini'],
    'MMLU': [88.7, 89.2, 84.5, 86.8],
})

inner = pd.merge(models, scores, on='model', how='inner')
print(f"内连接: {len(inner)} 行 (Gemini 和 Qwen 被排除)")

left = pd.merge(models, scores, on='model', how='left')
print(f"左连接: {len(left)} 行 (Gemini 的 MMLU 为 NaN)")
```

`how='inner'`（默认）只保留两边都有匹配的行——最严格也最安全，适合你需要完整数据的场景。`how='left'` 保留左表所有行——适合"以主表为基础补充信息"的场景，比如用模型信息表为主、去关联评测分数。
