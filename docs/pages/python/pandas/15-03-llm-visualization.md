---
title: LLM 数据可视化实战
description: 模型评分雷达图、训练损失曲线、数据质量分布图、API 监控仪表盘
---
# 实战：LLM 场景的可视化方案

这一节把 Pandas 绘图能力用在三个真实的 LLM 工程场景中。

## 场景一：模型评分对比

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

models = ['GPT-4o', 'Claude', 'Llama', 'Qwen']
metrics = ['MMLU', 'HumanEval', 'MATH', 'GPQA']

df = pd.DataFrame(np.random.uniform(70, 95, (len(models), len(metrics))),
                  index=models, columns=metrics)

ax = df.T.plot(kind='bar', figsize=(10, 5), rot=0,
                 title='模型评测对比', colormap='Set2')
ax.set_ylabel('Score')
```

转置（`.T`）让模型变成图例、评测指标变成 X 轴——这样每根柱子代表一个模型的某个维度分数，同一维度的不同模型柱子并排排列，对比一目了然。

## 场景二：训练曲线

```python
epochs = list(range(1, 51))
train_loss = [2.5 * np.exp(-i/10) + 0.1*np.random.randn() for i in epochs]
val_loss   = [2.7 * np.exp(-i/12) + 0.15*np.random.randn() for i in epochs]

df = pd.DataFrame({'epoch': epochs, 'train_loss': train_loss, 'val_loss': val_loss})
df.plot(x='epoch', y=['train_loss', 'val_loss'], title='Training Curve')
```

同时画 train 和 validation 的 loss 曲线，观察两者之间的 gap——gap 过大意味着过拟合，两者都居高不下则可能欠拟合。**这是每个 LLM 训练工程师每天都要看的图**。
