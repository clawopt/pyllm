---
title: 可视化基础：plot() 快速绘图
description: 折线图/柱状图/散点图/直方图基础用法、figure 与 axes、LLM 数据可视化入门
---
# Pandas 可视化快速入门

Pandas 内置了基于 Matplotlib 的 `plot()` 方法，让你不需要导入 Matplotlib 就能直接从 DataFrame/Series 生成图表。对于数据探索阶段的"看一眼"式可视化来说，这已经完全够用了。

## 基础折线图

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'day': range(1, 31),
    'requests': np.random.poisson(200, 30) + np.sin(np.arange(30)/5)*50,
})

df.plot(x='day', y='requests', title='API 调用量趋势', figsize=(10, 4))
```

`plot()` 默认画折线图。`figsize=(宽, 高)` 控制输出尺寸（单位是英寸）。对于时间序列数据（如 API 调用日志），折线图是最自然的选择——它能直观地展示趋势和波动。

## 柱状图与直方图

```python
df = pd.DataFrame({
    'model': ['GPT-4o', 'Claude', 'Llama', 'Qwen'],
    'score': [88.7, 89.2, 84.5, 83.5],
})

df.plot(kind='bar', x='model', y='score', title='模型评分对比', color='steelblue')

df['latency'] = np.random.normal(800, 150, 1000)
df['latency'].plot(kind='hist', bins=30, title='延迟分布', edgecolor='black')
```

柱状图适合比较分类变量（不同模型的分数），直方图适合查看数值变量的分布形态（延迟是否正态、是否有长尾等）。**在 LLM 场景中，这两种图是最高频的可视化类型**。
