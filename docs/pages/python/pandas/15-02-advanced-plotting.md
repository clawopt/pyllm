---
title: 高级绘图技巧
description: 多子图布局、颜色映射、标注与样式、保存图片
---
# 高级绘图：多子图与样式控制

单张图表往往不够——你需要把多个指标放在同一张图里对比，或者用多个子图展示不同维度的信息。

## 多子图

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame({
    'model': ['GPT-4o']*5 + ['Claude']*5,
    'latency': np.concatenate([np.random.normal(800, 150, 5),
                               np.random.normal(650, 120, 5)]),
})

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

df[df['model']=='GPT-4o']['latency'].plot(kind='hist', ax=axes[0],
                                          title='GPT-4o 延迟', alpha=0.7)
df[df['model']=='Claude']['latency'].plot(kind='hist', ax=axes[1],
                                           title='Claude 延迟', alpha=0.7)

plt.tight_layout()
```

`plt.subplots(行数, 列数)` 创建一个多子图画布，每个子图是一个 `axes` 对象。通过 `ax=` 参数把 Pandas 的 plot 输出定向到指定的子图中。这在"并排对比两个模型的延迟分布"这类场景中非常实用。

## 散点图与颜色映射

```python
df = pd.DataFrame({
    'quality': np.random.uniform(1, 5, 100),
    'tokens': np.random.randint(20, 2000, 100),
    'source': np.random.choice(['api', 'web'], 100),
})

df.plot.scatter(x='tokens', y='quality', c='source',
                cmap='viridis', alpha=0.6, figsize=(8, 5))
```

散点图的 `c` 参数可以按某列的值着色——比如按 source（api/web）区分不同颜色的点，一眼就能看出不同来源的数据在"质量-长度"二维空间中的分布差异。
