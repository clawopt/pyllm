---
title: 可视化最佳实践与导出
description: 样式规范 / 图表保存 / 中文显示 / 响应式尺寸 / 常见问题排查
---
# 可视化最佳实践


## 样式规范建议

```python
import pandas as pd
import numpy as np

def setup_plot_style():
    """统一图表风格"""
    pd.options.plotting.backend = 'matplotlib'

    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'figure.figsize': (10, 5),
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'axes.spines.top': False,      # 移除上边框
        'axes.spines.right': False,     # 移除右边框
        'axes.grid': True,
        'grid.alpha': 0.3,
        'figure.dpi': 100,
    })


np.random.seed(42)
df = pd.DataFrame({
    'model': ['GPT-4o', 'Claude', 'Llama', 'Qwen', 'DeepSeek'],
    'MMLU': [88.7, 89.2, 84.5, 83.5, 86.8],
}).set_index('model')

ax = df.plot.bar(
    title='MMLU Benchmark Scores',
    color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
    rot=0,
)
```

## 图表保存

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'x': range(20),
    'y': [i**0.5 + np.random.randn() for i in range(20)],
})

ax = df.plot.scatter(x='x', y='y', title='Sample Plot')

fig = ax.get_figure()
fig.savefig('/tmp/plot_output.png', dpi=150, bbox_inches='tight')
print("✓ 已保存: /tmp/plot_output.png")

fig.savefig('/tmp/plot_output.pdf', bbox_inches='tight')
print("✓ 已保存: /tmp/plot_output.pdf")

fig.savefig('/tmp/plot_output.svg', bbox_inches='tight')
print("✓ 已保存: /tmp/plot_output.svg")
```

## 中文显示配置

```python
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC',
                                    'Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


df = pd.DataFrame({
    '模型': ['GPT-4o', 'Claude', 'Llama', '通义千问', 'DeepSeek'],
    '得分': [88.7, 89.2, 84.5, 83.5, 86.8],
}).set_index('模型')

ax = df.plot.bar(title='模型基准测试对比')
```

## 不同场景的图表选择指南

| 分析目标 | 推荐图表 | Pandas kind | 关键参数 |
|---------|---------|-------------|---------|
| 时间趋势 | 折线图 | `line` 或默认 | `marker`, `linewidth` |
| 类别比较 | 柱状图 | `bar` / `barh` | `rot=0`, `color` |
| 占比分布 | 饼图 | `pie` | `autopct`, `explode` |
| 数值分布 | 直方图 | `hist` | `bins`, `alpha` |
| 分布概览 | 箱线图 | `box` | `vert=False` |
| 两变量关系 | 散点图 | `scatter` | `s`, `alpha`, `c` |
| 多变量趋势 | 面积图 | `area` | `stacked=True` |
| 相关矩阵 | 热力图 | 用 seaborn | — |
| 雷达图 | 极坐标 | 用 matplotlib | — |

## 常见问题排查

### 问题1：图表不显示（Jupyter 中）

```python
import matplotlib.pyplot as plt
plt.show()
```

### 问题2：图表重叠或裁切

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
plt.tight_layout()

plt.subplots_adjust(left=0.1, right=0.95, top=0.92, bottom=0.12)
```

### 问题3：大数据集绘图慢

```python
large_df = pd.DataFrame({'value': np.random.randn(1_000_000)})

sampled = large_df.sample(n=5000, random_state=42)
ax = sampled['value'].plot.hist(bins=50)

binned = pd.cut(large_df['value'], bins=50).value_counts().sort_index()
ax = binned.plot.bar(figsize=(14, 4))
```

### 问题4：时间轴标签拥挤

```python
import pandas as pd

ts = pd.DataFrame(
    {'value': range(365)},
    index=pd.date_range('2025-01-01', periods=365, freq='D'),
)

ax = ts.plot(xticks=ts.index[::30], rot=45)

ax = ts.plot()
ax.set_xticks(ax.get_xticks()[::len(ax.get_xticks())//8])
```
