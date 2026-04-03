---
title: Pandas 内置可视化基础
description: .plot() 接口 / 折线图 / 柱状图 / 散点图 / 直方图 / 基本样式定制
---
# 可视化基础绑定


## Pandas 可视化定位

Pandas 的 `.plot()` 是 **快速探索性分析** 的利器：
- 无需导入 matplotlib，一行代码出图
- 直接基于 DataFrame/Series，数据绑定零成本
- 适合：**看趋势、找分布、做对比**

**不适合**：出版级图表（用 matplotlib/seaborn/plotly）

## 折线图：时间序列首选

```python
import pandas as pd
import numpy as np

np.random.seed(42)

df = pd.DataFrame({
    'date': pd.date_range('2025-03-01', periods=30, freq='D'),
    'gpt4o_cost': np.cumsum(np.random.uniform(30, 60, 30)),
    'claude_cost': np.cumsum(np.random.uniform(25, 55, 30)),
    'deepseek_cost': np.cumsum(np.random.uniform(8, 20, 30)),
}).set_index('date')

ax = df.plot(
    figsize=(12, 5),
    title='Daily API Cost Trend (Mar 2025)',
    marker='o',
    markersize=3,
)
```

## 柱状图：类别对比

```python
import pandas as pd
import numpy as np

np.random.seed(42)

models = ['GPT-4o', 'Claude', 'Llama', 'Qwen', 'DeepSeek']
df = pd.DataFrame({
    'model': models,
    'MMLU': [88.7, 89.2, 84.5, 83.5, 86.8],
    'HumanEval': [92.0, 93.1, 82.4, 78.6, 87.2],
}).set_index('model')

ax = df.plot(
    kind='bar',
    figsize=(10, 5),
    title='Model Benchmark Comparison',
    rot=0,
    colormap='viridis',
)

ax_h = df.plot(
    kind='barh',
    figsize=(10, 4),
    title='Benchmark Scores by Model',
)
```

## 散点图：关系探索

```python
import pandas as pd
import numpy as np

np.random.seed(42)

n = 50
df = pd.DataFrame({
    'model_size_B': np.random.choice([8, 13, 35, 70, 176], n),
    'MMLU_score': [84 + np.random.randn() * 3 + (s/100) for s in np.random.choice([8, 13, 35, 70, 176], n)],
})

ax = df.plot.scatter(
    x='model_size_B',
    y='MMLU_score',
    figsize=(8, 6),
    title='Model Size vs MMLU Score',
    s=80,          # 点大小
    alpha=0.6,     # 透明度
    c='MMLU_score', # 颜色映射到值
    colormap='coolwarm',
)
```

## 直方图与箱线图：分布分析

```python
import pandas as pd
import numpy as np

np.random.seed(42)

latency_data = pd.DataFrame({
    'GPT-4o': np.random.exponential(800, 500).astype(int),
    'Claude': np.random.exponential(650, 500).astype(int),
    'Llama': np.random.exponential(350, 500).astype(int),
})

ax_hist = latency_data.plot.hist(
    bins=40,
    alpha=0.6,
    figsize=(10, 5),
    title='Latency Distribution by Model',
)

ax_box = latency_data.plot.box(
    figsize=(8, 5),
    title='Latency Box Plot by Model',
    vert=False,
)
```

## 面积图：堆叠占比

```python
import pandas as pd
import numpy as np

np.random.seed(42)

daily_usage = pd.DataFrame({
    'chat': np.random.randint(200, 500, 14),
    'code': np.random.randint(100, 300, 14),
    'analysis': np.random.randint(50, 200, 14),
}, index=pd.date_range('2025-03-01', periods=14, freq='D'))

ax_area = daily_usage.plot.area(
    figsize=(12, 5),
    title='API Usage by Task Type (Stacked)',
    alpha=0.7,
)
```

## 常用 plot 参数速查

| 参数 | 作用 | 示例值 |
|------|------|--------|
| `kind` | 图表类型 | `'line'`, `'bar'`, `'scatter'`, `'hist'`, `'box'` |
| `figsize` | 图尺寸 | `(10, 5)` 宽×高 |
| `title` | 标题 | `'My Chart'` |
| `xlabel/ylabel` | 轴标签 | `'Time'`, `'Value'` |
| `colormap` | 配色方案 | `'viridis'`, `'coolwarm'`, `'tab10'` |
| `alpha` | 透明度 | `0.6` |
| `rot` | x轴标签旋转角度 | `45`, `90`, `0` |
| `grid` | 网格线 | `True`, `False` |
| `legend` | 图例位置 | `'upper right'`, `'lower left'` |
| `subplots` | 子图模式 | `True`（每列一个子图） |
