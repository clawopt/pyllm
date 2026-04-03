---
title: 高级可视化技巧
description: 子图布局 / 双轴图 / 条件着色 / 注释标注 / 多面板对比
---
# 高级可视化技巧


## subplots=True：每列一个子图

```python
import pandas as pd
import numpy as np

np.random.seed(42)

metrics = pd.DataFrame({
    'accuracy': np.random.uniform(0.80, 0.96, 30),
    'latency': np.random.exponential(500, 30),
    'throughput': np.random.randint(40, 120, 30),
    'cost': np.random.uniform(1, 15, 30),
}, index=pd.date_range('2025-03-01', periods=30, freq='D'))

axes = metrics.plot(
    subplots=True,
    figsize=(12, 10),
    layout=(2, 2),      # 2行2列布局
    sharex=False,
    title=['Accuracy Trend', 'Latency (ms)', 'Throughput (req/s)', 'Cost ($)'],
)
```

## 双轴图：不同量纲的指标

```python
import pandas as pd
import numpy as np

np.random.seed(42)

df = pd.DataFrame({
    'date': pd.date_range('2025-03-01', periods=14, freq='D'),
    'api_calls': [100 + i*5 + np.random.randint(-10, 20) for i in range(14)],
    'avg_latency_ms': [500 - i*8 + np.random.randint(-50, 50) for i in range(14)],
}).set_index('date')

ax = df['api_calls'].plot(
    figsize=(11, 5),
    color='#2196F3',
    marker='o',
    label='API Calls (left)',
    title='API Volume vs Latency',
)

ax.set_ylabel('API Calls', color='#2196F3')
ax.tick_params(axis='y', labelcolor='#2196F3')

ax2 = ax.twinx()
df['avg_latency_ms'].plot(
    ax=ax2,
    color='#F44336',
    marker='s',
    linestyle='--',
    label='Latency ms (right)',
)
ax2.set_ylabel('Avg Latency (ms)', color='#F44336')
ax2.tick_params(axis='y', labelcolor='#F44336')

lines = ax.get_lines() + ax2.get_lines()
ax.legend(lines, [l.get_label() for l in lines], loc='upper center')
```

## 条件着色：根据值改变颜色

```python
import pandas as pd
import numpy as np

np.random.seed(42)

scores = pd.DataFrame({
    'model': ['GPT-4o', 'Claude', 'Llama', 'Qwen', 'DeepSeek', 'Gemini'],
    'score': [88.7, 89.2, 84.5, 83.5, 86.8, 87.0],
})

colors = ['#4CAF50' if s >= 87 else '#FF9800' if s >= 85 else '#F44336'
          for s in scores['score']]

ax = scores.plot.bar(
    x='model',
    y='score',
    figsize=(10, 5),
    title='Model Scores (Green≥87 / Orange≥85 / Red<85)',
    color=colors,
    legend=False,
)

for p in ax.patches:
    height = p.get_height()
    ax.annotate(f'{height:.1f}',
                xy=(p.get_x() + p.get_width()/2, height),
                ha='center', va='bottom',
                fontsize=11, fontweight='bold')
```

## 饼图：占比展示

```python
import pandas as pd

usage = pd.Series({
    'Chat/对话': 45,
    'Code/代码': 25,
    'Analysis/分析': 15,
    'Translation/翻译': 10,
    'Other/其他': 5,
})

ax = usage.plot.pie(
    figsize=(8, 8),
    autopct='%1.1f%%',
    startangle=90,
    colors=['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#607D8B'],
    explode=[0.02] * 5,
    title='API Usage Distribution by Task Type',
)
```
