---
title: 可视化对比分析
description: 雷达图、柱状对比图、热力图、散点矩阵、报告导出
---
# 可视化对比

```python
import matplotlib.pyplot as fig

ax = pivot[benchmarks].T.plot(kind='bar', figsize=(12, 5),
                               colormap='Set2', rot=0)
ax.set_title('模型评测对比')
ax.set_ylabel('Score')
ax.legend(loc='lower right')

fig2 = pivot[['mean', 'weighted_score']].plot(
    kind='bar', figsize=(8, 4), color=['steelblue', 'orange']
)
fig2.set_title('综合评分 vs 平均分')
```
