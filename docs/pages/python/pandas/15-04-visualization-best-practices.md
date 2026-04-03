---
title: 可视化最佳实践
description: 图表选择决策树、配色方案、标注规范、导出高质量图片
---
# 可视化最佳实践

画图容易，画好图难。这一节总结几个在实际工作中反复验证过的可视化原则。

## 图表选择速查

| 你想展示什么 | 用什么图 |
|-------------|---------|
| 趋势（随时间变化） | 折线图 `plot()` |
| 分类对比（模型 A vs B） | 柱状图 `plot(kind='bar')` |
| 分布形态（延迟/分数） | 直方图 `plot(kind='hist')` / KDE 图 |
| 两变量关系（质量 vs 长度） | 散点图 `plot.scatter()` |
| 组成比例（来源占比） | 饼图 `plot(kind='pie')` |
| 多变量分布（质量×长度×来源） | 散点矩阵 + 着色 |

**最常见的新手错误是用折线图展示分类数据**——比如把 model 名放在 X 轴上画折线。分类变量之间没有"顺序关系"，用柱状图才是正确的选择。

## 配色与可读性

```python
df.plot(kind='bar', colormap='viridis')
```

推荐使用感知均匀的色图（如 `viridis`、`Set2`、`tab10`），避免使用彩虹色图（`jet`/`rainbow`）——它们对色觉障碍人群不友好，而且会人为地制造数据中不存在的视觉差异。

## 导出图片

```python
ax = df.plot(kind='bar', figsize=(10, 5))
fig = ax.get_figure()
fig.savefig('report.png', dpi=300, bbox_inches='tight')
```

`dpi=300` 保证打印质量，`bbox_inches='tight'` 防止标签被截断。**在写报告或论文时，300dpi 是最低要求**。
