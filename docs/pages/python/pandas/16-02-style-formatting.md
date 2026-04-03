---
title: 样式与格式化
description: Style/Styler API、条件着色、格式化数值显示、导出美观的 Excel/HTML 表格
---
# 数据展示：让表格更好看

分析完数据之后，你需要把结果展示给别人看——可能是团队成员、产品经理、或者直接放进报告里。原始的 DataFrame `print()` 输出往往不够美观。Pandas 的 Style API 就是用来解决这个问题的。

## 条件高亮

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'model': ['GPT-4o', 'Claude', 'Llama', 'Qwen'],
    'MMLU': [88.7, 89.2, 84.5, 78.3],
    'price': [15, 15, 0.27, 0.14],
})

def highlight_max(s):
    return ['font-weight:bold; background-color:#d4edda' if v == s.max() else '' for v in s]

styled = df.style.apply(highlight_max, subset=['MMLU', 'price'])
```

`style.apply()` 对每个单元格应用 CSS 样式——上面的例子让每列的最大值加粗并标绿。这在"一眼找出最优模型"的场景中非常实用。

## 数值格式化

```python
(df.style
 .format({'MMLU': '{:.1f}', 'price': '${:.2f}'})
 .set_caption('模型对比表')
 .background_gradient(subset=['MMLU'], cmap='RdYlGn'))
```

`format()` 控制数值的显示格式（不改变底层值），`background_gradient()` 按数值大小自动生成颜色渐变（红色=低，绿色=高）。**这是做管理汇报时最常用的样式组合**。
