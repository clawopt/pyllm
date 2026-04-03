---
title: 样式与格式化输出
description: Style.apply() 条件格式 / 渐变色 / 数据条 / 高亮极值 / 导出 HTML/Excel 带样式
---
# 样式与格式化输出


## style 概述

Pandas 的 `.style` 属性返回一个 `Styler` 对象，用于对 DataFrame 进行**视觉美化**和**条件格式化**。它不修改数据本身，只改变显示方式。

## 基础样式

```python
import pandas as pd
import numpy as np

np.random.seed(42)

df = pd.DataFrame({
    'model': ['GPT-4o', 'Claude', 'Llama', 'Qwen', 'DeepSeek'],
    'MMLU': [88.7, 89.2, 84.5, 83.5, 86.8],
    'HumanEval': [92.0, 93.1, 82.4, 78.6, 87.2],
    'price': [2.50, 3.00, 0.27, 0.27, 0.14],
})

styled = df.style.highlight_max(subset=['MMLU', 'HumanEval'], color='#4CAF50')
```

## 条件格式化

### 按阈值着色

```python
def color_by_threshold(val, threshold=85):
    if val >= threshold:
        return 'color: green; font-weight: bold'
    elif val >= 80:
        return 'color: orange'
    else:
        return 'color: red'


styled = df.style.applymap(
    color_by_threshold,
    subset=['MMLU', 'HumanEval']
)
```

### 数据条（bar）

```python
with_bar = df.style.bar(
    subset=['MMLU', 'HumanEval'],
    color=['#FFCDD2', '#C8E6C9'],
)
```

### 背景渐变

```python
gradient = df.style.background_gradient(
    cmap='RdYlGn',
    subset=['MMLU', 'HumanEval'],
)
```

## 自定义 apply 函数

```python
import pandas as pd
import numpy as np

def highlight_best_model(s):
    """给每列最高分行加粗"""
    is_max = s == s.max()
    return ['font-weight: bold; background-color: #E8F5E9' if v else ''
            for v in is_max]


def format_price(v):
    """价格列格式化"""
    if isinstance(v, (int, float)):
        if v < 0.5:
            return f'${v:.2f} 💚'
        elif v < 2.0:
            return f'${v:.2f} 🟡'
        else:
            return f'${v:.2f} 🔴'
    return str(v)


df = pd.DataFrame({
    'model': ['GPT-4o', 'Claude', 'Llama', 'Qwen', 'DeepSeek', 'Gemini'],
    'MMLU': [88.7, 89.2, 84.5, 83.5, 86.8, 87.3],
    'price_per_1M': [12.50, 18.00, 0.87, 1.47, 0.42, 6.25],
})

styled = (
    df.style
    .apply(highlight_best_model, subset=['MMLU'])
    .format({'price_per_1M': format_price})
    .set_caption('LLM Benchmark & Pricing Comparison')
    .set_table_styles([
        {'selector': 'caption', 'props': [('font-size', '16px'),
                                            ('font-weight', 'bold')]}
    ])
)
```

## 导出带样式的 Excel

```python
import pandas as pd

output_path = '/tmp/llm_comparison.xlsx'

writer = pd.ExcelWriter(output_path, engine='openpyxl')

styled = df.style.apply(
    lambda s: ['background-color: #E8F5E9' if v == s.max() else '' for v in s],
    subset=['MMLU']
)

styled.to_excel(writer, sheet_name='Benchmarks', index=False)
writer.close()

print(f"✓ 已导出到 {output_path}")
```

## LLM 场景：评估报告自动格式化

```python
import pandas as pd
import numpy as np

class ReportFormatter:
    """评估报告格式化器"""

    @staticmethod
    def format_eval_report(eval_df):
        """格式化评估数据表"""

        def score_style(val):
            if not isinstance(val, (int, float)) or pd.isna(val):
                return ''
            if val >= 90:
                return 'background-color: #1B5E20; color: white; font-weight: bold'
            elif val >= 85:
                return 'background-color: #81C784'
            elif val >= 80:
                return 'background-color: #FFF59D'
            else:
                return 'background-color: #FFCDD2'

        def rank_badge(val):
            badges = {1: '🥇', 2: '🥈', 3: '🥉'}
            return badges.get(int(val), f'#{int(val)}')

        styled = (
            eval_df.style
            .applymap(score_style,
                      subset=[c for c in eval_df.columns
                              if c not in ['model', 'rank']])
            .format({'rank': rank_badge})
            .set_properties(**{
                'text-align': 'center',
                'font-size': '11pt',
            })
            .hide(axis='index')
        )
        return styled


np.random.seed(42)
eval_df = pd.DataFrame({
    'model': ['GPT-4o', 'Claude-3.5-Sonnet', 'Llama-3.1-70B',
              'Qwen2.5-72B', 'DeepSeek-V3'],
    'rank': [2, 1, 4, 5, 3],
    'MMLU': [88.7, 89.2, 84.5, 83.5, 86.8],
    'HumanEval': [92.0, 93.1, 82.4, 78.6, 87.2],
    'MATH': [76.6, 71.1, 50.4, 60.8, 65.2],
    'GPQA': [53.9, 59.3, 34.2, 40.1, 45.6],
})

formatted = ReportFormatter.format_eval_report(eval_df)
print("✓ 报告已格式化（在 Jupyter 中可直接渲染）")
```
