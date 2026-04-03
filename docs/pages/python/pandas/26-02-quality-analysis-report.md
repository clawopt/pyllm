---
title: 质量分析报告生成
description: 自动化质量报告、多维度统计分析、异常检测、可视化输出
---
# 质量分析报告

```python
import pandas as pd
import numpy as np

class QualityReport:
    def __init__(self, df):
        self.df = df
    
    def generate(self):
        lines = []
        n = len(self.df)
        lines.append(f"\n{'='*60}")
        lines.append(f"  千万级对话语料 — 质量分析报告")
        lines.append(f"  总样本: {n:,}")
        lines.append(f"{'='*60}")
        
        na = self.df.isna().sum()
        has_na = na[na > 0]
        if len(has_na) > 0:
            lines.append(f"\n❌ 缺失值 ({len(has_na)} 列):")
            for col, cnt in has_na.items():
                lines.append(f"  {col}: {cnt:,} ({cnt/n*100:.2f}%)")
        
        dup = self.df.duplicated(subset=['user_message', 'assistant_message']).sum()
        lines.append(f"\n🔁 内容重复: {dup:,} ({dup/n*100:.2f}%)")
        
        num_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            desc = self.df[num_cols].describe().T[['count','mean','std','min','50%','max']]
            lines.append(f"\n📊 数值统计:\n{desc.round(2).to_string()}")
        
        return '\n'.join(lines)

report = QualityReport(df)
print(report.generate())
```
