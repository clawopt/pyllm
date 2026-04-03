---
title: 生成评估报告
description: 自动生成 Markdown/HTML/Excel 报告、关键发现总结、决策建议
---
# 评估报告生成

```python
def generate_eval_report(pivot):
    lines = []
    lines.append("# LLM 模型评估报告\n")
    lines.append(f"## 综合排名\n")
    
    for rank, (model, row) in enumerate(pivot.sort_values('rank').iterrows(), 1):
        lines.append(f"{rank}. **{model}** — 加权分: {row['weighted_score']:.2f} | 均值: {row['mean']:.1f}")
    
    best = pivot['weighted_score'].idxmax()
    worst = pivot['weighted_score'].idxmin()
    
    lines.append(f"\n## 关键发现\n")
    lines.append(f"- 🏆 **最佳模型**: {best}（加权分 {pivot.loc[best, 'weighted_score']:.2f}）")
    lines.append(f"- ⚠️ **需改进**: {worst}（加权分 {pivot.loc[worst, 'weighted_score']:.2f}）")
    
    for b in benchmarks:
        winner = pivot[b].idxmax()
        lines.append(f"- {b} 冠军: {winner} ({pivot.loc[winner, b]:.1f})")
    
    return '\n'.join(lines)

report = generate_eval_report(result)
print(report)

with open('eval_report.md', 'w') as f:
    f.write(report)
```
