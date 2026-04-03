---
title: 迁移性能报告
description: 端到端迁移报告生成、成本效益分析、建议与总结
---
# 迁移报告

```python
def generate_migration_report(pandas_time, polars_time, pandas_mem, polars_mem):
    lines = []
    speedup = pandas_time / polars_time if polars_time > 0 else 0
    mem_save = (1 - polars_mem / pandas_mem) * 100 if pandas_mem > 0 else 0
    
    lines.append("# Pandas → Polars 迁移报告\n")
    lines.append(f"## 性能对比\n")
    lines.append(f"| 指标 | Pandas | Polars | 提升 |")
    lines.append(f"|------|--------|--------|------|")
    lines.append(f"| 处理时间 | {pandas_time:.2f}s | {polars_time:.2f}s | {speedup:.1f}x |")
    lines.append(f"| 内存占用 | {pandas_mem:.0f}MB | {polars_mem:.0f}MB | {mem_save:.0f}% 节省 |")
    
    lines.append(f"\n## 建议\n")
    if speedup > 5:
        lines.append("- ✅ Polars 在此场景下显著更快，建议迁移")
    elif speedup > 2:
        lines.append("- ⚡ Polars 有一定优势，可考虑逐步迁移")
    else:
        lines.append("- 📊 差异不大，Pandas 已足够，暂不需要迁移")
    
    return '\n'.join(lines)

print(generate_migration_report(3.45, 0.82, 1142, 286))
```
