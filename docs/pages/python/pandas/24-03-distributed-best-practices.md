---
title: 分布式处理最佳实践
description: 内存管理 / 分区调优 / Shuffle 优化 / 结果收集策略 / 故障排查
---
# 分布式处理最佳实践


## 核心原则

```
1. 能用 Pandas 就不用分布式
2. 小数据集上分布式更慢（调度开销）
3. 始终先优化单机性能再考虑分布式
4. 选择正确的分区键（数据倾斜是大敌）
```

## 数据倾斜（Skew）检测与解决

```python
import pandas as pd
import numpy as np


def detect_skew(df, group_col, target_col='value'):
    """检测数据倾斜"""
    grouped = df.groupby(group_col)[target_col].count()
    max_count = grouped.max()
    mean_count = grouped.mean()

    skew_ratio = max_count / mean_count if mean_count > 0 else 1

    print(f"=== 数据倾斜分析 ===")
    print(f"组数:       {len(grouped)}")
    print(f"最大组:     {max_count}")
    print(f"平均组:     {mean_count:.0f}")
    print(f"倾斜系数:   {skew_ratio:.2f}")

    if skew_ratio > 5:
        print(f"\n⚠️ 严重倾斜! (>{skew_ratio:.1f}x)")
        print("解决方案:")
        print("  1. 增加 partition 数")
        print("  2. 使用 salting_key 参数")
        print("  3. 预聚合/预过滤减少数据量")
        print(" 4. 考虑重新选择分组列")
        return False
    elif skew_ratio > 2:
        print(f"\n⚡ 中等倾斜 ({skew_ratio:.1f}x)")
        return True
    else:
        print(f"\n✅ 分布均匀")
        return True


np.random.seed(42)
n = 100_000
skewed_df = pd.DataFrame({
    'model': (['GPT-4o']*60000 + ['Claude']*30000 +
              ['Llama']*8000 + ['Qwen'*2000]),
    'value': np.random.randn(n),
})

detect_skew(skewed_df, 'model')
```

## 分区数选择指南

```python
def recommend_partitions(row_count, worker_cores=8):
    """推荐分区数"""
    if row_count < 10_000:
        return 1, "不需要分区"
    elif row_count < 1_000_000:
        parts = min(worker_cores * 4, row_count // 50_000)
        return max(parts, 1), f"每区 ~{row_count//parts:,} 行"
    elif row_count < 100_000_000:
        parts = worker_cores * 16
        return parts, f"每区 ~{row_count//parts:,} 行"
    else:
        parts = worker_cores * 64
        return parts, f"每区 ~{row_count//parts:,} 行"


scenarios = [
    ('SFT 清洗后', 500_000),
    ('API 日志月度', 5_000_000),
    ('全网爬虫', 200_000_000),
]

print("=== 分区建议 ===")
for name, n in scenarios:
    parts, note = recommend_partitions(n)
    bar = '█' * min(int(n / 1_000_000), 30)
    print(f"  {name:<20s} {n:>12,.0f} | {bar} | {parts} 区 | {note}")
```
