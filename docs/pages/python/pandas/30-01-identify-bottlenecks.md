---
title: 性能瓶颈识别
description: 内存/计算/I/O 瓶颈分析、cProfile + Pandas profiling、dtype 热图
---
# 案例 5：迁移至 Polars — 瓶颈识别

```python
import pandas as pd
import numpy as np

n = 5_000_000
df = pd.DataFrame({
    'id': range(n),
    'text': ['sample text data for benchmark'] * n,
    'value': np.random.randn(n),
    'category': np.random.choice(list('ABCDEFGHIJ'), n),
})

print("=== Pandas 基准 ===")
print(f"形状: {df.shape}")
print(f"内存: {df.memory_usage(deep=True).sum() / 1024**2:.0f} MB")
print(f"\ndtype 分布:")
for col in df.columns:
    mem = df[col].memory_usage(deep=True)
    print(f"  {col}: {mem / 1024**2:.1f} MB ({df[col].dtype})")
```
