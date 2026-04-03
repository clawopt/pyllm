---
title: 向量准备与分块
description: token 计数统计、chunk 大小分布分析、embedding 批处理管理
---
# 向量准备

```python
import pandas as pd
import numpy as np

n = 50_000
chunks = pd.DataFrame({
    'chunk_id': [f'ch_{i:06d}' for i in range(n)],
    'token_count': np.random.randint(50, 2000, n),
})

chunks['size_tier'] = pd.cut(
    chunks['token_count'],
    bins=[0, 200, 500, 1000, float('inf')],
    labels=['S', 'M', 'L', 'XL'],
)

print("Chunk 大小分布:")
print(chunks['size_tier'].value_counts().sort_index())

total_tokens = chunks['token_count'].sum()
print(f"\n总 tokens: {total_tokens:,}")
print(f"预估 embedding API 成本: ${total_tokens/1_000_000 * 0.02:.2f}")

oversized = chunks[chunks['token_count'] > 1500]
print(f"\n超长 chunks (>1500): {len(oversized)} ({len(oversized)/n*100:.1f}%)")
```
