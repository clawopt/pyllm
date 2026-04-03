---
title: nlargest() / nsmallest()：高效 TopN 选择
description: 堆排序原理、性能对比 sort_values+head、分组内 TopN、LLM 场景最佳模型选择
---
# TopN：只关心前几名时的高效选择

如果你只需要前 10 名而不是完整的排序列表，`nlargest()` 和 `nsmallest()` 是比 `sort_values().head(N)` 更好的选择——不仅代码更简洁，而且在数据量大时性能更好。

## 为什么它更快

```python
import pandas as pd
import numpy as np
import time

n = 2_000_000
df = pd.DataFrame({'value': np.random.randn(n)})

start = time.time()
r1 = df.sort_values('value', ascending=False).head(10)
t1 = time.time() - start

start = time.time()
r2 = df.nlargest(10, 'value')
t2 = time.time() - start

print(f"sort_values+head(): {t1:.4f}s")
print(f"nlargest():         {t2:.4f}s")
print(f"加速:               {t1/t2:.0f}x")
```

原因在于算法复杂度：`sort_values` 是 O(n log n)（需要完全排序），而 `nlargest` 使用堆选择（heap select）算法，复杂度是 O(n log k)。当 n=200 万、k=10 时，log k ≈ 3.3 而 log n ≈ 21——差距接近 7 倍。数据量越大这个优势越明显。

## 分组内 TopN

一个更实用的场景是：在每个组内取前 N 名。比如每个来源（api/web/export）各自质量最高的 5 条对话：

```python
import pandas as pd
import numpy as np

np.random.seed(42)
n = 100_000
df = pd.DataFrame({
    'source': np.random.choice(['api', 'web', 'export'], n),
    'quality': np.round(np.random.uniform(1, 5, n), 2),
    'text': [f'对话{i}' for i in range(n)],
})

top_per_source = df.groupby('source', group_keys=False)\
    .apply(lambda g: g.nlargest(3, 'quality'))

print(top_per_source[['source', 'quality', 'text']].head(9))
```

`group_keys=False` 防止 `groupby` 把分组键加回结果中。`apply(lambda g: g.nlargest(3, ...))` 对每个分组独立取 Top 3，最终结果是所有分组的 Top 3 拼接在一起。
