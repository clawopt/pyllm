---
title: 去重与对齐
description: drop_duplicates 在合并后的使用、reindex 对齐索引、align 方法族
---
# 合并后的数据整理

多个 DataFrame 拼接之后，经常需要做两件事：去重（因为不同来源的数据可能有重复记录）和对齐（确保索引或列的一致性）。

## 拼接后去重

```python
import pandas as pd

df1 = pd.DataFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
df2 = pd.DataFrame({'id': [3, 4, 5], 'val': ['c', 'd', 'e']})

combined = pd.concat([df1, df2], ignore_index=True)
print(f"拼接后: {len(combined)} 行")

deduped = combined.drop_duplicates(subset=['id'], keep='first')
print(f"去重后: {len(deduped)} 行")
```

注意 `keep='first'` 保留第一次出现的记录。如果你希望按某个优先级列来决定保留哪条，可以先排序再去重——我们在 07-02 节详细讨论过这个模式。

## reindex()：强制对齐

```python
import pandas as pd

s = pd.Series([10, 20, 30], index=['a', 'b', 'c'])
aligned = s.reindex(['a', 'b', 'c', 'd', 'e'])
print(aligned)
```

`reindex()` 让 Series 或 DataFrame 的索引变成你指定的样子——原来没有的索引位置会填 NaN。这在合并来自不同源、行标签不完全一致的数据时非常有用。
