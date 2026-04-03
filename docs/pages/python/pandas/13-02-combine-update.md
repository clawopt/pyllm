---
title: combine_first() 与 update()
description: 缺失值填充合并、覆盖更新数据、多版本数据融合
---
# combine_first() 与 update()

`concat()` 和 `merge()` 是最常用的合并方式，但还有两个更精细的工具：`combine_first()` 用于"用一张表填补另一张表的缺失值"，`update()` 用于"用新数据覆盖旧数据的非空值"。

## combine_first()：缺失值补全

```python
import pandas as pd
import numpy as np

primary = pd.DataFrame({
    'A': [1.0, np.nan, 3.0],
    'B': [np.nan, 20, np.nan],
})

backup = pd.DataFrame({
    'A': [10, 99, 30],
    'B': [90, 200, 300],
})

filled = primary.combine_first(backup)
print(filled)
```

规则很简单：**优先取 primary 的值，如果 primary 的某个位置是 NaN，就用 backup 对应位置的值来填充**。这在"有主数据源但需要用备用数据源补缺"的场景中特别有用——比如主数据库的记录可能不完整，但备份系统有完整版本。

## update()：就地更新

```python
import pandas as pd

df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
new_data = pd.DataFrame({'A': [10, np.nan, 30], 'B': [40, 50, 60]})

df.update(new_data)
print(df)
```

`update()` 用 new_data 中**非 NaN 的值**去覆盖 df 中对应位置的值。注意它是**原地修改**（返回 None）——所以不需要重新赋值。NaN 值不会覆盖原值（这和 `combine_first()` 正好相反）。
