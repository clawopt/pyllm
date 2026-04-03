---
title: concat() 基础：纵向与横向拼接
description: concat 的 axis 参数、ignore_index、keys 分组、join 参数处理列不对齐
---
# concat()：把多个 DataFrame 拼在一起

`merge()` 是按键值做"智能匹配"的合并，而 `concat()` 是更简单的"硬拼接"——把多个 DataFrame 按行或按列直接堆叠。当你需要合并多个结构相同的数据文件（比如每天一份日志）时，`concat()` 是正确的选择。

## 纵向拼接（axis=0）：最常用的模式

```python
import pandas as pd

jan = pd.DataFrame({'date': ['01-01']*3, 'users': [100, 120, 90]})
feb = pd.DataFrame({'date': ['02-01']*2, 'users': [110, 130]})

combined = pd.concat([jan, feb], ignore_index=True)
print(combined)
```

`ignore_index=True` 让结果重新生成从 0 开始的整数索引——否则你会看到两个 DataFrame 原来的索引被保留下来（可能有重复）。**在实际工作中几乎总是设置 `ignore_index=True`**。

## 横向拼接（axis=1）：并排连接

```python
import pandas as pd

df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
df2 = pd.DataFrame({'C': [5, 6], 'D': [7, 8]})

side_by_side = pd.concat([df1, df2], axis=1)
print(side_by_side)
```

`axis=1` 让 DataFrame 左右拼接。注意它和 `merge()` 的关键区别：`concat(axis=1)` 是**按位置**对齐行（第 0 行对第 0 行），而 `merge()` 是**按键值**匹配行。所以 `concat(axis=1)` 要求两个 DataFrame 有相同的行数和行顺序。
