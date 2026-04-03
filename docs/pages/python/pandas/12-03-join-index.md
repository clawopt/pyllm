---
title: join() 索引连接
description: DataFrame.join() 用法、lsuffix/rsuffix、多表链式 join、与 merge 的选择决策
---
# join()：基于索引的快速合并

`merge()` 按列值匹配，而 `join()` 按**索引**（Index）匹配。如果你的数据已经把连接键设为索引了，`join()` 比 `merge()` 更简洁。

## 基本用法

```python
import pandas as pd

benchmarks = pd.DataFrame(
    {'MMLU': [88.7, 89.2, 84.5]},
    index=['GPT-4o', 'Claude', 'Llama']
)

pricing = pd.DataFrame(
    {'price': [2.50, 3.00, 0.27]},
    index=['GPT-4o', 'Claude', 'Llama']
)

result = benchmarks.join(pricing)
print(result)
```

一行代码，不需要指定 `on` 参数——因为 `join()` 默认用索引作为连接键。这在"模型名作为索引"的场景下特别自然：你通常会在 `read_csv()` 时就把 model 列设为 `index_col=0`，之后所有关联操作都用 `join()` 即可。

## merge vs join 的选择

判断标准很简单：

- **连接键在普通列中** → 用 `merge(on='key')`
- **连接键在 Index 中** → 用 `join()`
- **需要 left_on / right_on** → 只能用 `merge()`
- **需要多表链式操作** → `join()` 支持链式调用更方便

实际工作中 `merge()` 的使用频率远高于 `join()`——因为大多数原始数据的连接键都在普通列里而非索引中。
