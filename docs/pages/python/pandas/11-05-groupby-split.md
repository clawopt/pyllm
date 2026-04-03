---
title: groupby 高级用法：分组迭代与自定义操作
description: 遍历 groupby 对象、get_group() 提取单组、并行分组处理、分组结果的合并策略
---
# 深入 groupby：遍历与自定义操作

大多数时候你用 `agg()` 或 `transform()` 就能完成分组计算。但有些场景需要你对每个组做更复杂的操作——比如对每个模型的数据分别跑一个独立的分析流程，或者把每组的结果保存为单独的文件。这时候就需要了解 `groupby` 对象的内部结构。

## 遍历 groupby 对象

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'model': ['GPT-4o']*5 + ['Claude']*5 + ['Llama']*5,
    'score': np.random.uniform(80, 95, 15),
})

for name, group in df.groupby('model'):
    print(f"\n=== {name} ({len(group)} 条) ===")
    print(f"  均值: {group['score'].mean():.1f}")
    print(f"  最高: {group['score'].max():.1f}")
```

`groupby` 对象是可迭代的，每次迭代返回一个 `(组名, 组DataFrame)` 元组。这让你能对每个组执行任意 Python 代码——不限于 Pandas 内置的聚合函数。

## get_group()：提取单个组

```python
gpt_group = df.groupby('model').get_group('GPT-4o')
print(gpt_group)
```

当你只需要某个特定组的数据时（而不是遍历所有组），`get_group()` 比 `for` 循环加 `if` 判断更直接也更高效——它不需要创建所有组的中间对象。

## 实战：按组导出独立文件

```python
import os
from pathlib import Path

output_dir = Path('./output/by_model')
output_dir.mkdir(parents=True, exist_ok=True)

for name, group in df.groupby('model'):
    out_path = output_dir / f'{name}_data.parquet'
    group.to_parquet(out_path, index=False)
    print(f"已导出: {out_path.name} ({len(group)} 条)")
```

这个模式在 LLM 数据处理中非常常见：按模型/来源/语言等维度把大数据集拆分成小文件，分发给不同的下游处理流程或团队成员。
