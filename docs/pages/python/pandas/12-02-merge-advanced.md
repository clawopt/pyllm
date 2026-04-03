---
title: merge() 高级技巧
description: left_on / right_on 不同列名合并、索引合并、批量多表合并、合并性能优化
---
# 高级合并操作


## 列名不同时的合并：left_on + right_on

```python
import pandas as pd

df_left = pd.DataFrame({
    'model_name': ['GPT-4o', 'Claude', 'Llama'],
    'params': [1760, 175, 70],
})

df_right = pd.DataFrame({
    'model': ['GPT-4o', 'Claude', 'Llama', 'Qwen'],
    'MMLU': [88.7, 89.2, 84.5, 83.5],
})

merged = pd.merge(
    df_left, df_right,
    left_on='model_name',
    right_on='model',
    how='left'
)
print(merged)
```

### 合并后清理重复键

```python
merged = pd.merge(
    df_left, df_right,
    left_on='model_name',
    right_on='model',
    how='inner'
).drop(columns=['model'])

print("清理后的结果:")
print(merged)
```

## 索引合并：left_index / right_index

```python
import pandas as pd

df1 = pd.DataFrame(
    {'val_a': [10, 20, 30]},
    index=['x', 'y', 'z']
)

df2 = pd.DataFrame(
    {'val_b': [100, 200, 300, 400]},
    index=['y', 'z', 'w', 'x']
)

idx_merge = pd.merge(
    df1, df2,
    left_index=True,
    right_index=True,
    how='outer'
)
print(idx_merge)
```

### DataFrame.join()（语法糖）

```python
joined = df1.join(df2, how='outer', lsuffix='_l', rsuffix='_r')
print(joined)
```

**`merge()` vs `join()` 的选择**：
- 用 **列值** 合并 → `merge(on=...)`
- 用 **索引** 合并 → `join()` 或 `merge(left_index=True)`

## 批量合并多个表

### 方式一：循环 reduce

```python
import pandas as pd
from functools import reduce

dfs = [
    pd.DataFrame({'id': [1, 2, 3], 'math': [90, 85, 78]}),
    pd.DataFrame({'id': [1, 2, 3], 'english': [88, 92, 80]}),
    pd.DataFrame({'id': [1, 2, 3], 'science': [95, 87, 82]}),
]

result = reduce(lambda left, right: pd.merge(left, right, on='id'), dfs)
print(result)
```

### 方式二：LLM 场景——多来源评估数据汇总

```python
import pandas as pd
import numpy as np

class MultiSourceMerger:
    """多来源数据合并器"""

    def __init__(self):
        self.sources = {}

    def add_source(self, name, df, key_col='model'):
        self.sources[name] = (df, key_col)
        return self

    def merge_all(self, how='outer'):
        if not self.sources:
            return pd.DataFrame()

        names = list(self.sources.keys())
        base_df, base_key = self.sources[names[0]]

        for name in names[1:]:
            curr_df, curr_key = self.sources[name]
            suffixes = ('', f'_{name}')
            base_df = pd.merge(
                base_df, curr_df,
                left_on=base_key,
                right_on=curr_key,
                how=how,
                suffixes=suffixes
            )
            if curr_key != base_key and curr_key in base_df.columns:
                base_df.drop(columns=[curr_key], inplace=True)

        return base_df


merger = MultiSourceMerger()
merger.add_source('meta', pd.DataFrame({
    'model': ['GPT-4o', 'Claude', 'Llama', 'Qwen'],
    'params_B': [1760, 175, 70, 72],
    'vendor': ['OpenAI', 'Anthropic', 'Meta', 'Alibaba'],
}))
merger.add_source('bench', pd.DataFrame({
    'model': ['GPT-4o', 'Claude', 'Llama', 'Qwen', 'DeepSeek'],
    'MMLU': [88.7, 89.2, 84.5, 83.5, 86.0],
    'HumanEval': [92.0, 93.1, 82.4, 78.6, 84.2],
}))
merger.add_source('cost', pd.DataFrame({
    'model': ['GPT-4o', 'Claude', 'Llama', 'Qwen'],
    'price_input': [2.50, 3.00, 0.27, 0.27],
    'price_output': [10.0, 15.0, 0.60, 1.20],
}))
merger.add_source('perf', pd.DataFrame({
    'model': ['GPT-4o', 'Claude', 'Llama', 'Qwen', 'DeepSeek'],
    'avg_latency_ms': [800, 650, 350, 400, 280],
    'throughput_tps': [45, 55, 120, 110, 150],
}))

full_data = merger.merge_all(how='outer')
print("=== 多源合并结果 ===")
print(full_data.to_string(index=False))
```

## 合并性能优化

### 大表合并策略

```python
import pandas as pd
import numpy as np
import time

n_small, n_large = 1000, 1_000_000

small_df = pd.DataFrame({
    'key': np.arange(n_small),
    'label': [f'item_{i}' for i in range(n_small)],
})

large_df = pd.DataFrame({
    'key': np.random.randint(0, n_small, n_large),
    'value': np.random.randn(n_large),
})

start = time.time()
large_sorted = large_df.sort_values('key').reset_index(drop=True)
small_sorted = small_df.sort_values('key').reset_index(drop_time.time() - start

start = time.time()
large_df['key_cat'] = pd.Categorical(large_df['key'])
small_df['key_cat'] = pd.Categorical(small_df['key'])
result_cat = pd.merge(small_df[['key_cat', 'label']],
                       large_df[['key_cat', 'value']],
                       on='key_cat',
                       how='inner')
t_cat = time.time() - start

print(f"普通合并耗时参考")
print(f"提示：大表合并时，确保键列 dtype 一致，可显著提升速度")
```

### 避免笛卡尔积陷阱

```python
import pandas as pd

df1 = pd.DataFrame({'A': [1, 2], 'B': ['a', 'b']})
df2 = pd.DataFrame({'C': [3, 4], 'D': ['c', 'd']})

cartesian = pd.merge(df1, df2, how='cross')  # 显式 cross join
print(f"笛卡尔积: {len(df1)} × {len(df2)} = {len(cartesian)} 行")

safe = pd.merge(df1, df2, left_on=df1.index, right_on=df2.index, how='inner')
print(f"正确合并: {len(safe)} 行")
```
