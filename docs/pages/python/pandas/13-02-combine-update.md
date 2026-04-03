---
title: combine_first() 与 update()：数据填充与更新
description: 缺失值填充 / 数据覆盖更新 / compare() 差异对比 / LLM 场景配置合并
---
# Combine 与 Update


## combine_first()：用另一个 DataFrame 填充 NaN

```python
import pandas as pd
import numpy as np

primary = pd.DataFrame({
    'model': ['GPT-4o', 'Claude', 'Llama', 'Qwen'],
    'MMLU': [88.7, 89.2, np.nan, 83.5],
    'HumanEval': [92.0, np.nan, 82.4, np.nan],
})

fallback = pd.DataFrame({
    'model': ['GPT-4o', 'Claude', 'Llama', 'Qwen'],
    'MMLU': [88.0, 89.0, 84.0, 83.0],
    'HumanEval': [91.0, 93.0, 82.0, 79.0],
    'MATH': [76.6, 71.1, 50.4, 60.8],
})

filled = primary.combine_first(fallback)
print("=== combine_first 结果 ===")
print(filled.round(1))
```

**适用场景**：
- 主数据源有缺失时，用备用数据源补充
- 新版数据 + 旧版兜底
- 高优先级评估结果 + 低优先级填充

## update()：就地覆盖更新

```python
import pandas as pd

original = pd.DataFrame({
    'A': [1, 2, 3, 4],
    'B': ['a', 'b', 'c', 'd'],
})

updates = pd.DataFrame({
    'A': [10, np.nan, 30],
    'B': ['X', 'Y', np.nan],
}, index=[0, 1, 2])

original_copy = original.copy()
original_copy.update(updates)
print("=== update 后 ===")
print(original_copy)
```

### overwrite 参数控制

```python
import pandas as pd

df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
new_vals = pd.DataFrame({'A': [10, np.nan, 30], 'B': [np.nan, 50, 60]}, index=[0, 1, 2])

df_default = df.copy()
df_default.update(new_vals)

df_preserve = df.copy()
df_preserve.update(new_vals, overwrite=False)

print("默认 (NaN 覆盖):")
print(df_default)
print("\noverwrite=False (保留原值):")
print(df_preserve)
```

## compare()：两个 DataFrame 的差异对比（Pandas 1.1+）

```python
import pandas as pd
import numpy as np

df1 = pd.DataFrame({
    'model': ['GPT-4o', 'Claude', 'Llama'],
    'v1_score': [88.7, 89.2, 84.5],
    'status': ['active', 'active', 'deprecated'],
})

df2 = pd.DataFrame({
    'model': ['GPT-4o', 'Claude', 'Llama'],
    'v1_score': [89.1, 88.8, 84.5],
    'status': ['active', 'updated', 'active'],
})

diff = df1.compare(df2, result_names=('v1', 'v2'), align_axis='columns')
print("=== 差异对比 ===")
print(diff)
```

## LLM 场景：配置文件合并与模型参数更新

```python
import pandas as pd
import numpy as np

class ModelConfigManager:
    """LLM 模型配置管理器"""

    def __init__(self):
        self.base_config = pd.DataFrame({
            'model': ['gpt-4o', 'claude-sonnet', 'llama-70b', 'qwen-72b'],
            'temperature': [0.7, 0.7, 0.7, 0.7],
            'max_tokens': [4096, 4096, 4096, 4096],
            'top_p': [1.0, 1.0, 1.0, 1.0],
            'presence_penalty': [0.0, 0.0, 0.0, 0.0],
        }).set_index('model')

    def apply_overrides(self, overrides_df):
        """应用自定义覆盖"""
        self.base_config.update(overrides_df)
        return self.base_config.reset_index()

    def merge_with_fallback(self, custom_config):
        """自定义配置 + 默认兜底"""
        result = custom_config.combine_first(self.base_config.reset_index())
        return result


manager = ModelConfigManager()

overrides = pd.DataFrame({
    'temperature': [0.3, 0.2],
    'max_tokens': [8192, 16000],
}, index=['gpt-4o', 'claude-sonnet'])

updated = manager.apply_overrides(overrides)
print("=== 更新后配置 ===")
print(updated.to_string(index=False))

custom = pd.DataFrame({
    'model': ['gpt-4o', 'llama-70b', 'new-model'],
    'temperature': [0.1, 0.5, 0.7],
    'max_tokens': [16384, np.nan, 2048],
}).set_index('model')

merged = manager.merge_with_fallback(custom)
print("\n=== 合并后（NaN 用默认填充）===")
print(merged.round(3).to_string(index=False))
```
