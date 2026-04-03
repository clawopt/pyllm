---
title: 多级索引（MultiIndex）深度操作
description: 创建 MultiIndex / set_index() 多级 / 交叉创建 / 层级操作 xs() / stack() unstack()
---
# MultiIndex 多级索引


## 什么是 MultiIndex

MultiIndex 允许 DataFrame 的行或列有**多个层级索引**，非常适合表示高维数据：

```
                    MMLU    HumanEval
model       vendor
GPT-4o      OpenAI   88.7     92.0
Claude      Anthropic 89.2     93.1
Llama       Meta     84.5     82.4
```

## 创建 MultiIndex

### 方式一：set_index() 从列创建

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'vendor': ['OpenAI', 'Anthropic', 'Meta', 'Alibaba'],
    'model': ['GPT-4o', 'Claude', 'Llama', 'Qwen'],
    'MMLU': [88.7, 89.2, 84.5, 83.5],
})

multi_idx = df.set_index(['vendor', 'model'])
print(multi_idx)
```

### 方式二：from_arrays / from_tuples / from_product

```python
import pandas as pd
import numpy as np

idx = pd.MultiIndex.from_arrays([
    ['A', 'A', 'B', 'B'],
    ['x', 'y', 'x', 'y']
], names=['L1', 'L2'])

idx_product = pd.MultiIndex.from_product(
    [['GPT-4o', 'Claude', 'Llama'], ['MMLU', 'HumanEval']],
    names=['model', 'metric']
)
print(f"笛卡尔积 MultiIndex ({len(idx_product)} 行):")
print(idx_product)
```

### 方式三：pivot_table 自动生成

```python
import pandas as pd
import numpy as np

np.random.seed(42)

raw = pd.DataFrame({
    'model': ['GPT-4o']*3 + ['Claude']*3 + ['Llama']*3,
    'task': ['chat', 'code', 'math'] * 3,
    'score': np.random.uniform(75, 95, 9),
})

pivot = raw.pivot_table(
    index='model',
    columns='task',
    values='score'
)
print(pivot)
print(f"\n列是 MultiIndex? {isinstance(pivot.columns, pd.MultiIndex)}")
```

## 层级操作

### xs(): 按层级值选取

```python
import pandas as pd

df = pd.DataFrame(
    {'value': range(8)},
    index=pd.MultiIndex.from_tuples([
        ('A', 'x', 1), ('A', 'x', 2),
        ('A', 'y', 1), ('A', 'y', 2),
        ('B', 'x', 1), ('B', 'x', 2),
        ('B', 'y', 1), ('B', 'y', 2),
    ], names=['L1', 'L2', 'L3'])
)

a_rows = df.xs('A', level='L1')
print("L1='A':")
print(a_rows)

ax = df.xs(('A', 'x'), level=['L1', 'L2'])
print("\nL1='A', L2='x':")
print(ax)
```

### swaplevel() / reorder_levels()

```python
import pandas as pd

df = pd.DataFrame(
    {'val': range(6)},
    index=pd.MultiIndex.from_product(
        [['GPT-4o', 'Claude'], ['v1', 'v2', 'v3']],
        names=['model', 'version']
    )
)

swapped = df.swaplevel('model', 'version')

reordered = df.reorder_levels(['version', 'model']).sort_index(level=0)
```

## LLM 场景：多维度评估数据管理

```python
import pandas as pd
import numpy as np

class MultiDimEvaluator:
    """多维度评估数据管理器"""

    def __init__(self):
        self.data = None

    def build_multi_index_data(self):
        """构建多级索引的评估数据"""
        np.random.seed(42)

        models = ['GPT-4o', 'Claude', 'Llama', 'Qwen']
        versions = ['Mar', 'Apr', 'May']
        tasks = ['MMLU', 'HumanEval', 'MATH']

        idx = pd.MultiIndex.from_product(
            [models, versions, tasks],
            names=['model', 'version', 'task']
        )

        base_map = {'GPT-4o': 88, 'Claude': 89, 'Llama': 84, 'Qwen': 83}
        task_mod = {'MMLU': 0, 'HumanEval': 4, 'MATH': -10}

        values = []
        for model, ver, task in idx:
            base = base_map[model] + task_mod.get(task, 0)
            month_boost = (versions.index(ver) * 0.5) if ver != 'Mar' else 0
            score = round(base + month_boost + np.random.randn() * 3, 1)
            values.append(max(score, 30))

        self.data = pd.DataFrame({'score': values}, index=idx)
        return self

    def query_by_model(self, model_name):
        return self.data.xs(model_name, level='model')

    def query_by_task(self, task_name):
        return self.data.xs(task_name, level='task')

    def pivot_to_wide(self):
        return self.data.unstack('task')['score'].round(1)

    def version_comparison(self):
        wide = self.pivot_to_wide()
        for model in wide.index:
            march_score = wide.loc[model, 'Mar'].mean()
            may_score = wide.loc[model, 'May'].mean()
            delta = may_score - march_score
            arrow = "↑" if delta > 0 else "↓"
            print(f"  {model:<12s} Mar→May: {march_score:.1f} → "
                  f"{may_score:.1f} ({arrow}{abs(delta):.1f})")


evaluator = MultiDimEvaluator()
evaluator.build_multi_index_data()

print("=== 多级索引结构 ===")
print(evaluator.data.head(12))

print("\n=== GPT-4o 各版本各任务 ===")
print(evaluator.query_by_model('GPT-4o'))

print("\n=== 版本间对比 ===")
evaluator.version_comparison()

print("\n=== 宽格式透视表 ===")
wide = evaluator.pivot_to_wide()
print(wide.round(1))
```
