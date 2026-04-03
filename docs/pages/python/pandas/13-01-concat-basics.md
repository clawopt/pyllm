---
title: concat() 数据拼接
description: 纵向/横向拼接、ignore_index、keys 多级索引、join 参数处理列不对齐
---
# Concat 拼接基础


## concat() vs merge() 的核心区别

| 操作 | merge() | concat() |
|------|---------|----------|
| 方向 | 横向（按列扩展） | 纵向或横向 |
| 连接方式 | 按键值匹配 | 按位置/索引对齐 |
| 典型场景 | SQL JOIN 式关联 | 堆叠多个同类数据集 |

## 纵向拼接（最常用）

```python
import pandas as pd
import numpy as np

batch1 = pd.DataFrame({
    'instruction': ['解释什么是 RAG', '如何安装 PyTorch'],
    'response': ['RAG 是检索增强生成...', 'PyTorch 安装步骤...'],
    'source': ['wiki', 'forum'],
})

batch2 = pd.DataFrame({
    'instruction': ['什么是 LoRA', 'BERT 和 GPT 的区别'],
    'response': ['LoRA 是低秩适配...', 'BERT 双向，GPT 单向...'],
    'source': ['textbook', 'paper'],
})

batch3 = pd.DataFrame({
    'instruction': ['什么是 KV Cache'],
    'response': ['KV Cache 缓存键值对...'],
    'source': ['blog'],
})

combined = pd.concat([batch1, batch2, batch3], ignore_index=True)
print(f"合并后: {len(combined)} 条")
print(combined)
```

### ignore_index 参数

```python
with_idx = pd.concat([batch1, batch2], ignore_index=False)
print("保留原始索引:")
print(with_idx.index.tolist())

reset = pd.concat([batch1, batch2], ignore_index=True)
print("\n重置索引:")
print(reset.index.tolist())
```

## 横向拼接

```python
import pandas as pd

df_a = pd.DataFrame({
    'model': ['GPT-4o', 'Claude', 'Llama'],
    'MMLU': [88.7, 89.2, 84.5],
}, index=[0, 1, 2])

df_b = pd.DataFrame({
    'HumanEval': [92.0, 93.1, 82.4],
    'MATH': [76.6, 71.1, 50.4],
}, index=[0, 1, 2])

horizontal = pd.concat([df_a, df_b], axis=1)
print(horizontal)
```

## keys 参数：添加多级索引

```python
import pandas as pd

batches = {
    'Q1': pd.DataFrame({'score': [85, 90, 78]}),
    'Q2': pd.DataFrame({'score': [92, 88, 95]}),
    'Q3': pd.DataFrame({'score': [80, 87, 91]}),
}

multi_idx = pd.concat(batches.values(), keys=batches.keys())
print(multi_idx)
print(f"\n索引层级: {multi_idx.index.names}")
```

### 用多级索引做分组分析

```python
quarterly_stats = multi_idx.groupby(level=0).agg(['mean', 'std', 'count'])
print(quarterly_stats.round(2))
```

## join 参数：处理列不对齐

```python
import pandas as pd

df1 = pd.DataFrame({
    'A': [1, 2],
    'B': [3, 4],
    'C': [5, 6],
})

df2 = pd.DataFrame({
    'B': [30, 40],
    'D': [50, 60],
})

inner_concat = pd.concat([df1, df2], axis=1, join='inner')
print("inner (交集列):")
print(inner_concat)

outer_concat = pd.concat([df1, df2], axis=1, join='outer')
print("\nouter (所有列):")
print(outer_concat)
```

## sort 参数控制排序

```python
import pandas as pd

df1 = pd.DataFrame({'A': [1]}, index=['c'])
df2 = pd.DataFrame({'A': [2]}, index=['a'])

unsorted = pd.concat([df1, df2], sort=False)
print(f"sort=False: {list(unsorted.index)}")

sorted_ = pd.concat([df1, df2], sort=True)
print(f"sort=True:  {list(sorted_.index)}")
```

## LLM 场景：多个评估批次汇总

```python
import pandas as pd
import numpy as np

class BatchAggregator:
    """多批次数据聚合器"""

    def __init__(self):
        self.batches = []

    def add_batch(self, df, batch_name=None):
        if batch_name:
            df = df.copy()
            df['_batch'] = batch_name
        self.batches.append(df)
        return self

    def combine(self, **kwargs):
        if not self.batches:
            return pd.DataFrame()
        return pd.concat(self.batches, ignore_index=True, **kwargs)

    def summary(self):
        combined = self.combine()
        if '_batch' in combined.columns:
            print(f"总记录: {len(combined)}, 批次数: {len(self.batches)}")
            print(f"\n各批次规模:")
            print(combined['_batch'].value_counts().sort_index())
        return combined


np.random.seed(42)

aggregator = BatchAggregator()

for i, source in enumerate(['web_crawl', 'textbook', 'wiki', 'paper']):
    batch_df = pd.DataFrame({
        'instruction': [f'{source}_指令_{j}' for j in range(np.random.randint(20, 50))],
        'instr_len': np.random.exponential(40, np.random.randint(20, 50)).astype(int) + 5,
        'response_len': np.random.exponential(250, np.random.randint(20, 50)).astype(int) + 20,
        'reward_score': np.random.beta(2.5, 4, np.random.randint(20, 50)),
    })
    aggregator.add_batch(batch_df, batch_name=f'batch_{i+1}_{source}')

full_dataset = aggregator.summary()

print(f"\n合并后数据概览:")
print(f"  总条目: {len(full_dataset)}")
print(f"  平均奖励分: {full_dataset['reward_score'].mean():.3f}")
print(f"  高质量占比: {(full_dataset['reward_score'] >= 0.7).mean()*100:.1f}%")
```
