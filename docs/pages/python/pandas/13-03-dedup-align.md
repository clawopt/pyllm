---
title: 数据去重合并与连接实战
description: 去重后合并 / 多源数据对齐 / SFT 数据集整合 / API 日志关联分析
---
# 去重与对齐


## 去重后再合并

```python
import pandas as pd

source_a = pd.DataFrame({
    'id': [1, 2, 3, 4],
    'value': [10, 20, 30, 40],
})

source_b = pd.DataFrame({
    'id': [3, 4, 5, 6],
    'score': [80, 90, 70, 60],
})

merged = pd.merge(
    source_a.drop_duplicates('id'),
    source_b.drop_duplicates('id'),
    on='id',
    how='outer',
    indicator=True,
)
print(merged)
```

## 多源数据对齐

```python
import pandas as pd
import numpy as np

class DataAligner:
    """多数据源对齐工具"""

    @staticmethod
    def align_columns(dfs, fill_value=np.nan):
        """统一所有 DataFrame 的列"""
        all_cols = set()
        for df in dfs:
            all_cols.update(df.columns)

        aligned = []
        for df in dfs:
            missing_cols = all_cols - set(df.columns)
            if missing_cols:
                for col in missing_cols:
                    df[col] = fill_value
            aligned.append(df[sorted(all_cols)])
        return aligned

    @staticmethod
    def safe_concat(dfs, **kwargs):
        """安全拼接（自动对齐列）"""
        aligned = DataAligner.align_columns(dfs)
        return pd.concat(aligned, ignore_index=True, **kwargs)


df1 = pd.DataFrame({'model': ['A', 'B'], 'score': [88, 92], 'latency': [800, 650]})
df2 = pd.DataFrame({'model': ['C', 'D'], 'score': [84, 83], 'cost': [2.5, 0.27]})
df3 = pd.DataFrame({'model': ['E', 'F'], 'latency': [350, 400], 'cost': [0.14, 0.27]})

aligned_dfs = DataAligner.align_columns([df1, df2, df3])
for i, adf in enumerate(aligned_dfs):
    print(f"DataFrame {i+1} 列: {list(adf.columns)}")

combined = DataAligner.safe_concat([df1, df2, df3])
print(f"\n对齐后合并: {len(combined)} 行 × {len(combined.columns)} 列")
print(combined.fillna('-').to_string(index=False))
```
