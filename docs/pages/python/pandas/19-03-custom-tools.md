---
title: 自定义 Tool 与 Agent 增强
description: 扩展 Pandas Agent 的能力、添加自定义分析工具、多 DataFrame 协作
---
# 增强 Agent：自定义工具与多表协作

默认的 Pandas Agent 只能操作单个 DataFrame。但在实际项目中你经常需要同时查询多个表（比如"用户表 + 对话表 + 标注表"），或者调用自定义的分析函数。

## 添加自定义工具

```python
from langchain_core.tools import tool
import pandas as pd

@tool
def get_quality_distribution(df) -> str:
    """获取质量分数的分布统计"""
    dist = df['quality_score'].value_counts().sort_index()
    return f"质量分布:\n{dist.to_string()}"

@tool
def detect_outliers(df, column, threshold=3):
    """检测某列的组内异常值"""
    mean = df[column].mean()
    std = df[column].std()
    outliers = df[abs(df[column] - mean) > threshold * std]
    return f"{column} 异常值: {len(outliers)} 条 (阈值: ±{threshold}σ)"
```

通过 `@tool` 装饰器，任何 Python 函数都可以变成 Agent 可调用的工具。Agent 在推理过程中会根据你的问题自动选择合适的工具——如果问"数据的质量分布怎么样"，它就会调用 `get_quality_distribution` 而不是自己去写 groupby 代码。
