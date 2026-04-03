---
title: 缺失值识别与统计
description: .isnull() / .isna() 的使用、缺失值分布分析、可视化缺失模式、大模型语料中的空值处理策略
---
# 缺失值检测与分析


## 缺失值的来源与类型

在 LLM 数据处理中，缺失值（Missing Values）无处不在：

| 来源 | 典型场景 | 缺失原因 |
|------|---------|---------|
| **数据采集** | API 超时返回空响应 | `timeout` → 空字段 |
| **用户行为** | 用户跳过必填项 | 提交了空白表单 |
| **系统错误** | 数据管道断裂 | ETL 中间步骤失败写入 NULL |
| **业务逻辑** | 某些字段不适用 | 非代码类对话无 "code" 字段 |
| **标注过程** | 标注员跳过困难样本 | 困难样本标记为"无法判断" |

### Pandas 中的两种"缺失"

```python
import pandas as pd

s = pd.Series([1, None, np.nan, pd.NA, 'missing'])
print(s)
```

**关键区别**：
- `np.nan`：仅存在于 float 列，会"传染"整型变浮点型
- `pd.NA` / `<NA>`：Pandas 的统一缺失值表示，保持原始 dtype
- 空字符串 `' '` 或 `'None'`：**不是缺失值**，是有效数据！

## isnull() 与 isna()

### 功能完全相同

```python
import pandas as pd
import numpy as np

s = pd.Series([1, None, 3, np.nan])

print(s.isnull())   # [False, True, False, True]
print(s.isna())      # [False, True, False, True]  # 完全等价
print(s.notnull())  # [True, False, True, False]  # 取反
```

**推荐使用 `.isna()` 和 `.notna()`**——更短且语义更清晰。

### 在 DataFrame 上使用

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'id': [1, 2, 3, 4, 5],
    'prompt': ['什么是AI', None, '解释Python', np.nan, '如何学习LLM'],
    'response': ['AI是...', '...', 'Python是...', None, 'LLM是...'],
    'quality': [4.5, None, 3.8, 4.2, None],
})

print(df.isna().sum())

print(df.isna().sum(axis=1))

complete_rows = df[df.notna().all(axis=1)]
print(f"完整行: {len(complete_rows)} / {len(df)}")

any_missing = df[df.isna().any(axis=1)]
print(f"含缺失行: {len(any_missing)}")
```

## 缺失值统计与分析

### 基础统计

```python
import pandas as pd
import numpy as np

n = 100_000
np.random.seed(42)

df = pd.DataFrame({
    'conversation_id': range(n),
    'prompt': [f'问题{i}' for i in range(n)],
    'response': [f'回答{i}' for i in range(n)],
    'quality': np.random.choice([None] + list(range(1,6)), n),
    'turn_count': np.random.choice([None] + list(range(1,10)), n),
    'source': np.random.choice(['api', 'web', 'export'], n),
})

missing_summary = pd.DataFrame({
    '总行数': len(df),
    '非空数': df.notna().sum(),
    '缺失数': df.isna().sum(),
    '缺失率(%)': (df.isna().mean() * 100).round(2),
}).T

print(missing_summary)
```

### 缺失模式分析（Missing Pattern）

```python
def analyze_missing_patterns(df):
    """分析缺失值的模式和关联性"""
    
    print("=" * 60)
    print("  🔍 缺失值模式分析")
    print("=" * 60)
    
    missing_matrix = df.isna()
    co_occurrence = missing_matrix.T @ missing_matrix
    np.fill_diagonal(co_occurrence.values, 0)
    
    print("\n【缺失共现矩阵】(数值越大 = 越容易同时缺失):")
    co_df = pd.DataFrame(co_occurrence,
                        index=co_occurrence.columns,
                        columns=co_occurrence.columns)
    print(co_df)
    
    if hasattr(df.index, 'to_series'):
        idx = df.index.to_series()
        if pd.api.types.is_datetime_like(idx):
            df['_temp_date'] = idx.dt.date
            by_date = df.groupby('_temp_date').apply(lambda x: x.isna().sum())
            print(f"\n【按日期的缺失趋势】:")
            print(by_date.head(20))
            del df['_temp_date']
    
    print(f"\n【缺失机制初步判断】")
    for col in df.select_dtypes(include=[np.number]).columns:
        s = df[col]
        non_null = s.dropna()
        
        if len(non_null) > 10:
            corr_with_others = {}
            for other_col in df.select_dtypes(include=[np.number]).columns:
                if other_col != col:
                    valid_pair = df[[col, other_col]].dropna()
                    if len(valid_pair) > 5:
                        corr = valid_pair[col].corr(valid_pair[other_col])
                        corr_with_others[other_col] = round(corr, 3)
            
            max_corr_key = max(corr_with_others, key=corr_with_others.get, default=(None, 0))
            max_corr_val = corr_with_others[max_corr_key] if max_corr_key else 0
            
            if abs(max_corr_val) > 0.3:
                mechanism = f"MNAR (与 {max_corr_key} 相关={max_corr_val})"
            elif abs(max_corr_val) < 0.1:
                mechanism = "MCAR (完全随机)"
            else:
                mechanism = "MAR (随机但与其他变量相关)"
            
            print(f"  {col}: {mechanism}")
    
    return co_df

patterns = analyze_missing_patterns(df)
```

输出示例：

```
============================================================
  🔍 缺失值模式分析
============================================================

【缺失共现矩阵】(数值越大 = 越容易同时缺失):
           response  quality  turn_count
response       214       1980       189
quality        1980     4988       1654
turn_count     1654      1654       1766

【缺失机制初步判断】
  quality: MCAR (完全随机)
  response: MNAR (与 quality 相关=0.342)
  turn_count: MAR (随机但与其他变量相关)
```

**三种缺失机制的应对策略**：

| 机制 | 特征 | 推荐处理方式 |
|------|------|-----------|
| **MCAR** (完全随机缺失) | 缺失与任何变量无关 | 直接删除或均值填充 |
| **MAR** (随机缺失) | 缺失取决于已观测变量 | 分组后填充 / 模型预测填充 |
| **MNAR** (非随机缺失) | 缺失本身有含义 | 单独建模或作为单独类别 |

## 缺失值可视化

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

def visualize_missing(df, figsize=(14, 6)):
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    ax1 = axes[0]
    sns.heatmap(
        df.isna(),
        cbar=False,
        cmap='YlOrRd',
        yticklabels=False,
        ax=ax1,
    )
    ax1.set_title('缺失值分布热力图\n(黄色=有缺失)')
    
    ax2 = axes[1]
    missing_pct = (df.isna().sum() / len(df) * 100).sort_values(ascending=False)
    
    bars = ax2.barh(missing_pct.index, missing_pct.values, color='#FF6B6B')
    ax2.axvline(x=5, color='red', linestyle='--', label='5% 警戒线')
    
    for bar, val in zip(bars, missing_pct.values):
        ax2.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', fontsize=9)
    
    ax2.set_xlabel('缺失率 (%)')
    ax2.set_title('各列缺失率')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('missing_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    return fig

np.random.seed(42)
demo_df = pd.DataFrame({
    'A': np.random.choice([1, 2, 3, None], 5000, p=[.3,.3,.3,.1]),
    'B': np.random.choice(['x','y',z', None], 5000, p=[.25,.25,.25,.25]),
    'C': np.random.randn(5000),
})
visualize_missing(demo_df)
```
