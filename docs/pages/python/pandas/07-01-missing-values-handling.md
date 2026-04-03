---
title: 缺失值处理
description: dropna() / fillna() / interpolate() 的使用策略，大模型场景下对话数据空响应的处理方案
---
# 缺失值处理策略


## 处理缺失值的三种基本策略

```
遇到缺失值时：
│
├─ 策略一：删除（dropna）
│   适用：缺失比例低 (< 5%) 且随机分布
│   风险：丢失信息
│
├─ 策略二：填充（fillna）
│   适用：需要保持数据量、缺失有规律
│   方法：均值/中位数/前值/后值/模型预测
│
└─ 策略三：保留（标记为特殊值）
    适用：缺失本身有意义
    方法：用 pd.NA 或单独类别表示
```

## dropna()：删除

### 基础用法

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'id': [1, 2, 3, 4, 5],
    'prompt': ['什么是AI', None, '解释Python', np.nan, '如何学习LLM'],
    'response': ['AI是...', '...', 'Python是...', None, 'LLM是...'],
    'quality': [4.5, None, 3.8, 4.2, None],
})

clean_all = df.dropna()
print(f"全量删除: {len(df)} → {len(clean_all)} (去除 {len(df)-len(clean_all)})")

clean_subset = df.dropna(subset=['prompt', 'response'])
print(f"按列删除: {len(df)} → {len(clean_subset)}")

clean_all_nan = df.dropna(how='all')
print(f"全空行删除: {len(df)} → {len(clean_all_nan)}")

clean_thresh = df.dropna(thresh=3)
print(f"阈值删除(>=3个非空): {len(df)} → {len(clean_thresh)}")
```

### LLM 场景的删除策略

```python
def smart_dropna_for_llm(df):
    """针对 LLM 对话数据的智能删除"""
    
    original = len(df)
    
    df = df[df['prompt'].notna()]
    df = df[df['prompt'].str.strip() != '']
    
    if 'response' in df.columns:
        df = df[df['response'].notna()]
    
    if 'quality_score' in df.columns:
        df = df.dropna(subset=['quality_score'])
    
    removed = original - len(df)
    print(f"智能去空: {original:,} → {len(df):,} (去除 {removed:,}, {removed/original*100:.1f}%)")
    
    return df.reset_index(drop=True)
```

## fillna()：填充

### 数值型填充

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'value': [1.0, np.nan, 3.0, np.nan, 5.0, np.nan],
    'score': [85, np.nan, 92, np.nan, 78, 88],
})

df['value_zero'] = df['value'].fillna(0)
df['value_neg1'] = df['value'].fillna(-1)

mean_val = df['value'].mean()
df['value_mean'] = df['value'].fillna(mean_val)

median_val = df['score'].median()
df['score_median'] = df['score'].fillna(median_val)

print(df.round(2))
```

### 前向/后向填充（时间序列常用）

```python
import pandas as pd

ts_df = pd.DataFrame({
    'date': pd.date_range('2025-01-01', periods=10),
    'daily_users': [1000, np.nan, 1200, np.nan, 1300,
                     1250, np.nan, 1400, 1350, np.nan],
})

ts_df['ffill'] = ts_df['daily_users'].ffill()

ts_df['bfill'] = ts_df['daily_users'].bfill()

ts_df['ffill_limit'] = ts_df['daily_users'].fillna(method='ffill', limit=2)

print(ts_df)
```

### 分类变量填充

```python
import pandas as pd

df = pd.DataFrame({
    'category': ['tech', 'finance', None, 'tech', None, 'medical'],
    'source': [None, 'api', 'web', 'api', None, 'export'],
})

mode_cat = df['category'].mode()[0]
df['cat_filled'] = df['category'].fillna(mode_cat)

df['cat_unknown'] = df['category'].fillna('<UNKNOWN>')

df['cat_category'] = df['category'].astype('category')
df['cat_category'] = df['cat_category'].cat.add_categories(['<MISSING>']).fillna('<MISSING>')

print(df)
```

### interpolate()：插值填充

适用于**有序数据**中的少量缺失：

```python
import pandas as pd
import numpy as np

s = pd.Series([1, np.nan, np.nan, 4, 5, np.nan, 7])

print(s.interpolate())

ts = pd.Series(
    [1, np.nan, np.nan, 4],
    index=[0, 1, 5, 10]  # 不等间距的时间戳
)
print(ts.interpolate(method='index'))
```

## LLM 数据的完整清洗流程

```python
import pandas as pd
import numpy as np

np.random.seed(42)
n = 500_000

raw = pd.DataFrame({
    'conversation_id': range(n),
    'user_message': [f'问题{i%200}' for i in range(n)],
    'assistant_message': [f'回答{i}' for i in range(n)],
    'turn_count': np.random.choice([None] + list(range(1, 8)), n),
    'quality_score': np.round(
        np.random.choice([None] + list(np.arange(1.0, 5.1, 0.5)), n), 1
    ),
    'token_count': np.random.randint(20, 2000, n).astype(float),
    'source': np.random.choice([None, 'api', 'web', 'export'], n),
})


class LLMMissingHandler:
    def __init__(self, df):
        self.df = df.copy()
        self.log = []
    
    def _log(self, msg):
        self.log.append(msg)
        print(f"  [{msg}]")
    
    def handle(self):
        before = len(self.df)
        
        self._log("Step 1: 删除关键字段缺失行")
        self.df = self.df[self.df['user_message'].notna()].copy()
        self.df = self.df[self.df['assistant_message'].notna()].copy()
        
        self._log("Step 2: 清理空白文本")
        self.df['user_message'] = self.df['user_message'].str.strip().replace(r'^\s*$', np.nan, regex=True)
        self.df = self.df.dropna(subset=['user_message', 'assistant_message'])
        
        self._log("Step 3: 填充 turn_count 缺失值为 1")
        self.df['turn_count'] = self.df['turn_count'].fillna(1).astype(int)
        
        self._log("Step 4: quality_score 缺失值用中位数填充")
        median_q = self.df['quality_score'].median()
        self.df['quality_was_missing'] = self.df['quality_score'].isna()
        self.df['quality_score'] = self.df['quality_score'].fillna(median_q)
        
        self._log("Step 5: source 缺失值填 unknown")
        self.df['source'] = self.df['source'].fillna('unknown').astype('category')
        
        after = len(self.df)
        self._log(f"完成: {before:,} → {after:,} (去除 {before-after:,})")
        
        return self.df.reset_index(drop=True)


handler = LLMMissingHandler(raw)
cleaned = handler.handle()

print("\n=== 最终统计 ===")
print(cleaned.isna().sum())
print(f"\n各列 dtype:")
print(cleaned.dtypes)
```
