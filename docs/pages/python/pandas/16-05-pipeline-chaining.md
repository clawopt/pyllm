---
title: 高级操作综合实战
description: 管道式链式操作 / pipe() 函数管道 / 方法链最佳实践 / 复杂 ETL 流水线
---
# 管道链式调用


## 链式调用（Method Chaining）模式

```python
import pandas as pd
import numpy as np

np.random.seed(42)

raw = pd.DataFrame({
    'model': ['GPT-4o', 'Claude', 'Llama', 'Qwen', 'DeepSeek', 'Gemini'],
    'MMLU': [88.7, 89.2, 84.5, 83.5, 86.8, 87.3],
    'HumanEval': [92.0, 93.1, 82.4, 78.6, 87.2, 85.1],
    'price': [12.50, 18.00, 0.87, 1.47, 0.42, 6.25],
    'params_B': [1760, 175, 70, 72, 37, None],
})

result = (
    raw.dropna(subset=['params_B'])
    .assign(
        avg_score=lambda d: (d['MMLU'] + d['HumanEval']) / 2,
        value_ratio=lambda d: d['avg_score'] / d['price'],
        tier=lambda d: pd.cut(
            d['avg_score'], bins=[0, 82, 86, 90, 100],
            labels=['C', 'B', 'A', 'S']
        ),
    )
    .sort_values('avg_score', ascending=False)
    .reset_index(drop=True)
)

print(result.to_string(index=False))
```

## pipe()：函数式管道

```python
import pandas as pd

def clean_data(df):
    """清洗数据"""
    return df.dropna().reset_index(drop=True)

def add_features(df):
    """添加特征"""
    df = df.copy()
    if 'MMLU' in df.columns and 'HumanEval' in df.columns:
        df['avg_score'] = (df['MMLU'] + df['HumanEval']) / 2
    return df

def rank_models(df):
    """排名"""
    if 'avg_score' in df.columns:
        df = df.sort_values('avg_score', ascending=False).reset_index(drop=True)
        df.insert(0, 'rank', range(1, len(df)+1))
    return df

raw_df = pd.DataFrame({
    'model': ['GPT-4o', 'Claude', 'Llama', None, 'Qwen'],
    'MMLU': [88.7, 89.2, 84.5, 83.5, np.nan],
    'HumanEval': [92.0, 93.1, 82.4, 78.6, 80.0],
})

result = (
    raw_df
    .pipe(clean_data)
    .pipe(add_features)
    .pipe(rank_models)
)

print(result.to_string(index=False))

def filter_by_tier(df, min_tier='B'):
    tier_order = {'S': 4, 'A': 3, 'B': 2, 'C': 1}
    if 'tier' in df.columns:
        min_val = tier_order.get(min_tier, 0)
        return df[df['tier'].map(tier_order).fillna(0) >= min_val]
    return df
```

## LLM 场景：完整的 SFT 数据处理流水线

```python
import pandas as pd
import numpy as np

class SFTPipeline:
    """SFT 数据处理流水线（链式 + pipe 模式）"""

    def __init__(self):
        self.steps = []

    def run(self, raw_df):
        result = raw_df
        for step_name, step_fn in self.steps:
            before = len(result)
            result = step_fn(result)
            after = len(result)
            print(f"  [{step_name}] {before} → {after} 行")
        return result

    @staticmethod
    def build_standard_pipeline():
        pipeline = SFTPipeline()

        def step_remove_empty(df):
            mask = True
            if 'instruction' in df.columns:
                mask &= df['instruction'].notna() & (df['instruction'].str.len() > 3)
            if 'response' in df.columns:
                mask &= df['response'].notna() & (df['response'].str.len() > 10)
            return df[mask].copy()

        def step_compute_features(df):
            df = df.copy()
            if 'instruction' in df.columns:
                df['instr_len'] = df['instruction'].str.len()
            if 'response' in df.columns:
                df['resp_len'] = df['response'].str.len()
                df['ratio'] = (df['resp_len'] / df['instr_len'].replace(0, 1)).round(2)
            return df

        def step_filter_outliers(df):
            if 'instr_len' in df.columns:
                Q1, Q3 = df['instr_len'].quantile([0.25, 0.75])
                IQR = Q3 - Q1
                df = df[(df['instr_len'] >= Q1 - 3*IQR) &
                        (df['instr_len'] <= Q3 + 3*IQR)]
            return df.reset_index(drop=True)

        def step_assign_quality(df):
            if 'ratio' in df.columns:
                df['quality'] = pd.cut(
                    df['ratio'],
                    bins=[0, 2, 5, 15, float('inf')],
                    labels=['too_short', 'normal', 'good', 'excellent'],
                )
            return df

        pipeline.steps = [
            ('去空值/过短', step_remove_empty),
            ('计算特征', step_compute_features),
            ('过滤异常值', step_filter_outliers),
            ('质量分级', step_assign_quality),
        ]
        return pipeline


np.random.seed(42)
n = 2000
raw_sft = pd.DataFrame({
    'instruction': [
        f'{"问题" * (i % 50 + 1)}_{i}' for i in range(n)
    ] + [None] * int(n * 0.02) + ['短'] * int(n * 0.01),
    'response': [
        f'{"回答" * (i % 30 + 1)}_{i}' for i in range(len(raw_sft) if isinstance(raw_sft.iloc[i % n], str) or True)
    ],
})

pipeline = SFTPipeline.build_standard_pipeline()
print("=== SFT 数据处理流水线 ===")
final = pipeline.run(raw_sft)

print(f"\n最终结果:")
print(f"总行数: {len(final)}")
if 'quality' in final.columns:
    print(f"质量分布:")
    print(final['quality'].value_counts().sort_index())
```
