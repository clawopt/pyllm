---
title: 重复值处理
description: drop_duplicates() 的保留策略选择、子集去重、智能去重保留最优样本，大模型语料去重实战
---
# 重复值处理方法


## 去重的决策树

```
发现重复数据
    │
    ▼
重复的"粒度"是什么？
    │
    ├─ 完全相同的行 → keep='first' / 'last' 删除
    │
    ├─ prompt 相同（不同回答）→ 按质量分排序后保留最好的
    │
    ├─ 用户+问题相同 → 同一用户只保留一次提问
    │
    └─ 内容近似但不完全相同 → 模糊去重（MinHash/Embedding）
```

## 按优先级去重

### 核心模式：按质量分数保留最佳版本

```python
import pandas as pd
import numpy as np

np.random.seed(42)
n = 100_000

df = pd.DataFrame({
    'id': range(n),
    'prompt': [f'问题{i%500}' for i in range(n)],
    'response': [f'回答{i%200}' for i in range(n)],
    'quality_score': np.round(np.random.uniform(1, 5, n), 1),
    'source': np.random.choice(['api', 'web'], n),
})

def dedup_by_priority(df, groupby_cols, priority_col='quality_score',
                       ascending=False):
    """
    按指定列排序后去重，保留每组中优先级最高的行
    
    参数:
        df: 输入 DataFrame
        groupby_cols: 用于判断是否重复的列列表
        priority_col: 用于决定保留哪一行的列名
        ascending: True=保留最小的, False=保留最大的
    """
    
    before = len(df)
    
    sorted_df = df.sort_values(
        by=priority_col,
        ascending=ascending,
        na_position='last'
    )
    
    deduped = sorted_df.drop_duplicates(
        subset=groupby_cols,
        keep='first'
    ).reset_index(drop=True)
    
    after = len(deduped)
    
    print(f"去重: {before:,} → {after:,} (去除 {before-after:,}, {(before-after)/before*100:.1f}%)")
    return deduped


print("=== 场景 A：完全相同 ===")
dedup_a = dedup_by_priority(df, ['prompt', 'response'], 'quality_score')

print("\n=== 场景 B：Prompt 级去重 ===")
dedup_b = dedup_by_priority(df, ['prompt'], 'quality_score')

print(f"\n原始平均质量分: {df['quality_score'].mean():.2f}")
print(f"去重后平均质量分: {dedup_b['quality_score'].mean():.2f}")
print(f"提升: +{dedup_b['quality_score'].mean() - df['quality_score'].mean():.2f}")
```

### 多级去重流水线

```python
import pandas as pd
import numpy as np

def multi_stage_dedup(df, stages):
    """
    多阶段去重
    
    stages: list of dict，每个 dict 包含:
      - name: 阶段名称
      - subset: 去重判断列
      - priority: 排序列
      - ascending: 排序方向
    """
    
    current = df.copy()
    original_len = len(current)
    
    for stage in stages:
        before = len(current)
        
        current = current.sort_values(
            by=stage.get('priority', 'quality_score'),
            ascending=stage.get('ascending', False)
        )
        
        current = current.drop_duplicates(
            subset=stage['subset'],
            keep='first'
        )
        
        removed = before - len(current)
        print(f"  [{stage['name']}] {before:,} → {len(current):,} "
              f"(去除 {removed:,})")
    
    total_removed = original_len - len(current)
    print(f"\n总计: {original_len:,} → {len(current):,} (去除 {total_removed:,}, "
          f"{total_removed/original_len*100:.1f}%)")
    
    return current.reset_index(drop=True)


stages = [
    {
        'name': '第1步 - 完全相同对话',
        'subset': ['prompt', 'response'],
        'priority': 'quality_score',
        'ascending': False,
    },
    {
        'name': '第2步 - 相同 Prompt',
        'subset': ['prompt'],
        'priority': ['quality_score', 'response_length'],
        'ascending': [False, False],
    },
]

clean_data = multi_stage_dedup(df, stages)
```

## 子集去重与条件组合

```python
import pandas as pd

df = pd.DataFrame({
    'user_id': [1, 1, 1, 2, 2, 3, 3, 4],
    'prompt': ['hi', 'hello', 'hi', 'hey', 'hey', 'yo', 'yo', 'sup'],
    'response': [...],
    'quality': [4.0, 3.5, 4.8, 3.9, 4.2, 3.0, 4.5, 3.7],
})

def user_prompt_dedup(df):
    """用户+问题级别去重"""
    
    result_df = []
    
    for (user_id, prompt), group in df.groupby(['user_id', 'prompt']):
        if len(group) == 1:
            result_df.append(group.iloc[0])
        else:
            best_idx = group['quality'].idxmax()
            result_df.append(group.loc[best_idx])
    
    return pd.DataFrame(result_df)


def user_prompt_dedup_fast(df):
    """用户+问题级别去重（高效版）"""
    
    sorted_df = df.sort_values(
        by=['quality_score', 'conversation_id'],
        ascending=[False, True]  # 质量降序，ID 升序（早提交的优先）
    )
    
    return sorted_df.drop_duplicates(
        subset=['user_id', 'prompt'],
        keep='first'
    ).reset_index(drop=True)


deduped = user_prompt_dedup_fast(df)
print(deduped[['user_id', 'prompt', 'quality']])
```

## 去重效果验证

```python
import pandas as pd
import numpy as np

def verify_dedup(original_df, deduped_df, check_columns=None):
    """验证去重结果的正确性"""
    
    print("=" * 50)
    print("  去重结果验证报告")
    print("=" * 50)
    
    checks = []
    
    checks.append(('行数减少', len(deduped_df) <= len(original_df)))
    
    if check_columns:
        remaining_dups = deduped_df.duplicated(subset=check_columns).sum()
        checks.append((f'{check_columns} 无剩余重复', remaining_dups == 0))
    else:
        remaining_dups = deduped_df.duplicated().sum()
        checks.append(('无完全重复行', remaining_dups == 0))
    
    if check_columns:
        orig_set = set(map(tuple, original_df[check_columns].values))
        dedup_set = set(map(tuple, deduped_df[check_columns].values))
        checks.append(('去重后是原数据的子集', dedup_set <= orig_set))
    
    if 'quality_score' in deduped_df.columns:
        avg_before = original_df['quality_score'].mean()
        avg_after = deduped_df['quality_score'].mean()
        checks.append((
            f'平均质量分不下降 ({avg_after:.2f} >= {avg_before:.2f})',
            avg_after >= avg_before * 0.99  # 允许 1% 浮动
        ))
    
    all_pass = all(passed for _, passed in checks)
    
    for name, passed in checks:
        icon = "✅" if passed else "❌"
        print(f"  {icon} {name}")
    
    status = "全部通过 🎉" if all_pass else "存在问题 ⚠️"
    print(f"\n  结果: {status}")
    
    return all_pass
```
