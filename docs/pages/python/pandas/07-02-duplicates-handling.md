---
title: 重复值处理
description: drop_duplicates() 的保留策略选择、子集去重、智能去重保留最优样本，大模型语料去重实战
---
# 去重实战：从检测到清理

上一节我们讨论了如何用 `duplicated()` 检测重复数据。这一节要把检测变成行动——用 `drop_duplicates()` 做实际的去重操作，并且重点讲解 LLM 场景下最实用的"按质量排序后去重"模式。

## 去重的核心决策：保留哪一条

当你发现数据中有重复时，最关键的问题不是"要不要删"而是"删掉之后保留哪一份"。`drop_duplicates()` 的 `keep` 参数给了你三个选项：

```python
import pandas as pd

df = pd.DataFrame({
    'text': ['A', 'B', 'A', 'C', 'B'],
    'quality': [4.5, 3.0, 4.8, 3.9, 2.5],
})

print("保留第一次出现的:")
print(df.drop_duplicates(subset=['text'], keep='first'))

print("\n保留最后一次出现的:")
print(df.drop_duplicates(subset=['text'], keep='last'))
```

`keep='first'` 保留每组中第一行出现的记录（默认），`keep='last'` 保留最后一行。但这两个都是"碰运气"的策略——你无法控制到底保留了质量好的还是差的。

## 按优先级排序后去重：生产级做法

在实际项目中，你几乎总是应该**先排序再去重**。这样 `keep='first'` 就能保证保留的是每组中优先级最高的那条：

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

def dedup_by_quality(df, group_cols):
    before = len(df)
    
    sorted_df = df.sort_values('quality_score', ascending=False, na_position='last')
    deduped = sorted_df.drop_duplicates(subset=group_cols, keep='first')
    deduped = deduped.reset_index(drop=True)
    
    after = len(deduped)
    print(f"去重: {before:,} → {after:,} (去除 {before-after:,}, {(before-after)/before*100:.1f}%)")
    print(f"平均质量分: {df['quality_score'].mean():.2f} → {deduped['quality_score'].mean():.2f}")
    
    return deduped

result = dedup_by_quality(df, ['prompt'])
```

注意输出中平均质量分的变化——因为去重时丢弃了低质量的重复项，保留的全是高质量版本，所以整体均值会上升。这就是"智能去重"的价值：它不仅减少了数据量，还提升了数据质量。

## 多级去重流水线

实际项目中的去重往往不是一步完成的。你可能需要先做严格去重（完全相同的行），再做内容级去重（prompt+response 相同），最后做 prompt 级去重。每一步的粒度不同，保留策略也可以不同：

```python
def multi_stage_dedup(df):
    original = len(df)
    
    stage1 = df.drop_duplicates(keep='first')
    print(f"Stage 1 (全量去重): {len(df):,} → {len(stage1):,}")
    
    stage2 = stage1.sort_values('quality_score', ascending=False)\
                  .drop_duplicates(subset=['prompt','response'], keep='first')\
                  .reset_index(drop=True)
    print(f"Stage 2 (内容去重): {len(stage1):,} → {len(stage2):,}")
    
    stage3 = stage2.sort_values('quality_score', ascending=False)\
                  .drop_duplicates(subset=['prompt'], keep='first')\
                  .reset_index(drop=True)
    print(f"Stage 3 (Prompt去重): {len(stage2):,} → {len(stage3):,}")
    
    total_removed = original - len(stage3)
    print(f"\n总计: {original:,} → {len(stage3):,} (去除 {total_removed:,}, {total_removed/original*100:.1f}%)")
    return stage3

result = multi_stage_dedup(df.copy())
```

多级去重的逻辑是逐级收紧去重条件：第一步去掉完全重复的数据录入错误；第二步去掉内容相同的多版本记录（可能来自不同采集时间）；第三步合并同一问题的多次提问（只保留最佳回答）。每一级的粒度都更粗、删除量都更大，但同时也越来越激进——你需要根据具体业务需求决定执行到哪一级。
