---
title: 重复值检测
description: .duplicated() / .drop_duplicates() 的使用、子集去重、保留策略选择，大模型语料去重实战
---
# 重复数据检测


## 为什么重复值是 LLM 数据的头号敌人

在构建训练数据集时，重复数据会导致：

| 问题 | 影响 | 严重程度 |
|------|------|---------|
| **过拟合** | 模型记住重复样本而非学习模式 | 🔴 高 |
| **评估偏差** | 测试集和训练集有重叠 | 🔴 高 |
| **资源浪费** | 相同数据重复计算梯度 | 🟡 中 |
| **分布失真** | 某类样本被过度代表 | 🟡 中 |

### 典型来源

```python
api_data = pd.DataFrame({'prompt': ['什么是AI'] * 100})
web_data = pd.DataFrame({'prompt': ['什么是AI'] * 50})
combined = pd.concat([api_data, web_data])

```

## duplicated()：识别重复

### 基础用法

```python
import pandas as pd

df = pd.DataFrame({
    'id': [1, 2, 3, 2, 5, 6, 1],
    'text': ['hello', 'world', 'foo', 'world', 'bar', 'baz', 'hello'],
    'cat': ['A', 'B', 'C', 'B', 'D', 'E', 'A'],
})

print(df.duplicated(keep=False))

print(df.duplicated())

print(df.duplicated(keep='last'))

n_duplicates = df.duplicated().sum()
total = len(df)
print(f"重复行数: {n_duplicates} ({n_duplicates/total*100:.1f}%)")
```

### 按子集检查重复

这是**最常用**的方式——判断哪些字段组合构成"同一条记录"：

```python
import pandas as pd

df = pd.DataFrame({
    'conversation_id': [1, 2, 3, 4, 5],
    'user_id': [101, 102, 101, 103, 104],
    'prompt': ['hi', 'hello', 'hi', 'hey', 'yo'],
    'response': ['there', 'world', 'there!', 'you', 'man'],
})

strict_dup = df.duplicated(subset=['prompt', 'response'])
print(f"严格重复: {strict_dup.sum()} 条")

user_prompt_dup = df.duplicated(subset=['user_id', 'prompt'])
print(f"用户+问题重复: {user_prompt_dup.sum()} 条")

prompt_only_dup = df.duplicated(subset=['prompt'])
print(f"仅 prompt 重复: {prompt_only_dup.sum()} 条")
```

### 去重策略矩阵

| 策略 | subset 参数 | 适用场景 | 风险 |
|------|-----------|---------|------|
| **全量去重** | 不指定（全部列） | 完全相同的记录 | 可能误删合理变体 |
| **内容去重** | `['prompt', 'response']` | 对话级去重 | 不同用户问同样的问题会被删 |
| **用户级去重** | `['user_id', 'prompt']` | 同一用户的重复提问 | 保留不同回答的版本 |
| **Prompt 去重** | `['prompt']` | 问题级去重 | 丢失多答案信息 |

## drop_duplicates()：删除重复

### 基本用法与保留策略

```python
import pandas as pd

df = pd.DataFrame({
    'id': [1, 2, 3, 4, 5, 6],
    'text': ['A', 'B', 'A', 'C', 'B', 'D'],
    'quality': [4.5, 3.0, 4.8, 3.9, 2.5, 4.0],
})

dedup_first = df.drop_duplicates(subset=['text'], keep='first')
print(dedup_first)

dedup_last = df.drop_duplicates(subset=['text'], keep='last')
print(dedup_last)

dedup_none = df.drop_duplicates(subset=['text'], keep=False)
print(dedup_none)
```

### 大模型场景：智能去重——保留质量最高的版本

```python
import pandas as pd
import numpy as np

np.random.seed(42)
n = 50_000

df = pd.DataFrame({
    'conversation_id': range(n),
    'prompt': [f'问题{i%100}' for i in range(n)],
    'response': [f'回答{i}' for i in range(n)],
    'quality_score': np.round(np.random.uniform(1, 5, n), 1),
    'source': np.random.choice(['api', 'web', 'export'], n),
})

def smart_deduplicate(df, subset_cols, priority_col='quality_score'):
    """按指定列排序后去重，保留优先级最高的行"""
    
    sorted_df = df.sort_values(by=priority_col, ascending=False)
    
    deduped = sorted_df.drop_duplicates(subset=subset_cols, keep='first')
    
    return deduped.reset_index(drop=True)


before = len(df)
after_dedup = smart_deduplicate(df, ['prompt'], 'quality_score')

print(f"去重前: {before:,} 条")
print(f"去重后: {len(after_dedup):,} 条")
print(f"去除: {before - len(after_dedup):,} 条 ({(before-len(after_dedup))/before*100:.1f}%)")

print(f"\n去重后的质量分布:")
print(after_dedup['quality_score'].describe().round(2))
```

### 基于哈希的高效去重

当数据量极大时，直接比较字符串很慢。用哈希加速：

```python
import pandas as pd
import hashlib

def hash_based_dedup(df, columns_to_check):
    """基于内容哈希的去重"""
    
    def compute_row_hash(row):
        content = '|'.join(str(row[c]) for c in columns_to_check)
        return hashlib.md5(content.encode()).hexdigest()
    
    df['_content_hash'] = df.apply(compute_row_hash, axis=1)
    
    before = len(df)
    deduped = df.drop_duplicates(subset='_content_hash').copy()
    after = len(deduped)
    
    deduped = deduped.drop(columns='_content_hash')
    
    print(f"哈希去重: {before:,} → {after:,} (去除 {before-after:,})")
    return deduped


import time

n_test = 500_000
test_df = pd.DataFrame({
    'a': [f'text_{i%10000}' for i in range(n_test)],
    'b': [f'data_{i%5000}' for i in range(n_test)],
})

start = time.time()
result_normal = test_df.drop_duplicates(subset=['a', 'b'])
t_normal = time.time() - start

start = time.time()
result_hash = hash_based_dedup(test_df.copy(), ['a', 'b'])
t_hash = time.time() - start

assert len(result_normal) == len(result_hash)
print(f"\n普通去重: {t_normal:.2f}s")
print(f"哈希去重: {t_hash:.2f}s")
print(f"速度比: {t_normal/t_hash:.1f}x")
```

## 近似去重（海量数据）

对于千万级行的数据，精确去重可能太慢。可以使用**近似去重**：

```python
from datasketch import MinHashLSH

def approximate_dedup(df, text_col, threshold=0.85,
                     num_perm=128, num_buckets=10):
    """
    使用 MinHash + LSH 进行近似文本去重
    
    原理：
    1. 将每条文本转为 MinHash 签名
    2. 用 LSH 快速找到相似的签名对
    3. 只对候选对做精确比较
    
    适用于：百万级以上文本数据的模糊去重
    """
    from datasketch import MinHash
    
    lsh = MinHashLSH(num_perm=num_perm, params=[threshold, num_buckets])
    minhashes = {}
    
    for idx, row in df.iterrows():
        text = str(row[text_col])
        
        words = set(text.lower().split())
        m = MinHash(num_perm=num_perm)
        for word in words:
            m.update(word.encode('utf-8'))
        
        minhashes[idx] = m
        lsh.insert(idx, m)
    
    duplicate_groups = []
    visited = set()
    
    for idx, m in minhashes.items():
        if idx in visited:
            continue
        
        results = lsh.query(m)
        if len(results) > 1:
            group = set(results)
            visited.update(group)
            duplicate_groups.append(group)
    
    rows_to_keep = set(range(len(df)))
    for group in duplicate_groups:
        best_idx = max(group, key=lambda x: df.iloc[x].get('quality_score', 0))
        group.remove(best_idx)
        rows_to_keep -= group
    
    return df.iloc[list(rows_to_keep)].reset_index(drop=True)
```
