---
title: 迭代器与高效遍历
description: iterrows() / itertuples() / items() / 向量化替代方案 / 大数据集分块处理
---
# 迭代与分块处理


## 遍历 DataFrame 的四种方式

```python
import pandas as pd
import numpy as np

np.random.seed(42)

df = pd.DataFrame({
    'model': ['GPT-4o', 'Claude', 'Llama', 'Qwen'],
    'MMLU': [88.7, 89.2, 84.5, 83.5],
    'price': [2.50, 3.00, 0.27, 0.27],
})
```

### 方式一：iterrows() — 最慢，但最灵活

```python
for idx, row in df.iterrows():
    print(f"{row['model']}: MMLU={row['MMLU']}, Price=${row['price']}")
```

**特点**：返回 `(index, Series)` 元组，每行是一个 Series。**最慢**（10-100x 慢于向量化），但在需要逐行做复杂逻辑时是唯一选择。

### 方式二：itertuples() — 快 10-50x，推荐用于只读遍历

```python
for row in df.itertuples(index=False):
    print(f"{row.model}: MMLU={row.MMLU}, Price=${row.price}")
```

**特点**：返回 `namedtuple`，通过属性访问。比 `iterrows()` **快 10-50 倍**，内存占用更小。

### 方式三：items() — 按列遍历

```python
for col_name, series in df.items():
    print(f"列 '{col_name}': {series.tolist()}")
```

### 方式四：直接向量化 — 最快（始终优先）

```python
result = df['MMLU'] / df['price']
print(result.round(1))
```

## 性能对比基准

```python
import pandas as pd
import numpy as np
import time

n = 10_000
df = pd.DataFrame({
    'a': np.random.randn(n),
    'b': np.random.randn(n),
})

start = time.time()
r1 = [x * y for _, (x, y) in df[['a', 'b']].iterrows()]
t1 = time.time() - start

start = time.time()
r2 = [x.a * x.b for x in df.itertuples(index=False)]
t2 = time.time() - start

start = time.time()
r3 = df['a'] * df['b']
t3 = time.time() - start

print(f"iterrows:   {t1:.4f}s")
print(f"itertuples: {t2:.4f}s ({t1/t2:.1f}x faster)")
print(f"向量化:     {t3:.4f}s ({t1/t3:.1f}x faster)")
```

## LLM 场景：大文件分块处理

```python
import pandas as pd
import numpy as np

class ChunkedProcessor:
    """大数据集分块处理器"""

    def __init__(self, filepath_or_df, chunksize=50_000):
        if isinstance(filepath_or_df, pd.DataFrame):
            self.df = filepath_or_df
            self.is_dataframe = True
        else:
            self.filepath = filepath_or_df
            self.chunksize = chunksize
            self.is_dataframe = False

    def process_in_chunks(self, process_fn, **read_kwargs):
        """分块处理并收集结果"""
        results = []

        if self.is_dataframe:
            df = self.df
            for i in range(0, len(df), self.chunksize):
                chunk = df.iloc[i:i + self.chunksize].copy()
                result = process_fn(chunk)
                if result is not None:
                    results.append(result)
        else:
            reader = pd.read_csv(
                self.filepath,
                chunksize=self.chunksize,
                **read_kwargs
            )
            for chunk in reader:
                result = process_fn(chunk)
                if result is not None:
                    results.append(result)

        if results:
            return pd.concat(results, ignore_index=True)
        return pd.DataFrame()


def clean_sft_chunk(chunk):
    """处理每个 SFT 数据块"""
    chunk = chunk.copy()

    if 'instruction' in chunk.columns:
        chunk['instr_len'] = chunk['instruction'].str.len()
        mask = chunk['instr_len'] > 3
        chunk = chunk[mask]

    if 'response' in chunk.columns:
        chunk['resp_len'] = chunk['response'].str.len()
        chunk = chunk[chunk['resp_len'] >= 10]

    chunk['_chunk_rows'] = len(chunk)
    return chunk


np.random.seed(42)
large_sft = pd.DataFrame({
    'instruction': [f'问题_{i}' for i in range(200_000)],
    'response': [f'回答内容_{i} ' * (i % 10 + 1) for i in range(200_000)],
    'source': np.random.choice(['web', 'book', 'wiki'], 200_000),
})

processor = ChunkedProcessor(large_sft, chunksize=30_000)
cleaned = processor.process_in_chunks(clean_sft_chunk)

print(f"原始: {len(large_sft):,} 条 → 清洗后: {len(cleaned):,} 条")
print(f"各块处理统计:")
if '_chunk_rows' in cleaned.columns:
    print(cleaned.groupby(cleaned.index // 30000)['_chunk_rows'].sum())
```
