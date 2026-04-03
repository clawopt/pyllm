---
title: I/O 性能优化
description: Parquet 优势 / 列裁剪 / 分区读取 / dtype 指定 / chunksize 策略
---
# I/O 性能优化


## I/O 格式性能对比

| 格式 | 100万行写入 | 100万行读取 | 文件大小 | 支持列裁剪 |
|------|-----------|-----------|---------|-----------|
| CSV | 基准(1x) | 基准(1x) | 100% | ❌ |
| JSONL | ~0.8x | ~0.7x | ~80% | ❌ |
| **Parquet** | **~3-5x** | **~5-10x** | **~23%** | ✅ |
| Feather | ~4x | ~8x | ~25% | ✅ |

## Parquet 最佳实践

### 写入优化

```python
import pandas as pd
import numpy as np

np.random.seed(42)
n = 1_000_000

df = pd.DataFrame({
    'model': np.random.choice(['GPT-4o', 'Claude', 'Llama', 'Qwen'], n),
    'task': np.random.choice(['chat', 'code', 'math'], n),
    'score': np.random.uniform(60, 98, n).round(2),
    'latency_ms': np.random.randint(100, 5000, n),
    'timestamp': pd.date_range('2025-01-01', periods=n, freq='s'),
})

df.to_parquet('/tmp/llm_data.parquet', engine='pyarrow', index=False)

df.to_parquet(
    '/tmp/llm_partitioned',
    partition_cols=['model'],
    engine='pyarrow',
    index=False,
)
```

### 读取优化：只读需要的列

```python
import pandas as pd

subset = pd.read_parquet(
    '/tmp/llm_data.parquet',
    columns=['model', 'score', 'latency_ms']
)

print(f"读取 {len(subset)} 行, {len(subset.columns)} 列")
print(subset.head())
```

### 读取优化：分区过滤下推

```python
import pandas as pd

gpt_only = pd.read_parquet(
    '/tmp/llm_partitioned',
    filters=[('model', '=', 'GPT-4o')],
    columns=['task', 'score', 'latency_ms'],
)
print(f"GPT-4o 数据: {len(gpt_only)} 行")
```

## CSV 读取优化技巧

```python
import pandas as pd
import numpy as np

n = 500_000
csv_file = '/tmp/large_llm.csv'

large_df = pd.DataFrame({
    'id': range(n),
    'model': np.random.choice(['GPT-4o', 'Claude', 'Llama'], n),
    'prompt': [f'测试数据{i}' for i in range(n)],
    'score': np.random.uniform(0.6, 0.99, n).round(4),
})
large_df.to_csv(csv_file, index=False)

dtypes = {
    'id': 'int32',
    'model': 'category',
    'score': 'float32',
}

start_opt = pd.read_csv(csv_file, dtype=dtypes, usecols=['id', 'model', 'score'])

sample = pd.read_csv(csv_file, nrows=1000)
print(f"采样: {len(sample)} 行")

chunk_results = []
for chunk in pd.read_csv(csv_file, chunksize=50_000):
    chunk_summary = chunk.groupby('model')['score'].mean()
    chunk_results.append(chunk_summary)

combined = pd.concat(chunk_results)
final = combined.groupby(combined.index).mean()
print(f"\n各模型平均分 (分块聚合):")
print(final.round(4))
```

## LLM 场景：大规模语料加载策略

```python
import pandas as pd
import os

class EfficientLoader:
    """高效数据加载器"""

    @staticmethod
    def load_sft_corpus(path_pattern, columns=None, sample_ratio=1.0):
        """高效加载 SFT 语料"""
        import glob

        files = sorted(glob.glob(path_pattern))
        if not files:
            raise FileNotFoundError(f"未找到匹配文件: {path_pattern}")

        print(f"找到 {len(files)} 个文件")

        all_dfs = []
        for f in files:
            file_size_mb = os.path.getsize(f) / 1024 / 1024
            ext = f.split('.')[-1].lower()

            if ext == 'parquet':
                df = pd.read_parquet(f, columns=columns)
            elif ext == 'jsonl':
                df = pd.read_json(f, lines=True, dtype={'instruction': 'string'})
            elif ext == 'csv':
                df = pd.read_csv(f, usecols=columns)
            else:
                continue

            df['_source_file'] = os.path.basename(f)
            all_dfs.append(df)
            print(f"  ✓ {os.path.basename(f)}: "
                  f"{file_size_mb:.1f} MB, {len(df):,} 行")

        combined = pd.concat(all_dfs, ignore_index=True)

        if sample_ratio < 1.0:
            n_sample = int(len(combined) * sample_ratio)
            combined = combined.sample(n_sample, random_state=42)

        total_mem = combined.memory_usage(deep=True).sum() / 1024 / 1024
        print(f"\n总计: {len(combined):,} 行 × {len(combined.columns)} 列")
        print(f"内存占用: {total_mem:.2f} MB")

        return combined


loader = EfficientLoader()
print("=== 高效加载策略 ===")
```
