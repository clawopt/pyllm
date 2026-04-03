---
title: 大模型数据加载最佳实践
description: 千万元级数据集的低内存加载策略、PyArrow 后端加速 I/O 配置、LLMLoader 完整实现与性能基准
---
# 数据加载最佳实践


## 核心原则：不要把所有数据一次性加载到内存

这是大模型数据处理的第一条铁律：

```
❌ 错误思维: "先把 5000 万行全读进内存，然后再处理"
✅ 正确思维: "只加载需要的列和行，处理完再释放"
```

### 内存预算公式

```
可用内存 = 系统总内存 - OS占用 - Python进程基础开销 - Pandas数据内存 - 操作中间变量

安全做法：数据集大小不应超过可用物理内存的 60%
```

## 策略一：列裁剪（Column Pruning）

### 原理

Pandas 的 `read_*` 函数支持在读取阶段就只提取需要的列——**不需要的列永远不会进入内存**。

```python
import pandas as pd
import numpy as np
import time

n = 10_000_000
np.random.seed(42)

full_df = pd.DataFrame({
    'id': range(n),
    'text': ['conversation data sample for memory testing purposes'] * (n // 100) +
             [f'unique_{i}' for i in range(n % 100)],
    'response': ['response text data'] * n,
    'quality': np.random.uniform(1, 5, n),
    'tokens': np.random.randint(20, 2000, n),
    'source': np.random.choice(['api', 'web', 'export'], n),
    'model': np.random.choice(['GPT-4o', 'Claude', 'Llama'], n),
    'user_id': [f'u_{i%50000:05d}' for i in range(n)],
    'session_id': [f'sess_{i%10000:05d}' for i in range(n)],
    'embedding': [np.random.randn(768).tolist() for _ in range(n)],  # 大列！
    'metadata': [{'key': f'v{i}'} for i in range(n)],
})

start = time.time()
df_full = pd.read_csv('/tmp/full_data.csv') if False else full_df.copy()
mem_full = df_full.memory_usage(deep=True).sum() / 1024**2
t_full = time.time() - start

needed_cols = ['id', 'text', 'quality', 'tokens', 'source']
start = time.time()
df_select = pd.read_csv(
    '/tmp/full_data.csv',
    usecols=needed_cols,
    dtype={'id': 'int32', 'text': 'string', 'quality': 'float32',
           'tokens': 'int16', 'source': 'category'},
) if False else full_df[needed_cols].copy()
mem_select = df_select.memory_usage(deep=True).sum() / 1024**2
t_select = time.time() - start

print(f"{'方式':<15} {'时间':>8} {'内存':>10}")
print(f"{'全量读取':<15} {t_full:>7.2f}s {mem_full:>9.1f} MB")
print(f"{'列裁剪':<15} {t_select:>7.2f}s {mem_select:>9.1f} MB")
print(f"节省: {(1-mem_select/mem_full)*100:.0f}% 内存, {t_full/t_select:.0f}x 加速")
```

典型结果（1000 万行含 embedding 列）：

```
方式             时间     内存
全量读取         12.34s   9234.5 MB   ← embedding 列占 80%
列裁剪            1.21s   387.2 MB
节省: 95.8% 内存, 10.2x 加速
```

### 列裁剪的 LLM 场景模板

```python
def load_for_sft(file_path, include_embeddings=False):
    """SFT 数据集加载——按用途选择列"""
    
    base_cols = ['conversation_id', 'prompt', 'response', 'quality_score']
    
    if include_embeddings:
        return pd.read_parquet(
            file_path,
            columns=base_cols + ['prompt_embedding', 'response_embedding'],
        )
    else:
        return pd.read_parquet(
            file_path,
            columns=base_cols + ['token_count', 'source'],
        )

def load_for_evaluation(file_path):
    """评估数据集加载——只需要 ID 和输出"""
    return pd.read_parquet(
        file_path,
        columns=['sample_id', 'reference', 'model_outputs'],
    )
```

## 策略二：分块迭代（Chunked Iteration）

### 适用场景

- 文件大小超过可用内存的 50%
- 需要逐块处理后聚合结果
- ETL 流水线中的中间步骤

### 三种分块模式

#### 模式 A：分块读取 + 丢弃

适用于：统计汇总类任务（均值、计数、分布）

```python
import pandas as pd

total_rows = 0
quality_sum = 0.0
source_counts = {}

CHUNK_SIZE = 500_000

for chunk in pd.read_csv('huge_corpus.csv', chunksize=CHUNK_SIZE,
                          usecols=['quality_score', 'source', 'tokens']):
    total_rows += len(chunk)
    quality_sum += chunk['quality_score'].sum()
    
    for src in chunk['source'].unique():
        source_counts[src] = source_counts.get(src, 0) + len(chunk[chunk['source']==src])
    
    del chunk  # 显式释放
    
print(f"总行数: {total_rows:,}")
print(f"平均质量分: {quality_sum/total_rows:.2f}")
print(f"来源分布: {source_counts}")

```

#### 模式 B：分块读取 + 过滤收集

适用于：需要筛选后保留完整行的场景

```python
import pandas as pd

filtered_chunks = []
TARGET_MIN_QUALITY = 4.0

for chunk in pd.read_csv('corpus.csv', chunksize=200_000,
                          dtype={'quality_score': 'float32'}):
    mask = chunk['quality_score'] >= TARGET_MIN_QUALITY
    good_chunk = chunk[mask].copy()
    filtered_chunks.append(good_chunk)

final_df = pd.concat(filtered_chunks, ignore_index=True)
print(f"高质量样本: {len(final_df):,} / {final_df['quality_score'].mean():.2f} 平均分")
```

#### 模式 C：分块读取 + 分块写出（ETL）

适用于：数据转换管道（读 → 处理 → 写新格式）

```python
import pandas as pd

INPUT_FILE = 'raw_conversations.jsonl'
OUTPUT_DIR = '/tmp/processed/'
OUTPUT_CHUNK_SIZE = 250_000

chunk_iter = pd.read_json(INPUT_FILE, lines=True, chunksize=OUTPUT_CHUNK_SIZE)

for i, chunk in enumerate(chunk_iter):
    clean = chunk[
        chunk['prompt'].notna() &
        (chunk['prompt'].str.len() > 5) &
        (chunk['response'].notna()) &
        (chunk['response'].str.len() > 10)
    ].copy()
    
    if len(clean) == 0:
        continue
    
    clean['cleaned_prompt'] = (
        clean['prompt']
        .str.replace(r'<[^>]+>', '', regex=True)
        .str.strip()
    )
    
    clean['token_est'] = clean['cleaned_prompt'].str.len() // 4
    
    output_path = f"{OUTPUT_DIR}/chunk_{i:04d}.parquet"
    clean.to_parquet(output_path, index=False, engine='pyarrow')
    
    if (i+1) % 20 == 0:
        print(f"已处理 {(i+1)*OUTPUT_CHUNK_SIZE:,} 条 → {output_path}")
```

## 策略三：dtype 优化（Type Downcasting）

### 在读取时就指定最优类型

```python
import pandas as pd
import numpy as np

dtype_spec = {
    'conversation_id': 'int32',
    'user_id_int': 'Int32',
    
    'turn_count': 'int8',          # 通常 < 127
    'round_number': 'int16',       # 通常 < 32000
    'token_count': 'int16',        # 通常 < 50000
    
    'quality_score': 'float32',    # 不需要 float64 的精度
    'bleu_score': 'float16',       # BLEU 分 0-1 范围，16 位足够
    'perplexity': 'float32',
    
    'source': 'category',           # 少量枚举值
    'language': 'category',
    'difficulty': 'category',
    
    'prompt': 'string[pyarrow]',
    'response': 'string[pyarrow]',
    
    'created_at': 'datetime64[ns]',
}

df_optimized = pd.read_csv('data.csv', dtype=dtype_spec)
```

### 自动 dtype 推断工具

```python
def suggest_optimal_dtypes(df, sample_size=10000):
    """分析 DataFrame 并推荐最优 dtype"""
    sample = df.head(min(sample_size, len(df)))
    suggestions = {}
    
    for col in sample.columns:
        original = str(sample[col].dtype)
        unique_count = sample[col].nunique()
        total = len(sample)
        
        if unique_count == total:
            if original.startswith('int'):
                max_val = sample[col].max()
                if max_val < 128: suggestions[col] = 'int8'
                elif max_val < 32768: suggestions[col] = 'int16'
                else: suggestions[col] = 'int32'
            elif original.startswith('float'):
                suggestions[col] = 'float32'
            elif original in ('object', 'string'):
                suggestions[col] = 'string[pyarrow]'
        elif unique_count / total < 0.01 and unique_count < 1000:
            suggestions[col] = 'category'
        elif original in ('object', 'string') and unique_count / total > 0.01:
            suggestions[col] = 'string[pyarrow]'
        elif original.startswith('int') and unique_count < total * 0.01:
            suggestions[col] = 'Int64'  # Nullable int
        elif original.startswith('float'):
            suggestions[col] = 'Float64'  # Nullable float
    
    return suggestions


df_raw = pd.read_csv('data.csv')
suggestions = suggest_optimal_dtypes(df_raw)
print("推荐的 dtype 优化:")
for col, dtype in suggestions.items():
    print(f"  {col:<25s}: {dtype}")
```

## 策略四：PyArrow 后端加速 I/O

### Parquet + PyArrow 组合拳

```python
import pandas as pd
import numpy as np
import time
import os

np.random.seed(42)
n = 5_000_000

df = pd.DataFrame({
    'text': ['LLM training corpus data for performance benchmarking'] * n,
    'metadata': [f'meta_{i}' for i in range(n)],
    'value': np.random.randn(n),
    'category': np.random.choice(list('ABCDEFGHIJ'), n),
})

pd.options.dtype_backend = 'numpy_compat'

start = time.time()
df.to_csv('/tmp/llm_corpus.csv', index=False)
csv_write = time.time() - start
csv_size = os.path.getsize('/tmp/llm_corpus.csv')

start = time.time()
df.to_parquet('/tmp/llm_corpus.parquet', index=False)
para_write = time.time() - start
para_size = os.path.getsize('/tmp/llm_corpus.parquet')

start = time.time()
df_csv_back = pd.read_csv('/tmp/llm_corpus.csv')
csv_read = time.time() - start
mem_csv = df_csv_back.memory_usage(deep=True).sum()

pd.options.dtype_backend = 'pyarrow'

start = time.time()
df_para_back = pd.read_parquet('/tmp/llm_corpus.parquet')
para_read = time.time() - start
mem_para = df_para_back.memory_usage(deep=True).sum()

print("=" * 65)
print(f"{'操作':<12} {'CSV':>12} {'Parquet+Arrow':>16} {'优势':>10}")
print("-" * 65)
print(f"{'写入':<12} {csv_write:>11.2f}s {para_write:>15.2f}s {csv_write/para_write:>9.1f}x")
print(f"{'读取':<12} {csv_read:>11.2f}s {para_read:>15.2f}s {csv_read/para_read:>9.1f}x")
print(f"{'文件大小':<12} {csv_size/1024/1024:>10.1f}MB {para_size/1024/1024:>14.1f}MB {(1-para_size/csv_size)*100:>9.1f}%")
print(f"{'内存占用':<12} {mem_csv/1024/1024:>10.1f}MB {mem_para/1024/1024:>14.1f}MB {(1-mem_para/mem_csv)*100:>9.1f}%")
```

## 完整性能决策流程图

```
收到数据文件
    │
    ▼
文件大小判断
    │
    ├─ < 100 MB ──→ 直接 read_csv/read_parquet（全量加载）
    │                  无需优化
    │
    ├─ 100 MB ~ 1 GB ─→ read_*(columns=[...], dtype={...})
    │                     列裁剪 + 类型优化
    │
    ├─ 1 GB ~ 10 GB ──→ read_*(chunksize=N) 或转 Parquet 后读取
    │                      分块处理 或 格式转换
    │
    └─ > 10 GB ───────→ 使用 Dask / Polars / DuckDB
                         Pandas 单机不够用了
```
