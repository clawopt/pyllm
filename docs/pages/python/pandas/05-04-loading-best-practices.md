---
title: 大模型数据加载最佳实践
description: 千万元级数据集的低内存加载策略、PyArrow 后端加速 I/O 配置、LLMLoader 完整实现与性能基准
---
# 数据加载最佳实践

前三节我们分别介绍了 CSV/JSONL、数据库、Parquet/Feather/Excel 这些格式的读写方法。这一节要把它们串起来，回答一个更宏观的问题：**当面对千万级行的 LLM 语料时，怎么设计数据加载策略才能既不爆内存又能跑得快？**

核心原则只有一条：**不要把所有数据一次性加载到内存**。听起来简单，但在实际操作中涉及列裁剪、分块迭代、dtype 优化、格式选择等多个维度的配合。下面我们逐一拆解。

## 策略一：列裁剪——最省事的优化

这是投入产出比最高的优化手段，没有之一。

它的原理很简单：`read_csv()` 和 `read_parquet()` 都支持 `usecols` / `columns` 参数，让你在读取阶段就只提取需要的列——不需要的列永远不会被解析，不会占用任何内存。假设你的原始数据有 20 列但你的清洗脚本只用其中 5 列，那 15 列的数据在 I/O 层面就被完全跳过了。

```python
import pandas as pd
import numpy as np
import time

n = 10_000_000
np.random.seed(42)

full_df = pd.DataFrame({
    'id': range(n),
    'text': ['conversation data for memory testing'] * (n // 100) +
             [f'unique_{i}' for i in range(n % 100)],
    'response': ['response text'] * n,
    'quality': np.random.uniform(1, 5, n),
    'tokens': np.random.randint(20, 2000, n),
    'source': np.random.choice(['api', 'web', 'export'], n),
    'model': np.random.choice(['GPT-4o', 'Claude', 'Llama'], n),
    'user_id': [f'u_{i%50000:05d}' for i in range(n)],
    'session_id': [f'sess_{i%10000:05d}' for i in range(n)],
    'embedding': [np.random.randn(768).tolist() for _ in range(n)],
    'metadata': [{'key': f'v{i}'} for i in range(n)],
})

needed = ['id', 'text', 'quality', 'tokens', 'source']
df_select = full_df[needed].copy()

mem_full = full_df.memory_usage(deep=True).sum() / 1024**2
mem_select = df_select.memory_usage(deep=True).sum() / 1024**2

print(f"全量: {mem_full:.0f} MB")
print(f"只读5列: {mem_select:.0f} MB")
print(f"节省: {(1 - mem_select/mem_full)*100:.0f}%")
```

在这个例子中，`embedding` 列存的是 768 维的向量（每行一个列表），一千万行的话光这一列就要吃掉 **60+ GB 内存**（每个 Python 列表对象约 50 字节开销 + 768 个 float）。如果你不需要 embedding 列，通过列裁剪直接跳过它，内存从不可能装下变成轻松搞定。

实际使用中，我建议按用途定义好列组合模板：

```python
def load_for_sft(file_path):
    return pd.read_parquet(file_path, columns=[
        'conversation_id', 'prompt', 'response',
        'quality_score', 'token_count', 'source',
    ])

def load_for_eval(file_path):
    return pd.read_parquet(file_path, columns=[
        'sample_id', 'reference', 'model_outputs',
        'gpt4o_score', 'claude_score', 'llama_score',
    ])

def load_for_analysis(file_path):
    return pd.read_parquet(file_path, columns=[
        'user_id', 'session_id', 'tokens', 'quality_score',
        'source', 'model', 'created_at',
    ])
```

不同的下游任务需要不同的列子集，提前定义好模板可以避免每次都手动指定列名，同时确保不会意外加载不需要的大列（比如 embedding）。

## 策略二：分块迭代——内存不够时的解法

当文件太大以至于即使做了列裁剪也装不下的时候（比如你需要对所有列做变换），就需要分块读取了。Pandas 的 `read_csv(chunksize=N)` 返回一个迭代器，每次 yield 一个有 N 行的 DataFrame 子集。

关键是要理解：**分块处理不是简单地切小块，而是要根据你的任务目标选择正确的模式**。

### 模式 A：分块聚合——只需要统计结果

当你只需要最终的均值、计数、分布等聚合结果而不需要保留原始行时：

```python
import pandas as pd

CHUNK_SIZE = 500_000
total_rows = 0
quality_sum = 0.0
source_counts = pd.Series(dtype=int)

for chunk in pd.read_csv('huge_corpus.csv', chunksize=CHUNK_SIZE,
                          usecols=['quality_score', 'source']):
    total_rows += len(chunk)
    quality_sum += chunk['quality_score'].sum()
    source_counts = source_counts.add(chunk['source'].value_counts(), fill_value=0)
    del chunk

print(f"总行数: {total_rows:,}")
print(f"平均质量分: {quality_sum/total_rows:.2f}")
print(f"\n来源分布:\n{source_counts.sort_values(ascending=False)}")
```

这里用 `Series.add()` 而不是自己维护 dict 来累加各 chunk 的 value_counts——好处是 Pandas 会自动对齐索引，某个 chunk 里不存在的来源不会导致 KeyError 或 NaN。

### 模式 B：分块过滤——保留满足条件的完整行

当你需要筛选出符合条件的完整行并拼成一张大表时：

```python
import pandas as pd

filtered_chunks = []
MIN_QUALITY = 4.0

for chunk in pd.read_csv('corpus.csv', chunksize=200_000,
                          dtype={'quality_score': 'float32'}):
    good = chunk[chunk['quality_score'] >= MIN_QUALITY].copy()
    if len(good) > 0:
        filtered_chunks.append(good)
    del chunk

final = pd.concat(filtered_chunks, ignore_index=True)
print(f"筛选后: {len(final):,} 条, 平均分: {final['quality_score'].mean():.2f}")
```

注意 `.copy()` 的作用：不加 copy 的话 `good` 只是 `chunk` 的一个视图（view），`del chunk` 之后 `good` 指向的内存就被释放了，后续 `pd.concat()` 读到的是已释放的垃圾数据。这是一个非常容易踩坑且很难排查的问题——因为它不会立即报错，而是产生随机乱码的结果。

### 模式 C：分块 ETL——读 → 处理 → 写出新格式

这是生产环境中最常见的模式：从原始格式读取，做清洗和特征工程，然后写出为高效的 Parquet 格式：

```python
import pandas as pd

INPUT = 'raw_conversations.jsonl'
OUT_DIR = '/tmp/processed/'

for i, chunk in enumerate(pd.read_json(INPUT, lines=True, chunksize=250_000)):
    mask = (
        chunk['prompt'].notna() & (chunk['prompt'].str.len() > 5) &
        chunk['response'].notna() & (chunk['response'].str.len() > 10)
    )
    clean = chunk[mask].copy()
    
    if len(clean) == 0:
        continue
    
    clean['cleaned_prompt'] = (
        clean['prompt']
        .str.replace(r'<[^>]+>', '', regex=True)
        .str.strip()
    )
    clean['token_est'] = clean['cleaned_prompt'].str.len() // 4
    
    clean.to_parquet(f"{OUT_DIR}/chunk_{i:04d}.parquet", index=False)
    
    if (i + 1) % 10 == 0:
        print(f"已处理 {(i+1)*250_000:,} 条 → {OUT_DIR}/chunk_{i:04d}.parquet")
```

这种模式下每个 chunk 处理完就写出并释放，理论上可以处理任意大小的文件——只要磁盘空间够用。处理完所有 chunk 后如果需要合并成一个文件，可以用 `pd.concat([pd.read_parquet(f) for f in Path(OUT_DIR).glob('*.parquet')])` 再统一写回。

## 策略三：dtype 优化——在读取时就选对类型

我们在 04-04 节详细讨论过 dtype 对内存的影响。这里的关键点是：**dtype 优化不应该在读取之后再做，而应该在读取的时候就指定**。因为 `read_csv()` 默认会用 `int64` 和 `float64` 来存储数值列，这意味着你读完之后再去 `astype('int16')` 实际上经历了"用大类型读进内存 → 创建新的小类型数组 → 复制数据 → 释放旧数组"的过程，峰值内存是两份。如果在 `read_csv()` 时就直接指定 dtype，数据只会以小类型的形态存在，峰值内存减半。

```python
import pandas as pd
import numpy as np

dtype_spec = {
    'conversation_id': 'int32',
    'turn_count': 'int8',
    'round_number': 'int16',
    'token_count': 'int16',
    'quality_score': 'float32',
    'bleu_score': 'float16',
    'perplexity': 'float32',
    'source': 'category',
    'language': 'category',
    'difficulty': 'category',
    'prompt': 'string[pyarrow]',
    'response': 'string[pyarrow]',
    'created_at': 'datetime64[ns]',
}

df = pd.read_csv('data.csv', dtype=dtype_spec)
```

这个 dtype 映射表是我在多个 LLM 项目中总结出来的经验值：对话轮次通常不超过 127 所以 `int8` 够用，token 数通常不超过 32000 所以 `int16` 够用，质量评分不需要 double 精度所以 `float32` 够用，BLEU 分数在 0-1 范围内所以 `float16` 就够了。把这些固定下来作为项目的标准配置，每次读数据时直接套用。

## 策略四：PyArrow 后端——I/O 加速的最后一块拼图

Pandas 3.0 引入的 PyArrow 后端可以从根本上改变 I/O 的行为方式。设置方法很简单：

```python
import pandas as pd

pd.options.dtype_backend = 'pyarrow'
```

一旦设置了这行代码，之后所有新创建的 DataFrame 都会自动使用 PyArrow 类型。它带来的好处有三个：第一，天然支持 Nullable 语义（不会再遇到 int64 遇 None 变 float64 的问题）；第二，字符串列默认使用 Arrow 的 string 类型而非 Python 的 object 类型，内存效率高得多；第三，与 Parquet 文件的交互是零拷贝的——读 Parquet 时不需类型转换，写 Parquet 时也不需序列化。

```python
import pandas as pd
import numpy as np
import time, os

np.random.seed(42)
n = 5_000_000

df = pd.DataFrame({
    'text': ['LLM corpus benchmark data'] * n,
    'metadata': [f'meta_{i}' for i in range(n)],
    'value': np.random.randn(n),
    'category': np.random.choice(list('ABCDEFGHIJ'), n),
})

pd.options.dtype_backend = 'numpy_compat'

start = time.time()
df.to_csv('/tmp/bench.csv', index=False)
t_csv_w = time.time() - start
sz_csv = os.path.getsize('/tmp/bench.csv')

start = time.time()
df.to_parquet('/tmp/bench.pq', index=False)
t_pq_w = time.time() - start
sz_pq = os.path.getsize('/tmp/bench.pq')

start = time.time()
df_csv_back = pd.read_csv('/tmp/bench.csv')
t_csv_r = time.time() - start
mem_csv = df_csv_back.memory_usage(deep=True).sum()

pd.options.dtype_backend = 'pyarrow'

start = time.time()
df_pq_back = pd.read_parquet('/tmp/bench.pq')
t_pq_r = time.time() - start
mem_pq = df_pq_back.memory_usage(deep=True).sum()

print(f"{'操作':<12} {'CSV':>10} {'Parquet+Arrow':>16} {'优势':>8}")
print("-" * 52)
print(f"{'写入':<12} {t_csv_w:>9.2f}s {t_pq_w:>15.2f}s {t_csv_w/t_pq_w:>7.1f}x")
print(f"{'读取':<12} {t_csv_r:>9.2f}s {t_pq_r:>15.2f}s {t_csv_r/t_pq_r:>7.1f}x")
print(f"{'大小':<12} {sz_csv/1024/1024:>9.1f}MB {sz_pq/1024/1024:>15.1f}MB {(1-sz_pq/sz_csv)*100:>7.1f}%")
print(f"{'内存':<12} {mem_csv/1024/1024:>9.1f}MB {mem_pq/1024/1024:>15.1f}MB {(1-mem_pq/mem_csv)*100:>7.1f}%")
```

对于新项目来说，我的建议是在代码入口处（比如 `__main__.py` 或者 Jupyter notebook 的第一个 cell）就设置 `pd.options.dtype_backend = 'pyarrow'`，然后忘记传统 NumPy 类型的那些坑。这是目前 Pandas 生态中最现代化的方案。

## 决策流程总结

把以上四种策略整合起来，就是一份完整的数据加载决策流程：

```
收到数据文件
│
├─ 先判断文件大小
│   │
│   ├─ < 100 MB ──→ 全量读取，无需特殊优化
│   │                  但仍建议指定 usecols + dtype
│   │
│   ├─ 100 MB ~ 1 GB ─→ usecols + dtype 优化后全量读取
│   │                     如果原始格式低效，先转 Parquet
│   │
│   ├─ 1 GB ~ 10 GB ───→ chunksize 分块处理
│   │                      或转 Parquet 后利用列式裁剪
│   │
│   └─ > 10 GB ────────→ 考虑 Dask / Polars / DuckDB
│                         Pandas 单机可能不够用了
│
├─ 再判断是否需要全部列
│   ├─ 只需部分列 ──→ usecols / columns 裁剪
│   └─ 需要全部列 ──→ chunksize 分块 或 dtype 降级
│
└─ 最后选择最优格式
    ├─ 原始是 CSV/JSON ──→ 首次处理后转 Parquet 缓存
    ├─ 已是 Parquet ────→ 直接用 columns + filters 读取
    └─ 来自数据库 ──────→ SQL 尽量预聚合，减少拉取量
```

到这里，第五章的全部四个部分就结束了。我们覆盖了从最基本的 CSV 读写到高性能的 Parquet 列式存储，从本地文件到远程数据库，从全量加载到分块迭代的完整知识体系。这些 I/O 技能是你后续所有数据处理工作的基础——无论你是要做数据清洗、特征工程还是模型评估，第一步永远是"正确高效地把数据加载进来"。接下来的第六章和第七章，我们将进入数据处理的实质性环节：如何发现并修复数据中的质量问题。
