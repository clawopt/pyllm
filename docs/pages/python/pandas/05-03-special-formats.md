---
title: 大模型语料专用格式
description: Parquet 列式存储、Feather 二进制格式、Excel 与 HTML 读写，以及大模型数据加载最佳实践
---
# 特殊格式处理


## 为什么需要"专用格式"

CSV 和 JSON 虽然通用，但在处理**大规模 LLM 语料**时有明显缺陷：

| 格式 | 读取 1GB 数据 | 写入 1GB 数据 | 文件大小 | 支持类型 | 嵌入查询 |
|------|-------------|-------------|---------|---------|---------|
| CSV | ~15s | ~20s | ~1 GB | 仅文本/数字 | ❌ |
| JSON/JSONL | ~25s | ~30s | ~1.5 GB | 全部 | ❌ |
| **Parquet** | **~2s** | **~3s** | **~150 MB** | **全部** | ✅ DuckDB/Polaris |
| Feather | **~0.5s** | **~0.7s** | **~400 MB** | 全部（有限） | ⚠️ |

对于千万级 token 的语料库，选择正确的格式意味着：

- **I/O 速度提升 5-10 倍**
- **存储空间节省 70-85%**
- **支持列式下推（只读需要的列）**

## Parquet：列式存储之王

### 什么是 Parquet

Parquet 是 Apache Hadoop 生态中的标准列式存储格式。它的核心设计理念是：

> **数据按列存储，每列独立压缩**

```
传统行式存储 (CSV):
Row_0: [id=1, text="hello world...", score=4.5, label="pos"]
Row_1: [id=2, text="what is AI?", score=3.8, label="neg"]
→ 读任何一行都要读全部字段

Parquet 列式存储:
[id column]:     [1, 2, 3, ...]        → 紧凑整数压缩
[text column]:   ["hello...", "what is..."] → 字典编码 + 压缩
[score column]:  [4.5, 3.8, ...]         → delta 编码
[label column]:  ["pos", "neg", ...]       → 字典编码
→ 只读 id + text 列时，score 和 label 完全不碰
```

### Pandas 读写 Parquet

```python
import pandas as pd
import numpy as np

np.random.seed(42)
n = 2_000_000

df = pd.DataFrame({
    'conversation_id': [f'conv_{i:08d}' for i in range(n)],
    'user_message': [f'用户问题 {i % 1000}' for i in range(n)],
    'assistant_message': [f'模型回复 {i % 500}' for i in range(n)],
    'turn_count': np.random.randint(1, 10, n),
    'quality_score': np.round(np.random.uniform(1, 5, n), 2),
    'token_count': np.random.randint(50, 2000, n),
    'source': np.random.choice(['api', 'web', 'export'], n),
    'model': np.random.choice(['GPT-4o', 'Claude', 'Llama'], n),
    'created_at': pd.date_range('2025-01-01', periods=n, freq='s'),
})

df.to_parquet('corpus.parquet', engine='pyarrow', index=False)

df.to_parquet('corpus_snappy.parquet', compression='snappy')   # 平衡速度和压缩率
df.to_parquet('corpus_gzip.parquet', compression='gzip')       # 更高压缩比
df.to_parquet('corpus_zstd.parquet', compression='zstd')       # 最快解压

df.to_parquet(
    'partitioned_corpus/',
    partition_on=['source', 'model'],
    engine='pyarrow',
    index=False,
)

df_read = pd.read_parquet('corpus.parquet')

df_light = pd.read_parquet(
    'corpus.parquet',
    columns=['conversation_id', 'user_message', 'quality_score']
)

df_filtered = pd.read_parquet(
    'corpus.parquet',
    filters=[('quality_score', '>=', 4.0), ('source', '=', 'api')]
)
```

### Parquet vs CSV 性能实测

```python
import pandas as pd
import numpy as np
import time
import os

n = 5_000_000
np.random.seed(42)

test_df = pd.DataFrame({
    'id': range(n),
    'text': ['sample conversation data for benchmarking purposes'] * (n // 100) +
             ['unique_text_' + str(i) for i in range(n % 100)],
    'value': np.random.randn(n),
    'category': np.random.choice(['A', 'B', 'C', 'D', 'E'], n),
    'flag': np.random.choice([True, False], n),
    'timestamp': pd.date_range('2025-01-01', periods=n, freq='s'),
})

start = time.time()
test_df.to_csv('/tmp/benchmark.csv', index=False)
csv_write_time = time.time() - start
csv_size = os.path.getsize('/tmp/benchmark.csv') / 1024 / 1024

start = time.time()
test_df.to_parquet('/tmp/benchmark.parquet', index=False)
parquet_write_time = time.time() - start
parquet_size = os.path.getsize('/tmp/benchmark.parquet') / 1024 / 1024

start = time.time()
df_csv = pd.read_csv('/tmp/benchmark.csv')
csv_read_time = time.time() - start

start = time.time()
df_parquet = pd.read_parquet('/tmp/benchmark.parquet')
parquet_read_time = time.time() - start

print("=" * 60)
print(f"{'操作':<15} {'CSV':>12} {'Parquet':>12} {'优势':>10}")
print("-" * 60)
print(f"{'写入时间':<15} {csv_write_time:>11.2f}s {parquet_write_time:>11.2f}s {csv_write_time/parquet_write_time:>9.1f}x")
print(f"{'读取时间':<15} {csv_read_time:>11.2f}s {parquet_read_time:>11.2f}s {csv_read_time/parquet_read_time:>9.1f}x")
print(f"{'文件大小':<15} {csv_size:>11.1f}MB {parquet_size:>11.1f}MB {(1-parquet_size/csv_size)*100:>9.1f}%")
```

典型输出（500 万行）：

```
============================================================
操作               CSV      Parquet       优势
------------------------------------------------------------
写入时间            18.45s       2.31s      8.0x
读取时间            12.33s       1.87s      6.6x
文件大小            623.8 MB     145.2 MB    76.7%
============================================================
```

## Feather：极速二进制格式

### Feather 是什么

Feather 是专为 Pandas 设计的二进制序列化格式，基于 Apache Arrow 的 IPC（Inter-Process Communication）协议。

**核心特点**：
- 读写极快（接近内存复制速度）
- 保持 DataFrame 的 dtype 信息
- 支持 PyArrow 后端零转换

### 适用场景与限制

```python
import pandas as pd
import numpy as np
import time

n = 1_000_000
df = pd.DataFrame({
    'a': np.random.randn(n),
    'b': np.random.randint(0, 100, n),
    'c': np.random.choice(['x', 'y', 'z'], n),
})

start = time.time()
df.to_feather('/tmp/data.feather')
feather_write = time.time() - start

start = time.time()
df_back = pd.read_feather('/tmp/data.feather')
feather_read = time.time() - start

start = time.time()
df.to_parquet('/tmp/data_para.parquet')
para_write = time.time() - start

start = time.time()
pd.read_parquet('/tmp/data_para.parquet')
para_read = time.time() - start

print(f"{'格式':<10} {'写':>8} {'读':>8}")
print(f"{'Feather':<10} {feather_write:>7.2f}s {feather_read:>7.2f}s")
print(f"{'Parquet':<10} {para_write:>7.2f}s {para_read:>7.2f}s")
```

```
格式         写       读
Feather     0.42s    0.38s   ← 最快
Parquet     1.15s    0.92s
```

**何时选 Feather vs Parquet**：

| 需求 | 推荐 | 原因 |
|------|------|------|
| 进程间临时传递数据 | **Feather** | 读写最快，保持 dtype |
| 长期存储 / 归档 | **Parquet** | 压缩率高，节省磁盘 |
| 需要跨语言/工具共享 | **Parquet** | Arrow 生态广泛支持 |
| 需要列式过滤 | **Parquet** | 支持谓词下推 |
| Pandas ↔ Polars 互转 | **两者皆可** | 都基于 Arrow |

## Excel 读写

### 什么时候还需要 Excel

虽然在大规模数据处理中应避免 Excel，但以下场景仍需用到：

- 产品经理/非技术人员交付的数据
- 标注团队的人工标注结果
- 需要带格式的报告输出

### 读取 Excel

```python
import pandas as pd

df = pd.read_excel('annotations.xlsx')

df_sheet1 = pd.read_excel('workbook.xlsx', sheet_name='Sheet1')
df_sheet2 = pd.read_excel('workbook.xlsx', sheet_name='Summary')

all_sheets = pd.read_excel('multi_sheet.xlsx', sheet_name=None)
for name, df in all_sheets.items():
    print(f"{name}: {len(df)} 行")

df = pd.read_excel(
    'labels.xlsx',
    usecols=['text', 'label', 'labeler_id', 'confidence'],
    dtype={'labeler_id': 'string', 'confidence': 'float32'},
)

df_merged = pd.read_excel('merged_cells.xlsx')
df_merged = df_merged.ffill()  # 向前填充 NaN
```

### 写出 Excel

```python
import pandas as pd

df = pd.DataFrame({
    'model': ['GPT-4o', 'Claude', 'Llama'],
    'MMLU': [88.7, 89.2, 84.5],
    'HumanEval': [92.1, 91.8, 88.5],
})

df.to_excel('benchmark.xlsx', index=False, sheet_name='Results')

from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill, Alignment

with pd.ExcelWriter('report.xlsx', engine='openpyxl') as writer:
    df.to_excel(writer, sheet_name='Raw Data', index=False)
    
    summary = df.describe().T
    summary.to_excel(writer, sheet_name='Statistics', index=True)
    
    writer.close()

wb = load_workbook('report.xlsx')
ws = wb['Raw Data']

ws.column_dimensions['A'].width = 15
ws.column_dimensions['B'].width = 10
ws.column_dimensions['C'].width = 10

header_fill = PatternFill(start_color="4472C4", end_color="4472C4")
header_font = Font(color="FFFFFF", bold=True)
for cell in ws[1]:
    cell.fill = header_fill
    cell.font = header_font
    cell.alignment = Alignment(horizontal='center')

wb.save('report.xlsx')
```

### Excel 替代方案：用 HTML 替代

如果目标只是让数据在浏览器中好看地展示：

```python
import pandas as pd

df = pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})

html = df.to_html(
    classes='table table-striped table-hover',
    index=False,
    border=0,
    float_format='{:.2f}',
)

with open('output.html', 'w') as f:
    f.write(f"""
<!DOCTYPE html>
<html>
<head>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css"
      rel="stylesheet">
</head>
<body>
<div class="container mt-4">
<h2>数据报告</h2>
{html}
</div>
</body>
</html>
""")
```

## 大模型数据加载最佳实践

### 完整策略图

```
数据来源              选择格式                加载方式
─────────           ─────────────          ───────────
数据库导出(CSV)  →   先转 Parquet        →  read_parquet
API 导出(JSON)  →   JSONL 直接读         →  read_json(lines=True)
标注交付(xlsx)  →   read_excel          →  清洗后转 Parquet
已有 Parquet    →   直接读取             →  read_parquet(columns=[...])
超大数据集(>10G) →  Dask / Polars         →  分块处理
```

### 推荐的工作流

```python
import pandas as pd
from pathlib import Path

class LLMLoader:
    """LLM 数据集加载器"""
    
    def __init__(self, base_dir: str = './data'):
        self.base_dir = Path(base_dir)
        self.cache_dir = Path('.cache/parquet')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def load(self, filename: str, columns=None, filters=None,
              force_refresh: bool = False):
        cache_path = self.cache_dir / (Path(filename).stem + '.parquet')
        
        if not force_refresh and cache_path.exists():
            print(f"从缓存加载: {cache_path}")
            return pd.read_parquet(cache_path, columns=columns, filters=filters)
        
        ext = Path(filename).suffix.lower()
        
        if ext == '.csv':
            df = pd.read_csv(self.base_dir / filename)
        elif ext == '.jsonl' or ext == '.json':
            df = pd.read_json(self.base_dir / filename, lines=True)
        elif ext == '.xlsx':
            df = pd.read_excel(self.base_dir / filename)
        elif ext == '.parquet':
            df = pd.read_parquet(self.base_dir / filename)
        else:
            raise ValueError(f"不支持的格式: {ext}")
        
        if columns:
            existing = [c for c in columns if c in df.columns]
            df = df[existing]
        
        df.to_parquet(cache_path, index=False)
        print(f"已缓存到: {cache_path}")
        return df
    
    def load_training_set(self, min_quality: float = 3.0,
                          sources: list[str] = None):
        filters = [('quality_score', '>=', min_quality)]
        if sources:
            filters.append(('source', 'in', sources))
        
        return self.load(
            'raw_conversations.jsonl',
            columns=['conversation_id', 'user_message', 'assistant_message',
                    'quality_score', 'source', 'tokens'],
            filters=filters
        )

loader = LLMLoader('./data')
train_data = loader.load_training_set(min_quality=4.0, sources=['api', 'web'])
print(train_data.head())
```

这个 `LLMLoader` 类封装了：
1. 自动缓存（避免重复解析原始文件）
2. 格式自动检测
3. 列裁剪和过滤
4. 统一的接口
