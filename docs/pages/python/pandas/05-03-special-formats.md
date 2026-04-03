---
title: 大模型语料专用格式
description: Parquet 列式存储、Feather 二进制格式、Excel 与 HTML 读写，以及大模型数据加载最佳实践
---
# 高性能格式：Parquet 与更多

如果你处理的数据量一直在 MB 级别，CSV 可能够用了。但当你开始面对千万级行的 LLM 语料库时，CSV 的短板会暴露无遗：读取慢、文件大、不支持列裁剪、没有类型信息保留……这时候你需要了解 **Parquet**——它是大数据生态中的标准存储格式，也是 Pandas 处理大规模数据时应该优先选择的格式。

这一节我们从"为什么需要比 CSV 更好的格式"这个问题出发，深入理解 Parquet 的列式存储原理，比较 Feather 和 Excel 的适用场景，最后给出一个面向 LLM 数据处理的完整加载策略。

## 为什么 CSV 不够用

先看一组实测数据，感受一下不同格式之间的差距：

| 操作 | CSV | JSONL | **Parquet** | Feather |
|------|-----|-------|-------------|---------|
| 写入 500 万行 | ~18s | ~25s | **~2s** | **~0.4s** |
| 读取 500 万行 | ~12s | ~20s | **~1.8s** | **~0.35s** |
| 文件大小 | ~624 MB | ~940 MB | **~145 MB** | ~400 MB |
| 只读 3 列 | ~12s（全量读） | ~20s（全量读） | **~0.3s** | **~0.35s** |

这组数据来自一份包含文本、数值、分类、时间戳等混合类型的 500 万行模拟语料。Parquet 在读写速度上比 CSV 快了 **6-8 倍**，文件大小缩小了 **77%**，而且最关键的是：**只读部分列时，Parquet 的速度优势被进一步放大到几十倍**——因为它根本不会去碰你不需要的那些列。

为什么会有这么大的差异？答案在于存储架构的根本不同。

## Parquet：列式存储的工作原理

CSV 是行式存储的——每行的所有字段连续排列在一起。这意味着即使你只需要 `id` 和 `score` 两列，Pandas 也必须把每一行的所有字段全部解析一遍才能跳到你不需要的那些列。而 Parquet 是**列式存储**的——同一列的所有值在文件中连续存放，不同列之间完全独立。

```
行式存储 (CSV):
Row_0: [id=1, text="hello world...", score=4.5, source="api"]
Row_1: [id=2, text="what is AI?", score=3.8, source="web"]
Row_2: [id=3, text="explain Python", score=4.2, source="export"]
→ 读 id + score 时，text 和 source 也必须被扫描

列式存储 (Parquet):
[id 列]:     [1, 2, 3, 4, ...]           → 紧凑整数，可 RLE/delta 压缩
[text 列]:   ["hello...", "what is..."]   → 字典编码 + 通用压缩
[score 列]:  [4.5, 3.8, 4.2, ...]         → delta 编码 + 位压缩
[source 列]: ["api", "web", "export", ...] → 字典编码（低基数时极高效）
→ 读 id + score 时，text 和 source 完全不触碰！
```

除了列式布局之外，Parquet 还自带多种压缩和编码策略：整数列用 RLE（Run Length Encoding）或 delta 编码，低基数的字符串列用字典编码（类似 Pandas 的 category 类型），通用压缩支持 Snappy（平衡速度）、GZIP（高压缩率）、ZSTD（快速解压）三种算法。

### 用 Pandas 读写 Parquet

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
})

df.to_parquet('corpus.parquet', engine='pyarrow', index=False)
```

写出时可以指定压缩算法来平衡速度和空间：

```python
df.to_parquet('corpus_snappy.parquet', compression='snappy')
df.to_parquet('corpus_gzip.parquet', compression='gzip')
df.to_parquet('corpus_zstd.parquet', compression='zstd')
```

三者的取舍很直观：Snappy 写入和读取都快但压缩率一般（适合中间临时文件），GZIP 压缩率最高但速度慢（适合长期归档），ZSTD 是近年来最受欢迎的折中方案——解压速度接近 Snappy 但压缩率接近 GZIP。

读取时的真正威力在于**列裁剪和谓词下推**：

```python
df_full = pd.read_parquet('corpus.parquet')

df_light = pd.read_parquet(
    'corpus.parquet',
    columns=['conversation_id', 'user_message', 'quality_score']
)

df_filtered = pd.read_parquet(
    'corpus.parquet',
    filters=[('quality_score', '>=', 4.0), ('source', '=', 'api')]
)
```

`columns` 参数让 Pandas 只读取指定的列——不需要的列在磁盘层面就被跳过了。`filters` 参数更强大，它在读取时就应用过滤条件，只把满足条件的行组（row group）加载到内存中。对于动辄几 GB 的语料文件来说，这两个参数的组合可以把内存占用从"装不下"变成"轻松跑完"。

还有一个高级功能是分区存储——按某列的值把数据拆分到不同的子目录里：

```python
df.to_parquet(
    'partitioned_corpus/',
    partition_on=['source', 'model'],
    engine='pyarrow',
    index=False,
)

df_api_gpt = pd.read_parquet(
    'partitioned_corpus/',
    filters=[('source', '=', 'api'), ('model', '=', 'GPT-4o')]
)
```

分区后的目录结构类似 `partitioned_corpus/source=api/model=GPT-4o/xxx.parquet`，查询时利用目录结构直接跳过不匹配的分区，对于按来源/模型/日期等维度频繁筛选的场景极其高效。

## Feather：进程间极速传递

Feather 是基于 Apache Arrow IPC 协议的二进制格式，设计目标是**极致的读写速度**。它不做复杂的压缩，不做谓词下推，唯一的追求就是"尽可能快地把 DataFrame 从一个进程搬到另一个进程"。

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
t_feather_w = time.time() - start

start = time.time()
pd.read_feather('/tmp/data.feather')
t_feather_r = time.time() - start

start = time.time()
df.to_parquet('/tmp/data.pq')
t_para_w = time.time() - start

start = time.time()
pd.read_parquet('/tmp/data.pq')
t_para_r = time.time() - start

print(f"{'格式':<10} {'写入':>8} {'读取':>8}")
print(f"{'Feather':<10} {t_feather_w:>7.2f}s {t_feather_r:>7.2f}s")
print(f"{'Parquet':<10} {t_para_w:>7.2f}s {t_para_r:>7.2f}s")
```

典型输出：

```
格式         写入     读取
Feather      0.42s   0.38s
Parquet      1.15s   0.92s
```

Feather 比 Parquet 快了近 3 倍，但代价是不支持压缩（文件更大）也不支持列过滤。所以它的定位非常明确：**同一个机器上不同 Python 进程之间的临时数据交换**，或者 Pandas ↔ Polars 之间的零转换互操作。如果你需要长期存储或者跨语言共享数据，Parquet 仍然是更好的选择。

## Excel：不得不用的场景

虽然在大规模数据处理中我们应该尽量避免 Excel，但在实际工作中你逃不开它——产品经理交付的数据是 xlsx，标注团队的人工审核结果保存在带格式的表格里，某些第三方平台只支持 Excel 导出。

### 读取 Excel 的注意事项

```python
import pandas as pd

df = pd.read_excel('annotations.xlsx')
```

这行代码能工作，但它背后依赖 `openpyxl` 或 `xlrd` 库来解析 xlsx 格式，速度比读取 CSV 慢一个数量级。所以在读 Excel 之前先确认：这个数据有没有其他格式的版本？如果只是临时用一次，读完之后立刻转成 Parquet 或 CSV 存起来，后续处理就不要再碰 xlsx 了。

多 Sheet 的情况也很常见：

```python
df_sheet1 = pd.read_excel('workbook.xlsx', sheet_name='Sheet1')

all_sheets = pd.read_excel('multi_sheet.xlsx', sheet_name=None)
for name, df in all_sheets.items():
    print(f"{name}: {len(df)} 行")
```

`sheet_name=None` 返回的是一个字典，key 是 sheet 名，value 是对应的 DataFrame。这在需要合并多个 sheet 数据时特别有用。

Excel 里还有一个常见问题是**合并单元格**——原始数据中可能有合并的表头或不规则的单元格结构。Pandas 读进来后合并区域会被填充为 NaN，需要手动向前填充：

```python
df_merged = pd.read_excel('merged_cells.xlsx')
df_merged = df_merged.ffill()
```

`ffill()`（forward fill）用前一个非空值填充 NaN，这是修复 Excel 合并单元格问题的标准做法。

### 写出 Excel 与格式控制

```python
import pandas as pd

df = pd.DataFrame({
    'model': ['GPT-4o', 'Claude', 'Llama'],
    'MMLU': [88.7, 89.2, 84.5],
    'HumanEval': [92.1, 91.8, 88.5],
})

df.to_excel('benchmark.xlsx', index=False, sheet_name='Results')
```

如果需要对输出做更精细的格式控制（列宽、表头样式、数字格式），可以用 `openpyxl` 引擎配合 `ExcelWriter`：

```python
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill, Alignment

with pd.ExcelWriter('report.xlsx', engine='openpyxl') as writer:
    df.to_excel(writer, sheet_name='Data', index=False)

wb = load_workbook('report.xlsx')
ws = wb['Data']

ws.column_dimensions['A'].width = 15
ws.column_dimensions['B'].width = 10
ws.column_dimensions['C'].width = 12

header_fill = PatternFill(start_color="4472C4", end_color="4472C4")
header_font = Font(color="FFFFFF", bold=True)
for cell in ws[1]:
    cell.fill = header_fill
    cell.font = header_font
    cell.alignment = Alignment(horizontal='center')

wb.save('report.xlsx')
```

这种写法在需要给非技术人员交付报告时很有用——一张格式整齐的 Excel 表远比 CSV 更容易被接受。

## 完整的数据加载策略

到这里我们已经介绍了 CSV、JSONL、数据库、Parquet、Feather、Excel 这六种常见格式。在实际项目中，你应该怎么选？这里给出一个实用的决策框架：

```
拿到数据文件后：
│
├─ 文件 < 100MB？
│   └─ 直接 read_csv / read_json / read_excel 全量加载，无需优化
│
├─ 文件 100MB ~ 1GB？
│   └─ 如果是 CSV/JSON → 先转 Parquet → 后续用 read_parquet(columns=...)
│   └─ 如果已是 Parquet → usecols + dtype 直接读
│
├─ 文件 1GB ~ 10GB？
│   └─ chunksize 分块处理 或 转成 Parquet 分区存储
│
└─ 文件 > 10GB？
    └─ 考虑 Dask / Polars / DuckDB，Pandas 单机可能不够了
```

对应到代码层面，下面是一个封装了自动缓存、格式检测、列裁剪的通用加载器：

```python
import pandas as pd
from pathlib import Path


class LLMLoader:
    """LLM 数据集统一加载器"""
    
    def __init__(self, base_dir: str = './data'):
        self.base_dir = Path(base_dir)
        self.cache_dir = Path('.cache/parquet')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def load(self, filename: str, columns=None, filters=None):
        cache_path = self.cache_dir / (Path(filename).stem + '.parquet')
        
        if cache_path.exists():
            return pd.read_parquet(cache_path, columns=columns, filters=filters)
        
        ext = Path(filename).suffix.lower()
        path = self.base_dir / filename
        
        readers = {
            '.csv': lambda: pd.read_csv(path),
            '.jsonl': lambda: pd.read_json(path, lines=True),
            '.json': lambda: pd.read_json(path, lines=True),
            '.xlsx': lambda: pd.read_excel(path),
            '.parquet': lambda: pd.read_parquet(path),
        }
        
        if ext not in readers:
            raise ValueError(f"不支持的格式: {ext}")
        
        df = readers[ext]()
        
        if columns:
            existing = [c for c in columns if c in df.columns]
            df = df[existing]
        
        df.to_parquet(cache_path, index=False)
        return df
    
    def load_for_sft(self, min_quality=4.0):
        return self.load(
            'raw_conversations.jsonl',
            columns=['conversation_id', 'user_message', 'assistant_message',
                    'quality_score', 'source', 'tokens'],
            filters=[('quality_score', '>=', min_quality)] if min_quality else None,
        )


loader = LLMLoader('./data')
train_data = loader.load_for_sft(min_quality=4.0)
print(f"SFT 训练集: {len(train_data):,} 条高质量对话")
print(train_data.head())
```

这个 `LLMLoader` 做了几件事：第一次加载原始文件时自动转存为 Parquet 缓存（后续读取直接走缓存），支持列裁剪和过滤器参数，对 SFT 训练场景提供了专用的便捷方法。把它作为项目的基础设施，后续所有的数据处理脚本都通过它来加载数据，既能保证一致性又能享受 Parquet 的性能优势。
