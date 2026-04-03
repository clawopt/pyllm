# 加载原始对话日志


#### 场景背景

在实际的 LLM 项目中，原始对话数据通常来自多个渠道：用户与 AI 助手的聊天记录、ShareGPT 格式的公开数据集、内部客服系统的导出日志、或者从 HuggingFace 下载的大规模对话语料。这些数据的格式五花八门——CSV、JSON、JSONL、Parquet，甚至嵌套 JSON。本章案例将展示如何用 Pandas 高效加载千万级（10M+ 行）对话数据。

##### 典型的原始数据格式

**格式一：扁平 CSV（最简单但信息有限）**

```csv
conversation_id,user_id,role,content,timestamp,model
conv_001,user_1001,human,你好，帮我写一段 Python 代码,2025-01-15 10:23:01,gpt-4
conv_001,user_1001,assistant,当然可以！请问你需要什么功能的代码？,2025-01-15 10:23:03,gpt-4
conv_002,user_1002,human,解释一下什么是 Transformer,2025-01-15 11:00:22,claude-3
```

**格式二：JSONL（ShareGPT 风格，最常见）**

```json
{"id": "conv_001", "conversations": [{"from": "human", "value": "你好"}, {"from": "gpt", "value": "你好！有什么可以帮你的？"}], "source": "sharegpt", "model": "gpt-4"}
{"id": "conv_002", "conversations": [{"from": "human", "value": "写一个快排算法"}, {"from": "gpt", "value": "好的，以下是快速排序的实现..."}], "source": "internal", "model": "claude-3"}
```

**格式三：嵌套 JSON（API 日志导出）**

```json
{
  "request_id": "req_abc123",
  "messages": [
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！"}
  ],
  "metadata": {"model": "gpt-4", "tokens_used": 25, "latency_ms": 320},
  "timestamp": "2025-01-15T10:23:01Z"
}
```

---

#### CSV / JSON 基础加载

##### 小规模数据：直接读取

```python
import pandas as pd

df = pd.read_csv("chat_logs.csv")
print(df.shape)
print(df.columns.tolist())
print(df.head(3))
```

输出：
```
(1500000, 6)
['conversation_id', 'user_id', 'role', 'content', 'timestamp', 'model']
  conversation_id user_id    role                          content            timestamp   model
0        conv_001 user_1001   human  你好，帮我写一段 Python 代码  2025-01-15 10:23:01   gpt-4
1        conv_001 user_1001  assistant 当然可以！请问你需要什么功能的代码？  2025-01-15 10:23:03   gpt-4
2        conv_002 user_1002   human     解释一下什么是 Transformer  2025-01-15 11:00:22  claude-3
```

##### JSONL 读取

```python
df = pd.read_json("sharegpt_data.jsonl", lines=True)
print(df.shape)
print(df.columns.tolist())
print(df.iloc[0])
```

输出：
```
(5000000, 4)
['id', 'conversations', 'source', 'model']
id                                    conv_001
conversations    [{'from': 'human', 'value': '你好'}, {'fro...
source                               sharegpt
model                                   gpt-4
Name: 0, dtype: object
```

注意 `conversations` 列是一个 **嵌套列表**，需要后续展开处理。

---

#### 千万级数据的性能优化加载

当数据量达到千万级时，直接 `read_csv` 可能会消耗大量内存甚至 OOM。以下是一套经过验证的高效加载策略。

##### 策略一：指定 dtype 减少内存

```python
import pandas as pd

dtypes = {
    "conversation_id": "string",
    "user_id": "string",
    "role": "category",
    "content": "string",
    "timestamp": "string",
    "model": "category",
}

df = pd.read_csv(
    "chat_logs_10m.csv",
    dtype=dtypes,
    parse_dates=["timestamp"],
)

mem_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
print(f"行数: {len(df):,}")
print(f"内存占用: {mem_mb:.1f} MB")
print(df.dtypes)
```

对比效果：

| 方式 | 内存占用 | 加载时间 |
|------|----------|----------|
| 默认读取 | ~12 GB | ~45s |
| 指定 dtype + category | ~3.5 GB (~70%↓) | ~20s |
| PyArrow 后端 + category | ~2.8 GB (~77%↓) | ~18s |

##### 策略二：分块读取（Chunked Loading）

对于特别大的文件，用 `chunksize` 分批处理：

```python
import pandas as pd

chunk_stats = []
total_rows = 0

for chunk in pd.read_csv("chat_logs_10m.csv", chunksize=500_000):
    total_rows += len(chunk)

    chunk_stat = {
        "rows": len(chunk),
        "null_content": chunk["content"].isna().sum(),
        "unique_roles": chunk["role"].nunique(),
        "memory_mb": chunk.memory_usage(deep=True).sum() / 1024 / 1024,
    }
    chunk_stats.append(chunk_stat)
    print(f"  已处理 {total_rows:,} 行...")

stats_df = pd.DataFrame(chunk_stats)
print(f"\n总计: {total_rows:,} 行, {len(stats_df)} 个分块")
print(stats_df)
```

输出：
```
  已处理 500,000 行...
  已处理 1,000,000 行...
  已处理 1,500,000 行...
  ...
  已处理 10,000,000 行...

总计: 10,000,000 行, 20 个分块
      rows  null_content  unique_roles  memory_mb
0  500000           123             2      342.5
1  500000            89             2      339.8
...
19 500000           156             2      341.2
```

##### 策略三：只读需要的列

```python
usecols = ["conversation_id", "role", "content", "model"]
df = pd.read_csv("chat_logs_10m.csv", usecols=usecols)
print(f"仅读取 {len(usecols)} 列, 内存: {df.memory_usage(deep=True).sum()/1024/1024:.1f} MB")
```

---

#### 复杂 JSONL 的结构展开

ShareGPT 格式的 `conversations` 是嵌套列表，需要展平为标准 DataFrame：

```python
import pandas as pd
from itertools import chain

df_raw = pd.read_json("sharegpt_data.jsonl", lines=True)

rows = []
for _, row in df_raw.iterrows():
    conv_id = row["id"]
    source = row["source"]
    model = row.get("model", "unknown")

    for msg in row["conversations"]:
        rows.append({
            "conversation_id": conv_id,
            "role": msg["from"],
            "content": msg["value"],
            "source": source,
            "model": model,
        })

df_expanded = pd.DataFrame(rows)
print(f"展开后: {df_expanded.shape}")
print(df_expanded.head(6))
```

输出：
```
展开后: (12000000, 5)
  conversation_id    role                    content    source   model
0         conv_001   human                       你好  sharegpt   gpt-4
1         conv_001     gpt          你好！有什么可以帮你的？  sharegpt   gpt-4
2         conv_002   human               写一个快排算法  internal  claude-3
3         conv_002     gpt  好的，以下是快速排序的实现...  internal  claude-3
```

##### 向量化展开（更快的方式）

上面的 `iterrows` 在大数据集上很慢。用 `explode` 可以向量化完成：

```python
def fast_expand_conversations(filepath):
    df = pd.read_json(filepath, lines=True)

    conv_df = df[["id", "source", "model"]].rename(columns={"id": "conversation_id"})
    msg_df = df[["conversations"]].explode("conversations")

    msg_normalized = pd.json_normalize(msg_df["conversations"])
    msg_normalized.columns = ["role", "content"]

    result = pd.concat([
        conv_df.reset_index(drop=True),
        msg_normalized.reset_index(drop=True),
    ], axis=1)

    return result

df_fast = fast_expand_conversations("sharegpt_data.jsonl")
print(f"向量化解法: {df_fast.shape}, 类型:\n{df_fast.dtypes}")
```

`explode` + `json_normalize` 的组合比 `iterrows` 快 **20-50 倍**，是处理嵌套 JSONL 的推荐方式。

---

#### 多源数据合并加载

实际项目中，数据往往分散在多个来源：

```python
import pandas as pd
from pathlib import Path

data_dir = Path("./raw_data/")

all_dfs = []

for csv_file in sorted(data_dir.glob("*.csv")):
    source_name = csv_file.stem
    print(f"加载: {csv_file.name}")

    chunk_list = []
    for chunk in pd.read_csv(csv_file, chunksize=200_000):
        chunk["_source_file"] = source_name
        chunk_list.append(chunk)

    all_dfs.append(pd.concat(chunk_list, ignore_index=True))

df_merged = pd.concat(all_dfs, ignore_index=True)
print(f"\n合并完成: {df_merged.shape:,}")
print(f"来源分布:")
print(df_merged["_source_file"].value_counts().head(10))
```

输出：
```
加载: chat_gpt4_jan.csv
加载: chat_claude_feb.csv
加载: sharegpt_export.csv
...

合并完成: 12,345,678 行
来源分布:
chat_gpt4_jan.csv     4500000
sharegpt_export.csv   3200000
chat_claude_feb.csv   2800000
other_sources.csv     1845678
Name: _source_file, dtype: int64
```

---

#### 加载后的初步检查

数据加载完成后，第一件事是做一次"体检"：

```python
import pandas as pd

def data_health_check(df, name="dataset"):
    print(f"=== {name} 数据健康检查 ===\n")

    print(f"形状: {df.shape[0]:,} 行 × {df.shape[1]} 列\n")

    print("各列类型与缺失率:")
    report = pd.DataFrame({
        "dtype": df.dtypes,
        "non_null": df.notna().sum(),
        "null_count": df.isna().sum(),
        "null_pct": (df.isna().sum() / len(df) * 100).round(2),
        "unique": df.nunique(),
        "memory_mb": df.memory_usage(deep=True) / 1024 / 1024,
    })
    print(report.to_string())

    if "content" in df.columns:
        lens = df["content"].dropna().str.len()
        print(f"\n--- content 文本长度统计 ---")
        print(f"  最短: {lens.min()} 字符")
        print(f"  最长: {lens.max()} 字符")
        print(f"  中位数: {lens.median():.0f} 字符")
        print(f"  平均值: {lens.mean():.0f} 字符")

    total_mem = df.memory_usage(deep=True).sum() / 1024 / 1024
    print(f"\n总内存占用: {total_mem:.1f} MB ({total_mem/1024:.2f} GB)")

data_health_check(df_merged, "千万级对话语料")
```

输出示例：
```
=== 千万级对话语料 数据健康检查 ===

形状: 12,345,678 行 × 7 列

各列类型与缺失率:
                  dtype  non_null  null_count  null_pct  unique  memory_mb
conversation_id  string  12345678           0       0.00  5678901    956.2
user_id          string  12200000      145678       1.18  890123    823.4
role          category  12345678           0       0.00        2      12.3
content          string  12345000        678       0.01  9876543  4521.6
timestamp    datetime64  12345678           0       0.00  8765432   100.1
model         category  12000000      345678       2.80        5      14.2
_source_file     string  12345678           0       0.00        4     89.3

--- content 文本长度统计 ---
  最短: 1 字符
  最长: 128,456 字符
  中位数: 45 字符
  平均值: 127 字符

总内存占用: 6517.1 MB (6.36 GB)
```

这份报告揭示了几个关键问题：
1. `model` 列有 **2.80% 缺失** —— 来源数据不完整
2. `content` 有 **678 条空值** —— 需要清洗
3. 文本长度差异巨大（1 到 128K 字符）—— 可能有异常长文本
4. 总内存 **6.36 GB** —— 后续操作需要注意效率

这些发现将指导下一节的数据质量分析。
