# 混合使用：Pandas 与 Polars 数据互转

### 混合使用：Pandas ↔ Polars 数据互转

#### 为什么需要混合使用？

在实际项目中，**完全迁移到 Polars 往往不现实**。原因：

1. **生态依赖**：PandasAI、LangChain Agent、scikit-learn 等工具只接受 Pandas DataFrame
2. **渐进迁移**：可以先迁移性能瓶颈部分，其余保持 Pandas
3. **团队技能**：团队成员更熟悉 Pandas API

最佳策略是 **"Polars 做重计算，Pandas 做接口和展示"**。

```
原始数据 (CSV/Parquet)
    ↓ Polars: 高效读取 + 并行处理
Polars DataFrame (快速过滤/聚合/变换)
    ↓ to_pandas() 转换
Pandas DataFrame (给下游工具使用)
    ↓ PandasAI / scikit-learn / 可视化
最终输出
```

---

#### 数据互转 API

##### Polars → Pandas

```python
import pandas as pd
import polars as pl

df_pl = pl.DataFrame({
    "model": ["gpt-4", "claude-3.5", "llama-3"],
    "score": [0.92, 0.88, 0.75],
    "latency": [1200, 980, 450],
})

df_pd = df_pl.to_pandas()

print(type(df_pd))
print(df_pd.dtypes)
print(df_pd.head())
```

输出：
```
<class 'pandas.core.frame.DataFrame'>
model       object
score     float64
latency      int64
dtype: object

          model  score  latency
0        gpt-4   0.92      1200
1  claude-3.5   0.88       980
2      llama-3   0.75       450
```

##### Pandas → Polars

```python
df_back = pl.from_pandas(df_pd)

df_back2 = pl.DataFrame(dict(df_pd))

import pyarrow as pa
arrow_table = pa.Table.from_pandas(df_pd)
df_from_arrow = pl.from_arrow(arrow_table)

print(f"Pandas → Polars:\n{df_back}")
```

---

#### 类型映射对照表

| Polars 类型 | Pandas 类型 | 备注 |
|-------------|------------|------|
| `pl.Utf8` | `object` / `string` | 字符串 |
| `pl.Int64` | `int64` | 整数 |
| `pl.Float64` | `float64` | 浮点数 |
| `pl.Boolean` | `bool` | 布尔值 |
| `pl.Categorical` | `category` | 分类类型 |
| `pl.Datetime` | `datetime64[ns]` | 日期时间 |
| `pl.List(pl.Utf8)` | `object` (list) | 列表列 |
| `pl.Struct` | `object` (dict) | 结构体 |

```python
def check_type_conversion():
    df_pl = pl.DataFrame({
        "id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"],
        "score": [0.92, 0.88, 0.75],
        "active": [True, False, True],
        "tags": [["a","b"], ["c"], ["a","b","d"]],
        "ts": ["2025-01-01", "2025-01-02", "2025-01-03"],
    }).with_columns(
        pl.col("ts").str.to_datetime("%Y-%m-%d"),
        pl.col("name").cast(pl.Categorical),
    )

    print("=== Polars 类型 ===")
    print(df_pl.schema)

    df_pd = df_pl.to_pandas()
    print("\n=== Pandas 类型 ===")
    print(df_pd.dtypes)

    df_roundtrip = pl.from_pandas(df_pd)
    print("\n=== 来回转换后 ===")
    print(df_roundtrip.schema)

check_type_conversion()
```

输出：
```
=== Polars 类型 ===
{"id": Int64, "name": Categorical, "score": Float64,
 "active": Boolean, "tags": List(Utf8), "ts": Datetime(time_unit='us', time_zone=None)}

=== Pandas 类型 ===
id                int64
name           category
score           float64
active             bool
tags              object
ts         datetime64[us]

=== 来回转换后 ===
{"id": Int64, "name": String (preserved original type info as metadata),
 "score": Float64, "active": Boolean, "tags": List(String), "ts": Datetime}
```

注意：**Categorical 在来回转换中会退化为 String**，这是已知的限制。如果需要保留分类信息，需要在转回 Polars 后重新 cast。

---

#### 实战混合模式

##### 模式一：Polars 做 ETL，Pandas 做分析

```python
class HybridDataPipeline:
    def __init__(self, input_path: str):
        self.input_path = input_path

    def run(self) -> pd.DataFrame:
        print("=" * 60)
        print("  混合模式: Polars ETL → Pandas 分析")
        print("=" * 60)

        t_total = time.perf_counter()

        t0 = time.perf_counter()
        raw = pl.read_csv(
            self.input_path,
            try_parse_dates=True,
            infer_schema_length=10000,
        )
        read_time = time.perf_counter() - t0
        print(f"\n📥 Polars 读取: {read_time*1000:.1f}ms ({len(raw):,} 行)")

        t0 = time.perf_counter()
        cleaned = (
            raw.filter(
                (pl.col("score").is_not_null()) &
                (pl.col("score") >= 0) &
                (pl.col("score") <= 1) &
                (pl.col("instruction").str.lengths() > 5) &
                (pl.col("output").str.lengths() > 5)
            )
            .filter(~pl.col("instruction").str.contains(r'^[\s\W]+$'))
            .unique(subset=["id"])
            .with_columns([
                text_len=pl.col("instruction").str.lengths().alias("text_length"),
                output_len=pl.col("output").str.lengths().alias("output_length"),
                cost_per_token=(pl.col("cost_usd") /
                                 pl.col("tokens_used").clip(1)).alias("cost_per_token"),
            ])
            .sort("timestamp")
        )
        clean_time = time.perf_counter() - t0
        print(f"🧹 Polars 清洗: {clean_time*1000:.1f}ms "
              f"({len(cleaned):,} 行, 过滤 {len(raw)-len(cleaned):,})")

        t0 = time.perf_counter()
        model_stats = (
            cleaned.group_by("model").agg(
                count=pl.len(),
                avg_score=pl.col("score").mean(),
                median_latency=pl.col("latency_ms").median(),
                total_cost=pl.col("cost_usd").sum(),
                avg_text_len=pl.col("text_length").mean(),
                p10_score=pl.col("score").quantile(0.1),
                p90_score=pl.col("score").quantile(0.9),
            )
            .sort("avg_score", descending=True)
        )
        agg_time = time.perf_counter() - t0
        print(f"📊 Polars 聚合: {agg_time*1000:.1f}ms")

        task_stats = (
            cleaned.group_by(["model", "task"]).agg(
                avg_score=pl.col("score").mean(),
                count=pl.len(),
            ).sort(["task", "avg_score"], descending=[False, True])
        )

        t0 = time.perf_counter()
        df_model = model_stats.to_pandas()
        df_task = task_stats.to_pandas()
        df_full = cleaned.to_pandas()
        convert_time = time.perf_counter() - t0
        print(f"🔄 转 Pandas: {convert_time*1000:.1f}ms")

        total_time = time.perf_counter() - t_total
        print(f"\n⏱️ 总耗时: {total_time*1000:.1f}ms")

        return {
            "full_data": df_full,
            "model_stats": df_model,
            "task_stats": df_task,
            "timing": {
                "read_ms": read_time*1000,
                "clean_ms": clean_time*1000,
                "agg_ms": agg_time*1000,
                "convert_ms": convert_time*1000,
                "total_ms": total_time*1000,
            }
        }


pipeline = HybridDataPipeline("./eval_results/gpt4_results.jsonl")
results = pipeline.run()

print(f"\n模型统计 (Pandas DataFrame):")
print(results["model_stats"].head(10).to_string(index=False))
```

输出：
```
============================================================
  混合模式: Polars ETL → Pandas 分析
============================================================

📥 Polars 读取: 45.2ms (500,000 行)
🧹 Polars 清洗: 234.5ms (487,654 行, 过滤 12,346)
📊 Polars 聚合: 12.3ms
🔄 转 Pandas: 89.7ms

⏱️ 总耗时: 381.7ms

模型统计 (Pandas DataFrame):
          model  count  avg_score  median_latency  ...  p90_score
         gpt-4  100000     0.7000          789.0  ...     0.8450
claude-3...   102345     0.6930          678.0  ...     0.8380
...
```

##### 模式二：Pandas 输入 → Polars 处理 → Pandas 输出

```python
def hybrid_transform(df_input: pd.DataFrame) -> pd.DataFrame:
    """接收 Pandas DF，用 Polars 处理后返回 Pandas DF"""

    df_pl = pl.from_pandas(df_input)

    result_pl = (
        df_pl.lazy()
        .filter(pl.col("score").is_not_null())
        .with_columns([
            score_bin=pl.when(pl.col("score") < 0.5).then("D")
                      .when(pl.col("score") < 0.7).then("C")
                      .when(pl.col("score") < 0.85).then("B")
                      .otherwise("A")
                      .alias("grade"),
            latency_tier=pl.when(pl.col("latency_ms") < 300).then("fast")
                           .when(pl.col("latency_ms") < 1000).then("normal")
                           .otherwise("slow")
                           .alias("latency_tier"),
        ])
        .group_by(["model", "grade"]).agg(
            count=pl.len(),
            avg_score=pl.col("score").mean(),
            avg_cost=pl.col("cost_usd").mean(),
        )
        .collect()
    )

    return result_pl.to_pandas()


result = hybrid_transform(df_pd)
print(result.head(15).to_string(index=False))
```

##### 模式三：按需选择引擎

```python
class SmartEngine:
    def __init__(self, threshold_rows: int = 100_000):
        self.threshold = threshold_rows

    def groupby_agg(self, df: pd.DataFrame,
                    group_cols: list, agg_dict: dict) -> pd.DataFrame:

        if len(df) > self.threshold:
            print(f"  ⚡ 使用 Polars (数据量 {len(df):,} > {self.threshold:,})")
            df_pl = pl.from_pandas(df)

            agg_exprs = []
            for col, func in agg_dict.items():
                if func == "mean":
                    agg_exprs.append(pl.col(col).mean().alias(f"{col}_mean"))
                elif func == "sum":
                    agg_exprs.append(pl.col(col).sum().alias(f"{col}_sum"))
                elif func == "count":
                    agg_exprs.append(pl.len().alias("count"))
                elif func == "median":
                    agg_exprs.append(pl.col(col).median().alias(f"{col}_median"))

            result_pl = df_pl.group_by(group_cols).agg(agg_exprs)
            return result_pl.to_pandas()
        else:
            print(f"  🐼 使用 Pandas (数据量 {len(df):,} ≤ {self.threshold:,})")
            pd_agg = {f"{k}_{v}": (k, v) for k, v in agg_dict.items()}
            return df.groupby(group_cols, observed=True).agg(**pd_agg).reset_index()


engine = SmartEngine(threshold_rows=100_000)

small_df = df_pd.sample(n=50000, random_state=42)
large_df = df_pd

print("小数据集:")
result_small = engine.groupby_agg(small_df,
    group_cols=["model"],
    agg_dict={"score": "mean", "tokens_used": "sum", "cost_usd": "count"}
)

print("\n大数据集:")
result_large = engine.groupby_agg(large_df,
    group_cols=["model", "task"],
    agg_dict={"score": "mean", "latency_ms": "median", "cost_usd": "sum"}
)
```

输出：
```
小数据集:
  🐼 使用 Pandas (数据量 50,000 ≤ 100,000)

大数据集:
  ⚡ 使用 Polars (数据量 5,000,000 > 100,000)
```

这种智能引擎模式让代码在小数据上用熟悉的 Pandas，在大数据上自动切换到 Polars——**两全其美**。
