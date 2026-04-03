# 语法迁移：关键函数对照与改写


#### 核心语法映射表

| 操作 | Pandas | Polars | 注意事项 |
|------|--------|--------|----------|
| **创建** | `pd.DataFrame(data)` | `pl.DataFrame(data)` | Polars 自动推断类型更精确 |
| **读取 CSV** | `pd.read_csv(f)` | `pl.read_csv(f)` | Polars 默认并行读取 |
| **选择列** | `df[["a","b"]]` | `df.select("a","b")` | Polars 用 select |
| **过滤行** | `df[df.a > 0]` | `df.filter(pl.col("a") > 0)` | Polars 用 filter + pl.col |
| **新列** | `df["c"] = a + b` | `df.with_columns(c=...)` | Polars 不可变，返回新 DF |
| **聚合** | `df.groupby("x").agg(...)` | `df.group_by("x").agg(...)` | Polars group_by |
| **排序** | `df.sort_values("a")` | df.sort("a") | 类似 |
| **Join** | `df1.merge(df2, on="k")` | `df1.join(df2, on="k")` | 默认 inner join |
| **字符串** | `df.col.str.len()` | `df.col.str.lengths()` | 方法名不同 |
| **缺失值** | `df.fillna(0)` | `df.fill_null(0)` | fillna → fill_null |

---

#### 逐类迁移指南

##### 数据读取与创建

```python
import pandas as pd
import polars as pl

df_pd = pd.read_csv("large_data.csv",
                     dtype={"id": "string", "model": "category"},
                     parse_dates=["timestamp"])

df_pl = pl.read_csv("large_data.csv",
                    dtypes={"id": pl.Utf8, "model": pl.Categorical},
                    try_parse_dates=True)

data = {"name": ["Alice", "Bob"], "age": [25, 30]}

pd_df = pd.DataFrame(data)
pl_df = pl.DataFrame(data)

print(pd_df)
print(pl_df)
```

##### 列操作

```python
pandas:   result = df[["model", "score", "latency_ms"]]
polars:   result = df.select("model", "score", "latency_ms")

pandas:   result = df.drop(columns=["_source_file", "_line_number"])
polars:   result = df.drop("_source_file", "_line_number")

pandas:   result = df.rename(columns={"latency_ms": "latency"})
polars:   result = df.rename({"latency_ms": "latency"})

pandas:
    df["cost_per_token"] = df["cost_usd"] / df["tokens_used"]
    df["is_high_score"] = df["score"] > 0.8

polars:
    df = df.with_columns([
        (pl.col("cost_usd") / pl.col("tokens_used")).alias("cost_per_token"),
        (pl.col("score") > 0.8).alias("is_high_score"),
    ])
```

##### 过滤操作

```python
pandas:   filtered = df[df["score"] > 0.8]
polars:   filtered = df.filter(pl.col("score") > 0.8)

pandas:   filtered = df[(df["score"] > 0.8) & (df["model"] == "gpt-4")]
polars:   filtered = df.filter(
              (pl.col("score") > 0.8) & (pl.col("model") == "gpt-4")
          )

pandas:   filtered = df[df["model"].isin(["gpt-4", "claude-3.5-sonnet"])]
polars:   filtered = df.filter(
              pl.col("model").is_in(["gpt-4", "claude-3.5-sonnet"])
          )

pandas:
    filtered = df.loc[
        (df["category"] == "hard") &
        (df["task"].str.contains("reasoning|math", regex=True)) &
        (df["latency_ms"] < 2000),
        ["id", "model", "task", "score"]
    ]

polars:
    filtered = df.filter(
        (pl.col("category") == "hard") &
        (pl.col("task").str.contains("(reasoning|math)")) &
        (pl.col("latency_ms") < 2000)
    ).select("id", "model", "task", "score")
```

##### GroupBy 聚合

```python
pandas:
    result = df.groupby("model").agg(
        avg_score=("score", "mean"),
        median_lat=("latency_ms", "median"),
        total_tokens=("tokens_used", "sum"),
        count=("id", "count"),
    ).reset_index()

polars:
    result = df.group_by("model").agg(
        avg_score=pl.col("score").mean(),
        median_lat=pl.col("latency_ms").median(),
        total_tokens=pl.col("tokens_used").sum(),
        count=pl.len(),
    )

pandas:
    result = (df.groupby(["model", "task"], observed=True)
               .agg(avg_score=("score", "mean"))
               .reset_index()
               .sort_values(["task", "avg_score"],
                            ascending=[True, False]))

polars:
    result = (df.group_by(["model", "task"])
               .agg(avg_score=pl.col("score").mean())
               .sort(["task", "avg_score"],
                      descending=[False, True]))

pandas:
    grouped = df.groupby("model").agg(count=("id", "count"))
    result = grouped[grouped["count"] >= 100]

polars:
    result = (df.group_by("model")
               .agg(count=pl.len())
               .filter(pl.col("count") >= 100))
```

##### 字符串操作

```python
pandas:   df["text_len"] = df["instruction"].str.len()
polars:   df = df.with_columns(pl.col("instruction").str.lengths().alias("text_len"))

pandas:   filtered = df[df["instruction"].str.contains("Python", case=False)]
polars:   filtered = df.filter(pl.col("instruction").str.contains("(?i)Python"))

pandas:   df["extracted"] = df["instruction"].str.extract(r'第(\d+)条', expand=False)
polars:   df = df.with_columns(
              pl.col("instruction").str.extract(r"第(\d+)条", 1).alias("extracted")
          )

pandas:   df["upper_model"] = df["model"].str.upper()
polars:   df = df.with_columns(pl.col("model").str.to_uppercase().alias("upper_model"))

pandas:   df["clean"] = df["output"].str.strip()
polars:   df = df.with_columns(pl.col("output").str.strip_chars().alias("clean"))
```

##### Join / Merge

```python
pandas:   merged = df1.merge(df2, on="id", how="inner")
polars:   merged = df1.join(df2, on="id", how="inner")

pandas:   merged = df1.merge(df2,
                             left_on="user_id",
                             right_on="uid",
                             how="left")
polars:   merged = df1.join(df2,
                              left_on="user_id",
                              right_on="uid",
                              how="left")

pandas:   merged = df1.merge(df2, on=["model", "task"])
polars:   merged = df1.join(df2, on=["model", "task"])
```

---

#### 完整迁移示例

```python
def pandas_version(df: pd.DataFrame) -> pd.DataFrame:

    step1 = df[
        (df["score"] > 0.5) &
        (df["latency_ms"] < 2000)
    ].copy()

    step2 = step1.assign(
        text_len=step1["instruction"].str.len(),
        cost_efficiency=step1["score"] / (step1["cost_usd"] + 1e-10),
    )

    step3 = step2.groupby(["model", "task"], observed=True).agg(
        avg_score=("score", "mean"),
        median_latency=("latency_ms", "median"),
        total_cost=("cost_usd", "sum"),
        count=("id", "count"),
        avg_text_len=("text_len", "mean"),
    ).reset_index()

    step4 = step3[step3["count"] >= 100].copy()
    step4["efficiency_rank"] = step4.groupby("task")["avg_score"].rank(
        method="dense", ascending=False
    )

    result = step4.sort_values(
        ["task", "avg_score"], ascending=[True, False]
    )
    return result


def polars_version(df: pl.DataFrame) -> pl.DataFrame:

    step1 = df.filter(
        (pl.col("score") > 0.5) & (pl.col("latency_ms") < 2000)
    )

    step2 = step1.with_columns([
        pl.col("instruction").str.lengths().alias("text_len"),
        (pl.col("score") / (pl.col("cost_usd") + 1e-10)).alias("cost_efficiency"),
    ])

    step3 = step2.group_by(["model", "task"]).agg(
        avg_score=pl.col("score").mean(),
        median_latency=pl.col("latency_ms").median(),
        total_cost=pl.col("cost_usd").sum(),
        count=pl.len(),
        avg_text_len=pl.col("text_len").mean(),
    )

    step4 = step3.filter(pl.col("count") >= 100).with_columns(
        efficiency_rank=pl.col("avg_score").rank(
            method="dense", descending=True
        ).over("task")
    )

    result = step4.sort(["task", "avg_score"], descending=[False, True])
    return result


import time

t0 = time.perf_counter()
result_pd = pandas_version(df_pd)
pd_time = time.perf_counter() - t0

t0 = time.perf_counter()
result_pl = polars_version(df_pl)
pl_time = time.perf_counter() - t0

print(f"Pandas: {pd_time*1000:.1f}ms")
print(f"Polars: {pl_time*1000:.1f}ms")
print(f"加速比: {pd_time/pl_time:.2f}x")
```

输出：
```
Pandas: 2345.6ms
Polars: 287.3ms
加速比: 8.16x
```

同样的逻辑，**8 倍加速**。代码结构几乎一一对应，迁移成本很低。
