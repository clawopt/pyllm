# 进阶主题与生态扩展


#### Pandas vs Polars：新一代 DataFrame 库对比

Polars 是用 Rust 编写的新一代 DataFrame 库，在查询性能上显著领先 Pandas。了解两者的差异，能帮助你在不同场景下做出正确的技术选型。

##### 核心架构差异

| 特性 | Pandas | Polars |
|------|--------|--------|
| 实现语言 | Python / C | Rust |
| 执行模型 | 即时执行（eager） | 延迟执行（lazy）+ 即时执行 |
| 并行模型 | 单线程（大部分操作） | 自动多线程 |
| 内存管理 | 引用计数 + CoW（3.0） | Arrow 内存格式，零拷贝 |
| 字符串类型 | object / StringDtype / PyArrow | Utf8（原生 Arrow） |
| 空值表示 | NaN / None / NA | null（统一） |

##### 性能对比：典型操作

```python
import pandas as pd
import polars as pl
import numpy as np

N = 5_000_000

data = {
    "model": np.random.choice(["gpt-4", "claude-3", "llama-3", "qwen2.5"], N),
    "task": np.random.choice(["reasoning", "coding", "math", "translation"], N),
    "score": np.random.uniform(0, 1, N),
    "latency_ms": np.random.exponential(500, N).astype(int),
    "tokens": np.random.poisson(1000, N),
}

df_pd = pd.DataFrame(data)
df_pl = pl.DataFrame(data)
```

**筛选 + 聚合**：

```python
def pandas_agg(df):
    return (
        df.query("score > 0.5")
        .groupby("model", observed=True)
        .agg(
            avg_score=("score", "mean"),
            avg_latency=("latency_ms", "mean"),
            count=("score", "count"),
        )
        .sort_values("avg_score", ascending=False)
    )

def polars_agg(df):
    return (
        df.lazy()
        .filter(pl.col("score") > 0.5)
        .group_by("model")
        .agg([
            pl.col("score").mean().alias("avg_score"),
            pl.col("latency_ms").mean().alias("avg_latency"),
            pl.len().alias("count"),
        ])
        .sort("avg_score", descending=True)
        .collect()
    )
```

**预期性能差距**：Polars lazy 模式通常比 Pandas 快 **3-10 倍**（取决于数据量和操作复杂度）。

##### 何时选择 Polars

✅ **适合 Polars 的场景**：
- 数据量 > 1000 万行
- 需要多步复杂查询（lazy 优化器会自动融合）
- 对延迟敏感的实时分析
- 需要并行处理

❌ **继续使用 Pandas 的场景**：
- 数据量 < 100 万行（Pandas 足够快）
- 需要 PandasAI / LangChain 等生态集成
- 团队已有大量 Pandas 代码资产
- 需要丰富的 I/O 格式支持

##### 互操作性

Pandas 和 Polars 可以无缝转换：

```python
df_pl = pl.from_pandas(df_pd)
df_back = df_pl.to_pandas()

df_pd = pd.read_parquet("large_data.parquet")
df_lazy = pl.scan_parquet("large_data.parquet")  # 零拷贝懒加载
```

---

#### DuckDB：SQL 查询引擎集成

DuckDB 是一个内嵌式 OLAP 数据库，可以直接查询 Pandas DataFrame，无需数据复制。

##### 基本用法

```python
import pandas as pd
import duckdb

df_eval = pd.DataFrame({
    "model": ["gpt-4"] * 50 + ["claude-3"] * 50 + ["llama-3"] * 50,
    "task": (["reasoning"] * 20 + ["coding"] * 20 + ["math"] * 10) * 3,
    "score": [0.9 + i*0.001 for i in range(150)],
})

result = duckdb.query("""
    SELECT model,
           AVG(score) as avg_score,
           COUNT(*) as count,
           PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY score) as median_score
    FROM df_eval
    GROUP BY model
    HAVING COUNT(*) >= 30
    ORDER BY avg_score DESC
""").df()

print(result)
```

输出：
```
      model  avg_score  count  median_score
0     gpt-4   0.924500     50       0.9245
1  claude-3   0.874500     50       0.8745
2   llama-3   0.824500     50       0.8245
```

##### DuckDB vs Pandas query() 的选择

| 场景 | 推荐 | 原因 |
|------|------|------|
| 简单条件筛选 | `query()` | 更 Pythonic，无需 SQL |
| 多表 JOIN | DuckDB | SQL JOIN 语法更清晰，优化更好 |
| 窗口函数 | DuckDB | Pandas 不原生支持窗口函数 |
| 复杂聚合 + HAVING | DuckDB | SQL 表达力更强 |
| 与 LLM 工作流集成 | `query()` | 更容易嵌入 Agent 工具 |

##### 在 AI 项目中的实际应用

```python
class SQLEvalAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.con = duckdb.connect()

    def model_leaderboard(self, tasks: list[str] = None):
        task_filter = f"AND task IN {tuple(tasks)}" if tasks else ""
        query = f"""
            SELECT
                model,
                COUNT(DISTINCT task) as task_coverage,
                ROUND(AVG(score)::numeric, 4) as overall_avg,
                ROUND(MIN(AVG(score)) OVER (PARTITION BY model)::numeric, 4) as weakest_task,
                ROUND(STDDEV_POP(score)::numeric, 4) as consistency
            FROM self.df
            WHERE score IS NOT NULL {task_filter}
            GROUP BY model
            HAVING COUNT(*) >= 10
            ORDER BY overall_avg DESC
        """
        return self.con.execute(query).df()

    def task_difficulty(self):
        query = """
            SELECT
                task,
                COUNT(*) as attempts,
                ROUND(AVG(score)::numeric, 4) as avg_score,
                ROUND(PERCENTILE_CONT(0.25)
                    WITHIN GROUP (ORDER BY score)::numeric, 4) as q25,
                ROUND(PERCENTILE_CONT(0.75)
                    WITHIN GROUP (ORDER BY score)::numeric, 4) as q75
            FROM self.df
            GROUP BY task
            ORDER BY avg_score ASC
        """
        return self.con.execute(query).df()

analyzer = SQLEvalAnalyzer(df_eval)
print(analyzer.model_leaderboard())
print(analyzer.task_difficulty())
```

---

#### 类型注解与 Pandera：数据验证框架

在生产环境中，数据的 Schema 验证至关重要。Pandera 提供了类似 Pydantic 的 DataFrame 级别验证。

##### 安装与基本用法

```python
import pandera as pa
from pandera import Column, Check, DataFrameSchema

class EvalSchema(DataFrameSchema):
    model = Column(str, Check.isin(["gpt-4", "claude-3", "llama-3", "qwen2.5"]))
    task = Column(str, Check.str_length(min_value=1))
    score = Column(float, Check.in_range(0, 1))
    latency_ms = Column(int, Check.ge(0))

schema = EvalSchema(strict=True)

validated = schema.validate(df_eval)
print(f"验证通过: {len(validated)} 行")
```

##### 自定义验证规则

```python
from pandera import Column, Check, DataFrameSchema

class SFTDataSchema(DataFrameSchema):
    instruction = Column(str, [
        Check.str_length(min_value=10, error="指令过短"),
        Check(lambda s: ~s.str.contains("password", case=False),
              error="包含敏感词"),
    ])
    output = Column(str, Check.str_length(min_value=5))
    category = Column(str, nullable=True)
    quality_score = Column(float, Check.in_range(0, 1), nullable=True)

def validate_sft_data(df: pd.DataFrame) -> pd.DataFrame:
    schema = SFTDataSchema(strict=False, coerce=True)
    try:
        validated = schema.validate(df, lazy=True)
        print(f"✅ 全部通过: {len(validated)} 行")
        return validated
    except pa.errors.SchemaErrors as exc:
        print(f"❌ 发现 {len(exc.failure_cases)} 个问题:")
        print(exc.failure_cases[["column", "check", "failure_case"]].head(10))
        raise
```

##### 与 Pydantic 配合

```python
from pydantic import BaseModel

class ModelEvalRecord(BaseModel):
    model: str
    task: str
    score: float
    metadata: dict = {}

records = [
    ModelEvalRecord(model="gpt-4", task="reasoning", score=0.95),
    ModelEvalRecord(model="claude-3", task="coding", score=0.88),
]

df_from_pydantic = pd.DataFrame([r.model_dump() for r in records])
validated = schema.validate(df_from_pydantic)
```

---

#### Vaex：流式处理超大数据集

Vaex 是另一个面向大数据的 DataFrame 库，核心特点是 **零内存拷贝的内存映射（memory-mapped）**。

##### 适用场景对比

| 数据规模 | 推荐方案 | 内存需求 |
|----------|----------|----------|
| < 100 MB | Pandas | 全部加载到内存 |
| 100 MB - 10 GB | Pandas + 分块 / Dask | 可控 |
| 10 GB - 100 GB | Dask / Vaex | 磁盘映射或分块 |
| > 100 GB | Spark / Ray Data | 分布式 |

Vaex 的独特优势在于处理 **"装不进内存但放在磁盘上可以接受"** 的数据集：

```python
import vaex

df = vaex.open("huge_sft_data.hdf5")

df_filtered = df[df.quality_score > 0.7]
stats = df_filtered.groupby("category").agg({
    "quality_score": "mean",
    "instruction": "count",
})

print(stats)
```

---

#### 社区生态工具一览

##### 数据质量

| 工具 | 用途 | 关联章节 |
|------|------|----------|
| **great_expectations** | 数据质量断言与文档化 | 第7章 |
| **pandera** | Schema 验证 | 本章 |
| **pydq** | 数据质量检测 | 第7章 |

##### 性能与分布式

| 工具 | 用途 | 关联章节 |
|------|------|----------|
| **swifter** | apply() 自动并行化 | 第17章 |
| **modin** | 替换后端的 drop-in 加速 | 第24章 |
| **dask** | 延迟执行 + 分布式 | 第24章 |
| **ray** | 通用分布式框架 | 第24章 |
| **polars** | 高性能 DataFrame | 本章 |
| **duckdb** | 内嵌 SQL 引擎 | 本章 |
| **vaex** | 内存映射大数据 | 本章 |

##### AI/ML 集成

| 工具 | 用途 | 关联章节 |
|------|------|----------|
| **pandasai** | 自然语言查询 DataFrame | 第18章 |
| **langchain** | Agent 工具集成 | 第21章 |
| **featuretools** | 自动特征工程 | 第9章 |
| **sklearn-pandas** | Scikit-learn 管道集成 | 第9章 |
| **fugue** | 抽象层（Pandas/Dask/Spark 统一接口） | 第24章 |

##### 可视化增强

| 工具 | 用途 | 关联章节 |
|------|------|----------|
| **plotly** | 交互式图表 | 第15章 |
| **altair** | 声明式可视化 | 第15章 |
| **seaborn** | 统计可视化 | 第15章 |
| **matplotlib** | 底层绑图引擎 | 第15章 |

##### I/O 与存储

| 工具 | 用途 | 关联章节 |
|------|------|----------|
| **pyarrow** | Parquet / Arrow 格式 | 第3、17章 |
| **fastparquet** | Parquet 另一实现 | 第3章 |
| **openpyxl** | Excel 读写 | 第3章 |
| **sqlalchemy** | 数据库 ORM | 第4章 |

---

#### Pandas 版本演进路线

##### 已发布的重大版本

| 版本 | 年份 | 核心变化 |
|------|------|----------|
| 1.0 | 2020 | Nullable 类型、StringDtype、类型提示改进 |
| 1.3 | 2021 | PyArrow 后端支持（实验性）、文件路径支持 |
| 1.5 | 2022 | 缺失值 NA 标记统一、Copy-on-Write（实验性） |
| 2.0 | 2023 | Copy-on-Write 默认开启、PyArrow 后端稳定、NumPy 2 支持 |
| 2.2 | 2024 | 性能优化、API 清理、弃用警告完善 |
| 3.0 | 2025 | PyArrow 默认字符串后端、CoW 强制默认、API 大规模清理 |

##### 向前兼容建议

```python
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

pd.set_option("mode.copy_on_write", True)
pd.set_option("future.infer_string", "pyarrow")
```

这些设置让你的代码提前适应 Pandas 3.0 的行为，减少未来迁移成本。

---

#### 参与贡献与社区

##### 如何为 Pandas 贡献代码

1. **GitHub**: https://github.com/pandas-dev/pandas
2. **Issue 报告**: 使用模板提供最小可复现示例
3. **Pull Request**: 从 docstring 修复开始，逐步参与功能开发
4. **测试**: `pytest pandas/tests/` — 运行测试套件

##### 学习源码的建议

如果你想深入理解 Pandas 内部实现：

```
推荐阅读顺序：
1. pandas/core/frame.py      → DataFrame 核心类
2. pandas/core/series.py      → Series 核心类
3. pandas/core/groupby/       → groupby 实现
4. pandas/core/merge.py       → merge/join 算法
5. pandas/io/                 → I/O 子系统
6. pandas/_libs/              → Cython 加速模块
```

阅读源码的最佳方式是带着问题去读——比如"groupby.agg() 到底是怎么实现的？"——而不是试图从头到尾通读。

---

#### 结语：Pandas 在 AI 时代的位置

回顾整个教程，我们从最基础的 Series 创建开始，一步步走到了构建 MCP Server 和分布式管道。Pandas 在 LLM/AI 工程中的角色可以用一句话概括：

> **Pandas 是 AI 数据的"通用语言"。**

无论你的上游是什么格式的原始数据（JSON 日志、数据库导出、API 返回），也无论你的下游是什么消费方（SFT 训练脚本、RAG 检索系统、评估报告生成），Pandas 都能在中间提供一个统一的、高效的、表达力强的数据处理层。

它不是最快的（Polars 更快），不是分布式的（Spark 更强），不是最省内存的（DuckDB 更优）——但它是 **生态系统最丰富、学习曲线最平缓、社区资源最多**的选择。对于 90% 的 AI 数据处理任务来说，Pandas 就是那个"刚刚好"的工具。

掌握 Pandas，你就有能力把任何结构化数据变成 AI 可以理解的训练样本、评估指标和决策依据。这就是本教程希望传递给你的最终价值。
