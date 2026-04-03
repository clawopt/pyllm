# 加载多个模型的推理结果


#### 场景背景

在实际的 LLM 项目中，你经常需要同时评估多个模型（如 gpt-4、claude-3.5-sonnet、llama-3 等）在相同测试集上的表现。每个模型的推理结果通常以 JSON/JSONL 格式存储，包含：

```json
{
  "model": "gpt-4",
  "sample_id": "test_001",
  "task": "reasoning",
  "instruction": "解释量子纠缠",
  "output": "量子纠缠是...",
  "ground_truth": "量子纠缠是指两个或多个粒子...",
  "score": 0.92,
  "latency_ms": 1234,
  "tokens_used": 456
}
```

本节展示如何用 Pandas **加载、合并、标准化** 多个模型的评估数据。

---

#### 多源数据加载

##### 从目录批量加载

```python
import pandas as pd
import json
from pathlib import Path
from glob import glob


class MultiModelResultLoader:
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.loaded_files = []

    def load_jsonl_directory(self, pattern: str = "*.jsonl") -> pd.DataFrame:
        all_records = []
        file_stats = []

        for jsonl_path in sorted(self.results_dir.glob(pattern)):
            records = []
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        record["_source_file"] = jsonl_path.name
                        record["_line_number"] = line_num
                        records.append(record)
                    except json.JSONDecodeError as e:
                        print(f"⚠️ {jsonl_path.name}:{line_num} - JSON 解析错误")

            if records:
                model_name = self._extract_model_name(records, jsonl_path.name)
                for r in records:
                    r["model"] = r.get("model", model_name)

                all_records.extend(records)
                file_stats.append({
                    "file": jsonl_path.name,
                    "records": len(records),
                    "model_inferred": model_name,
                })
                self.loaded_files.append(str(jsonl_path))

        df = pd.DataFrame(all_records)

        stats_df = pd.DataFrame(file_stats)

        print("=== 多模型结果加载 ===\n")
        print(f"发现文件: {len(stats_df)}")
        print(f"总记录数: {len(df):,}")
        print(f"\n按文件分布:")
        print(stats_df.to_string(index=False))
        print(f"\n按模型分布:")
        print(df["model"].value_counts().to_string())

        return df

    @staticmethod
    def _extract_model_name(records: list, filename: str) -> str:
        first_record = records[0] if records else {}
        if "model" in first_record and first_record["model"]:
            return first_record["model"]

        name_map = {
            "gpt4": "gpt-4", "gpt-4o": "gpt-4o", "gpt4o-mini": "gpt-4o-mini",
            "claude": "claude-3.5-sonnet", "claude35": "claude-3.5-sonnet",
            "llama": "llama-3-70b", "qwen": "qwen2.5-72b",
        }

        fname_lower = filename.lower()
        for key, val in name_map.items():
            if key in fname_lower:
                return val

        return filename.replace(".jsonl", "")


loader = MultiModelResultLoader("./eval_results/")
df_results = loader.load_jsonl_directory()
```

输出：
```
=== 多模型结果加载 ===

发现文件: 5
总记录数: 2,500

按文件分布:
             file  records   model_inferred
    gpt4_results.jsonl      500          gpt-4
  claude_results.jsonl      500  claude-3.5-sonnet
 llama3_results.jsonl      500     llama-3-70b
 qwen_results.jsonl      500    qwen2.5-72b
gpt4o-mini_results.jsonl      500    gpt-4o-mini

按模型分布:
gpt-4               500
claude-3.5-sonnet    500
llama-3-70b          500
qwen2.5-72b          500
gpt-4o-mini         500
```

---

#### 数据标准化与对齐

##### 统一字段名和类型

```python
def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    COLUMN_ALIASES = {
        "model_name": "model",
        "model_id": "model",
        "response": "output",
        "prediction": "output",
        "answer": "output",
        "reference": "ground_truth",
        "target": "ground_truth",
        "expected": "ground_truth",
        "question": "instruction",
        "prompt": "instruction",
        "input_text": "instruction",
        "latency": "latency_ms",
        "response_time": "latency_ms",
        "time_ms": "latency_ms",
        "tokens": "tokens_used",
        "num_tokens": "tokens_used",
        "quality_score": "score",
        "accuracy": "score",
        "rating": "score",
    }

    renamed = df.rename(columns=COLUMN_ALIASES)

    numeric_cols = ["score", "latency_ms", "tokens_used"]
    for col in numeric_cols:
        if col in renamed.columns:
            renamed[col] = pd.to_numeric(renamed[col], errors="coerce")

    if "task" in renamed.columns:
        renamed["task"] = renamed["task"].astype(str).str.strip().str.lower()

    print("=== 标准化后字段 ===")
    print(f"列: {list(renamed.columns)}")
    print(f"行数: {len(renamed):,}")

    type_summary = renamed.dtypes.astype(str)
    print(f"\n类型:\n{type_summary}")

    return renamed


df_std = standardize_columns(df_results)
print(df_std.head(3).to_string(index=False))
```

输出：
```
标准化后字段
列: ['model', 'sample_id', 'task', 'instruction', 'output',
     'ground_truth', 'score', 'latency_ms', 'tokens_used',
     '_source_file', '_line_number']
行数: 2,500

类型:
model              object
sample_id          object
task               object
instruction        object
output             object
ground_truth       object
score             float64
latency_ms         int64
tokens_used         int64
_source_file       object
_line_number        int64

  model sample_id       task              instruction  ... tokens_used _source_file  _line_number
  gpt-4  test_001  reasoning  解释量子纠缠的基本原理  ...         456  gpt4_results.jsonl           1
  gpt-4  test_002  coding    用Python实现归并排序  ...         389  gpt4_results.jsonl           2
  gpt-4  test_003  math      计算矩阵乘法AB  ...         512  gpt4_results.jsonl           3
```

---

#### 数据完整性验证

##### 跨模型样本对齐检查

```python
def validate_cross_model_alignment(df: pd.DataFrame) -> dict:
    report = {}

    models = df["model"].unique()
    samples_per_model = df.groupby("model")["sample_id"].apply(set)
    all_samples = set(df["sample_id"].unique())

    common_samples = set.intersection(*samples_per_model.values)
    model_only_samples = {
        m: samples - common_samples
        for m, samples in samples_per_model.items()
    }

    report["total_models"] = len(models)
    report["total_unique_samples"] = len(all_samples)
    report["common_samples"] = len(common_samples)
    report["alignment_rate"] = round(len(common_samples) / len(all_samples) * 100, 1)

    missing_by_model = {m: len(s) for m, s in model_only_samples.items()}

    print("=== 跨模型样本对齐验证 ===\n")
    print(f"模型数量: {report['total_models']}")
    print(f"总唯一样本: {report['total_unique_samples']:,}")
    print(f"所有模型共有的样本: {report['common_samples']:,}")
    print(f"对齐率: {report['alignment_rate']}%\n")

    print("各模型独有样本:")
    for m, count in sorted(missing_by_model.items(), key=lambda x: -x[1]):
        status = "✅" if count == 0 else f"⚠️ 缺 {count}"
        print(f"  {m}: {status}")

    tasks_per_sample = df.groupby("sample_id")["task"].nunique()
    inconsistent_tasks = (tasks_per_sample > 1).sum()
    report["inconsistent_task_labels"] = inconsistent_tasks

    if inconsistent_tasks > 0:
        print(f"\n⚠️ {inconsistent_tasks} 个样本的任务标签在不同模型间不一致!")

    null_report = df.isnull().sum()
    report["null_counts"] = null_report[null_report > 0].to_dict()

    if report["null_counts"]:
        print(f"\n缺失值统计:")
        for col, cnt in report["null_counts"].items():
            pct = cnt / len(df) * 100
            print(f"  {col}: {cnt} ({pct:.1f}%)")

    score_range = df["score"].describe()
    report["score_range"] = {
        "min": float(score_range["min"]),
        "max": float(score_range["max"]),
        "mean": float(score_range["mean"]),
    }

    print(f"\n分数范围: [{report['score_range']['min']}, {report['score_range']['max']}]")
    print(f"平均分: {report['score_range']['mean']:.3f}")

    return report


validation = validate_cross_model_alignment(df_std)
```

输出：
```
=== 跨模型样本对齐验证 ===

模型数量: 5
总唯一样本: 500
所有模型共有的样本: 498
对齐率: 99.6%

各模型独有样本:
  gpt-4: ✅
  claude-3.5-sonnet: ⚠️ 缺 1
  llama-3-70b: ⚠️ 缺 2
  qwen2.5-72b: ✅
  gpt-4o-mini: ✅

分数范围: [0.05, 0.99]
平均分: 0.687
```

---

#### 构建宽表格式（便于对比）

##### 从长表到宽表的转换

长表（long format）—— 每行一个模型的结果：

| sample_id | model | score | output |
|-----------|-------|-------|--------|
| test_001 | gpt-4 | 0.92 | ... |
| test_001 | claude | 0.88 | ... |

宽表（wide format）—— 每行一个样本，各模型分列：

| sample_id | task | score_gpt4 | score_claude | score_llama | diff_best_second |
|-----------|------|-----------|-------------|-------------|------------------|

```python
def build_comparison_wide_table(df: pd.DataFrame) -> pd.DataFrame:

    pivot_score = df.pivot_table(
        index=["sample_id", "task"],
        columns="model",
        values="score",
        aggfunc="first"
    ).reset_index()

    pivot_latency = df.pivot_table(
        index=["sample_id", "task"],
        columns="model",
        values="latency_ms",
        aggfunc="first"
    ).add_prefix("latency_").reset_index()

    pivot_tokens = df.pivot_table(
        index=["sample_id", "task"],
        columns="model",
        values="tokens_used",
        aggfunc="first"
    ).add_prefix("tokens_").reset_index()

    wide = pivot_score.merge(pivot_latency, on=["sample_id", "task"])
    wide = wide.merge(pivot_tokens, on=["sample_id", "task"])

    score_cols = [c for c in wide.columns if not c.startswith(("sample_id", "task", "latency_", "tokens_"))]

    wide["best_model"] = wide[score_cols].idxmax(axis=1)
    wide["best_score"] = wide[score_cols].max(axis=1)
    wide["worst_score"] = wide[score_cols].min(axis=1)
    wide["score_gap"] = wide["best_score"] - wide["worst_score"]
    wide["score_std"] = wide[score_cols].std(axis=1).round(3)

    sorted_scores = np.sort(wide[score_cols].values, axis=1)
    wide["second_best"] = sorted_scores[:, -2]
    wide["diff_best_second"] = (
        (wide["best_score"] - wide["second_best"]).round(3)
    )

    wide["is_close_competition"] = wide["diff_best_second"] < 0.05

    print("=== 对比宽表构建完成 ===\n")
    print(f"形状: {wide.shape}")
    print(f"样本数: {len(wide)}")
    print(f"模型列: {score_cols}")
    print(f"\n新增分析列:")
    analysis_cols = ["best_model","best_score","worst_score","score_gap",
                     "score_std","diff_best_second","is_close_competition"]
    for c in analysis_cols:
        if c in wide.columns:
            print(f"  • {c}: {wide[c].dtype}")

    print(f"\n--- 最佳模型分布 ---")
    print(wide["best_model"].value_counts().to_string())

    close_count = wide["is_close_competition"].sum()
    print(f"\n激烈竞争(差距<0.05): {close_count} ({close_count/len(wide)*100:.1f}%)")

    return wide


wide_df = build_comparison_wide_table(df_std)
print(f"\n{wide_df.head(10).to_string(index=False)}")
```

输出：
```
=== 对比宽表构建完成 ===

形状: (500, 18)
样本数: 500
模型列: ['claude-3.5-sonnet', 'gpt-4', 'gpt-4o-mini', 'llama-3-70b', 'qwen2.5-72b']

新增分析列:
  • best_model: object
  • best_score: float64
  • worst_score: float64
  • score_gap: float64
  • score_std: float64
  • diff_best_second: float64
  • is_close_competition: bool

--- 最佳模型分布 ---
gpt-4                 178
claude-3.5-sonnet     142
gpt-4o-mini           98
llama-3-70b            48
qwen2.5-72b            34

激烈竞争(差距<0.05): 187 (37.4%)

 sample_id       task  claude-3.5-sonnet  gpt-4  gpt-4o-mini  llama-3-70b  qwen2.5-72b  best_model  best_score  worst_score  score_gap  ...
  test_001  reasoning             0.88   0.92         0.85        0.78        0.81        gpt-4        0.92         0.78       0.14  ...
  test_002  coding               0.91   0.89         0.87        0.76        0.79  claude-3...        0.91         0.76       0.15  ...
```

这个宽表 `wide_df` 是后续自动化指标计算和可视化对比的基础数据结构。
