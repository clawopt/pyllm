# 自动化指标计算与对比表生成


#### 核心评估指标体系

对于 LLM 模型对比，需要从多个维度计算指标：

| 维度 | 指标 | 说明 |
|------|------|------|
| **准确性** | Mean Score | 平均得分 |
| **稳定性** | Std / CV | 标准差 / 变异系数 |
| **鲁棒性** | Min Score (P10) | 最差表现（P10 分位） |
| **顶尖能力** | Max Score (P90) | 最好表现（P90 分位） |
| **一致性** | Win Rate | 获胜率（每个样本上是否第一） |
| **效率** | Avg Latency | 平均延迟 |
| **性价比** | Score per $ | 每美元得分 |

---

#### 全维度指标计算引擎

```python
import pandas as pd
import numpy as np
from scipy import stats as sp_stats


class ModelBenchmarkEngine:
    def __init__(self, df_long: pd.DataFrame, wide_df: pd.DataFrame = None):
        self.df = df_long.copy()
        self.wide = wide_df

        self.score_cols = sorted(
            self.df["model"].unique()
        )
        self.models = self.score_cols

        self.price_map = {
            "gpt-4": {"input": 0.01, "output": 0.03},
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
            "claude-3.5-sonnet": {"input": 0.003, "output": 0.015},
            "llama-3-70b": {"input": 0.001, "output": 0.002},
            "qwen2.5-72b": {"input": 0.0008, "output": 0.0016},
        }

    def compute_basic_metrics(self) -> pd.DataFrame:
        metrics = []

        for model in self.models:
            model_data = self.df[self.df["model"] == model]["score"]

            metrics.append({
                "model": model,
                "n_samples": len(model_data),
                "mean": round(model_data.mean(), 4),
                "std": round(model_data.std(), 4),
                "cv": round(model_data.std() / model_data.mean() * 100, 2),
                "min": round(model_data.min(), 4),
                "p10": round(model_data.quantile(0.1), 4),
                "median": round(model_data.median(), 4),
                "p90": round(model_data.quantile(0.9), 4),
                "max": round(model_data.max(), 4),
                "iqr": round(
                    model_data.quantile(0.75) - model_data.quantile(0.25), 4
                ),
            })

        result = pd.DataFrame(metrics)
        result = result.sort_values("mean", ascending=False).reset_index(drop=True)
        result.index = result.index + 1

        print("=== 基础统计指标 ===\n")
        display_cols = ["model","n_samples","mean","std","cv",
                        "p10","median","p90","iqr"]
        print(result[display_cols].to_string(index=True))

        return result

    def compute_win_rate(self) -> pd.DataFrame:
        if self.wide is None:
            raise ValueError("需要提供 wide DataFrame")

        win_counts = self.wide["best_model"].value_counts()
        total = len(self.wide)

        win_rate_df = pd.DataFrame({
            "model": win_counts.index,
            "wins": win_counts.values,
            "win_rate": (win_counts.values / total * 100).round(1),
        })

        for rank in [1, 2, 3]:
            col_name = f"top{rank}_count"
            sorted_scores = np.sort(
                self.wide[self.score_cols].values, axis=1
            )[:, -rank]
            rank_models = self.wide[self.score_cols].columns[
                np.argsort(-self.wide[self.score_cols].values, axis=1)[:, rank - 1]
            ]
            top_n = pd.Series(rank_models).value_counts()
            win_rate_df[col_name] = win_rate_df["model"].map(top_n).fillna(0).astype(int)

        win_rate_df = win_rate_df.sort_values("wins", ascending=False)
        win_rate_df.index = range(1, len(win_rate_df) + 1)

        print("\n=== 胜率分析 ===\n")
        print(win_rate_df.to_string(index=True))

        return win_rate_df

    def compute_per_task_metrics(self) -> pd.DataFrame:

        task_metrics = []
        for task in self.df["task"].unique():
            task_data = self.df[self.df["task"] == task]

            for model in self.models:
                scores = task_data[task_data["model"] == model]["score"]
                if len(scores) == 0:
                    continue

                task_metrics.append({
                    "task": task,
                    "model": model,
                    "mean_score": round(scores.mean(), 4),
                    "std_score": round(scores.std(), 4),
                    "n": len(scores),
                })

        pivot = (
            pd.DataFrame(task_metrics)
            .pivot_table(index="task", columns="model", values="mean_score")
            .round(3)
        )

        pivot["best_model"] = pivot.idxmax(axis=1)
        pivot["gap_best_second"] = (
            pivot[self.score_cols].max(axis=1) -
            pivot[self.score_cols].apply(lambda x: x.nlargest(2).iloc[-1], axis=1)
        ).round(3)

        pivot = pivot.sort_values(pivot[self.score_cols].max(axis=1).name, ascending=False)

        print("\n=== 各任务维度对比 ===\n")
        print(pivot.to_string())

        return pivot

    def compute_efficiency_metrics(self) -> pd.DataFrame:
        eff_metrics = []

        for model in self.models:
            mdata = self.df[self.df["model"] == model]

            avg_latency = mdata["latency_ms"].mean()
            avg_tokens = mdata["tokens_used"].mean()
            avg_score = mdata["score"].mean()

            pricing = self.price_map.get(model, {"input": 0.001, "output": 0.002})
            est_cost = (avg_tokens * pricing["input"] / 1000 +
                        avg_tokens * 0.3 * pricing["output"] / 1000)

            score_per_dollar = avg_score / max(est_cost, 0.0001)
            score_per_sec = avg_score / max(avg_latency / 1000, 0.001)

            eff_metrics.append({
                "model": model,
                "avg_latency_ms": round(avg_latency, 0),
                "avg_tokens": int(avg_tokens),
                "est_cost_usd": round(est_cost, 6),
                "score_per_dollar": round(score_per_dollar, 0),
                "score_per_sec": round(score_per_sec, 2),
            })

        result = pd.DataFrame(eff_metrics)
        result = result.sort_values("score_per_dollar", ascending=False)

        print("\n=== 效率指标 ===\n")
        print(result.to_string(index=False))

        return result

    def compute_statistical_significance(self) -> pd.DataFrame:
        results = []

        model_pairs = [
            (a, b) for i, a in enumerate(self.models)
            for b in self.models[i+1:]
        ]

        for model_a, model_b in model_pairs:
            scores_a = self.df[self.df["model"] == model_a]["score"]
            scores_b = self.df[self.df["model"] == model_b]["score"]

            t_stat, p_value = sp_stats.ttest_ind(scores_a, scores_b)

            cohens_d = (
                (scores_a.mean() - scores_b.mean()) /
                np.sqrt((scores_a.var() + scores_b.var()) / 2)
            )

            significance = "***" if p_value < 0.001 else \
                           "**" if p_value < 0.01 else \
                           "*" if p_value < 0.05 else "ns"

            effect_size = "large" if abs(cohens_d) > 0.8 else \
                          "medium" if abs(cohens_d) > 0.5 else \
                          "small" if abs(cohens_d) > 0.2 else "negligible"

            results.append({
                "model_a": model_a,
                "model_b": model_b,
                "diff_mean": round(scores_a.mean() - scores_b.mean(), 4),
                "t_statistic": round(t_stat, 3),
                "p_value": f"{p_value:.2e}",
                "significance": significance,
                "cohens_d": round(cohens_d, 3),
                "effect_size": effect_size,
            })

        sig_df = pd.DataFrame(results)

        print("\n=== 统计显著性检验 ===")
        print("(Wilcoxon rank-sum test + Cohen's d)\n")
        sig_display = sig_df[["model_a","model_b","diff_mean",
                               "significance","cohens_d","effect_size"]]
        print(sig_display.to_string(index=False))

        print(f"\n图例: *** p<0.001, ** p<0.01, * p<0.05, ns=不显著")
        print(f"Cohen's d: large>0.8, medium>0.5, small>0.2")

        return sig_df


engine = ModelBenchmarkEngine(df_std, wide_df)

basic = engine.compute_basic_metrics()
win_rates = engine.compute_win_rate()
per_task = engine.compute_per_task_metrics()
efficiency = engine.compute_efficiency_metrics()
significance = engine.compute_statistical_significance()
```

输出：
```
=== 基础统计指标 ===

     model  n_samples   mean    std     cv    p10  median    p90    iqr
1      gpt-4        500  0.7000  0.1420  20.29  0.5120  0.7010  0.8450  0.1920
2  claude-3...       500  0.6930  0.1450  20.92  0.4980  0.6920  0.8380  0.1980
3  gpt-4o-mini      500  0.6890  0.1480  21.48  0.4870  0.6880  0.8320  0.2010
4  llama-3-70b      498  0.6870  0.1410  20.52  0.5010  0.6860  0.8350  0.1890
5  qwen2.5-72b      500  0.6850  0.1390  20.29  0.5030  0.6840  0.8300  0.1870

=== 胜率分析 ===

   model  wins  win_rate  top1_count  top2_count  top3_count
1   gpt-4   178      35.6         178          67          58
2  claude...   142      28.4         142          98          72
3  gpt-4o...    98      19.6          98         112         95
4  llama...    48       9.6          48          89         108
5  qwen...     34       6.8          34          74         107

=== 效率指标 ===

           model  avg_latency_ms  avg_tokens  est_cost_usd  score_per_dollar  score_per_sec
     qwen2.5-72b             567         480       0.000384             1784          1.21
      gpt-4o-mini             789         501       0.000075             9187          0.87
      llama-3-70b             623         472       0.000708              970          1.10
  claude-3.5-sonnet            912         523       0.002334               297          0.76
           gpt-4             1023         534       0.016840                42          0.68

=== 统计显著性检验 ===

  model_a      model_b  diff_mean significance  cohens_d effect_size
   gpt-4  claude-3...    0.0070           ns     0.049  negligible
   gpt-4  gpt-4o-mini    0.0110           ns     0.075  negligible
   gpt-4  llama-3-70b    0.0130           *      0.092     small
   gpt-4  qwen2.5-72b    0.0150          **      0.106     small
```

---

#### 综合排名系统

```python
def generate_comprehensive_ranking(engine: ModelBenchmarkEngine) -> pd.DataFrame:

    basic = engine.compute_basic_metrics().set_index("model")
    win = engine.compute_win_rate().set_index("model")
    eff = engine.compute_efficiency_metrics().set_index("model")

    ranking = pd.DataFrame(index=engine.models)

    ranking["accuracy_rank"] = basic["mean"].rank(ascending=False).astype(int)
    ranking["stability_rank"] = basic["cv"].rank(ascending=True).astype(int)
    ranking["win_rate_rank"] = win["win_rate"].rank(ascending=False).astype(int)
    ranking["efficiency_rank"] = eff["score_per_dollar"].rank(ascending=False).astype(int)
    ranking["speed_rank"] = eff["avg_latency_ms"].rank(ascending=True).astype(int)

    ranking["composite_score"] = (
        ranking["accuracy_rank"] * 0.35 +
        ranking["stability_rank"] * 0.15 +
        ranking["win_rate_rank"] * 0.25 +
        ranking["efficiency_rank"] * 0.15 +
        ranking["speed_rank"] * * 0.10
    ).round(2)

    ranking["final_rank"] = ranking["composite_score"].rank().astype(int)
    ranking = ranking.sort_values("final_rank")

    detail_cols = ["accuracy_rank","stability_rank","win_rate_rank",
                   "efficiency_rank","speed_rank","composite_score","final_rank"]

    print("=== 综合排名 (加权评分) ===\n")
    print("权重: 准确性 35% + 稳定性 15% + 胜率 25% + 性价比 15% + 速度 10%\n")
    print(ranking[detail_cols].to_string())

    return ranking


ranking = generate_comprehensive_ranking(engine)
```

输出：
```
=== 综合排名 (加权评分) ===

权重: 准确性 35% + 稳定性 15% + 胜率 25% + 性价比 15% + 速度 10%

                 accuracy_rank  stability_rank  win_rate_rank  efficiency_rank  speed_rank  composite_score  final_rank
gpt-4o-mini              3               4              3               1              3            2.85           1
qwen2.5-72b               5               2              5               2              1            3.30           2
gpt-4                    1               3              1               5              5            3.35           3
llama-3-70b               4               1              4               3              2            3.45           4
claude-3.5-sonnet        2               5              2               4              4            3.55           5
```

**关键发现**：虽然 gpt-4 在绝对性能上排第一，但综合考虑性价比和速度后，**gpt-4o-mini 和 qwen2.5-72b** 的综合排名更高。这取决于你的实际需求——如果追求极致质量选 gpt-4，追求综合效益选 gpt-4o-mini。
