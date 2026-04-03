# 可视化对比图表


#### 可视化策略

模型评估的可视化需要传达 **"谁更好？好多少？在哪些方面？"** 这三个核心问题。我们用 Pandas 内置绑图 + Matplotlib 扩展来实现。

---

#### 模型总体对比雷达图

```python
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False


def plot_radar_comparison(engine: ModelBenchmarkEngine):
    basic = engine.compute_basic_metrics().set_index("model")

    categories = ["平均分", "稳定性(CV)", "P10(鲁棒性)",
                  "P90(顶尖)", "胜率%", "性价比"]

    def normalize(series, higher_better=True):
        if higher_better:
            return (series - series.min()) / (series.max() - series.min() + 1e-10)
        else:
            return (series.max() - series) / (series.max() - series.min() + 1e-10)

    metrics_data = pd.DataFrame({
        "平均分": normalize(basic["mean"]),
        "稳定性(CV)": normalize(basic["cv"], higher_better=False),
        "P10(鲁棒性)": normalize(basic["p10"]),
        "P90(顶尖)": normalize(basic["p90"]),
    })

    win_rates = engine.compute_win_rate().set_index("model")
    eff = engine.compute_efficiency_metrics().set_index("model")

    metrics_data["胜率%"] = normalize(win_rates["win_rate"])
    metrics_data["性价比"] = normalize(eff["score_per_dollar"])

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for i, model in enumerate(metrics_data.index):
        values = metrics_data.loc[model].tolist()
        values += values[:1]

        ax.plot(angles, values, "o-", linewidth=2,
                label=model, color=colors[i % len(colors)])
        ax.fill(angles, values, alpha=0.15, color=colors[i % len(colors)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
    ax.set_title("多维度模型能力雷达图", fontsize=14, fontweight="bold", pad=20)

    plt.tight_layout()
    return fig


fig_radar = plot_radar_comparison(engine)
plt.show()
```

输出效果：一个五边形雷达图，每个顶点代表一个评估维度，5 个模型用不同颜色的多边形覆盖，面积越大表示综合能力越强。

---

#### 分组柱状图：按任务类型对比

```python
def plot_task_bar_chart(per_task_df: pd.DataFrame):
    score_cols = [c for c in per_task_df.columns
                  if c not in ["best_model", "gap_best_second"]]

    tasks = per_task_df.index.tolist()
    x = np.arange(len(tasks))
    width = 0.15
    n_models = len(score_cols)

    colors = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd"]
    fig, ax = plt.subplots(figsize=(14, 7))

    for i, col in enumerate(score_cols):
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, per_task_df[col], width,
                      label=col, color=colors[i % len(colors)], alpha=0.85)

        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.2f}",
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 2), textcoords="offset points",
                       ha="center", va="bottom", fontsize=7)

    ax.set_xlabel("任务类型", fontsize=12)
    ax.set_ylabel("平均得分", fontsize=12)
    ax.set_title("各模型在不同任务上的表现对比", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=30, ha="right")
    ax.legend(title="模型", loc="upper right")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(y=per_task_df[score_cols].values.mean(), color="red",
               linestyle="--", alpha=0.5, label="全局均值")

    plt.tight_layout()
    return fig


fig_bar = plot_task_bar_chart(per_task)
plt.show()
```

效果：分组柱状图，X 轴为任务类型（reasoning/coding/math 等），每组内各模型的柱子并排显示，可以直观看出哪个模型在哪个任务上领先。

---

#### 分布对比：箱线图与密度图

```python
def plot_distribution_comparison(df: pd.DataFrame):

    models = sorted(df["model"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    data_by_model = [df[df["model"] == m]["score"] for m in models]
    bp = ax1.boxplot(data_by_model, labels=models, patch_artist=True)

    colors = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6", "#f39c12"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax1.set_title("分数分布箱线图", fontweight="bold")
    ax1.set_ylabel("Score")
    ax1.grid(axis="y", alpha=0.3)

    for i, mdata in enumerate(data_by_model):
        y = mdata
        x = np.random.normal(i + 1, 0.04, size=len(y))
        ax1.scatter(x, y, alpha=0.15, s=8, color=colors[i])

    ax2 = axes[1]
    for i, model in enumerate(models):
        scores = df[df["model"] == model]["score"]
        scores.plot.kde(ax=ax2, label=model, linewidth=2,
                        color=colors[i % len(colors)])
        mean_val = scores.mean()
        ax2.axvline(mean_val, color=colors[i % len(colors)],
                    linestyle=":", alpha=0.7, linewidth=1)

    ax2.set_title("分数密度分布", fontweight="bold")
    ax2.set_xlabel("Score")
    ax2.set_ylabel("Density")
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_xlim(0, 1.05)

    plt.tight_layout()
    return fig


fig_dist = plot_distribution_comparison(df_std)
plt.show()
```

效果：左图为箱线图叠加散点（展示分布形态和离群点），右图为核密度估计曲线（展示分数分布的形状差异）。

---

#### 散点矩阵：两两相关性分析

```python
def plot_scatter_matrix(wide_df: pd.DataFrame, score_cols: list):

    n = len(score_cols)
    fig, axes = plt.subplots(n, n, figsize=(3 * n + 2, 3 * n + 2))

    for i in range(n):
        for j in range(n):
            ax = axes[i][j] if n > 1 else axes

            if i == j:
                data = wide_df[score_cols[j]]
                ax.hist(data, bins=25, color="#3498db", alpha=0.7, edgecolor="white")
                ax.axvline(data.mean(), color="red", linestyle="--", linewidth=1.5)
            else:
                ax.scatter(
                    wide_df[score_cols[j]], wide_df[score_cols[i]],
                    alpha=0.15, s=8, c="#2ecc71", edgecolors="none"
                )

                corr = wide_df[score_cols[j]].corr(wide_df[score_cols[i]])
                ax.set_title(f"r={corr:.2f}", fontsize=8, pad=2)

            if j == 0:
                ax.set_ylabel(score_cols[i], fontsize=8, rotation=0,
                              ha="right", va="center")
            if i == n - 1:
                ax.set_xlabel(score_cols[j], fontsize=8)
            ax.tick_params(labelsize=6)

    fig.suptitle("模型间分数相关性散点矩阵", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    return fig


score_cols = [c for c in wide_df.columns
              if c not in ("sample_id","task","best_model","best_score",
                           "worst_score","score_gap","score_std",
                           "diff_best_second","is_close_competition")]
fig_scatter = plot_scatter_matrix(wide_df, score_cols[:4])
plt.show()
```

效果：N×N 的散点矩阵，对角线为直方图，非对角线为两个模型间的散点图+相关系数。高相关系数意味着两个模型表现一致。

---

#### 热力图：任务×模型矩阵

```python
def plot_heatmap_matrix(per_task_df: pd.DataFrame):

    score_cols = [c for c in per_task_df.columns
                  if c not in ["best_model", "gap_best_second"]]

    matrix = per_task_df[score_cols].astype(float)

    fig, ax = plt.subplots(figsize=(max(6, len(score_cols)), max(5, len(matrix))))

    im = ax.imshow(matrix.values, cmap="YlGnBu", aspect="auto", vmin=0.5, vmax=0.85)

    ax.set_xticks(np.arange(len(score_cols)))
    ax.set_yticks(np.arange(len(matrix)))
    ax.set_xticklabels(score_cols, rotation=35, ha="right", fontsize=9)
    ax.set_yticklabels(matrix.index, fontsize=9)

    for i in range(len(matrix)):
        for j in range(len(score_cols)):
            val = matrix.iloc[i, j]
            text_color = "white" if val < 0.62 or val > 0.78 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                   color=text_color, fontsize=9, fontweight="bold")

    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel("Score", rotation=-90, va="bottom")

    best_per_row = matrix.idxmax(axis=1)
    for i, best_m in enumerate(best_per_row.items()):
        j = score_cols.index(best_m[1])
        rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                               fill=False, edgecolor="red", linewidth=2.5)
        ax.add_patch(rect)

    ax.set_title("任务 × 模型 得分热力图 (红框=最优)", fontweight="bold", pad=15)

    plt.tight_layout()
    return fig


fig_heatmap = plot_heatmap_matrix(per_task)
plt.show()
```

效果：热力图中颜色越深表示分数越高，红色方框标记每行的最高分（即该任务的最优模型），一目了然地看到每个任务的赢家。
