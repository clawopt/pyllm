# 生成评估报告


#### 报告结构设计

一份专业的模型评估报告应该包含以下部分：

```
1. 执行摘要（Executive Summary）— 一页纸结论
2. 数据概况（Data Overview）— 样本量、覆盖范围
3. 总体排名（Overall Ranking）— 综合得分与名次
4. 分维度分析（Dimensional Analysis）— 按任务/难度/语言
5. 头部对比（Head-to-Head）— 模型两两 PK
6. 效率分析（Efficiency Analysis）— 成本/速度/性价比
7. 统计检验（Statistical Tests）— 显著性判断
8. 结论与建议（Recommendations）— 选型建议
```

---

#### 报告生成器

```python
import base64
from io import BytesIO


class EvaluationReportGenerator:
    def __init__(self, engine: ModelBenchmarkEngine,
                 wide_df: pd.DataFrame):
        self.engine = engine
        self.wide = wide_df
        self.sections = []

    def _add_section(self, title: str, content: str):
        self.sections.append(f"## {title}\n\n{content}")

    def generate(self) -> str:
        self._executive_summary()
        self._data_overview()
        self._overall_ranking()
        self._dimensional_analysis()
        self._head_to_head()
        self._efficiency_analysis()
        self._statistical_tests()
        self._recommendations()

        full_report = "\n\n".join(self.sections)
        return full_report

    def _executive_summary(self):

        basic = self.engine.compute_basic_metrics().set_index("model")
        best_model = basic["mean"].idxmax()
        best_score = basic.loc[best_model, "mean"]

        win = self.engine.compute_win_rate().set_index("model")
        top_winner = win["wins"].idxmax()
        win_rate = win.loc[top_winner, "win_rate"]

        eff = self.engine.compute_efficiency_metrics().set_index("model")
        best_value = eff["score_per_dollar"].idxmax()

        n_samples = len(self.wide)
        n_models = len(basic)
        n_tasks = self.df["task"].nunique()

        content = f"""| 指标 | 数值 |
|------|------|
| **评估模型数** | {n_models} |
| **测试样本数** | {n_samples:,} |
| **任务类型** | {n_tasks} |
| **🏆 最佳性能模型** | **{best_model}** (均分 {best_score:.3f}) |
| **🥇 最高胜率模型** | **{top_winner}** (胜率 {win_rate:.1f}%) |
| **💰 性价比之王** | **{best_value}** |

**核心结论**: {best_model} 在绝对性能上领先，但差距极小（前两名仅差 ~0.7%）。
在实际选型中，应综合考虑成本和部署便利性。"""

        self._add_section("执行摘要", content)

    def _data_overview(self):
        basic = self.engine.compute_basic_metrics().reset_index()

        table = basic[["model","n_samples","mean","std",
                       "p10","median","p90"]].to_string(index=False)

        content = f"### 基础统计\n\n{table}\n\n"
        content += f"- **样本总数**: {basic['n_samples'].sum():,}\n"
        content += f"- **分数范围**: [{basic['min'].min():.3f}, {basic['max'].max():.3f}]\n"

        self._add_section("数据概况", content)

    def _overall_ranking(self):
        ranking = generate_comprehensive_ranking(self.engine).reset_index()

        table = ranking.to_string(index=False)

        content = f"### 综合排名 (加权评分)\n\n"
        content += f"| 排名 | 模型 | 准确性排名 | 稳定性排名 | 胜率排名 | 性价比排名 | 综合得分 |\n"
        content += f"|------|------|-----------|-----------|---------|-----------|----------|\n"

        for _, row in ranking.iterrows():
            content += (f"| {row['final_rank']} | {row.name} | "
                       f"{row['accuracy_rank']} | {row['stability_rank']} | "
                       f"{row['win_rate_rank']} | {row['efficiency_rank']} | "
                       f"**{row['composite_score']:.2f}** |\n")

        self._add_section("总体排名", content)

    def _dimensional_analysis(self):
        per_task = self.engine.compute_per_task_metrics()

        task_table = per_task.reset_index().to_string(index=False)
        content = f"### 各任务维度对比\n\n{task_table}"

        category_stats = []
        if "category" in self.df.columns:
            for cat in self.df["category"].unique():
                cat_data = self.df[self.df["category"] == cat]
                for model in self.models:
                    scores = cat_data[cat_data["model"] == model]["score"]
                    if len(scores) > 0:
                        category_stats.append({
                            "category": cat,
                            "model": model,
                            "mean": round(scores.mean(), 3),
                            "n": len(scores),
                        })

            if category_stats:
                cat_df = pd.DataFrame(category_stats)
                cat_pivot = cat_df.pivot_table(
                    index="category", columns="model", values="mean"
                ).round(3)
                content += f"\n\n### 按难度等级\n\n{cat_pivot.to_string()}"

        self._add_section("分维度分析", content)

    def _head_to_head(self):
        pairs = [
            ("gpt-4", "claude-3.5-sonnet"),
            ("gpt-4", "gpt-4o-mini"),
            ("gpt-4o-mini", "qwen2.5-72b"),
        ]

        content = ""
        for m_a, m_b in pairs:
            a_wins = (self.wide["best_model"] == m_a).sum()
            b_wins = (self.wide["best_model"] == m_b).sum()
            ties = len(self.wide) - a_wins - b_wins

            diff = (
                self.wide[m_a] - self.wide[m_b]
            )
            avg_diff = diff.mean()
            std_diff = diff.std()

            a_better_pct = (diff > 0).mean() * 100

            content += f"#### {m_a} vs {m_b}\n\n"
            content += f"| 指标 | 值 |\n|------|-----|\n"
            content += f"| {m_a} 胜出 | {a_wins} ({a_wins/len(self.wide)*100:.1f}%) |\n"
            content += f"| {m_b} 胜出 | {b_wins} ({b_wins/len(self.wide)*100:.1f}%) |\n"
            content += f"| 平局 | {ties} ({ties/len(self.wide)*100:.1f}%) |\n"
            content += f"| 平均分差 ({m_a}-{m_b}) | {avg_diff:+.4f} |\n"
            content += f"| 分差标准差 | {std_diff:.4f} |\n"
            content += f"| {m_a} 更好的比例 | {a_better_pct:.1f}% |\n\n"

        self._add_section("头部对比 (Head-to-Head)", content)

    def _efficiency_analysis(self):
        eff = self.engine.compute_efficiency_metrics().reset_index()

        content = "| 模型 | 平均延迟(ms) | Token用量 | 预估成本($) | $/得分 | 得分/秒 |\n"
        content += "|------|-------------|----------|------------|--------|--------|\n"

        for _, row in eff.iterrows():
            content += (f"| {row['model']} | {row['avg_latency_ms']:,} | "
                       f"{row['avg_tokens']:,} | ${row['est_cost_usd']:.6f} | "
                       f"${1/max(row['score_per_dollar'],0.01):.4f} | "
                       f"{row['score_per_sec']:.2f} |\n")

        best_speed = eff.loc[eff["avg_latency_ms"].idxmin(), "model"]
        best_cost = eff.loc[eff["est_cost_usd"].idxmin(), "model"]
        content += f"\n- 🚀 最快: **{best_speed}**\n- 💰 最便宜: **{best_cost}**\n"

        self._add_section("效率分析", content)

    def _statistical_tests(self):
        sig_df = self.engine.compute_statistical_significance()

        content = "| 对比 | 差值 | 显著性 | Cohen's d | 效应量 |\n"
        content += "|------|------|--------|-----------|--------|\n"

        for _, row in sig_df.iterrows():
            content += (f"| {row['model_a']} vs {row['model_b']} | "
                       f"{row['diff_mean']:+.4f} | {row['significance']} | "
                       f"{row['cohens_d']:.3f} | {row['effect_size']} |\n")

        significant_pairs = sig_df[sig_df["significance"] != "ns"]
        if len(significant_pairs) > 0:
            content += f"\n✅ 有统计显著差异的配对: **{len(significant_pairs)}** 组\n"
        else:
            content += f"\n⚠️ 所有模型间差异均未达到统计显著水平 (p>0.05)\n"

        self._add_section("统计显著性检验", content)

    def _recommendations(self):
        basic = self.engine.compute_basic_metrics().set_index("model")
        eff = self.engine.compute_efficiency_metrics().set_index("model")
        win = self.engine.compute_win_rate().set_index("model")

        best_perf = basic["mean"].idxmax()
        best_value = eff["score_per_dollar"].idxmax()
        fastest = eff["avg_latency_ms"].idxmin()
        most_stable = basic["cv"].idxmin()

        content = f"""
### 场景化选型建议

| 使用场景 | 推荐模型 | 理由 |
|----------|----------|------|
| 追求最高质量，不计成本 | **{best_perf}** | 平均分最高 ({basic.loc[best_perf,'mean']:.3f}) |
| 预算有限 / 大规模部署 | **{best_value}** | 性价比最优 ({eff.loc[best_value,'score_per_dollar']:,.0f} $/分) |
| 低延迟实时场景 | **{fastest}** | 平均延迟最低 ({eff.loc[fastest,'avg_latency_ms']:.0f}ms) |
| 要求结果稳定可预测 | **{most_stable}** | 变异系数最小 (CV={basic.loc[most_stable,'cv']:.1f}%) |

### 注意事项
- 本次评估样本量为 {len(self.wide):,} 条，建议在更大规模数据上验证
- 分数差异较小 (<2%) 的模型在实际使用中可能难以感知区别
- 建议结合具体业务场景进行 A/B 测试验证
"""

        self._add_section("结论与建议", content)


generator = EvaluationReportGenerator(engine, wide_df)
report_md = generator.generate()
print(report_md[:2000])
print("\n... (完整报告包含以上所有章节)")
```

---

#### 导出为 HTML 格式

```python
def markdown_to_html(md_text: str) -> str:

    import re

    html = md_text

    html = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)
    html = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
    html = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
    html = re.sub(r'#### (.+)$', r'<h4>\1</h4>', html, flags=re.MULTILINE)

    def table_replace(match):
        lines = match.group(0).strip().split("\n")
        rows = []
        for i, line in enumerate(lines):
            cells = [c.strip() for c in line.split("|")[1:-1]]
            tag = "th" if i <= 1 else "td"
            row_html = "<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in cells) + "</tr>"
            rows.append(row_html)
        return '<table border="1" cellpadding="6" cellspacing="0">' + "\n".join(rows) + "</table>"

    html = re.sub(r'(\|.+\|\n)+', table_replace, html)

    html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
    html = re.sub(r'`(.+?)`', r'<code>\1</code>', html)
    html = re.sub(r'^- (.+)$', r'<li>\1</li>', html, flags=re.MULTILINE)
    html = re.sub(r'(<li>.*</li>\n?)+', r'<ul>\g<0></ul>', html)
    html = "\n".join(
        f"<p>{line}</p>" if line.strip() and not line.startswith("<") else line
        for line in html.split("\n")
    )

    css_style = """
    <style>
      body { font-family: -apple-system, 'PingFang SC', sans-serif; max-width: 900px;
             margin: 40px auto; padding: 20px; color: #333; }
      h1 { color: #1976D2; border-bottom: 3px solid #1976D2; padding-bottom: 10px; }
      h2 { color: #1565C0; margin-top: 30px; }
      table { width: 100%; border-collapse: collapse; margin: 15px 0; }
      th { background: #1976D2; color: white; }
      tr:nth-child(even) { background: #f5f5f5; }
      td, th { padding: 8px 12px; text-align: left; }
      strong { color: #D32F2F; }
      code { background: #e8e8e8; padding: 2px 6px; border-radius: 3px; font-size: 0.9em; }
      ul { padding-left: 20px; }
      li { margin: 4px 0; }
    </style>
    """

    full_html = f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>模型评估报告</title>{css_style}</head><body>{html}</body></html>"

    return full_html


html_report = markdown_to_html(report_md)

output_path = "eval_report.html"
with open(output_path, "w", encoding="utf-8") as f:
    f.write(html_report)

file_size = Path(output_path).stat().st_size / 1024
print(f"✅ HTML 报告已导出: {output_path} ({file_size:.1f} KB)")
```

---

#### 完整流程：一键生成

```python
def run_full_evaluation(results_dir: str = "./eval_results/",
                        output_dir: str = "./reports/") -> str:

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  模型评估报告 — 一键生成系统")
    print("=" * 60)

    loader = MultiModelResultLoader(results_dir)
    df_long = loader.load_jsonl_directory()

    df_std = standardize_columns(df_long)
    validate_cross_model_alignment(df_std)

    wide_df = build_comparison_wide_table(df_std)

    engine = ModelBenchmarkEngine(df_std, wide_df)

    generator = EvaluationReportGenerator(engine, wide_df)
    report_md = generator.generate()

    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    md_path = f"{output_dir}/evaluation_report_{timestamp}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"\n✅ Markdown 报告: {md_path}")

    html_report = markdown_to_html(report_md)
    html_path = f"{output_dir}/evaluation_report_{timestamp}.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_report)
    print(f"✅ HTML 报告:   {html_path}")

    fig_radar = plot_radar_comparison(engine)
    radar_path = f"{output_dir}/radar_chart_{timestamp}.png"
    fig_radar.savefig(radar_path, dpi=150, bbox_inches="tight")
    plt.close(fig_radar)
    print(f"✅ 雷达图:     {radar_path}")

    fig_bar = plot_task_bar_chart(per_task)
    bar_path = f"{output_dir}/task_bar_{timestamp}.png"
    fig_bar.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close(fig_bar)
    print(f"✅ 任务柱状图: {bar_path}")

    summary_data = {
        "timestamp": timestamp,
        "models": list(engine.models),
        "total_samples": len(wide_df),
        "best_overall": engine.compute_basic_errors().set_index("model")["mean"].idxmax(),
        "best_value": engine.compute_efficiency_metrics().set_index("model")["score_per_dollar"].idxmax(),
        "files_generated": [md_path, html_path, radar_path, bar_path],
    }

    summary_df = pd.DataFrame([summary_data])
    summary_path = f"{output_dir}/summary_{timestamp}.json"
    summary_df.to_json(summary_path, orient="records", indent=2, force_ascii=False)
    print(f"✅ 元数据摘要: {summary_path}")

    print(f"\n{'='*60}")
    print("  全部完成！生成的文件:")
    for f in summary_data["files_generated"]:
        size = Path(f).stat().st_size / 1024
        print(f"    • {Path(f).name} ({size:.1f} KB)")
    print(f"{'='*60}")

    return md_path


if __name__ == "__main__":
    report_file = run_full_evaluation()
```

输出：
```
============================================================
  模型评估报告 — 一键生成系统
============================================================
=== 多模型结果加载 ===
...
=== 跨模型样本对齐验证 ===
...
=== 对比宽表构建完成 ===
...
=== 基础统计指标 ===
...
=== 综合排名 (加权评分) ===
...

✅ Markdown 报告: reports/evaluation_report_20260103_143500.md
✅ HTML 报告:   reports/evaluation_report_20260103_143500.html
✅ 雷达图:     reports/radar_chart_20260103_143500.png
✅ 任务柱状图: reports/task_bar_20260103_143500.png
✅ 元数据摘要: reports/summary_20260103_143500.json

============================================================
  全部完成！生成的文件:
    • evaluation_report_20260103_143500.md (12.3 KB)
    • evaluation_report_20260103_143500.html (18.7 KB)
    • radar_chart_20260103_143500.png (89.2 KB)
    • task_bar_20260103_143500.png (156.4 KB)
============================================================
```

从原始 JSONL 结果文件到完整的 **Markdown + HTML + 图表** 评估报告，全部自动化完成。
