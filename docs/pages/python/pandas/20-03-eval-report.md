---
title: 项目二：LLM 评估报告自动生成系统
description: 多模型多基准评估 / 自动排名 / 对比矩阵 / 报告导出 HTML/Markdown
---
# 模型评估报告系统


## 项目概述

构建一个**自动化 LLM 模型评估报告生成器**，输入评估数据，输出完整的对比分析报告。

```python
import pandas as pd
import numpy as np
from datetime import datetime

class ModelEvaluationReporter:
    """LLM 模型评估报告生成器"""

    def __init__(self):
        self.raw_data = None
        self.processed = None
        self.report = None

    def load_eval_data(self, eval_df):
        """加载原始评估数据"""
        required = ['model', 'benchmark', 'score']
        missing = [c for c in required if c not in eval_df.columns]
        if missing:
            raise ValueError(f"缺少必要列: {missing}")

        self.raw_data = eval_df.copy()
        print(f"加载 {len(eval_df):,} 条评估记录, "
              f"{eval_df['model'].nunique()} 个模型, "
              f"{eval_df['benchmark'].nunique()} 个基准")
        return self

    def process(self):
        """处理数据：计算综合指标"""
        df = self.raw_data.copy()

        pivot = df.pivot_table(
            index='model',
            columns='benchmark',
            values='score',
            aggfunc='mean'
        ).round(2)

        bench_cols = [c for c in pivot.columns if c not in ['rank']]
        if len(bench_cols) >= 2:
            pivot['avg_score'] = pivot[bench_cols].mean(axis=1).round(2)

            rank_matrix = pivot[bench_cols].rank(ascending=False, method='dense').astype(int)
            for col in rank_matrix.columns:
                pivot[f'{col}_rank'] = rank_matrix[col]

        self.processed = pivot.sort_values('avg_score', ascending=False)
        self.processed['overall_rank'] = (
            self.processed['avg_score'].rank(ascending=False, method='dense').astype(int)
        )
        return self

    def generate_report(self):
        """生成完整报告"""
        if self.processed is None:
            raise ValueError("请先调用 process()")

        df = self.processed
        report_lines = []
        report_lines.append("# LLM 模型评估报告")
        report_lines.append(f"\n> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        report_lines.append(f"\n## 总览")
        report_lines.append(f"- **参与模型**: {len(df)} 个")
        report_lines.append(f"- **基准测试**: {len([c for c in df.columns if not c.endswith('_rank') and c != 'avg_score' and c != 'overall_rank'])} 项")
        report_lines.append(f"")

        report_lines.append("## 综合排行榜")
        report_lines.append("")
        report_lines.append("| 排名 | 模型 | 均分 |")
        report_lines.append("|------|------|------|")
        for _, row in df.iterrows():
            medal = {1:'🥇', 2:'🥈', 3:'🥉'}.get(row['overall_rank'], '')
            report_lines.append(
                f"| {row['overall_rank']} | **{row.name}** {medal} "
                f"| {row.get('avg_score', 'N/A')} |"
            )

        report_lines.append("\n## 各基准测试最佳模型")
        bench_cols = [c for c in df.columns if c not in ['avg_score', 'overall_rank']
                       and not c.endswith('_rank')]
        for col in bench_cols:
            best = df[col].idxmax()
            score = df.loc[best, col]
            report_lines.append(f"- **{col}**: {best} ({score})")

        report_lines.append("\n## 详细分数矩阵")
        display_cols = [c for c in df.columns if not c.endswith('_rank')]
        report_lines.append(df[display_cols].to_markdown())

        report_lines.append("\n## 关键发现")
        if 'avg_score' in df.columns:
            best_model = df['avg_score'].idxmax()
            worst_model = df['avg_score'].idxmin()
            gap = df.loc[best_model, 'avg_score'] - df.loc[worst_model, 'avg_score']
            report_lines.append(f"- **最强模型**: {best_model}")
            report_lines.append(f"- **分差最大**: {gap:.1f} 分 ({best_model} vs {worst_model})")

        self.report = '\n'.join(report_lines)
        print(self.report)
        return self.report

    def export_html(self, output_path):
        """导出为 HTML 格式"""
        import os
        if self.report is None:
            self.generate_report()

        html_content = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>LLM Evaluation Report</title>
<style>
body {{ font-family: -apple-system, sans-serif; max-width:900px; margin: 40px auto;
         line-height: 1.6; color: #333; }}
table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
th {{ background: #f5f5f5; font-weight: bold; }}
tr:nth-child(even) {{ background: #fafafa; }}
h1 {{ border-bottom: 3px solid #1976D2; padding-bottom: 10px; }}
h2 {{ color: #1976D2; margin-top: 30px; }}
</style></head><body>
{self._markdown_to_html(self.report)}
</body></html>"""

        with open(output_path, 'w') as f:
            f.write(html_content)

        size_kb = os.path.getsize(output_path) / 1024
        print(f"✓ HTML 报告已导出: {output_path} ({size_kb:.0f} KB)")
        return output_path

    @staticmethod
    def _markdown_to_html(md_text):
        import re
        md = md_text
        md = re.sub(r'^# (.+)$', r'<h1>\1</h1>', md, flags=re.M)
        md = re.sub(r'^## (.+)$', r'<h2>\1</h2>', md, flags=re.M)
        md = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', md)
        md = re.sub(r'^- (.+)$', r'<li>\1</li>', md, flags=re.M)
        md = re.sub(r'\| (.+) \|', lambda m: '<td>' + m.group(1).replace('|', '</td><td>') + '</td>', md)
        md = re.sub(r'^\|.*\|$', '', md, flags=re.M)
        return md


np.random.seed(42)
models = ['GPT-4o', 'Claude-3.5-Sonnet', 'Llama-3.1-70B',
          'Qwen2.5-72B', 'DeepSeek-V3']
benchmarks = ['MMLU', 'HumanEval', 'MATH', 'GPQA', 'BBH']

base_scores = {'GPT-4o': 88, 'Claude-3.5-Sonnet': 89,
                'Llama-3.1-70B': 84, 'Qwen2.5-72B': 83, 'DeepSeek-V3': 86}

raw_eval = []
for _ in range(150):
    model = np.random.choice(models)
    bm = np.random.choice(benchmarks)
    base = base_scores[model] + {'MATH': -8, 'HumanEval': 4}.get(bm, 0)
    raw_eval.append({
        'model': model,
        'benchmark': bm,
        'score': round(np.clip(base + np.random.randn() * 4, 30, 99), 1),
    })

eval_df = pd.DataFrame(raw_eval)

reporter = ModelEvaluationReporter()
reporter.load_eval_data(eval_df)
reporter.process()
reporter.generate_report()
```
