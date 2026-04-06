# 8.5 完整评估工作流与生产实践

## 从实验室到生产线：评估体系的最后一公里

前面四节我们分别学习了 RAG 评估的整体框架、检索质量评估、生成质量评估、以及调试与可观测性技术。这些知识像散落的珍珠，现在需要一根线把它们串成一条完整的项链。这一节要解决的核心问题是：**如何把评估能力从"开发时手动跑一次"升级为"贯穿整个生命周期的自动化体系"？**

想象一下这样的场景：你的企业知识库系统已经上线运行了三个月，某天产品经理告诉你最近用户反馈回答质量下降了。你该怎么办？手动挑几个问题测试一下？那太不靠谱了。正确的做法是：你有一个持续运行的评估管道（Pipeline），每天自动用最新的测试集跑一遍完整评估，生成报告推送到钉钉/Slack，并且一旦某个指标跌破阈值就立刻告警。这就是本节要构建的东西。

## 评估全生命周期：五个阶段

一个成熟的 RAG 评估体系不是一次性的事情，而是覆盖从开发到上线再到运维的完整生命周期。我们可以把它划分为五个阶段：

```
┌─────────────────────────────────────────────────────────────┐
│                    RAG 评估全生命周期                          │
│                                                             │
│  阶段1: 基线建立    阶段2: 迭代优化    阶段3: 回归防护       │
│  ┌───────────┐   ┌───────────┐   ┌───────────┐              │
│  │ 构建初始   │→→│ 每次改动   │→→│ 自动化回归 │              │
│  │ 测试集     │   │ 跑评估     │   │ 测试       │              │
│  │ 跑基线     │   │ 对比差异   │   │ 防止退化   │              │
│  └───────────┘   └───────────┘   └───────────┘              │
│        ↓               ↓               ↓                    │
│  阶段4: 生产监控    阶段5: 持续改进                            │
│  ┌───────────┐   ┌───────────┐                               │
│  │ 在线采样   │→→│ 定期回顾   │                               │
│  │ 实时反馈   │   │ 更新测试集 │                               │
│  │ 异常告警   │   │ 优化策略   │                               │
│  └───────────┘   └───────────┘                               │
└─────────────────────────────────────────────────────────────┘
```

### 阶段一：基线建立

基线建立是所有后续工作的参照系。没有基线，你就不知道"好"的标准是什么。

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.evaluation import (
    DatasetGenerator,
    RetrieverEvaluator,
    BatchEvalRunner,
    FaithfulnessEvaluator,
    RelevancyEvaluator,
)
from llama_index.llms.openai import OpenAI
import json
from pathlib import Path
from datetime import datetime


class BaselineBuilder:
    """构建评估基线的完整流程"""

    def __init__(self, data_dir: str, output_dir: str = "./eval_baselines"):
        self.data_dir = data_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build_baseline(self, num_questions: int = 50):
        """完整的基线建立流程"""
        print("=== 阶段1: 加载数据并构建索引 ===")
        documents = SimpleDirectoryReader(self.data_dir).load_data()
        index = VectorStoreIndex.from_documents(documents)
        query_engine = index.as_query_engine(similarity_top_k=3)

        print("=== 阶段2: 生成评估数据集 ===")
        dataset_generator = DatasetGenerator.from_documents(
            documents,
            llm=OpenAI(model="gpt-4o", temperature=0.7),
            num_questions_per_doc=3,
            question_gen_query=(
                "请根据以下文档内容，生成能够测试RAG系统能力的多样化问题。"
                "包括：事实性问题、推理问题、对比问题和细节追问。"
            ),
        )
        eval_questions = dataset_generator.generate_questions_from_nodes(
            num=num_questions
        )
        print(f"生成了 {len(eval_questions)} 个评估问题")

        print("=== 阶段3: 运行全面评估 ===")
        faithfulness_evaluator = FaithfulnessEvaluator(
            llm=OpenAI(model="gpt-4o")
        )
        relevancy_evaluator = RelevancyEvaluator(
            llm=OpenAI(model="gpt-4o")
        )

        runner = BatchEvalRunner(
            {
                "faithfulness": faithfulness_evaluator,
                "relevancy": relevancy_evaluator,
            },
            workers=8,
        )

        eval_results = await runner.aevaluate_queries(
            query_engine, queries=eval_questions
        )

        print("=== 阶段4: 计算检索指标 ===")
        retriever = index.as_retriever(similarity_top_k=5)
        retriever_evaluator = RetrieverEvaluator.from_metric_names(
            ["mrr", "hit_rate"], retriever=retriever
        )
        retrieval_results = await retriever_evaluator.aevaluate_dataset(
            eval_questions, show_progress=True
        )

        print("=== 阶段5: 保存基线结果 ===")
        baseline_data = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "data_dir": str(self.data_dir),
                "num_questions": num_questions,
                "similarity_top_k": 3,
                "llm_model": "gpt-4o",
            },
            "generation_metrics": {
                "faithfulness": eval_results["faithfulness"].get_average_score(),
                "relevancy": eval_results["relevancy"].get_average_score(),
            },
            "retrieval_metrics": {
                "mrr": retrieval_results.get_average_score("mrr"),
                "hit_rate": retrieval_results.get_average_score("hit_rate"),
            },
            "per_question_details": self._extract_per_question_details(
                eval_results, retrieval_results, eval_questions
            ),
        }

        baseline_file = self.output_dir / f"baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(baseline_file, "w", encoding="utf-8") as f:
            json.dump(baseline_data, f, ensure_ascii=False, indent=2, default=str)

        latest_file = self.output_dir / "baseline_latest.json"
        with open(latest_file, "w", encoding="utf-8") as f:
            json.dump(baseline_data, f, ensure_ascii=False, indent=2, default=str)

        self._print_summary(baseline_data)
        return baseline_data

    def _extract_per_question_details(self, gen_results, ret_results, questions):
        details = []
        for i, q in enumerate(questions):
            detail = {
                "question": q,
                "faithfulness": gen_results["faithfulness"].results[i].score if i < len(gen_results["faithfulness"].results) else None,
                "relevancy": gen_results["relevancy"].results[i].score if i < len(gen_results["relevancy"].results) else None,
                "mrr": ret_results.results[i].metrics.get("mrr") if i < len(ret_results.results) else None,
                "hit_rate": ret_results.results[i].metrics.get("hit_rate") if i < len(ret_results.results) else None,
            }
            details.append(detail)
        return details

    def _print_summary(self, baseline_data):
        print("\n" + "=" * 60)
        print(f"📊 基线评估完成！时间戳: {baseline_data['timestamp']}")
        print("=" * 60)
        print("\n【生成质量指标】")
        print(f"  忠实度 (Faithfulness): {baseline_data['generation_metrics']['faithfulness']:.4f}")
        print(f"  相关性 (Relevancy):     {baseline_data['generation_metrics']['relevancy']:.4f}")
        print("\n【检索质量指标】")
        print(f"  MRR:  {baseline_data['retrieval_metrics']['mrr']:.4f}")
        print(f"  Hit Rate: {baseline_data['retrieval_metrics']['hit_rate']:.4f}")
        print("=" * 60)


# 使用示例
builder = BaselineBuilder("./data/company_kb")
baseline = builder.build_baseline(num_questions=50)
```

这个 `BaselineBuilder` 类做了五件事：加载数据构建索引、用 LLM 自动生成评估问题、同时跑生成质量和检索质量的批量评估、保存结果到 JSON 文件、打印摘要报告。其中有一个容易被忽略的细节：**每个问题的详细分数也被保存了下来**，而不仅仅是平均值。为什么这很重要？因为平均值会掩盖个体差异——比如你可能整体 Faithfulness 是 0.85 看起来不错，但其中有 10% 的问题得分低于 0.5，这些"长尾坏案例"恰恰是最需要关注的。

### 阶段二：迭代优化中的对比评估

当你修改了 chunking 策略、换了 embedding 模型、或者调整了 reranker 的参数之后，你需要知道这些改动到底是变好了还是变坏了。这就需要做 **A/B 对比评估**：

```python
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from enum import Enum


class ComparisonResult(Enum):
    IMPROVED = "improved"
    DEGRADED = "degraded"
    NEUTRAL = "neutral"


@dataclass
class MetricDelta:
    metric_name: str
    old_value: float
    new_value: float
    delta: float
    delta_pct: float
    result: ComparisonResult
    significance: bool


class EvaluationComparator:
    """对比两次评估结果的工具"""

    def __init__(
        self,
        baseline_path: str,
        threshold_pct: float = 2.0,
        strict_mode: bool = False,
    ):
        self.baseline_path = Path(baseline_path)
        self.threshold_pct = threshold_pct
        self.strict_mode = strict_mode

    def load_baseline(self):
        with open(self.baseline_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def compare(self, new_results: dict, baseline: Optional[dict] = None) -> dict:
        if baseline is None:
            baseline = self.load_baseline()

        report = {
            "comparison_time": datetime.now().isoformat(),
            "baseline_timestamp": baseline.get("timestamp", "unknown"),
            "threshold_pct": self.threshold_pct,
            "summary": {"improved": [], "degraded": [], "neutral": []},
            "deltas": [],
            "verdict": None,
            "recommendations": [],
        }

        gen_deltas = self._compare_metric_group(
            baseline["generation_metrics"],
            new_results.get("generation_metrics", {}),
            "generation",
        )
        report["deltas"].extend(gen_deltas)

        ret_deltas = self._compare_metric_group(
            baseline["retrieval_metrics"],
            new_results.get("retrieval_metrics", {}),
            "retrieval",
        )
        report["deltas"].extend(ret_deltas)

        for d in report["deltas"]:
            if d.result == ComparisonResult.IMPROVED:
                report["summary"]["improved"].append(d.metric_name)
            elif d.result == ComparisonResult.DEGRADED:
                report["summary"]["degraded"].append(d.metric_name)
            else:
                report["summary"]["neutral"].append(d.metric_name)

        report["verdict"] = self._make_verdict(report)
        report["recommendations"] = self._generate_recommendations(report)

        self._print_comparison_report(report)
        return report

    def _compare_metric_group(
        self, old_metrics: dict, new_metrics: dict, group_name: str
    ) -> list:
        deltas = []
        for name, old_val in old_metrics.items():
            if name not in new_metrics:
                continue
            new_val = new_metrics[name]
            delta = new_val - old_val
            delta_pct = (delta / old_val * 100) if old_val != 0 else 0

            if abs(delta_pct) >= self.threshold_pct:
                if delta > 0:
                    result = ComparisonResult.IMPROVED
                    significance = True
                elif self.strict_mode:
                    result = ComparisonResult.DEGRADED
                    significance = True
                else:
                    result = ComparisonResult.DEGRADED
                    significance = True
            else:
                result = ComparisonResult.NEUTRAL
                significance = False

            deltas.append(
                MetricDelta(
                    metric_name=f"{group_name}.{name}",
                    old_value=old_val,
                    new_value=new_val,
                    delta=delta,
                    delta_pct=delta_pct,
                    result=result,
                    significance=significance,
                )
            )
        return deltas

    def _make_verdict(self, report: dict) -> str:
        degraded_count = len(report["summary"]["degraded"])
        improved_count = len(report["summary"]["improved"])

        if degraded_count == 0:
            if improved_count > 0:
                return "✅ PASS — 所有指标持平或提升，可以发布"
            else:
                return "⚪ NEUTRAL — 无显著变化，风险可控"
        elif degraded_count <= 1 and improved_count >= 2:
            return "⚠️ CONDITIONAL — 有少量退化但整体改善明显，建议人工审核后决定"
        else:
            return "❌ FAIL — 存在显著退化，不建议发布"

    def _generate_recommendations(self, report: dict) -> list:
        recommendations = []
        for d in report["deltas"]:
            if d.result == ComparisonResult.DEGRADED:
                if "faithfulness" in d.metric_name:
                    recommendations.append(
                        f"🔴 {d.metric_name} 下降了 {abs(d.delta_pct):.1f}% "
                        f"({d.old_value:.4f} → {d.new_value:.4f})。"
                        f"建议检查：(1) 检索到的上下文是否包含足够信息；"
                        f"(2) Prompt是否引导模型基于上下文作答；"
                        f"(3) 是否需要增加reranking来过滤噪声节点。"
                    )
                elif "relevancy" in d.metric_name:
                    recommendations.append(
                        f"🔴 {d.metric_name} 下降了 {abs(d.delta_pct):.1f}%。"
                        f"建议检查：(1) top_k是否过小导致相关内容被截断；"
                        f"(2) embedding模型是否匹配当前语言/领域；"
                        f"(3) 是否需要启用Hybrid Search补充关键词召回。"
                    )
                elif "mrr" in d.metric_name or "hit_rate" in d.metric_name:
                    recommendations.append(
                        f"🔴 {d.metric_name} 下降了 {abs(d.delta_pct):.1f}%。"
                        f"建议检查：(1) 数据是否有更新导致embedding过期；"
                        f"(2) chunk_size是否合适（过大降低精度）；"
                        f"(3) 是否有新的查询模式未被现有索引覆盖。"
                    )
            elif d.result == ComparisonResult.IMPROVED and d.significance:
                recommendations.append(
                    f"🟢 {d.metric_name} 提升了 {d.delta_pct:.1f}% "
                    f"({d.old_value:.4f} → {d.new_value:.4f}) ✨"
                )
        return recommendations

    def _print_comparison_report(self, report: dict):
        print("\n" + "=" * 70)
        print("📋 评估对比报告")
        print("=" * 70)
        print(f"\n基线时间: {report['baseline_timestamp']}")
        print(f"阈值设置: ±{self.threshold_pct}% 为显著变化\n")

        print(f"{'指标':<30} {'基线值':>10} {'新值':>10} {'变化率':>10} {'状态':>10}")
        print("-" * 70)
        for d in report["deltas"]:
            status_icon = {
                ComparisonResult.IMPROVED: "🟢↑",
                ComparisonResult.DEGRADED: "🔴↓",
                ComparisonResult.NEUTRAL: "⚪→",
            }[d.result]
            print(
                f"{d.metric_name:<30} {d.old_value:>10.4f} {d.new_value:>10.4f} "
                f"{d.delta_pct:>+9.1f}% {status_icon:>10}"
            )

        print("\n" + "-" * 70)
        print(f"\n📊 总体判定: {report['verdict']}")

        if report["recommendations"]:
            print("\n💡 建议:")
            for rec in report["recommendations"]:
                print(f"  {rec}")

        print("=" * 70)


# 使用示例
comparator = EvaluationComparator(
    baseline_path="./eval_baselines/baseline_latest.json",
    threshold_pct=2.0,
    strict_mode=True,
)
# report = comparator.compare(new_results)
```

这个对比器的核心设计思路是：**不只是告诉你是好是坏，还要告诉你具体哪里变了、变化了多少、以及应该怎么修**。注意其中的 `_make_verdict` 方法——它不是简单地看有没有退化的指标，而是做了分级判断：零退化直接通过、少量退化但大量提升则条件通过（需要人工审核）、多维度退化则拒绝。这种判断逻辑在实际工程中非常实用，因为现实中的改动往往是有得有失的，一刀切的"有任何退化就不让上线"会导致团队不敢做任何尝试。

### 阶段三：回归防护——CI/CD 集成

当评估能力成熟之后，下一步就是把它嵌入到 CI/CD 流水线中，实现每次代码提交或合并请求都自动跑评估。下面是一个完整的 GitHub Actions 工作流配置：

```yaml
# .github/workflows/rag-evaluation.yml
name: RAG Evaluation Pipeline

on:
  pull_request:
    branches: [main]
  push:
    branches: [main]
    paths:
      - "rag_pipeline/**"
      - "evaluation/**"
      - "data/**"

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: "pip"

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install llama-index llama-index-embeddings-openai
          pip install llama-index-readers-file llama-index-vector-stores-chroma
          pip install ragas openai datasets

      - name: Load environment variables
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: |
          echo "OPENAI_API_KEY=$OPENAI_API_KEY" >> $GITHUB_ENV

      - name: Run baseline comparison
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: |
          python evaluation/run_evaluation.py \
            --data-dir ./data/knowledge_base \
            --baseline ./eval_baselines/baseline_latest.json \
            --output ./eval_results \
            --threshold 2.0 \
            --strict

      - name: Upload evaluation results
        uses: actions/upload-artifact@v4
        with:
          name: evaluation-results
          path: ./eval_results/
          retention-days: 30

      - name: Comment PR with results
        if: github.event_name == 'pull_request'
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            const resultPath = './eval_results/comparison_report.json';

            if (!fs.existsSync(resultPath)) {
              console.log('No evaluation results found');
              return;
            }

            const report = JSON.parse(fs.readFileSync(resultPath, 'utf8'));
            let body = '## 📊 RAG 评估报告\n\n';
            body += `**基线**: ${report.baseline_timestamp}\n\n`;
            body += `### 总体判定: ${report.verdict}\n\n`;

            body += '| 指标 | 基线 | 当前 | 变化 | 状态 |\n';
            body += '|------|------|------|------|------|\n';

            for (const d of report.deltas) {
              const icon = d.result === 'improved' ? '🟢' :
                           d.result === 'degraded' ? '🔴' : '⚪';
              body += `| ${d.metric_name} | ${d.old_value} | ${d.new_value} | ${d.delta_pct > 0 ? '+' : ''}${d.delta_pct}% | ${icon} |\n`;
            }

            if (report.recommendations && report.recommendations.length > 0) {
              body += '\n### 💡 建议\n';
              for (const rec of report.recommendations) {
                body += `- ${rec}\n`;
              }
            }

            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: body
            });

      - name: Check evaluation gate
        id: check_gate
        run: |
          python -c "
          import json
          with open('./eval_results/comparison_report.json') as f:
              report = json.load(f)
          
          verdict = report.get('verdict', '')
          if 'FAIL' in verdict:
              print('::error::Evaluation FAILED - significant degradation detected')
              exit(1)
          elif 'CONDITIONAL' in verdict:
              print('::warning::Evaluation CONDITIONAL - manual review required')
              exit(0)
          else:
              print('Evaluation PASSED')
              exit(0)
          "

      - name: Notify on failure
        if: failure()
        run: |
          echo "RAG 评估未通过！请在 PR 中查看详细报告。"
          echo "如果这是预期内的改动，请联系团队负责人审批豁免。"
```

对应的 Python 入口脚本 `run_evaluation.py`：

```python
#!/usr/bin/env python3
"""CI/CD 评估流水线入口"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.evaluation import (
    DatasetGenerator,
    RetrieverEvaluator,
    BatchEvalRunner,
    FaithfulnessEvaluator,
    RelevancyEvaluator,
)
from llama_index.llms.openai import OpenAI


async def main():
    parser = argparse.ArgumentParser(description="RAG 评估流水线")
    parser.add_argument("--data-dir", required=True, help="数据目录")
    parser.add_argument("--baseline", required=True, help="基线文件路径")
    parser.add_argument("--output", default="./eval_results", help="输出目录")
    parser.add_argument("--threshold", type=float, default=2.0, help="显著性阈值(%)")
    parser.add_argument("--questions", type=int, default=50, help="评估问题数量")
    parser.add_argument("--strict", action="store_true", help="严格模式")

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] 加载数据: {args.data_dir}")
    documents = SimpleDirectoryReader(args.data_dir).load_data()

    print(f"[2/4] 构建索引...")
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine(similarity_top_k=3)

    print(f"[3/4] 运行评估 ({args.questions} 个问题)...")
    faithfulness_eval = FaithfulnessEvaluator(llm=OpenAI(model="gpt-4o"))
    relevancy_eval = RelevancyEvaluator(llm=OpenAI(model="gpt-4o"))

    runner = BatchEvalRunner(
        {"faithfulness": faithfulness_eval, "relevancy": relevancy_eval},
        workers=8,
    )

    question_file = Path(args.baseline).parent / "eval_questions.json"
    if question_file.exists():
        with open(question_file, "r", encoding="utf-8") as f:
            eval_questions = json.load(f)
        print(f"  使用已有问题集: {len(eval_questions)} 个问题")
    else:
        print("  生成新的评估问题集...")
        generator = DatasetGenerator.from_documents(
            documents, llm=OpenAI(model="gpt-4o", temperature=0.7)
        )
        eval_questions = generator.generate_from_nodes(num=args.questions)
        with open(question_file, "w", encoding="utf-8") as f:
            json.dump(eval_questions, f, ensure_ascii=False)

    eval_results = await runner.aevaluate_queries(query_engine, queries=eval_questions)

    retriever = index.as_retriever(similarity_top_k=5)
    ret_evaluator = RetrieverEvaluator.from_metric_names(
        ["mrr", "hit_rate"], retriever=retriever
    )
    ret_results = await ret_evaluator.aevaluate_dataset(eval_questions)

    new_results = {
        "timestamp": datetime.now().isoformat(),
        "generation_metrics": {
            "faithfulness": eval_results["faithfulness"].get_average_score(),
            "relevancy": eval_results["relevancy"].get_average_score(),
        },
        "retrieval_metrics": {
            "mrr": ret_results.get_average_score("mrr"),
            "hit_rate": ret_results.get_average_score("hit_rate"),
        },
    }

    print(f"[4/4] 对比基线...")
    from evaluation.comparator import EvaluationComparator
    comparator = EvaluationComparator(
        baseline_path=args.baseline,
        threshold_pct=args.threshold,
        strict_mode=args.strict,
    )
    report = comparator.compare(new_results)

    report_file = output_dir / "comparison_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)

    raw_file = output_dir / "raw_results.json"
    with open(raw_file, "w", encoding="utf-8") as f:
        json.dump(new_results, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n报告已保存至: {report_file}")

    if "FAIL" in report["verdict"]:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
```

这里有几个工程上的关键点值得展开讨论。首先是 **评估问题集的管理策略**：上面的代码中，我们第一次运行时会用 LLM 生成问题集并缓存到文件里，后续每次 CI 跑的时候复用同一份问题集。这样做的原因是：如果每次都用 LLM 重新生成问题，那么两次评估之间的差异可能来自"问题不同"而不是"系统变化"，这会让对比失去意义。当然，问题集本身也需要定期更新——比如每两周人工审核一次，加入新出现的问题类型。

其次是 **退出码的设计**：评估脚本通过 `sys.exit(1)` 在 FAIL 时返回非零退出码，这样 GitHub Actions 的后续 step 就能感知到失败并触发通知。而 CONDITIONAL 的情况返回 0 但打印 warning，这意味着不会阻断 CI 但会在日志中留下痕迹。

第三个容易踩坑的地方是 **CI 环境中的 API 调用量控制**。上面这个流水线每次 PR 都会调用 OpenAI API 跑 50 个问题的评估，假设每个问题需要 2 次 LLM 调用（faithfulness + relevancy 各一次），那就是 100 次额外的 GPT-4o 调用。如果你的团队很活跃，一天可能有几十个 PR，API 费用会快速累积。解决方案包括：（1）只对 `main` 分支的推送跑完整评估，PR 上只跑轻量级抽样（比如 10 个问题）；（2）使用更便宜的模型（如 gpt-4o-mini）跑评估；（3）对同一个 PR 的重复提交做去重，只在最新 commit 上跑。

### 阶段四：生产环境监控

系统上线之后，评估并没有结束——相反，真正的挑战才刚刚开始。生产环境的评估核心难点在于：**你没有标准答案可以对比**。在开发阶段你可以精心编写 reference answer，但在生产中用户的提问千奇百怪，不可能预先准备好答案。所以生产监控需要换一套方法论。

#### 在线评估策略一：用户隐式反馈信号

最简单也最实用的在线评估方法是收集用户的隐式反馈——用户虽然不会明确告诉你"这个回答好不好"，但他们的行为会透露线索：

```python
from dataclasses import dataclass
from typing import Optional
from collections import defaultdict
from datetime import datetime, timedelta
import statistics
import json
from pathlib import Path


@dataclass
class QueryRecord:
    query_id: str
    query_text: str
    response_text: str
    source_nodes_count: int
    response_latency_ms: float
    timestamp: datetime
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    copied: bool = False
    liked: Optional[bool] = None
    follow_up_asked: bool = False
    rephrased: bool = False
    abandoned: bool = False
    feedback_delay_s: float = 0.0


class ProductionMonitor:
    """生产环境 RAG 质量监控器"""

    def __init__(self, window_minutes: int = 60):
        self.window_minutes = window_minutes
        self.records: list[QueryRecord] = []
        self.alert_thresholds = {
            "avg_latency_ms": 5000,
            "copy_rate": 0.15,
            "abandon_rate": 0.40,
            "rephrase_rate": 0.20,
        }

    def record_query(self, record: QueryRecord):
        self.records.append(record)
        cutoff = datetime.now() - timedelta(minutes=self.window_minutes)
        self.records = [r for r in self.records if r.timestamp >= cutoff]

    def get_health_report(self) -> dict:
        if not self.records:
            return {"status": "no_data", "message": "暂无数据"}

        window_start = min(r.timestamp for r in self.records)
        total = len(self.records)

        copy_count = sum(1 for r in self.records if r.copied)
        abandon_count = sum(1 for r in self.records if r.abandoned)
        rephrase_count = sum(1 for r in self.records if r.rephrased)
        follow_up_count = sum(1 for r in self.records if r.follow_up_asked)

        latencies = [r.response_latency_ms for r in self.records]

        report = {
            "status": "healthy",
            "window": f"{window_start.strftime('%H:%M')} ~ {datetime.now().strftime('%H:%M')}",
            "total_queries": total,
            "metrics": {
                "avg_latency_ms": statistics.mean(latencies),
                "p50_latency_ms": statistics.median(latencies),
                p95_key: sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0,
                p99_key: sorted(latencies)[int(len(latencies) * 0.99)] if latencies else 0,
                "copy_rate": copy_count / total,
                "abandon_rate": abandon_count / total,
                "rephrase_rate": rephrase_count / total,
                "follow_up_rate": follow_up_count / total,
            },
            "alerts": [],
            "trends": self._compute_trends(),
        }

        metrics = report["metrics"]
        if metrics["avg_latency_ms"] > self.alert_thresholds["avg_latency_ms"]:
            report["alerts"].append({
                "level": "warning",
                "metric": "avg_latency_ms",
                "value": metrics["avg_latency_ms"],
                "threshold": self.alert_thresholds["avg_latency_ms"],
                "message": f"平均响应延迟 {metrics['avg_latency_ms']:.0f}ms 超过阈值",
            })

        if metrics["copy_rate"] < self.alert_thresholds["copy_rate"]:
            report["alerts"].append({
                "level": "warning",
                "metric": "copy_rate",
                "value": metrics["copy_rate"],
                "message": f"复制率 {metrics['copy_rate']*100:.1f}% 偏低，回答质量可能下降",
            })

        if metrics["abandon_rate"] > self.alert_thresholds["abandon_rate"]:
            report["alerts"].append({
                "level": "critical",
                "metric": "abandon_rate",
                "value": metrics["abandon_rate"],
                "message": f"放弃率 {metrics['abandon_rate']*100:.1f}% 过高",
            })

        if metrics["rephrase_rate"] > self.alert_thresholds["rephrase_rate"]:
            report["alerts"].append({
                "level": "warning",
                "metric": "rephrase_rate",
                "value": metrics["rephrase_rate"],
                "message": f"改写重问率 {metrics['rephrase_rate']*100:.1f}% 偏高",
            })

        if report["alerts"]:
            max_level = max(a["level"] for a in report["alerts"])
            report["status"] = "critical" if max_level == "critical" else "degraded"

        return report

    def _compute_trends(self) -> dict:
        half = len(self.records) // 2
        if half < 10:
            return {"status": "insufficient_data"}

        first_half = self.records[:half]
        second_half = self.records[half:]
        trends = {}

        old_lat = statistics.mean(r.response_latency_ms for r in first_half)
        new_lat = statistics.mean(r.response_latency_ms for r in second_half)
        change_pct = ((new_lat - old_lat) / old_lat * 100) if old_lat > 0 else 0
        trends["latency"] = {"change_pct": change_pct, "direction": "↑" if change_pct > 3 else ("↓" if change_pct < -3 else "→")}

        old_copy = sum(1 for r in first_half if r.copied) / len(first_half)
        new_copy = sum(1 for r in second_half if r.copied) / len(second_half)
        copy_change = ((new_copy - old_copy) / old_copy * 100) if old_copy > 0 else 0
        trends["copy"] = {"change_pct": copy_change, "direction": "↑" if copy_change > 3 else ("↓" if copy_change < -3 else "→")}

        old_abandon = sum(1 for r in first_half if r.abandoned) / len(first_half)
        new_abandon = sum(1 for r in second_half if r.abandoned) / len(second_half)
        ab_change = ((new_abandon - old_abandon) / old_abandon * 100) if old_abandon > 0 else 0
        trends["abandon"] = {"change_pct": ab_change, "direction": "↑" if ab_change > 3 else ("↓" if ab_change < -3 else "→")}

        return trends

    def export_bad_cases(self, top_n: int = 20) -> list[dict]:
        scored = []
        for r in self.records:
            score = 0.0
            if r.abandoned and not r.copied:
                score += 3.0
            if r.rephrased:
                score += 2.0
            if r.response_latency_ms > 8000:
                score += 1.0
            if r.liked is False:
                score += 3.0
            scored.append((score, r))

        scored.sort(key=lambda x: x[0], reverse=True)
        bad_cases = []
        for score, r in scored[:top_n]:
            bad_cases.append({
                "risk_score": round(score, 2),
                "query": r.query_text,
                "response_preview": r.response_text[:200] + "...",
                "signals": {
                    "copied": r.copied,
                    "liked": r.liked,
                    "abandoned": r.abandoned,
                    "rephrased": r.rephrased,
                    "latency_ms": r.response_latency_ms,
                },
                "timestamp": r.timestamp.isoformat(),
            })
        return bad_cases


p95_key = "p95_latency_ms"
p99_key = "p99_latency_ms"


class MonitoredQueryEngine:
    """带监控功能的查询引擎包装器"""

    def __init__(self, query_engine, monitor: ProductionMonitor):
        self.query_engine = query_engine
        self.monitor = monitor
        self._id_counter = 0

    def query(self, query_str: str, user_id: str = None, session_id: str = None):
        import time
        start = time.perf_counter()
        response = self.query_engine.query(query_str)
        latency_ms = (time.perf_counter() - start) * 1000

        record = QueryRecord(
            query_id=f"q_{self._id_counter}",
            query_text=query_str,
            response_text=response.response,
            source_nodes_count=len(response.source_nodes),
            response_latency_ms=latency_ms,
            timestamp=datetime.now(),
            user_id=user_id,
            session_id=session_id,
        )
        self._id_counter += 1
        self.monitor.record_query(record)
        response.metadata["_monitor_record_id"] = record.query_id
        return response

    def record_feedback(self, record_id: str, **kwargs):
        for r in self.monitor.records:
            if r.query_id == record_id:
                for k, v in kwargs.items():
                    if hasattr(r, k):
                        setattr(r, k, v)
                break
```

这套隐式反馈系统的设计哲学是：**不依赖用户的主动评价行为（因为大多数用户懒得点好评/差评），而是从用户的行为模式中推断满意度**。其中几个信号的解读逻辑值得仔细说明：

- **复制率（Copy Rate）**：如果用户复制了回答的内容，这是一个强正信号——说明回答中有用户认为有价值的信息。一般来说，知识库问答场景的复制率应该在 20%-40% 之间，低于 15% 就需要警惕。
- **放弃率（Abandon Rate）**：用户收到回答后没有任何后续操作（不复制、不追问、不改写），直接离开了。这通常意味着回答完全没用。
- **改写重问率（Rephrase Rate）**：用户收到回答后换了一种方式再问同样的问题。这说明首次回答可能方向对了但不够准确，或者是用户觉得系统没理解他的意思。
- **响应延迟的 P99**：P99 延迟比平均延迟更重要，因为那代表了最慢的那 1% 用户体验。如果 P99 超过 10 秒，即使平均只有 2 秒，也会有相当比例的用户感到不耐烦。

#### 在线评估策略二：LLM-as-Judge 自动打分

对于重要的客户（比如 VIP 企业用户），隐式反馈可能不够精细。这时可以用 LLM-as-Judge 方法，在生产环境中对部分查询做实时质量打分：

```python
import asyncio
from typing import Optional
from dataclasses import dataclass
from llama_index.llms.openai import OpenAI
from llama_index.core.prompts import PromptTemplate


@dataclass
class QualityScore:
    query_id: str
    overall_score: float
    faithfulness: float
    relevance: float
    completeness: float
    reasoning: str
    improvement_suggestion: str


JUDGE_PROMPT = PromptTemplate(
    """你是一个专业的 RAG 系统质量评审员。请对以下问答进行评分。

## 用户问题
{query}

## 系统回答
{response}

## 检索到的参考上下文
{context}

## 评分标准（每个维度1-5分）
- 忠实度(Faithfulness)：回答是否严格基于提供的上下文，有无幻觉
- 相关性(Relevance)：回答是否针对用户问题，有无偏题
- 完整性(Completeness)：回答是否充分回应了问题的各个方面
- 综合质量(Overall)：综合考虑以上维度的整体质量

请以JSON格式输出评分结果：
```json
{
  "overall_score": <1-5>,
  "faithfulness": <1-5>,
  "relevance": <1-5>,
  "completeness": <1-5>,
  "reasoning": "<简要说明评分理由>",
  "improvement_suggestion": "<如果质量不佳，给出改进建议>"
}
```
"""
)


class OnlineQualityJudge:
    """在线质量评判器（异步，不阻塞主流程）"""

    def __init__(
        self,
        llm: OpenAI = None,
        sample_rate: float = 0.1,
        batch_size: int = 5,
    ):
        self.llm = llm or OpenAI(model="gpt-4o-mini")
        self.sample_rate = sample_rate
        self.batch_size = batch_size
        self.pending: list[dict] = []
        self.scores: list[QualityScore] = []

    def should_judge(self) -> bool:
        import random
        return random.random() < self.sample_rate

    async def submit_for_judging(
        self,
        query_id: str,
        query_text: str,
        response_text: str,
        context_texts: list[str],
    ):
        context = "\n---\n".join(context_texts[:3])
        self.pending.append({
            "query_id": query_id,
            "query_text": query_text,
            "response_text": response_text,
            "context": context,
        })

        if len(self.pending) >= self.batch_size:
            await self._process_batch()

    async def _process_batch(self):
        if not self.pending:
            return

        batch = self.pending[:self.batch_size]
        self.pending = self.pending[self.batch_size:]

        tasks = [self._judge_one(item) for item in batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, QualityScore):
                self.scores.append(result)
            else:
                print(f"评判失败: {result}")

    async def _judge_one(self, item: dict) -> QualityScore:
        prompt = JUDGE_PROMPT.format(
            query=item["query_text"],
            response=item["response_text"],
            context=item["context"],
        )

        response = await self.llm.acomplete(prompt)
        try:
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', response.text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(1))
            else:
                data = json.loads(response.text.strip())

            return QualityScore(
                query_id=item["query_id"],
                overall_score=data.get("overall_score", 0),
                faithfulness=data.get("faithfulness", 0),
                relevance=data.get("relevance", 0),
                completeness=data.get("completeness", 0),
                reasoning=data.get("reasoning", ""),
                improvement_suggestion=data.get("improvement_suggestion", ""),
            )
        except (json.JSONDecodeError, KeyError) as e:
            print(f"解析评判结果失败: {e}, 原始文本: {response.text[:200]}")
            return QualityScore(
                query_id=item["query_id"],
                overall_score=0, faithfulness=0, relevance=0,
                completeness=0, reasoning=f"Parse error: {e}",
                improvement_suggestion="",
            )

    def get_quality_stats(self, hours: int = 1) -> dict:
        from datetime import timedelta
        cutoff = datetime.now() - timedelta(hours=hours)
        recent = [s for s in self.scores]

        if not recent:
            return {"status": "no_recent_scores"}

        return {
            "total_judged": len(recent),
            "avg_overall": statistics.mean(s.overall_score for s in recent),
            "avg_faithfulness": statistics.mean(s.faithfulness for s in recent),
            "avg_relevance": statistics.mean(s.relevance for s in recent),
            "score_distribution": self._count_distribution(s.overall_score for s in recent),
            "common_issues": self._extract_common_issues(recent),
        }

    def _count_distribution(self, scores) -> dict:
        dist = {"5(优秀)": 0, "4(良好)": 0, "3(一般)": 0, "2(较差)": 0, "1(很差)": 0}
        for s in scores:
            bucket = int(s)
            key = list(dist.keys())[min(bucket - 1, 4)]
            dist[key] += 1
        return dist

    def _extract_common_issues(self, scores: list) -> list[str]:
        issues = []
        low_faith = sum(1 for s in scores if s.faithfulness <= 2)
        low_rel = sum(1 for s in scores if s.relevance <= 2)
        total = len(scores)

        if low_faith / total > 0.2:
            issues.append(f"忠实度偏低 ({low_faith}/{total})，可能存在幻觉问题")
        if low_rel / total > 0.2:
            issues.append(f"相关性偏低 ({low_rel}/{total})，检索或理解可能偏离主题")
        return issues
```

这个在线评判器的设计要点是：**抽样而非全量**（用 `sample_rate=0.1` 只评判 10% 的查询控制成本）、**批量处理**（攒够一批再发送减少 API 调用次数）、**异步非阻塞**（评判操作不影响主查询流程的延迟）、**用便宜模型**（评判用的是 gpt-4o-mini 而不是 gpt-4o，因为评判任务不需要最强的推理能力）。这里有一个实际部署时需要注意的问题：评判结果的解析要足够鲁棒——LLM 输出的 JSON 格式不一定完全规范，所以代码中用了正则来提取 ```json 代码块中的内容作为第一选择，再尝试直接 parse 整体文本作为 fallback。

### 阶段五：持续改进闭环

评估的最终目的不是为了得到一个漂亮的分数，而是为了驱动系统持续改进。下面是一个完整的持续改进工作流的实现：

```python
@dataclass
class ImprovementAction:
    id: str
    category: str
    title: str
    description: str
    priority: int
    estimated_impact: str
    status: str
    evidence: list[str]
    created_at: datetime
    completed_at: Optional[datetime] = None
    result_metrics: Optional[dict] = None


class ContinuousImprovementLoop:
    """RAG 系统持续改进闭环管理器"""

    def __init__(self, monitor: ProductionMonitor, judge: OnlineQualityJudge):
        self.monitor = monitor
        self.judge = judge
        self.actions: list[ImprovementAction] = []
        self.history: list[dict] = []

    def weekly_review(self) -> dict:
        health = self.monitor.get_health_report()
        bad_cases = self.monitor.export_bad_cases(top_n=30)
        quality_stats = self.judge.get_quality_stats(hours=168)

        review_input = {
            "review_date": datetime.now().isoformat(),
            "health_status": health["status"],
            "key_metrics": health["metrics"],
            "alerts": health["alerts"],
            "top_bad_cases": bad_cases[:10],
            "quality_stats": quality_stats,
            "proposed_actions": self._generate_proposals(health, bad_cases, quality_stats),
            "previous_actions_status": self._summarize_previous_actions(),
        }
        return review_input

    def _generate_proposals(self, health, bad_cases, quality_stats) -> list:
        proposals = []
        action_id = len(self.actions) + 1

        if health["metrics"]["abandon_rate"] > 0.35:
            proposals.append(ImprovementAction(
                id=f"A{action_id:03d}",
                category="retrieval",
                title="引入 Hybrid Search 提升召回率",
                description=(
                    "当前纯向量检索在高放弃率场景下表现不足，"
                    "建议引入 BM25 关键词检索与向量检索融合，"
                    "预计可将 hit_rate 提升 10-15%。"
                ),
                priority=4,
                estimated_impact="high",
                status="proposed",
                evidence=[
                    f"放弃率 {health['metrics']['abandon_rate']*100:.1f}% 超过阈值",
                ],
                created_at=datetime.now(),
            ))
            action_id += 1

        if health["metrics"].get(p99_key, 0) > 10000:
            proposals.append(ImprovementAction(
                id=f"A{action_id:03d}",
                category="infrastructure",
                title="优化查询链路延迟",
                description=(
                    f"P99 延迟达 {health['metrics'].get(p99_key, 0)/1000:.1f}s，"
                    "主要瓶颈可能在 embedding 推理、LLM 生成或向量数据库查询。"
                    "建议逐段 profiling 后针对性优化。"
                ),
                priority=3,
                estimated_impact="medium",
                status="proposed",
                evidence=[f"P99={health['metrics'].get(p99_key, 0):.0f}ms"],
                created_at=datetime.now(),
            ))
            action_id += 1

        if quality_stats.get("avg_faithfulness", 5) < 3.5:
            proposals.append(ImprovementAction(
                id=f"A{action_id:03d}",
                category="synthesis",
                title="增强幻觉检测与约束",
                description=(
                    "LLM 评判显示忠实度偏低，可能存在模型编造上下文中不存在的信息。"
                    "建议：(1) 在 system prompt 中强化基于上下文回答约束；"
                    "(2) 启用 FaithfulnessEvaluator 做后处理过滤；"
                    "(3) 考虑切换到 REFINE 合成模式。"
                ),
                priority=5,
                estimated_impact="high",
                status="proposed",
                evidence=[f"平均忠实度: {quality_stats.get('avg_faithfulness', 0):.2f}/5"],
                created_at=datetime.now(),
            ))
            action_id += 1

        if health["metrics"]["rephrase_rate"] > 0.15:
            proposals.append(ImprovementAction(
                id=f"A{action_id:03d}",
                category="retrieval",
                title="引入 HyDE 或 Query Rewriting",
                description=(
                    f"改写重问率达 {health['metrics']['rephrase_rate']*100:.1f}%，"
                    "说明首次检索的语义匹配不够精准。建议启用 HyDEQueryTransform "
                    "或 DecomposeQueryTransform 来改善查询理解。"
                ),
                priority=3,
                estimated_impact="medium",
                status="proposed",
                evidence=[f"重问率: {health['metrics']['rephrase_rate']*100:.1f}%"],
            ))

        self.actions.extend(proposals)
        return proposals

    def _summarize_previous_actions(self) -> list[dict]:
        summary = []
        completed = [a for a in self.actions if a.status == "completed"]
        in_progress = [a for a in self.actions if a.status == "in_progress"]

        if completed:
            summary.append({"status": "completed", "count": len(completed)})
        if in_progress:
            summary.append({
                "status": "in_progress",
                "count": len(in_progress),
                "items": [{"id": a.id, "title": a.title} for a in in_progress],
            })
        return summary

    def complete_action(self, action_id: str, result_metrics: dict = None):
        for a in self.actions:
            if a.id == action_id:
                a.status = "completed"
                a.completed_at = datetime.now()
                a.result_metrics = result_metrics
                self.history.append({
                    "action_id": action_id,
                    "title": a.title,
                    "completed_at": a.completed_at.isoformat(),
                    "result": result_metrics,
                })
                break

    def generate_weekly_report(self) -> str:
        review = self.weekly_review()
        lines = [
            "# RAG 系统周报",
            f"> 生成时间: {review['review_date']}",
            "",
            "## 📊 系统健康状态",
            f"- 整体状态: **{review['health_status'].upper()}**",
            f"- 本周总查询量: **{review['key_metrics'].get('total_queries', 'N/A')}**",
            f"- 平均延迟: **{review['key_metrics'].get('avg_latency_ms', 0):.0f}ms**",
            f"- 复制率: **{review['key_metrics'].get('copy_rate', 0)*100:.1f}%**",
            f"- 放弃率: **{review['key_metrics'].get('abandon_rate', 0)*100:.1f}%**",
            "",
        ]

        if review["alerts"]:
            lines.append("## 🚨 本周告警")
            for alert in review["alerts"]:
                icon = "🔴" if alert["level"] == "critical" else "⚠️"
                lines.append(f"- {icon} [{alert['metric']}] {alert['message']}")
            lines.append("")

        if review["proposed_actions"]:
            lines.append("## 📋 建议改进措施")
            for action in review["proposed_actions"]:
                stars = "⭐" * action.priority
                lines.append(f"### {action.id} {action.title} {stars}")
                lines.append(f"- 类别: {action.category}")
                lines.append(f"- 预期影响: {action.estimated_impact}")
                lines.append(f"- 依据: {'; '.join(action.evidence[:2])}")
                lines.append("")

        if review.get("previous_actions_status"):
            lines.append("## ✅ 历史措施进展")
            for s in review["previous_actions_status"]:
                tag = s.get("count", "")
                lines.append(f"- **{s['status']}**: {tag} 项")
            lines.append("")

        return "\n".join(lines)
```

`ContinuousImprovementLoop` 是整个评估体系的"大脑"——它把监控数据、评判数据和历史记录整合在一起，自动生成改进提案并追踪执行状态。其中的 `_generate_proposals` 方法展示了一种基于规则的简单决策逻辑：当某个监控指标超过预设阈值时，自动创建对应的改进工单。当然，在生产环境中你可能希望用 LLM 来做更智能的提案生成（让 GPT-4o 分析所有数据然后给出建议），但规则方式的好处是可解释性强、不会产生奇怪的提案、且运行成本几乎为零。

## 评估仪表盘：一站式可视化

有了各种评估数据和监控指标之后，最后一步是把它们整合到一个可视化的仪表盘中。下面是一个基于 FastAPI + Chart.js 的评估仪表盘实现：

```python
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import json


app = FastAPI(title="RAG Evaluation Dashboard")

monitor_instance = None
judge_instance = None
loop_instance = None


def get_dashboard_html(data: dict) -> str:
    quality_dist_json = json.dumps(list(data.get("quality_dist", {}).keys()))
    quality_vals_json = json.dumps(list(data.get("quality_dist", {}).values()))
    total_q = data["health"]["metrics"].get("total_queries", 0)
    abandon_r = data["health"]["metrics"].get("abandon_rate", 0)
    copy_r = data["health"]["metrics"].get("copy_rate", 0)

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG 评估仪表盘</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body class="bg-gray-900 text-gray-100 min-h-screen">
    <div class="container mx-auto px-4 py-8">
        <div class="flex justify-between items-center mb-8">
            <h1 class="text-3xl font-bold">🔍 RAG 系统评估仪表盘</h1>
            <div id="status-badge" class="px-4 py-2 rounded-full text-sm font-semibold
                {'bg-green-500' if data['health']['status'] == 'healthy' else
                 'bg-yellow-500' if data['health']['status'] == 'degraded' else 'bg-red-500'}">
                {data['health']['status'].upper()}
            </div>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            <div class="bg-gray-800 rounded-xl p-6 border border-gray-700">
                <div class="text-gray-400 text-sm mb-1">总查询量</div>
                <div class="text-3xl font-bold">{total_q}</div>
                <div class="text-xs text-gray-500 mt-1">窗口: {data['health'].get('window', '-')}</div>
            </div>
            <div class="bg-gray-800 rounded-xl p-6 border border-gray-700">
                <div class="text-gray-400 text-sm mb-1">平均延迟</div>
                <div class="text-3xl font-bold">{data['health']['metrics'].get('avg_latency_ms', 0):.0f}<span class="text-lg">ms</span></div>
                <div class="text-xs text-gray-500 mt-1">P99: {data['health']['metrics'].get(p99_key, 0):.0f}ms</div>
            </div>
            <div class="bg-gray-800 rounded-xl p-6 border border-gray-700">
                <div class="text-gray-400 text-sm mb-1">复制率</div>
                <div class="text-3xl font-bold">{copy_r*100:.1f}<span class="text-lg">%</span></div>
                <div class="text-xs mt-1">{'📈' if copy_r > 0.2 else '📉'}</div>
            </div>
            <div class="bg-gray-800 rounded-xl p-6 border border-gray-700">
                <div class="text-gray-400 text-sm mb-1">放弃率</div>
                <div class="text-3xl font-bold">{abandon_r*100:.1f}<span class="text-lg">%</span></div>
                <div class="text-xs mt-1">阈值: 40%</div>
            </div>
        </div>

        {'<div class="bg-red-900/30 border border-red-700 rounded-xl p-6 mb-8">' if data['health']['alerts'] else '<div class="hidden">'}
            <h2 class="text-lg font-semibold text-red-400 mb-3">🚨 活跃告警</h2>
            {"".join([f'''
            <div class="flex items-start gap-3 py-2 border-b border-red-800 last:border-0">
                <span class="{'text-red-400' if a['level']=='critical' else 'text-yellow-400'}">●</span>
                <div>
                    <div class="font-medium">{a['metric']}</div>
                    <div class="text-sm text-gray-400">{a['message']}</div>
                </div>
            </div>
            ''' for a in data['health']['alerts']])}
        </div>

        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
            <div class="bg-gray-800 rounded-xl p-6 border border-gray-700">
                <h2 class="text-lg font-semibold mb-4">质量评分分布</h2>
                <canvas id="qualityChart" height="200"></canvas>
            </div>
            <div class="bg-gray-800 rounded-xl p-6 border border-gray-700">
                <h2 class="text-lg font-semibold mb-4">用户行为漏斗</h2>
                <canvas id="funnelChart" height="200"></canvas>
            </div>
        </div>

        <div class="bg-gray-800 rounded-xl p-6 border border-gray-700">
            <h2 class="text-lg font-semibold mb-4">⚠️ 高风险案例 TOP 10</h2>
            <div class="overflow-x-auto">
                <table class="w-full text-sm">
                    <thead><tr class="text-left text-gray-400 border-b border-gray-700">
                        <th class="pb-2">风险分</th><th class="pb-2">问题</th>
                        <th class="pb-2">信号</th><th class="pb-2">时间</th>
                    </tr></thead>
                    <tbody>
                        {"".join([f'''
                        <tr class="border-b border-gray-700/50">
                            <td class="py-2 font-mono {'text-red-400' if c['risk_score'] >= 3 else 'text-yellow-400'}">{c['risk_score']}</td>
                            <td class="py-2 max-w-md truncate">{c['query'][:60]}</td>
                            <td class="py-2 text-xs">
                                {'✓复制' if c['signals']['copied'] else '✗'} {'✗放弃' if c['signals']['abandoned'] else ' '}
                                {'↻重问' if c['signals']['rephrased'] else ' '}
                            </td>
                            <td class="py-2 text-gray-500">{c['timestamp'][11:16]}</td>
                        </tr>
                        ''' for c in data.get('bad_cases', [])[:10]])}
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <script>
        new Chart(document.getElementById('qualityChart'), {{
            type: 'doughnut',
            data: {{
                labels: {quality_dist_json},
                datasets: [{{
                    data: {quality_vals_json},
                    backgroundColor: ['#22c55e', '#84cc16', '#eab308', '#f97316', '#ef4444']
                }}]
            }},
            options: {{ responsive: true, plugins: {{ legend: {{ position: 'bottom' }} }} }}
        }});

        new Chart(document.getElementById('funnelChart'), {{
            type: 'bar',
            data: {{
                labels: ['总查询', '有交互', '复制', '满意'],
                datasets: [{{
                    data: [
                        {total_q},
                        {int(total_q * (1 - abandon_r))},
                        {int(total_q * copy_r)},
                        {int(total_q * copy_r * 0.7)}
                    ],
                    backgroundColor: ['#6366f1', '#8b5cf6', '#a855f7', '#d946ef']
                }}]
            }},
            options: {{ responsive: true, indexAxis: 'y', plugins: {{ legend: {{ display: false }} }} }}
        }});
    </script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    if not monitor_instance:
        raise HTTPException(status_code=503, detail="Monitor not initialized")
    health = monitor_instance.get_health_report()
    bad_cases = monitor_instance.export_bad_cases(top_n=20)
    quality_stats = judge_instance.get_quality_stats(hours=1) if judge_instance else {}
    data = {
        "health": health,
        "bad_cases": bad_cases,
        "quality_dist": quality_stats.get("score_distribution", {}),
    }
    return get_dashboard_html(data)


@app.get("/api/health")
async def api_health():
    if not monitor_instance:
        raise HTTPException(status_code=503, detail="Monitor not initialized")
    return monitor_instance.get_health_report()


@app.get("/api/bad-cases")
async def api_bad_cases(limit: int = 20):
    if not monitor_instance:
        raise HTTPException(status_code=503, detail="Monitor not initialized")
    return monitor_instance.export_bad_cases(top_n=limit)


@app.get("/api/weekly-report")
async def api_weekly_report():
    if not loop_instance:
        raise HTTPException(status_code=503, detail="Improvement loop not initialized")
    return {"report": loop_instance.generate_weekly_report()}
```

启动仪表盘服务后，你就可以在浏览器中看到一个实时的 RAG 系统健康面板，包含核心指标卡片、活跃告警、质量分布饼图、用户行为漏斗图和高风险案例列表。这个仪表盘可以作为团队每日站会的讨论基础，也可以投屏到大屏幕上作为团队的"质量温度计"。FastAPI 提供的同时还有 `/api/health`、`/api/bad-cases`、`/api/weekly-report` 等 RESTful API 端点，方便与其他运维系统集成（比如 Prometheus 抓取、Grafana 展示、钉钉机器人推送等）。

## 常见误区与避坑指南

在搭建和使用 RAG 评估体系的过程中，有几个特别常见的错误值得单独提出来：

### 误区一：只看平均值不看分布

这是新手最容易犯的错误。假设你有 50 个测试问题，Faithfulness 平均分 0.90，看起来很棒对吧？但如果其中 5 个问题的得分是 0.0（完全胡说八道），另外 45 个是 1.0（完美），平均下来还是 0.90。这 5 个零分问题可能恰好是你最重要的业务场景（比如产品定价咨询），但你被平均值蒙蔽了双眼。

**正确做法**：始终关注指标的分布情况，特别是 P10 和 P90 分位数。在 `BaselineBuilder` 中我们已经保存了逐题详情，记得用它来做分布分析而不是只看均值。一个实用的技巧是：把所有问题的得分按从低到高排序画出来，如果曲线的左尾部有明显拖尾，那就说明存在系统性短板需要优先解决。

### 误区二：测试集泄露到训练数据

如果你用 LLM 生成评估问题集，然后又把这些问题和答案加入到知识库的数据源中，那就出现了严重的泄露——系统当然能答好这些问题，因为这些答案就在它的"课本"里。更隐蔽的泄露方式是：你在调试过程中反复用同一批问题测试，然后根据测试结果微调了 chunking 参数或 prompt 模板，这本质上也是一种过拟合。

**正确做法**：维护三套独立的数据集——开发集（日常调试用）、验证集（调参选模型用）、测试集（最终汇报用）。测试集在整个开发过程中只能使用一次，就像期末考试一样。如果你发现自己在反复查看测试集的结果来指导开发，那就应该立即停下来，换回开发集继续工作。

### 误区三：评估指标与业务目标脱节

Faithfulness 得分高不一定意味着用户满意。一个极端的例子：系统对每个问题都回答"抱歉，我没有找到相关信息"，Faithfulness 可能是 1.0（因为没有编造任何信息），但用户满意度显然是 0。另一个常见情况是：你的 MRR 从 0.75 提升到了 0.80，团队庆祝了一番，但客户那边客服转接率没有任何变化——因为 MRR 的提升主要来自于那些本来就能答好的问题变得更精准了，而真正导致用户转人工的"困难问题"并没有改善。

**正确做法**：建立"评估指标 → 业务指标"的映射关系表，定期做用户调研或 A/B 测试来验证映射的有效性。必要时引入业务侧指标（如客服工单转接率、用户次日留存、平均会话时长等）作为辅助校验。记住：**评估是为了服务于业务目标，而不是为了追求漂亮的数字**。

### 误区四：忽视评估本身的成本

完整的评估流程（特别是涉及 LLM-as-Judge 的）是不便宜的。前面提到过，50 个问题 × 2 个评价指标 = 100 次 GPT-4o 调用，按当前价格大约几美元。如果你想做更全面的评估（加上检索指标、多个 LLM judge 交叉验证、不同 prompt 变体的 A/B 测试），单次评估的成本轻松达到几十美元。如果还要在 CI 里每次 PR 都跑，一个月下来就是一笔不小的开支。

**正确做法**：建立分层评估策略——PR 阶段用小规模抽样（10 个问题）+ 便宜模型（gpt-4o-mini）做快速门禁；合入 main 后跑完整评估（50 个问题）+ 强模型（gpt-4o）做正式验收；每周或每两周做一次深度评估（100 个问题）+ 多 judge 投票做趋势分析。同时，尽可能缓存中间计算结果（如 embedding、检索结果），避免重复计算。

## 总结

