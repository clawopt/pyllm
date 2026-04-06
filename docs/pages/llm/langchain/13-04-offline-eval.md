---
title: 离线评估与持续回归测试
description: 离线评估工作流、pytest 集成、CI/CD 自动化、回归测试套件设计、性能基准测试
---
# 离线评估与持续回归测试

前三节我们讨论了评估的必要性、指标体系和 LangSmith 平台。但 LangSmith 是**在线评估**——每次运行都要调用 LLM API，这会产生成本和延迟。对于开发阶段的快速迭代，我们需要一套**离线评估 + 持续回归**的机制。

## 离线评估 vs 在线评估

| 维度 | 在线评估（LangSmith） | 离线评估（本地） |
|------|---------------------|------------------|
| **成本** | 每次调用 LLM API | 使用本地模型或缓存结果 |
| **速度** | 受网络延迟影响 | 本地执行，毫秒级 |
| **隐私** | 数据上传到云端 | 数据不离开本地 |
| **适用阶段** | 生产监控、A/B 测试 | 开发迭代、模型选择 |
| **评估能力** | 完整（支持 RAGAS） | 有限（需要自建） |

两者不是替代关系，而是**互补**的。开发阶段用离线评估快速验证代码变更，上线后用在线评估持续监控质量。

## 离线评估工作流

### 第一步：准备测试数据集

离线评估需要一份**标注好的测试数据**。格式与在线评估类似，但保存在本地文件中：

```python
# tests/eval_data/customer_service.json
{
    "test_cases": [
        {
            "id": "tc_001",
            "category": "pricing",
            "question": "免费版支持几个人？",
            "expected_answer": "免费版最多支持 5 名团队成员。",
            "context": "pricing.md",
            "metadata": {
                "priority": "high",
                "expected_intent": "product_inquiry",
            },
        },
        {
            "id": "tc_002",
            "category": "refund",
            "question": "订单完成后多久内可以申请退款？",
            "expected_answer": "订单完成后 30 天内可以申请退款。",
            "context": "policies.md",
            "metadata": {
                "priority": "high",
                "expected_intent": "refund_request",
            },
        },
        {
            "id": "tc_003",
            "category": "handoff",
            "question": "转人工",
            "expected_answer": "__HANDOFF__",
            "context": "N/A",
            "metadata": {
                "priority": "critical",
                "expected_intent": "handoff_request",
            },
        },
    ]
}
```

### 第二步：编写离线评估脚本

```python
import json
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class TestCase:
    id: str
    category: str
    question: str
    expected_answer: str
    context: str
    metadata: dict

@dataclass
class EvalResult:
    test_case_id: str
    actual_answer: str
    passed: bool
    score: float
    reason: str
    latency_ms: float

class OfflineEvaluator:
    def __init__(self, bot, eval_data_path: str):
        self.bot = bot
        self.test_cases: List[TestCase] = self._load_test_cases(eval_data_path)

    def _load_test_cases(self, path: str) -> List[TestCase]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [TestCase(**case) for case in data["test_cases"]]

    def evaluate_all(self) -> Dict[str, List[EvalResult]]:
        results = {}
        for case in self.test_cases:
            result = self._evaluate_single(case)
            if case.category not in results:
                results[case.category] = []
            results[case.category].append(result)
        return results

    def _evaluate_single(self, case: TestCase) -> EvalResult:
        import time
        start = time.time()

        self.bot.initialize()
        response = self.bot.process_message(case.question)

        latency = (time.time() - start) * 1000

        passed, score, reason = self._compare_answers(
            case.expected_answer,
            response["response"],
            case.category,
        )

        return EvalResult(
            test_case_id=case.id,
            actual_answer=response["response"],
            passed=passed,
            score=score,
            reason=reason,
            latency_ms=latency,
        )

    def _compare_answers(self, expected: str, actual: str,
                       category: str) -> tuple[bool, float, str]:
        if category == "handoff":
            is_handoff = actual == "__HANDOFF__" or "转人工" in actual
            return is_handoff, 1.0 if is_handoff else 0.0, (
                "正确触发 Handoff" if is_handoff else "未触发 Handoff"
            )

        if expected.lower() in actual.lower():
            return True, 1.0, "完全匹配"

        if any(keyword in actual.lower() for keyword in expected.lower().split()):
            return True, 0.8, "部分匹配"

        if category == "pricing":
            if "5" in actual and "人" in actual:
                return True, 0.9, "包含关键数字"
            if "免费" in actual:
                return True, 0.7, "回答了免费版相关信息"

        return False, 0.0, "不匹配"

    def generate_report(self, results: Dict[str, List[EvalResult]]) -> str:
        report_lines = ["# 离线评估报告\n"]
        report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        for category, case_results in results.items():
            report_lines.append(f"\n## {category.upper()} 测试\n")
            passed_count = sum(1 for r in case_results if r.passed)
            total_count = len(case_results)
            pass_rate = passed_count / total_count * 100 if total_count > 0 else 0
            avg_score = sum(r.score for r in case_results) / total_count if total_count > 0 else 0

            report_lines.append(f"通过率: {pass_rate:.1f}% ({passed_count}/{total_count})")
            report_lines.append(f"平均得分: {avg_score:.2f}")

            failed_cases = [r for r in case_results if not r.passed]
            if failed_cases:
                report_lines.append(f"\n### 失败案例\n")
                for fc in failed_cases[:5]:
                    report_lines.append(f"- **{fc.test_case_id}**: {fc.reason}")
                    report_lines.append(f"  预期: {fc.actual_answer[:100]}...")

        return "\n".join(report_lines)
```

### 第三步：集成到 pytest

为了让评估脚本更易用，我们把它包装成 pytest 测试：

```python
import pytest
from pathlib import Path

@pytest.fixture(scope="session")
def bot():
    from customer_service_bot import CustomerServiceBot
    bot = CustomerServiceBot()
    bot.initialize()
    return bot

@pytest.fixture
def evaluator(bot):
    from offline_evaluator import OfflineEvaluator
    return OfflineEvaluator(bot, "tests/eval_data/customer_service.json")

def test_pricing_questions(evaluator):
    results = evaluator.evaluate_all()
    pricing_results = results.get("pricing", [])

    passed = sum(1 for r in pricing_results if r.passed)
    total = len(pricing_results)
    pass_rate = passed / total * 100 if total > 0 else 0

    assert pass_rate >= 80, f"定价问题通过率仅 {pass_rate:.1f}%，期望 ≥80%"
    assert all(r.latency_ms < 2000 for r in pricing_results), "存在响应超过 2 秒的测试"

def test_refund_questions(evaluator):
    results = evaluator.evaluate_all()
    refund_results = results.get("refund", [])

    passed = sum(1 for r in refund_results if r.passed)
    total = len(refund_results)
    pass_rate = passed / total * 100 if total > 0 else 0

    assert pass_rate >= 75, f"退款问题通过率仅 {pass_rate:.1f}%，期望 ≥75%"

def test_handoff_scenarios(evaluator):
    results = evaluator.evaluate_all()
    handoff_results = results.get("handoff", [])

    passed = sum(1 for r in handoff_results if r.passed)
    total = len(handoff_results)
    pass_rate = passed / total * 100 if total > 0 else 0

    assert pass_rate == 100, "Handoff 场景应该 100% 通过"

def test_overall_performance(evaluator):
    results = evaluator.evaluate_all()
    all_results = [r for cat_results in results.values() for r in cat_results]

    total_passed = sum(1 for r in all_results if r.passed)
    total_tests = len(all_results)
    overall_pass_rate = total_passed / total_tests * 100 if total_tests > 0 else 0

    avg_latency = sum(r.latency_ms for r in all_results) / total_tests if total_tests > 0 else 0

    assert overall_pass_rate >= 75, f"整体通过率仅 {overall_pass_rate:.1f}%，期望 ≥75%"
    assert avg_latency < 1500, f"平均响应时间 {avg_latency:.0f}ms 超过阈值 1500ms"
```

运行评估：

```bash
cd /path/to/project
pytest tests/test_evaluator.py -v
```

输出：

```
tests/test_evaluator.py::test_pricing_questions PASSED
tests/test_evaluator.py::test_refund_questions PASSED
tests/test_evaluator.py::test_handoff_scenarios PASSED
tests/test_evaluator.py::test_overall_performance PASSED

======================== 5 passed in 3.42s ========================
```

## 持续回归测试

回归测试的核心思想是：**每次代码变更后，自动运行一套完整的测试用例，确保没有破坏已有的功能**。

### 回归测试套件设计

一个完整的回归测试套件应该覆盖：

| 测试类别 | 测试数量 | 典型用例 |
|---------|---------|----------|
| **核心功能测试** | 20-30 | 定价查询、退款流程、订单状态、Handoff 触发 |
| **边界情况测试** | 10-15 | 空输入、超长文本、特殊字符、并发请求 |
| **性能基准测试** | 5-10 | P95 延迟、Token 使用量、并发吞吐量 |
| **安全测试** | 5-10 | Prompt 注入、SQL 注入、恶意输入 |
| **集成测试** | 5-10 | 端到端场景（完整对话流程） |

### 性能基准测试

```python
import pytest
import statistics
from typing import List

def benchmark_p95_latency(bot, questions: List[str], iterations: int = 10):
    latencies = []
    for _ in range(iterations):
        start = time.time()
        for q in questions:
            bot.process_message(q)
        end = time.time()
        latencies.append((end - start) / len(questions) * 1000)

    p95 = statistics.quantiles(latencies, n=20)[18]
    avg = statistics.mean(latencies)

    print(f"P95 延迟: {p95:.0f}ms")
    print(f"平均延迟: {avg:.0f}ms")

    return p95

@pytest.mark.benchmark
def test_latency_baseline(bot):
    questions = [
        "免费版支持几个人？",
        "专业版多少钱？",
        "退款流程是什么？",
    ]
    p95 = benchmark_p95_latency(bot, questions)

    assert p95 < 1500, f"P95 延迟 {p95}ms 超过阈值 1500ms"

@pytest.mark.benchmark
def test_token_usage_efficiency(bot):
    total_tokens = 0
    for _ in range(10):
        result = bot.process_message("免费版支持几个人？")
        total_tokens += result.get("token_count", 100)

    avg_tokens = total_tokens / 10
    print(f"平均 Token 使用量: {avg_tokens}")

    assert avg_tokens < 200, f"平均 Token 使用量 {avg_tokens} 超过阈值 200"
```

### 集成到 CI/CD

回归测试应该每次代码 push 时自动运行。以下是 GitHub Actions 的示例配置：

```yaml
# .github/workflows/regression-test.yml
name: Regression Tests

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  regression:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt

      - name: Run regression tests
        run: |
          pytest tests/test_evaluator.py -v --tb=short
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}

      - name: Upload results
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: regression-results
          path: reports/
```

### 回归失败的处理策略

当回归测试失败时，不应该直接阻止合并，而是根据失败类型采取不同策略：

| 失败类型 | 策略 | 示例 |
|---------|------|------|
| **功能回归** | 阻止合并，要求修复 | 定价问题从 95% 降到 80% |
| **性能退化** | 警告但允许合并（需说明） | P95 从 1200ms 升到 1800ms，但仍在可接受范围 |
| **边界情况** | 记录但不阻止 | 新发现的特殊字符处理问题，标记为已知问题 |
| **测试本身问题** | 修复测试，不阻止代码合并 | 测试用例更新导致误判 |

## 离线评估的常见误区

**误区一：测试用例太少**。只准备 10-20 个测试 case，无法覆盖真实场景的多样性。一个健康的回归测试套件应该至少有 **50+ 个测试用例**，并且随着功能增加持续扩充。

**误区二：测试用例过于简单**。所有测试都是"免费版支持几个人？"这种直球问题，不测试边界和异常。好的测试套件应该包含：**正常路径 + 边界情况 + 错误输入 + 复杂组合场景**。

**误区三：只测不修**。测试失败后没有分析原因就直接跳过。正确的做法是：**失败 → 分析根因 → 修复 → 重新测试 → 验证修复**。这是一个闭环过程。

**误区四：硬编码的预期答案**。测试用例的 `expected_answer` 写死了"免费版最多支持 5 人"，但产品策略变了（改成了 3 人），测试就会失败。预期答案应该基于**业务规则文档**而不是写死具体数值。

## 评估体系的最终形态

把前面三节的内容整合起来，一个完整的评估体系应该是这样的：

```
开发阶段
├── 离线评估 (13.4)
│   ├── 本地测试套件 (pytest)
│   ├── 回归测试 (CI/CD)
│   └── 性能基准测试
│   └── 目标：快速迭代、成本控制
│
上线阶段
├── 在线评估 (13.3)
│   ├── LangSmith Trace 采集
│   ├── Dataset 管理
│   ├── 自动化评估器
│   └── 目标：持续监控、质量基线
│
优化阶段
├── 指标体系 (13.2)
│   ├── 检索相关性 (Context Relevance)
│   ├── 忠实度 (Faithfulness)
│   ├── 答案相关性 (Answer Relevance)
│   └── Agent 专用指标 (工具调用正确率、目标达成率）
│   └── 目标：多维衡量、精准诊断
│
反馈闭环
└── 人工审核 (13.1)
    ├── L1: 人工抽查
    ├── L2: 自动规则监控
    ├── L3: LLM-as-Judge
    └── L4: 黄金基准集
    └── 目标：发现盲点、建立 ground truth
```

这个体系覆盖了从**开发到上线、从离线到在线、从功能到性能**的全生命周期。它不是一次性的工作，而是需要持续维护和改进的基础设施。
