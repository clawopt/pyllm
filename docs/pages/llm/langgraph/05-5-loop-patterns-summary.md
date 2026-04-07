# 5.5 循环模式总结与实战指南

> 经过前面四节的深入探讨，我们已经全面了解了 LangGraph 中各种循环模式的实现方法——从基础的自省循环到重试容错、从状态管理优化到高级的嵌套和并行结构。这一节作为第5章的收尾，我们会把所有内容串联起来，提炼出循环设计的核心决策框架，并通过一个综合实战案例来展示如何在实际项目中灵活运用这些模式。

## 循环设计决策树

当你在面对一个需要循环结构的业务场景时，可以按照下面的决策树来快速确定应该使用哪种循环模式：

```
需要循环吗？
│
├─ 否 → 使用线性管道（第2章的模式）
│
└─ 是 → 循环的目的？
         │
         ├─ 质量改进（让结果越来越好）
         │   └─ 自省循环（Self-Reflection Loop）
         │       ├─ 规则评估 → 第5.1节
         │       ├─ LLM评估 → 第5.1节 LLM驱动版本
         │       └─ 混合评估 → 先规则后LLM
         │
         ├─ 错误恢复（从故障中恢复）
         │   └─ 重试/容错
         │       ├─ 简单重试 → 固定次数
         │       ├─ 指数退避 → 避免雪崩
         │       ├─ 断路器 → 防止级联失败
         │       └─ 多依赖容错 → 关键/非关键区分
         │
         ├─ 遍历处理（对每个元素做操作）
         │   └─ 遍历循环
         │       ├─ 单层遍历 → 简单 for-each
         │       ├─ 嵌套遍历 → 外层+内层子图
         │       └─ 并行遍历 → 扇出扇入结构
         │
         └─ 搜索/优化（寻找最优解）
             └─ 动态循环
                 ├─ 固定迭代 → 设定最大次数
                 ├─ 收敛检测 → 改善停滞时退出
                 └─ 多条件退出 → 综合多种退出策略
```

这个决策树虽然简化了实际情况，但它提供了一个清晰的思考框架。在实际应用中，你可能会组合使用多种模式——比如一个自省循环内部可能包含重试逻辑（调用 LLM 时自动重试），或者一个遍历循环中的每个元素可能都需要经过自省改进。

## 实战案例：智能文档处理流水线

让我们通过一个综合性的实战案例来展示如何在真实场景中组合运用多种循环模式。假设我们需要构建一个文档处理系统：批量导入一批文档，对每篇文档进行多轮质量检查和自动修复，最终生成处理报告。

```python
from typing import TypedDict, Annotated
import operator
import re
import time
from langgraph.graph import StateGraph, START, END

class DocumentState(TypedDict):
    doc_id: str
    raw_content: str
    cleaned_content: str
    quality_issues: list[str]
    quality_score: int
    fix_round: Annotated[int, operator.add]
    max_fix_rounds: int
    final_content: str
    processing_status: str
    doc_log: Annotated[list[str], operator.add]

def clean_document(state: DocumentState) -> dict:
    content = state["raw_content"]
    cleaned = re.sub(r'\s+', ' ', content.strip())
    cleaned = re.sub(r'[^\w\s\u4e00-\u9fff，。！？；：""''（）【】]', '', cleaned)
    return {
        "cleaned_content": cleaned,
        "doc_log": [f"[清洗] 文档已规范化 ({len(cleaned)} 字符)"]
    }

def check_quality(state: DocumentState) -> dict:
    content = state["cleaned_content"]
    issues = []

    if len(content) < 50:
        issues.append("文档过短 (不足50字符)")
    if len(content.split('\n')) < 3:
        issues.append("段落太少 (不足3段)")
    if not any(c.isalpha() for c in content):
        issues.append("缺少字母字符")
    if "TODO" in content or "FIXME" in content:
        issues.append("包含未完成的标记")
    if len(content) > 5000:
        issues.append("文档过长 (>5000字符)")

    score = 100 - len(issues) * 15

    iters = state["fix_round"] + 1
    log_msg = f"[质检 第{iters}轮] 得分: {max(0, score)} | 问题数: {len(issues)}"
    if issues:
        log_msg += "\n    " + "\n    ".join(f"• {i}" for i in issues)

    return {
        "quality_issues": issues,
        "quality_score": max(0, score),
        "doc_log": [log_msg]
    }

def should_fix_document(state: DocumentState) -> str:
    score = state["quality_score"]
    rounds = state["fix_round"] + 1
    max_r = state["max_fix_rounds"]

    if score >= 75:
        return "accept"
    if rounds >= max_r:
        return "force_accept"
    return "fix"

def auto_fix_document(state: DocumentState) -> dict:
    content = state["cleaned_content"]
    issues = state["quality_issues"]
    fixes_applied = []

    fixed = content

    for issue in issues:
        if "过短" in issue and len(fixed) < 50:
            fixed = fixed + (" 这是一段补充文本，用于满足最小长度要求。" * 3)
            fixes_applied.append("补充文本至最小长度")
        elif "段落太少" in issue:
            parts = fixed.split('\n')
            if len(parts) > 1:
                fixed = '\n\n'.join(parts)
                fixes_applied.append("增加段落间距")
        elif "TODO" in issue or "FIXME" in issue:
            fixed = fixed.replace("TODO", "[已完成]").replace("FIXME", "[已修复]")
            fixes_applied.append("标记 TODO/FIXME 为已解决")
        elif "过长" in issue:
            sentences = fixed.replace('。', '。\n').split('\n')
            fixed = '\n'.join(sentences[:10])
            fixes_applied.append("截断至前10句")

    iters = state["fix_round"] + 1
    log_msg = (
        f"[修复 第{iters}轮] 应用了 {len(fixes_applied)} 个修复:\n"
        + "    " + "\n    ".join(f"✓ {f}" for f in fixes_applied)
    )

    return {
        "cleaned_content": fixed,
        "fix_round": 1,
        "doc_log": [log_msg]
    }

def accept_document(state: DocumentState) -> dict:
    rounds = state["fix_round"]
    return {
        "final_content": state["cleaned_content"],
        "processing_status": f"✅ 通过 ({rounds}轮修复)",
        "doc_log": [f"[完成] 文档质量达标"]
    }

def force_accept(state: DocumentState) -> dict:
    rounds = state["fix_round"]
    remaining = state["quality_issues"]
    return {
        "final_content": state["cleaned_content"],
        "processing_status": f"⚠️ 强制接受 ({rounds}轮后仍有{len(remaining)}个问题)",
        "doc_log": [f"[完成] 达到修复上限，使用当前版本"]
    }

doc_graph = StateGraph(DocumentState)
doc_graph.add_node("clean", clean_document)
doc_graph.add_node("check", check_quality)
doc_graph.add_node("fix", auto_fix_document)
doc_graph.add_node("accept", accept_document)
doc_graph.add_node("force_accept", force_accept)

doc_graph.add_edge(START, "clean")
doc_graph.add_edge("clean", "check")
doc_graph.add_conditional_edges("check", should_fix_document, {
    "fix": "fix",
    "accept": "accept",
    "force_accept": "force_accept"
})
doc_graph.add_edge("fix", "check")  # 循环回到质检
doc_graph.add_edge("accept", END)
doc_graph.add_edge("force_accept", END)

compiled_doc_processor = doc_graph.compile()

class BatchProcessingState(TypedDict):
    documents: list[dict]
    current_idx: int
    results: list[dict]
    batch_log: Annotated[list[str], operator.add]
    summary_report: str

def get_next_doc(state: BatchProcessingState) -> dict:
    idx = state["current_idx"]
    docs = state["documents"]

    if idx >= len(docs):
        return {"batch_log": ["[批处理] 所有文档已处理完毕"]}

    doc = docs[idx]
    return {"batch_log": [
        f"[批处理] 处理文档 {idx+1}/{len(docs)}: {doc['id']}"
    ]}

def should_continue_batch(state: BatchProcessingState) -> str:
    if state["current_idx"] >= len(state["documents"]):
        return "generate_report"
    return "process_doc"

def process_single_doc(state: BatchProcessingState) -> dict:
    idx = state["current_idx"]
    doc = state["documents"][idx]

    result = compiled_doc_processor.invoke({
        "doc_id": doc["id"],
        "raw_content": doc["content"],
        "cleaned_content": "",
        "quality_issues": [],
        "quality_score": 0,
        "fix_round": 0,
        "max_fix_rounds": 4,
        "final_content": "",
        "processing_status": "",
        "doc_log": []
    })

    results = list(state["results"])
    results.append({
        "id": doc["id"],
        "status": result["processing_status"],
        "score": result["quality_score"],
        "rounds": result["fix_round"],
        "content_length": len(result["final_content"])
    })

    all_logs = list(result["doc_log"])
    all_logs.insert(0, f"{'='*40}")

    return {
        "results": results,
        "batch_log": all_logs
    }

def advance_batch_index(state: BatchProcessingState) -> dict:
    return {"current_idx": state["current_idx"] + 1}

def generate_summary_report(state: BatchProcessingState) -> dict:
    results = state["results"]

    total = len(results)
    passed = sum(1 for r in results if "✅" in r.get("status", ""))
    forced = sum(1 for r in results if "⚠️" in r.get("status", ""))
    avg_score = sum(r["score"] for r in results) / total if total else 0
    avg_rounds = sum(r["rounds"] for r in results) / total if total else 0

    report_lines = [
        "",
        "=" * 60,
        "📊 批量文档处理报告",
        "=" * 60,
        f"总文档数: {total}",
        f"✅ 质量达标: {passed} ({passed/total*100:.0f}%)" if total else "",
        f"⚠️ 强制接受: {forced} ({forced/total*100:.0f}%)" if total else "",
        f"平均质量分: {avg_score:.1f}",
        f"平均修复轮次: {avg_rounds:.1f}",
        "-" * 40,
        "各文档详情:",
    ]

    for i, r in enumerate(results, 1):
        report_lines.append(
            f"  {i}. [{r['id']}] 分数:{r['score']} "
            f"轮次:{r['rounds']} 长度:{r['content_length']} | {r['status']}"
        )

    report = "\n".join(report_lines)

    return {
        "summary_report": report,
        "batch_log": ["[报告] 处理报告已生成"]
    }

batch_graph = StateGraph(BatchProcessingState)
batch_graph.add_node("get_next", get_next_doc)
batch_graph.add_node("process_doc", process_single_doc)
batch_graph.add_node("advance", advance_batch_index)
batch_graph.add_node("generate_report", generate_summary_report)

batch_graph.add_edge(START, "get_next")
batch_graph.add_conditional_edges("get_next", should_continue_batch, {
    "process_doc": "process_doc",
    "generate_report": "generate_report"
})
batch_graph.add_edge("process_doc", "advance")
batch_graph.add_edge("advance", "get_next")  # 外层循环
batch_graph.add_edge("generate_report", END)

app = batch_graph.compile()

start_time = time.time()
result = app.invoke({
    "documents": [
        {"id": "DOC-001", "content": "这是一篇关于Python编程的短文。Python是一门优秀的编程语言。"},
        {"id": "DOC-002", "content": "TODO 需要完成的内容 FIXME 待修复的问题 这是一段很长的测试文本。" * 20},
        {"id": "DOC-003", "content": "正常的文档内容，包含足够的长度和多个段落。这是第二段内容。这是第三段内容。"},
        {"id": "DOC-004", "content": "短"},
        {"id": "DOC-005", "content": "这是一篇包含TODO标记的文档，需要被修复才能达到质量标准。"}
    ],
    "current_idx": 0,
    "results": [],
    "batch_log": [],
    "summary_report": ""
})
elapsed = time.time() - start_time

print(result["summary_report"])
print(f"\n⏱ 总耗时: {elapsed:.2f}s")
for entry in result["batch_log"]:
    print(entry)
```

这个综合实战案例展示了多层循环的组合使用：

**外层循环**（`batch_graph`）：遍历文档列表，对每个文档依次调用处理流程。这是一个简单的 for-each 遍历循环。

**内层循环**（`doc_graph`，封装为子图）：对单篇文档执行清洗→质检→修复→再质检的循环，直到质量达标或达到最大修复轮次。这是一个自省改进循环。

两层循环通过子图机制组合在一起——外层的 `process_doc` 节点调用了编译后的内层图 `compiled_doc_processor`，内层图的完整执行过程对外层来说是透明的。

最终输出包含一份详细的处理报告，汇总了所有文档的处理情况——包括通过率、强制接受率、平均质量分数、平均修复轮次以及每个文档的详细信息。

## 循环模式的性能基准与选择建议

不同循环模式在性能特征上有显著差异，下表总结了各模式的关键指标：

| 模式 | 典型延迟 | 适用数据规模 | 内存占用 | 容错能力 |
|------|---------|------------|---------|---------|
| 简单单层循环 | 低（毫秒级） | 小到中（<1000项） | 低 | 基础（try-except） |
| 自省循环（规则） | 低（毫秒级） | 小（<100项） | 低-中 | 高（有评分兜底） |
| 自省循环（LLM） | 高（秒级） | 很小（<20项） | 中 | 高（有重试） |
| 重试循环（固定） | 取决于退避时间 | 不适用 | 低 | 高（专门设计） |
| 重试循环（退避） | 中等（可配置） | 不适用 | 低 | 很高（防雪崩） |
| 嵌套循环 | O(外×内) | 小（<100×<100） | 中高 | 取决于实现 |
| 并行循环 | 取决于最慢分支 | 中（每个分支<500） | 中 | 各分支独立 |

基于这些性能特征，给出以下选择建议：

- **数据量小（<100）、逻辑简单** → 用简单的单层循环或规则自省循环
- **需要语义理解** → 用 LLM 驱动的自省循环，但限制迭代次数在 3-5 次
- **外部服务可能不稳定** → 用带指数退避的重试循环
- **多个独立数据源** → 用并行循环（扇出-扇入）
- **双层遍历需求** → 用嵌套循环（外层 + 内层子图）
- **搜索/优化问题** → 用动态循环（多条件退出）

## 常见陷阱终极清单

最后，我们把整个第5章讨论过的所有常见陷阱整理成一个终极清单，供你在设计和实现循环时逐条对照检查：

1. **没有退出条件或退出不可达** → 必须设置硬性上限（如最大迭代次数）
2. **计数器没有正确递增** → 使用 `Annotated[int, operator.add]` 自动管理
3. **状态无限膨胀** → 只保留必要信息，限制历史记录数量
4. **每次迭代重复相同操作** → 把不变的操作移到循环外部
5. **循环体中有副作用且未隔离** → 确保异常不会污染后续迭代的状态
6. **退避时间没有上限** → 设置 max_delay 防止无限等待
7. **临时性错误和永久性错误混为一谈** → 区分错误类型，只对临时性错误重试
8. **断路器冷却期太短或太长** → 根据服务的实际恢复时间来设定
9. **并行循环的结果合并顺序不确定** → 如果顺序重要，添加排序步骤
10. **嵌套循环深度超过 3 层** → 考虑用其他数据结构或算法替代
11. **动态循环的改善停滞阈值设置不当** → 太小会过早退出，太大会浪费时间
12. **错误处理后继续但未记录失败信息** → 必须记录以便事后分析
13. **循环日志过于详细影响性能** → 生产环境减少日志级别
14. **循环中使用了阻塞式 I/O 但没有超时控制** → 为所有 I/O 操作设置超时
15. **循环结束后没有清理临时资源** → 在结束节点中释放资源

遵循这些原则和建议，你就能够在 LangGraph 中构建出既强大又高效的循环系统——无论是简单的重试逻辑还是复杂的多层嵌套自省优化，都能游刃有余地应对。
