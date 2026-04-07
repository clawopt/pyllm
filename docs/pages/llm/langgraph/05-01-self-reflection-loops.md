# 5.1 自省循环基础：让 Agent 反思并改进

> 在 LangChain 的 Agent 章节中我们接触过 ReAct 模式——Agent 通过"思考→行动→观察"的循环来逐步完成任务。但 ReAct 循环有一个局限：它很少"回头审视"自己之前的行为是否正确。LangGraph 的图结构让我们可以构建更复杂的循环模式，其中最强大的一种就是**自省循环（Self-Reflection Loop）**——让 Agent 生成初步结果后，主动反思这个结果的质量，如果发现问题就自动修正，然后再反思、再修正，直到结果达到可接受的标准。这种"生成→评估→改进"的循环模式是构建高质量 AI 系统的核心范式。

## 从一个直观例子开始：代码自审与自动修复

先通过一个完整的例子来看看自省循环是如何工作的。假设我们要构建一个系统，让它写一段 Python 代码，然后自己审查这段代码是否有问题，如果有就自动修复，修复后再审查，直到代码质量达标。

```python
from typing import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, START, END

class CodeSelfReviewState(TypedDict):
    requirement: str
    generated_code: str
    review_findings: list[str]
    review_score: int
    iteration: Annotated[int, operator.add]
    max_iterations: int
    fix_actions: list[str]
    final_code: str
    final_status: str
    execution_log: Annotated[list[str], operator.add]

def generate_initial_code(state: CodeSelfReviewState) -> dict:
    req = state["requirement"]
    code = f"""def process_data(data):
    result = []
    for item in data:
        if item:
            print(f"Processing: {item}")
            result.append(item.upper())
    return result

def main():
    sample = ["hello", "world", "", "test", None]
    output = process_data(sample)
    print(output)
"""
    return {
        "generated_code": code,
        "execution_log": [f"[生成] 初始代码已生成 ({len(code)} 字符)"]
    }

def review_code(state: CodeSelfReviewState) -> dict:
    code = state["generated_code"]
    findings = []

    if "print(" in code and ("debug" in code.lower() or "Processing" in code):
        findings.append("包含调试打印语句 (print)")
    if "TODO" in code or "FIXME" in code:
        findings.append("存在 TODO/FIXME 标记")
    if 'except:' in code and ('Exception' not in code):
        findings.append("使用了裸 except 子句（未指定异常类型）")
    if not any(kw in code for kw in ["def ", "class "]):
        findings.append("缺少函数或类定义")

    has_type_hints = "-> " in code or ": " in code.split("def ")[-1].split("(")[0] if "def " in code else False
    if not has_type_hints:
        findings.append("函数缺少类型注解")

    has_docstring = '"""' in code or "'''" in code
    if not has_docstring:
        findings.append("函数缺少文档字符串")

    score = 100 - len(findings) * 12

    log_entry = (
        f"[审查 第{state['iteration']+1}轮] "
        f"发现 {len(findings)} 个问题 | 得分: {max(0, score)}"
    )
    if findings:
        log_entry += f"\n    问题列表:"
        for i, f in enumerate(findings, 1):
            log_entry += f"\n      {i}. {f}"

    return {
        "review_findings": findings,
        "review_score": max(0, score),
        "execution_log": [log_entry]
    }

def should_continue_refining(state: CodeSelfReviewState) -> str:
    score = state["review_score"]
    iteration = state["iteration"] + 1
    max_iter = state["max_iterations"]

    if score >= 80:
        return "accept"
    if iteration >= max_iter:
        return "max_reached"
    return "fix"

def apply_fixes(state: CodeSelfReviewState) -> dict:
    code = state["generated_code"]
    findings = state["review_findings"]
    actions = []

    fixed_code = code

    for finding in findings:
        if "打印语句" in finding:
            fixed_code = fixed_code.replace(
                'print(f"Processing: {item}")',
                '# Removed debug print'
            )
            actions.append("移除调试打印语句")
        elif "裸 except" in finding:
            fixed_code = fixed_code.replace(
                'except:',
                'except Exception as e:'
            )
            actions.append("补全异常类型为 Exception")
        elif "类型注解" in finding:
            fixed_code = fixed_code.replace(
                "def process_data(data):",
                "def process_data(data: list[str]) -> list[str]:"
            )
            actions.append("添加函数签名类型注解")
        elif "文档字符串" in finding:
            fixed_code = fixed_code.replace(
                "def process_data",
                '''def process_data"""
    处理输入数据列表，返回大写后的非空元素。

    Args:
        data: 输入字符串列表

    Returns:
        大写后的非空字符串列表
    """
def process_data'''
            )
            actions.append("添加函数文档字符串")

    iter_num = state["iteration"] + 1
    log_msg = (
        f"[修复 第{iter_num}轮] 应用了 {len(actions)} 个修复:\n"
        + "\n".join(f"    • {a}" for a in actions)
    )

    return {
        "generated_code": fixed_code,
        "fix_actions": actions,
        "iteration": 1,
        "execution_log": [log_msg]
    }

def accept_result(state: CodeSelfReviewState) -> dict:
    iters = state["iteration"]
    return {
        "final_code": state["generated_code"],
        "final_status": f"✅ 已接受 (经过{iters}轮迭代)",
        "execution_log": [f"[完成] 最终得分: {state['review_score']}"]
    }

def max_iterations_reached(state: CodeSelfReviewState) -> dict:
    iters = state["iteration"]
    remaining = state["review_findings"]
    return {
        "final_code": state["generated_code"],
        "final_status": f"⚠️ 达到最大迭代次数({iters}次)，仍有{len(remaining)}个问题",
        "execution_log": [
            f"[终止] 最大迭代次数已用尽",
            f"       剩余问题: {remaining}"
        ]
    }

loop_graph = StateGraph(CodeSelfReviewState)
loop_graph.add_node("generate", generate_initial_code)
loop_graph.add_node("review", review_code)
loop_graph.add_node("fix", apply_fixes)
loop_graph.add_node("accept", accept_result)
loop_graph.add_node("max_reached", max_iterations_reached)

loop_graph.add_edge(START, "generate")
loop_graph.add_edge("generate", "review")
loop_graph.add_conditional_edges("review", should_continue_refining, {
    "accept": "accept",
    "max_reached": "max_reached",
    "fix": "fix"
})
loop_graph.add_edge("fix", "review")  # 关键：修复后回到审查
loop_graph.add_edge("accept", END)
loop_graph.add_edge("max_reached", END)

app = loop_graph.compile()

print("=" * 60)
print("代码自省循环演示")
print("=" * 60)

result = app.invoke({
    "requirement": "编写一个处理数据列表的Python函数",
    "generated_code": "",
    "review_findings": [],
    "review_score": 0,
    "iteration": 0,
    "max_iterations": 5,
    "fix_actions": [],
    "final_code": "",
    "final_status": "",
    "execution_log": []
})

print(f"\n最终状态: {result['final_status']}")
print(f"\n完整执行日志:")
for entry in result["execution_log"]:
    print(entry)

print(f"\n{'='*60}")
print("最终代码:")
print(result["final_code"])
```

这段程序描述了自省循环的完整工作原理。执行路径是这样的：

```
START → generate(生成初始代码) → review(审查代码)
                                      ↓ 分数<80 且 未达上限
                                    fix(应用修复)
                                      ↓
                                    review(再次审查) ← 循环！
                                      ↓ 分数>=80 或 达到上限
                                  accept / max_reached → END
```

核心拓扑特征是 `fix` 节点通过一条普通边回到了 `review` 节点，形成了**循环边（cycle edge）**。每次循环中 `iteration` 计数器会自动加 1（因为用了 `Annotated[int, operator.add]`），`should_continue_refining` 条件函数根据当前分数和迭代次数决定是继续修复还是结束。注意这里有两个退出条件：分数达到 80 以上表示质量合格可以接受；迭代次数达到上限表示即使还有问题也必须停止（防止无限循环）。

## 自省循环的三个核心组件

一个完整的自省循环由三个核心组件构成：**生成器（Generator）**负责产出初始结果或改进后的结果；**评估器（Evaluator）**负责判断当前结果的质量并给出具体的改进建议；**控制器（Controller）**负责根据评估结果决定是继续改进还是结束循环。这三个组件各司其职，组合在一起形成了强大的自我优化能力。

```python
# 组件1: 生成器 —— 产生输出
def generator(state):
    current = state.get("current_output", "")
    feedback = state.get("feedback", "")

    if feedback:
        new_output = improve_based_on_feedback(current, feedback)
    else:
        new_output = generate_from_scratch(state["requirement"])

    return {"current_output": new_output}

# 组件2: 评估器 —— 检查质量
def evaluator(state):
    output = state["current_output"]

    criteria_results = {}
    for criterion in quality_criteria:
        criteria_results[criterion.name] = criterion.check(output)

    overall_score = weighted_average(criteria_results)
    issues = find_issues(criteria_results)

    return {
        "score": overall_score,
        "issues": issues,
        "feedback": format_feedback(issues),
        "criteria_details": criteria_results
    }

# 组件3: 控制器 —— 决定是否继续
def controller(state):
    score = state["score"]
    iterations = state["iterations"]
    max_iters = state["max_iterations"]
    improvement = get_recent_improvement_trend(state)

    if score >= threshold:
        return "accept"
    if iterations >= max_iters:
        return "force_accept_or_reject"
    if improvement < min_improvement_threshold:
        return "stagnated"  # 改善停滞，可能需要换策略
    return "continue"
```

这三个组件的实现方式可以非常灵活——生成器可以是 LLM 调用、模板渲染、规则引擎等任何能产出的机制；评估器可以是规则检查、LLM 打分、自动化测试套件等任何能做质量判断的机制；控制器可以是简单的阈值比较、复杂的趋势分析、甚至引入人类决策（通过 Interrupt）。这种灵活性意味着同样的循环拓扑结构可以适应各种不同的业务场景。

## 用 LLM 驱动的自省循环

上面的例子中生成器和评估器都是基于规则的，这在简单场景下足够了。但更强大的做法是用 LLM 来驱动这两个组件——让 LLM 来生成内容、让另一个 LLM（或者同一个 LLM 的不同 prompt）来评估质量。这样就能处理那些规则难以覆盖的复杂场景。

```python
from typing import TypedDict, Annotated
import operator
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class ArticleWritingState(TypedDict):
    topic: str
    draft: str
    evaluation: dict
    score: float
    revision_notes: str
    iteration: Annotated[int, operator.add]
    max_iterations: int
    final_article: str
    status: str
    log: Annotated[list[str], operator.add]

generate_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个专业的内容创作者。
请根据给定的主题写一篇300字以内的文章。
要求：结构清晰、论点有据、语言流畅。"""),
    ("user", "主题: {topic}")
])

evaluate_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个严格的内容编辑。
请从以下维度评估文章质量（每项1-10分）：
1. 内容准确性：信息是否准确无误
2. 结构清晰度：逻辑是否清晰、段落是否合理
3. 语言表达：是否流畅、是否有语法错误
4. 完整性：是否充分回答了主题

请以JSON格式返回评估结果：
{{"accuracy": <分>, "structure": <分>, "language": <分>,
  "completeness": <分>, "overall_score": <加权总分>,
  "issues": ["问题1", "问题2"],
  "suggestions": ["建议1", "建议2"]}}"""),
    ("user", "主题: {topic}\n\n文章:\n{draft}")
])

revise_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个专业的编辑。根据编辑反馈修改文章。
保持原文的核心观点和风格，只针对指出的问题进行修改。
修改后的文章应该在300字以内。"""),
    ("user", """原始文章:\n{draft}

编辑评估:\n{evaluation}

请输出修改后的完整文章:""")
])

generate_chain = generate_prompt | llm
evaluate_chain = evaluate_prompt | llm
revise_chain = revise_prompt | llm

def llm_generate(state: ArticleWritingState) -> dict:
    response = generate_chain.invoke({"topic": state["topic"]})
    return {
        "draft": response.content,
        "log": [f"[生成] 初稿完成 ({len(response.content)} 字)"]
    }

def llm_evaluate(state: ArticleWritingState) -> dict:
    response = evaluate_chain.invoke({
        "topic": state["topic"],
        "draft": state["draft"]
    })

    try:
        import json
        eval_result = json.loads(response.content)
        score = eval_result.get("overall_score", 0)
        issues = eval_result.get("issues", [])
        suggestions = eval_result.get("suggestions", [])
    except:
        score = 5.0
        issues = ["无法解析评估结果"]
        suggestions = []

    it = state["iteration"] + 1
    log_msg = (
        f"[评估 第{it}轮] 总分: {score}/10 | 问题数: {len(issues)}"
    )
    if issues:
        log_msg += "\n    " + "\n    ".join(f"• {i}" for i in issues[:3])
        if len(issues) > 3:
            log_msg += f"\n    ... 还有 {len(issues)-3} 个问题"

    return {
        "evaluation": eval_result if 'eval_result' in dir() else {},
        "score": score,
        "revision_notes": str(suggestions),
        "log": [log_msg]
    }

def should_revise(state: ArticleWritingState) -> str:
    score = state["score"]
    iters = state["iteration"] + 1
    max_it = state["max_iterations"]

    if score >= 8.0:
        return "accept"
    if iters >= max_it:
        return "finish"
    return "revise"

def llm_revise(state: ArticleWritingState) -> dict:
    response = revise_chain.invoke({
        "draft": state["draft"],
        "evaluation": state["revision_notes"]
    })

    it = state["iteration"] + 1
    return {
        "draft": response.content,
        "iteration": 1,
        "log": [f"[修订 第{it}轮] 文章已更新 ({len(response.content)} 字)"]
    }

def accept_article(state: ArticleWritingState) -> dict:
    return {
        "final_article": state["draft"],
        "status": "accepted",
        "log": [f"[完成] ✅ 文章已定稿 (共{state['iteration']}轮迭代)"]
    }

def finish_with_current(state: ArticleWritingState) -> dict:
    return {
        "final_article": state["draft"],
        "status": "completed_max_iterations",
        "log": [f"[完成] ⚠️ 达到最大迭代次数，使用当前版本"]
    }

article_graph = StateGraph(ArticleWritingState)
article_graph.add_node("generate", llm_generate)
article_graph.add_node("evaluate", llm_evaluate)
article_graph.add_node("revise", llm_revise)
article_graph.add_node("accept", accept_article)
article_graph.add_node("finish", finish_with_current)

article_graph.add_edge(START, "generate")
article_graph.add_edge("generate", "evaluate")
article_graph.add_conditional_edges("evaluate", should_revise, {
    "accept": "accept",
    "finish": "finish",
    "revise": "revise"
})
article_graph.add_edge("revise", "evaluate")  # 循环回到评估
article_graph.add_edge("accept", END)
article_graph.add_edge("finish", END)

app = article_graph.compile()

result = app.invoke({
    "topic": "远程办公对团队协作的影响",
    "draft": "",
    "evaluation": {},
    "score": 0.0,
    "revision_notes": "",
    "iteration": 0,
    "max_iterations": 3,
    "final_article": "",
    "status": "",
    "log": []
})

print(f"\n状态: {result['status']}")
for entry in result["log"]:
    print(entry)
if result["final_article"]:
    print(f"\n最终文章预览:\n{result['final_article'][:300]}...")
```

这个 LLM 驱动的写作自省循环展示了三个 LLM 调用的协作：第一个 LLM 根据主题生成初稿；第二个 LLM 以编辑的身份从多个维度评估初稿的质量并给出具体的问题和修改建议；第三个 LLM 根据评估反馈修改文章。整个循环最多执行 3 轮（由 `max_iterations` 控制），当评分达到 8.0/10 以上时提前结束。

## 自省循环的关键设计考量

在设计和实现自省循环时，有几个关键的设计决策需要仔细考虑。

**第一，退出条件的设定**。循环必须有明确的退出条件，否则就会变成无限循环。最常见的两种退出条件是：质量达标（如评分超过某个阈值）和次数上限（如最多迭代 N 次）。一个好的实践是同时设置这两个条件——质量达标时优雅地提前退出，次数到达上限时强制退出并标记结果可能不完美。有些高级实现还会加入第三个条件：改善停滞检测——如果连续几轮迭代的评分都没有明显提升，说明当前的改进策略可能无效，应该提前退出或切换策略。

**第二，评估标准的客观性**。评估器的质量直接决定了自省循环的效果。如果评估标准太宽松，低质量的输出也会被接受；如果评估标准太严格，循环会一直运行直到次数上限。理想情况下，评估标准应该是客观的、可量化的、与实际使用场景对齐的。对于可以用自动化测试验证的场景（如代码是否编译通过、测试是否全部通过），优先使用自动化测试作为评估标准；对于需要主观判断的场景（如文章质量、设计美观度），可以使用 LLM 辅助评估但要配合人工抽样审核。

**第三，每轮改进的幅度控制**。每轮迭代应该带来可见的改进，而不是微乎其微的变化。如果每轮改进幅度太小，需要很多轮才能达到目标，浪费时间和成本；如果改进幅度太大，可能会过度修改导致偏离原始意图。一个实用的技巧是在 prompt 中明确告诉 LLM"只针对指出的问题进行修改，不要改变其他部分"，这样可以控制改进的范围和幅度。

**第四，成本与质量的平衡**。每一轮迭代都涉及一次 LLM 调用（甚至多次），这意味着时间和金钱的成本。需要在迭代次数限制、每轮的 LLM 模型选择（大模型更准但更贵）、以及最终输出的质量要求之间找到平衡点。对于高价值场景（如生产环境的代码生成、正式发布的文章），值得多花几轮迭代来确保质量；对于低价值的辅助任务（如草稿生成、内部备忘录），1-2 轮迭代可能就够了。

## 常见误区与反模式

在使用自省循环时有几个常见的错误模式需要注意。

**误区一：没有退出条件或退出条件不可达**。最严重的 bug 就是写了循环但没有正确的退出条件，导致图永远运行下去直到被外部强制终止。一定要确保至少有一个退出条件是必然可达的——比如次数上限就是一个硬性的必然可达条件（只要你的计数器正确递增）。

**误区二：评估器和生成器串通作弊**。如果你用同一个 LLM 同时做生成和评估，而且评估的 prompt 不够严格，LLM 可能会"放水"给自己高分以尽快结束循环。解决方案是使用不同的 LLM（或同一个 LLM 但用完全不同的 system prompt 和角色设定）来做评估，或者在评估 prompt 中强调"严格打分、不要宽容"。

**误区三：每次迭代都从头开始**。有些实现中每轮迭代都丢弃之前的输出完全重新生成，这浪费了前面所有迭代的工作成果。正确的做法是基于上一轮的输出进行增量修改——保留好的部分、只修改有问题的部分。这不仅效率更高，也能避免在修改过程中引入新的问题。

**误区四：忽略了循环中的状态膨胀**。每次循环都会往状态中追加新的数据（日志、中间结果、历史记录等），如果循环次数很多，状态的大小会不断增长。对于那些只需要最新值的字段（如 `draft`），不需要用累积型合并；对于那些确实需要记录历史的字段（如 `log`），可以考虑限制保存的条目数量或定期截断。
