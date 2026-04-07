# 4.3 Interrupt 高级用法与交互模式

> 前面两节我们学习了 Interrupt 的基本用法和审批流场景，但 Interrupt 的能力远不止于此。它实际上是一种通用的"暂停-等待-恢复"机制，可以用来实现各种需要人机交互的模式——从简单的确认对话框到复杂的多轮对话，从单次输入到动态表单填写。这一节我们会探索 Interrupt 的更多高级用法，包括条件性中断、多轮交互、以及如何把 Interrupt 与其他 LangGraph 特性（如流式输出、子图）结合使用。

## 条件性中断：不是每次都需要停下来

在实际应用中，你可能不希望每次执行到某个节点都无条件地暂停——有时候只有满足特定条件时才需要人工介入，其他情况下应该自动通过。这种"条件性中断"可以通过在节点内部先做判断、只在必要时调用 `interrupt()` 来实现。

```python
from typing import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

class ConditionalInterruptState(TypedDict):
    content: str
    category: str
    confidence: float
    auto_approved: bool
    human_decision: str
    final_verdict: str
    processing_log: Annotated[list[str], operator.add]

def classify_content(state: ConditionalInterruptState) -> dict:
    content = state["content"].lower()

    if any(w in content for w in ["违规", "违法", "暴力", "色情"]):
        return {
            "category": "violated",
            "confidence": 0.95,
            "processing_log": ["[分类] 检测到敏感内容 (置信度: 95%)"]
        }
    if any(w in content for w in ["广告", "推广", "营销"]):
        return {
            "category": "advertisement",
            "confidence": 0.85,
            "processing_log": ["[分类] 检测到广告内容 (置信度: 85%)"]
        }
    if any(w in content for w in ["正常", "普通", "日常"]):
        return {
            "category": "normal",
            "confidence": 0.90,
            "processing_log": ["[分类] 正常内容 (置信度: 90%)"]
        }

    return {
        "category": "uncertain",
        "confidence": 0.40,
        "processing_log": ["[分类] 无法确定类别 (置信度: 40%)"]
    }

def conditional_review(state: ConditionalInterruptState) -> dict:
    category = state["category"]
    confidence = state["confidence"]

    should_interrupt = (
        category == "violated" or
        category == "uncertain" or
        confidence < 0.70
    )

    if not should_interrupt:
        return {
            "auto_approved": True,
            "human_decision": "auto_approve",
            "final_verdict": f"自动通过 ({category})",
            "processing_log": [f"[审核] ✅ 自动通过 | 类别: {category}, 置信度: {confidence:.0%}"]
        }

    prompt = (
        f"{'='*50}\n"
        f"需要人工审核\n"
        f"{'='*50}\n"
        f"内容: {state['content'][:100]}...\n"
        f"AI分类: {category} (置信度: {confidence:.0%})\n\n"
        f"请选择:\n"
        f"- approve: 批准\n"
        f"- reject: 拒绝并标记\n"
        f"- escalate: 升级处理\n"
        f"(可附加评论)"
    )

    user_input = interrupt(prompt)

    if not user_input or not user_input.strip():
        return {"processing_log": ["[审核] 等待人工输入..."]}

    parts = user_input.strip().split(maxsplit=1)
    decision = parts[0].lower()
    comment = parts[1] if len(parts) > 1 else ""

    verdict_map = {
        "approve": f"✅ 人工批准 ({comment})",
        "reject": f"❌ 人工拒绝 ({comment})",
        "escalate": f"⚠️ 已升级 ({comment})"
    }
    verdict = verdict_map.get(decision, f"⚠️ 无效指令: {decision}")

    return {
        "auto_approved": False,
        "human_decision": decision,
        "final_verdict": verdict,
        "processing_log": [f"[审核] {verdict}"]
    }

graph = StateGraph(ConditionalInterruptState)
graph.add_node("classify", classify_content)
graph.add_node("review", conditional_review)
graph.add_edge(START, "classify")
graph.add_edge("classify", "review")
graph.add_conditional_edges("review",
    lambda s: "end" if s["final_verdict"] else "review",
    {"end": END, "review": "review"}
)

app = graph.compile(checkpointer=MemorySaver())

test_cases = [
    ("今天天气真好，适合出门散步", "应该自动通过"),
    ("这个产品太棒了，推荐给大家购买！", "广告内容可能需要审核"),
    ("这是一些违规的内容测试", "必须人工审核"),
]

for content, expected in test_cases:
    config = {"configurable": {"thread_id": f"test-{hash(content) % 10000}"}}
    result = app.invoke({
        "content": content, "category": "", "confidence": 0.0,
        "auto_approved": False, "human_decision": "",
        "final_verdict": "", "processing_log": []
    }, config=config)

    print(f"\n内容: '{content[:30]}...'")
    print(f"  预期: {expected}")
    print(f"  结果: {result['final_verdict']}")
    for log in result["processing_log"]:
        print(f"    {log}")
```

这段程序描述了条件性中断的工作原理。`conditional_review` 节点首先检查是否真的需要人工介入——如果内容是正常的且 AI 分类置信度高（>=70%），就直接返回"自动通过"，根本不会调用 `interrupt()`。只有在检测到敏感内容、或者 AI 无法确定类别（低置信度）的情况下才会触发 `interrupt()` 暂停执行等待人工输入。

这种模式的价值在于它能显著减少对人工的打扰——大部分正常的内容都会被自动放行，只有那些真正需要人类判断的情况才会触发人工审核。在实际生产环境中，这能把人工审核的工作量减少 80% 甚至更多，同时又不牺牲对风险内容的控制力。

## 多轮交互模式

有些场景下一次 Interrupt 不够——你需要和用户进行多轮交互来收集完整的信息。比如一个复杂的表单填写过程，用户可能需要分步提供姓名、地址、支付方式等信息；或者一个技术支持场景，客服人员需要先了解问题类型，再根据类型收集具体的错误信息。

```python
from typing import TypedDict, Annotated, Optional
import operator
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

class MultiRoundState(TypedDict):
    current_step: str
    user_name: Optional[str]
    user_email: Optional[str]
    issue_category: Optional[str]
    issue_description: Optional[str]
    urgency_level: Optional[str]
    collected_data: dict
    form_complete: bool
    interaction_log: Annotated[list[str], operator.add]

FORM_STEPS = [
    {"key": "user_name", "prompt": "请输入您的姓名:", "validator": lambda x: len(x) >= 2},
    {"key": "user_email", "prompt": "请输入您的邮箱地址:",
     "validator": lambda x: "@" in x and "." in x},
    {"key": "issue_category", "prompt": "请选择问题类型 [technical/billing/account/other]:",
     "validator": lambda x: x.lower() in ["technical", "billing", "account", "other"]},
    {"key": "issue_description", "prompt": "请详细描述您遇到的问题:",
     "validator": lambda x: len(x) >= 10},
    {"key": "urgency_level", "prompt": "请选择紧急程度 [low/medium/high]:",
     "validator": lambda x: x.lower() in ["low", "medium", "high"]},
]

def determine_next_step(state: MultiRoundState) -> int:
    for i, step in enumerate(FORM_STEPS):
        key = step["key"]
        if not state.get(key):
            return i
    return -1

def collect_form_data(state: MultiRoundState) -> dict:
    next_step_idx = determine_next_step(state)

    if next_step_idx == -1:
        return {
            "form_complete": True,
            "current_step": "completed",
            "interaction_log": ["[表单] 所有信息已收集完毕"]
        }

    step = FORM_STEPS[next_step_idx]
    key = step["key"]
    prompt_text = step["prompt"]

    progress = f"\n[{next_step_idx + 1}/{len(FORM_STEPS)}] "
    collected_so_far = {k: v for k, v in state.items()
                        if v and k in [s["key"] for s in FORM_STEPS]}
    if collected_so_far:
        progress += f"(已收集: {', '.join(collected_so_far.keys())}) "

    user_input = interrupt(progress + prompt_text)

    if not user_input or not user_input.strip():
        return {"interaction_log": [f"[表单] 等待输入: {key}"]}

    value = user_input.strip()

    validator = step["validator"]
    if not validator(value):
        return {
            "interaction_log": [f"[表单] ❌ 输入无效，请重新输入: {key}"]
        }

    updates = {key: value}
    updates["current_step"] = key
    updates["collected_data"] = {**state.get("collected_data", {}), key: value}
    updates["interaction_log"] = [f"[表单] ✅ 已收集: {key} = {value}"]

    return updates

def route_form_collection(state: MultiRoundState) -> str:
    if state["form_complete"]:
        return "summarize"
    return "collect"

def summarize_submission(state: MultiRoundState) -> dict:
    data = state.get("collected_data", {})
    summary_lines = [
        "工单提交摘要:",
        f"  姓名: {data.get('user_name', 'N/A')}",
        f"  邮箱: {data.get('user_email', 'N/A')}",
        f"  问题类型: {data.get('issue_category', 'N/A')}",
        f"  问题描述: {data.get('issue_description', 'N/A')}",
        f"  紧急程度: {data.get('urgency_level', 'N/A')}",
    ]
    summary = "\n".join(summary_lines)

    confirm_prompt = f"{summary}\n\n确认提交吗? [yes/no]"
    confirmation = interrupt(confirm_prompt)

    if confirmation and confirmation.strip().lower() == "yes":
        return {
            "interaction_log": summary_lines + ["\n✅ 工单已确认提交!"]
        }
    else:
        return {
            "form_complete": False,
            "user_name": None,
            "interaction_log": summary_lines + ["\n已取消，可重新填写"]
        }

graph = StateGraph(MultiRoundState)
graph.add_node("collect", collect_form_data)
graph.add_node("summarize", summarize_submission)

graph.add_edge(START, "collect")
graph.add_conditional_edges("collect", route_form_collection, {
    "collect": "collect",
    "summarize": "summarize"
})
graph.add_conditional_edges("summarize",
    lambda s: "collect" if not s.get("form_complete", True) else END,
    {"collect": "collect", "__end__": END}
)

app = graph.compile(checkpointer=MemorySaver())
config = {"configurable": {"thread_id": "form-session-001"}}

print("=== 开始多轮表单收集 ===")
result = app.invoke({
    "current_step": "", "user_name": None, "user_email": None,
    "issue_category": None, "issue_description": None,
    "urgency_level": None, "collected_data": {},
    "form_complete": False, "interaction_log": []
}, config=config)

for log in result["interaction_log"]:
    print(log)

# 模拟用户逐步填写表单
simulated_inputs = [
    "张三",
    "zhangsan@example.com",
    "technical",
    "登录页面一直显示500错误，已经持续了2小时",
    "high"
]

print("\n--- 模拟用户填写 ---")
for user_input in simulated_inputs:
    r = app.invoke(Command(resume=user_input), config=config)
    last_log = r["interaction_log"][-1] if r["interaction_log"] else ""
    print(f"  用户输入: '{user_input}' → {last_log}")

# 最后确认
print("\n--- 确认提交 ---")
final = app.invoke(Command(resume="yes"), config=config)
for log in final["interaction_log"]:
    print(f"  {log}")
```

这个多轮表单收集系统展示了 Interrupt 在交互式场景中的强大能力。核心设计思路是用一个统一的 `collect_form_data` 节点来处理所有的表单字段收集——它通过 `determine_next_step()` 函数判断当前应该收集哪个字段，然后针对该字段调用 `interrupt()` 等待用户输入。每收到一个有效输入后更新状态中的对应字段，然后回到 `collect` 节点继续收集下一个字段，直到所有字段都收集完毕才进入汇总确认阶段。

注意这里的一个精妙之处：**同一个节点函数 `collect_form_data` 会被多次调用**，每次调用时它根据当前状态决定该做什么（收集哪个字段、还是标记完成）。这是 LangGraph 中一种非常强大的模式——让节点的行为随状态变化而变化，而不是为每个步骤创建独立的节点。

## Interrupt 与流式输出的结合

当你的图中包含 LLM 节点并且使用了流式输出时，Interrupt 的行为会有一些特殊之处需要注意。特别是当 LLM 节点和 Interrupt 节点相邻时，你需要考虑如何在流式输出完成后正确地切换到等待人工输入的状态。

```python
from typing import TypedDict, Annotated
import operator
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

class StreamWithInterruptState(TypedDict):
    topic: str
    draft_content: str
    human_feedback: str
    final_content: str
    status: str
    log: Annotated[list[str], operator.add]

def generate_draft(state: StreamWithInterruptState) -> dict:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个专业的内容写手。根据给定的主题写一段200字以内的草稿。"),
        ("user", "主题: {topic}")
    ])

    chain = prompt | llm
    response = chain.invoke({"topic": state["topic"]})

    return {
        "draft_content": response.content,
        "log": [f"[生成] 草稿已完成 ({len(response.content)} 字)"]
    }

def human_review(state: StreamWithInterruptState) -> dict:
    draft = state["draft_content"]

    display = (
        f"{'='*50}\n"
        f"内容草稿\n"
        f"{'='*50}\n"
        f"{draft}\n"
        f"{'='*50}\n\n"
        f"请审阅以上草稿:\n"
        f"- 输入 'approve' 直接使用\n"
        f"- 输入修改意见（将用于改进草稿）\n"
        f"- 输入 'rewrite' 要求完全重写"
    )

    feedback = interrupt(display)

    if not feedback or not feedback.strip():
        return {"status": "waiting", "log": ["[审核] 等待反馈..."]}

    fb = feedback.strip()

    if fb.lower() == "approve":
        return {
            "human_feedback": fb,
            "final_content": draft,
            "status": "approved",
            "log": ["[审核] ✅ 已批准使用原稿"]
        }

    if fb.lower() == "rewrite":
        return {
            "human_feedback": fb,
            "status": "needs_rewrite",
            "log": ["[审核] 🔄 要求重写"]
        }

    return {
        "human_feedback": fb,
        "status": "needs_revision",
        "log": [f"[审核] 📝 收到修改意见: {fb[:50]}..."]
    }

def apply_revision(state: StreamWithInterruptState) -> dict:
    revision_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是专业编辑。根据用户的修改意见改进以下草稿。"),
        ("user", "原稿:\n{draft}\n\n修改意见:\n{feedback}\n\n请输出改进后的版本:")
    ])

    chain = revision_prompt | llm
    revised = chain.invoke({
        "draft": state["draft_content"],
        "feedback": state["human_feedback"]
    })

    return {
        "final_content": revised.content,
        "status": "revised",
        "log": [f"[修订] 已应用修改意见 ({len(revised.content)} 字)"]
    }

def do_rewrite(state: StreamWithInterruptState) -> dict:
    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是专业写手。请完全重写以下主题的内容，采用不同的风格和角度。"),
        ("user", "主题: {topic}")
    ])

    chain = rewrite_prompt | llm
    new_content = chain.invoke({"topic": state["topic"]})

    return {
        "final_content": new_content.content,
        "status": "rewritten",
        "log": [f"[重写] 新版本完成 ({len(new_content.content)} 字)"]
    }

def route_after_review(state: StreamWithInterruptState) -> str:
    status = state["status"]
    mapping = {
        "approved": END,
        "needs_revision": "revise",
        "needs_rewrite": "rewrite",
        "waiting": "human_review"
    }
    return mapping.get(status, "human_review")

graph = StateGraph(StreamWithInterruptState)
graph.add_node("generate", generate_draft)
graph.add_node("human_review", human_review)
graph.add_node("revise", apply_revision)
graph.add_node("rewrite", do_rewrite)

graph.add_edge(START, "generate")
graph.add_edge("generate", "human_review")
graph.add_conditional_edges("human_review", route_after_review, {
    END: END,
    "revise": "revise",
    "rewrite": "rewrite",
    "human_review": "human_review"
})
graph.add_edge("revise", END)
graph.add_edge("rewrite", END)

app = graph.compile(checkpointer=MemorySaver())

config = {"configurable": {"thread_id": "stream-interrupt-001"}}

print("=== 内容生成+人工审核流程 ===\n")
result = app.invoke({
    "topic": "远程办公的优势与挑战",
    "draft_content": "",
    "human_feedback": "",
    "final_content": "",
    "status": "",
    "log": []
}, config=config)

for entry in result["log"]:
    print(entry)

if result["status"] == "waiting":
    print(f"\n--- 草稿预览 ---")
    print(result["draft_content"])
    print(f"\n--- 等待人工审核 ---")

# 模拟用户提供修改意见
r2 = app.invoke(
    Command(resume="请增加一些具体的数据和案例支撑"),
    config=config
)
print(f"\n最终状态: {r2['status']}")
print(f"最终内容预览:\n{r2['final_content'][:200]}...")
for entry in r2["log"]:
    print(f"  {entry}")
```

这段程序展示了 LLM 生成 + Interrupt 人工审核 + 根据反馈修改/重写的完整工作流。LLM 先生成一份草稿，然后 Interrupt 暂停执行等待人工审核；人类可以选择直接批准（使用原稿）、提供修改意见（LLM 根据意见修改）、或要求完全重写（LLM 用不同风格重新生成）。这种"生成→审核→迭代优化"的模式在内容创作、代码生成等场景中非常有用。

## Interrupt 在子图中的行为

当 Interrupt 出现在子图内部时，行为会变得更加有趣。因为子图本身是被当作父图中的一个节点使用的，所以子图内部的 Interrupt 实际上会导致整个父图的执行都被暂停——不仅仅是子图暂停，而是整个执行链路都在 Interrupt 处挂起。

```python
class SubGraphState(TypedDict):
    input_data: str
    processed_result: str
    human_verified: bool
    verification_comment: str

def sub_process(state: SubGraphState) -> dict:
    data = state["input_data"].upper()
    words = data.split()
    processed = " ".join(reversed(words))
    return {"processed_result": processed}

def sub_verify(state: SubGraphState) -> dict:
    result = interrupt(
        f"验证结果: {state['processed_result']}\n\n确认? [yes/no]"
    )
    if result and result.strip().lower() == "yes":
        return {"human_verified": True, "verification_comment": "已确认"}
    return {"human_verified": False, "verification_comment": "未确认"}

sub_graph = StateGraph(SubGraphState)
sub_graph.add_node("process", sub_process)
sub_graph.add_node("verify", sub_verify)
sub_graph.add_edge(START, "process")
sub_graph.add_edge("process", "verify")
sub_graph.add_conditional_edges("verify",
    lambda s: END if s["human_verified"] else "verify",
    {END: END, "verify": "verify"}
)
compiled_sub = sub_graph.compile(checkpointer=MemorySaver())

class ParentState(TypedDict):
    raw_input: str
    sub_output: str
    final_message: str

def parent_post_process(state: ParentState) -> dict:
    output = state["sub_output"]
    msg = f"处理完成! 结果: {output}"
    return {"final_message": msg}

parent_graph = StateGraph(ParentState)
parent_graph.add_node("run_sub", compiled_sub)
parent_graph.add_node("post_process", parent_post_process)
parent_graph.add_edge(START, "run_sub")
parent_graph.add_edge("run_sub", "post_process")
parent_graph.add_edge("post_process", END)

parent_app = parent_graph.compile(checkpointer=MemorySaver())
config = {"configurable": {"thread_id": "sub-interrupt-test"}}

result = parent_app.invoke({
    "raw_input": "hello world from subgraph",
    "sub_output": "",
    "final_message": ""
}, config=config)

print(f"子图处理结果: {result.get('sub_output', '(等待验证)')}")
print(f"状态: {'已完成' if result.get('final_message') else '子图内等待人工验证'}")
```

当子图内部的 `verify` 节点调用 `interrupt()` 时，整个父图的执行都会暂停。这是因为子图作为父图的一个节点运行，子图没有完成之前父图也无法继续。恢复执行时同样需要用相同的 config 和 `Command(resume=...)` 来传入人类的输入。这种行为对于构建多层嵌套的人机协作流程非常重要——你可以在深层子图中放置 Interrupt，而外层的调用者不需要知道 Interrupt 的存在，只需要按照标准的"invoke → 可能暂停 → resume"模式来操作即可。
