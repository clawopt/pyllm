# 1.1 为什么需要 LangGraph？

## 从"能跑通"到"能跑稳"：Agent 开发的进化之路

如果你已经学过 LangChain 的 Agent 章节，你应该对 `create_react_agent()` 不陌生——它让你能用几行代码就创建一个能调用工具、自主推理的 AI Agent。在 demo 阶段这很酷：你问一个问题，Agent 自动思考该用什么工具、调用工具、观察结果、再决定下一步，最后给出答案。一切看起来都很美好。

但当你试图把这个 Agent 从 demo 推向生产环境时，问题会接踵而至。比如你做了一个客服工单处理 Agent：用户提交工单后，Agent 需要先分类工单类型 → 查询知识库 → 尝试自动解决 → 解决不了则转人工 → 人工处理后需要确认结果 → 最终关闭工单。用 `create_react_agent()` 写出来的版本可能在测试时表现不错，但上线一周后你会发现：

- **状态丢失**：Agent 在第3步"尝试自动解决"时因为 API 超时崩溃了，重启之后它完全忘了自己之前做了什么，工单卡在半空中
- **无法人工介入**：某个工单被 Agent 错误地归类为"账单问题"（实际是技术故障），你想在 Agent 执行到"转人工"之前手动纠正它的判断——但 ReAct 模式不给你这个机会
- **流程不可控**：Agent 有时候会陷入死循环——反复调用同一个工具却得不到新信息，直到 token 用完才报错退出
- **无法并行**：你需要 Agent 同时调研三个竞品的信息来写对比报告，但 ReAct 只能串行执行
- **调试困难**：出问题时你只知道最终输出错了，但不知道它在哪一步的哪个决策出了偏差

这些问题的根源在于：**`create_react_agent()` 本质上是一个线性循环（Thought → Action → Observation），它没有真正的状态管理能力，也没有结构化的控制流**。而真实世界的复杂任务恰恰需要这些能力。

## LangGraph 是什么

LangGraph 是 LangChain 官方推出的**有状态多步工作流编排框架**。如果说 LangChain 的 `create_react_agent()` 让 AI 学会了"走几步"，那 LangGraph 就是让 AI 学会了"画地图并在上面导航"。它借鉴了图论和有限状态机的思想，把一个复杂的任务拆解为一张**有向图（Directed Graph）**：

```
LangGraph 的核心模型:

┌─────────────────────────────────────────────────────────┐
│                    State (状态/共享内存)                    │
│                                                             │
│   {                                                          │
│     messages: [...],      ← 对话历史                        │
│     current_step: "researching",                            │
│     findings: {},           ← 累积的研究结果                   │
│     error_count: 0,        ← 错误计数器                      │
│     approved: false        ← 审批状态                       │
│   }                                                          │
│                                                             │
└──────────────────────┬──────────────────────────────────────┘
                       │
          ┌────────────┼──────────────┐
          ▼            ▼              ▼
    ┌──────────┐ ┌──────────┐ ┌──────────┐
    │ Node A   │ │ Node B   │ │ Node C   │
    │ (节点)    │ │ (节点)    │ │ (节点)    │
    │ 读取状态   │ │ 修改状态   │ │ 做决策     │
    │ 执行逻辑   │ │ 调用工具   │ │ 条件分支   │
    └─────┬─────┘ └─────┬─────┘ └─────┬─────┘
          │            │              │
          ▼            ▼              ▼
    ┌─────────────────────────────────────┐
    │         Edge (边 / 转移规则)          │
    │                                       │
    │  A → B (总是)                         │
    │  B → C (条件: findings足够?)          │
    │  C → A (条件: 需要更多信息)          │
    │  B → END (条件: 任务完成)             │
    │                                       │
    └─────────────────────────────────────┘
```

这张图就是 LangGraph 编程的核心抽象。每一个圆圈是一个 **Node（节点）——代表一步操作或一个决策点；每一条箭头是一条 **Edge（边）**——定义了从当前节点可以转移到哪些下一个节点以及转移的条件；中间的 **State（状态）** 则是所有节点共享的"黑板"，每个节点都可以读取和修改上面的数据。

## 一个具体的例子：让概念落地

光看定义可能还是有点抽象，让我们用一个真实的场景走一遍完整的流程。假设你要构建一个**代码审查助手 Agent**：

```python
# 这是你最终要写出的 LangGraph 程序（先有个整体感受）
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END


# 第1步：定义 State —— 所有节点共享的数据结构
class CodeReviewState(TypedDict):
    repo_url: str
    files_content: dict       # {filename: code}
    issues_found: list[dict]    # 发现的问题列表
    review_summary: str        # 最终审查报告
    status: str                # 当前阶段
    iteration: int             # 当前迭代次数


# 第2步：定义 Node —— 图中的各个步骤
def clone_repo(state: CodeReviewState) -> dict:
    """克隆仓库并读取文件"""
    print(f"正在克隆仓库: {state['repo_url']}")
    # ... git clone 逻辑 ...
    return {"files_content": {"main.py": "...", "utils.py": "..."}, "status": "cloned"}


def analyze_code(state: CodeReviewState) -> dict:
    """调用 LLM 分析代码质量"""
    print("正在分析代码...")
    # ... LLM 分析逻辑 ...
    return {
        "issues_found": [
            {"file": "main.py", "line": 42, "type": "安全漏洞", "severity": "高"},
            {"file": "utils.py", "line": 15, "type": "性能问题", "severity": "中"},
        ],
        "status": "analyzed",
    }


def check_if_critical(state: CodeReviewState) -> dict:
    """检查是否有严重问题需要人工审核"""
    critical = any(i["severity"] == "高" for i in state["issues_found"])
    if critical:
        return {"status": "needs_human_review"}
    else:
        return {"status": "can_auto_fix"}


def auto_fix(state: CodeReviewState) -> dict:
    """自动修复非严重问题"""
    print("尝试自动修复...")
    # ... LLM 生成修复代码 ...
    return {"status": "fixed", "iteration": state["iteration"] + 1}


def human_review(state: CodeReviewState) -> dict:
    """等待人工审核严重问题"""
    # 这里会触发 INTERRUPT，暂停执行等待人类输入
    return {"status": "awaiting_human"}


def generate_report(state: CodeReviewState) -> dict:
    """生成最终审查报告"""
    report = f"## 代码审查报告\n\n"
    for issue in state["issues_found"]:
        report += f"- [{issue['severity']}] {issue['file']}:{issue['line']} - {issue['type']}\n"
    return {"review_summary": report, "status": "done"}


# 第3步：定义 Edge —— 节点之间的转移规则
def should_route_after_analysis(state: CodeReviewState) -> str:
    """分析完成后根据结果决定下一步"""
    if not state.get("issues_found"):
        return "generate_report"  # 没问题直接出报告
    else:
        return "check_critical"   # 有问题先检查严重程度


def should_continue_fix(state: CodeReviewState) -> str:
    """是否继续修复（限制最大迭代次数）"""
    if state.get("iteration", 0) >= 3:
        return "generate_report"  # 超过最大次数，直接出报告
    elif state["status"] == "needs_human_review":
        return "human_review"
    else:
        return "auto_fix"


# 第4步：组装 Graph
workflow = StateGraph(CodeReviewState)

workflow.add_node("clone_repo", clone_repo)
workflow.add_node("analyze_code", analyze_code)
workflow.add_node("check_critical", check_if_critical)
workflow.add_node("auto_fix", auto_fix)
workflow.add_node("human_review", human_review)
workflow.add_node("generate_report", generate_report)

# 入口
workflow.add_edge(START, "clone_repo")
workflow.add_edge("clone_repo", "analyze_code")

# 条件边
workflow.add_conditional_edges(
    "analyze_code",
    should_route_after_analysis,
    {
        "check_critical": "check_critical",
        "generate_report": "generate_report",
    },
)

workflow.add_conditional_edges(
    "check_critical",
    lambda s: "human_review" if s["status"] == "needs_human_review" else "auto_fix",
)

workflow.add_conditional_edges(
    "auto_fix",
    should_continue_fix,
    {
        "human_review": "human_review",
        "auto_fix": "auto_fix",       # 继续修复
        "generate_report": "generate_report",  # 达到上限
    },
)

workflow.add_edge("human_review", "generate_report")
workflow.add_edge("generate_report", END)

# 编译
app = workflow.compile()

# 运行！
result = app.invoke({
    "repo_url": "https://github.com/example/my-project",
    "status": "init",
    "iteration": 0,
})

print(result["review_summary"])
```

这段代码虽然看起来比 `create_react_agent()` 长，但它做的事情也完全不在一个量级上。让我们看看它解决了前面提到的所有痛点：

**状态不会丢失**：`CodeReviewState` 中记录了每一步的进展（克隆了哪些文件、发现了哪些问题、修到了第几次）。即使程序中途崩溃，只要配合 Checkpointer 就可以从最近的快照恢复。

**人机协作自然流畅**：当发现严重问题时，Agent 会走到 `human_review` 节点暂停，等你审核完之后再继续。你不需要在 Agent 开始前就把所有信息塞给它——可以在它执行过程中随时介入。

**流程完全可控**：`should_continue_fix` 函数限制了最大修复次数为 3 次，防止无限循环。每条边的转移条件都清晰可见，不存在"Agent 自己决定乱跳"的情况。

**调试可视化**：LangGraph 与 LangSmith 深度集成，你可以在 LangSmith 的界面上看到整个图的执行过程——每个节点的输入输出、状态的每次变化、走了哪条边、为什么走这条边。

## LangGraph vs 其他方案

你可能还会问：为什么不直接用 Airflow/DAG？不用 Temporal/Cadence？不用纯手写 while 循环？下面做一个全面的对比：

| 方案 | 适用场景 | 优势 | 劣势 |
|------|---------|------|------|
| **LangChain ReAct** | 单轮问答、简单工具调用 | 5 行代码跑通 | 无状态、无持久化、无分支 |
| **while 循环 + 字典** | 最简单的多步任务 | 不依赖任何框架 | 手动管理状态、容易出 bug、无法可视化 |
| **Airflow/DAG** | 定时批处理 ETL | 成熟稳定、丰富的 operator | 不适合实时交互、LLM 调用集成差 |
| **Temporal 工作流** | 长运行事务性工作流 | 强一致性保证 | 学习曲线陡、与 LLM 生态割裂 |
| **LangGraph** | **LLM 驱动的有状态 Agent** | **专为 LLM 设计、原生支持工具调用、内置人机协作** | 相对新框架，生态还在发展中 |

LangGraph 的独特定位是：**它是为 LLM Agent 这种特殊应用场景量身定制的**。普通的 workflow 引擎不知道什么是 LLM 的 tool call、什么是 message 格式、什么时候应该停下来等人类确认——但 LangGraph 知道。它是唯一一个把"LLM 作为一等公民"嵌入到工作流引擎中的框架。

## 安装与环境准备

```bash
# LangGraph 已经包含在 langchain 包中（核心部分）
pip install langchain langchain-openai langgraph

# 可选：用于可视化的 dev 包
pip install langgraph-dev

# 可选：用于持久化的 checkpointer 存储
pip install langgraph-checkpoint-sqlite

# 可选：用于 Studio 可视化界面
pip install langgraphstudio
```

验证安装：

```python
import langgraph
print(f"LangGraph version: {langgraph.__version__}")

from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
print("✅ 核心组件导入成功")
```

## 总结

