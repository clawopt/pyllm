# 2.1 State 设计模式：类型化状态的艺术

## State 是 LangGraph 的心脏

在上一章中我们初步接触了 State 的基本用法——用 `TypedDict` 定义字段、在 Node 函数中返回要更新的字典、然后由框架自动合并。但那只是最简单的用法。在实际项目中，State 的设计质量直接决定了你的 LangGraph 程序是否好维护、是否容易出 bug、以及能否支持持久化和人机协作。这一节我们要深入探讨 State 设计的最佳实践。

## 为什么 State 设计如此重要

先说一个真实的教训。我见过一个团队做的客服 Agent，它的 State 大概长这样：

```python
class BadAgentState(TypedDict):
    data: dict          # 所有数据都塞这里
    messages: list       # 对话历史
    tools: list         # 可用工具
    current: str        # 当前阶段
    result: str         # 最终结果
    error: str          # 错误信息
    retry_count: int     # 重试次数
    metadata: dict      # 元信息
```

看起来没什么问题对吧？但随着功能迭代，这个 State 开始膨胀：

```python
# 三个月后:
class ReallyBadAgentState(TypedDict):
    data: dict
    data_raw: str              # 原始数据（新增）
    data_processed: dict      # 处理后数据（新增）
    messages: list
    system_prompt: str        # 新增：系统提示词
    tools: list
    tool_results: list        # 新增：工具调用记录
    current: str
    sub_status: str           # 新增：子状态
    result: str
    result_draft: str         # 新增：草稿
    final_result: str         # 新增：最终版
    error: str
    last_error: str          # 新增：上一次错误
    retry_count: int
    max_retries: int          # 新增：最大重试数
    metadata: dict
    user_info: dict          # 新增：用户信息
    session_id: str           # 新增：会话 ID
    created_at: str           # 新增：创建时间
    updated_at: str           # 新增：更新时间
```

20 个字段，其中一半是最近三个月里"临时加进去"的。每次加新字段都要检查所有 Node 是否受影响，条件边函数是否需要修改，序列化/反序列化是否兼容旧版本。这就是**State 腐烂的代价**——它像一块没人打扫的公告板，上面贴满了各种过期的通知，真正重要的信息反而找不到。

好的 State 设计应该遵循一个核心原则：**State 只存放需要在多个节点之间传递的信息，且每个字段都有明确的语义和生命周期**。

## State 设计的分层模型

```
┌─────────────────────────────────────────────────────┐
│               State 分层设计                        │
│                                                     │
│  ┌─────────────────────────────────────────────┐   │
│  │ Layer 1: 输入与输出 (I/O)             │   │
│  │  · 用户输入（只读）                    │   │
│  │  · 最终结果（只写一次）                │   │
│  │  · 生命周期 = 整个图执行期间            │   │
│  └─────────────────────────────────────────────┘   │
│                                                     │
│  ┌─────────────────────────────────────────────┐   │
│  │ Layer 2: 任务上下文 (Task Context)       │   │
│  │  · 当前处于哪个阶段                   │   │
│  │  · 已收集的数据（累积式）             │   │
│  │  · 错误/重试状态                       │   │
│  │  · 生命周期 = 当前任务                 │   │
│  └─────────────────────────────────────────────┘   │
│                                                     │
│  ┌─────────────────────────────────────────────┐   │
│  │ Layer 3: 会话状态 (Conversation)          │   │
│  │  · 消息历史                             │   │
│  │  · 用户偏好设置                         │   │
│  │  · 生命周期 = 当前会话（可跨 invoke）  │   │
│  └─────────────────────────────────────────────┘   │
│                                                     │
│  ⚠️ 不应该在 State 中的:                          │
│  · 临时变量（只在单个 Node 内部使用）            │
│  · 不可序列化的对象（文件句柄、网络连接）      │
│  · 重复可计算的数据（每次用时可重新获取的）      │
│  ·  超大的二进制内容（应存外部存储，只存引用）    │
└─────────────────────────────────────────────────────┘
```

## 实战：设计一个完整的 Agent State

让我们用一个实际的例子来实践这个分层设计方法。假设我们要构建一个**代码审查 Agent**：

```python
from typing import TypedDict, Annotated, Literal
from operator import add
import datetime


# === Layer 1: 输入与输出 ===

class ReviewIO(TypedDict):
    """Layer 1: 用户的输入和最终输出"""
    
    # 输入（只写一次，之后只读）
    repo_url: Annotated[str, "用户的 GitHub 仓库地址"]
    review_type: Annotated[Literal["full", "quick", "security"], "审查类型"]
    focus_areas: list[str]  # 关注领域，如 ["安全", "性能", "风格"]
    
    # 输出（只在最后一步写入）
    report: str                  # 最终生成的审查报告
    score: float                # 综合评分 0-10
    summary: str                # 一句话总结


# === Layer 2: 任务上下文 ===

class ReviewContext(TypedDict):
    """Layer 2: 审查过程中的中间状态"""
    
    # 阶段追踪
    current_phase: Annotated[
        Literal[
            "cloning",
            "parsing",
            "analyzing",
            "reviewing",
            "reporting",
            "done",
        ],
        "当前所处阶段",
    ]
    
    # 数据累积（只增不减）
    files_found: Annotated[list[dict], "发现的文件列表"]
    issues_found: Annotated[list[dict], "发现的问题列表"]
    analysis_notes: Annotated[list[str], "分析笔记"]
    
    # 控制流
    error: Annotated[str, "", "最近的错误信息"]
    retry_count: Annotated[int, 0, "当前重试次数"]
    max_retries: int = 3


# === Layer 3: 会话状态 ===

class ConversationState(TypedDict):
    """Layer 3: 与人类交互相关的状态"""
    
    messages: Annotated[list[dict], "对话消息历史"]
    human_feedback: Annotated[str, "", "人类的反馈/审批意见"]
    approved_by: Annotated[str, "", "审批人"]
    created_at: str = datetime.datetime.utcnow().isoformat()
    updated_at: str = datetime.datetime.utcnow().isoformat()


# === 最终组合 ===

class CodeReviewState(ReviewIO, ReviewContext, ConversationState):
    """完整的 Agent State —— 组合三个层级"""
    pass
```

这个 State 设计有 18 个字段，但它们被清晰地分成了三个逻辑层。当你需要添加新功能时，可以快速定位到应该在哪一层添加：

- 需要记录新的用户输入参数？→ Layer 1 (`ReviewIO`)
- 需要在流程中传递新的中间数据？→ Layer 2 (`ReviewContext`)
- 需要与人类交互的新机制？→ Layer 3 (`ConversationState`)

## State 更新策略详解

前面提到 State 是"合并"而非"替换"，但这只是默认行为。LangGraph 提供了更细粒度的控制方式：

### 默认行为：覆盖合并（Override & Merge）

```python
class DefaultState(TypedDict):
    a: int = 0
    b: str = "hello"
    c: list[int] = [1, 2, 3]

def node_x(state: DefaultState) -> dict:
    return {"a": 10}  # 只更新 a

# 结果: {"a": 10, "b": "hello", "c": [1, 2, 3]}
# b 和 c 保持不变 ✅
```

### 列表字段的特殊处理

对于 `list` 类型的字段，默认行为是**扩展**而不是替换：

```python
class ListState(TypedDict):
    items: list[int] = []

def append_item(state: ListState) -> dict:
    return {"items": state["items"] + [42]}

# 初始: items = []
# 第一次调用: items = [42]
# 第二次调用: items = [42, 42]  ← 追加了！不是替换
# 这通常是你想要的行为
```

但如果你的意图是**替换**整个列表呢？你需要使用 `operator` 或显式的 Reducer：

```python
from operator import add
from typing import Annotated

class CounterState(TypedDict):
    count: Annotated[int, operator.add]
    history: list[str]


def increment(state: CounterState) -> dict:
    return {"count": 1}  # count += 1


def reset_and_add(state: CounterState) -> dict:
    # 先清零再增加
    return {
        "count": -state["count"] + 1,  # -3 + 1 = -2
        "history": ["reset at this point"],
    }
```

### 字段删除：Reducer 模式

LangGraph 还支持一种叫 **Reducer** 的模式，让你可以用函数来定义如何合并某个字段：

```python
from langgraph.graph import add_messages

class ChatState(TypedDict):
    messages: list  # 使用特殊的 add_messages reducer

def chat_node(state: ChatState) -> dict:
    from langchain_core.messages import HumanMessage, AIMessage
    
    response = llm.invoke(state["messages"][-1].content)
    
    return {
        "messages": [AIMessage(content=response)]  # 追加到列表末尾
    }

app = graph.compile()
result = app.invoke({"messages": [HumanMessage(content="你好")]})
# result["messages"] == [HumanMessage(...), AIMessage("你好！有什么可以帮您？")]
```

`add_messages` 是 LangGraph 内置的一个 Reducer，它会自动把新消息追加到 `messages` 列表的末尾。这种模式特别适合对话类应用。

### 何时使用 Annotated / operator？

| 场景 | 推荐方式 | 示例 |
|------|---------|------|
| 简单覆盖 | 直接返回字典 | `{"status": "done"}` |
| 数值累加 | `Annotated + operator.add` | 计数器、价格累加 |
| 列表追加 | 默认行为或 `add_messages` | 消息历史、日志 |
| 列表替换 | 需要用特殊处理 | 重置列表、排序后替换 |
| 字典深度合并 | 需要用 Reducer | 嵌套配置的合并 |

## 常见 State 反模式和修复

### 反模式一：State 变成"万能字典"

```python
# ❌ 差反模式
class GodState(TypedDict):
    everything: dict  # 一个字段包含所有东西

# 问题：
# 1. 类型不安全——everything 里什么都能塞
# 2. 条件边要写成 state["everything"]["subfield"] 这种脆弱的长链访问
# 3. 无法通过类型检查发现拼写错误
```

### 反模式二：把 LLM 原始输出直接存 State

```python
# ❌ 不推荐
class RawLLMState(TypedDict):
    llm_raw_response: str  # LLM 返回的原始 JSON
    llm_tokens_used: int
    llm_model: str

# 推荐：
class ProcessedLLMState(TypedDict):
    extracted_answer: str     # 从 LLM 输出中提取的结构化答案
    confidence_score: float    # 解析出的置信度
    sources_cited: list[str]  # 引用的来源
    # llm 内部细节不需要存在 State 中
```

原始 LLM 输出通常很大且格式不稳定（可能偶尔返回 markdown 格式有时返回 JSON），把它原封不动地存进 State 既浪费存储空间又增加了解析的不确定性。应该在 Node 内部完成"原始输出 → 结构化提取"这个过程，只把干净的结果写回 State。

### 反模式三：State 中存放文件内容

```python
# ❌ 绝对不要这样做
class FileContentState(TypedDict):
    file_content: str  # 可能几 MB 的文本！

# ✅ 正确做法
class FileReferenceState(TypedDict):
    file_path: str           # 文件路径或 URL
    file_hash: str           # 内容哈希（用于缓存判断）
    file_size: int           # 文件大小
    file_metadata: dict     # 其他元信息
    # 实际内容按需从 file_path 读取
```

Checkpoint 会序列化整个 State。如果你的 State 里有一个 5MB 的字符串，每次 checkpoint 都要序列化/反序列化这 5MB 数据，而且如果这个字段在图的早期节点就被填入了，后续所有 checkpoint 都会带着这个包袱。正确做法是：大块数据放外部存储（数据库、对象存储、文件系统），State 中只存引用。

## 总结

