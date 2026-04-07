# 2.2 Node 编程模式：如何写好一个节点

## Node 是 LangGraph 中你写得最多的代码

在整个 LangGraph 开发过程中，Node 函数是你最高频编写的代码——一个中等复杂度的图可能有 5-15 个 Node，每个 Node 平均 20-50 行代码。Node 的质量直接决定了图的可读性、可维护性和可调试性。这一节我们要建立一套 Node 编写的最佳实践。

## Node 的三种角色

在实际项目中，Node 通常扮演以下三种角色之一（或它们的组合）：

```
Node 的三种典型角色:

┌───────────┐   ┌───────────┐   ┌───────────┐
│  LLM Node │   │  Tool Node │   │ Logic Node│
│           │   │           │   │          │
│ 调用 LLM  │   │ 调用外部  │   │ 纯计算/判断│
│ 做决策    │   │ 工具/API  │   │ 数据转换  │
│           │   │           │   │          │
└───────────┘   └───────────┘   └───────────┘

组合示例:
  [LLM+Tool] → 先让 LLM 决定调用哪个工具，再执行
  [Logic→LLM] → 先做数据预处理，再把结果给 LLM 分析
  [Tool→Logic→LLM] → 取数据 → 处理 → 让 LLM 总结
```

不同角色的 Node 有不同的编写模式和注意事项。让我们逐一深入。

## 角色一：LLM Node —— 让 AI 做决策

LLM Node 是 Agent 系统中最核心也最昂贵的部分——它通常占整个图的 60%-80% 执行时间和 70%+ 的 token 消耗。写好 LLM Node 是优化 LangGraph 性能的最有效手段之一。

### 基础 LLM Node

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o", temperature=0.1)


def analyst_node(state: AnalysisState) -> dict:
    """基础版 LLM Node：直接调用并返回结果"""
    
    prompt = f"""你是一个资深分析师。请分析以下数据并给出结论。

数据:
{state['raw_data']}

要求:
1. 给出 3 个关键发现
2. 给出 1 个风险提示
3. 综合评分 (1-10)
"""
    
    response = llm.invoke(prompt)
    
    return {
        "analysis": response.content,
        "current_phase": "analyzed",
    }
```

这个版本能用但有几个问题：（1）prompt 是硬编码在函数内部的，不方便修改；（2）没有结构化输出——`response.content` 可能是任何格式；（3）错误处理完全缺失。

### 生产级 LLM Node

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


# Prompt 模板化 —— 与业务逻辑解耦
ANALYST_PROMPT = ChatPrompt.from_template("""
你是一个{role}。请分析以下数据并给出结论。

## 待分析数据
{data}

## 输出要求
1. 给出 {num_findings} 个关键发现
2. 给出 {num_risks} 个风险提示
3. 综合评分 ({scale_min}-{scale_max})

请以 JSON 格式输出分析结果。
""")

# 结构化输出解析器
ANALYSIS_PARSER = StrOutputParser()


class AnalystNodeConfig(TypedDict):
    role: str = "资深数据分析师"
    num_findings: int = 3
    num_risks: int = 1
    scale_min: int = 1
    scale_max: int = 10


def create_analystic_node(config: AnalystNodeConfig | None = None):
    """工厂函数：根据配置创建 LLM Node"""
    
    cfg = config or AnalystNodeConfig()
    prompt = ANALYST_PROMPT.partial(
        role=cfg.role,
        data="{raw_data}",  # 占位符，运行时替换
        num_findings=cfg.num_findings,
        num_risks=cfg.num_risks,
        scale_min=cfg.scale_min,
        scale_max=cfg.scale_max,
    )
    
    def node(state: AnalysisState) -> dict:
        """生产级 LLM Node"""
        try:
            # 1. 用格式化的 prompt 替换占位符
            formatted_prompt = prompt.format(raw_data=state["raw_data"])
            
            # 2. 调用 LLM（带结构化输出）
            response = llm.invoke(formatted_prompt)
            
            # 3. 解析结构化输出
            parsed = ANALYSIS_PARSER.parse(response.content)
            
            # 4. 验证解析结果
            if not parsed or "findings" not in parsed:
                raise ValueError("LLM 输出格式不符合预期")
            
            findings = parsed.get("findings", [])
            
            return {
                "analysis": response.content,       # 完整原文（用于引用溯源）
                "parsed_findings": findings,           # 结构化数据（供后续使用）
                "risk_alerts": parsed.get("risks", []),
                "score": float(parsed.get("score", 5)),
                "current_phase": "analyzed",
                "error": "",
            }
        
        except Exception as e:
            # 5. 错误不崩溃，而是记录到 State 中让后续节点处理
            return {
                "analysis": f"[ERROR] 分析失败: {str(e)}",
                "parsed_findings": [],
                "risk_alerts": [],
                "score": 0.0,
                "current_phase": "analyzed",
                "error": str(e),
            }
    
    return node
```

这个生产级版本展示了几个关键改进：

**Prompt 模板化（`ChatPromptTemplate.partial()`）**：把 prompt 的"固定部分"和"变量部分"分离开来。固定部分（角色定义、输出要求、格式约束）写在 Node 外部作为模板；变量部分（实际数据）在运行时填入。这让同一个 Node 函数可以复用于不同的分析场景——只需要改配置而不用改代码。

**结构化输出 + Parser**：要求 LLM 返回 JSON 格式并用 `StrOutputParser` 解析。这比自由文本输出可靠得多——你可以直接对 `parsed_findings[0]["severity"]` 做条件判断，而不需要用正则去从自然语言中提取信息。

**异常处理与优雅降级**：LLM 调用可能因为网络超时、API 限流、返回格式错误等各种原因失败。生产级的 Node 不应该让异常冒泡导致整个图中断——而是把错误信息记录到 State 的 `error` 字段中，让专门的 `error_handler_node` 来决定是重试、跳过还是放弃。

## 角色二：Tool Node —— 与外部世界交互

Tool Node 负责 LangGraph 与外部系统的交互——调用 API、查询数据库、读写文件、发送消息等。它是连接"AI 世界"和"真实世界"的桥梁。

```python
import httpx
from typing import TypedDict


class APICallState(TypedDict):
    api_base_url: str
    endpoint: str
    params: dict
    response_data: dict
    status_code: int
    error: str


def call_external_api(state: APICallState) -> dict:
    """Tool Node: 调用外部 REST API"""
    
    url = f"{state['api_base_url']}/{state['endpoint']}"
    
    try:
        response = httpx.post(
            url,
            json=state["params"],
            timeout=30.0,
        )
        response.raise_for_status()
        
        return {
            "status_code": response.status_code,
            "response_data": response.json(),
            "error": "",
        }
    
    except httpx.TimeoutException as e:
        return {
            "status_code": 0,
            "response_data": {},
            "error": f"API 超时: {e}",
        }
    
    except httpx.HTTPStatusError as e:
        return {
            "status_code": e.response.status_code,
            "response_data": {},
            "error": f"HTTP {e.response.status_code}: {e.response.text[:100]}",
        }
    
    except Exception as e:
        return {
            "status_code": -1,
            "response_data": {},
            "error": f"未知错误: {e}",
        }


def query_database(state: DBQueryState) -> dict:
    """Tool Node: 查询数据库"""
    
    import sqlite3
    
    conn = sqlite3.connect(state["db_path"])
    cursor = conn.cursor()
    
    try:
        cursor.execute(state["query_sql"], state.get("query_params", {}))
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        
        return {
            "query_result": [
                dict(zip(columns, row)) for row in rows
            ],
            "result_count": len(rows),
            "error": "",
        }
    
    finally:
        conn.close()
```

Tool Node 的编写要点：

**超时控制必须要有**：外部调用是最不可控的延迟来源。30 秒的超时对于大多数 HTTP API 是合理的；数据库查询通常应该更快（< 5 秒）。如果某个操作确实需要更长时间，考虑把它拆成两个 Node——第一个 Node 发起异步请求，第二个 Node 检查结果。

**资源清理要在 finally 中做**：上面的 `query_database` 展示了正确的模式——无论成功还是失败都要关闭数据库连接。忘记关闭连接是在 Tool Node 中最常见的资源泄漏原因。

**返回值要标准化**：不管外部调用成功还是失败，都返回统一的结构化字典（包含状态码、数据和错误信息）。这样后续的 Node 可以用统一的逻辑来判断"这次调用是否成功了"。

## 角色三：Logic Node —— 纯计算与路由辅助

Logic Node 不调用 LLM 也不调外部服务——它只做本地的计算、数据转换或路由判断：

```python
import re
from datetime import datetime


class EmailProcessingState(TypedDict):
    raw_email: str
    sender: str
    recipients: list[str]
    subject: str
    body: str
    is_spam: bool
    priority: int
    category: str


def extract_metadata(state: EmailProcessingState) -> dict:
    """Logic Node: 从原始邮件文本中提取元数据"""
    
    text = state["raw_email"]
    
    # 提取发件人
    sender_match = re.search(r'From:\s*(.+?)\n', text)
    sender = sender_match.group(1).strip() if sender_match else "unknown"
    
    # 提取主题
    subject_match = re.search(r'Subject:\s*(.+?)(?:\n|$)', text)
    subject = subject_match.group(1).strip() if subject_match else "(无主题)"
    
    # 简测是否为垃圾邮件
    spam_indicators = ["unsubscribe", "winner", "free money", "lottery"]
    is_spam = any(indicator in text.lower() for indicator in spam_indicators)
    
    # 判断优先级
    priority = 3  # 默认优先级
    if "urgent" in text.lower() or "ASAP" in text.upper():
        priority = 1
    elif "FYI" in text.upper():
        priority = 2
    
    return {
        "sender": sender,
        "recipients": [],  # 后续 Node 填充
        "subject": subject,
        "body": text,
        "is_spam": is_spam,
        "priority": priority,
        "category": "unclassified",
    }


def classify_and_route(state: EmailProcessingState) -> dict:
    """Logic Node + 路由判断：分类邮件并决定下一步"""
    
    # 这类邮件直接归档不需要处理
    if state["is_spam"]:
        return {"category": "spam", "current_phase": "archived"}
    
    # 内部邮件转给内部团队
    if "@" in state["sender"].endswith("mycompany.com"):
        return {"category": "internal", "current_phase": "routing"}
    
    # 外部客户邮件按优先级路由
    if state["priority"] == 1:
        return {"category": "urgent", "current_phase": "responding"}
    else:
        return {"category": "normal", "current_phase": "responding"}
```

Logic Node 的特点是**快且确定**——同样的输入永远产生同样的输出，不依赖外部服务，调试起来非常容易（给定输入就能预测输出）。这使得它们特别适合做：
- **数据清洗和转换**
- **路由前的预判断/预分类**
- **State 字段的计算和聚合**
- **输入验证**

## Node 组合模式：LLM + Tool 的协作

在实际 Agent 中，最常见的是 LLM Node 和 Tool Node 的组合使用。有两种经典的组合模式：

### 模式 A：LLM 决策 → Tool 执行

```python
def planner_node(state):
    """LLM Node: 制定计划——决定用什么工具查什么"""
    llm_with_tools = llm.bind_tools([search_tool, db_query_tool, calc_tool])
    response = llm_with_tools.invoke(f"制定研究计划: {state['question']}")
    
    return {
        "plan": response.content,
        "plan_steps": parse_plan(response.content),
        "current_step": 0,
    }

def executor_node(state):
    """Tool Node: 执行计划中的每一步"""
    steps = state["plan_steps"]
    current = state["current_step"]
    
    if current >= len(steps):
        return {"current_phase": "all_done"}
    
    step = steps[current]
    
    # 根据步骤类型选择工具
    if step["type"] == "search":
        result = search_tool.invoke(step["query"])
    elif step["type"] == "query":
        result = db_query_tool.invoke(step["sql"])
    elif step["type"] == "calc":
        result = calc_tool.invoke(step["expr"])
    
    results_so_far = state.get("execution_results", [])
    results_so_far.append({
        "step_idx": current,
        "step_desc": step["description"],
        "result": str(result),
    })
    
    return {
        "execution_results": results_so_far,
        "current_step": current + 1,
    }
```

### 模式 B：Tool 收集 → LLM 综合

```python
def gatherer_node(state):
    """Tool Node: 并行收集多个数据源的信息"""
    
    import asyncio
    
    tasks = [
        fetch_from_api_a(state["query"]),
        fetch_from_db(state["query"]),
        search_web(state["query"]),
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    return {
        "collected_data": {
            "api_result": results[0] if not isinstance(results[0], Exception) else None,
            "db_result": results[1] if not isinstance(results[1], Exception) else None,
            "web_result": results[2] if not isinstance(results[2], Exception) else None,
        },
        "current_phase": "synthesizing",
    }


def synthesizer_node(state):
    """LLM Node: 综合所有数据源生成最终答案"""
    
    context = format_multi_source(
        state["collected_data"]["api_result"],
        state["collected_data"]["db_result"],
        state["collected_data"]["web_result"],
    )
    
    response = llm.invoke(f"基于以下综合信息回答问题:\n\n{context}\n\n{state['question']}")
    
    return {
        "answer": response.content,
        "current_phase": "done",
    }
```

这两种模式的选择取决于你的任务特点：**如果每次需要的工具不同且由 LLM 动态决定**，用模式 A；**如果工具集相对固定且需要先收集再综合**，用模式 B**。

## Node 测试策略

由于 Node 就是普通 Python 函数，测试它们非常简单——你不需要启动整个 Graph 就能验证单个 Node 的行为：

```python
import pytest


def test_extract_metadata():
    """测试 Logic Node 的各种边界情况"""
    
    # 正常邮件
    state_normal = {
        "raw_email": "From: alice@example.com\nSubject: Q4 Report\n\nHi team,\nHere are the numbers...",
        "sender": "", "recipients": [], "subject": "", 
        "body": "", "is_spam": False, "priority": 3, "category": ""
    }
    
    result = extract_metadata(state_normal)
    assert result["sender"] == "alice@example.com"
    assert result["subject"] == "Q4 Report"
    assert result["is_spam"] is False
    assert result["priority"] == 3
    
    # 垃圾邮件
    state_spam = {
        "raw_email": "From: winner@lottery.com\nYou won!!!",
        "sender": "", "recipients": [], "subject": "",
        "body": "", "is_spam": False, "priority": 3, "category": ""
    }
    
    result_spam = extract_metadata(state_spam)
    assert result_spam["is_spam"] is True
    assert result_spam["category"] == "spam"
    
    # 加急邮件
    state_urgent = {
        "raw_email": "From: boss@company.com\nASAP: Server is down!!!",
        "sender": "", "recipients": [], "subject": "",
        "body": "", "is_spam": False, "priority": 3, "category": ""
    }
    
    result_urgent = extract_metadata(state_urgent)
    assert result_urgent["priority"] == 1


def test_llm_node_error_handling():
    """测试 LLM Node 的错误处理能力"""
    
    state_bad = {"raw_data": "INVALID_DATA_THAT_WILL_CRASH_LLM"}
    
    result = analyst_node(state_bad)  # 应该不抛异常
    assert result["error"] != ""  # 错误被记录了
    assert result["score"] == 0.0  # 降级为最低分
    assert result["analysis"].startswith("[ERROR]")  # 有明确标记
```

这种单元测试比集成测试快得多——你可以在几秒钟内验证 20+ 个 Node 的正确性，而跑一次完整的 Graph 可能要几分钟。

## 总结

