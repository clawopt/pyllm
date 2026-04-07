# 2.5 实战模式与最佳实践

> 前面四节我们分别深入了状态设计、节点编程、边路由和子图组合这四个核心维度，每一个都包含了大量的 API 细节和代码示例。但知道单个概念怎么用和能在真实项目中把它们有机地组合起来，中间还隔着一道"工程经验"的鸿沟。这一节的目标就是帮你跨过这道鸿沟——我们会把前面学到的所有知识串联起来，通过几个完整的实战模式来展示如何在复杂场景中综合运用 State、Node、Edge 和 Subgraph，同时也会总结出一些在长期开发中沉淀下来的最佳实践和常见陷阱。

## 模式一：管道-过滤器架构（Pipeline-Filter Pattern）

这是最基础也最常用的 LangGraph 架构模式。它的核心思想是：数据像水流一样经过一系列顺序执行的节点，每个节点对数据做一种特定的转换或过滤操作，最终输出处理后的结果。这种模式特别适合数据处理流水线、内容审核链路、多阶段分析等场景。

理解这个模式的关键在于把握"每个节点只做一件事"的原则——就像 Unix 管道中 `cat file | grep pattern | sort | uniq -c | head -10` 每个命令只负责一个环节一样，LangGraph 的管道模式中每个节点也应该有清晰单一的职责。

```python
from typing import TypedDict, Annotated
import operator
import re
import json
from langgraph.graph import StateGraph, START, END

class ArticleProcessingState(TypedDict):
    raw_text: str
    title: str
    cleaned_text: str
    paragraphs: list[str]
    keywords: list[str]
    summary: str
    sentiment: str
    processing_log: Annotated[list[str], operator.add]

def extract_title(state: ArticleProcessingState) -> dict:
    text = state["raw_text"].strip()
    lines = [l for l in text.split('\n') if l.strip()]
    title = lines[0] if lines else "无标题"
    return {"title": title, "processing_log": ["[1] 标题提取完成"]}

def clean_content(state: ArticleProcessingState) -> dict:
    text = state["raw_text"]
    cleaned = re.sub(r'\s+', ' ', text)
    cleaned = re.sub(r'[^\w\s\u4e00-\u9fff，。！？、；：""''（）《】\[\]]', '', cleaned)
    return {"cleaned_text": cleaned.strip(), "processing_log": ["[2] 内容清洗完成"]}

def split_paragraphs(state: ArticleProcessingState) -> dict:
    text = state["cleaned_text"]
    paras = [p.strip() for p in text.split('\n') if p.strip() and len(p.strip()) > 20]
    return {"paragraphs": paras, "processing_log": [f"[3] 分段完成，共{len(paras)}段"]}

def extract_keywords(state: ArticleProcessingState) -> dict:
    from collections import Counter
    text = state["cleaned_text"].lower()
    words = re.findall(r'\b\w{3,}\b', text)
    common_stopwords = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out'}
    filtered = [w for w in words if w not in common_stopwords]
    top_keywords = [word for word, count in Counter(filtered).most_common(8)]
    return {"keywords": top_keywords, "processing_log": ["[4] 关键词提取完成"]}

def analyze_sentiment(state: ArticleProcessingState) -> dict:
    text = state["cleaned_text"]
    positive_words = {'好', '优秀', '喜欢', '棒', '赞', 'good', 'great', 'excellent', 'love', 'amazing'}
    negative_words = {'差', '糟糕', '讨厌', '烂', 'bad', 'terrible', 'hate', 'awful', 'poor'}
    pos_count = sum(1 for w in positive_words if w in text.lower())
    neg_count = sum(1 for w in negative_words if w in text.lower())
    if pos_count > neg_count * 1.5:
        sentiment = "正面"
    elif neg_count > pos_count * 1.5:
        sentiment = "负面"
    else:
        sentiment = "中性"
    return {"sentiment": sentiment, "processing_log": [f"[5] 情感分析完成: {sentiment}"]}

def generate_summary(state: ArticleProcessingState) -> dict:
    paras = state["paragraphs"]
    if len(paras) >= 3:
        summary = paras[0][:200] + ("..." if len(paras[0]) > 200 else "")
    elif paras:
        summary = paras[0]
    else:
        summary = state["cleaned_text"][:200]
    return {"summary": summary, "processing_log": ["[6] 摘要生成完成"]}

pipeline_graph = StateGraph(ArticleProcessingState)
pipeline_graph.add_node("extract_title", extract_title)
pipeline_graph.add_node("clean_content", clean_content)
pipeline_graph.add_node("split_paragraphs", split_paragraphs)
pipeline_graph.add_node("extract_keywords", extract_keywords)
pipeline_graph.add_node("analyze_sentiment", analyze_sentiment)
pipeline_graph.add_node("generate_summary", generate_summary)

pipeline_graph.add_edge(START, "extract_title")
pipeline_graph.add_edge("extract_title", "clean_content")
pipeline_graph.add_edge("clean_content", "split_paragraphs")
pipeline_graph.add_edge("split_paragraphs", "extract_keywords")
pipeline_graph.add_edge("extract_keywords", "analyze_sentiment")
pipeline_graph.add_edge("analyze_sentiment", "generate_summary")
pipeline_graph.add_edge("generate_summary", END)

app = pipeline_graph.compile()

sample_article = """
# 深入理解 Python 异步编程

Python 的异步编程模型自从 3.5 版本引入 async/await 语法以来，
已经成为了构建高性能网络应用的核心技术。与传统的多线程模型相比，
协程（Coroutine）具有更低的内存开销和更简单的并发控制方式。

asyncio 库提供了事件循环、任务调度、 Future 对象等基础设施，
让开发者能够以同步代码的风格编写异步逻辑。
优秀的异步框架如 FastAPI、aiohttp 等，进一步简化了异步开发的门槛。

在实际项目中合理使用异步编程，可以显著提升 I/O 密集型应用的吞吐量，
但同时也要注意避免在异步函数中执行阻塞操作，否则会阻塞整个事件循环。
"""

result = app.invoke({
    "raw_text": sample_article,
    "title": "",
    "cleaned_text": "",
    "paragraphs": [],
    "keywords": [],
    "summary": "",
    "sentiment": "",
    "processing_log": []
})

print(f"标题: {result['title']}")
print(f"段落数: {len(result['paragraphs'])}")
print(f"关键词: {result['keywords']}")
print(f"情感倾向: {result['sentiment']}")
print(f"摘要: {result['summary'][:100]}...")
for log in result["processing_log"]:
    print(log)
```

这个例子完整展示了管道模式的运作方式：数据从 `raw_text` 字段流入，依次经过标题提取、内容清洗、分段、关键词提取、情感分析和摘要生成六个阶段，每个阶段的输出作为下一阶段的输入，最终所有结果汇总到同一个状态字典中返回。`processing_log` 字段使用了 `Annotated[list, operator.add]` 来累积每一步的处理日志，这在调试和生产监控中都非常有用。

管道模式的一个关键设计考量是**错误隔离**。如果第五个节点 `analyze_sentiment` 抛出了异常，前面的四个节点的产出是否会丢失呢？这取决于你是否配置了 checkpointing——如果没有，整个 invoke 调用会抛出异常且所有中间状态都会丢失；如果配置了 MemorySaver 或其他 checkpointer，你可以从最近的 checkpoint 恢复，并且可以跳过失败的节点继续执行后面的步骤（通过修改图结构加入条件边来实现）。

## 模式二：决策树路由（Decision Tree Routing）

当业务流程中存在多层条件判断时——比如先判断用户等级，再根据等级判断是否有权限，然后根据权限决定走哪个处理分支——线性管道就不够用了，需要引入基于条件边的决策树路由模式。这个模式的核心思想是用条件边来模拟 if-elif-else 的嵌套判断结构，把复杂的业务规则编码为图的拓扑结构。

```python
from typing import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, START, END

class LoanApplicationState(TypedDict):
    applicant_name: str
    income_monthly: float
    credit_score: int
    employment_years: float
    loan_amount: float
    loan_purpose: str
    risk_category: str
    interest_rate: float
    approval_status: str
    rejection_reason: str
    audit_trail: Annotated[list[str], operator.add]

def assess_credit_tier(state: LoanApplicationState) -> dict:
    score = state["credit_score"]
    income = state["income_monthly"]
    if score >= 750 and income >= 15000:
        tier = "excellent"
    elif score >= 680 and income >= 8000:
        tier = "good"
    elif score >= 600 and income >= 5000:
        tier = "fair"
    else:
        tier = "poor"
    return {
        "risk_category": tier,
        "audit_trail": [f"信用评级: {tier} (评分={score}, 月收入={income})"]
    }

def route_by_tier(state: LoanApplicationState) -> str:
    return state["risk_category"]

def process_excellent(state: LoanApplicationState) -> dict:
    base_rate = 4.5
    purpose = state["loan_purpose"]
    if purpose in ["房产", "房屋"]:
        rate = base_rate - 0.5
    elif purpose in ["教育"]:
        rate = base_rate - 0.3
    else:
        rate = base_rate
    return {
        "interest_rate": round(rate, 2),
        "approval_status": "approved",
        "audit_trail": [f"优质客户审批通过, 利率: {rate}%"]
    }

def process_good(state: LoanApplicationState) -> dict:
    amount = state["loan_amount"]
    dti = (amount / 36) / state["income_monthly"]
    if dti > 0.4:
        return {
            "approval_status": "rejected",
            "rejection_reason": f"债务收入比过高 ({dti:.1%})",
            "audit_trail": [f"良好客户拒绝: DTI={dti:.1%} > 40%"]
        }
    rate = 6.5
    return {
        "interest_rate": round(rate, 2),
        "approval_status": "approved_with_conditions",
        "audit_trail": [f"良好客户有条件通过, 利率: {rate}%"]
    }

def process_fair(state: LoanApplicationState) -> dict:
    years = state["employment_years"]
    if years < 2:
        return {
            "approval_status": "rejected",
            "rejection_reason": "工作年限不足 (需至少2年)",
            "audit_trail": [f"一般客户拒绝: 工作年限={years}年 < 2年"]
        }
    rate = 9.0
    return {
        "interest_rate": round(rate, 2),
        "approval_status": "approved_with_collateral",
        "audit_trail": [f"一般客户需担保通过, 利率: {rate}%"]
    }

def process_poor(state: LoanApplicationState) -> dict:
    return {
        "approval_status": "rejected",
        "rejection_reason": "信用评分和工作收入不满足最低要求",
        "audit_trail": ["差级客户直接拒绝"]
    }

def finalize_decision(state: LoanApplicationState) -> dict:
    status_map = {
        "approved": "✅ 自动批准",
        "approved_with_conditions": "⚠️ 有条件批准（需额外材料）",
        "approved_with_collateral": "⚠️ 需担保批准",
        "rejected": "❌ 已拒绝"
    }
    msg = status_map.get(state["approval_status"], "未知状态")
    reason = state.get("rejection_reason", "")
    detail = f"{msg}" + (f" | 原因: {reason}" if reason else "")
    return {"audit_trail": [f"最终决策: {detail}"]}

loan_graph = StateGraph(LoanApplicationState)
loan_graph.add_node("assess_tier", assess_credit_tier)
loan_graph.add_node("process_excellent", process_excellent)
loan_graph.add_node("process_good", process_good)
loan_graph.add_node("process_fair", process_fair)
loan_graph.add_node("process_poor", process_poor)
loan_graph.add_node("finalize", finalize_decision)

loan_graph.add_edge(START, "assess_tier")
loan_graph.add_conditional_edges(
    "assess_tier", route_by_tier,
    {
        "excellent": "process_excellent",
        "good": "process_good",
        "fair": "process_fair",
        "poor": "process_poor"
    }
)
loan_graph.add_edge("process_excellent", "finalize")
loan_graph.add_edge("process_good", "finalize")
loan_graph.add_edge("process_fair", "finalize")
loan_graph.add_edge("process_poor", "finalize")
loan_graph.add_edge("finalize", END)

app = loan_graph.compile()

result = app.invoke({
    "applicant_name": "王芳",
    "income_monthly": 12000.0,
    "credit_score": 720,
    "employment_years": 3.5,
    "loan_amount": 300000.0,
    "loan_purpose": "教育",
    "risk_category": "",
    "interest_rate": 0.0,
    "approval_status": "",
    "rejection_reason": "",
    "audit_trail": []
})

print(f"申请人: {result['applicant_name']}")
print(f"风险等级: {result['risk_category']}")
print(f"审批结果: {result['approval_status']}")
print(f"利率: {result['interest_rate']}%")
for entry in result["audit_trail"]:
    print(f"  {entry}")
```

这个贷款审批的例子展示了决策树模式的完整结构。注意几个关键设计点：第一，`route_by_tier` 函数返回的值必须精确匹配 `path_map` 中的 key（这里是 `"excellent"`、`"good"`、`"fair"`、`"poor"`），这是初学者最容易出错的地方——如果返回了一个不在 path_map 中的字符串，LangGraph 会直接抛出 ValueError。第二，四条分支最终都汇聚到 `"finalize"` 节点，这种**扇入（fan-in）**结构保证了无论走了哪条分支，最后的收尾逻辑都能统一执行。第三，`audit_trail` 字段贯穿全流程，每一步都追加自己的审计信息，这对于金融类应用的合规性要求来说至关重要。

## 模式三：带重试的自愈循环（Self-Healing Loop with Retry）

很多实际场景下某个操作可能不会一次成功——调用外部 API 可能遇到临时故障、LLM 生成的格式可能不符合预期、数据校验可能不通过等等。这时候就需要用循环结构来做自动重试，同时在达到最大重试次数后优雅地降级或报错。这个模式结合了自环边（self-loop edge）、计数器状态和条件退出三个机制。

```python
from typing import TypedDict, Annotated
import operator
import random
import time
from langgraph.graph import StateGraph, START, END

class APICallState(TypedDict):
    endpoint: str
    payload: dict
    response_data: dict
    error_message: str
    attempt_count: Annotated[int, operator.add]
    max_attempts: int
    success: bool
    final_status: str
    retry_log: Annotated[list[str], operator.add]

def simulate_api_call(endpoint: str, payload: dict) -> tuple[dict | None, str | None]:
    random.seed(hash(endpoint + str(payload)))
    if random.random() < 0.3:
        return None, f"API 临时不可用 (503 Service Unavailable)"
    if random.random() < 0.1:
        return None, "API 返回格式错误 (500 Internal Error)"
    return {"data": f"来自 {endpoint} 的响应", "timestamp": time.strftime("%H:%M:%S")}, None

def call_api(state: APICallState) -> dict:
    endpoint = state["endpoint"]
    payload = state["payload"]
    attempt = state["attempt_count"] + 1

    response, error = simulate_api_call(endpoint, payload)

    log_entry = f"[尝试 {attempt}/{state['max_attempts']}] 调用 {endpoint}"
    if error:
        log_entry += f" → 失败: {error}"
        return {
            "error_message": error,
            "attempt_count": 1,
            "success": False,
            "retry_log": [log_entry]
        }
    else:
        log_entry += f" → 成功"
        return {
            "response_data": response,
            "error_message": "",
            "attempt_count": 1,
            "success": True,
            "retry_log": [log_entry]
        }

def should_retry(state: APICallState) -> str:
    if state["success"]:
        return "success"
    if state["attempt_count"] >= state["max_attempts"]:
        return "exhausted"
    return "retry"

def handle_success(state: APICallState) -> dict:
    attempts = state["attempt_count"]
    return {
        "final_status": f"成功 (第{attempts}次尝试)",
        "retry_log": [f"✅ API 调用成功，共尝试 {attempts} 次"]
    }

def handle_exhaustion(state: APICallState) -> dict:
    attempts = state["attempt_count"]
    last_error = state["error_message"]
    return {
        "final_status": f"失败 ({attempts}次尝试后放弃)",
        "retry_log": [f"❌ 达到最大重试次数，最后错误: {last_error}"]
    }

retry_graph = StateGraph(APICallState)
retry_graph.add_node("call_api", call_api)
retry_graph.add_node("handle_success", handle_success)
retry_graph.add_node("handle_exhaustion", handle_exhaustion)

retry_graph.add_edge(START, "call_api")
retry_graph.add_conditional_edges(
    "call_api", should_retry,
    {"retry": "call_api", "success": "handle_success", "exhausted": "handle_exhaustion"}
)
retry_graph.add_edge("handle_success", END)
retry_graph.add_edge("handle_exhaustion", END)

app = retry_graph.compile()

result = app.invoke({
    "endpoint": "https://api.example.com/data",
    "payload": {"query": "test"},
    "response_data": {},
    "error_message": "",
    "attempt_count": 0,
    "max_attempts": 5,
    "success": False,
    "final_status": "",
    "retry_log": []
})

print(f"最终状态: {result['final_status']}")
print(f"响应数据: {result['response_data']}")
for entry in result["retry_log"]:
    print(entry)
```

这段程序描述了一个带最大重试次数限制的自愈循环。核心机制在于 `should_retry` 条件函数的三路分支设计：如果 API 调用成功了就走 `"success"` 分支结束；如果还没成功但还没达到最大次数就走 `"retry"` 分支回到 `call_api` 再试一次；如果已经达到了最大重试次数就走 `"exhausted"` 分支进入失败处理。由于 `attempt_count` 使用了 `Annotated[int, operator.add]`，每次经过 `call_api` 节点时计数器都会自动加 1，不需要手动管理计数器的递增逻辑。

这里有一个容易踩坑的地方：**重试之间的退避策略**。上面的例子中每次重试之间没有任何等待时间，这在实际生产环境中可能导致对下游服务造成过大压力。更健壮的做法是在 `call_api` 节点内部加入指数退避（exponential backoff）：

```python
def call_api_with_backoff(state: APICallState) -> dict:
    endpoint = state["endpoint"]
    payload = state["payload"]
    attempt = state["attempt_count"] + 1

    if attempt > 1:
        wait_time = min(2 ** (attempt - 1), 10)
        time.sleep(wait_time)

    response, error = simulate_api_call(endpoint, payload)
    # ... 其余逻辑相同
```

另一个需要关注的点是**幂等性**。如果你的 API 调用有副作用（比如创建订单、发送邮件），那重试可能会导致重复操作。在设计重试逻辑时必须确保被调用的接口是幂等的，或者在状态中记录已经成功完成的副作用操作，避免重复执行。

## 模式四：并行扇出-汇聚（Parallel Fan-out Fan-in）

有些任务的各个子任务之间没有依赖关系，可以并行执行以缩短总耗时。LangGraph 虽然本身是单线程执行的（默认情况下），但你可以通过 Send API 或者把并行任务封装为子图再配合异步执行框架来实现并行效果。不过即使不借助额外的并行机制，用图的结构来表达"这些任务是并行的"这件事本身就很有价值——它清晰地表达了任务之间的依赖关系（或者说无依赖关系），便于后续优化和阅读。

```python
from typing import TypedDict, Annotated
import operator
import time
from langgraph.graph import StateGraph, START, END

class ParallelAnalysisState(TypedDict):
    target_url: str
    content_text: str
    security_analysis: dict
    seo_analysis: dict
    performance_analysis: dict
    accessibility_analysis: dict
    overall_score: float
    recommendations: list[str]
    analysis_log: Annotated[list[str], operator.add]

def fetch_content(state: ParallelAnalysisState) -> dict:
    url = state["target_url"]
    start = time.time()
    time.sleep(0.1)
    elapsed = time.time() - start
    content = f"从 {url} 获取的模拟页面内容（HTML长度: ~15KB）..."
    return {
        "content_text": content,
        "analysis_log": [f"[抓取] {url} | 耗时 {elapsed:.2f}s"]
    }

def run_security_check(state: ParallelAnalysisState) -> dict:
    content = state["content_text"]
    start = time.time()
    time.sleep(0.15)
    elapsed = time.time() - start
    findings = []
    if "http://" in content:
        findings.append("发现混合内容（HTTP资源在HTTPS页面上）")
    if "eval(" in content:
        findings.append("检测到动态代码执行风险")
    has_form = "<form" in content
    has_csrf = "csrf" in content.lower()
    result = {
        "score": 85 if has_csrf else (60 if has_form else 75),
        "findings": findings,
        "has_https": "https" in content
    }
    return {
        "security_analysis": result,
        "analysis_log": [f"[安全检查] 发现 {len(findings)} 个问题 | 得分 {result['score']} | {elapsed:.2f}s"]
    }

def run_seo_check(state: ParallelAnalysisState) -> dict:
    content = state["content_text"]
    start = time.time()
    time.sleep(0.12)
    elapsed = time.time() - start
    has_title = "<title>" in content or "#" in content
    has_meta_desc = "description" in content.lower()
    has_h1 = "<h1" in content or "# " in content
    word_count = len(content.split())
    score = 0
    if has_title: score += 25
    if has_meta_desc: score += 25
    if has_h1: score += 25
    if 300 <= word_count <= 3000: score += 25
    result = {
        "score": score,
        "word_count": word_count,
        "issues": [] if has_title else ["缺少标题标签"],
        "meta_ok": has_meta_desc
    }
    return {
        "seo_analysis": result,
        "analysis_log": [f"[SEO检查] 词数:{word_count} | 得分:{score} | {elapsed:.2f}s"]
    }

def run_performance_check(state: ParallelAnalysisState) -> dict:
    content = state["content_text"]
    start = time.time()
    time.sleep(0.08)
    elapsed = time.time() - start
    size_kb = len(content.encode('utf-8')) / 1024
    img_tags = content.count("<img") + content.count("![")
    script_tags = content.count("<script")
    score = 100
    if size_kb > 500: score -= 20
    if img_tags > 20: score -= 15
    if script_tags > 10: score -= 15
    result = {
        "score": max(score, 0),
        "page_size_kb": round(size_kb, 1),
        "image_count": img_tags,
        "script_count": script_tags
    }
    return {
        "performance_analysis": result,
        "analysis_log": [f"[性能检查] 页面大小:{size_kb:.1f}KB | 图片:{img_tags} | 脚本:{script_tags} | {elapsed:.2f}s"]
    }

def run_accessibility_check(state: ParallelAnalysisState) -> dict:
    content = state["content_text"]
    start = time.time()
    time.sleep(0.1)
    elapsed = time.time() - start
    issues = []
    has_alt = 'alt=' in content
    has_aria = 'aria-' in content
    has_heading_structure = content.count("<h") + content.count("# ")
    if not has_alt:
        issues.append("图片缺少 alt 属性")
    if not has_aria:
        issues.append("缺少 ARIA 标签")
    if has_heading_structure < 2:
        issues.append("标题层级结构不完整")
    score = max(100 - len(issues) * 15, 0)
    result = {"score": score, "issue_count": len(issues), "issues": issues}
    return {
        "accessibility_analysis": result,
        "analysis_log": [f"[可访问性检查] 问题数:{len(issues)} | 得分:{score} | {elapsed:.2f}s"]
    }

def aggregate_results(state: ParallelAnalysisState) -> dict:
    sec = state["security_analysis"]["score"]
    seo = state["seo_analysis"]["score"]
    perf = state["performance_analysis"]["score"]
    a11y = state["accessibility_analysis"]["score"]
    overall = (sec * 0.3 + seo * 0.25 + perf * 0.25 + a11y * 0.2)
    recommendations = []
    if sec < 80:
        recommendations.append("优先修复安全问题：升级到 HTTPS、添加 CSRF 保护")
    if seo < 70:
        recommendations.append("SEO 优化建议：添加 meta description、确保 H1 标签存在")
    if perf < 70:
        recommendations.append("性能优化：压缩图片、减少脚本数量、启用 CDN")
    if a11y < 70:
        recommendations.append("可访问性改进：为图片添加 alt 文本、使用 ARIA 标签")
    if overall >= 85:
        recommendations.append("整体表现良好，继续保持！")
    return {
        "overall_score": round(overall, 1),
        "recommendations": recommendations,
        "analysis_log": [
            f"[汇总] 安全:{sec} SEO:{seo} 性能:{perf} 无障碍:{a11y}",
            f"[汇总] 综合得分: {overall:.1f} | 建议 {len(recommendations)} 条"
        ]
    }

parallel_graph = StateGraph(ParallelAnalysisState)
parallel_graph.add_node("fetch_content", fetch_content)
parallel_graph.add_node("security_check", run_security_check)
parallel_graph.add_node("seo_check", run_seo_check)
parallel_graph.add_node("performance_check", run_performance_check)
parallel_graph.add_node("accessibility_check", run_accessibility_check)
parallel_graph.add_node("aggregate", aggregate_results)

parallel_graph.add_edge(START, "fetch_content")
parallel_graph.add_edge("fetch_content", "security_check")
parallel_graph.add_edge("fetch_content", "seo_check")
parallel_graph.add_edge("fetch_content", "performance_check")
parallel_graph.add_edge("fetch_content", "accessibility_check")
parallel_graph.add_edge("security_check", "aggregate")
parallel_graph.add_edge("seo_check", "aggregate")
parallel_graph.add_edge("performance_check", "aggregate")
parallel_graph.add_edge("accessibility_check", "aggregate")
parallel_graph.add_edge("aggregate", END)

app = parallel_graph.compile()

start_time = time.time()
result = app.invoke({
    "target_url": "https://example.com/product-page",
    "content_text": "",
    "security_analysis": {},
    "seo_analysis": {},
    "performance_analysis": {},
    "accessibility_analysis": {},
    "overall_score": 0.0,
    "recommendations": [],
    "analysis_log": []
})
total_time = time.time() - start_time

print(f"\n{'='*50}")
print(f"网站综合分析报告 — 总耗时: {total_time:.2f}s")
print(f"{'='*50}")
print(f"综合得分: {result['overall_score']}")
for rec in result["recommendations"]:
    print(f"  → {rec}")
for entry in result["analysis_log"]:
    print(entry)
```

这个网站分析器展示了经典的扇出-汇聚模式。`fetch_content` 是唯一的串行前置步骤——必须先拿到页面内容才能开始各种分析。之后从 `fetch_content` 同时引出了四条边到四个分析节点（安全、SEO、性能、可访问性），这就是**扇出（fan-out）**。四个分析节点各自独立运行，互不依赖，它们的结果最后全部汇聚到 `aggregate` 节点进行加权打分和建议生成，这就是**汇聚（fan-in）**。

虽然在这个简单示例中四个分析节点实际上是串行执行的（因为 LangGraph 默认的单线程执行模型），但图的结构清楚地表达出了"这四个分析之间没有依赖关系"这一语义信息。如果你后续需要真正的并行执行，可以把这四个节点替换为一个使用 `asyncio.gather()` 的包装节点，或者使用 LangGraph 的 Send API 来实现真正的并行分派，而图的结构完全不需要改变。

## 最佳实践清单

经过上面四种模式的演练，我们来总结一下在使用 LangGraph 构建实际系统时应该遵循的核心原则。这些原则不是教条，而是从大量实践经验中提炼出来的指导方针，帮助你在面对设计选择时有清晰的判断依据。

**关于状态设计的最佳实践**：首先，保持状态的扁平化——避免在 TypedDict 中定义超过两层的嵌套结构，因为深层嵌套会让状态更新路径变得难以追踪，也增加了序列化/反序列化的出错概率。其次，区分"输入型字段"和"累积型字段"——输入型字段（如用户请求参数）通常只在初始化时写入一次，之后只读；累积型字段（如日志列表、计数值）会在多个节点中被反复更新，需要配合 `Annotated` 类型使用正确的合并操作符。第三，不要在状态中存储大块二进制数据或完整的 LLM 响应原始文本——用引用（文件路径、URL、数据库 ID）代替，需要时再按需加载。第四，为每个状态字段添加清晰的类型注解和文档字符串说明其用途和更新时机，这在团队协作中尤其重要。

**关于节点设计的最佳实践**：每个节点函数应该遵循单一职责原则，理想情况下不超过 30 行核心逻辑。如果一个节点的功能过于复杂，考虑把它拆成一个子图。节点函数应该是纯函数风格的——除了读写状态之外，不应该有其他副作用（如修改全局变量、直接写数据库）。所有的外部交互（API 调用、数据库操作、文件 I/O）都应该在节点内部完成并通过异常捕获来保证健壮性。对于 LLM 节点，一定要做输出解析的容错处理——LLM 返回的格式永远不能百分之百信任，始终要有 fallback 逻辑。

**关于边路由的最佳实践**：条件路由函数的返回值必须严格匹配 path_map 中定义的 key，最安全的做法是用枚举类型或常量来管理这些路由键，避免手写字符串导致的拼写错误。不要在路由函数里执行耗时的计算或外部调用——路由决策应该是轻量级的、确定性的，基于状态中已有的数据就能做出的判断。如果路由逻辑确实很复杂（比如需要 LLM 来辅助决策），把这个决策过程抽取为一个独立的节点，而不是塞进路由函数里。

**关于子图使用的最佳实践**：只有当一个逻辑块包含 3 个及以上节点、且可能在多处被复用时才值得抽取为子图。父子图之间的状态映射要显式且明确，依赖隐式字段名匹配是很多难以排查的 bug 的根源。每个子图都应该能够独立运行和独立测试——这是衡量子图划分是否合理的黄金标准。嵌套层数建议不超过 3 层，过深的嵌套会增加调试难度和状态追踪的复杂度。

## 性能考量

最后我们来谈谈性能这个容易被忽视但又至关重要的话题。LangGraph 本身的执行引擎是非常轻量的——节点函数之间的调度开销通常在微秒级别，状态合并的开销取决于状态的大小和复杂度。真正影响性能的瓶颈几乎总是在节点函数内部的业务逻辑上，特别是涉及 LLM 调用和外部 I/O 的地方。

第一个性能要点是**减少不必要的 LLM 调用**。每次 LLM 调动不仅花费金钱（token 费用），还有显著的时间延迟（通常几百毫秒到几秒）。在设计图结构时，应该尽可能用普通的逻辑节点来做那些不需要 AI 判断的工作（如数据格式化、正则匹配、规则过滤），只在真正需要语义理解、创意生成或复杂推理的地方才引入 LLM 节点。比如在前面的文章处理管道中，标题提取、内容清洗、分段都可以用规则完成，只有摘要生成这种需要语义理解的步骤才值得调用 LLM。

第二个要点是**利用 checkpointing 来避免重复计算**。当你的图包含耗时的节点（如 LLM 调用、大数据处理）并且配置了持久化检查点时，如果某次执行在中途失败了，你不需要从头开始重跑——可以从最近的 checkpoint 恢复，跳过已经成功完成的节点。这在长流程（包含 10+ 个节点的图）中能节省大量的时间和成本。

第三个要点是**关注状态的大小**。每次节点执行完毕后，LangGraph 都会把该节点的输出合并到全局状态中。如果你的状态中包含了大量数据（比如完整的文档内容、长对话历史），每一次状态合并都会有内存复制和序列化的开销。一个实用的技巧是把大型数据存储在外部（数据库、对象存储），状态中只保留引用标识。

第四个要点是**流式输出的合理使用**。对于面向用户的交互式应用，使用 `stream_mode="messages"` 可以让用户在 LLM 还没生成完完整回复时就看到部分输出，大幅提升感知响应速度。但对于后端批处理任务，使用普通的 `invoke()` 往往更简单也更高效，除非你需要实时监控进度。

总的来说，LangGraph 的性能优化思路和传统的后端性能优化是一致的：减少不必要的计算、避免重复工作、关注 I/O 瓶颈、合理使用缓存。只不过在 LangGraph 的语境下，这些原则需要落实到图结构的设计和节点函数的实现上。
