# 2. 核心模块实现

## 从设计到代码：搭建研究引擎的骨架

上一节我们完成了 DeepResearch 项目的需求分析和架构设计，明确了系统的功能范围、非功能约束、整体架构和图拓扑结构。现在，让我们把这些设计转化为可运行的代码。这一节将聚焦于最核心的部分——状态定义、节点函数实现以及图的组装——这三者构成了 LangGraph 应用的骨架，也是理解整个系统运作机制的关键。

### 状态设计：三层模型在研究场景中的落地

在第二章中我们详细讨论过状态设计的最佳实践，其中最重要的原则是**分层设计**——把不同生命周期的数据放在不同的层中，避免状态膨胀和耦合。对于研究助手这个场景，我们可以把这个原则具体化为以下三层：

```python
from typing import TypedDict, Annotated, List, Optional, Any, Dict
import operator
from datetime import datetime
from dataclasses import dataclass, field
import enum

class ResearchStatus(str, enum.Enum):
    PLANNING = "planning"
    RESEARCHING = "researching"
    EVALUATING = "evaluating"
    REPORTING = "reporting"
    COMPLETED = "completed"
    FAILED = "failed"

class SourceType(str, enum.Enum):
    ACADEMIC_PAPER = "academic_paper"
    OFFICIAL_DOC = "official_doc"
    NEWS_ARTICLE = "news_article"
    BLOG_POST = "blog_post"
    WIKIPEDIA = "wikipedia"
    FORUM = "forum"
    OTHER = "other"

# ========== 第一层：输入/输出层（生命周期：全程）==========

class ResearchInput(TypedDict):
    user_query: str
    max_rounds: int
    max_sources: int
    depth_level: str
    required_source_types: List[str]
    excluded_domains: List[str]
    output_format: str
    language: str

# ========== 第二层：任务上下文层（生命周期：单次研究任务）==========

class TaskContext(TypedDict):
    plan_id: str
    started_at: str
    current_round: int
    current_focus_index: int
    total_searches_performed: int
    total_sources_collected: int
    total_facts_extracted: int
    token_usage: Dict[str, int]
    estimated_cost_usd: float
    rounds_without_new_info: int

# ========== 第三层：研究成果层（生命周期：持续累积）==========

@dataclass
class Fact:
    subject: str
    predicate: str
    object_value: str
    confidence: float
    source_urls: List[str] = field(default_factory=list)
    extracted_at: str = field(default_factory=lambda: datetime.now().isoformat())
    round_number: int = 0
    sub_question: str = ""
    
    def to_dict(self) -> dict:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object_value": self.object_value,
            "confidence": self.confidence,
            "source_urls": self.source_urls,
            "extracted_at": self.extracted_at,
            "round_number": self.round_number,
            "sub_question": self.sub_question
        }

@dataclass  
class Source:
    url: str
    title: str
    source_type: SourceType
    content_summary: str
    full_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    quality_score: float = 0.0
    relevance_score: float = 0.0
    collected_in_round: int = 0
    
    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "title": self.title,
            "source_type": self.source_type.value if isinstance(self.source_type, SourceType) else self.source_type,
            "content_summary": self.content_summary[:500],
            "full_content_length": len(self.full_content),
            "metadata": self.metadata,
            "quality_score": self.quality_score,
            "relevance_score": self.relevance_score,
            "collected_in_round": self.collected_in_round
        }

# ========== 主状态：整合三层 ==========

class ResearchState(TypedDict):
    # 第一层：I/O
    input_config: ResearchInput
    
    # 第二层：任务上下文
    task_context: TaskContext
    
    # 研究计划
    topic: str
    sub_questions: List[str]
    search_strategies: List[Dict[str, Any]]
    
    # 第三层：持续累积的研究成果
    collected_sources: Annotated[List[Dict], operator.add]
    extracted_facts: Annotated[List[Dict], operator.add]
    
    # 当前轮次的临时数据
    current_search_results: List[Dict]
    current_facts_batch: List[Dict]
    
    # 评估结果
    coverage_map: Dict[str, float]
    sufficiency_verdict: str
    stuck_reason: str
    
    # 最终输出
    final_report: str
    report_metadata: Dict[str, Any]
    
    # 运行时状态
    status: ResearchStatus
    research_log: Annotated[List[str], operator.add]
```

这个状态设计的层次感非常清晰：

**第一层 `ResearchInput`** 存储的是用户提供的配置信息——查询内容、最大搜索轮次、最大信息源数量、深度级别等。这些数据在整个研究过程中都是只读的，不会改变。

**第二层 `TaskContext`** 记录的是当前任务的执行进度——第几轮了、搜索了多少次、收集了多少源、花了多少 Token 和费用、连续几轮没有新发现等。这些数据会随着研究的推进不断更新，但在任务结束后就没有意义了。

**第三层是两个用 `Annotated + operator.add` 标记的累加列表**：`collected_sources` 和 `extracted_facts`。每执行一轮搜索-提取循环，新的信息源和事实就会被追加到这些列表中。这是整个系统最有价值的产出物——它们最终会被报告生成器消费，转化为最终的研究报告。

另外还有几个重要的字段值得注意：
- `coverage_map` 是一个字典，记录每个子问题的信息覆盖程度（0.0 到 1.0），用于判断是否需要继续搜索
- `sufficiency_verdict` 是评估节点输出的判断结果（"continue" / "sufficient" / "stuck"）
- `research_log` 同样使用累加模式，记录每一步的操作日志，方便调试和审计

### 节点函数实现：从解析到报告的完整链路

有了状态定义之后，接下来就是实现各个节点的处理函数。每个节点接收完整的 `ResearchState` 作为输入，处理后返回一个包含更新字段的字典。让我们按执行顺序逐一实现。

#### 节点一：parse_query —— 解析研究查询

这是图的入口节点，负责把用户的自然语言查询转化为结构化的研究意图。它的核心任务有三个：提取主题、识别约束条件、估算复杂度。

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

query_parser_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

PARSE_QUERY_SYSTEM_PROMPT = """你是一个专业的研究助手分析师。你的任务是分析用户的研究查询，并输出结构化的分析结果。
请严格按照 JSON 格式返回，不要添加任何额外的文字。

你需要分析的内容包括：
1. core_topic: 核心研究主题（一句话概括）
2. key_dimensions: 关键研究维度/角度列表（3-6个）
3. constraints: 用户提到的约束条件（时间范围、地域、特定方面等）
4. complexity: 复杂度评估 (simple/medium/complex)
5. suggested_sub_questions: 建议的子问题列表（4-8个）
6. estimated_rounds: 建议的搜索轮次（2-5）
7. potential_source_types: 可能需要的信息源类型"""

def parse_query_node(state: ResearchState) -> dict:
    query = state["input_config"]["user_query"]
    
    messages = [
        SystemMessage(content=PARSE_QUERY_SYSTEM_PROMPT),
        HumanMessage(content=f"请分析以下研究查询：\n\n{query}")
    ]
    
    response = query_parser_llm.invoke(messages)
    
    try:
        import json
        analysis = json.loads(response.content.strip())
    except json.JSONDecodeError:
        analysis = {
            "core_topic": query,
            "key_dimensions": [query],
            "constraints": [],
            "complexity": "medium",
            "suggested_sub_questions": [f"关于 {query} 的基本情况"],
            "estimated_rounds": 3,
            "potential_source_types": ["news_article", "official_doc", "blog_post"]
        }
    
    max_rounds = state["input_config"].get("max_rounds", analysis.get("estimated_rounds", 3))
    depth = state["input_config"].get("depth_level", analysis.get("complexity", "medium"))
    
    log_entry = f"[Round 0] 解析查询完成。主题: {analysis['core_topic']}, 维度数: {len(analysis['key_dimensions'])}, 复杂度: {analysis['complexity']}"
    
    return {
        "topic": analysis["core_topic"],
        "sub_questions": analysis.get("suggested_sub_questions", []),
        "task_context": {
            **state.get("task_context", {}),
            "current_round": 0,
            "total_searches_performed": 0,
            "total_sources_collected": 0,
            "total_facts_extracted": 0,
            "token_usage": {"input": 0, "output": 0},
            "estimated_cost_usd": 0.0,
            "rounds_without_new_info": 0,
            "started_at": datetime.now().isoformat()
        },
        "status": ResearchStatus.PLANNING,
        "coverage_map": {sq: 0.0 for sq in analysis.get("suggested_sub_questions", [])},
        "research_log": [log_entry],
        "search_strategies": [
            {
                "sub_question": sq,
                "keywords": generate_keywords_for_subquestion(sq, analysis["key_dimensions"]),
                "source_types": analysis.get("potential_source_types", ["news_article", "official_doc"]),
                "priority": i + 1
            }
            for i, sq in enumerate(analysis.get("suggested_sub_questions", []))
        ]
    }

def generate_keywords_for_subquestion(sub_question: str, dimensions: list) -> list:
    base_keywords = sub_question.split()
    dimension_keywords = []
    for dim in dimensions:
        if any(word in sub_question.lower() for word in dim.lower().split()):
            dimension_keywords.extend(dim.split())
    return list(set(base_keywords + dimension_keywords))[:8]
```

`parse_query_node` 的实现思路是这样的：首先用 LLM 对用户的查询进行结构化分析，得到核心主题、关键维度、子问题建议等信息；然后把 LLM 的输出解析为 Python 字典（这里做了容错处理——如果 LLM 返回的不是合法 JSON 就用默认值兜底）；最后初始化 `task_context`、`coverage_map`、`search_strategies` 等字段，并写入第一条研究日志。

一个容易忽略但很重要的细节是 **`search_strategies` 的生成**。我们不是简单地拿子问题当搜索关键词，而是通过 `generate_keywords_for_subquestion` 函数结合全局的关键维度来生成更丰富的关键词组合。比如子问题是"大语言模型的安全风险"，而关键维度包含"对抗攻击"、"隐私泄露"、"偏见"，那么生成的关键词就会包含这些维度词，从而让后续的搜索更加精准。

#### 节点二：create_plan —— 生成并优化研究计划

这个节点负责基于解析结果生成一份详细的可执行研究计划，并在必要时对计划进行调整和优化。

```python
PLANNER_SYSTEM_PROMPT = """你是一个资深的研究方法论专家。根据给定的研究主题和子问题，制定一份详细的研究执行计划。

对于每个子问题，请提供：
1. primary_keywords: 主要搜索关键词（3-5个）
2. secondary_keywords: 辅助/扩展关键词（3-5个）  
3. preferred_source_types: 首选的信息源类型及优先级排序
4. expected_findings: 预期可能找到什么类型的信息
5. minimum_sources_needed: 至少需要多少个独立来源才能得出可靠结论

返回格式为 JSON 数组，每个元素对应一个子问题的策略。"""

def create_plan_node(state: ResearchState) -> dict:
    topic = state["topic"]
    sub_questions = state["sub_questions"]
    existing_strategies = state.get("search_strategies", [])
    
    if existing_strategies and len(existing_strategies) >= len(sub_questions):
        log_entry = "[Round 0] 研究计划已存在，跳过重新生成"
        return {"research_log": [log_entry]}
    
    prompt = f"""研究主题：{topic}

子问题列表：
{chr(10).join(f'{i+1}. {q}' for i, q in enumerate(sub_questions))}

请为每个子问题制定详细的搜索策略。"""
    
    messages = [
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ]
    
    response = query_parser_llm.invoke(messages)
    
    import json
    try:
        strategies = json.loads(response.content.strip())
    except:
        strategies = [
            {
                "primary_keywords": q.split()[:4],
                "secondary_keywords": [],
                "preferred_source_types": ["news_article", "official_doc"],
                "expected_findings": f"关于 {q} 的相关信息",
                "minimum_sources_needed": 3
            }
            for q in sub_questions
        ]
    
    enriched_strategies = []
    for i, (sq, strat) in enumerate(zip(sub_questions, strategies)):
        enriched = {
            "sub_question": sq,
            "index": i,
            "primary_keywords": strat.get("primary_keywords", sq.split()[:4]),
            "secondary_keywords": strat.get("secondary_keywords", []),
            "preferred_source_types": strat.get("preferred_source_types", ["news_article"]),
            "expected_findings": strat.get("expected_findings", ""),
            "minimum_sources_needed": strat.get("minimum_sources_needed", 3),
            "sources_found_so_far": 0,
            "status": "pending"
        }
        enriched_strategies.append(enriched)
    
    log_entry = (
        f"[Round 0] 研究计划生成完成。共 {len(enriched_strategies)} 个子问题，"
        f"预计总搜索轮次: {state['input_config'].get('max_rounds', 3)}"
    )
    
    return {
        "search_strategies": enriched_strategies,
        "status": ResearchStatus.RESEARCHING,
        "research_log": [log_entry]
    }
```

`create_plan_node` 的逻辑相对直观：它调用 LLM 为每个子问题生成详细的搜索策略（包括主关键词、辅助关键词、首选来源类型、预期发现、最少所需来源数），然后把这些策略与已有的子问题列表进行配对和丰富化，最终输出完整的 `search_strategies` 列表。

注意其中的一个优化细节：**如果 `search_strategies` 已经存在且数量足够，就直接跳过重新生成**。这在调整策略后重新进入 create_plan 节点的场景下很有用——避免重复消耗 Token 去生成相同的计划。

#### 节点三：select_focus —— 智能选择研究焦点

当研究循环决定要继续时，我们需要确定下一轮应该聚焦于哪个子问题。这不是简单的轮流坐庄，而是应该基于当前的**信息覆盖率**来做智能决策——覆盖率最低的子问题最需要补充信息。

```python
FOCUS_SELECTOR_PROMPT = """你是一个研究策略协调器。给定当前的研究进展信息，请选择下一轮应该重点研究的子问题。

考虑因素：
1. 信息覆盖率最低的子问题优先
2. 如果某个子问题已经有足够多的来源（>= minimum_sources_needed），可以暂时跳过
3. 优先选择那些"预期有高价值发现"的方向
4. 避免连续多轮聚焦于同一个子问题（除非它确实很薄弱）

返回格式：{"selected_index": 数字, "reason": "选择原因"}"""

def select_focus_node(state: ResearchState) -> dict:
    strategies = state["search_strategies"]
    coverage_map = state["coverage_map"]
    task_ctx = state["task_context"]
    current_round = task_ctx["current_round"]
    
    candidates = []
    for i, strat in enumerate(strategies):
        coverage = coverage_map.get(strat["sub_question"], 0.0)
        sources_found = strat.get("sources_found_so_far", 0)
        min_needed = strat.get("minimum_sources_needed", 3)
        
        if sources_found >= min_needed and coverage >= 0.7:
            continue
        
        score = (1.0 - coverage) * 50 + (min(1.0, sources_found / min_needed)) * 20
        candidates.append({
            "index": i,
            "sub_question": strat["sub_question"],
            "coverage": coverage,
            "sources_found": sources_found,
            "score": score
        })
    
    if not candidates:
        selected_idx = 0
        reason = "所有子问题都已达到目标覆盖率，默认选择第一个"
    else:
        candidates.sort(key=lambda x: x["score"], reverse=True)
        selected = candidates[0]
        selected_idx = selected["index"]
        reason = (
            f"选择了 '{selected['sub_question']}'，"
            f"覆盖率仅 {selected['coverage']:.1%}，"
            f"已找到 {selected['sources_found']} 个来源"
        )
    
    log_entry = f"[Round {current_round + 1}] 选择焦点: 子问题 #{selected_idx + 1} - {strategies[selected_idx]['sub_question']}. 原因: {reason}"
    
    return {
        "task_context": {
            **task_ctx,
            "current_focus_index": selected_idx,
            "current_round": current_round + 1
        },
        "research_log": [log_entry]
    }
```

`select_focus_node` 实现了一个基于评分的焦点选择算法。它会遍历所有子问题，计算每个候选的"优先级分数"——公式是 `(1 - 覆盖率) × 50 + (已找来源 / 所需来源) × 20`。这个公式的含义是：覆盖率越低得分越高（越需要关注），但同时已经找到了一些来源的也会获得一定加分（说明这个方向是有信息的）。然后选出分数最高的那个作为下一轮的焦点。

当然，在实际生产环境中，这个选择逻辑还可以更加复杂——比如考虑子问题之间的依赖关系（某些问题需要在其他问题有答案后才能深入研究）、考虑历史选择的分布（避免总是选前几个）、甚至可以让 LLM 参与决策。但对于我们的教程来说，这个基于规则的版本已经足以展示核心思想了。

#### 节点四：search_sources —— 多源并行搜索

这是系统中对外部依赖最强的节点，负责实际调用搜索引擎 API 来获取候选信息源列表。为了提高效率，我们应该支持多种搜索引擎的并行调用和结果合并。

```python
import httpx
import asyncio
from typing import List, Dict, Any

class WebSearchTool:
    def __init__(self, api_key: str = None, provider: str = "tavily"):
        self.api_key = api_key
        self.provider = provider
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def search(self, query: str, num_results: int = 10) -> List[Dict]:
        if self.provider == "tavily":
            return await self._search_tavily(query, num_results)
        elif self.provider == "serpapi":
            return await self._search_serpapi(query, num_results)
        else:
            return await self._search_mock(query, num_results)
    
    async def _search_tavily(self, query: str, num_results: int) -> List[Dict]:
        url = "https://api.tavily.com/search"
        payload = {
            "api_key": self.api_key,
            "query": query,
            "max_results": min(num_results, 20),
            "include_answer": False,
            "include_raw_content": True,
            "include_images": False
        }
        try:
            resp = await self.client.post(url, json=payload)
            data = resp.json()
            results = []
            for item in data.get("results", []):
                results.append({
                    "url": item.get("url", ""),
                    "title": item.get("title", ""),
                    "snippet": item.get("content", "")[:500],
                    "score": item.get("score", 0),
                    "published_date": item.get("published_date", "")
                })
            return results
        except Exception as e:
            print(f"Tavily 搜索失败: {e}")
            return []
    
    async def _search_mock(self, query: str, num_results: int) -> List[Dict]:
        mock_results = [
            {
                "url": f"https://example.com/article-{i+1}",
                "title": f"{query} - 深度分析文章 {i+1}",
                "snippet": f"这是一篇关于 {query} 的详细分析文章，涵盖了最新的发展趋势和实践案例...",
                "score": 0.9 - i * 0.05,
                "published_date": "2025-01-15"
            }
            for i in range(min(num_results, 8))
        ]
        return mock_results
    
    async def close(self):
        await self.client.aclose()

search_tool = WebSearchTool(provider="mock")

async def execute_search_round(
    primary_keywords: List[str],
    secondary_keywords: List[str],
    source_types: List[str],
    max_sources: int
) -> List[Dict]:
    all_results = []
    
    queries_to_try = [
        " ".join(primary_keywords[:4]),
        " ".join(primary_keywords[:3] + secondary_keywords[:2]),
        " ".join([primary_keywords[0]] + secondary_keywords[:3]) if secondary_keywords else primary_keywords[0]
    ]
    
    tasks = [search_tool.search(q, max_sources // len(queries_to_try) + 2) for q in queries_to_try]
    results_lists = await asyncio.gather(*tasks, return_exceptions=True)
    
    seen_urls = set()
    for results in results_lists:
        if isinstance(results, Exception):
            continue
        for r in results:
            if r["url"] not in seen_urls and len(all_results) < max_sources:
                r["source_type"] = guess_source_type(r["url"])
                all_results.append(r)
                seen_urls.add(r["url"])
    
    return all_results

def guess_source_type(url: str) -> str:
    url_lower = url.lower()
    if any(domain in url_lower for domain in ["arxiv.org", "semanticscholar", "ieee", "acm.org"]):
        return "academic_paper"
    elif any(domain in url_lower for domain in ["wikipedia.org", "wiki"]):
        return "wikipedia"
    elif any(domain in url_lower for domain in [".gov", ".org", "official"]):
        return "official_doc"
    elif any(domain in url_lower for domain in ["reddit", "stack overflow", "forum"]):
        return "forum"
    elif any(domain in url_lower for domain in ["blog.medium", "substack", ".io/blog"]):
        return "blog_post"
    else:
        return "news_article"
```

`search_sources` 节点的核心逻辑封装在 `execute_search_round` 函数中。它的工作流程如下：

1. **构建多个搜索查询**：不只是用一个关键词组合去搜，而是生成 3 个不同侧重的查询——纯主关键词、主+辅混合、主首词+辅关键词。这能增加搜索结果的多样性，避免因为单一查询的偏差而遗漏重要信息。

2. **并行执行搜索**：用 `asyncio.gather` 同时发起多个搜索请求，而不是串行等待。假设每个请求需要 2 秒，3 个串行就是 6 秒，并行只需要约 2 秒。

3. **结果去重**：通过 `seen_urls` 集合确保同一个 URL 不会被重复收录。这是一个常见的陷阱——不同的关键词组合可能会返回相同的结果。

4. **猜测来源类型**：`guess_source_type` 函数通过 URL 的域名特征来推断来源类型（学术论文、维基百科、官方文档、论坛、博客、新闻）。虽然不如人工标注准确，但作为初步分类已经够用了。

为了演示目的，代码中使用了一个 `_search_mock` 方法来模拟搜索结果。在生产环境中，你需要替换为真实的 Tavily 或 SerpAPI 调用。

现在我们把搜索逻辑包装成正式的 LangGraph 节点函数：

```python
async def search_sources_node(state: ResearchState) -> dict:
    task_ctx = state["task_context"]
    focus_idx = task_ctx["current_focus_index"]
    strategy = state["search_strategies"][focus_idx]
    max_sources = state["input_config"]["max_sources"]
    
    primary_kw = strategy.get("primary_keywords", [])
    secondary_kw = strategy.get("secondary_keywords", [])
    preferred_types = strategy.get("preferred_source_types", [])
    
    round_budget = min(max_sources // task_ctx["current_round"] + 3, 10)
    
    search_results = await execute_search_round(
        primary_keywords=primary_kw,
        secondary_keywords=secondary_kw,
        source_types=preferred_types,
        max_sources=round_budget
    )
    
    new_task_ctx = dict(task_ctx)
    new_task_ctx["total_searches_performed"] += 1
    
    updated_strategies = state["search_strategies"].copy()
    updated_strategies[focus_idx] = {
        **updated_strategies[focus_idx],
        "sources_found_so_far": updated_strategies[focus_idx].get("sources_found_so_far", 0) + len(search_results)
    }
    
    log_entry = (
        f"[Round {task_ctx['current_round']}] 搜索完成。"
        f"焦点: '{strategy['sub_question']}'，"
        f"找到 {len(search_results)} 个候选来源"
    )
    
    return {
        "current_search_results": search_results,
        "task_context": new_task_ctx,
        "search_strategies": updated_strategies,
        "research_log": [log_entry]
    }
```

注意这个节点是一个 **async 函数**——因为它内部调用了异步的搜索 API。LangGraph 完全支持 async 节点函数，在编译图时会自动处理异步调度。

#### 节点五：extract_facts —— 从原始内容中提炼知识

搜索到了候选信息源之后，下一步是从中提取结构化的事实。这是从"原始数据"到"可用知识"的关键转化步骤。

```python
EXTRACTOR_SYSTEM_PROMPT = """你是一个专业的信息提取专家。从给定的文本中提取关键事实和信息点。

提取规则：
1. 只提取明确陈述的事实、数据、观点，不要推断或臆测
2. 每个事实用 Subject-Predicate-Object 格式表示
3. 标注置信度（0.5-1.0）：直接引用的高，推测的低
4. 区分事实(fact)和观点(opinion)，在 predicate 中体现
5. 忽略广告、导航元素等无关内容

返回 JSON 数组格式：
[
  {
    "subject": "...",
    "predicate": "...", 
    "object_value": "...",
    "confidence": 0.9,
    "context": "原文中的相关句子"
  }
]"""

fact_extractor_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

async def fetch_and_extract(source_info: Dict, sub_question: str) -> List[Dict]:
    url = source_info.get("url", "")
    title = source_info.get("title", "")
    snippet = source_info.get("snippet", "")
    
    content = await fetch_page_content(url)
    if not content or len(content) < 50:
        content = snippet
    
    truncated_content = content[:4000]
    
    prompt = f"""请从以下文本中提取与 "{sub_question}" 相关的关键事实。

标题：{title}
URL：{url}
文本内容：
{truncated_content}
"""
    
    messages = [
        SystemMessage(content=EXTRACTOR_SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ]
    
    response = fact_extractor_llm.invoke(messages)
    
    import json
    try:
        facts_raw = json.loads(response.content.strip())
    except:
        facts_raw = [{"subject": title, "predicate": "discusses", "object_value": content[:200], "confidence": 0.6, "context": ""}]
    
    enriched_facts = []
    for f in facts_raw:
        enriched = Fact(
            subject=f.get("subject", ""),
            predicate=f.get("predicate", ""),
            object_value=f.get("object_value", ""),
            confidence=min(1.0, max(0.5, f.get("confidence", 0.7))),
            source_urls=[url],
            sub_question=sub_question
        )
        enriched_facts.append(enriched.to_dict())
    
    return enriched_facts

async def fetch_page_content(url: str) -> str:
    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            resp = await client.get(url, headers={
                "User-Agent": "DeepResearch/1.0 (Research Assistant)"
            })
            if resp.status_code == 200:
                html = resp.text
                text = simple_html_to_text(html)
                return text
    except Exception as e:
        print(f"抓取页面失败 [{url}]: {e}")
    return ""

def simple_html_to_text(html: str) -> str:
    import re
    text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'&nbsp;', ' ', text)
    text = unescape_html(text)
    return text.strip()

def unescape_html(text: str) -> str:
    replacements = {
        '&amp;': '&', '&lt;': '<', '&gt;': '>',
        '&quot;': '"', '&#39;': "'", '&nbsp;': ' '
    }
    for entity, char in replacements.items():
        text = text.replace(entity, char)
    return text

async def extract_facts_node(state: ResearchState) -> dict:
    search_results = state["current_search_results"]
    task_ctx = state["task_context"]
    focus_idx = task_ctx["current_focus_index"]
    sub_question = state["search_strategies"][focus_idx]["sub_question"]
    
    all_new_facts = []
    all_new_sources = []
    
    extraction_tasks = [
        fetch_and_extract(src, sub_question)
        for src in search_results[:8]
    ]
    
    batches = await asyncio.gather(*extraction_tasks, return_exceptions=True)
    
    for result, src in zip(batches, search_results[:8]):
        if isinstance(result, Exception):
            continue
        
        facts = result
        all_new_facts.extend(facts)
        
        source_obj = Source(
            url=src["url"],
            title=src["title"],
            source_type=SourceType(src.get("source_type", "other")),
            content_summary=src.get("snippet", ""),
            full_content="",
            metadata={"published_date": src.get("published_date", ""), "score": src.get("score", 0)},
            collected_in_round=task_ctx["current_round"]
        )
        all_new_sources.append(source_obj.to_dict())
    
    new_task_ctx = dict(task_ctx)
    new_task_ctx["total_sources_collected"] += len(all_new_sources)
    new_task_ctx["total_facts_extracted"] += len(all_new_facts)
    
    log_entry = (
        f"[Round {task_ctx['current_round']}] 提取完成。"
        f"处理了 {len(search_results[:8])} 个来源，"
        f"提取了 {len(all_new_facts)} 条事实"
    )
    
    return {
        "current_facts_batch": all_new_facts,
        "collected_sources": all_new_sources,
        "extracted_facts": all_new_facts,
        "task_context": new_task_ctx,
        "research_log": [log_entry]
    }
```

`extract_facts_node` 是另一个 async 节点，因为它涉及网络 I/O（抓取网页内容）和 LLM 调用（提取事实）。它的内部流程是：

1. 对本轮搜索到的每个候选信息源（最多 8 个），并行地执行 `fetch_and_extract`
2. `fetch_and_extract` 先尝试抓取网页全文（带超时和重定向处理），如果抓取失败就用搜索结果中的摘要片段代替
3. 把获取到的文本（截断到 4000 字符以内以控制 Token 消耗）发送给 LLM 进行事实提取
4. LLM 返回的是 SPO 格式的三元组数组，我们把它转换为 `Fact` dataclass 并序列化为字典
5. 同时创建 `Source` 对象记录元信息
6. 最后汇总所有新事实和新来源，更新统计计数器和日志

其中 `simple_html_to_text` 函数实现了一个简易的 HTML 清洗——去除 script/style 标签、去掉所有 HTML 标签、合并空白字符、解码 HTML 实体。在生产环境中你可能想用更成熟的库如 `trafilatura` 或 `readability-lxml`，但这个简化版足够展示原理。

#### 节点六：evaluate_findings —— 信息充足性判断与冲突检测

这是整个研究循环中最关键的决策节点——它决定了研究是该继续、该结束、还是该换方向。

```python
EVALUATOR_SYSTEM_PROMPT = """你是一个研究质量评估专家。给定当前的研究进展信息，判断是否应该继续搜索更多信息。

评估维度：
1. 覆盖完整性：每个子问题是否有足够的信息支撑？
2. 来源多样性：信息是否来自多种独立的可信来源？
3. 事实一致性：是否存在明显的矛盾或冲突？
4. 新信息增量：最近几轮是否还在发现有价值的新信息？

返回 JSON 格式：
{
    "verdict": "continue" | "sufficient" | "stuck",
    "confidence": 0.0-1.0,
    "reason": "详细理由",
    "coverage_update": {"子问题1": 0.0-1.0, ...},
    "conflicts_found": [...],
    "key_gaps_remaining": ["..."]
}"""

evaluator_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

def evaluate_findings_node(state: ResearchState) -> dict:
    task_ctx = state["task_context"]
    all_facts = state["extracted_facts"]
    all_sources = state["collected_sources"]
    strategies = state["search_strategies"]
    current_coverage = state["coverage_map"]
    max_rounds = state["input_config"]["max_rounds"]
    
    facts_summary = format_facts_for_evaluation(all_facts[-50:] if len(all_facts) > 50 else all_facts)
    sources_summary = format_sources_for_evaluation(all_sources[-20:] if len(all_sources) > 20 else all_sources)
    
    prompt = f"""研究主题：{state['topic']}

当前轮次：{task_ctx['current_round']} / 最大轮次 {max_rounds}
已收集来源数：{len(all_sources)}
已提取事实数：{len(all_facts)}
连续无新信息轮次：{task_ctx['rounds_without_new_info']}

子问题及当前覆盖率：
{chr(10).join(f'- {s[\"sub_question\"]}: {current_coverage.get(s[\"sub_question\"], 0):.1%}' for s in strategies)}

最近的提取事实（最近30条）：
{facts_summary}

最近的收集来源（最近15个）：
{sources_summary}

请评估当前研究进展并给出判断。"""
    
    messages = [
        SystemMessage(content=EVALUATOR_SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ]
    
    response = evaluator_llm.invoke(messages)
    
    import json
    try:
        evaluation = json.loads(response.content.strip())
    except:
        evaluation = {
            "verdict": "continue" if task_ctx["current_round"] < max_rounds else "sufficient",
            "confidence": 0.7,
            "reason": "评估解析失败，使用默认策略",
            "coverage_update": dict(current_coverage),
            "conflicts_found": [],
            "key_gaps_remaining": []
        }
    
    verdict = evaluation.get("verdict", "continue")
    
    if task_ctx["current_round"] >= max_rounds and verdict == "continue":
        verdict = "sufficient"
        evaluation["reason"] += "（已达最大轮次限制）"
    
    new_rounds_no_new = task_ctx["rounds_without_new_info"]
    new_facts_count = len(state.get("current_facts_batch", []))
    if new_facts_count == 0:
        new_rounds_no_new += 1
    else:
        new_rounds_no_new = 0
    
    if new_rounds_no_new >= 2 and verdict == "continue":
        verdict = "stuck"
        evaluation["reason"] = f"连续 {new_rounds_no_new} 轮无新信息，判定为陷入僵局"
    
    new_coverage = current_coverage.copy()
    coverage_update = evaluation.get("coverage_update", {})
    for sq, cov in coverage_update.items():
        if sq in new_coverage:
            new_coverage[sq] = max(new_coverage[sq], cov)
    
    conflicts = evaluation.get("conflicts_found", [])
    
    log_entry = (
        f"[Round {task_ctx['current_round']}] 评估完成。"
        f"判定: {verdict.upper()}，置信度: {evaluation.get('confidence', 0):.1%}。"
        f"原因: {evaluation.get('reason', '')[:100]}"
    )
    
    return {
        "sufficiency_verdict": verdict,
        "coverage_map": new_coverage,
        "task_context": {
            **task_ctx,
            "rounds_without_new_info": new_rounds_no_new
        },
        "research_log": [log_entry],
        "current_conflicts": conflicts
    }

def format_facts_for_evaluation(facts: List[Dict]) -> str:
    lines = []
    for f in facts[-30:]:
        lines.append(f"- {f.get('subject', '?')} {f.get('predicate', '?')} {f.get('object_value', '?')} (置信度: {f.get('confidence', 0)})")
    return chr(10).join(lines) if lines else "(暂无事实)"

def format_sources_for_evaluation(sources: List[Dict]) -> str:
    lines = []
    for s in sources[-15:]:
        lines.append(f"- [{s.get('source_type', '?')}] {s.get('title', '?')} ({s.get('url', '?')[:60]}...)")
    return chr(10).join(lines) if lines else "(暂无来源)"
```

`evaluate_findings_node` 是整个图中逻辑最复杂的节点之一。让我拆解它的几个关键决策点：

**LLM 辅助评估**：我们把当前的研究状态（轮次、已收集的来源和事实、各子问题的覆盖率、最近的新增内容）打包成一个精心构造的 prompt 发送给 LLM，让它给出综合判断。LLM 能做人类难以用简单规则表达的事情——比如判断"虽然子问题 A 有 80% 的覆盖率，但这些信息都来自同一类来源，缺乏多样性"。

**硬性安全阀**：无论 LLM 说"continue"，一旦达到 `max_rounds` 上限就强制切换为"sufficient"。这是防止无限循环的重要保护措施。

**僵局检测**：如果连续两轮都没有提取到新事实（`rounds_without_new_info >= 2`），就判定为"stuck"。这说明当前的搜索策略可能已经穷尽了可用信息，需要调整方向。

**覆盖率更新**：LLM 返回的 `coverage_update` 会与现有的 `coverage_map` 做 merge（取较大值），确保覆盖率只会上升不会下降。

#### 节点七：adjust_strategy —— 策略调整

当评估结果为 "stuck" 时，这个节点会被触发来尝试打破僵局。

```python
ADJUSTER_SYSTEM_PROMPT = """你是研究策略优化专家。当前研究陷入了僵局（连续多轮搜索未能获得有价值的新信息）。
请分析原因并提出调整方案。

可能的调整策略：
1. 扩展/更换关键词：当前的关键词可能太窄或太泛
2. 切换信息源类型：如果一直在搜新闻，试试学术论文或官方文档
3. 拆分或合并子问题：当前的问题粒度可能不合适
4. 尝试全新的角度：从相关领域或交叉学科寻找突破口
5. 降低标准接受已有信息：也许现有信息已经足够好了

返回 JSON 格式：
{
    "diagnosis": "僵局原因诊断",
    "adjustments": [{"strategy": "调整描述", "target_sub_question_index": 数字}],
    "new_keywords_to_try": [["关键词1", "关键词2"], ...],
    "should_add_new_subquestions": false,
    "new_subquestions_if_any": ["..."]
}"""

def adjust_strategy_node(state: ResearchState) -> dict:
    task_ctx = state["task_context"]
    strategies = state["search_strategies"]
    coverage_map = state["coverage_map"]
    
    weakest_idx = min(range(len(strategies)), 
                       key=lambda i: coverage_map.get(strategies[i]["sub_question"], 0))
    weakest_sq = strategies[weakest_idx]["sub_question"]
    
    prompt = f"""研究主题：{state['topic']}
当前轮次：{task_ctx['current_round']}
僵局原因：连续 {task_ctx['rounds_without_new_info']} 轮无新信息

最薄弱的子问题：{weakest_sq}（覆盖率: {coverage_map.get(weakest_sq, 0):.1%}）

当前搜索策略概览：
{chr(10).join(f'{i+1}. {s[\"sub_question\"]}: keywords={s.get(\"primary_keywords\", [])[:3]}, types={s.get(\"preferred_source_types\", [])}' for i, s in enumerate(strategies))}

已发现的冲突点：{state.get('current_conflicts', [])}

请提出具体的策略调整方案。"""
    
    messages = [
        SystemMessage(content=ADJUSTER_SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ]
    
    response = evaluator_llm.invoke(messages)
    
    import json
    try:
        adjustment = json.loads(response.content.strip())
    except:
        adjustment = {
            "diagnosis": "无法解析调整方案",
            "adjustments": [{"strategy": "扩大搜索关键词范围", "target_sub_question_index": weakest_idx}],
            "new_keywords_to_try": [[weakest_sq] + ["overview", "analysis", "latest"]],
            "should_add_new_subquestions": False,
            "new_subquestions_if_any": []
        }
    
    updated_strategies = strategies.copy()
    for adj in adjustment.get("adjustments", []):
        target_idx = adj.get("target_sub_question_index", weakest_idx)
        if 0 <= target_idx < len(updated_strategies):
            old_kw = updated_strategies[target_idx].get("primary_keywords", [])
            new_kw_list = adjustment.get("new_keywords_to_try", [[]])
            if new_kw_list and target_idx < len(new_kw_list):
                updated_strategies[target_idx]["primary_keywords"] = (
                    list(set(old_kw + new_kw_list[target_idx]))
                )
            
            old_types = updated_strategies[target_idx].get("preferred_source_types", [])
            if "academic_paper" not in old_types and len(old_types) > 0:
                updated_strategies[target_idx]["preferred_source_types"] = (
                    old_types + ["academic_paper"]
                )
    
    log_entry = (
        f"[Round {task_ctx['current_round']}] 策略调整。"
        f"原因: {adjustment.get('diagnosis', '未知')}。"
        f"调整了 {len(adjustment.get('adjustments', []))} 个子问题的搜索策略"
    )
    
    return {
        "search_strategies": updated_strategies,
        "task_context": {
            **task_ctx,
            "rounds_without_new_info": 0
        },
        "research_log": [log_entry],
        "sufficiency_verdict": "continue"
    }
```

`adjust_strategy_node` 的核心思路是：找出覆盖率最低的那个子问题（最可能是瓶颈所在），让 LLM 分析为什么在这个方向上找不到新信息，然后针对性地调整搜索策略——可能是扩大关键词范围、增加新的信息源类型、或者微调子问题的表述方式。调整完成后，把 `sufficiency_verdict` 重置为 `"continue"` 让循环继续下去。

### 图的组装：把所有节点串联起来

所有节点函数都实现完毕之后，最后一步就是把它们组装成一张完整可执行的图。

```python
from langgraph.graph import StateGraph, START, END

def build_research_graph():
    graph = StateGraph(ResearchState)
    
    graph.add_node("parse_query", parse_query_node)
    graph.add_node("create_plan", create_plan_node)
    graph.add_node("select_focus", select_focus_node)
    graph.add_node("search_sources", search_sources_node)
    graph.add_node("extract_facts", extract_facts_node)
    graph.add_node("evaluate_findings", evaluate_findings_node)
    graph.add_node("adjust_strategy", adjust_strategy_node)
    graph.add_node("generate_report", generate_report_node)
    
    graph.add_edge(START, "parse_query")
    graph.add_edge("parse_query", "create_plan")
    
    graph.add_conditional_edges(
        "create_plan",
        should_start_research,
        {
            "start": "select_focus",
            "end": END
        }
    )
    
    graph.add_edge("select_focus", "search_sources")
    graph.add_edge("search_sources", "extract_facts")
    graph.add_edge("extract_facts", "evaluate_findings")
    
    graph.add_conditional_edges(
        "evaluate_findings",
        lambda state: state.get("sufficiency_verdict", "continue"),
        {
            "continue": "select_focus",
            "sufficient": "generate_report",
            "stuck": "adjust_strategy"
        }
    )
    
    graph.add_edge("adjust_strategy", "select_focus")
    graph.add_edge("generate_report", END)
    
    return graph.compile()

def should_start_research(state: ResearchState) -> str:
    if not state.get("sub_questions"):
        return "end"
    return "start"

async def generate_report_node(state: ResearchState) -> dict:
    from datetime import datetime
    
    report = build_markdown_report(state)
    
    report_metadata = {
        "generated_at": datetime.now().isoformat(),
        "topic": state["topic"],
        "total_rounds": state["task_context"]["current_round"],
        "total_sources": len(state["collected_sources"]),
        "total_facts": len(state["extracted_facts"]),
        "sub_questions_count": len(state["sub_questions"]),
        "final_coverage": state["coverage_map"]
    }
    
    log_entry = f"[DONE] 报告生成完成。共 {report_metadata['total_rounds']} 轮，{report_metadata['total_sources']} 个来源，{report_metadata['total_facts']} 条事实"
    
    return {
        "final_report": report,
        "report_metadata": report_metadata,
        "status": ResearchStatus.COMPLETED,
        "research_log": [log_entry]
    }

def build_markdown_report(state: ResearchState) -> str:
    topic = state["topic"]
    sub_questions = state["sub_questions"]
    facts = state["extracted_facts"]
    sources = state["collected_sources"]
    coverage = state["coverage_map"]
    task_ctx = state["task_context"]
    
    lines = []
    lines.append(f"# 深度研究报告：{topic}\n")
    lines.append(f"> 由 DeepResearch 自动生成 | {task_ctx['started_at'][:10]}\n")
    
    lines.append("## 执行摘要\n")
    summary = generate_executive_summary(topic, facts, coverage)
    lines.append(summary)
    
    lines.append("\n## 研究方法\n")
    lines.append(f"- **研究轮次**: {task_ctx['current_round']} 轮")
    lines.append(f"- **信息来源**: 共 {len(sources)} 个独立来源")
    lines.append(f"- **提取事实**: 共 {len(facts)} 条结构化知识点")
    lines.append(f"- **子问题覆盖**: {', '.join(f'{sq}: {cov:.0%}' for sq, cov in coverage.items())}\n")
    
    lines.append("---\n")
    
    for i, sq in enumerate(sub_questions):
        lines.append(f"\n## {i+1}. {sq}\n")
        
        sq_facts = [f for f in facts if f.get("sub_question") == sq]
        if not sq_facts:
            lines.append("*（暂未找到充分信息）*\n")
            continue
        
        unique_subjects = {}
        for f in sq_facts:
            key = (f.get("subject", ""), f.get("predicate", ""))
            if key not in unique_subjects:
                unique_subjects[key] = f
        
        for (subj, pred), fact in list(unique_subjects.items())[:15]:
            lines.append(f"### {subj}\n")
            lines.append(f"- **{pred}**: {fact.get('object_value', '')}")
            lines.append(f"  - 置信度: {fact.get('confidence', 0):.0%}")
            if fact.get("source_urls"):
                lines.append(f"  - 来源: {', '.join(fact['source_urls'][:2])}")
            lines.append("")
    
    lines.append("\n---\n")
    lines.append("\n## 参考资料\n")
    seen_urls = set()
    for src in sources:
        if src.get("url") not in seen_urls:
            lines.append(f"- [{src.get('title', 'Untitled')}]({src.get('url', '#')}) "
                        f"[{src.get('source_type', '?')}]")
            seen_urls.add(src.get("url"))
    
    lines.append(f"\n---\n*报告由 DeepResearch v1.0 自动生成 | "
                f"研究耗时 {task_ctx['current_round']} 轮*")
    
    return "\n".join(lines)

def generate_executive_summary(topic: str, facts: List[Dict], coverage: Dict[str, float]) -> str:
    high_confidence_facts = [f for f in facts if f.get("confidence", 0) >= 0.8]
    
    key_findings = []
    subjects_seen = set()
    for f in high_confidence_facts[:20]:
        subj = f.get("subject", "")
        if subj not in subjects_seen:
            key_findings.append(f"- {subj} {f.get('predicate', '')} {f.get('object_value', '')}")
            subjects_seen.add(subj)
    
    avg_coverage = sum(coverage.values()) / len(coverage) if coverage else 0
    
    summary_parts = [
        f"本报告围绕 **{topic}** 这一主题进行了系统性研究。",
        f"经过多轮信息采集与分析，共收集 **{len(facts)}** 条关键事实，",
        f"信息来源于 **{len(set(s.get('url','') for f in facts for s in [f]))}** 个独立渠道。",
        f"各子问题的平均信息覆盖率达到 **{avg_coverage:.0%}**。\n",
        "**主要发现：**\n"
    ]
    summary_parts.extend(key_findings[:8])
    
    if avg_coverage < 0.7:
        summary_parts.append(f"\n> ⚠️ 部分子问题的信息覆盖尚不完全（平均 {avg_coverage:.0%}），建议进一步深入研究。\n")
    
    return "\n".join(summary_parts)

research_graph = build_research_graph()
```

图的组装过程清晰地反映了我们在架构设计中定义的拓扑结构：

1. **START → parse_query → create_plan**：线性启动阶段，解析查询并生成计划
2. **create_plan → select_focus | END**：如果没有子问题就直接结束（异常情况）
3. **select_focus → search_sources → extract_facts → evaluate_findings**：一轮完整的研究循环
4. **evaluate_findings → select_focus（continue）/ generate_report（sufficient）/ adjust_strategy（stuck）**：核心的条件分支
5. **adjust_strategy → select_focus**：调整后回到焦点选择重新开始循环
6. **generate_report → END**：最终输出

### 快速验证：端到端运行测试

代码写完了，最重要的是验证它能不能真正跑起来。下面是一个完整的端到端测试示例：

```python
import asyncio

async def run_research_test():
    initial_state: ResearchState = {
        "input_config": {
            "user_query": "2025年AI Agent的技术路线图和主要玩家分析",
            "max_rounds": 3,
            "max_sources": 15,
            "depth_level": "medium",
            "required_source_types": [],
            "excluded_domains": [],
            "output_format": "markdown",
            "language": "zh"
        },
        "task_context": {
            "plan_id": f"test_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "started_at": "",
            "current_round": 0,
            "current_focus_index": 0,
            "total_searches_performed": 0,
            "total_sources_collected": 0,
            "total_facts_extracted": 0,
            "token_usage": {"input": 0, "output": 0},
            "estimated_cost_usd": 0.0,
            "rounds_without_new_info": 0
        },
        "topic": "",
        "sub_questions": [],
        "search_strategies": [],
        "collected_sources": [],
        "extracted_facts": [],
        "current_search_results": [],
        "current_facts_batch": [],
        "coverage_map": {},
        "sufficiency_verdict": "",
        "stuck_reason": "",
        "final_report": "",
        "report_metadata": {},
        "status": ResearchStatus.PLANNING,
        "research_log": [],
        "current_conflicts": []
    }
    
    print("=" * 60)
    print("DeepResearch 端到端测试")
    print("=" * 60)
    print(f"查询: {initial_state['input_config']['user_query']}")
    print(f"最大轮次: {initial_state['input_config']['max_rounds']}")
    print("-" * 60)
    
    result = await research_graph.ainvoke(initial_state)
    
    print("-" * 60)
    print("研究完成！")
    print(f"状态: {result['status']}")
    print(f"总轮次: {result['task_context']['current_round']}")
    print(f"收集来源: {len(result['collected_sources'])}")
    print(f"提取事实: {len(result['extracted_facts'])}")
    print("-" * 60)
    
    print("\n=== 研究日志 ===")
    for log in result["research_log"]:
        print(log)
    
    print("\n=== 报告预览（前2000字符）===")
    print(result["final_report"][:2000])
    
    return result

if __name__ == "__main__":
    test_result = asyncio.run(run_research_test())
```

当你运行这段代码时（前提是你已经设置了 `OPENAI_API_KEY` 环境变量），你会看到类似这样的输出：

```
============================================================
DeepResearch 端到端测试
============================================================
查询: 2025年AI Agent的技术路线图和主要玩家分析
最大轮次: 3
------------------------------------------------------------
[Round 0] 解析查询完成。主题: AI Agent技术路线图与生态格局, 维度数: 5, 复杂度: medium
[Round 0] 研究计划生成完成。共 5 个子问题，预计总搜索轮次: 3
[Round 1] 选择焦点: 子问题 #1 - AI Agent的核心技术架构。原因: 覆盖率仅 0.0%, 已找到 0 个来源
[Round 1] 搜索完成。焦点: 'AI Agent的核心技术架构'，找到 8 个候选来源
[Round 1] 提取完成。处理了 8 个来源，提取了 24 条事实
[Round 1] 评估完成。判定: CONTINUE，置信度: 85%。原因: 核心技术架构维度信息不足...
[Round 2] 选择焦点: 子问题 #2 - 主要厂商和产品布局。原因: 覆盖率仅 5.0%...
[Round 2] 搜索完成。焦点: '主要厂商和产品布局'，找到 7 个候选来源
[Round 2] 提取完成。处理了 7 个来源，提取了 19 条事实
[Round 2] 评估完成。判定: SUFFICIENT，置信度: 90%。原因: 各子问题均有适度覆盖...
------------------------------------------------------------
研究完成！
状态: completed
总轮次: 2
收集来源: 15
提取事实: 43
------------------------------------------------------------

=== 报告预览 ===
# 深度研究报告：AI Agent技术路线图与生态格局

> 由 DeepResearch 自动生成 | 2025-04-06

## 执行摘要

本报告围绕 **AI Agent技术路线图与生态格局** 这一主题进行了系统性研究。
经过多轮信息采集与分析，共收集 **43** 条关键事实，
信息来源于 **15** 个独立渠道。
各子问题的平均信息覆盖率达到 **72%**。

**主要发现：**
- OpenAI 正在开发 Agent 平台...
- Anthropic 的 Claude 具备工具使用能力...
- Google 的 Project Astra ...
...

## 研究方法
- **研究轮次**: 2 轮
- **信息来源**: 共 15 个独立来源
- **提取事实**: 共 43 条结构化知识点
- **子问题覆盖**: 核心技术架构: 85%, 主要厂商: 78%, 商业模式: 65%, ...

---
```

到这里，DeepResearch 项目最核心的部分——状态设计和节点实现——就全部完成了。我们已经拥有了一个能够自主进行多轮迭代研究的完整系统。接下来的章节将进一步增强它的能力：加入更强大的多步推理机制、完善结果整合与可视化展示、以及最终的部署与扩展方案。
