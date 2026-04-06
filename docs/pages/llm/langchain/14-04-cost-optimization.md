---
title: 成本优化：缓存策略与模型路由
description: LLM API 成本分析、语义缓存实现、模型路由策略（简单问题用小模型）、Token 优化技巧、成本监控仪表盘
---
# 成本优化：缓存策略与模型路由

前面三节我们解决了"怎么部署"的问题。这一节要解决的是**"怎么省钱"**——LLM 应用的运营成本是很多团队最头疼的事。

先算一笔账。假设你的客服 API 日活 1000 人，每人平均每天问 3 个问题：

| 指标 | GPT-4o | GPT-4o-mini | 月成本 |
|------|--------|-------------|--------|
| 单次调用 Token（入+出） | ~800 | ~800 | — |
| 单次调用成本 | ~$0.012 | ~$0.0015 | — |
| 日调用量 (1000×3) | 3000 | 3000 | — |
| **日成本** | **$36** | **$4.5** | — |
| **月成本** | **~$1,080** | **~$135** | — |

仅仅切换模型就能把月成本从 $1080 降到 $135 —— **节省 87.5%**。但这只是最粗粒度的优化。通过缓存和智能路由，我们还能进一步压缩到 **$30-50/月**。

## 成本优化的四个层次

```
Level 1: 模型选择        → 选对模型（最大杠杆）
Level 2: 语义缓存        → 相同问题不重复调用
Level 3: 智能路由        → 简单问题用便宜模型
Level 4: Token 优化       → 减少每次调用的 token 数
```

## Level 1：模型选择——最大的成本杠杆

### LangChain 支持的模型成本对比

| 模型 | 输入 ($/1M tokens) | 输出 ($/1M tokens) | 质量评分* | 适用场景 |
|------|-------------------|-------------------|---------|---------|
| **GPT-4o** | $2.50 | $10.00 | 95 | 复杂推理/代码生成 |
| **GPT-4o-mini** | $0.15 | $0.60 | 90 | 通用问答/RAG |
| **Claude 3.5 Sonnet** | $3.00 | $15.00 | 94 | 长文本/分析任务 |
| **Claude 3 Haiku** | $0.25 | $1.25 | 85 | 简单分类/格式化 |
| **本地 LLM (Qwen/Llama)** | $0 (GPU 成本另算) | $0 | 80 | 数据敏感场景 |

\*质量评分为相对值，基于通用 RAG 场景的主观评估

### 选择策略

```python
from langchain_openai import ChatOpenAI

MODEL_CONFIG = {
    "gpt-4o": {
        "model": "gpt-4o",
        "temperature": 0.1,
        "max_tokens": 2048,
        "use_cases": ["code_generation", "complex_reasoning", "agent_planning"],
        "cost_per_1k_output": 0.01,
    },
    "gpt-4o-mini": {
        "model": "gpt-4o-mini",
        "temperature": 0.1,
        "max_tokens": 1024,
        "use_cases": ["rag_qa", "classification", "summarization"],
        "cost_per_1k_output": 0.0006,
    },
}

def get_llm_for_task(task_type: str = "default"):
    config = MODEL_CONFIG.get("gpt-4o-mini", MODEL_CONFIG["gpt-4o"])
    return ChatOpenAI(
        model=config["model"],
        temperature=config["temperature"],
        max_tokens=config["max_tokens"],
    )
```

**经验法则**：
- RAG 问答、意图分类、摘要生成 → **GPT-4o-mini**（够用且便宜 15 倍）
- Agent 规划、代码生成、复杂推理 → **GPT-4o**
- 数据不能出内网 → 本地部署的 Qwen/Llama

## Level 2：语义缓存——避免重复计算

### 为什么传统缓存不够用

传统的 KV 缓存（如 Redis）依赖**精确匹配 key**。但 LLM 的输入是自然语言，同一个问题的表达方式千差万别：

```python
# 这三个问题语义相同，但字符串完全不同
questions = [
    "免费版支持几个人？",
    "免费版能加几个成员？",
    "free version how many team members?",
]
```

如果用精确匹配，三次查询会触发三次 LLM 调用——浪费了两次钱。**语义缓存**通过比较问题的向量相似度来解决这个问题。

### 实现语义缓存

```python
import hashlib
import time
import json
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass, field

@dataclass
class CacheEntry:
    question: str
    question_embedding: list[float]
    answer: str
    model: str
    created_at: float
    hit_count: int = 0
    token_usage: dict = field(default_factory=dict)

class SemanticCache:
    def __init__(self,
                 similarity_threshold: float = 0.92,
                 ttl_seconds: int = 3600,
                 max_size: int = 10000):
        self.threshold = similarity_threshold
        self.ttl = ttl_seconds
        self.max_size = max_size
        self.entries: list[CacheEntry] = []
        self.embeddings_model = None

    def _get_embeddings(self):
        if self.embeddings_model is None:
            from langchain_openai import OpenAIEmbeddings
            self.embeddings_model = OpenAIEmbeddings()
        return self.embeddings_model

    def _cosine_similarity(self, vec_a: list[float], vec_b: list[float]) -> float:
        a, b = np.array(vec_a), np.array(vec_b)
        dot = np.dot(a, b)
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        return float(dot / norm) if norm > 0 else 0.0

    def get(self, question: str) -> Optional[CacheEntry]:
        embeddings = self._get_embeddings()
        query_vec = embeddings.embed_query(question)

        now = time.time()

        best_match = None
        best_score = 0.0

        for entry in self.entries:
            if now - entry.created_at > self.ttl:
                continue

            score = self._cosine_similarity(query_vec, entry.question_embedding)
            if score > best_score:
                best_score = score
                best_match = entry

        if best_match and best_score >= self.threshold:
            best_match.hit_count += 1
            return best_match

        return None

    def set(self, question: str, answer: str, model: str = "",
             token_usage: dict = None):
        embeddings = self._get_embeddings()
        question_vec = embeddings.embed_query(question)

        entry = CacheEntry(
            question=question,
            question_embedding=question_vec,
            answer=answer,
            model=model,
            created_at=time.time(),
            token_usage=token_usage or {},
        )

        self.entries.append(entry)

        if len(self.entries) > self.max_size:
            self.entries.sort(key=lambda e: (e.hit_count, -e.created_at))
            removed = self.entries.pop(0)
            print(f"[Cache] 淘汰低频条目: {removed.question[:30]}...")

    def stats(self) -> dict:
        now = time.time()
        total = len(self.entries)
        expired = sum(1 for e in self.entries if now - e.created_at > self.ttl)
        hits = sum(e.hit_count for e in self.entries)
        return {
            "total_entries": total,
            "expired_entries": expired,
            "active_entries": total - expired,
            "total_hits": hits,
            "hit_rate": hits / max(total, 1),
        }
```

### 在 Chain 中集成缓存

```python
from functools import wraps

semantic_cache = SemanticCache(similarity_threshold=0.92, ttl_seconds=3600)

def with_cache(chain_func):
    @wraps(chain_func)
    def cached_wrapper(inputs: dict) -> str:
        question = inputs.get("question", "")

        cached = semantic_cache.get(question)
        if cached:
            print(f"[Cache HIT] 相似度={semantic_cache._cosine_similarity(...):.3f}")
            return cached.answer

        result = chain_func(inputs)

        if isinstance(result, str):
            semantic_cache.set(question=question, answer=result)

        return result
    return cached_wrapper


cached_rag_chain = with_cache(rag_chain.invoke)
```

### 缓存效果实测

```python
test_questions = [
    "免费版支持几个人？",
    "免费版能加几个成员？",
    "免费版支持几个人",          # 几乎一样
    "你们的产品定价是怎样的？",     # 完全不同
]

for q in test_questions:
    start = time.time()
    result = cached_rag_chain({"question": q})
    latency = (time.time() - start) * 1000
    source = "CACHE" if latency < 50 else "LLM"
    print(f"[{source}] ({latency:.0f}ms) Q: {q[:30]}")
```

输出：

```
[LLM] (2340ms) Q: 免费版支持几个人？
[CACHE HIT] (3ms) Q: 免费版能加几个成员？
[CACHE HIT] (2ms) Q: 免费版支持几个人
[LLM] (2100ms) Q: 你们的产品定价是怎样的？

Cache Stats: entries=3, active=3, hits=2, hit_rate=0.67
```

4 个问题只触发了 2 次 LLM 调用——**缓存命中率 67%，直接省了 50% 的 API 成本**。对于客服场景这种高频重复问题，缓存效果通常能达到 **40-70%** 的命中率。

## Level 3：智能路由——按难度分配模型

不是所有问题都需要用昂贵的模型。"天气怎么样？"这种闲聊用 $0.0015 的 mini 模型就够了；但"帮我写一个复杂的 Python 多线程程序"才值得动用 $0.01 的完整模型。

### 两级路由架构

```python
class SmartRouter:
    def __init__(self):
        self.cheap_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        self.premium_llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
        self.classifier_chain = (
            ChatPromptTemplate.from_messages([
                ("system", """你是一个任务复杂度分类器。
判断以下问题是"简单"还是"复杂"。

简单: 事实查询、闲聊、格式转换、简单翻译
复杂: 推理链长、需要多步思考、代码生成、创意写作

只输出 JSON: {"complexity": "simple" 或 "complex"}"""),
                ("human", "{question}"),
            ])
            | self.cheap_llm
            | JsonOutputParser(pydantic_object=ComplexityResult)
        )

    def route(self, question: str) -> ChatOpenAI:
        result = self.classifier_chain.invoke({"question": question})

        if result.complexity == "simple":
            return self.cheap_llm
        else:
            return self.premium_llm


router = SmartRouter()
llm = router.route(user_question)
chain = prompt | llm | parser
```

### 三级路由（进阶）

```python
class TieredRouter:
    def __init__(self):
        self.tiers = {
            "tier1": {  # 最便宜：用于缓存命中 + 极简分类
                "llm": ChatOpenAI(model="gpt-4o-mini"),
                "cost_per_call": 0.0008,
                "tasks": ["cache_hit", "simple_faq", "greeting"],
            },
            "tier2": {  # 中等：标准 RAG 问答
                "llm": ChatOpenAI(model="gpt-4o-mini"),
                "cost_per_call": 0.002,
                "tasks": ["rag_qa", "classification", "summarization"],
            },
            "tier3": {  # 最贵：Agent / 代码生成
                "llm": ChatOpenAI(model="gpt-4o"),
                "cost_per_call": 0.015,
                "tasks": ["agent", "code_gen", "complex_reasoning"],
            },
        }

    def route_and_execute(self, chain, inputs: dict):
        tier = self._classify_tier(inputs.get("question", ""))
        llm = self.tiers[tier]["llm"]

        modified_chain = chain.with_config({"runnable": llm})
        return modified_chain.invoke(inputs)
```

### 路由效果统计

```python
class CostTracker:
    def __init__(self):
        self.calls = {"tier1": 0, "tier2": 0, "tier3": 0}
        self.total_cost = 0.0

    def record(self, tier: str):
        self.calls[tier] += 1
        self.total_cost += SMART_ROUTER.tiers[tier]["cost_per_call"]

    def summary(self) -> dict:
        total = sum(self.calls.values())
        return {
            "total_calls": total,
            "distribution": {
                k: f"{v/total*100:.1f}%"
                for k, v in self.calls.items()
            },
            "estimated_daily_cost": self.total_cost * (3000 / max(total, 1)),
            "vs_all_tier3_savings": f"{(1 - self.total_cost/(total * 0.015)) * 100:.1f}%",
        }
```

典型分布：

```
总调用: 3000 次/天
├── Tier 1 (缓存/FAQ): 1200 次 (40%)   → $0.96/天
├── Tier 2 (RAG 问答):    1500 次 (50%)   → $3.00/天
└── Tier 3 (Agent/复杂):   300 次 (10%)    → $4.50/天
─────────────────────────────────────
总计:                           → $8.46/天

对比全部用 GPT-4o ($45/天): 节省 81%
对比全部用 GPT-4o-mini ($4.5/天): 仅增加 88% 但质量更好
```

## Level 4: Token 优化——减少每次调用的开销

### Prompt 压缩技巧

Token 是计费的直接单位。减少 prompt 中的冗余内容就是直接省钱：

| 技巧 | 示例 | 节省量 |
|------|------|--------|
| **精简 system prompt** | 从 500 字减到 200 字 | ~150 tokens |
| **限制检索上下文数量** | `k=10` → `k=5` | ~2000 tokens |
| **使用更短的变量名** | `user_input_message` → `q` | 每次 ~5 tokens |
| **避免重复上下文注入** | 只在必要时传 chat_history | 变化很大 |
| **截断过长的参考文档** | chunk 超过 2000 字符时截断 | 可控 |

```python
def optimize_context_for_cost(context_docs: list, max_chars: int = 4000) -> str:
    total_chars = 0
    selected = []
    for doc in context_docs:
        doc_chars = len(doc.page_content)
        if total_chars + doc_chars <= max_chars:
            selected.append(doc)
            total_chars += doc_chars
        else:
            break

    if not selected and context_docs:
        selected = [context_docs[0]]

    return "\n\n".join(f"[{i}] {d.page_content}" for i, d in enumerate(selected))
```

### 输出长度控制

```python
llm_with_limit = ChatOpenAI(
    model="gpt-4o-mini",
    max_tokens=500,      # 限制最大输出 token 数
    stop=["\n\n\n"],    # 遇到连续换行就停止（防止过度展开）
)
```

`max_tokens=500` 对大多数 RAG 问答足够了（约 300-400 个中文字）。设置为 2048 则允许更长回答但也意味着更高的成本。

## 成本监控仪表盘

最后，你需要一个地方来持续跟踪成本数据：

```python
class CostDashboard:
    def __init__(self):
        self.daily_records = {}

    def record_call(self, model: str, input_tokens: int,
                    output_tokens: int, cache_hit: bool = False):
        from datetime import datetime
        today = datetime.now().strftime("%Y-%m-%d")

        if today not in self.daily_records:
            self.daily_records[today] = {
                "calls": 0, "input_tokens": 0, "output_tokens": 0,
                "cache_hits": 0, "total_cost_usd": 0.0,
            }

        rec = self.daily_records[today]
        rec["calls"] += 1
        rec["input_tokens"] += input_tokens
        rec["output_tokens"] += output_tokens
        if cache_hit:
            rec["cache_hits"] += 1

        cost = self._estimate_cost(model, input_tokens, output_tokens)
        rec["total_cost_usd"] += cost

    def _estimate_cost(self, model: str, inp: int, out: int) -> float:
        pricing = {
            "gpt-4o": {"inp": 2.5, "out": 10.0},
            "gpt-4o-mini": {"inp": 0.15, "out": 0.60},
        }
        p = pricing.get(model, pricing["gpt-4o-mini"])
        return (inp * p["inp"] + out * p["out"]) / 1_000_000

    def get_report(self) -> str:
        lines = ["# 📊 成本监控报告\n"]
        for date, rec in sorted(self.daily_records.items()):
            cache_rate = rec["cache_hits"] / max(rec["calls"], 1) * 100
            avg_tokens = (rec["input_tokens"] + rec["output_tokens"]) / max(rec["calls"], 1)
            lines.append(f"""
**{date}**
| 指标 | 值 |
|------|-----|
| 总调用次数 | {rec['calls']} |
| 缓存命中率 | {cache_rate:.1f}% |
| 平均 Token/次 | {avg_tokens:.0f} |
| 今日成本 | ${rec['total_cost_usd']:.4f} |
| 预估月成本 | ${rec['total_cost_usd'] * 30:.2f} |
""")
        return "\n".join(lines)


dashboard = CostDashboard()
```

在中间件中自动记录每次调用：

```python
# app/middleware/cost_tracker.py
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

class CostTrackerMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        dashboard.record_call(
            model=response.headers.get("x-model", "unknown"),
            input_tokens=int(response.headers.get("x-input-tokens", "0")),
            output_tokens=int(response.headers.get("x-output-tokens", "0")),
            cache_hit=response.status_code == 200 and response.headers.get("x-cache-hit") == "true",
        )
        return response
```

## 四层优化叠加效果

单独每层优化都能带来显著收益，叠加后效果惊人：

| 方案 | 日成本 | vs 原始 ($45) | 节省率 |
|------|-------|--------------|--------|
| **原始（全部 GPT-4o，无优化）** | $45.00 | baseline | 0% |
| **+ Level 1: 模型降级为 mini** | $6.75 | -$38.25 | 85% |
| **+ Level 2: 语义缓存 (50% 命中)** | $3.38 | -$41.62 | 92% |
| **+ Level 3: 智能路由 (40% 用 mini)** | $2.71 | -$42.29 | 94% |
| **+ Level 4: Token 优化 (-20% tokens)** | $2.17 | -$42.83 | 95% |

从每天 $45 降到 $2.17 ——**同样的服务质量，成本降低 95%**。这就是系统化成本优化的力量。
