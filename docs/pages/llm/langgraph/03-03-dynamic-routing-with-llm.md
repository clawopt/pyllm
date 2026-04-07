# 3.3 动态路由与 LLM 驱动的决策

> 前面两节我们讨论的条件边都是基于规则的路由——路由函数通过一系列 if-elif-else 判断来决定下一步去哪里。这种方式在逻辑清晰、规则明确的场景下非常高效，但面对那些需要语义理解、模糊判断或需要从大量选项中选择最佳路径的场景时，规则路由就显得力不从心了。这时候就需要引入 LLM 来做路由决策——让大模型根据当前状态的语义内容来动态决定下一步应该执行哪个分支。这种"LLM 驱动的动态路由"是 LangGraph 最强大的能力之一，也是构建智能 Agent 系统的核心技术。

## 从规则路由到 LLM 路由

先通过一个对比来直观感受一下规则路由和 LLM 路由的差异。假设你有一个客服消息处理系统，需要根据用户输入的内容自动分发给不同的处理队列。用规则路由的话，你需要定义一系列关键词匹配规则：

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class MessageState(TypedDict):
    user_message: str
    intent: str
    confidence: float
    assigned_queue: str
    processing_log: list[str]

def rule_based_router(state: MessageState) -> str:
    msg = state["user_message"].lower()

    if any(kw in msg for kw in ["bug", "error", "crash", "失败", "错误", "崩溃"]):
        return "technical_support"
    if any(kw in msg for kw in ["bill", "payment", "charge", "账单", "支付", "收费"]):
        return "billing"
    if any(kw in msg for kw in ["cancel", "refund", "退货", "退款", "取消"]):
        return "refund"
    if any(kw in msg for kw in ["how", "what", "why", "如何", "什么", "为什么"]):
        return "faq"
    if any(kw in msg for kw in ["please", "can you", "help", "请", "帮忙", "帮助"]):
        return "general_inquiry"

    return "general_inquiry"
```

这种规则路由有几个明显的局限性：第一，需要手动维护关键词列表，新出现的表达方式（比如"系统挂了"、"出问题了"）无法被识别；第二，无法处理语义相近但用词不同的表达（比如"价格太贵"和"收费不合理"应该都走 billing 队列，但规则很难穷举）；第三，无法判断用户的真实意图——"你们的系统怎么这么烂"这句话虽然包含"系统"这个技术词，但实际可能是在发泄情绪而不是报告技术问题。

现在来看看用 LLM 做路由的版本：

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

router_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个客服消息路由专家。根据用户消息的内容，判断应该将其分发到哪个处理队列。

可选队列及其适用场景：
- technical_support: 技术问题、系统故障、功能异常、API 错误
- billing: 账单查询、支付问题、费用争议、退款申请
- account_management: 账户设置、权限管理、密码重置、个人信息修改
- product_inquiry: 产品功能咨询、使用方法、特性介绍
- complaint: 投诉、建议、意见反馈
- general_inquiry: 其他一般性咨询

请只返回队列名称，不要返回任何其他内容。"""),
    ("user", "用户消息: {message}")
])

router_chain = router_prompt | llm | StrOutputParser()

def llm_based_router(state: MessageState) -> str:
    msg = state["user_message"]
    queue = router_chain.invoke({"message": msg})
    return queue.strip().lower()

graph = StateGraph(MessageState)
graph.add_node("route", lambda s: {"intent": llm_based_router(s)})
graph.add_edge(START, "route")
graph.add_edge("route", END)

app = graph.compile()

test_messages = [
    "我的页面一直显示 500 错误，已经半小时了",
    "上个月被多扣了 50 块钱，怎么退回来？",
    "你们的 AI 助手怎么用？有没有教程？",
    "系统太烂了，每次用都卡死",
    "我想修改绑定的邮箱地址"
]

for msg in test_messages:
    result = app.invoke({"user_message": msg, "intent": "",
                         "confidence": 0.0, "assigned_queue": "", "processing_log": []})
    print(f"消息: {msg[:40]}...")
    print(f"  → 路由到: {result['intent']}\n")
```

这个 LLM 路由器的优势非常明显：它能理解语义而非仅仅匹配关键词，能处理从未见过的表达方式，能判断用户的真实意图（比如"系统太烂了"会被正确识别为 complaint 而不是 technical_support）。而且你不需要手动维护规则列表，只需要给 LLM 提供清晰的队列定义和适用场景描述，它就能自动做出合理的路由决策。

## LLM 路由的标准化输出处理

虽然 LLM 路由很强大，但有一个关键的技术挑战：**如何保证 LLM 的输出严格符合 path_map 中定义的键值**。LLM 的输出具有不确定性——它可能返回 "technical_support"，也可能返回 "Technical Support"（大小写不同），甚至可能返回 "技术支持"（中文），这些都会导致路由失败。

解决这个问题的标准做法是**结构化输出**——让 LLM 以 JSON 格式输出，并且用 Pydantic 模型来约束输出字段。LangChain 提供了 `with_structured_output()` 方法来实现这一点：

```python
from typing import Literal
from pydantic import BaseModel, Field

class RoutingDecision(BaseModel):
    queue: Literal["technical_support", "billing", "account_management",
                   "product_inquiry", "complaint", "general_inquiry"] = Field(
        description="目标队列名称，必须是以下之一: technical_support, billing, account_management, product_inquiry, complaint, general_inquiry"
    )
    confidence: float = Field(
        description="路由决策的置信度，0.0 到 1.0 之间",
        ge=0.0, le=1.0
    )
    reasoning: str = Field(
        description="简短说明做出这个决策的理由",
        max_length=100
    )

structured_llm = llm.with_structured_output(RoutingDecision)

router_prompt_structured = ChatPromptTemplate.from_messages([
    ("system", """你是一个客服消息路由专家。分析用户消息，决定应该将其分发到哪个处理队列。

队列定义：
- technical_support: 技术问题、系统故障、功能异常、API 错误
- billing: 账单查询、支付问题、费用争议、退款申请
- account_management: 账户设置、权限管理、密码重置、个人信息修改
- product_inquiry: 产品功能咨询、使用方法、特性介绍
- complaint: 投诉、建议、意见反馈
- general_inquiry: 其他一般性咨询

请返回一个 JSON 对象，包含 queue（队列名称）、confidence（置信度）、reasoning（理由）三个字段。"""),
    ("user", "用户消息: {message}")
])

router_chain_structured = router_prompt_structured | structured_llm

def robust_llm_router(state: MessageState) -> str:
    msg = state["user_message"]
    try:
        decision = router_chain_structured.invoke({"message": msg})
        return {
            "intent": decision.queue,
            "confidence": decision.confidence,
            "assigned_queue": decision.queue,
            "processing_log": [f"LLM路由: {decision.queue} (置信度:{decision.confidence:.2f}) 理由:{decision.reasoning}"]
        }
    except Exception as e:
        return {
            "intent": "general_inquiry",
            "confidence": 0.0,
            "assigned_queue": "general_inquiry",
            "processing_log": [f"LLM路由失败: {str(e)}, 使用默认队列 general_inquiry"]
        }

graph = StateGraph(MessageState)
graph.add_node("route", robust_llm_router)
graph.add_edge(START, "route")
graph.add_edge("route", END)

app = graph.compile()

result = app.invoke({
    "user_message": "我的 API 密钥好像失效了，一直返回 401",
    "intent": "", "confidence": 0.0, "assigned_queue": "", "processing_log": []
})
print(f"路由结果: {result['intent']}")
print(f"置信度: {result['confidence']}")
for log in result["processing_log"]:
    print(log)
```

这段程序描述了结构化 LLM 路由的完整流程。`RoutingDecision` 是一个 Pydantic 模型，用 `Literal` 类型约束了 `queue` 字段只能取预定义的 6 个值之一，用 `Field` 添加了描述信息和取值范围约束。`structured_llm = llm.with_structured_output(RoutingDecision)` 这一行让 LLM 的输出自动解析为 `RoutingDecision` 对象——如果 LLM 返回的 JSON 不符合模型定义，LangChain 会自动触发重试（最多 3 次）。

在 `robust_llm_router` 函数中，我们用 try-except 包裹了 LLM 调用，即使 LLM 完全失败（比如网络超时、API 配额耗尽），也能优雅地降级到默认队列 `general_inquiry`，而不是让整个图执行中断。这种防御性编程在生产环境中非常重要。

## 多轮 LLM 路由：逐步细化决策

有些场景下，一次 LLM 路由可能不够——你需要先做一个粗粒度的分类，然后在每个大类内部再做细粒度的子分类。这种多轮 LLM 路由虽然会增加一些成本（因为要调用多次 LLM），但能显著提高路由的准确性和可解释性。

```python
from typing import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, START, END

class MultiStageRoutingState(TypedDict):
    user_query: str
    broad_category: str
    specific_subcategory: str
    confidence: float
    routing_path: list[str]
    final_destination: str
    trace: Annotated[list[str], operator.add]

broad_categories = {
    "technical": "技术相关问题",
    "business": "业务流程问题",
    "account": "账户与权限问题",
    "billing": "计费与支付问题",
    "other": "其他问题"
}

subcategories = {
    "technical": ["api", "integration", "bug_report", "feature_request", "performance"],
    "business": ["workflow", "data_management", "reporting", "automation"],
    "account": ["login", "permissions", "profile", "security"],
    "billing": ["invoice", "payment_method", "refund", "pricing"],
    "other": ["feedback", "complaint", "general_inquiry"]
}

def first_stage_broad_routing(state: MultiStageRoutingState) -> dict:
    query = state["user_query"]

    prompt = f"""分析用户查询，判断它属于哪个大类。

大类定义：
{chr(10).join(f'- {k}: {v}' for k, v in broad_categories.items())}

用户查询: {query}

只返回大类名称（technical/business/account/billing/other），不要其他内容。"""

    try:
        category = llm.invoke(prompt).content.strip().lower()
        if category not in broad_categories:
            category = "other"
    except:
        category = "other"

    return {
        "broad_category": category,
        "routing_path": [category],
        "trace": [f"[第一阶段] 大类: {category} ({broad_categories[category]})"]
    }

def route_to_second_stage(state: MultiStageRoutingState) -> str:
    return state["broad_category"]

def second_stage_sub_routing(state: MultiStageRoutingState) -> dict:
    query = state["user_query"]
    broad = state["broad_category"]

    available_subs = subcategories.get(broad, ["general"])
    subs_list = ", ".join(available_subs)

    prompt = f"""用户查询已被归类为 {broad} 类。现在需要进一步细分子类别。

可用子类别: {subs_list}

用户查询: {query}

只返回子类别名称，不要其他内容。如果都不匹配，返回 general。"""

    try:
        sub = llm.invoke(prompt).content.strip().lower()
        if sub not in available_subs:
            sub = "general"
    except:
        sub = "general"

    final_dest = f"{broad}_{sub}"

    return {
        "specific_subcategory": sub,
        "final_destination": final_dest,
        "routing_path": [broad, sub],
        "confidence": 0.85,
        "trace": [f"[第二阶段] 子类: {sub} → 最终目标: {final_dest}"]
    }

def route_to_handler(state: MultiStageRoutingState) -> str:
    return state["final_destination"]

graph = StateGraph(MultiStageRoutingState)
graph.add_node("broad_routing", first_stage_broad_routing)
graph.add_node("sub_routing", second_stage_sub_routing)

graph.add_edge(START, "broad_routing")
graph.add_conditional_edges("broad_routing", route_to_second_stage, {
    cat: "sub_routing" for cat in broad_categories.keys()
})
graph.add_edge("sub_routing", END)

app = graph.compile()

queries = [
    "你们的 API 怎么对接？有没有 Python SDK？",
    "我想导出上个月的销售报表，怎么操作？",
    "我的登录密码忘了，怎么重置？",
    "上个月的发票什么时候能开？"
]

for q in queries:
    result = app.invoke({
        "user_query": q, "broad_category": "", "specific_subcategory": "",
        "confidence": 0.0, "routing_path": [], "final_destination": "", "trace": []
    })
    print(f"\n查询: {q}")
    print(f"  → {result['broad_category']} > {result['specific_subcategory']}")
    print(f"  → 最终: {result['final_destination']}")
    for t in result["trace"]:
        print(f"    {t}")
```

这个两阶段路由系统先做粗粒度的大类分类（5 个大类），然后在每个大类内部做细粒度的子分类（每个大类有 4-5 个子类）。最终的路由目标是一个组合字符串，比如 `technical_api`、`business_workflow` 等，这样既保证了路由的细粒度，又避免了单次 LLM 调用需要从 20+ 个选项中选择（这会降低准确率）。

在实际项目中，这种多阶段路由的变体非常常见——比如先判断是"需要人工处理"还是"可以自动处理"，如果是自动处理再判断具体走哪个自动化流程；或者先判断"用户类型"（新用户/老用户/VIP用户），再根据用户类型走不同的处理逻辑。关键是要合理地划分阶段，让每个阶段 LLM 的决策空间保持在 5-7 个选项以内，这样既能保证准确率，又能控制成本。

## LLM 路由的性能优化

LLM 路由虽然强大，但每次调用都需要几百毫秒的时间和一定的 token 费用。在高并发场景下，这些成本会累积成显著的负担。有几个常用的优化策略可以缓解这个问题：

**第一，路由结果缓存**。如果相同的或高度相似的用户输入反复出现，可以缓存 LLM 的路由决策结果，避免重复调用。缓存可以用简单的内存字典（适合短期缓存）或 Redis（适合长期、多实例共享的缓存）来实现：

```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=1000)
def cached_llm_router(message_hash: str, message: str) -> str:
    return router_chain.invoke({"message": message})

def cached_router(state: MessageState) -> str:
    msg = state["user_message"]
    msg_hash = hashlib.md5(msg.encode()).hexdigest()
    return cached_llm_router(msg_hash, msg)
```

**第二，混合路由策略**——先用规则路由处理那些明显、高频的输入，只有规则无法确定的情况下才调用 LLM。这能显著减少 LLM 的调用次数：

```python
def hybrid_router(state: MessageState) -> str:
    msg = state["user_message"].lower()

    if "密码" in msg or "登录" in msg:
        return "account_management"
    if "账单" in msg or "支付" in msg:
        return "billing"
    if "api" in msg or "sdk" in msg:
        return "technical_support"

    return llm_based_router(state)
```

**第三，使用更小、更快的模型**。对于路由这种相对简单的任务，不需要动用 GPT-4 这样的大模型。GPT-4o-mini、Claude Haiku、甚至一些开源的小模型（如 Llama-3-8B）往往就足够了，而且速度更快、成本更低。

**第四，批量路由**。如果你有多个独立的输入需要路由，可以把它们打包成一个 batch 一次性发给 LLM，让 LLM 一次返回多个路由决策：

```python
def batch_router(messages: list[str]) -> list[str]:
    prompt = f"""为以下每条用户消息选择合适的队列。

可选队列: technical_support, billing, account_management, product_inquiry, complaint, general_inquiry

用户消息:
{chr(10).join(f'{i+1}. {msg}' for i, msg in enumerate(messages))}

请返回一个列表，每条消息对应一个队列名称，格式如: ["queue1", "queue2", ...]"""

    response = llm.invoke(prompt).content
    import json
    try:
        queues = json.loads(response)
        return queues if len(queues) == len(messages) else ["general_inquiry"] * len(messages)
    except:
        return ["general_inquiry"] * len(messages)
```

## LLM 路由的常见陷阱

最后我们来总结一下在使用 LLM 路由时容易踩的坑。第一个问题是**路由幻觉**——LLM 有时会返回一个不在预定义列表中的队列名称，比如返回 "customer_service" 而不是 "general_inquiry"。这通常是因为 prompt 描述不够清晰或者 LLM 产生了幻觉。解决方法是使用结构化输出（前面已经展示过），或者在路由函数中加一层映射逻辑，把未知队列映射到默认队列。

第二个问题是**置信度误判**。LLM 返回的置信度并不总是可靠的——有时 LLM 会对自己错误的判断给出很高的置信度。不要完全依赖 LLM 的置信度来做决策，而是把它作为一个参考指标。真正可靠的判断标准是多次调用的一致性——如果同一个输入多次路由的结果都一样，那这个结果的可信度就比较高。

第三个问题是**上下文污染**。如果路由函数能访问到完整的状态（包括之前所有节点的输出），而之前的状态中包含了一些误导性的信息，可能会影响 LLM 的路由决策。一个实用的技巧是只把当前节点相关的、干净的数据传给 LLM，而不是把整个状态字典都塞进去。

第四个问题是**成本失控**。在循环结构中使用 LLM 路由时要特别小心——如果循环次数不可控，LLM 的调用次数可能指数级增长。一定要设置最大迭代次数的上限，并且在路由函数中记录调用次数，超过阈值就强制走默认路径。

总的来说，LLM 路由是一个强大的工具，但它不是银弹。在规则路由和 LLM 路由之间找到平衡点，根据具体场景选择合适的策略，才能构建出既智能又高效的路由系统。
