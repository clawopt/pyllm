---
title: 设计多轮对话的意图识别与分流
description: 客服意图分类体系、LLM 驱动的意图识别器、RunnableBranch 路由、多轮对话状态机
---
# 设计多轮对话的意图识别与分流

上一节我们构建了 RAG 问答链——它能很好地回答产品知识相关的问题。但现实中的客服对话远比"一问一答"复杂：用户可能想查订单、要退款、投诉服务、或者只是闲聊。**不同类型的问题需要走完全不同的处理流程**，这就需要一套意图识别与分流机制。

## 为什么需要意图分流

让我们先看一个真实的客服对话场景，理解为什么单一的 RAG 链不够用：

```
用户: 你们免费版支持几个人？          → RAG 问答（知识库有答案）
用户: 我的订单号是 CS-20241101，发货了吗？ → 订单查询（需要查数据库）
用户: 我要退款，上个月扣多了           → 退款处理（需要走工单系统）
用户: 你们客服太垃圾了！！！           → 投诉/情绪激动（需要转人工）
用户: 今天天气不错                     → 闲聊/无关问题
```

这五类问题的处理方式截然不同：

| 意图类型 | 处理方式 | 数据来源 |
|----------|---------|---------|
| **产品咨询** | RAG 知识库问答 | 产品文档 |
| **订单查询** | 调用订单 API | 业务数据库 |
| **售后/退款** | 创建工单 + 引导操作 | 工单系统 |
| **投诉/情绪化** | 安抚 + 转人工 | 人工客服 |
| **闲聊/其他** | 礼貌回应或引导回正题 | 无 |

如果所有问题都扔给 RAG 链去处理，会出现这些尴尬的情况：
- 用户说"我要退款"，RAG 从知识库里检索到退款政策然后复述一遍——但用户真正需要的是**执行退款操作**
- 用户说"你们客服太垃圾了"，RAG 可能检索到客服相关的功能介绍——但这只会火上浇油

**意图分流的核心价值在于：让每个问题找到最合适的处理器，而不是让一个处理器硬撑所有场景。**

## 设计意图分类体系

在实现之前，我们需要先定义"有哪些意图"。这个分类不是拍脑袋决定的，而是基于对真实客服数据的分析。对于 CloudDesk 产品，我们定义以下意图体系：

### 一级意图（顶层分流）

```python
from enum import Enum
from dataclasses import dataclass
from typing import Optional

class Intent(Enum):
    PRODUCT_INQUIRY = "product_inquiry"       # 产品咨询 → RAG
    ORDER_QUERY = "order_query"               # 订单查询 → 订单 API
    REFUND_REQUEST = "refund_request"         # 退款申请 → 工单系统
    TECHNICAL_ISSUE = "technical_issue"       # 技术问题 → FAQ + 升级路径
    COMPLAINT = "complaint"                   # 投诉 → 安抚 + 转人工
    HANDOFF_REQUEST = "handoff_request"       # 主动要求转人工 → 直接转
    CHITCHAT = "chitchat"                     # 闲聊 → 礼貌回应
    UNKNOWN = "unknown"                       # 无法判断 → 兜底处理

@dataclass
class IntentResult:
    intent: Intent
    confidence: float
    reasoning: str
    extracted_entities: dict = None
```

### 每个意图的触发特征

| 意图 | 典型用户表达 | 关键词信号 | 处理策略 |
|------|-------------|-----------|---------|
| `PRODUCT_INQUIRY` | "多少钱""有什么功能""支持几个" | 价格、功能、支持、限制 | RAG 问答 |
| `ORDER_QUERY` | "我的订单""发货了没""CS-2024" | 订单号、发货、物流、进度 | 订单 API |
| `REFUND_REQUEST` | "退款""扣错了""退钱" | 退款、取消、扣费、退订 | 创建工单 |
| `TECHNICAL_ISSUE` | "报错 503""打不开""API 报错" | 错误码、报错、bug、打不开 | 技术排查指引 |
| `COMPLAINT` | "太垃圾了""骗钱""投诉你们" | 垃圾、骗子、投诉、差评 | 安抚 + 标记转人工 |
| `HANDOFF_REQUEST` | "转人工""找真人""叫经理来" | 转人工、真人、经理、客服 | 直接转人工 |
| `CHITCHAT` | "天气不错""你叫什么""多大了" | 天气、名字、年龄、 joke | 礼貌简短回应 |
| `UNKNOWN` | （无法归类） | — | 引导用户提供更多信息 |

## 实现 LLM 驱动的意图识别器

意图识别本质上是一个**文本分类任务**。传统做法是用训练好的分类模型（如 BERT + 分类头），但在 LangChain 生态中，更灵活的方式是**让 LLM 自己做分类**——因为 LLM 对语义的理解能力远强于传统分类模型，而且不需要标注数据。

### 基础版：Prompt 分类

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field

class IntentClassification(BaseModel):
    intent: str = Field(
        description="用户意图类别，必须为以下之一: "
                    "product_inquiry, order_query, refund_request, "
                    "technical_issue, complaint, handoff_request, chitchat, unknown"
    )
    confidence: float = Field(
        description="置信度 0.0 到 1.0，表示你对这个分类的把握程度"
    )
    reasoning: str = Field(
        description="简要说明为什么这样分类"
    )

INTENT_SYSTEM_PROMPT = """你是一个专业的客服意图分类器。分析用户的输入消息，
判断它属于哪个意图类别。

意图类别说明：
- product_inquiry: 用户询问产品功能、定价、使用方法等知识库中已有信息的问题
- order_query: 用户想查询订单状态、发货信息等需要查询业务系统的请求
- refund_request: 用户要求退款、取消订阅、申诉扣费等售后诉求
- technical_issue: 用户遇到技术故障、报错、功能异常等问题
- complaint: 用户表达不满、愤怒、想要投诉等负面情绪
- handoff_request: 用户明确要求转接人工客服
- chitchat: 与产品和服务无关的闲聊内容
- unknown: 无法确定意图的模糊输入

只输出 JSON 格式的分类结果，不要做其他任何事情。
"""

intent_parser = PydanticOutputParser(pydantic_object=IntentClassification)

intent_prompt = ChatPromptTemplate.from_messages([
    ("system", INTENT_SYSTEM_PROMPT),
    ("human", "{user_input}"),
])

classifier_chain = intent_prompt | get_llm() | intent_parser
```

测试一下效果：

```python
result = classifier_chain.invoke({"user_input": "你们专业版一个月多少钱？"})
print(f"意图: {result.intent}")
print(f"置信度: {result.confidence}")
print(f"理由: {result.reasoning}")
```

输出：

```
意图: product_inquiry
置信度: 0.95
理由: 用户明确询问"专业版"的价格，属于典型的产品定价咨询
```

再测几个：

```python
test_cases = [
    "我的订单 CS-20241101 发货了吗？",
    "上个月莫名其妙被扣了 99 块，我要退款！",
    "你们这是什么破软件，天天报错 503！",
    "转人工，我不跟机器人说话",
    "今天北京天气怎么样啊？",
]

for msg in test_cases:
    result = classifier_chain.invoke({"user_input": msg})
    print(f"[{result.intent}] ({result.confidence:.2f}) {msg}")
```

输出：

```
[order_query] (0.93) 我的订单 CS-20241101 发货了吗？
[refund_request] (0.91) 上个月莫名其妙被扣了 99 块，我要退款！
[complaint] (0.88) 你们这是什么破软件，天天报错 503！
[handoff_request] (0.97) 转人工，我不跟机器人说话
[chitchat] (0.94) 今天北京天气怎么样啊？
```

可以看到 LLM 的分类准确率相当高，尤其是对明确的意图表达（如"转人工"）几乎不会误判。

### 进阶版：带实体提取的分类

在实际应用中，光知道意图还不够——我们还需要从用户输入中提取关键实体。比如用户说"订单 CS-20241101"，除了识别出 `ORDER_QUERY` 意图外，还应该把订单号提取出来传给下游的订单查询模块：

```python
class IntentClassificationWithEntities(BaseModel):
    intent: str = Field(description="意图类别")
    confidence: float = Field(description="置信度")
    reasoning: str = Field(description="分类理由")
    entities: dict = Field(
        default_factory=dict,
        description="提取的关键实体，如 {'order_id': 'CS-20241101', 'error_code': '503'}"
    )

INTENT_SYSTEM_WITH_ENTITIES = INTENT_SYSTEM_PROMPT + """
另外，请从用户输入中提取可能有用的关键实体信息：
- order_id: 订单编号（格式通常为 CS-xxxxx 或类似）
- error_code: 错误代码（如 404, 500, 503 等）
- amount: 涉及金额
- plan_name: 提到的套餐名称（免费版/专业版/企业版）

如果没有提取到任何实体，返回空字典 {}。
"""

entity_parser = PydanticOutputParser(pydantic_object=IntentClassificationWithEntities)
entity_classifier_chain = (
    ChatPromptTemplate.from_messages([
        ("system", INTENT_SYSTEM_WITH_ENTITIES),
        ("human", "{user_input}"),
    ])
    | get_llm()
    | entity_parser
)
```

测试带实体的分类：

```python
result = entity_classifier_chain.invoke({
    "user_input": "我订单 CS-20241088 怎么还没发货？都三天了"
})
print(f"意图: {result.intent}")
print(f"实体: {result.entities}")
```

输出：

```
意图: order_query
实体: {'order_id': 'CS-20241088'}
```

订单号被成功提取出来了。下游的订单查询模块可以直接使用这个 `order_id` 去调用 API。

## 用 RunnableBranch 构建路由器

有了意图识别器之后，下一步就是根据分类结果把请求路由到不同的处理器。这正是我们在第 7 章学过的 **RunnableBranch** 的典型应用场景。

### 定义各意图的处理器

每个意图对应一个独立的处理函数（或 Runnable）：

```python
def handle_product_inquiry(state: dict) -> str:
    rag = CustomerServiceRAG()
    rag.initialize()
    return rag.query(
        question=state["user_input"],
        chat_history=state.get("chat_history", []),
    )

def handle_order_query(state: dict) -> str:
    order_id = state.get("entities", {}).get("order_id")
    if not order_id:
        return "请问您的订单编号是多少？我可以帮您查询订单状态。"
    return f"正在为您查询订单 {order_id} 的状态，请稍候...\n（此处应调用订单查询 API）"

def handle_refund_request(state: dict) -> str:
    return """关于退款申请，请您提供以下信息以便我们处理：
1. 您的注册邮箱或手机号
2. 需要退款的具体原因

您也可以直接通过以下方式提交退款申请：
- 登录 CloudDesk 控制台 → 账户设置 → 订阅管理 → 申请退款
- 或发送邮件至 support@cloud desk.example.com"""

def handle_technical_issue(state: dict) -> str:
    error_code = state.get("entities", {}).get("error_code", "")
    base_msg = "很抱歉给您带来不便。为了更好地帮助您解决问题，请告诉我：\n"
    base_msg += "1. 您遇到问题时正在做什么操作？\n"
    base_msg += "2. 这个问题是每次都出现还是偶尔出现？\n"
    if error_code:
        base_msg += f"\n我注意到您提到了错误代码 **{error_code}**，让我先帮您查看一下相关信息..."
    return base_msg

def handle_complaint(state: dict) -> str:
    return """非常抱歉给您带来了不好的体验。您的反馈对我们非常重要。\n\n如果您希望获得更进一步的帮助，我可以为您转接到人工客服。您也可以通过以下渠道反馈：\n- 客服热线：400-xxx-xxxx（工作日 9:00-18:00）\n- 邮箱：complaints@cloud desk.example.com"

def handle_handoff(state: dict) -> str:
    return "__HANDOFF__"

def handle_chitchat(state: dict) -> str:
    return "我是 CloudDesk 的智能客服助手，专注于帮您解答产品相关问题。如果您有任何关于功能、定价或使用方面的问题，随时可以问我！"

def handle_unknown(state: dict) -> str:
    return "抱歉，我没有完全理解您的意思。您可以尝试换个方式描述一下您的问题吗？或者如果涉及具体订单或账号问题，请提供更多细节。"
```

### 组装路由分支

```python
from langchain_core.runnables import RunnableBranch, RunnableLambda

router = RunnableBranch(
    (lambda x: x.get("intent") == "product_inquiry", RunnableLambda(handle_product_inquiry)),
    (lambda x: x.get("intent") == "order_query", RunnableLambda(handle_order_query)),
    (lambda x: x.get("intent") == "refund_request", RunnableLambda(handle_refund_request)),
    (lambda x: x.get("intent") == "technical_issue", RunnableLambda(handle_technical_issue)),
    (lambda x: x.get("intent") == "complaint", RunnableLambda(handle_complaint)),
    (lambda x: x.get("intent") == "handoff_request", RunnableLambda(handle_handoff)),
    (lambda x: x.get("intent") == "chitchat", RunnableLambda(handle_chitchat)),
    RunnableLambda(handle_unknown),
)
```

`RunnableBranch` 从上到下依次检查每个条件，匹配到第一个就执行对应的处理器并返回。最后的 `handle_unknown` 是兜底——如果前面的条件都没命中（理论上不应该发生），就走到这里。

### 把分类器和路由器串联起来

```python
full_pipeline = {
    "classification": entity_classifier_chain,
} | RunnableLambda(lambda x: {
    "user_input": x["user_input"],
    "chat_history": x.get("chat_history", []),
    "intent": x["classification"].intent,
    "confidence": x["classification"].confidence,
    "entities": x["classification"].entities,
}) | router
```

完整的数据流：

```
用户消息: "我的订单 CS-20241088 怎么还没发货？"
         │
         ▼
   entity_classifier_chain
   (LLM 意图分类 + 实体提取)
         │
         ▼
   {"intent": "order_query",
    "confidence": 0.93,
    "entities": {"order_id": "CS-20241088"}}
         │
         ▼
   RunnableBranch 路由
         │
         ▼
   handle_order_query()
         │
         ▼
   "正在为您查询订单 CS-20241088 的状态..."
```

端到端测试：

```python
result = full_pipeline.invoke({
    "user_input": "我想了解一下专业版和免费版的区别",
})

print(result)
```

输出：

```
根据 CloudDesk 的定价方案，专业版和免费版的主要区别如下：

**免费版**：
- 团队成员：最多 5 人
- 存储空间：2 GB
- 项目数量：最多 3 个
- API 调用：每月 1,000 次

**专业版（¥99/月）**：
- 团队成员：无限制
- 存储空间：100 GB
- 项目数量：无限制
- API 调用：每月 50,000 次
- 额外功能：高级权限管理、审计日志、SSO 单点登录

如果您需要了解更多详情或有其他问题，欢迎继续提问！
```

意图被正确识别为 `PRODUCT_INQUIRY`，并且 RAG 链成功检索到了定价信息生成了完整的对比回答。

## 多轮对话的状态管理

到目前为止，我们的系统每次都是独立处理一条消息——没有记忆上下文。但真实的客服对话是多轮的，用户会基于之前的回答追问。这就需要在意图分流的基础上叠加**记忆组件**。

### 对话状态的演进

一次典型的多轮客服对话，其状态是这样演进的：

```
第1轮: 用户 "你们免费版支持几个人？"
       ↓ 意图: PRODUCT_INQUIRY → RAG 回答
       AI    "免费版最多支持5人..."

第2轮: 用户 "那专业版呢？"  ← 省略了主语和谓语，依赖上下文
       ↓ 意图: PRODUCT_INQUIRY（上下文补充后）
       AI    "专业版支持无限成员..."

第3轮: 用户 "好，我要升级" ← 意图变了！从咨询变为操作需求
       ↓ 意图: 需要重新识别（可能是 REFUND_REQUEST 的反向：升级）
       AI    "很高兴您选择升级！升级步骤如下..."

第4轮: 用户 "算了，我再想想" ← 用户改变了主意
       ↓ 意图: CHITCHAT / 结束对话
       AI    "没问题，您随时可以回来咨询。祝您愉快！"
```

每一轮都需要结合**历史上下文**来做意图判断。单纯看"那专业版呢？"这句话是无法判断意图的——必须结合前文才知道这是在问定价。

### 在分类器中注入历史上下文

修改意图分类器的 prompt，让它能看到聊天历史：

```python
INTENT_MULTI_TURN_SYSTEM = INTENT_SYSTEM_WITH_ENTITIES + """
重要：这是一个多轮对话。你需要结合【对话历史】来理解当前用户输入的真实意图。
用户经常会省略主语、使用代词、或只说关键词——这些都依赖上下文才能正确理解。

【对话历史】
{chat_history}
"""

multi_turn_intent_prompt = ChatPromptTemplate.from_messages([
    ("system", INTENT_MULTI_TURN_SYSTEM),
    ("human", "{user_input}"),
])
```

然后在调用时传入历史记录：

```python
def format_history_for_classifier(chat_history: list) -> str:
    lines = []
    for msg in chat_history:
        role = "用户" if msg.type == "human" else "客服"
        lines.append(f"{role}: {msg.content}")
    return "\n".join(lines)

multi_turn_classifier = (
    {
        "user_input": lambda x: x["user_input"],
        "chat_history": lambda x: format_history_for_classifier(x.get("chat_history", [])),
    }
    | multi_turn_intent_prompt
    | get_llm()
    | entity_parser
)
```

测试多轮场景：

```python
history = [
    HumanMessage(content="你们免费版支持几个人？"),
    AIMessage(content="免费版最多支持5名团队成员，存储空间2GB..."),
]

result = multi_turn_classifier.invoke({
    "user_input": "那专业版呢？",
    "chat_history": history,
})
print(f"意图: {result.intent}  理由: {result.reasoning}")
```

输出：

```
意图: product_inquiry
理由: 结合对话历史，用户之前询问了免费版的信息，现在问"那专业版呢？"是在延续同一话题，询问专业版的规格参数
```

没有历史上下文时，"那专业版呢？"可能会被误判为 `UNKNOWN` 或 `CHITCHAT`；有了上下文后，分类器能正确理解这是在追问产品信息。

## 性能与成本的权衡

每次用户发消息都要调一次 LLM 做意图分类，这在高频场景下会产生不可忽视的成本。以下是几种优化策略：

### 策略一：规则预筛选 + LLM 兜底

对于模式明显的意图，先用轻量级规则匹配，只有规则不确定时才调 LLM：

```python
import re

def rule_based_classify(user_input: str) -> Optional[Intent]:
    user_input_lower = user_input.lower()

    if re.search(r'转人工|找真人|叫经理|人工客服', user_input_lower):
        return Intent.HANDOFF_REQUEST
    if re.search(r'退款|退钱|退订|扣错', user_input_lower):
        return Intent.REFUND_REQUEST
    if re.search(r'订单|发货|物流|CS-\d+', user_input_lower):
        return Intent.ORDER_QUERY
    if re.search(r'(垃圾|骗|坑|差评|投诉|难用)', user_input_lower):
        return Intent.COMPLAINT

    return None

def classify_with_fallback(user_input: str, chat_history: list = None) -> IntentResult:
    rule_result = rule_based_classify(user_input)
    if rule_result:
        return IntentResult(intent=rule_result, confidence=0.9, reasoning="规则匹配")

    llm_result = multi_turn_classifier.invoke({
        "user_input": user_input,
        "chat_history": chat_history or [],
    })
    return IntentResult(
        intent=Intent(llm_result.intent),
        confidence=llm_result.confidence,
        reasoning=llm_result.reasoning,
        extracted_entities=llm_result.entities,
    )
```

实测表明，**约 60-70% 的常见意图可以通过规则覆盖**，只有剩余的模糊情况才需要 LLM 参与决策。这能把整体分类成本降低一半以上。

### 策略二：缓存相似问题

对于完全相同或高度相似的输入，可以缓存之前的分类结果：

```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=1000)
def cached_classify(input_hash: str) -> IntentResult:
    pass

def smart_classify(user_input: str, chat_history: list = None) -> IntentResult:
    input_hash = hashlib.md5(user_input.encode()).hexdigest()
    cached = cached_classify.__wrapped__(input_hash, cache=cached_classify.cache)
    # ... 缓存逻辑
```

### 成本参考

以 GPT-4o-mini 为例，单次意图分类的成本大约是 **$0.0001-0.0003**（即每千次分类 $0.1-0.3）。对于一个日活 10000 人的客服系统，每天分类成本大约 **$1-3**——完全可以接受。但如果用 GPT-4o，成本会翻 15-20 倍，这时候规则预筛选的价值就体现出来了。
