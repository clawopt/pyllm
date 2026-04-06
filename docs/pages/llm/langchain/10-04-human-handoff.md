---
title: 集成人工接管（Handoff）机制
description: Handoff 触发条件设计、会话上下文无缝传递、完整客服系统组装、CLI 与 FastAPI 部署
---
# 集成人工接管（Handoff）机制

前面三节我们分别实现了 RAG 知识库问答、意图识别与分流、多轮对话状态管理。现在是时候把所有模块组装在一起，并加上最后一个关键能力——**人工接管（Handoff）**。

## 为什么 Handoff 是客服系统的生命线

无论你的 AI 客服做得多么智能，它永远无法处理所有场景。以下情况必须由人来接手：

- **用户情绪激动**：投诉、威胁差评、涉及法律风险
- **问题超出知识范围**：AI 连续多次无法回答
- **涉及敏感操作**：退款大额资金、账号封禁/解封
- **用户明确要求**："转人工""我要找真人"
- **安全边界触发**：检测到 prompt 注入或恶意试探

**一个没有 Handoff 机制的客服系统，就像一家没有紧急出口的建筑——平时看不出问题，一旦出事就是灾难。** 用户被 AI 气得要死却找不到真人，这种体验比没有 AI 客服更糟糕。

## Handoff 的触发条件设计

我们设计三层触发机制：**主动请求 → 被动检测 → 安全兜底**。

### 第一层：用户主动请求

这是最简单的——用户直接说"转人工"或类似表达。我们在意图分类器中已经覆盖了 `HANDOFF_REQUEST`：

```python
HANDOFF_TRIGGERS_EXPLICIT = [
    "转人工", "找真人", "叫经理", "人工客服",
    "我要找人", "不跟机器人说", "接人工",
    "transfer", "human agent", "speak to person",
]
```

### 第二层：被动条件检测

系统自动判断是否需要转人工，不需要用户明确要求：

```python
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional
from enum import Enum

class HandoffReason(Enum):
    USER_REQUEST = "user_request"           # 用户主动要求
    EMOTIONAL_OVERFLOW = "emotional"        # 情绪过于激动
    CONSECUTIVE_FAILURES = "failures"       # 连续多次回答失败
    SENSITIVE_TOPIC = "sensitive"           # 敏感话题
    SECURITY_ALERT = "security"             # 安全告警
    MAX_TURNS_EXCEEDED = "max_turns"        # 对话轮次超限

@dataclass
class HandoffDecision:
    should_handoff: bool
    reason: Optional[HandoffReason] = None
    message_to_user: str = ""
    conversation_summary: str = ""
    urgency: str = "normal"

class HandoffDetector:
    def __init__(self,
                 max_failures: int = 3,
                 max_turns: int = 20,
                 emotion_threshold: float = 0.7):
        self.max_failures = max_failures
        self.max_turns = max_turns
        self.emotion_threshold = emotion_threshold

    def check(self, session_state: dict) -> HandoffDecision:
        state = session_state

        if state.get("turn_count", 0) >= self.max_turns:
            return HandoffDecision(
                should_handoff=True,
                reason=HandoffReason.MAX_TURNS_EXCEEDED,
                message_to_user="为了更好地帮助您，我已为您安排了人工客服，请稍候...",
                urgency="low",
            )

        if state.get("consecutive_failures", 0) >= self.max_failures:
            return HandoffDecision(
                should_handoff=True,
                reason=HandoffReason.CONSECUTIVE_FAILURES,
                message_to_user="抱歉，我似乎无法很好地解答您的问题。让我为您转接人工客服...",
                urgency="high",
            )

        emotion_score = self._detect_emotion(state.get("last_user_input", ""))
        if emotion_score > self.emotion_threshold:
            return HandoffDecision(
                should_handoff=True,
                reason=HandoffReason.EMOTIONAL_OVERFLOW,
                message_to_user="我感受到您可能有些着急。让我立刻为您转接一位专业的人工客服。",
                urgency="urgent",
            )

        sensitive = self._check_sensitive_topics(state.get("last_user_input", ""))
        if sensitive:
            return HandoffDecision(
                should_handoff=True,
                reason=HandoffReason.SENSITIVE_TOPIC,
                message_to_user="这个问题需要人工客服来为您处理，正在为您转接...",
                urgency="high",
            )

        return HandoffDecision(should_handoff=False)

    def _detect_emotion(self, text: str) -> float:
        negative_words = [
            "垃圾", "骗", "坑", "差评", "投诉", "难用",
            "退钱", "骗子", "恶心", "愤怒", "举报",
            "!!!", "？？？", "操", "妈的",
        ]
        text_lower = text.lower()
        score = 0.0
        for word in negative_words:
            if word in text_lower:
                score += 0.15
        if text.count("!") >= 2 or text.count("?") >= 2:
            score += 0.1
        return min(score, 1.0)

    def _check_sensitive_topics(self, text: str) -> bool:
        sensitive_patterns = [
            r"退款.*\d{4,}",          # 大额退款
            r"封号|解封|冻结",         # 账号操作
            r"律师|起诉|法律",         # 法律威胁
            r"媒体|曝光|记者",         # 威胁曝光
        ]
        import re
        for pattern in sensitive_patterns:
            if re.search(pattern, text):
                return True
        return False
```

### 第三层：安全兜底

在内容审核中间件中检测到恶意输入时强制转人工（或不做任何响应，取决于安全策略）。这部分我们在第 9 章已经讨论过，这里不再展开。

## 会话上下文的无缝传递

Handoff 不是简单地说一句"请稍等"就完事了。**最关键的是要把 AI 和用户之前的对话上下文传递给人工客服**，这样客服人员才能无缝接续对话，不用让用户重复说明问题。

### 会话摘要生成

当触发 Handoff 时，系统应该自动生成一份简洁但信息完整的会话摘要：

```python
SUMMARY_PROMPT = """你是一个客服会话摘要生成器。请根据以下对话历史，
为人工客服生成一份简洁的摘要。

摘要应包含：
1. 用户的核心诉求（一句话）
2. 已尝试过的解决方案和结果
3. 用户的关键信息（订单号、错误码等）
4. 当前情绪状态判断

请用中文输出，控制在 200 字以内。
"""

def generate_handoff_summary(chat_history: list, session_metadata: dict) -> str:
    summary_chain = (
        ChatPromptTemplate.from_messages([
            ("system", SUMMARY_PROMPT),
            ("human", "对话历史：\n{history}\n\n会话元数据：{metadata}"),
        ])
        | get_llm()
        | StrOutputParser()
    )

    history_text = "\n".join([
        f"{'用户' if m.type == 'human' else 'AI'}: {m.content}"
        for m in chat_history[-10:]
    ])

    metadata_text = (
        f"会话ID: {session_metadata.get('session_id', 'N/A')}\n"
        f"对话轮数: {session_metadata.get('turn_count', 0)}\n"
        f"连续失败次数: {session_metadata.get('consecutive_failures', 0)}"
    )

    return summary_chain.invoke({
        "history": history_text,
        "metadata": metadata_text,
    })
```

测试一下摘要效果：

```python
test_history = [
    HumanMessage(content="你们免费版支持几个人？"),
    AIMessage(content="免费版最多5人..."),
    HumanMessage(content="那专业版呢？"),
    AIMessage(content="专业版无限制成员...月费99元..."),
    HumanMessage(content="好 我要升级 但是我不知道怎么操作"),
    AIMessage(content="升级步骤如下：登录控制台→账户设置→订阅管理→选择专业版..."),
    HumanMessage(content="我找不到订阅管理这个选项！！你们页面是不是有问题！"),
    AIMessage(content="请问您使用的是网页版还是桌面客户端？不同版本的菜单位置略有不同..."),
    HumanMessage(content="网页版！我都说了三遍了找不到找不到找不到！！！"),
]

summary = generate_handoff_summary(
    test_history,
    {"session_id": "sess_001", "turn_count": 6, "consecutive_failures": 1}
)
print(summary)
```

输出：

```
【会话摘要】
核心诉求：用户想从免费版升级到专业版，但在网页版控制台中找不到"订阅管理"入口。

已尝试方案：
- AI 提供了标准升级路径（控制台→账户设置→订阅管理），用户反馈找不到该选项
- AI 追问了客户端类型（确认是网页版），但尚未给出针对性解决方案

关键信息：无订单号；使用网页版客户端

情绪状态：明显焦虑/烦躁，连续使用了感叹号和重复表达（"找不到""三遍"），建议优先安抚情绪后引导操作。
```

这份摘要让人工客服一眼就能掌握全局，无需翻阅完整聊天记录。

## 完整系统组装

现在我们把所有组件——RAG 问答链、意图分类器、路由分支、记忆管理、Handoff 检测器——组装成一个完整的 `CustomerServiceBot` 类。

### 主系统类

```python
import os
import time
import uuid
from typing import Dict, List, Optional
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

class CustomerServiceBot:
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.rag = CustomerServiceRAG()
        self.rag.initialize()
        self.handoff_detector = HandoffDetector(
            max_failures=self.config.get("max_failures", 3),
            max_turns=self.config.get("max_turns", 20),
        )
        self.session_store: Dict[str, dict] = {}
        self.chat_history_store: Dict[str, InMemoryChatMessageHistory] = {}

    def _get_or_create_session(self, session_id: str = None) -> str:
        if session_id is None:
            session_id = f"sess_{uuid.uuid4().hex[:8]}"
        if session_id not in self.session_store:
            self.session_store[session_id] = {
                "session_id": session_id,
                "created_at": time.time(),
                "turn_count": 0,
                "consecutive_failures": 0,
                "handoff_triggered": False,
                "messages": [],
            }
            self.chat_history_store[session_id] = InMemoryChatMessageHistory()
        return session_id

    def _get_chat_history(self, session_id: str) -> list:
        store = self.chat_history_store.get(session_id)
        if not store:
            return []
        return store.messages

    def process_message(self, user_input: str, session_id: str = None) -> dict:
        session_id = self._get_or_create_session(session_id)
        session = self.session_store[session_id]
        history_store = self.chat_history_store[session_id]

        session["turn_count"] += 1
        session["last_user_input"] = user_input

        history_store.add_message(HumanMessage(content=user_input))

        handoff_decision = self.handoff_detector.check(session)
        if handoff_decision.should_handoff:
            session["handoff_triggered"] = True
            full_history = self._get_chat_history(session_id)
            summary = generate_handoff_summary(full_history, session)

            response = {
                "response": handoff_decision.message_to_user,
                "session_id": session_id,
                "intent": "handoff",
                "handoff": True,
                "handoff_reason": handoff_decision.reason.value if handoff_decision.reason else None,
                "handoff_summary": summary,
                "urgency": handoff_decision.urgency,
            }

            history_store.add_message(AIMessage(content=response["response"]))
            session["messages"].append({"role": "user", "content": user_input})
            session["messages"].append({"role": "assistant", "content": response["response"]})
            return response

        intent_result = classify_with_fallback(user_input, self._get_chat_history(session_id))
        session["last_intent"] = intent_result.intent.value

        state = {
            "user_input": user_input,
            "chat_history": self._get_chat_history(session_id),
            "intent": intent_result.intent.value,
            "entities": intent_result.extracted_entities or {},
        }

        try:
            router_response = self._route_by_intent(state)

            if router_response == "__HANDOFF__":
                session["handoff_triggered"] = True
                full_history = self._get_chat_history(session_id)
                summary = generate_handoff_summary(full_history, session)
                ai_reply = "好的，正在为您转接人工客服，请稍候..."
                response = {
                    "response": ai_reply,
                    "session_id": session_id,
                    "intent": "handoff",
                    "handoff": True,
                    "handoff_summary": summary,
                    "urgency": "normal",
                }
            else:
                session["consecutive_failures"] = 0
                response = {
                    "response": router_response,
                    "session_id": session_id,
                    "intent": intent_result.intent.value,
                    "confidence": intent_result.confidence,
                    "handoff": False,
                }

        except Exception as e:
            session["consecutive_failures"] = session.get("consecutive_failures", 0) + 1
            response = {
                "response": "抱歉，处理您的请求时遇到了一些问题。您可以换个方式再试一次，或者输入「转人工」获取帮助。",
                "session_id": session_id,
                "intent": "error",
                "handoff": False,
                "error": str(e),
            }

        history_store.add_message(AIMessage(content=response["response"]))
        session["messages"].append({"role": "user", "content": user_input})
        session["messages"].append({"role": "assistant", "content": response["response"]})

        return response

    def _route_by_intent(self, state: dict) -> str:
        intent = state.get("intent")

        handlers = {
            "product_inquiry": lambda s: self.rag.query(
                s["user_input"], s["chat_history"]
            ),
            "order_query": handle_order_query,
            "refund_request": handle_refund_request,
            "technical_issue": handle_technical_issue,
            "complaint": handle_complaint,
            "handoff_request": lambda s: "__HANDOFF__",
            "chitchat": handle_chitchat,
            "unknown": handle_unknown,
        }

        handler = handlers.get(intent, handle_unknown)
        return handler(state)

    def get_session_info(self, session_id: str) -> Optional[dict]:
        return self.session_store.get(session_id)

    def list_sessions(self) -> List[dict]:
        return [
            {"session_id": s["session_id"], "turns": s["turn_count"],
             "handoff": s["handoff_triggered"]}
            for s in self.session_store.values()
        ]
```

### CLI 交互程序

有了主系统类之后，我们可以写一个命令行交互界面来体验完整的客服流程：

```python
def run_cli():
    print("=" * 60)
    print("   CloudDesk 智能客服系统")
    print("   输入 'quit' 退出 | 输入 '/sessions' 查看所有会话")
    print("=" * 60)

    bot = CustomerServiceBot()
    current_session = None

    while True:
        try:
            user_input = input("\n您: ").strip()

            if user_input.lower() in ("quit", "exit", "q"):
                print("感谢您的咨询，再见！")
                break

            if user_input == "/sessions":
                sessions = bot.list_sessions()
                if not sessions:
                    print("当前没有活跃会话")
                else:
                    for s in sessions:
                        flag = " [已转人工]" if s["handoff"] else ""
                        print(f"  {s['session_id']} ({s['turns']}轮){flag}")
                continue

            if user_input.startswith("/switch "):
                current_session = user_input.split(maxsplit=1)[1]
                print(f"已切换到会话: {current_session}")
                continue

            result = bot.process_message(user_input, current_session)
            current_session = result["session_id"]

            print(f"\n[{result['intent'].upper()}] AI:", result["response"])

            if result.get("handoff"):
                print(f"\n{'!' * 50}")
                print(f"  ⚠️  已触发人工接管")
                print(f"  原因: {result.get('handoff_reason', '未知')}")
                print(f"  紧急度: {result.get('urgency', 'normal')}")
                if result.get("handoff_summary"):
                    print(f"\n  📋 会话摘要:")
                    print(f"  {result['handoff_summary']}")
                print(f"{'!' * 50}")

        except KeyboardInterrupt:
            print("\n\n已中断，再见！")
            break
        except Exception as e:
            print(f"\n系统错误: {e}")

if __name__ == "__main__":
    run_cli()
```

### 端到端演示

让我们模拟一次完整的客服对话，看看整个系统的运转效果：

```
============================================================
   CloudDesk 智能客服系统
   输入 'quit' 退出 | 输入 '/sessions' 查看所有会话
============================================================

您: 你们免费版有什么限制？

[PRODUCT_INQUIRY] AI: 根据我们的定价方案，**免费版**包含以下限制：
- 团队成员：最多 5 人
- 存储空间：2 GB
- 项目数量：最多 3 个
- API 调用：每月 1,000 次
- 技术支持：社区论坛

如果您团队规模超过 5 人或需要更多资源，可以考虑升级到**专业版（¥99/月）**。需要了解更多详情吗？

您: 专业版多少钱？

[PRODUCT_INQUIRY] AI: **专业版的定价是 ¥99/月**，相比免费版有以下优势：
- 团队成员：无限制
- 存储空间：100 GB
- API 调用：每月 50,000 次
- 额外功能：高级权限管理、审计日志、SSO 单点登录
- 技术支持：邮件支持（24 小时内响应）

升级立即生效，按剩余天数比例计费。如果您是学生，使用 .edu 邮箱注册还可享受 **5 折优惠**！

您: 我的订单 CS-20241088 怎么还没发货啊？都三天了！

[ORDER_QUERY] AI: 正在为您查询订单 **CS-20241088** 的状态，请稍候...
（此处应调用订单查询 API）

您: 如果没发货我要退款

[REFUND_REQUEST] AI: 关于退款申请，请您提供以下信息以便我们处理：
1. 您的注册邮箱或手机号
2. 需要退款的具体原因

您也可以通过以下方式提交退款申请：
- 登录 CloudDesk 控制台 → 账户设置 → 订阅管理 → 申请退款
- 或发送邮件至 support@cloud desk.example.com

你们这是什么破服务！！订单不发货退款还这么麻烦！！！

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  ⚠️  已触发人工接管
  原因: emotional
  紧急度: urgent

  📋 会话摘要:
  【会话摘要】
  核心诉求：用户查询订单 CS-20241088 的发货状态（已等待3天未发货），
  在得知发货延迟后提出退款诉求，对退款流程表示强烈不满。

  已尝试方案：
  - 回答了产品定价相关问题（免费版/专业版）
  - 查询了订单状态（告知需调用API）
  - 提供了退款申请的标准流程

  关键信息：订单号 CS-20241088；等待时间3天

  情绪状态：高度激动/愤怒，使用了"破服务""!!!"等强烈表达，
  强烈建议优先安排资深客服介入，先安抚情绪再处理业务问题。
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
```

可以看到，系统在用户情绪从平静 → 疑虑 → 不满 → 爆发的演进过程中，始终保持了正确的意图识别和合理的回复策略，并在关键时刻准确触发了 Handoff 并生成了高质量的会话摘要。

## FastAPI 服务端部署

CLI 适合开发和调试，生产环境需要一个 Web 服务。我们用 FastAPI 来部署，同时支持流式输出：

```python
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import json

app = FastAPI(title="CloudDesk 智能客服 API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

bot = CustomerServiceBot()

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    intent: str
    handoff: bool = False
    handoff_summary: Optional[str] = None

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    result = bot.process_message(req.message, req.session_id)
    return ChatResponse(**result)

@app.get("/sessions")
async def list_sessions():
    return {"sessions": bot.list_sessions()}

@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    info = bot.get_session_info(session_id)
    if not info:
        raise HTTPException(status_code=404, detail="Session not found")
    return info

@app.get("/health")
async def health_check():
    return {"status": "ok", "active_sessions": len(bot.session_store)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

启动服务后，可以通过 HTTP 接口与客服系统交互：

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "你们免费版支持几个人？"}'
```

返回：

```json
{
  "response": "根据 CloudDesk 的定价方案，免费版最多支持 5 名团队成员...",
  "session_id": "sess_a3f8b2c1",
  "intent": "product_inquiry",
  "handoff": false
}
```

## 扩展方向

本章实现的智能客服系统已经具备了核心功能，但要达到生产级水平，还有几个重要的扩展方向：

**第一，接入真实的外部系统**。目前订单查询、退款处理都是模拟的。实际部署时需要对接公司的 CRM 系统、工单系统、支付网关等。建议通过统一的 `ExternalServiceClient` 封装这些接口，保持核心逻辑不变。

**第二，引入评估与反馈闭环**。每次对话结束后让用户评价回复质量（👍/👎），收集"坏案例"用于优化知识库和 prompt。这是第 12 章（评估与可观测性）的重点内容。

**第三，A/B 测试不同的 LLM 和 Prompt**。在生产环境中同时运行两个版本的模型（比如 GPT-4o-mini vs Claude Haiku），对比它们的首次解决率、平均对话轮数、用户满意度等指标。

**第四，多语言支持**。如果产品面向国际用户，需要在意图分类器和 RAG prompt 中加入多语言处理能力，或者在入口处先做语言检测再分流到对应语言的处理器。

**第五，监控仪表盘**。基于第 9 章学到的回调机制，收集每轮对话的 token 用量、响应延迟、意图分布、Handoff 率等指标，构建实时监控面板。
