---
title: LangSmith 实战：追踪、调试与性能评估
description: LangSmith 平台入门、Trace 数据采集、运行时可视化调试、Dataset 与评估器集成、性能基线监控
---
# LangSmith 实战：追踪、调试与性能评估

前两节我们讨论了评估的理论框架和具体指标。但指标只是数字——你需要一个平台来**采集数据、可视化执行过程、对比不同版本、自动化运行测试**。这就是 **LangSmith** 登场的时刻。

LangSmith 是 LangChain 官方推出的 LLM 应用开发平台，核心能力覆盖了从开发到生产的全生命周期：**追踪（Tracing）→ 调试（Debugging）→ 评估（Evaluating）→ 监控（Monitoring）**。

## LangSmith 是什么

用一句话概括：**LangSmith 是 LLM 应用的 "Chrome DevTools"**。

就像前端开发者用 Chrome DevTools 查看 HTTP 请求、DOM 结构、控制台日志一样，LLM 应用开发者需要类似的工具来查看：

- 每次调用传给 LLM 的完整 prompt 是什么？
- RAG 检索到了哪些文档片段？排序分数是多少？
- Agent 在每一步选择了哪个工具？输入输出是什么？
- Token 用量、延迟、成本分别是多少？
- 哪些请求失败了？错误信息是什么？

这些信息散落在各个组件的内部，没有一个统一的观测入口就很难排查问题。LangSmith 通过 **自动 Trace 采集**解决了这个问题。

## 快速开始：启用 Trace

### 第一步：获取 API Key

访问 [smith.langchain.com](https://smith.langchain.com) 注册账号（免费额度足够个人开发和测试），然后在 Settings → API Keys 中创建一个 Key。

### 第二步：配置环境变量

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "lsv2_pt_xxxxxxxxxxxx"
# 可选：设置项目名称，用于在 LangSmith 中分组
os.environ["LANGCHAIN_PROJECT"] = "customer-service-bot"
```

只需要这两行配置——**不需要修改任何业务代码**。LangChain 的所有核心组件（ChatModel、Chain、Agent、Retriever、Tool 等）都内置了 Trace 上报逻辑，只要环境变量开启就会自动采集。

### 第三步：运行你的应用

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是 CloudDesk 智能客服助手。"),
    ("human", "{question}"),
])
chain = prompt | llm

result = chain.invoke({"question": "免费版支持几个人？"})
print(result.content)
```

现在打开 [smith.langchain.com](https://smith.langchain.com) 的 Projects 页面，你会看到：

```
📁 customer-service-bot (刚刚)
  └─ 🟢 Chain Run #001  (1.2s)
       ├─ 📝 Prompt: ChatPromptTemplate
       │   └─ 输入: {"question": "免费版支持几个人？"}
       ├─ 🤖 LLM: ChatOpenAI (gpt-4o-mini)
       │   ├── input messages: [System, Human]
       │   ├── output: "根据 CloudDesk 定价方案..."
       │   ├── token_usage: {input: 25, output: 128}
       │   └── latency: 0.89s
       └─ 📤 Output: "根据 CloudDesk 定价方案..."
```

每一次 `invoke()` 调用都会生成一条完整的 Trace 记录，包含**完整的输入、输出、中间状态、Token 统计、耗时**等信息。

## Trace 的核心概念

理解 LangSmith 需要掌握几个核心概念：

### Run（运行记录）

每一次可追踪的操作都是一个 **Run**。Run 有层级关系——一个 Chain Run 包含多个子 Run：

```
Chain Run (agent.invoke)
├── LLM Run (意图分类)
│   ├── input: [system prompt, user message]
│   └── output: {"intent": "product_inquiry"}
├── Retriever Run (RAG 检索)
│   ├── input: "免费版支持几个人？"
│   └── output: [doc1, doc2, doc3]
├── LLM Run (答案生成)
│   ├── input: [system + context + question]
│   └── output: "免费版最多5人..."
└── Chain Run (总耗时: 2.3s)
```

每个 Run 都有唯一的 ID、类型（chain/llm/retriever/tool）、名称、输入输出、元数据（metadata）、开始/结束时间等属性。

### Trace（追踪链）

一条 **Trace** 是一次端到端用户请求产生的所有 Run 的有向无环图（DAG）。它完整地还原了一次调用的执行路径：

```python
from langsmith import traceable

@traceable(name="customer_service_handler")
def handle_customer_query(question: str, session_id: str):
    intent = classify_intent(question)
    if intent == "product_inquiry":
        context = retrieve_context(question)
        answer = generate_answer(question, context)
    else:
        answer = route_to_handler(intent, question)
    return answer
```

加上 `@traceable` 装饰器后，这个函数的整个执行过程都会被记录为一条 Trace，即使函数内部调用了非 LangChain 的代码（比如自定义的分类函数或数据库查询）。

### Project（项目）

Project 是 Trace 的逻辑分组容器。你可以按用途创建不同的 Project：
- `cs-bot-dev` — 开发环境的客服机器人
- `cs-bot-prod` — 生产环境的客服机器人
- `code-agent-eval` — 代码分析助手的评估测试
- `ab-test-v1-vs-v2` — A/B 测试对比

## 可视化调试：在 LangSmith 中排查问题

### 场景一：回答质量差——查看检索内容

用户反馈："问定价问题经常答错"。你在 LangSmith 中找到几条相关的 Trace：

```
Trace #1234 "专业版多少钱？"
├── Retriever Run
│   └── output:
│       [1] pricing.md: "免费版...专业版 ¥99/月...企业版..."
│       [2] faq.md: "学生优惠 5 折..."  
│       [3] policies.md: "退款政策..."
├── LLM Run (生成)
│   └── output: "专业版的定价是 ¥79/月..."  ← ❌ 错误！应该是 ¥99
```

问题一目了然：检索到的上下文中明确写着"¥99/月"，但 LLM 生成了"¥79"。这说明是**忠实度问题（幻觉）**而非检索问题。解决方案：在 system prompt 中加强约束——"必须严格使用参考材料中的数字"。

### 场景二：响应慢——定位瓶颈

某次调用耗时 8 秒，远超正常水平（通常 2-3 秒）。在 Trace 详情中查看每个 Run 的耗时：

```
Trace #5678 (总耗时: 8.2s)
├── Intent Classification: 0.8s     ← 正常
├── RAG Retrieval: 0.3s           ← 正常
├── Answer Generation: 6.9s       ← ⚠️ 异常！
│   ├── first_token_time: 1.2s    ← TTFT 偏高
│   ├── total_tokens: 512         ← 输出较长
│   └── model: gpt-4o (不是 mini)
└── Metadata: user_tier=vip
```

发现两个问题：
1. 这次调用用了 `gpt-4o` 而不是 `gpt-4o-mini`（可能是 VIP 用户路由逻辑触发了模型升级）
2. 输出了 512 个 token，比平均（~150）多很多

### 场景三：Agent 循环——查看推理过程

Agent 的多步推理过程在 LangSmith 中以树状图展示：

```
Trace #9012 "帮我查订单 CS-20241088 并退款"
│
├── 🔵 Step 1: LLM (意图识别)
│   Thought: 用户想查订单并退款，需要先查订单状态
│   Action: query_order("CS-20241088")
│
├── 🟢 Step 2: Tool (order_query)
│   Input: {"order_id": "CS-20241088"}
│   Output: {"status": "shipped", "amount": 299.00, ...}
│
├── 🔵 Step 3: LLM (决策)
│   Thought: 订单已发货不能直接退款，需要走退货流程...
│   Action: search_knowledge("已发货订单退货流程")
│
├── 🟢 Step 4: Tool (rag_search)
│   Input: "已发货订单退货流程"
│   Output: [退货政策文档片段...]
│
├── 🔵 Step 5: LLM (最终回答)
│   Final Answer: 您的订单已经发货了...
│
└── ✅ 完成 (5 步, 4.5s)
```

每一步的输入输出都清晰可见。如果 Agent 在某一步做出了错误的决策（比如应该查订单却去搜索知识库），你可以精确定位到那一步并分析原因。

## Dataset：构建评估基准集

LangSmith 不仅是一个 Trace 平台，还是一个**评估平台**。它的 Dataset 功能让你可以管理评估用的测试数据和标准答案。

### 创建 Dataset

```python
from langsmith import Client

client = Client()

dataset_name = "cs-bot-eval-v1"
dataset = client.create_dataset(
    dataset_name=dataset_name,
    description="智能客服系统评估数据集 v1.0",
    data_type="kv",
)

test_cases = [
    {"input": {"question": "免费版支持几个人？"},
     "output": "免费版最多支持5名团队成员"},
    {"input": {"question": "专业版一个月多少钱？"},
     "output": "专业版月费99元"},
    {"input": {"question": "我想退款，怎么操作？"},
     "output": "退款需在订单完成后30天内申请"},
    {"input": {"question": "转人工"},
     "output": "__HANDOFF__"},
    {"input": {"question": "你们的产品能离线使用吗？"},
     "output": "暂时无法解答"},
]

for case in test_cases:
    client.create_example(
        inputs=case["input"],
        outputs=case["output"],
        dataset_id=dataset.id,
    )

print(f"✅ 数据集创建完成: {dataset_name} ({len(test_cases)} 条)")
```

### 运行在线评估

有了 Dataset 之后，可以用 LangSmith 内置的评估器对每条测试数据自动打分：

```python
from langsmith.evaluation import evaluate, LangSmithStringEvaluator

def cs_bot_predict(inputs: dict) -> dict:
    bot = CustomerServiceBot()
    bot.initialize()
    result = bot.process_message(inputs["question"])
    return {"response": result["response"]}

faithfulness_evaluator = LangSmithStringEvaluator(
    evaluator="criteria:faithfulness",
)

answer_relevance_evaluator = LangSmithStringEvaluator(
    evaluator="criteria:relevance",
)

dataset_name = "cs-bot-eval-v1"

results = evaluate(
    cs_bot_predict,
    data=dataset_name,
    evaluators=[faithfulness_evaluator, answer_relevance_evaluator],
    max_concurrency=4,
    experiment_prefix="eval-run-20250406",
)

print(f"\n评估完成！")
print(f"  忠实度平均分: {results.get('faithfulness_scores', {}).get('mean', 'N/A')}")
print(f"  相关性平均分: {results.get('relevance_scores', {}).get('mean', 'N/A')}")
```

评估结果会自动保存在 LangSmith 的 Experiments 页面中，你可以：
- 对比不同版本的评估结果
- 查看每条测试 case 的详细评分和评语
- 筛选出得分低的 case 进行重点分析
- 设置质量门禁（低于阈值时阻止部署）

## 性能基线与回归检测

除了功能正确性，**性能稳定性**同样重要。LangSmith 可以帮你建立性能基线并在每次变更后做回归检测。

### 收集性能指标

```python
@traceable(name="cs_bot_with_metrics")
def tracked_bot_process(question: str, session_id: str = None):
    import time
    start = time.time()

    bot = CustomerServiceBot()
    bot.initialize()
    result = bot.process_message(question, session_id)

    latency = time.time() - start

    return {
        "response": result["response"],
        "intent": result.get("intent"),
        "handoff": result.get("handoff", False),
        "metrics": {
            "latency_seconds": round(latency, 3),
            "token_count_estimate": len(result["response"]) // 4,
            "session_turns": bot.session_store.get(session_id, {}).get("turn_count", 1),
        },
    }
```

### 在 LangSmith 中设置告警

LangSmith 支持基于 Trace 数据的自动告警规则：

| 监控项 | 告警条件 | 含义 |
|--------|---------|------|
| P95 延迟 | > 5s | 大部分请求变慢 |
| 错误率 | > 5% | 系统不稳定 |
| Handoff 率 | > 20% | AI 解决能力不足 |
| 平均 Token 数 | > 基线 × 2 | 可能被 prompt 注入攻击 |
| 成本/请求 | > 基线 × 1.5 | 成本异常 |

当触发告警时，LangSmith 可以发送通知到 Slack、Email 或 webhook。

## 本地替代方案：LangSmith 不想用怎么办？

LangSmith 不是唯一的选择。如果你的场景不适合使用云端服务（如数据安全要求严格），以下替代方案值得考虑：

### 方案一：LangFuse（开源自托管）

```python
# pip install langfuse
import os
os.environ["LANGfuse_PUBLIC_KEY"] = "pk-lf-..."
os.environ["LANGUSE_SECRET_KEY"]="sk-lf-..."
os.environ["LANGUSE_HOST"] = "http://localhost:3000"

from langfuse.callback import CallbackHandler
langfuse_handler = CallbackHandler()

chain.invoke({"question": "测试"}, config={"callbacks": [langfuse_handler]})
```

LangFuse 是开源的，可以自托管在你的基础设施上。功能与 LangSmith 类似，支持 Trace、评估、用户反馈收集等。

### 方案二：纯本地 Callback（第 9 章方案）

如果连自托管都不想要，我们在第 9 章学过的 `BaseCallbackHandler` 仍然是最轻量的选择：

```python
class LocalMetricsCollector(BaseCallbackHandler):
    def __init__(self):
        self.runs = []

    def on_llm_end(self, response, **kwargs):
        self.runs.append({
            "type": "llm",
            "tokens": response.llm_output["token_usage"],
            "latency": kwargs.get("run_id"),
        })

    def on_chain_end(self, outputs, **kwargs):
        self.runs.append({
            "type": "chain",
            "output_keys": list(outputs.keys()),
        })

    def get_summary(self):
        total_tokens = sum(r.get("tokens", {}).get("total_tokens", 0)
                         for r in self.runs if r["type"] == "llm")
        return {"total_runs": len(self.runs), "total_tokens": total_tokens}
```

### 三种方案对比

| 维度 | LangSmith | LangFuse | 本地 Callback |
|------|-----------|----------|--------------|
| **部署** | 云端 SaaS | 自托管 | 无需部署 |
| **功能完整性** | 最全 | 接近完整 | 基础 |
| **评估集成** | 内置评估器 | 支持 | 手动实现 |
| **团队协作** | 原生支持 | 支持 | 不支持 |
| **数据安全** | 数据上传到云端 | 数据在你手中 | 数据不离开本机 |
| **成本** | 免费额度+付费 | 服务器成本 | 免费 |
| **推荐场景** | 团队协作 / 生产级 | 有安全合规要求 | 个人开发 / 快速验证 |

无论选择哪种方案，关键原则不变：**可观测性应该是内置的而不是事后补丁**。从一开始就把 Trace / 日志 / 指标采集作为架构的一部分，后续的调试和优化才能事半功倍。
