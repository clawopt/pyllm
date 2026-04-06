---
title: 评估指标：准确率、忠实度、答案相关性
description: RAG 三角评估框架、各指标的精确定义与计算方法、LLM-as-Judge 实现方案、RAGAS 集成使用
---
# 评估指标：准确率、忠实度、答案相关性

上一节我们讨论了"为什么要评估"和评估的层次体系。这一节要解决的是**具体怎么衡量**——用哪些指标、如何定义、如何计算。

## RAG 评估的"铁三角"

学术界和工业界经过大量实践，逐渐形成了一套被广泛接受的 RAG 评估框架——**RAG 三角（RAG Triad）**：

```
                 ┌──────────────┐
                 │   准确率     │
                 │ (Answer      │
                 │  Relevance)  │
                 └──────┬───────┘
                        │
           ┌────────────┼────────────┐
           │            │            │
    ┌──────▼──────┐    │    ┌───────▼──────┐
    │   忠实度     │    │    │   检索质量    │
    │(Faithfulness│    │    │(Context       │
    │  to Context) │    │    │ Relevance)   │
    └─────────────┘    │    └──────────────┘
                       │
              用户问题 (Query)
```

三个顶点分别对应一个核心指标，每个指标回答一个关键问题：

| 指标 | 核心问题 | 一句话解释 |
|------|---------|-----------|
| **检索相关性** (Context Relevance) | 检索到的内容是否与问题相关？ | "找对资料了吗？" |
| **忠实度** (Faithfulness) | 回答是否基于检索到的内容？ | "有没有瞎编？" |
| **答案相关性** (Answer Relevance) | 回答是否解决了用户的问题？ | "有没有答到点子上？" |

下面我们逐一深入每个指标，给出精确定义和可执行的实现代码。

## 指标一：检索相关性（Context Relevance）

### 定义

**检索相关性**衡量的是：系统从知识库中检索出来的文档片段，有多少比例是真正与用户问题相关的。

为什么这个指标很重要？因为 **RAG 的质量上限由检索阶段决定**。如果检索阶段就漏掉了包含正确答案的文档，无论 LLM 多强大都不可能生成正确的回答。

### 计算方法

给定一个问题 Q 和检索返回的一组文档片段 C = {c₁, c₂, ..., cₙ}：

```
Context Relevance = (相关片段数) / n × 100%
```

其中"相关"的判定标准是：该片段中**至少有一句话**直接有助于回答问题 Q。

### 实现方式

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

class ContextRelevanceResult(BaseModel):
    relevant_chunks: list[bool] = Field(
        description="每个检索片段是否相关的布尔列表"
    )
    relevance_score: float = Field(
        description="整体检索相关性得分 (0.0 - 1.0)"
    )
    reasoning: str = Field(description="评分理由")

CONTEXT_RELEVANCE_PROMPT = """你是一个 RAG 系统的质量评估专家。
请评估以下检索结果与用户问题的相关性。

【用户问题】
{question}

【检索到的文档片段】
{contexts}

## 评估规则
对于每个文档片段，判断它是否包含能够帮助回答用户问题的信息：
- 即使只包含部分相关信息，也标记为 True
- 如果完全无关或跑题，标记为 False

然后计算整体相关性得分 = 相关片段数 / 总片段数。"""

context_relevance_chain = (
    ChatPromptTemplate.from_messages([
        ("system", CONTEXT_RELEVANCE_PROMPT),
        ("human", "请评估"),
    ])
    | ChatOpenAI(model="gpt-4o", temperature=0.0)
)
```

测试示例：

```python
question = "免费版支持几个人？"
retrieved_contexts = [
    "[1] 免费版最多支持 5 名团队成员，存储空间 2 GB...",
    "[2] 专业版（¥99/月）支持无限成员，存储空间 100 GB...",
    "[3] 企业版需要联系销售获取报价，支持定制集成开发...",
]

result = context_relevance_chain.invoke({
    "question": question,
    "contexts": "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(retrieved_contexts)),
})

print(result.relevant_chunks)     # [True, False, False]
print(result.relevance_score)     # 0.333...
print(result.reasoning)
# "片段1明确包含了'5名团队成员'的信息，直接回答了问题。
#  片段2和3讨论的是其他版本，与'免费版人数限制'无直接关系。"
```

### 常见优化策略

当检索相关性偏低时，通常意味着以下问题之一：
1. **chunk_size 不合适**：太小的 chunk 导致信息碎片化，太大的 chunk 引入噪声
2. **embedding 模型不匹配**：领域术语在通用 embedding 中语义距离远
3. **k 值设置不当**：k 太小可能遗漏，k 太大引入噪声
4. **metadata 过滤缺失**：没有利用文件类型/日期等元信息缩小搜索范围

## 指标二：忠实度（Faithfulness）

### 定义

**忠实度**（也叫 Groundedness 或 Attribution）衡量的是：模型生成的回答中的每一条事实性声明，是否都能在检索到的上下文中找到依据。

这是 RAG 评估中最重要也最独特的指标——传统 NLP 评估中没有对应概念。它的核心诉求是：**RAG 系统不应该产生幻觉**。如果知识库说"5个人"，回答就不能说"10个人"；如果知识库没提到某个功能，回答就不应该声称有这个功能。

### 计算方法

```
Faithfulness = (可验证为真的声明数) / (总声明数) × 100%
```

具体步骤：
1. 把模型的回答拆解为若干条独立的**事实性声明**
2. 对每条声明，检查它是否能从检索上下文中推断出来
3. 统计比例

### 实现方式

```python
class FaithfulnessResult(BaseModel):
    claims: list[str] = Field(
        description="从回答中提取的事实性声明列表"
    )
    verdicts: list[bool] = Field(
        description="每条声明能否被上下文支持的布尔列表"
    )
    faithfulness_score: float = Field(
        description="忠实度得分 (0.0 - 1.0)"
    )

FAITHFULNESS_PROMPT = """你是一个严格的事实核查员。
你的任务是判断 AI 的回答是否忠实地基于给定的参考材料。

【参考材料（检索到的上下文）】
{context}

【AI 的回答】
{answer}

## 评估步骤
1. 从 AI 回答中提取所有独立的事实性声明（数字、名称、关系、属性等）
2. 对每条声明，判断它是否能从参考材料中直接找到依据
3. 注意：
   - 如果参考材料没有提及某事，而回答中提到了 → 不忠实 ❌
   - 如果回答中的数字与参考材料不一致 → 不忠实 ❌
   - 如果回答做了合理的概括但不改变原意 → 忠实 ✅
   - 如果回答添加了外部知识（即使正确）→ 不忠实 ❌"""

faithfulness_chain = (
    ChatPromptTemplate.from_messages([
        ("system", FAITHFULNESS_PROMPT),
        ("human", "开始评估"),
    ])
    | ChatOpenAI(model="gpt-4o", temperature=0.0)
)
```

测试——一个有幻觉的回答：

```python
context = "免费版最多支持 5 名团队成员，存储空间 2 GB，项目数量最多 3 个"

answer_with_hallucination = (
    "根据产品信息，免费版支持 5 名团队成员和 2 GB 存储空间。"
    "此外，免费版还支持无限的项目数量和每月 5000 次 API 调用。"
    "如果需要更多功能，可以升级到专业版，月费 79 元。"
)

result = faithfulness_chain.invoke({
    "context": context,
    "answer": answer_with_hallucination,
})

for claim, verdict in zip(result.claims, result.verdicts):
    status = "✓" if verdict else "✗"
    print(f"{status} {claim}")

print(f"\n忠实度得分: {result.faithfulness_score:.2f}")
```

输出：

```
✓ 免费版支持 5 名团队成员
✓ 存储空间为 2 GB
✗ 支持无限的项目数量（参考材料说是"最多3个"，不是"无限"）
✗ 每月 5000 次 API 调用（参考材料未提及此信息）
✗ 专业版月费 79 元（参考材料说的是 ¥99，不是 ¥79）

忠实度得分: 0.40
```

5 条声明中有 3 条有问题——两条是纯 hallucination（编造了不存在的信息），一条是数值错误。忠实度得分只有 40%，说明这个回答质量很差。

### 为什么忠实度比准确率更根本

你可能想问："为什么不直接对比回答和标准答案？"原因在于：

1. **标准答案不一定存在**。很多 RAG 场景下没有唯一的"正确答案"
2. **表达方式可以不同**。"5人"和"五名团队成员"意思一样
3. **忠实度检测的是过程而非结果**——它确保回答**来源可靠**，这比"碰巧答对"更有意义

## 指标三：答案相关性（Answer Relevance）

### 定义

**答案相关性**衡量的是：生成的回答是否完整地回应了用户的问题。

注意这与"回答是否正确"不完全相同。一个回答可能是**正确的但不完整的**：

```
Q: "免费版和专业版的区别是什么？"

A_部分相关: "免费版支持5人，专业版支持无限人。"
  → 只回答了人数差异，遗漏了价格、存储、API 等维度 → 相关性低

A_高度相关: "免费版和专业版的主要区别如下：
  1. 团队人数：免费版5人 vs 专业版无限
  2. 存储：2GB vs 100GB
  3. 价格：免费 vs ¥99/月
  4. API 调用：1,000次/月 vs 50,000次/月
  ..."
  → 全面覆盖了各个维度的差异 → 相关性高
```

### 计算方法

```
Answer Relevance = (回答覆盖的用户意图维度数) / (应该覆盖的总维度数) × 100%
```

实际操作中，我们让 LLM 判断回答是否充分回应了问题：

```python
class AnswerRelevanceResult(BaseModel):
    is_relevant: bool = Field("回答是否总体上解决了用户的问题")
    coverage_score: float = Field("完整性得分 (0.0 - 1.0)")
    missing_aspects: list[str] = Field("回答中遗漏的关键方面")
    redundant_info: list[str] = Field("回答中的冗余/无关信息")

ANSWER_RELEVANCE_PROMPT = """你是一个问答质量评估专家。
评估以下回答是否有效回应用户的问题。

【用户问题】
{question}

【AI 回答】
{answer}

## 评估维度
1. **直接性**: 是否正面回答了问题？
2. **完整性**: 是否涵盖了问题的所有关键方面？
3. **精确性**: 提供的信息是否有足够的细节？
4. **简洁性**: 是否包含不必要的冗余信息？

请列出回答遗漏的关键方面和多余的无关信息。"""

answer_relevance_chain = (
    ChatPromptTemplate.from_messages([
        ("system", ANSWER_RELEVANCE_PROMPT),
        ("human", "请评估"),
    ])
    | ChatOpenAI(model="gpt-4o", temperature=0.0)
)
```

## 用 RAGAS 实现一键评估

上面的实现都是手写的 prompt + 解析器。在生产环境中，你可以使用 **RAGAS（Retrieval Augmented Generation Assessment）** 这个开源库——它是目前最成熟的 RAG 自动评估框架：

```python
# pip install ragas

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    context_relevancy,
    faithfulness,
    answer_relevancy,
)

eval_dataset = Dataset.from_dict({
    "question": [
        "免费版支持几个人？",
        "专业版多少钱一个月？",
        "退款政策是什么？",
    ],
    "answer": [
        "免费版最多支持5名团队成员。",
        "专业版月费99元。",
        "订单完成后30天内可申请退款。",
    ],
    "contexts": [
        ["免费版最多支持5名团队成员..."],
        ["专业版（¥99/月）..."],
        ["退款需在订单完成后30天内..."],
    ],
    "ground_truth": [
        "免费版最多5名团队成员",
        "专业版99元/月",
        "30天内可申请退款",
    ],
})

result = evaluate(
    eval_dataset,
    metrics=[
        context_relevancy,
        faithfulness,
        answer_relevancy,
    ],
)

df_result = result.to_pandas()
print(df_result[["context_relevancy", "faithfulness", "answer_relevancy"]])
```

输出：

```
   context_relevancy  faithfulness  answer_relevancy
0             0.95         1.00            0.90
1             1.00         1.00            0.85
2             0.80         0.75            0.70
```

RAGAS 会自动处理声明提取、逐句核验、维度覆盖分析这些繁琐工作，输出 0-1 之间的标准化分数。

### RAGAS 支持的完整指标清单

| 指标 | 英文名 | 衡量什么 | 适用场景 |
|------|--------|---------|---------|
| 检索相关性 | Context Precision / Recall | 检索质量 | 所有 RAG |
| 忠实度 | Faithfulness | 有无幻觉 | 所有 RAG |
| 答案相关性 | Answer Relevance | 回答完整性 | 所有 RAG |
| 答案相似度 | Answer Similarity | 与标准答案的语义相似度 | 有 ground truth 时 |
| 答案正确性 | Answer Correctness | 与标准答案的一致性 | 有 ground truth 时 |
| 上下文实体召回 | Context Entity Recall | 关键实体是否都被检索到 | 结构化数据场景 |

## Agent 专用指标

除了 RAG 三角之外，Agent 应用还需要额外的评估维度：

### 工具调用正确率（Tool Call Accuracy）

Agent 的核心能力是选择正确的工具并传入正确的参数：

```python
def evaluate_tool_calls(expected_tools: list, actual_tools: list) -> dict:
    expected_set = {(t["name"], frozenset(t.get("args", {}).items()))
                    for t in expected_tools}
    actual_set = {(t["name"], frozenset(t.get("args", {}).items()))
                  for t in actual_tools}

    correct = len(expected_set & actual_set)
    total_expected = len(expected_set)
    extra_calls = len(actual_set - expected_set)

    return {
        "precision": correct / len(actual_set) if actual_set else 0,
        "recall": correct / total_expected if total_expected else 0,
        "f1": 2 * correct / (len(actual_set) + total_expected) if (actual_set or total_expected) else 0,
        "extra_tool_calls": extra_calls,
    }
```

### 目标达成率（Goal Achievement Rate）

对于多步任务（如"帮我调研 Python 和 Go 的并发性能差异并写报告"），最终判断标准是：**任务目标是否达成**？

```python
GOAL_ACHIEVEMENT_PROMPT = """以下是一个 Agent 执行任务的记录。
判断 Agent 是否成功完成了用户的原始目标。

【用户目标】
{goal}

【Agent 执行过程】
{trajectory}

【最终输出】
{final_output}

请给出:
1. goal_achieved: true/false
2. achievement_score: 0.0-1.0
3. missing_steps: 未完成的关键步骤
4. quality_notes: 输出质量评价"""
```

### 对话轮次效率（Turn Efficiency）

衡量 Agent 解决一个问题平均需要多少轮交互：

```python
def calculate_turn_efficiency(conversations: list) -> dict:
    turns_per_conversation = [len(conv["messages"]) // 2 for conv in conversations]
    return {
        "avg_turns": sum(turns_per_conversation) / len(turns_per_conversation),
        "max_turns": max(turns_per_conversation),
        "min_turns": min(turns_per_conversation),
        "single_turn_rate": sum(1 for t in turns_per_conversation if t == 1) / len(turns_per_conversation),
    }
```

轮次越少通常越好——说明 Agent 能快速定位问题并给出答案。但如果 single_turn_rate 过高（比如 >80%），也可能说明 Agent 在回避复杂问题而不是真正解决它们。

## 三个指标的综合解读

拿到三个分数之后，如何综合判断系统的健康状态？以下是常见的组合模式及其含义：

| Context Relevance | Faithfulness | Answer Relevance | 诊断 | 优先行动 |
|-------------------|-------------|------------------|------|---------|
| 高 | 高 | 高 | ✅ 健康 | 保持监控 |
| 低 | 高 | 中 | ⚠️ 检索弱 | 优化 embedding / 分块 / k 值 |
| 高 | 低 | 中 | 🟠 幻觉多 | 加强 prompt 约束（"不知道就说不知道"） |
| 高 | 高 | 低 | 🔵 回答不全 | 优化生成 prompt（要求全面覆盖） |
| 低 | 低 | 低 | 🔴 全面问题 | 从头排查整个 pipeline |

这种诊断矩阵能帮你**精准定位瓶颈所在**，避免盲目优化。
