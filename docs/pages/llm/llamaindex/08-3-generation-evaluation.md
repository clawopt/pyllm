---
title: 答案质量评估：忠实度 / 准确性 / 有用性
description: RAGAS 框架集成、LLM-as-Judge 模式、Faithfulness/Accuracy/Relevance 指标详解、多维度评分体系
---
# 答案质量评估：忠实度 / 准确性 / 有用性

上一节我们评估了检索阶段——"系统是否找到了正确的文档"。但找到正确的文档只是成功的一半，另一半是**合成阶段能否基于这些文档生成高质量的答案**。这一节我们来评估答案本身的质量。

## 答案质量的三个核心维度

### 维度一：忠实度（Faithfulness）

**定义：** LLM 生成的答案中的事实性主张是否都有检索到的文档作为依据？有没有编造文档中没有的信息？

**为什么这是第一重要的指标？** 因为 RAG 系统的核心价值主张就是"基于文档的可信答案"。如果 LLM 在回答中编造信息，这个价值主张就崩塌了——用户之所以信任 RAG 系统，正是因为它声称"只基于你的文档回答"。

**评估方法一：RAGAS**

[RAGAS](https://github.com/explodinggradients/ragas)（Retrieval Augmented Generation Assessment）是目前最广泛使用的 RAG 评估框架：

```bash
pip install ragas
```

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    answer_correctness,
)

# 准备数据
test_set = [
    {
        "query": "S1 的退货政策是什么？",
        "reference_answer": "支持30天无理由退款，定制产品除外...",
        "retrieved_context": [node.text for node in retrieved_nodes],
        "response": response_text,
    },
    # ... 更多测试用例
]

# 执行评估
result = evaluate(
    test_set,
    metrics=[
        faithfulness,       # 忠实度：答案是否忠于源文档
        answer_relevancy,   # 相关性：答案是否与问题相关
        context_precision,   # 上下文精度：是否利用了相关上下文
        answer_correctness,   # 正确性：答案的事实准确性
    ],
)

print(result)
# 输出示例:
# faithfulness: 0.95  ───────── 大部分内容有据可查 ✅
# answer_relevancy: 0.90  ────── 回答切题 ✅
# context_precision: 0.85  ──── 利用了大部分相关上下文 ✅
# answer_correctness: 0.88  ── 具体事实基本正确 ⚠️ 有小幅偏差
```

**RAGAS 的四大指标解读：**

| 指标 | 衡量什么 | 分数范围 | 好的阈值 |
|------|---------|---------|----------|
| **faithfulness** | 是否编造了文档中没有的信息 | 0-1 | > 0.9 |
| **answer_relevancy** | 答案是否真正回应了查询意图 | 0-1 | > 0.85 |
| **context_precision** | 是否有效利用了检索到的上下文 | 0-1 | > 0.8 |
| **answer_correctness** | 事实性细节（数字、名称等）的准确程度 | 0-1 | > 0.85 |

**评估方法二：LLM-as-Judge（大模型评判）**

如果你不想安装额外的依赖，可以用一个强大的 LLM（如 GPT-4o）来充当评判者：

```python
def llm_as_judge(query, response, reference, source_text, judge_llm):
    """使用 LLM 作为评判者来评估答案质量"""
    prompt = (
        f"你是一个严格但公正的 RAG 答案质量评判员。\n\n"
        f"【任务】\n"
        f"请根据以下信息评判给定答案的质量。\n\n"
        f"【用户问题】\n{query}\n\n"
        f"【参考答案（标准答案）】\n{reference}\n\n"
        f"【参考资料】\n{source_text}\n\n"
        f"【待评估答案】\n{response}\n\n"
        f"请从以下 5 个维度打分（1-10分）：\n"
        f"1. 忠实度(Faithfulness): 答案是否完全基于参考资料，有无编造？\n"
        f"2. 完整性(Completeness): 是否覆盖了参考答案的所有关键点？\n"
        f"3. 准确性(Accuracy): 具体事实（数字、名称）是否正确？\n"
        f"4. 相关性(Relevance): 是否直接回答了用户的问题？\n"
        f"5. 可读性(Readability): 表达是否清晰易懂？\n\n"
        f"请以 JSON 格式输出评分和详细评语:\n"
        f'{{"faithfulness": <score>, "comment": "..."}, ...}}'
    )

    result = judge_llm.complete(prompt)
    return json.loads(result.text)
```

**两种方法的对比：**

| 维度 | RAGAS | LLM-as-Judge |
|------|-------|--------------|
| **速度** | 快（本地运行） | 较慢（API 调用） |
| **成本** | 免费 | 按 token 收费 |
| **一致性** | 高（规则固定） | 中（可能不稳定） |
| **灵活性** | 低（固定 4 指标） | 高（可自定义维度） |
| **推荐场景** | 大批量自动化评估 | 小样本深度分析 |

**建议：** 日常自动化评估用 RASG，定期抽样深度分析用 LLM-as-Judge。

## 构建高质量的标准答案

无论使用哪种评估方法，你都需要为每个评估条目提供**参考答案（ground truth）**。参考答案的质量直接决定了评估的有效性。

### 参考答案编写规范

```python
GOOD_EXAMPLE = {
    "query": "S1 支持哪些无线连接协议？",
    "reference_answer": (
        "S1 智能音箱支持以下无线连接协议：\n"
        "1. Wi-Fi 6 (802.11ax)：主要的数据传输方式，理论速率可达 9.6Gbps\n"
        "2. Bluetooth 5.3：用于音频外设连接（如耳机、音箱），"
        "支持 A2DP 和 LE Audio 两种模式\n"
        "3. Zigbee 3.0：用于智能家居设备互联（如智能灯泡、传感器），"
        "需配合专用网关使用\n"
        "4. Matter（即将通过固件更新支持）：新兴的智能家居统一协议\n\n"
        "注意：S1 不支持以太网、Thread、Z-Wave 或 NFC。"
    ),
}
```

好的参考答案应该：
- **完整**：涵盖所有关键信息点
- **准确**：所有事实性信息正确无误
- **结构化**：分点或分段清晰易读
- **标注例外**：明确说明不支持或不适用的情况

### 差答相似度评估

有时候用户的查询不是只有一个"标准答案"，而是可能有多个合理的回答方式。这时需要引入**答案相似度（Answer Similarity）**的概念——即使措辞不同，只要语义等价就应该算对：

```python
def semantic_similarity(answer, reference):
    """计算答案与参考答案的语义相似度"""
    # 方法一：使用嵌入模型的余弦相似度
    emb_a = embed(answer)
    emb_r = embed(reference)
    return cosine_similarity(emb_a, emb_r)

    # 方法二：使用 LLM 判断
    prompt = f"判断以下两个答案在语义上是否等价：\nA: {answer}\nB: {reference}"
    result = judge_llm.complete(prompt)
    return "等价" in result.text
```

## 多维度综合评分

单一指标往往不能全面反映答案质量。建立一个加权综合评分体系：

```python
class ComprehensiveScorer:
    """多维度答案质量评分器"""

    def __init__(self):
        self.weights = {
            "faithfulness": 0.30,     # 最重要：不编造
            "accuracy": 0.25,           # 很重要：事实正确
            "completeness": 0.20,         # 重要：信息完整
            "relevance": 0.15,            # 较题相关性
            "readability": 0.10,          # 可读性
        }

    def score(self, query, response, reference, source_nodes):
        scores = {}

        # 1. 忠实度评估
        scores["faithfulness"] = self._check_faithfulness(
            response, source_nodes
        )

        # 2. 准确性评估
        scores["accuracy"] = self._check_accuracy(response, reference)

        # 3. 完整性评估
        scores["completeness"] = self._check_completeness(
            response, reference
        )

        # 4. 相关性评估
        scores["relevance"] = self._check_relevance(query, response)

        # 5. 可读性评估
        scores["readability"] = self._check_readability(response)

        # 加权总分
        total = sum(
            score * weight for score, weight in self.weights.items()
        )

        return {
            "total": round(total, 2),
            "breakdown": {k: round(v, 2) for k, v in scores.items()},
        }

    def _check_faithfulness(self, response, sources):
        claims = extract_claims(response)
        source_text = "\n".join([s.text for s in sources])
        supported = sum(1 for c in claims if is_in_source(c, source_text))
        return supported / max(len(claims), 1)

    def _check_accuracy(self, response, reference):
        ref_numbers = extract_numbers(reference)
        ans_numbers = extract_numbers(response)
        matches = sum(1 for n in ans_numbers if any(
            abs(n - r) < r * 0.05 for r in ref_numbers
        ))
        return matches / max(len(ref_numbers), 1)

    def _check_completeness(self, response, reference):
        ref_points = reference.split("。")
        ans_points = response.split("。")
        covered = sum(1 for p in ref_points if any(
            similar(p.strip(), a.strip()) for a in ans_points
        ))
        return covered / max(len(ref_points), 1)

    def _check_relevance(self, query, response):
        prompt = f"以下回答是否充分回应了'{query}'？只需回答 YES 或 NO。"
        result = llm.complete(f"{prompt}\n\n{response}")
        return 1.0 if "YES" in result.text.upper() else 0.0

    def _check_readability(self, response):
        sentences = [s for s in response.split("。") if len(s.strip()) > 10]
        avg_len = sum(len(s) for s in sentences) / max(len(sentences), 1)
        # 太长或太短都不好
        if avg_len < 20:
            return 0.5
        elif avg_len > 200:
            return 0.7
        else:
            return 1.0


scorer = ComprehensiveScorer()
result = scorer.score(query, response, reference, nodes)
print(f"综合得分: {result['total']}/10")
print(f"各维度: {result['breakdown']}")
```

## 从评估到优化的行动路径

评估的价值最终体现在它如何指导优化。以下是评估结果到具体行动的映射关系：

| 评估发现的问题 | 可能的原因 | 建议的优化行动 |
|-------------|-----------|-------------|
| Recall 低但 Precision 正常 | 相关文档没被检索到 | 检查 chunk_size/embedding/top_k；尝试 HyDE；检查文档是否已索引 |
| Faithfulness 低 (< 0.8) | LLM 在编造信息 | 强化 Prompt 约束；启用事后校验；换更强的 LLM |
| Accuracy 低 但 Faithfulness 高 | 细节信息出错 | 检查源文档是否有误；增加精确性约束 Prompt |
| Completeness 低 | 遗漏重要信息 | 增加 top_k；改用 REFINE/TREE_SUMMARIZE；检查 Node 质量 |
| Relevance 低 | 答非所问 | 检查 Query Engine 路由；优化检索查询 |
| 延迟高 (> 5s) | 合成模式太重或 Node 太多 | 切换轻量模式；压缩上下文；启用缓存 |

记住这个表格——每次评估后都应该对照它来确定下一步的优化方向。
