---
title: 合成质量优化与常见问题
description: 合成质量的评估指标、幻觉检测与抑制、长答案的质量控制、Token 效率优化、生产级调优清单
---
# 合成质量优化与常见问题

经过前面几节的学习，你已经掌握了响应合成的核心概念、四种内置模式、自定义 Synthesizer、Prompt 工程和结构化输出。最后一节我们来讨论一个实际工程中最关心的问题：**如何确保合成质量达到生产级别的要求？出了问题怎么排查和修复？**

## 合成质量的评估维度

在优化之前，我们需要定义"什么是好的合成质量"。我认为应该从以下五个维度来评估：

### 维度一：忠实度（Faithfulness）

**定义：** 答案中的每一项事实主张都能在检索到的文档中找到依据。

**评估方法：**
```python
def evaluate_faithfulness(response, source_nodes):
    """检查答案是否有超出源文档的主张"""
    claims = extract_claims(response.response)  # 提取事实主张
    source_text = "\n".join([n.text for n in source_nodes])

    unsupported = []
    for claim in claims:
        if not is_substantiated(claim, source_text):
            unsupported.append(claim)

    faith_score = 1 - len(unsupported) / max(len(claims), 1)
    return faith_score, unsupported
```

**目标值：** > 0.95（即少于 5% 的主张无法在源文档中找到依据）

### 维度二：完整性（Completeness）

**定义：** 检索到的相关文档中的重要信息是否都被包含在了答案中。

**评估方法：**
```python
def evaluate_completeness(response, source_nodes, expected_info):
    """检查期望的信息点是否都在答案中出现"""
    expected_points = expected_info.split(",")  # 预期的关键信息点
    answer_text = response.response.lower()

    found = sum(1 for p in expected_points if p.lower() in answer_text)
    completeness = found / len(expected_points)
    return completeness, found, expected_points
```

**目标值：** > 0.85（大部分关键信息都被覆盖）

### 维度三：准确性（Accuracy）

**定义：** 答案中的事实信息是否正确（没有张冠李戴、没有数字错误）。

**评估方法：** 通常需要人工评估或使用另一个更强的 LLM 作为评判者（LLM-as-Judge）。第八章会详细讲评估体系。

**目标值：** > 0.90

### 维度四：可读性（Readability）

**定义：** 答案是否通顺易读、结构清晰、符合目标用户的阅读习惯。

**评估指标：**
- 平均句长（适中为佳）
- 专业术语的使用是否恰当
- 是否有清晰的段落/列表结构
- 语法和拼写错误率

### 维度五：效率（Efficiency）

**定义：** 生成答案所消耗的 token 数和时间是否合理。

**评估指标：**
- 总 token 消耗（输入 + 输出）
- LLM 调用次数
- 端到端延迟

## 幻觉检测与抑制

**幻觉（Hallucination）** 是 RAG 系统最大的敌人之一——LLM 可能"自信地"编造文档中不存在的信息。这在响应合成阶段尤为危险，因为用户倾向于信任 RAG 系统（毕竟它声称"基于文档"）。

### 幻觉的常见形式

**形式一：编造细节。**
```
文档说: "保修期 24 个月"
LLM 回答: "保修期 24 个月，且在此期间免费提供一次上门服务"
                                              ↑ 这部分是编造的！
```

**形式二：混淆归属。**
```
文档 A 说: "S1 价格 299 元"
文档 B 说: "S2 支持 Bluetooth LE Audio"
LLM 回答: "S1 支持 Bluetooth LE Audio 且价格为 299 元"
                         ↑ 把 B 的属性安到了 A 上
```

**形式三：过度推断。**
```
文档说: "支持 Wi-Fi 6"
LLM 回答: "支持 Wi-Fi 6 (802.11ax)，理论速率可达 9.6Gbps"
                                  ↑ 具体标准是文档没提的
```

### 抑制策略

**策略一：强化 Prompt 约束**

```python
no_hallucination_template = PromptTemplate(
    "基于以下资料回答问题。严格遵守以下规则：\n\n"
    "【零容忍规则】\n"
    "- 绝对不要编造文档中没有的信息\n"
    "- 如果文档中没有答案，直接说'根据现有资料无法确定'\n"
    "- 不要使用你的通用知识来补充细节\n"
    "- 不要猜测数值、日期或具体名称\n"
    "- 如果某个信息只出现了一部分，只回答这部分，不要补全\n"
    "- 对每个关键事实，在心里标记'(来源: xxx)'确保有据可查\n\n"
    "资料:\n{context_str}\n\n"
    "问题:\n{query_str}"
)
```

**策略二：引用强制**

要求 LLM 在每段陈述后面显式标注来源：

```python
citation_template = PromptTemplate(
    ...
    "回答要求：\n"
    "- 每个关键事实陈述后必须标注 (来源: 文件名)\n"
    "- 格式如: '...价格为299元 (来源: price_list.md)'\n"
    "- 未标注来源的信息将被视为不可信\n"
    ...
)
```

**策略三：事后校验（Post-hoc Validation）**

在合成完成后，用一个独立的 LLM 调用来检验答案的忠实度：

```python
def post_hoc_validation(response, source_nodes, validator_llm):
    """事后校验：检查答案是否忠于源文档"""
    source_text = "\n".join([
        f"[Source {i+1}] {n.text}" for i, n in enumerate(source_nodes)
    ])

    validation_prompt = (
        f"你是一个事实核查员。请核查以下回答是否忠实地基于给定的参考资料。\n\n"
        f"参考资料:\n{source_text}\n\n"
        f"待核查的回答:\n{response.response}\n\n"
        f"请逐一检查：\n"
        f"1. 回答中的每个事实主张是否出现在参考资料中？\n"
        f"2. 有没有任何编造或添加的细节吗？\n"
        f"3. 有没有把 A 文档的信息错归于 B 文档？\n\n"
        f"输出格式：\n"
        f"- 如果完全忠实，输出: VALIDATED:PASS\n"
        f"- 如果有问题，输出: VALIDATED: FAIL 及具体问题描述"
    )

    result = validator_llm.complete(validation_prompt).text
    is_valid = "PASS" in result.upper()

    return is_valid, result
```

注意：事后校验会增加一次额外的 LLM 调用（和对应的成本），所以建议只在**高风险场景**（法律、医疗、金融）中使用，或者**抽样校验**（比如每 10 个请求校验 1 个）。

## 长答案的质量控制

当答案很长时（超过 500 字），质量问题会被放大——越长越容易出错、越容易前后矛盾、越容易让读者迷失方向。

### 控制技巧一：前置摘要

要求 LLM 在详细展开之前先给出一句话的摘要：

```python
long_answer_template = PromptTemplate(
    ...
    "输出要求：\n"
    "- 先用一句话概括核心结论（20字以内）\n"
    "- 然后分点展开详细信息\n"
    "- 最后给出行动建议（如有）\n"
    "- 总长度控制在 800 字以内\n"
    ...
)
```

### 控制技巧二：信号词分段

在 Prompt 中定义清晰的结构标记：

```python
structured_long_template = PromptTemplate(
    ...
    "请按以下结构组织你的回答：\n\n"
    "=== 概述 ===\n"
    "[一句话总结]\n\n"
    "=== 详细说明 ===\n"
    "[分点展开]\n\n"
    "=== 注意事项 ===\n"
    "[例外情况和提醒]\n\n"
    "=== 信息来源 ===\n"
    "[关键信息来自哪些文件]"
    ...
)
```

### 控制技巧三：迭代截断

对于 TREE_SUMMARIZE 或 REFINE 模式生成的超长答案，可以在后处理阶段做截断：

```python
def truncate_long_answer(response, max_chars=1500):
    if len(response.response) <= max_chars:
        return response

    # 保留开头（通常是最重要的概述部分）
    truncated = response.response[:max_chars]

    # 找到最后一个完整的句子结尾
    last_period = truncated.rfind("。")
    if last_period > max_chars * 0.7:
        truncated = truncated[:last_period + 1]

    truncated += "\n\n...(答案较长，已截断。如需完整版本请针对性提问)"

    response.response = truncated
    return response
```

## Token 效率优化

Token 是 RAG 系统的主要成本之一。优化 Token 消耗直接影响运营成本。

### 优化一：选择合适的合成模式

| 场景 | 推荐 | 预估 Token 消耗（5 个 Node） |
|------|------|---------------------------|
| 简单事实查询 | SIMPLE_SUMMARIZE | ~2000 |
| 中等复杂度 | REFINE | ~4000-6000 |
| 大量信息 | TREE_SUMMARIZE | ~4500-6000 |

### 优化二：压缩上下文

在发送给 Synthesizer 之前，先压缩 Node 文本：

```python
def compress_nodes_for_synthesis(nodes, max_total_chars=3000):
    """压缩 Node 文本以减少 Token 消耗"""
    compressed = []
    total = 0

    for node in nodes:
        text = node.text
        if total + len(text) > max_total_chars:
            # 截断最后一个 Node
            remaining = max_total_chars - total
            compressed.append(TextNode(text=text[:remaining], metadata=node.metadata))
            break
        compressed.append(node)
        total += len(text)

    return compressed
```

### 优化三：缓存相似查询的合成结果

如果多个用户问了相同或高度相似的问题，没必要每次都重新合成：

```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=100)
def cached_synthesize(query_hash, nodes_hash_tuple):
    """带缓存的合成"""
    nodes = [TextNode(text=t, metadata={}) for t in eval(nodes_hash_tuple)]
    return actual_synthesizer.synthesize(query, nodes)


def get_query_hash(query):
    return hashlib.md5(query.encode()).hexdigest()


def smart_synthesize(query, nodes):
    qhash = get_query_hash(query)
    nhash = str(hash(tuple((n.text for n in nodes))))
    return cached_synthesize(qhash, nhash)
```

## 生产级 Synthesizer 配置清单

最后，这是一个经过实战检验的生产级 Synthesizer 配置清单：

```python
class ProductionSynthesizerConfig:
    """生产级响应合成配置"""

    # === 模式选择 ===
    RESPONSE_MODE = ResponseMode.REFINE  # 默认首选

    # === Prompt 配置 ===
    SYSTEM_PROMPT_PARTS = [
        "角色设定：你是{role_name}，服务于{target_audience}",
        "约束条件：只使用参考资料，不编造，不猜测",
        "格式要求：Markdown格式，关键信息加粗，操作步骤编号",
        "安全规则：敏感信息脱敏，不确定时建议转人工",
        "引用规范：每段关键信息标注(来源: 文件名)",
    ]

    # === 质量门控 ===
    MAX_OUTPUT_CHARS = 2000       # 最大输出长度
    MIN_SOURCE_NODES = 2          # 最少参考文档数
    ENABLE_POST_HOC_VALIDATION = False  # 是否启用事后校验
    POST_HOC_SAMPLE_RATE = 0.05      # 抽样校验比例

    # === 性能配置 ===
    USE_ASYNC = True               # 启用异步
    STREAMING = False              # 是否流式（API 场景通常关闭）
    LLM_TIMEOUT = 30.0             # 单次调用超时（秒）
    MAX_RETRIES = 2                 # 失败重试次数

    # === 降级策略 ===
    FALLBACK_TO_TEXT = True         # 结构化解析失败时降级
    FALLBACK_MODE = "simple_summarize"  # REFINE 失败时的备选模式

    # === 监控 ===
    LOG_SYNTHESIS_DETAILS = True     # 记录合成过程的详细信息
    ALERT_ON_LONG_DURATION = 10.0     # 超过 10 秒告警
```

## 常见问题速查表

| 现象 | 可能原因 | 解决方案 |
|------|---------|---------|
| 答案太短/信息量不足 | Node 相关性低或数量不够 | 增加 similarity_top_k；改用 REFINE 模式 |
| 答案中有明显错误信息 | 幻觉问题 | 加强 Prompt 约束；启用事后校验 |
| 答案前后矛盾 | REFINE 路径依赖 | 先 rerank 排序；考虑 TREE_SUMMARIZE |
| 答案格式混乱 | Prompt 缺少格式要求 | 使用结构化输出模板 |
| 合成耗时过长 | Node 太多或模式不当 | 切换更轻量的模式；压缩上下文 |
| 结构化解析失败 | LLM 输出格式不规范 | 重试 + 降级到纯文本 |
| Token 消耗过高 | 模式选择不当或上下文未压缩 | 选择合适的模式；启用缓存 |

掌握这些优化技巧后，你的 RAG 系统的响应质量将从"能用"提升到"专业水准"——这也是区分一个 demo 级 RAG 和生产级 RAG 的关键差距之一。
