---
title: 解析质量评估与调优
description: 解析质量的量化指标、基于检索效果的端到端评估方法、A/B 测试框架、持续优化流程
---
# 解析质量评估与调优

前面几节我们学习了各种文档解析技术和策略，但你可能会问：**我怎么知道我的解析方案到底好不好？** chunk_size 设成 512 到底比 256 好还是差？层级解析真的比扁平切分有效吗？这些问题不能靠直觉回答——需要数据驱动的评估方法。

这一节我们来建立一个完整的**解析质量评估体系**，包括直接指标（边界准确性、完整性）、间接指标（检索质量、答案准确率），以及如何通过 A/B 测试来持续优化解析参数。

## 直接质量指标

直接指标关注的是解析过程本身的产出质量，不需要涉及下游的检索和生成环节。

### 指标一：边界准确率（Boundary Accuracy）

这是最基本的指标——检查 Parser 是否在合理的语义边界处进行切分。

```python
import random
from typing import List

def evaluate_boundary_accuracy(nodes, sample_size=50):
    """
    评估边界准确率
    对随机抽样的节点进行人工或规则判断
    """
    sampled = random.sample(nodes, min(sample_size, len(nodes)))
    good_boundaries = 0
    issues = []

    for i, node in enumerate(sampled):
        text = node.text
        verdict, reason = check_boundary_quality(text)
        if verdict:
            good_boundaries += 1
        else:
            issues.append({
                "node_id": node.node_id[:8],
                "reason": reason,
                "first_50_chars": text[:50],
                "last_50_chars": text[-50:],
            })

    accuracy = good_boundaries / len(sampled)
    print(f"边界准确率: {accuracy:.1%} ({good_boundaries}/{len(sampled)})")

    if issues:
        print(f"\n问题样本 ({len(issues)} 个):")
        for issue in issues[:5]:
            print(f"  [{issue['node_id']}] {issue['reason']}")
            print(f"    前: ...{issue['first_50_chars']}...")
            print(f"    后: ...{issue['last_50_chars']}...")

    return accuracy, issues


def check_boundary_quality(text: str):
    """
    检查一个节点的起止位置是否在合理的语义边界上
    返回 (是否良好, 原因描述)
    """
    first_line = text.lstrip()[:80]
    last_line = text.rstrip()[-80:]

    bad_patterns = [
        (text.startswith(("the ", "The ", "这是一个", "其中")),
         "开头像是句子的中间（缺少上下文）"),
        (text.endswith((",", "，", " and ", " 以及", " of ")),
         "结尾在句子或短语中间（句子未完成）"),
        (not text[0].isupper() and not text[0] in ("#", "-", "*", "(", "[", "{", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0")),
         "首字母不是大写也不是特殊符号（可能是断词）"),
    ]

    for pattern, reason in bad_patterns:
        if pattern:
            return False, reason

    return True, "OK"


# 使用
accuracy, issues = evaluate_boundary_accuracy(nodes)
```

**目标值：边界准确率 > 90%。** 如果低于这个阈值，说明你的 Parser 在很多地方都不恰当地截断了内容，这会直接影响下游的检索和生成质量。

### 指标二：上下文完整性（Context Completeness）

即使边界位置合理，chunk 本身也可能缺乏理解自身所需的上下文。

```python
def evaluate_context_completeness(nodes, llm):
    """
    使用 LLM 评估每个节点的上下文完整性
    """
    total = len(nodes)
    complete = 0
    incomplete_examples = []

    prompt_template = """请判断以下文本片段是否包含了足够理解自身内容的上下文。
只回答"完整"或"不完整"，不需要解释。

文本片段:
{text}

判断:"""

    for node in nodes[:100]:  # 抽样评估，避免成本过高
        response = llm.complete(
            prompt_template.format(text=node.text[:500])
        )
        is_complete = "完整" in response.text
        if is_complete:
            complete += 1
        else:
            incomplete_examples.append(node.text[:150])

    score = complete / min(total, 100)
    print(f"上下文完整性: {score:.1%}")

    if incomplete_examples:
        print(f"\n不完整样本示例:")
        for ex in incomplete_examples[:3]:
            print(f"  \"{ex}...\"")

    return score
```

### 指标三：覆盖率（Coverage Rate）

覆盖率衡量的是原始文档中有多少内容被保留在了解析结果中：

```python
def evaluate_coverage(original_text: str, nodes) -> float:
    """计算解析后内容对原文的覆盖率"""
    original_chars = len(original_text.strip())
    combined_text = "".join(n.text for n in nodes)
    combined_chars = len(combined_text.strip())

    coverage = combined_chars / original_chars if original_chars > 0 else 0

    # 检查是否有重大遗漏（某些段落完全消失）
    original_paragraphs = [p.strip() for p in original_text.split("\n\n") if p.strip()]
    covered_paragraphs = 0
    for para in original_paragraphs:
        if para[:50] in combined_text:
            covered_paragraphs += 1

    paragraph_coverage = covered_paragraphs / len(original_paragraphs) \
        if original_paragraphs else 0

    print(f"字符覆盖率: {coverage:.1%}")
    print(f"段落覆盖率: {paragraph_coverage:.1%}")

    return coverage, paragraph_coverage
```

**目标值：字符覆盖率应在 95%-105% 之间。** 低于 95% 说明有内容丢失（Parser 可能跳过了某些部分）；高于 105% 说明有大量重复内容（overlap 设置过大或去重不彻底）。段落覆盖率应接近 100%——任何段落的完全丢失都是严重问题。

## 间接质量指标：端到端检索评估

直接指标告诉我们 Parser 本身工作得怎么样，但我们真正关心的是**解析质量对最终的 RAG 效果有多大影响**。这就需要端到端的评估——用真实的查询来测试不同解析方案的检索质量。

### 构建评估数据集

首先需要一批"标准问答对"——已知正确答案的问题集合：

```python
EVALUATION_DATASET = [
    {
        "query": "智能音箱 S1 的处理器型号是什么？",
        "expected_answer_keywords": ["Cortex-A53", "四核", "ARM"],
        "expected_source_section": "1.1 核心规格",
        "difficulty": "easy",
    },
    {
        "query": "S1 支持哪些智能家居协议？",
        "expected_answer_keywords": ["Wi-Fi", "Bluetooth", "Zigbee", "Matter"],
        "expected_source_section": "1.2 支持的智能家居协议",
        "difficulty": "medium",
    },
    {
        "query": "如果设备离线了该怎么排查？",
        "expected_answer_keywords": ["路由器", "网络", "重启", "同一网络"],
        "expected_source_section": "2.2 常见问题",
        "difficulty": "medium",
    },
    {
        "query": "总结一下这款产品的保修政策",
        "expected_answer_keywords": ["24个月", "硬件故障", "制造缺陷"],
        "expected_source_section": "第三章 保修政策",
        "difficulty": "hard",  # 需要综合多段信息
    },
    # ... 更多测试用例
]
```

### 评估函数

```python
def evaluate_retrieval_quality(query_engine, dataset, top_k=5):
    """评估检索质量"""
    results = []

    for item in dataset:
        response = query_engine.query(item["query"])

        retrieved_texts = [node.text for node in response.source_nodes]
        retrieved_sections = [
            node.metadata.get("header_path", ["未知"])[-1]
            for node in response.source_nodes
        ]

        keyword_hits = sum(
            1 for kw in item["expected_answer_keywords"]
            if any(kw.lower() in t.lower() for t in retrieved_texts)
        )
        keyword_recall = keyword_hits / len(item["expected_answer_keywords"])

        section_match = any(
            item["expected_source_section"] in s
            for s in retrieved_sections
        )

        results.append({
            "query": item["query"],
            "keyword_recall": keyword_recall,
            "section_match": section_match,
            "retrieved_count": len(response.source_nodes),
            "difficulty": item["difficulty"],
        })

    # 汇总统计
    easy_results = [r for r in results if r["difficulty"] == "easy"]
    med_results = [r for r in results if r["difficulty"] == "medium"]
    hard_results = [r for r in results if r["difficulty"] == "hard"]

    def avg_recall(group):
        return sum(r["keyword_recall"] for r in group) / len(group) if group else 0

    print("=" * 60)
    print("检索质量评估报告")
    print("=" * 60)
    print(f"总查询数: {len(results)}")
    print(f"平均关键词召回率: {sum(r['keyword_recall'] for r in results)/len(results):.1%}")
    print(f"  简单查询: {avg_recall(easy_results):.1%}")
    print(f"  中等查询: {avg_recall(med_results):.1%}")
    print(f"  困难查询: {avg_recall(hard_results):.1%}")
    print(f"章节命中率: {sum(r['section_match'] for r in results)/len(results):.1%}")

    for r in results:
        status = "✅" if r["keyword_recall"] >= 0.8 else \
                 "⚠️" if r["keyword_recall"] >= 0.5 else "❌"
        print(f"  {status} [{r['difficulty']:5s}] "
              f"召回:{r['keyword_recall']:.0%} 章节:{'✓' if r['section_match'] else '✗'} "
              f"\"{r['query'][:40]}...\"")

    return results
```

### A/B 测试框架

有了评估函数，我们就可以对不同解析方案做 A/B 对比了：

```python
def ab_test_parsing_strategies(documents, eval_dataset):
    """A/B 测试不同的解析策略"""

    strategies = {
        "small_chunks": SentenceSplitter(chunk_size=128, chunk_overlap=25),
        "medium_chunks": SentenceSplitter(chunk_size=512, chunk_overlap=100),
        "large_chunks": SentenceSplitter(chunk_size=1024, chunk_overlap=200),
        "hierarchical": MarkdownNodeParser(),
        "code_aware": CodeSplitter(language="python", chunk_lines=60),
    }

    results = {}

    for name, parser in strategies.items():
        print(f"\n{'='*60}")
        print(f"测试策略: {name}")
        print(f"{'='*60}")

        nodes = parser.get_nodes_from_documents(documents)
        index = VectorStoreIndex(nodes)
        qe = index.as_query_engine(similarity_top_k=5)

        strategy_results = evaluate_retrieval_quality(qe, eval_dataset)
        results[name] = {
            "results": strategy_results,
            "node_count": len(nodes),
            "avg_node_len": sum(len(n.text) for n in nodes) / len(nodes),
        }

    # 对比表格
    print(f"\n{'='*60}")
    print("策略对比总览")
    print(f"{'='*60}")
    print(f"{'策略':20s} {'节点数':>8s} {'平均长度':>8s} {'平均召回':>8s}")
    print("-" * 56)

    for name, data in sorted(results.items(),
                              key=lambda x: -sum(
                                  r['keyword_recall']
                                  for r in x[1]['results']
                              ) / len(x[1]['results'])):
        avg_r = sum(r['keyword_recall'] for r in data['results']) \
                / len(data['results'])
        print(f"{name:20s} {data['node_count']:8d} "
              f"{data['avg_node_len']:8.0f} {avg_r:8.1%}")

    return results
```

运行这个 A/B 测试后，你会得到一张清晰的对比表：

```
============================================================
策略对比总览
============================================================
策略                   节点数   平均长度    平均召回
--------------------------------------------------------
hierarchical             347      489      87.3%
medium_chunks            523      498      82.1%
large_chunks             268      987      78.5%
code_aware               412      356      71.2%
small_chunks            1043      124      63.8%
```

从这个假设的结果可以看出：**层级解析（hierarchical）在这个数据集上表现最好**，因为它保留了文档的结构信息；中等大小的 chunks（medium_chunks）次之；而过小的 chunks（small_chunks）虽然数量多，但由于上下文不足导致召回率最低。

## 基于评估结果的调优流程

评估不是为了得到一个数字就结束——它的目的是指导你**有针对性地改进解析策略**。以下是常见的评估→改进循环：

### 场景一：边界准确率低

**症状：** 边界准确率 < 85%，大量节点在不恰当的位置被截断。

**诊断方向：**
1. 检查 `chunk_size` 是否太小——太小会导致频繁在句子中间切断
2. 检查文档语言——中文文档的句子边界可能与英文不同（中文不依赖空格分词）
3. 检查是否有特殊格式干扰了句子检测（如代码块、列表项）

**改进措施：**

```python
# 措施一：增大 chunk_size
splitter = SentenceSplitter(chunk_size=768, chunk_overlap=150)

# 措施二：使用支持中文的分句工具
import jieba

def chinese_sentence_split(text):
    sentences = []
    for sent in jieba.cut(text):
        sent = sent.strip()
        if sent:
            sentences.append(sent)
    return sentences

# 措施三：排除特殊格式的干扰
splitter = SentenceSplitter(
    chunk_size=512,
    chunk_overlap=100,
    secondary_chunking_regex="[^\\n]+",  # 优先在换行处断开
)
```

### 场景二：关键词召回率低

**症状：** 端到端评估中，预期关键词在检索结果中出现频率低。

**诊断方向：**
1. 关键词是否被拆散到了不同节点？（chunk_size 或 overlap 问题）
2. 关键词所在的节点是否因为其他不相关的内容而降低了相似度分数？（噪音问题）
3. 关键词的表达方式是否与用户查询差异太大？（同义词/改写问题）

**改进措施：**

```python
# 措施一：增加 overlap
splitter = SentenceSplitter(chunk_size=512, chunk_overlap=200)

# 措施二：使用 Hypothetical Document Embeddings (HyDE)
# 在检索时生成假想答案再搜索（第五章会详细讲）
# 措施三：添加查询扩展
# 将用户的查询扩展为多个变体后再分别检索
```

### 场景三：困难查询表现差

**症状：** 简单和中等查询都还行，但需要综合信息的困难查询（如"总结""比较""列举所有"）效果很差。

**诊断方向：** 困难查询通常需要看到更大范围的信息。当前的 chunk 粒度太细，无法提供足够的综合视角。

**改进措施：**

```python
# 措施一：使用 Tree Summarize 响应模式
# （第七章会详细讲）
synthesizer = get_response_synthesizer(
    response_mode=ResponseMode.TREE_SUMMARIZE
)

# 措施二：增大 similarity_top_k
qe = index.as_query_engine(similarity_top_k=10)

# 措施三：使用 SummaryIndex 作为补充索引
summary_index = SummaryIndex.from_documents(documents)
summary_qe = summary_index.as_query_engine()
# 对于概括性查询，路由到 summary_qe
```

## 持续优化的闭环

解析调优不是一个一次性任务，而应该是持续的过程：

```
┌──────────────┐
│  当前解析配置  │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  解析文档      │ ← 新文档不断加入
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  运行评估      │ ← 定期执行（每周/每月）
│  (直接+间接)   │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  分析结果      │
│  发现退化点    │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  调整参数/策略  │
└──────┴───────┘
       │
       └──────────→ 回到顶部
```

建议在生产环境中建立这样的自动化流水线：
1. **每日：** 监控基础指标（节点数量、平均长度、解析耗时）
2. **每周：** 运行抽样评估（边界准确率、覆盖率）
3. **每月：** 运行完整的端到端评估（使用固定的评估数据集）
4. **每季度：** 审查并更新评估数据集（加入新的查询模式，淘汰过时的用例）

## 常见误区

**误区一："解析只需要设置一次"。** 文档特征会变化（新作者加入、新模板启用），查询模式会演进（新产品上线带来新的问题类型），模型能力会升级（新的嵌入模型可能有不同的最佳 chunk size）。**定期重新评估和调整是必要的。**

**误区二："评估数据集越大越好"。** 一个精心设计的 50 条评估数据集远胜过一个随意收集的 500 条数据集。关键是**覆盖面要广**（简单/中等/困难、事实型/推理型/概括型）且**答案要可靠**（由领域专家确认的正确答案）。

**误区三:"只看平均指标就够了"。** 平均指标可能掩盖严重的长尾问题——也许 90% 的查询都很好，但剩下 10% 的关键业务查询全部失败。**一定要看分位数和最差案例**，它们往往揭示了最有价值的改进方向。
