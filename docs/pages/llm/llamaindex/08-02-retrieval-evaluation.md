---
title: 检索质量评估：Recall / Precision / MRR / Hits@K
description: 检索评估指标详解、评估数据集构建方法、LlamaIndex 内置评估工具、BM25 与向量的联合评估
---
# 检索质量评估：Recall / Precision / MRR / Hits@K

在上一节我们建立了 RAG 评估的整体框架，现在让我们深入第一个也是最关键的评估维度——**检索质量（Retrieval Quality）**。检索是 RAG 管道的入口，检索质量直接决定了后续所有环节的上限。如果检索阶段就遗漏了相关文档，后面的合成再优秀也无济于事。

## 核心指标详解

### Recall@K（召回率）

**定义：** 在全部相关文档中，有多少比例出现在了检索返回的 Top-K 结果里。

```
假设知识库中有 10 个与查询"退货政策"相关的文档
检索返回了 Top-5 结果，其中 4 个是相关的

Recall@5 = 4/10 = 0.40 (40%)
```

**直观理解：** Recall 衡量的是**"有没有漏掉重要的东西"**。Recall 越高说明检索系统越不容易遗漏相关信息。

**计算公式：**
```python
def recall_at_k(relevant_ids, retrieved_ids, k):
    """计算 Recall@K"""
    relevant_set = set(relevant_ids)
    retrieved_set = set(retrieved_ids[:k])
    hits = relevant_set & retrieved_set
    return len(hits) / max(len(relevant_set), 1)
```

**目标值：**
| 场景 | Recall@5 目标 | Recall@10 目标 |
|------|-------------|--------------|
| FAQ 类精确查询 | > 0.90 | > 0.95 |
| 技术文档搜索 | > 0.80 | > 0.90 |
| 法律/合规查询 | > 0.95 | > 0.98 |
| 开放式概览类 | > 0.60 | > 0.80 |

### Precision@K（精确率）

**定义：** 在检索返回的 Top-K 结果中，有多少比例是真正相关的。

```
检索返回 Top-5 结果，其中 4 个是相关的，1 个不相关

Precision@5 = 4/5 = 0.80 (80%)
```

**直观理解：** Precision 衡量的是**"找到的东西中有多少是有用的"**。Precision 越高说明检索结果的噪音越少。

**注意 Recall 和 Precision 的权衡关系：** 增加 `similarity_top_k` 通常会提高 Recall（更多候选 → 更可能包含相关文档）但降低 Precision（更多候选 → 更多噪音混入）。这就是所谓的 **Precision-Recall Tradeoff**。

### MRR（Mean Reciprocal Rank，平均倒数排名）

**定义：** 相关文档在检索结果中排名位置的倒数的平均值。

```
查询: "退货政策"
相关文档: [D3, D7, D12, D15]

检索结果排序:
  #1 D8  (不相关)
  #2 D3  ← 相关！rank=2
  #3 D15 (不相关)
  #4 D7  ← 相关！rank=4
  #5 D12 ← 相关！rank=5

MRR = (1/2 + 1/4 + 1/5) / 3 = 0.25 + 0.25 + 0.20 / 3 = 0.233
```

**直观理解：** MRR 同时考虑了"能不能找到"（Recall）和"排在多前面"（排序质量）。MRR 越高说明相关文档不仅被找到了，而且排在了更靠前的位置——这对用户体验很重要，因为用户通常只看前几个结果。

**目标值：** > 0.70（即相关文档平均排在第 1-3 名）

### Hits@K（命中率）

**定义：** 至少有一个相关文档出现在 Top-K 结果中的查询占总查询的比例。

```python
def hits_at_k(queries, k):
    """计算 Hits@K"""
    hits = 0
    for q in queries:
        if any(r.is_relevant and r.rank <= k for r in q.results[:k]):
            hits += 1
    return hits / len(queries)
```

**目标值：** > 0.90 @ K=5

### NDCG（Normalized Discounted Cumulative Gain）

**定义：** 一种同时考虑相关性分数和排名位置的高级指标。排名越高的结果权重越大。

```
结果: [#1 score=0.9(相关), #2 score=0.3(不相关), #3 score=0.8(相关)]

NDCG@3 = (0.9/log2 + 0/log4 + 0.8/log4) / ideal_NDCG
       ≈ (0.9/0.69 + 0 + 0.8/1.39) / 1.0
       ≈ (1.30 + 0 + 0.58) / 1.0
       ≈ 0.62
```

**目标值：** > 0.65

NDCG 是信息检索领域最权威的综合指标之一，但它计算相对复杂。对于大多数 RAG 项目来说，**Recall + MRR 的组合已经能提供足够好的评估信号**，NDCG 可以作为补充参考。

## 构建评估数据集

评估指标的计算需要一个**评估数据集（Benchmark Dataset）**——一组已知答案的"问题-答案"对。

### 数据集设计原则

**原则一：覆盖面要广。** 评估数据集应该涵盖你的 RAG 系统可能遇到的各种查询类型：

```python
EVALUATION_DATASET = [
    # === 事实型查询（精确匹配）===
    {
        "query": "S1 的价格是多少？",
        "expected_answer": "299 元",
        "expected_keywords": ["299", "价格", "¥", "CNY"],
        "relevant_docs": ["price_list.pdf", "product_S1.md"],
        "difficulty": "easy",
        "category": "factual",
    },
    {
        "query": "API 的 rate_limit 参数默认值？",
        "expected_answer": "100",
        "expected_keywords": ["rate", "limit", "100"],
        "relevant_docs": ["api_reference.md"],
        "difficulty": "easy",
        "category": "factual",
    },

    # === 推理型查询（需要综合分析）===
    {
        "query": "S1 和 S2 各自适合什么用户群体？",
        "expected_answer": "S1 适合预算敏感的家庭用户...",
        "expected_keywords": ["家庭", "预算", "企业"],
        "relevant_docs": ["comparison.md", "S1_spec.md", "S2_spec.md"],
        "difficulty": "hard",
        "category": "reasoning",
    },

    # === 概括型查询（全局视角）===
    {
        "query": "这份季度报告的核心结论是什么？",
        "expected_answer": "Q3 收入增长 23%...",
        "expected_keywords": ["收入", "增长", "Q3"],
        "relevant_docs": ["Q3_report.pdf", "exec_summary.md"],
        "difficulty": "medium",
        "category": "summary",
    },

    # === 操作型查询（步骤指引）===
    {
        "query": "怎么申请退款？",
        "expected_answer": "登录账户→订单管理→点击申请退款...",
        "expected_keywords": ["登录", "订单", "申请", "退款"],
        "relevant_docs": ["refund_guide.md", "user_manual.pdf"],
        "difficulty": "medium",
        "category": "procedural",
    },

    # === 否定型查询（边界情况）===
    {
        "query": "这款产品支持水下操作吗？",
        "expected_answer": "不支持水下操作（IP67 防水等级）",
        "expected_keywords": ["不支持", "水下", "IP67"],
        "relevant_docs": ["specifications.pdf"],
        "difficulty": "easy",
        "category": "negative",
    },
]
```

**原则二：答案必须可靠。** 每个评估条目的"预期答案"都应该经过人工审核（最好由领域专家确认）。错误的预期答案会导致整个评估失去意义——你会基于错误的标准来判断系统的表现。

**原则三：要有区分度。** 好的评估数据集应该能够**区分好坏系统**。如果你换了一个不同的嵌入模型或调整了 chunk_size，评估指标应该能反映出变化。如果无论怎么改指标都不变，说明数据集缺乏敏感性。

**原则四：定期更新。** 随着知识库内容的增长和用户查询模式的演变，旧的评估数据集可能不再具有代表性。建议每季度审核并更新一次评估数据集。

### 数据集规模建议

| 项目阶段 | 建议数量 | 说明 |
|---------|---------|------|
| 原型验证 | 20-30 条 | 快速验证基本功能 |
| 开发期 | 50-100 条 | 覆盖主要场景 |
| 预发布 | 100-200 条 | 全面回归测试 |
| 生产期 | 50-100 条（抽样） | 持续监控 |

## LlamaIndex 中的检索评估实现

LlamaIndex 提供了内置的工具来简化检索质量的评估：

### 方法一：手动构建评估

```python
from llama_index.core.evaluation import (
    DatasetGenerator,
    EmbeddingQAFinetuneEvaluator,
    RetrieverEvaluator,
)

# Step 1: 准备评估数据集
documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)

# 生成查询-文档对的评估数据集
dataset_generator = DatasetGenerator()
eval_questions = dataset_generator.generate_questions_from_documents(num=25)

# Step 2: 创建评估器
retriever_evaluator = RetrieverEvaluator(
    metric="hit_rate",     # 或 "mrr", "ndcg"
    retriever=index.as_retriever(similarity_top_k=5),
)

# Step 3: 执行评估
eval_result = await retriever_evaluator.aevaluate(
    queries=eval_queries,
)
print(f"Hits@5: {eval_result['hit_rate']:.2%}")
print(f"MRR: {eval_result['mrr']:.3f}")
```

### 方法二：使用内置的 BatchEvalRunner

```python
from llama_index.core.evaluation import (
    BatchEvalRunner,
    EmbeddingQAFinetuneEvaluator,
    RetrieverEvaluator,
)

# 配置评估器列表
evaluators = [
    # 检索质量评估
    RetrieverEvaluator(
        metric="hit_rate",
        retriever=index.as_retriever(similarity_top_k=5),
    ),
    RetrieverEvaluator(
        metric="mrr",
        retriever=index.as_retriever(similarity_top_k=5),
    ),
    # 生成质量评估（可选）
    EmbeddingQAFinetuneEvaluator(
        embed_model="local:BAAI/bge-large-zh-v1.5",
    ),
]

# 运行批量评估
runner = BatchEvalRunner(
    evaluators=evaluators,
    workers=4,  # 并行 worker 数量
)

# 对每个查询执行评估
response = await runner.aevaluate_queries(
    queries=[q.query for q in eval_questions],
)

print(response["results"])
```

## 混合检索的评估

如果你的系统使用了混合检索（向量 + BM25），评估需要相应地调整：

```python
from llama_index.core.evaluation import (
    RetrieverEvaluator,
)
from llama_index.retrieversers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever

vector_retriever = index.as_retriever(similarity_top_k=20)
bm25_retriever = BM25Retriever.from_defaults(
    nodes=list(index.index_struct.nodes_dict.values()),
    similarity_top_k=20,
)
hybrid_retriever = QueryFusionRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    mode="reciprocal_rank",
    similarity_top_k=5,
)

hybrid_qe = RetrieverQueryEngine(
    retriever=hybrid_retriever,
    response_synthesizer=get_response_synthesizer("refine"),
)

evaluator = RetrieverEvaluator(
    metric="hit_rate",
    retriever=hybrid_retriever,  # 用混合检索器评估
)
```

这样评估出来的指标反映的是**混合检索管道的整体效果**，而不是单一检索方法的贡献。

## 评估结果分析与行动

拿到评估数据后，关键是**从数字中提取洞察并转化为行动**：

```python
def analyze_and_suggest(eval_results):
    """分析评估结果并给出优化建议"""

    print("=" * 60)
    print("检索质量评估报告")
    print("=" * )

    # 按 difficulty 分组统计
    easy_hits = [r for r in eval_results if r.difficulty == "easy"]
    medium_hits = [r for r in eval_results if r.difficulty == "medium"]
    hard_hits = [r for r in eval_results if r.difficulty == "hard"]

    def hit_rate(group):
        if not group:
            return 0
        return sum(1 for r in group if r.is_hit) / len(group)

    print(f"\n{'='*60}")
    print(f"{'总体 Hit Rate':=^25} {'值':^10}")
    print(f"{'='*60}")
    print(f"{'全部':^25} {hit_rate(eval_results):>10.2%}")
    print(f"{'简单':^25} {hit_rate(easy_hits):>10.2%}")
    print(f"{'中等':^25} {hit_rate(medium_hits):>10.2%}")
    print(f{"困难":^25} {hit_rate(hard_hits):>10.2%}")

    # 分析未命中的查询
    misses = [r for r in eval_results if not r.is_hit]
    if misses:
        print(f"\n❌ 未命中查询 ({len(misses)}个):")
        for m in misses[:5]:
            print(f"   - [{m.difficulty}] {m.query[:50]}...")

        # 给出针对性建议
        easy_misses = [m for m in misses if m.difficulty == "easy"]
        if len(easy_misses) > 2:
            print("\n⚠️ 建议: 简单查询命中率过低，检查:")
            print("   1. 文档是否已正确加载到索引？")
            print("   2. chunk_size 是否合适？（可能切断了关键信息）")
            print("   3. 嵌入模型是否能覆盖查询领域的术语？")

        hard_misses = [m for m in misses if m.difficulty == "hard"]
        if len(hard_misses) > 2:
            print("\n⚠️ 建议: 困难查询命中率低，考虑:")
            print("   1. 启用 HyDE 扩展短查询")
            print("   2. 增加 similarity_top_k 后加 reranker")
            print("   3. 尝试混合检索（+BM25）")


# 假设我们有评估结果
class MockResult:
    def __init__(self, query, is_hit, difficulty):
        self.query = query
        self.is_hit = is_hit
        self.difficulty = difficulty


mock_results = [
    MockResult("价格是多少？", True, "easy"),
    MockResult("价格是多少？", True, "easy"),
    MockResult("怎么退款？", True, "medium"),
    MockResult("比较 S1/S2 功能", False, "hard"),
    MockResult("比较 S1/S2 功能", False, "hard"),
    MockResult("API 超时设置", False, "easy"),
    MockResult("水下操作？", True, "easy"),
    MockResult("Q3 收入趋势", False, "medium"),
    MockResult("Q3 收入趋势", False, "medium"),
    MockResult("定制产品退款", False, "hard"),
]

analyze_and_suggest(mock_results)
```

这种"评估 → 分析 → 建议"的闭环是持续优化的引擎。每次迭代都会让你更了解系统的行为特征，从而做出更有针对性的改进。
