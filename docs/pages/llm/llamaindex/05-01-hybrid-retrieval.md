---
title: 混合检索：向量搜索 + 关键词搜索的协同
description: Hybrid Search 原理、BM25 与向量的互补性、LlamaIndex 实现方式、权重调优
---
# 混合检索：向量搜索 + 关键词搜索的协同

在第四章中我们深入了 VectorStoreIndex 的内部机制，也了解了 KeywordTableIndex 作为精确关键词匹配的价值。但当时我们提到过一个关键观点：**这两种索引各有各的强项和弱项，最好的策略是把它们组合起来使用**。这一节我们就来详细讨论这种"混合检索"（Hybrid Search）——如何让语义搜索和关键词搜索协同工作，取长补短，获得比任何单一方法都更好的检索效果。

## 为什么单一搜索方法不够好？

让我们通过一组具体的例子来感受纯向量搜索和纯关键词搜索各自的局限性。

### 纯向量搜索的失败案例

**案例一：专业术语失配。**

```
用户查询: "根据《消费者权益保护法》第二十四条，退货的条件是什么？"

文档中的内容: "消费者在收到商品之日起七日内可无理由要求退货..."
```

问题在于：用户用了精确的法律条文引用（《消法》第二十四条），而文档中并没有重复这个引用——它只是描述了这个条款的内容。嵌入模型可能无法将"第二十四条"和"七日内无理由退货"这两个表达编码到足够接近的向量空间位置上。结果就是：**明明文档中有答案，向量搜索却找不到它。**

**案例二：罕见实体名称。**

```
用户查询: "xgboost 的 n_estimators 参数应该设多少？"

文档内容: "XGBoost 的树的数量（n_estimators）建议设置为 100-1000 之间..."
```

"xgboost" 和 "XGBoost" 在大小写上有差异，"n_estimators" 这个精确参数名可能在 embedding 过程中被稀释或变形。对于包含精确专有名词的查询，纯向量搜索经常表现不佳。

### 纯关键词搜索的失败案例

**案例一：同义词无法匹配。**

```
用户查询: "怎么退钱？"

文档内容: "如需办理退款，请登录账户进入订单管理页面..."
```

用户说的"退钱"和文档中写的"退款"是同一个意思，但字面上完全不同。基于关键词匹配的方法（如 BM25）只能匹配完全相同或高度相似的词汇，对同义词无能为力。

**案例二：语义理解需求。**

```
用户查询: "我的设备连不上网了怎么办？"

文档内容: "若产品无法连接 Wi-Fi 网络，请按以下步骤排查：
1. 确认路由器正常工作
2. 检查设备的网络设置
3. 尝试重启设备和路由器"
```

用户问的是"连不上网"，文档写的是"无法连接 Wi-Fi"。虽然 BM25 可能因为"连接"和"网络"这两个词的部分匹配而返回结果，但它无法像向量搜索那样真正理解"连不上网"和"无法连接 Wi-Fi"是同一回事。

### 总结对比

| 场景 | 向量搜索 | 关键词搜索 | 谁更好？ |
|------|---------|-----------|---------|
| "怎么退款？" vs 文档说"退货流程" | ✅ 理解同义 | ❌ 词不匹配 | **向量** |
| "《消法》第二十四条" vs 文档描述内容 | ⚠️ 可能遗漏 | ✅ 如果有该词出现 | **关键词** |
| "API 返回 503 错误" | ⚠️ 数字可能被稀释 | ✅ 精确匹配 | **关键词** |
| "系统整体运行慢的原因" | ✅ 综合理解语义 | ❌ 太泛泛 | **向量** |
| 产品型号 "S1-Pro-Max" | ⚠️ 特殊编号处理不稳定 | ✅ 精确匹配 | **关键词** |

结论很清楚：**没有一种方法能在所有场景下都表现最好。** 它们是互补的关系，而非替代关系。

## 混合检索的原理

混合检索的核心思想很简单：**同时执行两种检索，然后合并它们的结果**。

```
用户查询: "S1 产品的保修期和退货政策"

┌─────────────────────┐     ┌─────────────────────┐
│   向量搜索 (dense)    │     │   关键词搜索 (sparse) │
│                     │     │                     │
│  Query → Embedding  │     │  Query → 分词       │
│       ↓             │     │       ↓             │
│  余弦相似度 Top-10   │     │  BM25 得分 Top-10    │
│       ↓             │     │       ↓             │
│  结果 A (score:0.89) │     │  结果 B (score:8.5) │
│  结果 C (score:0.85) │     │  结果 D (score:7.2) │
│  结果 E (score:0.82) │     │  结果 F (score:6.8) │
│  ...                │     │  ...                │
└──────────┬──────────┘     └──────────┬──────────┘
           │                           │
           ▼                           ▼
    ┌──────────────────────────────────────┐
    │         Reciprocal Rank Fusion        │
    │         (倒数排名融合算法)              │
    │                                       │
    │  对每个结果的两个分数进行融合          │
    │  生成统一的排序                        │
    └──────────────────┬───────────────────┘
                       │
                       ▼
              最终 Top-K 结果
              (融合了两种信号的优势)
```

最关键的步骤是**分数融合（Score Fusion）**——两种搜索方法的评分体系和量纲完全不同（余弦相似度范围 [-1, 1]，BM25 分数可能是 [0, 30]），不能直接相加或比较。需要一种融合算法来统一它们。

### Reciprocal Rank Fusion (RRF)

RRF 是目前最流行且效果最好的融合算法之一。它的公式非常优雅：

```
RRF_score(d) = Σ 1 / (k + rank_i(d))

其中:
  d = 一个候选文档
  i = 第 i 个检索方法（向量搜索或关键词搜索）
  rank_i(d) = 文档 d 在第 i 种方法中的排名（从 1 开始）
  k = 平滑常数（通常取 60）
```

举个例子：

| 文档 | 向量搜索排名 | 关键词搜索排名 | RRF 分数 |
|------|------------|-------------|----------|
| Doc A | 1 (最好) | 3 | 1/(60+1) + 1/(60+3) = 0.0317 |
| Doc B | 2 | 1 | 1/(60+2) + 1/(60+1) = 0.0326 |
| Doc C | 3 | 15 | 1/(60+3) + 1/(60+15) = 0.0249 |
| Doc D | 10 | 2 | 1/(60+10) + 1/(60+2) = 0.0264 |

最终排序：Doc B > Doc A > Doc D > Doc C

注意 Doc B 虽然在向量搜索中只排第 2，但在关键词搜索中排第 1，综合得分反而超过了在向量搜索中排第 1 的 Doc A。这就是 RRF 的价值——**它能让在任一方法中表现优异的结果脱颖而出**。

k=60 是经验值，适用于大多数场景。较小的 k 值会让排名靠前的结果获得更高的权重；较大的 k 值则更平滑地对待所有排名位置。

## LlamaIndex 中的混合检索实现

LlamaIndex 从 0.10 版本开始原生支持混合检索。实现方式如下：

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.retrievers.bm25 import BM25Retriever

documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)
nodes = index.node_parser.get_nodes_from_documents(documents)

# 创建两种检索器
vector_retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=10,
)
bm25_retriever = BM25Retriever.from_defaults(
    nodes=nodes,
    similarity_top_k=10,
    verbose=True,
)

# 组合为混合检索器
from llama_index.core.retrievers import QueryFusionRetriever

hybrid_retriever = QueryFusionRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    mode="reciprocal_rank",  # RRF 融合模式
    similarity_top_k=5,      # 最终返回 top 5
    num_queries=1,            # 不做查询扩展（下一节讲）
    verbose=True,
)

# 使用混合检索器创建查询引擎
query_engine = RetrieverQueryEngine.from_args(
    hybrid_retriever,
    response_mode="compact",
)

response = query_engine.query("S1 产品的保修期是多少？")
print(response.response)

print("\n--- 检索详情 ---")
for node in response.source_nodes:
    print(f"[{node.score:.3f}] {node.text[:80]}...")
```

### 关键组件解析

**BM25Retriever：** 这是 LlamaIndex 内置的关键词检索器，实现了经典的 BM25 算法（Best Matching 25）。BM25 是信息检索领域几十年来经过验证的算法，它在以下方面表现出色：
- 精确术语匹配（包括短语匹配）
- 稀有词的高权重（稀有词的匹配更有区分度）
- 文档长度归一化（避免长文档因包含更多词而占优势）

`BM25Retriever.from_defaults()` 会自动对所有 Node 构建倒排索引。你也可以传入自定义参数：

```python
bm25_retriever = BM25Retriever.from_defaults(
    nodes=nodes,
    similarity_top_k=10,
    k1=1.5,    # 词频饱和参数（默认 1.5）
    b=0.75,    # 文档长度归一化参数（默认 0.75),
    verbose=True,
)
```

**QueryFusionRetriever：** 这是混合检索的核心协调者。它的 `mode` 参数支持多种融合策略：

| mode | 说明 | 适用场景 |
|------|------|----------|
| `reciprocal_rank` | RRF（推荐） | 大多数场景 |
| `relative_score` | 相对分数融合 | 需要保留原始分数比例时 |
| `simple` | 简单平均 | 快速原型 |

### 完整的生产级示例

下面是一个更完整的生产级混合检索配置：

```python
class HybridSearchRAG:
    def __init__(self, documents):
        self.index = VectorStoreIndex.from_documents(documents)
        self.nodes = self.index._index_struct.nodes_dict
        self._setup_retrievers()
        self._setup_query_engine()

    def _setup_retrievers(self):
        self.vector_retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=20,  # 多取一些，给融合留余地
        )
        self.bm25_retriever = BM25Retriever.from_defaults(
            nodes=list(self.nodes.values()),
            similarity_top_k=20,
        )
        self.hybrid_retriever = QueryFusionRetriever(
            retrievers=[self.vector_retriever, self.bm25_retriever],
            mode="reciprocal_rank",
            similarity_top_k=5,  # 最终只返回 top 5
            verbose=True,
        )

    def _setup_query_engine(self):
        from llama_index.core.response_synthesizers import get_response_synthesizer
        synthesizer = get_response_synthesizer(
            response_mode=ResponseMode.REFINE,
        )
        from llama_index.core.query_engine import RetrieverQueryEngine
        self.query_engine = RetrieverQueryEngine(
            retriever=self.hybrid_retriever,
            response_synthesizer=synthesizer,
        )

    def query(self, question: str) -> dict:
        response = self.query_engine.query(question)
        return {
            "answer": response.response,
            "sources": [
                {
                    "text": node.text[:150],
                    "score": node.score,
                    "source": node.metadata.get("file_name", "未知"),
                }
                for node in response.source_nodes
            ],
        }
```

## 权重调优

虽然 RRF 使用固定的 k=60 通常效果不错，但在某些场景下你可能希望调整两种检索方法的相对权重。LlamaIndex 允许通过自定义融合函数来实现：

```python
def weighted_fusion(results_list, weights=None):
    """自定义加权融合"""
    if weights is None:
        weights = [0.6, 0.4]  # 向量搜索 60%，关键词搜索 40%

    fused_scores = {}
    for result_set, weight in zip(results_list, weights):
        for node_with_score in result_set:
            node_id = node_with_score.node.node_id
            raw_score = node_with_score.score if node_with_score.score else 0
            if node_id not in fused_scores:
                fused_scores[node_id] = {
                    "node": node_with_score.node,
                    "score": 0,
                }
            fused_scores[node_id]["score"] += raw_score * weight

    sorted_results = sorted(
        fused_scores.values(), key=lambda x: x["score"], reverse=True
    )

    return [NodeWithScore(node=item["node"], score=item["score"])
            for item in sorted_results]


# 使用自定义融合
hybrid_retriever = QueryFusionRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    mode="relative_score",
    similarity_top_k=5,
    vector_weight=0.7,   # 向量搜索权重更高（适合语义丰富的查询）
    bm25_weight=0.3,     # 关键词搜索权重较低
)
```

权重的选择取决于你的数据特征：
- **如果用户的查询倾向于使用自然语言（口语化、同义词多）** → 提高向量搜索权重（0.7+）
- **如果用户的查询包含大量精确术语（型号、编号、法规条文名）** → 提高关键词搜索权重（0.5+）
- **不确定的话** → 使用默认的 RRF（不设权重）

## 性能考量

混合检索的性能开销主要来自两个方面：

**额外内存：** BM25Retriever 需要在内存中维护一个倒排索引。对于百万级节点，这通常需要几百 MB 的额外内存。对于更大的数据集，可以考虑使用外部搜索引擎（如 Elasticsearch）来提供 BM25 能力。

**额外延迟：** 需要执行两次检索 + 一次融合操作。在实际测试中，相比纯向量搜索，混合检索通常增加 20-50ms 的延迟（取决于数据规模和硬件配置）。对于大多数应用来说这是可以接受的。

## 常见误区

**误区一:"混合检索总是比单一检索更好"。** 不是绝对的。如果你的查询类型非常一致（比如全部都是自然语言问答），且数据特征也很均匀，额外的关键词搜索可能不会带来明显提升，反而增加了系统复杂度和延迟。**先评估再决定是否引入混合检索。**

**误区二:"BM25 只能用于英文"。** LlamaIndex 的 BM25Retriever 支持中文分词——它内部使用了 jieba 或其他分词器来处理中文文本。不过中文分词的质量会影响 BM25 的效果，对于专业领域的中文文档，可能需要自定义词典。

**误区三:"RRF 的 k 值需要仔细调优"。** 实践中 k=60 在绝大多数场景下都是很好的默认值。除非你有明确的证据表明调整 k 带来了显著的提升，否则不要花时间在这个参数上。**把调优精力放在更有影响力的地方（如 chunk_size、嵌入模型选择、top_k 值等）。**
