---
title: 自定义 Query Engine 与高级配置
description: SubQuestionQueryEngine / RouterQueryEngine / MultiStepQueryEngine 的深度使用、Tool-based Query Engine
---
# 自定义 Query Engine 与高级配置

前面两节我们学习了两种最基本的 Query Engine：RetrieverQueryEngine（单次问答）和 ChatEngine（多轮对话）。但 LlamaIndex 的 Query Engine 体系远不止这两种——还有专门处理复杂问题的 SubQuestionQueryEngine、支持多索引路由的 RouterQueryEngine、以及可以自由组装任何组件的自定义 Query Engine。

这一节我们来探索这些高级 Query Engine 类型，以及如何根据具体需求定制自己的查询引擎。

## SubQuestionQueryEngine：分解复杂问题

有些问题太复杂，无法通过一次检索就得到好的答案。比如："比较一下我们三个主要竞品的功能差异，并结合我们的产品定位给出建议。"这个问题实际上包含了至少 4 个子问题：
1. 我们的产品定位是什么？
2. 竞品 A 的功能特点？
3. 竞品 B 的功能特点？
4. 竞品 C 的功能特点？

`SubQuestionQueryEngine` 能自动完成这种分解：

```python
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool

base_index = VectorStoreIndex.from_documents(all_documents)
base_qe = base_index.as_query_engine(similarity_top_k=5)

sub_qe = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=[
        QueryEngineTool.from_defaults(
            base_qe,
            name="knowledge_base",
            description="公司内部知识库，包含产品、技术、市场等信息",
        ),
    ],
    verbose=True,  # 打印子问题生成和执行过程
)

response = sub_qe.query(
    "比较 S1 和 S2 两款产品的功能差异，"
    "并说明哪款更适合中小企业客户"
)
print(response.response)
```

### 执行流程详解

当你提交上述复杂问题时，SubQuestionQueryEngine 的内部工作流程如下：

```
原始问题: 比较 S1 和 S2 的功能差异，说明哪款更适合中小企业

═══ Step 1: 问题分解 (LLM) ═══
生成的子问题列表:
  Q1: "S1 产品的主要功能和特性是什么？"
  Q2: "S2 产品的主要功能和特性是什么？"
  Q3: "中小企业客户选购智能音箱的核心需求是什么？"

═══ Step 2: 并行执行子查询 ═══
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│ Q1 → 检索   │ │ Q2 → 检索   │ │ Q3 → 检索   │
│ → 回答      │ │ → 回答      │ │ → 回答      │
│ "S1 支持..."│ │ "S2 定位..."│ │ "中小企业..."│
└─────────────┘ └─────────────┘ └─────────────┘

═══ Step 3: 综合答案 (LLM) ═══
基于三个子问题的答案，生成最终的综合回答:

"S1 和 S2 的功能差异主要体现在：
1. S1 定位入门级市场，主打性价比...
2. S2 面向企业级用户，提供更多高级功能...

对于中小企业客户，推荐 S1 因为..."
```

### 何时使用 SubQuestionQueryEngine

| 场景 | 推荐 | 原因 |
|------|------|------|
| 单个事实查询（"X 是多少？"） | ❌ 不需要 | 一次检索就够了 |
| 比较类查询（"A 和 B 的区别？"） | ✅ 推荐 | 需要分别了解 A 和 B |
| 列举类查询（"列出所有..."） | ⚠️ 视情况 | 如果信息分散在多处则有用 |
| 分析类查询（"为什么 X 发生了？"）| ✅ 推荐 | 通常需要多角度信息 |
| 聚合类查询（"总体趋势如何？"）| ✅ 推荐 | 需要从多个数据点综合 |

### 性能与成本考量

SubQuestionQueryEngine 的代价是显著的：
- **额外的 LLM 调用：** 至少 1 次（问题分解）+ N 次（子查询合成）+ 1 次（最终综合）
- **更多的检索操作：** 每个子问题都会触发一次完整的检索+合成流程
- **更高的延迟：** 总耗时约为单次查询的 3-5 倍

建议只在对答案质量要求高且愿意承担额外成本的场景中使用。

## RouterQueryEngine：多索引路由

第四章我们已经介绍过 RouterQueryEngine 的基本用法。这里我们从 Query Engine 的视角再深入一层：

```python
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import PydanticSingleSelector
from llama_index.core.tools import QueryEngineTool

tech_index = VectorStoreIndex.from_documents(tech_docs)
report_index = SummaryIndex.from_documents(report_docs)
contract_index = KeywordTableIndex.from_documents(contract_docs)

router_qe = RouterQueryEngine(
    selector=PydanticSingleSelector(),
    query_engine_tools=[
        QueryEngineTool.from_defaults(
            tech_index.as_query_engine(),
            name="technical_search",
            description="搜索技术文档，适合 API 参考、配置指南等技术问题",
        ),
        QueryEngineTool.from_defaults(
            report_index.as_query_engine(),
            name="report_summary",
            "生成业务报告摘要，适合财务、运营等宏观分析问题",
        ),
        QueryEngineTool.from_defaults(
            contract_index.as_query_engine(),
            name="contract_search",
            "搜索合同条款，适合法律、合规相关问题",
        ),
    ],
    verbose=True,
)
```

### Selector 的决策质量

RouterQueryEngine 的效果高度依赖 Selector 的决策质量。以下是提升决策质量的实用技巧：

**技巧一：详细的工具描述**

```python
description = (
    "用于在公司内部技术文档库中进行语义搜索的工具。"
    "覆盖范围包括：API 参考手册、开发者指南、架构设计文档、"
    "故障排查手册等。输入应为具体的技术问题，"
    "输出为来自相关文档的技术细节。"
    "不适合：业务数据查询、法律条款查找、非技术性的概览性问题。"
)
```

描述越详细，LLM 做路由决策时就越不容易出错。

**技巧二：提供示例**

某些版本的 Selector 支持通过 `examples` 参数提供少量示例：

```python
tool = QueryEngineTool.from_defaults(
    qe,
    name="tech_search",
    description="...",
    examples=[
        ("API 的 timeout 参数怎么设？", "tech_search"),
        ("数据库连接池大小应该是多少？", "tech_search"),
        ("Q3 收入是多少？", "report_summary"),  # 反面示例
    ],
)
```

**技巧三：监控路由决策**

```python
def query_with_routing_info(router_qe, question):
    response = router_qe.query(question)
    print(f"[路由] 问题被发往: {response.metadata.get('_selected_tool', 'unknown')}")
    return response
```

定期统计路由分布可以帮助你发现 Selector 是否倾向于过度选择某个工具（可能是描述不够均衡导致的偏差）。

## TransformQueryEngine：查询转换管道

第五章学的 HyDE、Multi-Query 等查询转换技术也可以封装为一个独立的 Query Engine：

```python
from llama_index.core.query_engine import TransformQueryEngine
from llama_index.core.indices.query.query_transform import HyDEQueryTransform

base_qe = index.as_query_engine(similarity_top_k=5)

transformed_qe = TransformQueryEngine(
    base_query_engine=base_qe,
    query_transform=HyDEQueryTransform(  # 查询转换层
        llm=Settings.llm,
        num_hypotheticals=1,
        include_original=True,
    ),
    transform_metadata=True,  # 在 metadata 中标记使用了哪种转换
)

response = transformed_qe.query("S1 的蓝牙协议")
# response.metadata 中会包含原始查询和 HyDE 生成的假想文档
```

`TransformQueryEngine` 的价值在于**解耦**——它把"查询应该是什么样"的逻辑从 Query Engine 本身分离出来，使得你可以独立地修改、替换或组合不同的查询转换策略。

## 自定义 Query Engine 类

如果你有非常特殊的需求，可以从头实现自己的 Query Engine：

```python
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core import Response
from typing import Sequence
from llama_index.core.schema import TextNode, NodeWithScore, QueryBundle
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.base_response_synthesizer import (
    BaseResponseSynthesizer,
)


class RerankThenSummarizeQueryEngine(CustomQueryEngine):
    """
    自定义 Query Engine:
    1. 先用轻量级检索获取大量候选
    2. 再用强力的 Reranker 精排
    3. 最后只对 top-3 做详细合成
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        reranker,  # 任意 reranker 对象
        synthesizer: BaseResponseSynthesizer,
        initial_top_k: int = 30,
        final_top_k: int = 3,
    ):
        self._retriever = retriever
        self._reranker = reranker
        self._synthesizer = synthesizer
        self.initial_top_k = initial_top_k
        self.final_top_k = final_top_k

    def custom_query(self, query_str: str) -> Response:
        # Stage 1: 粗排
        raw_nodes = self._retriever.retrieve(query_str)
        print(f"[Stage 1] 粗排返回 {len(raw_nodes)} 个节点")

        # Stage 2: Rerank
        reranked = self._reranker.postprocess_nodes(
            raw_nodes,
            query_str=query_str,
        )
        print(f"[Stage 2] Rerank 后剩余 {len(reranked)} 个节点")

        # Stage 3: 只取 top-N 合成
        top_nodes = reranked[:self.final_top_k]
        print(f"[Stage 3] 送入合成的节点: {len(top_nodes)} 个")

        # Stage 4: 合成答案
        response = self._synthesizer.synthesize(
            query=query_str,
            nodes=top_nodes,
        )

        return response

    async def aquery(self, query_str: str) -> Response:
        return self.custom_query(query_str)


# 使用
custom_qe = RerankThenSummarizeQueryEngine(
    retriever=index.as_retriever(similarity_top_k=30),
    reranker=CohereRerank(model="rerank-v3.5"),
    synthesizer=get_response_synthesizer(response_mode="refine"),
    initial_top_k=30,
    final_top_k=3,
)

response = custom_qe.query("产品的保修政策")
```

这个自定义 Query Engine 实现了一个"漏斗式"处理模式：先用低成本方法获取大量候选，再用精确但昂贵的方法逐步过滤，最后只对少数高质量结果做深度合成。这种模式在高并发、成本敏感的生产环境中特别有价值。

## 常见误区

**误区一:"越复杂的 Query Engine 越好"。** 不是的。SubQuestionQueryEngine 虽然强大，但对简单问题来说就是杀鸡用牛刀——不仅慢而且贵。**从最简单的方案开始（RetrieverQueryEngine），只在遇到明确的需求时才升级到更复杂的类型。**

**误区二:"RouterQueryEngine 只能做路由"。** RouterQueryEngine 的每个"工具"本身就是一个完整的 Query Engine——它可以有自己的 Retriever、Synthesizer、后处理器等完整配置。所以 RouterQueryEngine 实质上是在**多个完整的查询流水线之间做路由**，而不只是简单的分发。

**误区三:"自定义 Query Engine 很难写"。** 只要继承 `CustomQueryEngine` 并实现 `custom_query()` 方法就可以了。大部分情况下你只是以不同顺序或方式调用已有的 Retriever 和 Synthesizer，并不需要从头实现检索或合成逻辑。
