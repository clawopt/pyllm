---
title: 索引组合策略：如何组合多种索引应对复杂查询
description: Router Query Engine、多索引协作模式、查询路由策略、实际架构案例
---
# 索引组合策略：如何组合多种索引应对复杂查询

上一节我们学习了六种不同的索引类型，每种都有自己的强项和弱项。但在真实项目中，用户的查询类型是多种多样的——有时候问一个精确的事实（"S1 的价格是多少？"），有时候要一个全局概括（"这份报告的核心观点是什么？"），有时候需要关系推理（"谁负责这个产品线？"）。

如果只用一种索引来应对所有这些查询，必然会在某些场景下表现不佳。**解决方案是组合使用多种索引，然后根据查询的类型智能地路由到最合适的索引**——这就是 LlamaIndex 的"索引组合"策略。

## 为什么需要组合？

让我们用一个具体场景来说明单一索引的局限性：

> 一家科技公司的内部知识库包含以下内容：
> - **500 份技术文档**（API 参考、架构设计文档）
> - **200 份 PDF 合同**（客户协议、供应商合同）
> - **3000 条 FAQ**（客服常见问题及答案）
> - **50 份季度报告**（财务数据、业务分析）

现在考虑以下四种完全不同的用户查询：

| 查询 | 最佳索引类型 | 原因 |
|------|-------------|------|
| "API 的 rate limit 参数默认值是多少？" | VectorStoreIndex | 精确的语义匹配 |
| "列出所有提到 'GDPR' 的合同条款" | KeywordTableIndex | 精确关键词匹配 + 法律术语 |
| "Q3 季度报告的整体表现如何？" | SummaryIndex | 需要全局概括能力 |
| "张三负责的产品线有哪些？他的下属是谁？" | GraphIndex | 需要多跳关系推理 |

如果你只用了 VectorStoreIndex：
- 第一种查询 ✅ 表现很好
- 第二种查询 ⚠️ 可能遗漏（"GDPR" 这个精确词可能被语义近似但不同的表达替代）
- 第三种查询 ❌ 只返回几个片段而非全局视角
- 第四种查询 ❌ 完全无法处理

这就是为什么需要组合。

## Router Query Engine：查询路由器

LlamaIndex 提供了 `RouterQueryEngine` 作为组合多种索引的核心组件。它的工作方式类似于网络中的路由器——接收一个查询，分析其特征，然后将它转发到最合适的目标"端口"（子 Query Engine）。

```python
from llama_index.core import VectorStoreIndex, SummaryIndex, KeywordTableIndex
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import PydanticSingleSelector

# ===== 构建多个专用索引 =====

# 索引 A: 技术文档 → 向量搜索
tech_docs = SimpleDirectoryReader("./docs/technical").load_data()
tech_index = VectorStoreIndex.from_documents(tech_docs)
tech_qe = tech_index.as_query_engine(similarity_top_k=5)

# 索引 B: 季度报告 → 全局摘要
report_docs = SimpleDirectoryReader("./docs/reports").load_data()
report_index = SummaryIndex.from_documents(report_docs)
report_qe = report_index.as_query_engine()

# 索引 C: 合同文档 → 关键词搜索
contract_docs = SimpleDirectoryReader("./docs/contracts").load_data()
contract_index = KeywordTableIndex.from_documents(contract_docs)
contract_qe = contract_index.as_query_engine()

# ===== 组合成路由查询引擎 =====

router_query_engine = RouterQueryEngine(
    selector=PydanticSingleSelector(),
    query_engine_tools=[
        tech_qe,      # 工具 1: 技术文档搜索
        report_qe,    # 工具 2: 报告摘要
        contract_qe,  # 工具 3: 合同关键词搜索
    ],
)

# 使用 — 用户不需要知道背后有多个索引
response = router_query_engine.query("Q3 的收入增长率是多少？")
print(response.response)
```

### 路由选择器（Selector）的工作原理

`RouterQueryEngine` 的核心是 **Selector（选择器）**——它决定了每个查询应该被发送到哪个子 Query Engine。LlamaIndex 提供了几种 Selector 实现：

**PydanticSingleSelector（LLM 驱动的单选）：** 使用 LLM 分析查询并选择唯一最佳的目标：

```python
from llama_index.core.selectors import PydanticSingleSelector

selector = PydanticSingleSelector()
# 内部 Prompt 类似于:
# "给定以下工具描述和用户查询，选择最适合处理该查询的工具:
#  - tool_1: 用于搜索技术文档...
#  - tool_2: 用于生成报告摘要...
#  查询: {user_query}
#  请返回最佳工具的编号"
```

这种方式最灵活——LLM 能理解查询的语义意图并做出合理的选择。代价是每次查询都需要一次额外的 LLM 调用（用于路由决策），增加了延迟和成本。

**PydanticMultiSelector（LLM 驱动的多选）：** 允许同时选择多个目标：

```python
from llama_index.core.selectors import PydanticMultiSelector

selector = PydanticMultiSelector()
# 一个查询可能同时需要搜索技术文档 AND 搜索合同
# 例如: "API 文档中关于认证的部分，以及合同中的相关条款"
```

**LLMSingleSelector / LLMMultiSelector（旧版接口）：** 功能类似但使用较旧的接口格式，新项目推荐使用 Pydantic 版本。

### 为路由提供更好的提示信息

Selector 的决策质量高度依赖于它对每个子 Query Engine 能力的了解程度。你可以通过 `metadata` 参数提供更丰富的描述：

```python
from llama_index.core.tools import QueryEngineTool

tools = [
    QueryEngineTool.from_defaults(
        tech_qe,
        name="tech_doc_search",
        description=(
            "用于搜索公司内部的技术文档，包括 API 参考手册、"
            "架构设计文档、开发指南等。适合回答关于产品功能、"
            "技术实现、参数配置等技术问题。"
        ),
    ),
    QueryEngineTool.from_defaults(
        report_qe,
        name="report_summary",
        description=(
            "用于生成季度和年度业务报告的全局摘要。"
            "适合回答关于财务表现、业务趋势、KPI 达成情况等"
            "需要宏观视角的问题。不适用于查找具体的数字或细节。"
        ),
    ),
    QueryEngineTool.from_defaults(
        contract_qe,
        name="contract_search",
        description=(
            "用于在法律合同中搜索特定条款或关键词。"
            "适合回答关于合同条款、法律责任、合规要求等问题。"
            "支持精确的关键词匹配。"
        ),
    ),
]

router_engine = RouterQueryEngine(
    selector=PydanticSingleSelector(),
    query_engine_tools=tools,
)
```

`description` 字段越详细准确，Selector 做出正确路由决策的概率就越高。这是调优 Router Query Engine 最重要的一环。

## SubQuestionQueryEngine：分解复杂问题

有些问题太复杂，无法通过单个索引的一次检索来回答。比如："比较一下我们三个主要竞品的功能差异，并结合我们的产品定位给出建议。"这个问题需要：
1. 找到三个竞品的信息
2. 找到我们自己产品的定位信息
3. 进行比较分析
4. 给出建议

`SubQuestionQueryEngine` 就是用来处理这类复杂问题的——它会把一个大问题自动拆解为多个子问题，分别查询后再综合答案：

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
            description="公司知识库，包含产品、技术、市场等信息",
        ),
    ],
)

response = sub_qe.query(
    "比较 S1 和 S2 两款产品的功能差异，"
    "并说明哪款更适合中小企业客户"
)
print(response.response)
```

SubQuestionQueryEngine 的工作流程如下：

```
原始问题: "比较 S1 和 S2 的功能差异，说明哪个更适合中小企业"

Step 1: 问题分解 (LLM)
┌──────────────────────────────────────────┐
│ 子问题 1: "S1 产品的主要功能有哪些？"      │
│ 子问题 2: "S2 产品的主要功能有哪些？"      │
│ 子问题 3: "中小企业的典型需求是什么？"     │
│ 子问题 4: "S1 和 S2 在哪些方面有差异？"   │
└──────────────────────────────────────────┘
       │
       ▼
Step 2: 并行执行子查询
┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│ Q1 → 检索   │ │ Q2 → 检索   │ │ Q3 → 检索   │ │ Q4 → 检索   │
│ → 回答: ... │ │ → 回答: ... │ │ → 回答: ... │ │ → 回答: ... │
└─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘
       │
       ▼
Step 3: 综合答案 (LLM)
基于所有子问题的答案，生成最终的综合回答
```

这种"分而治之"的策略对于复杂分析类查询特别有效。但要注意的是，它会显著增加 LLM 调用次数（每个子问题至少一次），因此成本和延迟都会相应增加。

## MultiDocumentQueryEngine：跨文档聚合

当你的数据分散在多个独立的数据源中时（如第二章讲的多源融合场景），你可能需要一个能同时查询多个数据源的 Query Engine：

```python
from llama_index.core.query_engine import MultiDocumentQueryEngine
from llama_index.core.response_synthesizers import ResponseMode

mdqe = MultiDocumentQueryEngine(
    query_engine_tools=[
        QueryEngineTool.from_defaults(file_qe, name="file_search"),
        QueryEngineTool.from_defaults(db_qe, name="db_search"),
        QueryEngineTool.from_defaults(notion_qe, name="notion_search"),
    ],
    response_mode=ResponseMode.TREE_SUMMARIZE,  # 用树状汇总整合多源结果
    verbose=True,  # 打印中间过程便于调试
)

response = mdqe.query("退款政策的完整流程是什么？")
# 会同时在文件搜索、数据库搜索、Notion搜索中查找
# 然后汇总所有来源的结果
```

## 实际架构案例：企业级多索引 RAG 系统

下面是一个生产级的多索引 RAG 系统架构示例，展示了如何在实际项目中组合使用各种索引：

```python
class EnterpriseRAGSystem:
    """企业级多索引 RAG 系统"""

    def __init__(self):
        self.indexes = {}
        self.router = None
        self._build_indexes()
        self._build_router()

    def _build_indexes(self):
        # 1. 主力索引：向量搜索（覆盖大部分通用查询）
        all_docs = self._load_all_sources()
        self.indexes["vector"] = VectorStoreIndex.from_documents(all_docs)

        # 2. 辅助索引：关键词搜索（用于精确术语匹配）
        contract_docs = self._load_contracts()
        parser = KeywordNodeParser(keywords=10)
        nodes = parser.get_nodes_from_documents(contract_docs)
        self.indexes["keyword"] = KeywordTableIndex(nodes=nodes)

        # 3. 辅助索引：摘要索引（用于概览类查询）
        report_docs = self._load_reports()
        self.indexes["summary"] = SummaryIndex.from_documents(report_docs)

        # 4. 特殊索引：知识图谱（用于关系查询）
        org_docs = self._load_org_data()
        graph_store = SimpleGraphStore()
        self.indexes["graph"] = KnowledgeGraphIndex.from_documents(
            org_docs, graph_store=graph_store
        )

    def _build_router(self):
        tools = [
            QueryEngineTool.from_defaults(
                self.indexes["vector"].as_query_engine(),
                name="general_search",
                description="通用语义搜索，适合大多数问题",
            ),
            QueryEngineTool.from_defaults(
                self.indexes["keyword"].as_query_engine(),
                name="keyword_search",
                description="精确关键词搜索，适合查找法条、编号、专有名词",
            ),
            QueryEngineTool.from_defaults(
                self.indexes["summary"].as_query_engine(),
                name="summary",
                description="文档摘要，适合概括性问题",
            ),
            QueryEngineTool.from_defaults(
                self.indexes["graph"].as_query_engine(),
                name="graph_query",
                description="知识图谱查询，适合人员组织、实体关系等问题",
            ),
        ]

        self.router = RouterQueryEngine(
            selector=PydanticSingleSelector(),
            query_engine_tools=tools,
        )

    def query(self, question: str) -> str:
        response = self.router.query(question)
        return response.response


# 使用
rag = EnterpriseRAGSystem()
answer = rag.query("我们的 GDPR 合规措施有哪些？")
```

## 常见误区

**误区一:"路由越多越好"。** 不是的。每增加一个路由目标就增加了一次 LLM 调用（用于路由决策），也增加了系统复杂度和出错概率。从 2-3 个核心索引开始，只在确实遇到某些查询类型无法有效处理时才添加新的索引。**YAGNI 原则（You Aren't Gonna Need It）同样适用于索引设计。**

**误区二:"Selector 总是能做出正确的路由决策"。** 不可能 100% 正确。LLM 对查询意图的理解可能出错，特别是对于模糊或混合意图的查询（"告诉我关于 X 的所有信息"——这到底是该去向量搜索还是摘要索引？）。缓解方法包括：提供更详细的工具描述、收集错误路由的案例来优化 prompt、以及在某些情况下允许 fallback 到默认索引。

**误区三:"不同索引之间是完全独立的"。** 实际上它们经常共享相同的基础数据。你不需要为每种索引类型维护一份独立的数据副本——可以从同一份 Document 列表构建多个 Index，或者让一个 Index 的输出作为另一个 Index 的输入。**共享数据源、差异化索引策略**是多索引系统的核心设计原则。
