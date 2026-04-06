---
title: RetrieverQueryEngine：检索型查询引擎详解
description: 手动组装 Query Engine、自定义 Retriever/Synthesizer、高级配置选项、调试技巧
---
# RetrieverQueryEngine：检索型查询引擎详解

上一节我们从宏观层面介绍了 Query Engine 的架构。这一节来聚焦于最常用也最重要的具体实现——`RetrieverQueryEngine`。它是你在 95% 的 RAG 应用中会使用的 Query Engine 类型，理解它的每一个配置选项和内部行为，对于构建高质量的 RAG 系统至关重要。

## RetrieverQueryEngine 的完整创建方式

虽然 `index.as_query_engine()` 是最快捷的方式，但了解手动组装的过程能让你获得完全的控制权：

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizer import get_response_synthesizer
from llama_index.core.response_synthesizers import ResponseMode

# 准备数据
documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)

# Step 1: 配置 Retriever
retriever = index.as_retriever(
    similarity_top_k=20,  # 多取一些候选
)

# Step 2: 配置 Response Synthesizer
synthesizer = get_response_synthesizer(
    response_mode=ResponseMode.REFINE,
    use_async=True,         # 异步加速
    verbose=True,           # 打印中间过程
)

# Step 3: 组装
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=synthesizer,
    node_postprocessors=[      # 后处理器（也可放在这里）
        CohereRerank(top_n=8),
        DeduplicateNodePostProcessor(),
    ],
)

# 使用
response = query_engine.query("公司的退款政策是什么？")
print(response.response)
```

### 为什么选择 REFINE 作为默认合成模式？

`ResponseMode.REFINE` 是 LlamaIndex 推荐的默认合成模式，原因如下：

**第一，它能充分利用所有检索到的信息。** 与 SIMPLE_SUMMARIZE（只看一遍全部内容）不同，REFINE 模式会对每个 Node 进行迭代式的精炼，确保即使排在后面的 Node 中的关键信息也不会被忽略。

**第二，它的输出质量通常更高。** 由于每次精炼都是基于前一次结果的改进，最终答案往往比一次性生成更加完整和准确。

**第三，它的 token 效率相对合理。** 虽然 TREE_SUMMARIZE 对大量 Node 更有效，但它需要更多的 LLM 调用次数。REFINE 在质量和效率之间取得了较好的平衡。

### REFINE 模式的内部工作机制

让我们深入看看 REFINE 模式到底是怎么工作的：

```python
# 假设检索到了 5 个 Node
nodes = [
    Node(text="公司提供30天无理由退款服务...", score=0.92),
    Node(text="退款流程：登录账户→订单管理→申请...", score=0.89),
    Node(text="特殊商品不适用无理由退款政策...", score=0.85),
    Node(text="退款将在3-5个工作日内原路退回...", score=0.82),
    Node(text="如有疑问请联系客服400-xxx...", score=0.78),
]
query = "退款政策是什么？"
```

REFINE 的执行过程如下：

```
=== Iteration 1 (初始生成) ===
Input to LLM:
  System: 你是一个有帮助的助手。请根据以下上下文信息回答用户的问题。
  Context: 公司提供30天无理由退款服务...
  Question: 退款政策是什么？

Output from LLM:
  "根据公司政策，客户可在购买后30天内申请全额退款。"

=== Iteration 2 (第一次精炼) ===
Input to LLM:
  System: 我们有一个关于'{query}'的现有答案：
  '{existing_answer}'
  我们有机会通过以下更多上下文来精炼这个答案：
  '{next_node_text}'
  如果新上下文有用，请精炼答案。如果没有用，保持原样。

  Existing Answer: 根据公司政策，客户可在购买后30天内申请全额退款。
  New Context: 退款流程：登录账户进入订单管理页面，
             点击对应订单的"申请退款"按钮...

Output from LLM:
  "根据公司政策，客户可在购买后30天内申请全额退款。
   具体操作流程为：登录账户进入订单管理页面，
   点击对应订单的'申请退款'按钮填写退款原因后提交审核。"

=== Iteration 3 (第二次精炼) ===
... (继续用第3个Node精炼)

=== Iteration 4 (第三次精炼) ===
... (继续用第4个Node精炼)

=== Iteration 5 (第四次精炼) ===
... (继续用第5个Node精炼)

=== Final Output ===
"根据公司政策，客户可在购买后30天内申请全额退款。
 具体操作流程为：登录账户进入订单管理页面...
 特殊商品（如定制产品、已拆封商品）不适用此政策。
 退款将在3-5个工作日内原路退回至原支付账户。
 如有任何疑问，请联系客服热线400-xxx..."
```

注意答案是如何随着每次迭代逐渐丰富起来的——从最初的"一句话"，变成了包含流程、例外情况、到账时间、联系方式等完整信息的全面回答。

### 自定义 REFINE Prompt

你可以完全控制 REFINE 过程中使用的 Prompt 模板：

```python
from llama_index.core.prompts.prompt_template import PromptTemplate

text_qa_template = PromptTemplate(
    "你是一个专业的客服助手。请严格基于以下上下文信息回答问题。\n\n"
    "上下文信息:\n{context_str}\n\n"
    "用户问题:\n{query_str}\n\n"
    "要求：\n"
    "- 只使用上下文中的信息，不要编造\n"
    "- 如果上下文中没有答案，直接说'我不知道'\n"
    "- 回答时引用具体的条款或规定\n"
)

refine_template = PromptTemplate(
    "我们有一个关于'{query_str}'的现有答案：\n"
    "'{existing_answer}'\n\n"
    "以下是新的参考信息：\n"
    "'{context_msg}'\n\n"
    "请基于新的信息精炼原有答案。\n"
    "如果新信息与已有答案矛盾，以新信息为准并注明变化。\n"
    "如果新信息没有帮助，保持原样不变。\n"
)

synthesizer = get_response_synthesizer(
    response_mode=ResponseMode.REFINE,
    text_qa_template=text_qa_template,
    refine_template=refine_template,
)
```

通过自定义 Prompt，你可以：
- 控制 LLM 的"人设"和语气风格
- 添加领域特定的指令（如法律免责声明）
- 要求特定格式的输出（如 Markdown 表格、JSON 等）
- 注入安全约束（如"不要泄露敏感信息"）

## 使用自定义 Retriever

`RetrieverQueryEngine` 接受任何实现了 `BaseRetriever` 接口的检索器对象。这意味着你可以轻松地把第五章学到的高级检索技术集成进来：

```python
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.query_engine import RetrieverQueryEngine

# 创建混合检索器
vector_retriever = index.as_retriever(similarity_top_k=20)
bm25_retriever = BM25Retriever.from_defaults(
    nodes=list(index.index_struct.nodes_dict.values()),
    similarity_top_k=20,
)
hybrid_retriever = QueryFusionRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    mode="reciprocal_rank",
    similarity_top_k=10,
)

# 用自定义 Retriever 组装 Query Engine
query_engine = RetrieverQueryEngine(
    retriever=hybrid_retriever,  # ← 这里传入自定义的混合检索器
    response_synthesizer=get_response_synthesizer(
        response_mode=ResponseMode.COMPACT_ACCUMULATE,
    ),
)

response = query_engine.query("S1 支持哪些无线协议？")
```

这段代码的关键在于：**RetrieverQueryEngine 不关心你的 Retriever 是怎么实现的**——它是向量搜索、BM25、混合检索还是某种完全自定义的逻辑都不重要。只要它实现了 `retrieve(query) -> List[NodeWithScore]` 这个接口，就能无缝集成。

## 调试 Query Engine

当 Query Engine 的输出不如预期时，你需要系统地排查问题出在哪一层。以下是一些实用的调试技巧：

### 技巧一：分离 Retriever 和 Synthesizer

```python
# 先单独测试 Retriever
nodes = query_engine._retriever.retrieve("问题？")
print(f"检索到 {len(nodes)} 个节点:")
for i, n in enumerate(nodes):
    print(f"  [{i}] score={n.score:.3f} | {n.text[:100]}...")

# 如果检索结果没问题，再单独测试 Synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine
synth_response = query_engine._response_synthesizer.synthesize(
    query="问题？",
    nodes=nodes[:5],  # 手动指定节点
)
print(f"\n合成结果:\n{synth_response.response}")
```

通过这种方式可以快速定位问题是出在**检索阶段**（没找到正确的内容）还是**合成阶段**（找到了但没组织好答案）。

### 技巧二：启用 Verbose 模式

```python
query_engine = index.as_query_engine(
    similarity_top_k=5,
    verbose=True,  # 打印详细的中间过程
)

response = query_engine.query("问题？")
# 会打印类似:
# > 检索阶段: 找到 5 个节点
# > 后处理阶段: 经过 reranker 后剩余 5 个节点
# > 合成阶段: 使用 REFINE 模式，共 5 次迭代
# > 第 1 次: 生成初始答案 (128 tokens)
# > 第 2 次: 精炼答案 (156 tokens)
# ...
```

### 技巧三：使用 Callbacks 监控

```python
from llama_index.core.callbacks import CallbackManager, CBEventHandler

handler = CBEventHandler(event_starts_to_ignore=[], event_ends_to_ignore=[])
callback_manager = CallbackManager([handler])

query_engine = index.as_query_engine(callback_manager=callback_manager)
response = query_engine.query("问题？")

# handler 中记录了所有事件的时间戳和详细信息
for event in handler.event_pairs:
    print(f"{event.id}: {event.duration_ms:.0f}ms")
```

## 异步与批量查询

对于高并发场景或需要同时处理多个查询的情况，异步 API 至关重要：

```python
import asyncio

async def batch_query(queries: list[str]) -> list[str]:
    """批量异步查询"""
    query_engine = index.as_query_engine()

    tasks = [query_engine.aquery(q) for q in queries]
    responses = await asyncio.gather(*tasks)

    return [r.response for r in responses]


queries = [
    "产品的价格是多少？",
    "怎么申请退货？",
    "支持哪些支付方式？",
    "保修期多长？",
]

results = asyncio.run(batch_query(queries))
for q, r in zip(queries, results):
    print(f"Q: {q}")
    print(f"A: {r[:150]}...")
    print()
```

`aquery()` 是 `query()` 的异步版本，底层使用了 Python 的 asyncio 来实现并发执行。对于 IO 密集型的 RAG 查询（网络调用 LLM API、向量数据库查询等），异步可以将吞吐量提升 3-5 倍。

## 常见误区

**误区一:"response_mode 设一次就不用管了"。** 不同的查询类型适合不同的合成模式。事实型查询用 SIMPLE_SUMMARIZE 就够了（速度快），复杂分析型查询用 REFINE 或 TREE_SUMMARIZE 效果更好。**考虑根据查询类型动态选择合成模式。**

**误区二:"similarity_top_k 越大越好"。** 不是的。top_k 太大会引入噪音（低分结果），增加合成阶段的 token 消耗和 LLM 调用次数，还可能降低答案质量（LLM 被无关信息干扰）。**经过 reranker 后 top_k=5 到 top_k=10 通常是最优区间。**

**误区三:"Query Engine 创建后就固定不变了"。** 你可以在运行时动态修改 Query Engine 的组件。比如在检测到某个查询类型后临时切换 Synthesizer 的模式，或者根据用户权限动态调整 Retriever 的过滤条件。**Query Engine 是可组合、可替换的，不是一成不变的。**
