# 6.2 子图间的状态映射与数据流

> 在上一节中我们讨论了如何从架构层面划分子图、定义接口和组装父图。但有一个关键的技术细节还没有深入探讨：**父图和子图之间的状态是如何传递的？** 当父图的状态字段名和子图的状态字段名不同时怎么办？当一个子图的输出需要作为另一个子图的输入时，数据应该如何流转？这一节我们会系统地解决这些子图间通信的核心问题。

## 状态映射的三种模式

LangGraph 中父图和子图之间的状态传递有三种基本模式，根据复杂度递增排列：

### 模式一：字段名自动匹配（最简单）

当父图和子图恰好有一些同名字段时，LangGraph 会自动在这些字段之间进行值的传递。这是最简单的情况，不需要任何额外的配置。

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class SharedFieldState(TypedDict):
    user_input: str
    processed_result: str
    confidence: float
    parent_only_field: str

class SubgraphState(TypedDict):
    user_input: str
    processed_result: str
    confidence: float
    _subgraph_internal: str

def sub_node(state: SubgraphState) -> dict:
    text = state["user_input"].upper()
    return {
        "processed_result": text,
        "confidence": 0.95,
        "_subgraph_internal": "some internal data"
    }

sub_graph = StateGraph(SubgraphState)
sub_graph.add_node("process", sub_node)
sub_graph.add_edge(START, "process")
sub_graph.add_edge("process", END)
compiled_sub = sub_graph.compile()

def parent_node(state: SharedFieldState) -> dict:
    result = compiled_sub.invoke({
        "user_input": state["user_input"],
        "processed_result": "",
        "confidence": 0.0,
        "_subgraph_internal": ""
    })
    return {
        "processed_result": result["processed_result"],
        "confidence": result["confidence"]
    }

parent_graph = StateGraph(SharedFieldState)
parent_graph.add_node("run_sub", parent_node)
parent_graph.add_edge(START, "run_sub")
parent_graph.add_edge("run_sub", END)

app = parent_graph.compile()
result = app.invoke({
    "user_input": "hello world",
    "processed_result": "",
    "confidence": 0.0,
    "parent_only_field": "secret"
})
print(f"结果: {result['processed_result']}")  # HELLO WORLD
print(f"置信度: {result['confidence']}")     # 0.95
```

在这个例子中，`SharedFieldState` 和 `SubgraphState` 共享了三个字段：`user_input`、`processed_result`、`confidence`。当 `compiled_sub` 被执行时，这三个字段的值会自动在父子图之间传递。注意 `_subgraph_internal` 字段只存在于子图中，它不会泄漏到父图；同样 `parent_only_field` 只存在于父图中，子图看不到它。

### 模式二：包装函数手动映射（最灵活）

当父图和子图的字段名完全不同时，或者需要在传递过程中做数据转换时，就需要用包装函数来手动处理映射关系。这是最灵活的方式，也是实际项目中最常用的方式。

```python
class OrderParentState(TypedDict):
    order_id: str
    customer_name: str
    items: list[dict]
    total_amount: float
    risk_score: float
    approval_status: str
    audit_trail: list[str]

class RiskCheckSubState(TypedDict):
    transaction_id: str
    buyer_name: str
    purchase_items: list[dict]
    transaction_value: float
    risk_flags: list[str]
    risk_level: str
    risk_score: float

def check_fraud_risk(state: RiskCheckSubState) -> dict:
    value = state["transaction_value"]
    flags = []
    if value > 10000:
        flags.append("大额交易")
    if value > 50000:
        flags.append("超大额-需人工审核")

    score = len(flags) * 20
    level = "high" if score >= 40 else ("medium" if score >= 20 else "low")

    return {
        "risk_flags": flags,
        "risk_level": level,
        "risk_score": score
    }

risk_subgraph = StateGraph(RiskCheckSubState)
risk_subgraph.add_node("check", check_fraud_risk)
risk_subgraph.add_edge(START, "check")
risk_subgraph.add_edge("check", END)
compiled_risk = risk_subgraph.compile()

def risk_check_wrapper(state: OrderParentState) -> dict:
    mapped_input = {
        "transaction_id": state["order_id"],
        "buyer_name": state["customer_name"],
        "purchase_items": state["items"],
        "transaction_value": state["total_amount"],
        "risk_flags": [],
        "risk_level": "low",
        "risk_score": 0.0
    }

    sub_result = compiled_risk.invoke(mapped_input)

    return {
        "risk_score": sub_result["risk_score"],
        "approval_status": "approved" if sub_result["risk_level"] != "high"
                           else "needs_review",
        "audit_trail": [f"风控检查: {sub_result['risk_level']} (分数:{sub_result['risk_score']})"]
    }
```

这个包装函数 `risk_check_wrapper` 做了三件事：第一，把父图的字段（`order_id`/`customer_name`/`items`/`total_amount`）映射为子图需要的字段名（`transaction_id`/`buyer_name`/`purchase_items`/`transaction_value`）；第二，调用子图并获取结果；第三，把子图的输出字段映射回父图的字段名（`risk_score`/`approval_status`/`audit_trail`）。整个映射逻辑完全由你控制，可以做任意复杂的数据转换。

### 模式三：共享基类继承（最规范）

第三种方式是通过类型继承来实现字段共享——定义一个包含公共字段的基类，然后父图和子图分别继承并扩展自己的特有字段。这种方式在大型项目中特别有价值，因为它保证了公共字段的一致性。

```python
from typing import TypedDict

class BaseMessageState(TypedDict):
    message_id: str
    content: str
    sender_id: str
    timestamp: str
    metadata: dict

class ClassificationSubState(BaseMessageState):
    intent: str
    confidence: float
    entities: list[dict]

class ResponseGenSubState(BaseMessageState):
    context_sources: list[dict]
    generated_response: str
    response_tokens: int

class QualityCheckSubState(BaseMessageState):
    quality_score: float
    issues_found: list[str]
    improvement_suggestions: list[str]

class ParentOrchestratorState(BaseMessageState):
    classification_result: dict
    generation_result: dict
    quality_result: dict
    final_output: str
    pipeline_status: str
```

通过这种继承结构，所有子图都天然拥有 `message_id`、`content`、`sender_id`、`timestamp`、`metadata` 这些公共字段，而各自的特有字段（如 `intent`、`generated_response`、`quality_score`）保持独立。在包装函数中，公共字段可以直接透传，只有特有字段需要显式映射。

## 多子图之间的数据流转

在实际系统中，一个父图通常包含多个子图，而且前一个子图的输出往往需要作为后一个子图的输入。这就涉及到了**多子图之间的数据流转**问题。

```python
from typing import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, START, END

class PipelineParentState(TypedDict):
    raw_text: str
    classification: dict
    retrieval_results: list[dict]
    generated_answer: str
    quality_report: dict
    pipeline_log: Annotated[list[str], operator.add]

# 子图1: 文本分类
class ClassifyState(TypedDict):
    input_text: str
    category: str
    subcategories: list[str]
    confidence: float

classify_graph = StateGraph(ClassifyState)
classify_graph.add_node("categorize", lambda s: {
    "category": "technical",
    "subcategories": ["api", "integration"],
    "confidence": 0.87
})
classify_graph.add_edge(START, "categorize")
classify_graph.add_edge("categorize", END)
compiled_classify = classify_graph.compile()

# 子图2: 知识库检索
class RetrieveState(TypedDict):
    query: str
    category: str
    results: list[dict]

retrieve_graph = StateGraph(RetrieveState)
retrieve_graph.add_node("search", lambda s: {
    "results": [
        {"id": "doc-1", "title": f"{s['category']} 指南",
         "relevance": 0.92, "content": "详细内容..."},
        {"id": "doc-2", "title": f"{s['category']} 最佳实践",
         "relevance": 0.85, "content": "更多内容..."}
    ]
})
retrieve_graph.add_edge(START, "search")
retrieve_graph.add_edge("search", END)
compiled_retrieve = retrieve_graph.compile()

# 子图3: 回答生成
class GenerateState(TypedDict):
    question: str
    context: list[dict]
    answer: str

generate_graph = StateGraph(GenerateState)
generate_graph.add_node("llm_generate", lambda s: {
    "answer": f"基于 {len(s['context'])} 条参考文档的回答: {s['question']}"
})
generate_graph.add_edge(START, "llm_generate")
generate_graph.add_edge("llm_generate", END)
compiled_generate = generate_graph.compile()

# 父图：编排三个子图的数据流
def step1_classify(state: PipelineParentState) -> dict:
    result = compiled_classify.invoke({
        "input_text": state["raw_text"],
        "category": "", "subcategories": [], "confidence": 0.0
    })
    return {
        "classification": result,
        "pipeline_log": [f"[步骤1] 分类: {result['category']} (置信度: {result['confidence']})"]
    }

def step2_retrieve(state: PipelineParentState) -> dict:
    cls_result = state["classification"]
    result = compiled_retrieve.invoke({
        "query": state["raw_text"],
        "category": cls_result.get("category", "general"),
        "results": []
    })
    return {
        "retrieval_results": result["results"],
        "pipeline_log": [f"[步骤2] 检索到 {len(result['results'])} 条结果"]
    }

def step3_generate(state: PipelineParentState) -> dict:
    result = compiled_generate.invoke({
        "question": state["raw_text"],
        "context": state["retrieval_results"],
        "answer": ""
    })
    return {
        "generated_answer": result["answer"],
        "pipeline_log": [f"[步骤3] 生成回答完成 ({len(result['answer'])} 字符)"]
    }

pipeline_graph = StateGraph(PipelineParentState)
pipeline_graph.add_node("classify", step1_classify)
pipeline_graph.add_node("retrieve", step2_retrieve)
pipeline_graph.add_node("generate", step3_generate)

pipeline_graph.add_edge(START, "classify")
pipeline_graph.add_edge("classify", "retrieve")   # 分类结果 → 检索输入
pipeline_graph.add_edge("retrieve", "generate")   # 检索结果 → 生成输入
pipeline_graph.add_edge("generate", END)

app = pipeline_graph.compile()

result = app.invoke({
    "raw_text": "如何使用 Python 的 asyncio 库进行并发编程？",
    "classification": {}, "retrieval_results": [],
    "generated_answer": "", "quality_report": {},
    "pipeline_log": []
})

for entry in result["pipeline_log"]:
    print(entry)
print(f"\n最终回答:\n{result['generated_answer']}")
```

这个流水线示例展示了经典的多子图数据流转模式：

```
raw_text ──→ [分类子图] ──→ classification (category, confidence)
                                    │
                                    ↓ (category 作为查询条件)
                              [检索子图] ──→ retrieval_results
                                                    │
                                                    ↓ (context 作为生成素材)
                                              [生成子图] ──→ generated_answer
```

每个子图的包装节点负责两件事：**把前序子图的输出映射为本子图的输入**，以及**把本子图的输出存入父图状态供后续使用**。这种"适配器"模式让每个子图都可以独立开发和测试——只要接口不变，内部实现可以随意替换。

## 数据流的常见陷阱与解决方案

在处理多子图数据流时，有几个容易出问题的点值得特别注意。

**陷阱一：字段名拼写错误导致静默丢失数据**。如果你在映射函数中把目标字段名写错了（比如写成了 `retrieval_result` 而不是 `retrieval_results`），数据会被写入到一个不存在的字段中，后续节点读取时拿到的是空值或默认值，而且不会报任何错误。防御性做法是在包装函数的最后打印一下关键映射的摘要日志。

**陷阱二：子图输出的数据被后续子图意外覆盖**。如果两个子图往父图状态的同一个字段写入数据，后执行的会覆盖先执行的。确保每个子图写入的字段是唯一的，或者在父图中做合并处理。

**陷阱三：循环引用导致无限嵌套**。如果图 A 包含子图 B，而子图 B 又反过来包含图 A，就会形成无限递归。LangGraph 编译时不会检测这种跨图的循环依赖（因为它是运行时的函数调用），只有在执行时才会触发栈溢出。解决方法是始终保持依赖关系的单向性。

**陷阱四：大对象在子图间反复序列化**。如果一个子图输出了很大的数据（比如完整的文档列表），而这个数据又被传入下一个子图再传回来，每次经过子图边界都会有一次序列化/反序列化的开销。对于大块数据，考虑在外部存储（数据库/文件系统）中保存，状态中只传递引用 ID。
