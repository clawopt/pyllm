# 2.4 子图与模块化组合

> 当你的图从 3 个节点增长到 15 个节点时，你会发现主图的定义文件已经变得难以阅读和维护。这时候就需要"子图"（Subgraph）出场了——它就像编程中的函数封装，把一组相关的节点打包成一个独立的、可复用的图单元，然后在更大的图中像调用普通节点一样使用它。LangGraph 的子图机制让你能够真正实现"组合优于继承"的设计哲学。

## 从一个痛点说起：为什么需要子图

假设你在构建一个代码审查系统，整个流程包含：克隆仓库 → 分析代码结构 → 检查安全漏洞 → 检查性能问题 → 检查代码风格 → 汇总报告 → 生成修复建议 → 人工审核。如果把这 8 个节点全部平铺在一个 StateGraph 里，不仅定义文件会变得很长，而且其中"检查安全漏洞"、"检查性能问题"、"检查代码风格"这三个步骤的逻辑结构几乎一模一样——都是"运行检查器 → 解析结果 → 判断严重程度 → 决定是否阻断流程"。这种重复的模式天然就适合抽取成子图。

更实际地说，子图解决的是三个层面的问题：第一是**可读性**，把复杂的图拆分成有语义的模块，每个模块只关注自己的职责；第二是**复用性**，同一个检查逻辑可以在多个不同的主图中被引用，而不需要复制粘贴节点定义；第三是**隔离性**，子图有自己的内部状态空间，它的内部变化不会意外地污染外层的状态。下面我们通过具体的例子来逐步理解这些概念。

## 子图的基本用法：将图作为节点

在 LangGraph 中创建子图的思路非常直观：你先像往常一样用 StateGraph 定义一个完整的子图，编译它得到 CompiledGraph 对象，然后把这个编译后的图作为一个节点添加到另一个父图中。父图并不关心这个节点背后是一个普通函数还是一个完整的子图——对它来说两者没有区别，都是接收状态输入、返回状态更新的处理单元。

让我们从一个最简单的例子开始。假设我们有一个数据处理流水线，其中包含两个阶段：数据清洗和数据验证。我们把验证逻辑抽取为一个独立的子图：

```python
from typing import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, START, END

class ValidationState(TypedDict):
    raw_data: str
    cleaned_data: str
    validation_errors: list[str]
    is_valid: bool
    validation_score: float

def format_checker(state: ValidationState) -> dict:
    data = state["cleaned_data"]
    errors = []
    if len(data) < 10:
        errors.append("数据长度不足10个字符")
    if not any(c.isalpha() for c in data):
        errors.append("数据不包含字母")
    return {"validation_errors": errors}

def content_checker(state: ValidationState) -> dict:
    data = state["cleaned_data"]
    errors = list(state["validation_errors"])
    forbidden = ["密码", "secret", "token"]
    for word in forbidden:
        if word.lower() in data.lower():
            errors.append(f"包含敏感词: {word}")
    return {"validation_errors": errors}

def finalize_validation(state: ValidationState) -> dict:
    errors = state["validation_errors"]
    score = max(0, 100 - len(errors) * 15)
    return {"is_valid": len(errors) == 0, "validation_score": score}

validation_graph = StateGraph(ValidationState)
validation_graph.add_node("format_check", format_checker)
validation_graph.add_node("content_check", content_checker)
validation_graph.add_node("finalize", finalize_validation)
validation_graph.add_edge(START, "format_check")
validation_graph.add_edge("format_check", "content_check")
validation_graph.add_edge("content_check", "finalize")
validation_graph.add_edge("finalize", END)

compiled_validation = validation_graph.compile()
```

到这里为止，`compiled_validation` 就是一个完全独立的、可以单独运行的子图。你可以直接用它来验证数据：

```python
result = compiled_validation.invoke({
    "raw_data": "test",
    "cleaned_data": "hello world",
    "validation_errors": [],
    "is_valid": False,
    "validation_score": 0.0
})
print(result["is_valid"])        # True
print(result["validation_score"]) # 100.0
```

但更有趣的是，我们可以把它嵌入到一个更大的父图中，作为其中的一个节点来使用。关键在于理解**状态映射**的问题：父图和子图可能使用不同的 State 类型，LangGraph 需要知道如何把父图的状态字段映射到子图的状态字段上。

```python
class PipelineState(TypedDict):
    input_text: str
    processed_text: Annotated[str, operator.add]
    quality_report: str
    passed_validation: bool

def clean_data(state: PipelineState) -> dict:
    text = state["input_text"].strip().lower()
    return {"processed_text": text}

def generate_report(state: PipelineState) -> dict:
    valid = state["passed_validation"]
    status = "✅ 通过" if valid else "❌ 未通过"
    report = f"数据处理完成 | 验证状态: {status}"
    return {"quality_report": report}
```

现在把子图作为节点加入父图。这里有一个非常重要的细节：当子图被当作节点使用时，LangGraph 会自动尝试按**字段名匹配**来做状态映射——也就是说，如果父图和子图的 State 类型中有同名字段，这些字段的值会在进入子图时自动传递过去，子图的输出也会自动写回父图的同名字段。但如果字段名不同呢？这就需要用到我们在后面会详细讨论的状态映射机制了。对于当前这个例子，我们故意让 `PipelineState` 和 `ValidationState` 共享了一些字段名（比如 `cleaned_data` 和 `processed_text` 的映射关系需要在节点函数中手动处理），先看最简单的情况：

```python
class SharedState(TypedDict):
    input_text: str
    cleaned_data: str
    validation_errors: Annotated[list[str], operator.add]
    is_valid: bool
    validation_score: float
    final_report: str

def clean_step(state: SharedState) -> dict:
    return {"cleaned_data": state["input_text"].strip()}

def report_step(state: SharedState) -> dict:
    score = state["validation_score"]
    valid = state["is_valid"]
    report = f"验证得分: {score:.1f} | 结果: {'通过' if valid else '失败'}"
    return {"final_report": report}

main_graph = StateGraph(SharedState)
main_graph.add_node("clean", clean_step)
main_graph.add_node("validate", compiled_validation)
main_graph.add_node("report", report_step)
main_graph.add_edge(START, "clean")
main_graph.add_edge("clean", "validate")
main_graph.add_edge("validate", "report")
main_graph.add_edge("report", END)

app = main_graph.compile()

result = app.invoke({
    "input_text": "  Hello World  ",
    "cleaned_data": "",
    "validation_errors": [],
    "is_valid": False,
    "validation_score": 0.0,
    "final_report": ""
})

print(result["cleaned_data"])     # "Hello World"
print(result["is_valid"])         # True (长度>10? 不，这里只有11字符... 等等)
print(result["final_report"])      # 验证得分: 100.0 | 结果: 通过
```

这段代码展示了子图最核心的使用方式：`compiled_validation` 被当作一个普通节点 `"validate"` 添加到 `main_graph` 中。当执行流到达 `"validate"` 节点时，LangGraph 会把当前的 `SharedState` 传入子图，子图内部的三个节点依次执行，最终子图的输出结果会被合并回父图的状态中。对外部的观察者来说，`"validate"` 节点和普通的函数节点没有任何区别——它同样接收一个状态字典，返回更新后的状态字典。

## 状态映射：父子图之间的桥梁

上面那个例子里我们故意让父图和子图共享了相同的字段名，这样 LangGraph 可以自动完成映射。但在真实项目中，父图和子图往往有不同的状态结构，这时就需要显式地指定状态映射规则。LangGraph 提供了几种方式来处理这个问题，其中最常用的是在将子图添加为节点时传入状态映射函数。

理解状态映射的关键在于想清楚数据的流向：当父图的执行流进入子图节点时，需要从父图的状态中提取出子图需要的字段；当子图执行完毕后，需要把子图的输出结果写回到父图的状态中。这两个方向都可以自定义。

```python
from typing import TypedDict, Any
import operator
from langgraph.graph import StateGraph, START, END

class OrderState(TypedDict):
    order_id: str
    customer_name: str
    items: list[dict]
    total_amount: float
    payment_status: str
    risk_level: str
    fraud_score: float
    order_notes: Annotated[str, operator.add]

class RiskCheckState(TypedDict):
    transaction_id: str
    buyer_name: str
    amount: float
    item_count: int
    risk_flags: list[str]
    calculated_risk: str
    risk_score: float

def check_amount_risk(state: RiskCheckState) -> dict:
    amount = state["amount"]
    flags = []
    if amount > 10000:
        flags.append("大额交易")
    if amount > 50000:
        flags.append("超大额交易-需人工审核")
    return {"risk_flags": flags}

def check_frequency_risk(state: RiskCheckState) -> dict:
    item_count = state["item_count"]
    flags = list(state["risk_flags"])
    if item_count > 20:
        flags.append("异常商品数量")
    return {"risk_flags": flags}

def calculate_final_risk(state: RiskCheckState) -> dict:
    flags = state["risk_flags"]
    score = len(flags) * 20
    if score >= 60:
        level = "high"
    elif score >= 20:
        level = "medium"
    else:
        level = "low"
    return {"calculated_risk": level, "risk_score": score}

risk_subgraph = StateGraph(RiskCheckState)
risk_subgraph.add_node("check_amount", check_amount_risk)
risk_subgraph.add_node("check_frequency", check_frequency_risk)
risk_subgraph.add_node("calculate", calculate_final_risk)
risk_subgraph.add_edge(START, "check_amount")
risk_subgraph.add_edge("check_amount", "check_frequency")
risk_subgraph.add_edge("check_frequency", "calculate")
risk_subgraph.add_edge("calculate, END")

compiled_risk = risk_subgraph.compile()
```

注意这里的 `OrderState` 和 `RiskCheckState` 字段名完全不同——`order_id` 对应 `transaction_id`，`customer_name` 对应 `buyer_name`，`total_amount` 对应 `amount`，`items` 的长度对应 `item_count`。我们需要建立这两套状态 schema 之间的映射关系。在 LangGraph 中，这可以通过包装函数来实现：

```python
def risk_check_node(state: OrderState) -> dict:
    mapped_input = {
        "transaction_id": state["order_id"],
        "buyer_name": state["customer_name"],
        "amount": state["total_amount"],
        "item_count": len(state["items"]),
        "risk_flags": [],
        "calculated_risk": "low",
        "risk_score": 0.0
    }
    sub_result = compiled_risk.invoke(mapped_input)
    return {
        "risk_level": sub_result["calculated_risk"],
        "fraud_score": sub_result["risk_score"],
        "order_notes": f"风控检查完成 | 等级: {sub_result['calculated_risk']}"
    }

order_graph = StateGraph(OrderState)
order_graph.add_node("risk_check", risk_check_node)
```

这种包装函数的方式是最灵活的，你可以做任意复杂的数据转换逻辑。但它也有一个缺点：你需要手写映射代码，当字段很多时容易出错。另一种思路是在设计阶段就让子图和父图共享一部分状态字段，减少映射的工作量。实际项目中常见的做法是定义一个"基础状态类"包含公共字段，然后父图和子图分别继承并扩展自己特有的字段。

## 嵌套子图：子图中再嵌套子图

既然子图可以被当作节点使用，那自然也就可以在子图里面再嵌套子图——LangGraph 对嵌套层级没有硬性限制。这种能力在构建多层抽象的系统时特别有用。想象一下一个订单处理系统：顶层是"订单生命周期管理"图，其中包含"支付处理"子图，而"支付处理"子图里面又包含了"风控检查"子图和"金额计算"子图。

```python
class PaymentState(TypedDict):
    order_id: str
    amount: float
    currency: str
    payment_method: str
    risk_level: str
    fraud_score: float
    fee: float
    final_amount: float
    payment_status: str

def calculate_fee(state: PaymentState) -> dict:
    amount = state["amount"]
    method = state["payment_method"]
    if method == "credit_card":
        fee = amount * 0.029 + 0.30
    elif method == "paypal":
        fee = amount * 0.039
    else:
        fee = amount * 0.015
    return {"fee": round(fee, 2), "final_amount": round(amount + fee, 2)}

def process_payment(state: PaymentState) -> dict:
    final = state["final_amount"]
    risk = state["risk_level"]
    if risk == "high":
        status = "pending_manual_review"
    else:
        status = "completed"
    return {"payment_status": status}

payment_graph = StateGraph(PaymentState)
payment_graph.add_node("risk_assessment", compiled_risk)
payment_graph.add_node("calculate_fee", calculate_fee)
payment_graph.add_node("process_payment", process_payment)
payment_graph.add_edge(START, "risk_assessment")
payment_graph.add_edge("risk_assessment", "calculate_fee")
payment_graph.add_edge("calculate_fee", "process_payment")
payment_graph.add_edge("process_payment, END")

compiled_payment = payment_graph.compile()
```

这里 `compiled_risk`（风控子图）被嵌套在了 `compiled_payment`（支付处理子图）内部。当我们运行顶层图时，调用链路是这样的：顶层图 → 支付子图 → 风控子图（内部3个节点）→ 回到支付子图继续执行 → 回到顶层图。每一层都维护着自己的状态边界，层与层之间通过预定义的接口交换数据。

## 子图的独立测试能力

子图最大的工程价值之一就是**可独立测试**。因为子图本身就是一个完整的 CompiledGraph，你不需要启动整个系统就能对它进行端到端的测试。这对持续集成和快速迭代来说非常重要——当你修改了某个子图的内部逻辑后，只需要针对这个子图跑单元测试，而不需要担心影响到其他部分。

```python
import pytest

def test_risk_subgraph_low_risk():
    result = compiled_risk.invoke({
        "transaction_id": "ORD-001",
        "buyer_name": "张三",
        "amount": 100.0,
        "item_count": 2,
        "risk_flags": [],
        "calculated_risk": "low",
        "risk_score": 0.0
    })
    assert result["calculated_risk"] == "low"
    assert result["risk_score"] == 0

def test_risk_subgraph_high_risk():
    result = compiled_risk.invoke({
        "transaction_id": "ORD-002",
        "buyer_name": "李四",
        "amount": 80000.0,
        "item_count": 30,
        "risk_flags": [],
        "calculated_risk": "low",
        "risk_score": 0.0
    })
    assert result["calculated_risk"] == "high"
    assert result["risk_score"] >= 80

def test_validation_subgraph_empty_data():
    result = compiled_validation.invoke({
        "raw_data": "",
        "cleaned_data": "",
        "validation_errors": [],
        "is_valid": False,
        "validation_score": 0.0
    })
    assert result["is_valid"] == False
    assert len(result["validation_errors"]) >= 1
```

这种测试方式和测试普通函数几乎没有区别，但覆盖的是一整套多节点的业务流程。在实际项目中，建议为每个子图编写至少三类测试用例：正常路径（happy path）、边界条件（空数据/极大值/特殊字符）、以及错误路径（预期会触发错误标记的输入）。这三类测试能覆盖绝大多数的回归场景。

## 动态子图选择：根据状态决定使用哪个子图

还有一种更高级的用法：不在编译期固定死使用哪个子图，而是在运行时根据当前状态动态选择要执行的子图。这在"策略模式"场景下非常有用——比如同一个"数据处理"节点，根据数据类型的不同，可能需要走文本处理子图、图像处理子图或结构化数据处理子图。

```python
from typing import TypedDict, Annotated, Union
import operator
from langgraph.graph import StateGraph, START, END

class DataProcessingState(TypedDict):
    input_data: str
    data_type: str
    processing_result: str
    metadata: dict

text_processor = StateGraph(TypedDict(input_data=str, result=str, word_count=int))
def process_text(s): return {"result": s.upper(), "word_count": len(s.split())}
text_processor.add_node("process", process_text)
text_processor.add_edge(START, "process")
text_processor.add_edge("process", END)
compiled_text = text_processor.compile()

number_processor = StateGraph(TypedDict(input_data=str, result=str, total=float))
def process_numbers(s):
    nums = [float(x) for x in s.split() if x.replace('.','').isdigit()]
    return {"result": f"求和={sum(nums)}, 均值={sum(nums)/len(nums) if nums else 0}", "total": sum(nums)}
number_processor.add_node("process", process_numbers)
number_processor.add_edge(START, "process")
number_processor.add_edge("process", END)
compiled_number = number_processor.compile()

def dynamic_router(state: DataProcessingState) -> dict:
    dtype = state["data_type"]
    data = state["input_data"]
    if dtype == "text":
        sub_result = compiled_text.invoke({"input_data": data, "result": "", "word_count": 0})
        return {
            "processing_result": sub_result["result"],
            "metadata": {"processor": "text", "word_count": sub_result["word_count"]}
        }
    elif dtype == "number":
        sub_result = compiled_number.invoke({"input_data": data, "result": "", "total": 0.0})
        return {
            "processing_result": sub_result["result"],
            "metadata": {"processor": "number", "total": sub_result["total"]}
        }
    else:
        return {"processing_result": f"[未知类型] {data}", "metadata": {"processor": "passthrough"}}

router_graph = StateGraph(DataProcessingState)
router_graph.add_node("dynamic_process", dynamic_router)
router_graph.add_edge(START, "dynamic_process")
router_graph.add_edge("dynamic_process", END)
app = router_graph.compile()
```

在这个例子中，`dynamic_router` 节点内部根据 `state["data_type"]` 的值选择调用不同的子图。虽然从 LangGraph 的角度看这只是一个普通的节点函数，但从架构设计的角度看，它实现了**策略模式**的动态分发——新增一种数据类型只需要增加一个新的子图和在路由函数里加一个分支即可，不需要修改主图的结构。

## 子图的 Checkpointing 隔离性

当父图配置了 Checkpointer（比如 MemorySaver 或 PostgresSaver）时，子图的执行过程是否也会被记录下来呢？答案是**会的，而且是以嵌套的方式记录**。这意味着你可以在 LangSmith 的 trace 视图中看到子图内部的每一个节点执行详情，也可以在做时间旅行调试时精确地回退到子图内部的某一步。

但这种嵌套记录也带来一个需要注意的点：**子图的 checkpoint key 是基于父图的 thread_id 加上子图自身的执行上下文生成的**。也就是说，如果你在不同的父图执行中使用相同的 thread_id，它们各自的子图执行记录是相互隔离的，不会互相干扰。这种隔离性保证了并发执行时的安全性。

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
parent_app = main_graph.compile(checkpointer=checkpointer)

config1 = {"configurable": {"thread_id": "session-001"}}
config2 = {"configurable": {"thread_id": "session-002"}}

result1 = parent_app.invoke(initial_state, config=config1)
result2 = parent_app.invoke(initial_state, config=config2)

checkpoint1 = checkpointer.get(config1)
checkpoint2 = checkpointer.get(config2)
# 两个 checkpoint 完全独立，互不影响
```

## 常见误用与陷阱

在使用子图的过程中，有几个容易踩坑的地方值得特别注意。第一个常见问题是**循环依赖**——如果图 A 嵌套了图 B，而图 B 又反过来嵌套了图 A，就会形成无限递归。LangGraph 在编译阶段并不会检测这种跨图的循环依赖（因为它发生在运行时的函数调用中），所以只有在实际执行时才会触发 `RecursionError` 或栈溢出。避免的方法很简单：保持依赖关系的单向性，用 DAG（有向无环图）的方式来组织子图之间的嵌套关系。

第二个问题是**状态字段冲突**。当父图和子图有同名字段但类型不同时（比如父图的 `items` 是 `list[dict]` 而子图的 `items` 是 `list[str]`），自动映射可能会导致静默的类型错误或数据丢失。解决方案是要么彻底区分字段名，要么在包装函数中显式地做类型转换。

第三个容易被忽视的问题是**子图的错误传播**。默认情况下，如果子图内部某个节点抛出了未捕获的异常，这个异常会原样向上冒泡到父图，导致整个父图执行中断。如果你希望子图内部的失败不要影响父图的其余流程，需要在子图内部做好错误处理——通常的做法是在子图的每个节点函数外面套一层 try-except，把异常转化为状态中的错误信息字段：

```python
def safe_format_checker(state: ValidationState) -> dict:
    try:
        data = state["cleaned_data"]
        errors = []
        if len(data) < 10:
            errors.append("数据长度不足10个字符")
        if not any(c.isalpha() for c in data):
            errors.append("数据不包含字母")
        return {"validation_errors": errors}
    except Exception as e:
        return {"validation_errors": [f"格式检查器异常: {str(e)}"]]}
```

第四个问题是关于**子图的冷启动开销**。每次父图执行到子图节点时，LangGraph 都需要初始化子图的执行上下文，包括解析子图的结构、准备状态快照等。如果你的子图非常小（只有1-2个节点），把它抽成子图反而可能引入不必要的性能开销。一个经验法则是：如果一个逻辑块包含 3 个或以上的节点，并且可能在多处被复用，那么抽取为子图才是划算的。

最后还要提一点：**子图不能直接访问父图的全部状态**。这是设计上的有意为之——子图只能看到通过映射传进来的字段，这种信息隐藏保证了模块间的低耦合。如果你发现自己在子图里需要越来越多地访问父图的字段，那可能说明你的状态划分不够合理，应该重新审视一下哪些数据应该是子图的私有状态、哪些应该是跨层共享的。
