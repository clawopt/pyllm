# 6.1 子图架构设计：从可复用组件到完整系统

> 在第2章中我们已经了解了子图的基本概念——把一个 StateGraph 编译后当作节点添加到另一个图中使用。但那只是最简单的用法。在实际的大型系统中，子图不仅仅是"被复用的代码块"，它更是一种**架构分层和组织策略**。通过合理地划分子图，你可以把一个包含几十个节点的复杂系统分解为多个职责清晰、边界明确、独立可测试的模块。这一节我们会从架构设计的角度深入探讨如何规划子图的划分、接口定义和组合方式。

## 为什么需要子图架构

在讨论具体的设计方法之前，先理解为什么随着系统复杂度的增长，子图架构变得不可或缺。

想象一下你正在构建一个完整的客服工单处理系统，它包含以下所有功能：接收用户消息 → 意图分类 → 知识库检索 → LLM 生成回复 → 敏感词过滤 → 质量评分 → 存储到数据库 → 发送通知 → 更新统计指标。如果把这所有的功能都放在一张平铺的图中，你最终会得到一个有 15-20 个节点的巨型图。这样的图会面临几个严重问题：

**第一，认知负担过重**。没有任何人能够一眼看懂一个 20 节点的图的全部逻辑——每个节点做什么、它们之间的依赖关系是什么、数据是如何流转的，这些信息散落在整个图的定义中，需要花费大量时间才能理清。

**第二，修改风险高**。当你需要修改其中一个功能（比如换一种意图分类方式）时，很难预测这个修改会不会影响到其他不相关的部分。因为所有节点共享同一个状态空间，一个字段的变更可能波及到很多地方。

**第三，无法独立测试**。如果你想测试"意图分类→知识库检索→生成回复"这条链路是否正常工作，你必须启动整个系统（包括敏感词过滤、质量评分等完全不相关的部分），这大大降低了测试效率和反馈速度。

**第四，团队协作困难**。如果多人同时在这个巨型图上工作，冲突几乎是不可避免的——两个人可能同时修改同一个状态字段或同一条边。

子图架构正是为了解决这些问题而生的。它让你能够按照功能域或业务流程来组织代码，每个子图是一个独立的、自包含的单元，有自己的状态定义和内部逻辑。父图只负责协调各子图之间的交互，而不关心它们的内部实现细节。

## 子图划分的三大原则

好的子图划分应该遵循三个核心原则：**单一职责**、**高内聚低耦合**、**接口稳定**。

**原则一：单一职责**。每个子图应该只负责一件事。这件事可以很复杂（比如"处理一次完整的对话"），但它的范围应该是清晰的、可以用一句话描述的。如果一个子图的名字需要用"和"、"以及"、"还有"来连接多个动词，那就说明它承担了太多职责，应该拆分。

```python
# ❌ 违反单一职责：一个子图做了太多事情
class GodGraphState(TypedDict):
    user_message: str
    intent: str
    kb_results: list
    reply: str
    filtered_reply: str
    quality_score: int
    db_record_id: str
    notification_sent: bool
    stats_updated: bool

god_graph = StateGraph(GodGraphState)
# ... 15个节点 ...

# ✅ 遵循单一职责：每个子图做一件事
class ClassificationState(TypedDict):
    user_message: str
    intent: str
    confidence: float

classification_graph = StateGraph(ClassificationState)
# 只做意图分类

class RetrievalState(TypedDict):
    intent: str
    query: str
    results: list[dict]

retrieval_graph = StateGraph(RetrievalState)
# 只做知识库检索

class GenerationState(TypedDict):
    context: list[dict]
    reply: str

generation_graph = StateGraph(GenerationState)
# 只做回复生成
```

**原则二：高内聚低耦合**。内聚指的是子图内部的节点之间应该紧密相关——它们共同完成同一个任务，彼此之间有大量的数据交换。耦合指的是不同子图之间应该尽量减少直接的依赖——它们之间通过明确定义的接口来通信，而不是直接访问对方的内部状态。

```python
# ❌ 低内聚高耦合：子图内部松散，子图之间紧耦合
class LooseSubgraphState(TypedDict):
    raw_input: str
    temp_var_a: str      # 只被 node1 使用
    temp_var_b: str      # 只被 node2 使用
    shared_with_other: dict  # 直接暴露给其他子图

# ✅ 高内聚低耦合：子图内部紧密，子图之间通过接口通信
class CohesiveClassificationState(TypedDict):
    input_text: str       # 输入接口
    output_intent: str    # 输出接口
    output_confidence: float  # 输出接口
    _internal_context: str     # 内部状态，不对外暴露
```

**原则三：接口稳定**。一旦子图的输入输出接口确定下来，就应该尽量避免频繁改动。因为接口的变化会影响所有使用这个子图的父图和其他调用者。内部实现可以随时重构（比如换一种算法、增加中间步骤），但接口要保持稳定。这是软件工程中经典的"面向接口编程"思想在 LangGraph 中的体现。

## 实战案例：电商订单处理的子图架构

让我们用一个具体的例子来展示子图架构设计的完整过程。假设我们要构建一个电商订单处理系统，从用户下单到完成交付的全过程。

### 第一步：识别功能域

首先分析业务流程，识别出相对独立的功能域：

1. **订单验证**：检查商品库存、验证价格、校验优惠券
2. **支付处理**：调用支付网关、确认扣款成功
3. **物流调度**：分配仓库、生成运单、通知快递公司
4. **通知推送**：发送确认短信/邮件、更新 App 推送
5. **售后保障**：自动确认收货、触发评价邀请、处理退款申请

这五个功能域就是五个候选子图。每个功能域内部可能有多个步骤，但它们都服务于同一个目标。

### 第二步：定义每个子图的接口

为每个子图定义清晰的输入输出接口：

```python
from typing import TypedDict

class OrderValidationInput(TypedDict):
    order_id: str
    user_id: str
    items: list[dict]
    coupon_code: str | None
    shipping_address: dict

class OrderValidationOutput(TypedDict):
    is_valid: bool
    validated_items: list[dict]
    final_price: float
    discount_amount: float
    validation_errors: list[str]

class PaymentInput(TypedDict):
    order_id: str
    amount: float
    payment_method: str
    user_id: str

class PaymentOutput(TypedDict):
    payment_success: bool
    transaction_id: str
    payment_time: str
    error_code: str | None

class LogisticsInput(TypedDict):
    order_id: str
    items: list[dict]
    shipping_address: dict
    warehouse_preference: str | None

class LogisticsOutput(TypedDict):
    tracking_number: str
    estimated_delivery: str
    carrier_name: str
    dispatch_status: str

class NotificationInput(TypedDict):
    order_id: str
    user_id: str
    notification_type: str
    template_data: dict

class NotificationOutput(TypedDict):
    sent_successfully: bool
    sent_channels: list[str]
    sent_timestamp: str
```

注意这里的一个关键设计决策：**每个子图都有独立的 Input 和 Output 类型**，而不是共享一个全局的状态类型。这样做的好处是每个子图的接口是显式的、自文档化的——你不需要阅读子图内部的代码就能知道它需要什么输入、会产生什么输出。

### 第三步：实现各个子图

接下来实现每个子图的内部逻辑。以订单验证子图为例：

```python
from langgraph.graph import StateGraph, START, END

class ValidationInternalState(OrderValidationInput, OrderValidationOutput):
    stock_checked: bool
    price_verified: bool
    coupon_applied: bool
    validation_log: list[str]

def check_stock(state: ValidationInternalState) -> dict:
    items = state["items"]
    validated = []
    for item in items:
        available = check_inventory(item["product_id"], item["quantity"])
        if available:
            validated.append({**item, "in_stock": True})
        else:
            validated.append({**item, "in_stock": False})

    return {
        "stock_checked": True,
        "validated_items": validated,
        "validation_log": [f"[库存] 检查 {len(items)} 个商品"]
    }

def verify_price(state: ValidationInternalState) -> dict:
    items = state.get("validated_items", [])
    total = 0
    for item in items:
        if item.get("in_stock"):
            current_price = get_current_price(item["product_id"])
            if abs(current_price - item["unit_price"]) < 0.01:
                total += current_price * item["quantity"]
            else:
                return {
                    "is_valid": False,
                    "validation_errors": [f"价格变动: {item['product_id']}"],
                    "validation_log": [f"[价格] ⚠️ 价格不一致"]
                }
    return {"price_verified": True, "final_price": total}

def apply_coupon(state: ValidationInternalState) -> dict:
    code = state.get("coupon_code")
    price = state.get("final_price", 0)

    if not code:
        return {"coupon_applied": True, "discount_amount": 0}

    discount = validate_and_calculate_discount(code, price)
    return {
        "coupon_applied": True,
        "discount_amount": discount,
        "final_price": price - discount,
        "validation_log": [f"[优惠] 优惠券抵扣 ¥{discount:.2f}"]
    }

def finalize_validation(state: ValidationInternalState) -> dict:
    errors = state.get("validation_errors", [])
    valid = len(errors) == 0 and state.get("price_verified", False)

    return {
        "is_valid": valid,
        "validation_log": [f"[完成] {'✅ 验证通过' if valid else '❌ 验证失败'}"]
    }

validation_graph = StateGraph(ValidationInternalState)
validation_graph.add_node("check_stock", check_stock)
validation_graph.add_node("verify_price", verify_price)
validation_graph.add_node("apply_coupon", apply_coupon)
validation_graph.add_node("finalize", finalize_validation)

validation_graph.add_edge(START, "check_stock")
validation_graph.add_edge("check_stock", "verify_price")
validation_graph.add_conditional_edges("verify_price",
    lambda s: "apply_coupon" if s.get("price_verified") else "finalize",
    {"apply_coupon": "apply_coupon", "finalize": "finalize"}
)
validation_graph.add_edge("apply_coupon", "finalize")
validation_graph.add_edge("finalize", END)

compiled_validation = validation_graph.compile()
```

这个订单验证子图包含了四个内部节点，它们共同完成了订单验证的全部工作。外部调用者只需要关心 `OrderValidationInput` 和 `OrderValidationOutput` 中定义的字段，不需要了解 `stock_checked`、`price_verified` 这些内部状态的存在。

### 第四步：组装父图

最后，在父图中把各个子图组装起来：

```python
class OrderProcessingState(TypedDict):
    order_input: dict
    validation_result: OrderValidationOutput
    payment_result: PaymentOutput
    logistics_result: LogisticsOutput
    notification_result: NotificationOutput
    order_status: str
    processing_log: list[str]

def run_validation(state: OrderProcessingState) -> dict:
    inp = state["order_input"]
    result = compiled_validation.invoke({
        **inp,
        "is_valid": False, "validated_items": [], "final_price": 0,
        "discount_amount": 0, "validation_errors": [],
        "stock_checked": False, "price_verified": False,
        "coupon_applied": False, "validation_log": []
    })
    return {
        "validation_result": result,
        "processing_log": result.get("validation_log", [])
    }

def route_after_validation(state: OrderProcessingState) -> str:
    if not state["validation_result"].get("is_valid"):
        return "handle_rejection"
    return "run_payment"

def run_payment(state: OrderProcessingState) -> dict:
    val_result = state["validation_result"]
    payment_inp = {
        "order_id": state["order_input"]["order_id"],
        "amount": val_result["final_price"],
        "payment_method": state["order_input"].get("payment_method", "alipay"),
        "user_id": state["order_input"]["user_id"]
    }
    result = compiled_payment.invoke(payment_inp)
    return {"payment_result": result}

def run_logistics(state: OrderProcessingState) -> dict:
    log_inp = {
        "order_id": state["order_input"]["order_id"],
        "items": state["validation_result"]["validated_items"],
        "shipping_address": state["order_input"]["shipping_address"],
        "warehouse_preference": None
    }
    result = compiled_logistics.invoke(log_inp)
    return {"logistics_result": result}

def send_notifications(state: OrderProcessingState) -> dict:
    notif_inp = {
        "order_id": state["order_input"]["order_id"],
        "user_id": state["order_input"]["user_id"],
        "notification_type": "order_confirmed",
        "template_data": {
            "tracking": state["logistics_result"].get("tracking_number"),
            "eta": state["logistics_result"].get("estimated_delivery")
        }
    }
    result = compiled_notification.invoke(notif_inp)
    return {"notification_result": result}

def handle_rejection(state: OrderProcessingState) -> dict:
    errors = state["validation_result"].get("validation_errors", [])
    return {
        "order_status": "rejected",
        "processing_log": [f"[拒绝] 原因: {'; '.join(errors)}"]
    }

def mark_completed(state: OrderProcessingState) -> dict:
    return {
        "order_status": "completed",
        "processing_log": ["[完成] 订单处理完毕"]
    }

parent_graph = StateGraph(OrderProcessingState)
parent_graph.add_node("validate", run_validation)
parent_graph.add_node("pay", run_payment)
parent_graph.add_node("ship", run_logistics)
parent_graph.add_node("notify", send_notifications)
parent_graph.add_node("reject", handle_rejection)
parent_graph.add_node("complete", mark_completed)

parent_graph.add_edge(START, "validate")
parent_graph.add_conditional_edges("validate", route_after_validation, {
    "run_payment": "pay",
    "handle_rejection": "reject"
})
parent_graph.add_edge("pay", "ship")
parent_graph.add_edge("ship", "notify")
parent_graph.add_edge("notify", "complete")
parent_graph.add_edge("reject", END)
parent_graph.add_edge("complete", END)

app = parent_graph.compile()
```

这个父图的结构非常清晰：它只有 7 个节点（其中 4 个是对应子图的包装节点），边的走向一目了然——验证→支付→物流→通知→完成，或者验证→拒绝。每个子图的内部复杂性被完全封装起来了，父图的维护者只需要关注子图之间的协调逻辑。

## 子图 vs 函数：什么时候该用子图

一个常见的困惑是：既然子图本质上也是被当作节点调用的，为什么不直接用一个普通函数来实现同样的功能？确实，对于非常简单的逻辑（比如一个纯计算函数），用普通函数就足够了。但在以下几种情况下，子图明显优于函数：

**情况一：内部有多步顺序执行**。如果你的逻辑包含 3 个及以上按固定顺序执行的步骤，而且这些步骤之间有状态传递，用子图来表达这种顺序关系比在一个大函数里依次调用要清晰得多。

**情况二：内部有条件分支**。如果你的逻辑根据某些条件走不同的分支路径，子图的条件边能清晰地表达这些分支关系，而函数内部的 if-else 嵌套会让代码难以阅读。

**情况三：需要独立测试**。如果你希望对某个功能模块进行端到端的测试（而不是逐函数测试），子图可以作为一个独立的可编译单元进行测试。

**情况四：可能在多处被复用**。如果同一个逻辑需要在不同的父图中被引用，把它封装为子图比复制粘贴函数要好得多。

**情况五：需要独立配置 checkpointing**。如果某个模块需要自己的持久化策略（比如不同的保存频率、不同的存储后端），子图可以独立配置 checkpointer。

作为一条经验法则：**当一个逻辑块包含 3 个及以上节点，或者可能在多处被复用时，就值得抽取为子图**。
