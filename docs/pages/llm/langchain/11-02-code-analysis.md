---
title: 实现代码解释、Bug 定位与单元测试生成
description: 三大核心能力：代码自然语言解释、智能 Bug 检测与定位、自动生成单元测试、Prompt 工程与输出解析
---
# 实现代码解释、Bug 定位与单元测试生成

上一节我们构建了代码知识库——能够把一个 Git 仓库中的 Python 代码按函数/类级别切分、增强上下文、向量化存入 Chroma。现在这些"积木"要派上用场了。

本章实现代码分析助手的**三大核心能力**：**代码解释**（把代码翻译成人话）、**Bug 定位**（找出代码中的问题）、**测试生成**（自动写出单元测试）。这三个能力覆盖了开发者日常最频繁的代码理解需求。

## 能力一：代码解释——让 LLM 当你的"结对编程伙伴"

### 为什么需要代码解释

你接手了一个新项目，打开一个 300 行的 `payment_processor.py`，里面有一堆业务逻辑。你想知道"这个退款函数的完整流程是什么？"传统做法是逐行阅读，在脑子里模拟执行。但如果有一个人坐在你旁边，你指着代码问"这里做了什么？"他立刻用自然语言解释给你听——这就是代码解释工具要做的事。

### 基础版：单函数解释

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

EXPLAIN_SYSTEM = """你是一个资深软件工程师，擅长用简洁清晰的中文解释代码。
用户会提供一段代码和相关的上下文信息，你需要：
1. 用一段话概括这段代码的功能（一句话）
2. 分步骤解释核心逻辑（3-5 步）
3. 指出关键的设计决策或值得注意的地方
4. 如果有潜在问题或改进建议，简要提及

要求：
- 使用中文，面向有基础的开发者
- 不要逐行翻译代码，而是解释"为什么这样做"
- 保持技术准确性"""

EXPLAIN_PROMPT = ChatPromptTemplate.from_messages([
    ("system", EXPLAIN_SYSTEM),
    ("human", "请解释以下代码：\n\n{code}\n\n【相关上下文】\n{context}"),
])

explain_chain = EXPLAIN_PROMPT | get_llm() | StrOutputParser()
```

使用示例：

```python
code_snippet = """def process_refund(self, order_id: str, reason: str) -> RefundResult:
    order = self.db.get_order(order_id)
    if not order or order.status != "completed":
        return RefundResult(success=False, error="订单不存在或未完成")

    if datetime.now() - order.completed_at > timedelta(days=30):
        return RefundResult(success=False, error="已超过30天退款期限")

    refund_amount = order.total * 0.9
    if self.payment_gateway.refund(order.payment_id, refund_amount):
        order.status = "refunded"
        self.db.update_order(order)
        self.notify_user(order.user_id, f"退款 ¥{refund_amount:.2f} 已处理")
        return RefundResult(success=True, amount=refund_amount)
    else:
        self.alert_team(f"退款失败: {order_id}")
        return RefundResult(success=False, error="退款处理失败")"""

context_info = "这是 PaymentProcessor 类的方法，用于处理电商订单退款。RefundResult 是返回值数据类。"

result = explain_chain.invoke({
    "code": code_snippet,
    "context": context_info,
})

print(result)
```

输出：

```
## 功能概述
`process_refund` 实现了带业务规则校验的订单退款流程：验证订单有效性 → 检查退款时限 →
计算退款金额（扣除10%手续费）→ 调用支付网关执行退款 → 更新订单状态 → 通知用户。

## 核心逻辑拆解

1. **订单有效性校验**（第2-3行）：先查数据库确认订单存在且状态为已完成（`completed`），
   防止对未完成或已退款的订单重复操作。

2. **退款时效检查**（第5-6行）：限制在订单完成后 30 天内才能申请退款，
   这是一个常见的风控策略，防止长期后的争议性退款。

3. **退款金额计算**（第8行）：按订单总额的 90% 退款，即扣除 10% 手续费。
   这个比例硬编码在逻辑中——如果未来需要根据会员等级动态调整手续费率，
   这里需要重构为可配置的策略模式。

4. **原子性退款操作**（第9-13行）：调用支付网关 API 执行实际退款，
   成功后立即更新本地数据库状态并发送用户通知。
   注意这里没有事务包裹——如果 `notify_user` 失败，订单状态已经改为 refunded 但用户没收到通知，
   存在轻微的数据一致性问题。

5. **异常降级**（第14-16行）：支付网关调用失败时不会抛异常，
   而是记录告警并返回错误结果，保证了接口的健壮性。

## 值得注意的设计点
- 采用 **Early Return 模式**：每个校验不通过都直接返回错误，避免深层嵌套
- 手续费比例（0.9）是 **Magic Number**，建议提取为常量或配置项
- `notify_user` 与数据库更新之间缺少 **事务一致性保障**
```

这个解释的质量相当高：它不是简单地把 `if` 翻译成"如果"，而是解释了**每一步的业务含义**和**设计取舍**，甚至还指出了潜在的改进点。

### 进阶版：结合 RAG 的全项目级解释

单个函数的解释只是起点。真正的价值在于：**用户问一个高层问题时，系统能从整个项目中检索相关代码并给出综合解释**：

```python
PROJECT_EXPLAIN_SYSTEM = EXPLAIN_SYSTEM + """
你现在面对的是一个完整的代码仓库。系统会从代码库中检索与用户问题相关的代码片段。
请综合所有检索到的代码片段，给出全面而连贯的解释。
如果不同片段之间存在调用关系或数据流转，请明确指出。"""

project_explain_prompt = ChatPromptTemplate.from_messages([
    ("system", PROJECT_EXPLAIN_SYSTEM),
    ("human", """用户问题: {question}

【从代码库中检索到的相关代码】

{retrieved_code}

请基于以上代码片段回答用户的问题。"""),
])

project_explain_chain = (
    {
        "question": lambda x: x["question"],
        "retrieved_code": retriever | (lambda docs: "\n\n---\n\n".join(
            f"文件: {d.metadata.get('source', '?')} "
            f"(L{d.metadata.get('start_line', '?')}-{d.metadata.get('end_line', '?')})\n```python\n{d.page_content}\n```"
            for d in docs
        )),
    }
    | project_explain_prompt
    | get_llm()
    | StrOutputParser()
)
```

测试一下跨文件的复杂问题：

```python
result = project_explain_chain.invoke({
    "question": "用户从登录到完成一次支付的完整流程是怎样的？涉及哪些模块？"
})
print(result)
```

输出会综合 `auth_service.py`（认证）、`api_routes.py`（路由）、`payment_processor.py`（支付）等多个文件的内容，画出一条完整的数据流链路。这种**跨文件的全局视角**正是 RAG + LLM 结合的最大优势——人类阅读代码时很难同时在脑子里装下十几个文件，但 LLM 可以。

## 能力二：Bug 定位——AI 代码审查员

### 从"解释代码"到"找 Bug"

代码解释是被动的（你说什么我解释什么），Bug 定位是主动的（我来帮你审查代码有没有问题）。这需要 LLM 不仅理解代码的语义，还要能识别**常见的安全漏洞、性能陷阱、逻辑错误和反模式**。

### Bug 检测 Prompt 设计

```python
BUG_FINDER_SYSTEM = """你是一个严格的代码审查专家，擅长发现代码中的各类问题。
分析给定的代码片段，按照以下格式输出发现的问题：

对于每个问题，必须包含：
1. **严重程度**: 🔴 Critical / 🟠 High / 🟡 Medium / 🔵 Low / ⚪ Info
2. **问题类型**: Security / Logic / Performance / Reliability / Style / Best Practice
3. **位置描述**: 具体到行号范围或代码片段
4. **问题描述**: 清楚说明问题是什么
5. **修复建议**: 给出具体的修复代码或方案

注意：
- 不要误报！只有确定有问题才报告
- 区分"确实的 bug"和"可以改进的地方"
- 如果代码质量很好，就如实说"未发现明显问题"
- 用中文回答"""

BUG_FINDER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", BUG_FINDER_SYSTEM),
    ("human", "请审查以下代码：\n\n```python\n{code}\n```\n\n上下文: {context}"),
])

bug_finder_chain = BUG_FINDER_PROMPT | get_llm() | StrOutputParser()
```

### 测试：注入一段有问题的代码

```python
buggy_code = """class UserAPI:
    def __init__(self):
        self.conn = sqlite3.connect("users.db")

    def get_user(self, user_id: str):
        query = f"SELECT * FROM users WHERE id = {user_id}"
        cursor = self.conn.execute(query)
        return dict(cursor.fetchone())

    def login(self, username, password):
        user = self.db.find_one({"name": username, "pwd": password})
        if user:
            token = jwt.encode({"user_id": user["id"]}, "secret-key")
            return {"token": token}
        return None

    def reset_password(self, email: str, new_pwd: str):
        import smtplib
        server = smtplib.SMTP("smtp.example.com", 587)
        server.sendmail(
            from_addr="noreply@example.com",
            to_addrs=email,
            msg=f"您的新密码是: {new_pwd}"
        )
        server.quit()
        return True"""

result = bug_finder_chain.invoke({
    "code": buggy_code,
    "context": "UserAPI 类处理用户认证相关操作",
})
print(result)
```

输出：

```
## 代码审查报告 — UserAPI 类

共发现 **6 个问题**:

---

### 1. 🔴 SQL 注入漏洞 (Security)

**位置**: `get_user()` 方法第 3 行
```python
query = f"SELECT * FROM users WHERE id = {user_id}"
```

**问题**: 直接将 `user_id` 拼接到 SQL 字符串中，攻击者可以通过传入
`1 OR 1=1` 来获取全部用户数据。这是经典的 SQL 注入漏洞。

**修复建议**:
```python
cursor = self.conn.execute("SELECT * FROM users WHERE id = ?", (user_id,))
```

---

### 2. 🔴 JWT 密钥硬编码 (Security)

**位置**: `login()` 方法第 11 行
```python
token = jwt.encode({"user_id": user["id"]}, "secret-key")
```

**问题**: JWT 签名密钥直接写在源码中。如果代码泄露到公开仓库，
任何人都可以伪造合法 token。应使用环境变量或密钥管理服务存储。

**修复建议**:
```python
secret = os.environ.get("JWT_SECRET_KEY")
token = jwt.encode({"user_id": user["id"]}, secret)
```

---

### 3. 🟠 明文密码比较 (Security)

**位置**: `login()` 方法第 9 行
```python
user = self.db.find_one({"name": username, "pwd": password})
```

**问题**: 数据库中存储的是明文密码，并且直接做字符串比较。
即使数据库未被攻破，运维人员也能看到所有用户的原始密码。

**修复建议**: 存储 bcrypt 哈希值，验证时使用 `bcrypt.checkpw()`。

---

### 4. 🟡 密码通过邮件明文传输 (Reliability)

**位置**: `reset_password()` 方法
```python
msg=f"您的新密码是: {new_pwd}"
```

**问题**: 新密码以纯文本形式通过邮件发送。邮件传输过程中可能被中间人截获，
而且邮件服务器上会保留明文副本。

**修复建议**: 发送密码重置链接（含临时 token），让用户自行设置新密码。

---

### 5. 🟡 SMTP 连接无异常处理 (Reliability)

**位置**: `reset_password()` 方法

**问题**: `smtplib.SMTP()` 和后续的 `sendmail()` 都没有 try-except 包裹。
如果邮件服务器不可用，会导致整个请求崩溃返回 500 错误。

**修复建议**: 用 try-finally 确保 `server.quit()` 总是执行。

---

### 6. 🔵 缺少类型注解 / 类型不一致 (Style)

**位置**: 多处

**问题**: 构造函数中使用了 `self.conn`（SQLite），但 `login()` 中用了 `self.db`
（看起来像 MongoDB）。同一个类混用了两种数据库连接方式，可能是复制粘贴遗留的 bug。

---

## 总结
严重程度分布: 🔴×2 🟠×1 🟡×2 🔵×1
**首要修复优先级**: 先解决两个 🔴 安全漏洞（SQL 注入 + JWT 密钥）
```

这份审查报告的质量接近中级工程师的水平——不仅找到了问题，还给出了**具体的修复代码**和**优先级排序**。特别是第 6 个问题（`self.conn` vs `self.db` 的不一致）展示了 LLM 的**跨函数上下文推理能力**——这不是简单的模式匹配能做到的。

## 能力三：单元测试生成——让 AI 写测试

写单元测试是开发者的痛点之一：**都知道应该写，但总是没时间写**。LLM 在这方面有天然优势——它同时理解被测代码的逻辑和主流测试框架的用法，可以快速生成高质量的测试骨架。

### 测试生成 Prompt

```python
TEST_GEN_SYSTEM = """你是一个测试工程专家。根据提供的源代码，生成完整的 Python 单元测试。

要求：
1. 使用 pytest 框架
2. 覆盖正常路径和主要边界情况
3. 使用 pytest 的 fixture 和 parametrize 特性
4. 对外部依赖（数据库、API、文件系统）使用 mock
5. 每个测试函数要有清晰的 docstring 说明测试目的
6. 只输出测试代码，不要其他解释文字"""

TEST_GEN_PROMPT = ChatPromptTemplate.from_messages([
    ("system", TEST_GEN_SYSTEM),
    ("human", """请为以下代码生成单元测试：

```python
{code}
```

上下文信息: {context}"""),
])

test_gen_chain = TEST_GEN_PROMPT | get_llm() | StrOutputParser()
```

### 测试生成效果

```python
source_code = """class DiscountCalculator:
    def __init__(self):
        self.tiers = [
            {"min_purchase": 0, "max_purchase": 100, "rate": 0},
            {"min_purchase": 100, "max_purchase": 500, "rate": 0.05},
            {"min_purchase": 500, "max_purchase": 2000, "rate": 0.10},
            {"min_purchase": 2000, "max_purchase": float("inf"), "rate": 0.15},
        ]

    def calculate(self, purchase_amount: float) -> dict:
        if purchase_amount < 0:
            raise ValueError("购买金额不能为负数")

        for tier in self.tiers:
            if tier["min_purchase"] <= purchase_amount < tier["max_purchase"]:
                discount = purchase_amount * tier["rate"]
                final_amount = purchase_amount - discount
                return {
                    "original": purchase_amount,
                    "discount_rate": tier["rate"],
                    "discount_amount": round(discount, 2),
                    "final_amount": round(final_amount, 2),
                }

        return {
            "original": purchase_amount,
            "discount_rate": 0.15,
            "discount_amount": round(purchase_amount * 0.15, 2),
            "final_amount": round(purchase_amount * 0.85, 2),
        }"""

result = test_gen_chain.invoke({
    "code": source_code,
    "context": "DiscountCalculator 根据购买金额计算阶梯折扣",
})
print(result)
```

输出：

```python
import pytest
from decimal import Decimal
from discount import DiscountCalculator


@pytest.fixture
def calculator():
    return DiscountCalculator()


class TestDiscountCalculatorNormalPaths:
    """正常路径测试：各折扣区间的正确性"""

    @pytest.mark.parametrize("amount,expected_rate,expected_final", [
        (50, 0.0, 50.0),          # 无折扣区间
        (100, 0.05, 95.0),       # 区间左边界（100 属于第二档）
        (299.99, 0.05, 284.99),  # 第二档内部
        (500, 0.10, 450.0),      # 第三档左边界
        (1500, 0.10, 1350.0),    # 第三档内部
        (2000, 0.15, 1700.0),    # 最高档左边界
        (9999, 0.15, 8499.15),   # 大额购买
    ])
    def test_tier_discounts(self, calculator, amount, expected_rate, expected_final):
        result = calculator.calculate(amount)
        assert result["discount_rate"] == expected_rate
        assert result["final_amount"] == expected_final

    def test_zero_purchase(self, calculator):
        """金额为 0 时不应报错"""
        result = calculator.calculate(0)
        assert result["final_amount"] == 0.0


class TestDiscountCalculatorEdgeCases:
    """边界情况和异常输入"""

    def test_negative_amount_raises(self, calculator):
        """负数金额应抛出 ValueError"""
        with pytest.raises(ValueError, match="不能为负数"):
            calculator.calculate(-1)

    def test_negative_large_amount(self, calculator):
        """很大的负数也应抛出异常"""
        with pytest.raises(ValueError):
            calculator.calculate(-99999)

    def test_float_precision(self, calculator):
        """浮点数精度：结果应四舍五入到两位小数"""
        result = calculator.calculate(333.33)
        assert result["discount_amount"] == round(333.33 * 0.05, 2)


class TestDiscountCalculatorReturnStructure:
    """返回值结构完整性"""

    def test_return_keys(self, calculator):
        """返回字典应包含所有预期字段"""
        result = calculator.calculate(500)
        expected_keys = {"original", "discount_rate", "discount_amount", "final_amount"}
        assert set(result.keys()) == expected_keys

    def test_original_amount_preserved(self, calculator):
        """原始金额应在返回值中原样保留"""
        result = calculator.calculate(777.77)
        assert result["original"] == 777.77

    def test_discount_matches_formula(self, calculator):
        """折扣额应等于 原价 × 折扣率"""
        result = calculator.calculate(888)
        assert result["discount_amount"] == round(888 * result["discount_rate"], 2)
```

生成的测试代码质量很高：
- 使用了 `@pytest.fixture` 共享实例
- 用 `parametrize` 覆盖了 7 个代表性金额点
- 正常路径、边界情况、异常输入、返回结构验证四个维度都有覆盖
- 每个测试方法都有清晰的 docstring

### 把三个能力组装成统一入口

现在我们把代码解释、Bug 检测、测试生成封装成一个统一的 `CodeAnalyzer` 类：

```python
from enum import Enum
from typing import Optional

class AnalysisTask(Enum):
    EXPLAIN = "explain"
    FIND_BUGS = "find_bugs"
    GENERATE_TESTS = "generate_tests"

class CodeAnalyzer:
    def __init__(self, retriever=None):
        self.retriever = retriever

    def analyze(self, task: AnalysisTask, code: str, context: str = "",
               question: str = None) -> str:
        chains = {
            AnalysisTask.EXPLAIN: explain_chain,
            AnalysisTask.FIND_BUGS: bug_finder_chain,
            AnalysisTask.GENERATE_TESTS: test_gen_chain,
        }

        chain = chains[task]

        if task == AnalysisTask.EXPLAIN and question and self.retriever:
            return project_explain_chain.invoke({
                "question": question,
            })

        return chain.invoke({
            "code": code,
            "context": context,
        })

    def analyze_from_repo(self, task: AnalysisTask, file_path: str,
                          target_name: str = None) -> str:
        if not self.retriever:
            raise ValueError("需要提供 retriever 才能进行仓库级别分析")

        query = f"{target_name or file_path} 的功能和实现细节"
        results = self.retriever.invoke(query)

        target_chunk = None
        for doc in results:
            name = doc.metadata.get("name", "")
            source = doc.metadata.get("source", "")
            if (target_name and name == target_name) or (target_name in source):
                target_chunk = doc
                break

        if not target_chunk and results:
            target_chunk = results[0]

        if not target_chunk:
            return "未找到相关代码"

        context = f"来自文件 {target_chunk.metadata.get('source', 'unknown')}"
        return self.analyze(task, target_chunk.page_content, context)


analyzer = CodeAnalyzer(retriever=retriever)
```

使用体验：

```python
analyzer.analyze(AnalysisTask.FIND_BUGS, buggy_code, "UserAPI 认证类")
# → 输出 Bug 审查报告

analyzer.analyze(AnalysisTask.GENERATE_TESTS, source_code, "折扣计算器")
# → 输出 pytest 测试代码

analyzer.analyze_from_repo(AnalysisTask.EXPLAIN, "auth_service.py", "authenticate")
# → 输出 authenticate 函数的项目级解释
```

三种能力统一入口，根据 `AnalysisTask` 枚举分发到不同的处理链路。下一节我们将在这个基础上更进一步——不只是分析和解释代码，而是**真正执行代码**来验证我们的分析和修复是否正确。
