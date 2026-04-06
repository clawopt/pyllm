---
title: 中间件：横切关注点（日志、限流、重试、审核）
description: RunnablePassthrough.assign / 自定义中间件、日志中间件、重试机制、内容审核
---
# 中间件：横切关注点（日志、限流、重试、审核）

前两节我们学习了流式和异步——它们解决的是**性能层面**的问题（感知延迟、并发吞吐）。这一节我们将学习 **中间件（Middleware）**——它解决的是**横切关注点（Cross-cutting Concerns）**的问题。

什么是横切关注点？简单说就是那些**与核心业务逻辑无关但每个请求都需要执行的操作**：
- 日志记录
- 输入/输出校验
- 速率限制
- 错误处理和重试
- 内容安全审核

如果没有中间件，这些逻辑会散落在代码的各个角落，难以统一管理和修改。有了中间件，你可以把它们声明式地"插入"到任何 Chain 中。

## LCEL 中的中间件模式

在 LCEL 中，中间件的实现方式非常优雅——**它就是一个被插入管道中的 Runnable**：

```python
# 基础 Chain
chain = prompt | chat | parser

# 带中间件的 Chain — 在管道中插入额外组件
chain_with_middleware = (
    input_logger       # 中间件1: 记录输入
    | prompt
    | llm_logger        # 中间件2: 记录 LLM 调用
    | chat
    | output_logger     # 中间件3: 记录输出
    | parser
)
```

每个中间件本身都是一个 `Runnable`，接收上游的输出作为输入，处理后传给下游。数据流是线性的：

```
输入 → [中间件1] → prompt → [中间件2] → chat → [中间件3] → parser → 最终输出
```

## 实战一：日志中间件

最常用的中间件类型——记录每次调用的输入输出：

```python
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
import time
import json

def create_logging_chain(base_chain):
    """为任意 Chain 添加完整的日志记录"""

    def log_input(x):
        print(f"\n{'='*50}")
        print(f"📥 INPUT: {json.dumps(x, ensure_ascii=False)[:200]}")
        return x

    def log_llm_input(x):
        if hasattr(x, 'to_messages'):
            msgs = x.to_messages()
            print(f"🤖 PROMPT ({len(msgs)} messages)")
            for m in msgs:
                content = getattr(m, 'content', '')[:100]
                print(f"   [{m.__class__.__name__}] {content}")
        return x

    def log_output(x):
        print(f"📤 OUTPUT: {str(x)[:200]}")
        print(f"{'='*50}")
        return x

    def measure_time(x):
        start = time.time()
        result = base_chain.invoke(x)  # 注意：这里调用的是原始 chain
        elapsed = time.time() - start
        print(f"\n⏱️  耗时: {elapsed:.2f}s")
        return result

    logged_chain = (
        RunnableLambda(log_input)      # 入口日志
        | base_chain                 # 原始业务逻辑
        | RunnableLambda(log_output)  # 出口日志
    )

    return logged_chain
```

使用：

```python
chat = ChatOpenAI(model="gpt-4o-mini", temperature=0)
prompt = ChatPromptTemplate.from_template("回答: {question}")
base = prompt | chat | StrOutputParser()

logged_chain = create_logging_chain(base)

result = logged_chain.invoke({"question": "什么是 RAG？"})
```

输出：

```
==================================================
📥 INPUT: {"question": "什么是 RAG？"}
🤖 PROMPT (1 messages)
   [HumanMessage] 回答: 什么是 RAG？
📤 OUTPUT: RAG 是一种检索增强生成技术...
==================================================

⏱️  耗时: 2.83s
```

每条日志都清晰展示了数据在管道中流转的过程。这在调试复杂 Chain 时极其有用。

## 实战二：重试中间件

LLM API 偶尔会返回错误（网络超时、服务端过载、速率限制等）。与其让错误直接抛给用户，不如自动重试几次：

```python
import time
from langchain_core.runnables import RunnableLambda

def with_retry(runnable, max_retries=3, delay=1.0):
    """添加自动重试能力的包装器"""

    def retry_wrapper(input_data):
        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                result = runnable.invoke(input_data)
                if attempt > 1:
                    print(f"  ✅ 第 {attempt} 次尝试成功")
                return result
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    print(f"  ❌ 第 {attempt} 次失败: {e}，{delay}秒后重试...")
                    time.sleep(delay)
        
        raise last_error

    return RunnableLambda(retry_wrapper)


# 使用
base_chain = prompt | chat | StrOutputParser()
retry_chain = with_retry(base_chain, max_retries=3)

result = retry_chain.invoke({"question": "解释量子计算"})
```

当第一次调用失败时：

```
❌ 第 1 次失败: RateLimitError, 1.0s后重试...
✅ 第 2 次尝试成功
```

用户完全感知不到底层错误——他们只看到最终的成功结果。

## 实战三：内容审核中间件

对于面向公众的 AI 应用，你需要确保输出内容不包含有害信息。审核中间件可以在结果返回前进行检查和过滤：

```python
from langchain_core.runnables import RunnableLambda

FORBIDDEN_WORDS = ["密码", "secret", "api_key", "token", "管理员"]

def content_auditor(base_chain):
    """内容审核中间件"""

    def audit(output_text):
        text_lower = output_text.lower()
        
        # 检查敏感词
        violations = [w for w in FORBIDDEN_WORDS if w in text_lower]
        
        if violations:
            # 发现敏感词 — 返回安全的替代响应
            print(f"⚠️ 审核拦截: 检测到敏感词 {violations}")
            return "抱歉，该回复包含敏感信息，无法显示。"
        
        # 长度检查
        if len(output_text) > 5000:
            output_text = output_text[:5000] + "\n...(内容过长已截断)"
        
        return output_text

    return base_chain | RunnableLambda(audit)


# 使用
safe_chain = content_auditor(prompt | chat | StrOutputParser())
result = safe_chain.invoke({"question": "告诉我数据库的密码"})
# ⚠️ 审核拦截: 检测到敏感词 ['密码']
# 输出: "抱歉，该回复包含敏感信息，无法显示。"
```

## 组合多个中间件

实际项目中你通常需要同时启用多种中间件。LCEL 让你可以自由组合：

```python
def build_production_chain(base_chain):
    """组装生产级 Chain（含完整中间件栈）"""

    return (
        # === 输入层 ===
        RunnableLambda(validate_input)      # 参数校验
        | RunnableLambda(rate_limit_check)     # 速率限制
        
        # === 业务层 ===
        base_chain                           # 核心逻辑 (prompt|chat|parser)
        
        # === 输出层 ===
        RunnableLambda(content_audit)         # 内容审核
        | RunnableLambda(log_to_database)      # 日志持久化
        | RunnableLambda(measure_metrics)      # 性能指标收集
    )
```

这个 Chain 的数据流是这样的：

```
用户输入
    ↓
[validate_input] ← 格式是否合法？字段是否完整？
    ↓ 合格
[rate_limit_check] ← 是否超频？是否需要排队等待？
    ↓ 通过
[prompt → chat → parser] ← 核心业务逻辑
    ↓ 结果
[content_audit] ← 有没有敏感词？长度是否合理？
    ↓ 安全
[log_to_database] ← 写入审计日志表
    ↓
[measure_metrics] ← 记录耗时、token 消耗等
    ↓
返回给用户
```

每一层中间件只关心自己的职责，互不干扰。这就是 **关注点分离（Separation of Concerns）** 的威力。

## LangChain 内置回调机制

除了手动用 `RunnableLambda` 创建中间件，LangChain 还提供了更底层的 **Callbacks API**，能让你深入到组件内部运行流程的每一个环节：

```python
from langchain.callbacks import StdOutCallbackHandler
from langchain_core.tracers import ConsoleCallbackHandler

handler = StdOutCallbackHandler()

# 方式一：在 invoke 时传入 callbacks
result = chain.invoke(
    {"question": "你好"},
    config={"callbacks": [handler]}
)

# 方式二：用 trace 函数追踪完整链路
from langchain_core.tracers import ConsoleCallbackHandler

result = await chain.ainvoke(
    {"question": "RAG"},
    config={"callbacks": [ConsoleCallbackHandler()]}
)
```

`ConsoleCallbackHandler` 会打印 Chain 执行过程中**每一个 Runnable 的输入和输出**，包括嵌套的子链路。这对于调试复杂的 Agent 或多分支 Chain 特别有价值——你能看到数据在系统内部的完整旅程。

下一节我们将深入学习 Callbacks 机制的更多细节，以及如何用它构建一个带对话审核功能的完整应用。
