---
title: 条件分支与路由（RunnableBranch）
description: RunnableBranch 条件路由、动态选择执行路径、路由在 RAG/Agent 中的应用
---
# 条件分支与路由（RunnableBranch）

前两节我们学习了线性管道（`|`）和并行执行（RunnableParallel）。但现实中的很多应用需要**根据输入内容动态地选择不同的处理路径**——比如数学问题走计算器、知识问题走搜索、闲聊走普通 LLM。

这一节我们学习 **RunnableBranch**——LCEL 中实现条件路由的标准方式。

## 为什么需要条件分支

先看一个具体的场景：你正在构建一个智能问答助手，用户的问题可能是以下几种类型：

```python
questions = [
    "175 * 23 + 456 = ?",        # 数学计算 → 需要计算器工具
    "北京今天天气怎么样？",       # 实时信息 → 需要搜索工具
    "什么是装饰器？",             # 知识问题 → 直接用 LLM 回答
    "帮我写一个排序函数"         # 编程任务 → 可能需要代码生成
]
```

如果用 if-else 来处理：

```python
def route_and_answer(question):
    if is_math(question):
        return calculator_chain.invoke(question)
    elif needs_realtime_info(question):
        return search_chain.invoke(question)
    elif is_coding(question):
        return code_gen_chain.invoke(question)
    else:
        return general_llm.invoke(question)
```

这种方式有几个问题：
1. **判断逻辑和业务逻辑耦合**——每次新增一种问题类型都要修改这个函数
2. **难以测试**——每个分支都需要单独构造输入来验证
3. **无法组合**——不能把这个路由逻辑作为组件嵌入更大的 Chain

## RunnableBranch 基础用法

`RunnableBranch` 接受一组 **(条件函数, 目标 Runnable)** 对，按顺序逐一评估条件，返回第一个匹配条件的 Runnable 的输出：

```python
from langchain_core.runnables import RunnableBranch

branch = RunnableBranch(
    (lambda x: is_math(x["question"]), math_chain),
    (lambda x: needs_search(x["question"]), search_chain),
    general_chain   # 默认分支（所有条件都不匹配时）
)

result = branch.invoke({"question": "175 * 23 = ?"})
```

### 工作机制详解

当 `branch.invoke(input)` 被调用时，RunnableBranch 内部会：

1. 取第一个条件函数 `(lambda x: is_math(x["question"]))`
2. 用 input 调用它 → 返回 True 或 False
3. 如果 True → 调用对应的 `math_chain` 并返回结果
4. 如果 False → 继续检查下一个条件
5. 如果所有条件都不匹配 → 调用默认的最后一个 Runnable（没有条件前缀的那个）

## 实战：智能问答路由器

让我们构建一个完整的条件路由示例：

```python
"""
lcel_branch_demo.py — 条件分支实战
"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableBranch, RunnablePassthrough, RunnableLambda,
    RunnableParallel
)

load_dotenv()

chat = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# === 定义各分支的 Chain ===

# 分支一：数学计算（模拟）
math_prompt = ChatPromptTemplate.from_template("""
你是一个计算器。只输出数学表达式的结果，不要解释。
表达式：{question}
""") | chat | StrOutputParser()

# 分支二：搜索模式（模拟）
search_prompt = ChatPromptTemplate.from_template("""
你是搜索引擎助手。基于你的知识回答以下查询。
注意：对于时效性问题，请说明信息可能不是最新的。
查询：{question}
""") | chat | StrOutputParser()

# 分支三：通用问答（默认）
general_prompt = ChatPromptTemplate.from_template(
    "你是一个有帮助的助手。简洁地回答：{question}"
) | chat | StrOutputParser()

# === 判断函数 ===

def is_math_question(text: str) -> bool:
    """检测是否包含数学运算特征"""
    math_keywords = ["+", "-", "*", "/", "=", "计算", "多少", "平方", "开方"]
    text_lower = text.lower()
    return any(kw in text_lower for kw in math_keywords)

def is_realtime_query(text: str) -> bool:
    """检测是否询问实时信息"""
    realtime_keywords = ["天气", "股价", "新闻", "今天", "最新", "现在几点"]
    text_lower = text.lower()
    return any(kw in text_lower for kw in realtime_keywords)

# === 创建路由 ===

router = RunnableBranch(
    (lambda x: is_math_question(x["question"]), math_prompt),
    (lambda x: is_realtime_query(x["question"]), search_prompt),
    general_prompt   # 默认
)

# === 包装为完整 Chain ===
chain = (
    {"question": RunnablePassthrough()}
    | router
)

# === 测试 ===
if __name__ == "__main__":
    test_questions = [
        "175 * 23 + 456 = ?",
        "上海今天天气怎么样？",
        "什么是 Python 装饰器？",
        "帮我写一个快速排序",
        "1+1等于几"
    ]

    for q in test_questions:
        print(f"\n❓ {q}")
        result = chain.invoke({"question": q})
        print(f"💡 {result[:100]}")
        print("-" * 40)
```

运行效果：

```
❓ 175 * 23 + 456 = ?
💡 4481
----------------------------------------

❓ 上海今天天气怎么样？
💡 关于上海的天气情况：我无法提供实时的天气数据。
建议您通过天气预报应用或网站获取上海今天的具体天气...
----------------------------------------

❓ 什么是 Python 装饰器？
💡 Python 装饰器是一种特殊的语法糖...
----------------------------------------

❓ 帮我写一个快速排序
💡 快速排序是一种高效的排序算法...
----------------------------------------

❓ 1+1等于几
💡 2
----------------------------------------
```

可以看到：
- 数学题走了计算器分支（直接给出数字答案）
- 天气搜索走了搜索分支（提示信息可能不最新）
- 知识问题和编程问题走了通用 LLM 分支
- 简单算术也被正确识别为数学问题

## 高级用法：路由 + 并行 组合

RunnableBranch 可以和其他 LCEL 模式自由组合。比如**先并行做多维度分析，再根据分析结果路由到不同的报告生成器**：

```python
# 先并行做摘要和情感分析
pre_analysis = RunnableParallel(
    summary=summarizer,
    sentiment=sentiment_analyzer
)

# 根据情感倾向决定报告风格
style_router = RunnableBranch(
    (lambda x: "正面" in x.get("sentiment", ""), positive_report_chain),
    (lambda x: "负面" in x.get("sentiment", ""), negative_report_chain),
    neutral_report_chain
)

full_chain = (
    {"text": RunnablePassthrough()}
    | pre_analysis      # 并行：摘要 + 情感
    | style_router       # 根据情感路由到不同风格的报告生成器
)
```

这种**"先并行预处理 → 再条件路由"** 的模式在生产级应用中非常强大——它让同一个 Chain 能根据内容的特性自适应地调整后续行为。

## 路由 vs Agent 的选择

看到这里你可能会想：**这不就是上一章学的 Agent 吗？Agent 也是自动选择工具啊。**

两者确实有相似之处，但有本质区别：

| 维度 | RunnableBranch | Agent |
|------|---------------|-------|
| 决策方式 | 基于**预定义规则**（纯 Python 函数） | 基于 **LLM 推理**（模型自己决定） |
| 灵活性 | 固定的分支结构 | 动态的多步规划 |
| 成本 | 低（无额外 LLM 调用） | 高（每轮循环都调 LLM） |
| 适用场景 | 规则清晰、分类明确 | 需要灵活推理的复杂任务 |

经验法则：
- **规则简单且固定** → 用 RunnableBranch（快、便宜、可控）
- **规则复杂或不可预测** → 用 Agent（灵活但成本高）

很多生产系统是**两者结合**的：先用 RunnableBranch 做粗分类（如区分数学/搜索/常识），再在特定分支内部使用 Agent 处理复杂推理。

下一节我们将综合本章所有知识，用 LCEL 从零重构一个复杂的问答 Chain，展示声明式编排的完整威力。
