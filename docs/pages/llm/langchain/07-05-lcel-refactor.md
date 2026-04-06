---
title: 实战：用 LCEL 重构一个复杂的问答链
description: 综合运用管道/并行/路由，从零构建一个多模式智能问答系统
---
# 实战：用 LCEL 重构一个复杂的问答链

前面四节我们学习了 LCEL 的全部核心能力：Runnable 接口、`|` 管道、RunnableParallel 并行化、RunnableBranch 条件路由。这一节我们将综合运用这些知识，从零构建一个**多模式智能问答系统**——它能自动识别问题类型，选择最优的处理路径，并给出结构化的回答。

## 需求分析

我们的智能问答系统需要处理以下类型的问题：

| 问题类型 | 处理方式 | 示例 |
|---------|---------|------|
| 数学计算 | 计算器分支 | "175 * 23 + 456 = ?" |
| 实时信息 | 搜索分支 | "今天北京天气？" |
| 代码生成 | 编程分支 | "写一个 Python 快速排序" |
| 文档查询 | RAG 分支（检索知识库） | "公司的报销政策是什么？" |
| 通用知识 | LLM 默认分支 | "什么是机器学习？" |

## 第一步：定义各分支 Chain

```python
"""
lcel_smart_qa.py — 基于 LCEL 的多模式智能问答系统
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

# --- 分支一：数学计算 ---

math_chain = ChatPromptTemplate.from_template("""
你是一个精确的计算器。
1. 只输出计算结果，不要解释过程
2. 如果表达式有误，返回 "无法计算"
3. 对于复杂表达式，分步给出中间结果

算式：{question}
""") | chat | StrOutputParser()

# --- 分支二：实时搜索（模拟）---

search_chain = ChatPromptTemplate.from_template("""
你是信息助手。基于你的知识回答以下实时类查询：
- 天气：给出一般性建议和注意事项
- 股价：说明你无法提供实时数据，建议查看专业平台
- 新闻：说明你的知识可能不是最新的

查询：{question}
""") | chat | StrOutputParser()

# --- 分支三：代码生成 ---

code_chain = ChatPromptTemplate.from_template(
    "你是一个编程助手。请用{lang}语言完成以下任务：\n{question}\n"
    "只输出代码，不需要解释。"
) | chat | StrOutputParser()

# --- 分支四：RAG 文档查询（模拟）---

rag_chain = ChatPromptTemplate.from_template("""
你是公司内部文档助手。基于以下参考资料回答问题。

参考资料：
- 员工手册：年假5天，病假10天；差旅住宿一线城市≤500元/晚
- 报销政策：超2000元需总监审批，超5000元需CEO审批
- API限制：免费版20次/分，专业版200次/分，企业版1000次/分

如果资料中没有相关信息，明确说明。回答要简洁。

问题：{question}
""") | chat | StrOutputParser()

# --- 分支五：通用问答（默认）---

general_chain = ChatPromptTemplate.from_template(
    "你是一个有帮助的助手。简洁地回答：{question}"
) | chat | StrOutputParser()
```

## 第二步：构建分类器 + 路由

```python
def classify_question(text: str) -> int:
    """
    返回问题类型编号：
    0 = 数学计算, 1 = 实时搜索, 2 = 代码生成, 3 = 内部文档, 4 = 通用
    """
    text_lower = text.lower()

    # 数学特征检测
    math_ops = ["+", "-", "*", "/", "=", "**", "//", "sqrt", "pow", "计算", "等于", "多少"]
    if any(op in text_lower for op in math_ops):
        return 0

    # 实时信息特征
    realtime_kw = ["天气", "股价", "新闻", "今天", "现在几点", "最新", "汇率"]
    if any(kw in text_lower for kw in realtime_kw):
        return 1

    # 代码生成特征
    code_kw = ["写", "实现", "函数", "代码", "程序", "算法", "排序", "class ", "def "]
    if any(kw in text_lower for kw in code_kw):
        return 2

    # 内部文档特征
    internal_kw = ["报销", "请假", "工资", "考勤", "API", "休假", "差旅", "审批"]
    if any(kw in text_lower for kw in internal_kw):
        return 3

    return 4  # 默认：通用


# 创建路由器
router = RunnableBranch(
    (lambda x: classify_question(x["question"]) == 0, math_chain),
    (lambda x: classify_question(x["question"]) == 1, search_chain),
    (lambda x: classify_question(x["question"]) == 2, code_chain),
    (lambda x: classify_question(x["question"]) == 3, rag_chain),
    general_chain   # type 4 及其他 → 通用
)
```

## 第三步：增强版 —— 先并行分析再路由

上面的版本直接根据关键词做简单分类。让我们升级它——**先并行做多维度分析，再基于分析结果做更智能的路由决策**：

```python
# === 增强版：先分析再路由 ===

# Step A: 并行做轻量级预分析
pre_analyzer = RunnableParallel(
    length=RunnableLambda(lambda x: f"长度:{len(x['question'])}"),
    has_math=RunnableLambda(lambda x: {
        "yes" if any(o in x["question"] for o in "+-*/=计算")
        else "no"}),
    keywords=RunnableLambda(lambda x: x["question"][:30])
)

# Step B: 基于分析结果的增强路由
smart_router = RunnableBranch(
    (lambda x: x.get("has_math") == "yes", math_chain),
    (lambda x: classify_question(x["keywords"]) == 1, search_chain),
    (lambda x: classify_question(x["keywords"]) == 2, code_chain),
    (lambda x: classify_question(x["keywords"]) == 3, rag_chain),
    general_chain
)

# 组装完整 pipeline
smart_pipeline = (
    {"question": RunnablePassthrough()}
    | pre_analyzer      # 并行：长度 + 是否数学 + 关键词预览
    | smart_router     # 基于多维度信号做路由决策
)
```

## 第四步：交互式主程序

```python
def main():
    print("=" * 56)
    print("   🧠 多模式智能问答系统 (LCEL 版)")
    print("=" * 56)
    print("  支持能力:")
    print("  🔢 数学计算    🔍 实时搜索")
    print("  💻 代码生成    📚 文档查询")
    print("  💬 通用问答")
    print("=" * 56)

    while True:
        try:
            question = input("\n❓ 你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 再见！")
            break

        if not question:
            continue
        if question.lower() in ["退出", "exit", "quit"]:
            break

        print("\n🧠 正在分析问题类型...")

        try:
            result = smart_pipeline.invoke({"question": question})
            print(f"\n💡 {result}")
        except Exception as e:
            print(f"\n⚠️ 出错了: {e}")

        print()


if __name__ == "__main__":
    main()
```

## 运行效果演示

```
========================================================
   🧠 多模式智能问答系统 (LCEL 版)
========================================================

❓ you: 175 * 23 + 456 等于多少？

🧠 正在分析问题类型...

💡 4481

--------------------------------------------------------

❓ you: 上海今天天气怎么样？

🧠 正在分析问题类型...

💡 关于上海的天气情况：我无法提供实时的天气数据...
建议您通过天气预报应用或网站获取上海今天的具体温度...

--------------------------------------------------------

❓ you: 写一个 Python 二叉树遍历

🧠 正在分析问题类型...

💡 class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
    
    def preorder(self):
        if self.val: print(self.val)
        if self.left: self.left.preorder()
        if self.right: self.right.preorder()

--------------------------------------------------------

❓ you: 差旅费超过 5000 怎么审批？

🧠 正在分析问题类型...

💡 根据员工手册的报销政策：
- 单笔超过 **2,000 元** → 需要 **部门总监审批**
- 单笔超过 **5,000 元** → 需要 **CEO 审批**

你的情况属于 CEO 审批范畴。建议准备详细的预算说明...

--------------------------------------------------------

❓ you: 什么是深度学习？

🧠 正在分析问题类型...

💡 深度学习是机器学习的一个子领域...
```

每个问题都被自动识别类型并路由到最合适的处理链路——用户完全不用关心背后发生了什么。

## 项目结构与扩展方向

```
lcel-smart-qa/
├── .env
├── lcel_smart_qa.py          # 主程序
└── requirements.txt
```

### 扩展思路

**方向一：加入 RAG 真实检索**

把 `rag_chain` 从模拟数据替换为真实的向量库检索：

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

vectorstore = Chroma(persist_directory="./chroma_db",
                     embedding_function=OpenAIEmbeddings())
retriever = vectorstore.as_retriever(k=3)

real_rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | chat
    | parser
)

# 替换 router 中的 rag_chain
```

**方向二：添加执行日志中间件**

```python
from langchain_core.runnables import RunnableLambda

logged_pipeline = (
    {"question": RunnablePassthrough()}
    | RunnableLambda(lambda x: print(f"\n[INPUT] {x['question']}") or x)
    | pre_analyzer
    | RunnableLambda(lambda x: print(f"[ROUTE] math={x.get('has_math')}") or x)
    | smart_router
    | RunnableLambda(lambda x: print(f"[OUTPUT] {x[:80]}") or x)
)
```

**方向三：接入记忆模块**

用 `RunnableWithMessageHistory` 包装整个 smart_pipeline，让系统能记住之前的对话上下文：

```python
from langchain.runnables.history import RunnableWithMessageHistory

with_memory = RunnableWithMessageHistory(
    runnable=smart_pipeline,
    get_session_history=get_session_history,
    input_messages_key="question"
)
```

到这里，LCEL 章的全部 5 个小节就结束了。我们从"为什么需要 LCEL"出发，学习了 Runnable 统一接口、`|` 管道操作符、RunnableParallel 并行化、RunnableBranch 条件路由，最终构建了一个能自动识别问题类型并路由到最优处理路径的多模式智能问答系统。

LCEL 是 LangChain v1.0 最核心也最强大的抽象——掌握它意味着你可以用声明式的方式组合任意复杂度的 AI 应用流水线。接下来的一章，我们将进入 Agent 的世界，让应用从"被动应答"进化为"自主思考和行动"。
