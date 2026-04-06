---
title: 管道操作符与并行化（RunnableParallel）
description: \| 管道的深入理解、RunnableParallel 并行执行、结果合并与引用
---
# 管道操作符与并行化（RunnableParallel）

前两节我们学习了 LCEL 的基本理念和 Runnable 接口。这一节我们将深入两个最强大的组合模式：**管道操作符 `|` 的工作细节**和 **RunnableParallel 并行化**。

## 管道操作符 `|` 深度解析

### 数据如何流过管道

当你在写 `A | B | C` 时，LCEL 在底层做了什么？让我们追踪数据流的完整路径：

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

chat = ChatOpenAI(model="gpt-4o-mini", temperature=0)
prompt = ChatPromptTemplate.from_template("解释{topic}")
parser = StrOutputParser()

chain = prompt | chat | parser

result = chain.invoke({"topic": "RAG"})
```

这个 `invoke({"topic": "RAG"})` 触发了以下过程：

```
Step 1: prompt.invoke({"topic": "RAG"})
    输入: {"topic": "RAG"}
    输出: ChatPromptValue(messages=[HumanMessage(content="解释RAG")])

Step 2: chat.invoke(Step 1 的输出)
    输入: ChatPromptValue(messages=[...])
    输出: AIMessage(content="RAG 是一种检索增强生成技术...")

Step 3: parser.invoke(Step 2 的输出)
    输入: AIMessage(content="...")
    输出: "RAG 是一种检索增强生成技术..." (纯字符串)

最终返回 Step 3 的输出
```

关键点：**每一步的输出类型必须与下一步的输入类型兼容**。LCEL 会自动处理大多数常见转换，但你需要了解基本的类型匹配规则：

| 组件 A 的输出 | 组件 B 的输入 | 兼容性 |
|-------------|------------|--------|
| `dict` | `ChatPromptTemplate` | ✅ 自动匹配变量 |
| `str` / `dict` | `LLM` / `ChatModel` | ✅ 自动包装为消息 |
| `AIMessage` | `StrOutputParser` | ✅ 提取 `.content` |
| `AIMessage` | 另一个 `LLM` | ✅ 作为消息历史传入 |

### 字典分发：同时传给多个组件

当你需要在管道中把同一个输入分发给多个下游组件时，可以用字典语法：

```python
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

chain = (
    {
        "topic": RunnablePassthrough(),        # 原样传递
        "style": RunnableLambda(lambda x: "简洁")   # 固定值
    }
    | prompt    # prompt 同时收到 topic 和 style
    | chat
    | StrOutputParser()
)

result = chain.invoke("Python")
# prompt 收到 {"topic": "Python", "style": "简洁"}
```

字典中的每个 key 对应下游组件的一个变量名——这是 LCEL 中"一对多"数据分发的标准方式。

## RunnableParallel：并行执行

很多时候你的 Chain 需要同时做几件独立的事情，然后把结果汇总。比如对同一篇文章同时做摘要、提取关键词、分析情感：

### 基础用法

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel

chat = ChatOpenAI(model="gpt-4o-mini", temperature=0)

summary_prompt = ChatPromptTemplate.from_template("用一句话总结:\n{text}")
keyword_prompt = ChatPromptTemplate.from_template("提取5个关键词:\n{text}")
sentiment_prompt = ChatPromptTemplate.from_template("判断情感倾向（正面/负面/中性）:\n{text}")

parallel = RunnableParallel(
    summary=summary_prompt | chat | StrOutputParser(),
    keywords=keyword_prompt | chat | StrOutputParser(),
    sentiment=sentiment_prompt | chat | StrOutputParser()
)

article = """
LangChain 是一个用于开发大语言模型应用的开源框架。
它提供了模块化的组件来构建端到端的 AI 应用链路，
包括模型接口、提示词管理、输出解析等功能。
"""

result = parallel.invoke({"text": article})

print(f"摘要: {result['summary']}")
print(f"关键词: {result['keywords']}")
print(f"情感: {result['sentiment']}")
```

输出：

```
摘要: LangChain 是一个帮助开发者构建 LLM 应用的开源框架，提供模块化组件支持端到端 AI 应用开发。
关键词: LangChain, 大语言模型, 开源框架, 模块化组件, AI 应用开发
情感: 正面
```

三个分析任务**同时执行**，总耗时约等于最慢的那个任务（而不是三者之和）。这就是并行的威力。

### 性能对比

```python
import time

# 串行版本
start = time.time()
s = summary_chain.invoke({"text": article})
k = keyword_chain.invoke({"text": article})
sen = sentiment_chain.invoke({"text": article})
serial_time = time.time() - start

# 并行版本
start = time.time()
all_results = parallel.invoke({"text": article})
parallel_time = time.time() - start

print(f"串行: {serial_time:.2f}s")
print(f"并行: {parallel_time:.2f}s")
print(f"加速比: {serial_time/parallel_time:.1f}x")
```

典型结果（网络延迟下）：
```
串行: 4.52s
并行: 1.83s
加速比: 2.5x
```

对于 3 个任务就有 2.5 倍加速；任务越多、单个任务越慢，加速效果越明显。

## 引用并行结果

`RunnableParallel` 的输出是一个字典，key 就是你定义时用的名称。在后续的管道中你可以通过 key 来引用某个分支的结果：

```python
analysis = RunnableParallel(
    summary=summary_chain,
    sentiment=sentiment_chain
)

# 用并行结果组装最终的回答
final_prompt = ChatPromptTemplate.from_template("""
基于以下分析结果，用{tone}的语气回复用户：

文章摘要：{summary}
情感倾向：{sentiment}

用户问题：{question}
""")

chain = (
    {
        "text": RunnablePassthrough(),
        "question": RunnablePassthrough(),
        "tone": RunnableLambda(lambda x: "专业")
    }
    | analysis     # 并行执行摘要+情感
    | final_prompt # 把并行结果 + 原始问题组合成新 prompt
    | chat
)

result = chain.invoke({
    "text": long_article,
    "question": "这篇文章主要讲了什么？"
})
```

注意这里的数据流：
1. 用户输入的 `text` 和 `question` 通过 `RunnablePassthrough()` 透传
2. `text` 被送入 `analysis`（RunnableParallel），内部拆分为 summary + sentiment 两个并行分支
3. `analysis` 的输出（包含 summary 和 sentiment）加上原始 question 和固定 tone 一起送入 `final_prompt`
4. 最终由 chat 生成回答

这种**先并行分析再综合生成**的模式在生产级 RAG 和 Agent 应用中非常常见。

## 实战：用 LCEL 构建多维度分析 Chain

让我们把上面的知识整合成一个完整的实战例子——一个能同时对文本做多维度分析的 Chain：

```python
"""
lcel_parallel_demo.py — LCEL 并行处理演示
"""
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableParallel, RunnablePassthrough, RunnableLambda
)

load_dotenv()

chat = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# === 定义各个分析子任务 ===

summarizer = ChatPromptTemplate.from_template("""
将以下内容总结为一段话（不超过100字）：
{text}
""") | chat | StrOutputParser()

keyword_extractor = ChatPromptTemplate.from_template("""
从以下文本中提取最重要的 5 个关键词，用逗号分隔：
{text}
""") | chat | StrOutputParser()

translator = ChatPromptTemplate.from_template("""
将以下中文翻译成英文：
{text}
""") | chat | StrOutputParser()

# === 并行组合 ===

analyzer = RunnableParallel(
    summary=summarizer,
    keywords=keyword_extractor,
    translation=translator
)

# === 最终综合 ===

final_prompt = ChatPromptTemplate.from_template("""
你是一个文档分析师。请根据以下多维度的分析结果，
用{format}格式输出一份简短的分析报告。

===== 分析结果 =====
📝 摘要：{summary}
🏷️ 关键词：{keywords}
🌐 英文翻译：{translation}

请基于以上信息，给出一个整体评价。
""")

full_pipeline = (
    {
        "text": RunnablePassthrough(),
        "format": RunnableLambda(lambda x: "Markdown")
    }
    | analyzer       # 三路并行
    | final_prompt    # 合并结果
    | chat           # 生成报告
    | StrOutputParser()
)

# === 运行 ===
sample_text = """
检索增强生成（Retrieval-Augmented Generation，简称 RAG）
是一种人工智能技术框架，旨在提升大型语言模型的准确性和可靠性。
其核心思想是：在模型生成答案之前，先从一个外部知识库中检索相关的参考信息，
然后基于这些参考信息来生成最终回答。
这种方式有效缓解了模型的幻觉问题，并使答案能够溯源到具体的数据来源。
"""

if __name__ == "__main__":
    result = full_pipeline.invoke({"text": sample_text})
    print(result)
    print("\n" + "=" * 50)
    print("✅ 多维度分析完成！")
```

运行后你会看到一段结构化的分析报告——摘要、关键词、翻译一应俱全，全部由三个并行分支自动完成后再综合生成的。

到这里，我们已经掌握了 LCEL 的线性管道、并行执行两种核心模式。下一节我们将学习最后一个关键能力——**条件分支与路由（RunnableBranch）**，它让 Chain 能够根据输入动态选择不同的执行路径。
