---
title: 实战：你的第一个 LangChain 应用
description: Hello World 程序、流式输出实现、完整的项目结构模板
---
# 实战：你的第一个 LangChain 应用

前面三节我们搭好了环境、装好了依赖、接通了模型。现在终于到了最激动人心的部分——**写出你的第一个能跑的 LangChain 应用**。

这一节我们从最简单的"Hello World"开始，逐步加入更多功能，让你亲身体验 LangChain 的核心工作方式。

## 第一个程序：Hello World

创建一个新文件 `hello.py`：

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

response = llm.invoke("你好，请用一句话介绍你自己")
print("=" * 40)
print(response.content)
print("=" * 40)
```

运行它：

```bash
python hello.py
```

你应该会看到类似这样的输出：

```
========================================
我是 GPT-4o mini，由 OpenAI 开发的多模态大语言模型。
========================================
```

恭喜！**你刚刚完成了与 GPT-4o 的第一次对话**——虽然是通过 LangChain 做的中间人。这段代码虽然简单，但它已经包含了 LangChain 应用的三个核心要素：

1. **模型初始化**（`ChatOpenAI`）—— 选择用哪个 LLM
2. **输入发送**（`.invoke()`）—— 把用户消息传给模型
3. **结果获取**（`response.content`）—— 取出模型的文本回复

## 第二步：加上提示词模板

上面的例子中，我们把问题硬编码在 `invoke()` 里。但在实际应用中，你通常需要更灵活的提示词——比如每次问的问题不同，或者你想给模型设定一个固定的角色/风格。这就需要用到 **PromptTemplate（提示词模板）**：

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 定义一个带有占位符的模板
prompt_template = ChatPromptTemplate.from_template(
    "你是一个{role}。请用{style}的风格回答问题。\n\n问题：{question}"
)

# 用不同的参数填充模板，生成不同的行为
result_casual = prompt_template.invoke({
    "role": "Python 助教",
    "style": "轻松活泼",
    "question": "什么是装饰器？"
})

result_formal = prompt_template.invoke({
    "role": "技术文档撰写专家",
    "style": "严谨专业",
    "question": "解释一下 Python 的 GIL 机制"
})

print("=== 轻松版 ===")
print(result_casual.content)
print("\n=== 严谨版 ===")
print(result_formal.content)
```

运行后你会看到两个风格截然不同的回答——同样的底层模型，仅仅因为提示词不同，输出的语气和深度就完全不同了。这就是**提示词工程（Prompt Engineering）** 的威力：**你通过控制输入来控制输出**，而 LangChain 的 PromptTemplate 让这个过程变得结构化和可复用。

注意看 `invoke()` 的参数是一个字典 `{...}` —— 这里的 key 必须和模板中的 `{变量名}` 完全对应。如果模板里有 `{question}` 但你传的参数里没有这个 key，LangChain 会直接报错。这是初学者最容易犯的错误之一。

## 第三步：体验流式输出

到目前为止，我们的程序都是"等模型全部生成完毕后一次性返回结果"。如果模型的回复很长（比如一篇长文章），你可能要等好几秒甚至十几秒才能看到任何输出——这对用户体验来说是很糟糕的。

**流式输出（Streaming）** 解决的就是这个问题：它让模型一边生成一边把已生成的片段推送给你，而不是等到全部完成才一次性返回：

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
prompt = ChatPromptTemplate.from_template("用三句话解释什么是 RAG")

# stream=True 让响应变成逐 token 推送
for chunk in llm.stream(prompt.invoke({"input": "用三句话解释什么是 RAG"})):
    print(chunk.content, end="", flush=True)  # end="" 让每个片段紧跟在前一段后面

print()  # 最后换行
```

运行效果大概是这样的（文字会像打字一样逐步出现）：

```
RAG（检索增强生成）是一种 AI 架构模式，它的核心思想是...
```

对比非流式版本：你要干等几秒然后突然收到全部内容；而流式版本是**字逐个出现**，体验上流畅得多。对于聊天类应用来说，流式输出几乎是必须的功能——没有用户愿意盯着空白屏幕等 10 秒钟。

> **关于 `flush=True`**：在终端里，Python 的 `print()` 默认是有缓冲区的——也就是说内容可能不会立即显示到屏幕上。`flush=True` 强制刷新缓冲区，确保每个 token 一生成就立刻显示出来。这在流式输出中至关重要。

## 一个完整的项目结构模板

到这里你已经掌握了 LangChain 应用的基本要素。下面是一个推荐的项目目录结构，后续章节的所有代码都可以基于这个结构来组织：

```
your-project/
├── .env                          # API Key 等敏感信息（不提交 Git）
├── .gitignore                     # Git 忽略规则
├── hello.py                       # 你的第一个程序（刚才写的）
├── chain_basic.py                 # 提示词模板示例
└── streaming.py                  # 流式输出示例
```

`.gitignore` 文件至少应该包含以下内容：

```
.env
__pycache__/
*.pyc
.venv/
```

好了，第 2 章的全部 4 个小节都已完成。下一章我们将深入 LangChain 的核心 I/O 机制——理解如何精确地控制输入格式、如何可靠地解析输出结构化数据，以及如何构建一个有实际价值的情感分析器。
