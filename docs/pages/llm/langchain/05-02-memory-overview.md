---
title: LangChain 记忆组件全景
description: Memory 类层次结构、BufferMemory / WindowMemory / SummaryMemory / TokenBufferMemory
---
# LangChain 记忆组件全景

上一节我们理解了为什么需要 Memory 以及它解决的核心问题。这一节我们将系统地学习 LangChain 提供的各种记忆组件——它们就像一个工具箱，每种工具适合不同的场景。掌握它们的区别和选择方法，是构建高质量对话应用的关键。

## Memory 的两种工作模式

在深入具体类型之前，先要理解 LangChain 中 Memory 组件的两种基本工作模式，这决定了你在不同场景下该怎么用它：

**模式一：独立使用（Standalone）**。Memory 作为独立的组件，你手动调用它的方法来存取对话历史：

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()

# 手动保存每一轮
memory.save_context({"input": "你好"}, {"output": "你好！有什么可以帮你的？"})
memory.save_context({"input": "我叫张三"}, {"output": "你好张三！"})

# 查看记忆内容
print(memory.load_memory_variables({}))
# {'history': 'Human: 你好\nAI: 你好！有什么可以帮你的？\nHuman: 我叫张三\nAI: 你好张三！'}
```

这种模式给你最大的控制权——你想什么时候存、存什么格式、怎么用这些数据，都由你自己决定。

**模式二：集成到 Chain 中（Integrated）**。Memory 被直接嵌入到 Chain 的管道中，自动在每次调用时管理上下文：

```python
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

chain = LLMChain(
    llm=chat,
    prompt=prompt,
    memory=ConversationBufferMemory()   # 自动管理
)

chain.invoke("你好")      # 自动保存上下文
chain.invoke("我叫什么？") # 自动加载之前的上下文
```

这种模式最省心——你完全不用关心消息列表的管理，Memory 在后台默默地完成一切。

> **重要提示**：在 LangChain v1.0 + LCEL 架构中，推荐的模式是使用 `RunnableWithMessageHistory` 来包装 Chain，而不是旧式的 `LLMChain` + `memory=` 参数。我们会在后续小节中详细介绍这种方式。这里先了解传统模式是为了建立完整的概念体系。

## ConversationBufferMemory：完整的原始记录

这是最简单也最直观的记忆类型——它把**所有对话历史原封不动地保留下来**，不做任何删减或压缩：

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()

memory.save_context({"input": "我叫张三"}, {"output": "你好，张三！"})
memory.save_context({"input": "我喜欢Python"}, {"output": "Python 是一门很棒的语言"})

print(memory.load_memory_variables({}))
```

输出：

```
{
  'history': 'Human: 我叫张三\nAI: 你好，张三！\nHuman: 我喜欢Python\nAI: Python 是一门很棒的语言'
}
```

可以看到，`ConversationBufferMemory` 把每轮对话以 `"Human: ... \nAI: ..."` 的纯文本格式拼接成一段连续的历史字符串。当 Chain 需要上下文时，就把这段字符串注入到提示词中。

### 返回消息对象版本

除了默认的字符串格式，还可以让 BufferMemory 返回结构化的 Message 对象列表（这在 ChatModel 场景下更有用）：

```python
memory = ConversationBufferMemory(return_messages=True)

memory.save_context({"input": "你好"}, {"output": "嗨！"})
memory.save_context({"input": "叫什么"}, {"output": "我是助手"})

result = memory.load_memory_variables({})
print(result)
# {'history': [HumanMessage(content='你好'), AIMessage(content='嗨!'), 
#            HumanMessage(content='叫什么'), AIMessage(content='我是助手')]}
```

设置 `return_messages=True` 后，`history` 字段从纯文本变成了 Message 对象列表。这让它可以直接传给 ChatModel 的 `invoke()` 方法，无需额外的格式转换。

### 适用场景与局限

**适用场景**：
- 短期对话（通常不超过 10-20 轮）
- 需要完整保留每个字的原样（如法律/医疗场景需要审计追踪）
- 对话内容较短，不会触及 token 上限
- 开发和调试阶段（方便查看完整上下文）

**不适用场景**：
- 长对话（几十轮以上）—— token 消耗过大
- 成本敏感的应用 —— 每轮都发送全部历史，API 费用线性增长
- 上下文窗口较小的模型

## ConversationBufferWindowMemory：滑动窗口

如果你需要限制记忆的长度但又不想做复杂的摘要处理，`ConversationBufferWindowMemory` 是一个很好的折中选择——它只保留**最近 N 轮对话**，更早的内容自动丢弃：

```python
from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(k=2)   # 只保留最近 2 轮

# 模拟 4 轮对话
rounds = [
    ("我叫张三", "你好张三！"),
    ("我今年25岁", "正是年轻有为的年纪"),
    ("我在学Python", "Python是个很好的入门语言"),
    ("GIL是什么", "全局解释器锁，是Python的一个特殊机制"),
]

for user_msg, ai_msg in rounds:
    memory.save_context({"input": user_msg}, {"output": ai_msg})

print(memory.load_memory_variables({}))
```

输出：

```
{
  'history': 'Human: 我在学Python\nAI: Python是个很好的入门语言\nHuman: GIL是什么\nAI: 全局解释器锁，是Python的一个特殊机制'
}
```

注意看——前两轮（"我叫张三"和"我今年25岁"）已经被丢弃了，只保留了最后两轮。这就是"滑动窗口"的含义：**窗口大小固定为 k=2，新对话进来时最旧的对话被挤出窗口**。

### 窗口大小的选择

`k` 值的选择需要在"上下文完整性"和"token 效率"之间做权衡：

| k 值 | 保留轮数 | 大约 token 数 | 适用场景 |
|------|---------|--------------|---------|
| 2-3 | 2-3 轮 | ~500-1000 | 简单问答，只需最近上下文 |
| 5 | 5 轮 | ~1500-2500 | 一般对话，需要中等长度上下文 |
| 10 | 10 轮 | ~3000-6000 | 复杂讨论，需要较长的推理链 |

一个实用的经验法则是：**k 设为你认为模型回答当前问题所需要回顾的最大轮数**。如果用户的问题通常是"刚才那个东西怎么改？"这样的短距离引用，k=3~5 就够了；如果是"我们一开始讨论的那个方案……"这样的长距离引用，就需要更大的 k 值。

## ConversationTokenBufferMemory：按 Token 限制

`BufferWindowMemory` 按轮数截断，但不同轮次的对话长度差异可能很大——有的只有几个字，有的是一大段代码。更精细的做法是**按 token 数量来控制记忆的大小**：

```python
from langchain.memory import ConversationTokenBufferMemory
from langchain_openai import ChatOpenAI

chat = ChatOpenAI(model="gpt-4o-mini", temperature=0)

memory = ConversationTokenBufferMemory(
    llm=chat,           # 需要 llm 来计算 token 数
    max_token_limit=200 # 记忆最多占 200 个 token
)

rounds = [
    ("你好", "你好！"),
    ("我叫张三", "你好张三"),
    ("帮我写个排序函数", "好的，这是一个快速排序的实现...（很长一段代码）"),
    ("改成稳定的版本", "..."),
]

for u, a in rounds:
    memory.save_context({"input": u}, {"output": a})

print(memory.load_memory_variables({}))
```

输出大概是这样的：

```
{
  'history': 'Human: 帮我写个排序函数\nAI: 好的，这是一个快速排序的实现...\nHuman: 改成稳定的版本\nAI: ...'
}
```

前面的短对话（"你好"、"我叫张三"）被丢弃了，因为那段长代码占了大量 token，200 token 的配额只能容纳最近的几轮。这就是 **Token Buffer Memory** 相对于 Window Memory 的优势——**它能根据实际内容长度动态调整保留范围**，而不是机械地按轮数切割。

> **关于 `llm` 参数**：`ConversationTokenBufferMemory` 需要知道怎么计算 token 数量，所以必须传入一个 LLM 实例。这个 LLM 只用于 token 计数，不会产生任何 API 调用费用（它用的是本地计数逻辑）。

## ConversationSummaryMemory：摘要式记忆

当对话非常长时，即使只保留最近几轮也可能占用太多 token。这时候更好的策略是：**对旧对话做摘要压缩，只保留摘要 + 最近几轮原文**。这就是 `ConversationSummaryMemory` 的设计思想：

```python
from langchain.memory import ConversationSummaryMemory
from langchain_openai import ChatOpenAI

chat = ChatOpenAI(model="gpt-4o-mini", temperature=0)

memory = ConversationSummaryMemory(llm=chat)

# 模拟多轮对话
rounds = [
    ("你好，我想了解一下Python", "好的，Python是一门高级编程语言..."),
    ("它的主要特点是什么", "Python的主要特点包括：简洁易读、动态类型..."),
    ("那GIL呢", "GIL是全局解释器锁，它使得同一时间只有一个线程执行Python字节码..."),
    ("怎么绕过GIL的限制", "有几种方法：1. 使用multiprocessing代替threading..."),
    ("asyncio呢", "asyncio通过事件循环实现并发，可以绕过GIL的限制..."),
]

for u, a in rounds:
    memory.save_context({"input": u}, {"output": a})
```

查看记忆内容：

```python
result = memory.load_memory_variables({})
print(result['history'])
```

输出类似这样：

```
用户询问了关于 Python 的问题。首先了解了 Python 是一门高级编程语言，
然后讨论了其主要特点包括简洁易读和动态类型等。
接着用户问了 GIL（全局解释器锁），了解到 GIL 使得同一时刻只有一个线程执行字节码，
以及绕过 GIL 的方法包括使用 multiprocessing 和 asyncio 等。
```

看到了吗？5 轮对话被**压缩成了 4 行摘要文字**。原来的几百字对话变成了精炼的概要，token 消耗大幅降低。而且摘要是由 LLM 生成的，能够捕捉对话中的关键信息点。

### 摘要的质量权衡

SummaryMemory 的优势很明显——无论对话多长，记忆的 size 都保持在可控范围内。但它也有代价：

**第一，摘要会丢失细节**。上面的摘要中，具体的绕过 GIL 的方法名称（multiprocessing、asyncio）虽然保留了，但详细的代码示例和参数说明肯定丢失了。如果用户问"刚才说的第一种方法具体怎么用？"，仅凭摘要模型是无法给出准确答案的。

**第二，每次更新摘要都需要调用 LLM**。这意味着 SummaryMemory 有额外的 API 调用成本和延迟。对于高频对话的场景，这个开销不可忽视。

**第三，摘要质量取决于 LLM 的能力**。如果摘要遗漏了关键信息或者产生了偏差，后续的所有回答都会基于这个不准确的摘要，造成错误累积。

因此，SummaryMemory 最适合的场景是：**对话很长、但不需要精确回溯之前的具体内容**。比如闲聊机器人、通用咨询助手等。而对于需要精确引用之前内容的场景（如编程辅导、法律咨询），Buffer 或 Window 类型更合适。

## 组合策略：摘要 + 缓冲

实际项目中最高效的做法往往是**组合使用多种 Memory**——用 SummaryMemory 保存早期对话的摘要，用 BufferMemory 保留最近几轮的原文：

```python
from langchain.memory import ConversationSummaryBufferMemory
from langchain_openai import ChatOpenAI

chat = ChatOpenAI(model="gpt-4o-mini", temperature=0)

memory = ConversationSummaryBufferMemory(
    llm=chat,
    max_token_limit=150,     # 总预算 150 token
    # 超出预算的部分自动转为摘要，未超出的部分保持原文
)
```

`ConversationSummaryBufferMemory` 就是专门为此设计的混合型记忆：它在内部维护两个区域——一个是摘要区（存放旧对话的压缩摘要），另一个是缓冲区（存放最近几轮的原始对话）。当总 token 数超出 `max_token_limit` 时，自动把缓冲区中最旧的对话移入摘要区并压缩。

```python
for u, a in rounds:
    memory.save_context({"input": u}, {"output": a})

print(memory.load_memory_variables({}))
```

输出同时包含摘要和近期原文：

```
[摘要部分]
用户询问了 Python 的基本信息和特点，然后深入讨论了 GIL 机制...

[缓冲部分 - 最近 1-2 轮原文]
Human: asyncio呢
AI: asyncio 通过事件循环实现并发...
```

这种"摘要+缓冲"的组合策略在生产环境中被广泛采用，因为它兼顾了**长期记忆的紧凑性**和**短期记忆的精确性**。

## 各类 Memory 快速对比

为了帮你快速做出选择，我们把目前介绍过的 Memory 类型做一个总结对比：

| Memory 类型 | 存储方式 | Token 消耗 | 信息完整度 | 适用场景 |
|------------|---------|-----------|-----------|---------|
| BufferMemory | 全部原文 | 高（线性增长） | 100% | 短对话 / 调试 |
| BufferWindowMemory | 最近 k 轮 | 固定上限 | 近期 100% | 中等长度对话 |
| TokenBufferMemory | 最近 N token | 固定上限 | 近期 100% | 不规则长度对话 |
| SummaryMemory | LLM 摘要 | 低且稳定 | 较低 | 长对话 / 概览 |
| SummaryBufferMemory | 摘要 + 近期原文 | 可控上限 | 混合 | **生产环境首选** |

下一节我们将把这些 Memory 组件真正集成到 Chain 中，编写可运行的对话程序，并展示 LangChain v1.0 推荐的 `RunnableWithMessageHistory` 用法。
