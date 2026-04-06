---
title: 输入管理：提示模板的多种玩法
description: PromptTemplate 基础、Few-shot 模板、聊天消息模板、部分变量
---
# 输入管理：提示模板的多种玩法

上一节我们搞清楚了 LLM 和 ChatModel 的区别，也知道了 ChatModel 的输入是结构化的消息列表。但在实际开发中，你不可能每次都手动拼装 `SystemMessage`、`HumanMessage`——那样太繁琐了，而且容易出错。LangChain 提供了 **提示词模板（Prompt Template）** 机制来解决这个问题：它让你像写带有占位符的文档模板一样定义提示词，然后在运行时用实际数据填充这些占位符。

这一节我们将系统地学习 LangChain 中三种最重要的提示词模板，以及它们各自适用的场景。

## 基础模板：PromptTemplate 和 ChatPromptTemplate

我们在第二章已经初步接触过 `ChatPromptTemplate`，现在让我们更深入地理解它的工作机制。

`ChatPromptTemplate.from_template()` 是最常用的创建方式，它接受一个包含 `{变量名}` 占位符的字符串模板：

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template(
    "你是一个{role}。请用{style}的风格回答以下问题：\n\n{question}"
)

# 查看模板的结构
print(prompt.input_variables)
# ['role', 'style', 'question']

# 用字典填充变量
result = prompt.invoke({
    "role": "Python 助教",
    "style": "轻松幽默",
    "question": "什么是生成器？"
})

print(result)
# messages=[...]
print(result.to_messages())
```

当你调用 `prompt.invoke({...})` 时，LangChain 会做两件事：首先把字典中的值填入模板的占位符位置，然后将结果包装成一个 `HumanMessage`（因为 `from_template()` 默认把整个模板内容当作用户消息）。最终产出的是一个 `ChatPromptValue` 对象，你可以通过 `.to_messages()` 方法把它转换成消息列表——这个列表可以直接传给 `ChatOpenAI.invoke()`。

让我们看看完整的管道效果：

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

chat = ChatOpenAI(model="gpt-4o-mini", temperature=0)
prompt = ChatPromptTemplate.from_template("用一句话解释{topic}")

chain = prompt | chat
response = chain.invoke({"topic": "装饰器"})
print(response.content)
# 装饰器是一种语法糖，它允许你在不修改原函数代码的情况下扩展其功能。
```

这里 `|` 管道操作符的作用是：**把前一个组件的输出自动作为后一个组件的输入**。`prompt.invoke({"topic": "..."})` 产出的 `ChatPromptValue`（本质上是消息列表）会被自动传入 `chat.invoke()`，整个过程一气呵成。

### 多角色模板

上面的例子中，整个模板都被当作一条 `HumanMessage`。但在很多场景下，你需要同时定义 System 消息和 Human 消息——比如设定系统角色 + 用户问题。这时可以用 `from_messages()` 方法：

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的{domain}专家，回答要准确且通俗易懂"),
    ("human", "{question}")
])

result = prompt.invoke({
    "domain": "机器学习",
    "question": "什么是梯度下降？"
})

print(result.to_messages())
```

输出会是这样：

```
[
    SystemMessage(content='你是一个专业的机器学习专家，回答要准确且通俗易懂'),
    HumanMessage(content='什么是梯度下降？')
]
```

注意 `from_messages()` 接受的是一个**元组列表**，每个元组的格式是 `(角色, 模板内容)`。支持的角色包括 `"system"`、`"human"`、`"ai"` 以及我们稍后会详细介绍的 `"placeholder"`。这种写法比 `from_template()` 更灵活，因为它能精确控制每条消息的角色类型。

### 部分变量预填充

有时候你的模板中有一些变量的值是固定的（比如系统角色描述），只有少数变量需要在运行时动态填充。LangChain 提供了 **partial_variables** 机制来处理这种情况：

```python
prompt = ChatPromptTemplate.from_template(
    "你是一个{role}。\n\n请回答：{question}"
)

# 预先填充 role 变量
partial_prompt = prompt.partial(role="资深 Python 工程师")

# 之后只需要填剩下的变量
result1 = partial_prompt.invoke({"question": "什么是 async/await？"})
result2 = partial_prompt.invoke({"question": "typing 模块有什么用？"})

print(result1.to_messages()[0].content)
# 你是一个资深 Python 工程师。
#
# 请回答：什么是 async/await？

print(result2.to_messages()[0].content)
# 你是一个资深 Python 工程师。
#
# 请回答：typing 模块有什么用？
```

`partial()` 返回一个新的模板对象，其中 `role` 已经被锁定为"资深 Python 工程师"，你只需要在后续调用时提供 `question` 的值就行。这种模式在实际项目中非常常见——比如你有一个客服机器人，它的"人设"是固定的，但用户的问题每次都不同，这时候就可以用 `partial()` 把人设预先绑定好。

你也可以在创建模板时就指定 partial variables：

```python
prompt = ChatPromptTemplate.from_template(
    "你是一个{role}。\n\n请回答：{question}",
    partial_variables={"role": "数据分析师"}
)

# 直接调用，不需要再传 role
result = prompt.invoke({"question": "解释一下 SQL 的 JOIN"})
```

## Few-shot 模板：让模型从示例中学习

**Few-shot Prompting（少样本提示）** 是一种强大的技术：你在提示词中给模型展示几个"输入→输出"的示例，然后让它按照相同的模式处理新的输入。这相当于在"考试前给了几道例题"，模型会模仿这些例题的风格和格式来作答。

LangChain 提供了专门的 Few-shot 模板支持。来看一个实际的例子——假设你想让模型把自然语言翻译成 SQL 查询语句：

```python
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

# 定义几个示例（例题）
examples = [
    {"input": "列出所有用户", "sql": "SELECT * FROM users"},
    {"input": "查找叫张三的用户", "sql": "SELECT * FROM users WHERE name = '张三'"},
    {"input": "统计订单总数", "sql": "SELECT COUNT(*) FROM orders"},
]

# 定义示例的格式模板
example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{sql}")
])

# 创建 few-shot 提示词模板
few_shot_prompt = FewShotChatMessagePromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
)

# 构建完整的提示词：系统指令 + 示例 + 实际问题
final_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个 SQL 专家。根据用户的自然语言描述生成对应的 SQL 查询。"),
    few_shot_prompt,           # 这里会展开为所有的示例对话
    ("human", "{input}")
])

# 测试效果
result = final_prompt.invoke({"input": "找出消费超过1000元的用户"})
print(result.to_messages())
```

输出大概是这样的：

```
[
    SystemMessage('你是一个 SQL 专家。根据用户的自然语言描述生成对应的 SQL 查询。'),
    # --- 下面是 few_shot_prompt 展开的内容 ---
    HumanMessage('列出所有用户'),
    AIMessage('SELECT * FROM users'),
    HumanMessage('查找叫张三的用户'),
    AIMessage("SELECT * FROM users WHERE name = '张三'"),
    HumanMessage('统计订单总数'),
    AIMessage('SELECT COUNT(*) FROM orders'),
    # --- 实际问题 ---
    HumanMessage('找出消费超过1000元的用户')
]
```

看到了吗？`FewShotChatMessagePromptTemplate` 把 `examples` 列表中的每一项都按照 `example_prompt` 的格式展开成一对 `HumanMessage` + `AIMessage`，然后插入到最终提示词的对应位置。模型看到这些示例后，就会学会"用户说自然语言 → 我输出 SQL"的模式，从而对新问题给出符合预期的格式。

让我们把它接入 Chain 跑一下完整流程：

```python
from langchain_openai import ChatOpenAI

chat = ChatOpenAI(model="gpt-4o-mini", temperature=0)
chain = final_prompt | chat

response = chain.invoke({"input": "找出消费超过1000元的用户"})
print(response.content)
# SELECT * FROM users WHERE total_spend > 1000
```

模型准确地输出了 SQL 语句，而且格式和示例完全一致。这就是 Few-shot 的威力——**不需要微调模型，仅仅通过在提示词中提供几个精心设计的示例，就能显著提升输出的质量和一致性**。

### 动态选择示例

当你的示例库很大时，每次都把所有示例塞进提示词既浪费 token 又可能造成干扰。LangChain 提供了 **ExampleSelector** 机制，可以根据输入动态选择最相关的示例：

```python
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstore import InMemoryVectorStore

examples = [
    {"input": "列出所有用户", "sql": "SELECT * FROM users"},
    {"input": "查找叫张三的用户", "sql": "SELECT * FROM users WHERE name = '张三'"},
    {"input": "统计订单总数", "sql": "SELECT COUNT(*) FROM orders"},
    {"input": "按金额降序排列订单", "sql": "SELECT * FROM orders ORDER BY amount DESC"},
    {"input": "计算每个用户的平均消费", "sql": "SELECT user_id, AVG(amount) FROM orders GROUP BY user_id"},
]

# 用语义相似度选择最相关的示例
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    OpenAIEmbeddings(),        # 用于计算语义相似度的嵌入模型
    InMemoryVectorStore,       # 向量存储后端
    k=2,                       # 只选最相关的 2 个示例
)

example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{sql}")
])

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_selector=example_selector,   # 注意这里用的是 selector 而不是 examples
    example_prompt=example_prompt,
)
```

现在当我们问"查找消费最高的三个用户"时，selector 会自动从 5 个示例中选出语义最接近的 2 个（可能是关于排序和查询的示例），而不是把全部 5 个都塞进去。这样既节省了 token，又能保证示例的相关性。

> **注意**：`SemanticSimilarityExampleSelector` 需要一个嵌入模型（Embedding Model）来计算文本之间的语义相似度。上面的例子用了 `OpenAIEmbeddings()`，这意味着每次选择示例时都会调用 OpenAI 的 API 来做向量化计算。如果你的项目对延迟或成本敏感，可以考虑使用本地嵌入模型或者基于规则的简单选择器。

## MessagesPlaceholder：动态消息列表

前面介绍的模板都是静态的——模板的结构在编写时就确定了，运行时只是填充变量值。但有些场景下，你需要**动态地插入数量不定的消息**，比如多轮对话中的历史消息——你事先不知道会有多少轮对话，所以无法在模板里写死消息的数量。

`MessagesPlaceholder` 就是解决这个问题的工具。它像一个"插槽"，可以在运行时插入任意数量的消息：

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有帮助的助手"),
    MessagesPlaceholder(variable_name="history"),   # 动态插槽
    ("human", "{question}")
])

# 第一次调用：没有历史消息
result1 = prompt.invoke({
    "history": [],
    "question": "你好"
})
print([m.content for m in result1.to_messages()])
# ['你是一个有帮助的助手', '你好']

# 第二次调用：有历史消息
result2 = prompt.invoke({
    "history": [
        ("human", "你好"),
        ("ai", "你好！有什么可以帮你的？")
    ],
    "question": "什么是 LangChain？"
})
print([m.content for m in result2.to_messages()])
# ['你是一个有帮助的助手', '你好', '你好！有什么可以帮你的？', '什么是 LangChain？']
```

第一次调用时 `history` 是空列表，所以 `MessagesPlaceholder` 不插入任何消息；第二次调用时 `history` 包含了一轮对话的两条消息，它们被完整地插入到了 system 消息和当前问题之间。这种灵活性使得 `MessagesPlaceholder` 成为构建**有状态的对话应用**的核心组件。

下面是一个更完整的对话循环示例：

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

chat = ChatOpenAI(model="gpt-4o-mini", temperature=0)
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个简洁的 Python 助教"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])
chain = prompt | chat

chat_history = []

while True:
    user_input = input("\n你: ")
    if user_input.lower() in ["退出", "exit", "quit"]:
        break

    response = chain.invoke({
        "chat_history": chat_history,
        "question": user_input
    })

    print(f"\n助手: {response.content}")

    # 把本轮对话追加到历史中
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=response.content))
```

运行效果类似这样：

```
你: 什么是装饰器？
助手: 装饰器是一种特殊的函数，它可以修改其他函数的行为而不改变其源代码...

你: 给个例子
助手: 这是一个简单的计时装饰器：
import time
def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        ...
```

每轮对话结束后，我们把用户的输入和 AI 的回复都以 Message 对象的形式追加到 `chat_history` 列表中。下一轮调用时，这些历史消息会被 `MessagesPlaceholder` 自动插入到提示词中，让模型能够"记住"之前的对话内容。

> **性能提醒**：随着对话轮次的增加，`chat_history` 会越来越长，每次调用的 token 消耗也会线性增长。在生产环境中，你需要实现**上下文窗口管理策略**——比如只保留最近 N 轮对话、对历史消息做摘要压缩等。这是构建生产级对话系统的关键挑战之一，我们会在后续章节深入讨论。

## 常见用法与误区

**最佳实践一：始终使用 ChatPromptTemplate 而非 PromptTemplate**。LangChain 中还有一个纯文本版本的 `PromptTemplate`（来自 `langchain_core.prompts.PromptTemplate`），它产出的是纯字符串而不是消息列表。如果你用的是 ChatModel（你应该用），那么 `ChatPromptTemplate` 是唯一正确的选择——它能直接与 ChatModel 的接口对接，而 `PromptTemplate` 还需要额外的转换步骤。

```python
# 不推荐 — 产出的是纯字符串，需要额外转换
from langchain_core.prompts import PromptTemplate
p = PromptTemplate.from_template("解释{topic}")  # 返回 StringPromptValue

# 推荐 — 产出的是消息列表，可直接传给 ChatModel
from langchain_core.prompts import ChatPromptTemplate
p = ChatPromptTemplate.from_template("解释{topic}")  # 返回 ChatPromptValue
```

**最佳实践二：模板变量命名要有意义**。不要用 `{a}`、`{x}` 这种无意义的变量名。好的变量名应该让人一眼就能看出这个位置要填什么内容：`{question}`、`{topic}`、`{user_query}`、`{code_snippet}` 等。这不仅提高代码可读性，也能减少填错参数的概率。

**常见错误一：变量名拼写不一致**。模板里写的是 `{question}` 但 invoke 时传的是 `{"quesiton"}`（少了个 t），LangChain 会抛出 `MissingInputVariableError` 或 `UnexpectedInputVariableError`。这类拼写错误是初学者最高频的报错原因之一。

```python
prompt = ChatPromptTemplate.from_template("回答: {question}")
# ❌ 错误 — 少了一个 t
prompt.invoke({"quesiton": "什么是 AI"})

# ✅ 正确
prompt.invoke({"question": "什么是 AI"})
```

**常见错误二：混淆 from_template 和 from_messages**。`from_template()` 把整个字符串作为一条 HumanMessage；`from_messages()` 接受元组列表，可以精确控制每条消息的角色。如果你需要 System 消息，必须用 `from_messages()` 或者组合使用。

```python
# 只有 human message
p1 = ChatPromptTemplate.from_template("你是{role}。回答{q}")
# p1.invoke() 产生的只有一条 HumanMessage

# 有 system + human messages
p2 = ChatPromptTemplate.from_messages([
    ("system", "你是{role}"),
    ("human", "回答{q}")
])
# p2.invoke() 产生 SystemMessage + HumanMessage
```

到这里，我们已经掌握了发送给模型的"输入端"的各种精细控制手段。接下来的一节，我们将转向另一端——**如何解析和处理模型的输出**，把自由格式的文本转换为程序可用的结构化数据。
