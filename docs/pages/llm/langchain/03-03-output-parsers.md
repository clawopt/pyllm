---
title: 输出处理：解析器的妙用
description: StrOutputParser、PydanticOutputParser、CommaSeparatedListOutputParser 及自定义解析器
---
# 输出处理：解析器的妙用

前两节我们解决了"怎么选模型"和"怎么组织输入"的问题。现在模型已经返回了结果，但这个结果通常是一段自由格式的文本——如果你的程序需要把这段文本做进一步处理（比如存入数据库、传给另一个函数、在界面上展示），自由文本就不够用了。你需要的是**结构化的数据**：一个字符串、一个列表、一个字典、或者一个自定义的 Python 对象。

LangChain 的 **Output Parser（输出解析器）** 就是干这件事的：它位于 Chain 的末端，负责把模型的原始输出转换成你想要的格式。这一节我们将学习 LangChain 内置的三种最常用的解析器，以及如何编写自定义解析器来满足特殊需求。

## StrOutputParser：提取纯文本

这是最简单也最常用的解析器。`ChatOpenAI.invoke()` 返回的是一个 `AIMessage` 对象，而 `StrOutputParser` 把它变成纯粹的字符串：

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

chat = ChatOpenAI(model="gpt-4o-mini", temperature=0)
prompt = ChatPromptTemplate.from_template("用一句话解释{topic}")
parser = StrOutputParser()

# 方式一：手动调用 parser
response = chat.invoke(prompt.invoke({"topic": "RAG"}))
text = parser.parse(response)
print(text)
print(type(text))  # <class 'str'>

# 方式二（推荐）：用管道串联
chain = prompt | chat | parser
result = chain.invoke({"topic": "RAG"})
print(result)
print(type(result))  # <class 'str'>
```

注意看方式二的输出类型——是 `str`，不是 `AIMessage`。`StrOutputParser` 做的事情就是从 `AIMessage.content` 中提取纯文本并返回。这看起来很简单，但在实际开发中非常有用，因为后续的数据处理流程通常期望接收字符串而不是 Message 对象。

让我们对比一下有无解析器的区别：

```python
prompt = ChatPromptTemplate.from_template("说一个{color}色的水果")
chat = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 没有 parser — 返回 AIMessage 对象
chain1 = prompt | chat
r1 = chain1.invoke({"color": "红"})
print(r1)          # AIMessage(content='苹果', ...)
print(r1.content)   # 苹果
print(type(r1))    # <class 'langchain_core.messages.ai.AIMessage'>

# 有 parser — 返回纯字符串
chain2 = prompt | chat | StrOutputParser()
r2 = chain2.invoke({"color": "红"})
print(r2)           # 苹果
print(type(r2))     # <class 'str'>
```

当你需要把结果直接用于字符串操作（拼接、写入文件、发送到 API 等）时，有 `StrOutputParser` 的版本明显更方便。

## PydanticOutputParser：结构化数据提取

如果你需要的不只是纯文本，而是一个**有明确字段结构的对象**——比如从一段评论中提取情感、主题和关键词——那么 `PydanticOutputParser` 就是你的最佳选择。它利用 Pydantic 的数据验证能力，强制模型按照你定义的 schema 来输出数据。

先定义你的数据结构：

```python
from pydantic import BaseModel, Field

class SentimentAnalysis(BaseModel):
    sentiment: str = Field(description="情感倾向: 正面/负面/中性")
    confidence: float = Field(description="置信度 0-1 之间的数值")
    keywords: list[str] = Field(description="从文本中提取的关键词列表")
    summary: str = Field(description:"一句话总结文本主旨")
```

然后用 `PydanticOutputParser` 包装它：

```python
from langchain_core.output_parsers import PydanticOutputParser

parser = PydanticOutputParser(pydantic_object=SentimentAnalysis)

# 查看解析器自动生成的格式指令
print(parser.get_format_instructions())
```

`get_format_instructions()` 会输出一段详细的格式说明，类似这样：

```
The output should be formatted as a JSON instance that conforms to the JSON schema below.

As an example, for the schema {"properties": {"foo": {"title": "Foo", "type": "string"}}}, "required": ["foo"]}
the object {"foo": "bar"} is a well-formatted instance of the object itself...
```

这段说明的作用是告诉模型："请按这个 JSON 格式来组织你的输出"。你需要把这段说明注入到提示词中，让模型知道该输出什么格式的数据：

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

chat = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_template("""
分析以下文本的情感倾向：

{text}

{format_instructions}
""")

chain = prompt | chat | parser

result = chain.invoke({
    "text": "这家咖啡店的拿铁做得非常地道，环境也很舒适，就是价格稍微有点贵",
    "format_instructions": parser.get_format_instructions()
})

print(result)
# sentiment='正面' confidence=0.85 keywords=['拿铁', '环境舒适', '价格贵'] summary='整体体验良好但价格偏高'
print(type(result))
# <class '__main__.SentimentAnalysis'>

# 可以像普通 Python 对象一样访问属性
print(result.sentiment)     # 正面
print(result.confidence)    # 0.85
print(result.keywords)      # ['拿铁', '环境舒适', '价格贵']
print(result.summary)       # 整体体验良好但价格偏高
```

注意输出的 `result` 不再是字符串或 Message，而是一个 **真正的 `SentimentAnalysis` 实例**——你可以用点号访问它的每个字段，把它传给其他函数，甚至直接序列化为 JSON。这就是 PydanticOutputParser 的核心价值：**把 LLM 的自由文本输出变成了类型安全的 Python 对象**。

### 处理解析失败的情况

有时候模型可能不会严格遵循你要求的 JSON 格式输出（尤其是当提示词设计得不够清晰时）。PydanticOutputParser 在遇到无法解析的输出时会抛出 `OutputParserException`：

```python
from langchain_core.exceptions import OutputParserException

try:
    result = chain.invoke({
        "text": "今天天气不错",
        "format_instructions": parser.get_format_instructions()
    })
except OutputParserException as e:
    print(f"解析失败: {e}")
    # 你可以在这里实现重试逻辑或降级处理
```

在生产环境中，建议对解析器的调用加上异常处理，或者使用 LangChain 提供的 **OutputFixingParser** 来自动修复格式错误：

```python
from langchain_core.output_parsers import OutputFixingParser

# 用另一个 LLM 来修复格式错误的输出
fixing_parser = OutputFixingParser.from_llm(
    parser=parser,
    llm=ChatOpenAI(model="gpt-4o-mini", temperature=0)
)

# 如果原始解析失败，fixing_parser 会调用 LLM 尝试修复输出格式
chain = prompt | chat | fixing_parser
```

## CommaSeparatedListOutputParser：列表提取

第三个常用解析器专注于一种特定但高频的场景：**让模型返回逗号分隔的列表，然后自动解析为 Python 列表**。这在需要获取多个标签、多个选项、多个关键字的场景下特别好用：

```python
from langchain_core.output_parsers import CommaSeparatedListOutputParser

parser = CommaSeparatedListOutputParser()

# 查看格式指令
print(parser.get_format_instructions())
# Your response should be a list of comma separated values,
# eg: `foo, bar, baz`
```

完整的使用示例：

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

chat = ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = CommaSeparatedListOutputParser()

prompt = ChatPromptTemplate.from_template("""
列举5个{category}相关的技术术语。

{format_instructions}
""")

chain = prompt | chat | parser

result = chain.invoke({
    "category": "深度学习",
    "format_instructions": parser.get_format_instructions()
})

print(result)
# ['神经网络', '反向传播', '卷积层', '梯度下降', '注意力机制']
print(type(result))  # <class 'list'>
```

模型输出的是一段包含逗号分隔值的文本（如 `"神经网络, 反向传播, 卷积层, 梯度下降, 注意力机制"`），解析器自动把它拆分成了一个 Python `list`。这个过程包括去除首尾空格、按逗号分割、清理每个元素的空白字符等——这些琐碎的处理工作都被封装好了。

### 与 PydanticOutputParser 的选择

你可能会有疑问：既然 PydanticOutputParser 也能返回列表（通过定义 `list[str]` 字段），为什么还需要专门的 `CommaSeparatedListOutputParser`？

答案是**简洁性和可靠性**。CommaSeparatedListOutputParser 的格式指令更简短（只需要模型输出逗号分隔值），模型遵循这个简单格式的成功率更高。而 PydanticOutputParser 要求完整的 JSON 格式，对于简单的列表需求来说有些"杀鸡用牛刀"。作为一般规则：
- 只需要一个扁平列表 → 用 `CommaSeparatedListOutputParser`
- 需要复杂的嵌套结构 → 用 `PydanticOutputParser`

```python
# 简单场景 — 用 CommaSeparatedListOutputParser
parser = CommaSeparatedListOutputParser()
# 输出: ['item1', 'item2', 'item3']

# 复杂场景 — 用 PydanticOutputParser
class ComplexData(BaseModel):
    items: list[str]
    metadata: dict[str, str]
    count: int
parser = PydanticOutputParser(pydantic_object=ComplexData)
# 输出: ComplexData(items=[...], metadata={...}, count=3)
```

## 自定义解析器

内置的三个解析器覆盖了大部分常见需求，但总有一些特殊场景需要你自己写解析逻辑。LangChain 让编写自定义解析器变得非常简单——你只需要继承 `BaseOutputParser` 并实现两个方法：

```python
from langchain_core.output_parsers import BaseOutputParser
from typing import List

class BracketExtractor(BaseOutputParser[List[str]]):
    """提取文本中被方括号包裹的内容"""

    def parse(self, text: str) -> List[str]:
        import re
        return re.findall(r'\[(.+?)\]', text)

    @property
    def _type(self) -> str:
        return "bracket_extractor"
```

使用方式和内置解析器完全一样：

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

chat = ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = BracketExtractor()

prompt = ChatPromptTemplate.from_template("""
从以下文本中提取所有被方括号标记的关键信息：

{text}
""")

chain = prompt | chat | StrOutputParser() | parser

result = chain.invoke({
    "text": "会议记录：[张三]提出新方案，[李四]表示赞同，[王五]建议下周讨论"
})

print(result)
# ['张三', '李四', '王五']
```

这里有一个细节值得注意：我们在 Chain 中用了 `StrOutputParser() | parser` —— 先用 `StrOutputParser` 把 `AIMessage` 转成纯字符串，再用我们的自定义 `BracketExtractor` 从中提取内容。这种**多解析器串联**的模式在实际项目中很常见，体现了 LangChain 管道设计的灵活性。

### 更实用的自定义解析器示例

来看一个更有实际价值的例子——从模型的输出中提取代码块：

```python
import re
from langchain_core.output_parsers import BaseOutputParser

class CodeBlockParser(BaseOutputParser[str]):
    """提取 Markdown 代码块中的内容"""

    def parse(self, text: str) -> str:
        match = re.search(r'```(?:python)?\n(.*?)```', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text.strip()  # 找不到代码块就返回原文

    @property
    def _type(self) -> str:
        return "code_block_parser"
```

测试效果：

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

chat = ChatOpenAI(model="gpt-4o-mini", temperature=0)
prompt = ChatPromptTemplate.from_template("写一个Python快速排序函数")
chain = prompt | chat | StrOutputParser() | CodeBlockParser()

code = chain.invoke({})
print(code)

# def quicksort(arr):
#     if len(arr) <= 1:
#         return arr
#     pivot = arr[len(arr) // 2]
#     left = [x for x in arr if x < pivot]
#     middle = [x for x in arr if x == pivot]
#     right = [x for x in arr if x > pivot]
#     return quicksort(left) + middle + quicksort(right)
```

模型通常会输出一大段包含解释文字和代码块的混合内容，而 `CodeBlockParser` 帮我们把纯粹的代码部分提取了出来——这对于"生成代码后直接执行"的场景特别有用。

## 解析器在 Chain 中的位置

到目前为止，我们看到的 Chain 都是线性的：`prompt | model | parser`。但解析器不一定非要放在最后面。根据你的需求，它可以出现在 Chain 的任何位置：

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, CommaSeparatedListOutputParser

chat = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 模式一：标准用法 — 放在末尾
chain1 = (
    ChatPromptTemplate.from_template("列出3个{topic}")
    | chat
    | CommaSeparatedListOutputParser()
)

# 模式二：中间处理 — 先转字符串再做进一步处理
chain2 = (
    ChatPromptTemplate.from_template("描述一下{topic}")
    | chat
    | StrOutputParser()       # 先转为纯文本
    # 后续可以接其他 Runnable 组件
)
```

理解了解析器的工作原理和使用方法之后，下一节我们将把这些知识整合起来，构建一个**完整的、可配置的情感分析应用**——它将综合运用模型选择、提示词模板和输出解析，展示 LangChain I/O 全链路的实际工作方式。
