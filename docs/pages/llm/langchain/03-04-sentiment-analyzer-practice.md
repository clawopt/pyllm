---
title: 实战：构建一个可配置参数的情感分析器
description: 综合运用模型、模板和解析器，从零搭建一个生产级的情感分析工具
---
# 实战：构建一个可配置参数的情感分析器

前面三节我们分别学习了模型类型、提示词模板和输出解析器这三个独立的概念。现在到了把它们串起来的时候了——这一节我们将从零开始，逐步构建一个**完整的、可配置的情感分析器**。这个分析器不仅能判断文本的情感倾向，还能通过调整参数来控制分析的粒度、输出格式和分析维度，让你真正体会 LangChain I/O 全链路的协作方式。

## 需求分析

我们要构建的情感分析器需要满足以下功能：

1. **情感分类**：判断文本是正面、负面还是中性
2. **置信度评分**：给出 0-1 之间的置信度值
3. **多维度分析**：不仅判断整体情感，还能分析特定维度（如产品质量、服务态度、性价比等）
4. **可配置性**：通过参数控制分析的语言、详细程度、是否提取关键词等
5. **结构化输出**：返回一个 Python 对象，方便程序后续处理

## 第一步：定义数据结构

首先用 Pydantic 定义我们期望的输出格式。好的数据结构设计是整个应用的基石：

```python
from pydantic import BaseModel, Field
from typing import Optional

class DimensionScore(BaseModel):
    """单个维度的评分"""
    dimension: str = Field(description="维度名称")
    sentiment: str = Field(description="该维度的情感: 正面/负面/中性")
    score: float = Field(description="该维度的得分 0-1, 越高越正面")
    reason: str = Field(description="该维度情感的理由")

class SentimentResult(BaseModel):
    """完整的情感分析结果"""
    overall_sentiment: str = Field(description="整体情感倾向: 正面/负面/中性")
    overall_score: float = Field(description="整体得分 0-1, 越高越正面")
    confidence: float = Field(description="分析置信度 0-1")
    dimensions: list[DimensionScore] = Field(description="各维度的详细分析")
    keywords: list[str] = Field(description="从文本中提取的关键词")
    summary: str = Field(description="一句话总结")
```

注意 `DimensionScore` 和 `SentimentResult` 的嵌套关系——这是一个**有层次的结构化数据**，纯靠字符串处理几乎不可能可靠地解析出来，而 PydanticOutputParser 能很好地处理这种复杂结构。

## 第二步：构建可配置的提示词模板

核心的设计思路是：**把所有可配置的选项都作为模板变量**，这样用户可以在不修改代码的情况下调整分析器的行为：

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

parser = PydanticOutputParser(pydantic_object=SentimentResult)

system_template = """你是一个专业的情感分析专家。你的任务是对用户提供的文本进行深入的情感分析。

分析要求：
- 分析语言：{language}
- 详细程度：{detail_level}
- 是否提取关键词：{extract_keywords}
- 分析维度：{dimensions}

请严格按照 JSON 格式输出分析结果。
{format_instructions}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_template),
    ("human", "请分析以下文本的情感：\n\n{text}")
])
```

这个模板包含了 5 个可配置参数：

| 参数 | 说明 | 可选值示例 |
|------|------|-----------|
| `language` | 输出语言 | `"中文"`, `"English"` |
| `detail_level` | 详细程度 | `"简洁"`, `"标准"`, `"详细"` |
| `extract_keywords` | 是否提取关键词 | `"是"`, `"否"` |
| `dimensions` | 要分析的维度 | `"整体", "产品质量,服务态度,价格"` |
| `text` | 待分析文本 | 任意字符串 |

## 第三步：组装 Chain

现在把所有组件串联起来：

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

chat = ChatOpenAI(model="gpt-4o-mini", temperature=0)

analyzer = prompt | chat | parser
```

就这四行——`prompt`（输入格式化）→ `chat`（模型推理）→ `parser`（输出解析）。LangChain 的管道操作符 `|` 把三个独立的组件无缝地连接在一起，数据在它们之间自动流转。

## 第四步：使用分析器

### 基础用法

```python
result = analyzer.invoke({
    "language": "中文",
    "detail_level": "标准",
    "extract_keywords": "是",
    "dimensions": "整体情感",
    "text": "这家餐厅的菜品味道很好，环境也不错，但服务员反应太慢了，等了20分钟才点单",
    "format_instructions": parser.get_format_instructions()
})

print(f"整体情感: {result.overall_sentiment}")
print(f"整体得分: {result.overall_score:.2f}")
print(f"置信度:   {result.confidence:.2f}")
print(f"关键词:   {result.keywords}")
print(f"总结:     {result.summary}")
```

典型输出：

```
整体情感: 正面
整体得分: 0.70
置信度:   0.88
关键词:   ['菜品味道好', '环境不错', '服务员慢', '等20分钟']
总结:     整体体验偏正面，但服务效率问题明显拉低了满意度
```

### 多维度分析

切换到多维度模式，只需要改变 `dimensions` 参数的值：

```python
result = analyzer.invoke({
    "language": "中文",
    "detail_level": "详细",
    "extract_keywords": "是",
    "dimensions": "菜品质量, 环境氛围, 服务态度, 性价比",
    "text": "这家餐厅的菜品味道很好，环境也不错，但服务员反应太慢了，等了20分钟才点单，价格倒是挺合理的",
    "format_instructions": parser.get_format_instructions()
})

print(f"=== 总体 ===")
print(f"情感: {result.overall_sentiment} | 得分: {result.overall_score:.2f}")

print(f"\n=== 各维度 ===")
for dim in result.dimensions:
    print(f"{dim.dimension:8s} | {dim.sentiment:4s} | {dim.score:.2f} | {dim.reason}")
```

输出：

```
=== 总体 ===
情感: 正面 | 得分: 0.72

=== 各维度的 ===
菜品质量  | 正面 | 0.90 | 用户明确表示"味道很好"
环境氛围  | 正面 | 0.85 | 用户认为"环境不错"
服务态度  | 负面 | 0.30 | "等了20分钟才点单"，等待时间过长
性价比    | 正面 | 0.80 | 用户认可"价格合理"
```

看到了吗？同样的文本，仅仅因为 `dimensions` 参数不同，输出的信息丰富度就有了质的飞跃。这就是**可配置性**的价值——一套代码适配多种分析场景。

### 切换语言

把 `language` 改为 `"English"`，整个输出就会变成英文：

```python
result = analyzer.invoke({
    "language": "English",
    "detail_level": "标准",
    "extract_keywords": "是",
    "dimensions": "Overall",
    "text": "The product quality is excellent but shipping took forever",
    "format_instructions": parser.get_format_instructions()
})

print(result.summary)
# Overall positive sentiment despite significant frustration with delivery speed
```

## 第五步：封装为可复用的类

到目前为止我们的分析器还是"裸奔"状态——每次调用都要手动传一堆参数。在实际项目中，你应该把它封装成一个整洁的类：

```python
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional, List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()

class DimensionScore(BaseModel):
    dimension: str = Field(description="维度名称")
    sentiment: str = Field(description="情感: 正面/负面/中性")
    score: float = Field(description="得分 0-1")
    reason: str = Field(description="理由")

class SentimentResult(BaseModel):
    overall_sentiment: str = Field(description="整体情感")
    overall_score: float = Field(description="整体得分 0-1")
    confidence: float = Field(description="置信度 0-1")
    dimensions: List[DimensionScore] = Field(description="各维度分析")
    keywords: List[str] = Field(description="关键词")
    summary: str = Field(description="一句话总结")

class SentimentAnalyzer:
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        language: str = "中文",
        detail_level: str = "标准",
        extract_keywords: bool = True,
        dimensions: str = "整体情感",
        temperature: float = 0
    ):
        self.chat = ChatOpenAI(model=model_name, temperature=temperature)
        self.language = language
        self.detail_level = detail_level
        self.extract_keywords = "是" if extract_keywords else "否"
        self.dimensions = dimensions

        self.parser = PydanticOutputParser(pydantic_object=SentimentResult)

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个专业的情感分析专家。

分析要求：
- 语言：{language}
- 详细程度：{detail_level}
- 提取关键词：{extract_keywords}
- 维度：{dimensions}

{format_instructions}
"""),
            ("human", "分析以下文本的情感：\n\n{text}")
        ])

        self.chain = self.prompt | self.chat | self.parser

    def analyze(self, text: str) -> SentimentResult:
        return self.chain.invoke({
            "language": self.language,
            "detail_level": self.detail_level,
            "extract_keywords": self.extract_keywords,
            "dimensions": self.dimensions,
            "text": text,
            "format_instructions": self.parser.get_format_instructions()
        })

    def analyze_batch(self, texts: List[str]) -> List[SentimentResult]:
        return self.chain.batch([
            {
                "language": self.language,
                "detail_level": self.detail_level,
                "extract_keywords": self.extract_keywords,
                "dimensions": self.dimensions,
                "text": t,
                "format_instructions": self.parser.get_format_instructions()
            }
            for t in texts
        ])
```

封装后的用法变得非常简洁：

```python
# 创建分析器实例 — 所有配置在这里一次性完成
analyzer = SentimentAnalyzer(
    model_name="gpt-4o-mini",
    language="中文",
    detail_level="详细",
    dimensions="产品质量, 服务态度, 性价比"
)

# 单条分析
result = analyzer.analyze("手机屏幕很清晰，但电池不耐用，客服态度还不错")
print(result.summary)

# 批量分析
texts = [
    "非常好用，强烈推荐！",
    "一般般，没有宣传的那么好",
    "垃圾产品，再也不买了"
]
results = analyzer.analyze_batch(texts)

for i, r in enumerate(results):
    print(f"\n--- 文本 {i+1} ---")
    print(f"情感: {r.overall_sentiment} | 得分: {r.overall_score:.2f}")
    print(r.summary)
```

注意 `analyze_batch()` 方法——它利用了 LangChain 的 `.batch()` 接口，可以**并行地发送多个分析请求**，比逐个调用 `analyze()` 快得多（尤其是在分析大量文本时）。这是 LangChain 内置的性能优化能力之一。

## 第六步：添加实用工具方法

为了让分析器更好用，我们可以给它加一些辅助方法：

```python
class SentimentAnalyzer:
    # ... (前面的代码保持不变) ...

    def format_report(self, result: SentimentResult) -> str:
        """生成易读的分析报告"""
        lines = [
            f"{'='*50}",
            f"情感分析报告",
            f"{'='*50}",
            f"整体情感: {result.overall_sentiment}",
            f"整体得分: {result.overall_score:.2f}/1.00",
            f"置信度:   {result.confidence:.2f}",
        ]

        if result.dimensions:
            lines.append(f"\n{'─'*40}")
            lines.append("维度详情:")
            for d in result.dimensions:
                lines.append(
                    f"  [{d.dimension}] {d.sentiment} "
                    f"(得分 {d.score:.2f}) - {d.reason}"
                )

        if result.keywords:
            lines.append(f"\n关键词: {', '.join(result.keywords)}")
        lines.append(f"\n总结: {result.summary}")

        return "\n".join(lines)

    def compare(self, text_a: str, text_b: str) -> dict:
        """对比两条文本的情感差异"""
        r1 = self.analyze(text_a)
        r2 = self.analyze(text_b)

        return {
            "text_a": {"sentiment": r1.overall_sentiment, "score": r1.overall_score},
            "text_b": {"sentiment": r2.overall_sentiment, "score": r2.overall_score},
            "diff": r1.overall_score - r2.overall_score,
            "more_positive": "A" if r1.overall_score > r2.overall_score else "B"
        }
```

使用效果：

```python
analyzer = SentimentAnalyzer(dimensions="整体情感")

report = analyzer.format_report(
    analyzer.analyze("这款耳机音质很棒，降噪效果好，就是戴久了耳朵有点疼")
)
print(report)
```

输出：

```
==================================================
情感分析报告
==================================================
整体情感: 正面
整体得分: 0.78/1.00
置信度:   0.90

────────────────────────────────────────────────
维度详情:
  [整体情感] 正面 (得分 0.78) - 音质和降噪获得高度评价，但佩戴舒适度扣分

关键词: 音质很棒, 降噪效果好, 戴久耳朵疼

总结: 产品核心功能出色但存在人体工学缺陷，整体评价偏正面
```

## 完整项目文件结构

到这里，我们的情感分析器已经是一个功能完整的小型应用了。推荐的项目组织方式如下：

```
sentiment-analyzer/
├── .env                          # API Key
├── .gitignore
├── requirements.txt              # langchain-openai, python-dotenv, pydantic
├── analyzer.py                   # SentimentAnalyzer 类定义
├── demo.py                       # 演示脚本
└── test_texts.txt                # 测试用的样本文本
```

`demo.py` 的内容大概是这样：

```python
from analyzer import SentimentAnalyzer

def main():
    analyzer = SentimentAnalyzer(
        language="中文",
        detail_level="详细",
        dimensions="产品质量, 服务态度, 性价比"
    )

    while True:
        text = input("\n请输入要分析的文本（输入 q 退出）: ").strip()
        if text.lower() == 'q':
            break
        if not text:
            continue

        result = analyzer.analyze(text)
        print(analyzer.format_report(result))

if __name__ == "__main__":
    main()
```

运行后你就可以不断地输入文本，即时看到专业的情感分析报告了。

## 性能优化建议

当你的分析器投入实际使用时，有几个性能方面的考虑值得了解：

**第一，模型选择对成本和速度的影响巨大**。上面的例子用的是 `gpt-4o-mini`——它速度快、成本低，适合大多数场景。如果你需要更高的准确率，可以换成 `gpt-4o`，但成本会增加约 10 倍，速度也会慢一些。对于简单的二分类任务（只判断正面/负面），甚至可以考虑更小的模型。

**第二，批量处理优于逐个处理**。当你需要分析 100 条评论时，`analyze_batch()` 比 100 次 `analyze()` 快得多——因为 batch 模式可以利用并发请求来减少总的等待时间。

**第三，缓存重复请求**。如果相同的文本可能被多次分析（比如同一个商品的评价被多个用户查看），可以考虑加入缓存层，避免重复调用 API：

```python
from functools import lru_cache

class SentimentAnalyzer:
    def __init__(self, ..., cache_size: int = 128):
        ...
        self._cache = {}

    def analyze(self, text: str) -> SentimentResult:
        if text in self._cache:
            return self._cache[text]
        result = self.chain.invoke({...})
        self._cache[text] = result
        return result
```

## 常见问题和调试技巧

**问题一：输出格式不符合预期**。如果模型的输出无法被 Pydantic 解析，通常是因为提示词中的格式指令不够明确。解决方法是：
1. 确保 `format_instructions` 被正确注入到提示词中
2. 在系统提示中强调"必须严格遵循 JSON 格式"
3. 使用 `OutputFixingParser` 作为后备方案（上一节介绍过）

**问题二：分析结果不一致**。即使 `temperature=0`，对于复杂的分析任务，模型也可能在不同调用间产生细微差异。如果需要完全确定性的结果，可以考虑：
- 使用 `seed` 参数（如果模型支持）
- 对结果做后处理归一化
- 采用投票机制（多次分析取众数）

**问题三：token 超限**。当待分析的文本很长时（比如一篇几千字的文章），可能会超过模型的上下文窗口限制。解决方案：
- 在发送前截断或摘要文本
- 使用支持更长上下文的模型（如 `gpt-4o` 支持 128K token）
- 分段分析后合并结果

到这里，第三章的全部内容就结束了。我们从模型接口的选择讲起，学习了如何精细化管理输入（提示词模板）、如何可靠地解析输出（输出解析器），最终把它们整合成了一个有实际价值的情感分析应用。这些知识构成了 LangChain I/O 操作的完整基础——接下来的章节将在此基础上引入更强大的概念：Memory（记忆）、Chain（链式组合）、以及 Agent（智能体），让我们的应用从"能对话"进化到"能思考"。
