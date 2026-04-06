---
title: 结构化输出：JSON / 表格 / 代码
description: 强制 LLM 输出结构化格式、Pydantic 输出解析器、表格自动生成、代码块格式化、前端集成方案
---
# 结构化输出：JSON / 表格 / 代码

到目前为止，我们的 RAG 系统返回的都是纯文本答案——自然语言段落，人读起来很舒服。但在很多实际应用场景中，调用方需要的不是一段文字，而是**结构化的数据**：前端要渲染成表格或表单、下游系统要解析成 API 响应、数据分析工具要导入 CSV 或 JSON。

这一节我们来学习如何在 LlamaIndex 的响应合成阶段强制 LLM 输出结构化格式，以及如何可靠地解析这些结构化输出。

## 为什么需要结构化输出？

考虑以下几个真实场景：

**场景一：产品参数查询**

用户问："S1 的完整技术规格是什么？"

纯文本输出：
```
S1 使用四核 ARM Cortex-A53 处理器，配备 2GB DDR4 内存和 16GB eMMC 存储。
音频方面搭载 40mm 全频单元 × 2 加上低音辐射器...
```

结构化输出（JSON）：
```json
{
  "product_name": "智能音箱 S1",
  "specs": {
    "processor": "ARM Cortex-A53 四核 @ 1.8GHz",
    "memory": "2GB DDR4",
    "storage": "16GB eMMC",
    "audio": "40mm 全频 × 2 + 低音辐射器",
    "connectivity": ["Wi-Fi 6", "Bluetooth 5.3", "Zigbee 3.0"],
    "dimensions": "98×98×145mm",
    "weight": "380g"
  },
  "price": {"currency": "CNY", "amount": 299}
}
```

前端可以直接用这个 JSON 渲染一个漂亮的参数卡片，而不是试图从自由文本中正则提取字段值。

**场景二：对比表格**

用户问："S1 和 S2 的功能对比"

纯文本输出可能是一段冗长的叙述，而结构化输出可以是一个 Markdown 表格：

| 功能 | S1 | S2 |
|------|-----|-----|
| 价格 | ¥299 | ¥399 |
| 处理器 | 4 核 ARM | 8 核 ARM |
| 蓝牙 | 5.3 | 5.3 + LE Audio |
| 显示屏 | 无 | 3.5" 触控屏 |
| 电池 | 无 | 内置电池 12h |

**场景三：代码生成**

用户问："怎么用 Python 连接数据库？"

输出应该是一段可直接运行的代码块，而不是对代码的文字描述。

## OutputProcessor：LlamaIndex 的结构化输出方案

LlamaIndex 通过 `OutputProcessor` 机制来实现结构化输出。核心思路是：
1. 在 Prompt 中要求 LLM 以特定格式输出
2. 用 `OutputProcessor` 解析和验证输出
3. 返回解析后的结构化对象

### JSON 输出

```python
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core.response_synthesizers import get_response_synthesizer
from pydantic import BaseModel, Field
from typing import List


class ProductSpecs(BaseModel):
    """产品规格的结构化输出模型"""
    product_name: str = Field(description="产品名称")
    processor: str = Field(description="处理器型号")
    memory: str = Field(description="内存大小")
    storage: str = Field(description="存储容量")
    connectivity: List[str] = Field(description="支持的连接协议")
    price_cny: int = Field(description="人民币价格（分）")


output_parser = PydanticOutputParser(output_cls=ProductSpecs)

synthesizer = get_response_synthesizer(
    response_mode="compact",
    output_parser=output_parser,
)

query_engine = index.as_query_engine(response_synthesizer=synthesizer)

response = query_engine.query("S1 的技术规格和价格？")

if hasattr(response, "parsed"):
    specs: ProductSpecs = response.parsed  # 类型安全的 Pydantic 对象
    print(f"产品: {specs.product_name}")
    print(f"处理器: {specs.processor}")
    print(f"连接: {', '.join(specs.connectivity)}")
    print(f"价格: ¥{specs.price_cny / 100}")
else:
    # 解析失败时的降级处理
    print(f"无法解析为结构化数据，原文: {response.response}")
```

### 工作原理

当你指定 `output_parser` 时，LlamaIndex 会自动做以下事情：

```
Step 1: 在 Prompt 中追加格式指令
  "请严格按照以下 JSON 格式输出你的答案：
  {output_format_instruction}"

  其中 output_format_instruction 由 PydanticOutputParser 根据
  ProductSpecs 模型自动生成，类似：
  {
    "product_name": "string (产品名称)",
    "processor": "string (处理器型号)",
    ...
  }

Step 2: LLM 生成带 JSON 格式的回答
  {..."product_name": "S1 智能音箱", "processor": "ARM Cortex-A53"...}

Step 3: PydanticOutputParser 尝试用 Pydantic 解析
  → 成功: 返回 ProductSpecs 对象
  → 失败: 返回原始文本 + 错误信息
```

### Markdown 表格输出

对于表格类的结构化输出，可以使用类似的机制：

```python
from llama_index.core.output_parsers import MarkdownTableOutputParser

table_parser = MarkdownTableOutputParser()

synthesizer = get_response_synthesizer(
    response_mode="simple_summarize",
    output_parser=table_parser,
)

response = query_engine.query("对比 S1 和 S2 的主要参数")

if hasattr(response, "parsed"):
    table = response.parsed  # 解析后的表格数据
    print(table)  # 可以直接渲染为 HTML 表格
```

### 代码块输出

对于需要返回可执行代码的场景：

```python
from llama_index.core.output_parsers import CodeOutputParser

code_parser = CodeOutputParser(language="python")

synthesizer = get_response_synthesizer(
    response_mode="simple_summarize",
    output_parser=code_parser,
)

response = query_engine.query("写一个 Python 函数计算斐波那契数列")

if hasattr(response, "parsed"):
    code = response.parsed  # 提取出的代码字符串
    exec(code)  # 可选：直接执行验证
```

## 手动实现结构化输出解析

如果你需要比内置 `OutputProcessor` 更精细的控制，可以手动实现解析逻辑：

```python
import re
import json
from typing import Optional, Any
from llama_index.core import Response
from llama_index.core.schema import NodeWithScore
from llama_index.core.response_synthesizers import BaseResponseSynthesizer
from llama_index.core.schema import TextNode


class StructuredQueryEngine:
    """支持多种结构化输出格式的查询引擎"""

    def __init__(self, index, default_format="text"):
        self.index = index
        self.default_format = default_format

    def query_json(self, question: str, schema_model) -> Any:
        """查询并以 JSON 格式返回"""
        from pydantic import BaseModel

        output_parser = PydanticOutputParser(output_cls=schema_model)
        synthesizer = get_response_synthesizer(
            response_mode="compact",
            output_parser=output_parser,
        )
        qe = self.index.as_query_engine(response_synthesizer=synthesizer)
        response = qe.query(question)

        if hasattr(response, "parsed") and response.parsed is not None:
            return response.parsed
        else:
            # Fallback: 尝试从文本中提取 JSON
            return self._extract_json_fallback(response.response)

    def query_table(self, question: str, columns: list[str]) -> dict:
        """查询并以表格格式返回"""
        table_parser = MarkdownTableOutputParser()
        synthesizer = get_response_synthesizer(
            response_mode="simple_summarize",
            output_parser=table_parser,
        )
        qe = self.index.as_query_engine(response_synthesizer=synthesizer)
        response = qe.query(question)

        if hasattr(response, "parsed"):
            return {
                "columns": columns,
                "rows": response.parsed,
                "raw_markdown": response.response,
            }
        return {"columns": columns, "rows": [], "raw": response.response}

    def _extract_json_fallback(self, text: str) -> Optional[dict]:
        """当 Pydantic 解析失败时，尝试从文本中提取 JSON"""
        patterns = [
            r"\{[\s\S]*\}",           # 标准 JSON 对象
            r"```json\s*\n([\s\S]*?)\n\s*```",  # 代码块中的 JSON
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    continue
        return None
```

## Web API 中的结构化输出

当 RAG 系统作为后端 API 服务时，结构化输出尤其有价值：

```python
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List
from enum import Enum

app = FastAPI()

class QueryFormat(str, Enum):
    TEXT = "text"
    JSON = "json"
    TABLE = "table"


class QueryRequest(BaseModel):
    question: str
    format: QueryFormat = QueryFormat.TEXT


class StructuredResponse(BaseModel):
    answer: str
    data: Optional[dict] = None
    table: Optional[dict] = None
    sources: List[dict] = []


@app.post("/api/query")
def structured_query(req: QuestionRequest) -> StructuredResponse:
    if req.format == QueryFormat.JSON:
        data = engine.query_json(req.question, ProductSpecs)
        return StructuredResponse(
            answer=data.model_dump_json(),
            data=data.model_dump(),
            sources=[],  # 可添加来源信息
        )

    elif req.format == QueryFormat.TABLE:
        result = engine.query_table(
            req.question,
            columns=["参数", "S1", "S2"],
        )
        return StructuredResponse(
            answer=result["raw_markdown"],
            table=result,
        )

    else:
        response = engine.query(req.question)
        return StructuredResponse(
            answer=response.response,
            sources=[
                {"file": n.metadata.get("file_name"), "score": n.score}
                for n in response.source_nodes
            ],
        )
```

前端可以根据 `format` 参数决定如何渲染结果——JSON 数据用 React 组件渲染卡片，Markdown 表格直接用 HTML 渲染，纯文本用 Markdown 渲染器显示。

## 结构化输出的可靠性策略

LLM 生成的结构化格式并不总是完美的——它可能会：
- **格式错误：** 缺少引号、多了逗号、嵌套层级不对
- **字段缺失：** 忘记输出某个必需的字段
- **类型错误：** 把数字写成文字、把列表写成字符串
- **幻觉字段：** 编造了 Schema 中不存在的字段

应对这些问题的策略：

**策略一：Schema 约束 + 重试**

```python
from tenacity import retry, stop_after_attempt

@retry(stop=stop_after_attempt(3), retry=lambda e: isinstance(e, ValueError))
def robust_json_query(question):
    data = engine.query_json(question, ProductSpecs)
    if not data:
        raise ValueError("Failed to parse JSON output")
    return data
```

**策略二：多模型验证**

```python
def validated_json_query(question, schema_model):
    primary_data = engine.query_json(question, schema_model)
    if not primary_data:
        return None

    # 用第二个 LLM（可能是更强的模型）验证和修复
    verification_prompt = (
        f"以下是一个从文档提取的 JSON 数据：\n{primary_data}\n\n"
        f"请检查其是否符合以下 Schema 要求：\n"
        f"{schema_model.model_json_schema()}\n\n"
        f"如果有错误，请输出修正后的合法 JSON。\n"
        f"如果没有错误，直接原样输出即可。"
    )
    verified = strong_llm.complete(verification_prompt).text
    try:
        return schema_model.model_validate_json(json.loads(verified))
    except Exception:
        return primary_data  # 返回原始结果作为最佳努力
```

**策略三：优雅降级**

```python
def query_with_fallback(question):
    # 尝试结构化输出
    try:
        data = engine.query_json(question, ProductSpecs)
        if data:
            return {"status": "structured", "data": data}
    except Exception as e:
        pass

    # 降级到纯文本
    response = engine.query(question)
    return {
        "status": "text",
        "data": response.response,
        "sources": [n.metadata.get("file_name") for n in response.source_nodes],
        "warning": "结构化解析失败，已降级为纯文本输出",
    }
```

## 常见误区

**误区一:"结构化输出 100% 可靠"。** 不是的。即使使用了 Pydantic 校验和重试机制，LLM 仍然可能持续生成格式错误的输出（特别是当 Schema 很复杂或上下文不足时）。**始终要有降级到纯文本输出的 fallback 方案。**

**误区二:"所有查询都适合结构化输出"。** 不适合。开放式的问题（"总结一下这篇文章"、"给我一些建议"）强行要求 JSON 输出会导致 LLM 要么丢失大量信息（为了塞进固定 Schema），要么产生虚假的字段填充。**只在查询预期有明确结构化答案时使用结构化输出。**

**误区三:"OutputProcessor 只能在 Synthesizer 层面使用"。** 不完全对。你也可以在获取 `response.response` 后手动调用 `output_parser.parse(response.text)` 来做后处理式的结构化解析。这种方式更灵活——你可以根据 response 的内容动态决定是否尝试结构化解析。
