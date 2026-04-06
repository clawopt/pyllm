---
title: Node Parser 详解：SentenceSplitter 与其变体
description: SentenceSplitter 内部机制、参数调优指南、CodeSplitter/MetadataAwareSeparator、自定义 Parser 实现
---
# Node Parser 详解：SentenceSplitter 与其变体

上一节我们分析了文档解析面临的种种挑战，这一节来深入 LlamaIndex 提供的解决方案——Node Parser（节点解析器）。如果说 Document 是原材料，那 Node Parser 就是加工机床，它的任务是把粗糙的原材料切割成规格统一的、适合后续工序的零件（Nodes）。

LlamaIndex 内置了多种 Node Parser，每种适用于不同的文档类型和场景。其中最重要、最常用的是 `SentenceSplitter`，我们会花大部分篇幅来深入理解它；然后再介绍其他几种专用 Parser。

## SentenceSplitter 的工作原理

`SentenceSplitter` 是 LlamaIndex 的默认 Node Parser，它的名字暗示了它的基本策略——**按照句子边界来切分文本**。但实际的算法比名字表达的要精妙得多。

```python
from llama_index.core.node_parser import SentenceSplitter

splitter = SentenceSplitter(
    chunk_size=512,       # 每个 chunk 的最大字符数
    chunk_overlap=100,    # 相邻 chunk 之间的重叠字符数
)
nodes = splitter.get_nodes_from_documents(documents)
```

### 内部切分流程

当你调用 `get_nodes_from_documents()` 时，SentenceSplitter 内部执行以下流程：

```
输入: Document(text="很长的一篇文档...")
        │
        ▼
┌───────────────────────┐
│ Step 1: 句子分割         │
│                        │
│ 用正则表达式识别句子边界 │
│ （句号、问号、感叹号等）  │
└───────────┬───────────┘
            │
            ▼ sentences: ["句子1。", "句子2？", "句子3！", ...]
┌───────────────────────┐
│ Step 2: 分组（chunking） │
│                        │
│ 将连续的句子组合成 group │
│ 直到达到 chunk_size     │
└───────────┬───────────┘
            │
            ▼ chunks: ["句子1。句子2。", "句子3。句子4。", ...]
┌───────────────────────┐
│ Step 3: 重叠处理         │
│                        │
│ 相邻 chunk 共享 overlap │
│ 字符的尾部/头部内容     │
└───────────┬───────────┘
            │
            ▼ nodes: [Node(text=...), Node(text=...), ...]
```

### 句子分割的细节

Step 1 的句子分割是整个流程中最关键的一步。SentenceSplitter 内部使用的分割逻辑大致如下（简化版）：

```python
import re

def split_sentences(text: str) -> list[str]:
    """简化的句子分割逻辑"""
    pattern = r"""
        (?<=[。！？.!?])              # 前瞻：前面是句子结束标点
        \s*                          # 可选的空白字符
        (?=[^a-zA-Z\d]|$)           # 后瞻：后面不是字母数字（避免在缩写处断开）
        |
        (?<=\n)\n+                  # 或者：换行符（段落边界）
    """
    return re.split(pattern, text, flags=re.VERBOSE)
```

这个正则表达式处理了几种常见的边界情况：
- 中文句号（`。`）、感叹号（`！`）、问号（`？`）
- 英文句号（`.`）、感叹号（`!`）、问号（`?`）
- 换行符（作为段落分隔）

但它也有已知的局限：
- **不会在分号（；）处断开**——因为分号不一定代表句子结束
- **不会在冒号（：）后断开**——同上
- **对缩写的处理不完美**——如 "U.S." 可能被错误地在 "." 后断开
- **列表项之间的边界不会被识别**——"1. 第一项\n2. 第二项" 不会被拆成两个句子

### 分组策略

Step 2 的分组逻辑采用了一种贪心算法：

```python
def chunk_sentences(sentences: list[str], chunk_size: int) -> list[str]:
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_len = len(sentence)
        if current_length + sentence_len > chunk_size and current_chunk:
            chunks.append("".join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_len
        else:
            current_chunk.append(sentence)
            current_length += sentence_len

    if current_chunk:
        chunks.append("".join(current_chunk))

    return chunks
```

核心思想是：**尽可能多地往当前 chunk 里塞句子，直到塞不下为止，然后开启一个新的 chunk**。这是一种简单但有效的策略——它保证了每个 chunk 都接近但不超出 `chunk_size` 限制（最后一个句子可能会略微超出）。

### 重叠（Overlap）的实现

Step 3 的重叠处理确保了相邻 chunks 之间有共享的内容：

```python
def add_overlap(chunks: list[str], overlap: int) -> list[str]:
    if overlap <= 0:
        return chunks

    result = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_chunk = result[-1]
        curr_chunk = chunks[i]

        # 从前一个 chunk 的末尾取 overlap 个字符
        overlap_text = prev_chunk[-overlap:] if len(prev_chunk) > overlap else prev_chunk
        result.append(overlap_text + curr_chunk)

    return result
```

假设 `chunk_size=20`, `overlap=5`，原文是 `"ABCDEFGHIJ KLMNOPQRST UVWXYZ"`，切分结果如下：

```
Chunk #1: "ABCDEFGHIJ KLMNO"  (19 字符，加上下一句会超限)
Chunk #2: "MNOP UVWXYZ"       (前 5 字符 "MNOP " 来自 Chunk #1 的尾部)
```

注意 Chunk #2 开头的 "MNOP " 就是 overlap 区域——它既出现在 Chunk #1 的末尾，又出现在 Chunk #2 的开头。这样做的意义在于：**如果一个信息点恰好落在两个 chunk 的边界附近，它在两个 chunk 中都会出现，保证了至少有一个 chunk 包含了这个信息点的完整上下文。**

## 参数调优实战

`chunk_size` 和 `chunk_overlap` 是 SentenceSplitter 最重要的两个参数，它们的取值直接影响 RAG 系统的效果。让我们通过实验来感受不同参数的影响。

### 实验：chunk_size 的影响

假设我们有以下技术文档片段：

```
Redis 是一个开源的内存数据结构存储系统，可以用作数据库、缓存和消息中间件。
它支持字符串、哈希、列表、集合、有序集合等多种数据类型。
Redis 提供了主从复制、Sentinel 哨兵和 Cluster 集群三种高可用方案。
在生产环境中推荐使用 Cluster 模式，因为它提供了自动分片和故障转移能力。
Cluster 模式最少需要 3 个主节点才能保证选举的正常进行。
```

**chunk_size=128 时：**

```
Node #1:
Redis 是一个开源的内存数据结构存储系统，可以用作数据库、缓存和消息中间件。
它支持字符串、哈希、列表、集合、有序集合等多种数据类型。

Node #2:
多种数据类型。Redis 提供了主从复制、Sentinel 哨兵和 Cluster 集群三种
高可用方案。在生产环境中推荐使用 Cluster 模式，因为它提供了自动分片和故障转
移能力。Cluster 模式最少需要 3 个主节点才能保证选举的正常进行。
```

**chunk_size=256 时：**

```
Node #1:
Redis 是一个开源的内存数据结构存储系统，可以用作数据库、缓存和消息中间件。
它支持字符串、哈希、列表、集合、有序集合等多种数据类型。
Redis 提供了主从复制、Sentinel 哨兵和 Cluster 集群三种高可用方案。
在生产环境中推荐使用 Cluster 模式，因为它提供了自动分片和故障转移能力。
```

**chunk_size=64 时：**

```
Node #1:
Redis 是一个开源的内存数据结构存储系统，可以用作数据库、缓存和消息中间件。

Node #2:
消息中间件。它支持字符串、哈希、列表、集合、有序集合等多种数据类型。

Node #3:
Redis 提供了主从分割、Sentinel 哨兵和 Cluster 集群三种高可用方案。
...
```

现在模拟几个查询来看看不同 chunk_size 下的表现：

| 用户查询 | chunk_size=64 | chunk_size=128 | chunk_size=256 |
|----------|--------------|----------------|----------------|
| "Redis 是什么？" | ✅ Node #1 完美匹配 | ✅ Node #1 匹配 | ✅ Node #1 匹配 |
| "支持哪些数据类型？" | ⚠️ Node #2 只有后半截 | ✅ Node #1 完整覆盖 | ✅ 完整覆盖 |
| "Cluster 至少需要几个节点？" | ❌ 信息分散在多个 Node | ✅ Node #2 包含答案 | ✅ 完整覆盖 |
| "总结 Redis 的高可用方案" | ❌ 需要 3+ 个 Node | ⚠️ 跨越 2 个 Node | ✅ 单个 Node 足够 |

从这个实验可以看出：**没有万能的 chunk_size**。小 chunk 有利于精确匹配（"Redis 是什么？"），大 chunk 有利于全面回答（"总结高可用方案"）。

### 经验法则

基于大量实践，以下是一些经验参考值：

| 场景 | 推荐 chunk_size | 推荐 overlap | 原因 |
|------|----------------|-------------|------|
| 短问答（FAQ 类） | 128-256 | 32-64 | 问题短小精悍，答案通常也在一两句话内 |
| 技术文档 | 512-1024 | 100-200 | 需要保留一定的概念完整性 |
| 法律合同 | 1024-2048 | 200-400 | 条款通常较长且相互关联 |
| 学术论文 | 512-768 | 100-150 | 段落长度适中，但引用关系重要 |
| 新闻文章 | 256-512 | 50-100 | 段落较短，信息密度中等 |
| 聊天记录 | 128-256 | 30-50 | 对话轮次短，但需要保留上下文 |

**最重要的建议：用你的实际数据和查询来做 A/B 测试。** 上面的表格只是起点，最优值取决于你的具体数据特征和用户查询模式。

## CodeSplitter：专为代码设计

如果你的知识库中包含大量代码文件（API 文档中的示例代码、内部技术方案的代码片段等），普通的 SentenceSplitter 效果会很差——它会在函数定义的中间、类的属性列表中、或者 import 语句的中间切断代码，导致每个 chunk 都是无法运行的残缺代码。

`CodeSplitter` 是专门为程序代码设计的 Node Parser：

```python
from llama_index.core.node_parser import CodeSplitter

splitter = CodeSplitter(
    language="python",       # 编程语言
    max_chars_per_line=100,  # 每行最大字符数（超过则折行）
    chunk_lines=80,          # 每个 chunk 最大行数
    chunk_lines_overlap=15,  # 行级重叠
)
nodes = splitter.get_nodes_from_documents(documents)
```

CodeSplitter 的特点：

**按语法单元切分而非按字符数。** 它理解代码的结构——函数定义、类定义、import 块、控制流块等。切分会尽量发生在这些语法单元的边界上：

```python
# 原始代码
import os
import json
from typing import List


class DataProcessor:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)

    def _load_config(self, path: str) -> dict:
        with open(path) as f:
            return json.load(f)


    def process(self, data: List[dict]) -> List[dict]:
        results = []
        for item in data:
            results.append(self._transform(item))
        return results

# CodeSplitter 切分后（示意）
Chunk #1:
import os
import json
from typing import List

class DataProcessor:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)

Chunk #2:
    def _load_config(self, path: str) -> dict:
        with open(path) as f:
            return json.load(f)

    def process(self, data: List[dict]) -> List[dict]:
```

可以看到，切分发生在方法之间而不是方法内部——每个 chunk 都是**语法上相对完整**的代码片段。

**支持的语言：** CodeSplitter 支持主流编程语言的语言特定切分规则：Python、JavaScript、TypeScript、Java、C/C++、Go、Rust、Ruby、PHP 等。每种语言都有针对其语法的定制化切分逻辑。

## MetadataAwareSentenceSplitter：感知元数据的切分

有时候，同一个文档中不同部分的切分策略应该是不同的。比如一份年度报告中，执行摘要部分应该用较大的 chunk（因为每段话信息密度高），而财务报表附注部分应该用较小的 chunk（因为每条附注是独立的）。`MetadataAwareSentenceSplitter` 允许你根据 metadata 来动态调整切分参数：

```python
from llama_index.core.node_parser import MetadataAwareSentenceSplitter

splitter = MetadataAwareSentenceSplitter(
    breakpoint_generator_metadata_key="section_type",
    breakpoints={
        "executive_summary": {"chunk_size": 1024, "chunk_overlap": 200},
        "financial_notes": {"chunk_size": 256, "chunk_overlap": 50},
        "default": {"chunk_size": 512, "chunk_overlap": 100},
    },
)
```

使用前需要先给 Document 的 metadata 中添加分段标识：

```python
for doc in documents:
    section_type = classify_section(doc.text)  # 你自己实现的分类逻辑
    doc.metadata["section_type"] = section_type

nodes = splitter.get_nodes_from_documents(documents)
```

这种"先分类再差异化切分"的策略在处理结构复杂的复合型文档时非常有效。

## 自定义 Node Parser

当内置的 Parser 都不能满足需求时，你可以编写自己的 Node Parser。只需要继承 `NodeParser` 基类并实现 `get_nodes_from_documents` 方法：

```python
from typing import List, Optional
from llama_index.core import Document
from llama.index.core.node_parser import NodeParser
from llama_index.core.schema import TextNode, BaseNode


class TableAwareNodeParser(NodeParser):
    """能够识别并保留表格结构的自定义 Parser"""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 100,
        table_strategy: str = "preserve",  # "preserve" | "expand" | "flatten"
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.table_strategy = table_strategy

    def _parse_table(self, text: str) -> tuple:
        """检测并解析 Markdown 表格"""
        lines = text.split("\n")
        table_start = None
        for i, line in enumerate(lines):
            if "|" in line and "---" in lines[i + 1] if i + 1 < len(lines) else False:
                table_start = i
                break

        if table_start is None:
            return None, text  # 没有表格

        table_end = table_start + 1
        while table_end < len(lines) and "|" in lines[table_end]:
            table_end += 1

        table_text = "\n".join(lines[table_start:table_end])
        remaining_text = "\n".join(lines[:table_start] + lines[table_end:])
        return table_text, remaining_text

    def _process_table(self, table_text: str) -> List[str]:
        """根据策略处理表格"""
        if self.table_strategy == "preserve":
            return [table_text]  # 保持表格原样
        elif self.table_strategy == "expand":
            return self._expand_table_to_rows(table_text)
        elif self.table_strategy == "flatten":
            return [self._flatten_table(table_text)]
        return [table_text]

    def _expand_table_to_rows(self, table_text: str) -> List[str]:
        """将表格展开为逐行叙述"""
        rows = table_text.strip().split("\n")
        headers = [h.strip() for h in rows[0].split("|") if h.strip()]
        narratives = []
        for row in rows[2:]:  # 跳过分隔行
            cells = [c.strip() for c in row.split("|") if c.strip()]
            narrative = "；".join(
                f"{h}是{c}" for h, c in zip(headers, cells)
            )
            narratives.append(narrative)
        return narratives

    def _flatten_table(self, table_text: str) -> str:
        """将表格转为紧凑文本"""
        lines = [l.strip() for l in table_text.split("\n") if l.strip()]
        return " | ".join(lines)

    def get_nodes_from_documents(
        self,
        documents: List[Document],
        show_progress: bool = False,
    ) -> List[BaseNode]:
        all_nodes = []

        for doc in documents:
            table_text, remaining_text = self._parse_table(doc.text)

            nodes = []

            if table_text:
                table_parts = self._process_table(table_text)
                for part in table_parts:
                    nodes.append(TextNode(
                        text=part,
                        metadata={**doc.metadata, "content_type": "table"},
                    ))

            if remaining_text.strip():
                fallback_splitter = SentenceSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                )
                text_nodes = fallback_splitter.get_nodes_from_documents([
                    Document(text=remaining_text, metadata=doc.metadata)
                ])
                nodes.extend(text_nodes)

            all_nodes.extend(nodes)

        return all_nodes


# 使用
parser = TableAwareNodeParser(
    chunk_size=512,
    chunk_overlap=100,
    table_strategy="expand",  # 表格展开为逐行叙述
)
nodes = parser.get_nodes_from_documents(documents)
```

这个自定义 Parser 展示了一个实用的模式：**先用自定义逻辑处理特殊内容（表格），然后用内置的 SentenceSplitter 处理剩余的普通文本**。你不必从头实现所有的切分逻辑——站在巨人的肩膀上，只补充框架缺失的那一块即可。

## 常见误区

**误区一："chunk_size 越大越好"。** 很多初学者倾向于设置很大的 chunk_size（如 2048 或更大），认为"信息越多越好"。但实际上过大的 chunk 会引入大量无关信息，降低向量表征的质量（一个谈论十个话题的 chunk，其 embedding 无法准确反映任何一个话题），同时浪费 LLM 的上下文窗口。**正确的做法是根据查询类型和数据特征来选择合适的粒度。**

**误区二:"overlap 越大越好"。** overlap 的目的是防止边界信息丢失，不是越多越好。过大的 overlap（如 chunk_size 的 50% 以上）会导致严重的冗余——大部分内容都被重复存储和检索，增加了存储成本和检索延迟，却没有带来相应的质量提升。**overlap 一般控制在 chunk_size 的 10%-25% 之间。**

**误区三:"一种 Parser 通吃所有文档类型"。** 技术文档、法律合同、聊天记录、代码文件的"自然切分边界"完全不同。用 SentenceSplitter 切代码、用 CodeSplitter 切法律合同都不会有好结果。**根据文档类型选择合适的 Parser，或者在混合文档中使用 MetadataAwareSentenceSplitter 做差异化处理。**

**误区四:"Parser 配置好后就不需要再调整了"。** 数据会变化、查询模式会演进、新的文档类型会出现。建立定期评估机制（第八章会详细讲），监控 Parser 的切分质量并在必要时调整参数，是保证长期效果的关键。
