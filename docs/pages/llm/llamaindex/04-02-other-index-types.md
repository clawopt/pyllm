---
title: 多种索引类型详解：ListIndex/TreeIndex/KeywordTableIndex/SummaryIndex/GraphIndex
description: 五种非向量索引的原理、适用场景、代码示例与性能分析
---
# 多种索引类型详解：ListIndex/TreeIndex/KeywordTableIndex/SummaryIndex/GraphIndex

上一节我们深入研究了 VectorStoreIndex——这个最常用也最强大的索引类型。但 LlamaIndex 的索引体系远不止向量索引一种。实际上，LlamaIndex 提供了 **6 种原生索引类型**，每种都针对不同的数据特征和查询模式做了专门的优化。

如果说 VectorStoreIndex 是一把瑞士军刀的主刀片（最常用、最锋利），那其他五种索引就是瑞士军刀上的螺丝刀、剪刀、开瓶器——它们在各自的特定场景下比主刀片更好用。

这一节我们来逐一认识这五种"特殊工具"。

## ListIndex：顺序遍历索引

`ListIndex` 是最简单的索引类型——它把所有 Node 按顺序存放在一个列表中，查询时依次遍历每个 Node。

```python
from llama_index.core import ListIndex

documents = SimpleDirectoryReader("./data").load_data()
index = ListIndex.from_documents(documents)
query_engine = index.as_query_engine()

response = query_engine.query("总结这篇文档的主要内容")
print(response.response)
```

### 工作原理

```
ListIndex 内部结构:
┌─────────────────────────────┐
│  Node[0]: "第一章 引言..."    │
│  Node[1]: "第二章 方法..."    │
│  Node[2]: "第三章 实验..."    │
│  Node[3]: "第四章 结论..."    │
│  ...                         │
│  Node[N]: "参考文献..."       │
└─────────────────────────────┘
       │
       ▼ 查询时
按顺序遍历所有 Node → 将全部内容发送给 LLM 合成答案
```

ListIndex 的查询过程非常简单粗暴：**把所有 Node 的内容一股脑塞给 LLM，让它自己从中找答案**。这听起来很傻，但对于某些类型的查询来说，这恰恰是最有效的方式。

### 适用场景

**场景一：全文摘要。** 当用户的查询本质上是"告诉我这篇文章讲了什么"时，LLM 确实需要看到全部内容才能给出好的摘要。VectorStoreIndex 只会返回 top-k 个最相似的 chunk，很可能遗漏了文章的重要部分。

**场景二：短文档集合。** 如果你的知识库总共只有几十个 Node（比如几篇短文章），遍历全部内容的开销完全可以接受。这时候 ListIndex 比 VectorStoreIndex 更简单可靠。

**场景三：需要全局信息的查询。** 如"这篇文章的主要论点有哪些？""作者的观点前后是否有矛盾？"这类需要纵观全文才能回答的问题。

### 不适用的场景

- 文档数量大（>100 个 Node）——每次查询都要读取全部内容，太慢且浪费 token
- 需要精确查找具体信息——"API 的超时参数是多少？"这类查询用 ListIndex 是杀鸡用牛刀
- 有严格的响应时间要求——ListIndex 的查询时间随文档数量线性增长

### 性能特征

| 指标 | 表现 |
|------|------|
| 构建速度 | ⚡⚡⚡⚡⚡ 最快（无需计算 embedding） |
| 查询速度 | 🐢 随文档数量线性增长 |
| 存储占用 | ⚡⚡⚡⚡ 较小（只存原文） |
| 摘要类查询质量 | ⭐⭐⭐⭐⭐ 最佳 |
| 精确查找质量 | ⭐ 很差 |

## TreeIndex：层级摘要树索引

`TreeIndex` 把文档组织成一棵树形结构，其中每个节点是其子节点的摘要。查询时从根节点出发，沿着树向下导航到最相关的叶子节点。

```python
from llama_index.core import TreeIndex

documents = SimpleDirectoryReader("./books/novel.txt").load_data()
index = TreeIndex.from_documents(documents, num_children=10)
query_engine = index.as_query_engine()

response = query_engine.query("主角在第十章做了什么决定？")
```

### 工作原理

TreeIndex 的构建过程如下：

```
原始文档（假设有 100 个 Node）

Layer 0 (根):
┌──────────────────────────────────┐
│ 摘要: 这是一部关于...的小说，     │
│ 包含 A、B、C 三条故事线...        │
└──────────┬───────────────────────┘
           │
    ┌──────┼──────┐
    ▼      ▼      ▼
Layer 1 (3个中间节点):
┌──────────┐ ┌──────────┐ ┌──────────┐
│ 故事线A  │ │ 故事线B  │ │ 故事线C  │
│ 的摘要   │ │ 的摘要   │ │ 的摘要   │
└────┬─────┘ └────┬─────┘ └────┬─────┘
     │            │            │
  ... (继续分裂直到叶子节点) ...

Layer N (叶子节点):
┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐
│N1  │ │N2  │ │N3  │ │... │ │N10 │
└────┘ └────┘ └────┘ └────┘ └────┘
(原始 Node 或小的 Node 组)
```

查询时的导航过程：

```
Query: "主角在第十章的决定"
       │
       ▼
根节点摘要: "...包含 A、B、C 三条故事线..."
       │
       ▼ LLM 判断: 与哪条故事线最相关？→ 故事线A
       │
       ▼
中间节点A摘要: "故事线A涵盖第1-15章..."
       │
       ▼ LLM 判断: 第十章属于哪个子区间？→ 第8-12章
       │
       ▼
叶子节点(第8-12章的内容)
       │
       ▼
最终答案
```

### 关键参数：num_children

`num_children` 控制每个父节点有多少个子节点。较小的值（如 3-5）会产生更高更窄的树（更多层级，每层的摘要更聚焦）；较大的值（如 10-20）会产生更矮更宽的树（更少的层级，但每层的摘要覆盖面更大）。

```python
# 窄而深的树 — 适合层级清晰的文档（如书籍、法规）
index = TreeIndex.from_documents(docs, num_children=4)

# 宽而浅的树 — 适合扁平但较长的文档
index = TreeIndex.from_documents(docs, num_children=20)
```

### 适用场景

- **书籍、长篇报告、法规文件**等具有天然层级结构的文档
- 需要"由粗到细"浏览式探索的查询模式
- 文档较长但需要控制单次查询的 token 消耗

### 局限性

- **构建成本高：** 需要对每一层生成摘要，涉及大量的 LLM 调用
- **导航可能出错：** 如果某层的摘要不够准确，可能导致后续导航走错分支
- **不适合频繁更新：** 任何文档变更都可能需要重建整棵树

## KeywordTableIndex：关键词倒排索引

`KeywordTableIndex` 是传统的信息检索技术在 LlamaIndex 中的实现——它从每个 Node 中提取关键词，建立"关键词 → Node 列表"的倒排索引，查询时通过关键词匹配来定位相关 Node。

```python
from llama_index.core import KeywordTableIndex
from llama_index.core.node_parser import KeywordNodeParser

parser = KeywordNodeParser(
    keywords=5,            # 每个 Node 提取前 5 个关键词
)
nodes = parser.get_nodes_from_documents(documents)

index = KeywordTableIndex(nodes=nodes)
query_engine = index.as_query_engine()

response = query_engine.query("退款政策中的退货流程")
print(response.response)
```

### 工作原理

```
关键词提取阶段:
Node A ("退款政策说明...") → 关键词: [退款, 退货, 流程, 政策, 申请]
Node B ("产品规格参数...")  → 关键词: [产品, 规格, 参数, 尺寸, 重量]
Node C ("安装指南...")      → 关键词: [安装, 指南, 步骤, 设置, 配置]

倒排索引:
┌──────────┬───────────────────────┐
│ 关键词   │ Node 列表             │
├──────────┼───────────────────────┤
│ 退款     │ [A]                   │
│ 退货     │ [A]                   │
│ 流程     │ [A, C]                │
│ 产品     │ [B]                   │
│ 安装     │ [C]                   │
│ ...      │                       │
└──────────┴───────────────────────┘

查询阶段:
Query: "退款政策中的退货流程"
    ↓ 提取查询关键词: [退款, 退货, 匹配, 政策, 流程]
    ↓ 在倒排索引中查找
    ↓ 返回命中最多的 Node: A (命中 4 个关键词)
```

### 适用场景

- **精确关键词匹配**——用户查询和文档中都出现相同的关键词
- **专业术语密集的领域**——法律、医学、技术文档中有大量标准化术语
- **作为 VectorStoreIndex 的补充**——弥补向量搜索在精确术语匹配上的不足

### 与 VectorStoreIndex 的互补关系

这是一个非常重要的实践模式：**KeywordTableIndex 和 VectorStoreIndex 互相弥补对方的弱点**。

| 场景 | VectorStoreIndex | KeywordTableIndex |
|------|------------------|-------------------|
| "退货怎么办"（口语化） | ✅ 强项 | ❌ 弱项 |
| "依据《消费者权益保护法》第二十四条"（精确法条引用） | ⚠️ 可能遗漏 | ✅ 强项 |
| "API 返回 503 错误码"（含精确编号） | ⚠️ 可能被稀释 | ✅ 强项 |
| "系统整体运行缓慢的原因分析"（需要语义理解） | ✅ 强项 | ❌ 弱项 |

第五章的"混合检索"部分会讲解如何同时使用两种索引来获得两者的优势。

## SummaryIndex：全局摘要索引

`SummaryIndex` 为整个文档集生成一个全局性的摘要，适用于"大局观"类的查询。

```python
from llama_index.core import SummaryIndex

documents = SimpleDirectoryReader("./reports/q4_report.pdf").load_data()
index = SummaryIndex.from_documents(documents)
query_engine = index.as_query_engine()

response = query_engine.query("这份季度报告的核心结论是什么？")
```

### 工作原理

```
构建阶段:
所有 Documents → LLM 生成一份全局摘要
                → 存储为 SummaryIndex 的核心内容

查询阶段:
用户提问 → 全局摘要 + 问题 → LLM 生成答案
```

SummaryIndex 和 ListIndex 很像——都是把全部（或大部分）内容交给 LLM 处理。区别在于 SummaryIndex **预先计算了一份摘要**，使得查询时不需要传输全部原始文本，节省了 token 消耗。

### 适用场景

- **仪表板概览**——"我们这个季度的整体表现如何？"
- **快速浏览**——用户想先了解文档的大致内容再决定是否深入阅读
- **监控告警摘要**——"过去 24 小时最重要的系统事件是什么？"

### 注意事项

SummaryIndex 的摘要质量**严重依赖于 LLM 的能力**。对于特别长的文档集（超过 LLM 上下文窗口限制的），摘要可能会丢失重要细节。此外，当文档更新后，摘要需要重新生成——这是一笔不小的 LLM 调用成本。

## GraphIndex：知识图谱索引

`GraphIndex` 是最复杂也最强大的索引类型之一。它从文档中提取实体（entities）和关系（relationships），构建成知识图谱，然后支持基于图谱结构的查询。

```python
from llama_index.core import KnowledgeGraphIndex
from llama_index.core.graph_stores import SimpleGraphStore

graph_store = SimpleGraphStore()
index = KnowledgeGraphIndex.from_documents(
    documents,
    graph_store=graph_store,
    include_text=True,  # 同时保留原始文本
)

# 查询示例
query_engine = index.as_query_engine(
    include_text=True,  # 答案中包含原始文本证据
)

response = query_engine.query("张三是谁的上司？")
```

### 工作原理

```
知识抽取阶段:
文档: "张三是产品部的经理，李四是他的下属。
       李四负责 S1 产品的开发工作。"
       ↓
实体: [张三, 李四, 产品部, 经理, 下属, S1 产品, 开发]
关系: [张三 --经理→ 产品部,
       张三 --上司→ 李四,
       李四 --下属→ 张三,
       李四 --负责→ S1 产品]

图谱存储:
  张三 ──(经理)──▶ 产品部
   │
   ├──(上司)──▶ 李四 ──(负责)──▶ S1 产品
   │               │
   └──(下属)◀──    └──(开发)

查询阶段:
"张三是谁的上司？"
    ↓ 图谱查询: 找到 张三 --上司→ X
    ↓ 结果: X = 李四
    ↓ 结合原始文本生成自然语言回答
```

### 适用场景

- **实体关系查询**——"A 公司的 CEO 是谁？""产品和部门之间是什么关系？"
- **多跳推理**——"张三的下属负责的产品有什么功能？"（需要两跳：张三→李四→S1产品→功能）
- **领域知识建模**——医疗诊断、法律推理、供应链追踪等需要结构化知识的场景

### 局限性

- **构建复杂度高：** 需要高质量的实体和关系抽取（通常依赖 LLM），成本较高
- **图谱维护困难：** 文档更新后需要增量更新图谱，处理冲突和一致性不简单
- **对抽取质量敏感：** 如果实体识别或关系抽取出错，整个图谱的质量都会受影响

## 索引类型选择决策树

面对一个具体的 RAG 需求，如何选择合适的索引类型？以下是实用的决策流程：

```
你的主要查询类型是什么？
       │
       ├─ 精确事实查找（"X 是多少？"）
       │    → VectorStoreIndex + 元数据过滤
       │
       ├─ 全局概括（"这篇文章讲什么？"）
       │    → SummaryIndex 或 ListIndex（文档少时）
       │
       ├─ 关系推理（"A 和 B 是什么关系？"）
       │    → GraphIndex
       │
       ├─ 关键词精确匹配（"找到所有提到 XXX 的段落"）
       │    → KeywordTableIndex
       │
       ├─ 层级浏览（"这本书的第3章讲了什么？"）
       │    → TreeIndex
       │
       └─ 以上都有（最常见的实际情况！）
            → 组合使用（下一节详细讲）
```

记住：**在实际项目中，很少只用一种索引类型**。最强大的方案是根据查询类型动态路由到最合适的索引——这正是第五章要讲的"高级检索技术"的核心内容之一。
