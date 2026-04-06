---
title: HyDE：假设性文档嵌入
description: HyDE 原理与实现、查询扩展技术、Multi-Query 检索、Decompose Transform
---
# HyDE：假设性文档嵌入

前两节我们学习了混合检索（组合多种检索方法）和重排序（对结果做精细排序）。这一节我们来探索另一种完全不同但同样强大的检索增强思路：**不是改进"怎么搜"，而是改进"搜什么"**。

具体来说，用户的原始查询往往很短、很模糊、缺乏上下文。"S1 怎么样？"——这个查询本身携带的信息量极少，任何检索系统都很难从中推断出用户的真实意图。但如果有一个"魔法"能把这个短查询扩展成一段完整的、详细的描述性文本，然后再用这段文本来做检索，效果会不会好得多？

**HyDE（Hypothetical Document Embeddings，假设性文档嵌入）** 就是这样的"魔法"——它利用 LLM 先生成一个"假想的答案文档"，然后用这个假想文档的 embedding 来代替原始查询的 embedding 进行检索。

## HyDE 的核心思想

传统检索流程：
```
用户查询: "S1 的保修政策"
    ↓
Query Embedding → [0.12, -0.34, ...]
    ↓
向量搜索 → 找到最相似的文档
```

HyDE 检索流程：
```
用户查询: "S1 的保修政策"
    ↓
LLM 生成假想答案:
"根据公司政策，智能音箱 S1 提供 24 个月的官方保修服务。
保修范围包括硬件故障和制造缺陷导致的性能问题。
用户需要在购买后 30 内完成产品注册..."
    ↓
假想答案 Embedding → [0.21, -0.45, ...]  ← 更丰富的语义表征
    ↓
向量搜索 → 找到最相似的文档（通常是更好的匹配）
```

关键洞察：**假想答案的 embedding 比原始查询的 embedding 携带了更多的语义信息**。即使 LLM 生成的假想答案内容与真实答案不完全一致，它的"语义方向"通常也比简短的查询更接近真实的答案文档。

### 为什么这有效？

想象一下你在图书馆找书。你走到咨询台说："我想找一本关于保修的书。"图书管理员可能会给你指一堆书——汽车保修、电子产品保修、房屋保修……因为你说的太笼统了。

但如果你说："我想找一本关于智能音箱产品保修政策的书，里面应该提到保修期限、保修范围和申请流程。"图书管理员就能更精准地定位到你需要的书了。

HyDE 做的事情就是帮你的查询"补充细节"——它不会凭空捏造事实（那是幻觉），而是生成一段**在语义方向上与真实答案一致的、信息密度更高的文本**。

## LlamaIndex 中的 HyDE 实现

LlamaIndex 通过 `HyDEQueryTransform` 组件原生支持 HyDE：

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.indices.query.query_transform import HyDEQueryTransform

documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)

hyde = HyDEQueryTransform(
    llm=Settings.llm,              # 用于生成假想答案的 LLM
    include_original=True,          # 同时保留原始查询
    num_hypotheticals=3,            # 生成 3 个不同的假想答案
)

query_engine = index.as_query_engine(
    transform_queries=hyde,         # 启用 HyDE 查询转换
    similarity_top_k=5,
)

response = query_engine.query("S1 的保修期是多久？")
print(response.response)
```

### 关键参数解析

**`num_hypotheticals`（假想答案数量）：** 默认为 1，即只生成一个假想答案。增加到 2-3 可以覆盖更多的语义角度，但也会增加 LLM 调用次数和检索开销。

当 `num_hypotheticals=3` 时，LlamaIndex 会：
1. 用 LLM 生成 3 个不同版本的假想答案
2. 对每个假想答案分别做 embedding
3. 分别执行 3 次向量搜索
4. 合并去重所有结果并返回 top-k

**`include_original`（是否包含原始查询）：** 设为 True 时，除了假想答案的检索结果外，还会包含原始查询的检索结果。这是一种保守策略——确保不会因为 HyDE 生成的假想答案跑偏而丢失原始查询可能找到的相关结果。

### 查看 HyDE 的内部工作过程

```python
hyde = HyDEQueryTransform(
    llm=Settings.llm,
    include_original=False,
    num_hypotheticals=1,
    verbose=True,  # 打印详细信息
)

transformed = hyde("S1 的保修期是多久？")
# verbose=True 会打印类似如下信息:
#
# === HyDE Query Transform ===
# Original query: S1 的保修期是多久？
#
# Hypothetical document #1:
# 根据公司的产品保修政策，智能音箱 S1 为用户提供自购买之日起
# 24 个月的免费保修服务。保修范围涵盖因材料缺陷或工艺问题导致
# 的功能故障。用户可通过官方客服渠道申请保修...
#
# Transformed queries:
# 1. [original] S1 的保修期是多久？
# 2. [hypothetical #1] 根据公司的产品保修政策...
```

通过 `verbose=True` 你可以看到 LLM 到底生成了什么样的假想答案——这对于调试 HyDE 的效果非常有帮助。如果发现生成的假想答案偏离了主题，可能需要调整 prompt 或换一个更强的 LLM。

## Multi-Query：多角度查询扩展

HyDE 是一种特殊的"查询扩展"技术——它把一个查询扩展为一个或多个假想文档。还有一种更直接的查询扩展方法：**Multi-Query（多查询）**——让 LLM 把原始查询改写成多个不同角度的变体。

```python
from llama_index.core.indices.query.query_transform import DecomposeQueryTransform

multi_query = DecomposeQueryTransform(
    llm=Settings.llm,
    verbose=True,
)

query_engine = index.as_query_engine(
    transform_queries=multi_query,
    similarity_top_k=5,
)

response = query_engine.query("S1 的网络连接有问题怎么办？")
```

当用户问"S1 的网络连接有问题怎么办？"时，DecomposeQueryTransform 可能会产生这样的多角度查询：

```
Original: S1 的网络连接有问题怎么办？

Generated sub-queries:
1. 智能音箱 S1 无法连接 Wi-Fi 的排查步骤是什么？
2. S1 设备显示离线状态应该如何解决？
3. S1 蓝牙连接失败的常见原因和解决方法？
4. 如何重置 S1 的网络设置以解决连接问题？
```

然后系统会对每个子查询分别执行检索，合并所有结果。这样做的好处是：
- **覆盖更多同义表达**："网络连接有问题" → "无法连接 Wi-Fi"、"显示离线"、"蓝牙连接失败"
- **分解复杂问题**：一个问题拆解为多个具体的子问题
- **提高召回率**：从多个角度搜索，降低遗漏相关信息的风险

### HyDE vs Multi-Query 的选择

| 维度 | HyDE | Multi-Query |
|------|------|-------------|
| **扩展形式** | 生成假想答案文档 | 生成多个查询变体 |
| **适用场景** | 需要"丰富描述"的查询 | 可以"多角度解读"的查询 |
| **输出长度** | 较长（段落级） | 较短（句子级） |
| **LLM 调用次数** | num_hypotheticals 次 | 通常 1 次（批量生成） |
| **语义方向性** | 强（指向答案的方向） | 中（指向问题的不同侧面） |

**经验法则：** 对于事实型、信息查找型的查询（"X 是什么？""Y 怎么做？"），HyDE 效果更好；对于开放式的、探索性的查询（"关于 X 我应该了解什么？"），Multi-Query 更合适。当然，你也可以同时使用两者。

## Step-Decompose Transform：分步分解查询

有些查询天然就是复合型的——包含多个子问题或需要多步推理。`StepDecomposeQueryTransform` 能将这类查询自动分解为有序的子查询链：

```python
from llama_index.core.indices.query.query_transform import (
    StepDecomposeQueryTransform,
)

step_decompose = StepDecomposeQueryTransform(
    llm=Settings.llm,
    verbose=True,
)

query_engine = index.as_query_engine(
    transform_queries=step_decompose,
    similarity_top_k=5,
)

response = query_engine.query(
    "比较 S1 和 S2 两款产品的功能和价格，"
    "并推荐哪款更适合家庭用户"
)
```

StepDecompose 可能产生这样的分解：

```
Original: 比较 S1 和 S2 的功能和价格，推荐适合家庭的

Step 1: S1 产品的主要功能有哪些？价格范围是多少？
  → 检索结果 A
Step 2: S2 产品的主要功能有哪些？价格范围是多少？
  → 检索结果 B
Step 3: 家庭用户选购智能音箱的主要需求是什么？
  → 检索结果 C
Step 4: 基于 A+B+C 的信息，比较并给出推荐
  → 综合回答
```

这与第五章前面讲的 SubQuestionQueryEngine 有相似之处，区别在于 StepDecompose 是在**查询转换层面**做的（改变的是发给检索器的查询），而 SubQuestionQueryEngine 是在**查询引擎层面**做的（改变的是整个问答流程）。两者可以配合使用。

## 查询转换的组合使用

你可以同时应用多种查询转换技术：

```python
from llama_index.core.indices.query.query_transform import (
    HyDEQueryTransform,
    DecomposeQueryTransform,
)
from llama_index.core.indices.query.transform import ComposableQueryTransform

combined_transform = ComposableQueryTransform(
    transforms=[
        HyDEQueryTransform(      # 第一步：生成假想答案
            llm=Settings.llm,
            num_hypotheticals=1,
        ),
        DecomposeQueryTransform(  # 第二步：多角度扩展
            llm=Settings.llm,
        ),
    ],
)

query_engine = index.as_query_engine(
    transform_queries=combined_transform,
    similarity_top_k=5,
)
```

不过要注意：**每增加一层转换就增加一次 LLM 调用和一次检索操作**。组合使用的代价是延迟和成本的线性增长。建议只在确实需要时才使用组合转换，并且始终监控其对端到端性能的影响。

## 常见误区

**误区一:"HyDE 生成的假想答案就是最终答案"。** 不是的。HyDE 生成的假想答案只是用来做检索的"中间产物"——它的作用是改善 embedding 的质量，而不是直接作为回答返回给用户。最终的答案仍然来自检索到的真实文档 + LLM 的综合生成。

**误区二:"HyDE 总是能提升检索质量"。** 不一定。当 LLM 对某个领域不够了解时，它生成的假想答案可能是错误的或误导性的，反而会把检索引向错误的方向。**HyDE 在 LLM 对查询领域有一定认知的情况下效果最好**——如果你的知识库是非常专业化的领域（如医学、法律），考虑用该领域的数据微调过 LLM 后再用于 HyDE。

**误区三:"查询转换越多越好"。** 查询转换是一把双刃剑。它能改善检索召回率，但也引入了额外的 LLM 调用延迟和成本，而且过多的子查询可能引入噪音结果。**从最简单的方案开始（不加任何转换），只有在评估中发现明确的改进空间时才逐步添加转换层。**
