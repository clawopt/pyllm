---
title: 四种合成模式深度解析
description: REFINE / COMPACT_ACCUMULATE / TREE_SUMMARIZE / SIMPLE_SUMMARIZE 的内部机制、Prompt 结构、Token 消耗分析与选择指南
---
# 四种合成模式深度解析

上一节我们建立了对响应合成的基本认知。这一节来深入 LlamaIndex 的四种内置合成模式——逐个拆解它们的内部工作机制、使用的 Prompt 模板、Token 消耗特征以及各自的适用边界。

## SIMPLE_SUMMARIZE：最简单直接的模式

**一句话概括：** 把所有检索到的 Node 文本拼接在一起，一次性发给 LLM 生成答案。

### 工作流程

```
Nodes: [N1, N2, N3, N4, N5]
Query: "退货政策是什么？"

Step 1: 拼接所有 Node 文本
Context = N1.text + "\n\n" + N2.text + "\n\n" + ... + N5.text

Step 2: 组装 Prompt
Prompt = """
[System] 你是一个有帮助的助手。请根据以下上下文回答问题。
上下文:
{Context}

问题: {Query}
"""

Step 3: 调用 LLM 一次 → 得到答案
```

### 实际代码

```python
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.response_synthesizers import ResponseMode

synthesizer = get_response_synthesizer(
    response_mode=ResponseMode.SIMPLE_SUMMARIZE,
    use_async=False,
)

query_engine = index.as_query_engine(
    response_synthesizer=synthesizer,
    similarity_top_k=5,
)
```

### 优点
- **最快：** 只需 1 次 LLM 调用
- **最简单：** 逻辑直观，没有复杂的中间状态
- **最低延迟：** 单次 API 往返即可得到答案

### 局限性
- **上下文窗口压力：** 所有 Node 文本全部进入 Prompt，Node 多了容易超限
- **信息稀释：** 当 Node 数量较多时，重要信息被大量文本淹没，LLM 难以聚焦
- **无渐进优化：** 一次性生成，没有机会根据后续信息修正前面的表述

### 适用场景
- Node 数量少（≤ 5 个）
- 追求速度优先于答案深度
- 查询类型简单（事实型单问题）

---

## REFINE：迭代精炼模式（默认推荐）

**一句话概括：** 先用第一个 Node 生成初始答案，然后用后续 Node 逐一精炼完善。

### 工作流程

```
Nodes: [N1, N2, N3, N4, N5]
Query: "退货政策是什么？"

═══ Iteration 1: 初始生成 ═══
Input:
  System: 你是一个有帮助的助手...
  Context: [N1.text]
  Question: 退货政策是什么？

Output (Initial Answer):
  "公司提供30天无理由退款服务。"

═══ Iteration 2: 第一次精炼 ═══
Input:
  我们有一个关于"退货政策是什么？"的现有答案：
  "公司提供30天无理由退款服务。"
  
  以下是新的参考信息（可能有助于改进答案）：
  "[N2.text: 登录账户进入订单管理页面...]"
  
  如果新信息有用，请精炼原有答案；如果没有用，保持不变。

Output (Refined Answer):
  "公司提供30天无理由退款服务。具体操作为登录账户进入
   订单管理页面，点击对应订单的'申请退款'按钮..."

═══ Iteration 3: 第二次精炼 ═══
... (继续用 N3 精炼)

═══ Iteration 4: 第三次精炼 ═══
... (继续用 N4 精炼)

═══ Iteration 5: 第四次精炼 ═══
... (继续用 N5 精炼)

Final Output: 经过 5 轮精炼后的完整答案
```

### 实际代码

```python
synthesizer = get_response_synthesizer(
    response_mode=ResponseMode.REFINE,
    verbose=True,  # 打印每次迭代的中间结果
)
```

### Token 消耗分析

假设每个 Node 平均 300 tokens，共 5 个 Node：

| 迭代 | 输入 tokens (约) | 输出 tokens (约) |
|------|-----------------|-----------------|
| 初始生成 | ~400 (system + N1 + query) | ~100 |
| 精炼 1 | ~700 (system + answer + N2 + query) | ~150 |
| 精炼 2 | ~850 (system + longer_answer + N3 + query) | ~120 |
| 精炼 3 | ~950 | ~80 |
| 精炼 4 | ~1000 | ~60 |
| **总计** | **~3900** | **~510** |

注意随着迭代进行，输入 token 逐渐增加（因为答案在不断变长），这也是 REFINE 模式在 Node 很多时成本较高的原因之一。

### 优点
- **充分利用每个 Node：** 每个 Node 都有机会影响最终答案
- **答案逐步丰富：** 从粗到细，层次分明
- **天然支持长答案：** 答案随迭代自然增长

### 局限性
- **成本较高：** N 个 Node = 1(初始) + N-1(精炼) = N 次 LLM 调用
- **顺序依赖：** 后面的 Node 可能过度受前面答案的影响方向（路径依赖）
- **首 Node 效应偏差：** 如果第一个 Node 质量差，初始答案跑偏，后续精炼可能难以纠正

### 适用场景
- Node 数量中等（5-20 个）
- 需要综合多个信息源的复杂问答
- 答案需要有层次感和完整性

---

## COMPACT_ACCUMULATE：压缩累积模式

**一句话概括：** 尝试把所有 Node 塞进上下文窗口；如果超限则压缩/截断后再生成。

### 工作流程

```
Nodes: [N1, N2, ..., N20]  (假设有 20 个 Node)
Query: "退货政策是什么？"
Context Window Limit: 4096 tokens

Step 1: 尝试拼接所有 Node
All_Text = N1 + N2 + ... + N20
Total = 8000 tokens → 超过限制！

Step 2: 自动压缩
Option A: 截断 — 只保留前 N 个 Node 使总 token < 4096
Option B: 用 LLM 总结每个 Node 为更短的版本
Option C: 按重要性排序后取 top-K

Step 3: 用压缩后的上下文生成答案
```

### 实际代码

```python
synthesizer = get_response_synthesizer(
    response_mode=ResponseMode.COMPACT_ACCUMULATE,
)
```

### 与 SIMPLE_SUMMARIZE 的区别

SIMPLE_SUMMARIZE 直接拼接所有文本，超了就报错或截断。COMPACT_ACCUMULATE 会**智能地处理超限情况**——通过内部机制（如 `compact()` 方法）将文本压缩到上下文窗口内。

### 优点
- **自适应性强：** 不管来多少 Node 都能处理
- **容错性好：** 不会因为 Node 太多而崩溃
- **相对高效：** 通常只需要 1-2 次 LLM 调用

### 局限性
- **压缩可能丢失信息：** 截断或摘要过程中可能丢掉关键细节
- **压缩本身消耗资源：** 压缩步骤可能涉及额外的 LLM 调用
- **难以预测行为：** 同样的输入在不同运行中可能因压缩策略差异而有不同输出

### 适用场景
- Node 数量不确定（可能多也可能少）
- 需要一个"万能"方案不想手动切换模式
- 对答案深度要求不是极高

---

## TREE_SUMMARIZE：树状汇总模式

**一句话概括：** 先对每个 Node 生成摘要，再递归合并摘要，直到得到最终答案。

### 工作流程

```
Nodes: [N1, N2, N3, N4, N5, N6, N7, N8]

Level 0 (原始 Node): 8 个 Node

Level 1 (Node 级摘要):
┌──────────────────────────────────────┐
│ N1 摘要: "公司提供30天退款..."      │
│ N2 摘要: "退款需通过订单管理页面..."  │
│ N3 摘要: "定制产品例外..."          │
│ N4 摘要: "S2 退款流程..."           │
│ N5 摘要: "退款时效3-5个工作日..."    │
│ N6 摘要: "部分商品不支持..."        │
│ N7 摘要: "退款原因选项..."          │
│ N8 摄要: "特殊情况处理..."          │
└──────────────────────────────────────┘
       (8 个摘要)

Level 2 (两两合并):
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│ S1: N1+N2 │ │ S2: N3+N4 │ │ S3: N5+N6 │ │ S4: N7+N8 │
│ "退款政策  │ │ "例外与    │ │ "时效与    │ │ "原因与    │
│  与流程"  │ │ 其他产品"  │ │ 限制"     │ │ 特殊情况"  │
└──────────┘ └──────────┘ └──────────┘ └──────────┘
       (4 个合并摘要)

Level 3 (再次合并):
┌──────────────────┐ ┌──────────────────┐
│ M1: S1+S2        │ │ M2: S3+S4        │
│ "完整的退款政策"  │ │ "补充信息"        │
└──────────────────┘ └──────────────────┘
       (2 个合并摘要)

Level 4 (最终合并):
┌──────────────────────────────────────┐
│ Final: M1+M2                         │
│ "完整的退货政策说明（含流程、例外、  │
│  时效、原因等所有方面）"             │
└──────────────────────────────────────┘
```

### 实际代码

```python
from llama_index.core.response_synthesizers import TreeSummarize

synthesizer = get_response_synthesizer(
    response_mode=ResponseMode.TREE_SUMMARIZE,
    verbose=True,  # 打印每一层的摘要
)
```

### Token 消耗分析

对于 8 个 Node 的 TREE_SUMMARIZE：

| 层级 | LLM 调用次数 | 每次 tokens (约) | 总 tokens (约) |
|------|------------|-----------------|---------------|
| Node 摘要 | 8 | ~200 (每个 Node → 一句话摘要) | ~1600 |
| Level 2 合并 | 4 | ~350 (两个摘要 → 一个合并摘要) | ~1400 |
| Level 3 合并 | 2 | ~500 | ~1000 |
| Level 4 合并 | 1 | ~600 | ~600 |
| **总计** | **15** | | **~4600** |

注意：总 LLM 调用次数 = 2N-1（N 个 Node），但**每次调用的输入都相对较小**（因为处理的是摘要而非原文）。这与 REFINE 形成了有趣的对比：REFINE 是次数少但每次输入大，TREE_SUMMARIZE 是次数多但每次输入小。

### 优点
- **适合大规模 Node：** 即使有几十个 Node 也能有效处理
- **结构化输出：** 天然产生层次清晰的答案
- **抗噪音强：** 摘要过程会自然过滤掉不重要的细节

### 层限性
- **最高成本：** LLM 调用次数最多
- **可能过度抽象：** 多层摘要后可能丢失具体的数字或细节
- **不适合精确事实查询：** "价格是多少？"这种需要精确数字的回答，经过多层摘要后可能变得模糊

### 适用场景
- Node 数量很多（>15 个）
- 需要"全局概览"类的答案（"这篇文章主要讲了什么？"）
- 需要对大量信息做结构化总结

## 模式选择决策树

```
你的检索结果有多少个 Node？
       │
       ├─ ≤ 3 个
       │    → SIMPLE_SUMMARIZE（最快最简单）
       │
       ├─ 4-10 个
       │    → REFINE（默认推荐，效果最好）
       │
       ├─ 11-20 个
       │    → REFINE 或 COMPACT_ACCUMULATE
       │    （REFINE 如果答案需要深度综合性；
       │      COMPACT 如果想控制成本）
       │
       ├─ 21-50 个
       │    → COMPACT_ACCUMULATE 或 TREE_SUMMARIZE
       │    （COMPACT 如果大部分 Node 相关；
       │      TREE 如果需要结构化总结）
       │
       └─ > 50 个
            → TREE_SUMARIZE（唯一能优雅处理的模式）
               或考虑先做 rerank 减少 Node 数量再用 REFINE
```

## 常见误区

**误区一:"模式选错了会导致报错"。** 不会。任何模式都能在任何数量的 Node 下工作——只是效果有好有坏。SIMPLE_SUMMARIZE 给它 50 个 Node 也能跑（只是答案质量可能很差），TREE_SUMMARIZE 给它 2 个 Node 也能跑（只是浪费了几次额外的 LLM 调用）。

**误区二:"REFINE 的迭代顺序会影响最终答案"。** 是的，这是一个已知的行为特性。Node 的顺序由 Retriever 决定（通常按相关性分数排序），REFINE 按此顺序依次精炼。如果最重要的信息排在最后的 Node 中，它对答案的影响可能被削弱。**解决方案：先用 Reranker 确保最相关的 Node 排在前面。**

**误区三:"四种模式已经覆盖所有需求。"** 对于某些特殊需求（如需要严格按照特定格式输出 JSON、需要在合成过程中查询外部 API 等），内置模式可能不够用。这时就需要自定义 Synthesizer（上一节的框架代码可以作为起点）。
