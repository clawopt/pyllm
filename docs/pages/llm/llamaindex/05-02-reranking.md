---
title: 重排序（Reranking）：从粗排到精排
description: Reranker 原理、Cross-Encoder vs Bi-Encoder、Cohere Rerank / bge-reranker 集成、多阶段检索管道
---
# 重排序（Reranking）：从粗排到精排

在上一节的混合检索中，我们讨论了如何结合向量搜索和关键词搜索来获得更好的候选结果。但无论你的初始检索（粗排）做得多好，返回的 top-k 结果中通常都会混入一些**看似相关但实际上不相关**的条目——可能是因为某些文档碰巧包含了查询中的几个高频词，或者其向量表示恰好与查询向量在空间中距离较近。

**重排序（Reranking）** 就是为了解决这个问题而设计的"第二道关卡"：它对粗排返回的候选结果进行更精细的重新评分和排序，把真正相关的结果推到前面，把噪音结果压到后面或直接过滤掉。

## 为什么需要重排序？

让我们用一个直观的例子来说明。假设用户问："S1 产品的蓝牙连接支持哪些协议？"

### 粗排阶段（向量搜索 top-10）

| 排名 | 文档片段 | 相似度分数 | 真正相关？ |
|------|---------|-----------|-----------|
| 1 | "S1 支持 Bluetooth 5.3 协议..." | 0.92 | ✅ 完美匹配 |
| 2 | "S1 的 Wi-Fi 配置说明..." | 0.88 | ⚠️ 提到了 S1 但不是蓝牙 |
| 3 | "所有产品都支持蓝牙连接..." | 0.85 | ⚠️ 太泛泛，没有具体信息 |
| 4 | "S2 产品的蓝牙配对教程..." | 0.83 | ❌ 是 S2 不是 S1 |
| 5 | "S1 支持的无线协议列表：Wi-Fi 6、Bluetooth 5.3、Zigbee..." | 0.81 | ✅ 相关但排名太低！ |
| 6 | "蓝牙技术的发展历史..." | 0.78 | ❌ 无关背景知识 |
| 7 | "S1 的包装清单..." | 0.75 | ❌ 不相关 |
| 8 | "S1 的固件更新日志 v2.1：修复了蓝牙连接问题..." | 0.73 | ⚠️ 间接相关 |
| 9 | "S1 产品对比表：S1 vs S2 vs S3" | 0.71 | ⚠️ 可能包含有用信息 |
| 10 | "客户评价：S1 的音质很好但蓝牙偶尔断连" | 0.69 | ⚠️ 用户反馈非官方信息 |

注意看排名第 5 的结果——它实际上包含了用户需要的完整答案（Bluetooth 5.3），但因为它的表述方式与查询不完全一致（用了"无线协议列表"而非直接说"蓝牙"），在向量相似度上反而排到了第 5 位。而排名第 2 和第 3 的结果虽然分数更高，但包含的信息量远不如第 5。

### 重排序之后

经过一个高质量的重排序模型处理后：

| 排名 | 文档片段 | 重排分数 | 变化 |
|------|---------|---------|------|
| **1** | "S1 支持的无线协议列表：Wi-Fi 6、Bluetooth 5.3、Zigbee..." | **0.98** | ↑ 从第 5 升到第 1 |
| 2 | "S1 支持 Bluetooth 5.3 协议..." | 0.95 | ↓ 从第 1 降到第 2 |
| 3 | "S1 产品对比表：S1 vs S2 vs S3" | 0.82 | ↑ 从第 9 升到第 3 |
| 4 | "所有产品都支持蓝牙连接..." | 0.72 | ↓ 从第 3 降到第 4 |
| ... | (其余被压制) | < 0.5 | — |

最相关的结果（第 5 名）被提升到了第 1 名！这就是重排序的价值——**它用更精确但更慢的模型来纠正粗排阶段的错误排序。**

## Bi-Encoder 与 Cross-Encoder 的区别

要理解重排序的工作原理，需要先理解两种不同的编码器架构：

### Bi-Encoder（双编码器）— 用于粗排

这就是我们在向量搜索中使用的嵌入模型（如 `text-embedding-3-small`）：

```
Query: "S1 蓝牙协议"
        │
        ▼  Embedding Model
   [0.12, -0.34, ...]    ← Query 向量 (独立编码)

Doc: "S1 支持 Bluetooth 5.3..."
        │
        ▼  同一个 Embedding Model
   [0.15, -0.31, ...]    ← Doc 向量 (独立编码)

相似度 = cosine(query_vec, doc_vec)
```

关键特征：
- Query 和 Document **分别独立编码**
- 编码后的向量可以**预先计算并存储**（这就是为什么向量搜索很快）
- 但 Query 和 Document 之间**没有交互**——模型在编码 Doc 时不知道 Query 具体问了什么

### Cross-Encoder（交叉编码器）— 用于精排/重排

```
Input: "[CLS] S1 蓝牙协议 [SEP] S1 支持 Bluetooth 5.3... [SEP]"
        │
        ▼ Cross-Encoder Model (如 BERT)
   注意力机制让 Query 和 Doc 的每个 token 都能相互关注
        │
        ▼ Output: 0.98 (相关性得分)
```

关键特征：
- Query 和 Document **一起输入同一个模型**
- 模型的注意力机制能让 Query 中的每个词与 Document 中的每个词**充分交互**
- 因此能捕捉更细粒度的语义匹配关系
- **但不能预计算**——每次查询都需要对每个候选文档运行一次推理

### 性能与质量的权衡

| 特性 | Bi-Encoder (粗排) | Cross-Encoder (精排) |
|------|-------------------|---------------------|
| 计算时机 | 可离线预处理 | 必须在线实时计算 |
| 单次查询延迟 | ~10ms (向量搜索) | ~50-200ms / 候选文档 |
| 候选集规模 | 全部文档 (百万级) | 通常 20-50 个 (粗排结果) |
| 匹配质量 | 较好（语义级别） | 优秀（细粒度 token 级别） |
| 适用场景 | 第一轮筛选 | 第二轮精细排序 |

这就是为什么实际系统采用**两阶段检索**：先用快速的 Bi-Encoder 从海量数据中筛选出几十个候选，再用精确但较慢的 Cross-Encoder 对这几十个候选做精细排序。

## LlamaIndex 中的 Reranker 使用

LlamaIndex 通过**后处理器（Postprocessor）**机制来集成 Reranker。使用方式非常简洁：

```python
from llama_index.core.postprocessor import CohereRerank
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine(
    similarity_top_k=20,       # 粗排多取一些（给 rerank 留余地）
    node_postprocessors=[
        CohereRerank(
            model="rerank-v3.5",     # Cohere 最新的 rerank 模型
            top_n=5,                  # rerank 后只保留 top 5
            api_key=os.getenv("COHERE_API_KEY"),
        ),
    ],
)

response = query_engine.query("S1 支持哪些蓝牙协议？")
print(response.response)
```

注意这里的关键参数变化：
- `similarity_top_k=20`：粗排阶段取 20 个候选（比平时多）
- `CohereRerank(top_n=5)`：rerank 后只保留最好的 5 个
- 中间的 15 个候选被 reranker 过滤掉了

### NodePostprocessor 管道

`node_postprocessors` 参数接受一个处理器列表，它们会按顺序组成一个处理管道：

```python
query_engine = index.as_query_engine(
    similarity_top_k=30,
    node_postprocessors=[
        # Stage 1: 过滤掉分数太低的（可选）
        SimilarityPostProcessor(similarity_cutoff=0.5),

        # Stage 2: 重排序（核心）
        CohereRerank(model="rerank-v3.5", top_n=10),

        # Stage 3: 去除重复内容（可选）
        DeduplicateNodePostprocessor(),
    ],
)
```

这种管道式设计让你可以灵活地组合各种后处理操作。

## 主流 Reranker 模型选择

### Cohere Rerank（商业 API）

Cohere 提供了目前业界效果最好的 rerank 模型之一：

```python
from llama_index.postprocessor.cohere_rerank import CohereRerank

reranker = CohereRerank(
    api_key=os.getenv("COHERE_API_KEY"),
    model="rerank-v3.5",      # 最新版本
    top_n=5,
)
```

**优势：**
- 效果顶尖，尤其在多语言场景下
- API 简单易用，无需本地部署 GPU
- 持续优化和更新

**劣势：**
- 商业服务，按调用次数计费（约 $1/1000 次搜索）
- 数据需要发送到 Cohere 的服务器
- 有速率限制（免费层较严格）

### bge-reranker（开源本地部署）

如果因为成本、隐私或延迟原因无法使用云服务，开源的 `bge-reranker` 是最佳替代方案：

```bash
pip install llama-index-postprocessor-flag-embedding-reranker sentence-transformers
```

```python
from llama_index.postprocessor.flag_embedding_reranker import (
    FlagEmbeddingReranker,
)

reranker = FlagEmbeddingReranker(
    model="BAAI/bge-reranker-v2-m3",  # 多语言版本
    top_n=5,
    use_async=True,  # 异步加速
)
```

**bge-reranker 系列模型对比：**

| 模型 | 语言 | 参数量 | 推荐硬件 | 相对质量 |
|------|------|--------|---------|---------|
| `bge-reranker-base` | 中英 | ~278M | CPU 即可 | ★★★☆☆ |
| `bge-reranker-large` | 中英 | ~560M | 推荐 GPU | ★★★★☆ |
| `bge-reranker-v2-m3` | 多语言 | ~568M | 推荐 GPU | ★★★★★ |

对于纯中文场景，也可以考虑 `maidalun1024/bce-reranker-base_v1-zh`（中文专用，参数量更小）。

### 自定义 Reranker

如果你有特殊需求（比如使用了微调过的交叉编码器），可以实现自定义的 Reranker：

```python
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.postprocessor import BaseNodePostprocessor
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch


class CustomReranker(BaseNodePostprocessor):
    """基于自定义 Cross-Encoder 模型的 Reranker"""

    def __init__(self, model_name: str, top_n: int = 5):
        super().__init__()
        self.top_n = top_n
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        if torch.cuda.is_available():
            self.model.cuda()

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: QueryBundle | None = None,
    ) -> list[NodeWithScore]:
        if not nodes or not query_bundle:
            return nodes

        query = query.query_str if query_bundle else ""
        pairs = [(query, node.node.text) for node in nodes]

        inputs = self.tokenizer(
            pairs, padding=True, truncation=True,
            max_length=512, return_tensors="pt",
        )
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            scores = self.model(**inputs).logits.squeeze(-1)

        scored_nodes = [
            NodeWithScore(node=n.node, score=s.item())
            for n, s in zip(nodes, scores)
        ]
        scored_nodes.sort(key=lambda x: x.score, reverse=True)

        return scored_nodes[:self.top_n]


# 使用
reranker = CustomReranker(
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_n=5,
)
query_engine = index.as_query_engine(
    similarity_top_k=20,
    node_postprocessors=[reranker],
)
```

## Reranker 的性能优化策略

**策略一：减少候选数量。** Reranker 的耗时与候选数量成正比。如果你确定粗排的质量已经不错（比如用了混合检索），可以把 `similarity_top_k` 从 20 降到 10，这样 reranker 只需处理一半的候选。

**策略二：异步执行。** 如果使用本地 reranker 模型且有多核 CPU 或 GPU，启用异步并行可以显著加速：

```python
reranker = FlagEmbeddingReranker(
    model="BAAI/bge-reranker-v2-m3",
    top_n=5,
    use_async=True,          # 异步执行
    batch_size=32,           # 批量大小
)
```

**策略三：缓存 rerank 结果。** 对于相同或高度相似的查询，缓存 rerank 的结果可以避免重复计算：

```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=1000)
def cached_rerank(query_hash, doc_hashes_tuple):
    """带缓存的 rerank"""
    return actual_rerank_function(...)
```

## 常见误区

**误区一:"有了 reranker 就不需要好的粗排了"。** 不对。Reranker 的效果受限于输入的候选质量——如果粗排阶段就把真正的相关文档漏掉了（没进入 top-k），reranker 再强大也无能为力。**粗排负责"不遗漏"，reranker 负责"排精准"，两者缺一不可。**

**误区二:"reranker 越大越好"。** 不是的。更大的模型确实可能带来更好的效果，但边际收益递减。而且更大意味着更高的延迟和成本。在实际项目中，`bge-reranker-large` 或 `Cohere rerank-v3.5` 已经能在大多数场景下提供接近最优的效果。

**误区三:"所有查询都需要 reranker"。** 对于简单的、有明确答案的事实型查询（"S1 的价格是多少？"），纯向量搜索可能就已经足够好了。Reranker 在以下场景中价值最大：查询模糊、答案分散在多个文档中、需要对多个候选做精细区分。**根据查询类型动态决定是否启用 reranker**是高级优化的方向。
