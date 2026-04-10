# 3.1 Embedding Function 机制

> **从文本到向量——理解 Chroma 的 Embedding Function 是如何把你的文字变成机器能理解的数字的**

---

## 这一节在讲什么？

在前两章的示例中，你可能注意到了一个有趣的现象：我们调用 `collection.add(documents=["一些文本"])` 时，从来没有手动传入过向量——Chroma 自动就把文本变成了向量。这背后就是 Embedding Function（嵌入函数）在工作。它就像一个翻译官，把人类可读的文本翻译成机器可计算的浮点数数组，使得"语义相近的文本在向量空间中距离更近"这一核心假设得以成立。

这一节我们要深入理解 Embedding Function 的工作机制：Chroma 默认用了什么模型、如何切换到其他 embedding provider、自定义 Embedding Function 的接口契约是什么、以及不同 provider 在精度、速度、成本之间的权衡。这些知识不仅影响你日常使用 Chroma 的方式，也是面试中关于向量数据库的高频考点——"你们用的什么 embedding 模型？为什么选它？换一个模型会怎样？"

---

## Embedding Function 的工作流程

当你调用 `collection.add(documents=["文本"])` 或 `collection.query(query_texts=["查询"])` 时，如果文档或查询是以文本形式传入的，Chroma 会自动调用 Embedding Function 将文本转化为向量。整个流程如下：

```
┌─────────────────────────────────────────────────────────────────┐
│  collection.add(documents=["退款政策规定..."])                    │
│       ↓                                                         │
│  Chroma 检测到 documents 参数（文本，非向量）                     │
│       ↓                                                         │
│  调用 Embedding Function: ["退款政策规定..."] → [[0.12, -0.34, ...]]│
│       ↓                                                         │
│  将向量存入 HNSW 索引，文本存入 SQLite                            │
│       ↓                                                         │
│  返回（无输出）                                                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  collection.query(query_texts=["如何退款？"])                     │
│       ↓                                                         │
│  Chroma 检测到 query_texts 参数（文本，非向量）                   │
│       ↓                                                         │
│  调用 Embedding Function: ["如何退款？"] → [[0.15, -0.31, ...]]  │
│       ↓                                                         │
│  用查询向量在 HNSW 索引中搜索最近邻                               │
│       ↓                                                         │
│  返回 top-K 结果                                                │
└─────────────────────────────────────────────────────────────────┘
```

关键点在于：**添加文档时和查询时必须使用同一个 Embedding Function**。如果添加时用模型 A 编码，查询时用模型 B 编码，那么两个向量空间不兼容，搜索结果会完全不可靠。Chroma 通过在 Collection 级别绑定 Embedding Function 来保证这一点——一旦 Collection 创建时指定了某个 EF，后续所有 add 和 query 都会自动使用同一个 EF。

---

## Chroma 的默认 Embedding Function

当你不指定任何 Embedding Function 时，Chroma 使用 `all-MiniLM-L6-v2` 模型作为默认值。这是 SentenceTransformers 库提供的一个轻量级模型，具有以下特点：

| 属性 | 值 |
|------|-----|
| 模型名称 | `all-MiniLM-L6-v2` |
| 输出维度 | 384 |
| 模型大小 | ~80MB |
| 推理速度 | ~10ms/句（CPU） |
| 训练数据 | 英文为主，10亿+句子对 |
| 多语言支持 | ❌ 主要支持英文 |

```python
import chromadb

# 不指定 EF，使用默认的 all-MiniLM-L6-v2
client = chromadb.Client()
collection = client.create_collection(name="default_ef")

collection.add(
    documents=["This is an English document about machine learning"],
    ids=["en_doc"]
)

results = collection.query(query_texts=["What is ML?"], n_results=1)
print(results['documents'][0])  # 能正常命中
```

**默认 EF 的局限**：`all-MiniLM-L6-v2` 是一个英文模型，对中文的支持非常有限。如果你直接用它处理中文文本，embedding 质量会大幅下降——因为模型在训练时几乎没有见过中文语料，它无法正确编码中文的语义信息。比如下面的程序展示了中文场景下默认 EF 的问题：

```python
# 默认 EF 对中文的支持很差
collection.add(
    documents=[
        "机器学习是人工智能的一个分支",
        "今天天气真好，适合出门散步",
        "深度学习使用多层神经网络"
    ],
    ids=["zh_1", "zh_2", "zh_3"]
)

# 查询"人工智能"，可能返回不相关的结果
results = collection.query(query_texts=["人工智能"], n_results=2)
for doc in results['documents'][0]:
    print(doc)
# 可能输出（因为默认模型对中文编码质量差）：
# "今天天气真好，适合出门散步"  ← 语义上完全不相关！
# "机器学习是人工智能的一个分支"  ← 这个才应该排第一
```

所以如果你的应用需要处理中文（或者多语言），**必须替换默认的 Embedding Function**。

---

## 切换 Embedding Function 的三种方式

### 方式一：使用 Chroma 内置的 Embedding Function

Chroma 提供了 `chromadb.utils.embedding_functions` 模块，内置了几个常用的 EF：

```python
from chromadb.utils import embedding_functions

# 1. Sentence Transformers（本地推理，支持多语言模型）
ef_st = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"  # 多语言模型
)

# 2. OpenAI API（远程调用，质量高但需付费）
ef_openai = embedding_functions.OpenAIEmbeddingFunction(
    api_key="sk-...",
    model_name="text-embedding-3-small"  # 或 text-embedding-3-large
)

# 3. Hugging Face 推理 API（远程调用，免费额度有限）
ef_hf = embedding_functions.HuggingFaceEmbeddingFunction(
    api_key="hf_...",
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 创建 Collection 时指定 EF
collection = client.create_collection(
    name="multilingual_docs",
    embedding_function=ef_st,  # 使用多语言模型
    metadata={"hnsw:space": "cosine"}
)

# 后续 add 和 query 自动使用指定的 EF
collection.add(
    documents=["机器学习是人工智能的一个分支"],
    ids=["zh_1"]
)

results = collection.query(query_texts=["什么是AI"], n_results=1)
print(results['documents'][0][0])  # 现在能正确命中中文文档了
```

### 方式二：自定义 Embedding Function

如果内置的 EF 不满足需求（比如你想用自己微调的 embedding 模型、或者用本地部署的推理服务），可以实现 Chroma 的 `EmbeddingFunction` 接口：

```python
import chromadb
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
from typing import List

class MyCustomEmbeddingFunction(EmbeddingFunction):
    """自定义 Embedding Function 示例"""

    def __init__(self, model_path: str, device: str = "cpu"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_path, device=device)

    def __call__(self, input: Documents) -> Embeddings:
        """
        核心接口：接收文本列表，返回向量列表

        参数:
            input: List[str] - 待编码的文本列表
        返回:
            List[List[float]] - 对应的向量列表

        约束:
            - 返回的向量数量必须等于输入的文本数量
            - 所有向量的维度必须一致
            - 不能返回 None 或空列表
        """
        embeddings = self.model.encode(input, normalize_embeddings=True)
        return embeddings.tolist()


# 使用自定义 EF
custom_ef = MyCustomEmbeddingFunction(
    model_path="BAAI/bge-small-zh-v1.5",  # 中文 embedding 模型
    device="cpu"
)

collection = client.create_collection(
    name="custom_ef_collection",
    embedding_function=custom_ef
)

collection.add(
    documents=["中文文档的语义编码效果会好很多"],
    ids=["custom_1"]
)
```

接口契约非常简洁——只要实现 `__call__(self, input: Documents) -> Embeddings` 方法即可。其中 `Documents` 是 `List[str]` 的类型别名，`Embeddings` 是 `List[List[float]]` 的类型别名。这个设计使得你可以用任何方式生成向量：本地模型、远程 API、甚至是基于规则的伪向量（测试用）。

### 方式三：跳过 EF，直接传入预计算的向量

如果你已经有了预计算好的向量（比如离线批量计算的结果），可以完全跳过 Embedding Function，直接传入 embeddings：

```python
import numpy as np

# 假设你已经有了预计算的向量（比如从其他系统导出的）
precomputed_vectors = np.random.randn(3, 384).tolist()  # 3 条 384 维向量

collection.add(
    documents=["文档1", "文档2", "文档3"],
    ids=["pre_1", "pre_2", "pre_3"],
    embeddings=precomputed_vectors  # 直接传入向量，跳过 EF
)

# 查询时也需要传入向量（因为 Collection 没有绑定 EF）
query_vector = np.random.randn(384).tolist()
results = collection.query(
    query_embeddings=[query_vector],  # 注意：用 query_embeddings 而不是 query_texts
    n_results=2
)
```

**注意**：如果你在创建 Collection 时没有指定 EF（或指定了 `None`），那么查询时只能用 `query_embeddings` 传入向量，不能用 `query_texts`——因为没有 EF 来把文本转成向量。

---

## 常用 Embedding Provider 对比

选择 Embedding Function 时，需要在精度、速度、成本、语言支持之间做权衡。下面是几种常见方案的对比：

```
┌──────────────────────────────────────────────────────────────────────┐
│  Embedding Provider 选型决策树                                       │
│                                                                      │
│  你的场景是什么？                                                     │
│  │                                                                   │
│  ├─ 快速原型 / 本地开发                                               │
│  │   → SentenceTransformers (all-MiniLM-L6-v2)                      │
│  │   → 免费、离线、80MB、384维、英文                                   │
│  │                                                                   │
│  ├─ 中文或多语言场景                                                   │
│  │   → SentenceTransformers (paraphrase-multilingual-MiniLM-L12-v2)  │
│  │   → 免费、离线、420MB、384维、50+语言                               │
│  │   → 或 BAAI/bge-small-zh-v1.5（中文专用，512维）                   │
│  │                                                                   │
│  ├─ 追求最高质量                                                      │
│  │   → OpenAI text-embedding-3-large                                 │
│  │   → 付费、API调用、3072维、多语言                                   │
│  │   → 或 BAAI/bge-large-zh-v1.5（本地中文最优）                      │
│  │                                                                   │
│  ├─ 大规模批量处理                                                     │
│  │   → 本地 GPU + SentenceTransformers                               │
│  │   → 或预计算 + 直接传 embeddings                                   │
│  │                                                                   │
│  └─ 多模态（图文混合）                                                 │
│      → CLIP (openai/clip-vit-base-patch32)                           │
│      → 图像和文本共享同一向量空间                                       │
└──────────────────────────────────────────────────────────────────────┘
```

| Provider | 模型 | 维度 | 中文支持 | 成本 | 延迟/句 | 适用场景 |
|----------|------|------|---------|------|---------|---------|
| SentenceTransformers | all-MiniLM-L6-v2 | 384 | ❌ 差 | 免费 | ~10ms(CPU) | 英文原型 |
| SentenceTransformers | multilingual-MiniLM-L12-v2 | 384 | ✅ 良好 | 免费 | ~15ms(CPU) | 多语言开发 |
| SentenceTransformers | bge-small-zh-v1.5 | 512 | ✅ 优秀 | 免费 | ~12ms(CPU) | 中文首选 |
| SentenceTransformers | bge-large-zh-v1.5 | 1024 | ✅ 最佳 | 免费 | ~30ms(CPU) | 中文高质量 |
| OpenAI | text-embedding-3-small | 1536 | ✅ 良好 | $0.02/1M tokens | ~100ms(网络) | 快速上线 |
| OpenAI | text-embedding-3-large | 3072 | ✅ 优秀 | $0.13/1M tokens | ~150ms(网络) | 最高质量 |
| 本地部署 | bge-m3 (BAAI) | 1024 | ✅ 最佳 | 免费(GPU) | ~5ms(GPU) | 大规模生产 |

### 实战：中文场景的 EF 配置

```python
from chromadb.utils import embedding_functions
import chromadb

# 方案 A：用 Chroma 内置的 SentenceTransformers EF（推荐入门）
ef_multilingual = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)

# 方案 B：用自定义 EF 加载中文专用模型（推荐生产）
class ChineseEmbeddingFunction(embedding_functions.EmbeddingFunction):
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(
            "BAAI/bge-small-zh-v1.5",
            device="cpu"
        )

    def __call__(self, input):
        return self.model.encode(
            input,
            normalize_embeddings=True,  # 归一化后 cosine 和 ip 等价
            show_progress_bar=False
        ).tolist()

ef_chinese = ChineseEmbeddingFunction()

# 创建 Collection
client = chromadb.Client()
collection = client.create_collection(
    name="chinese_kb",
    embedding_function=ef_chinese,
    metadata={"hnsw:space": "cosine"}
)

# 添加中文文档
collection.add(
    documents=[
        "机器学习是人工智能的核心技术之一",
        "深度学习通过多层神经网络提取特征",
        "自然语言处理让计算机理解人类语言",
        "计算机视觉用于图像和视频的分析与理解"
    ],
    ids=["ml", "dl", "nlp", "cv"],
    metadatas=[
        {"category": "ai_fundamental"},
        {"category": "ai_fundamental"},
        {"category": "ai_application"},
        {"category": "ai_application"}
    ]
)

# 中文语义搜索——现在效果好多了
results = collection.query(
    query_texts=["语言理解技术"],
    n_results=3,
    include=["documents", "distances", "metadatas"]
)

for i in range(len(results['ids'][0])):
    dist = results['distances'][0][i]
    doc = results['documents'][0][i]
    cat = results['metadatas'][0][i]["category"]
    print(f"  [{dist:.4f}] ({cat}) {doc}")
```

输出：

```
  [0.2134] (ai_application) 自然语言处理让计算机理解人类语言
  [0.4521] (ai_fundamental) 深度学习通过多层神经网络提取特征
  [0.5867] (ai_fundamental) 机器学习是人工智能的核心技术之一
```

可以看到，"语言理解技术"最匹配的是"自然语言处理"这条文档，距离只有 0.2134，远好于之前用默认英文模型时的结果。

---

## Embedding 维度对性能的影响

Embedding 模型的输出维度直接影响 Chroma 的存储和查询性能。维度越高，向量占用的存储空间越大，HNSW 索引的构建和搜索也越慢。但维度并不总是越高越好——维度高意味着模型能编码更多信息，但也可能引入噪声。

```
存储开销估算：
  单条向量 = dim × 4 bytes (float32)
  10万条 384维 = 384 × 4 × 100,000 ≈ 146 MB
  10万条 1536维 = 1536 × 4 × 100,000 ≈ 586 MB
  10万条 3072维 = 3072 × 4 × 100,000 ≈ 1.17 GB

HNSW 索引额外开销（约 1.5~2x 向量数据）：
  10万条 384维 总计 ≈ 220~290 MB
  10万条 1536维 总计 ≈ 880~1170 MB
```

比如下面的程序对比了不同维度下的查询延迟：

```python
import chromadb
import time
import numpy as np

def benchmark_dimension(dim, n_docs=10000, n_queries=100):
    """测试不同维度下的查询性能"""
    client = chromadb.Client()
    col = client.create_collection(name=f"bench_{dim}")

    # 生成随机向量（模拟不同维度的 embedding）
    vectors = np.random.randn(n_docs, dim).tolist()
    docs = [f"文档 {i}" for i in range(n_docs)]
    ids = [f"d_{i}" for i in range(n_docs)]

    # 分批添加
    batch = 1000
    for i in range(0, n_docs, batch):
        col.add(
            documents=docs[i:i+batch],
            ids=ids[i:i+batch],
            embeddings=vectors[i:i+batch]
        )

    # 查询测试
    query_vecs = np.random.randn(n_queries, dim).tolist()
    start = time.time()
    for qv in query_vecs:
        col.query(query_embeddings=[qv], n_results=10)
    elapsed = time.time() - start

    qps = n_queries / elapsed
    avg_ms = (elapsed / n_queries) * 1000
    print(f"维度={dim}: {avg_ms:.1f}ms/query, {qps:.0f} QPS")
    client.delete_collection(f"bench_{dim}")

# benchmark_dimension(384)   # ~5ms/query, ~200 QPS
# benchmark_dimension(768)   # ~8ms/query, ~125 QPS
# benchmark_dimension(1536)  # ~15ms/query, ~67 QPS
```

**实践建议**：对于大多数 RAG 场景，384~768 维已经足够。只有当你发现检索质量不够（比如 top-5 的召回率低于 80%）时，才考虑升级到更高维度的模型。

---

## 常见误区与排查

### 误区 1：不同 Collection 可以混用不同的 EF

**可以**，但要注意：每个 Collection 绑定自己的 EF，不同 Collection 完全可以用不同的模型。但**同一个 Collection 内**，add 和 query 必须用同一个 EF——否则向量空间不兼容，搜索结果不可靠。

```python
# ✅ 不同 Collection 用不同 EF，完全合法
col_en = client.create_collection(name="english", embedding_function=ef_english)
col_zh = client.create_collection(name="chinese", embedding_function=ef_chinese)

# ❌ 同一个 Collection 混用 EF（Chroma 会阻止这种情况）
# 因为 EF 在创建时绑定，后续无法更改
```

### 误区 2：换了 EF 后旧数据还能用

**不能**。如果你删除旧 Collection 并用同名但不同 EF 重建，旧数据的向量与新 EF 的向量空间不兼容。你必须用新的 EF 重新编码所有文档并重新入库。

```python
# ❌ 错误做法：换 EF 后期望旧数据还能查
client.delete_collection("my_docs")
col = client.create_collection(name="my_docs", embedding_function=new_ef)
# 此时 Collection 是空的！旧向量已经随旧 Collection 一起删除了

# ✅ 正确做法：重新编码所有文档
all_old_docs = [...]  # 从备份或其他来源获取原文
col.add(documents=all_old_docs, ids=[...], metadatas=[...])
```

### 误区 3：EF 的 API 调用失败导致 add/query 报错

使用远程 EF（如 OpenAI）时，网络问题或 API 限流会导致 embedding 计算失败，进而让 add 或 query 抛出异常。建议在生产环境中加入重试机制：

```python
import time
from tenacity import retry, stop_after_attempt, wait_exponential

class RobustOpenAIEF(embedding_functions.EmbeddingFunction):
    def __init__(self, api_key, model_name="text-embedding-3-small"):
        self.ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name=model_name
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def __call__(self, input):
        try:
            return self.ef(input)
        except Exception as e:
            print(f"Embedding 调用失败: {e}，正在重试...")
            raise

# 使用
robust_ef = RobustOpenAIEF(api_key="sk-...")
```

### 误区 4：忽略 EF 的最大输入长度

每个 embedding 模型都有最大输入 token 长度限制。比如 `all-MiniLM-L6-v2` 的上限是 256 tokens，`bge-small-zh-v1.5` 是 512 tokens。如果你传入超长文本，模型会自动截断——截断后的 embedding 丢失了后半段的信息，导致检索质量下降。

```python
# ❌ 超长文本被自动截断
very_long_text = "这是一段很长的文本..." * 500  # 远超 512 tokens
collection.add(documents=[very_long_text], ids=["long_doc"])
# embedding 只编码了前 512 tokens 的内容！

# ✅ 正确做法：先切分再入库
chunks = split_text(very_long_text, max_tokens=400, overlap=50)
for i, chunk in enumerate(chunks):
    collection.add(
        documents=[chunk],
        ids=[f"long_doc_chunk_{i}"],
        metadatas=[{"chunk_index": i, "total_chunks": len(chunks)}]
    )
```

---

## 本章小结

Embedding Function 是 Chroma 的"翻译层"——它把人类语言翻译成机器可计算的向量，使得语义搜索成为可能。理解 EF 的工作机制是使用 Chroma 的基本功，也是面试中的高频考点。

核心要点回顾：第一，Chroma 默认使用 `all-MiniLM-L6-v2`，这是一个英文模型，对中文支持很差，中文场景必须替换；第二，EF 在 Collection 创建时绑定，后续 add 和 query 自动使用同一个 EF，保证向量空间一致性；第三，自定义 EF 只需实现 `__call__(input: Documents) -> Embeddings` 接口，可以对接任何 embedding 模型或 API；第四，维度越高存储和查询开销越大，384~768 维对大多数 RAG 场景足够；第五，远程 EF 需要处理网络异常和限流，生产环境务必加重试机制；第六，超长文本会被模型截断，必须先切分再入库。

下一节我们将深入文档切分策略——如何把一篇长文档切成合适的 chunk，使得每个 chunk 既保留足够的语义完整性，又不会超出 embedding 模型的输入长度限制。
