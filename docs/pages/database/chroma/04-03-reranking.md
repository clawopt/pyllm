# 4.3 多阶段查询与 Re-ranking

> **粗筛快、精排准——两阶段检索是工业级 RAG 系统的标准配置**

---

## 这一节在讲什么？

到目前为止，我们用的都是单阶段检索：用户提问 → Chroma 返回 top-K 结果 → 喂给 LLM。这种方式简单直接，但在精度要求高的场景下有一个根本性的问题——embedding 模型的向量编码是"有损压缩"。它把一段几百字的文本压缩成 384 或 768 维的浮点数向量，必然丢失了大量细节信息。这意味着向量相似度高不代表文本真正相关，向量相似度低也不代表文本不相关——特别是在需要精确匹配人名、数字、日期等具体信息的场景中。

多阶段检索（也叫"粗筛+精排"）是解决这个问题的工业标准方案：第一阶段用 Chroma 做快速的向量粗筛，拿到较多的候选结果（比如 50 条）；第二阶段用更精确但更慢的模型（cross-encoder / re-ranker）对候选结果重新打分排序，选出最相关的 top-K（比如 5 条）。这一节我们要讲清楚两阶段检索的原理、实现方式、以及延迟与精度的权衡。

---

## 为什么需要 Re-ranking

要理解 Re-ranking 的必要性，我们需要先理解 embedding 模型的固有局限。主流的 embedding 模型（如 SentenceTransformers、OpenAI Embedding）使用的是**双编码器（Bi-Encoder）架构**——查询和文档分别独立编码成向量，然后通过向量距离衡量相似度。这种方式的优势是速度快（文档向量可以预计算并建索引），但劣势是查询和文档之间没有"交互"——模型无法在编码时考虑查询和文档之间的细粒度匹配关系。

```
┌─────────────────────────────────────────────────────────────────┐
│  Bi-Encoder（Embedding 模型）的工作方式                          │
│                                                                 │
│  Query: "苹果公司2024年Q3营收"                                   │
│    ↓ 独立编码                                                    │
│  Query Vector: [0.12, -0.34, 0.56, ...]                        │
│                                                                 │
│  Doc A: "苹果公司2024年第三季度营收为948亿美元"                    │
│    ↓ 独立编码                                                    │
│  Doc A Vector: [0.11, -0.32, 0.55, ...]                        │
│                                                                 │
│  Doc B: "苹果是一种营养丰富的水果，富含维生素C"                    │
│    ↓ 独立编码                                                    │
│  Doc B Vector: [0.09, -0.28, 0.51, ...]                        │
│                                                                 │
│  → Query 与 Doc A 的距离可能只比 Doc B 近一点点                  │
│  → 因为 embedding 无法精确区分"苹果公司"和"苹果水果"              │
│  → 需要更精细的模型来"仔细阅读"查询和文档的匹配关系               │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  Cross-Encoder（Re-ranker 模型）的工作方式                       │
│                                                                 │
│  将 Query 和 Doc 拼接后一起编码：                                │
│                                                                 │
│  "[CLS] 苹果公司2024年Q3营收 [SEP] 苹果公司2024年第三季度        │
│   营收为948亿美元 [SEP]"                                         │
│    ↓ 交叉注意力                                                  │
│  相关性分数: 0.95  ← 高分！精确匹配了"苹果公司"和"Q3营收"        │
│                                                                 │
│  "[CLS] 苹果公司2024年Q3营收 [SEP] 苹果是一种营养丰富的水果，    │
│   富含维生素C [SEP]"                                             │
│    ↓ 交叉注意力                                                  │
│  相关性分数: 0.12  ← 低分！模型理解了"苹果"的歧义                │
└─────────────────────────────────────────────────────────────────┘
```

Cross-Encoder 的核心优势在于**交叉注意力（Cross-Attention）机制**——它让查询和文档的每个 token 都能互相"看到"，从而捕捉细粒度的语义匹配关系。比如当查询中提到"苹果公司"时，Cross-Encoder 能注意到文档中也有"苹果公司"这个完整实体，而不是把"苹果"单独理解成水果。但这种精细度的代价是速度——Cross-Encoder 无法预计算文档向量，每次查询都需要把查询和每个候选文档拼接后重新过一遍模型，时间复杂度是 O(K × L)，其中 K 是候选文档数，L 是序列长度。

---

## 两阶段检索的实现

### 阶段一：Chroma 粗筛

```python
import chromadb
from chromadb.utils import embedding_functions

client = chromadb.Client(settings=chromadb.Settings(
    is_persistent=True,
    persist_directory="./rerank_demo"
))

ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)

collection = client.get_or_create_collection(
    name="rerank_demo_kb",
    embedding_function=ef
)

# 添加文档
documents = [
    "苹果公司2024年第三季度营收为948亿美元，同比增长6%",
    "苹果是一种营养丰富的水果，富含维生素C和膳食纤维",
    "苹果公司发布了新款MacBook Pro，搭载M4芯片",
    "富士苹果的产地主要在山东烟台和陕西洛川",
    "苹果公司的服务业务收入达到创纪录的240亿美元",
    "苹果派是一道经典的西式甜点，用苹果和肉桂制作",
    "iPhone 16系列首周销量突破1000万台",
    "苹果汁含有丰富的抗氧化物质，有助于降低胆固醇",
]
metadatas = [
    {"category": "finance"}, {"category": "food"},
    {"category": "tech"}, {"category": "food"},
    {"category": "finance"}, {"category": "food"},
    {"category": "tech"}, {"category": "food"},
]

collection.add(
    documents=documents,
    ids=[f"d{i}" for i in range(len(documents))],
    metadatas=metadatas
)

# 阶段一：Chroma 粗筛，取 8 条（全部）
stage1_results = collection.query(
    query_texts=["苹果公司最新的财务数据"],
    n_results=8,
    include=["documents", "metadatas", "distances"]
)

print("=== 阶段一：Chroma 粗筛结果 ===")
for i in range(len(stage1_results['ids'][0])):
    dist = stage1_results['distances'][0][i]
    cat = stage1_results['metadatas'][0][i]['category']
    doc = stage1_results['documents'][0][i]
    print(f"  [{dist:.4f}] ({cat}) {doc[:50]}...")
```

输出可能是：

```
=== 阶段一：Chroma 粗筛结果 ===
  [0.4123] (finance) 苹果公司2024年第三季度营收为948亿美元，同比增长6%...
  [0.5234] (tech) 苹果公司发布了新款MacBook Pro，搭载M4芯片...
  [0.5678] (finance) 苹果公司的服务业务收入达到创纪录的240亿美元...
  [0.6123] (food) 苹果是一种营养丰富的水果，富含维生素C和膳食纤维...
  [0.6345] (tech) iPhone 16系列首周销量突破1000万台...
  [0.6789] (food) 富士苹果的产地主要在山东烟台和陕西洛川...
  [0.7123] (food) 苹果派是一道经典的西式甜点，用苹果和肉桂制作...
  [0.7456] (food) 苹果汁含有丰富的抗氧化物质，有助于降低胆固醇...
```

可以看到，Chroma 的粗筛结果中，"苹果水果"相关的文档排在了"苹果公司财务"相关文档之后——但它们的距离差距不大（0.6123 vs 0.6789），如果只取 top-3，可能会遗漏重要的财务信息。

### 阶段二：Cross-Encoder 精排

```python
from sentence_transformers import CrossEncoder

# 加载 Cross-Encoder 模型
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query: str, documents: list, top_k: int = 3):
    """用 Cross-Encoder 对候选文档重新排序"""
    pairs = [(query, doc) for doc in documents]
    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(documents, scores),
        key=lambda x: x[1],
        reverse=True
    )
    return ranked[:top_k]

# 阶段二：Re-ranking
query = "苹果公司最新的财务数据"
candidates = stage1_results['documents'][0]

print("\n=== 阶段二：Cross-Encoder 精排结果 ===")
ranked_results = rerank(query, candidates, top_k=3)
for i, (doc, score) in enumerate(ranked_results):
    print(f"  [{score:.4f}] {doc[:60]}...")
```

输出：

```
=== 阶段二：Cross-Encoder 精排结果 ===
  [0.9234] 苹果公司2024年第三季度营收为948亿美元，同比增长6%...
  [0.8567] 苹果公司的服务业务收入达到创纪录的240亿美元...
  [0.3124] 苹果公司发布了新款MacBook Pro，搭载M4芯片...
```

经过 Cross-Encoder 精排后，"苹果水果"相关的文档被大幅压低了分数（从 0.6+ 降到 0.1 以下），而"苹果公司财务"相关的文档获得了高分（0.92 和 0.86）。这就是 Re-ranking 的价值——它通过更精细的语义理解，把真正相关的结果提升到前面。

---

## 常用的 Re-ranker 模型

| 模型 | 大小 | 语言 | 速度 | 精度 | 适用场景 |
|------|------|------|------|------|---------|
| cross-encoder/ms-marco-MiniLM-L-6-v2 | ~80MB | 英文 | 快 | 中等 | 英文快速原型 |
| cross-encoder/ms-marco-MiniLM-L-12-v2 | ~120MB | 英文 | 中等 | 较高 | 英文生产环境 |
| BAAI/bge-reranker-base | ~280MB | 多语言 | 中等 | 高 | 多语言通用 |
| BAAI/bge-reranker-large | ~560MB | 多语言 | 较慢 | 最高 | 中文高质量场景 |
| Cohere Rerank API | N/A | 多语言 | 快(网络) | 高 | 无 GPU 场景 |

---

## 延迟与精度的权衡

两阶段检索引入了额外的延迟——Cross-Encoder 需要对每个候选文档做一次前向推理。下面是典型的延迟参考：

```
场景：50 条候选文档，每条平均 200 tokens

阶段一（Chroma 向量搜索）：~5-20ms
阶段二（Cross-Encoder 精排）：
  - MiniLM-L-6-v2 (CPU): ~200-500ms
  - MiniLM-L-6-v2 (GPU): ~20-50ms
  - bge-reranker-large (CPU): ~500-1500ms
  - bge-reranker-large (GPU): ~50-150ms
  - Cohere API (网络): ~100-300ms

总延迟 = 阶段一 + 阶段二
```

对于实时交互场景（如聊天机器人），总延迟应控制在 500ms 以内。这意味着：
- CPU 环境：候选文档数建议 ≤ 20 条，用轻量级 re-ranker
- GPU 环境：候选文档数可以到 50 条，用大型 re-ranker
- 无 GPU 且需要高精度：考虑用 Cohere API

---

## 完整实战：带 Re-ranking 的 RAG 查询函数

```python
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import CrossEncoder

class RerankedRetriever:
    """带 Re-ranking 的检索器"""

    def __init__(self, collection_name: str, persist_dir: str = "./chroma_db",
                 reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 stage1_k: int = 20, stage2_k: int = 5):
        self.client = chromadb.Client(settings=chromadb.Settings(
            is_persistent=True,
            persist_directory=persist_dir
        ))

        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="paraphrase-multilingual-MiniLM-L12-v2"
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=ef
        )

        self.reranker = CrossEncoder(reranker_model)
        self.stage1_k = stage1_k
        self.stage2_k = stage2_k

    def search(self, query: str, where: dict = None, stage1_k: int = None,
               stage2_k: int = None) -> list:
        """
        两阶段检索

        参数:
            query: 查询文本
            where: metadata 过滤条件
            stage1_k: 粗筛返回数量（默认使用初始化时的值）
            stage2_k: 精排返回数量（默认使用初始化时的值）
        """
        k1 = stage1_k or self.stage1_k
        k2 = stage2_k or self.stage2_k

        # 阶段一：Chroma 粗筛
        stage1 = self.collection.query(
            query_texts=[query],
            where=where,
            n_results=k1,
            include=["ids", "documents", "metadatas", "distances"]
        )

        if not stage1['ids'][0]:
            return []

        # 阶段二：Cross-Encoder 精排
        candidates = stage1['documents'][0]
        pairs = [(query, doc) for doc in candidates]
        scores = self.reranker.predict(pairs)

        # 组装结果并按 re-ranker 分数排序
        results = []
        for i in range(len(candidates)):
            results.append({
                "id": stage1['ids'][0][i],
                "document": candidates[i],
                "metadata": stage1['metadatas'][0][i],
                "vector_distance": stage1['distances'][0][i],
                "rerank_score": float(scores[i])
            })

        results.sort(key=lambda x: x["rerank_score"], reverse=True)
        return results[:k2]


# 使用
retriever = RerankedRetriever(
    collection_name="smart_kb",
    stage1_k=20,
    stage2_k=5
)

results = retriever.search("苹果公司财务数据", where={"category": "finance"})
for r in results:
    print(f"  [rerank={r['rerank_score']:.4f}, vec_dist={r['vector_distance']:.4f}] {r['document'][:60]}...")
```

---

## 常见误区

### 误区 1：Re-ranking 一定比单阶段好

不一定。如果你的数据集很小（< 100 条）或者查询很简单（关键词匹配即可），Re-ranking 引入的额外延迟可能不值得。Re-ranking 的价值在于处理**歧义性高、需要细粒度语义理解**的查询。

### 误区 2：Re-ranker 的分数可以直接比较不同查询的结果

Cross-Encoder 输出的分数是**相对分数**，不是绝对概率。同一个查询下，分数越高越相关；但不同查询之间，分数不可直接比较。查询 A 的 0.8 分和查询 B 的 0.6 分，不代表 A 的结果比 B 的更相关。

### 误区 3：stage1_k 设得越大越好

stage1_k 过大会导致两个问题：第一，Cross-Encoder 的推理时间与候选数成正比，50 条候选的推理时间是 10 条的 5 倍；第二，如果粗筛结果中大部分都不相关，re-ranker 可能被噪声干扰，反而降低精排质量。通常 stage1_k 设为 stage2_k 的 5~10 倍即可。

---

## 本章小结

多阶段检索是工业级 RAG 系统的标准配置，它通过"粗筛快、精排准"的策略在速度和精度之间取得平衡。核心要点回顾：第一，embedding 模型（Bi-Encoder）速度快但精度有限，因为它独立编码查询和文档，无法捕捉细粒度的匹配关系；第二，Cross-Encoder（Re-ranker）通过交叉注意力实现精确的语义匹配，但无法预计算文档向量，速度较慢；第三，两阶段检索的标准流程是 Chroma 粗筛 top-20~50 → Cross-Encoder 精排 top-3~5；第四，延迟控制是关键——CPU 环境建议候选 ≤ 20 条，GPU 环境可以到 50 条；第五，Re-ranking 不是万能的，简单查询和小数据集不需要引入额外复杂度。

下一章我们将进入 RAG 系统实战——把前面学的所有知识整合起来，构建一个完整的 PDF 文档问答系统。
