# 5.2 多向量搜索与重排序

> **一个 Collection 多个向量字段——多模态搜索的基础能力**

---

## 这一节在讲什么？

pgvector 和 Chroma 的一个 Collection 只能有一个向量字段——如果你想同时搜索文本 embedding 和图像 embedding，只能建两个 Collection 然后在应用层合并结果。Milvus 支持一个 Collection 有多个向量字段，搜索时可以同时搜索多个字段并用 RRF（Reciprocal Rank Fusion）或加权方式合并结果。这是多模态 RAG 的基础能力，也是 Milvus 相比 pgvector/Chroma 的重要优势。这一节我们要聊多向量字段的设计、Hybrid Search 的用法、以及重排序策略。

---

## 多向量字段的设计

一个 Collection 可以有多个向量字段——每个字段可以有不同的维度和不同的索引：

```python
from pymilvus import FieldSchema, CollectionSchema, DataType, MilvusClient

client = MilvusClient(uri="http://localhost:19530")

# 多模态文档：文本 embedding + 图像 embedding
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="text_embedding", dtype=DataType.FLOAT_VECTOR, dim=768),   # 文本向量
    FieldSchema(name="image_embedding", dtype=DataType.FLOAT_VECTOR, dim=512),  # 图像向量
]

schema = CollectionSchema(fields=fields)
client.create_collection(collection_name="multimodal_docs", schema=schema)

# 为两个向量字段分别创建索引
index_params = client.prepare_index_params()
index_params.add_index(
    field_name="text_embedding",
    index_type="HNSW",
    metric_type="COSINE",
    params={"M": 16, "efConstruction": 256}
)
index_params.add_index(
    field_name="image_embedding",
    index_type="HNSW",
    metric_type="COSINE",
    params={"M": 16, "efConstruction": 256}
)
client.create_index(collection_name="multimodal_docs", index_params=index_params)
client.load_collection("multimodal_docs")
```

插入数据时，两个向量字段都需要提供：

```python
client.insert(
    collection_name="multimodal_docs",
    data=[
        {
            "title": "AI breakthrough",
            "text_embedding": [0.1] * 768,
            "image_embedding": [0.2] * 512
        }
    ]
)
```

---

## Hybrid Search：多向量搜索

Hybrid Search 允许你同时搜索多个向量字段，然后合并结果。Milvus 提供了两种合并策略：

### WeightedRanker：加权合并

WeightedRanker 按权重加权合并多个搜索结果的距离值——权重越大，该路搜索的结果对最终排序影响越大：

```python
from pymilvus import AnnSearchRequest, WeightedRanker

# 第1路搜索：文本向量
text_search = AnnSearchRequest(
    data=[[0.1] * 768],
    anns_field="text_embedding",
    param={"metric_type": "COSINE", "params": {"ef": 100}},
    limit=20
)

# 第2路搜索：图像向量
image_search = AnnSearchRequest(
    data=[[0.2] * 512],
    anns_field="image_embedding",
    param={"metric_type": "COSINE", "params": {"ef": 100}},
    limit=20
)

# 加权合并：文本 70%，图像 30%
results = client.hybrid_search(
    collection_name="multimodal_docs",
    reqs=[text_search, image_search],
    ranker=WeightedRanker(0.7, 0.3),
    limit=5,
    output_fields=["title"]
)
```

WeightedRanker 的缺点是你需要手动调权重——不同场景的最优权重可能不同，而且权重对结果的影响不是线性的。

### RRFRanker：Reciprocal Rank Fusion

RRFRanker 按"排名倒数"合并结果——不需要调权重，更鲁棒：

```python
from pymilvus import AnnSearchRequest, RRFRanker

results = client.hybrid_search(
    collection_name="multimodal_docs",
    reqs=[text_search, image_search],
    ranker=RRFRanker(k=60),  # k 是平滑参数，默认 60
    limit=5,
    output_fields=["title"]
)
```

RRF 的计算公式是：`score = Σ 1/(k + rank_i)`，其中 `rank_i` 是第 i 路搜索中该结果的排名。k 值越大，排名差异的影响越小——默认值 60 在大多数场景下效果不错。

比如，下面的例子展示了 RRF 的计算过程，由于 RRF 按排名而非距离合并，所以即使两路搜索的距离尺度不同，也能公平地融合结果：

```
RRF 合并示例（k=60）：

  文本搜索结果：        图像搜索结果：
  ID=1, rank=1         ID=3, rank=1
  ID=2, rank=2         ID=1, rank=2
  ID=3, rank=3         ID=5, rank=3

  RRF 分数计算：
  ID=1: 1/(60+1) + 1/(60+2) = 0.0164 + 0.0161 = 0.0325
  ID=2: 1/(60+2) + 0        = 0.0161
  ID=3: 1/(60+3) + 1/(60+1) = 0.0159 + 0.0164 = 0.0323
  ID=5: 0          + 1/(60+3) = 0.0159

  最终排序：ID=1 > ID=3 > ID=2 > ID=5
```

---

## 重排序策略

Hybrid Search 返回的结果是合并排序后的 Top-K，但这个排序是基于 ANN 索引的近似距离——可能不够精确。重排序（Reranking）的策略是：先用 ANN 索引快速召回更多候选，再用精确距离或交叉编码器重排。

### 策略1：ANN 召回 + 精确距离重排

```python
# 先用 ANN 索引召回 Top-50
results = client.search(
    collection_name="documents",
    data=[query_vec],
    limit=50,  # 召回更多候选
    search_params={"metric_type": "COSINE", "params": {"ef": 200}},
    output_fields=["content", "embedding"]  # 返回向量用于精确计算
)

# 用精确距离重排 Top-50 → Top-5
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

candidates = []
for hit in results[0]:
    emb = hit['entity']['embedding']
    sim = cosine_similarity(query_vec, emb)
    candidates.append((hit['id'], sim, hit['entity']['content']))

# 按精确相似度排序
candidates.sort(key=lambda x: x[1], reverse=True)
top5 = candidates[:5]
```

### 策略2：多路召回 + 交叉编码器重排

```python
from sentence_transformers import CrossEncoder

# 用 Cross-Encoder 重排序
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

candidates_text = [c[2] for c in candidates[:20]]  # 取前 20 个候选
pairs = [[query_text, text] for text in candidates_text]
scores = cross_encoder.predict(pairs)

# 按 Cross-Encoder 分数排序
ranked = sorted(zip(candidates[:20], scores), key=lambda x: x[1], reverse=True)
top5 = [r[0] for r in ranked[:5]]
```

---

## 常见误区：Hybrid Search 一定比单路搜索好

Hybrid Search 的优势在于融合多路信息——但如果你只有一种向量（比如只有文本 embedding），Hybrid Search 没有意义。而且 Hybrid Search 的计算量是单路搜索的 2~3 倍（需要分别搜索多个字段再合并），如果两路搜索的结果高度重叠，Hybrid Search 的收益可能抵不上额外的计算开销。

---

## 小结

这一节我们聊了 Milvus 的多向量搜索：一个 Collection 可以有多个向量字段，Hybrid Search 同时搜索多个字段并合并结果。合并策略有 WeightedRanker（加权合并，需要调权重）和 RRFRanker（排名倒数合并，更鲁棒）。重排序策略有 ANN 召回 + 精确距离重排和多路召回 + 交叉编码器重排。下一节我们聊动态字段与 JSON 过滤。
