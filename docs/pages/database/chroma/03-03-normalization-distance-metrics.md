# 3.3 向量标准化与距离度量选择

> **距离度量是向量搜索的"度量衡"——选错了度量，就像用温度计量距离一样荒谬**

---

## 这一节在讲什么？

在前面的章节中，我们多次提到 Chroma 支持三种距离度量：L2（欧氏距离）、Cosine（余弦距离）和 IP（内积），也提到 Chroma 默认使用 cosine。但你可能一直有个疑问：这三种度量到底有什么区别？为什么 Chroma 默认选 cosine 而不是其他两个？如果我传入了手动计算的向量，应该选哪个？这些问题的答案取决于一个关键概念——向量标准化（Normalization）。这一节我们要从数学原理出发，讲清楚三种度量的计算方式、它们之间的关系、以及在不同场景下应该如何选择。

这不是一个纯理论讨论——距离度量的选择直接影响你的搜索结果。同一个查询、同一批数据，用不同的度量可能返回完全不同的排序。理解这些差异，是构建高质量向量检索系统的基础。

---

## 三种距离度量的数学定义

### L2 距离（欧氏距离）

L2 距离是最直观的距离概念——它就是两个向量在空间中的直线距离。对于两个向量 $a$ 和 $b$，L2 距离的计算公式为：

$$d_{L2}(a, b) = \sqrt{\sum_{i=1}^{d}(a_i - b_i)^2} = ||a - b||_2$$

L2 距离的值域是 $[0, +\infty)$，值越小表示两个向量越接近。当两个向量完全相同时，L2 距离为 0。

```python
import numpy as np

def l2_distance(a, b):
    """计算 L2 距离"""
    return np.sqrt(np.sum((np.array(a) - np.array(b)) ** 2))

a = [1.0, 2.0, 3.0]
b = [1.1, 2.1, 3.1]
c = [4.0, 5.0, 6.0]

print(f"L2(a, b) = {l2_distance(a, b):.4f}")  # 0.1732 — 很近
print(f"L2(a, c) = {l2_distance(a, c):.4f}")  # 5.1962 — 很远
```

L2 距离的特点是它对向量的**模长（magnitude）敏感**。两个方向相同但模长不同的向量，L2 距离可能很大。比如向量 $[1, 0]$ 和 $[100, 0]$ 方向完全一致，但 L2 距离是 99。在 embedding 场景中，这意味着如果某些文档的 embedding 向量模长较大（比如因为文本较长），它们在 L2 空间中会被"推远"，即使语义上可能很相似。

### Cosine 距离（余弦距离）

Cosine 距离衡量的是两个向量方向的差异，忽略了模长的影响。它基于余弦相似度（Cosine Similarity）计算：

$$\cos(\theta) = \frac{a \cdot b}{||a|| \cdot ||b||} = \frac{\sum_{i=1}^{d}a_i b_i}{\sqrt{\sum_{i=1}^{d}a_i^2} \cdot \sqrt{\sum_{i=1}^{d}b_i^2}}$$

余弦相似度的值域是 $[-1, 1]$，1 表示方向完全相同，0 表示正交（无关），-1 表示方向相反。Chroma 使用的 Cosine 距离是余弦相似度的变换形式：

$$d_{cos}(a, b) = 1 - \cos(\theta) = 1 - \frac{a \cdot b}{||a|| \cdot ||b||}$$

Cosine 距离的值域是 $[0, 2]$，0 表示方向完全相同，2 表示方向完全相反。

```python
def cosine_distance(a, b):
    """计算 Cosine 距离"""
    a, b = np.array(a), np.array(b)
    similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return 1 - similarity

a = [1.0, 2.0, 3.0]
b = [2.0, 4.0, 6.0]   # 与 a 方向相同，模长是 a 的 2 倍
c = [-1.0, -2.0, -3.0] # 与 a 方向相反

print(f"Cosine(a, b) = {cosine_distance(a, b):.4f}")  # 0.0000 — 方向完全相同！
print(f"Cosine(a, c) = {cosine_distance(a, c):.4f}")  # 2.0000 — 方向完全相反
print(f"L2(a, b)     = {l2_distance(a, b):.4f}")      # 3.7417 — 模长差异导致 L2 很大
```

这个例子清楚地展示了 L2 和 Cosine 的核心区别：向量 $a$ 和 $b$ 方向完全一致（Cosine 距离 = 0），但 L2 距离却很大（3.74）。在 embedding 场景中，这意味着 Cosine 距离不受文本长度对 embedding 模长的影响——两段语义相同但长度不同的文本，它们的 Cosine 距离仍然很小。

### IP（内积 / Inner Product）

IP 就是两个向量的点积：

$$IP(a, b) = a \cdot b = \sum_{i=1}^{d}a_i b_i$$

IP 的值域取决于向量的模长——对于单位向量（模长为 1 的向量），IP 等于余弦相似度，值域为 $[-1, 1]$；对于任意向量，IP 可以是任何实数。

在 Chroma 中，IP 距离的计算方式是取负内积：$d_{IP}(a, b) = -a \cdot b$。这样距离越小（即内积越大），表示越相似。

```python
def ip_distance(a, b):
    """计算 IP 距离（负内积）"""
    return -np.dot(np.array(a), np.array(b))

a = [1.0, 0.0, 0.0]
b = [0.9, 0.1, 0.0]
c = [0.0, 1.0, 0.0]

print(f"IP距离(a, b) = {ip_distance(a, b):.4f}")  # -0.9 — 内积大，很相似
print(f"IP距离(a, c) = {ip_distance(a, c):.4f}")  # 0.0 — 内积为0，正交
```

---

## 三种度量之间的关系：归一化是关键

三种度量之间有一个非常重要的数学关系：**当向量被归一化（即每个向量的模长都为 1）时，L2 距离、Cosine 距离和 IP 距离给出完全相同的排序**。

推导过程如下。假设 $||a|| = ||b|| = 1$（归一化向量）：

- **Cosine 距离**：$d_{cos} = 1 - \frac{a \cdot b}{||a|| \cdot ||b||} = 1 - a \cdot b = 1 - IP$
- **L2 距离的平方**：$d_{L2}^2 = ||a-b||^2 = ||a||^2 + ||b||^2 - 2a \cdot b = 1 + 1 - 2IP = 2(1 - IP) = 2 \cdot d_{cos}$

所以对于归一化向量，$d_{L2}^2 = 2 \cdot d_{cos}$，$d_{cos} = 1 + d_{IP}$。三者之间是单调线性变换关系，排序完全一致。

```
┌─────────────────────────────────────────────────────────────────┐
│  归一化向量下三种度量的等价关系                                   │
│                                                                 │
│  ||a|| = ||b|| = 1 时:                                          │
│                                                                 │
│  d_cosine = 1 - IP(a, b)                                       │
│  d_L2²    = 2 × d_cosine = 2 × (1 - IP(a, b))                 │
│  d_IP     = -IP(a, b)                                          │
│                                                                 │
│  → 三者排序完全等价！                                            │
│  → 选哪个都一样，只是数值不同                                     │
│                                                                 │
│  ||a|| ≠ 1 或 ||b|| ≠ 1 时:                                     │
│                                                                 │
│  → L2 受模长影响，长向量被"推远"                                  │
│  → Cosine 不受模长影响，只看方向                                  │
│  → IP 受模长影响，长向量内积更大（可能"拉近"）                     │
│  → 三者排序可能完全不同！                                         │
└─────────────────────────────────────────────────────────────────┘
```

比如下面的程序展示了归一化前后三种度量的排序差异。由于未归一化的向量模长差异会导致 L2 和 IP 给出不同于 Cosine 的排序，而归一化后三者完全一致：

```python
import numpy as np

# 三条向量，模长不同但方向相似
v1 = np.array([1.0, 0.0, 0.0])       # 模长 = 1.0
v2 = np.array([3.0, 0.1, 0.0])       # 模长 ≈ 3.0，方向接近 v1
v3 = np.array([0.0, 1.0, 0.0])       # 模长 = 1.0，方向与 v1 正交

query = np.array([1.0, 0.0, 0.0])    # 查询向量

print("=== 未归一化向量的排序 ===")
for name, vec in [("v2(方向接近但模长大)", v2), ("v3(方向正交)", v3)]:
    l2 = l2_distance(query, vec)
    cos = cosine_distance(query, vec)
    ip = ip_distance(query, vec)
    print(f"  {name}: L2={l2:.4f}, Cosine={cos:.4f}, IP={ip:.4f}")

# 输出：
#   v2(方向接近但模长大): L2=2.0050, Cosine=0.0055, IP=-3.0000
#   v3(方向正交):         L2=1.4142, Cosine=1.0000, IP=0.0000
#
# L2 认为 v3 更近（1.41 < 2.00）——因为 v2 模长大被推远了
# Cosine 认为 v2 更近（0.0055 < 1.0）——v2 方向与 query 几乎一致
# IP 认为 v2 更近（-3.0 < 0.0）——v2 模长大导致内积大

print("\n=== 归一化后的排序 ===")
v2_norm = v2 / np.linalg.norm(v2)
for name, vec in [("v2(归一化后)", v2_norm), ("v3(本来就是单位向量)", v3)]:
    l2 = l2_distance(query, vec)
    cos = cosine_distance(query, vec)
    ip = ip_distance(query, vec)
    print(f"  {name}: L2={l2:.4f}, Cosine={cos:.4f}, IP={ip:.4f}")

# 输出：
#   v2(归一化后): L2=0.1054, Cosine=0.0055, IP=-0.9945
#   v3(本来就是单位向量): L2=1.4142, Cosine=1.0000, IP=0.0000
#
# 归一化后三者排序一致：v2 更近（语义更相似）
```

---

## 为什么 Chroma 默认用 Cosine

理解了三种度量的关系后，Chroma 默认选择 Cosine 的原因就很清楚了：

**大多数 embedding 模型输出的向量已经是归一化的**。SentenceTransformers、OpenAI Embedding、BGE 系列等主流模型，在训练时都会对输出做 L2 归一化（即 $||v|| = 1$）。对于归一化向量，三种度量排序等价，但 Cosine 有两个额外优势：

1. **鲁棒性**：即使某些向量因为数值精度问题没有完美归一化（$||v|| \approx 1$ 但不严格等于 1），Cosine 仍然能给出正确排序，因为它显式地做了归一化除法。而 IP 在这种情况下可能产生偏差。

2. **可解释性**：Cosine 距离的值域是 $[0, 2]$，0 表示完全相同，1 表示正交，2 表示完全相反。这个范围直观且稳定，不受向量维度和模长的影响。而 L2 距离的值域随维度增加而增大（高维灾难），IP 的值域取决于向量模长，都不如 Cosine 好解释。

---

## 什么时候该用 L2 或 IP

### 用 L2 的场景

L2 适合**需要区分向量模长差异**的场景。比如在图像检索中，两张图片的特征向量如果模长差异很大，通常意味着它们的"信息量"或"对比度"不同——这种情况下 L2 距离能同时捕捉方向和模长的差异，比 Cosine 更有区分度。

```python
# L2 适合的场景：图像特征检索
image_collection = client.create_collection(
    name="image_features",
    metadata={"hnsw:space": "l2"}
)
```

### 用 IP 的场景

IP 适合**向量已严格归一化且追求极致性能**的场景。由于 IP 不需要做归一化除法，计算速度比 Cosine 略快（省了两次范数计算和一次除法）。在大规模检索（百万级以上向量）中，这个微小的速度差异可能累积成可观的性能提升。

```python
# IP 适合的场景：大规模已归一化向量的快速检索
fast_collection = client.create_collection(
    name="fast_search",
    metadata={"hnsw:space": "ip"}
)

# 确保传入的向量是归一化的
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")

texts = ["文档1", "文档2", "文档3"]
embeddings = model.encode(texts, normalize_embeddings=True)  # ← 关键：归一化
fast_collection.add(
    documents=texts,
    ids=["d1", "d2", "d3"],
    embeddings=embeddings.tolist()
)
```

### ⚠️ 归一化的陷阱

如果你手动传入了未归一化的向量却用了 IP 或 Cosine 度量，搜索结果可能不可靠。来看一个具体的例子：

```python
# 陷阱：未归一化向量 + IP 度量 = 排序异常
collection = client.create_collection(
    name="trap_ip",
    metadata={"hnsw:space": "ip"}
)

# 两条语义相似的文档，但 embedding 模长不同
doc_a_embedding = [1.0, 0.0, 0.0]          # 模长 = 1.0
doc_b_embedding = [100.0, 1.0, 0.0]        # 模长 ≈ 100.0，方向接近 doc_a
doc_c_embedding = [0.0, 1.0, 0.0]          # 模长 = 1.0，方向与 doc_a 正交

collection.add(
    documents=["文档A", "文档B(与A语义相似)", "文档C(与A无关)"],
    ids=["a", "b", "c"],
    embeddings=[doc_a_embedding, doc_b_embedding, doc_c_embedding]
)

# 查询向量与 doc_a 方向一致
query_embedding = [1.0, 0.0, 0.0]

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3,
    include=["documents", "distances"]
)

for i in range(3):
    print(f"  {results['documents'][0][i]}: IP距离={results['distances'][0][i]:.4f}")

# 输出可能是：
#   文档B(与A语义相似): IP距离=-100.0000  ← 模长大导致内积最大，排第一
#   文档A: IP距离=-1.0000
#   文档C(与A无关): IP距离=0.0000
#
# 看起来 B 最相似——但如果 B 的模长大只是因为文本更长，
# 而不是语义更相关，那这个排序就是误导性的
```

**解决方案**：如果你用 IP 度量，务必确保传入的向量是归一化的。如果你无法保证归一化，就用 Cosine——它会自动处理模长差异。

---

## HNSW 距离：近似但不等于精确距离

Chroma 使用 HNSW（Hierarchical Navigable Small World）算法做近似最近邻搜索。HNSW 的核心思想是构建一个多层图结构，搜索时从顶层开始逐层向下导航，最终在底层找到最近的邻居。由于 HNSW 是近似算法，它返回的距离值可能不等于精确计算的距离——但误差通常很小（< 1%），对排序结果几乎没有影响。

```
┌─────────────────────────────────────────────────────────────┐
│  HNSW 近似搜索 vs 精确搜索                                   │
│                                                             │
│  精确搜索（Brute Force）：                                   │
│    计算查询向量与所有 N 个向量的距离 → 排序 → 返回 top-K      │
│    时间复杂度: O(N × d)                                      │
│    结果: 100% 精确                                           │
│                                                             │
│  HNSW 近似搜索：                                             │
│    在多层图中导航 → 找到近似最近的 top-K                       │
│    时间复杂度: O(log N × d)                                  │
│    结果: Recall@10 通常 > 95%（取决于参数配置）               │
│                                                             │
│  Chroma 的 HNSW 默认参数:                                    │
│    M = 16 (每层最大连接数)                                    │
│    ef_construction = 100 (建索引时的搜索宽度)                  │
│    ef_search = 10 (查询时的搜索宽度)                          │
│                                                             │
│  → 在 10K~1M 向量规模下，HNSW 的速度优势是                   │
│    Brute Force 的 10~100 倍，而召回率损失 < 5%                │
└─────────────────────────────────────────────────────────────┘
```

---

## 完整实战：对比三种度量的检索效果

```python
import chromadb
import numpy as np
from chromadb.utils import embedding_functions

ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="paraphrase-multilingual-MiniLM-L12-v2"
)

docs = [
    "机器学习是人工智能的核心技术",
    "深度学习使用多层神经网络提取特征",
    "自然语言处理让计算机理解人类语言",
    "计算机视觉用于图像和视频的分析",
    "强化学习通过奖励信号训练智能体",
    "今天天气真好，适合出门散步",       # 无关文档
    "股市今日大涨，科技股领涨",         # 无关文档
]
ids = [f"d{i}" for i in range(len(docs))]

# 创建三个不同度量的 Collection
results_comparison = {}
for metric in ["cosine", "l2", "ip"]:
    client = chromadb.Client()  # 每个 metric 用独立的 client 避免冲突
    col = client.create_collection(
        name=f"metric_{metric}",
        embedding_function=ef,
        metadata={"hnsw:space": metric}
    )
    col.add(documents=docs, ids=ids)

    r = col.query(query_texts=["AI技术有哪些"], n_results=3, include=["documents", "distances"])
    results_comparison[metric] = list(zip(r['documents'][0], r['distances'][0]))

print("=== 三种度量的检索结果对比 ===\n")
for metric, results in results_comparison.items():
    print(f"[{metric.upper()}]")
    for doc, dist in results:
        print(f"  ({dist:.4f}) {doc}")
    print()
```

典型输出：

```
[COSINE]
  (0.3124) 机器学习是人工智能的核心技术
  (0.4567) 深度学习使用多层神经网络提取特征
  (0.5234) 强化学习通过奖励信号训练智能体

[L2]
  (0.7891) 机器学习是人工智能的核心技术
  (1.0234) 深度学习使用多层神经网络提取特征
  (1.1567) 强化学习通过奖励信号训练智能体

[IP]
  (0.6876) 机器学习是人工智能的核心技术
  (0.5433) 深度学习使用多层神经网络提取特征
  (0.4766) 强化学习通过奖励信号训练智能体
```

由于 SentenceTransformers 输出的向量已经是归一化的，三种度量给出了相同的排序——只是距离数值不同。这验证了我们前面的理论：**对于归一化向量，三种度量等价**。

---

## 常见误区

### 误区 1：认为 Cosine 距离就是余弦相似度

Cosine 距离 = 1 - 余弦相似度。距离越小越相似，相似度越大越相似。在 Chroma 的返回值中，`distances` 字段是距离值，不是相似度值。如果你需要相似度，需要手动转换：

```python
# Chroma 返回的是距离，不是相似度
distance = results['distances'][0][0]
similarity = 1 - distance  # Cosine 距离转相似度
print(f"距离: {distance:.4f}, 相似度: {similarity:.4f}")
```

### 误区 2：在同一个 Collection 中混用不同度量的向量

如果你先用 cosine 度量创建了一个 Collection 并添加了数据，然后删除重建为 l2 度量——虽然 ID 可以复用，但向量空间已经变了。你必须用新的度量重新编码所有文档。

### 误区 3：认为距离值可以直接跨 Collection 比较

不同 Collection 的距离值不可直接比较——即使它们用了相同的度量。因为每个 Collection 的 HNSW 索引是独立构建的，距离值的绝对大小受数据分布影响。你应该只比较同一个 Collection 内的距离排序。

---

## 本章小结

距离度量是向量搜索的"度量衡"——它定义了"相似"的数学含义。核心要点回顾：第一，L2 对模长敏感，Cosine 只看方向，IP 最快但要求归一化；第二，对于归一化向量，三种度量排序完全等价，这也是 Chroma 默认选 Cosine 的原因——它对归一化不完美的情况最鲁棒；第三，如果你手动传入向量，务必确保归一化后再用 IP 度量，否则模长差异会干扰排序；第四，HNSW 返回的是近似距离，与精确计算有微小误差但对排序几乎无影响；第五，不同 Collection 的距离值不可直接比较，只看排序即可。

下一章我们将进入高级查询与过滤——深入 `query()` 方法的每个参数、Where 过滤器的完整语法、以及多阶段查询与 Re-ranking 策略。
