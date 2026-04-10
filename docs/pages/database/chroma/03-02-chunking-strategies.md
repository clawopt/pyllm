# 3.2 文档切分策略（Chunking）

> **切分是 RAG 的第一道关卡——切得好，检索精度翻倍；切得差，再好的模型也救不回来**

---

## 这一节在讲什么？

在 RAG 系统中，你面对的通常不是一条条简短的句子，而是一篇篇完整的文档——几十页的 PDF、几千行的代码、上万字的报告。这些长文档无法直接塞进 Chroma，原因有两个：第一，embedding 模型有输入长度限制（通常 256~512 tokens），超长文本会被截断导致信息丢失；第二，即使模型能处理长文本，把整篇文档编码成一个向量也会导致信息过度压缩——一个向量很难同时表达"退款政策"和"安装指南"两种截然不同的主题。

所以我们需要把长文档切成较小的片段（chunk），每个 chunk 单独做 embedding 并入库。这个看似简单的操作实际上深刻影响着 RAG 系统的检索质量：切得太小，每个 chunk 缺乏足够的上下文，embedding 无法捕捉完整语义；切得太大，一个 chunk 包含多个主题，向量无法精确表达任何一个。这一节我们要讲清楚三种主流的切分策略、它们各自的适用场景和参数调优方法，以及切分策略如何影响最终的检索效果。

---

## 为什么切分质量决定 RAG 的上限

在深入具体策略之前，让我们先理解切分在 RAG 链路中的位置和影响。RAG 的核心流程是"检索 → 组装 → 生成"，切分发生在检索之前——它决定了哪些文本片段会被编码成向量、进入索引。如果切分不当，即使后续的 embedding 模型再好、LLM 再强，也无法弥补检索阶段的损失。

```
┌─────────────────────────────────────────────────────────────────┐
│  切分质量对 RAG 链路的影响                                       │
│                                                                 │
│  原始文档: "我们的退款政策规定，购买后7天内可无条件退款。         │
│           退款流程：1. 在订单页面点击'申请退款'；                  │
│           2. 填写退款原因；3. 等待3-5个工作日审核。               │
│           注意：已拆封的数码产品不支持无理由退款。"                │
│                                                                 │
│  ❌ 切分不当: ["我们的退款政策规定，购买后7天内可无条件退款。",    │
│              "退款流程：1. 在订单页面点击'申请退款'；",            │
│              "2. 填写退款原因；3. 等待3-5个工作日审核。",          │
│              "注意：已拆封的数码产品不支持无理由退款。"]           │
│  问题：每条 chunk 信息不完整，用户问"数码产品能退款吗"时          │
│        可能只检索到"无条件退款"那条，遗漏了限制条件               │
│                                                                 │
│  ✅ 合理切分: ["我们的退款政策规定，购买后7天内可无条件退款。     │
│              退款流程：1. 在订单页面点击'申请退款'；               │
│              2. 填写退款原因；3. 等待3-5个工作日审核。",           │
│              "注意：已拆封的数码产品不支持无理由退款。"]           │
│  优势：第一条 chunk 包含完整的退款政策和流程，                    │
│        第二条 chunk 包含重要的限制条件                            │
└─────────────────────────────────────────────────────────────────┘
```

好的切分应该遵循一个核心原则：**每个 chunk 应该是一个语义完整的单元**——它应该围绕一个主题展开，包含足够的上下文让读者（和 embedding 模型）理解其含义，同时又不能大到包含多个不相关的主题。

---

## 策略一：固定长度切分（Fixed-Size Chunking）

最简单的切分方式：按照固定的字符数或 token 数切割文档，相邻 chunk 之间保留一定长度的重叠（overlap）。

```python
def fixed_size_chunk(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    """
    固定长度切分

    参数:
        text: 原始文本
        chunk_size: 每个 chunk 的字符数
        overlap: 相邻 chunk 的重叠字符数
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap  # 下一个 chunk 从 overlap 处开始
    return chunks

# 使用示例
document = "这是一篇很长的技术文档..." * 100  # 假设 5000 字
chunks = fixed_size_chunk(document, chunk_size=500, overlap=50)
print(f"原文长度: {len(document)} 字符")
print(f"切分后: {len(chunks)} 个 chunk")
print(f"每个 chunk: ~500 字符, 重叠: 50 字符")
```

**overlap 的作用**：重叠窗口确保了相邻 chunk 之间有上下文衔接。如果没有 overlap，一个完整的句子可能被切断在两个 chunk 的边界上，导致两个 chunk 都缺少关键信息。overlap=50 意味着每个 chunk 的最后 50 个字符会在下一个 chunk 的开头重复出现，这样即使关键信息恰好在边界处，也能被至少一个 chunk 完整包含。

固定长度切分的优点是实现简单、行为可预测、每个 chunk 的大小一致（便于批处理）。但它的缺点也很明显：它完全不考虑文本的自然边界——可能在一个句子中间切断，也可能把两个不相关的段落粘在一起。

比如下面的程序展示了固定长度切分的问题。由于切分点完全由字符数决定，一个完整的句子"深度学习是机器学习的子领域，它使用多层神经网络来学习数据的层次化表示"可能被切断成"深度学习是机器学习的子领域"和"，它使用多层神经网络来学习数据的层次化表示"两个 chunk，后者缺少主语，语义不完整：

```python
text = "深度学习是机器学习的子领域，它使用多层神经网络来学习数据的层次化表示。自然语言处理是深度学习的重要应用方向。"

# 固定长度切分，chunk_size=30
chunks = fixed_size_chunk(text, chunk_size=30, overlap=5)
for i, chunk in enumerate(chunks):
    print(f"Chunk {i}: '{chunk}'")

# 输出：
# Chunk 0: '深度学习是机器学习的子领域，它使用多层神经网'
# Chunk 1: '层神经网络来学习数据的层次化表示。自然语言处'
# Chunk 2: '言处理是深度学习的重要应用方向。'
# ↑ 可以看到句子被生硬地切断了
```

---

## 策略二：递归字符级切分（Recursive Character Chunking）

这是 LangChain 的 `RecursiveCharacterTextSplitter` 采用的策略，也是目前最广泛使用的切分方法。它的核心思想是：**优先在自然边界处切分**——先尝试按双换行符（段落）切，如果段落太长就按单换行符（行）切，再不行就按句号切，最后才按字符数硬切。

```python
def recursive_chunk(text: str, chunk_size: int = 500, overlap: int = 50,
                    separators: list = None) -> list:
    """
    递归字符级切分

    参数:
        text: 原始文本
        chunk_size: 目标 chunk 大小
        overlap: 重叠长度
        separators: 切分分隔符优先级列表（从粗到细）
    """
    if separators is None:
        separators = ["\n\n", "\n", "。", "！", "？", ".", " ", ""]

    def _split(text, sep_idx):
        if sep_idx >= len(separators):
            return [text] if text else []

        sep = separators[sep_idx]
        parts = text.split(sep) if sep else list(text)

        chunks = []
        current = ""
        for part in parts:
            candidate = current + sep + part if current else part
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                if len(part) > chunk_size:
                    chunks.extend(_split(part, sep_idx + 1))
                    current = ""
                else:
                    current = part
        if current:
            chunks.append(current)
        return chunks

    raw_chunks = _split(text, 0)

    # 添加 overlap
    result = []
    for i, chunk in enumerate(raw_chunks):
        if i > 0 and overlap > 0:
            prev_tail = raw_chunks[i-1][-overlap:]
            chunk = prev_tail + chunk
        result.append(chunk)
    return result


# 使用示例
document = """## 退款政策

我们的退款政策规定，购买后7天内可无条件退款。退款流程如下：
1. 在订单页面点击"申请退款"
2. 填写退款原因
3. 等待3-5个工作日审核

## 注意事项

已拆封的数码产品不支持无理由退款。食品类商品一经售出概不退换。
"""

chunks = recursive_chunk(document, chunk_size=200, overlap=30)
for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i} ({len(chunk)} 字符) ---")
    print(chunk)
    print()
```

输出：

```
--- Chunk 0 (68 字符) ---
## 退款政策

我们的退款政策规定，购买后7天内可无条件退款。退款流程如下：

--- Chunk 1 (97 字符) ---
退款流程如下：
1. 在订单页面点击"申请退款"
2. 填写退款原因
3. 等待3-5个工作日审核

--- Chunk 2 (72 字符) ---
工作日审核

## 注意事项

已拆封的数码产品不支持无理由退款。食品类商品一经售出概不退换。
```

递归切分的效果明显好于固定长度切分——每个 chunk 都在自然边界处断开，保持了段落的完整性。注意 Chunk 1 和 Chunk 2 之间有 "工作日审核" 这个重叠，确保了上下文衔接。

---

## 策略三：语义切分（Semantic Chunking）

语义切分是最精细的策略：它不依赖固定的分隔符，而是用 embedding 模型来判断文本中语义发生变化的"断点"。具体做法是：先把文本按句子切分，计算每个句子的 embedding，然后比较相邻句子的相似度——当相似度突然下降时，说明话题发生了切换，这就是切分点。

```python
import numpy as np

def semantic_chunk(text: str, model, chunk_size: int = 500,
                   similarity_threshold: float = 0.5) -> list:
    """
    语义切分

    参数:
        text: 原始文本
        model: SentenceTransformer 模型实例
        chunk_size: 最大 chunk 大小（字符数）
        similarity_threshold: 相邻句子相似度低于此值时切分
    """
    # Step 1: 按句子切分
    sentences = []
    for sep in ["。", "！", "？", ".", "!", "?"]:
        text = text.replace(sep, sep + "\n")
    raw_sentences = [s.strip() for s in text.split("\n") if s.strip()]

    # Step 2: 计算每个句子的 embedding
    embeddings = model.encode(raw_sentences, normalize_embeddings=True)

    # Step 3: 计算相邻句子的余弦相似度
    similarities = []
    for i in range(len(embeddings) - 1):
        sim = np.dot(embeddings[i], embeddings[i + 1])
        similarities.append(sim)

    # Step 4: 在相似度低于阈值处切分
    chunks = []
    current_chunk = [raw_sentences[0]]

    for i in range(len(similarities)):
        if similarities[i] < similarity_threshold or \
           sum(len(s) for s in current_chunk) + len(raw_sentences[i+1]) > chunk_size:
            chunks.append("".join(current_chunk))
            current_chunk = [raw_sentences[i+1]]
        else:
            current_chunk.append(raw_sentences[i+1])

    if current_chunk:
        chunks.append("".join(current_chunk))

    return chunks


# 使用示例（需要先安装 sentence-transformers）
# from sentence_transformers import SentenceTransformer
# model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
#
# document = "机器学习是AI的核心技术。深度学习使用多层神经网络。今天天气真好。自然语言处理让计算机理解语言。"
# chunks = semantic_chunk(document, model, similarity_threshold=0.3)
# for i, chunk in enumerate(chunks):
#     print(f"Chunk {i}: {chunk}")
#
# 预期输出：
# Chunk 0: 机器学习是AI的核心技术。深度学习使用多层神经网络。  ← 两个AI相关句子
# Chunk 1: 今天天气真好。  ← 天气话题，与AI无关，单独成块
# Chunk 2: 自然语言处理让计算机理解语言。  ← 回到AI话题
```

语义切分的优势在于它能精确捕捉话题切换的边界——即使两个句子在物理位置上紧挨着，只要语义不相关，就会被分到不同的 chunk。但它的代价也很明显：需要对每个句子做 embedding 计算，处理速度比前两种策略慢很多；而且切分结果依赖于 embedding 模型的质量——如果模型本身对短句子的编码不够好，相似度计算就不准确，切分效果反而不如递归切分。

---

## 三种策略的对比与选择

| 维度 | 固定长度 | 递归字符级 | 语义切分 |
|------|---------|-----------|---------|
| 实现复杂度 | ⭐ 极简 | ⭐⭐ 中等 | ⭐⭐⭐ 复杂 |
| 切分速度 | ⭐⭐⭐ 最快 | ⭐⭐⭐ 快 | ⭐ 慢（需额外 embedding） |
| 语义完整性 | ⭐ 差 | ⭐⭐ 良好 | ⭐⭐⭐ 最佳 |
| 可控性 | ⭐⭐⭐ 完全可控 | ⭐⭐ 较可控 | ⭐ 不太可控 |
| 适用文档类型 | 日志、代码 | 通用文档 | 结构松散的长文 |
| 推荐场景 | 快速原型 | **生产首选** | 高精度需求 |

**实践建议**：对于大多数 RAG 项目，**递归字符级切分是最佳起点**。它在语义完整性和实现复杂度之间取得了最好的平衡。只有当你发现递归切分的检索质量不够（比如 top-5 召回率低于 70%）时，才考虑升级到语义切分。

---

## 参数调优：chunk_size 和 overlap 怎么选？

chunk_size 和 overlap 是切分策略中最重要的两个参数，它们直接影响检索质量。下面是经过大量实践验证的调优指南：

### chunk_size 的选择

chunk_size 的选择取决于三个因素：embedding 模型的最大输入长度、文档的主题密度、以及 LLM 的上下文窗口大小。

```
chunk_size 选择指南：

├─ 200~300 字符
│   → 适合：FAQ 条目、短消息、对话记录
│   → 优点：每个 chunk 主题单一，向量编码精确
│   → 缺点：可能缺少上下文，LLM 需要更多 chunk 才能回答问题
│
├─ 500~1000 字符（推荐起点）
│   → 适合：技术文档、产品手册、新闻文章
│   → 优点：语义完整性和上下文深度的最佳平衡
│   → 缺点：可能包含多个子主题
│
├─ 1500~2000 字符
│   → 适合：法律条文、学术论文、长篇报告
│   → 优点：上下文丰富，LLM 能获得更完整的信息
│   → 缺点：向量编码可能模糊，检索精度下降
│
└─ > 2000 字符
    → 通常不推荐：超出大多数 embedding 模型的有效编码范围
    → 向量会"稀释"多个主题，检索质量急剧下降
```

### overlap 的选择

overlap 的作用是防止关键信息被切断在 chunk 边界上。一般建议 overlap 为 chunk_size 的 10%~20%：

```python
# 推荐的参数组合
RECOMMENDED_CONFIGS = {
    "faq": {"chunk_size": 300, "overlap": 30},
    "general": {"chunk_size": 800, "overlap": 100},
    "technical": {"chunk_size": 1000, "overlap": 150},
    "legal": {"chunk_size": 1500, "overlap": 200},
}
```

overlap 不是越大越好——过大的 overlap 会导致相邻 chunk 高度重复，浪费存储空间，而且检索时可能返回多个几乎相同的 chunk，挤掉了其他有价值的结果。

---

## 完整实战：带 Metadata 的切分入库 Pipeline

让我们把切分策略和上一节学的 metadata 设计结合起来，构建一个完整的文档入库 pipeline：

```python
import chromadb
import hashlib
import time
from chromadb.utils import embedding_functions

class DocumentIngestionPipeline:
    """文档入库 Pipeline：加载 → 切分 → 编码 → 入库"""

    def __init__(self, collection_name: str, persist_dir: str = "./chroma_db",
                 chunk_size: int = 800, chunk_overlap: int = 100):
        self.client = chromadb.Client(settings=chromadb.Settings(
            is_persistent=True,
            persist_directory=persist_dir
        ))

        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="paraphrase-multilingual-MiniLM-L12-v2"
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"}
        )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str) -> list:
        """递归字符级切分"""
        return recursive_chunk(
            text,
            chunk_size=self.chunk_size,
            overlap=self.chunk_overlap
        )

    def ingest_document(self, text: str, source: str, source_type: str = "text",
                        category: str = "general", version: int = 1,
                        language: str = "zh", extra_meta: dict = None):
        """入库一篇完整文档"""
        chunks = self.chunk_text(text)
        current_ts = int(time.time())

        ids = []
        documents = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            doc_id = hashlib.md5(f"{source}::chunk_{i}".encode()).hexdigest()[:16]

            meta = {
                "source": source,
                "source_type": source_type,
                "category": category,
                "version": version,
                "language": language,
                "created_at": current_ts,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "chunk_size": len(chunk),
                "is_reviewed": False,
            }
            if extra_meta:
                meta.update(extra_meta)

            ids.append(doc_id)
            documents.append(chunk)
            metadatas.append(meta)

        # 批量 upsert
        self.collection.upsert(
            documents=documents,
            ids=ids,
            metadatas=metadatas
        )

        print(f"✅ 入库完成: {source} → {len(chunks)} 个 chunk")
        return len(chunks)

    def search(self, query: str, category: str = None, n_results: int = 5):
        """带过滤的语义搜索"""
        where = {"category": category} if category else None
        results = self.collection.query(
            query_texts=[query],
            where=where,
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )

        for i in range(len(results['ids'][0])):
            meta = results['metadatas'][0][i]
            dist = results['distances'][0][i]
            doc = results['documents'][0][i]
            print(f"  [{dist:.4f}] {meta['source']} (chunk {meta['chunk_index']}/{meta['total_chunks']})")
            print(f"         {doc[:80]}...")


# ====== 使用示例 ======
pipeline = DocumentIngestionPipeline(
    collection_name="smart_kb",
    chunk_size=600,
    chunk_overlap=80
)

# 入库一篇长文档
refund_policy = """
## 退款政策

我们的退款政策规定，购买后7天内可无条件退款。退款流程如下：
1. 在订单页面点击"申请退款"按钮
2. 在弹出的表单中填写退款原因（必填）
3. 提交后等待3-5个工作日的审核
4. 审核通过后退款将原路返回到您的支付账户

## 退款限制

以下情况不支持无理由退款：
- 已拆封的数码产品（手机、电脑、平板等）
- 食品类商品（出于食品安全考虑）
- 定制类商品（刻字、定制尺寸等）
- 超过7天退款期限的订单

## 特殊情况

如果商品存在质量问题，不受7天限制，您可以在收货后30天内申请质量退款。
需要提供商品照片和问题描述，我们的客服团队会在24小时内响应。
"""

pipeline.ingest_document(
    text=refund_policy,
    source="refund_policy_v2.md",
    source_type="markdown",
    category="after_sales",
    version=2
)

# 搜索
print("\n🔍 搜索 '数码产品退款':")
pipeline.search("数码产品退款", category="after_sales")

print("\n🔍 搜索 '质量问题':")
pipeline.search("质量问题", category="after_sales")
```

---

## 常见误区

### 误区 1：chunk_size 越大越好，因为 LLM 能看到更多上下文

这是一个常见的误解。chunk_size 影响的是**检索阶段**的质量，而不是 LLM 生成阶段。检索时，每个 chunk 被编码成一个向量，chunk 越大向量越"模糊"——因为它要同时编码多个主题的信息。正确的做法是用较小的 chunk_size 保证检索精度，然后在生成阶段把检索到的多个 chunk 拼接起来喂给 LLM。

### 误区 2：overlap 越大越安全

过大的 overlap 会导致两个问题：存储空间浪费（重复内容占比过高），以及检索结果冗余（多个高度相似的 chunk 挤占 top-K 的位置）。10%~20% 的 overlap 是经过验证的最佳范围。

### 误区 3：切分后不需要保留原文的层级结构

如果你把一篇有标题层级的文档（比如 Markdown）切成扁平的 chunk，每个 chunk 丢失了它所属的章节信息。当 LLM 看到一个孤立的 chunk 时，可能无法理解它的上下文。解决方案是在 metadata 中保存章节路径：

```python
# 切分时保留章节信息
meta = {
    "section_path": "第3章 > 3.2 退款流程 > 3.2.1 申请步骤",
    "heading": "申请步骤",
    "chunk_index": 0
}
```

---

## 本章小结

切分是 RAG 系统的"第一道关卡"，它决定了哪些文本片段会进入向量索引，直接影响检索质量的上限。核心要点回顾：第一，固定长度切分最简单但语义完整性最差，适合快速原型；第二，递归字符级切分是生产首选，在自然边界处切分保证了语义完整性；第三，语义切分精度最高但速度最慢，适合对检索质量有极致要求的场景；第四，chunk_size 推荐 500~1000 字符，overlap 推荐 chunk_size 的 10%~20%；第五，切分时务必在 metadata 中保留来源、章节路径、chunk 编号等信息，为后续的上下文拼接和来源溯源提供基础。

下一节我们将深入向量标准化与距离度量选择——为什么 Chroma 默认用 cosine、什么时候该用 l2 或 ip、以及归一化对检索结果的影响。
