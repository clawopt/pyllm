---
title: 向量存储与嵌入模型
description: Embeddings 嵌入模型原理、Chroma / FAISS / Milvus Lite 对比、嵌入模型选择指南
---
# 向量存储与嵌入模型

上一节我们把文档切成了文本块。但这些块目前还只是普通的字符串——计算机无法理解"这段文字和那段文字在语义上是否相近"。要让检索成为可能，我们需要完成 RAG 中最关键的技术转换：**把文本变成向量**。

## 嵌入（Embedding）：文本到向量的翻译官

**嵌入模型**的作用是把一段文字映射到一个高维空间中的点（一个数字数组），使得**语义相似的文字在空间中的距离更近，语义不同的文字距离更远**：

```python
from langchain_openai import OpenAIEmbeddings

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

vector = embedding_model.embed_query("Python 是一门编程语言")

print(f"向量维度: {len(vector)}")  # 1536
print(f"前 5 个值: {vector[:5]}")
# [0.0023, -0.0145, 0.0087, 0.0231, -0.0056]
```

这个向量有 **1536 维**——"Python 是一门编程语言"这句话被编码成了 1536 个浮点数。每个数字都承载着某种语义特征的强度，虽然人类无法直接解读这些数字的含义，但数学运算可以很好地利用它们来衡量文本之间的相似度。

### 验证语义相似度

```python
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

pairs = [
    ("猫坐在垫子上", "一只小猫躺在地毯上"),       # 语义相近
    ("Python 编程语言", "Java 程序设计"),           # 同领域
    ("今天天气很好", "深度学习的梯度下降算法"),      # 完全无关
]

vectors = embedding_model.embed_documents([p[0] for p in pairs] + [p[1] for p in pairs])

for i, (a, b) in enumerate(pairs):
    sim = cosine_similarity(vectors[i], vectors[i + len(pairs)])
    print(f'"{a}" vs "{b}" → {sim:.4f}')
```

典型输出：

```
"猫坐在垫子上" vs "一只小猫躺在地毯下" → 0.9234   ← 很接近
"Python 编程语言" vs "Java 程序设计" → 0.7856     ← 比较接近
"今天天气很好" vs "深度学习的梯度下降算法" → 0.2143  ← 差很远
```

**语义越接近的文本，余弦相似度越接近 1；语义无关的文本，相似度接近 0。** 这就是向量检索能够工作的根本原理。

## 嵌入模型的选择

| 模型 | 提供方 | 维度 | 特点 | 适用场景 |
|------|--------|------|------|---------|
| `text-embedding-3-small` | OpenAI | 1536 | 性价比最高 | **通用首选** |
| `text-embedding-3-large` | OpenAI | 3072 | 精度更高 | 高精度需求 |
| `bge-small-zh-v1.5` | BAAI (开源) | 512 | 中文优化、免费本地运行 | 中文场景 / 隐私敏感 |
| `nomic-embed-text` | Nomic (开源) | 768 | 多语言支持 | 本地多语言场景 |
| `e5-mistral-7b-instruct` | Mistral (开源) | 1024 | 指令跟随能力强 | 复杂查询理解 |

### OpenAI 嵌入模型（云端）

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 单条嵌入
vec = embeddings.embed_query("Hello world")

# 批量嵌入
vecs = embeddings.embed_texts(["Hello", "World", "Test"])
```

优点：质量高、API 简单、无需维护基础设施。
缺点：需要网络调用、有 API 成本、数据离开本地机器。

### 本地嵌入模型（离线/隐私优先）

```python
# 方案一：通过 Ollama 运行本地模型
from langchain_ollama import OllamaEmbeddings

local_emb = OllamaEmbeddings(model="nomic-embed-text")
vec = local_emb.embed_query("你好世界")

# 方案二：通过 HuggingFace 直接加载
from langchain_huggingface import HuggingFaceEmbeddings

hf_emb = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5"
)
vec = hf_emb.embed_query("Python是一门什么语言？")
```

优点：零 API 成本、数据不出本地、无网络依赖。
缺点：需要足够的内存和 GPU/CPU 资源、精度通常略低于大模型。

## 向量数据库（Vector Store）

有了文本向量后，需要一个专门的数据结构来高效地存储和搜索它们——这就是 **向量数据库**。LangChain 对主流向量库做了统一抽象，切换后端几乎不需要改代码。

### Chroma：最易上手的选择

```bash
pip install chromadb langchain-chroma
```

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

documents = [
    "Python 是一门解释型、动态类型的编程语言",
    "Java 使用静态类型检查，编译后运行在 JVM 上",
    "Go 语言由 Google 开发，强调简洁和并发性能",
]

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory="./chroma_db"   # 持久化到磁盘
)

# 相似度搜索
results = vectorstore.similarity_search("动态类型语言的特点", k=2)
for doc in results:
    print(doc.page_content)
```

输出：

```
Python 是一门解释型、动态类型的编程语言
```

只返回了最相关的那个结果（因为只有它包含"动态类型"这个关键词）。

### FAISS：高性能内存搜索

```bash
pip install faiss-cpu langchain-community
```

```python
from langchain_community.vectorstores import FAISS

faiss_index = FAISS.from_texts(documents, embeddings)
results = faiss_index.similarity_search("并发编程", k=2)

# 保存到磁盘 / 从磁盘加载
faiss_index.save_local("faiss_index")
loaded = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
```

### Milvus Lite：轻量级生产方案

```bash
pip install pymilvus langchain-community
```

```python
from langchain_community.vectorstores import Milvus

milvus_store = Milvus.from_texts(
    texts=documents,
    embedding=embeddings,
    connection_args={"uri": "./milvus_demo.db"},
    collection_name="demo_collection"
)

results = milvus_store.similarity_search("Google开发的", k=1)
```

Milvus Lite 是 Milvus 的嵌入式版本，不需要单独的服务进程，适合中小型项目。当数据量增长到百万级以上时，可以无缝升级到完整的 Milvus 集群部署。

### 三种向量库对比

| 特性 | Chroma | FAISS | Milvus Lite |
|------|-------|------|-----------|
| 部署难度 | ⭐ 最简单 | ⭐⭐ 简单 | ⭐⭐ 中等 |
| 持久化 | ✅ 内置 | 需手动 save/load | ✅ 内置 |
| 元数据过滤 | ✅ 支持 | ❌ 不支持 | ✅ 支持 |
| 规模上限 | 十万级 | 百万级（内存） | 百万级+ |
| 生产就绪 | 原型/小项目 | 中小型项目 | 可扩展到生产 |
| 推荐场景 | 学习/原型开发 | 快速实验/临时使用 | 正式项目 |

对于学习阶段和大多数原型项目，**Chroma** 是最佳选择——零配置、自带持久化、支持元数据过滤。当你对数据规模或查询灵活性有更高要求时，再考虑迁移到 Milvus 或其他方案。

## 从 Vector Store 创建 Retriector

在 LangChain 的标准 Chain 架构中，检索功能通过 **Retriever** 接口来使用。每个 Vector Store 都有 `.as_retriever()` 方法：

```python
retriever = vectorstore.as_retriever(
    search_type="similarity",   # 搜索策略
    k=3                         # 返回 top-k 个结果
)

results = retriever.invoke("什么是装饰器？")
for doc in results:
    print(f"[{doc.metadata.get('source', '?')}] {doc.page_content[:60]}...")
```

到这里，RAG 数据准备流水线的前半部分（加载 → 分块 → 向量化 → 存储）已经完成。下一节我们将学习如何从向量库中精准地检索相关信息——包括基础的相似度搜索和高级的 MMR 多样性搜索、上下文压缩等策略。
