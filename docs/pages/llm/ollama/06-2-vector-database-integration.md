# 06-2 向量数据库集成

## 为什么需要向量数据库

上一节我们学会了如何用 Ollama 的 Embedding 模型把文本变成向量。但如果你只有几十个文档，用 numpy 做一次暴力相似度计算（两两比较所有向量）完全没问题——这叫**穷举搜索（Brute Force Search）**，时间复杂度是 O(n²)。

但当你有 10 万、100 万甚至更多文档时，每次查询都要和所有文档向量做一遍余弦相似度计算，延迟会从毫秒级飙升到秒级甚至分钟级。这时候就需要**向量数据库（Vector Database）**了——它通过 **ANN（Approximate Nearest Neighbor，近似最近邻）** 算法，能在亚毫秒到毫秒级别从百万级向量中找到最相似的 K 个结果。

```
┌─────────────────────────────────────────────────────────────┐
│              暴力搜索 vs 向量数据库                           │
│                                                             │
│  文档数量: 10,000                                           │
│  向量维度: 768                                             │
│                                                             │
│  暴力搜索 (numpy):                                          │
│  ├── 计算量: 10,000 × 768 = 768万次浮点运算                 │
│  ├── 延迟: ~50ms (还可以)                                   │
│  └── 内存: 全部加载 (76MB)                                  │
│                                                             │
│  文档数量: 1,000,000                                        │
│                                                             │
│  暴力搜索 (numpy):                                          │
│  ├── 计算量: 1M × 768 = 7.68亿次浮点运算 (!!)              │
│  ├── 延迟: ~5-15秒 (不可接受)                               │
│  └── 内存: 全部加载 (7.6GB)                                 │
│                                                             │
│  向量数据库 (Chroma/FAISS/Qdrant):                          │
│  ├── 通过索引跳过大部分无关向量                              │
│  ├── 延迟: 5-50ms (可接受)                                  │
│  └── 内存: 只加载索引 + 部分向量                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Ollama + ChromaDB 集成（推荐入门方案）

ChromaDB 是目前最易用的开源向量数据库之一——它是一个纯 Python 库，安装即用，无需额外服务，非常适合本地 RAG 开发。

### 安装与基本使用

```bash
pip install chromadb
```

### 完整集成代码：Ollama Embedding + ChromaDB

```python
#!/usr/bin/env python3
"""
Ollama + ChromaDB 完整集成示例
功能: 文档 → 分块 → Ollama Embedding → ChromaDB 存储 → 语义检索
"""

import chromadb
from chromadb.config import Settings
import requests
import time
import json
from typing import List, Dict, Optional


class OllamaChromaRAG:
    """基于 Ollama Embedding 和 ChromaDB 的 RAG 引擎"""
    
    def __init__(self,
                 collection_name="documents",
                 embedding_model="nomic-embed-text",
                 llm_model="qwen2.5:7b",
                 persist_dir="./chroma_data",
                 ollama_url="http://localhost:11434"):
        
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.ollama_url = ollama_url
        
        # 初始化 ChromaDB 客户端（持久化模式）
        self.client = chromadb.PersistentClient(path=persist_dir)
        
        # 获取或创建 collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # 使用余弦距离
        )
    
    def _embed_with_ollama(self, text: str) -> List[float]:
        """调用 Ollama API 生成 Embedding"""
        resp = requests.post(
            f"{self.ollama_url}/api/embeddings",
            json={
                "model": self.embedding_model,
                "prompt": text
            },
            timeout=30
        )
        return resp.json()["embedding"]
    
    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """批量生成 Embedding（串行，Ollama 不支持并行嵌入）"""
        embeddings = []
        for i, text in enumerate(texts):
            if (i + 1) % 20 == 0:
                print(f"    Embedding 进度: {i+1}/{len(texts)}")
            vec = self._embed_with_ollama(text)
            embeddings.append(vec)
            time.sleep(0.005)  # 避免请求过快
        return embeddings
    
    def add_documents(self, documents: List[Dict]):
        """
        添加文档到向量库
        
        Args:
            documents: 文档列表，每个元素包含:
                - content: 文档内容
                - metadata: 元数据字典 (可选)
                - id: 文档唯一ID (可选)
        """
        
        ids = []
        contents = []
        metadatas = []
        
        for i, doc in enumerate(documents):
            ids.append(doc.get("id", f"doc_{i}"))
            contents.append(doc["content"])
            metadatas.append(doc.get("metadata", {}))
        
        print(f"\n📥 正在添加 {len(documents)} 个文档...")
        print(f"   正在生成 Embedding (模型: {self.embedding_model})...")
        
        start = time.time()
        embeddings = self._embed_batch(contents)
        embed_time = time.time() - start
        
        print(f"   Embedding 完成 ({embed_time:.1f}s)")
        print(f"   正在写入 ChromaDB...")
        
        self.collection.add(
            ids=ids,
            documents=contents,
            embeddings=embeddings,
            metadatas=metadatas
        )
        
        total_time = time.time() - start
        print(f"   ✅ 完成! 总耗时: {total_time:.1f}s")
        print(f"   当前集合大小: {self.collection.count()} 个文档")
    
    def query(self, 
               text: str, 
               n_results: int = 5,
               where: Optional[Dict] = None,
               min_similarity: float = 0.0) -> List[Dict]:
        """
        语义搜索
        
        Args:
            text: 查询文本
            n_results: 返回结果数量
            where: 元数据过滤条件
            min_similarity: 最小相似度阈值
            
        Returns:
            检索结果列表
        """
        
        # 生成查询向量
        query_vec = self._embed_with_ollama(text)
        
        # 执行查询
        results = self.collection.query(
            query_embeddings=[query_vec],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"]
        )
        
        formatted_results = []
        
        for i in range(len(results["ids"][0])):
            distance = results["distances"][0][i]
            similarity = 1 - distance  # Chroma 返回的是距离，转换为相似度
            
            if similarity < min_similarity:
                continue
            
            formatted_results.append({
                "id": results["ids"][0][i],
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "similarity": round(similarity, 4),
                "distance": round(distance, 4)
            })
        
        return formatted_results
    
    def generate_answer(self, question: str, context_docs: List[Dict]) -> str:
        """
        基于检索到的上下文生成答案
        
        Args:
            question: 用户问题
            context_docs: 检索到的相关文档
            
        Returns:
            LLM 生成的答案
        """
        
        # 构建 Prompt
        context_text = "\n\n".join([
            f"[文档{i+1}] {doc['content']}"
            for i, doc in enumerate(context_docs)
        ])
        
        prompt = f"""基于以下参考文档回答用户的问题。
如果文档中没有相关信息，请明确说明，不要编造信息。

## 参考文档
{context_text}

## 用户问题
{question}

请根据文档内容给出详细、准确的回答。引用具体文档时请标注来源。"""
        
        resp = requests.post(
            f"{self.ollama_url}/api/chat",
            json={
                "model": self.llm_model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0.3}
            },
            timeout=120
        )
        
        return resp.json()["message"]["content"]


def demo_chroma_integration():
    """完整的 Ollama + ChromaDB 演示"""
    
    rag = OllamaChromaRAG(
        collection_name="tech_docs",
        embedding_model="nomic-embed-text",
        llm_model="qwen2.5:7b"
    )
    
    # 示例文档库
    documents = [
        {
            "id": "doc_001",
            "content": (
                "Ollama 是一个运行大语言模型的工具。它支持 LLaMA、Qwen、"
                "Mistral 等多种开源模型，可以通过命令行或 API 进行交互。"
                "安装非常简单，macOS 用户可以使用 Homebrew 一键安装。"
            ),
            "metadata": {"source": "ollama-guide", "category": "tool"}
        },
        {
            "id": "doc_002",
            "content": (
                "ChromaDB 是一个开源的嵌入式 AI 数据库，专为存储和查询"
                "向量嵌入而设计。它支持多种距离度量方式（余弦、L2、内积），"
                "提供 Python 和 JavaScript 客户端，适合本地开发和小规模部署。"
            ),
            "metadata": {"source": "vector-db-guide", "category": "database"}
        },
        {
            "id": "doc_003",
            "content": (
                "RAG（Retrieval-Augmented Generation）是一种结合了信息检索"
                "和文本生成的技术架构。它首先从知识库中检索相关文档，然后将这些"
                "文档作为上下文提供给大语言模型，从而生成更准确、更有依据的回答。"
            ),
            "metadata": {"source": "rag-tutorial", "category": "architecture"}
        },
        {
            "id": "doc_004",
            "content": (
                "Embedding 是将文本转换为数值向量的技术。语义相似的文本在向量空间中"
                "距离较近，这使得我们可以通过计算向量之间的距离来判断文本的相关性。"
                "常用的 Embedding 模型包括 OpenAI 的 text-embedding-ada-002、"
                "HuggingFace 的 sentence-transformers 系列，以及 Nomic 的 nomic-embed-text。"
            ),
            "metadata": {"source": "embedding-guide", "category": "ml-concept"}
        },
        {
            "id": "doc_005",
            "content": (
                "Docker 容器化部署 Ollama 的步骤：首先创建 docker-compose.yml 文件，"
                "配置端口映射(11434)、GPU 直通和数据卷挂载。然后执行 "
                "'docker compose up -d' 启动服务。生产环境建议配置健康检查、"
                "日志轮转和网络隔离策略。"
            ),
            "metadata": {"source": "docker-deploy", "category": "deployment"}
        }
    ]
    
    # 添加文档
    rag.add_documents(documents)
    
    # 测试查询
    queries = [
        "什么是 RAG？它如何工作？",
        "如何用 Docker 部署 Ollama？",
        "Embedding 模型有哪些选择？",
        "Python 中如何连接和使用 ChromaDB？"
    ]
    
    print("\n" + "=" * 70)
    print("🔍 语义检索测试")
    print("=" * 70)
    
    for query in queries:
        print(f"\n❓ 问题: {query}")
        
        results = rag.query(query, n_results=3)
        
        if results:
            print("\n📄 检索到的相关文档:")
            for r in results:
                bar = "█" * int(r["similarity"] * 20)
                print(f"  [{r['similarity']:.3f}] {bar}")
                print(f"  来源: {r['metadata'].get('source', '?')}")
                print(f"  内容: {r['content'][:120]}...")
            
            # 生成答案
            print(f"\n💡 生成的回答:")
            answer = rag.generate_answer(query, results[:2])
            print(answer[:500])
            if len(answer) > 500:
                print("...")
        else:
            print("  ⚠️ 未找到相关文档")
        
        print("-" * 70)


if __name__ == "__main__":
    demo_chroma_integration()
```

## Ollama + FAISS 集成（高性能 CPU 方案）

FAISS（Facebook AI Similarity Search）是 Meta 出品的高性能向量搜索库，特别适合**纯 CPU 环境**下的大规模向量检索：

```python
#!/usr/bin/env python3
"""Ollama + FAISS 高性能向量检索"""

import requests
import numpy as np
import faiss
import pickle
import os
from typing import List, Dict


class OllamaFAISSIndex:
    """基于 FAISS 的向量索引"""
    
    def __init__(self, 
                 embedding_model="nomic-embed-text",
                 dimension=768,
                 index_type="flat"):
        
        self.embedding_model = embedding_model
        self.dimension = dimension
        self.documents = []  # 存储原始文本
        
        # 创建 FAISS 索引
        if index_type == "flat":
            # 精确搜索（适合 < 10万 文档）
            self.index = faiss.IndexFlatIP(dimension)  # 内积索引
        elif index_type == "ivf":
            # IVF 索引（适合 10万-1000万 文档）
            nlist = 100  # 聚类中心数
            quantizer = faiss.IndexFlatIP(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
        elif index_type == "hnsw":
            # HNSW 索引（高质量+快速，内存占用较大）
            self.index = faiss.IndexHNSWFlat(dimension, M=32)
            self.index.hnsw.efConstruction = 200
            self.index.hnsw.efSearch = 100
    
    def _embed(self, text: str) -> np.ndarray:
        """生成 Embedding 并归一化"""
        resp = requests.post(
            "http://localhost:11434/api/embeddings",
            json={"model": self.embedding_model, "prompt": text},
            timeout=30
        )
        vec = np.array(resp.json()["embedding"], dtype=np.float32)
        # L2 归一化（使内积等价于余弦相似度）
        vec /= np.linalg.norm(vec)
        return vec
    
    def add(self, texts: List[str], ids: List[str] = None):
        """添加文档到索引"""
        
        if ids is None:
            ids = [str(i) for i in range(len(texts))]
        
        # 批量生成 Embedding
        print(f"正在为 {len(texts)} 个文档生成 Embedding...")
        embeddings = np.array([self._embed(t) for t in texts], dtype=np.float32)
        
        # 如果是 IVF 索引，需要先训练
        if hasattr(self.index, 'train') and not self.index.is_trained:
            print("训练 IVF 索引...")
            self.index.train(embeddings)
        
        # 添加到索引
        self.index.add(embeddings)
        self.documents.extend(list(zip(ids, texts)))
        
        print(f"✅ 已添加 {len(texts)} 个文档，当前总计: {self.index.ntotal}")
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """搜索最相似的文档"""
        
        query_vec = self._embed(query).reshape(1, -1)
        
        scores, indices = self.index.search(query_vec, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.documents):
                doc_id, doc_text = self.documents[idx]
                results.append({
                    "id": doc_id,
                    "content": doc_text,
                    "score": float(score),  # 内积值（已归一化 ≈ 余弦相似度）
                })
        
        return results
    
    def save(self, path: str):
        """保存索引到磁盘"""
        faiss.write_index(self.index, f"{path}.index")
        with open(f"{path}.docs", "wb") as f:
            pickle.dump(self.documents, f)
        print(f"✅ 索引已保存到 {path}")
    
    @classmethod
    def load(cls, path: str, embedding_model="nomic-embed-text"):
        """从磁盘加载索引"""
        instance = cls.__new__(cls)
        instance.index = faiss.read_index(f"{path}.index")
        with open(f"{path}.docs", "rb") as f:
            instance.documents = pickle.load(f)
        instance.embedding_model = embedding_model
        instance.dimension = instance.index.d
        print(f"✅ 已加载索引，共 {instance.index.ntotal} 个文档")
        return instance


# 使用示例
if __name__ == "__main__":
    # 创建索引
    index = OllamaFAISSIndex(embedding_model="nomic-embed-text", 
                            dimension=768, index_type="flat")
    
    # 添加文档
    docs = [
        "Python 是一种高级编程语言",
        "Java 是面向对象的编程语言",
        "JavaScript 用于 Web 前端开发",
        "Go 语言以并发性能著称",
        "Rust 注重内存安全",
        "C++ 是系统级编程语言",
    ]
    index.add(docs)
    
    # 搜索
    results = index.search("哪种语言适合写后端服务？", k=3)
    
    for r in results:
        print(f"[{r['score']:.3f}] {r['content']}")
```

## 其他向量数据库速查

除了 ChromaDB 和 FAISS，以下向量数据库也可以与 Ollama 配合使用：

| 数据库 | 特点 | 适用场景 | 安装方式 |
|--------|------|---------|---------|
| **LanceDB** | 无服务器嵌入式，支持 SQL | 边缘 / 本地应用 | `pip install lancedb` |
| **Qdrant** | Rust 编写，高性能，支持过滤 | 生产环境大规模部署 | Docker 或 `pip install qdrant-client` |
| **Milvus** | 分布式，云原生 | 企业级海量数据 | Docker Compose / K8s |
| **Weaviate** | 多模态原生支持 | 图像+文本混合检索 | Docker |

### LanceDB 快速示例

```python
import lancedb
import requests
import pyarrow as pa

db = lancedb.connect("./lance_data")
table = db.create_table("docs", schema=[
    pa.field("id", pa.string()),
    pa.field("text", pa.string()),
    pa.field("vector", pa.list_(pa.float32(), 768)),
], exist_ok=True)

def embed(text):
    resp = requests.post("http://localhost:11434/api/embeddings",
                        json={"model": "nomic-embed-text", "prompt": text})
    return resp.json()["embedding"]

table.add([{"id": "1", "text": "Hello world", "vector": embed("Hello world")}])
results = table.search().where("text LIKE '%world%'").limit(5).to_list()
```

## 持久化策略

无论你选择哪种向量数据库，都需要考虑数据持久化：

```
持久化策略对比:

┌──────────┬─────────────┬──────────┬──────────┬──────────────┐
│ 策略      │ 写入速度     │ 读速度    │ 存储占用  │ 适用场景      │
├──────────┼─────────────┼──────────┼──────────┼──────────────┤
│ 内存      │ 最快         │ 最快      │ 重启丢失  │ 缓存/临时     │
│ 文件持久化 │ 中等         │ 快        │ 中等      │ 本地开发      │
│ 远程服务   │ 慢(网络IO)  │ 中等      │ 可扩展    │ 生产环境      │
│ 混合       │ 内存+异步落盘│ 快        │ 大       │ 高吞吐场景    │
└──────────┴─────────────┴──────────┴──────────┴──────────────┘
```

## 本章小结

这一节我们学习了如何将 Ollama 的 Embedding 能力与各种向量数据库集成：

1. **暴力搜索只适合小规模数据**（< 1万文档），大规模需要 ANN 向量数据库
2. **ChromaDB 是最佳入门选择**——纯 Python、零配置、持久化支持
3. **FAISS 是高性能 CPU 方案**——Meta 出品，支持 Flat/IVF/HNSW 多种索引类型
4. **核心流程**：文档 → 分块 → Ollama Embedding → 向量库存储 → 查询向量化 → 相似度搜索 → 返回 Top-K 结果
5. **LanceDB/Qdrant/Milvus** 提供了不同场景下的替代方案
6. **持久化策略**决定了你的数据是否能在重启后保留

下一节我们将把这些组件串联起来，构建一个完整的端到端 RAG 系统。
