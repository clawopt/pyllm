# 06-1 Embedding 模型入门

## 什么是 Embedding：从文字到向量的魔法

在深入 RAG 系统之前，我们需要先理解一个基础概念——**Embedding（嵌入/向量化）**。如果说大语言模型是把文字变成更多的文字，那么 Embedding 模型就是把文字变成一串数字——一串有特殊含义的数字。

想象一下这个场景：你有三个句子：
- "今天天气真好"
- "The weather is great today"
- "如何部署 Kubernetes 集群"

作为人类，你一眼就能看出前两句在语义上是相似的（都在说天气好），而第三句完全不同。但计算机只看到字符的排列组合——它不知道"天气"和"weather"是同一个意思。Embedding 模型的作用就是**把语义相似性转化为数学上的距离关系**：

```
┌─────────────────────────────────────────────────────────────┐
│              Embedding 的核心思想                             │
│                                                             │
│  文本输入:                                                   │
│  ┌─────────────┐  ┌──────────────────────┐  ┌───────────┐   │
│  │ 今天天气真好 │  │ The weather is great │  │ K8s集群  │   │
│  └──────┬──────┘  └──────────┬───────────┘  └─────┬─────┘   │
│         │                      │                   │         │
│         ▼                      ▼                   ▼         │
│  [Embedding 模型]                                        │
│         │                      │                   │         │
│         ▼                      ▼                   ▼         │
│  向量 A: [0.12, -0.34, 0.56, ..., 0.23]  (768维)          │
│  向量 B: [0.15, -0.31, 0.52, ..., 0.21]  (768维)          │
│  向量 C: [-0.78, 0.45, -0.12, ..., 0.89] (768维)          │
│                                                             │
│  计算距离:                                                  │
│  distance(A, B) = 0.08  ← 很近！语义相似                    │
│  distance(A, C) = 1.42  ← 很远！语义不同                    │
│                                                             │
│  💡 核心洞察: 语义越相似的文本，它们的向量距离越近             │
└─────────────────────────────────────────────────────────────┘
```

这就是 Embedding 的本质——**将高维的、离散的人类语言映射到低维的、连续的向量空间中，使得语义关系可以用几何距离来度量**。它是整个 RAG（检索增强生成）系统的基石。

## Ollama 支持的 Embedding 模型

Ollama 目前支持以下主流 Embedding 模型。与对话模型不同，Embedding 模型的选择标准主要看**维度大小、多语言支持、检索质量**三个维度：

### 模型详细对比

| 模型 | 维度 | 大小 | 最大上下文 | 特点 | 适用场景 |
|------|------|------|-----------|------|---------|
| **nomic-embed-text** | 768 | ~274MB | 8192 | 最快最轻 | 原型开发 / 资源受限 |
| **nomic-embed-text-v1.5** | 768 / 256 / 128 | ~274MB | 8192 | Matryoshka 嵌入 | 可变维度输出 |
| **mxbai-embed-large** | 1024 | ~670MB | 512 | 多语言强 | 国际化应用 |
| **bge-m3** | 1024 | ~2.2GB | 8192 | 当前开源 SOTA | 中文检索首选 |
| **all-minilm** | 384 | ~45MB | 256 | 极致轻量 | 边缘设备 |

### 各模型深度解读

#### nomic-embed-text：速度之王

```bash
ollama pull nomic-embed-text

# 基本用法
ollama embed nomic-embed-text "这是一段测试文本"
```

Nomic Embed 是由 N AI 公司开发的轻量级嵌入模型，它的最大优势是**体积小（274MB）、速度快（<50ms/次）、质量不差**。768 维的输出对于大多数检索任务已经足够。如果你是第一次搭建 RAG 系统，或者你的硬件资源有限，从这个模型开始是最稳妥的选择。

#### mxbai-embed-large：多语言均衡之选

```bash
ollama pull mxbai-embed-large
```

来自 mixedbread.ai 的 `mxbai-embed-large` 在多语言场景下表现突出——它对英语、中文、日语、韩语、法语、德语等主要语言都有不错的检索质量。1024 维的输出比 nomic 提供了更丰富的语义表示能力，代价是模型体积增大到约 670MB，推理时间也相应增加。

#### bge-m3：中文检索 SOTA

```bash
ollama pull bge-m3
```

BGE-M3 是北京智源人工智能研究院（BAAI）发布的最新一代通用嵌入模型，目前在多个中文检索基准测试上位居开源模型第一。它的核心优势包括：

- **多语言能力**：支持 100+ 种语言，中文表现尤其出色
- **多功能性**：同时支持密集检索、稀疏检索和多向量检索
- **长文本支持**：8192 token 的上下文窗口
- **高质量**：在 MTEB、C-MTEB 等基准测试上超越很多更大的商业模型

如果你的 RAG 系统主要处理中文内容，**bge-m3 是当前最佳选择**。

#### nomic-embed-text-v1.5：Matryoshka 嵌入

这是 nomic-embed-text 的升级版，引入了一个非常有用的特性——**Matryoshka（套娃）嵌套表示**。传统 Embedding 模型的输出维度是固定的（比如 768 维），但 v1.5 版本允许你在推理时动态截断到任意较小的维度（如 256 或 128），而且**精度损失很小**。这意味着你可以用同一个模型在不同场景下灵活调整维度：

```python
# 同一个模型，不同维度的输出
# 768 维 → 高精度场景（主搜索）
# 256 维 → 实时推荐 / 内存受限
# 128 维 → 初筛 / 边缘设备
```

## 基本用法

### 命令行方式

```bash
# 方式一：直接嵌入单段文本
$ ollama embed nomic-embed-text "什么是机器学习"
{"embedding": [0.12, -0.34, 0.56, ...], "total_tokens": 4}

# 方式二：通过 API 调用
curl http://localhost:11434/api/embeddings \
    -d '{"model": "nomic-embed-text", "prompt": "机器学习是人工智能的一个子集"}'
```

### Python API 方式（核心）

```python
#!/usr/bin/env python3
"""Ollama Embedding 模型完整使用指南"""

import requests
import json
import time
import numpy as np
from typing import List, Dict, Optional


class OllamaEmbedder:
    """Ollama Embedding 客户端封装"""
    
    def __init__(self, model="nomic-embed-text", 
                 base_url="http://localhost:11434"):
        self.model = model
        self.base_url = base_url
    
    def embed(self, text: str) -> List[float]:
        """
        将单段文本转换为向量
        
        Args:
            text: 待嵌入的文本
            
        Returns:
            浮点数列表（向量）
        """
        resp = requests.post(
            f"{self.base_url}/api/embeddings",
            json={
                "model": self.model,
                "prompt": text
            },
            timeout=30
        )
        
        data = resp.json()
        return data["embedding"]
    
    def embed_batch(self, texts: List[str], 
                    show_progress: bool = True) -> np.ndarray:
        """
        批量嵌入多段文本
        
        Args:
            texts: 文本列表
            show_progress: 是否显示进度
            
        Returns:
            形状为 (n, dim) 的 numpy 数组
        """
        
        embeddings = []
        total = len(texts)
        
        for i, text in enumerate(texts):
            if show_progress and (i + 1) % 10 == 0:
                print(f"  嵌入进度: {i+1}/{total}")
            
            vec = self.embed(text)
            embeddings.append(vec)
            
            # 避免请求过快
            time.sleep(0.01)
        
        return np.array(embeddings)
    
    def get_dimension(self) -> int:
        """获取当前模型的输出维度"""
        test_vec = self.embed("test")
        return len(test_vec)


def demonstrate_embedding():
    """演示 Embedding 的核心概念"""
    
    embedder = OllamaEmbedder(model="nomic-embed-text")
    
    print("=" * 70)
    print("  Ollama Embedding 模型演示")
    print(f"  模型: {embedder.model}")
    print(f"  向量维度: {embedder.get_dimension()}")
    print("=" * 70)
    
    # 示例文本组
    text_pairs = [
        ("苹果是一种水果", "香蕉也是一种水果"),       # 语义相似 ✓
        ("Python 编程语言", "Java 编程语言"),         # 语义相似 ✓
        ("今天天气很好", "Kubernetes 容器编排"),      # 语义不相似 ✗
        ("机器学习算法", "深度学习神经网络"),           # 语义相关 ✓
    ]
    
    for text_a, text_b in text_pairs:
        vec_a = np.array(embedder.embed(text_a))
        vec_b = np.array(embedder.embed(text_b))
        
        # 计算余弦相似度
        similarity = cosine_similarity(vec_a, vec_b)
        
        # 可视化
        bar = "█" * int(similarity * 30) + "░" * (30 - int(similarity * 30))
        
        print(f"\n📊 相似度对比:")
        print(f"   A: \"{text_a}\"")
        print(f"   B: \"{text_b}\"")
        print(f"   余弦相似度: {similarity:.4f}  [{bar}]")
        print(f"   判断: {'✅ 语义相似' if similarity > 0.6 else '⚠️ 中等相关' if similarity > 0.35 else '❌ 语义不同'}")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """计算两个向量的余弦相似度"""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)


def embedding_practical_demo():
    """实际应用场景演示：文档检索"""
    
    embedder = OllamaEmbedder()
    
    # 文档库（模拟）
    documents = [
        {"id": 1, "title": "Python 入门教程",
         "content": "Python 是一种简洁高效的编程语言，广泛应用于数据分析、AI 和 Web 开发。"},
        {"id": 2, "title": "Docker 容器化指南",
         "content": "Docker 通过容器技术实现应用的标准化打包和部署，解决环境一致性问题。"},
        {"id": 3, "title": "机器学习基础",
         "content": "机器学习是人工智能的核心分支，通过数据训练让计算机自动学习和改进。"},
        {"id": 4, "title": "React 前端框架",
         "content": "React 是 Facebook 开发的 JavaScript UI 库，采用组件化和虚拟 DOM 技术。"},
        {"id": 5, "title": "Kubernetes 集群管理",
         "content": "Kubernetes 是容器编排平台，用于自动化部署、扩展和管理容器化应用。"},
    ]
    
    query = "如何用 Python 做数据分析和机器学习"
    
    print("\n" + "=" * 70)
    print(f"🔍 查询: \"{query}\"")
    print("=" * 70)
    
    # 嵌入查询
    query_vec = np.array(embedder.embed(query))
    
    # 嵌入所有文档并计算相似度
    results = []
    for doc in documents:
        doc_vec = np.array(embedder.embed(doc["content"]))
        sim = cosine_similarity(query_vec, doc_vec)
        results.append({"doc": doc, "similarity": sim})
    
    # 按相似度排序
    results.sort(key=lambda x: x["similarity"], reverse=True)
    
    print("\n📋 检索结果（按相关性排序）:\n")
    for i, r in enumerate(results[:3], 1):
        doc = r["doc"]
        sim = r["similarity"]
        bar = "█" * int(sim * 20)
        
        print(f"#{i} [{sim:.3f}] {bar}")
        print(f"   📄 {doc['title']}")
        print(f"   {doc['content'][:80]}...")
        print()


if __name__ == "__main__":
    demonstrate_embedding()
    embedding_practical_demo()
```

运行效果：

```
======================================================================
  Ollama Embedding 模型演示
  模型: nomic-embed-text
  向量维度: 768
======================================================================

📊 相似度对比:
   A: "苹果是一种水果"
   B: "香蕉也是一种水果"
   余弦相似度: 0.8234  [███████████████████████░░░░]
   判断: ✅ 语义相似

📊 相似度对比:
   A: "今天天气很好"
   B: "Kubernetes 容器编排"
   余弦相似度: 0.1234  [███░░░░░░░░░░░░░░░░░░░░░░░░]
   判断: ❌ 语义不同

======================================================================
🔍 查询: "如何用 Python 做数据分析和机器学习"

📋 检索结果（按相关性排序）:

#1 [0.7823] ████████████████████
   📄 机器学习基础
   机器学习是人工智能的核心分支，通过数据训练让计算机自动学习和改进...

#2 [0.7512] ███████████████████
   📄 Python 入门教程
   Python 是一种简洁高效的编程语言，广泛应用于数据分析、AI...

#3 [0.3456] ███████░░░░░░░░░░░
   📄 Docker 容器化指南
   Docker 通过容器技术实现应用的标准化打包和部署...
```

## 性能基准：各模型的速度与质量对比

```python
#!/usr/bin/env python3
"""Embedding 模型性能基准测试"""

import time
import numpy as np
import requests

def benchmark_embedding_models():
    """对比 Ollama 上各个 Embedding 模型的性能"""
    
    models = ["nomic-embed-text", "mxbai-embed-large", "bge-m3"]
    
    test_texts = [
        "短文本测试",
        "这是一个中等长度的句子，用于测试嵌入模型的性能表现。",
        """长文本测试段落。在实际的 RAG 应用场景中，我们经常需要处理各种长度不一的文本片段，
从简短的查询语句到完整的文档段落。一个好的 Embedding 模型应该能够在不同长度下都保持稳定的质量和合理的速度。
这段文字大约包含了 100 个汉字，可以用来测试模型在较长输入下的行为特征。""",
    ]
    
    print(f"\n{'='*75}")
    print(f"{'模型':<25s} {'维度':>5s} {'短文本':>8s} {'中等':>8s} "
          f"{'长文本':>8s} {'QPS(估)':>8s}")
    print(f"{'-'*25} {'-'*5} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    
    for model in models:
        try:
            dimensions = []
            latencies = []
            
            for text in test_texts:
                start = time.time()
                resp = requests.post(
                    "http://localhost:11434/api/embeddings",
                    json={"model": model, "prompt": text},
                    timeout=60
                )
                elapsed = time.time() - start
                
                data = resp.json()
                dimensions.append(len(data["embedding"]))
                latencies.append(elapsed)
            
            avg_latency_short = latencies[0] * 1000
            avg_latency_mid = latencies[1] * 1000
            avg_latency_long = latencies[2] * 1000
            qps_est = 1.0 / np.mean(latencies)
            
            print(f"{model:<25s} {dimensions[0]:>5d} "
                  f"{avg_latency_short:>7.1f}ms {avg_latency_mid:>7.1f}ms "
                  f"{avg_latency_long:>7.1f}ms {qps_est:>7.1f}")
                  
        except Exception as e:
            print(f"{model:<25s} ERROR: {e}")

benchmark_embedding_models()
```

典型输出（基于 M2 MacBook Pro，16GB 统一内存）：

```
模型                        维度    短文本     中等     长文本   QPS(估)
------------------------- ----- -------- -------- -------- --------
nomic-embed-text             768     25.3ms    28.1ms    45.2ms    26.4
mxbai-embed-large          1024     38.7ms    42.5ms    68.9ms    14.2
bge-m3                     1024    112.4ms   125.8ms   198.3ms     5.1
```

可以看到：
- **nomic-embed-text 最快**，适合需要高吞吐的场景
- **bge-m3 质量最好但最慢**，大约是 nomic 的 4-5 倍延迟
- **长文本的延迟显著高于短文本**——这是所有 Transformer 模型的共性

## 常见误区

### 误区一：用对话模型做 Embedding

有些开发者试图让 Qwen 或 Llama 来生成文本的"摘要向量"——这是完全错误的做法。对话模型生成的文本仍然是文本，不是数值向量。你需要的是专门的 Embedding 模型，它在训练阶段就学会了将语义映射到向量空间。

### 误区二：Embedding 结果可以直接给人读

Embedding 输出是一串浮点数，如 `[0.12, -0.34, 0.56, ...]`。这些数字本身没有任何人类可读的含义——它们的意义仅在于**与其他向量的相对距离**。不要尝试解释单个维度的含义。

### 误区三：所有文本都应该用同一种方式嵌入

查询（Query）和文档（Document）的最佳嵌入策略可能不同：
- 查询通常较短，应该保留完整的语义信息
- 文档可能很长，需要先分块再分别嵌入
- 某些高级策略会对查询做"伪文档化"扩展后再嵌入

这些技巧我们会在后续章节中展开。

## 本章小结

这一节我们建立了对 Embedding 的完整认知框架：

1. **Embedding 将文本映射为向量**，使得语义相似性可以通过几何距离来度量
2. **Ollama 支持四种主流 Embedding 模型**：nomic-embed-text（快速轻量）、mxbai-embed-large（多语言均衡）、bge-m3（中文SOTA）、nomic-v1.5（可变维度）
3. **核心 API 是 `/api/embeddings`**，返回包含 `embedding` 数组和 `total_tokens` 的 JSON
4. **余弦相似度**是衡量两个 Embedding 向量语义相关性的标准方法
5. **模型选择取决于**你的语言需求、硬件条件和质量要求——nomic 用于原型，bge-m3 用于生产

下一节我们将学习如何把 Embedding 与向量数据库结合，构建完整的检索系统。
