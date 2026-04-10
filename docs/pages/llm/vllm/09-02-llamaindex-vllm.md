# LlamaIndex + vLLM

> **白板时间**：如果说 LangChain 是"瑞士军刀"——什么都能做但需要自己组装，那 LlamaIndex 就是"手术刀"——专为 RAG（检索增强生成）场景设计，开箱即用。LlamaIndex 的核心优势在于：更精细的文档处理、更灵活的索引策略、更强大的查询引擎。当它与 vLLM 结合时，你可以在本地构建一个媲美企业级产品的 RAG 系统。

## 一、初始化配置

### 1.1 LLM 连接

```python
from llama_index.llms.vllm import Vllm

llm = Vllm(
    model="Qwen/Qwen2.5-7B-Instruct",
    tokenizer="Qwen/Qwen2.5-7B-Instruct",
    tokenizer_mode="auto",
    context_window=32768,
    generate_kwargs={
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 512,
    },
    verbose=False,
)

response = llm.complete("解释什么是 PagedAttention")
print(response)
```

### 1.2 Embedding 模型连接

```python
from llama_index.embeddings.vllm import VllmEmbedding

embed_model = VllmEmbedding(
    model="BAAI/bge-m3",
    embed_batch_size=64,
)

embeddings = embed_model.get_text_embedding("人工智能正在改变世界")
print(f"向量维度: {len(embeddings)}")
print(f"范数: {sum(x**2 for x in embeddings)**0.5:.4f}")
```

### 1.3 同时使用 LLM + Embedding

```python
from llama_index.core import Settings
from llama_index.llms.vllm import Vllm
from llama_index.embeddings.vllm import VllmEmbedding

Settings.llm = Vllm(
    model="Qwen/Qwen2.5-7B-Instruct",
    context_window=16384,
    generate_kwargs={"temperature": 0.3, "max_tokens": 256},
)

Settings.embed_model = VllmEmbedding(
    model="BAAI/bge-m3",
)

# 之后所有 LlamaIndex 操作自动使用这两个模型
```

## 二、索引构建与查询

### 2.1 基础 VectorStoreIndex

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
import os


def build_and_query():
    """构建索引并查询"""
    
    # ===== 加载文档 =====
    documents = SimpleDirectoryReader(
        input_dir="./data/docs",
        required_exts=[".md", ".txt", ".pdf"],
        recursive=True,
    ).load_data()
    
    print(f"[加载] {len(documents)} 个文档")
    
    # ===== 构建索引 =====
    index = VectorStoreIndex.from_documents(
        documents,
        show_progress=True,
    )
    
    index.storage_context.persist(persist_dir="./index_storage")
    
    print("[索引] 构建完成并持久化")
    
    # ===== 查询引擎 =====
    query_engine = index.as_query_engine(
        similarity_top_k=3,
        response_mode="compact",     # compact / refine / tree_summarize
        streaming=True,              # 流式输出
    )
    
    # ===== 执行查询 =====
    queries = [
        "PagedAttention 如何解决 KV Cache 内存浪费问题？",
        "Continuous Batching 和 Static Batching 的区别？",
        "vLLM 支持哪些量化格式？",
    ]
    
    for q in queries:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        print(f"{'='*60}")
        
        response = query_engine.query(q)
        
        print(f"A: {response.response}")
        print(f"\n参考来源:")
        for node in response.source_nodes:
            preview = node.get_content()[:100].replace('\n', ' ')
            score = node.score if hasattr(node, 'score') else 'N/A'
            print(f"  [{score}] {preview}...")


build_and_query()
```

### 2.2 高级查询模式

```python
def advanced_query_modes():
    """不同查询模式的对比"""
    
    modes_info = """
    ════════════════════════════════════════
          LlamaIndex 查询模式对比
    ════════════════════════════════════════
    
    ┌──────────────┬──────────┬──────────┬──────────────┐
    │ 模式          │ 速度      │ 质量      │ 适用场景       │
    ├──────────────┼──────────┼──────────┼──────────────┤
    │ "default"     │ 快        │ 中等      │ 简单问答       │
    │ "compact"     │ 快        │ 良好      │ ⭐ 推荐       │
    │ "refine"      │ 较慢      │ 很好      │ 需要详细回答   │
    │ "tree_...     │ 最慢      │ 最好      │ 复杂分析任务   │
    └──────────────┴──────────┴──────────┴──────────────┘
    
    使用方式:
      query_engine = index.as_query_engine(response_mode="refine")
    """
    print(modes_info)


def refine_mode_demo():
    """Refine 模式演示（两阶段：先检索后精炼）"""
    
    from llama_index.core import VectorStoreIndex, Document
    
    docs = [
        Document(text="PagedAttention 是 vLLM 的核心技术。它借鉴了操作系统的虚拟内存分页机制。"),
        Document(text="KV Cache 存储了每个 token 的 Key 和 Value 向量。传统方案按最大长度预分配导致大量浪费。"),
        Document(text="PagedAttention 将 KV Cache 划分为固定大小的 Block（通常16个token）。Block 可以按需分配和释放。"),
        Document(text="通过 PagedAttention，显存利用率从传统的 50-60% 提升到 90% 以上。"),
        Document(text="Continuous Batching 是 vLLM 的另一项创新。它允许在推理过程中动态加入新请求。"),
    ]
    
    index = VectorStoreIndex.from_documents(docs)
    query_engine = index.as_query_engine(
        response_mode="refine",
        similarity_top_k=3,
        streaming=True,
    )
    
    response = query_engine.query("PagedAttention 的核心优势是什么？")
    
    print(f"[Refine Mode]")
    print(f"\n初步回答:\n{response.response}")
    print(f"\n参考了 {len(response.source_nodes)} 个文档片段")


refine_mode_demo()
```

## 三、自定义 RAG Pipeline

### 3.1 完整可控的 Pipeline

```python
from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader,
    PromptTemplate,
)
from llama_index.llms.vllm import Vllm
from llama_index.embeddings.vllm import VllmEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.retrievers import VectorIndexRetriever
from llama_index.response_synthesizers import ResponseMode, get_response_synthesizer
from llama_index.core import QueryBundle


class CustomRAGPipeline:
    """完全自定义的 RAG Pipeline"""
    
    def __init__(self, vllm_base_url: str = "http://localhost:8000"):
        self.llm = Vllm(
            model="Qwen/Qwen2.5-7B-Instruct",
            context_window=8192,
            generate_kwargs={"temperature": 0.3, "max_tokens": 256},
        )
        self.embed_model = VllmEmbedding(model="BAAI/bge-m3")
        self.index = None
        self.splitter = SentenceSplitter(chunk_size=384, chunk_overlap=64)
    
    def ingest(self, directory: str):
        """加载文档并构建索引"""
        
        documents = SimpleDirectoryReader(
            input_dir=directory,
            required_exts=[".md", ".txt"],
        ).load_data()
        
        nodes = self.splitter.get_nodes_from_documents(documents)
        
        self.index = VectorStoreIndex(nodes=nodes)
        
        return len(nodes)
    
    def retrieve(self, query: str, top_k: int = 3):
        """检索相关文档片段"""
        
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=top_k,
        )
        
        retrieved_nodes = retriever.retrieve(QueryBundle(query_str=query))
        
        return retrieved_nodes
    
    def synthesize(self, query: str, nodes):
        """基于检索结果生成回答"""
        
        context_text = "\n\n".join([
            f"[{i+1}] {node.get_content()}"
            for i, node in enumerate(nodes)
        ])
        
        template = """
请基于以下参考信息回答用户的问题。
如果参考信息不足以回答，请明确说明。

参考信息:
{context}

用户问题: {query}
"""
        
        prompt = PromptTemplate(template).format(
            context=context_text,
            query=query,
        )
        
        response = self.llm.complete(prompt)
        return response.text
    
    def query(self, question: str) -> dict:
        """完整查询流程"""
        
        nodes = self.retrieve(question, top_k=3)
        answer = self.synthesize(question, nodes)
        
        return {
            "question": question,
            "answer": answer,
            "sources": [n.get_content()[:80] for n in nodes],
            "num_sources": len(nodes),
        }


# 使用示例
pipeline = CustomRAGPipeline(vllm_base_url="http://localhost:8000")

count = pipeline.ingest("./docs/vllm_guide")
print(f"[索引] {count} 个节点已创建")

result = pipeline.query("如何优化 vLLM 的 TTFT？")
print(f"\nQ: {result['question']}")
print(f"A: {result['answer'][:200]}...")
print(f"\n引用了 {result['num_sources']} 个来源")
```

## 四、多模态支持

### 4.1 VLM (视觉语言模型) + vLLM

```python
from llama_index.multi_modal_llms.vllm import VllmMultiModal
from llama_index.core import Document, ImageDocument

mm_llm = VllmMultiModal(
    model="Qwen/Qwen2-VL-7B-Instruct",
    max_new_tokens=256,
)

image_doc = ImageDocument(
    image_path="./images/chart.png",
    text="请描述这张图片的内容。",
)

response = mm_llm.complete([image_doc])
print(response)
```

---

## 五、LangChain vs LlamaIndex 选型

| 维度 | LangChain | LlamaIndex |
|------|-----------|------------|
| **定位** | 通用 LLM 编排框架 | **专注 RAG** |
| **优势** | Agent/Tool/Chain 生态丰富 | 文档处理/索引/查询深度优化 |
| **学习曲线** | 中等 | 较低（RAG 场景） |
| **vLLM 集成** | ChatOpenAI 兼容 | 原生 Vllm/VllmEmbedding 类 |
| **适用场景** | Agent 系统、复杂工作流 | **知识库问答、文档搜索** |
| **社区规模** | 更大 | RAG 领域更专业 |

```python
def framework_decision():
    """框架选型决策"""
    
    decision = """
    你的主要需求是什么？
    │
    ├─ 构建 RAG / 知识库问答系统
    │   └─→ 🎯 **LlamaIndex** (原生 RAG 支持)
    │
    ├─ 构建 Agent / 工具调用系统
    │   └─→ 🎯 **LangChain** (Agent 生态成熟)
    │
    ├─ 需要 Agent + RAG 组合
    │   └─→ LangChain + LlamaIndex (混合使用)
    │       LangChain 做 Agent 和编排
    │       LlamaIndex 做检索和索引
    │
    ├─ 简单对话 / 单轮生成
    │   └─→ 直接用 OpenAI SDK 或两者皆可
    │
    └─ 多模态 (图像理解)
        └─→ LlamaIndex MultiModal (原生支持更好)
    """
    print(decision)

framework_decision()
```

---

## 六、总结

本节完成了 LlamaIndex 与 vLLM 的集成：

| 能力 | 代码 | 说明 |
|------|------|------|
| **LLM 连接** | `Vllm()` from `llama_index.llms.vllm` | 原生类，细粒度控制 |
| **Embedding** | `VllmEmbedding()` from `llama_index.embeddings.vllm` | 同一服务提供 LLM+Embedding |
| **VectorStoreIndex** | `VectorStoreIndex.from_documents()` | 一行构建向量索引 |
| **查询引擎** | `index.as_query_engine(response_mode=...)` | default/compact/refine/tree |
| **自定义 Pipeline** | Retriever → Synthesizer 两阶段分离 | 完全控制每个环节 |
| **多模态** | `VllmMultiModal` | 图文混合输入 |

**核心要点回顾**：

1. **LlamaIndex 是 RAG 场景的最优选择**——从文档到答案的全流程都有深度优化
2. **Settings 全局配置**让所有组件自动使用 vLLM 后端——一次配置全局生效
3. **Refine 模式是质量与速度的最佳平衡**——两阶段检索保证回答准确性
4. **与 LangChain 不冲突**——可以同时使用两者各取所长
5. **自定义 Pipeline 让你有 100% 控制权**——适合生产级定制需求

下一节我们将学习 **其他框架集成与生态工具链**。
