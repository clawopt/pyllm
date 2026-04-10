# 07-2 LlamaIndex + Ollama

## LlamaIndex vs LangChain：定位差异

在深入 LlamaIndex 之前，有必要先澄清它与 LangChain 的关系。很多开发者会困惑：**"它们不是做同样的事情吗？我该选哪一个？"**

简单来说：

| 维度 | LangChain | LlamaIndex |
|------|-----------|------------|
| **核心定位** | 通用 LLM 应用框架 | **数据连接 + RAG 专用框架** |
| **最强场景** | Agent / Tool Calling / Chain 编排 | **文档索引 / 检索 / 知识图谱** |
| **抽象层级** | 高层抽象（Chain/Agent） | 中低层抽象（Node/Index/Query Engine） |
| **数据处理** | 基础的 DocumentLoader | **极其丰富的 Connector（150+ 数据源）** |
| **索引策略** | 简单的 VectorStore | **多种索引类型（Vector/Keyword/Tree/Graph/Knowledge Graph）** |
| **适用人群** | 想快速构建通用 AI 应用的开发者 | **需要精细控制 RAG 行为的开发者** |

**结论**：如果你的核心需求是 RAG 和知识库问答，LlamaIndex 往往是更好的选择——它为数据索引和检索做了更深度的优化。

## 初始化与基础用法

```bash
pip install llama-index llama-index-embeddings-ollama llama-index-llms-ollama
```

```python
#!/usr/bin/env python3
"""LlamaIndex + Ollama 完整集成"""

from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader,
    Settings,
    StorageContext,
    load_index_from_storage
)
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding


def basic_usage():
    """基础用法: 加载文档 → 构建索引 → 查询"""
    
    # 配置全局设置
    Settings.llm = Ollama(model="qwen2.5:7b", request_timeout=120)
    Settings.embed_model = OllamaEmbedding(model="nomic-embed-text")
    
    # 加载文档
    documents = SimpleDirectoryReader("./my_docs").load_data()
    print(f"📄 加载了 {len(documents)} 个文档")
    
    # 构建向量索引（自动完成分块+Embedding+存储）
    index = VectorStoreIndex.from_documents(documents)
    
    # 创建查询引擎
    query_engine = index.as_query_engine(similarity_top_k=3)
    
    # 查询
    response = query_engine.query("Ollama 如何配置端口？")
    
    print("\n" + "=" * 60)
    print("回答:")
    print(response.response)
    
    print("\n引用来源:")
    for node in response.source_nodes:
        print(f"  [{node.score:.3f}] {node.text[:150]}...")


def chat_mode():
    """对话模式: 支持多轮上下文"""
    
    from llama_index.core import VectorStoreIndex
    
    Settings.llm = Ollama(model="qwen2.5:7b")
    Settings.embed_model = OllamaEmbedding(model="nomic-embed-text")
    
    documents = SimpleDirectoryReader("./docs").load_data()
    index = VectorStoreIndex.from_documents(documents)
    
    # 对话引擎（维护多轮上下文）
    chat_engine = index.as_chat_engine(
        similarity_top_k=3,
        chat_mode="condense_question",  # 压缩问题模式
        verbose=True
    )
    
    # 多轮对话
    response1 = chat_engine.chat("什么是 Docker？")
    print(f"\nQ1: 什么是 Docker？\nA1: {response1.response[:200]}...")
    
    response2 = chat_engine.chat("它和虚拟机有什么区别？")
    print(f"\nQ2: 它和虚拟机有什么区别？\nA2: {response2.response[:200]}...")
    # 注意: Q2 中的 "它" 被正确理解为 Docker，因为 chat_engine 保留了上下文


def streaming():
    """流式输出"""
    
    Settings.llm = Ollama(model="qwen2.5:7b")
    Settings.embed_model = OllamaEmbedding(model="nomic-embed-text")
    
    documents = SimpleDirectoryReader("./docs").load_data()
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine(streaming=True)
    
    streaming_response = query_engine.query("解释微服务架构")
    
    print("📖 流式输出:\n")
    for text in streaming_response.response_gen:
        print(text, end="", flush=True)
    print()


if __name__ == "__main__":
    basic_usage()
    chat_mode()
    streaming()
```

## 本地 RAG 完整示例（100 行代码）

```python
#!/usr/bin/env python3
"""
📚 LocalRAG-LI — 基于 LlamaIndex + Ollama 的本地知识库系统
100行代码实现完整的 RAG 功能
"""

from llama_index.core import (
    VectorStoreIndex, SimpleDirectoryReader,
    Settings, PromptTemplate
)
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.node_parser import SentenceSplitter
import os


class LocalRAGLI:
    """基于 LlamaIndex 的本地知识库"""
    
    def __init__(self,
                 docs_dir="./knowledge_base",
                 persist_dir="./li_index",
                 llm_model="qwen2.5:7b",
                 embed_model="nomic-embed-text"):
        
        self.docs_dir = docs_dir
        self.persist_dir = persist_dir
        
        # 配置模型
        Settings.llm = Ollama(model=llm_model, request_timeout=180)
        Settings.embed_model = OllamaEmbedding(model=embed_model)
        
        # 自定义分块器
        self.transformations = [SentenceSplitter(chunk_size=512, chunk_overlap=64)]
        
        self.index = None
    
    def build_index(self):
        """构建或加载索引"""
        
        if os.path.exists(self.persist_dir) and \
           len(os.listdir(self.persist_dir)) > 0:
            print(f"📂 从磁盘加载已有索引...")
            storage_context = StorageContext.from_defaults(persist_dir=self.persist_dir)
            self.index = load_index_from_storage(
                storage_context, 
                index_id="default"
            )
            print(f"✅ 索引已加载")
        else:
            if not os.path.exists(self.docs_dir):
                os.makedirs(self.docs_dir)
                print(f"⚠️ 文档目录为空: {self.docs_dir}")
                print(f"   请将文档放入该目录后重新构建索引")
                return
            
            print(f"📥 正在从 {self.docs_dir} 加载文档...")
            documents = SimpleDirectoryReader(
                self.docs_dir,
                required_exts=[".md", ".txt", ".pdf", ".docx"],
                recursive=True
            ).load_data()
            
            if not documents:
                print("❌ 未找到任何文档")
                return
            
            print(f"   找到 {len(documents)} 个文档片段")
            
            print("🔨 正在分块、嵌入、建索引...")
            self.index = VectorStoreIndex.from_documents(
                documents,
                transformations=self.transformations,
                show_progress=True
            )
            
            print(f"💾 持久化索引到 {self.persist_dir}")
            self.index.storage_context.persist(persist_dir=self.persist_dir)
        
        return self.index
    
    def query(self, question: str, top_k: int = 3):
        """查询"""
        
        if not self.index:
            self.build_index()
        
        query_engine = self.index.as_query_engine(
            similarity_top_k=top_k,
            response_mode="compact"  # 紧凑模式
        )
        
        response = query_engine.query(question)
        
        return {
            "answer": response.response,
            "sources": [
                {
                    "text": node.text[:200] + "...",
                    "score": round(node.score, 4),
                    "metadata": node.metadata
                }
                for node in response.source_nodes
            ]
        }


# 自定义 Prompt 模板
CUSTOM_QA_PROMPT = """\
上下文信息如下：
---------------------
{context_str}
---------------------

根据上述上下文信息回答用户问题。如果上下文中没有相关信息，
请明确说明不要编造。回答时请引用具体的来源。

用户问题: {query_str}
答案: """


if __name__ == "__main__":
    rag = LocalRAGLI(docs_dir="./my_knowledge")
    rag.build_index()
    
    while True:
        question = input("\n❓ 问题 (q退出): ").strip()
        if question.lower() in ["q", "quit"]:
            break
        
        result = rag.query(question)
        
        print(f"\n{'='*60}")
        print(result["answer"])
        
        if result["sources"]:
            print(f"\n📚 引用来源 ({len(result['sources'])}):")
            for src in result["sources"]:
                print(f"  [{src['score']:.3f}] {src['text']}")
```

## 多模态支持

LlamaIndex 对多模态的支持比 LangChain 更成熟：

```python
#!/usr/bin/env python3
"""LlamaIndex 多模态 + Ollama (LLaVA)"""

from llama_index.core import SimpleDirectoryReader
from llama_index.multi_modal_llms.ollama import OllamaMultiModal
from llama_index.core import VectorStoreIndex


def multimodal_rag():
    """多模态 RAG: 图像 + 文档混合检索"""
    
    # 使用多模态模型
    Settings.llm = OllamaMultiModal(model="minicpm-v")
    Settings.embed_model = OllamaEmbedding(model="nomic-embed-text")
    
    # 加载混合内容（文本 + 图片）
    documents = SimpleDirectoryReader(
        "./multimodal_docs",
        required_exts=[".png", ".jpg", ".md"]
    ).load_data()
    
    # 构建索引
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine(similarity_top_k=3)
    
    # 可以用文字或图片提问
    response = query_engine.query("截图中显示的错误是什么？如何修复？")
    print(response.response)


if __name__ == "__main__":
    multimodal_rag()
```

## 本章小结

这一节介绍了 LlamaIndex 与 Ollama 的集成：

1. **LlamaIndex 定位为 RAG 专用框架**，在数据索引和检索方面比 LangChain 更专业
2. **三步标准流程**：`SimpleDirectoryReader` → `VectorStoreIndex.from_documents()` → `index.as_query_engine()`
3. **三种查询模式**：`query_engine`（单次）、`chat_engine`（多轮）、`streaming=True`（流式）
4. **索引持久化**通过 `storage_context.persist()` 实现，重启后无需重建
5. **自定义 Prompt** 通过 `response_mode` 和 `PromptTemplate` 控制
6. **多模态支持**通过 `OllamaMultiModal` 类实现图像理解能力

下一节我们将快速浏览其他主流框架对 Ollama 的集成方式。
