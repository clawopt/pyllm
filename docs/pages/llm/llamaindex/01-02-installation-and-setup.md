---
title: 安装与环境配置
description: LlamaIndex 安装方式、环境变量配置、多 LLM 后端切换、常见安装问题排查
---
# 安装与环境配置

在上一节我们讨论了"为什么需要 LlamaIndex"，这一节就来实际动手把环境搭好。好消息是，LlamaIndex 的安装过程非常简单——它遵循了 Python 社区的主流实践，如果你之前用过 `pip` 安装过任何库，整个过程对你来说会非常熟悉。

## 基础安装

LlamaIndex 的核心包叫做 `llama-index-core`，这是最基础的依赖，只包含框架本身而不包含任何连接器或 LLM 集成：

```bash
pip install llama-index-core
```

但说实话，**单独安装 core 包在实际开发中几乎不会用到**——因为你至少还需要一个 LLM（比如 OpenAI）和一个向量存储后端。所以更常见的做法是直接安装"一站式"的集合包：

```bash
pip install llama-index
```

这个 `llama-index` 包是一个元包（meta-package），它会自动拉入以下核心依赖：
- `llama-index-core` — 框架核心
- `llama-index-embeddings-openai` — OpenAI 的文本嵌入模型
- `llama-index-llms-openai` — OpenAI 的对话模型
- `llama-index-vector-stores-chroma` — ChromaDB 向量存储（本地开发用）

安装完成后，可以用以下代码验证环境是否就绪：

```python
import llama_index.core

print(f"LlamaIndex 版本: {llama_index.core.__version__}")
```

如果能看到版本号输出，说明基础安装成功了。

## 配置 API Key

LlamaIndex 本身不提供 LLM 能力，它需要调用外部的大语言模型服务。最常用的是 OpenAI，所以你需要先配置 API Key。

**方式一：环境变量（推荐用于生产环境）**

```bash
export OPENAI_API_KEY="sk-your-key-here"
```

在 Python 中读取：

```python
import os
from llama_index.core import Settings

Settings.llm = "openai"  # 使用 OpenAI 作为默认 LLM
# 或者更精细的控制：
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

Settings.llm = OpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
```

**方式二：直接在代码中传入（适合快速实验）**

```python
from llama_index.llms.openai import OpenAI

llm = OpenAI(
    model="gpt-4o-mini",
    api_key="sk-your-key-here",  # 仅限本地测试，不要提交到代码仓库
    temperature=0.7,
    max_tokens=2048
)
```

这里有一个非常重要的设计概念需要理解：**Settings 对象是 LlamaIndex 的全局配置中心**。你可以在程序启动时一次性设置好 LLM、嵌入模型、分块策略等参数，之后所有组件都会自动使用这些全局设置，不需要每次都重复传参。这和 LangChain 中每个 Chain 都要手动传参的方式形成了鲜明对比。

```python
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter

Settings.llm = OpenAI(model="gpt-4o-mini")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.chunk_size = 512
Settings.chunk_overlap = 50
Settings.node_parser = SentenceSplitter(
    chunk_size=Settings.chunk_size,
    chunk_overlap=Settings.chunk_overlap
)
```

一旦设置了 `Settings`，后续创建 Index 或 Query Engine 时就不需要再重复指定这些参数了。当然，你也可以在特定场景下覆盖全局设置——局部参数永远优先于全局配置。

## 切换不同的 LLM 后端

OpenAI 不是唯一选择。LlamaIndex 支持几乎所有主流 LLM 提供商，而且切换方式非常统一：

```python
# 使用 Anthropic Claude
from llama_index.llms.anthropic import Anthropic
Settings.llm = Anthropic(model="claude-sonnet-4-20250514")

# 使用本地 Ollama
from llama_index.llms.ollama import Ollama
Settings.llm = Ollama(model="qwen2.5:7b", request_timeout=120.0)

# 使用 Google Gemini
from llama_index.llms.gemini import Gemini
Settings.llm = Gemini(model="gemini-2.0-flash")

# 使用 Azure OpenAI
from llama_index.llms.azure_openai import AzureOpenAI
Settings.llm = AzureOpenAI(
    engine="your-deployment-name",
    model="gpt-4o-mini",
    azure_endpoint="https://xxx.openai.azure.com/",
    api_key="...",
    api_version="2024-02-15-preview"
)

# 使用开源模型通过 HuggingFace 推理
from llama_index.llms.huggingface import HuggingFaceLLM
Settings.llm = HuggingFaceLLM(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    tokenizer_name="Qwen/Qwen2.5-7B-Instruct",
    context_window=32768,
    max_new_tokens=2048,
    device_map="auto"
)
```

你可以看到，无论使用哪个 LLM 后端，接口都是一致的——都继承自 `BaseLLM` 基类，提供 `complete()` 和 `chat()` 方法。这种抽象让你可以在不改动业务逻辑的情况下自由切换底层模型，这在做 A/B 测试或者成本优化时非常有用。

嵌入模型同样支持多种后端：

```python
# OpenAI 嵌入
from llama_index.embeddings.openai import OpenAIEmbedding
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# 本地嵌入模型（无需 API 费用）
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-large-zh-v1.5"
)

# Cohere 嵌入
from llama_index.embeddings.cohere import CohereEmbedding
Settings.embed_model = CohereEmbedding(
    cohere_api_key="...",
    model_name="embed-multilingual-v3.0"
)
```

## 可选依赖按需安装

LlamaIndex 采用了一种很聪明的包管理策略：**核心包保持轻量，所有额外功能都以独立包的形式提供**。这意味着你只需要安装自己用到的功能，不会因为一个 PDF 解析器而把整个 PyTorch 拉进来。

以下是常用的可选包及其用途：

| 包名 | 用途 | 安装命令 |
|------|------|---------|
| `llama-index-readers-file` | 文件解析（PDF/Word/PPT/Excel等） | `pip install llama-index-readers-file` |
| `llama-index-readers-database` | 数据库连接器 | `pip install llama-index-readers-database` |
| `llama-index-readers-web` | 网页爬取 | `pip install llama-index-readers-web` |
| `llama-index-vector-stores-qdrant` | Qdrant 向量数据库 | `pip install llama-index-vector-stores-qdrant` |
| `llama-index-vector-stores-pinecone` | Pinecone 向量数据库 | `pip install llama-index-vector-stores-pinecone` |
| `llama-index-postprocessor-cohere-rerank` | Cohere 重排序 | `pip install llama-index-postprocessor-cohere-rerank` |
| `llama-index-graph-stores-neo4j` | Neo4j 知识图谱存储 | `pip install llama-index-graph-stores-neo4j` |

如果你想一次性安装所有常用功能（适合学习阶段），可以执行：

```bash
pip install "llama-index[all]"
```

但这会安装大量依赖，**生产环境中强烈建议按需安装**，以避免不必要的依赖冲突和安全风险。

## 虚拟环境管理

在实际项目中，管理 Python 虚拟环境是必不可少的。推荐使用以下几种方式之一：

**方式一：venv（Python 内置，零依赖）**

```bash
python -m venv llamaindex-env
source llamaindex-env/bin/activate  # Linux/Mac
# llamaindex-env\Scripts\activate   # Windows

pip install llama-index
```

**方式二：conda（适合数据科学场景）**

```bash
conda create -n llamaindex python=3.11
conda activate llamaindex
pip install llama-index
```

**方式三：uv（新一代包管理器，速度快）**

```bash
uv venv llamaindex-env
source .venv/bin/activate
uv pip install llama-index
```

**关于 Python 版本的要求：** LlamaIndex 要求 **Python 3.8+**，但建议使用 **Python 3.10 或 3.11** 以获得最佳兼容性。Python 3.12 虽然也支持，但部分依赖（特别是某些 C 扩展）可能还没有完全适配。

## 常见安装问题与排查

**问题一：安装速度慢或超时。**

这是因为 PyPI 服务器在国外，国内访问可能不稳定。解决方案是使用镜像源：

```bash
pip install llama-index -i https://pypi.tuna.tsinghua.edu.cn/simple
```

或者在系统中永久配置：

```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

**问题二：`ModuleNotFoundError: No module named 'llama_index.xxx'`。**

这是最常见的错误之一。原因是 LlamaIndex 的模块命名有特定的规范：

```python
# 正确的导入方式
from llama_index.core import VectorStoreIndex  # 核心模块
from llama_index.readers.file import PDFReader   # 文件读取器
from llama_index.vector_stores.chroma import ChromaVectorStore  # 向量存储

# 错误的导入方式（新手常犯）
import llama_index  # 这个包不存在！
from llama_index import VectorStoreIndex  # 旧版本写法，已废弃
```

记住这个规律：**所有模块都在 `llama_index.*` 命名空间下**，而不是顶层 `llama_index` 包。具体来说：
- 核心功能 → `llama_index.core`
- 读取器 → `llama_index.readers.*`
- LLM 集成 → `llama_index.llms.*`
- 嵌入模型 → `llama_index.embeddings.*`
- 向量存储 → `llama_index.vector_stores.*`
- 后处理器 → `llama_index.postprocessor.*`

**问题三：PDF 读取报错或中文乱码。**

PDF 解析需要额外的系统依赖。如果你用的是 `PyMuPDF`（LlamaIndex 默认的 PDF 引擎），需要确保安装了完整版：

```bash
pip install llama-index-readers-file
# 如果还有问题，尝试手动安装 PyMuPDF
pip install PyMuPDF
```

对于中文 PDF，有时会遇到编码问题。这时可以考虑换用其他 PDF 引擎：

```python
from llama_index.readers.file import PDFReader
# 使用 pdfplumber 引擎（对中文表格支持更好）
reader = PDFReader(backend="pdfplumber")
# 或者使用 marker（基于 AI 的 PDF 解析，效果最好）
reader = PDFReader(backend="marker")
```

**问题四：内存不足（OOM）。**

这通常发生在加载大文件或处理大量文档时。解决方案包括：
1. 减小 `chunk_size`（从默认的 1024 降到 512 或 256）
2. 使用流式处理而非一次性加载全部文档
3. 对于嵌入操作，考虑使用 CPU 推理而非 GPU（GPU 反而可能占用更多显存）

**问题五：API Key 配置了但仍然报认证错误。**

检查以下几点：
1. 确认环境变量的名字完全正确（注意大小写：`OPENAI_API_KEY` 不是 `openai_api_key`）
2. 如果你用的是 `.env` 文件，确保用 `python-dotenv` 加载了它
3. 检查是否有 shell 配置文件中的旧值覆盖了你的设置：`echo $OPENAI_API_KEY`

## 完整的环境配置示例

下面是一个完整的、可用于生产项目的环境初始化脚本示例。把它放在项目根目录叫 `config.py`，然后在应用入口处导入即可：

```python
import os
from typing import Optional
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter


def init_settings(
    openai_api_key: Optional[str] = None,
    llm_model: str = "gpt-4o-mini",
    embed_model: str = "text-embedding-3-small",
    chunk_size: int = 512,
    chunk_overlap: int = 50,
):
    """初始化 LlamaIndex 全局配置"""
    api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("请设置 OPENAI_API_KEY 环境变量")

    Settings.llm = OpenAI(
        model=llm_model,
        api_key=api_key,
        temperature=0.1,  # RAG 场景建议低温度以保证事实准确性
        max_tokens=4096,
    )
    Settings.embed_model = OpenAIEmbedding(
        model=embed_model,
        api_key=api_key,
    )
    Settings.chunk_size = chunk_size
    Settings.chunk_overlap = chunk_overlap
    Settings.node_parser = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    print(f"✅ LLM: {llm_model}")
    print(f"✅ Embedding: {embed_model}")
    print(f"✅ Chunk: {chunk_size}/{chunk_overlap}")


if __name__ == "__main__":
    init_settings()
```

这样配置好后，下一节我们就可以用真正的代码来构建第一个 RAG 应用了。
