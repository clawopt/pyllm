---
title: 统一数据接入：LlamaIndex 的 Connector 体系
description: LlamaIndex Connector 架构设计、Reader/Loader 抽象、数据接入模式、与 LangChain Loader 的对比
---
# 统一数据接入：LlamaIndex 的 Connector 体系

在上一章的第一个 RAG 示例中，我们用了 `SimpleDirectoryReader` 从本地文件目录加载数据。你可能觉得这很简单——不就是读个文件嘛，有什么好讲的？但真实世界中的数据远比"几个 Markdown 文件"复杂得多：你的知识库可能散落在 PostgreSQL 数据库里、存在 AWS S3 的对象存储中、藏在 Notion 的团队空间里、或者需要通过 GitHub API 去拉取仓库文档。

如果每种数据源都要你手写解析和加载逻辑，那光是数据接入就能耗掉项目一半的开发时间。**LlamaIndex 的 Connector（连接器）体系正是为了解决这个问题而设计的**——它提供了一套统一的数据接入抽象，让你用几乎相同的代码从数百种不同的数据源中获取数据。

## 为什么需要统一的 Connector 体系？

让我们先看一个没有 Connector 体系的"原始社会"是什么样子的。假设你需要从三种不同来源加载数据：

```python
# 场景一：从本地 PDF 文件加载（用 PyMuPDF）
import fitz  # PyMuPDF
def load_from_pdf(filepath):
    doc = fitz.open(filepath)
    text = ""
    for page in doc:
        text += page.get_text()
    return Document(text=text, metadata={"source": filepath})

# 场景二：从数据库加载（用 psycopg2）
import psycopg2
def load_from_postgres(query):
    conn = psycopg2.connect("dbname=kb user=postgres")
    cursor = conn.cursor()
    cursor.execute(query)
    rows = cursor.fetchall()
    # 手动把每一行转成文本...
    return [Document(text=str(row)) for row in rows]

# 场景三：从网页加载（用 requests + BeautifulSoup）
import requests
from bs4 import BeautifulSoup
def load_from_web(url):
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, "html.parser")
    # 需要手动去掉导航栏、广告、脚本等噪音...
    text = soup.get_text(separator="\n")
    return Document(text=text, metadata={"url": url})
```

每个数据源都有自己完全不同的 API、错误处理方式、数据格式。更糟糕的是，当数据源升级或更换时（比如公司从 MySQL 迁移到 PostgreSQL），你得重写所有相关的加载逻辑。

LlamaIndex 的做法是定义一个**统一的接口契约**——无论数据来自哪里，最终都返回相同格式的 `Document` 对象列表：

```python
# LlamaIndex 的统一接口 — 所有 Reader 都返回同样的格式
documents = reader.load_data()  # 返回 List[Document]
```

这个简单的 `load_data()` 接口背后，隐藏着对数百种数据源的适配逻辑。作为使用者，你只需要知道两件事：
1. **选哪个 Reader**（根据数据源类型）
2. **传什么参数**（比如文件路径、数据库连接字符串、API URL 等）

剩下的全部由框架处理。

## Connector 的架构层次

LlamaIndex 的 Connector 体系分为三个层次：

```
┌─────────────────────────────────────────────┐
│           LlamaHub (社区生态)                  │
│   第三方开发者贡献的 Reader / Loader          │
│   (DatabaseReader, NotionReader, ...)        │
└──────────────────┬──────────────────────────┘
                   │ 继承 / 实现
                   ▼
┌─────────────────────────────────────────────┐
│       llama-index-readers-* (官方包)          │
│                                             │
│  ┌─────────────┐ ┌──────────────┐           │
│  │ File Readers │ │ DB Readers   │ ...      │
│  │ PDF/Word/... │ │ PG/Mongo/... │           │
│  └─────────────┘ └──────────────┘           │
└──────────────────┬──────────────────────────┘
                   │ 调用
                   ▼
┌─────────────────────────────────────────────┐
│         BaseReader (基类抽象)                │
│                                             │
│   def load_data() -> List[Document]          │
│   def lazy_load() -> Generator[Document]     │
└─────────────────────────────────────────────┘
```

**最底层是 `BaseReader` 抽象类**，定义了所有 Reader 必须实现的接口。核心方法只有两个：

```python
from abc import ABC, abstractmethod
from typing import List, Generator
from llama_index.core import Document

class BaseReader(ABC):
    @abstractmethod
    def load_data(self) -> List[Document]:
        """加载所有数据并返回 Document 列表"""
        pass

    def lazy_load(self) -> Generator[Document, None, None]:
        """惰性加载，逐个 yield Document（适合大数据量场景）"""
        for doc in self.load_data():
            yield doc
```

**中间层是官方维护的 `llama-index-readers-*` 系列包**，按数据源类型分组：
- `llama-index-readers-file` — 文件类型（PDF、Word、PPT、Excel、图片等）
- `llama-index-readers-database` — 数据库（PostgreSQL、MySQL、MongoDB、Redis 等）
- `llama-index-readers-web` — Web 数据源（网页爬取、RSS、Sitemap 等）
- `llama-index-readers-api` — API 服务（GitHub、Notion、Slack、Jira 等）
- `llama-index-readers-cloud` — 云服务（AWS S3、Google Drive、OneDrive 等）

**最顶层是 LlamaHub 社区生态**，任何开发者都可以编写自己的 Reader 并发布到 LlamaHub 上供其他人使用。这意味着即使官方没有覆盖的数据源，你也可以在社区中找到解决方案。

## SimpleDirectoryReader：最常用的入口

在实际开发中，**80% 的场景都可以用 `SimpleDirectoryReader` 解决**。它是 LlamaIndex 提供的一个"智能目录读取器"，能自动识别文件类型并调用相应的解析器：

```python
from llama_index.core import SimpleDirectoryReader

# 最简用法：读取目录下所有支持的文件
reader = SimpleDirectoryReader("./data")
documents = reader.load_data()

# 更精细的控制
reader = SimpleDirectoryReader(
    input_dir="./knowledge_base",
    required_exts=[".pdf", ".md", ".txt"],  # 只加载这些扩展名
    exclude=["*.tmp", "draft_*"],            # 排除匹配的文件
    recursive=True,                          # 递归子目录
    num_files_limit=100,                     # 最多加载 100 个文件
    file_metadata=lambda file_name: {        # 自定义元数据
        "category": "internal_docs",
        "loaded_at": "2025-01-15",
    },
)
documents = reader.load_data()
```

`SimpleDirectoryReader` 内部的文件类型识别流程如下：

```
文件 "report.pdf"
    │
    ▼
判断扩展名 → .pdf
    │
    ▼
查找已注册的解析器 → PDFReader (PyMuPDF 后端)
    │
    ▼
调用 PDFReader.load_data() → List[Document]
    │
    ▼
添加元数据 (file_name, file_path, creation_time...)
    │
    ▼
返回 Document 对象
```

它支持开箱即用的文件格式包括：`.txt`, `.md`, `.pdf`, `.csv`, `.docx`, `.pptx`, `.xlsx`, `.html`, `.htm`, `.ipynb`, `.epub`, `.jpg`, `.png` 等。对于不在此列表中的格式，你可以通过 `file_extractor` 参数注册自定义解析器：

```python
from llama_index.readers.file import (
    PDFReader,
    EPUBReader,
    MarkdownReader,
)

reader = SimpleDirectoryReader(
    input_dir="./data",
    file_extractor={
        ".pdf": PDFReader(backend="pymupdf"),     # PDF 用 PyMuPDF
        ".epub": EPUBReader(),                    # EPUB 用默认解析器
        ".md": MarkdownReader(),                 # Markdown 特殊处理
    },
)
```

这种可扩展的设计让 `SimpleDirectoryReader` 既能开箱即用，又能按需定制——它是 LlamaIndex 数据接入层最重要的"瑞士军刀"。

## Reader 与 Loader 的关系

如果你之前用过 LangChain，可能会注意到 LangChain 中叫 `DocumentLoader`，而 LlamaIndex 中叫 `Reader`。两者功能类似但有一些重要的设计差异：

| 维度 | LangChain DocumentLoader | LlamaIndex Reader |
|------|--------------------------|-------------------|
| 返回类型 | `List[Document]` | `List[Document]` |
| 元数据处理 | 较弱（通常只有 source） | 丰富（自动提取文件属性） |
| 懒加载 | `lazy_load()` 方法 | `lazy_load()` 方法 |
| 异步支持 | `alazy_load()` | `async_load_data()` |
| 错误恢复 | 通常自行处理 | 内置 `raise_on_error` 参数 |
| 组合使用 | 通过 Chain 组合 | 通过 Index 直接消费 |

一个关键的区别在于**错误处理策略**。LangChain 的 Loader 在遇到无法解析的文件时通常会抛出异常，导致整个加载过程中断。而 LlamaIndex 的 Reader 默认更加宽容：

```python
from llama_index.core import SimpleDirectoryReader

# 默认行为：跳过无法解析的文件，继续处理其他文件
reader = SimpleDirectoryReader("./data")
documents = reader.load_data()
# 即使有 1 个损坏的 PDF，其余 99 个文件仍会正常加载

# 严格模式：遇到错误立即抛出异常
reader = SimpleDirectoryReader(
    "./data",
    raise_on_error=True,  # 任何一个文件出错就停止
)
documents = reader.load_data()
```

这种"宽容默认值 + 可选严格模式"的设计在生产环境中非常实用——你不会因为一个坏文件而导致整个数据管道崩溃。

## 数据接入的性能考量

当数据量较大时，数据加载阶段可能成为性能瓶颈。以下是几个关键的优化策略：

**策略一：惰性加载（Lazy Loading）。**

如果你的知识库有数万份文档，一次性全部加载到内存中可能导致 OOM。使用惰性加载可以逐批处理：

```python
reader = SimpleDirectoryReader("./large_data")

for document in reader.lazy_load():
    index.insert(document)  # 逐个插入索引，内存占用恒定
```

注意这里用了 `index.insert()` 而不是 `VectorStoreIndex.from_documents()`——后者需要一次性接收所有文档，而前者支持增量插入。

**策略二：并行加载。**

对于 I/O 密集型的加载任务（如网络请求、大文件读取），可以利用异步 IO 加速：

```python
import asyncio
from llama_index.core import SimpleDirectoryReader

async def load_async():
    reader = SimpleDirectoryReader("./data")
    documents = await reader.aload_data()  # 异步版本
    return documents

documents = asyncio.run(load_async())
```

`aload_data()` 是 `load_data()` 的异步版本，内部使用 asyncio 并发执行多个文件的加载操作，在网络延迟较高或文件数量较多时能显著提升速度。

**策略三：选择性加载。**

不需要每次都重新加载所有数据。通过 `num_files_limit` 和文件过滤参数，只加载新增或变更的文件：

```python
import os
from pathlib import Path

def get_recent_files(directory, days=7):
    """只获取最近 N 天内修改过的文件"""
    cutoff = time.time() - days * 86400
    return [
        str(f) for f in Path(directory).rglob("*")
        if f.is_file() and f.stat().st_mtime > cutoff
    ]

reader = SimpleDirectoryReader(
    input_dir="./data",
    required_exts=[".pdf", ".md"],
    files=get_recent_files("./data", days=3),  # 只加载最近 3 天的文件
)
new_documents = reader.load_data()
# 然后增量更新已有索引
for doc in new_documents:
    existing_index.insert(doc)
```

## LlamaHub：社区驱动的生态系统

LlamaIndex 官方提供的 Reader 覆盖了常见的数据源，但世界上有太多的数据格式和服务不可能全部被官方覆盖。这就是 **LlamaHub** 存在的意义——它是一个类似 PyPI 但专门面向 LLM 应用数据加载的包仓库。

在 LlamaHub 上，你可以找到：
- **企业工具集成**：Salesforce Reader、ServiceNow Reader、Zendesk Reader
- **专业格式解析**：CAD 文件 Reader、医学影像 DICOM Reader、乐谱 Score Reader
- **小众平台**：Discord Reader、Figma Reader、Miro Reader
- **实验性项目**：各种新发布的数据源适配器

使用 LlamaHub 上的第三方 Reader 和使用官方 Reader 一样简单：

```bash
pip install llama-index-readers-servicenow
```

```python
from llama_index.readers.servicenow import ServiceNowReader

reader = ServiceNowReader(
    instance_name="mycompany",
    username="admin",
    password="xxx",
)
documents = reader.load_data(
    query="state=resolved^short_description=login issue",
)
```

如果你需要的某个数据源还没有现成的 Reader，LlamaIndex 也让编写自定义 Reader 变得非常容易——我们会在 2.6 节详细讲解如何为私有数据源编写 Reader。

## 常见误区

**误区一："SimpleDirectoryReader 能处理所有文件"。** 它能处理的只是常见格式。对于高度定制化的格式（如旧版 Lotus Notes 数据库、加密的容器格式等），你需要编写自定义解析器或寻找社区方案。不要假设它能"魔法般地"理解任意文件。

**误区二："所有 Reader 都在 llama-index 包里"。** 不是的。大部分 Reader 都在独立的包里（如 `llama-index-readers-file`），需要单独安装。只有 `SimpleDirectoryReader` 因为太常用而被包含在了 core 包中。遇到 `ModuleNotFoundError` 时，第一反应应该是检查是否安装了对应的 readers 包。

**误区三："load_data() 总是很快的"。** 对于包含大量 PDF 或图片的目录，`load_data()` 可能需要几分钟甚至更长时间，因为 PDF 解析和图片 OCR 都是计算密集型操作。在大规模数据加载时，一定要考虑进度显示和超时控制。

**误区四:"Connector 只是读数据，没什么技术含量"。** 一个好的 Connector 需要处理很多细节：字符编码检测、损坏文件容错、元数据提取、增量同步、认证管理、速率限制等等。这些"看不见的工作"往往是决定数据质量的关键因素。LlamaIndex 的 Connector 体系的价值正在于把这些复杂性封装起来，让你专注于业务逻辑。
