---
title: 文档加载器：从 PDF、网页、Notion 等 100+ 数据源读取
description: TextLoader / PyPDFLoader / WebBaseLoader / DirectoryLoader / NotionLoader 等主流加载器详解
---
# 文档加载器：从 PDF、网页、Notion 等 100+ 数据源读取

上一节我们了解了 RAG 的完整架构。现在开始动手实现第一步——**把各种格式的原始数据加载进系统**。LangChain 提供了大量的 **Document Loader（文档加载器）**，能够从几乎任何数据源中读取内容。

## 所有 Loader 的统一输出格式

不管数据来源是什么，所有 Loader 的输出都是统一的——一个 `Document` 对象的列表：

```python
from langchain_core.documents import Document

doc = Document(
    page_content="这是文档的实际文本内容",
    metadata={
        "source": "report.pdf",      # 来源文件
        "page": 3,                    # 页码
        "title": "Q3销售报告"         # 其他元信息
    }
)
```

每个 Document 有两个核心属性：
- **`page_content`**：字符串，存放实际的文本内容
- **`metadata`**：字典，存放来源信息（文件名、页码、URL、创建时间等）

metadata 在后续的检索和溯源中非常重要——当 RAG 系统基于某页的内容回答问题时，你可以告诉用户"答案来自报告第 X 页"。

## 加载文本文件

最简单的情况是加载纯文本文件（`.txt`、`.md` 等）：

```python
from langchain_community.document_loaders import TextLoader

loader = TextLoader("data/intro.md", encoding="utf-8")
docs = loader.load()

print(f"加载了 {len(docs)} 个文档")
print(f"内容预览: {docs[0].page_content[:100]}")
print(f"元数据: {docs[0].metadata}")
```

输出：

```
加载了 1 个文档
内容预览: # Python 入门指南

Python 是一门简洁而强大的编程语言...
元数据: {'source': 'data/intro.md'}
```

注意 `TextLoader` 会把整个文件内容作为一个 Document 返回。如果文件很大，这一个 Document 可能有几千甚至上万字——这就是为什么下一步需要分块的原因。

## 加载 PDF 文件

PDF 是企业知识库中最常见的格式之一。LangChain 使用 `PyPDFLoader` 来处理 PDF：

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("data/company-handbook.pdf")
pages = loader.load()

print(f"共 {len(pages)} 页")

for i, page in enumerate(pages[:3]):
    print(f"\n--- 第 {page.metadata['page']+1} 页 ---")
    print(page.page_content[:150] + "...")
```

输出：

```
共 42 页

--- 第 1 页 ---
# 公司员工手册 v2.3

欢迎加入我们的团队！本手册将帮助你快速了解...

--- 第 2 页 ---
## 第一章 公司概况

公司成立于 2015 年，专注于企业级 SaaS 解决方案...

--- 第 3 页 ---
## 第二章 薪酬与福利
```

关键区别：`PyPDFLoader` 按**页**拆分，每页是一个独立的 Document，`metadata` 中自动包含 `page`（页码）。这对后续溯源非常有用。

> **安装依赖**：`pip install pypdf`

### 其他 PDF 加载器选择

PyPDFLoader 不是唯一选项。根据 PDF 类型不同，可以选择不同的加载器：

| 加载器 | 特点 | 适用场景 |
|--------|------|---------|
| `PyPDFLoader` | 速度快，支持标准 PDF | 大多数常规 PDF |
| `PyMuPDFLoader` (fitz) | 渲染更精确，支持提取图片 | 复杂排版的 PDF |
| `PDFMinerLoader` | 解析能力强，容错性好 | 损坏或不规范的 PDF |
| `UnstructuredPDFLoader` | 支持表格和复杂结构 | 含表格/图表的 PDF |

```python
from langchain_community.document_loaders import PyMuPDFLoader

loader = PyMuPDFLoader("data/report.pdf", extract_images=True)
pages = loader.load()
# 可以同时提取文字和图片
```

## 加载网页内容

如果你的知识来源是公开网页或内部 Wiki：

```python
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://docs.python.org/3/tutorial/")
docs = loader.load()

print(f"网页标题: {docs[0].metadata.get('title', '未知')}")
print(f"内容长度: {len(docs[0].page_content)} 字符")
```

`WebBaseLoader` 底层会下载 HTML 并自动提取正文内容（去掉导航栏、广告、脚本等噪音），产出干净的文本。

## 批量加载目录下的文件

实际项目中你的知识库通常是一个文件夹里堆了几十个甚至上百个文件。`DirectoryLoader` 可以递归扫描目录，根据文件扩展名自动选择对应的加载器：

```python
from langchain_community.document_loaders import DirectoryLoader, TextLoader

loader = DirectoryLoader(
    path="data/knowledge-base/",
    glob="**/*.md",
    loader_cls=TextLoader,
    encoding="utf-8",
    show_progress=True,
    use_multithreading=True
)

docs = loader.load()
print(f"共加载 {len(docs)} 个文档")
```

参数说明：
- **`glob`**：文件匹配模式。`**/*.md` 表示递归查找所有子目录中的 Markdown 文件
- **`show_progress=True`**：显示加载进度条
- **`use_multithreading=True`**：多线程加速（对大量小文件效果明显）

## 加载 Notion 页面

Notion 是很多团队的知识管理工具。LangChain 提供了专门的 Notion 加载器：

```python
from langchain_community.document_loaders import NotionDirectoryLoader

loader = NotionDirectoryLoader("data/notion-export/")
docs = loader.load()

for doc in docs[:3]:
    print(f"[{doc.metadata.get('title', '?')}] {doc.page_content[:80]}...")
```

使用前需要先从 Notion 导出数据（Notion → Export → Markdown & CSV），然后将导出的文件夹路径传给 Loader。

## 加载 CSV 和 JSON 数据

结构化数据也可以作为 RAG 的知识来源：

```python
from langchain_community.document_loaders import CSVLoader
from langchain_community.document_loaders import JSONLoader

# CSV — 每行变成一个 Document
csv_loader = CSVLoader(file_path="data/sales.csv", source_column="product")
csv_docs = csv_loader.load()
# source_column 指定哪一列作为 metadata 中的 source 标识

# JSON — 通过 jq_schema 指定如何提取内容
json_loader = JSONLoader(
    file_path="data/users.json",
    jq_schema=".[] | {content: .bio, metadata: {name: .name, id: .id}}",
    text_content=False
)
json_docs = json_loader.load()
```

## 加载代码仓库

如果你想让 RAG 系统能搜索和理解代码库：

```python
from langchain_community.document_loaders import TextLoader
import os

def load_codebase(repo_path: str, extensions: list[str] = None):
    """加载代码仓库中指定类型的文件"""
    if extensions is None:
        extensions = [".py", ".js", ".ts", ".java", ".go", ".rs"]

    all_docs = []
    for root, dirs, files in os.walk(repo_path):
        # 排除常见的不需要索引的目录
        dirs[:] = [d for d in dirs if d not in {
            "__pycache__", "node_modules", ".git", "dist", "build"
        }]

        for f in files:
            if any(f.endswith(ext) for ext in extensions):
                filepath = os.path.join(root, f)
                try:
                    loader = TextLoader(filepath, encoding="utf-8")
                    docs = loader.load()
                    all_docs.extend(docs)
                except Exception as e:
                    print(f"跳过 {filepath}: {e}")

    return all_docs

code_docs = load_codebase("my-project/", extensions=[".py"])
print(f"加载了 {len(code_docs)} 个 Python 文件")
```

这个函数会递归遍历代码仓库，只加载指定扩展名的文件，并自动排除 `__pycache__`、`node_modules`、`.git` 等无关目录。

## 常见问题与解决

**问题一：PDF 加载出来是乱码**

这是最常见的问题之一。原因可能是：
- PDF 是**扫描版图片**（不是文字 PDF）→ 需要先用 OCR 工具提取文字
- PDF 使用了**特殊字体编码** → 尝试换一个 Loader（如 PyMuPDFLoader）
- PDF 有**密码保护** → 需要先解密

```python
# 方案一：尝试不同的加载器
from langchain_community.document_loaders import PyMuPDFLoader, PDFMinerLoader

loaders_to_try = [
    ("PyPDF", PyPDFLoader),
    ("PyMuPDF", PyMuPDFLoader),
    ("PDFMiner", PDFMinerLoader),
]

for name, LoaderClass in loaders_to_try:
    try:
        docs = LoaderClass("problem.pdf").load()
        if docs and len(docs[0].page_content.strip()) > 10:
            print(f"{name} 加载成功！")
            break
    except Exception as e:
        print(f"{name} 失败: {e}")
```

**问题二：网页加载了太多噪音内容**

默认情况下 `WebBaseLoader` 可能会抓取到导航栏、侧边栏、广告等内容。可以通过自定义 BS4 解析器来精确控制提取范围：

```python
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader(
    "https://example.com/article",
    bs_kwargs={
        "parse_only": {"class": ("article-content", "post-body")}
    }
)
# 只提取 class 为 article-content 或 post-body 的元素
```

**问题三：大文件导致内存不足**

如果一个文件特别大（如几百 MB 的日志文件），不要一次性全部加载。可以分块读取或使用懒加载模式：

```python
from langchain_community.document_loaders import TextLoader

# 懒加载模式 — 不一次性读入全部内容
loader = TextLoader("large_file.txt", encoding="utf-8")
# 在后续分块时才会真正读取内容
```

到这里，我们已经能把各种格式的数据加载成统一的 Document 列表了。下一节我们将学习如何把这些长文档合理地切分成适合检索的小块。
