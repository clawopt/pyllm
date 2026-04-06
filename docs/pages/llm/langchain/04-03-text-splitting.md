---
title: 文本分割：语义切分策略（按字符、递归、按语义）
description: RecursiveCharacterTextSplitter / MarkdownHeaderTextSplitter / 按语义切分 / 组合策略
---
# 文本分割：语义切分策略（按字符、递归、按语义）

上一节我们把各种格式的文档加载成了 Document 列表。但这些 Document 可能非常长——一篇完整的文章可能有几千甚至上万字。直接把整篇文章塞进模型的上下文窗口不仅会超出 token 限制，还会引入大量无关信息干扰模型判断。

这一节我们学习如何把长文档**合理地切分成较小的文本块（chunk）**。这是影响 RAG 检索质量的最重要因素之一——切得好，检索精准；切得差，要么信息不完整，要么噪音太多。

## 为什么需要分块

先看一个直观的例子。假设你的知识库中有这样一段文字：

```
Python 是一门高级编程语言，由 Guido van Rossum 于 1991 年创建。
它以简洁的语法和强大的标准库著称。
Java 是一门静态类型语言，由 James Gosling 在 1995 年开发。
Java 程序编译为字节码后在 JVM 上运行，具有"一次编写，到处运行"的特性。
Go 语言由 Google 在 2009 年发布，专注于高并发和网络编程。
```

如果用户问："Go 语言有什么特点？"，而这段文字被作为一个整体存储在向量库中：

- **不分块的问题**：检索时整个段落都被返回给模型，其中关于 Python 和 Java 的内容都是无关噪音
- **正确分块后**：每段语言独立成块，搜索 "Go 语言特点" 时只返回 Go 相关的那个块

## RecursiveCharacterTextSplitter：最通用的分块器

LangChain 推荐的分核器是 `RecursiveCharacterTextSplitter`——它的设计思路很聪明：**按照一组优先级从高到低的分隔符依次尝试分割**，优先在"自然的边界处"切开：

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,        # 每个块的最大字符数
    chunk_overlap=50,      # 块之间的重叠字符数
    separators=["\n\n", "\n", "。", " ", ""]
)
```

三个核心参数的含义：

### chunk_size：目标大小

每个文本块的目标大小，以字符数为单位。这个值通常设在 **300~1000** 之间：
- 太小（<200）：每个块的信息量不足，可能丢失完整语义
- 太大（>1500）：单次检索引入太多无关信息
- 经验起点：**500** 是大多数场景下的安全选择

### chunk_overlap：重叠区域

相邻两个块之间的重叠部分。设为非零值确保不会因为刚好在边界处切断而导致信息丢失：

```
Chunk 0: "...GIL使得同一时刻只有一个线程执行Python字节码，
         Python的多线程无法利用多核CPU的优势..."

Chunk 1: "Python的多线程无法利用多核CPU的优势。
         但对于I/O密集型任务（如网络请求、文件读写）..."
```

"Python的多线程无法利用多核CPU的优势" 这句话同时出现在两个块的末尾和开头——这就是 `chunk_overlap=50` 的作用。

经验值：**50~100** 对于中文内容比较合适。

### separators：分隔符优先级列表

`"\n\n"`（空行/段落边界）→ `"\n"`（换行）→ `"。"`（句号）→ `" "`（空格）→ `""`（按字符强制切割）

分块器会**优先用高优先级的分隔符来切**，只有当前面的分隔符都找不到足够多的切割点时，才降级使用下一个。这保证了文本尽可能在**自然的位置断开**。

### 实际运行示例

```python
text = """
第一章 Python 简介

Python 是一门高级编程语言，由 Guido van Rossum 于 1991 年首次发布。
它的设计哲学强调代码的可读性和简洁性。"优雅胜于丑陋"、"显式胜于隐式"
是 Python 社区的核心价值观。

Python 广泛应用于 Web 开发、数据分析、人工智能、自动化运维等领域。
它的语法简洁优雅，学习曲线平缓，非常适合初学者入门编程。

第二章 核心数据类型

Python 中最基本的数据类型包括整数(int)、浮点数(float)、字符串(str)和布尔(bool)。
每种类型都有丰富的内置方法可供调用。
"""

chunks = splitter.split_text(text)

for i, chunk in enumerate(chunks):
    print(f"\n=== Chunk {i} (长度: {len(chunk)}) ===")
    print(chunk)
```

输出：

```
=== Chunk 0 (长度: 487) ===
第一章 Python 简介

Python 是一门高级编程语言，由 Guido van Rossum 于 1991 年首次发布。
它的设计哲学强调代码的可读性和简洁性。"优雅胜于丑陋"、"显式胜于隐式"
是 Python 社区的核心价值观。

Python 广泛应用于 Web 开发、数据分析、人工智能、自动化运维等领域。

=== Chunk 1 (长度: 496) ===
应用于 Web 开发、数据分析、人工智能、自动化运维等领域。
它的语法简洁优雅，学习曲线平缓，非常适合初学者入门编程。

第二章 核心数据类型

Python 中最基本的数据类型包括整数(int)、浮点数(float)、字符串(str)和布尔(bool)。
每种类型都有丰富的内置方法可供调用。
```

注意 Chunk 0 和 Chunk 1 有重叠部分了吗？是的——"应用于 Web 开发..."那段同时出现在两块中。

## MarkdownHeaderTextSplitter：按标题结构分割

如果你的知识库主要是 Markdown 文档（如技术文档、Wiki 页面），按标题层级分割比纯字符分割更合理——它能让每个块对应一个完整的章节：

```python
from langchain_text_splitters import MarkdownHeaderTextSplitter

markdown_text = """
# 第一章 基础概念

## 1.1 变量与类型

变量是存储数据的容器...

## 1.2 运算符

Python 支持算术运算符、比较运算符...

# 第二章 控制流

## 2.1 条件语句

if/elif/else 用于条件判断...

## 2.2 循环语句

for 和 while 循环...
"""

md_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "h1"),
        ("##", "h2"),
    ]
)

md_chunks = md_splitter.split_text(markdown_text)

for chunk in md_chunks:
    print(f"[{chunk.metadata}]")
    print(chunk.page_content[:60] + "...\n")
```

输出：

```
[{'h1': '第一章 基础概念'}]
## 1.1 变量与类型
变量是存储数据的容器...

[{'h1': '第一章 基础概念', 'h2': '1.1 变量与类型'}]
变量是存储数据的容器...

[{'h1': '第二章 控制流'}]
## 2.1 条件语句
if/elif/else 用于条件判断...
```

每个块自动带上了所属的标题路径作为 metadata。这在 RAG 检索中非常有价值——当用户问"控制流相关的问题"时，系统可以优先从 `h1=第二章 控制流` 的那些块中搜索。

## 生产级做法：两种分块器组合使用

最佳实践通常是**先用 Markdown 结构做粗粒度分割，再用字符分块器把过长的章节进一步切小**：

```python
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter
)

def smart_split(documents, chunk_size=500, chunk_overlap=50):
    """智能分块：先按结构切，再按大小切"""
    
    # 第一步：按 Markdown 标题粗切
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2")]
    )
    coarse_chunks = []
    
    for doc in documents:
        if doc.page_content.count("#") > 0:
            # 包含 Markdown 标记 → 用标题分割
            coarse_chunks.extend(md_splitter.split_text(doc.page_content))
        else:
            # 纯文本 → 直接保留
            coarse_chunks.append(doc.page_content)
    
    # 第二步：对超长的块继续细切
    fine_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    final_chunks = fine_splitter.create_documents(coarse_chunks)
    
    return final_chunks


# 使用
raw_docs = TextLoader("data/guide.md").load()
chunks = smart_split(raw_docs, chunk_size=500, chunk_overlap=50)
print(f"最终生成 {len(chunks)} 个文本块")
```

## 不同内容类型的推荐参数

| 内容类型 | 推荐 chunk_size | 推荐 overlap | 推荐策略 |
|---------|----------------|-------------|---------|
| 技术文档 | 500-800 | 50-100 | 先按标题切，再按字符切 |
| 新闻文章 | 800-1200 | 100-200 | 按段落切（`\n\n` 优先） |
| 代码文件 | 1000-2000 | 100-200 | 按函数/类级别保持完整 |
| FAQ / Q&A | 按问答对分割 | 0 | 每个问答对天然独立 |
| 法律合同 | 200-400 | 50 | 条款要精确匹配 |
| API 文档 | 300-600 | 30-50 | 按 endpoint 分组 |

## 对 Document 列表的分块

实际流程中，你通常是直接对 Loader 产出的 Document 列表做分块：

```python
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = DirectoryLoader("data/knowledge-base/", glob="**/*.md", loader_cls=TextLoader)
raw_docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(raw_docs)

print(f"原始: {len(raw_docs)} 个文档")
print(f"分块后: {len(chunks)} 个块")

# 每个 chunk 都保留了原始 metadata
print(chunks[0].metadata)  # {'source': 'guide/ch1.md'}
```

关键方法区别：
- `split_text(text_string)` → 返回纯文本字符串列表
- `split_documents(doc_list)` → 返回 Document 对象列表（**保留 metadata**）

在 RAG 场景中始终应该用 `split_documents()`，因为后续检索结果展示来源时需要 metadata 信息。

## 常见误区

**误区一：chunk_size 越大越好**。很多人觉得"塞进去的信息越多越好"，但实际上过多无关信息会严重干扰模型的判断能力。实验表明，对于大多数问答任务，3-5 个高质量的小块比 1 个大块效果好得多。

**误区二：overlap 设为 0 可以节省 token**。确实能省一点 token，但代价是在边界处切断的关键信息可能永远找不到。比如一个专有名词恰好被切成两半，前后各一半——没有 overlap 就意味着这个词在任何一块里都不完整，导致检索失败。

**误区三：所有文档用同一种分块策略**。代码文件和技术文档的分块逻辑完全不同。好的 RAG 系统会根据文档类型自动选择或组合不同的分块策略。

到这里，我们已经有了干净、分好块的文档数据。下一节我们将把这些文本块转换成向量并存入向量数据库，让 RAG 系统具备检索能力。
