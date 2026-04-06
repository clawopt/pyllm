---
title: 加载代码仓库并构建代码知识库
description: 代码 RAG 的特殊性、仓库加载策略、AST 感知分块、代码嵌入模型选择、索引构建与检索优化
---
# 加载代码仓库并构建代码知识库

第十章我们用 RAG 构建了一个客服问答系统——知识来源是 Markdown 格式的产品文档。本章我们要把 RAG 的应用场景从"自然语言文档"切换到**代码**，构建一个能够理解、分析甚至执行代码的智能助手。

这是一个本质上不同的挑战。产品文档是给人读的自然语言，而代码是给机器执行的形式化语言——它有严格的语法结构、类型系统、依赖关系和语义约束。把代码当普通文本扔进 RAG，效果会很差。

## 代码 RAG 与文档 RAG 的本质区别

在动手之前，我们需要理解为什么"代码知识库"不能简单地套用第 4 章的文档 RAG 方案。

### 对比：同一套 RAG 流程在不同数据上的表现

假设我们有一个 Python 项目，其中包含一个用户认证模块：

```python
class AuthService:
    def __init__(self, db: Database, secret_key: str):
        self.db = db
        self.secret_key = secret_key

    def authenticate(self, username: str, password: str) -> Optional[User]:
        user = self.db.find_user(username)
        if not user or not self._verify_password(password, user.password_hash):
            return None
        token = self._generate_jwt(user.id, expires_in=3600)
        return User(id=user.id, token=token)

    def _verify_password(self, plain: str, hashed: str) -> bool:
        import bcrypt
        return bcrypt.checkpw(plain.encode(), hashed.encode())
```

如果用标准的 `RecursiveCharacterTextSplitter`（按 `\n\n` → `\n` → 句号切分），这段代码可能会被切成这样：

```
Chunk 0:
class AuthService:
    def __init__(self, db: Database, secret_key: str):
        self.db = db
        self.secret_key = secret_key

Chunk 1:
    def authenticate(self, username: str, password: str) -> Optional[User]:
        user = self.db.find_user(username)

Chunk 2:
        if not user or not self._verify_password(password, user.password_hash):
            return None
        token = self._generate_jwt(user.id, expires_in=3600)
        return User(id=user.id, token=token)
```

问题很明显：**`authenticate` 方法被切成两半了**。如果用户问"这个认证方法返回什么？"，检索可能只拿到 Chunk 1（只有前半段），看不到 `return User(...)` 这行关键代码。

这就是代码 RAG 的核心挑战：

| 维度 | 文档 RAG | 代码 RAG |
|------|---------|---------|
| **语义单元** | 段落/章节 | 函数/类/模块 |
| **切分依据** | 空行、句号 | 缩进层级、def/class 声明 |
| **上下文依赖** | 弱（段落相对独立） | 强（函数依赖类定义、import） |
| **嵌入模型** | 通用文本嵌入 | 需要理解代码语义 |
| **查询模式** | 自然语言提问 | "这个函数做了什么""哪里有 bug" |

## 加载代码仓库

LangChain 生态中加载代码有几种方式，从简单到复杂依次介绍。

### 方式一：TextLoader 加载单文件

最简单的方式——把代码文件当作纯文本加载：

```python
from langchain_community.document_loaders import TextLoader

loader = TextLoader("auth_service.py", encoding="utf-8")
doc = loader.load()
print(f"文件: {doc[0].metadata['source']}")
print(f"字符数: {len(doc[0].page_content)}")
```

这种方式的问题是没有代码结构信息——你只知道这是一坨文本，不知道哪部分是类定义、哪部分是方法实现。

### 方式二：DirectoryLoader 批量加载整个项目

```python
from langchain_community.document_loaders import DirectoryLoader

loader = DirectoryLoader(
    path="./my_project",
    glob="**/*.py",
    loader_cls=TextLoader,
    show_progress=True,
    exclude=["__pycache__", "*.pyc", ".git", "node_modules", "venv"],
)
docs = loader.load()
print(f"共加载 {len(docs)} 个 Python 文件")
```

实际项目中需要排除大量无关目录（`__pycache__`, `.git`, `venv`, `node_modules` 等）。但即使过滤后，一个中型项目也可能有几百个 `.py` 文件——全部加载进内存并分块，成本不低。

### 方式三：基于 Git 仓库的智能加载（推荐）

更实用的方式是利用 Git 元数据来筛选和加载代码。我们可以只加载特定分支、特定时间范围内修改的文件，或者排除测试文件和生成文件：

```python
import subprocess
import os
from pathlib import Path
from typing import List
from langchain_core.documents import Document

class CodeRepositoryLoader:
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path).resolve()

    def load_tracked_files(self, extensions: List[str] = None) -> List[Document]:
        extensions = extensions or [".py", ".ts", ".js", ".java"]
        result = subprocess.run(
            ["git", "ls-files"],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
        )
        documents = []
        for filepath in result.stdout.strip().split("\n"):
            if not filepath:
                continue
            ext = os.path.splitext(filepath)[1]
            if ext not in extensions:
                continue
            full_path = self.repo_path / filepath
            if not full_path.exists() or full_path.stat().st_size > 100_000:
                continue
            try:
                content = full_path.read_text(encoding="utf-8", errors="ignore")
                documents.append(Document(
                    page_content=content,
                    metadata={
                        "source": filepath,
                        "extension": ext,
                        "lines": content.count("\n") + 1,
                        "size_bytes": len(content.encode("utf-8")),
                    }
                ))
            except Exception:
                continue
        print(f"从 Git 仓库加载了 {len(documents)} 个源文件")
        return documents

    def load_recent_changes(self, days: int = 7) -> List[Document]:
        since = f"{days} days ago"
        result = subprocess.run(
            ["git", "diff", "--name-only", f"--since={since}", "HEAD"],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
        )
        changed_files = result.stdout.strip().split("\n") if result.stdout.strip() else []
        documents = []
        for filepath in changed_files:
            full_path = self.repo_path / filepath
            if full_path.exists() and full_path.suffix == ".py":
                try:
                    content = full_path.read_text(encoding="utf-8", errors="ignore")
                    documents.append(Document(
                        page_content=content,
                        metadata={"source": filepath, "recent_change": True},
                    ))
                except Exception:
                    continue
        print(f"最近 {days} 天内修改了 {len(documents)} 个文件")
        return documents


repo_loader = CodeRepositoryLoader("./my_project")
docs = repo_loader.load_tracked_files(extensions=[".py"])
# 输出: 从 Git 仓库加载了 47 个源文件
```

这种方式的几个优势：
- 只加载 Git 跟踪的文件，自动排除 `.gitignore` 中的内容
- 可以按扩展名过滤（只加载 Python，忽略配置文件）
- 可以只加载近期变更的文件（增量更新场景）
- 自动收集 metadata（文件路径、行数、大小）

## 代码感知的分块策略

这是代码 RAG 最关键的一步。好的分块策略应该保证**每个代码块是一个完整的语义单元**——通常是一个完整的函数或类。

### 基于 AST 的智能分块

Python 标准库提供了 `ast` 模块，可以解析代码的抽象语法树。我们利用它来实现按函数/类级别切分：

```python
import ast
import textwrap
from typing import List, Tuple

class CodeSplitter:
    def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_document(self, document: Document) -> List[Document]:
        source = document.page_content
        source_path = document.metadata.get("source", "unknown")

        try:
            tree = ast.parse(source)
        except SyntaxError:
            return self._fallback_split(document)

        chunks = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start_line = node.lineno - 1
                end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line + 10
                lines = source.split("\n")[start_line:end_line]
                code_text = "\n".join(lines)
                if len(code_text) < 20:
                    continue
                chunks.append(Document(
                    page_content=code_text,
                    metadata={
                        **document.metadata,
                        "node_type": type(node).__name__,
                        "name": node.name,
                        "start_line": start_line + 1,
                        "end_line": end_line,
                    }
                ))

        if not chunks:
            return self._fallback_split(document)

        chunks.sort(key=lambda x: x.metadata["start_line"])
        print(f"  {source_path}: 提取了 {len(chunks)} 个代码单元 "
              f"({sum(len(c.page_content) for c in chunks)} 字符)")
        return chunks

    def _fallback_split(self, document: Document) -> List[Document]:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\ndef ", "\n\nclass ", "\n\n", "\n", " ", ""],
        )
        return splitter.split_documents([document])


splitter = CodeSplitter()
all_chunks = []
for doc in docs:
    chunks = splitter.split_document(doc)
    all_chunks.extend(chunks)

print(f"\n总计: {len(all_chunks)} 个代码块")
```

输出示例：

```
从 Git 仓库加载了 47 个源文件
  auth_service.py: 提取了 4 个代码单元 (892 字符)
  user_model.py: 提取了 3 个代码单元 (456 字符)
  api_routes.py: 提取了 12 个代码单元 (3210 字符)
  ...

总计: 187 个代码块
```

看看提取出来的代码块长什么样：

```python
for chunk in all_chunks[:3]:
    print(f"\n--- {chunk.metadata['node_type']}: {chunk.metadata['name']} "
          f"(L{chunk.metadata['start_line']}-{chunk.metadata['end_line']}) ---")
    print(chunk.page_content[:300])
    print("...")
```

输出：

```
--- ClassDef: AuthService (L1-L25) ---
class AuthService:
    def __init__(self, db: Database, secret_key: str):
        self.db = db
        self.secret_key = secret_key

    def authenticate(self, username: str, password: str) -> Optional[User]:
...

--- FunctionDef: authenticate (L7-L18) ---
    def authenticate(self, username: str, password: str) -> Optional[User]:
        user = self.db.find_user(username)
        if not user or not self._verify_password(password, user.password_hash):
            return None
        token = self._generate_jwt(user.id, expires_in=3600)
        return User(id=user.id, token=token)

--- FunctionDef: _verify_password (L20-L23) ---
    def _verify_password(self, plain: str, hashed: str) -> bool:
        import bcrypt
        return bcrypt.checkpw(plain.encode(), hashed.encode())
```

可以看到：
- **类定义**被完整提取（包含所有方法）
- **每个函数**也被单独提取为独立块
- metadata 中记录了节点类型、名称、行号范围

这种"既保留完整类又拆出独立函数"的策略保证了检索时的灵活性：问"AuthService 类有哪些方法？"能匹配到 ClassDef 块；问"`authenticate` 方法怎么验证密码？"能匹配到 FunctionDef 块。

### 处理超大文件

有些文件特别大（比如一个 500 行的路由注册文件），按函数拆分后单个块仍然很长。这时需要二次分割：

```python
class SmartCodeSplitter(CodeSplitter):
    def split_document(self, document: Document) -> List[Document]:
        base_chunks = super().split_document(document)
        final_chunks = []

        for chunk in base_chunks:
            if len(chunk.page_content) <= self.chunk_size:
                final_chunks.append(chunk)
                continue

            sub_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n    def ", "\n    async def ", "\n\n", "\n", ""],
            )
            sub_chunks = sub_splitter.split_documents([chunk])
            for sc in sub_chunks:
                sc.metadata["is_sub_chunk"] = True
            final_chunks.extend(sub_chunks)

        return final_chunks
```

## 选择合适的代码嵌入模型

普通文本嵌入模型（如 `text-embedding-3-small`）对代码的理解能力有限。它们擅长捕捉自然语言的语义相似性，但对 `user.authenticate(username, password)` 和 `auth.verify_credentials(uid, pwd)` 这两行代码是否在做同一件事，判断力不如专门的代码嵌入模型。

### 主流代码嵌入模型对比

| 模型 | 来源 | 特点 | 推荐场景 |
|------|------|------|---------|
| **text-embedding-3-small** | OpenAI | 通用性好、生态兼容 | 代码量小、混合语言/代码 |
| **code-retrieval-model** | OpenAI Codex | 专为代码设计 | 纯代码检索 ✅ 推荐 |
| **Voyage-code** | Voyage AI | 代码+文档混合检索 | 有注释的代码库 |
| **BGE-large-zh-v1.5** | BAAI | 中文友好 | 中文注释为主的中文项目 |
| **jina-embeddings-v2-base-code** | Jina AI | 支持 8K 上下文 | 长函数/大文件 |

对于本项目，我们使用 OpenAI 的嵌入模型作为默认方案，同时展示如何切换到其他模型：

```python
from langchain_openai import OpenAIEmbeddings

def get_code_embeddings(model_name: str = None):
    model_name = model_name or os.getenv("CODE_EMBEDDING_MODEL", "text-embedding-3-small")
    return OpenAIEmbeddings(model=model_name, dimensions=1536)

embeddings = get_code_embeddings()
```

### 为代码块添加上下文增强

代码的一个特殊之处是：**同一个变量名在不同上下文中含义完全不同**。`id` 在用户模块中指用户 ID，在订单模块中指订单 ID。为了让嵌入模型更好地理解这一点，我们在存入向量数据库之前给每个代码块添加上下文描述：

```python
def enhance_code_chunk(chunk: Document) -> Document:
    node_type = chunk.metadata.get("node_type", "")
    name = chunk.metadata.get("name", "")
    source = chunk.metadata.get("source", "")

    context_prefix = ""
    if node_type == "ClassDef":
        context_prefix = f"# Class definition: {name} in {source}\n"
    elif node_type in ("FunctionDef", "AsyncFunctionDef"):
        context_prefix = f"# Function: {name}() in {source}\n"
    else:
        context_prefix = f"# Code from {source}\n"

    first_lines = chunk.page_content[:200].strip()
    enhanced_content = context_prefix + chunk.page_content

    return Document(
        page_content=enhanced_content,
        metadata=chunk.metadata,
    )

enhanced_chunks = [enhance_code_chunk(c) for c in all_chunks]
```

这样，存储到向量数据库中的内容变成了：

```
# Function: authenticate() in auth_service.py
    def authenticate(self, username: str, password: str) -> Optional[User]:
        user = self.db.find_user(username)
        ...
```

开头的注释行帮助嵌入模型理解"这段代码是什么、在哪里"，大幅提升检索精度。

## 构建代码向量索引

有了增强后的代码块，接下来就是标准的向量化流程：

```python
from langchain_chroma import Chroma

def build_code_index(chunks: list, persist_dir: str = "./chroma_code_db"):
    embeddings = get_code_embeddings()

    vectorstore = Chroma(
        collection_name="code_knowledge_base",
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )

    vectorstore.add_documents(chunks)

    stats = {
        "total_chunks": len(chunks),
        "classes": sum(1 for c in chunks if c.metadata.get("node_type") == "ClassDef"),
        "functions": sum(1 for c in chunks if c.metadata.get("node_type") in ("FunctionDef", "AsyncFunctionDef")),
        "total_chars": sum(len(c.page_content) for c in chunks),
    }

    print(f"\n代码索引构建完成:")
    print(f"  总代码块: {stats['total_chunks']}")
    print(f"  其中类定义: {stats['classes']}, 函数定义: {stats['functions']}")
    print(f"  总字符数: {stats['total_chars']}")

    return vectorstore

vectorstore = build_code_index(enhanced_chunks)
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 6},
)
```

### 测试代码检索效果

```python
query = "用户登录认证是怎么实现的？"
results = retriever.invoke(query)

for i, doc in enumerate(results):
    meta = doc.metadata
    print(f"\n[{i+1}] {meta.get('node_type', '?')}: {meta.get('name', '?')} "
          f"({meta.get('source', '')}:{meta.get('start_line', '?')})")
    preview = doc.page_content.replace("\n", " ")[:200]
    print(f"    {preview}...")
```

输出：

```
[1] ClassDef: AuthService (auth_service.py:1)
    # Class definition: AuthService in auth_service.py class AuthService: def __init__...
[2] FunctionDef: authenticate (auth_service.py:7)
    # Function: authenticate() in auth_service.py     def authenticate(self, username: str...
[3] FunctionDef: _verify_password (auth_service.py:20)
    # Function: _verify_password() in auth_service.py     def _verify_password(self...
[4] FunctionDef: login (api_routes.py:42)
    # Function: login() in api_routes.py     @router.post("/login") async def login(request:...
```

检索结果高度相关——排在前面的正是 `AuthService` 类和它的 `authenticate` 方法，以及 API 层的 `login` 路由处理函数。

再试一个更具体的查询：

```python
query = "JWT token 是怎么生成的？"
results = retriever.invoke(query)

for i, doc in enumerate(results[:3]):
    print(f"[{i+1}] {doc.metadata.get('name', '?')}: ")
    print(f"    {doc.page_content[100:400]}")
```

输出：

```
[1] _generate_jwt:
        token = self._generate_jwt(user.id, expires_in=3600)
```

成功定位到了调用 JWT 生成的具体位置。虽然 `_generate_jwt` 的完整实现可能在另一个块中（或者还没写出来），但检索已经帮我们把目光聚焦到了正确的区域。

## 常见误区

**误区一：把代码当纯文本做 RAG**。直接用 `TextLoader` + 标准 `RecursiveCharacterTextSplitter` 处理代码，不考虑语法结构。结果就是函数被拦腰截断，检索时只能拿到半个实现，回答质量大打折扣。

**误区二：忽略文件间的依赖关系**。代码不是孤立存在的——`auth_service.py` 引用了 `user_model.py` 中的 `User` 类，又依赖 `config.py` 中的数据库连接配置。理想的代码 RAG 应该在 metadata 中记录这些 import 依赖关系，在检索时一并返回相关文件。这属于高级优化方向，初学者至少要做到按函数/类级别的正确分块。

**误区三：用太小的 chunk_size**。有些人照搬文档 RAG 的经验，把代码切成 200-300 字符的小块。但一个有意义的代码单元（一个完整的函数体）通常需要 300-1000 字符才能表达完整逻辑。切得太碎会导致每个块都缺乏足够的语义信息，反而降低检索质量。
