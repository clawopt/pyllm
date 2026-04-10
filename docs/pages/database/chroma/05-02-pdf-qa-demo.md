# 5.2 端到端 RAG Demo：PDF 文档问答

> **从 PDF 文件到智能问答——用 Chroma 构建一个真正能用的 RAG 系统**

---

## 这一节在讲什么？

前面我们讲了 RAG 的架构原理和 Chroma 在其中的角色定位，但一直停留在概念层面。这一节我们要动手实现一个完整的、可运行的 PDF 文档问答系统——从加载 PDF 文件、切分文本、向量化入库，到用户提问、检索相关文档、组装 prompt、调用 LLM 生成回答，走通 RAG 的全流程。这个 Demo 不仅是学习成果的检验，也是你后续构建生产级 RAG 系统的起点。

---

## 项目结构

```
pdf_qa_demo/
├── data/                    # PDF 文档存放目录
│   └── sample.pdf
├── ingest.py               # 文档加载+切分+入库
├── query.py                # 用户提问→检索→生成回答
├── config.py               # 配置管理
└── requirements.txt        # 依赖
```

### requirements.txt

```
chromadb>=0.4.0
sentence-transformers>=2.2.0
pypdf>=3.0.0
openai>=1.0.0
```

---

## Step 1：配置管理

把所有可配置的参数集中管理，避免散落在代码各处：

```python
# config.py
import os

class Config:
    # Chroma 配置
    CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "pdf_qa")

    # Embedding 配置
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")

    # 切分配置
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))

    # 检索配置
    DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))
    MAX_DISTANCE = float(os.getenv("MAX_DISTANCE", "1.2"))

    # LLM 配置
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # openai / local
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))

    # 数据目录
    DATA_DIR = os.getenv("DATA_DIR", "./data")
```

---

## Step 2：文档加载与切分

PDF 文件的加载是 RAG 系统的入口。我们使用 `pypdf` 库提取文本，然后按递归字符级策略切分：

```python
# ingest.py
import os
import hashlib
import time
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
from config import Config


def load_pdf(file_path: str) -> list:
    """
    加载 PDF 文件，按页提取文本

    返回:
        list of dict，每个元素包含 page_number 和 text
    """
    reader = PdfReader(file_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            pages.append({
                "page_number": i + 1,
                "text": text.strip()
            })
    return pages


def recursive_chunk(text: str, chunk_size: int = 600, overlap: int = 80) -> list:
    """递归字符级切分"""
    separators = ["\n\n", "\n", "。", "！", "？", ".", " ", ""]

    def _split(text, sep_idx):
        if sep_idx >= len(separators):
            return [text] if text else []
        sep = separators[sep_idx]
        parts = text.split(sep) if sep else list(text)
        chunks = []
        current = ""
        for part in parts:
            candidate = current + sep + part if current else part
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                if len(part) > chunk_size:
                    chunks.extend(_split(part, sep_idx + 1))
                    current = ""
                else:
                    current = part
        if current:
            chunks.append(current)
        return chunks

    raw_chunks = _split(text, 0)
    result = []
    for i, chunk in enumerate(raw_chunks):
        if i > 0 and overlap > 0:
            prev_tail = raw_chunks[i-1][-overlap:]
            chunk = prev_tail + chunk
        result.append(chunk.strip())
    return [c for c in result if c]


def ingest_pdf(file_path: str, category: str = "general"):
    """
    加载 PDF → 切分 → 入库到 Chroma

    参数:
        file_path: PDF 文件路径
        category: 文档分类
    """
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return 0

    # 加载 PDF
    pages = load_pdf(file_path)
    if not pages:
        print(f"❌ 无法从 {file_path} 提取文本")
        return 0

    print(f"📖 加载 {file_path}: {len(pages)} 页")

    # 初始化 Chroma
    client = chromadb.Client(settings=chromadb.Settings(
        is_persistent=True,
        persist_directory=Config.CHROMA_PERSIST_DIR
    ))

    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=Config.EMBEDDING_MODEL
    )

    collection = client.get_or_create_collection(
        name=Config.CHROMA_COLLECTION_NAME,
        embedding_function=ef
    )

    # 切分并入库
    source_name = os.path.basename(file_path)
    current_ts = int(time.time())
    total_chunks = 0

    for page in pages:
        chunks = recursive_chunk(
            page["text"],
            chunk_size=Config.CHUNK_SIZE,
            overlap=Config.CHUNK_OVERLAP
        )

        ids = []
        documents = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            chunk_id = hashlib.md5(
                f"{source_name}::p{page['page_number']}::c{i}".encode()
            ).hexdigest()[:16]

            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append({
                "source": source_name,
                "category": category,
                "page": page["page_number"],
                "chunk_index": i,
                "total_chunks": len(chunks),
                "created_at": current_ts,
                "version": 1,
                "language": "zh",
            })

        if ids:
            collection.upsert(documents=documents, ids=ids, metadatas=metadatas)
            total_chunks += len(ids)

    print(f"✅ 入库完成: {source_name} → {total_chunks} 个 chunk")
    print(f"📊 Collection 总文档数: {collection.count()}")
    return total_chunks


def ingest_directory(directory: str = None, category: str = "general"):
    """批量入库目录下所有 PDF 文件"""
    directory = directory or Config.DATA_DIR
    if not os.path.exists(directory):
        print(f"❌ 目录不存在: {directory}")
        return

    pdf_files = [f for f in os.listdir(directory) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"⚠️ 目录中没有 PDF 文件: {directory}")
        return

    print(f"📂 发现 {len(pdf_files)} 个 PDF 文件")
    total = 0
    for pdf_file in pdf_files:
        file_path = os.path.join(directory, pdf_file)
        total += ingest_pdf(file_path, category)

    print(f"\n🎉 全部入库完成: 共 {total} 个 chunk")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        ingest_pdf(sys.argv[1])
    else:
        ingest_directory()
```

---

## Step 3：检索与生成

```python
# query.py
import chromadb
from chromadb.utils import embedding_functions
from config import Config


class PDFQueryEngine:
    """PDF 文档问答引擎"""

    def __init__(self):
        self.client = chromadb.Client(settings=chromadb.Settings(
            is_persistent=True,
            persist_directory=Config.CHROMA_PERSIST_DIR
        ))

        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=Config.EMBEDDING_MODEL
        )

        self.collection = self.client.get_or_create_collection(
            name=Config.CHROMA_COLLECTION_NAME,
            embedding_function=ef
        )

    def retrieve(self, question: str, category: str = None,
                 n_results: int = None, max_distance: float = None):
        """检索相关文档片段"""
        n_results = n_results or Config.DEFAULT_TOP_K
        max_distance = max_distance or Config.MAX_DISTANCE

        where = {"category": category} if category else None

        results = self.collection.query(
            query_texts=[question],
            where=where,
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )

        # 过滤距离过远的结果
        relevant = []
        for i in range(len(results['ids'][0])):
            dist = results['distances'][0][i]
            if dist <= max_distance:
                relevant.append({
                    "document": results['documents'][0][i],
                    "source": results['metadatas'][0][i]["source"],
                    "page": results['metadatas'][0][i]["page"],
                    "distance": dist
                })

        return relevant

    def build_prompt(self, question: str, contexts: list) -> str:
        """组装 RAG prompt"""
        context_text = "\n\n".join([
            f"[来源: {c['source']} 第{c['page']}页]\n{c['document']}"
            for c in contexts
        ])

        prompt = f"""你是一个专业的文档问答助手。请基于以下参考信息回答用户的问题。

要求：
1. 只基于参考信息回答，不要编造信息
2. 如果参考信息中没有相关内容，请回答"我没有在文档中找到相关信息"
3. 在回答中引用来源（文件名和页码）

参考信息：
{context_text}

用户问题：{question}

回答："""
        return prompt

    def generate(self, prompt: str) -> str:
        """调用 LLM 生成回答"""
        if Config.LLM_PROVIDER == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=Config.OPENAI_API_KEY)
            response = client.chat.completions.create(
                model=Config.OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=Config.LLM_MAX_TOKENS,
                temperature=Config.LLM_TEMPERATURE
            )
            return response.choices[0].message.content
        else:
            return "[LLM 生成功能需要配置 OPENAI_API_KEY]"

    def ask(self, question: str, category: str = None, verbose: bool = True):
        """完整的 RAG 查询流程：检索 → 组装 → 生成"""
        # 检索
        contexts = self.retrieve(question, category=category)

        if not contexts:
            return {
                "question": question,
                "answer": "我没有在文档中找到与您问题相关的信息。",
                "sources": [],
                "n_contexts": 0
            }

        # 组装 prompt
        prompt = self.build_prompt(question, contexts)

        # 生成回答
        answer = self.generate(prompt)

        if verbose:
            print(f"\n{'='*60}")
            print(f"❓ 问题: {question}")
            print(f"📄 检索到 {len(contexts)} 条相关文档:")
            for c in contexts:
                print(f"   - {c['source']} 第{c['page']}页 (距离: {c['distance']:.4f})")
            print(f"\n💬 回答:\n{answer}")
            print(f"{'='*60}")

        return {
            "question": question,
            "answer": answer,
            "sources": [{"source": c["source"], "page": c["page"]} for c in contexts],
            "n_contexts": len(contexts)
        }


if __name__ == "__main__":
    engine = PDFQueryEngine()

    print(f"📊 知识库文档数: {engine.collection.count()}")

    while True:
        question = input("\n❓ 请输入问题 (输入 q 退出): ").strip()
        if question.lower() in ("q", "quit", "exit"):
            break
        if not question:
            continue
        engine.ask(question)
```

---

## Step 4：运行 Demo

```bash
# 1. 安装依赖
pip install chromadb sentence-transformers pypdf openai

# 2. 准备 PDF 文件
mkdir -p data
cp ~/Documents/some_document.pdf data/

# 3. 入库文档
python ingest.py data/some_document.pdf

# 4. 开始问答
export OPENAI_API_KEY="sk-..."
python query.py
```

---

## 常见误区

### 误区 1：PDF 提取的文本质量都很好

实际上 PDF 的文本提取质量差异很大——扫描件 PDF 无法提取文本（需要 OCR）、多栏排版的 PDF 可能提取出交错文本、表格和图表的文本通常是乱序的。对于这些情况，需要使用更专业的工具（如 `pdfplumber`、`unstructured`、`marker`）做预处理。

### 误区 2：所有 PDF 都用同一个切分参数

不同类型的 PDF 适合不同的切分参数。技术手册段落较长，适合 chunk_size=800~1000；FAQ 文档条目较短，适合 chunk_size=300~500。建议按文档类型调整参数。

### 误区 3：入库一次就再也不用更新

文档会更新、新文档会加入、旧版本需要归档。建议设计一个增量更新机制：用 metadata 中的 version 字段标记版本，新版本入库后用 where 过滤只查最新版本。

---

## 本章小结

这一节我们实现了一个完整的 PDF 文档问答系统，走通了 RAG 的全流程。核心要点回顾：第一，PDF 加载使用 `pypdf` 按页提取文本，然后递归字符级切分为 chunk；第二，入库时为每个 chunk 生成确定性 ID（基于 source+page+chunk_index 的哈希），支持幂等的 upsert 操作；第三，检索时设定距离阈值过滤不相关结果，避免 LLM 基于错误信息产生幻觉；第四，prompt 组装时加入来源引用指令，让 LLM 在回答中标注信息出处；第五，PDF 文本提取质量是 RAG 系统的隐性瓶颈，扫描件和复杂排版需要专门的预处理工具。

下一节我们将讲对话历史管理与 Memory Layer——如何让 RAG 系统拥有"记忆"，实现多轮对话和用户画像。
