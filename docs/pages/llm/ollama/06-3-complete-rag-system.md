# 06-3 完整 RAG 系统实战

## 项目：本地私有知识库问答系统

前面两节我们分别学习了 Embedding 模型和向量数据库。现在到了最激动人心的部分——**把所有组件串联起来，构建一个完整的、可运行的 RAG（检索增强生成）系统**。这个系统将能够：

1. 加载本地文档（PDF / Markdown / TXT）
2. 自动分块和向量化
3. 存入向量数据库
4. 接收用户问题
5. 语义检索相关文档
6. 将检索结果作为上下文送给 LLM
7. 生成有依据的准确回答

```
┌─────────────────────────────────────────────────────────────┐
│           完整 RAG 系统架构图                                │
│                                                             │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐               │
│  │ PDF/MD/  │ → │ 文档     │ → │ Ollama   │               │
│  │ TXT/TXT  │   │ 分块器    │   │ Embedding│               │
│  └──────────┘   └──────────┘   └────┬─────┘               │
│                                      │                      │
│                                      ▼                      │
│                               ┌──────────┐                │
│                               │ 向量数据库 │ (ChromaDB)      │
│                               │ (存储+索引)│                │
│                               └────┬─────┘                │
│                                    │                       │
│  用户问题 ──────────────────────→│                       │
│  "什么是RAG?"              ┌────┴─────┐                │
│                           │ 语义检索  │                │
│                           │ Top-K=3  │                │
│                           └────┬─────┘                │
│                                │                       │
│                                ▼                       │
│                         ┌──────────┐                 │
│                         │ 上下文    │                 │
│                         │ 组装器    │                 │
│                         └────┬─────┘                 │
│                              │                        │
│                              ▼                        │
│                       ┌──────────┐                  │
│                       │ Ollama   │                  │
│                       │ LLM      │                  │
│                       │ (生成回答) │                  │
│                       └────┬─────┘                  │
│                            │                         │
│                            ▼                         │
│                     📝 有据可查的回答                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 完整代码实现（约 250 行）

```python
#!/usr/bin/env python3
"""
🏠 LocalRAG - 本地私有知识库问答系统
完全基于 Ollama 运行，数据不出本机

功能:
- 支持 PDF / Markdown / TXT / DOCX 文档加载
- 智能文档分块（固定长度 / 语义分段）
- Ollama Embedding 向量化
- ChromaDB 向量存储与检索
- Ollama LLM 生成回答
- 引用来源标注

依赖: pip install chromadb PyPDF2 python-docx
"""

import os
import re
import json
import time
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Generator

import requests
import chromadb
from chromadb.config import Settings


# ============================================================
# 第一部分：文档加载器
# ============================================================

class DocumentLoader:
    """多格式文档加载器"""
    
    @staticmethod
    def load(file_path: str) -> str:
        """
        根据文件扩展名自动选择加载方式
        
        Args:
            file_path: 文件路径
            
        Returns:
            文档的纯文本内容
        """
        
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        ext = path.suffix.lower()
        
        if ext == ".pdf":
            return DocumentLoader._load_pdf(path)
        elif ext in [".md", ".markdown"]:
            return DocumentLoader._load_markdown(path)
        elif ext == ".txt":
            return path.read_text(encoding="utf-8")
        elif ext in [".docx", ".doc"]:
            return DocumentLoader._load_docx(path)
        else:
            # 尝试作为纯文本读取
            return path.read_text(encoding="utf-8", errors="ignore")
    
    @staticmethod
    def _load_pdf(path: Path) -> str:
        """加载 PDF 文件"""
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(str(path))
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return text
        except ImportError:
            print("⚠️ 需要安装 PyPDF2: pip install PyPDF2")
            return ""
    
    @staticmethod
    def _load_markdown(path: Path) -> str:
        """加载 Markdown 文件，保留结构信息"""
        content = path.read_text(encoding="utf-8")
        return content
    
    @staticmethod
    def _load_docx(path: Path) -> str:
        """加载 Word 文档"""
        try:
            from docx import Document
            doc = Document(str(path))
            text = "\n".join([para.text for para in doc.paragraphs])
            return text
        except ImportError:
            print("⚠️ 需要安装 python-docx: pip install python-docx")
            return ""


# ============================================================
# 第二部分：文档分块器
# ============================================================

class TextChunker:
    """智能文本分块器"""
    
    def __init__(self, 
                 chunk_size: int = 512,
                 chunk_overlap: int = 64,
                 separators: List[str] = None):
        """
        Args:
            chunk_size: 每个块的最大字符数
            chunk_overlap: 相邻块之间的重叠字符数
            separators: 分隔符优先级列表（从高到低）
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", "。", "！", "？", ".", " ", ""]
    
    def chunk(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        将文本分割为多个块
        
        Args:
            text: 原始文本
            metadata: 附加到每个块的元数据
            
        Returns:
            块列表，每个块包含 content 和 metadata
        """
        
        metadata = metadata or {}
        chunks = []
        
        # 先按段落分割
        paragraphs = self._split_by_separators(text)
        
        current_chunk = ""
        chunk_index = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # 如果当前段落加上已有内容不超过限制
            if len(current_chunk) + len(para) <= self.chunk_size:
                current_chunk += ("\n" if current_chunk else "") + para
            else:
                # 保存当前块
                if current_chunk.strip():
                    chunks.append({
                        "content": current_chunk.strip(),
                        "metadata": {
                            **metadata,
                            "chunk_id": chunk_index,
                            "char_count": len(current_chunk.strip())
                        }
                    })
                    chunk_index += 1
                
                # 如果单个段落就超过限制，强制切割
                if len(para) > self.chunk_size:
                    sub_chunks = self._force_split(para)
                    for sub in sub_chunks[:-1]:
                        chunks.append({
                            "content": sub.strip(),
                            "metadata": {
                                **metadata,
                                "chunk_id": chunk_index,
                                "char_count": len(sub.strip())
                            }
                        })
                        chunk_index += 1
                    current_chunk = sub_chunks[-1] if sub_chunks else ""
                else:
                    current_chunk = para
        
        # 处理最后一个块
        if current_chunk.strip():
            chunks.append({
                "content": current_chunk.strip(),
                "metadata": {
                    **metadata,
                    "chunk_id": chunk_index,
                    "char_count": len(current_chunk.strip())
                }
            })
        
        return chunks
    
    def _split_by_separators(self, text: str) -> List[str]:
        """按分隔符优先级递归分割"""
        for sep in self.separators:
            if sep in text:
                parts = text.split(sep)
                if len(parts) > 1:
                    return [p for p in parts if p.strip()]
        return [text]
    
    def _force_split(self, text: str) -> List[str]:
        """强制按长度分割长文本"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # 尝试在单词边界处断开
            if end < len(text):
                last_space = text.rfind(" ", start, end)
                if last_space > start:
                    end = last_space
            
            chunks.append(text[start:end])
            
            # 下一个块的起点（考虑 overlap）
            start = end - self.chunk_overlap
            if start < 0:
                start = 0
        
        return chunks


# ============================================================
# 第三部分：Ollama 客户端封装
# ============================================================

class OllamaClient:
    """Ollama API 客户端（Embedding + Chat）"""
    
    def __init__(self, 
                 embedding_model: str = "nomic-embed-text",
                 chat_model: str = "qwen2.5:7b",
                 base_url: str = "http://localhost:11434"):
        
        self.embedding_model = embedding_model
        self.chat_model = chat_model
        self.base_url = base_url
    
    def embed(self, text: str) -> List[float]:
        """文本向量化"""
        resp = requests.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.embedding_model, "prompt": text},
            timeout=30
        )
        return resp.json()["embedding"]
    
    def embed_batch(self, texts: List[str], show_progress: bool = True) -> List[List[float]]:
        """批量向量化"""
        embeddings = []
        total = len(texts)
        
        for i, text in enumerate(texts):
            if show_progress and (i + 1) % 50 == 0:
                print(f"    Embedding: {i+1}/{total}")
            embeddings.append(self.embed(text))
            time.sleep(0.003)
        
        return embeddings
    
    def chat(self, messages: List[Dict], temperature: float = 0.3,
             system_prompt: str = None) -> str:
        """对话生成"""
        
        payload = {
            "model": self.chat_model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature}
        }
        
        if system_prompt:
            payload["messages"].insert(0, {
                "role": "system", "content": system_prompt
            })
        
        resp = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=180
        )
        
        return resp.json()["message"]["content"]


# ============================================================
# 第四部分：核心 RAG 引擎
# ============================================================

class LocalRAG:
    """本地私有知识库问答系统"""
    
    def __init__(self,
                 collection_name: str = "knowledge_base",
                 persist_dir: str = "./rag_data",
                 embedding_model: str = "nomic-embed-text",
                 chat_model: str = "qwen2.5:7b",
                 chunk_size: int = 512):
        
        self.client = OllamaClient(embedding_model, chat_model)
        self.chunker = TextChunker(chunk_size=chunk_size)
        
        # 初始化 ChromaDB
        db_client = chromadb.PersistentClient(path=persist_dir)
        self.collection = db_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        self.persist_dir = persist_dir
        print(f"📚 LocalRAG 初始化完成")
        print(f"   Embedding: {embedding_model}")
        print(f"   LLM: {chat_model}")
        print(f"   数据目录: {persist_dir}")
        print(f"   已有文档: {self.collection.count()} 个块")
    
    def ingest(self, file_path: str, batch_size: int = 100):
        """
        加载并索引一个文档
        
        Args:
            file_path: 文件路径
            batch_size: 每批写入 ChromaDB 的数量
        """
        
        print(f"\n📥 正在处理: {file_path}")
        
        # Step 1: 加载文档
        raw_text = DocumentLoader.load(file_path)
        if not raw_text or not raw_text.strip():
            print(f"❌ 文件为空或无法读取")
            return
        
        print(f"   原始文本长度: {len(raw_text)} 字符")
        
        # Step 2: 分块
        file_hash = hashlib.md5(raw_text.encode()).hexdigest()[:8]
        chunks = self.chunker.chunk(raw_text, metadata={
            "source": file_path,
            "hash": file_hash
        })
        
        print(f"   分块数量: {len(chunks)}")
        
        # Step 3: 批量嵌入
        print(f"   正在生成 Embedding...")
        texts = [c["content"] for c in chunks]
        embeddings = self.client.embed_batch(texts)
        
        # Step 4: 写入向量库
        ids = []
        contents = []
        metadatas = []
        
        for i, chunk in enumerate(chunks):
            ids.append(f"{file_hash}_{i:04d}")
            contents.append(chunk["content"])
            metadatas.append(chunk["metadata"])
        
        # 分批写入
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i+batch_size]
            batch_contents = contents[i:i+batch_size]
            batch_embeddings = embeddings[i:i+batch_size]
            batch_metadatas = metadatas[i:i+batch_size]
            
            self.collection.add(
                ids=batch_ids,
                documents=batch_contents,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas
            )
        
        print(f"   ✅ 完成! 当前总文档数: {self.collection.count()}")
    
    def ingest_directory(self, dir_path: str, extensions=None):
        """批量加载目录下的所有文档"""
        
        if extensions is None:
            extensions = {".pdf", ".md", ".txt", ".docx"}
        
        directory = Path(dir_path)
        files = [f for f in directory.iterdir() 
                  if f.is_file() and f.suffix.lower() in extensions]
        
        print(f"\n📂 发现 {len(files)} 个文档:")
        for f in sorted(files):
            print(f"   - {f.name} ({f.stat().st_size / 1024:.1f} KB)")
        
        for f in sorted(files):
            self.ingest(str(f))
    
    def query(self, question: str, top_k: int = 5,
              min_similarity: float = 0.3) -> Dict:
        """
        查询知识库并生成回答
        
        Returns:
            包含 answer, sources, query_time 的字典
        """
        
        start_time = time.time()
        
        # Step 1: 检索
        query_vec = self.client.embed(question)
        
        results = self.collection.query(
            query_embeddings=[query_vec],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        # 格式化检索结果
        sources = []
        context_parts = []
        
        for i in range(len(results["ids"][0])):
            dist = results["distances"][0][i]
            similarity = 1 - dist
            
            if similarity < min_similarity:
                continue
            
            doc_content = results["documents"][0][i]
            meta = results["metadatas"][0][i]
            
            sources.append({
                "content": doc_content[:200] + "...",
                "source": meta.get("source", "?"),
                "similarity": round(similarity, 4),
                "chunk_id": meta.get("chunk_id", i)
            })
            
            context_parts.append(f"[来源{i+1}: {meta.get('source', '?')}]\n{doc_content}")
        
        retrieve_time = time.time() - start_time
        
        if not sources:
            return {
                "answer": "抱歉，我在知识库中没有找到与您问题相关的信息。",
                "sources": [],
                "query_time": round(retrieve_time, 2),
                "found_relevant": False
            }
        
        # Step 2: 构建 Prompt 并生成回答
        context_text = "\n\n---\n\n".join(context_parts)
        
        system_prompt = """你是一个专业的知识库问答助手。
你的任务是基于提供的参考文档来回答用户的问题。

规则：
1. 只依据参考文档中的信息回答，不要编造文档中没有的内容
2. 如果文档中的信息不足以完整回答，请明确说明
3. 回答时引用具体的来源编号（如 [来源1]、[来源2]）
4. 使用清晰的结构化格式输出
5. 保持专业但友好的语气"""
        
        user_prompt = f"""## 参考文档
{context_text}

## 用户问题
{question}

请根据以上参考文档回答用户的问题。"""
        
        answer = self.client.chat(
            [{"role": "user", "content": user_prompt}],
            temperature=0.3,
            system_prompt=system_prompt
        )
        
        total_time = time.time() - start_time
        
        return {
            "answer": answer,
            "sources": sources,
            "query_time": round(total_time, 2),
            "retrieve_time": round(retrieve_time, 2),
            "generate_time": round(total_time - retrieve_time, 2),
            "found_relevant": True
        }


# ============================================================
# 第五部分：交互式命令行界面
# ============================================================

def interactive_cli():
    """命令行交互界面"""
    
    print("""
╔══════════════════════════════════════════════════════════╗
║           🏠 LocalRAG - 本地私有知识库问答系统             ║
║                                                          ║
║   所有数据存储在本地，通过 Ollama 处理，隐私安全          ║
╚════════════════════════════════════════════════════════╝
""")
    
    rag = LocalRAG(
        collection_name="my_knowledge",
        persist_dir="./rag_data",
        embedding_model="nomic-embed-text",
        chat_model="qwen2.5:7b"
    )
    
    while True:
        print("\n" + "─" * 60)
        cmd = input("📌 命令 (load/query/quit/help): ").strip().lower()
        
        if cmd in ["q", "quit", "exit", "退出"]:
            print("👋 再见!")
            break
        
        elif cmd in ["h", "help", "帮助"]:
            print("""
可用命令:
  load <文件路径>       - 加载单个文档到知识库
  loaddir <目录路径>    - 批量加载目录下所有文档
  query <你的问题>      - 提问
  stats                 - 显示知识库统计信息
  quit                  - 退出""")
        
        elif cmd.startswith("load "):
            file_path = cmd[5:].strip().strip('"').strip("'")
            try:
                rag.ingest(file_path)
            except Exception as e:
                print(f"❌ 错误: {e}")
        
        elif cmd.startswith("loaddir "):
            dir_path = cmd[8:].strip().strip('"').strip("'")
            try:
                rag.ingest_directory(dir_path)
            except Exception as e:
                print(f"❌ 错误: {e}")
        
        elif cmd.startswith("query ") or cmd.startswith("q "):
            question = cmd.split(" ", 1)[1].strip()
            
            if not question:
                print("⚠️ 请输入问题")
                continue
            
            print(f"\n🔍 问题: {question}")
            print("⏳ 思考中...\n")
            
            result = rag.query(question, top_k=3)
            
            print("=" * 60)
            print(result["answer"])
            print("=" * 60)
            
            if result.get("sources"):
                print(f"\n📚 参考来源 ({len(result['sources'])} 条):")
                for s in result["sources"]:
                    bar = "█" * int(s["similarity"] * 20)
                    print(f"  [{s['similarity']:.3f}] {bar}")
                    print(f"    来源: {s['source']}")
                    print(f"    内容: {s['content']}")
            
            print(f"\n⏱️ 耗时: 检索 {result.get('retrieve_time', 0)}s"
                  f" + 生成 {result.get('generate_time', 0)}s"
                  f" = 总计 {result.get('query_time', 0)}s")
        
        elif cmd == "stats":
            count = rag.collection.count()
            print(f"\n📊 知识库统计:")
            print(f"   文档块总数: {count}")
            print(f"   数据目录: {rag.persist_dir}")
            print(f"   Embedding 模型: {rag.client.embedding_model}")
            print(f"   LLM 模型: {rag.client.chat_model}")
        
        else:
            print("❓ 未知命令。输入 'help' 查看帮助。")


if __name__ == "__main__":
    interactive_cli()
```

## 使用流程演示

```bash
# 启动系统
$ python3 local_rag.py

╔══════════════════════════════════════════════════════════╗
║           🏠 LocalRAG - 本地私有知识库问答系统             ║
╚════════════════════════════════════════════════════════╝

📌 命令: load ./docs/product-manual.md

📥 正在处理: ./docs/product-manual.md
   原始文本长度: 45678 字符
   分块数量: 89
   正在生成 Embedding...
   ✅ 完成! 当前总文档数: 89

📌 命令: query 如何重置管理员密码？

🔍 问题: 如何重置管理员密码？
⏳ 思考中...

============================================================
根据产品手册中的说明，重置管理员密码的步骤如下：

## 重置步骤

### 方法一：通过控制台重置（推荐）

1. 登录服务器管理控制台 [来源1]
2. 导航至「系统设置」→「账户管理」页面
3. 点击管理员账户旁边的「...」菜单
4. 选择「重置密码」选项
5. 输入新密码（需满足：8位以上、包含大小写字母和数字）
6. 点击确认完成重置 [来源1]

### 方法二：命令行方式

如果无法访问控制台，可以通过 SSH 登录服务器后执行以下命令：

```bash
sudo admin-tool reset-password --user admin
```

系统会生成一个临时密码，首次登录时需要修改。

### 注意事项
- 重置密码后会话中所有已登录的用户需要重新登录 [来源2]
- 建议定期更换密码，且不要使用默认密码 [来源2]

============================================================

📚 参考来源 (2 条):
  [0.8234] ████████████████████████
    来源: ./docs/product-manual.md
    内容: ## 第三章 账户管理...

  [0.7562] ████████████████████░░░
    来源: ./docs/product-manual.md
    内容: ## 安全策略配置...

⏱️ 耗时: 检索 0.23s + 生成 3.45s = 总计 3.68s
```

## 评估方法

如何判断你的 RAG 系统效果好不好？以下是三个关键指标：

```python
#!/usr/bin/env python3
"""RAG 系统评估框架"""

def evaluate_rag(rag_system, test_cases):
    """
    评估 RAG 系统的质量
    
    test_cases 格式:
    [
        {
            "question": "问题",
            "expected_answer_keywords": ["关键词1", "关键词2"],
            "expected_source_files": ["doc1.pdf"],
            "min_expected_similarity": 0.6
        },
        ...
    ]
    """
    
    metrics = {
        "total": len(test_cases),
        "retrieval_hit": 0,      # 检索命中率
        "answer_relevant": 0,    # 回答相关性
        "source_correct": 0,     # 来源正确性
        "avg_latency": 0,        # 平均延迟
        "details": []
    }
    
    for tc in test_cases:
        result = rag_system.query(tc["question"], top_k=3)
        
        detail = {"question": tc["question"], "passed": True}
        
        # 检查是否检索到相关文档
        if result.get("found_relevant") and result["sources"]:
            best_sim = max(s["similarity"] for s in result["sources"])
            expected_min = tc.get("min_expected_similarity", 0.5)
            
            if best_sim >= expected_min:
                metrics["retrieval_hit"] += 1
            else:
                detail["passed"] = False
                detail["reason"] = f"相似度不足: {best_sim} < {expected_min}"
        else:
            detail["passed"] = False
            detail["reason"] = "未检索到任何文档"
        
        # 检查回答是否包含期望的关键词
        if result.get("answer"):
            answer_lower = result["answer"].lower()
            keywords_found = sum(
                1 for kw in tc.get("expected_answer_keywords", [])
                if kw.lower() in answer_lower
            )
            if keywords_found >= len(tc.get("expected_answer_keywords", [])) * 0.5:
                metrics["answer_relevant"] += 1
        
        # 检查来源文件是否匹配
        source_files = set(s.get("source", "") for s in result.get("sources", []))
        expected_files = set(tc.get("expected_source_files", []))
        if source_files & expected_files:
            metrics["source_correct"] += 1
        
        metrics["avg_latency"] += result.get("query_time", 0)
        metrics["details"].append(detail)
    
    n = metrics["total"]
    metrics["avg_latency"] /= n if n > 0 else 1
    
    print("\n" + "=" * 60)
    print("📊 RAG 系统评估报告")
    print("=" * 60)
    print(f"  总测试用例: {n}")
    print(f"  检索命中率: {metrics['retrieval_hit']}/{n} "
          f"({metrics['retrieval_hit']/n*100:.1f}%)")
    print(f"  回答相关性: {metrics['answer_relevant']}/{n} "
          f"({metrics['answer_relevant']/n*100:.1f}%)")
    print(f"  来源正确性: {metrics['source_correct']}/{n} "
          f"({metrics['source_correct']/n*100:.1f}%)")
    print(f"  平均延迟: {metrics['avg_latency']:.2f}s")
    
    return metrics
```

## 本章小结

这一节我们构建了一个完整的端到端 RAG 系统：

1. **四层架构**：DocumentLoader（加载）→ TextChunker（分块）→ OllamaClient（Embedding+LLM）→ LocalRAG（编排）
2. **支持多种文档格式**：PDF（PyPDF2）、Markdown（原生）、TXT（原生）、DOCX（python-docx）
3. **智能分块策略**：优先按段落分割，超长文本强制截断，保持 chunk_overlap 重叠避免语义断裂
4. **ChromaDB 作为持久化向量存储**，支持增量添加和持久化重启
5. **引用来源标注**让每条回答都有据可查
6. **评估框架**从检索命中率、回答相关性、来源正确性三个维度衡量系统质量

下一节我们将学习高级 RAG 技术——混合检索、重排序、父-子分块等进阶方案。
