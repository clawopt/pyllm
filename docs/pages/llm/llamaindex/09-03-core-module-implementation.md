# 9.3 核心模块实现

## 从设计到代码：核心模块的落地

前两节我们完成了需求分析和架构设计，有了清晰的蓝图。现在要进入最实质性的阶段——把设计图变成可运行的代码。这一节会涵盖企业知识库系统最核心的几个模块：统一数据加载器、RAG 引擎主类、权限过滤检索器、引用构建器和查询分类器。每个模块都会给出完整的实现代码，并解释关键的设计决策和容易踩的坑。

## 统一数据加载器：DataLoaderHub

在第二章中我们学习了 LlamaIndex 的各种 Connector，但在企业场景中不能每次都手动选择用哪个 Reader——我们需要一个统一的入口来管理所有数据源。`DataLoaderHub` 就是这个角色：

```python
from abc import ABC, abstractmethod
from typing import Any, Generator
from pathlib import Path
from dataclasses import dataclass, field
from llama_index.core import Document
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core.schema import TextNode
import logging

logger = logging.getLogger(__name__)


@dataclass
class LoadResult:
    """数据加载结果"""
    source_id: int
    documents: list[Document] = field(default_factory=list)
    total_chars: int = 0
    file_count: int = 0
    errors: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class BaseSourceLoader(ABC):
    """数据源加载器的基类"""

    def __init__(self, config: dict):
        self.config = config
        self.source_type = self.__class__.__name__

    @abstractmethod
    def load(self) -> LoadResult:
        """加载数据并返回 Document 列表"""
        ...

    @abstractmethod
    def detect_changes(self) -> list[str]:
        """检测自上次加载以来的变更文件/记录"""
        ...

    @abstractmethod
    def get_source_info(self) -> dict:
        """返回数据源的元信息（文档数量、总大小等）"""
        ...


class FileDirectoryLoader(BaseSourceLoader):
    """文件目录加载器——支持 PDF/Word/Markdown/HTML/Excel/PPT"""

    SUPPORTED_EXTENSIONS = {
        ".pdf": {"engine": "pymupdf", "description": "PDF 文档"},
        ".docx": {"engine": "python-docx", "description": "Word 文档"},
        ".doc": {"engine": "antiword", "description": "旧版 Word（需安装 antiword）"},
        ".md": {"engine": "default", "description": "Markdown 文件"},
        ".markdown": {"engine": "default", "description": "Markdown 文件"},
        ".html": {"engine": "default", "description": "HTML 网页"},
        ".htm": {"engine": "default", "description": "HTML 网页"},
        ".xlsx": {"engine": "openpyxl", "description": "Excel 表格"},
        ".xls": {"engine": "xlrd", "description": "旧版 Excel"},
        ".csv": {"engine": "default", "description": "CSV 数据"},
        ".pptx": {"engine": "python-pptx", "description": "PowerPoint"},
        ".txt": {"engine": "default", "description": "纯文本"},
        ".rst": {"engine": "default", "description": "reStructuredText"},
    }

    def __init__(self, config: dict):
        super().__init__(config)
        self.directory = Path(config["path"])
        self.file_patterns = config.get("file_patterns", ["**/*"])
        self.exclude_patterns = config.get(
            "exclude_patterns",
            ["**/.git/**", "**/node_modules/**", "**/__pycache__/**", "**/*.tmp"],
        )
        self.recursive = config.get("recursive", True)
        self.max_file_size_mb = config.get("max_file_size_mb", 50)

    def load(self) -> LoadResult:
        result = LoadResult(source_id=self.config.get("source_id", 0))

        if not self.directory.exists():
            result.errors.append({
                "type": "directory_not_found",
                "message": f"目录不存在: {self.directory}",
            })
            logger.error(f"目录不存在: {self.directory}")
            return result

        try:
            reader = SimpleDirectoryReader(
                input_dir=str(self.directory),
                required_exts=list(self.SUPPORTED_EXTENSIONS.keys()),
                recursive=self.recursive,
                exclude=self.exclude_patterns,
                num_files_limit=self.config.get("num_files_limit"),
                file_metadata=lambda filepath: {
                    "file_name": Path(filepath).name,
                    "file_path": str(filepath),
                    "file_size_bytes": Path(filepath).stat().st_size,
                },
            )

            documents = reader.load_data()
            result.documents = documents
            result.file_count = len(documents)
            result.total_chars = sum(len(doc.text) for doc in documents)

            # 检查是否有超大文件被跳过
            for doc in documents:
                size_mb = doc.metadata.get("file_size_bytes", 0) / (1024 * 1024)
                if size_mb > self.max_file_size_mb:
                    result.warnings.append(
                        f"大文件 {doc.metadata.get('file_name')} ({size_mb:.1f}MB)"
                        f" 超过建议大小上限 {self.max_file_size_mb}MB"
                    )

            logger.info(
                f"文件目录加载完成: {result.file_count} 个文档, "
                f"{result.total_chars} 字符"
            )

        except Exception as e:
            result.errors.append({"type": "load_error", "message": str(e)})
            logger.error(f"文件目录加载失败: {e}", exc_info=True)

        return result

    def detect_changes(self) -> list[str]:
        """通过文件修改时间检测变更"""
        import os
        from datetime import datetime

        changed_files = []
        cutoff = self.config.get("last_sync_time")

        if not cutoff:
            return [str(p) for p in self.directory.rglob("*") if p.is_file()]

        cutoff_dt = datetime.fromisoformat(cutoff) if isinstance(cutoff, str) else cutoff

        for pattern in self.file_patterns:
            for file_path in self.directory.glob(pattern):
                if file_path.is_file():
                    mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if mtime > cutoff_dt:
                        changed_files.append(str(file_path))

        return changed_files

    def get_source_info(self) -> dict:
        total_files = sum(1 for _ in self.directory.rglob("*") if _.is_file())
        total_size = sum(
            f.stat().st_size for f in self.directory.rglob("*") if f.is_file()
        )
        return {
            "type": "file_directory",
            "path": str(self.directory),
            "total_files": total_files,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "supported_formats": list(self.SUPPORTED_EXTENSIONS.keys()),
        }


class DatabaseLoader(BaseSourceLoader):
    """数据库加载器——将数据库记录转为 Document 对象"""

    def __init__(self, config: dict):
        super().__init__(config)
        self.db_type = config["db_type"]  # postgresql / mysql / sqlite
        self.connection_string = config["connection_string"]
        self.query_template = config.get("query")
        self.tables = config.get("tables", [])
        self.id_column = config.get("id_column", "id")
        self.content_column = config.get("content_column", "content")
        self.title_column = config.get("title_column", "title")
        self.metadata_columns = config.get("metadata_columns", [])
        self.batch_size = config.get("batch_size", 500)

    def _get_connection(self):
        """根据数据库类型创建连接"""
        if self.db_type == "postgresql":
            import psycopg2
            return psycopg2.connect(self.connection_string)
        elif self.db_type == "mysql":
            import mysql.connector
            return mysql.connector.connect(**self._parse_mysql_config())
        elif self.db_type == "sqlite":
            import sqlite3
            return sqlite3.connect(self.connection_string)
        else:
            raise ValueError(f"不支持的数据库类型: {self.db_type}")

    def _parse_mysql_config(self) -> dict:
        """从连接字符串解析 MySQL 配置"""
        import re
        match = re.match(
            r"mysql\+(?:pymysql|mysqlconnector)://([^:]+):([^@]+)@([^:]+):(\d+)/(.+)",
            self.connection_string,
        )
        if match:
            return {
                "user": match.group(1),
                "password": match.group(2),
                "host": match.group(3),
                "port": int(match.group(4)),
                "database": match.group(5),
            }
        raise ValueError(f"无法解析 MySQL 连接字符串: {self.connection_string}")

    def load(self) -> LoadResult:
        result = LoadResult(source_id=self.config.get("source_id", 0))

        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            if self.query_template:
                query = self.query_template
            else:
                columns = [self.id_column, self.content_column]
                if self.title_column:
                    columns.append(self.title_column)
                columns.extend(self.metadata_columns)

                table_name = self.tables[0] if self.tables else "documents"
                query = f"SELECT {', '.join(columns)} FROM {table_name}"

            cursor.execute(query)

            documents = []
            while True:
                rows = cursor.fetchmany(self.batch_size)
                if not rows:
                    break

                for row in rows:
                    metadata = {}
                    col_names = [desc[0] for desc in cursor.description]

                    for i, col_name in enumerate(col_names):
                        if col_name == self.content_column:
                            content = str(row[i]) if row[i] is not None else ""
                        elif col_name == self.title_column:
                            metadata["title"] = str(row[i]) if row[i] is not None else ""
                        elif col_name == self.id_column:
                            metadata["db_id"] = str(row[i])
                        elif col_name in self.metadata_columns or col_name not in (
                            self.id_column,
                            self.content_column,
                            self.title_column,
                        ):
                            metadata[f"db_{col_name}"] = row[i]

                    if content.strip():
                        doc = Document(text=content, metadata=metadata)
                        documents.append(doc)

            result.documents = documents
            result.file_count = len(documents)
            result.total_chars = sum(len(doc.text) for doc in documents)

            cursor.close()
            conn.close()

            logger.info(
                f"数据库加载完成 ({self.db_type}): "
                f"{result.file_count} 条记录, {result.total_chars} 字符"
            )

        except Exception as e:
            result.errors.append({"type": "db_error", "message": str(e)})
            logger.error(f"数据库加载失败: {e}", exc_info=True)

        return result

    def detect_changes(self) -> list[str]:
        """基于 updated_at 或 id 范围检测变更"""
        conn = self._get_connection()
        cursor = conn.cursor()

        last_sync_id = self.config.get("last_sync_id", "0")
        table = self.tables[0] if self.tables else "documents"

        try:
            cursor.execute(
                f"SELECT {self.id_column} FROM {table} "
                f"WHERE {self.id_column} > %s ORDER BY {self.id_column}",
                (last_sync_id,),
            )
            changed_ids = [str(row[0]) for row in cursor.fetchall()]
            cursor.close()
            conn.close()
            return changed_ids
        except Exception as e:
            logger.warning(f"变更检测失败: {e}")
            cursor.close()
            conn.close()
            return []

    def get_source_info(self) -> dict:
        conn = self._get_connection()
        cursor = conn.cursor()
        table = self.tables[0] if self.tables else "documents"

        try:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]

            cursor.close()
            conn.close()

            return {
                "type": f"database_{self.db_type}",
                "table": table,
                "total_records": count,
                "columns": self.metadata_columns + [
                    self.id_column,
                    self.content_column,
                    self.title_column,
                ],
            }
        except Exception:
            cursor.close()
            conn.close()
            return {"type": f"database_{self.db_type}", "error": "无法获取表信息"}


class DataLoaderHub:
    """统一数据加载中心——注册和管理所有数据源加载器"""

    _loaders: dict[str, type[BaseSourceLoader]] = {}

    @classmethod
    def register(cls, name: str, loader_class: type[BaseSourceLoader]):
        cls._loaders[name] = loader_class
        logger.info(f"已注册数据源加载器: {name}")

    @classmethod
    def get_loader(cls, source_type: str, config: dict) -> BaseSourceLoader:
        if source_type not in cls._loaders:
            raise ValueError(f"未知的数据源类型: {source_type}, 可用: {list(cls._loaders.keys())}")
        return cls._loaders[source_type](config)

    @classmethod
    def load_all(cls, sources: list[dict]) -> dict[int, LoadResult]:
        """批量加载多个数据源"""
        results = {}
        for source_config in sources:
            source_type = source_config["source_type"]
            loader = cls.get_loader(source_type, source_config)
            result = loader.load()
            results[source_config.get("source_id", 0)] = result
        return results


# 注册内置加载器
DataLoaderHub.register("file_directory", FileDirectoryLoader)
DataLoaderHub.register("database_postgres", DatabaseLoader)
DataLoaderHub.register("database_mysql", DatabaseLoader)
```

这个 `DataLoaderHub` 的设计采用了**策略模式（Strategy Pattern）+ 工厂模式（Factory Pattern）**的组合。每种数据源类型对应一个 Loader 类（策略），`DataLoaderHub` 作为工厂根据 `source_type` 创建对应的实例。新增一种数据源只需要：（1）继承 `BaseSourceLoader` 实现新类；（2）调用 `DataLoaderHub.register()` 注册。不需要修改任何已有代码——这符合开闭原则（Open-Closed Principle）。

有一个容易被忽略的细节是 `FileDirectoryLoader` 中的 `max_file_size_mb` 检查。在实际项目中经常有人上传几百 MB 的 PDF 扫描件（比如整本产品手册的扫描版），这种文件不仅解析极慢，而且 OCR 后的质量通常很差——满篇错别字和乱码。与其让它在解析阶段卡住整个流程，不如提前检测并发出警告。50MB 的阈值是根据经验设定的——正常的产品 PDF 手册一般在 5-20MB，超过 50MB 的很可能是扫描件或包含大量高清图片的非文本 PDF。

## RAG 引擎主类：EnterpriseRAGEngine

这是整个系统的大脑——它把前面学过的所有 LlamaIndex 组件组装成一个完整的问答流水线：

```python
import time
import hashlib
from typing import Optional
from dataclasses import dataclass, field
from enum import Enum

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
    Document,
    QueryBundle,
)
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.response_synthesizers import ResponseMode, get_response_synthesizer
from llama_index.core.postprocessor import SimilarityPostProcessor, LongContextReorder
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from app.data.loaders import DataLoaderHub
from app.core.retriever import HybridRetriever
from app.core.citation_builder import CitationBuilder
from app.core.query_classifier import QueryClassifier


class RetrievalMode(Enum):
    VECTOR_ONLY = "vector_only"
    HYBRID = "hybrid"
    KEYWORD_ONLY = "keyword_only"


@dataclass
class RAGConfig:
    """RAG 引擎配置"""
    embedding_model: str = "text-embedding-3-large"
    llm_model: str = "gpt-4o"
    chunk_size: int = 512
    chunk_overlap: int = 50
    similarity_top_k: int = 15
    rerank_top_k: int = 5
    similarity_cutoff: float = 0.5
    retrieval_mode: RetrievalMode = RetrievalMode.HYBRID
    use_hyde: bool = False
    use_reranker: bool = True
    reranker_model: str = "cohere-rerank-v3.5"
    response_mode: ResponseMode = ResponseMode.REFINE
    enable_citation: bool = True
    max_context_tokens: int = 8000


@dataclass
class QueryResult:
    """查询结果"""
    answer: str
    sources: list[dict]
    citations: list[dict]
    confidence: float
    latency_ms: float
    nodes_used: int
    tokens_used: int
    retrieval_details: dict = field(default_factory=dict)


class EnterpriseRAGEngine:
    """
    企业级 RAG 引擎

    将 LlamaIndex 的各组件组装为完整的问答流水线，
    支持混合检索、权限过滤、重排序、引用溯源等功能。
    """

    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        self._initialize_settings()
        self._index: Optional[VectorStoreIndex] = None
        self._retriever: Optional[HybridRetriever] = None
        self._citation_builder = CitationBuilder()
        self._query_classifier = QueryClassifier()

    def _initialize_settings(self):
        """初始化 LlamaIndex 全局设置"""
        Settings.llm = OpenAI(model=self.config.llm_model, temperature=0.1)
        Settings.embed_model = OpenAIEmbedding(
            model=self.config.embedding_model,
            dimensions=1024 if "large" in self.config.embedding_model else None,
        )
        Settings.node_parser = SentenceSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )

    def build_index_from_sources(self, sources: list[dict]) -> dict:
        """从多个数据源构建索引"""
        start = time.perf_counter()

        all_documents = []
        load_results = DataLoaderHub.load_all(sources)

        total_docs = 0
        total_errors = 0
        for source_id, result in load_results.items():
            all_documents.extend(result.documents)
            total_docs += result.file_count
            total_errors += len(result.errors)
            if result.errors:
                for err in result.errors:
                    logger.warning(f"数据源 {source_id} 加载错误: {err}")

        if not all_documents:
            return {"status": "error", "message": "没有成功加载任何文档"}

        logger.info(f"共加载 {len(all_documents)} 个文档, 开始构建向量索引...")

        qdrant_client = QdrantClient(url="http://qdrant:6333")
        vector_store = QdrantVectorStore(
            client=qdrant_client,
            collection_name="kb_main",
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        self._index = VectorStoreIndex.from_documents(
            documents=all_documents,
            storage_context=storage_context,
            show_progress=True,
        )

        self._setup_retriever()

        elapsed = time.perf_counter() - start
        logger.info(f"索引构建完成, 耗时 {elapsed:.1f}s, 文档数: {len(all_documents)}")

        return {
            "status": "success",
            "document_count": len(all_documents),
            "node_count": len(list(self._index.docstore.docs.values())),
            "build_time_seconds": round(elapsed, 2),
            "errors": total_errors,
        }

    def _setup_retriever(self):
        """配置检索器"""
        if self.config.retrieval_mode == RetrievalMode.HYBRID:
            self._retriever = HybridRetriever(
                index=self._index,
                similarity_top_k=self.config.similarity_top_k,
                sparse_top_k=20,
            )
        else:
            self._retriever = self._index.as_retriever(
                similarity_top_k=self.config.similarity_top_k
            )

    async def query(
        self,
        question: str,
        user_context: Optional[dict] = None,
        chat_history: Optional[list[dict]] = None,
        allowed_doc_ids: Optional[list[str]] = None,
        stream: bool = False,
    ) -> QueryResult:
        """
        执行完整 RAG 查询

        Args:
            question: 用户问题
            user_context: 用户上下文（部门、角色等）
            chat_history: 多轮对话历史
            allowed_doc_ids: 用户有权访问的文档ID列表（权限过滤）
            stream: 是否流式输出
        """
        start = time.perf_counter()

        if not self._index or not self._retriever:
            raise RuntimeError("索引尚未构建，请先调用 build_index_from_sources()")

        # Step 1: 问题预处理与分类
        query_type = self._query_classifier.classify(question)
        enriched_query = self._enrich_query(question, chat_history, user_context)

        # Step 2: 检索
        raw_nodes = await self._retriever.aretrieve(enriched_query)

        # Step 3: 权限过滤
        if allowed_doc_ids:
            filtered_nodes = self._filter_by_permission(raw_nodes, allowed_doc_ids)
        else:
            filtered_nodes = raw_nodes

        # Step 4: 重排序
        if self.config.use_reranker and filtered_nodes:
            reranked_nodes = await self._rerank(filtered_nodes, enriched_query)
        else:
            reranked_nodes = filtered_nodes[: self.config.rerank_top_k]

        # Step 5: 后处理
        postprocessed = self._postprocess(reranked_nodes)

        # Step 6: 回答合成
        synthesizer = get_response_synthesizer(
            response_mode=self.config.response_mode,
            use_async=True,
        )

        if stream:
            response = await synthesizer.asynthesize(
                query=enriched_query,
                nodes=postprocessed,
            )
            # 流式处理逻辑...
            answer = str(response)
        else:
            response = await synthesizer.asynthesize(
                query=enriched_query,
                nodes=postprocessed,
            )
            answer = str(response)

        # Step 7: 构建引用
        citations = []
        if self.config.enable_citation and postprocessed:
            citations = self._citation_builder.build(answer, postprocessed)

        # Step 8: 计算置信度
        confidence = self._compute_confidence(postprocessed, answer)

        latency = (time.perf_counter() - start) * 1000

        return QueryResult(
            answer=answer,
            sources=[
                {
                    "doc_id": n.node.metadata.get("doc_id", ""),
                    "title": n.node.metadata.get("file_name", n.node.metadata.get("title", "")),
                    "score": round(n.score, 4),
                    "snippet": n.node.text[:200],
                }
                for n in postprocessed[:3]
            ],
            citations=citations,
            confidence=confidence,
            latency_ms=round(latency, 1),
            nodes_used=len(postprocessed),
            tokens_used=response.metadata.get("total_tokens", 0) if hasattr(response, 'metadata') else 0,
            retrieval_details={
                "query_type": query_type.value,
                "raw_retrieved": len(raw_nodes),
                "after_permission_filter": len(filtered_nodes),
                "after_rerank": len(reranked_nodes),
                "after_postprocess": len(postprocessed),
            },
        )

    def _enrich_query(
        self, question: str, chat_history: list[dict], user_context: dict
    ) -> str:
        """丰富查询上下文"""
        parts = []

        if chat_history and len(chat_history) > 0:
            context_str = "\n".join([
                f"Q: {h['question']}\nA: {h['answer']}"
                for h in chat_history[-4:]  # 只取最近4轮
            ])
            parts.append(f"[对话历史]\n{context_str}")

        if user_context:
            dept = user_context.get("department", "")
            role = user_context.get("role", "")
            if dept or role:
                parts.append(f"[用户背景] 部门={dept}, 角色={role}")

        parts.append(f"[当前问题] {question}")
        return "\n\n".join(parts)

    def _filter_by_permission(
        self, nodes: list[NodeWithScore], allowed_doc_ids: list[str]
    ) -> list[NodeWithScore]:
        """基于文档 ID 过滤无权访问的节点"""
        allowed_set = set(allowed_doc_ids)
        filtered = [
            n for n in nodes
            if n.node.metadata.get("doc_id", "") in allowed_set
            or n.node.metadata.get("source_doc_id", "") in allowed_set
        ]

        if len(filtered) < len(nodes):
            dropped = len(nodes) - len(filtered)
            logger.debug(f"权限过滤: 移除了 {dropped} 个无权访问的节点")

        return filtered

    async def _rerank(
        self, nodes: list[NodeWithScore], query: str
    ) -> list[NodeWithScore]:
        """使用 Cross-Encoder 进行重排序"""
        try:
            from llama_index.postprocessor.cohere_rerank import CohereRerank

            reranker = CohereRerank(
                api_key=self.config.get("cohere_api_key"),
                top_n=self.config.rerank_top_k,
                model=self.config.reranker_model,
            )
            reranked = reranker.postprocess_nodes(nodes, query_str=query)
            return reranked
        except ImportError:
            logger.warning("Cohere Rerank 不可用，使用分数截断作为替代")
            return sorted(nodes, key=lambda x: x.score, reverse=True)[
                : self.config.rerank_top_k
            ]
        except Exception as e:
            logger.error(f"重排序失败: {e}，回退到原始排序")
            return nodes[: self.config.rerank_top_k]

    def _postprocess(self, nodes: list[NodeWithScore]) -> list[NodeWithScore]:
        """后处理：相似度过滤 + Lost in the Middle 重排序"""
        if not nodes:
            return nodes

        # 相似度阈值过滤
        above_threshold = [
            n for n in nodes if n.score >= self.config.similarity_cutoff
        ]
        if len(above_threshold) < len(nodes):
            logger.debug(
                f"相似度过滤: {len(nodes)} → {len(above_threshold)}"
                f" (阈值={self.config.similarity_cutoff})"
            )

        # Lost in the Middle: 把最好的节点放到首尾
        if len(above_threshold) >= 3:
            sorted_by_score = sorted(
                above_threshold, key=lambda x: x.score, reverse=True
            )
            reordered = [sorted_by_score[0]]
            mid_start = 1
            mid_end = len(sorted_by_score)
            step = 2
            left, right = mid_start, mid_end - 1
            while left < right:
                reordered.append(sorted_by_score[right])
                if left < right:
                    reordered.append(sorted_by_score[left])
                left += step
                right -= step
            if left == right:
                reordered.append(sorted_by_score[left])
            return reordered

        return above_threshold

    def _compute_confidence(
        self, nodes: list[NodeWithScore], answer: str
    ) -> float:
        """计算回答置信度"""
        if not nodes:
            return 0.0

        avg_score = sum(n.score for n in nodes) / len(nodes)
        max_score = max(n.score for n in nodes)

        avg_weight = 0.6
        max_weight = 0.4
        confidence = avg_score * avg_weight + max_score * max_weight

        if len(answer) < 20:
            confidence *= 0.7  # 太短的回答可能不够充分

        return round(min(confidence, 1.0), 3)

    def incremental_index(self, source_id: int, changed_docs: list[str]) -> dict:
        """增量更新索引"""
        if not self._index:
            return {"status": "error", "message": "索引未初始化"}

        inserted = 0
        deleted = 0

        for doc_ref in changed_docs:
            try:
                new_docs = DataLoaderHub.load_single(doc_ref, source_id)
                for doc in new_docs:
                    self._index.insert(doc)
                    inserted += 1
            except Exception as e:
                logger.warning(f"增量插入失败 {doc_ref}: {e}")

        return {
            "status": "success",
            "inserted": inserted,
            "deleted": deleted,
        }

    def get_stats(self) -> dict:
        """获取引擎统计信息"""
        if not self._index:
            return {"status": "not_initialized"}

        try:
            docstore = self._index.docstore
            index_store = self._index.index_store
            return {
                "status": "ready",
                "total_documents": len(docstore.docs),
                "total_nodes": len(docstore.docs),
                "index_count": len(index_store.index_structs),
                "config": {
                    "embedding_model": self.config.embedding_model,
                    "llm_model": self.config.llm_model,
                    "chunk_size": self.config.chunk_size,
                    "retrieval_mode": self.config.retrieval_mode.value,
                    "use_reranker": self.config.use_reranker,
                },
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}
```

`EnterpriseRAGEngine` 的 `query()` 方法就是上一节架构图中那个 11 步流水线的具体实现。有几个值得深入讨论的实现细节：

### 权限过滤的位置选择

你可能会问：为什么权限过滤放在检索之后而不是之前？能不能在 Qdrant 查询时就通过 filter 参数直接排除掉不允许的文档？答案是：两种方式各有优劣，我们在实现中选择了"先粗检后细滤"的两阶段策略。

如果在 Qdrant 查询时就做严格的权限过滤（把 `allowed_doc_ids` 作为 filter 条件传入），好处是减少网络传输的数据量——Qdrant 只返回允许访问的节点。但问题是：当用户的权限范围很大（比如 admin 可以看所有文档，allowed_doc_ids 可能有几千个），把这个列表传给 Qdrant 的 filter 会显著降低查询性能。而且 Qdrant 的 filter 不支持超大规模的 IN 列表（通常建议控制在 100 以内）。

所以我们采用的策略是：**检索时不做过滤（或少做粗粒度过滤），拿到候选集后在应用层做精确的权限过滤**。对于普通用户（允许访问的文档较少），可以在检索时加上 filter；对于管理员（允许访问全部文档），直接跳过过滤步骤。这个判断逻辑可以进一步优化：

```python
def _smart_permission_filter(
    self, nodes: list[NodeWithScore], allowed_doc_ids: list[str]
) -> list[NodeWithScore]:
    """智能权限过滤：根据允许列表大小选择过滤策略"""
    FILTER_THRESHOLD = 100  # 允许列表超过此值则延迟过滤

    if len(allowed_doc_ids) <= FILTER_THRESHOLD:
        # 小范围权限：在检索时就过滤（更高效）
        return self._filter_by_permission(nodes, allowed_doc_ids)
    elif len(allowed_doc_ids) > len(nodes) * 10:
        # 超大范围权限（如 admin）：可能无需过滤
        return nodes
    else:
        # 中等范围：应用层过滤
        return self._filter_by_permission(nodes, allowed_doc_ids)
```

### 置信度计算的启发式方法

`_compute_confidence()` 方法中的置信度计算是一个简单的启发式公式：`0.6 × 平均分 + 0.4 × 最高分`。为什么这样设计？因为平均分反映了整体检索质量（如果大部分节点的相关度都不高，说明系统对这个问题的理解可能有问题），而最高分代表了最好的一条证据有多强（即使平均分一般，只要有一条非常匹配的证据，回答仍然可能是好的）。0.6 和 0.4 的权重分配是基于经验调优的结果——你可以根据自己的数据和反馈调整这些参数。

另外注意那个"短回答惩罚"：如果最终生成的回答少于 20 个字符，置信度会被乘以 0.7。这是因为太短的回答（如"是的"、"不清楚"、"请参考手册"）往往意味着检索没有找到足够的信息，或者模型选择了保守的回避策略。

## 混合检索器：HybridRetriever

第五章中我们详细学习了 Hybrid Search 的原理。下面是在项目中实际使用的混合检索器实现：

```python
from typing import List, Optional
from llama_index.core import VectorStoreIndex
from llama_core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.core.node_parser import SentenceSplitter
import math
import re
import jieba  # 中文分词


class BM25Retriever(BaseRetriever):
    """BM25 关键词检索器（内存索引实现）"""

    def __init__(
        self,
        nodes: List[TextNode],
        similarity_top_k: int = 20,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self._nodes = nodes
        self._similarity_top_k = similarity_top_k
        self.k1 = k1
        self.b = b
        self._index = self._build_bm25_index()

    def _build_bm25_index(self) -> dict:
        """构建 BM25 倒排索引"""
        from collections import defaultdict
        import math

        df = defaultdict(int)  # document frequency
        tf_list = []  # term frequency per document
        doc_lengths = []

        for node in self._nodes:
            tokens = self._tokenize(node.text)
            tf = defaultdict(int)
            for token in tokens:
                tf[token] += 1
            tf_list.append(dict(tf))
            doc_lengths.append(len(tokens))

            for token in set(tokens):
                df[token] += 1

        avg_dl = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 1
        N = len(self._nodes)

        # 预计算 IDF
        idf = {}
        for term, freq in df.items():
            idf[term] = math.log((N - freq + 0.5) / (freq + 0.5) + 1)

        return {
            "tf_list": tf_list,
            "df": dict(df),
            "idf": idf,
            "avg_dl": avg_dl,
            "doc_lengths": doc_lengths,
            "N": N,
        }

    def _tokenize(self, text: str) -> list[str]:
        """中文分词 + 英文小写化"""
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', text)
        chinese_tokens = list(jieba.cut_for_search(text))
        english_tokens = re.findall(r'[a-zA-Z]+', text)
        tokens = [t.lower() for t in chinese_tokens + english_tokens if len(t.strip()) > 0]
        return tokens

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        query_text = query.query_str if hasattr(query_bundle, 'query_str') else str(query_bundle)
        query_tokens = self._tokenize(query_text)
        idx = self._index

        scores = []
        for i, node in enumerate(self._nodes):
            score = self._bm25_score(query_tokens, idx["tf_list"][i], idx["doc_lengths"][i], idx)
            scores.append((score, node))

        scores.sort(key=lambda x: x[0], reverse=True)
        return [
            NodeWithScore(node=node, score=score)
            for score, node in scores[: self._similarity_top_k]
        ]

    def _bm25_score(self, query_tokens: list, tf: dict, dl: int, idx: dict) -> float:
        """计算单个文档的 BM25 分数"""
        score = 0.0
        for token in query_tokens:
            if token not in idx["idf"]:
                continue
            freq = tf.get(token, 0)
            if freq == 0:
                continue
            idf = idx["idf"][token]
            tf_component = (freq * (self.k1 + 1)) / (freq + self.k1 * (1 - self.b + self.b * dl / idx["avg_dl"]))
            score += idf * tf_component
        return score

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        query_str = query_bundle.query_str
        query_tokens = self._tokenize(query_str)
        idx = self._index

        scores = []
        for i, node in enumerate(self._nodes):
            score = self._bm25_score(query_tokens, idx["tf_list"][i], idx["doc_lengths"][i], idx)
            scores.append((score, node))

        scores.sort(key=lambda x: x[0], reverse=True)
        return [
            NodeWithScore(node=node, score=score)
            for score, node in scores[: self._similarity_top_k]
        ]


class HybridRetriever(BaseRetriever):
    """混合检索器：融合向量搜索和 BM25 关键词搜索"""

    def __init__(
        self,
        index: VectorStoreIndex,
        similarity_top_k: int = 15,
        sparse_top_k: int = 20,
        rrf_k: int = 60,  # RRF 参数
    ):
        self._vector_retriever = index.as_retriever(similarity_top_k=similarity_top_k)
        self._sparse_top_k = sparse_top_k
        self._rrf_k = rrf_k
        self._bm25_retriever: Optional[BM25Retriever] = None
        self._all_nodes: Optional[List[TextNode]] = None

    def _ensure_bm25_ready(self):
        """懒初始化 BM25 索引（只在第一次查询时构建）"""
        if self._bm25_retriever is None:
            all_nodes = list(self._vector_retriever._index.docstore.docs.values())
            self._all_nodes = all_nodes
            self._bm25_retriever = BM25Retriever(
                nodes=all_nodes,
                similarity_top_k=self._sparse_top_k,
            )
            logger.info(f"BM25 索引构建完成, 共 {len(all_nodes)} 个节点")

    def _rrf_fusion(
        self,
        vector_results: List[NodeWithScore],
        bm25_results: List[NodeWithScore],
    ) -> List[NodeWithScore]:
        """
        Reciprocal Rank Fusion 融合算法

        公式: RRF(d) = Σ 1/(k + rank_i(d))
        其中 k 是平滑常数（默认60），rank_i 是文档在第 i 个结果列表中的排名
        """
        rrf_scores: dict[str, float] = {}
        node_map: dict[str, NodeWithScore] = {}

        for rank, result in enumerate(vector_results):
            node_id = result.node.node_id
            if node_id not in rrf_scores:
                rrf_scores[node_id] = 0
                node_map[node_id] = result
            rrf_scores[node_id] += 1.0 / (self._rrf_k + rank + 1)

        for rank, result in enumerate(bm25_results):
            node_id = result.node.node_id
            if node_id not in rrf_scores:
                rrf_scores[node_id] = 0
                node_map[node_id] = result
            rrf_scores[node_id] += 1.0 / (self._rrf_k + rank + 1)

        fused = sorted(
            [(node_id, score) for node_id, score in rrf_scores.items()],
            key=lambda x: x[1],
            reverse=True,
        )

        return [
            NodeWithScore(node=node_map[node_id].node, score=score)
            for node_id, score in fused[: self._vector_retriever._similarity_top_k]
        ]

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        self._ensure_bm25_ready()

        vector_results = self._vector_retriever.retrieve(query_bundle)
        bm25_results = self._bm25_retriever.retrieve(query_bundle)

        logger.debug(
            f"向量检索返回 {len(vector_results)} 结果, "
            f"BM25检索返回 {len(bm25_results)} 结果"
        )

        fused = self._rrf_fusion(vector_results, bm25_results)
        logger.debug(f"RRF融合后返回 {len(fused)} 结果")

        return fused
```

关于这个混合检索器的实现，有几个工程上的要点：

**BM25 的懒初始化（Lazy Initialization）**：BM25 索引需要遍历所有节点来构建倒排索引，对于几千个节点来说这个过程需要几秒钟。如果我们把它放在引擎启动时同步执行，会导致服务启动变慢。所以采用懒初始化策略——只在第一次查询请求到达时才构建 BM25 索引，后续请求直接复用。代价是第一个用户会多等几秒，但可以通过预热（warm-up）机制来解决——服务启动后自动发一个 dummy query 来触发初始化。

**jieba 中文分词的使用**：BM25 的效果高度依赖分词质量。对于英文来说空格分词就够了，但中文没有天然的词边界。`jieba.cut_for_search()` 是搜索引擎模式的分词，它会生成较细粒度的词元（比如"知识库问答"会被切分为"知识"/"库"/"问答"三个词），这对提高召回率有帮助。如果你对分词精度有更高要求，可以考虑用 pkuseg 或 LAC 这些更先进的分词器。

**RRF 参数 k=60 的选择**：这个值来自原始 RRF 论文的推荐范围（通常在 20-100 之间，60 是经验最优值）。k 越小，排名靠前的结果权重越大（更激进）；k 越大，不同排名之间的差异越小（更保守）。在我们的场景中 60 效果不错——既不会让某一个检索方法完全主导结果，也不会过度稀释高质量结果的贡献。

## 引用构建器：CitationBuilder

引用溯源是企业知识库区别于通用 AI 助手的核心差异化功能之一：

```python
import re
from dataclasses import dataclass
from typing import Optional
from llama_index.core.schema import NodeWithScore


@dataclass
class Citation:
    """单条引用"""
    source_title: str
    source_type: str  # pdf/docx/md/database
    section: Optional[str] = None
    snippet: Optional[str] = None
    relevance_score: float = 0.0
    url: Optional[str] = None  # 原文链接（如果有）


class CitationBuilder:
    """引用构建器——从回答和检索节点中提取格式化的引用"""

    def __init__(self, max_citations: int = 5, min_snippet_length: int = 50, max_snippet_length: int = 150):
        self.max_citations = max_citations
        self.min_snippet_length = min_snippet_length
        self.max_snippet_length = max_snippet_length

    def build(
        self,
        answer: str,
        nodes: list[NodeWithScore],
    ) -> list[dict]:
        """
        从回答文本和检索节点中构建引用列表

        核心思路：
        1. 提取回答中的关键陈述句
        2. 将每个陈述映射到最相关的源节点
        3. 格式化为标准引用格式
        """
        if not nodes:
            return []

        citations = []
        used_node_ids = set()

        # 方法一：按节点相关性取 top-N
        sorted_nodes = sorted(nodes, key=lambda x: x.score, reverse=True)

        for node in sorted_nodes[: self.max_citations]:
            if node.node.node_id in used_node_ids:
                continue

            citation = self._format_citation(node)
            citations.append(citation)
            used_node_ids.add(node.node.node_id)

        return citations

    def _format_citation(self, node: NodeWithScore) -> dict:
        """将单个节点格式化为引用条目"""
        meta = node.node.metadata

        title = (
            meta.get("title")
            or meta.get("file_name")
            or meta.get("db_title")
            or "未知来源"
        )

        source_type = self._detect_source_type(meta)

        section = meta.get("header_hierarchy") or meta.get("page_label") or None

        snippet = self._extract_snippet(node.node.text)

        return {
            "id": f"cite-{len(citations) + 1}" if 'citations' in dir() else f"cite-{id(node) % 1000}",
            "title": title,
            "source_type": source_type,
            "section": section,
            "snippet": snippet,
            "relevance_score": round(node.score, 3),
            "doc_id": meta.get("doc_id", "") or meta.get("source_doc_id", ""),
        }

    def _detect_source_type(self, metadata: dict) -> str:
        """根据元数据推断文档类型"""
        file_name = metadata.get("file_name", "")
        if file_name.endswith(".pdf"):
            return "pdf"
        elif file_name.endswith(".docx"):
            return "word"
        elif file_name.endswith(".md") or file_name.endswith(".markdown"):
            return "markdown"
        elif "db_" in str(metadata.keys()):
            return "database"
        elif file_name.endswith(".html"):
            return "webpage"
        else:
            return "unknown"

    def _extract_snippet(self, text: str) -> str:
        """提取用于展示的原文片段"""
        if len(text) <= self.max_snippet_length:
            return text.strip()

        # 尝试在第一句话处截断
        sentences = re.split(r'(?<=[。！？.!?\n])', text)
        snippet = ""
        for sent in sentences:
            if len(snippet) + len(sent) > self.max_snippet_length:
                break
            snippet += sent

        if len(snippet) < self.min_snippet_length:
            snippet = text[: self.max_snippet_length]

        if len(text) > len(snippet):
            snippet += "..."

        return snippet.strip()
```

引用构建看起来简单，但在实际项目中有很多细节需要注意：

**snippet 截断策略**：上面的 `_extract_snippet` 方法优先按句子边界截断，而不是简单地按字符数硬切。这是因为按句子截断能保持语义完整性——用户看到的引用片段是一段有意义的话，而不是半截话。正则表达式 `(?<=[。！？.!?\n])` 使用了 lookbehind 来匹配中文和英文的句末标点，确保断句位置自然。

**去重处理**：同一个文档可能被切成多个 chunk，检索时可能有多个来自同一文档的节点都进入了 top-k。如果不做去重，引用列表可能会出现"《员工手册》第 3 章"出现三次的情况。`used_node_ids` 集合用来追踪已经引用过的节点，避免重复。更进一步的做法是按文档聚合——同一文档只保留得分最高的那条引用。

## 查询分类器：QueryClassifier

```python
from enum import Enum
from dataclasses import dataclass
from typing import Optional
import re


class QuestionType(Enum):
    FACTUAL = "factual"           # 事实性："公司成立于哪一年？"
    PROCEDURAL = "procedural"     # 流程性："如何申请报销？"
    COMPARATIVE = "comparative"   # 对比性："Pro 版和企业版的区别？"
    TROUBLESHOOTING = "troubleshooting"  # 排查类："API 返回 403 怎么办？"
    DEFINITIONAL = "definitional" # 定义性："什么是 RBAC？"
    OPINION = "opinion"           # 观点性："你觉得哪个方案更好？"
    AMBIGUOUS = "ambiguous"       # 歧义性："我们的产品怎么样？"


@dataclass
class ClassificationResult:
    question_type: QuestionType
    confidence: float
    clarification_needed: bool
    suggested_clarification: Optional[str] = None
    keywords_extracted: list[str] = None


class QueryClassifier:
    """基于规则 + 启发的查询分类器"""

    PROCEDURAL_PATTERNS = [
        r"如何.*?[办做申请操作流程步骤]",
        r".*?怎么.*?",
        r".*?如何.*?",
        r".*?怎样.*?",
        r".*?步骤",
        r".*?流程",
        r".*?指南",
        r".*?教程",
        r".*?方法",
    ]

    COMPARATIVE_PATTERNS = [
        r".*?和.*?的区别",
        r".*?与.*?对比",
        r".*?vs\.?",
        r".*?比较.*?",
        r".*?差异",
        r".*?哪个好",
        r".*?选哪个",
        r".*?有什么不同",
    ]

    TROUBLESHOOTING_PATTERNS = [
        r".*?报错",
        r".*?错误",
        r".*?失败",
        r".*?异常",
        r".*?不行",
        r".*?解决.*?问题",
        r".*?修复",
        r".*?bug",
        r"\d{3}\s*(错误|error)",
        r".*?40[0-9]\b",
        r".*?50[0-9]\b",
    ]

    FACTUAL_PATTERNS = [
        r".*?是什么",
        r".*?是多少",
        r".*?有没有",
        r".*?是否",
        r".*?哪些",
        r".*?谁",
        r".*?哪里",
        r".*?什么时候",
        r".*?几[个天月年]",
        r".*?[是否是不是].*?[吗呢]",
    ]

    OPINION_PATTERNS = [
        r".*?觉得.*?好",
        r".*?推荐.*?",
        r".*?应该.*?选",
        r".*?值得.*?",
        r".*?评价.*?",
        r".*?看法",
        r".*?建议.*?",
    ]

    def classify(self, question: str) -> ClassificationResult:
        """
        分类用户问题

        Returns:
            ClassificationResult 包含问题类型、置信度和是否需要澄清
        """
        question = question.strip()

        if len(question) < 3:
            return ClassificationResult(
                question_type=QuestionType.AMBIGUOUS,
                confidence=0.3,
                clarification_needed=True,
                suggested_clarification="您的问题太简短了，能否详细描述一下您想了解什么？",
            )

        scores = {
            QuestionType.PROCEDURAL: self._match_score(
                question, self.PROCEDURAL_PATTERNS
            ),
            QuestionType.COMPARATIVE: self._match_score(
                question, self.COMPARATIVE_PATTERNS
            ),
            QuestionType.TROUBLESHOOTING: self._match_score(
                question, self.TROUBLESHOOTING_PATTERNS
            ),
            QuestionType.FACTUAL: self._match_score(
                question, self.FACTUAL_PATTERNS
            ),
            QuestionType.OPINION: self._match_score(
                question, self.OPINION_PATTERNS
            ),
        }

        best_type = max(scores, key=scores.get)
        best_score = scores[best_type]

        if best_score < 0.3:
            best_type = QuestionType.AMBIGUOUS
            best_score = 0.2

        clarification = self._check_need_clarification(question, best_type)

        keywords = self._extract_keywords(question)

        return ClassificationResult(
            question_type=best_type,
            confidence=round(best_score, 3),
            clarification_needed=clarification["needed"],
            suggested_clarification=clarification["suggestion"],
            keywords_extracted=keywords,
        )

    def _match_score(self, text: str, patterns: list[str]) -> float:
        """计算文本与一组正则模式的匹配分数"""
        matches = 0
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                matches += 1
        return matches / len(patterns) if patterns else 0

    def _check_need_clarification(
        self, question: str, qtype: QuestionType
    ) -> dict:
        """检查问题是否过于模糊需要澄清"""
        ambiguous_indicators = [
            len(question) < 8,
            "什么" in question and len(question) < 15,
            "怎么样" in question and len(question) < 15,
            qtype == QuestionType.AMBIGUOUS,
        ]

        needed = any(ambiguous_indicators)
        suggestion = None

        if needed:
            if "产品" in question or "功能" in question:
                suggestion = (
                    "您想了解哪个产品的信息？或者您可以描述一下"
                    "具体的业务场景，我会为您提供更精准的回答。"
                )
            elif "政策" in question or "规定" in question:
                suggestion = (
                    "请问您想了解哪方面的制度规范？"
                    "比如人事、财务、IT 还是其他领域？"
                )
            else:
                suggestion = (
                    "您的问题比较宽泛，能否提供更多细节？"
                    "例如涉及的具体产品、部门或场景。"
                )

        return {"needed": needed, "suggestion": suggestion}

    def _extract_keywords(self, question: str) -> list[str]:
        """提取问题中的关键词（用于日志分析和后续优化）"""
        import jieba
        import jieba.posseg as pseg

        words = pseg.cut(question)
        keywords = []
        stop_words = {
            "的", "了", "是", "在", "我", "有", "和", "就",
            "不", "人", "都", "一", "一个", "上", "也", "很",
            "到", "说", "要", "去", "你", "会", "着", "没有",
            "什么", "怎么", "如何", "哪", "哪些", "这个", "那个",
            "吗", "呢", "吧", "啊", "呀", "哦", "嗯",
        }

        for word, flag in words:
            if len(word) >= 2 and word not in stop_words:
                if flag.startswith(("n", "v", "nr", "nz")):
                    keywords.append(word)

        return keywords[:10]
```

这个查询分类器采用的是**规则匹配**而非 LLM 调用的方式。原因是：分类操作会在每次查询时执行，如果用 LLM 来分类会增加一次 API 调用和约 200-500ms 的额外延迟。对于企业内部的知识库问答，问题的类型分布相对固定且可预测（事实性问题最多约占 40%，流程性约 25%，其他类型合计约 35%），规则方法的准确率可以达到 85% 以上，足以满足需求。

当然，如果你的预算充足且对分类准确率有更高要求（>95%），可以用 LLM 替换规则方法。LlamaIndex 本身也提供了 `LLMPredictor` 可以方便地封装分类逻辑。一个折中方案是：规则方法作为快速路径（覆盖 85% 的常见情况），对于规则无法确定的问题（confidence < 0.5）再 fallback 到 LLM 分类。

## 总结

