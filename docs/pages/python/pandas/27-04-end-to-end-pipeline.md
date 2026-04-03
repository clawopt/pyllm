# 构建端到端的自动化流水线


#### 流水线架构总览

将前面三节的所有组件串联成一个完整的 **RAG 知识库构建流水线**：

```
┌─────────────┐    ┌──────────────┐    ┌───────────────┐
│ 文档扫描发现 │ →  │ 解析+元数据   │ →  │ 分块+质量评分  │
│ DocumentScan│    │ Extractor    │    │ Chunker       │
└─────────────┘    └──────────────┘    └───────────────┘
                                               ↓
┌─────────────┐    ┌──────────────┐    ┌───────────────┐
│ 导出+报告    │ ←  │ 向量化准备    │ ←  │ 过滤+优化      │
│ Exporter    │    │ VectorPrep   │    │ Filter        │
└─────────────┘    └──────────────┘    └───────────────┘
```

---

#### 核心流水线类

```python
import pandas as pd
import json
from pathlib import Path
from datetime import datetime


class RAGKnowledgePipeline:
    def __init__(self, knowledge_dir: str, output_dir: str = "./rag_output/"):
        self.knowledge_dir = Path(knowledge_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.scanner = DocumentScanner(str(self.knowledge_dir))
        self.pdf_parser = PDFParser()
        self.md_parser = MarkdownParser()
        self.meta_extractor = MetadataExtractor()
        self.chunker = RecursiveCharacterChunker(
            chunk_size=500,
            min_chunk_size=30,
        )

        self.metrics = {}
        self.stage_data = {}

    def run(self, config: dict = None) -> pd.DataFrame:
        cfg = config or {
            "min_quality_score": 30,
            "chunk_size": 500,
            "export_format": ["parquet", "jsonl"],
            "generate_report": True,
        }

        print("=" * 65)
        print("  RAG 知识库构建流水线")
        print(f"  输入目录: {self.knowledge_dir}")
        print(f"  输出目录: {self.output_dir}")
        print("=" * 65)

        self._stage1_scan()
        self._stage2_parse()
        self._stage3_metadata()
        self._stage4_chunking(cfg["min_quality_score"], cfg["chunk_size"])
        self._stage5_export(cfg["export_format"])

        if cfg.get("generate_report"):
            self._generate_report()

        final = self.stage_data.get("final_chunks")
        if final is not None:
            print(f"\n{'='*65}")
            print(f"  ✅ 流水线完成！最终输出: {len(final):,} 个高质量 chunk")
            print(f"{'='*65}")

        return final

    def _stage1_scan(self):
        print(f"\n{'─'*65}")
        print(f" 📂 阶段 1: 文档扫描")
        print(f"{'─'*65}")

        doc_index = self.scanner.scan(recursive=True)
        self.stage_data["doc_index"] = doc_index

        self.metrics["scan"] = {
            "total_files": len(doc_index),
            "total_size_mb": doc_index["size_bytes"].sum() / 1024 / 1024,
            "by_type": doc_index.groupby("extension").size().to_dict(),
        }

        return doc_index

    def _stage2_parse(self):
        print(f"\n{'─'*65}")
        print(f" 📄 阶段 2: 文档解析")
        print(f"{'─'*65}")

        doc_index = self.stage_data["doc_index"]

        pdf_result = self.pdf_parser.parse_batch(doc_index)
        md_result = self.md_parser.parse_batch(doc_index)

        other_mask = ~doc_index["extension"].isin([".pdf", ".md"])
        other_docs = doc_index[other_mask].copy()
        other_docs["text"] = ""
        other_docs["char_count"] = 0
        other_docs["page_count"] = 0
        other_docs["section_count"] = 0
        other_docs["success"] = True

        unified_cols = [
            "file_path", "full_path", "file_name", "extension",
            "mime_type", "size_bytes", "size_kb", "modified_time",
            "file_hash", "size_category", "text", "success",
            "char_count", "page_count", "section_count"
        ]

        all_parsed = pd.concat([
            pdf_result[unified_cols],
            md_result[unified_cols],
            other_docs[[c for c in unified_cols if c in other_docs.columns]],
        ], ignore_index=True)

        self.stage_data["parsed"] = all_parsed

        success_rate = all_parsed["success"].mean() * 100
        total_chars = all_parsed.loc[all_parsed["success"], "char_count"].sum()

        self.metrics["parse"] = {
            "success_rate": round(success_rate, 1),
            "total_chars": int(total_chars),
            "failed": int((~all_parsed["success"]).sum()),
        }

        print(f"  成功率: {success_rate:.1f}% | 总字符: {total_chars:,}")

    def _stage3_metadata(self):
        print(f"\n{'─'*65}")
        print(f" 🏷️  阶段 3: 元数据提取与标签")
        print(f"{'─'*65}")

        parsed = self.stage_data["parsed"]

        enriched = enrich_metadata(parsed, self.meta_extractor)
        enriched["tags"] = TagManager.batch_tag(enriched)
        enriched = fix_metadata_issues(enriched)

        self.stage_data["enriched"] = enriched

        self.metrics["metadata"] = {
            "after_fix": len(enriched),
            "with_tags": (enriched["tags"].apply(len) > 0).sum(),
            "tag_coverage": round(
                (enriched["tags"].apply(len) > 0).sum() / len(enriched) * 100, 1
            ),
        }

    def _stage4_chunking(self, min_quality: float, chunk_size: int):
        print(f"\n{'─'*65}")
        print(f" ✂️  阶段 4: 分块 + 质量评分")
        print(f"{'─'*65}")

        enriched = self.stage_data["enriched"]
        self.chunker.chunk_size = chunk_size

        chunks = self.chunker.chunk_dataframe(enriched)
        scored = score_chunks(chunks)
        filtered = prepare_for_vector_db(scored, min_quality=min_quality)

        self.stage_data["final_chunks"] = filtered

        self.metrics["chunking"] = {
            "total_chunks": len(chunks),
            "after_filter": len(filtered),
            "removed_low_quality": len(chunks) - len(filtered),
            "avg_score": round(filtered["quality_score"].mean(), 1),
            "avg_chars": round(filtered["char_count"].mean(), 0),
        }

    def _stage5_export(self, formats: list):
        print(f"\n{'─'*65}")
        print(f" 💾 阶段 5: 导出")
        print(f"{'─'*65}")

        final = self.stage_data["final_chunks"]
        if final is None or len(final) == 0:
            print("⚠️ 无数据可导出")
            return

        export_meta = {"chunk_id", "doc_id", "content", "source",
                       "title", "section_title", "chunk_index",
                       "total_chunks", "char_count", "token_estimate",
                       "tags", "language", "quality_score", "quality_grade"}

        available_export = [c for c in export_meta if c in final.columns]
        extra_cols = [c for c in final.columns if c not in export_meta]

        export_df = final[available_export + extra_cols].copy()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        exported_files = []

        if "parquet" in formats:
            parquet_path = self.output_dir / f"rag_chunks_{timestamp}.parquet"
            export_df.to_parquet(parquet_path, index=False)
            size_mb = parquet_path.stat().st_size / 1024 / 1024
            exported_files.append(f"{parquet_path} ({size_mb:.1f} MB)")
            print(f"  ✅ Parquet: {parquet_path.name} ({size_mb:.1f} MB)")

        if "jsonl" in formats:
            jsonl_path = self.output_dir / f"rag_chunks_{timestamp}.jsonl"

            non_serializable = []
            for col in export_df.columns:
                sample = export_df[col].iloc[0] if len(export_df) > 0 else None
                try:
                    json.dumps(sample, ensure_ascii=False, default=str)
                except (TypeError, ValueError):
                    non_serializable.append(col)

            safe_df = export_df.drop(columns=non_serializable, errors="ignore")

            with open(jsonl_path, "w", encoding="utf-8") as f:
                for _, row in safe_df.iterrows():
                    record = row.to_dict()
                    record = {k: (v if not pd.isna(v) else None)
                              for k, v in record.items()}
                    f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

            size_mb = jsonl_path.stat().st_size / 1024 / 1024
            exported_files.append(f"{jsonl_path} ({size_mb:.1f} MB)")
            print(f"  ✅ JSONL: {jsonl_path.name} ({size_mb:.1f} MB)")

        self.metrics["export"] = {
            "files": exported_files,
            "formats": formats,
            "record_count": len(export_df),
        }

    def _generate_report(self):
        report_lines = []
        report_lines.append("# RAG 知识库构建报告\n")
        report_lines.append(f"- 构建时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"- 输入目录: `{self.knowledge_dir}`\n")

        scan = self.metrics.get("scan", {})
        parse = self.metrics.get("parse", {})
        meta = self.metrics.get("metadata", {})
        chunk = self.metrics.get("chunking", {})
        export = self.metrics.get("export", {})

        report_lines.append("## 各阶段统计\n")
        report_lines.append("| 阶段 | 指标 | 数值 |")
        report_lines.append("|------|------|------|")
        report_lines.append(f"| 文件扫描 | 发现文件 | {scan.get('total_files', 0):,} |")
        report_lines.append(f"| 文件扫描 | 总大小 | {scan.get('total_size_mb', 0):.1f} MB |")
        report_lines.append(f"| 文档解析 | 成功率 | {parse.get('success_rate', 0):.1f}% |")
        report_lines.append(f"| 文档解析 | 总字符数 | {parse.get('total_chars', 0):,} |")
        report_lines.append(f"| 元数据丰富 | 处理后文档 | {meta.get('after_fix', 0):,} |")
        report_lines.append(f"| 元数据丰富 | 标签覆盖率 | {meta.get('tag_coverage', 0):.1f}% |")
        report_lines.append(f"| 分块 | 原始 chunk | {chunk.get('total_chunks', 0):,} |")
        report_lines.append(f"| 分块 | 过滤后 chunk | {chunk.get('after_filter', 0):,} |")
        report_lines.append(f"| 分块 | 平均质量分 | {chunk.get('avg_score', 0)} |")
        report_lines.append(f"| 导出 | 记录数 | {export.get('record_count', 0):,} |")

        report_text = "\n".join(report_lines)

        report_path = self.output_dir / "build_report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text)

        print(f"\n📋 报告已生成: {report_path}")
        print(report_text)
```

---

#### 运行完整流水线

```python
if __name__ == "__main__":
    pipeline = RAGKnowledgePipeline(
        knowledge_dir="./knowledge_base/",
        output_dir="./rag_output/",
    )

    result = pipeline.run(config={
        "min_quality_score": 30,
        "chunk_size": 500,
        "export_format": ["parquet", "jsonl"],
        "generate_report": True,
    })
```

输出：
```
=================================================================
  RAG 知识库构建流水线
  输入目录: knowledge_base/
  输出目录: rag_output/
=================================================================

─────────────────────────────────────────────────────────────────
 📂 阶段 1: 文档扫描
─────────────────────────────────────────────────────────────────
=== 文档扫描结果 ===

目录: knowledge_base/
总文件数: 1,234
总大小: 456.7 MB
...

─────────────────────────────────────────────────────────────────
 📄 阶段 2: 文档解析
─────────────────────────────────────────────────────────────────
=== PDF 解析结果 ===
成功: 318 | 失败: 3
=== Markdown 解析结果 ===
文件数: 567 | 章节数: 8,901
  成功率: 99.8% | 总字符: 67,890,123

─────────────────────────────────────────────────────────────────
 🏷️  阶段 3: 元数据提取与标签
─────────────────────────────────────────────────────────────────
=== 元数据丰富完成 ===
处理文件数: 1180
语言分布: zh=678, en=456
标题提取率: 80.0%
标签分布: pandas=234, python=198, ...

─────────────────────────────────────────────────────────────────
 ✂️  阶段 4: 分块 + 质量评分
─────────────────────────────────────────────────────────────────
=== 递归字符分块结果 ===
处理文档: 1,180
生成 chunk: 142,567
=== Chunk 质量评分 ===
A-优秀: 28,864 | B-良好: 67,890 | C-合格: 34,567 | ...
平均分: 62.3
=== 向量化准备完成 ===
最终 chunk 数: 131,321

─────────────────────────────────────────────────────────────────
 💾 阶段 5: 导出
─────────────────────────────────────────────────────────────────
  ✅ Parquet: rag_chunks_20260103_143000.parquet (178.5 MB)
  ✅ JSONL: rag_chunks_20260103_143000.jsonl (312.3 MB)

📋 报告已生成: rag_output/build_report.md

=================================================================
  ✅ 流水线完成！最终输出: 131,321 个高质量 chunk
=================================================================
```

---

#### 与向量数据库的对接

Pandas 准备好的数据可以直接批量导入主流向量数据库：

##### ChromaDB 批量导入

```python
def import_to_chroma(chunk_df: pd.DataFrame, collection_name: str = "knowledge_base"):
    import chromadb
    from chromadb.config import Settings

    client = chromadb.Client(Settings(
        chroma_db_impl="duckdb+parquet:",
        persist_directory="./chroma_db/"
    ))

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    batch_size = 1000
    total = len(chunk_df)

    for i in range(0, total, batch_size):
        batch = chunk_df.iloc[i:i + batch_size]

        ids = batch["chunk_id"].tolist()
        documents = batch["content"].tolist()
        metadatas = batch[[
            c for c in ["doc_id","source","title","section_title",
                        "tags","language","quality_score"]
            if c in batch.columns
        ]].to_dict("records")

        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

        if (i // batch_size + 1) % 10 == 0 or i + batch_size >= total:
            print(f"  已导入: {min(i + batch_size, total)}/{total}")

    print(f"\n✅ ChromaDB 导入完成！集合: {collection_name}")
    print(f"   总文档数: {collection.count()}")

    return collection
```

##### LanceDB 批量导入（推荐用于生产环境）

```python
def import_to_lancedb(chunk_df: pd.DataFrame, db_path: str = "./lancedb_kb"):
    import lancedb

    db = lancedb.connect(db_path)

    schema_fields = [
        lancedb.vector(len(chunk_df.iloc[0].get("embedding", [0]*768])),
        lancedb.field("chunk_id", lancedb.str),
        lancedb.field("doc_id", lancedb.str),
        lancedb.field("content", lancedb.str),
        lancedb.field("source", lancedb.str),
        lancedb.field("title", lancedb.str),
        lancedb.field("quality_score", lancedb.float32),
    ]

    tbl = db.create_table("chunks", schema=schema_fields, exist_ok=True)

    has_embedding = "embedding" in chunk_df.columns
    if not has_embedding:
        print("⚠️ 尚无 embedding 列，先创建占位符")
        chunk_df = chunk_df.copy()
        chunk_df["embedding"] = [None] * len(chunk_df)

    records = chunk_df[["chunk_id","doc_id","content","source",
                         "title","quality_score","embedding"]].to_dict("records")

    valid_records = [r for r in records if r.get("embedding") is not None]

    if valid_records:
        tbl.add(valid_records)
        print(f"✅ LanceDB 导入完成: {len(valid_records)} 条 (有 embedding)")
    else:
        print(f"ℹ️ 表结构已创建，等待后续 embedding 填充")
        print(f"   表路径: {db_path}/chunks")

    return tbl
```

---

#### 流水线的增量更新模式

对于持续运营的知识库，需要支持增量更新而非每次全量重建：

```python
class IncrementalRAGPipeline(RAGKnowledgePipeline):
    def run_incremental(self, existing_db_path: str = None):
        existing_hashes = set()

        if existing_db_path and Path(existing_db_path).exists():
            existing = pd.read_parquet(existing_db_path)
            existing_hashes = set(existing.get("file_hash", []))

        doc_index = self.scanner.scan()

        new_docs = doc_index[~doc_index["file_hash"].isin(existing_hashes)]

        if len(new_docs) == 0:
            print("✅ 无新增文档，无需更新")
            return None

        print(f"📥 发现 {len(new_docs)} 个新文档")

        new_parsed = (
            self.pdf_parser.parse_batch(new_docs)
            .pipe(lambda df: pd.concat([df, self.md_parser.parse_batch(new_docs)]))
        )

        new_enriched = enrich_metadata(new_parsed, self.meta_extractor)
        new_enriched["tags"] = TagManager.batch_tag(new_enriched)

        new_chunks = self.chunker.chunk_dataframe(new_enriched)
        new_scored = score_chunks(new_chunks)
        new_final = prepare_for_vector_db(new_scored)

        if existing_db_path and Path(existing_db_path).exists():
            merged = pd.concat([existing, new_final], ignore_index=True)
            merged = merged.drop_duplicates(subset=["content"], keep="last")
        else:
            merged = new_final

        output_path = self.output_dir / "rag_chunks_incremental.parquet"
        merged.to_parquet(output_path, index=False)

        print(f"\n✅ 增量更新完成:")
        print(f"   新增 chunk: {len(new_final):,}")
        print(f"   总计 chunk: {len(merged):,}")

        return merged
```

从原始文档到可检索的知识库，整个端到端流程只需要一个 `pipeline.run()` 调用。Pandas 在其中承担了数据编排、状态跟踪和质量控制的核心角色。
