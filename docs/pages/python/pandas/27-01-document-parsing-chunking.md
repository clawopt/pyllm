# 文档解析与分块


#### RAG 知识库的数据流

RAG（Retrieval-Augmented Generation）系统的核心是知识库，而知识库的质量取决于 **文档解析 → 分块 → 元数据管理 → 向量化** 这条流水线。Pandas 在其中扮演"数据编排者"的角色——管理文档元数据、跟踪分块状态、协调批量处理。

```
原始文档 (PDF/Markdown/HTML/DOCX)
    ↓ 文档解析器 (Unstructured / PyMuPDF / BeautifulSoup)
    ↓ 纯文本 + 结构化元数据
    ↓ Pandas DataFrame 管理
    ↓ 分块策略 (固定长度 / 语义 / 递归)
    ↓ 每个 chunk 带有完整元数据
    ↓ 导出为向量数据库格式
```

---

#### 文档加载与初步解析

##### 支持的文件类型与解析策略

| 格式 | 推荐解析库 | Pandas 角色 |
|------|-----------|-------------|
| PDF | PyMuPDF / pdfplumber | 元数据提取、页面级统计 |
| Markdown | 直接读取 | 标题层级分析 |
| HTML | BeautifulSoup / Readability | 正文提取、标签清理 |
| DOCX | python-docx | 段落/表格分离 |
| JSON/YAML | json / yaml 库 | 结构化字段映射 |
| CSV/Excel | Pandas 原生 | 表格数据直接入库 |

##### 文档扫描器：批量发现和分类

```python
import pandas as pd
from pathlib import Path
import hashlib
from datetime import datetime

class DocumentScanner:
    def __init__(self, root_dir: str):
        self.root = Path(root_dir)
        self.extensions = {
            ".pdf": "application/pdf",
            ".md": "text/markdown",
            ".html": "text/html",
            ".htm": "text/html",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".txt": "text/plain",
            ".json": "application/json",
            ".csv": "text/csv",
            ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        }

    def scan(self, recursive: bool = True) -> pd.DataFrame:
        pattern = "**/*" if recursive else "*"
        files = []

        for ext, mime in self.extensions.items():
            for fpath in self.root.glob(f"{pattern}{ext}"):
                stat = fpath.stat()
                files.append({
                    "file_path": str(fpath.relative_to(self.root)),
                    "full_path": str(fpath),
                    "file_name": fpath.name,
                    "extension": ext.lower(),
                    "mime_type": mime,
                    "size_bytes": stat.st_size,
                    "size_kb": round(stat.st_size / 1024, 1),
                    "modified_time": datetime.fromtimestamp(stat.st_mtime),
                    "file_hash": self._hash_file(fpath),
                })

        df = pd.DataFrame(files)

        if len(df) == 0:
            print("⚠️ 未找到任何支持的文档")
            return df

        df["size_category"] = pd.cut(
            df["size_bytes"],
            bins=[0, 1024, 10240, 102400, 1048576, float('inf')],
            labels=["<1KB", "1-10KB", "10-100KB", "100KB-1MB", ">1MB"],
        )

        self._print_summary(df)
        return df

    def _hash_file(self, fpath: Path) -> str:
        hasher = hashlib.md5()
        with open(fpath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()[:12]

    def _print_summary(self, df):
        print("=== 文档扫描结果 ===\n")
        print(f"目录: {self.root}")
        print(f"总文件数: {len(df):,}")
        print(f"总大小: {df['size_bytes'].sum() / 1024 / 1024:.1f} MB\n")

        print("按类型分布:")
        ext_stats = df.groupby("extension").agg(
            count=("file_path", "count"),
            total_size=("size_kb", "sum"),
            avg_size=("size_kb", "mean"),
        ).round(1)
        ext_stats["total_size_mb"] = (ext_stats["total_size"] / 1024).round(2)
        print(ext_stats[["count", "total_size_mb", "avg_size"]].to_string())

        print(f"\n按大小分布:")
        print(df["size_category"].value_counts().sort_index().to_string())


scanner = DocumentScanner("./knowledge_base/")
doc_index = scanner.scan(recursive=True)
```

输出：
```
=== 文档扫描结果 ===

目录: knowledge_base/
总文件数: 1,234
总大小: 456.7 MB

按类型分布:
   count  total_size_mb  avg_size
extension
.md         567       123.4      217.7
.pdf        321       289.0      900.3
.html       198        34.2      172.7
.docx        89         8.9      100.0
.txt         42         0.8      19.0
.json        12         0.3      25.0
.csv          5         0.1      20.0

按大小分布:
<1KB         23
1-10KB       456
10-100KB     512
100KB-1MB     189
>1MB           54
```

---

#### PDF 文档解析与内容提取

##### PyMuPDF 批量解析

```python
import fitz  # PyMuPDF

class PDFParser:
    def __init__(self):
        self.stats = {
            "parsed": 0,
            "failed": 0,
            "total_pages": 0,
            "total_chars": 0,
        }

    def parse_single(self, file_path: str) -> dict:
        try:
            doc = fitz.open(file_path)
            pages_text = []
            page_data = []

            for page_num, page in enumerate(doc):
                text = page.get_text()
                pages_text.append(text)

                page_info = {
                    "page_number": page_num + 1,
                    "char_count": len(text),
                    "has_images": len(page.get_images()) > 0,
                }
                page_data.append(page_info)

            full_text = "\n\n".join(pages_text)

            self.stats["parsed"] += 1
            self.stats["total_pages"] += len(doc)
            self.stats["total_chars"] += len(full_text)

            return {
                "success": True,
                "text": full_text,
                "page_count": len(doc),
                "char_count": len(full_text),
                "pages": page_data,
            }

        except Exception as e:
            self.stats["failed"] += 1
            return {"success": False, "error": str(e)}

    def parse_batch(self, doc_index: pd.DataFrame) -> pd.DataFrame:
        pdf_docs = doc_index[doc_index["extension"] == ".pdf"].copy()

        results = []
        for _, row in pdf_docs.iterrows():
            result = self.parse_single(row["full_path"])
            results.append({
                **row.to_dict(),
                **result,
                "parsed_at": pd.Timestamp.now(),
            })

        result_df = pd.DataFrame(results)

        print(f"\n=== PDF 解析结果 ===")
        print(f"成功: {self.stats['parsed']:,}")
        print(f"失败: {self.stats['failed']:,}")
        print(f"总页数: {self.stats['total_pages']:,}")
        print(f"总字符数: {self.stats['total_chars']:,}")

        if len(result_df) > 0:
            print(f"\n页数分布:")
            print(result_df[result_df["success"]]["page_count"].describe().round(1).to_string())

        return result_df


parser = PDFParser()
pdf_results = parser.parse_batch(doc_index)
```

输出：
```
=== PDF 解析结果 ===

成功: 318
失败: 3
总页数: 15,678
总字符数: 45,678,901

页数分布:
count     318.000000
mean       49.300000
std        32.100000
min         1.000000
25%        24.000000
50%        42.000000
75%        68.000000
max       256.000000
```

---

#### Markdown 文档结构化解析

Markdown 文件天然具有标题层级结构，可以保留为分块的边界信息：

```python
import re
from dataclasses import dataclass
from typing import Optional

@dataclass
class MarkdownSection:
    level: int
    title: str
    content: str
    start_line: int
    end_line: int

class MarkdownParser:
    HEADING_PATTERN = re.compile(r'^(#{1,6})\s+(.+)$')

    def parse_file(self, file_path: str) -> dict:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        full_text = "".join(lines)
        sections = []
        current_section = None

        for i, line in enumerate(lines):
            match = self.HEADING_PATTERN.match(line)
            if match:
                if current_section is not None:
                    current_section.end_line = i - 1
                    sections.append(current_section)

                level = len(match.group(1))
                title = match.group(2).strip()
                current_section = MarkdownSection(
                    level=level, title=title,
                    content="", start_line=i, end_line=i
                )
            elif current_section is not None:
                current_section.content += line

        if current_section is not None:
            current_section.end_line = len(lines) - 1
            sections.append(current_section)

        return {
            "text": full_text,
            "char_count": len(full_text),
            "line_count": len(lines),
            "sections": [
                {
                    "level": s.level,
                    "title": s.title,
                    "content_length": len(s.content.strip()),
                    "start_line": s.start_line,
                    "end_line": s.end_line,
                }
                for s in sections
            ],
            "section_count": len(sections),
        }

    def parse_batch(self, doc_index: pd.DataFrame) -> pd.DataFrame:
        md_docs = doc_index[doc_index["extension"] == ".md"].copy()

        results = []
        for _, row in md_docs.iterrows():
            parsed = self.parse_file(row["full_path"])
            results.append({**row.to_dict(), **parsed})

        result_df = pd.DataFrame(results)

        total_sections = result_df["section_count"].sum()
        avg_sections = result_df["section_count"].mean()

        print(f"\n=== Markdown 解析结果 ===")
        print(f"文件数: {len(result_df):,}")
        print(f"总章节数: {int(total_sections):,}")
        print(f"平均每文件章节: {avg_sections:.1f}")

        return result_df


md_parser = MarkdownParser()
md_results = md_parser.parse_batch(doc_index)
print(md_results[["file_name", "char_count", "section_count"]].head(10).to_string(index=False))
```

输出：
```
=== Markdown 解析结果 ===

文件数: 567
总章节数: 8,901
平均每文件章节: 15.7

              file_name  char_count  section_count
        pandas-guide.md        45678             23
        numpy-tutorial.md        89234             31
        llm-fine-tuning.md       123456             18
         rag-system.md        67890             27
```

---

#### 统一文档解析入口

将所有格式的解析结果统一到一个 DataFrame：

```python
class UnifiedDocumentParser:
    def __init__(self, root_dir: str):
        self.scanner = DocumentScanner(root_dir)
        self.pdf_parser = PDFParser()
        self.md_parser = MarkdownParser()

    def parse_all(self) -> pd.DataFrame:
        doc_index = self.scanner.scan()

        pdf_result = self.pdf_parser.parse_batch(doc_index)
        md_result = self.md_parser.parse_batch(doc_index)

        other_docs = doc_index[
            ~doc_index["extension"].isin([".pdf", ".md"])
        ].copy()

        other_docs["text"] = ""
        other_docs["success"] = True
        other_docs["char_count"] = 0
        other_docs["page_count"] = 0
        other_docs["section_count"] = 0

        all_parsed = pd.concat([
            pdf_result,
            md_result,
            other_docs[["file_path","full_path","file_name","extension",
                        "mime_type","size_bytes","size_kb",
                        "modified_time","file_hash","size_category",
                        "text","success","char_count","page_count","section_count"]],
        ], ignore_index=True)

        all_parsed["parse_status"] = all_parsed["success"].map(
            {True: "✅ 成功", False: "❌ 失败"}
        )

        success_rate = all_parsed["success"].mean() * 100
        total_chars = all_parsed.loc[all_parsed["success"], "char_count"].sum()

        print(f"\n{'='*50}")
        print(f"  统一解析完成")
        print(f"  成功率: {success_rate:.1f}%")
        print(f"  总字符数: {total_chars:,}")
        print(f"{'='*50}")

        return all_parsed


unified_parser = UnifiedDocumentParser("./knowledge_base/")
all_documents = unified_parser.parse_all()
```

输出：
```
=== 文档扫描结果 ===
...
=== PDF 解析结果 ===
...
=== Markdown 解析结果 ===
...

==================================================
  统一解析完成
  成功率: 99.8%
  总字符数: 67,890,123
==================================================
```

解析完成的 `all_documents` DataFrame 将作为下一节分块的输入。
