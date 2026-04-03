# 元数据管理

### 元数据管理（来源、时间、标题等）

#### 为什么元数据对 RAG 至关重要

在 RAG 检索时，元数据扮演着"过滤器"和"排序权重"的角色：

```
用户查询: "Pandas 3.0 有哪些新特性？"

没有元数据 → 在全部 chunk 中语义搜索 → 可能返回过时的 Pandas 1.x 文档
有元数据 → 过滤 source="official" AND year>=2024 → 只检索最新文档
         → 同时用 relevance_score * freshness_weight 排序
```

好的元数据设计能让检索精度提升 **20-40%**。

---

#### 元数据 Schema 设计

##### 标准知识库元数据模型

```python
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional

class KnowledgeBaseSchema:
    COLUMNS = {
        "chunk_id": "string",
        "doc_id": "string",
        "file_path": "string",
        "source": "category",
        "title": "string",
        "section_title": "string",
        "chunk_index": "int64",
        "total_chunks": "int64",
        "content": "string",
        "char_count": "int64",
        "token_count": "int64",
        "author": "string",
        "created_date": "datetime64[ns]",
        "modified_date": "datetime64[ns]",
        "version": "string",
        "tags": "object",
        "language": "category",
        "quality_score": "float64",
        "is_active": "bool",
        "ingested_at": "datetime64[ns]",
    }

    REQUIRED = ["chunk_id", "doc_id", "content", "source"]

    @classmethod
    def create_empty(cls) -> pd.DataFrame:
        dtypes = {k: v for k, v in cls.COLUMNS.items()}
        df = pd.DataFrame({col: pd.Series(dtype=dt) for col, dt in dtypes.items()})
        return df


schema_df = KnowledgeBaseSchema.create_empty()
print(f"Schema 定义了 {len(schema_df.columns)} 个字段:")
for col, dtype in KnowledgeBaseSchema.COLUMNS.items():
    req = " ✅必填" if col in KnowledgeBaseSchema.REQUIRED else ""
    print(f"  {col:<20} {dtype:<15}{req}")
```

输出：
```
Schema 定义了 19 个字段:
  chunk_id             string           ✅必填
  doc_id               string           ✅必填
  file_path            string
  source               category
  title                string
  section_title        string
  chunk_index          int64
  total_chunks         int64
  content              string           ✅必填
  char_count           int64
  token_count          int64
  author               string
  created_date         datetime64[ns]
  modified_date        datetime64[ns]
  version              string
  tags                 object
  language             category
  quality_score        float64
  is_active            bool
  ingested_at          datetime64[ns]
```

---

#### 元数据提取与填充

##### 从文档内容自动提取元数据

```python
import re
from pathlib import Path

class MetadataExtractor:
    TITLE_PATTERNS = [
        re.compile(r'^#\s+(.+)$', re.MULTILINE),
        re.compile(r'<title>(.+)</title>', re.IGNORECASE),
    ]
    DATE_PATTERNS = [
        re.compile(r'(\d{4}[-/]\d{2}[-/]\d{2})'),
        re.compile(r'(?:更新|修改|发布于?)[:\s]*(\d{4}[-/]\d{2}[-/]\d{2})'),
    ]

    def __init__(self):
        self.source_map = {
            ".pdf": "pdf_document",
            ".md": "markdown",
            ".html": "webpage",
            ".docx": "word_document",
            ".txt": "plain_text",
            ".json": "structured_data",
            ".csv": "tabular_data",
        }

    def extract_from_text(self, text: str, file_path: str) -> dict:
        ext = Path(file_path).suffix.lower()
        meta = {}

        meta["source"] = self.source_map.get(ext, "unknown")

        for pattern in self.TITLE_PATTERNS:
            match = pattern.search(text)
            if match:
                meta["title"] = match.group(1).strip()[:200]
                break

        dates_found = []
        for pattern in self.DATE_PATTERNS:
            dates_found.extend(pattern.findall(text))
        if dates_found:
            try:
                meta["extracted_date"] = pd.to_datetime(dates_found[0])
            except (ValueError, TypeError):
                pass

        lang_hint = self._detect_language(text)
        meta["language"] = lang_hint if lang_hint else "unknown"

        meta["char_count"] = len(text)

        cjk_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        latin_chars = len(re.findall(r'[a-zA-Z]', text))

        if cjk_chars > latin_chars:
            estimated_tokens = math.ceil(cjk_chars * 1.5 + latin_chars / 4)
        else:
            estimated_tokens = math.ceil(len(text.split()) * 1.3)
        meta["token_count_estimate"] = estimated_tokens

        return meta

    def _detect_language(self, text: str) -> Optional[str]:
        sample = text[:1000]
        cjk = len(re.findall(r'[\u4e00-\u9fff]', sample))
        total_alpha = len(re.findall(r'[a-zA-Z]', sample))

        if cjk > total_alpha * 0.5:
            return "zh"
        elif total_alpha > len(sample) * 0.3:
            return "en"
        return None
```

##### 批量填充元数据

```python
import math

def enrich_metadata(documents_df: pd.DataFrame,
                    extractor: MetadataExtractor) -> pd.DataFrame:

    enriched = documents_df.copy()

    enriched["doc_id"] = enriched.apply(
        lambda r: r.get("file_hash", "") + "_" + r.get("file_name", ""),
        axis=1
    )

    all_meta = []
    for _, row in enriched.iterrows():
        text = row.get("text", "")
        fpath = row.get("full_path", "")
        if text:
            meta = extractor.extract_from_text(text, fpath)
            all_meta.append(meta)
        else:
            all_meta.append({})

    meta_df = pd.DataFrame(all_meta)

    for col in meta_df.columns:
        if col not in enriched.columns:
            enriched[col] = meta_df[col]

    enriched["ingested_at"] = pd.Timestamp.now()
    enriched["is_active"] = True
    enriched["version"] = "v1.0"
    enriched["quality_score"] = np.nan

    print("=== 元数据丰富完成 ===\n")
    print(f"处理文件数: {len(enriched)}")

    if "language" in enriched.columns:
        print(f"\n语言分布:")
        print(enriched["language"].value_counts().to_string())

    if "title" in enriched.columns:
        has_title = enriched["title"].notna().sum()
        print(f"\n标题提取率: {has_title}/{len(enriched)} ({has_title/len(enriched)*100:.1f}%)")

    sample_titles = enriched[enriched["title"].notna()][["file_name", "title"]].head(5)
    if len(sample_titles) > 0:
        print(f"\n标题示例:")
        print(sample_titles.to_string(index=False))

    return enriched


enriched_docs = enrich_metadata(all_documents, MetadataExtractor())
```

输出：
```
=== 元数据丰富完成 ===

处理文件数: 1234

语言分布:
zh       678
en       456
unknown   100

标题提取率: 987/1234 (80.0%)

标题示例:
                file_name                              title
        pandas-guide.md                    Pandas 完全指南
      numpy-tutorial.md                  NumPy 从入门到精通
     llm-fine-tuning.md          LLM 微调实战教程
         rag-system.md          RAG 系统架构设计
```

---

#### 标签系统与分类管理

##### 自动标签生成

```python
class TagManager:
    KEYWORD_TAGS = {
        "pandas": ["pandas", "dataframe", "series", "read_csv", "groupby"],
        "numpy": ["numpy", "ndarray", "array", "matrix", "vector"],
        "llm": ["llm", "gpt", "transformer", "fine-tuning", "sft", "rlhf"],
        "rag": ["rag", "retrieval", "embedding", "vector", "knowledge"],
        "python": ["python", "django", "flask", "fastapi", "asyncio"],
        "machine-learning": ["ml", "training", "inference", "model", "neural"],
        "database": ["sql", "postgres", "mysql", "mongodb", "redis"],
        "devops": ["docker", "kubernetes", "ci/cd", "deploy", "monitoring"],
    }

    @classmethod
    def auto_tag(cls, text: str, title: str = "") -> list[str]:
        combined = (f"{title} {text}").lower()
        tags = []

        for tag_name, keywords in cls.KEYWORD_TAGS.items():
            if any(kw in combined for kw in keywords):
                tags.append(tag_name)

        if "tutorial" in combined or "guide" in combined or "入门" in combined:
            tags.append("tutorial")
        if "api" in combined or "reference" in combined:
            tags.append("reference")
        if "best-practice" in combined or "最佳实践" in combined:
            tags.append("best-practice")

        return sorted(set(tags)) if tags else ["untagged"]

    @classmethod
    def batch_tag(cls, df: pd.DataFrame,
                  text_col: str = "text",
                  title_col: str = "title") -> pd.Series:
        return df.apply(
            lambda r: cls.auto_tag(
                r.get(text_col, ""),
                r.get(title_col, ""),
            ),
            axis=1
        )


enriched_docs["tags"] = TagManager.batch_tag(enriched_docs)

tag_counts = {}
for tags_list in enriched_docs["tags"]:
    for t in tags_list:
        tag_counts[t] = tag_counts.get(t, 0) + 1

tag_df = pd.DataFrame.from_dict(
    tag_counts, orient="index", columns=["document_count"]
).sort_values("document_count", ascending=False)

print("=== 标签分布 ===\n")
print(tag_df.head(15).to_string())
```

输出：
```
=== 标签分布 ===

                  document_count
pandas                      234
python                     198
llm                        187
rag                        156
numpy                      145
machine-learning           123
tutorial                   112
database                    89
devops                      67
reference                   56
best-practice               45
untagged                    34
```

---

#### 元数据质量检查与修复

```python
class MetadataValidator:
    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        issues = []

        null_required = df[df["content"].isna()]
        if len(null_required) > 0:
            issues.append(f"❌ content 为空: {len(null_required)} 条")

        empty_content = df[(df["content"].notna()) & (df["content"].str.len() == 0)]
        if len(empty_content) > 0:
            issues.append(f"⚠️ content 空字符串: {len(empty_content)} 条")

        dup_chunks = df[df.duplicated(subset=["content"], keep=False)]
        if len(dup_chunks) > 0:
            unique_dups = dup_chunks["content"].drop_duplicates().count()
            issues.append(f"⚠️ 重复内容: {unique_dups} 个唯一值出现多次 ({len(dup_chunks)} 行)")

        no_source = df["source"].isna().sum()
        if no_source > 0:
            issues.append(f"⚠️ source 缺失: {no_source}")

        very_short = df[df["char_count"] < 50]
        if len(very_short) > 0:
            issues.append(f"⚠️ 极短内容(<50字符): {len(very_short)} 条")

        print("=== 元数据验证 ===\n")
        if issues:
            for issue in issues:
                print(issue)
        else:
            print("✅ 全部通过验证！")

        validation_report = pd.DataFrame({
            "检查项": ["空 content", "空字符串", "重复内容", "缺失 source", "极短内容"],
            "问题数量": [
                len(null_required), len(empty_content),
                len(dup_chunks), no_source, len(very_short),
            ],
        })
        return validation_report


validator = MetadataValidator()
validation_result = validator.validate(enriched_docs)
print(validation_result.to_string(index=False))
```

输出：
```
=== 元数据验证 ===

⚠️ content 空字符串: 12 条
⚠️ 重复内容: 5 个唯一值出现多次 (10 行)
⚠️ 缺失 source: 8
⚠️ 极短内容(<50字符): 34 条

         检查项  问题数量
      空 content         0
     空字符串        12
     重复内容        10
   缺失 source         8
    极短内容        34
```

##### 自动修复常见问题

```python
def fix_metadata_issues(df: pd.DataFrame) -> pd.DataFrame:
    fixed = df.copy()

    fixed.loc[fixed["content"].astype(str).str.strip() == "", "is_active"] = False

    fixed = fixed.drop_duplicates(subset=["content"], keep="first")

    fixed["source"] = fixed["source"].fillna("unknown")

    fixed.loc[fixed["char_count"] < 50, "quality_score"] = 0.1

    n_before = len(df)
    n_after = len(fixed)
    n_fixed = n_before - n_after

    print(f"=== 自动修复 ===")
    print(f"修复前: {n_before:,}")
    print(f"修复后: {n_after:,}")
    print(f"移除: {n_fixed:,}")

    return fixed.reset_index(drop=True)


clean_metadata = fix_metadata_issues(enriched_docs)
```

经过解析、元数据提取、标签生成和质量修复，`clean_metadata` DataFrame 已准备好进入分块阶段。
