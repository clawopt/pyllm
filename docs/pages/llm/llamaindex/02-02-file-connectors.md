---
title: 文件连接器：PDF / Word / Markdown / HTML / Excel / 图片
description: 各种文件格式的深度解析、多引擎选择、表格数据提取、图片 OCR 集成
---
# 文件连接器：PDF / Word / Markdown / HTML / Excel / 图片

上一节我们从宏观层面介绍了 LlamaIndex 的 Connector 体系，这一节来深入最常见的文件类型——毕竟在企业知识库场景中，**80% 以上的数据都以文件形式存在**：产品手册是 PDF、合同是 Word、技术文档是 Markdown、财务报表是 Excel、宣传材料里有 PNG/JPG 图片。

每种文件格式都有自己的"性格"：PDF 是出了名的难解析（因为它本质上是一系列绘图指令而非文本流）、Word 有复杂的嵌套结构（标题、段落、表格、图片混在一起）、HTML 充满了导航栏和广告这类噪音、Excel 的数据是结构化的但语义需要解读、图片则需要 OCR 才能变成文字。LlamaIndex 为每种格式都提供了专门的 Reader 和可配置的解析引擎。

## PDF 解析：最头疼也最重要

PDF 无疑是企业环境中最常见的文档格式，也是解析难度最大的。LlamaIndex 提供了多种 PDF 引擎供选择，每种有不同的适用场景。

### PyMuPDF（默认引擎）

这是 LlamaIndex 的默认 PDF 引擎，基于 PyMuPDF（也叫 fitz）库，特点是速度快且对大多数 PDF 效果不错：

```python
from llama_index.readers.file import PDFReader

reader = PDFReader(backend="pymupdf")  # backend="pymupdf" 是默认值，可以省略
documents = reader.load_data(file_path="./docs/report.pdf")

print(f"共 {len(documents)} 页")
for i, doc in enumerate(documents[:3]):
    print(f"\n--- 第 {i+1} 页 ---")
    print(doc.text[:200])
    print(f"元数据: {doc.metadata}")
```

PyMuPDF 会将 PDF 的每一页转换为一个独立的 Document 对象，每页的文本按照阅读顺序提取。元数据中自动包含了页码（`page_label`）、文件路径等信息。

但对于中文 PDF 或者包含复杂表格的 PDF，PyMuPDF 可能会遇到以下问题：
- 中文乱码或显示为方框（□□□）
- 表格内容被打散成无意义的文本片段
- 双栏排版的内容被错误地左右拼接

### pdfplumber（表格友好）

如果 PDF 中包含大量表格，`pdfplumber` 引擎通常是更好的选择：

```bash
pip install pdfplumber
```

```python
from llama_index.readers.file import PDFReader

reader = PDFReader(backend="pdfplumber")
documents = reader.load_data(file_path="./docs/financial_report.pdf")

for doc in documents:
    print(doc.text[:300])
```

`pdfplumber` 的优势在于它能更好地保留表格的结构信息。它会尝试将表格转换为类似 Markdown 表格的文本格式：

```
| 项目 | Q1 收入 | Q2 收入 | 同比增长 |
|------|---------|---------|----------|
| 产品A | 120万   | 150万   | +25%     |
| 产品B | 80万    | 95万    | +18.75%  |
```

而不是像 PyMuPDF 那样输出一堆散乱的数字。

### Marker（AI 驱动的解析）

对于质量要求极高的场景（比如法律文书、学术论文），可以使用 **Marker**——一个基于 AI 的 PDF 解析引擎，它不仅能提取文本，还能理解文档的布局结构：

```bash
pip install marker-pdf
```

```python
from llama_index.readers.file import PDFReader

reader = PDFReader(backend="marker")
documents = reader.load_data(file_path="./docs/legal_contract.pdf")
```

Marker 的特点：
- 使用视觉模型理解页面布局（双栏、页眉页脚、脚注等）
- 能准确区分正文、标题、图表说明等不同区域
- 表格提取质量接近人工水平
- 支持公式（LaTeX）和代码块的识别

代价是速度较慢（每页需要几秒的推理时间）和需要 GPU 加速才能达到合理速度。适合对质量敏感但不追求实时性的离线预处理场景。

### PDF 解析引擎对比

| 引擎 | 速度 | 表格质量 | 中文支持 | 复杂布局 | 适用场景 |
|------|------|----------|----------|----------|----------|
| **PyMuPDF** | ⚡⚡⚡⚡⚡ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | 通用场景，快速预览 |
| **pdfplumber** | ⚡⚡⚡⚡ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | 表格密集型文档 |
| **Marker** | ⚡ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 高质量要求的正式文档 |

**实际建议：** 先用 PyMuPDF 快速验证效果，如果发现表格或布局有问题，再切换到 pdfplumber 或 Marker。不要一开始就用最重的引擎——那样会浪费大量计算资源。

### 多文件批量加载

实际项目中很少只有一个 PDF 文件。以下是批量处理 PDF 的常用模式：

```python
from llama_index.core import SimpleDirectoryReader
from llama_index.readers.file import PDFReader

reader = SimpleDirectoryReader(
    input_dir="./pdf_library",
    required_exts=[".pdf"],
    file_extractor={
        ".pdf": PDFReader(backend="pdfplumber"),  # 所有 PDF 用 pdfplumber
    },
)
documents = reader.load_data()
print(f"成功加载 {len(documents)} 个文档/页")
```

## Word 文档解析

Word 文档（`.docx`）在企业环境中同样普遍——合同、提案、会议纪要大多以 Word 格式存在。LlamaIndex 使用 `python-docx` 库来解析 Word 文档：

```python
from llama_index.readers.file import DocxReader

reader = DocxReader()
documents = reader.load_data(file_path="./docs/proposal.docx")

for doc in documents:
    print(f"文档长度: {len(doc.text)} 字符")
    print(doc.text[:500])
```

DocxReader 的解析特点：
- **保留层级结构**：标题（Heading 1/2/3）、段落、列表都会被保留
- **表格转为文本**：Word 中的表格会被转换为带分隔符的文本表示
- **忽略样式信息**：字体、颜色、缩进等视觉样式不会被保留（因为对 RAG 没有意义）
- **页眉页脚处理**：默认会提取页眉页脚内容，可以通过参数控制是否包含

```python
reader = DocxReader(
    include_header=False,   # 不包含页眉
    include_footer=False,   # 不包含页脚
)
```

### Word 文档的特殊注意事项

**问题一：嵌套表格。** 如果 Word 文档中有表格套表格的情况（在一些复杂报告中很常见），`python-docx` 可能无法完美还原嵌套结构。这时可以考虑先把 Word 转换为 PDF 再用 Marker 解析。

**问题二：文档保护/加密。** 受密码保护的 Word 文件无法直接解析。你需要先通过其他方式解除保护（如用 `msoffcrypto-python` 库解密），然后再交给 LlamaIndex 处理。

**问题三：`.doc` 格式（旧版 Word）。** LlamaIndex 的 DocxReader 只支持 `.docx` 格式（基于 XML 的新版格式）。对于旧的 `.doc` 二进制格式，需要先用 `libreoffice --headless --convert-to docx` 进行转换。

## Markdown 解析

Markdown 是技术团队最爱的文档格式——README、API 文档、Wiki 页面、博客文章都是 Markdown。好消息是 Markdown 本身就是纯文本，解析起来几乎没有困难：

```python
from llama_index.readers.file import MarkdownReader

reader = MarkdownReader()
documents = reader.load_data(file_path="./docs/api_guide.md")

for doc in documents:
    print(doc.text[:400])
```

MarkdownReader 的特殊之处在于它**会保留 Markdown 的结构信息**，将其转化为特殊的标记，方便后续的分块和检索：

```markdown
原文:
# 第一章 简介
这是一个简介段落。

## 1.1 背景
背景信息...

解析后（简化展示）:
<Document>
  <h1>第一章 简介</h1>
  <p>这是一个简介段落。</p>
  <h2>1.1 背景</h2>
  <p>背景信息...</p>
</Document>
```

这种结构保留使得后续的 Node Parser 可以"感知"到标题层级，从而在切分时避免把同一个章节的内容拆散到不同的 Node 中。

### MarkdownReader 的高级选项

```python
from llama_index.readers.file import MarkdownReader

reader = MarkdownReader(
    remove_hyperlinks=True,       # 移除链接但保留链接文本
    remove_images=True,           # 移除图片标记 ![alt](url)
    as_tuples=False,              # 是否返回 (text, path) 元组
)
documents = reader.load_data(file_path="./docs/readme.md")
```

## HTML 解析

HTML 网页是一种特殊的数据源——它不仅包含有用的正文内容，还混杂了大量"噪音"：导航栏、侧边栏、广告、评论区域、Cookie 提示条、脚本和样式标签等等。如果直接把整个 HTML 当作文本喂给 RAG 系统，检索质量会大打折扣。

LlamaIndex 提供了 `HtmlReader`，它在内部使用 `beautifulsoup4` 来解析 HTML 并提取主要内容：

```python
from llama_index.readers.file import HtmlReader

reader = HtmlReader(
    tag_strategy="default",      # 内容提取策略
    ignore_unknown_tags=False,
)
documents = reader.load_data(file_path="./docs/page.html")

for doc in documents:
    print(doc.text[:400])
```

### tag_strategy 选项

`tag_strategy` 控制 HtmlReader 如何决定哪些标签内容应该被保留：

```python
# default: 保留 <p>, <div>, <section>, <article>, <li> 等内容标签
reader = HtmlReader(tag_strategy="default")

# basic: 更加激进地过滤，只保留最基本的文本内容
reader = HtmlReader(tag_strategy="basic")

# custom: 自定义要保留的标签
from html_reader import HtmlReader
reader = HtmlReader(tag_strategy="custom", custom_tags=["article", "main"])
```

### 从 URL 直接加载 HTML

很多时候 HTML 不是本地文件而是远程网页。虽然严格来说这属于"Web 连接器"的范畴（2.4 节会详细讲），但 HtmlReader 本身也支持直接从 URL 加载：

```python
from llama_index.readers.file import HtmlReader

reader = HtmlReader()
documents = reader.load_data(file_path="https://example.com/blog/post")
```

不过对于生产环境的网页抓取，推荐使用专门的 `BeautifulSoupWebReader` 或 `PlaywrightReader`（下一节详细介绍），它们提供了更好的反爬虫处理和 JavaScript 渲染支持。

## Excel 解析

Excel 文件（`.xlsx`/`.xls`）在企业中承载着大量的结构化数据：销售报表、客户名单、库存记录、财务数据等。与其他文件格式不同，Excel 的数据天然具有**行列结构**，这种结构在 RAG 场景下既是优势也是挑战。

```python
from llama_index.readers.file import PandasCSVReader  # CSV 也归此类

reader = PandasCSVReader(
    concat_rows=False,  # False: 每个 sheet 一行; True: 合并为一个
)
documents = reader.load_data(file_path="./data/sales.xlsx")

for doc in documents:
    print(doc.metadata)  # 包含 sheet_name 等信息
    print(doc.text[:300])
```

### Excel 解析的关键决策：如何将表格转为文本？

这是 Excel 解析中最重要的问题。一张表格可以用多种方式表达为文本，每种方式会影响后续的检索效果：

**方式一：原始格式（默认）。** PandasCSVReader 默认会将 DataFrame 转为其字符串表示：

```
   产品名称  季度  销售额(万元)
0  智能音箱  Q1   150
1  智能音箱  Q2   180
2  智能手表  Q1   95
```

这种方式保留了完整的结构信息，但如果表格很大，生成的文本会很长。

**方式二：逐行叙述。** 将每一行转换为自然语言描述：

```
记录1: 智能音箱在Q1季度的销售额为150万元。
记录2: 智能音箱在Q2季度的销售额为180万元。
记录3: 智能手表在Q1季度的销售额为95万元。
```

这种方式的优点是对语义搜索更友好——用户问"智能手表卖了多少钱"，系统能精确匹配到"智能手表...95万元"这条记录。缺点是丢失了表格的整体视图（难以回答"总销售额是多少"这类聚合查询）。

```python
import pandas as pd
from llama_index.core import Document

def excel_to_narrative_rows(filepath):
    """将 Excel 转为逐行叙述格式"""
    df = pd.read_excel(filepath)
    documents = []
    for _, row in df.iterrows():
        narrative = "，".join([f"{col}是{val}" for col, val in row.items()])
        documents.append(Document(
            text=narrative,
            metadata={"source": filepath, "row_index": row.name},
        ))
    return documents

documents = excel_to_narrative_rows("./data/sales.xlsx")
```

**方式三：摘要式。** 为整个表格生成一段摘要性描述，配合原始数据一起存入索引：

```python
def excel_with_summary(filepath):
    df = pd.read_excel(filepath)
    summary = f"该表格包含{len(df)}行记录，"
    summary += f"列名为：{', '.join(df.columns.tolist())}。"

    original = df.to_string()
    documents = [
        Document(text=summary, metadata={"type": "table_summary"}),
        Document(text=original, metadata={"type": "table_data"}),
    ]
    return documents
```

到底选哪种方式取决于你的查询模式：如果是精确字段查询，方式二最好；如果是趋势分析类查询，方式三更合适；如果不确定，可以把两种都存进去让检索系统自己选择。

## 图片解析与 OCR

现代企业文档中经常嵌入图片：产品截图、流程图、组织架构图、扫描件等。要让 RAG 系统理解这些图片，需要用到 **OCR（光学字符识别）** 技术。

LlamaIndex 通过 `llama-index-multi-modal-llms-openai` 包支持多模态能力，可以将图片发送给 GPT-4V 等视觉模型进行理解：

```bash
pip install llama-index-multi-modal-llms-openai
```

```python
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.core import SimpleDirectoryReader

openai_mm_llm = OpenAIMultiLLM(model="gpt-4o", max_new_tokens=1024)

reader = SimpleDirectoryReader(
    input_dir="./images",
    required_exts=[".png", ".jpg", ".jpeg"],
)
documents = reader.load_data()

for doc in documents:
    image_path = doc.metadata["file_path"]
    response = openai_mm_llm.complete(
        prompt="请详细描述这张图片中的所有文字内容和图表信息",
        image_documents=[doc],
        )
    doc.text = response.text  # 将图片描述替换为文本

# 之后就可以正常建索引了
index = VectorStoreIndex.from_documents(documents)
```

这种方式的效果很好但成本较高（每次调用 GPT-4V 都需要付费）。对于批量处理大量图片的场景，可以考虑先用传统的 OCR 引擎（如 Tesseract 或 PaddleOCR）做初步提取，只在必要时才调用视觉模型：

```python
import pytesseract
from PIL import Image
from llama_index.core import Document

def ocr_image(image_path):
    """使用 Tesseract 做 OCR"""
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img, lang='chi_sim+eng')
    return text.strip()

reader = SimpleDirectoryReader("./images", required_exts=[".png", ".jpg"])
raw_docs = reader.load_data()

ocr_documents = []
for doc in raw_docs:
    ocr_text = ocr_image(doc.metadata["file_path"])
    if ocr_text:
        ocr_documents.append(Document(
            text=ocr_text,
            metadata={**doc.metadata, "ocr_engine": "tesseract"},
        ))

print(f"OCR 完成，共处理 {len(ocr_documents)} 张图片")
```

## SimpleDirectoryReader 的一站式方案

回顾一下，前面介绍了各种文件格式的专门 Reader。但在实际项目中，你很少需要逐一调用它们——**`SimpleDirectoryReader` 已经帮你把这些整合到了一起**：

```python
from llama_index.core import SimpleDirectoryReader
from llama_index.readers.file import PDFReader, DocxReader

reader = SimpleDirectoryReader(
    input_dir="./company_knowledge_base",
    recursive=True,
    file_extractor={
        ".pdf": PDFReader(backend="pdfplumber"),  # PDF 用 pdfplumber（表格友好）
        ".docx": DocxReader(),                     # Word 用默认解析器
        # .md, .html, .xlsx, .png 等使用内置默认解析器
    },
    file_metadata=lambda fname: {
        "department": "product_team",
        "source_type": fname.split(".")[-1],
    },
)
all_documents = reader.load_data()

# 查看加载统计
from collections import Counter
file_types = Counter(doc.metadata.get("file_type", "unknown") for doc in all_documents)
print(f"总共加载 {len(all_documents)} 个文档")
for ftype, count in file_types.most_common():
    print(f"  {ftype}: {count} 个")
```

输出示例：
```
总共加载 247 个文档
  pdf: 89 个
  md: 72 个
  docx: 45 个
  xlsx: 23 个
  html: 12 个
  png: 6 个
```

## 常见误区与最佳实践

**误区一："PDF 解析效果不好是 LlamaIndex 的问题"。** 大部分情况下 PDF 解析质量差是因为 PDF 本身的问题——扫描件、加密、复杂布局、非标准编码等。这不是框架的错，而是 PDF 这种格式的固有缺陷。解决方法是选择合适的解析引擎（如 Marker 用于高质量需求）或在源头改善 PDF 质量（如从原始 Word 导出而非打印为 PDF）。

**误区二："所有文件都应该加载进来"。** 不是所有文件都有价值。临时文件、草稿、重复文件、过时版本不仅浪费存储和计算资源，还会降低检索质量（引入噪音）。建立数据清洗流程——在加载前过滤掉低质量文件——是构建高质量 RAG 系统的重要一步。

**误区三："Excel 数据直接塞进向量数据库就行"。** 结构化数据和半结构化数据的处理方式应该不同。Excel 中的数值数据也许更适合存入传统数据库并通过 SQL 检索，而非强行转向量。第二章后面讲数据库连接器时会再回到这个问题。

**误区四："图片必须用 GPT-4V"。** 对于纯文本类的图片（如扫描文档、截图），传统 OCR 引擎（Tesseract、PaddleOCR）的速度快几个数量级，成本几乎为零。只有在需要"理解"图片内容（如图表分析、流程图解读）时才有必要使用多模态 LLM。**根据图片类型选择合适的工具，不要一刀切。**
