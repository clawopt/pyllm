# 10.2 多模态数据接入与解析

## 当文档不只是文字：多模态数据接入的挑战

在上一节中我们确定了多模态 RAG 系统要处理的数据类型：文本、图片、表格、图表以及它们的混合体。现在的问题是：这些不同模态的数据从哪里来？来了之后怎么把它们变成系统能够理解和检索的统一格式？这一节就要解决这两个问题——**多模态数据的接入管道和智能解析器**。

首先需要理解一个核心概念：在多模态 RAG 中，"解析"的含义比传统文本 RAG 宽泛得多。对于纯文本 PDF，解析意味着提取文字内容并切分成 chunk；但对于一个包含架构图、参数表和操作步骤的混合文档，解析意味着：（1）识别页面上有哪些元素（这里是图片、那里是表格、周围是文字）；（2）对每种元素用最合适的方式提取信息（图片要做视觉理解，表格要做结构化提取，文字照常处理）；（3）保留元素之间的空间关系（这张图在第 3 段文字的下方、这个表在同一页的右侧）。

## 多模态文档解析器：MultiModalDocumentParser

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path
from enum import Enum
import base64
import json
import logging

from llama_index.core import Document
from llama_index.core.schema import ImageDocument, TextNode

logger = logging.getLogger(__name__)


class ContentType(Enum):
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    CHART = "chart"
    FORMULA = "formula"
    MIXED = "mixed"


@dataclass
class ParsedRegion:
    """解析出的单个区域"""
    content_type: ContentType
    content: str                           # 文本内容 / 图片描述 / 表格Markdown
    raw_data: Optional[bytes] = None       # 原始图片二进制数据
    image_path: Optional[str] = None       # 图片存储路径(如果有)
    bbox: Optional[tuple] = None           # 边界框 (x1, y1, x2, y2) 在页面中的位置
    page_number: int = 0
    confidence: float = 0.0                # 解析置信度
    metadata: dict = field(default_factory=dict)


@dataclass
class ParsedDocument:
    """完整的多模态解析结果"""
    source_path: str
    total_pages: int = 0
    regions: List[ParsedRegion] = field(default_factory=list)
    text_regions: List[ParsedRegion] = field(default_factory=list)
    image_regions: List[ParsedRegion] = field(default_factory=list)
    table_regions: List[ParsedRegion] = field(default_factory=list)
    chart_regions: List[ParsedRegion] = field(default_factory=list)

    def to_llama_documents(self) -> List[Document]:
        """将解析结果转换为 LlamaIndex Document 列表"""
        documents = []

        for region in self.regions:
            if region.content_type == ContentType.IMAGE and region.raw_data:
                doc = ImageDocument(
                    image=region.raw_data,
                    image_path=region.image_path,
                    metadata={
                        **region.metadata,
                        "content_type": "image",
                        "page": region.page_number,
                        "bbox": str(region.bbox) if region.bbox else "",
                    },
                )
            else:
                doc = Document(
                    text=region.content,
                    metadata={
                        **region.metadata,
                        "content_type": region.content_type.value,
                        "page": region.page_number,
                    },
                )
            documents.append(doc)

        return documents


class BaseMultiModalParser(ABC):
    """多模态文档解析器基类"""

    @abstractmethod
    def parse(self, file_path: str) -> ParsedDocument:
        ...

    @abstractmethod
    def supported_formats(self) -> List[str]:
        ...
```

### PDF 多模态解析器

PDF 是企业知识库中最常见也最复杂的格式——它可能同时包含文字、图片、表格、图表，而且布局千变万化。我们的策略是使用 PyMuPDF（fitz）做底层页面分析，结合多种专用工具来处理不同类型的区域：

```python
import fitz  # PyMuPDF
from PIL import Image
import io
import os


class PDFMultiModalParser(BaseMultiModalParser):
    """
    PDF 多模态解析器

    解析流程：
    1. 用 PyMuPDF 提取每页的文字块、图片块的位置信息
    2. 根据位置和尺寸判断每个区域的类型（图/表/文字）
    3. 对图片区域提取图像数据
    4. 对疑似表格区域调用表格提取工具
    5. 对所有非文字区域生成语义描述
    """

    def __init__(
        self,
        image_output_dir: str = "./data/images",
        min_image_size: tuple = (100, 100),      # 最小图片尺寸 (宽,高)
        max_image_size_mb: float = 10.0,          # 最大图片大小 MB
        ocr_enabled: bool = True,
        table_extraction_engine: str = "camelot", # camelot / pdfplumber / marker
    ):
        self.image_output_dir = Path(image_output_dir)
        self.image_output_dir.mkdir(parents=True, exist_ok=True)
        self.min_image_size = min_image_size
        self.max_image_size_bytes = int(max_image_size_mb * 1024 * 1024)
        self.ocr_enabled = ocr_enabled
        self.table_engine = table_extraction_engine

    def supported_formats(self) -> List[str]:
        return [".pdf"]

    def parse(self, file_path: str) -> ParsedDocument:
        doc = fitz.open(file_path)
        result = ParsedDocument(source_path=file_path, total_pages=len(doc))

        for page_num in range(len(doc)):
            page = doc[page_num]
            page_regions = self._parse_page(page, page_num, file_path)
            result.regions.extend(page_regions)

            for r in page_regions:
                if r.content_type == ContentType.TEXT:
                    result.text_regions.append(r)
                elif r.content_type == ContentType.IMAGE:
                    result.image_regions.append(r)
                elif r.content_type == ContentType.TABLE:
                    result.table_regions.append(r)
                elif r.content_type == ContentType.CHART:
                    result.chart_regions.append(r)

        doc.close()
        logger.info(
            f"PDF 解析完成: {file_path} → "
            f"{len(result.text_regions)} 文本区 + "
            f"{len(result.image_regions)} 图片区 + "
            f"{len(result.table_regions)} 表格区 + "
            f"{len(result.chart_regions)} 图表区"
        )

        return result

    def _parse_page(self, page, page_num: int, source_path: str) -> List[ParsedRegion]:
        """解析单页 PDF"""
        regions = []

        # Step 1: 获取页面上的所有图片
        image_list = page.get_images(full=True)
        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]

                if len(image_bytes) > self.max_image_size_bytes:
                    logger.debug(f"跳过大图片 (page {page_num}, img {img_index})")
                    continue

                img_pil = Image.open(io.BytesIO(image_bytes))
                width, height = img_pil.size

                if width < self.min_image_size[0] or height < self.min_image_size[1]:
                    continue

                # 保存图片到磁盘
                safe_name = Path(source_path).stem
                image_filename = f"{safe_name}_p{page_num}_img{img_index}.{image_ext}"
                image_full_path = self.image_output_dir / image_filename

                with open(image_full_path, "wb") as f:
                    f.write(image_bytes)

                # 获取图片在页面中的位置
                img_rects = page.get_image_rects(xref)
                bbox = img_rects[0] if img_rects else None

                # 初步分类：是表格还是普通图片？
                content_type = self._classify_image_region(img_pil, page, bbox)

                region = ParsedRegion(
                    content_type=content_type,
                    content=f"[图片: {image_filename}]",
                    raw_data=image_bytes,
                    image_path=str(image_full_path),
                    bbox=bbox,
                    page_number=page_num,
                    metadata={
                        "source_file": source_path,
                        "image_filename": image_filename,
                        "width": width,
                        "height": height,
                        "size_bytes": len(image_bytes),
                    },
                )
                regions.append(region)

            except Exception as e:
                logger.warning(f"提取图片失败 (page {page_num}, xref {xref}): {e}")

        # Step 2: 提取页面文字
        text_blocks = page.get_text("dict")["blocks"]
        for block in text_blocks:
            if block["type"] == 0:  # 文本块
                for line in block["lines"]:
                    line_text = "".join([span["text"] for span in line["spans"]]).strip()
                    if line_text:
                        bbox = line["bbox"]  # (x0, y0, x1, y1)
                        regions.append(ParsedRegion(
                            content_type=ContentType.TEXT,
                            content=line_text,
                            bbox=tuple(bbox),
                            page_number=page_num,
                            metadata={"source_file": source_path},
                        ))

        return regions

    def _classify_image_region(self, img_pil: Image.Image, page, bbox) -> ContentType:
        """
        判断一个图片区域是普通图片、表格还是图表

        启发式规则：
        - 宽高比接近且内部有网格线 → 可能是表格
        - 包含坐标轴/图例/数据系列标记 → 可能是图表
        - 其他 → 普通图片（截图/照片/图标）
        """
        width, height = img_pil.size
        aspect_ratio = width / height if height > 0 else 1

        # 规则1: 接似正方形或宽扁形 + 大尺寸 → 倾向于表格
        if 0.3 < aspect_ratio < 5.0 and width > 200 and height > 150:
            # 进一步检查是否像表格（检查是否有规则的行列结构）
            is_table_like = self._check_table_like_pattern(img_pil)
            if is_table_like:
                return ContentType.TABLE

        # 规则2: 检查是否像图表
        is_chart_like = self._check_chart_like_pattern(img_pil)
        if is_chart_like:
            return ContentType.CHART

        return ContentType.IMAGE

    def _check_table_like_pattern(self, img_pil: Image.Image) -> bool:
        """检查图片是否有表格特征"""
        import numpy as np

        img_array = np.array(img_pil.convert("L"))

        if img_array.shape[0] < 50 or img_array.shape[1] < 50:
            return False

        # 检测水平和垂直线条（表格的特征）
        from scipy.ndimage import sobel

        dx = sobel(img_array, axis=1)
        dy = sobel(img_array, axis=0)

        horizontal_lines = np.sum(np.abs(dx) > 50) / dx.size
        vertical_lines = np.sum(np.abs(dy) > 50) / dy.size

        # 如果有明显的横线和竖线交叉模式，很可能是表格
        return horizontal_lines > 0.02 and vertical_lines > 0.02

    def _check_chart_like_pattern(self, img_pil: Image.Image) -> bool:
        """检查图片是否有图表特征（简化版启发式）"""
        width, height = img_pil.size

        # 图表通常有较大的空白边距（坐标轴标签区域）
        # 这里做一个简单的尺寸比例检查
        if width > 300 and height > 200:
            return True

        return False
```

`PDFMultiModalParser` 的核心设计思路是**先粗分后精析**：先用 PyMuPDF 的 `get_images()` 和 `get_text("dict")` 把页面分解为图片块和文字块，然后通过启发式规则（宽高比、线条检测等）初步判断图片块的类型。这种两阶段策略的好处是速度快——不需要对每张图片都跑一次深度学习模型来做分类，先用轻量规则过滤掉明显的情况。

但启发式规则当然不是完美的——它会把一些复杂的截图误判为表格，或者把某些特殊格式的图表漏判为普通图片。所以在生产环境中，你可以在 `_classify_image_region` 方法中加入一个基于 CLIP 或轻量 CNN 的二次分类器来提高准确率。

### 图片理解与描述生成

当一张图片被提取出来后，我们需要让系统"理解"它说了什么。这是通过调用 GPT-4V 的视觉能力来实现的：

```python
import httpx
import base64
from typing import Optional


class ImageDescriber:
    """
    图片语义描述生成器

    使用 GPT-4V 对图片进行视觉理解，
    生成结构化的文本描述用于后续检索和问答。
    """

    def __init__(self, api_key: str, model: str = "gpt-4o", max_retries: int = 2):
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self.client = httpx.AsyncClient(timeout=120.0)

    async def describe(
        self,
        image_path: str,
        context_hint: Optional[str] = None,
        detail_level: str = "auto",
    ) -> dict:
        """
        生成图片的详细语义描述

        Args:
            image_path: 图片文件路径
            context_hint: 额外的上下文提示（如"这是第3页的架构图"）
            detail_level: OpenAI API 的 detail 参数 (low/high/auto)

        Returns:
            {
                "description": "自然语言描述",
                "key_elements": ["元素1", "元素2", ...],
                "text_content": "图中包含的文字 (OCR)",
                "type_inference": "inferred type (diagram/screenshot/photo/chart/table)",
                "structured_data": {...},  # 如果是图表/表格
            }
        """
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        prompt = self._build_description_prompt(context_hint)

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}",
                                "detail": detail_level,
                            },
                        },
                    ],
                }
            ],
            "max_tokens": 1000,
        }

        for attempt in range(self.max_retries + 1):
            try:
                response = await self.client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                response.raise_for_status()
                result = response.json()

                description_text = result["choices"][0]["message"]["content"]

                parsed = self._parse_description_response(description_text)
                parsed["raw_response"] = description_text
                parsed["image_path"] = image_path
                return parsed

            except Exception as e:
                if attempt < self.max_retries:
                    logger.warning(f"图片描述失败 (尝试 {attempt+1}/{self.max_retries+1}): {e}")
                    await asyncio.sleep(1 * (attempt + 1))
                else:
                    logger.error(f"图片描述最终失败: {image_path}, {e}")
                    return {
                        "description": f"[图片分析失败: {str(e)}]",
                        "key_elements": [],
                        "text_content": "",
                        "type_inference": "unknown",
                        "structured_data": {},
                    }

    def _build_description_prompt(self, context_hint: Optional[str]) -> str:
        """构建图片分析的 Prompt 模板"""
        base_prompt = """请仔细分析这张图片，并以 JSON 格式输出以下信息：

{
  "description": "用2-3句话详细描述图片的内容，包括主要对象、关系和关键细节",
  "key_elements": ["列出图片中的关键元素/对象/概念"],
  "text_content": "如果图片中包含可读文字，请完整转录",
  "type_inference": "推断图片的类型，可选值: diagram/screenshot/photo/chart/table/formula/icon/map/other",
  "structured_data": {
    "如果是图表": {"chart_type": "折线图/柱状图/饼图/散点图", "trends": "趋势描述"},
    "如果是表格": {"headers": ["列头"], "rows_count": 行数, "summary": "表格内容摘要"},
    "其他": {}
  }
}

要求：
- description 要足够详细，使仅凭这段文字就能理解图片的核心信息
- key_elements 要具体，便于后续检索匹配
- 如果图片质量太低无法辨认，请在 description 中说明"""

        if context_hint:
            base_prompt = f"背景信息: {context_hint}\n\n{base_prompt}"

        return base_prompt

    def _parse_description_response(self, text: str) -> dict:
        """解析 GPT-4V 返回的 JSON 描述"""
        import re

        try:
            json_match = re.search(r'\{[\s\S]*\}', text)
            if json_match:
                return json.loads(json_match.group())
        except (json.JSONDecodeError, IndexError):
            pass

        return {
            "description": text,
            "key_elements": [],
            "text_content": "",
            "type_inference": "unknown",
            "structured_data": {},
        }

    async def batch_describe(
        self,
        image_paths: list[str],
        concurrency: int = 5,
    ) -> list[dict]:
        """批量描述多张图片（并发控制）"""
        semaphore = asyncio.Semaphore(concurrency)
        results = []

        async def process_one(path):
            async with semaphore:
                return await self.describe(path)

        tasks = [process_one(p) for p in image_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        final_results = []
        for path, result in zip(image_paths, results):
            if isinstance(result, Exception):
                final_results.append({
                    "description": f"[错误: {result}]",
                    "image_path": path,
                })
            else:
                final_results.append(result)

        return final_results
```

`ImageDescriber` 是整个多模态管道中最关键也最昂贵的组件——每次调用 GPT-4V 分析一张图片的成本大约在 0.01-0.05 美元之间，延迟约 2-5 秒。所以它的设计有几个优化点：

**`detail_level` 参数**：OpenAI 的 Vision API 支持 `low`、`high`、`auto` 三种 detail 模式。`low` 模式使用较低的分辨率（更快更便宜），适合图标、logo 等简单图片；`high` 模式使用完整的分辨率并裁剪多个细节窗口（更慢更贵但更精确），适合需要读取密集文字的截图；`auto` 让模型自动选择。在我们的系统中，可以先根据图片的大小和预估复杂度来决定用哪种模式——小图片用 low，大图片用 auto，已知含密集文字的用 high。

**批量处理的并发控制**：`batch_describe` 方法使用 `asyncio.Semaphore` 来限制同时进行的 GPT-4V 调用数量。OpenAI 对 GPT-4V 有速率限制（通常 RPM 约 500），如果不控制并发可能在批量处理时触发 429 错误。`concurrency=5` 意味着最多同时有 5 个请求在飞，这是一个保守但安全的值。

**Prompt 工程的重要性**：`_build_description_prompt` 中的 prompt 设计直接影响返回结果的质量。我们要求模型以 JSON 格式输出（方便后续程序化处理）、要求 description 足够详细（确保检索时能匹配到）、要求提取关键元素（支持细粒度检索）。一个好的 prompt 可以把 GPT-4V 的准确率从 70% 提升到 90% 以上。

### 表格提取与结构化

表格是企业文档中承载结构化信息的核心载体——API 参数定义、配置项列表、状态码对照、版本差异对比都以表格形式存在。但表格也是最难处理的内容之一：

```python
import camelot
import pdfplumber
import pandas as pd
from io import StringIO


class TableExtractor:
    """
    表格提取器

    支持多种来源的表格提取：
    - PDF 内嵌表格（camelot/pdfplumber）
    - Excel/CSV 文件（pandas）
    - HTML 表格（BeautifulSoup）
    - 图片中的表格（GPT-4V + 结构化输出）
    """

    def __init__(self, preferred_engine: str = "camelot"):
        self.preferred_engine = preferred_engine

    def extract_from_pdf_page(self, page, page_num: int = 0) -> list[ParsedRegion]:
        """从 PDF 单页中提取表格"""
        tables = []

        if self.preferred_engine == "camelot":
            tables = self._extract_with_camelot(page, page_num)
        elif self.preferred_engine == "pdfplumber":
            tables = self._extract_with_pdfplumber(page, page_num)
        else:
            tables = self._extract_with_camelot(page, page_num)

        if not tables:
            tables = self._fallback_extract(page, page_num)

        return tables

    def _extract_with_camelot(self, page, page_num: int) -> list[ParsedRegion]:
        """使用 Camelot 提取 PDF 表格"""
        try:
            camelot_tables = camelot.read_pdf(
                str(page.parent.doc.name),
                pages=str(page_num + 1),
                flavor="lattice",  # lattice 更适合线条明确的表格
            )

            regions = []
            for i, table in enumerate(camelot_tables):
                df = table.df
                markdown_table = self._df_to_markdown(df)

                regions.append(ParsedRegion(
                    content_type=ContentType.TABLE,
                    content=markdown_table,
                    page_number=page_num,
                    confidence=table.accuracy,
                    metadata={
                        "rows": len(df),
                        "columns": len(df.columns),
                        "extraction_method": "camelot-lattice",
                        "accuracy": round(table.accuracy, 3),
                    },
                ))
            return regions
        except Exception as e:
            logger.warning(f"Camelot 提取失败 (page {page_num}): {e}")
            return []

    def _extract_with_pdfplumber(self, page, page_num: int) -> list[ParsedRegion]:
        """使用 pdfplumber 提取 PDF 表格（备用方案）"""
        try:
            import pdfplumber

            pdf = pdfplumber.open(str(page.parent.doc.name))
            pdf_page = pdf.pages[page_num]
            extracted_tables = pdf_page.extract_tables()

            regions = []
            for i, table in enumerate(extracted_tables):
                if not table or len(table) < 2:
                    continue

                df = pd.DataFrame(table[1:], columns=table[0])
                markdown_table = self._df_to_markdown(df)

                regions.append(ParsedRegion(
                    content_type=ContentType.TABLE,
                    content=markdown_table,
                    page_number=page_num,
                    metadata={
                        "rows": len(df),
                        "columns": len(df.columns),
                        "extraction_method": "pdfplumber",
                    },
                ))
            pdf.close()
            return regions
        except Exception as e:
            logger.warning(f"pdfplumber 提取失败 (page {page_num}): {e}")
            return []

    def _fallback_extract(self, page, page_num: int) -> list[ParsedRegion]:
        """回退方案：将页面上的规则排列文本块当作潜在表格"""
        text_dict = page.get_text("dict")
        blocks = text_dict["blocks"]

        text_blocks_by_y = []
        for block in blocks:
            if block["type"] == 0:
                for line in block["lines"]:
                    text = "".join([s["text"] for s in line["spans"]])
                    if text.strip():
                        y_coord = line["bbox"][1]
                        text_blocks_by_y.append((y_coord, text))

        if not text_blocks_by_y:
            return []

        text_blocks_by_y.sort(key=lambda x: x[0])

        lines_as_rows = []
        current_y = None
        current_line_texts = []

        for y, text in text_blocks_by_y:
            if current_y is None or abs(y - current_y) < 5:
                current_line_texts.append(text)
                current_y = y
            else:
                if len(current_line_texts) >= 2:
                    lines_as_rows.append(current_line_texts)
                current_line_texts = [text]
                current_y = y

        if len(current_line_texts) >= 2:
            lines_as_rows.append(current_line_texts)

        if len(lines_as_rows) >= 3:
            df = pd.DataFrame(lines_as_rows[1:], columns=lines_as_rows[0])
            markdown_table = self._df_to_markdown(df)
            return [ParsedRegion(
                content_type=ContentType.TABLE,
                content=markdown_table,
                page_number=page_num,
                confidence=0.5,
                metadata={"extraction_method": "heuristic-fallback"},
            )]

        return []

    def _df_to_markdown(self, df: pd.DataFrame) -> str:
        """将 DataFrame 转换为 Markdown 格式的表格字符串"""
        output = StringIO()
        df.to_markdown(buf=output, index=False, tablefmt="pipe")
        return output.getvalue().strip()

    def extract_from_excel(self, file_path: str, sheet_name: str = None) -> list[ParsedRegion]:
        """从 Excel 文件中提取表格"""
        regions = []

        if sheet_name:
            sheets = [sheet_name]
        else:
            xl = pd.ExcelFile(file_path)
            sheets = xl.sheet_names

        for sheet in sheets:
            try:
                df = pd.read_excel(file_path, sheet_name=sheet)
                markdown_table = self._df_to_markdown(df)

                regions.append(ParsedRegion(
                    content_type=ContentType.TABLE,
                    content=markdown_table,
                    metadata={
                        "source_file": file_path,
                        "sheet_name": sheet,
                        "rows": len(df),
                        "columns": len(df.columns),
                        "extraction_method": "pandas-excel",
                    },
                ))
            except Exception as e:
                logger.warning(f"Excel sheet '{sheet}' 提取失败: {e}")

        return regions

    def extract_from_image(self, image_path: str, describer: ImageDescriber) -> ParsedRegion:
        """
        从图片中提取表格（适用于扫描版 PDF 或截图中的表格）

        使用 GPT-4V 来识别图片中的表格结构
        """
        prompt_override = """这是一张包含表格的图片。
请以严格的 JSON 格式输出：
{
  "table_headers": ["列1标题", "列2标题", ...],
  "table_rows": [["行1列1", "行1列2", ...], ["行2列1", ...]],
  "table_title": "表格标题（如果有）",
  "notes": "表格下方的注释（如果有）"
}

要求：
- 准确识别所有单元格的内容
- 保持原始的语言（不要翻译）
- 如果单元格为空，用空字符串表示
- 合并单元格的内容只在第一个出现的格子中填写"""

        result = await describer.describe(
            image_path=image_path,
            context_hint="这是一张表格图片，请提取表格结构",
        )

        if result.get("structured_data") and "table_headers" in result.get("structured_data", {}):
            sd = result["structured_data"]
            headers = sd.get("table_headers", [])
            rows = sd.get("table_rows", [])

            if headers and rows:
                df = pd.DataFrame(rows, columns=headers)
                markdown_table = self._df_to_markdown(df)

                return ParsedRegion(
                    content_type=ContentType.TABLE,
                    content=markdown_table,
                    raw_data=result.get("raw_data"),
                    image_path=image_path,
                    confidence=0.8,
                    metadata={
                        "extraction_method": "gpt4v-vision",
                        "title": sd.get("table_title", ""),
                        "notes": sd.get("notes", ""),
                    },
                )

        return ParsedRegion(
            content_type=ContentType.TABLE,
            content=result.get("description", "[无法提取表格结构]"),
            image_path=image_path,
            confidence=0.3,
            metadata={"extraction_method": "gpt4v-fallback"},
        )
```

表格提取是一个经典的"没有银弹"的问题——每种方法都有自己的适用场景和盲区：

| 方法 | 适用场景 | 优势 | 劣势 |
|------|---------|------|------|
| Camelot (lattice) | 有明确边线的表格 | 高精度，保留结构 | 无法处理无边线表格 |
| Camelot (stream) | 无边线但有规律的表格 | 适应性好 | 复杂排版容易出错 |
| pdfplumber | 一般性 PDF 表格 | 简单易用 | 精度一般 |
| GPT-4V Vision | 扫描件/截图/手写表格 | 最灵活 | 成本最高，速度最慢 |

实际生产中建议采用**级联策略**：先用 Camelot 尝试（快且准），失败后 fallback 到 pdfplumber（覆盖更多情况），再失败才用 GPT-4V（兜底但贵）。上面的 `TableExtractor.extract_from_pdf_page()` 就实现了这个逻辑。

## 统一多模态数据加载管道

有了各种解析器之后，最后需要一个统一的管道来编排整个加载流程：

```python
class MultiModalDataPipeline:
    """统一的多模态数据处理管道"""

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.image_describer = ImageDescriber(
            api_key=self.config.get("openai_api_key", ""),
        )
        self.table_extractor = TableExtractor(
            preferred_engine=self.config.get("table_engine", "camelot"),
        )
        self.pdf_parser = PDFMultiModalParser(
            image_output_dir=self.config.get("image_output_dir", "./data/images"),
        )

    async def process_document(self, file_path: str) -> ParsedDocument:
        """
        完整的多模态文档处理流水线

        流程:
        1. 根据文件类型选择解析器
        2. 提取各区域的原始内容和元数据
        3. 对图片区域生成语义描述
        4. 对表格区域做结构化提取
        5. 返回统一的 ParsedDocument 结果
        """
        ext = Path(file_path).suffix.lower()

        if ext == ".pdf":
            parsed = self.pdf_parser.parse(file_path)
        elif ext in [".xlsx", ".xls"]:
            tables = self.table_extractor.extract_from_excel(file_path)
            parsed = ParsedDocument(
                source_path=file_path,
                regions=tables,
                table_regions=tables,
            )
        elif ext in [".png", ".jpg", ".jpeg", ".webp", ".gif"]:
            desc = await self.image_describer.describe(file_path)
            region = ParsedRegion(
                content_type=ContentType.IMAGE,
                content=desc.get("description", ""),
                image_path=file_path,
                metadata=desc,
            )
            parsed = ParsedDocument(source_path=file_path, regions=[region], image_regions=[region])
        else:
            raise ValueError(f"不支持的文件格式: {ext}")

        # 后处理：对图片区域补充语义描述
        if parsed.image_regions:
            need_describe = [
                r for r in parsed.image_regions
                if not r.metadata.get("description_generated")
            ]
            if need_describe:
                descriptions = await self.image_describer.batch_describe(
                    [r.image_path for r in need_describe if r.image_path]
                )
                for region, desc in zip(need_describe, descriptions):
                    region.content = desc.get("description", region.content)
                    region.metadata.update({
                        "description_generated": True,
                        "key_elements": desc.get("key_elements", []),
                        "type_inference": desc.get("type_inference", ""),
                    })

        return parsed
```

## 总结

