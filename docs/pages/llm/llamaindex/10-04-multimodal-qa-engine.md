# 10.4 多模态问答引擎实现

## 把所有组件组装成完整的问答系统

前三节我们分别完成了多模态数据的接入解析（10.2）、跨模态检索与融合策略（10.3）。现在到了最关键的一步——把这些零件组装成一个能实际工作的多模态问答引擎。这一节要实现的是 `MultiModalQAEngine`，它是整个项目的核心类，负责协调数据管道、检索器、多模态 LLM 和响应格式化器，对外提供简洁的 `query()` 接口。

## 引擎主类：MultiModalQAEngine

```python
import time
import base64
import asyncio
from typing import Optional, List
from dataclasses import dataclass, field
from enum import Enum

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore, ImageNode, TextNode
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

from app.multimodal.retriever import MultiModalRetriever, RetrievalMode
from app.multimodal.data_pipeline import MultiModalDataPipeline, ParsedDocument
from app.multimodal.image_describer import ImageDescriber


class AnswerSource(Enum):
    TEXT_ONLY = "text_only"
    IMAGE_ONLY = "image_only"
    MIXED = "mixed"


@dataclass
class MultiModalQueryResult:
    """多模态查询结果"""
    answer: str
    sources: List[dict] = field(default_factory=list)
    image_citations: List[dict] = field(default_factory=list)
    table_references: List[dict] = field(default_factory=list)
    confidence: float = 0.0
    latency_ms: float = 0.0
    source_type: AnswerSource = AnswerSource.TEXT_ONLY
    retrieval_details: dict = field(default_factory=dict)


@dataclass
class MultiModalConfig:
    """多模态 RAG 配置"""
    vision_model: str = "gpt-4o"
    text_model: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-large"
    clip_model: str = "ViT-B/32"
    max_images_in_context: int = 5
    max_tokens_answer: int = 2048
    retrieval_mode: RetrievalMode = RetrievalMode.ADAPTIVE
    text_top_k: int = 10
    image_top_k: int = 5
    enable_rerank: bool = True


class MultiModalQAEngine:
    """
    多模态问答引擎

    完整的问答流水线:
    1. 查询分析 (文本/图片输入 → 意图识别)
    2. 多模态检索 (文本索引 + 图片索引 → 融合结果)
    3. 上下文组装 (文本节点 + 图片节点 + 原始图片数据)
    4. 多模态生成 (GPT-4V 理解图片 + 生成回答)
    5. 结果格式化 (引用、置信度、来源标注)
    """

    def __init__(self, config: Optional[MultiModalConfig] = None):
        self.config = config or MultiModalConfig()
        self._initialize_models()
        self._retriever: Optional[MultiModalRetriever] = None
        self._data_pipeline: Optional[MultiModalDataPipeline] = None

    def _initialize_models(self):
        """初始化所有模型"""
        self.vision_llm = OpenAIMultiModal(model=self.config.vision_model)
        self.text_llm = OpenAI(model=self.config.text_model, temperature=0.1)

    def set_retriever(self, retriever: MultiModalRetriever):
        """设置检索器"""
        self._retriever = retriever

    def set_data_pipeline(self, pipeline: MultiModalDataPipeline):
        """设置数据处理管道"""
        self._data_pipeline = pipeline

    async def query(
        self,
        question: str,
        image_input: Optional[str] = None,
        image_bytes: Optional[bytes] = None,
        stream: bool = False,
    ) -> MultiModalQueryResult:
        """
        执行多模态问答

        Args:
            question: 用户问题（文本）
            image_input: 用户上传的图片路径
            image_bytes: 用户上传的图片二进制数据
            stream: 是否流式输出
        """
        start_time = time.perf_counter()

        if not self._retriever:
            raise RuntimeError("检索器未初始化，请先调用 set_retriever()")

        # Step 1: 执行多模态检索
        retrieval_result = await self._retriever.retrieve(
            query=question,
            mode=self.config.retrieval_mode,
            query_image=image_input,
        )

        fused_nodes = retrieval_result.fused_results or []
        text_nodes = [n for n in fused_nodes if isinstance(n.node, TextNode)]
        image_nodes = [n for n in fused_nodes if isinstance(n.node, ImageNode)]

        # Step 2: 组装多模态上下文
        context = self._build_multimodal_context(
            question=question,
            text_nodes=text_nodes,
            image_nodes=image_nodes[:self.config.max_images_in_context],
            user_image_path=image_input,
            user_image_bytes=image_bytes,
        )

        # Step 3: 调用多模态 LLM 生成回答
        answer = await self._generate_answer(context)

        # Step 4: 构建引用和来源信息
        latency = (time.perf_counter() - start_time) * 1000

        source_type = AnswerSource.TEXT_ONLY
        if image_nodes and text_nodes:
            source_type = AnswerSource.MIXED
        elif image_nodes and not text_nodes:
            source_type = AnswerSource.IMAGE_ONLY

        result = MultiModalQueryResult(
            answer=answer["response"],
            sources=self._format_sources(text_nodes, image_nodes),
            image_citations=answer.get("image_citations", []),
            table_references=self._extract_table_refs(text_nodes),
            confidence=answer.get("confidence", 0.5),
            latency_ms=round(latency, 1),
            source_type=source_type,
            retrieval_details={
                "query_type": retrieval_result.query_type,
                "mode_used": retrieval_result.mode_used.value,
                "text_results": len(text_nodes),
                "image_results": len(image_nodes),
                "total_fused": len(fused_nodes),
            },
        )

        return result

    def _build_multimodal_context(
        self,
        question: str,
        text_nodes: List[NodeWithScore],
        image_nodes: List[NodeWithScore],
        user_image_path: Optional[str],
        user_image_bytes: Optional[bytes],
    ) -> dict:
        """
        组装发送给 GPT-4V 的多模态上下文

        GPT-4V 的 messages 格式要求：
        - 文本内容放在 content 数组的 text 类型元素中
        - 图片内容放在 content 数组的 image_url 类型元素中（base64 编码）
        - 图片和文字可以交错排列以保持上下文关联性
        """
        context_parts = []

        # 用户问题
        context_parts.append({
            "type": "text",
            "text": f"## 用户问题\n{question}",
        })

        # 用户上传的图片（如果有）
        if user_image_path or user_image_bytes:
            context_parts.append({
                "type": "text",
                "text": "\n## 用户提供的参考图片\n请仔细查看用户上传的图片，结合下方的知识库内容来回答问题。",
            })
            if user_image_bytes:
                b64 = base64.b64encode(user_image_bytes).decode()
            else:
                with open(user_image_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()

            context_parts.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}",
                    "detail": "high",
                },
            })

        # 检索到的文本内容
        if text_nodes:
            context_parts.append({
                "type": "text",
                "text": f"\n## 来自知识库的文本参考\n\n"
                        + "\n\n---\n\n".join([
                            f"[相关度: {n.score:.3f}] {n.node.text[:500]}"
                            for n in text_nodes[:8]
                        ]),
            })

        # 检索到的图片内容
        if image_nodes:
            context_parts.append({
                "type": "text",
                "text": f"\n## 来自知识库的图片参考（共 {len(image_nodes)} 张）",
            })

            for i, img_node in enumerate(image_nodes):
                img_node_obj = img_node.node
                desc = img_node_obj.metadata.get("description", "")
                source = img_node_obj.metadata.get("source_file", "未知来源")
                page = img_node_obj.metadata.get("page", "")

                context_parts.append({
                    "type": "text",
                    "text": f"\n### 图片 {i+1}: {desc}\n来源: {source} 第{page}页",
                })

                if hasattr(img_node_obj, 'image') and img_node_obj.image:
                    b64 = base64.b64encode(img_node_obj.image).decode()
                elif hasattr(img_node_obj, 'image_path') and img_node_obj.image_path:
                    try:
                        with open(img_node_obj.image_path, "rb") as f:
                            b64 = base64.b64encode(f.read()).decode()
                    except Exception:
                        continue
                else:
                    continue

                context_parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64}",
                        "detail": "auto",
                    },
                })

        return {
            "question": question,
            "context_parts": context_parts,
            "text_node_count": len(text_nodes),
            "image_node_count": len(image_nodes),
        }

    async def _generate_answer(self, context: dict) -> dict:
        """使用 GPT-4V 生成多模态回答"""
        system_prompt = """你是一个专业的多模态知识库问答助手。
你的任务是基于提供的文本资料和图片来准确回答用户的问题。

核心原则：
1. **严格基于给定材料**：只使用下方提供的文本和图片中的信息作答，不要编造或推测
2. **明确标注图片来源**：如果答案主要依据某张图片，请在回答中注明"根据图片 X 所示..."
3. **图文结合**：当文字和图片都包含相关信息时，综合两者给出完整回答
4. **处理不确定性**：如果材料中没有足够的信息回答问题，明确说明而不是猜测
5. **结构清晰**：对于复杂问题，用分点或分步骤的方式组织回答

回答格式要求：
- 直接回答问题，不要有过多寒暄
- 如果涉及具体数值或步骤，尽量精确
- 如果参考了图片，用 [图片N] 的形式标注"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context["context_parts"]},
        ]

        try:
            response = await self.vision_llm.acomplete(
                prompt=context["context_parts"],
                image_documents=[],
                system_prompt=system_prompt,
            )
            answer_text = str(response) if response else "抱歉，无法生成回答。"

            return {
                "response": answer_text,
                "confidence": self._estimate_confidence(context, answer_text),
                "image_citations": self._extract_image_citations(answer_text, context),
            }

        except Exception as e:
            logger.error(f"多模态 LLM 调用失败: {e}")
            return {
                "response": f"抱歉，在生成回答时遇到了问题: {str(e)}",
                "confidence": 0.0,
                "image_citations": [],
            }

    def _estimate_confidence(self, context: dict, answer: str) -> float:
        """估算回答置信度"""
        score = 0.5

        node_count = context.get("text_node_count", 0) + context.get("image_node_count", 0)
        if node_count >= 5:
            score += 0.15
        elif node_count >= 2:
            score += 0.08

        if len(answer) > 100:
            score += 0.1
        if len(answer) < 20:
            score -= 0.15

        if context.get("image_node_count", 0) > 0:
            score += 0.05

        return min(max(score, 0.0), 1.0)

    def _extract_image_citations(self, answer: str, context: dict) -> list[dict]:
        """从回答中提取图片引用"""
        import re
        citations = []
        pattern = r'\[图片(\d+)\]'
        matches = re.findall(pattern, answer)

        image_nodes = [n for n in (getattr(self._retriever, '_last_image_results', []) or [])
                     if isinstance(n.node, ImageNode)]

        for match in matches:
            idx = int(match) - 1
            if 0 <= idx < len(image_nodes):
                node = image_nodes[idx].node
                citations.append({
                    "index": int(match),
                    "description": node.metadata.get("description", ""),
                    "source": node.metadata.get("source_file", ""),
                    "page": node.metadata.get("page", ""),
                    "image_path": getattr(node, 'image_path', None),
                })

        return citations

    def _format_sources(self, text_nodes, image_nodes) -> list[dict]:
        """格式化来源列表"""
        sources = []

        for n in text_nodes[:5]:
            sources.append({
                "type": "text",
                "title": n.node.metadata.get("file_name", n.node.metadata.get("source_file", "")),
                "content_preview": n.node.text[:150],
                "score": round(n.score, 3),
                "page": n.node.metadata.get("page", ""),
            })

        for n in image_nodes[:3]:
            sources.append({
                "type": "image",
                "title": n.node.metadata.get("description", "")[:80] or "图片",
                "source": n.node.metadata.get("source_file", ""),
                "score": round(n.score, 3),
                "page": n.node.metadata.get("page", ""),
                "image_path": getattr(n.node, 'image_path', None),
            })

        return sources

    def _extract_table_refs(self, text_nodes) -> list[dict]:
        """从文本节点中提取表格引用"""
        tables = []
        for n in text_nodes:
            if n.node.metadata.get("content_type") == "table":
                tables.append({
                    "content": n.node.text[:500],
                    "source": n.node.metadata.get("source_file", ""),
                    "rows": n.node.metadata.get("rows", "?"),
                    "columns": n.node.metadata.get("columns", "?"),
                })
        return tables
```

`MultiModalQAEngine.query()` 方法是多模态问答的核心入口。它的执行流程可以概括为以下五个阶段：

**阶段一：多模态检索**。通过 `MultiModalRetriever` 同时搜索文本索引和图片索引。返回的结果是融合排序后的 Node 列表——可能同时包含 TextNode 和 ImageNode。

**阶段二：上下文组装**。这是与传统 RAG 最不同的地方。GPT-4V 要求输入采用特定的 multimodal messages 格式——一个 content 数组中交错排列 text 类型和 image_url 类型的元素。我们需要把检索到的文本内容作为 text 元素、把图片节点的原始图片数据（base64 编码后）作为 image_url 元素，按照逻辑顺序组装在一起。注意每张图片前面都附上一段描述性文字（如"### 图片 1: 系统架构拓扑图 来源: api-docs.pdf 第3页"），这有助于 GPT-4V 理解每张图片在整体上下文中的角色。

**阶段三：多模态生成**。将组装好的上下文发给 GPT-4V。这里的 system prompt 特别强调了几个要点：严格基于给定材料（防止幻觉）、明确标注图片来源（可追溯性）、图文结合回答（充分利用多模态信息）、结构化组织（提高可读性）。

**阶段四：后处理**。包括置信度估算、图片引用提取、来源格式化等。图片引用提取通过正则匹配 `[图片N]` 标记来实现——这依赖于 GPT-4V 在回答时遵循我们在 prompt 中要求的标注格式。

## 支持图片上传的 API 端点

```python
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
import tempfile
import os

router = APIRouter(prefix="/api/v1/multimodal", tags=["multimodal"])


class MultimodalChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    stream: bool = False


@router.post("/chat")
async def multimodal_chat(
    message: str = Form(...),
    file: Optional[UploadFile] = File(None),
    session_id: Optional[str] = Form(None),
    stream: bool = Form(False),
    current_user: dict = Depends(get_current_user),
):
    """
    多模态问答端点

    支持两种输入模式：
    1. 纯文本提问 — 行为与传统 RAG 相同
    2. 文本 + 图片 — 图片可以是截图/照片/图表等任何格式

    使用 multipart/form-data 格式提交（支持文件上传）
    """
    global mm_engine

    image_path = None
    image_bytes = None

    if file:
        allowed_extensions = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
        file_ext = os.path.splitext(file.filename)[1].lower()

        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的图片格式: {file_ext}，允许: {allowed_extensions}",
            )

        if file.size > 20 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="图片大小不能超过 20MB")

        contents = await file.read()
        image_bytes = contents

        temp_dir = tempfile.mkdtemp(prefix="mm_upload_")
        image_path = os.path.join(temp_dir, f"{file.filename}")

        with open(image_path, "wb") as f:
            f.write(contents)

    try:
        result = await mm_engine.query(
            question=message,
            image_input=image_path,
            image_bytes=image_bytes,
            stream=stream,
        )

        return {
            "answer": result.answer,
            "session_id": session_id or f"mm_{id(result)}",
            "sources": result.sources,
            "image_citations": result.image_citations,
            "table_references": result.table_references,
            "confidence": result.confidence,
            "latency_ms": result.latency_ms,
            "source_type": result.source_type.value,
            "retrieval": result.retrieval_details,
        }

    finally:
        if image_path and os.path.exists(image_path):
            os.unlink(image_path)
            parent_dir = os.path.dirname(image_path)
            if os.path.isdir(parent_dir) and parent_dir.startswith(tempfile.gettempdir()):
                os.rmdir(parent_dir)


@router.post("/analyze-image")
async def analyze_standalone_image(
    file: UploadFile = File(...),
    question: str = Form("请描述这张图片的内容"),
    detail: str = Form("auto"),
    current_user: dict = Depends(get_current_user),
):
    """
    独立图片分析端点

    不走 RAG 检索流程，直接让 GPT-4V 分析用户上传的图片。
    适用于："帮我看看这个报错""这是什么图"等场景。
    """
    from app.multimodal.image_describer import ImageDescriber

    contents = await file.read()
    temp_path = f"/tmp/mm_analyze_{id(file)}_{file.filename}"

    with open(temp_path, "wb") as f:
        f.write(contents)

    try:
        describer = ImageDescriber(api_key=settings.OPENAI_API_KEY)
        result = await describer.describe(
            image_path=temp_path,
            context_hint=f"用户问题: {question}",
            detail_level=detail,
        )
        return result
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
```

两个 API 端点的区别值得说明：

**`/chat` 端点**是完整的 RAG 流程——用户的图片（如果有）会作为额外的查询条件参与检索，系统会在知识库中找相关的文本和图片，然后把所有相关信息一起送给 GPT-4V 生成回答。适用于"结合文档和我的截图帮我排查这个问题"的场景。

**`/analyze-image` 端点**是轻量级的纯图片分析——不走检索流程，直接把用户上传的图片发给 GPT-4V 分析。适用于快速查看场景——用户只是想让 AI 描述一下图片内容、读取图片中的文字或者识别图片中的物体。这个端点的响应速度更快（省去了检索步骤），成本也更低（不需要 embedding 和向量搜索）。

## 前端集成要点

多模态问答的前端需要在传统聊天界面的基础上增加图片上传能力：

```javascript
// 关键的前端交互代码片段

// 1. 图片上传区域（拖拽 + 点击）
const dropZone = document.getElementById('drop-zone');
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
});
dropZone.addEventListener('drop', async (e) => {
    e.preventDefault();
    const files = e.dataTransfer.files;
    if (files.length > 0 && files[0].type.startsWith('image/')) {
        await handleImageUpload(files[0]);
    }
});

// 2. 发送带图片的多模态请求
async function sendMultimodalMessage(text, imageFile) {
    const formData = new FormData();
    formData.append('message', text);
    if (imageFile) {
        formData.append('file', imageFile);
    }
    formData.append('session_id', currentSessionId);
    formData.append('stream', 'false');

    const response = await fetch('/api/v1/multimodal/chat', {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${getToken()}` },
        body: formData,  // 不要设 Content-Type，浏览器自动设置 boundary
    });

    const data = await response.json();

    // 渲染回答（含图片引用卡片）
    renderAnswer(data);
}

// 3. 渲染图片引用卡片
function renderImageCitation(citation) {
    return `
        <div class="image-citation-card">
            <div class="citation-header">📷 参考图片 #${citation.index}</div>
            ${citation.image_path
                ? `<img src="/api/images/${encodeURIComponent(citation.image_path)}"
                         class="citation-thumbnail"
                         onclick="showFullImage('${citation.image_path}')">`
                : '<div class="no-preview">预览不可用</div>'}
            <div class="citation-meta">
                来源: ${citation.source}
                ${citation.page ? `第${citation.page}页` : ''}
            </div>
            <div class="citation-desc">${citation.description}</div>
        </div>
    `;
}
```

前端的关键设计考量是**图片预览体验**。当系统返回的答案中引用了知识库中的图片时，前端应该能够展示这些图片的缩略图，点击后可以放大查看原图。这需要后端提供一个静态文件服务端点来安全地提供图片访问（不能直接暴露文件系统路径）。

## 总结

