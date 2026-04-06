# 10.3 多模态检索与融合

## 让文本和图片在同一个空间里"对话"

上一节我们解决了多模态数据的接入和解析问题——PDF 被拆解成了文本块、图片、表格和图表，每种元素都有了结构化的表示。现在的问题是：当用户提出一个问题时，系统怎么知道应该去检索哪种模态的信息？更根本的问题是，**文本和图片存在于完全不同的表征空间中，如何让它们能够被统一地检索和比较？**

这一节的核心就是回答这个问题。我们会从 CLIP 模型的图文对齐原理讲起，逐步构建一个支持跨模态语义检索的完整方案，包括多模态 embedding、混合索引策略、以及查询时的智能路由机制。

## CLIP：连接文本和图片的桥梁

要理解多模态检索的技术基础，必须先理解 CLIP（Contrastive Language-Image Pre-training）模型。CLIP 是 OpenAI 在 2021 年发布的一个开创性模型，它的训练思路简洁而优雅：

```
CLIP 训练原理（简化版）：

训练数据: 4亿对 (图片, 文本描述) 对
例如:
  (一张猫的照片, "a photo of a cat sitting on a windowsill")
  (一张架构图, "system architecture diagram showing microservices")
  (一张折线图, "sales revenue growth from Q1 to Q4")

训练目标: 让匹配的(图,文)对在向量空间中距离近，
         不匹配的(图,文)对距离远

训练后的效果:
  文本向量 "一只猫坐在窗台上"  ←→ 很近 →  图片向量 [猫的照片]
  文本向量 "销售增长趋势图"    ←→ 很近 →  图片向量 [折线图]
  文本向量 "一只猫坐在窗台上"  ←→ 很远 →  图片向量 [折线图]
```

CLIP 的关键洞察是：**不需要显式地标注图片里有什么物体、什么颜色、什么位置——只需要大量的图文配对数据，模型自己就能学会把语义相近的文本和图片映射到嵌入空间的相邻区域**。这意味着我们可以用同一段文字去同时检索文本文档和相关图片——因为它们在这个共享空间中的位置是接近的。

LlamaIndex 对 CLIP 有原生支持：

```python
from llama_index.embeddings.clip import ClipEmbedding
from llama_index.multi_modal_llms.openai import OpenAIMultiModal

# CLIP Embedding —— 同时处理文本和图片
clip_embedding = ClipEmbedding(
    model_name="ViT-B/32",      # 可选: ViT-B/32, ViT-B/16, ViT-L/14
    embed_batch_size=10,
)

# 文本 embedding（和普通 text-embedding 用法一样）
text_embedding = clip_embedding.get_text_embedding("架构图展示了系统的微服务设计")

# 图片 embedding（传入图片路径或 bytes）
image_embedding = clip_embedding.get_image_embedding("./images/architecture.png")

# 两者都在同一个 512 维的空间中！
import numpy as np
similarity = np.dot(text_embedding, image_embedding) / (
    np.linalg.norm(text_embedding) * np.linalg.norm(image_embedding)
)
print(f"图文相似度: {similarity:.4f}")
# 输出可能类似: 图文相似度: 0.8234 (如果确实相关)
```

但这里有一个重要的实际考量：**CLIP 的 ViT-B/32 模型的 embedding 维度只有 512 维，且其语义理解能力远不如 GPT 系列的 text-embedding-3-large（3072 维）**。在实际项目中，我们通常采用双 embedding 策略：文本用强大的 OpenAI Embedding 做精确匹配，图片用 CLIP 做跨模态桥接。

## 多模态索引策略

```python
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core import Document, ImageDocument
from llama_index.core.schema import TextNode, ImageNode
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.clip import ClipEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from typing import List


class MultiModalIndexBuilder:
    """
    多模态索引构建器

    核心策略：
    - 文本节点用 OpenAI Embedding (高精度)
    - 图片节点用 CLIP Embedding (跨模态能力)
    - 两者存储在同一 Qdrant Collection 中
    - 通过 metadata 的 content_type 字段区分
    """

    def __init__(self, config: dict):
        self.qdrant_client = QdrantClient(url=config.get("qdrant_url", "http://localhost:6333"))
        self.collection_name = config.get("collection_name", "multimodal_kb")

        self.text_embed_model = OpenAIEmbedding(
            model="text-embedding-3-large",
            dimensions=1024,
        )
        self.image_embed_model = ClipEmbedding(
            model_name="ViT-B/32",
        )

    def build_index(self, parsed_docs: List[ParsedDocument]) -> dict:
        """从解析结果构建多模态向量索引"""
        all_text_nodes = []
        all_image_nodes = []

        for parsed in parsed_docs:
            for region in parsed.regions:
                if region.content_type in (ContentType.TEXT, ContentType.TABLE):
                    node = TextNode(
                        text=region.content,
                        metadata={
                            **region.metadata,
                            "content_type": region.content_type.value,
                            "source_file": parsed.source_path,
                            "page": str(region.page_number),
                        },
                    )
                    all_text_nodes.append(node)

                elif region.content_type == ContentType.IMAGE:
                    if region.raw_data or region.image_path:
                        img_node = ImageNode(
                            image=region.raw_data,
                            image_path=region.image_path,
                            metadata={
                                **region.metadata,
                                "content_type": "image",
                                "description": region.content,
                                "source_file": parsed.source_path,
                                "page": str(region.page_number),
                            },
                        )
                        all_image_nodes.append(img_node)

                elif region.content_type == ContentType.CHART:
                    # 图表同时创建文本节点（描述）和图片节点（原始图）
                    text_node = TextNode(
                        text=region.content,
                        metadata={
                            **region.metadata,
                            "content_type": "chart_description",
                            "source_file": parsed.source_path,
                        },
                    )
                    all_text_nodes.append(text_node)

                    if region.raw_data or region.image_path:
                        img_node = ImageNode(
                            image=region.raw_data,
                            image_path=region.image_path,
                            metadata={
                                **region.metadata,
                                "content_type": "chart",
                                "source_file": parsed.source_path,
                            },
                        )
                        all_image_nodes.append(img_node)

        print(f"准备索引: {len(all_text_nodes)} 个文本节点 + {len(all_image_nodes)} 个图片节点")

        # 创建 Qdrant 向量存储
        vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=self.collection_name,
        )

        # 构建纯文本索引
        text_index = None
        if all_text_nodes:
            from llama_index.core import Settings
            Settings.embed_model = self.text_embed_model

            text_documents = [
                Document(text=n.text, metadata=n.metadata)
                for n in all_text_nodes
            ]
            text_index = VectorStoreIndex.from_documents(
                text_documents,
                storage_context=StorageContext.from_defaults(vector_store=vector_store),
                show_progress=True,
            )

        # 将图片节点追加到同一索引
        if all_image_nodes and text_index:
            for img_node in all_image_nodes:
                text_index.insert(img_node)

        stats = {
            "status": "success",
            "text_nodes": len(all_text_nodes),
            "image_nodes": len(all_image_nodes),
            "total_nodes": len(all_text_nodes) + len(all_image_nodes),
            "collection": self.collection_name,
        }

        return stats
```

上面的代码展示了一个关键的工程决策：**将文本节点和图片节点放在同一个 Qdrant Collection 中**。这样做的好处是查询时可以用一次搜索同时获取相关的文本和图片结果，不需要分别查两个集合再合并。代价是文本 embedding 和图片 embedding 的维度不同（OpenAI 是 1024 维，CLIP 是 512 维），Qdrant 需要在同一个 collection 中处理不同维度的向量——这可以通过 Qdrant 的 named vectors 功能来实现（每个向量有一个名称标识，如 `text_vector` 和 `image_vector`）。

不过 LlamaIndex 目前的抽象层对 named vectors 的支持还不够完善，所以在生产环境中更稳妥的做法是使用**两个独立的 collection**：`kb_text` 存文本节点（用 OpenAI embedding）、`kb_image` 存图片节点（用 CLIP embedding）。查询时并行搜索两个 collection 再合并结果。

## 多模态检索器：MultiModalRetriever

```python
import asyncio
from typing import List, Optional
from dataclasses import dataclass
from enum import Enum

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode, ImageNode
from llama_index.embeddings.clip import ClipEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding


class RetrievalMode(Enum):
    TEXT_ONLY = "text_only"
    IMAGE_ONLY = "image_only"
    FUSED = "fused"
    ADAPTIVE = "adaptive"


@dataclass
class MultiModalResult:
    """多模态检索结果"""
    text_results: List[NodeWithScore] = None
    image_results: List[NodeWithScore] = None
    fused_results: List[NodeWithScore] = None
    query_type: str = ""
    mode_used: RetrievalMode = RetrievalMode.FUSED


class MultiModalRetriever:
    """
    多模态检索器

    支持三种检索模式：
    1. TEXT_ONLY — 只检索文本（传统 RAG）
    2. IMAGE_ONLY — 只检索图片（用户明确需要视觉信息）
    3. FUSED — 融合检索（同时搜索文本和图片，合并排序）
    4. ADAPTIVE — 自适应模式（根据查询内容自动选择策略）
    """

    def __init__(
        self,
        text_index: VectorStoreIndex,
        image_index: Optional[VectorStoreIndex] = None,
        clip_model: str = "ViT-B/32",
        text_top_k: int = 10,
        image_top_k: int = 5,
        fusion_strategy: str = "rrf",
    ):
        self.text_retriever = text_index.as_retriever(similarity_top_k=text_top_k)
        self.image_retriever = None
        self.text_top_k = text_top_k
        self.image_top_k = image_top_k
        self.fusion_strategy = fusion_strategy

        if image_index:
            self.clip_embedding = ClipEmbedding(model_name=clip_model)
            self.image_retriever = image_index.as_retriever(similarity_top_k=image_top_k)

    async def retrieve(
        self,
        query: str,
        mode: RetrievalMode = RetrievalMode.ADAPTIVE,
        query_image: Optional[str] = None,
    ) -> MultiModalResult:
        """
        执行多模态检索

        Args:
            query: 用户的问题文本
            mode: 检索模式
            query_image: 用户上传的参考图片（可选）
        """
        query_type = self._classify_query_intent(query, query_image)

        if mode == RetrievalMode.ADAPTIVE:
            mode = self._determine_mode(query_type)

        tasks = {}

        if mode in (RetrievalMode.TEXT_ONLY, RetrievalMode.FUSED, RetrievalMode.ADAPTIVE):
            tasks["text"] = asyncio.create_task(
                self._retrieve_text(query)
            )

        if mode in (RetrievalMode.IMAGE_ONLY, RetrievalMode.FUSED) and self.image_retriever:
            if query_image:
                tasks["image"] = asyncio.create_task(
                    self._retrieve_image_by_reference(query, query_image)
                )
            else:
                tasks["image"] = asyncio.create_task(
                    self._retrieve_image_by_text(query)
                )

        results = {}
        if tasks:
            done_tasks = await asyncio.gather(*tasks.values(), return_exceptions=True)
            for key, result in zip(tasks.keys(), done_tasks):
                if isinstance(result, Exception):
                    logger.error(f"{key} 检索失败: {result}")
                    results[key] = []
                else:
                    results[key] = result

        text_results = results.get("text", [])
        image_results = results.get("image", [])

        if mode == RetrievalMode.FUSED and text_results and image_results:
            fused = self._fuse_results(text_results, image_results, query)
        elif mode == RetrievalMode.TEXT_ONLY:
            fused = text_results
        elif mode == RetrievalMode.IMAGE_ONLY:
            fused = image_results
        else:
            fused = (text_results or []) + (image_results or [])

        return MultiModalResult(
            text_results=text_results,
            image_results=image_results,
            fused_results=fused[:self.text_top_k],
            query_type=query_type,
            mode_used=mode,
        )

    def _classify_query_intent(self, query: str, has_image: bool) -> str:
        """判断用户的查询意图类型"""
        query_lower = query.lower()

        visual_indicators = [
            "图片", "截图", "照片", "图", "样例", "示例",
            "长什么样", "看起来", "界面", "布局",
            "image", "screenshot", "photo", "show me",
            "diagram", "chart", "graph", "figure",
        ]

        table_indicators = [
            "表格", "表", "参数", "字段", "配置项",
            "对比", "差异", "table", "parameter",
            "column", "config",
        ]

        chart_indicators = [
            "趋势", "增长", "下降", "占比", "分布",
            "chart", "graph", "trend", "统计",
        ]

        has_visual_keyword = any(kw in query_lower for kw in visual_indicators)
        has_table_keyword = any(kw in query_lower for kw in table_indicators)
        has_chart_keyword = any(kw in query_lower for kw in chart_indicators)

        if has_image or has_visual_keyword:
            return "visual_query"
        elif has_table_keyword:
            return "table_query"
        elif has_chart_keyword:
            return "chart_query"
        else:
            return "text_query"

    def _determine_mode(self, query_type: str) -> RetrievalMode:
        """根据查询意图自动决定检索模式"""
        mode_map = {
            "text_query": RetrievalMode.TEXT_ONLY,
            "table_query": RetrievalMode.TEXT_ONLY,
            "chart_query": RetrievalMode.FUSED,
            "visual_query": RetrievalMode.FUSED,
        }
        return mode_map.get(query_type, RetrievalMode.FUSED)

    async def _retrieve_text(self, query: str) -> List[NodeWithScore]:
        """文本检索"""
        try:
            return self.text_retriever.retrieve(query)
        except Exception as e:
            logger.error(f"文本检索失败: {e}")
            return []

    async def _retrieve_image_by_text(self, query: str) -> List[NodeWithScore]:
        """基于文本查询检索图片（通过 CLIP embedding 匹配）"""
        if not self.image_retriever:
            return []

        try:
            return self.image_retriever.retrieve(query)
        except Exception as e:
            logger.error(f"图片检索失败: {e}")
            return []

    async def _retrieve_image_by_reference(self, query: str, image_path: str) -> List[NodeWithScore]:
        """基于参考图片做以图搜图"""
        if not self.image_retriever:
            return []

        try:
            query_image_embedding = self.clip_embedding.get_image_embedding(image_path)
            # 使用 embedding 直接搜索（需要底层 retriever 支持）
            return self.image_retriever.retrieve(image_path)
        except Exception as e:
            logger.error(f"以图搜图失败: {e}")
            return []

    def _fuse_results(
        self,
        text_results: List[NodeWithScore],
        image_results: List[NodeWithScore],
        query: str,
    ) -> List[NodeWithScore]:
        """
        融合文本和图片检索结果

        策略: RRF (Reciprocal Rank Fusion)
        但对图片结果施加一个权重提升因子，
        因为当用户问视觉问题时图片通常更有价值。
        """
        k = 60

        scored_items = {}
        node_id_map = {}

        for rank, result in enumerate(text_results):
            node_id = result.node.node_id
            rrf_score = 1.0 / (k + rank + 1)
            if node_id not in scored_items:
                scored_items[node_id] = rrf_score
                node_id_map[node_id] = result
            else:
                scored_items[node_id] += rrf_score

        # 图片结果给予权重加成
        image_boost = 1.3
        for rank, result in enumerate(image_results):
            node_id = result.node.node_id
            rrf_score = image_boost / (k + rank + 1)
            if node_id not in scored_items:
                scored_items[node_id] = rrf_score
                node_id_map[node_id] = result
            else:
                scored_items[node_id] += rrf_score

        fused = sorted(
            scored_items.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        return [
            NodeWithScore(
                node=node_id_map[node_id].node,
                score=score,
            )
            for node_id, score in fused[: (self.text_top_k + self.image_top_k)]
        ]
```

`MultiModalRetriever` 的设计中有几个值得注意的点：

**自适应模式（ADAPTIVE）**：这是最用户友好的模式——用户不需要知道系统内部有文本和图片两种数据源，也不需要手动指定"我要搜图片"。系统通过 `_classify_query_intent()` 分析查询文本中的关键词来自动判断意图。"这个接口返回值是什么？"会被归类为 text_query，只搜文本；"部署架构图在哪里？"会被归类为 visual_query，触发融合检索。这种自动路由大大降低了使用门槛。

**图片结果的权重加成 (`image_boost = 1.3`)**：这是一个经验参数。原因是：当用户的问题被判定为需要视觉信息时（如 visual_query），图片结果通常比文本结果更有价值——一张架构图抵得上一千字的文字描述。所以在 RRF 融合时给图片结果乘以 1.3 的加权系数，让它们在最终排名中更容易排到前面。这个值不是固定的——你可以根据实际效果调整，范围一般在 1.0-2.0 之间。

**以图搜图的支持**：`_retrieve_image_by_reference()` 方法允许用户上传一张图片作为查询条件（比如"这张报错截图对应哪个已知问题？"）。这时系统会用 CLIP 提取上传图片的 embedding，然后在图片索引中找最相似的已有图片。这在客服场景中特别有用——用户直接截个图发过来，系统就能找到最匹配的知识库条目。

## 查询改写与多模态增强

有时候用户的问题本身不足以触发最优的检索策略。比如用户问"这个功能怎么配置？"——这个问题没有明确的视觉关键词，但实际上最相关的答案可能是一张配置界面的截图加上对应的说明文字。这时候就需要查询改写来补充信息：

```python
class MultiModalQueryRewriter:
    """多模态感知的查询改写器"""

    def __init__(self, llm):
        self.llm = llm

    async def rewrite_for_multimodal(self, query: str) -> dict:
        """
        改写查询以优化多模态检索效果

        Returns:
            {
                "original": 原始查询,
                "rewritten_for_text": 用于文本检索的改写版,
                "rewritten_for_image": 用于图片检索的改写版,
                "should_search_images": bool,
                "should_search_tables": bool,
                "should_search_charts": bool,
                "visual_hints": ["可能的视觉关键词"],
            }
        """
        prompt = f"""你是一个多模态检索系统的查询分析器。
请分析以下用户问题，判断它可能需要哪些类型的知识来回答。

用户问题: {query}

请以 JSON 格式输出：
{{
    "rewritten_for_text": "改写为更适合文本检索的形式（添加关键词、补全简称等）",
    "rewritten_for_image": "改写为适合图片检索的形式（描述期望看到的视觉内容）",
    "should_search_images": true/false,
    "should_search_tables": true/false,
    "should_search_charts": true/false,
    "visual_hints": ["可能相关的图片内容描述"],
    "reasoning": "简短说明你的判断依据"
}}"""

        response = await self.llm.acomplete(prompt)
        try:
            import re
            json_match = re.search(r'\{[\s\S]*\}', response.text)
            if json_match:
                return json.loads(json_match.group())
        except (json.JSONDecodeError, AttributeError):
            pass

        return {
            "original": query,
            "rewritten_for_text": query,
            "rewritten_for_image": query,
            "should_search_images": any(
                kw in query.lower()
                for kw in ["图", "截图", "界面", "样子", "show"]
            ),
            "should_search_tables": False,
            "should_search_charts": False,
            "visual_hints": [],
        }
```

这个查询改写器的价值在于它能把隐式的多模态需求显式化。比如用户问"微服务的部署步骤"，改写器可能会输出：

```json
{
    "rewritten_for_text": "微服务部署步骤 Docker Kubernetes 配置流程",
    "rewritten_for_image": "微服务架构部署拓扑图 容器编排示意图",
    "should_search_images": true,
    "should_search_tables": false,
    "should_search_charts": false,
    "visual_hints": ["deployment architecture diagram", "kubernetes topology"],
    "reasoning": "部署步骤通常配有架构图和流程图，视觉信息有助于理解整体流程"
}
```

有了这些改写后的信息，`MultiModalRetriever` 就能做出更精准的检索决策——即使原始查询中没有出现"图""截图"之类的关键词。

## 总结

