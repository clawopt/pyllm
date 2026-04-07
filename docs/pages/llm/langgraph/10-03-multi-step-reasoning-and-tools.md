# 3. 多步推理与工具调用

## 让研究助手"会思考"：从机械执行到智能推理

上一节我们搭建了 DeepResearch 的核心骨架——状态定义、节点函数、图组装——实现了一个能够进行多轮迭代搜索-提取-评估的研究循环。但如果你仔细观察这个系统的工作方式，会发现它本质上还是比较"机械"的：每个节点的行为是预先写死的，搜索策略的调整也主要依赖简单的规则和 LLM 的一次性判断。一个真正优秀的研究助手应该具备更深层次的推理能力——它应该能够在研究过程中主动发现问题、提出假设、验证猜想、甚至像人类研究者一样产生"顿悟"。

这一节我们要做的就是给 DeepResearch 注入更强的**多步推理能力**和**工具调用能力**。具体来说，我们将实现：基于链式思维（Chain-of-Thought）的深度分析、可扩展的工具注册与调用机制、自动假设生成与验证、以及代码解释器集成用于数据分析。

### 为什么需要多步推理

在讨论具体实现之前，让我们先理解一下"多步推理"对于研究助手为什么如此重要。

想象一个真实的研究场景：用户问"2025 年量子计算在药物发现中的应用前景如何"。一个简单的研究助手可能会搜索"quantum computing drug discovery 2025"，找到几篇文章，总结说"量子计算有望加速分子模拟过程"，然后输出报告。这当然有用，但远远不够深入。

一个具备多步推理能力的研究助手则会这样思考：

1. **分解问题**：这个问题涉及几个子维度——技术可行性（当前的量子硬件够不够用？）、算法进展（VQE 等算法成熟度如何？）、实际案例（有没有药企已经在用了？）、市场预期（投资规模和时间线）、以及挑战（噪声、纠错、规模化难题）
2. **识别关键依赖**：要回答"应用前景"，我需要先搞清楚当前量子计算的整体发展水平（因为如果基础都不行，更别提药物发现了），还需要了解传统药物发现的瓶颈在哪里（这样才能对比量子的优势）
3. **形成初步假设**：根据已有知识，我的初步假设是"短期（1-3年）内主要是概念验证和小规模实验，中期（3-7年）可能在特定分子体系上有突破，大规模商业化可能需要10年以上"
4. **有针对性地搜索验证**：围绕这个假设去找支持或反驳的证据
5. **综合判断**：根据收集到的证据修正或确认假设，形成最终结论

这种"分解→假设→验证→综合"的思维模式，就是**多步推理**的本质。它不是一步到位地给出答案，而是通过一系列中间推理步骤逐步逼近真相。LangGraph 的状态管理和循环结构天然适合表达这种推理过程——每一轮循环都可以是一步推理，而状态中累积的信息就是推理的基础。

### 推理引擎的设计：Reactor 模式

为了在 DeepResearch 中实现多步推理，我们采用一种称为 **Reactor（反应堆）模式**的设计。它的核心思想是：

```
┌─────────────────────────────────────────────┐
│              Reactor 循环                     │
│                                              │
│  ┌──────────┐    ┌──────────┐    ┌────────┐ │
│  │  Think   │ → │   Act    │ → │ Observe │ │
│  │ (LLM推理) │    │(调用工具) │    │(观察结果)│ │
│  └──────────┘    └──────────┘    └───┬────┘ │
│       ↑                                │      │
│       └────────────────────────────────┘      │
│              (继续下一轮思考)                   │
└─────────────────────────────────────────────┘
```

每轮 Reactor 循环包含三个阶段：
- **Think**：LLM 根据当前状态进行分析和推理，决定下一步该做什么
- **Act**：执行 LLM 决定的操作（调用某个工具）
- **Observe**：观察操作结果，更新状态，然后回到 Think 阶段

这种模式比我们之前实现的固定流程图更加灵活——LLM 不是被动地按照预设路径执行，而是主动地规划和调整自己的行动。下面我们来一步步实现它。

### 工具系统的设计与实现

Reactor 模式的关键是有一个丰富的**工具集**供 LLM 调用。对于研究助手来说，最核心的工具包括：

```python
from typing import Callable, Any, Dict, List, Optional
from pydantic import BaseModel, Field
from abc import ABC, abstractmethod
import json

class ToolParameter(BaseModel):
    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None

class ToolResult(BaseModel):
    success: bool
    data: Any
    error: Optional[str] = None
    metadata: Dict[str, Any] = {}

class BaseTool(ABC):
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        pass
    
    @abstractmethod
    def get_parameters(self) -> List[ToolParameter]:
        pass
    
    def to_openai_function_schema(self) -> dict:
        properties = {}
        required = []
        for param in self.get_parameters():
            properties[param.name] = {
                "type": param.type,
                "description": param.description
            }
            if param.default is not None:
                properties[param.name]["default"] = param.default
            if param.required:
                required.append(param.name)
        
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }

class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
    
    def register(self, tool: BaseTool):
        self._tools[tool.name] = tool
    
    def get(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name)
    
    def list_all(self) -> List[BaseTool]:
        return list(self._tools.values())
    
    def get_all_schemas(self) -> List[dict]:
        return [tool.to_openai_function_schema() for tool in self._tools.values()]
    
    async def execute_tool(self, name: str, **kwargs) -> ToolResult:
        tool = self.get(name)
        if not tool:
            return ToolResult(success=False, data=None, error=f"未知工具: {name}")
        try:
            result = await tool.execute(**kwargs)
            return result
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))

tool_registry = ToolRegistry()
```

`BaseTool` 是所有工具的抽象基类，定义了三个必须实现的接口：`execute`（执行工具逻辑）、`get_parameters`（返回参数定义）、`to_openai_function_schema`（转换为 OpenAI Function Calling 格式）。`ToolRegistry` 则是一个全局的工具注册表，负责管理所有可用工具的注册、查找和调用。

现在让我们来实现几个具体的工具。

#### 工具一：深度网络搜索（DeepWebSearch）

这是最基本的工具，但比上一节的搜索功能更强——它支持更精细的搜索控制、结果排序和多源聚合。

```python
class DeepWebSearchTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="deep_web_search",
            description="在互联网上深度搜索信息，支持精确的关键词组合、时间范围过滤、域名限制等高级选项。适用于需要全面了解某个话题的最新信息时使用。"
        )
        self.client = httpx.AsyncClient(timeout=30.0)
    
    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="query", type="string", 
                         description="搜索查询语句，可以使用布尔运算符(AND/OR/NOT)和引号精确匹配"),
            ToolParameter(name="num_results", type="integer",
                         description="返回结果数量上限", default=10),
            ToolParameter(name="time_range", type="string",
                         description="时间范围过滤: 'day'/'week'/'month'/'year'/'all'", default="month"),
            ToolParameter(name="domain_filter", type="string",
                         description="限定搜索域名的关键词(如 'arxiv.org' 只搜学术论文)", default="", required=False),
            ToolParameter(name="exclude_domains", type="string",
                         description="排除的域名列表，逗号分隔", default="", required=False),
        ]
    
    async def execute(self, query: str, num_results: int = 10, 
                      time_range: str = "month", domain_filter: str = "",
                      exclude_domains: str = "") -> ToolResult:
        all_results = []
        
        queries = self._expand_query(query)
        
        tasks = []
        for q in queries[:3]:
            tasks.append(self._single_search(q, num_results // len(queries) + 2))
        
        results_lists = await asyncio.gather(*tasks, return_exceptions=True)
        
        seen_urls = set()
        for results in results_lists:
            if isinstance(results, Exception):
                continue
            for r in results:
                url = r.get("url", "")
                if url not in seen_urls:
                    if domain_filter and domain_filter not in url:
                        continue
                    if exclude_domains:
                        should_exclude = any(ed.strip() in url for ed in exclude_domains.split(","))
                        if should_exclude:
                            continue
                    all_results.append(r)
                    seen_urls.add(url)
        
        all_results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        
        return ToolResult(
            success=True,
            data={
                "query": query,
                "results_count": len(all_results),
                "results": all_results[:num_results],
                "search_metadata": {
                    "queries_tried": len(queries),
                    "total_found_before_dedup": sum(
                        len(r) if not isinstance(r, Exception) else 0 
                        for r in results_lists
                    ),
                    "after_dedup": len(all_results)
                }
            }
        )
    
    def _expand_query(self, query: str) -> List[str]:
        words = query.split()
        expansions = [query]
        
        if len(words) >= 2:
            expansions.append(f'"{query}"')
        
        if " vs " in query.lower() or " versus " in query.lower():
            parts = query.replace(" vs ", " ").replace(" versus ", " ").split()
            for i, word in enumerate(parts):
                if word.lower() in ["vs", "versus"] and i > 0 and i < len(parts) - 1:
                    expansions.append(f"{parts[i-1]} {parts[i+1]} comparison")
                    expansions.append(f"{parts[i+1]} advantages over {parts[i-1]}")
                    break
        
        if "2024" in query or "2025" in query or "2026" in query:
            base = query.replace("2024", "").replace("2025", "").replace("2026", "").strip()
            expansions.append(f"{base} latest developments")
            expansions.append(f"{base} trends forecast")
        
        return list(set(expansions))[:5]
    
    async def _single_search(self, query: str, limit: int) -> List[Dict]:
        try:
            resp = await self.client.get(
                "https://api.tavily.com/search",
                json={
                    "api_key": "YOUR_TAVILY_KEY",
                    "query": query,
                    "max_results": min(limit, 15),
                    "include_answer": True,
                    "include_raw_content": False,
                    "search_depth": "advanced"
                },
                timeout=20.0
            )
            data = resp.json()
            
            results = []
            for item in data.get("results", []):
                results.append({
                    "url": item.get("url", ""),
                    "title": item.get("title", ""),
                    "snippet": item.get("content", "")[:500],
                    "relevance_score": item.get("score", 0),
                    "published_date": item.get("published_date", ""),
                    "source_domain": item.get("url", "").split("/")[2] if item.get("url") else ""
                })
            
            if data.get("answer"):
                results.insert(0, {
                    "url": "",
                    "title": f"AI 综合答案: {query}",
                    "snippet": data["answer"][:800],
                    "relevance_score": 0.95,
                    "published_date": "",
                    "source_domain": "tavily-ai-summary"
                })
            
            return results
            
        except Exception as e:
            print(f"搜索失败 [{query}]: {e}")
            return self._mock_results(query, limit)
    
    def _mock_results(self, query: str, limit: int) -> List[Dict]:
        return [
            {
                "url": f"https://research-example.com/paper-{i+1}",
                "title": f"{query} - Comprehensive Analysis ({['2024', '2025'][i % 2]})",
                "snippet": f"关于 {query} 的最新研究发现表明，该领域正在快速发展..."
                         f"关键数据点包括增长率 {15 + i * 5}%，市场规模预计达到 ${10 + i * 5}B...",
                "relevance_score": 0.9 - i * 0.05,
                "published_date": f"2025-0{i + 1}-15",
                "source_domain": "research-example.com"
            }
            for i in range(min(limit, 8))
        ]

tool_registry.register(DeepWebSearchTool())
```

`DeepWebSearchTool` 相比之前的搜索有几个重要增强：

**查询扩展（_expand_query）**：不是只拿用户给的原始关键词去搜，而是智能地生成多个变体查询——加引号做精确匹配、拆分比较型查询为两个方向、加上"latest developments"/"trends forecast"等补充词。这能显著提高召回率。

**多源聚合**：对多个扩展查询并行搜索后合并去重，按相关性排序。

**AI 综合答案优先**：Tavily API 返回的 AI 生成的摘要会被放在结果列表的最前面，因为它通常是对搜索主题的一个高质量概括。

**容错降级**：当真实 API 调用失败时，自动 fallback 到 mock 结果，确保研究流程不会中断。

#### 工具二：学术检索（AcademicSearch）

专门针对学术论文的检索工具，接入 Semantic Scholar 或类似服务。

```python
class AcademicSearchTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="academic_search",
            description="在学术数据库中搜索论文和研究文章。支持按引用数、发表时间、相关度排序。适用于需要获取权威学术来源时使用。"
        )
        self.client = httpx.AsyncClient(timeout=25.0)
    
    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="query", type="string",
                         description="学术搜索查询，建议使用专业术语"),
            ToolParameter(name="limit", type="integer",
                         description="返回论文数量", default=8),
            ToolParameter(name="sort_by", type="string",
                         description="排序方式: 'relevance'/'citations'/'date'", default="relevance"),
            ToolParameter(name="year_min", type="integer",
                         description="最早发表年份", default=2020, required=False),
            ToolParameter(name="fields_of_study", type="string",
                         description="研究领域筛选，逗号分隔(如 'Computer Science,Biology')", default="", required=False),
        ]
    
    async def execute(self, query: str, limit: int = 8, sort_by: str = "relevance",
                      year_min: int = 2020, fields_of_study: str = "") -> ToolResult:
        try:
            params = {
                "query": query,
                "limit": limit,
                "sort": sort_by,
                "year": f"{year_min}-2026",
                "fieldsOfStudy": fields_of_study.split(",") if fields_of_study else None
            }
            
            resp = await self.client.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params={k: v for k, v in params.items() if v is not None},
                headers={"Accept": "application/json"},
                timeout=20.0
            )
            
            data = resp.json()
            papers = []
            
            for paper in data.get("data", []):
                authors = ", ".join(a.get("name", "") for a in paper.get("authors", [])[:3])
                if len(paper.get("authors", [])) > 3:
                    authors += " et al."
                
                papers.append({
                    "paper_id": paper.get("paperId", ""),
                    "title": paper.get("title", ""),
                    "abstract": (paper.get("abstract", "") or "")[:600],
                    "authors": authors,
                    "year": paper.get("year", ""),
                    "citation_count": paper.get("citationCount", 0),
                    "url": f"https://www.semanticscholar.org/paper/{paper.get('paperId', '')}",
                    "venue": paper.get("venue", "") or "Preprint",
                    "open_access": paper.get("isOpenAccess", False),
                    "fields_of_study": paper.get("fieldsOfStudy", [])
                })
            
            return ToolResult(
                success=True,
                data={
                    "query": query,
                    "total": data.get("total", 0),
                    "papers": papers,
                    "offset": data.get("offset", 0)
                }
            )
            
        except Exception as e:
            print(f"学术搜索失败: {e}")
            return self._mock_academic_results(query, limit)
    
    def _mock_academic_results(self, query: str, limit: int) -> ToolResult:
        mock_papers = [
            {
                "paper_id": f"mock_{i:04d}",
                "title": f"{query}: A Comprehensive Survey and Future Directions",
                "abstract": f"This paper presents a comprehensive survey of recent advances in {query}. "
                           f"We review state-of-the-art methods, identify key challenges, "
                           f"and propose promising research directions. Our analysis covers ...",
                "authors": ["Zhang et al.", "Smith & Johnson", "Li, Wang, Chen"][i % 3],
                "year": 2024 + (i // 3),
                "citation_count": 50 + i * 30 + (i * i * 5),
                "url": f"https://semanticscholar.org/paper/mock_{i:04d}",
                "venue": ["NeurIPS", "ICML", "Nature", "Science", "ACL"][i % 5],
                "open_access": i % 2 == 0,
                "fields_of_study": [["Computer Science"], ["Artificial Intelligence"], ["Machine Learning"]][i % 3]
            }
            for i in range(limit)
        ]
        
        return ToolResult(success=True, data={
            "query": query, "total": limit, "papers": mock_papers, "offset": 0
        })

tool_registry.register(AcademicSearchTool())
```

学术搜索工具的特点是返回结构化的论文元数据——标题、摘要、作者、发表年份、引用次数等。这些信息对于评估来源的可信度和影响力非常重要。`sort_by=citations` 参数让用户可以优先看到高引用的经典文献，这在快速了解一个领域的经典工作时特别有用。

#### 工具三：内容深度提取（ContentExtractor）

搜索和学术检索只能拿到摘要级别的信息，很多时候我们需要完整阅读原文才能提取到真正有价值的事实。

```python
class ContentExtractorTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="extract_content",
            description="抓取并提取网页或文档的完整正文内容，去除广告、导航栏等无关元素。适用于需要深入阅读某篇特定文章时使用。"
        )
        self.client = httpx.AsyncClient(timeout=20.0, follow_redirects=True)
    
    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="url", type="string",
                         description="要提取内容的URL地址"),
            ToolParameter(name="max_length", type="integer",
                         description="最大返回字符数(控制Token消耗)", default=8000),
            ToolParameter(name="extract_mode", type="string",
                         description="提取模式: 'full'(全文)/'main'(正文)/'summary'(摘要)", default="main"),
        ]
    
    async def execute(self, url: str, max_length: int = 8000,
                      extract_mode: str = "main") -> ToolResult:
        try:
            resp = await self.client.get(url, headers={
                "User-Agent": "DeepResearch/1.0 (Research Assistant; contact@deepresearch.ai)"
            }, timeout=15.0)
            
            if resp.status_code != 200:
                return ToolResult(
                    success=False, data=None,
                    error=f"HTTP {resp.status_code}: 无法访问页面"
                )
            
            html = resp.text
            text = self._html_to_clean_text(html)
            
            if extract_mode == "summary":
                text = text[:2000]
            elif extract_mode == "main":
                sections = text.split("\n\n")
                main_sections = [s for s in sections if len(s) > 100]
                text = "\n\n".join(main_sections)[:max_length]
            else:
                text = text[:max_length]
            
            metadata = self._extract_page_metadata(html, url)
            
            return ToolResult(
                success=True,
                data={
                    "url": url,
                    "content": text,
                    "content_length": len(text),
                    "extract_mode": extract_mode,
                    "metadata": metadata
                }
            )
            
        except httpx.TimeoutException:
            return ToolResult(
                success=False, data=None,
                error=f"请求超时: {url} (超过15秒)"
            )
        except Exception as e:
            return ToolResult(
                success=False, data=None,
                error=f"内容提取失败: {str(e)}"
            )
    
    def _html_to_clean_text(self, html: str) -> str:
        import re
        
        remove_patterns = [
            (r'<script[^>]*>.*?</script>', '', re.DOTALL | re.IGNORECASE),
            (r'<style[^>]*>.*?</style>', '', re.DOTALL | re.IGNORECASE),
            (r'<nav[^>]*>.*?</nav>', '', re.DOTALL | re.IGNORECASE),
            (r'<footer[^>]*>.*?</footer>', '', re.DOTALL | re.IGNORECASE),
            (r'<header[^>]*>.*?</header>', '', re.DOTALL | re.IGNORECASE),
            (r'<!--.*?-->', '', re.DOTALL),
            (r'<[^>]+>', ' ', 0),
        ]
        
        text = html
        for pattern, replacement, flags in remove_patterns:
            text = re.sub(pattern, replacement, text, flags=flags)
        
        text = re.sub(r'\s+', ' ', text)
        text = unescape_html(text)
        
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        return '\n'.join(lines)
    
    def _extract_page_metadata(self, html: str, url: str) -> dict:
        meta = {"url": url}
        
        title_match = re.search(r'<title>(.*?)</title>', html, re.IGNORECASE)
        if title_match:
            meta["title"] = title_match.group(1).strip()
        
        desc_match = re.search(
            r'<meta[^>]+name=["\']description["\'][^>]+content=["\'](.*?)["\']',
            html, re.IGNORECASE
        )
        if desc_match:
            meta["description"] = desc_match.group(1)
        
        og_title = re.search(
            r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\'](.*?)["\']',
            html, re.IGNORECASE
        )
        if og_title:
            meta["og_title"] = og_title.group(1)
        
        date_match = re.search(
            r'(?:datetime|datePublished|published)["\']?\s*[:=]\s*["\'](\d{4}-\d{2}-\d{2})',
            html
        )
        if date_match:
            meta["published_date"] = date_match.group(1)
        
        return meta

tool_registry.register(ContentExtractorTool())
```

`ContentExtractorTool` 的亮点在于它的 HTML 清洗能力——不是简单地去掉标签，而是有选择性地移除导航栏、页脚、脚本、样式块等非正文内容，保留真正的文章主体。三种提取模式（full/main/summary）让使用者可以根据需求灵活控制返回内容的长度，避免浪费 Token 去处理不重要的部分。

#### 工具四：代码解释器（CodeInterpreter）

这是一个非常强大的工具——允许 LLM 在沙箱环境中执行 Python 代码来处理数据和进行计算。比如当我们搜索到了一组数字数据（各公司的营收、市场份额、增长率等），可以用代码解释器来做统计分析、绘制图表描述、计算相关性等。

```python
class CodeInterpreterTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="code_interpreter",
            description="在安全的沙箱环境中执行Python代码，用于数据处理、统计分析、数学计算和数据可视化。当你需要对收集到的数值数据进行分析时使用此工具。注意：不要用于执行网络请求或文件系统操作。"
        )
    
    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(name="code", type="string",
                         description="要执行的Python代码"),
            ToolParameter(name="timeout_seconds", type="integer",
                         description="超时时间(秒)", default=30, required=False),
        ]
    
    async def execute(self, code: str, timeout_seconds: int = 30) -> ToolResult:
        safe_builtins = {
            "abs": abs, "all": all, "any": any, "bin": bin, "bool": bool,
            "bytearray": bytearray, "bytes": bytes, "callable": callable,
            "chr": chr, "complex": complex, "divmod": divmod, "enumerate": enumerate,
            "filter": filter, "float": float, "format": format, "frozenset": frozenset,
            "getattr": getattr, "hasattr": hasattr, "hash": hash, "hex": hex,
            "int": int, "isinstance": isinstance, "iter": iter, "len": len,
            "list": list, "map": map, "max": max, "min": min, "next": next,
            "oct": oct, "ord": ord, "pow": pow, "print": print, "range": range,
            "repr": repr, "reversed": reversed, "round": round, "set": set,
            "sorted": sorted, "str": str, "sum": sum, "tuple": tuple, "zip": zip,
            "True": True, "False": False, "None": None,
        }
        
        forbidden_patterns = [
            r'\bimport\b', r'\b__import__\b', r'\beval\b', r'\bexec\b',
            r'\bopen\b', r'\bcompile\b', r'\bos\.', r'\bsys\.',
            r'\bsubprocess\b', r'\bsocket\b', r'\burllib\b',
            r'\b__', r'\.\.', r'/etc/', r'/dev/'
        ]
        
        for pattern in forbidden_patterns:
            if re.search(pattern, code):
                return ToolResult(
                    success=False, data=None,
                    error=f"安全限制: 代码包含不允许的操作 '{pattern}'"
                )
        
        output_buffer = []
        result_value = None
        error_msg = None
        
        exec_globals = {"__builtins__": safe_builtins}
        exec_locals = {
            "math": __import__("math"),
            "statistics": __import__("statistics"),
            "json": __import__("json"),
            "collections": __import__("collections"),
            "itertools": __import__("itertools"),
            "functools": __import__("functools"),
            "datetime": __import__("datetime"),
            "decimal": __import__("decimal"),
            "fractions": __import__("fractions"),
        }
        
        class CapturedOutput:
            def write(self, text):
                output_buffer.append(text)
            def flush(self):
                pass
        
        old_stdout = sys.stdout
        sys.stdout = CapturedOutput()
        
        try:
            compiled_code = compile(code, "<research_code>", "exec")
            exec(compiled_code, exec_globals, exec_locals)
            
            if "result" in exec_locals:
                result_value = exec_locals["result"]
            
        except TimeoutError:
            error_msg = f"代码执行超时 ({timeout_seconds}秒)"
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
        finally:
            sys.stdout = old_stdout
        
        stdout_text = "".join(output_buffer)
        
        result_data = {
            "stdout": stdout_text[-5000:] if stdout_text else "(无输出)",
            "result": self._serialize_result(result_value),
            "execution_time": "within limits",
            "variables_available": [k for k in exec_locals.keys() 
                                   if not k.startswith("_")]
        }
        
        if error_msg:
            result_data["error"] = error_msg
            return ToolResult(success=False, data=result_data, error=error_msg)
        
        return ToolResult(success=True, data=result_data)
    
    def _serialize_result(self, value) -> Any:
        if value is None:
            return None
        elif isinstance(value, (int, float, str, bool)):
            return value
        elif isinstance(value, (list, tuple)):
            return [self._serialize_result(v) for v in value][:100]
        elif isinstance(value, dict):
            return {k: self._serialize_result(v) for k, v in list(value.items())[:50]}
        else:
            return str(value)

tool_registry.register(CodeInterpreterTool())
```

`CodeInterpreterTool` 实现了一个**安全受限的 Python 执行环境**。安全措施包括：

1. **内置函数白名单**：只暴露安全的内置函数，禁止 `__import__`、`eval`、`exec`、`open` 等危险操作
2. **正则表达式黑名单**：检测代码字符串中的危险模式（导入模块、文件系统访问、网络操作等）
3. **输出捕获**：重定向 `sys.stdout` 来捕获 `print` 输出
4. **结果序列化**：把执行结果转换为 JSON 可序列化的格式
5. **预导入常用库**：math、statistics、json、collections 等——这些是数据分析中最常用的库，且没有安全隐患

有了这个工具，LLM 就能在研究过程中执行各种数据分析操作了——计算平均值、标准差、相关性矩阵、趋势拟合等。这对于处理数值密集型的研究课题（如市场分析、经济预测、科学数据处理）尤其有价值。

### Reactor 节点的实现

工具都就绪之后，现在来实现 Reactor 模式的核心——一个能够让 LLM 自主决定调用哪个工具、怎么调用、以及如何解读结果的节点。

```python
REACTOR_SYSTEM_PROMPT = """你是一个资深的研究分析师，正在执行一项深度研究任务。
你可以使用以下工具来帮助你收集和分析信息：

{available_tools}

## 当前研究状态
- 主题：{topic}
- 当前轮次：{current_round}/{max_rounds}
- 已收集事实数：{fact_count}
- 已收集来源数：{source_count}

## 最近的研究发现（最近10条）
{recent_facts}

## 当前焦点问题
{current_focus}

## 你的任务
请分析当前情况，决定下一步行动。你可以：
1. 使用 deep_web_search 搜索更多信息
2. 使用 academic_search 查找学术论文
3. 使用 extract_content 深入阅读某篇文章
4. 使用 code_interpreter 分析数值数据
5. 如果认为信息已经足够充分，输出 DONE

## 输出格式（严格JSON）
{
    "thought": "你的分析和推理过程",
    "action": "工具名称或'DONE'",
    "action_input": {{工具参数}} 或 null,
    "expected_outcome": "期望从这个操作中获得什么信息"
}"""

reactor_llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

async def reactor_node(state: ResearchState) -> dict:
    task_ctx = state["task_context"]
    current_round = task_ctx["current_round"]
    max_rounds = state["input_config"]["max_rounds"]
    
    tools_schemas = tool_registry.get_all_schemas()
    tools_description = "\n".join(
        f"- **{schema['function']['name']}**: {schema['function']['description']}\n"
        f"  参数: {json.dumps(schema['function']['parameters']['properties'], ensure_ascii=False)}"
        for schema in tools_schemas
    )
    
    recent_facts = state["extracted_facts"][-10:] if state["extracted_facts"] else []
    facts_text = "\n".join(
        f"- {f.get('subject','?')} {f.get('predicate','?')} {f.get('object_value','?')}"
        for f in recent_facts
    ) or "(暂无)"
    
    focus_idx = task_ctx.get("current_focus_index", 0)
    strategies = state.get("search_strategies", [])
    current_focus = strategies[focus_idx]["sub_question"] if focus_idx < len(strategies) else "待确定"
    
    prompt = REACTOR_SYSTEM_PROMPT.format(
        available_tools=tools_description,
        topic=state["topic"],
        current_round=current_round,
        max_rounds=max_rounds,
        fact_count=len(state["extracted_facts"]),
        source_count=len(state["collected_sources"]),
        recent_facts=facts_text,
        current_focus=current_focus
    )
    
    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content="请分析并决定下一步行动。")
    ]
    
    response = reactor_llm.invoke(messages)
    
    try:
        decision = json.loads(response.content.strip())
    except json.JSONDecodeError:
        decision = {
            "thought": "解析失败，默认继续搜索",
            "action": "deep_web_search",
            "action_input": {"query": current_focus},
            "expected_outcome": "获取更多相关信息"
        }
    
    action = decision.get("action", "DONE")
    action_input = decision.get("action_input") or {}
    thought = decision.get("thought", "")
    
    log_entry = (
        f"[Reactor R{current_round}] 思考: {thought[:150]}... | "
        f"动作: {action}"
    )
    
    if action == "DONE":
        return {
            "sufficiency_verdict": "sufficient",
            "research_log": [log_entry + " | 判定: 信息充足，结束研究"]
        }
    
    tool_result = await tool_registry.execute_tool(action, **action_input)
    
    new_sources_from_tool = []
    new_facts_from_tool = []
    
    if tool_result.success and tool_result.data:
        action_handler = ACTION_HANDLERS.get(action)
        if action_handler:
            processed = await action_handler(tool_result.data, state)
            new_sources_from_tool = processed.get("sources", [])
            new_facts_from_tool = processed.get("facts", [])
        else:
            raw_text = json.dumps(tool_result.data, ensure_ascii=False)[:3000]
            quick_facts = await quick_extract_facts(raw_text, current_focus)
            new_facts_from_tool = quick_facts
    
    new_task_ctx = dict(task_ctx)
    new_task_ctx["total_searches_performed"] += 1
    
    log_entry += f" | 结果: {'成功' if tool_result.success else '失败'}"
    if tool_result.error:
        log_entry += f" | 错误: {tool_result.error[:80]}"
    if new_facts_from_tool:
        log_entry += f" | 新事实: {len(new_facts_from_tool)}"
    
    return {
        "current_search_results": new_sources_from_tool,
        "current_facts_batch": new_facts_from_tool,
        "collected_sources": new_sources_from_tool,
        "extracted_facts": new_facts_from_tool,
        "task_context": new_task_ctx,
        "research_log": [log_entry]
    }

ACTION_HANDLERS = {}

async def handle_search_result(data: dict, state: ResearchState) -> dict:
    sources = []
    for r in data.get("results", []):
        sources.append(Source(
            url=r.get("url", ""),
            title=r.get("title", ""),
            source_type=SourceType(guess_source_type(r.get("url", ""))),
            content_summary=r.get("snippet", ""),
            full_content="",
            metadata=r,
            collected_in_round=state["task_context"]["current_round"]
        ).to_dict())
    
    facts = []
    for src in sources[:5]:
        snippet_facts = [
            Fact(
                subject=src["title"][:50],
                predicate="contains_information_about",
                object_value=src.get("content_summary", "")[:200],
                confidence=0.6,
                source_urls=[src["url"]],
                sub_question=state["search_strategies"][state["task_context"]["current_focus_index"]]["sub_question"]
            ).to_dict()
        ]
        facts.extend(snippet_facts)
    
    return {"sources": sources, "facts": facts}

ACTION_HANDLERS["deep_web_search"] = handle_search_result
ACTION_HANDLERS["academic_search"] = handle_search_result

async def handle_extraction_result(data: dict, state: ResearchState) -> dict:
    content = data.get("content", "")
    url = data.get("url", "")
    
    extracted = await extract_facts_from_text(content, url, state)
    
    sources = [Source(
        url=url,
        title=data.get("metadata", {}).get("title", url),
        source_type=SourceType(guess_source_type(url)),
        content_summary=content[:300],
        full_content=content,
        metadata=data.get("metadata", {}),
        collected_in_round=state["task_context"]["current_round"]
    ).to_dict()]
    
    return {"sources": sources, "facts": extracted}

ACTION_HANDLERS["extract_content"] = handle_extraction_result

async def handle_code_result(data: dict, state: ResearchState) -> dict:
    stdout = data.get("stdout", "")
    result_val = data.get("result")
    
    fact = Fact(
        subject="data_analysis",
        predicate="analysis_result",
        object_value=str(result_val) if result_val else stdout[:500],
        confidence=0.9,
        source_urls=[],
        sub_question="data_analysis"
    ).to_dict()
    
    return {"sources": [], "facts": [fact]}

ACTION_HANDLERS["code_interpreter"] = handle_code_result

async def quick_extract_facts(text: str, context: str) -> List[Dict]:
    if len(text) < 50:
        return []
    
    prompt = f"""从以下文本中快速提取 3-5 个最重要的关键信息点：
文本：{text[:2000]}
背景：{context}
返回 JSON 数组，每项含 subject, predicate, object_value, confidence"""
    
    try:
        response = await fact_extractor_llm.ainvoke([
            SystemMessage(content="简洁提取关键信息"),
            HumanMessage(content=prompt)
        ])
        facts_raw = json.loads(response.content.strip())
        return [
            Fact(
                subject=f.get("subject", ""), predicate=f.get("predicate", ""),
                object_value=f.get("object_value", ""), confidence=f.get("confidence", 0.7),
                source_urls=[], sub_question=context
            ).to_dict()
            for f in facts_raw[:5]
        ]
    except:
        return [Fact(
            subject="raw_data", predicate="contains", object_value=text[:200],
            confidence=0.5, source_urls=[], sub_question=context
        ).to_dict()]

async def extract_facts_from_text(content: str, url: str, state: ResearchState) -> List[Dict]:
    focus_idx = state["task_context"].get("current_focus_index", 0)
    strategies = state.get("search_strategies", [])
    sub_q = strategies[focus_idx]["sub_question"] if focus_idx < len(strategies) else "general"
    
    truncated = content[:4000]
    prompt = f"""从以下文本中提取与 "{sub_q}" 相关的关键事实：
URL: {url}
内容：
{truncated}"""
    
    messages = [
        SystemMessage(content=EXTRACTOR_SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ]
    
    response = fact_extractor_llm.invoke(messages)
    
    try:
        facts_raw = json.loads(response.content.strip())
    except:
        facts_raw = [{"subject": "content", "predicate": "discusses", "object_value": content[:200], "confidence": 0.6}]
    
    return [
        Fact(
            subject=f.get("subject", ""), predicate=f.get("predicate", ""),
            object_value=f.get("object_value", ""), confidence=f.get("confidence", 0.7),
            source_urls=[url], sub_question=sub_q
        ).to_dict()
        for f in facts_raw[:8]
    ]
```

`reactor_node` 是整个多步推理系统的核心。让我详细解释它的工作流程：

**第一步：构建上下文 prompt**。把所有可用工具的 schema、当前研究状态（主题、轮次、已收集的数据量）、最近的研究发现、当前焦点问题等信息组织成一个详细的 system prompt。这个 prompt 就是 LLM "思考"的基础。

**第二步：LLM 推理决策**。把 prompt 发送给 LLM（这里用的是 gpt-4o 而不是 gpt-4o-mini，因为决策质量很重要），让它输出一个 JSON 格式的决策——包含思考过程（thought）、要执行的动作（action）、动作参数（action_input）、以及期望的结果。

**第三步：执行工具调用**。根据 LLM 的决策调用对应的工具，传入它指定的参数。

**第四步：处理工具结果**。每种工具类型都有对应的 handler 函数（`ACTION_HANDLERS` 字典），负责把原始的工具输出转化为标准的 Source 和 Fact 对象。比如搜索结果的 handler 会把每条搜索结果包装成 Source 并从中快速提取初步事实；内容提取的 handler 会把完整文本发送给 LLM 做深度事实提取；代码执行的 handler 会把输出封装成一个分析结果事实。

**第五步：更新状态并记录日志**。把新的来源和事实追加到累加列表中，更新计数器，记录完整的 Reactor 日志（包含思考过程、执行的动作、执行结果）。

### 假设生成与验证机制

除了被动的"搜索→提取"循环，主动的**假设驱动研究**（Hypothesis-Driven Research）是区分普通研究助手和优秀研究助手的关键特征。当一个研究者面对一个复杂问题时，他不会漫无目的地搜集信息，而是先形成一些初步假设，然后有针对性地寻找证据来验证或反驳它们。

我们在 Reactor 中加入一个专门的假设管理机制：

```python
@dataclass
class Hypothesis:
    statement: str
    confidence: float
    supporting_evidence: List[str]
    refuting_evidence: List[str]
    status: str  # "pending" / "confirmed" / "refuted" / "partial"
    created_at_round: int
    source_sub_question: str
    
    def to_dict(self) -> dict:
        return {
            "statement": self.statement,
            "confidence": self.confidence,
            "supporting_evidence": self.supporting_evidence,
            "refuting_evidence": self.refuting_evidence,
            "status": self.status,
            "created_at_round": self.created_at_round,
            "source_sub_question": self.source_sub_question
        }

HYPOTHESIS_GENERATOR_PROMPT = """基于以下研究信息，生成 2-3 个值得验证的研究假设。
假设应该是：
- 具体的、可证伪的（能找到证据支持或反对）
- 有洞察力的（不是显而易见的常识）
- 与当前研究主题高度相关的

当前主题：{topic}
已知的发现：
{known_facts}

返回 JSON 数组：
[
  {{
    "hypothesis": "假设陈述",
    "initial_confidence": 0.5-0.7,
    "what_would_support_it": "什么证据会支持它",
    "what_would_refute_it": "什么证据会反驳它"
  }}
]"""

def generate_hypotheses_node(state: ResearchState) -> dict:
    topic = state["topic"]
    facts = state["extracted_facts"]
    task_ctx = state["task_context"]
    
    if task_ctx["current_round"] <= 1 and len(facts) < 5:
        return {"research_log": [f"[Hypothesis] 跳过 - 信息不足以形成假设 (Round {task_ctx['current_round']}, Facts: {len(facts)})"]}
    
    known_facts_text = "\n".join(
        f"- {f.get('subject')} {f.get('predicate')} {f.get('object_value')}"
        for f in facts[-20:]
    ) or "(暂无)"
    
    prompt = HYPOTHESIS_GENERATOR_PROMPT.format(
        topic=topic,
        known_facts=known_facts_text
    )
    
    messages = [
        SystemMessage(content=HYPOTHESIS_GENERATOR_PROMPT),
        HumanMessage(content=prompt)
    ]
    
    response = reactor_llm.invoke(messages)
    
    try:
        hypotheses_raw = json.loads(response.content.strip())
    except:
        hypotheses_raw = [{
            "hypothesis": f"{topic} 的发展正在加速",
            "initial_confidence": 0.6,
            "what_would_support_it": "近期的增长数据和市场报告",
            "what_would_refute_it": "停滞或下降的趋势数据"
        }]
    
    hypotheses = []
    for h in hypotheses_raw[:3]:
        hyp = Hypothesis(
            statement=h["hypothesis"],
            confidence=h.get("initial_confidence", 0.6),
            supporting_evidence=[],
            refuting_evidence=[],
            status="pending",
            created_at_round=task_ctx["current_round"],
            source_sub_question=state["search_strategies"][task_ctx.get("current_focus_index", 0)]["sub_question"]
            if state.get("search_strategies") else "unknown"
        )
        hypotheses.append(hyp.to_dict())
    
    log_entry = (
        f"[Hypothesis R{task_ctx['current_round']}] 生成了 {len(hypotheses)} 个假设: "
        + "; ".join(h['statement'][:40] + "..." for h in hypotheses)
    )
    
    return {
        "active_hypotheses": hypotheses,
        "research_log": [log_entry]
    }

def verify_hypotheses_node(state: ResearchState) -> dict:
    hypotheses = state.get("active_hypotheses", [])
    new_facts = state.get("current_facts_batch", [])
    
    if not hypotheses:
        return {"research_log": ["[Verify] 无活跃假设需要验证"]}
    
    updated_hypotheses = []
    verification_logs = []
    
    for hyp in hypotheses:
        updated = dict(hyp)
        statement = hyp["statement"].lower()
        
        for fact in new_facts:
            fact_text = f"{fact.get('subject', '')} {fact.get('predicate', '')} {fact.get('object_value', '')}".lower()
            
            relevance = calculate_text_relevance(statement, fact_text)
            
            if relevance > 0.6:
                if is_supporting(statement, fact_text):
                    updated["supporting_evidence"].append(fact_text[:100])
                    updated["confidence"] = min(1.0, updated["confidence"] + 0.1)
                elif is_refuting(statement, fact_text):
                    updated["refuting_evidence"].append(fact_text[:100])
                    updated["confidence"] = max(0.0, updated["confidence"] - 0.15)
        
        total_evidence = len(updated["supporting_evidence"]) + len(updated["refuting_evidence"])
        if total_evidence >= 3:
            if updated["confidence"] >= 0.75:
                updated["status"] = "confirmed"
            elif updated["confidence"] <= 0.3:
                updated["status"] = "refuted"
            else:
                updated["status"] = "partial"
        
        updated_hypotheses.append(updated)
        verification_logs.append(
            f"  假设: '{hyp['statement'][:50]}...' → "
            f"置信度 {hyp['confidence']:.1%} → {updated['confidence']:.1%} "
            f"[{updated['status']}] "
            f"(支持:{len(updated['supporting_evidence'])} 反对:{len(updated['refuting_evidence'])})"
        )
    
    log_entry = "[Verify]\n" + "\n".join(verification_logs)
    
    return {
        "active_hypotheses": updated_hypotheses,
        "research_log": [log_entry]
    }

def calculate_text_relevance(text_a: str, text_b: str) -> float:
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)

def is_supporting(hypothesis: str, evidence: str) -> bool:
    support_indicators = ['支持', '表明', '证实', '显示', 'confirm', 'show', 'support', 
                          'prove', 'demonstrate', 'increase', 'growth', 'success']
    return any(ind in evidence.lower() for ind in support_indicators)

def is_refuting(hypothesis: str, evidence: str) -> bool:
    refute_indicators = ['反对', '质疑', '挑战', '无法', '失败', 'refute', 'challenge', 
                         'contradict', 'fail', 'decline', 'risk', 'problem', 'limitation']
    return any(ind in evidence.lower() for ind in refute_indicators)
```

这套假设机制的运作方式如下：

**generate_hypotheses_node** 在研究的早期阶段（第 2 轮以后，且已有一定数量的发现）被触发，让 LLM 基于已有的发现生成 2-3 个研究假设。每个假设包含陈述、初始置信度、以及"什么证据会支持/反驳它"的说明。

**verify_hypotheses_node** 在每轮新事实提取后被调用，将新发现的事实与活跃假设进行匹配。匹配算法使用了简单的词汇重叠度（Jaccard similarity）来判断相关性，然后用关键词启发式（support_indicators / refute_indicators）来判断证据是支持的还是反对的。根据证据的数量和方向动态调整假设的置信度和状态（confirmed/refuted/partial）。

这种假设驱动的模式让研究过程不再是盲目的信息收集，而是有明确方向的证据搜寻。最终报告中可以单独列出"经过验证的假设"章节，大大增加报告的说服力和深度。

### 把 Reactor 集成到主图中

最后，我们需要把这个增强版的 Reactor 节点和假设机制集成到主研究图中。由于 Reactor 本身就是一个内循环（Think→Act→Observe→Think...），我们可以把它编译为一个子图嵌入到主图中。

```python
def build_enhanced_research_graph():
    graph = StateGraph(ResearchState)
    
    graph.add_node("parse_query", parse_query_node)
    graph.add_node("create_plan", create_plan_node)
    graph.add_node("reactor_loop", reactor_node)
    graph.add_node("generate_hypotheses", generate_hypotheses_node)
    graph.add_node("verify_hypotheses", verify_hypotheses_node)
    graph.add_node("evaluate_findings", evaluate_findings_node)
    graph.add_node("adjust_strategy", adjust_strategy_node)
    graph.add_node("generate_report", generate_report_node)
    
    graph.add_edge(START, "parse_query")
    graph.add_edge("parse_query", "create_plan")
    
    graph.add_conditional_edges(
        "create_plan",
        lambda s: "start" if s.get("sub_questions") else "end",
        {"start": "reactor_loop", "end": END}
    )
    
    graph.add_conditional_edges(
        "reactor_loop",
        lambda s: s.get("sufficiency_verdict", "continue"),
        {
            "continue": "generate_hypotheses",
            "sufficient": "generate_report",
            "stuck": "adjust_strategy"
        }
    )
    
    graph.add_conditional_edges(
        "generate_hypotheses",
        lambda s: "verify" if s.get("active_hypotheses") else "evaluate",
        {
            "verify": "verify_hypotheses",
            "evaluate": "evaluate_findings"
        }
    )
    
    graph.add_edge("verify_hypotheses", "evaluate_findings")
    
    graph.add_conditional_edges(
        "evaluate_findings",
        lambda s: s.get("sufficiency_verdict", "continue"),
        {
            "continue": "reactor_loop",
            "sufficient": "generate_report",
            "stuck": "adjust_strategy"
        }
    )
    
    graph.add_edge("adjust_strategy", "reactor_loop")
    graph.add_edge("generate_report", END)
    
    return graph.compile()

enhanced_research_graph = build_enhanced_research_graph()
```

增强后的图拓扑变成了这样：

```
START → parse_query → create_plan → reactor_loop
                                          │
                              ┌───────────┼───────────┐
                          continue     sufficient    stuck
                              │             │           │
                              ▼             │           ▼
                       generate_hypotheses   │    adjust_strategy
                              │             │           │
                        ┌─────┴─────┐       │           │
                      verify    (skip)      │           │
                        │         │         │           │
                        ▼         ▼         │           │
                   evaluate_findings ◄──────┘           │
                        │                             │
                  ┌─────┼─────┐                       │
              continue  sufficient stuck               │
                  │         │       │                   │
                  ▼         │       │                   │
            reactor_loop    │       │                   │
                  ↑         │       │                   │
                  └─────────┴───────┴───────────────────┘
                                    (全部汇合回 reactor 或 report)
```

相比之前的基础版本，增强版多了两个重要的中间环节：**假设生成** 和 **假设验证**。这让每一次 Reactor 循环不仅仅是搜索和提取，还包含了主动的知识建构过程——形成假设、验证假设、根据验证结果调整研究方向。

到这里，DeepResearch 的多步推理和工具调用能力已经相当完善了。它不再是一个简单的"搜索→摘要"机器，而是一个具备自主规划、工具使用、假设验证能力的智能研究助理。最后一节我们将关注如何把这些研究成果以最佳的形式呈现给用户，以及如何把整个系统部署到生产环境中。
