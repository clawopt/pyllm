# 6.4 动态子图选择与注册表模式

> 前面几节讨论的子图使用方式都是静态的——在定义父图时就确定了要使用哪些子图、它们之间的连接关系是怎样的。但在某些场景下，你需要根据运行时的状态来动态决定使用哪个子图。比如一个内容处理系统，根据文档类型（PDF/Word/Markdown）来选择不同的解析子图；或者一个客服系统，根据用户问题类别来路由到不同的处理流程。这一节我们会探讨如何实现这种动态的子图选择机制。

## 基础模式：字典查找

最简单的动态子图选择是用字典（registry）把所有可用的子图注册起来，然后根据某个键值来查找和调用。

```python
from typing import TypedDict, Any
from langgraph.graph import StateGraph, START, END

class DocumentProcessingState(TypedDict):
    file_path: str
    file_type: str
    raw_content: str
    parsed_content: dict
    processing_log: list[str]

class PDFParserState(TypedDict):
    file_path: str
    pages: list[str]
    metadata: dict
    text_content: str

def parse_pdf(state: PDFParserState) -> dict:
    return {
        "pages": [f"PDF 第{i+1}页的内容" for i in range(3)],
        "metadata": {"format": "PDF", "pages": 3, "author": "未知"},
        "text_content": "这是从 PDF 中提取的文本内容..."
    }

pdf_parser_graph = StateGraph(PDFParserState)
pdf_parser_graph.add_node("extract", parse_pdf)
pdf_parser_graph.add_edge(START, "extract")
pdf_parser_graph.add_edge("extract", END)
compiled_pdf = pdf_parser_graph.compile()

class WordParserState(TypedDict):
    file_path: str
    paragraphs: list[str]
    styles: list[dict]
    text_content: str

def parse_word(state: WordParserState) -> dict:
    return {
        "paragraphs": ["Word 第一段", "Word 第二段", "Word 第三段"],
        "styles": [{"type": "heading", "level": 1}, {"type": "normal"}],
        "text_content": "这是从 Word 文档中提取的文本内容..."
    }

word_parser_graph = StateGraph(WordParserState)
word_parser_graph.add_node("extract", parse_word)
word_parser_graph.add_edge(START, "extract")
word_parser_graph.add_edge("extract", END)
compiled_word = word_parser_graph.compile()

class MarkdownParserState(TypedDict):
    file_path: str
    headings: list[str]
    code_blocks: list[str]
    text_content: str

def parse_markdown(state: MarkdownParserState) -> dict:
    return {
        "headings": ["# 标题1", "## 标题2"],
        "code_blocks": ["```python\nprint('hello')\n```"],
        "text_content": "这是从 Markdown 文件中提取的内容..."
    }

md_parser_graph = StateGraph(MarkdownParserState)
md_parser_graph.add_node("extract", parse_markdown)
md_parser_graph.add_edge(START, "extract")
md_parser_graph.add_edge("extract", END)
compiled_md = md_parser_graph.compile()

# 注册表：文件类型 → 编译后的子图
PARSER_REGISTRY: dict[str, Any] = {
    "pdf": compiled_pdf,
    "docx": compiled_word,
    "doc": compiled_word,
    "markdown": compiled_md,
    "md": compiled_md,
}

DEFAULT_PARSER = compiled_md

def dynamic_parse_node(state: DocumentProcessingState) -> dict:
    file_type = state["file_type"].lower()
    file_path = state["file_path"]

    parser = PARSER_REGISTRY.get(file_type)

    if parser is None:
        return {
            "parsed_content": {"error": f"不支持的文件类型: {file_type}"},
            "processing_log": [f"[解析] ⚠️ 不支持的类型: {file_type}，使用默认解析器"]
        }

    try:
        if file_type in ("pdf",):
            result = parser.invoke({
                "file_path": file_path,
                "pages": [], "metadata": {}, "text_content": ""
            })
            output = {
                "content_type": "structured",
                "pages": result.get("pages", []),
                "metadata": result.get("metadata", {}),
                "text": result.get("text_content", "")
            }
        elif file_type in ("docx", "doc"):
            result = parser.invoke({
                "file_path": file_path,
                "paragraphs": [], "styles": [], "text_content": ""
            })
            output = {
                "content_type": "rich_text",
                "paragraphs": result.get("paragraphs", []),
                "styles": result.get("styles", []),
                "text": result.get("text_content", "")
            }
        else:
            result = parser.invoke({
                "file_path": file_path,
                "headings": [], "code_blocks": [], "text_content": ""
            })
            output = {
                "content_type": "plain",
                "headings": result.get("headings", []),
                "code_blocks": result.get("code_blocks", []),
                "text": result.get("text_content", "")
            }

        return {
            "parsed_content": output,
            "processing_log": [f"[解析] ✅ {file_type} 文件解析成功"]
        }

    except Exception as e:
        return {
            "parsed_content": {"error": str(e)},
            "processing_log": [f"[解析] ❌ 解析失败: {str(e)}"]
        }

router_graph = StateGraph(DocumentProcessingState)
router_graph.add_node("dynamic_parse", dynamic_parse_node)
router_graph.add_edge(START, "dynamic_parse")
router_graph.add_edge("dynamic_parse", END)

app = router_graph.compile()

test_files = [
    ("report.pdf", "pdf"),
    ("proposal.docx", "docx"),
    ("README.md", "markdown"),
    ("data.csv", "csv"),
]

for filename, filetype in test_files:
    result = app.invoke({
        "file_path": f"/docs/{filename}",
        "file_type": filetype,
        "raw_content": "",
        "parsed_content": {},
        "processing_log": []
    })
    print(f"{filename:20} ({filetype:5}) → {result['processing_log'][0]}")
```

这个动态解析器的核心是 `PARSER_REGISTRY` 字典，它把文件类型字符串映射到对应的编译后子图对象。`dynamic_parse_node` 节点在运行时从注册表中查找合适的解析器，调用它，然后把结果统一转换为父图需要的格式。

这种注册表模式有几个明显的优势：第一，新增一种文件类型只需要编写新的子图并在注册表中添加一行映射，不需要修改任何现有的代码；第二，每个解析器可以有自己的状态结构和内部逻辑，互不干扰；第三，不支持的类型会优雅地降级而不是崩溃。

## 高级模式：策略链与回退机制

在实际应用中，单一的字典查找可能不够——你可能需要尝试多种策略直到有一个成功为止。比如 PDF 解析器首选 PyMuPDF，如果它失败了就回退到 pdfplumber，再失败就用 OCR。这种**策略链（Strategy Chain）+ 回退（Fallback）**模式可以通过组合多个子图来实现。

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class RobustParserState(TypedDict):
    file_path: str
    content: str
    parser_used: str
    attempts: list[dict]
    success: bool
    log: list[str]

# 策略1: PyMuPDF 解析器 (最快)
def pymupdf_parse(state) -> dict:
    import random
    if random.random() < 0.3:
        raise Exception("PyMuPDF: 文件损坏")
    return {"content": "[PyMuPDF] 提取的完整内容...", "success": True}

# 策略2: pdfplumber 解析器 (较慢但更健壮)
def pdfplumber_parse(state) -> dict:
    import random
    if random.random() < 0.15:
        raise Exception("pdfplumber: 格式不支持")
    return {"content": "[pdfplumber] 提取的完整内容...", "success": True}

# 策略3: OCR 解析器 (最慢但能处理扫描件)
def ocr_parse(state) -> dict:
    return {"content": "[OCR] 识别出的文字内容...", "success": True}

STRATEGIES = [
    ("pymupdf", pymupdf_parse),
    ("pdfplumber", pdfplumber_parse),
    ("ocr", ocr_parse),
]

def try_strategies_chain(state: RobustParserState) -> dict:
    attempts = []
    file_path = state["file_path"]

    for strategy_name, strategy_func in STRATEGIES:
        attempt_record = {"strategy": strategy_name, "status": "pending"}
        try:
            result = strategy_func(state)
            if result.get("success"):
                attempt_record["status"] = "success"
                attempts.append(attempt_record)
                return {
                    "content": result["content"],
                    "parser_used": strategy_name,
                    "attempts": attempts,
                    "success": True,
                    "log": [f"✅ {strategy_name} 成功"]
                }
            else:
                attempt_record["status"] = "no_result"
                attempts.append(attempt_record)
        except Exception as e:
            attempt_record["status"] = f"error: {str(e)}"
            attempts.append(attempt_record)

    return {
        "content": "",
        "parser_used": "none",
        "attempts": attempts,
        "success": False,
        "log": [f"❌ 所有策略均失败 ({len(attempts)} 次尝试)"]
    }

robust_graph = StateGraph(RobustParserState)
robust_graph.add_node("try_strategies", try_strategies_chain)
robust_graph.add_edge(START, "try_strategies")
robust_graph.add_edge("try_strategies", END")

app = robust_graph.compile()

result = app.invoke({
    "file_path": "/docs/scanned-document.pdf",
    content": "", "parser_used": "", "attempts": [], "success": False, "log": []
})

print(f"最终结果: {'成功' if result['success'] else '失败'}")
print(f"使用的解析器: {result['parser_used']}")
print(f"\n尝试历史:")
for att in result["attempts"]:
    status_icon = "✅" if att["status"] == "success" else "⚠️"
    print(f"  {status_icon} {att['strategy']}: {att['status']}")
```

这个策略链示例展示了如何按优先级依次尝试多种解析策略：首先尝试最快的 PyMuPDF，失败了就回退到更健壮的 pdfplumber，再失败就用最后的兜底方案 OCR。每次尝试的结果都被记录下来，便于事后分析哪种策略在什么情况下更容易成功。

## 工厂模式：动态创建子图

有些场景下，你不仅需要选择已有的子图，还需要根据运行时参数**动态创建**子图。比如不同的客户可能有不同的审批流程配置，你需要根据客户的配置来动态构建对应的审批子图。

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

def create_approval_workflow(config: dict):
    """工厂函数：根据配置动态创建审批工作流子图"""

    class DynamicApprovalState(TypedDict):
        request_data: dict
        current_level: int
        approvers: list[str]
        decisions: list[dict]
        final_status: str

    nodes = {}
    level_configs = config.get("levels", [])

    for i, level_cfg in enumerate(level_configs):
        level_name = level_cfg["name"]
        required_role = level_cfg.get("required_role", "any")

        def make_approver_node(idx, name, role):
            def node_fn(state: DynamicApprovalState) -> dict:
                decision = {
                    "level": idx + 1,
                    "approver_name": name,
                    "role": role,
                    "decision": "approved",
                    "timestamp": "2024-01-01T12:00:00"
                }
                decisions = list(state["decisions"])
                decisions.append(decision)
                return {
                    "decisions": decisions,
                    "current_level": idx + 2
                }
            return node_fn

        node_name = f"level_{i+1}_{level_name}"
        nodes[node_name] = make_approver_node(i, level_name, required_role)

    def make_finalizer():
        def finalize(state: DynamicApprovalState) -> dict:
            all_approved = all(
                d["decision"] == "approved" for d in state["decisions"]
            )
            return {
                "final_status": "approved" if all_approved else "rejected"
            }
        return finalize

    nodes["finalize"] = make_finalizer()

    workflow = StateGraph(DynamicApprovalState)

    for node_name, node_fn in nodes.items():
        workflow.add_node(node_name, node_fn)

    first_level = list(nodes.keys())[0]
    workflow.add_edge(START, first_level)

    for i in range(len(level_configs)):
        current = list(nodes.keys())[i]
        if i + 1 < len(nodes):
            next_node = list(nodes.keys())[i + 1]
            workflow.add_edge(current, next_node)
        else:
            workflow.add_edge(current, "finalize")

    workflow.add_edge("finalize", END)

    return workflow.compile()

# 使用示例：为两个不同客户创建不同的审批流程
client_a_config = {
    "client_id": "client-A",
    "levels": [
        {"name": "supervisor", "required_role": "team_lead"},
        {"name": "manager", "required_role": "department_manager"},
    ]
}

client_b_config = {
    "client_id": "client-B",
    "levels": [
        {"name": "peer_review", "required_role": "senior"},
        {"name": "tech_lead", "required_role": "tech_lead"},
        {"name": "director", "required_role": "director"},
        {"name": "vp", "required_role": "vp"},
    ]
}

workflow_a = create_approval_workflow(client_a_config)
workflow_b = create_approval_workflow(client_b_config)

print(f"Client A 审批流程: {len(client_a_config['levels'])} 级")
result_a = workflow_a.invoke({
    "request_data": {"amount": 5000},
    "current_level": 1,
    "approvers": [],
    "decisions": [],
    "final_status": ""
})
print(f"  结果: {result_a['final_status']} | 经过了 {len(result_a['decisions'])} 个审批节点")

print(f"\nClient B 审批流程: {len(client_b_config['levels'])} 级")
result_b = workflow_b.invoke({
    "request_data": {"amount": 50000},
    "current_level": 1,
    "approvers": [],
    "decisions": [],
    "final_status": ""
})
print(f"  结果: {result_b['final_status']} | 经过了 {len(result_b['decisions'])} 个审批节点")
```

这个工厂模式的例子展示了如何用 Python 的闭包特性来动态创建子图。`create_approval_workflow` 函数接收一个配置字典，根据其中的 `levels` 定义动态生成对应数量的审批节点，并按顺序连接起来。Client A 有 2 级审批，Client B 有 4 级审批——两者使用同一个工厂函数创建，但生成的子图结构完全不同。

工厂模式特别适合以下场景：
- **多租户系统**：不同租户有不同的业务流程配置
- **A/B 测试**：需要动态切换不同的处理逻辑
- **插件化架构**：核心系统不知道未来会有哪些具体的处理器，由插件在运行时注册

## 注册表模式的最佳实践

在使用动态子图选择和注册表模式时，有几个最佳实践值得遵循：

**第一，注册表应该集中管理**。不要在每个模块中各自维护一份注册表副本——这会导致不一致和遗漏。建议在一个专门的模块中集中定义全局注册表，并提供统一的注册和查询接口：

```python
# registry.py - 集中管理的子图注册表
from typing import Any, Callable

class SubgraphRegistry:
    _instance = None
    _registry: dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def register(cls, name: str, subgraph: Any) -> None:
        cls._registry[name.lower()] = subgraph

    @classmethod
    def get(cls, name: str, default: Any = None) -> Any:
        return cls._registry.get(name.lower(), default)

    @classmethod
    def list_all(cls) -> list[str]:
        return list(cls._registry.keys())

    @classmethod
    def clear(cls) -> None:
        cls._registry.clear()
```

**第二，提供默认回退选项**。当请求的类型不在注册表中时，应该有明确的默认行为——要么返回一个通用的默认子图，要么抛出一个清晰的异常说明缺少哪个类型的处理器。永远不要静默地返回空或 None。

**第三，支持热注册（可选）**。对于需要在不重启服务的情况下添加新处理器的系统，可以提供运行时注册的能力。但这需要额外的线程安全保障和一致性检查。

**第四，记录注册和使用日志**。在生产环境中，记录每次子图的查找和调用日志对于排查问题非常有帮助——特别是当某个类型的请求总是找不到对应的处理器时。
