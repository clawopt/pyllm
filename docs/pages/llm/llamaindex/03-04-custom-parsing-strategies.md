---
title: 自定义解析策略：针对不同文档类型的优化
description: 按文档类型选择解析器、复合解析管道、JSON/日志/邮件等特殊格式的处理、解析质量监控
---
# 自定义解析策略：针对不同文档类型的优化

经过前三节的学习，你现在应该对 LlamaIndex 的 Node Parser 体系有了全面的了解：`SentenceSplitter` 处理通用文本、`CodeSplitter` 处理代码、`HTMLHierarchicalSplitter` 处理网页、`MarkdownNodeParser` 处理 Markdown。但在实际项目中，你遇到的文档类型远不止这些——JSON 配置文件、服务器日志、电子邮件、聊天记录、扫描件、数据流……每种格式都有其独特的结构特征，需要定制化的解析策略。

这一节我们将学习如何**根据文档类型构建差异化的解析管道**，以及如何处理一些常见但容易被忽视的特殊格式。

## 文档分类与路由解析

最实用的策略是**先识别文档类型，再路由到对应的 Parser**：

```python
from typing import List
from llama_index.core import Document
from llama_index.core.node_parser import (
    SentenceSplitter,
    CodeSplitter,
    MarkdownNodeParser,
)
from llama_index.core.schema import TextNode


class RoutingNodeParser:
    """根据文档类型自动选择解析策略的路由解析器"""

    def __init__(self):
        self.parsers = {
            "markdown": MarkdownNodeParser(),
            "code": CodeSplitter(language="python", chunk_lines=80),
            "default": SentenceSplitter(chunk_size=512, chunk_overlap=100),
        }

        self.classifier = self._build_classifier()

    def _classify(self, document: Document) -> str:
        """判断文档类型"""
        text = document.text
        metadata = document.metadata

        # 1. 根据文件扩展名
        ext = metadata.get("file_name", "").lower().split(".")[-1]
        if ext in ["md", "markdown"]:
            return "markdown"
        if ext in ["py", "js", "ts", "java", "go", "rs"]:
            return "code"

        # 2. 根据内容特征
        code_indicators = ["def ", "class ", "import ", "function ",
                           "const ", "let ", "public ", "private "]
        md_indicators = ["# ", "## ", "```", "- [ ]", "| "]

        code_score = sum(1 for w in code_indicators if w in text)
        md_score = sum(1 for w in md_indicators if w in text)

        if code_score >= 3:
            return "code"
        if md_score >= 3:
            return "markdown"

        # 3. 默认
        return "default"

    def get_nodes_from_documents(
        self, documents: List[Document], **kwargs
    ) -> List[TextNode]:
        all_nodes = []

        for doc in documents:
            doc_type = self._classify(doc)
            parser = self.parsers[doc_type]
            nodes = parser.get_nodes_from_documents([doc])
            for node in nodes:
                node.metadata["parser_used"] = doc_type
                node.metadata["doc_type"] = doc_type
            all_nodes.extend(nodes)

        stats = {}
        for node in all_nodes:
            t = node.metadata.get("doc_type", "unknown")
            stats[t] = stats.get(t, 0) + 1

        print(f"解析统计: {stats}")
        return all_nodes


# 使用
router = RoutingNodeParser()
documents = SimpleDirectoryReader("./mixed_content").load_data()
nodes = router.get_nodes_from_documents(documents)
```

这个 `RoutingNodeParser` 的设计思路是：
1. **先用简单规则做快速分类**（文件扩展名 → 内容特征匹配）
2. **然后分发给专门处理该类型的 Parser**
3. **最后在 metadata 中记录使用了哪个 Parser**（方便后续调试和分析）

这种"分类→分发"的模式在处理混合类型的知识库时非常有效——你的 `./data` 目录下可能同时有 `.md` 文档、`.py` 代码文件、`.txt` 日志和 `.json` 配置，路由解析器能确保每种格式都得到最适合的处理。

## JSON 数据的解析

JSON 是现代应用中最常见的数据交换格式之一。API 返回的是 JSON、配置文件是 JSON、日志格式化输出是 JSON、NoSQL 数据库存的是 JSON。但 JSON 数据直接塞进 RAG 系统通常效果不好——原始的 JSON 包含大量语法噪音（花括号、引号、逗号、嵌套结构），而且关键字段的语义可能被深层嵌套所掩盖。

### 策略一：展平为键值对叙述

```python
import json

def json_to_narrative(json_str: str, prefix: str = "") -> list[str]:
    """将 JSON 递归地转换为自然语言叙述"""
    data = json.loads(json_str) if isinstance(json_str, str) else json_str
    narratives = []

    if isinstance(data, dict):
        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, (dict, list)):
                narratives.extend(json_to_narrative(value, full_key))
            else:
                narratives.append(f"{full_key} 的值是 {value}")

    elif isinstance(data, list):
        for i, item in enumerate(data):
            full_key = f"{prefix}[{i}]" if prefix else f"[{i}]"
            narratives.extend(json_to_narrative(item, full_key))

    else:
        narratives.append(f"{prefix} 的值是 {data}")

    return narratives


# 示例
api_response = '''
{
    "product": {
        "name": "智能音箱 S1",
        "specs": {
            "processor": "ARM Cortex-A53 四核",
            "memory": "2GB DDR4",
            "storage": "16GB eMMC"
        },
        "price": {
            "currency": "CNY",
            "amount": 299,
            "discount": 0.15
        },
        "availability": true
    }
}
'''

narratives = json_to_narrative(api_response)
for n in narratives:
    print(n)
```

输出：
```
product.name 的值是 智能音箱 S1
product.specs.processor 的值是 ARM Cortex-A53 四核
product.specs.memory 的值是 2GB DDR4
product.specs.storage 的值是 16GB eMMC
product.price.currency 的值是 CNY
product.price.amount 的值是 299
product.price.discount 的值是 0.15
product.availability 的值是 True
```

每一条都是一个自包含的完整陈述，非常适合作为 RAG 索引的单元。当用户问"S1 的处理器是什么？"时，系统能精确匹配到"product.specs.processor 的值是 ARM Cortex-A53 四核"这一条。

### 策略二：保留结构的半结构化文本

有时候完全展平会丢失有用的结构信息。比如一个包含多个产品的列表，你可能希望每个产品保持为一个整体：

```python
def json_to_structured_docs(json_str: str, root_key: str = "item") -> list:
    """将 JSON 数组中的每个元素转为一个 Document"""
    data = json.loads(json_str)
    documents = []

    items = data if isinstance(data, list) else [data]

    for i, item in enumerate(items):
        text = format_item(item, indent=0)
        doc = Document(
            text=text,
            metadata={
                "source_type": "json",
                "item_index": i,
                "root_key": root_key,
            },
        )
        documents.append(doc)

    return documents


def format_item(obj, indent=0):
    """递归地将对象格式化为可读文本"""
    lines = []
    prefix = "  " * indent

    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                lines.append(f"{prefix}{k}:")
                lines.append(format_item(v, indent + 1))
            else:
                lines.append(f"{prefix}{k}: {v}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            lines.append(f"{prefix}- [{i}]")
            lines.append(format_item(v, indent + 1))

    return "\n".join(lines)
```

## 电子邮件的解析

企业邮箱中沉淀了大量的决策记录、方案讨论、任务分配等信息。但邮件有其特殊的结构——发件人、收件人、主题、时间戳、正文、签名档、回复链等——需要专门的解析逻辑。

```python
import re
from email import policy
from email.parser import BytesParser
from llama_index.core import Document


class EmailParser:
    """电子邮件解析器"""

    SIGNATURE_PATTERNS = [
        r"^--\s*$",
        r"^Best regards.*$",
        r"^Cheers.*$",
        r"^Sent from my.*$",
        r"^_\d+_$",
        r"^Disclaimer:.*$",
    ]

    QUOTE_PATTERN = r"^>.*$"

    def parse_email(self, raw_email: bytes) -> Document:
        msg = BytesParser(policy=policy.default).parsebytes(raw_email)

        subject = msg["Subject"] or "(无主题)"
        from_addr = msg["From"] or "未知"
        to_addr = msg["To"] or "未知"
        date = msg["Date"] or "未知"

        body = self._extract_body(msg)
        clean_body = self._clean_body(body)

        text = (
            f"邮件主题: {subject}\n"
            f"发件人: {from_addr}\n"
            f"收件人: {to_addr}\n"
            f"日期: {date}\n\n"
            f"正文:\n{clean_body}"
        )

        return Document(
            text=text,
            metadata={
                "subject": subject,
                "from": from_addr,
                "to": to_addr,
                "date": date,
                "thread_id": msg.get("Message-ID"),
                "source_type": "email",
            },
        )

    def _extract_body(self, msg):
        """提取邮件正文（处理 multipart）"""
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                if content_type == "text/plain":
                    payload = part.get_payload(decode=True)
                    if payload:
                        charset = part.get_content_charset() or "utf-8"
                        return payload.decode(charset, errors="replace")
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                charset = msg.get_content_charset() or "utf-8"
                return payload.decode(charset, errors="replace")
        return ""

    def _clean_body(self, body: str) -> str:
        """清理邮件正文：去除签名档、引用前文等"""
        lines = body.split("\n")
        clean_lines = []
        in_signature = False

        for line in lines:
            if any(re.match(p, line.strip()) for p in self.SIGNATURE_PATTERNS):
                in_signature = True
                continue

            if in_signature:
                continue

            if re.match(self.QUOTE_PATTERN, line):
                continue  # 跳过引用的前文（以 > 开头的行）

            clean_lines.append(line)

        result = "\n".join(clean_lines).strip()
        return result


# 使用示例（假设 .eml 文件）
with open("./emails/project_decision.eml", "rb") as f:
    raw = f.read()

parser = EmailParser()
doc = parser.parse_email(raw)
print(doc.text[:500])
```

EmailParser 做了几件关键的事情：
1. **提取结构化字段**——主题、发件人、收件人、日期等进入 metadata 和文本头部
2. **处理 multipart 邮件**——优先提取纯文本部分
3. **清理噪音**——去除签名档（"Best regards"、"- -" 等）、引用前文（"> "开头的行）
4. **编码处理**——正确处理各种字符集

这些步骤看似琐碎，但对最终检索质量影响巨大。一封没有经过清理的邮件可能 50% 的内容都是签名档和引用前文——全是噪音。

## 日志文件的解析

服务器日志是运维知识的宝库——错误消息及其解决方案、性能基线、系统行为模式等都隐藏在日志中。但日志有其独特的挑战：单条记录信息密度低（一行可能只有一个时间戳和一个状态码）、有价值的信息分散在多行关联记录中、大量重复的模板化内容。

```python
import re
from datetime import datetime
from collections import defaultdict
from llama_index.core import Document


class LogFileParser:
    """结构化日志文件解析器"""

    LOG_PATTERN = re.compile(
        r'(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+'
        r'\[(?P<level>\w+)\]\s+'
        r'(?P<logger>[^\]]+)\s+-\s+'
        r'(?P<message>.*)'
    )

    SESSION_WINDOW_SECONDS = 300  # 5分钟内的相关日志归为一个会话

    def parse_log_file(self, filepath: str) -> List[Document]:
        with open(filepath) as f:
            raw_lines = f.readlines()

        entries = []
        for line_num, line in enumerate(raw_lines, 1):
            match = self.LOG_PATTERN.match(line.strip())
            if match:
                entry = match.groupdict()
                entry["line_number"] = line_num
                entries.append(entry)

        sessions = self._group_into_sessions(entries)
        documents = []

        for session_id, session_entries in sessions.items():
            text = self._format_session(session_entries)
            doc = Document(
                text=text,
                metadata={
                    "session_id": session_id,
                    "start_time": session_entries[0]["timestamp"],
                    "end_time": session_entries[-1]["timestamp"],
                    "entry_count": len(session_entries),
                    "log_levels": list(set(e["level"] for e in session_entries)),
                    "source_file": filepath,
                    "source_type": "log",
                },
            )
            documents.append(doc)

        return documents

    def _group_into_sessions(self, entries):
        """按时间窗口将日志条目分组为会话"""
        sessions = defaultdict(list)
        current_session_id = 0
        session_start = None

        for entry in entries:
            ts = datetime.strptime(entry["timestamp"], "%Y-%m-%d %H:%M:%S")

            if session_start is None:
                session_start = ts
            elif (ts - session_start).total_seconds() > self.SESSION_WINDOW_SECONDS:
                current_session_id += 1
                session_start = ts

            sessions[current_session_id].append(entry)

        return sessions

    def _format_session(self, entries) -> str:
        """将会话格式化为可读文本"""
        lines = [f"日志会话 ({entries[0]['timestamp']} ~ {entries[-1]['timestamp']})"]
        lines.append(f"共 {len(entries)} 条日志\n")

        error_entries = [e for e in entries if e["level"] == "ERROR"]
        if error_entries:
            lines.append("[错误摘要]")
            for e in error_entries:
                lines.append(f"  [{e['timestamp']}] {e['message']}")
            lines.append("")

        lines.append("[详细日志]")
        for e in entries:
            level_marker = "❌" if e["level"] == "ERROR" else "⚠️" if e["level"] == "WARN" else "✅"
            lines.append(f"  {level_marker} [{e['timestamp']}] [{e['level']}] {e['message']}")

        return "\n".join(lines)


parser = LogFileParser()
log_docs = parser.parse_log_file("/var/log/application.log")
print(f"从日志文件中生成了 {len(log_docs)} 个会话文档")
```

LogFileParser 的核心思想是**把离散的日志行聚合为有意义的"会话"**——一个由相关事件组成的序列，而不是孤立的单行。这样做的原因是：单个日志行几乎不包含任何可回答问题的信息（"2025-01-15 10:23:01 [ERROR] Connection timeout" 这一行本身没什么用），但一组相关的日志行可以讲述一个完整的故事（连接超时 → 重试 → 成功/失败 → 触发告警）。

## 复合解析管道

对于特别复杂的场景，你可以把多个 Parser 串联成一个**处理管道**：

```python
class CompositeParsingPipeline:
    """复合解析管道 — 多阶段处理"""

    def __init__(self):
        self.stages = [
            ("preprocess", self._preprocess),
            ("classify", self._classify),
            ("parse", self._parse_by_type),
            ("enrich", self._enrich_metadata),
            ("validate", self._validate),
        ]

    def process(self, documents: List[Document]) -> List[TextNode]:
        results = documents

        for stage_name, stage_fn in self.stages:
            print(f"├─ 执行阶段: {stage_name}")
            results = stage_fn(results)
            print(f"│  └─ 输出: {len(results)} 个节点")

        return results

    def _preprocess(self, docs):
        """预处理：统一编码、去空内容、标准化空白"""
        cleaned = []
        for doc in docs:
            text = doc.text.strip()
            if len(text) < 10:
                continue
            text = re.sub(r'\n{3,}', '\n\n', text)  # 去除多余空行
            text = re.sub(r'[ \t]+', ' ', text)       # 标准化空白
            cleaned.append(Document(text=text, metadata=doc.metadata))
        return cleaned

    def _classify(self, docs):
        """分类：标记每个文档的类型"""
        for doc in docs:
            doc.metadata["_doc_type"] = classify_document(doc)
        return docs

    def _parse_by_type(self, docs):
        """按类型分发到对应 Parser"""
        parsed = []
        for doc in docs:
            doc_type = doc.metadata.get("_doc_type")
            parser = self._get_parser_for_type(doc_type)
            nodes = parser.get_nodes_from_documents([doc])
            parsed.extend(nodes)
        return parsed

    def _enrich_metadata(self, nodes):
        """元数据增强"""
        for node in nodes:
            node.metadata["char_count"] = len(node.text)
            node.metadata["estimated_tokens"] = len(node.text) // 3
            node.metadata["parsed_at"] = datetime.now().isoformat()
        return nodes

    def _validate(self, nodes):
        """质量验证"""
        valid = []
        for node in nodes:
            if self._is_valid_node(node):
                valid.append(node)
            else:
                print(f"⚠️ 节点被过滤: {node.node_id[:8]}... "
                      f"(长度={len(node.text)})")
        return valid

    def _is_valid_node(self, node):
        """验证节点是否合格"""
        if len(node.text) < 20:
            return False
        if node.text.count("\n") > len(node.text) / 10:
            return False  # 换行过多，可能是格式问题
        return True


pipeline = CompositeParsingPipeline()
documents = SimpleDirectoryReader("./raw_data").load_data()
nodes = pipeline.process(documents)
```

这个管道展示了生产级解析系统的典型架构——**多阶段、可插拔、每个阶段职责单一**。你可以轻松地在任意位置插入新的处理阶段（如敏感信息脱敏、PII 检测）或替换现有阶段的实现，而不影响其他部分。

## 解析质量监控

无论你的 Parser 设计得多精妙，都需要持续监控其效果。以下是一些实用的监控指标：

```python
class ParsingQualityMonitor:
    """解析质量监控器"""

    def __init__(self):
        self.metrics = {
            "total_documents": 0,
            "total_nodes": 0,
            "avg_node_length": 0,
            "nodes_too_short": 0,
            "nodes_too_long": 0,
            "type_distribution": {},
        }

    def record(self, documents, nodes):
        """记录一次解析的结果"""
        self.metrics["total_documents"] += len(documents)
        self.metrics["total_nodes"] += len(nodes)

        lengths = [len(n.text) for n in nodes]
        self.metrics["avg_node_length"] = sum(lengths) / max(len(lengths), 1)
        self.metrics["nodes_too_short"] = sum(1 for l in lengths if l < 30)
        self.metrics["nodes_too_long"] = sum(1 for l in lengths if l > 2000)

        for n in nodes:
            t = n.metadata.get("doc_type", "unknown")
            self.metrics["type_distribution"][t] = \
                self.metrics["type_distribution"].get(t, 0) + 1

    def report(self):
        """生成质量报告"""
        m = self.metrics
        print("=" * 50)
        print("解析质量报告")
        print("=" * 50)
        print(f"输入文档数: {m['total_documents']}")
        print(f"输出节点数: {m['total_nodes']}")
        print(f"平均节点长度: {m['avg_node_length']:.0f} 字符")
        print(f"过短节点 (<30字符): {m['nodes_too_short']} "
              f"({m['nodes_too_short']/max(m['total_nodes'],1)*100:.1f}%)")
        print(f"过长节点 (>2000字符): {m['nodes_too_long']} "
              f"({m['nodes_too_long']/max(m['total_nodes'],1)*100:.1f}%)")
        print(f"\n类型分布:")
        for t, count in sorted(m["type_distribution"].items(),
                                key=lambda x: -x[1]):
            pct = count / max(m["total_nodes"], 1) * 100
            bar = "█" * int(pct / 2)
            print(f"  {t:20s}: {count:5d} ({pct:5.1f}%) {bar}")


monitor = ParsingQualityMonitor()
nodes = router.get_nodes_from_documents(documents)
monitor.record(documents, nodes)
monitor.report()
```

典型的健康输出应该类似这样：

```
==================================================
解析质量报告
==================================================
输入文档数: 147
输出节点数: 2,891
平均节点长度: 347 字符
过短节点 (<30字符): 23 (0.8%)     ← 应该 < 5%
过长节点 (>2000字符): 45 (1.6%)   ← 应该 < 5%

类型分布:
  markdown           :   890 (30.8%) ███████████████████
  default            :   756 (26.1%) █████████████████
  code               :   680 (23.5%) ██████████████
  json               :   321 (11.1%) ███████
  email              :   168 (5.8%)  ███
  log                :    76 (2.6%)  ██
```

如果发现某类指标异常（如过短节点比例超过 5%），就应该回头检查对应的 Parser 配置是否合理。
