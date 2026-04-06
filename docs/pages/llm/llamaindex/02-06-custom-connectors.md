---
title: 自定义连接器：如何为私有数据源编写 Reader
description: BaseReader 接口实现、自定义数据加载逻辑、LlamaHub 发布流程、企业内部数据源适配
---
# 自定义连接器：如何为私有数据源编写 Reader

前面几节我们介绍了 LlamaIndex 内置的各种连接器——文件、数据库、API、云服务，覆盖了市面上绝大多数常见的数据源。但现实世界中总有一些"特殊分子"：公司内部用了十年的一套 legacy 系统、某个小众行业的专用软件、或者一个刚上线还没有公开 SDK 的内部平台。这些数据源没有现成的连接器可用，怎么办？

答案是：**自己写一个**。

好消息是，LlamaIndex 的连接器体系设计得非常开放，编写一个自定义 Reader 只需要实现一个基类方法。这一节我们就来手把手教你如何为任意私有数据源创建自定义连接器。

## 理解 BaseReader 接口

所有 LlamaIndex 的 Reader 都继承自 `BaseReader` 类，它的定义非常简洁：

```python
from abc import ABC, abstractmethod
from typing import List, Any, Generator, Optional
from llama_index.core import Document

class BaseReader(ABC):
    @abstractmethod
    def load_data(self, *args: Any, **kwargs: Any) -> List[Document]:
        """加载数据并返回 Document 列表"""
        pass

    def lazy_load(self, *args: Any, **kwargs: Any) -> Generator[Document, None, None]:
        """惰性加载版本（可选重写）"""
        for doc in self.load_data(*args, **kwargs):
            yield doc
```

就这么简单。你唯一必须实现的方法就是 `load_data()`，它的职责很明确：**不管数据从哪里来，最终都要返回一个 `List[Document]`**。

## 实战一：内部 REST API 的 Reader

假设你们公司有一个内部的工单系统，提供了一个 REST API 来查询工单数据。API 的文档如下：

```
GET https://ticket.internal.company.com/api/v1/tickets
Headers:
  Authorization: Bearer <api-token>
  X-Department: <department-code>

Query Parameters:
  status: open/closed/resolved (可选)
  limit: 最大返回数量 (默认100, 最大1000)
  offset: 分页偏移量 (默认0)

Response:
{
  "tickets": [
    {
      "id": "TKT-001234",
      "title": "无法登录系统",
      "description": "用户反馈点击登录按钮后...",
      "status": "resolved",
      "priority": "high",
      "assignee": "zhangsan",
      "created_at": "2025-01-10T08:30:00Z",
      "resolution": "已重置用户密码...",
      "tags": ["login", "account"]
    }
  ],
  "total": 1500,
  "has_more": true
}
```

现在我们来为这个 API 写一个 Reader：

```python
import requests
from typing import List, Optional, Any
from llama_index.core import Document
from llama_index.core.readers.base import BaseReader


class InternalTicketReader(BaseReader):
    """公司内部工单系统的数据读取器"""

    def __init__(
        self,
        api_base_url: str,
        api_token: str,
        department: str,
        status: Optional[str] = None,
        max_tickets: int = 500,
    ):
        self.api_base_url = api_base_url.rstrip("/")
        self.api_token = api_token
        self.department = department
        self.status = status
        self.max_tickets = max_tickets

    def load_data(self) -> List[Document]:
        documents = []
        offset = 0
        batch_size = 100  # 每批请求数量

        while len(documents) < self.max_tickets:
            params = {
                "limit": min(batch_size, self.max_tickets - len(documents)),
                "offset": offset,
            }
            if self.status:
                params["status"] = self.status

            response = requests.get(
                f"{self.api_base_url}/api/v1/tickets",
                headers={
                    "Authorization": f"Bearer {self.api_token}",
                    "X-Department": self.department,
                },
                params=params,
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

            for ticket in data.get("tickets", []):
                text = (
                    f"# {ticket['title']}\n\n"
                    f"**问题描述:** {ticket['description']}\n\n"
                    f"**解决方案:** {ticket.get('resolution', '未解决')}\n\n"
                    f"**标签:** {', '.join(ticket.get('tags', []))}"
                )

                doc = Document(
                    text=text,
                    metadata={
                        "id": ticket["id"],
                        "title": ticket["title"],
                        "status": ticket["status"],
                        "priority": ticket["priority"],
                        "assignee": ticket["assignee"],
                        "created_at": ticket["created_at"],
                        "tags": ticket.get("tags", []),
                        "source": f"{self.api_base_url}/tickets/{ticket['id']}",
                    },
                )
                documents.append(doc)

            if not data.get("has_more", False):
                break

            offset += batch_size

        return documents


# 使用示例
reader = InternalTicketReader(
    api_base_url="https://ticket.internal.company.com",
    api_token=os.getenv("TICKET_API_TOKEN"),
    department="tech-support",
    status="resolved",  # 只加载已解决的工单（最有知识价值）
    max_tickets=200,
)

documents = reader.load_data()
print(f"加载了 {len(documents)} 条工单记录")
```

让我们来分析一下这个 Reader 的几个关键设计决策：

**分页处理：** 内部 API 通常不会一次返回所有数据（出于性能和服务稳定性考虑），所以我们的 Reader 必须处理分页逻辑。这里使用了 `offset` + `limit` 的经典分页模式，循环请求直到拿到全部数据或达到上限。

**文本格式化：** 原始的 JSON 数据不适合直接作为 RAG 的输入——字段名和值混在一起，缺乏可读性。我们在 Reader 内部就把数据格式化为结构化的 Markdown 文本（标题用 `#`，关键字段用 `**加粗**`），这样 LLM 在生成答案时更容易理解上下文。

**元数据丰富：** 工单的 ID、状态、优先级、负责人等信息被放入 metadata 而非正文文本。这样做的目的是：当用户问"高优先级的登录问题怎么解决"时，系统可以通过 metadata 过滤快速定位相关工单，而不需要在全文中搜索。

**错误处理：** 使用了 `raise_for_status()` 让 HTTP 错误以异常形式抛出，以及 `timeout=30` 防止请求无限挂起。在生产环境中你可能还想加入重试逻辑（如 `tenacity` 库）。

## 实战二：遗留数据库系统的 Reader

假设公司有一个老库存管理系统，运行在一个古老的 FoxPro 数据库上，没有标准的数据库驱动，只能通过一个命令行工具 `foxquery.exe` 来执行查询：

```python
import subprocess
import json
from typing import List
from llama_index.core import Document
from llama_index.core.readers.base import BaseReader


class LegacyInventoryReader(BaseReader):
    """遗留 FoxPro 库存系统的数据读取器"""

    def __init__(self, query_tool_path: str, db_path: str):
        self.query_tool = query_tool_path
        self.db_path = db_path

    def _execute_query(self, sql: str) -> List[dict]:
        """调用 foxquery 工具执行 SQL 并返回结果"""
        result = subprocess.run(
            [self.query_tool, "-d", self.db_path, "-q", sql],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            raise RuntimeError(f"查询失败: {result.stderr}")
        return json.loads(result.stdout)

    def load_data(self) -> List[Document]:
        rows = self._execute_query("""
            SELECT product_code, product_name, category,
                   spec, warehouse_location, quantity,
                   min_stock_level, last_restock_date
            FROM inventory
            WHERE quantity > 0
        """)

        documents = []
        for row in rows:
            text = (
                f"产品: {row['product_name']} ({row['product_code']})\n"
                f"分类: {row['category']}\n"
                f"规格: {row['spec']}\n"
                f"仓库位置: {row['warehouse_location']}\n"
                f"当前库存: {row['quantity']} 件\n"
                f"安全库存: {row['min_stock_level']} 件\n"
                f"最近补货: {row['last_restock_date']}"
            )

            doc = Document(
                text=text,
                metadata={
                    "product_code": row["product_code"],
                    "category": row["category"],
                    "warehouse": row["warehouse_location"],
                    "quantity": row["quantity"],
                    "source": "legacy_inventory_system",
                },
            )
            documents.append(doc)

        return documents


reader = LegacyInventoryReader(
    query_tool_path="/opt/tools/foxquery.exe",
    db_path="/data/inventory/inventory.dbc",
)
documents = reader.load_data()
```

这个例子展示了 LlamaIndex Reader 的强大之处：**它不关心数据到底从哪里来**——REST API、命令行工具、消息队列、串口设备……只要你能在 Python 中拿到数据，就能包装成一个 Reader。这就是抽象的力量。

## 实战三：实时数据流的 Reader

有些数据源不是静态的"一批数据"，而是持续产生的流式数据——比如聊天室的消息、IoT 设备的传感器读数、实时的股票行情。对于这类场景，可以实现一个支持持续读取的 Reader：

```python
import websocket
import json
import time
from typing import Generator
from llama_index.core import Document
from llama_index.core.readers.base import BaseReader


class WebSocketFeedReader(BaseReader):
    """基于 WebSocket 实时数据流的读取器"""

    def __init__(self, ws_url: str, auth_token: str, duration_seconds: int = 60):
        self.ws_url = ws_url
        self.auth_token = auth_token
        self.duration = duration_seconds

    def load_data(self) -> List[Document]:
        """加载指定时间窗口内的数据"""
        documents = []
        start_time = time.time()

        ws = websocket.WebSocket()
        ws.connect(
            self.ws_url,
            header={"Authorization": f"Bearer {self.auth_token}"},
        )
        ws.send(json.dumps({"type": "subscribe", "channels": ["messages"]}))

        try:
            while time.time() - start_time < self.duration:
                result = ws.recv(timeout=1)
                data = json.loads(result)

                if data.get("type") == "message":
                    doc = Document(
                        text=f"[{data['sender']}] {data['content']}",
                        metadata={
                            "channel": data.get("channel"),
                            "sender": data.get("sender"),
                            "timestamp": data.get("timestamp"),
                            "msg_id": data.get("id"),
                            "source": "websocket_feed",
                        },
                    )
                    documents.append(doc)
        except websocket.WebSocketTimeoutException:
            pass  # 超时是正常的退出条件
        finally:
            ws.close()

        return documents

    def lazy_load(self) -> Generator[Document, None, None]:
        """持续产出数据的惰性版本"""
        ws = websocket.WebSocket()
        ws.connect(
            self.ws_url,
            header={"Authorization": f"Bearer {self.auth_token}"},
        )
        ws.send(json.dumps({"type": "subscribe", "channels": ["messages"]}))

        try:
            while True:
                result = ws.recv(timeout=None)
                data = json.loads(result)
                if data.get("type") == "message":
                    yield Document(
                        text=f"[{data['sender']}] {data['content']}",
                        metadata={
                            "channel": data.get("channel"),
                            "sender": data.get("sender"),
                            "timestamp": data.get("timestamp"),
                            "source": "websocket_feed",
                        },
                    )
        except KeyboardInterrupt:
            print("\n停止接收...")
        finally:
            ws.close()


# 用法一：收集 60 秒的数据后批量处理
reader = WebSocketFeedReader(
    ws_url="wss://chat.internal.com/ws",
    auth_token=os.getenv("WS_TOKEN"),
    duration_seconds=60,
)
recent_messages = reader.load_data()

# 用法二：持续接收并实时处理
reader = WebSocketFeedReader(
    ws_url="wss://chat.internal.com/ws",
    auth_token=os.getenv("WS_TOKEN"),
)
for message_doc in reader.lazy_load():
    index.insert(message_doc)  # 实时插入索引
```

注意这里我们同时实现了 `load_data()` 和 `lazy_load()`——前者适合批处理场景（收集一段时间的数据后统一建索引），后者适合实时场景（数据来了就立即索引）。

## Reader 的设计最佳实践

### 原则一：单一职责

一个好的 Reader 应该只做一件事：**把外部数据转化为 Document 列表**。不要在 Reader 里做索引构建、向量计算、LLM 调用等事情。保持 Reader 的纯粹性让它更容易测试、复用和维护。

```python
# ❌ 错误做法：Reader 里做了太多事
class BadReader(BaseReader):
    def load_data(self):
        data = self._fetch_from_api()     # OK: 获取数据
        docs = self._format_documents(data)  # OK: 格式化
        index = VectorStoreIndex.from_documents(docs)  # ❌ 不该在这里！
        return index.query("总结")              # ❌ 更不该在这里！

# ✅ 正确做法：Reader 只负责数据转换
class GoodReader(BaseReader):
    def load_data(self):
        data = self._fetch_from_api()
        return self._format_documents(data)  # 就这么多
```

### 原则二：容错性

外部数据源是不可靠的——网络会断、API 会报错、数据会有脏值。好的 Reader 应该能够优雅地处理这些情况：

```python
class RobustReader(BaseReader):
    def load_data(self) -> List[Document]:
        documents = []
        raw_items = self._fetch_all_items()

        for item in raw_items:
            try:
                doc = self._to_document(item)
                if doc and len(doc.text.strip()) > 10:  # 过滤太短的内容
                    documents.append(doc)
            except Exception as e:
                print(f"跳过无效项 {item.get('id')}: {e}")
                continue

        return documents
```

### 原则三：可配置性

不要把任何东西写死——API 地址、字段映射、过滤条件等都应该是可配置的参数：

```python
class ConfigurableReader(BaseReader):
    def __init__(
        self,
        endpoint: str,
        auth_token: str,
        fields_to_include: List[str] = None,
        text_template: str = None,
        max_items: int = 1000,
        exclude_empty: bool = True,
    ):
        self.endpoint = endpoint
        self.auth_token = auth_token
        self.fields_to_include = fields_to_include or []
        self.text_template = text_template or "{title}: {body}"
        self.max_items = max_items
        self.exclude_empty = exclude_empty
```

## 测试自定义 Reader

写完 Reader 后，测试非常重要。建议至少覆盖以下场景：

```python
import pytest
from unittest.mock import patch, MagicMock

def test_ticket_reader_basic():
    """测试基本功能"""
    with patch("requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "tickets": [{
                "id": "TKT-001",
                "title": "测试工单",
                "description": "这是一个测试",
                "status": "resolved",
                "priority": "medium",
                "assignee": "test_user",
                "created_at": "2025-01-01T00:00:00Z",
                "resolution": "已解决",
                "tags": ["test"],
            }],
            "total": 1,
            "has_more": False,
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        reader = InternalTicketReader(
            api_base_url="https://test.api.com",
            api_token="test-token",
            department="test",
        )
        docs = reader.load_data()

        assert len(docs) == 1
        assert docs[0].metadata["id"] == "TKT-001"
        assert "测试工单" in docs[0].text
        assert "resolved" in docs[0].metadata["status"]


def test_ticket_reader_pagination():
    """测试分页逻辑"""
    # Mock 多次请求模拟分页...
    pass


def test_ticket_reader_empty_result():
    """测试空结果的处理"""
    pass


def test_ticket_reader_api_error():
    """测试 API 错误的处理"""
    pass
```

## 发布到 LlamaHub（可选）

如果你写的 Reader 可能对其他人也有用，可以考虑发布到 LlamaHub 社区。步骤大致如下：

1. **Fork LlamaIndex 仓库**并在 `llama-index-integrations/` 下创建新包
2. **遵循命名规范**：`llama-index-readers-<name>`
3. **实现 BaseReader** 并添加完整的文档字符串
4. **编写测试**（覆盖率 > 80%）
5. **提交 PR** 到主仓库

发布后，其他开发者就可以通过 `pip install llama-index-readers-<your-name>` 安装并使用你的 Reader 了。

当然，如果只是公司内部使用，完全没有必要发布——放在项目的 `readers/` 目录下即可。

## 常见误区

**误区一："自定义 Reader 很难写"。** 不难。最简单的 Reader 只需要实现一个 `load_data()` 方法，返回 `List[Document]` 就行了。复杂度主要在外部数据源的接入逻辑上（认证、分页、错误处理），而不是 LlamaIndex 框架本身。

**误区二："必须继承 BaseReader"。** 技术上不强制——只要你的对象有 `load_data()` 方法返回正确的类型，LlamaIndex 就能用。但继承 BaseReader 是推荐的做法，因为它提供了 `lazy_load()` 的默认实现和一些类型提示。

**误区三:"Reader 应该处理所有边缘情况"。** 不现实的期望。Reader 应该处理常见的错误情况（网络超时、空结果、格式异常），但不应该尝试修复数据源本身的问题（如数据不一致、业务逻辑错误）。**Reader 是数据搬运工，不是数据清洁工。**
