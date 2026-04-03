---
title: MCP Server 与 Pandas 集成
description: Model Context Protocol 基础、构建 Pandas MCP Server、让 Claude/Cursor 等工具直接操作 DataFrame
---
# MCP Server：让 AI 工具直接操作你的数据

MCP（Model Context Protocol）是一个开放协议，让 LLM 应用（如 Claude Desktop、Cursor、VS Code Copilot）能够通过标准化的方式连接外部数据源。这一节我们用 Python 构建一个 **Pandas MCP Server**——让任何支持 MCP 的 AI 工具都能直接查询和分析你的 DataFrame。

## 为什么需要 MCP

没有 MCP 时，如果你想让 Claude 分析一份 CSV 文件，你需要：
1. 手动把文件内容复制粘贴到对话框里（文件大了就放不下）
2. 或者写代码把数据导出成某种格式再导入

有了 MCP Server，Claude 可以**直接通过协议调用你本地运行的 Python 函数**来读取和操作 DataFrame——不需要复制粘贴，也不需要导出导入。

## 最简实现

```python
from mcp.server import Server
from mcp.types import Tool, TextContent
import pandas as pd

app = Server("pandas-server")

@app.list_tools()
async def list_tools():
    return [
        Tool(name="head", description="查看 DataFrame 前 N 行"),
        Tool(name="describe", description="获取统计摘要"),
        Tool(name="query", description="用 pandas 查询数据"),
    ]

@app.call_tool()
async def call_tool(name, arguments):
    if name == "head":
        df = pd.read_csv(arguments["path"])
        return [TextContent(type="text", text=df.head(int(arguments.get("n", 5))).to_string())]
    elif name == "describe":
        df = pd.read_csv(arguments["path"])
        return [TextContent(type="text", text=df.describe().to_string())]
```

这个最简 Server 暴露了三个工具：`head`（看前几行）、`describe`（统计摘要）、`query`（自定义查询）。每个工具接收参数后执行对应的 Pandas 操作，把结果以文本形式返回给 LLM。
