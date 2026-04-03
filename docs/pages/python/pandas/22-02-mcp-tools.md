---
title: MCP Server 工具开发
description: 定义 Tool、实现 handler、错误处理、资源管理、完整 Pandas Server 代码
---
# 开发 Pandas MCP Server

这一节我们写一个完整的、可以实际运行的 Pandas MCP Server。它暴露四个工具：查看数据、统计摘要、执行查询和导出结果。

## 完整实现

```python
from mcp.server import Server
from mcp.types import Tool, TextContent, ImageContent
from mcp.server.stdio import stdio_server
import pandas as pd

app = Server("pandas-mcp-server")

@app.list_tools()
async def list_tools():
    return [
        Tool(name="head", description="查看 DataFrame 前 N 行",
             inputSchema={"type":"object","properties":{
                 "path":{"type":"string"},"n":{"type":"number"}}}),
        Tool(name="describe", description="获取数值列的统计摘要",
             inputSchema={"type":"object","properties":{"path":{"type":"string"}}}),
        Tool(name="query", description="执行 pandas 查询表达式",
             inputSchema={"type":"object","properties":{
                 "path":{"type":"string"},
                 "expr":{"type":"string","description":"如 df[df['score']>90]"}}}),
        Tool(name="groupby_summary", description="按列分组并汇总",
             inputSchema={"type":"object","properties":{
                 "path":{"type":"string"},
                 "col":{"type":"string"}}}),
    ]

@app.call_tool()
async def call_tool(name, arguments):
    try:
        if name == "head":
            df = pd.read_csv(arguments["path"])
            n = int(arguments.get("n", 5))
            return [TextContent(type="text", text=df.head(n).to_string())]
        elif name == "describe":
            df = pd.read_csv(arguments["path"])
            return [TextContent(type="text", text=df.describe().to_string())]
        elif name == "query":
            df = pd.read_csv(arguments["path"])
            result = eval(arguments["expr"], {"df": df}, {"pd": pd})
            return [TextContent(type="text", text=str(result))]
        elif name == "groupby_summary":
            df = pd.read_csv(arguments["path"])
            summary = df.groupby(arguments["col"]).size().sort_values(ascending=False)
            return [TextContent(type="text", text=summary.head(20).to_string())]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {e}")]

if __name__ == "__main__":
    import asyncio
    asyncio.run(stdio_server(app))
```

这个 Server 可以直接用 `python server.py` 启动，然后在 Claude Desktop 的配置文件中添加：
```json
{"mcpServers": {"pandas": {"command": "python", "args": ["server.py"]}}}
```
之后就可以在 Claude 对话中直接问"帮我看看 data.csv 里有哪些数据"了。
