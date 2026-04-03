---
title: MCP 完整实战
description: 端到端部署指南、Claude Desktop / Cursor 集成、调试技巧与常见问题
---
# MCP 实战：从开发到部署

这一节把 MCP Server 的开发和安全加固整合成一个完整的部署方案。

## 一键启动脚本

```python
import asyncio
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
import pandas as pd

app = Server("pandas-server", version="1.0.0")

@app.list_tools()
async def list_tools():
    return [
        Tool(name="head", description="查看数据前 N 行",
             inputSchema={"type":"object","properties":{
                 "path":{"type":"string"},"n":{"type":"number","default":5}}}),
        Tool(name="stats", description="统计摘要 + 缺失值报告"),
        Tool(name="top_n", description="按某列取 Top N",
             inputSchema={"type":"object","properties":{
                 "path":{"type":"string"},
                 "col":{"type":"string"},
                 "n":{"type":"number","default":10}}}),
    ]

@app.call_tool()
async def call_tool(name, args):
    try:
        df = pd.read_csv(args["path"])
        if name == "head":
            return [TextContent(text=df.head(int(args.get("n", 5))).to_string())]
        elif name == "stats":
            info = f"Shape: {df.shape}\n"
            info += f"Memory: {df.memory_usage(deep=True).sum()/1024**2:.1f} MB\n"
            info += f"Missing:\n{df.isna().sum().to_string()}\n"
            if len(df.select_dtypes('number').columns) > 0:
                info += f"\n{df.describe().to_string()}"
            return [TextContent(text=info)]
        elif name == "top_n":
            col = args["col"]
            n = int(args.get("n", 10))
            top = df.nlargest(n, col)
            return [TextContent(text=top[[col]].head(n).to_string())]
    except Exception as e:
        return [TextContent(text=f"Error: {e}")]

if __name__ == "__main__":
    asyncio.run(stdio_server(app))
```

## Claude Desktop 配置

在 `~/Library/Application Support/Claude/claude_desktop_config.json` 中添加：

```json
{
  "mcpServers": {
    "pandas-data": {
      "command": "python3",
      "args": ["/path/to/pandas_mcp_server.py"],
      "env": {"DATA_DIR": "/Users/you/data"}
    }
  }
}
```

重启 Claude Desktop 后，你就可以在对话中说"帮我看看 data 目录下最新 CSV 的前 10 行和统计摘要"，Claude 会自动调用你的 MCP Server 来完成查询。
