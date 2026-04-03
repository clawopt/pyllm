---
title: MCP 协议与 Pandas
description: MCP 核心概念、为什么需要 MCP、Pandas MCP Server 的架构设计
---
# MCP：让 AI 工具连接你的 Pandas 数据

MCP（Model Context Protocol）是 Anthropic 在 2024 年底推出的开放协议，目标是解决一个根本问题：**每个 AI 工具都有自己的插件格式，开发者需要为每个工具单独写集成**。MCP 统一了这个接口——你写一次 Server，所有支持 MCP 的 Client（Claude Desktop、Cursor、VS Code、Zed 等）都能用。

## MCP 的核心模型

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ Claude      │     │ Cursor      │     │ VS Code     │
│ Desktop     │     │             │     │ Copilot     │
└──────┬──────┘     └──────┬──────┘     └──────┬──────┘
       │                   │                   │
       └───────────────────┼───────────────────┘
                           │  MCP Protocol (JSON-RPC)
                           ▼
                   ┌───────────────┐
                   │  MCP Server   │
                   │  (你的代码)    │
                   │               │
                   │  ├── head()   │
                   │  ├── query()  │
                   │  ├── stats()  │
                   │  └── export() │
                   └───────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  Pandas DF    │
                    │  / CSV / DB   │
                    └───────────────┘
```

MCP 定义了两种角色：**Client**（AI 工具端）和 **Server**（数据提供端）。Server 暴露一组"工具"（Tools），Client 可以按需调用。协议基于 JSON-RPC 2.0，传输层支持 stdio（本地进程间通信）和 SSE（网络通信）。
