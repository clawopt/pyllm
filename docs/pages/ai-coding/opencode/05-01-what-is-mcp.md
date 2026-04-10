# 5.1 MCP 是什么？Model Context Protocol 详解

> **MCP 是 AI 工具扩展的"USB 接口"——有了它，AI 不再是孤岛，而是能连接 GitHub、数据库、浏览器等任何外部工具的开放平台。**

---

## 这一节在讲什么？

前面几章我们用的都是 OpenCode 的内置工具——READ、WRITE、RUN、SEARCH、DIAGNOSTICS。这些工具覆盖了基本的编码操作，但它们有一个根本限制：**AI 只能操作本地文件和命令，无法连接外部服务**。它不能帮你创建 GitHub Issue，不能查询数据库，不能操作浏览器。MCP（Model Context Protocol）就是为了打破这个限制而生的——它是一个开放协议，让 AI 能通过标准化的接口连接任意外部工具和数据源。这一节我们把 MCP 的定位、架构、能力讲清楚，帮你理解为什么 MCP 是 OpenCode 最强大的扩展能力。

---

## MCP 的定位：AI 工具扩展的标准协议

MCP（Model Context Protocol）是 Anthropic 在 2024 年底提出的一个开放协议，它的目标是解决一个核心问题：**AI 怎么跟外部工具交互？**

在 MCP 出现之前，每个 AI 工具都有自己的"插件系统"——Cursor 有 Cursor Extensions，Copilot 有 Copilot Plugins，Claude Code 没有插件系统。这些系统互不兼容——你给 Cursor 写的插件不能在 Copilot 里用，反之亦然。

MCP 的思路完全不同——它定义了一个**标准化的协议**，让任何 AI 工具都能通过这个协议连接任何外部工具。就像 USB 接口统一了外设连接一样，MCP 统一了 AI 工具的扩展方式：

```
MCP 之前的世界：

  Cursor Extensions ←→ Cursor
  Copilot Plugins  ←→ Copilot
  Claude Code      ←→ 无扩展

  每个工具都有自己的扩展方式，互不兼容

MCP 之后的世界：

  GitHub MCP Server  ──┐
  PostgreSQL MCP     ──┤
  Playwright MCP     ──┼──→ MCP 协议 ──→ OpenCode / Claude Desktop / Cursor / ...
  Filesystem MCP     ──┤
  Custom MCP         ──┘

  一套 MCP 服务器，所有 AI 工具都能用
```

---

## MCP 的架构：Client-Server 模式

MCP 采用 Client-Server 架构——OpenCode 是 Client，外部工具是 Server：

```
┌──────────────────┐     MCP 协议      ┌──────────────────┐
│                  │  ←─────────────→  │                  │
│  OpenCode        │                   │  MCP Server      │
│  (MCP Client)    │                   │  (GitHub/DB/...) │
│                  │                   │                  │
└──────────────────┘                   └──────────────────┘

  - OpenCode 发送请求                  - MCP Server 处理请求
  - 接收响应                           - 返回结果
  - 展示给用户                         - 执行实际操作
```

**MCP Client（OpenCode）** 的职责：
- 发现 MCP Server 提供的工具列表
- 将 AI 的工具调用请求发送给 MCP Server
- 接收 MCP Server 的响应并返回给 AI

**MCP Server（外部工具）** 的职责：
- 声明自己提供哪些工具
- 接收并执行工具调用请求
- 返回执行结果

这种架构的好处是**解耦**——OpenCode 不需要知道每个外部工具的具体实现，只需要遵循 MCP 协议。同样，MCP Server 不需要知道谁在调用它，只需要遵循 MCP 协议。

---

## MCP 的三种能力

MCP 定义了三种能力类型：

### Tools：AI 可以调用的函数

这是最常用的能力类型。Tools 让 AI 能执行操作——创建 Issue、查询数据库、操作浏览器等。

```
Tools 的典型示例：

  GitHub MCP Server 提供的 Tools：
  - create_issue：创建 GitHub Issue
  - create_pull_request：创建 Pull Request
  - search_code：搜索 GitHub 代码
  - list_commits：查看提交历史

  PostgreSQL MCP Server 提供的 Tools：
  - query：执行 SQL 查询
  - list_tables：列出所有表
  - describe_table：查看表结构

  Playwright MCP Server 提供的 Tools：
  - navigate：打开网页
  - screenshot：截取屏幕截图
  - click：点击元素
  - fill：填写表单
```

### Resources：AI 可以读取的数据

Resources 让 AI 能读取外部数据——文档、API 文档、配置信息等。跟 Tools 不同，Resources 是只读的。

```
Resources 的典型示例：

  - API 文档：让 AI 了解你的 API 规范
  - 数据库 Schema：让 AI 知道表结构
  - 项目配置：让 AI 了解部署配置
```

### Prompts：预定义的提示模板

Prompts 是预定义的提示词模板，让用户能快速执行常见操作。

```
Prompts 的典型示例：

  - "review-pr"：审查 PR 的标准提示词
  - "debug-error"：调试错误的标准提示词
  - "write-test"：编写测试的标准提示词
```

---

## MCP 与 OpenAI Function Calling 的对比

如果你用过 OpenAI 的 Function Calling，可能会问：MCP 跟 Function Calling 有什么区别？

| 维度 | MCP | OpenAI Function Calling |
|------|-----|------------------------|
| 协议性质 | 开放标准，任何 AI 工具都能用 | OpenAI 私有 API，只适用于 OpenAI 模型 |
| 工具定义 | Server 端声明，Client 自动发现 | 开发者在代码中硬编码 |
| 工具执行 | Server 端执行，Client 只负责调用 | 开发者自己实现执行逻辑 |
| 扩展性 | 添加新工具只需启动新 MCP Server | 需要修改代码并重新部署 |
| 跨工具兼容 | 一套 MCP Server，所有 AI 工具都能用 | 只能在 OpenAI 的 API 中使用 |

简单来说：**Function Calling 是 OpenAI 的私有方案，MCP 是行业开放标准**。如果你用 Function Calling 写了一个工具，它只能在 OpenAI 的 API 中使用；如果你用 MCP 写了一个工具，它可以在 OpenCode、Claude Desktop、Cursor 等任何支持 MCP 的 AI 工具中使用。

---

## MCP 的生态现状

MCP 虽然是 Anthropic 提出的，但它已经成为行业共识——越来越多的 AI 工具开始支持 MCP：

**支持 MCP 的 AI 工具**：
- OpenCode（原生支持）
- Claude Desktop（原生支持）
- Cursor（部分支持）
- Windsurf（部分支持）
- VS Code + Cline 扩展（通过扩展支持）

**常见的 MCP Server**：
- `@modelcontextprotocol/server-github`：GitHub API 集成
- `@modelcontextprotocol/server-postgres`：PostgreSQL 数据库
- `@modelcontextprotocol/server-filesystem`：文件系统访问
- `@playwright/mcp`：浏览器自动化
- `context7`：上下文增强
- `shadcn`：UI 组件添加
- `supabase`：Supabase 数据库

MCP 生态正在快速发展——新的 MCP Server 不断涌现，覆盖越来越多的场景。

---

## 常见误区

**误区一：MCP 只是 OpenCode 的功能**

不是。MCP 是 Anthropic 提出的开放协议，很多 AI 工具都在支持——Claude Desktop、Cursor、Windsurf 等。你在 OpenCode 里配置的 MCP Server，也可以在 Claude Desktop 里用。MCP 的价值在于"写一次，到处用"。

**误区二：MCP 很复杂，只有高级用户才能用**

不是。MCP 的配置很简单——你只需要在 opencode.json 中添加几行 JSON，指定 MCP Server 的启动命令或 URL。不需要写代码，不需要理解协议细节。社区已经提供了大量现成的 MCP Server，你直接配置就能用。

**误区三：MCP 不安全——让 AI 连接外部工具太危险**

MCP 有完善的安全模型：工具权限控制（你可以在配置中限制 AI 能使用哪些 MCP 工具）、确认机制（敏感操作需要用户确认）、最小权限原则（MCP Server 只声明自己需要的权限）。安全风险是可控的——关键是只启用你信任的 MCP Server，不要随便添加来源不明的 MCP。

**误区四：MCP 跟 OpenCode 的内置工具重复**

不重复。内置工具操作的是本地文件和命令（READ、WRITE、RUN、SEARCH、DIAGNOSTICS），MCP 工具操作的是外部服务（GitHub、数据库、浏览器）。它们是互补的——内置工具是"基础能力"，MCP 工具是"扩展能力"。

---

## 小结

这一节我们建立了对 MCP 的完整认知：它是 AI 工具扩展的开放标准协议，采用 Client-Server 架构，提供三种能力（Tools 执行操作、Resources 读取数据、Prompts 预定义模板）。MCP 跟 OpenAI Function Calling 的根本区别在于"开放 vs 私有"——MCP 是行业标准，一套 Server 所有 AI 工具都能用。下一节我们学习怎么在 OpenCode 中配置 MCP 服务器。
