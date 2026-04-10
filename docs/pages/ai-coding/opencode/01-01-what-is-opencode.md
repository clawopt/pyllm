# 1.1 OpenCode 是什么？终端原生 AI 编程助手

> **如果你已经习惯了在 IDE 里点按钮让 AI 写代码，OpenCode 会让你重新思考——AI 编程助手为什么一定要绑在编辑器上？**

---

## 这一节在讲什么？

你可能已经用过 Cursor、GitHub Copilot 或者 Claude Code，它们各有各的好——Cursor 嵌在 VS Code 里，Copilot 活在编辑器侧边栏，Claude Code 跑在终端里。但你有没有想过一个问题：**为什么 AI 编程助手一定要和某个编辑器或者某个模型绑定？** OpenCode 给出了一个不同的答案：它是一个完全开源的、运行在终端里的 AI 编程助手，支持 75+ 个 LLM 提供商，通过 MCP 协议连接外部工具，不绑定任何厂商、不强制订阅、不锁定编辑器。这一节我们要把 OpenCode 的定位讲清楚，帮你理解它为什么存在、它解决了什么问题、以及它跟其他 AI 编程工具有什么根本区别。

---

## OpenCode 的定位：开源、终端、多模型

OpenCode 是一个开源的 AI Coding Agent，用 Go 语言编写，运行在你的终端里。它不是一个 IDE 插件——你不需要安装 VS Code，不需要打开任何编辑器，只需要在终端里输入 `opencode`，就能获得一个拥有完整 Agent 能力的 AI 编程助手。

所谓"Agent 能力"，是指 OpenCode 不只是帮你补全代码或者回答问题——它能理解你的项目上下文，读取和修改文件，执行 shell 命令，搜索代码，操作 Git，甚至通过 MCP 协议连接 GitHub、数据库、浏览器等外部工具。你可以把它理解为一个"住在终端里的 AI 搭档"，你跟它对话，它帮你干活。

OpenCode 的三个核心关键词：

**开源（Open Source）**：OpenCode 的代码完全开源，采用 MIT 许可证。这意味着你可以审查它的代码、修改它的行为、甚至自己编译一个定制版本。相比之下，Cursor 和 Claude Code 都是闭源的——你不知道它们把你的代码发送到了哪里、做了什么处理。

**终端（Terminal）**：OpenCode 运行在终端里，提供 TUI（Terminal User Interface）界面。这不是简陋的命令行交互——OpenCode 的 TUI 界面有对话区、文件变更区、会话列表，甚至支持主题切换。终端意味着它不依赖任何编辑器，你可以在 SSH 远程服务器上用它，可以在 tmux 里用它，可以在任何你能打开终端的地方用它。

**多模型（Multi-Provider）**：OpenCode 支持 75+ 个 LLM 提供商，包括 Anthropic Claude、OpenAI GPT-4、Google Gemini、DeepSeek、Groq、Ollama 本地模型、AWS Bedrock、Azure OpenAI、OpenRouter 等。你可以在同一个会话里切换模型，简单任务用便宜模型，复杂任务用强模型。相比之下，Claude Code 只能用 Claude，Copilot 只能用 OpenAI 的模型。

---

## 与其他 AI 编程工具的对比

理解 OpenCode 最好的方式是把它放在 AI 编程工具的生态里对比。目前主流的 AI 编程工具有这么几类：

```
AI 编程工具的分类：

  1. IDE 内嵌型：Cursor、Windsurf
     → 嵌入 IDE，提供可视化编辑体验
     → 闭源，订阅制，模型选择有限

  2. 编辑器插件型：GitHub Copilot、Codeium
     → 作为 VS Code/JetBrains 插件运行
     → 闭源，订阅制，主要做代码补全

  3. 终端 Agent 型：Claude Code、OpenCode、Aider
     → 运行在终端，Agent 模式
     → 能读写文件、执行命令、操作 Git

  4. Web 对话型：ChatGPT、Claude.ai
     → 浏览器里对话，代码需要手动复制
     → 无法直接操作项目文件
```

让我们把 OpenCode 和几个最接近的竞品做详细对比：

| 维度 | OpenCode | Cursor | GitHub Copilot | Claude Code | Aider |
|------|----------|--------|---------------|-------------|-------|
| 开源 | ✅ MIT | ❌ | ❌ | ❌ | ✅ Apache 2.0 |
| 运行环境 | 终端 TUI | IDE | IDE 插件 | 终端 CLI | 终端 CLI |
| 模型选择 | 75+ 提供商 | 有限 | 仅 OpenAI | 仅 Claude | 多模型 |
| MCP 支持 | ✅ | 有限 | ❌ | ❌ | ❌ |
| LSP 集成 | ✅ | ✅ | ✅ | ❌ | ❌ |
| 本地模型 | ✅ Ollama | ❌ | ❌ | ❌ | ✅ |
| 会话管理 | ✅ 多会话 | ❌ | ❌ | ❌ | ❌ |
| Plan/Build 模式 | ✅ | ❌ | ❌ | ❌ | ❌ |
| 价格 | 免费（模型费自付） | $20/月 | $10/月 | 按量付费 | 免费 |

从这个对比中你能看出 OpenCode 的独特价值：**它是唯一一个同时具备开源、终端原生、多模型、MCP 扩展、LSP 集成这五个特性的 AI 编程工具**。

Claude Code 也是终端 Agent，但它闭源且只能用 Claude 模型；Aider 也开源，但它没有 TUI 界面、没有 MCP 支持、没有 LSP 集成；Cursor 有最好的 IDE 集成体验，但它闭源、订阅制、模型选择有限。OpenCode 的定位是——**给那些想要完全掌控 AI 编程体验的开发者，一个开源的、可扩展的、不绑定厂商的终端 Agent**。

---

## OpenCode 的核心能力

OpenCode 的能力可以分成五个层面来理解：

**1. 多模型支持（75+ LLM 提供商）**

这是 OpenCode 最核心的差异化能力。你不需要为不同的模型安装不同的工具——在 OpenCode 里，你可以随时切换模型：

```
# 在 TUI 里用 /models 命令切换模型
/models

# 或者用 -m 参数启动时指定模型
opencode -m anthropic/claude-sonnet-4-20250514
opencode -m openai/gpt-4o
opencode -m google/gemini-2.5-pro
opencode -m ollama/deepseek-coder:6.7b
```

你甚至可以配置三档模型——default（日常使用）、big（复杂任务）、fast（简单任务）——让 OpenCode 根据任务复杂度自动选择：

```json
{
  "agents": {
    "coder": {
      "model": "anthropic/claude-sonnet-4-20250514",
      "maxTokens": 16384
    },
    "task": {
      "model": "anthropic/claude-sonnet-4-20250514",
      "maxTokens": 16384
    }
  }
}
```

**2. 终端 TUI 界面**

OpenCode 的 TUI 界面不是简陋的命令行问答——它有完整的交互体验：

- **对话区**：显示你和 AI 的对话，支持 Markdown 渲染
- **文件变更区**：实时显示 AI 修改了哪些文件，diff 高亮
- **会话列表**：管理多个并行会话，随时切换
- **MCP 状态栏**：显示 MCP 服务器的连接状态
- **模式指示器**：右下角显示当前是 Plan 模式还是 Build 模式

**3. Agent 工具系统**

OpenCode 的 AI 不是只会聊天——它能调用一系列工具来操作你的项目：

| 工具 | 能力 | 示例 |
|------|------|------|
| READ | 读取文件内容 | AI 读取你的源码来理解上下文 |
| WRITE | 创建或修改文件 | AI 直接修改你的代码文件 |
| RUN | 执行 shell 命令 | AI 运行测试、安装依赖 |
| SEARCH | 搜索代码 | AI 在项目中搜索相关代码 |
| DIAGNOSTICS | 代码诊断（LSP） | AI 获取类型错误、lint 警告 |

**4. MCP 协议扩展**

MCP（Model Context Protocol）是 Anthropic 提出的开放协议，让 AI 能连接外部工具和数据源。OpenCode 原生支持 MCP——你可以配置 GitHub MCP 让 AI 直接操作 Issue 和 PR，配置 PostgreSQL MCP 让 AI 查询数据库，配置 Playwright MCP 让 AI 操作浏览器。这个能力是 Cursor、Copilot、Claude Code 都不具备的。

**5. LSP 集成**

OpenCode 集成了 Language Server Protocol，这意味着 AI 能获取代码的语义信息——跳转定义、查找引用、类型检查。当 AI 修改代码时，它能通过 LSP 知道修改是否引入了类型错误，而不是盲目地改了再说。

---

## 谁适合用 OpenCode

OpenCode 不是万能的——它有自己的目标用户群：

**适合用 OpenCode 的人：**
- 后端开发者：大部分时间在终端里，不需要 IDE 的可视化功能
- DevOps 工程师：经常在远程服务器上工作，SSH + tmux 是日常
- 开源贡献者：需要开源透明的工具，不想把代码发给闭源服务
- 偏好终端工作流的开发者：Vim/Neovim 用户、tmux 重度用户
- 需要多模型切换的团队：不同任务用不同模型，控制成本

**不太适合用 OpenCode 的人：**
- 前端开发者：需要 IDE 的实时预览、组件可视化，终端体验不够直观
- 编程新手：终端操作本身就有门槛，IDE 的可视化引导更友好
- 需要极致 IDE 集成的人：如果你离不开 VS Code 的调试器、Git 可视化、扩展生态，Cursor 可能更适合

---

## 常见误区

**误区一：OpenCode 是 IDE 插件**

不是。OpenCode 是一个独立的终端应用，不依赖任何编辑器。你不需要安装 VS Code，也不需要安装任何 IDE。它在终端里运行，通过 TUI 界面交互。这意味着你可以在 SSH 远程服务器上用它——这是 IDE 插件做不到的。

**误区二：OpenCode 只能用 Claude**

不是。OpenCode 支持 75+ 个 LLM 提供商，包括 OpenAI、Google Gemini、DeepSeek、Groq、Ollama 本地模型等。Claude 只是其中一个选项。你甚至可以用 GitHub Copilot 的模型——如果你有 Copilot 订阅，可以直接在 OpenCode 里用，不需要额外付费。

**误区三：OpenCode 和 Claude Code 一样**

不一样。虽然它们都是终端 Agent，但有几个关键区别：OpenCode 开源（Claude Code 闭源）、OpenCode 支持多模型（Claude Code 只能用 Claude）、OpenCode 支持 MCP 扩展（Claude Code 不支持）、OpenCode 有 TUI 界面（Claude Code 是纯 CLI）。你可以把 OpenCode 理解为"开源版的多模型 Claude Code"。

**误区四：终端里的 AI 肯定不好用**

这是很多人的第一反应，但实际体验会改变你的看法。OpenCode 的 TUI 界面有完整的对话区、文件变更追踪、会话管理，交互体验远比"命令行问答"丰富。而且终端有一个 IDE 无法比拟的优势——你可以用 SSH 在远程服务器上使用它，你可以在 tmux 里同时开多个会话，你可以用 `opencode run` 做非交互式自动化。终端不是限制，而是自由。

---

## 小结

这一节我们建立了对 OpenCode 的基本认知：它是一个开源的、终端原生的、支持 75+ 模型的 AI Coding Agent，具备 MCP 扩展和 LSP 集成能力。它跟 Cursor、Copilot、Claude Code 的根本区别在于——**开源让你可审计，终端让你不受限，多模型让你自由选，MCP 让你无限扩展**。下一节我们深入 OpenCode 的工作原理，看看它的 Agent 架构、工具系统和上下文管理是怎么运作的。
