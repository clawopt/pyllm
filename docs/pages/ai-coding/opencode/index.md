# OpenCode 教程大纲

> **终端里的 AI 编程搭档——OpenCode，开源、多模型、MCP 可扩展的 AI Coding Agent**

---

## 📖 教程概述

如果你已经用过 Cursor、GitHub Copilot 或者 Claude Code，你可能会问：为什么还需要 OpenCode？答案在于三个关键词——**开源、终端、MCP**。OpenCode 是一个完全开源的 AI 编程助手，它运行在你的终端里（TUI 界面），支持 75+ LLM 提供商（Claude、GPT-4、Gemini、DeepSeek、Ollama 本地模型等），并且通过 MCP（Model Context Protocol）协议实现了强大的工具扩展能力。它不绑定任何模型厂商、不强制订阅、不锁定编辑器——你在终端里用它，就像跟一个懂代码的搭档结对编程。

OpenCode 的核心定位是"终端原生的 AI Coding Agent"——它不只是帮你补全代码（那是 Copilot 的事），而是能理解你的项目上下文、执行命令、读写文件、搜索代码、管理 Git，甚至通过 MCP 连接外部工具（如 GitHub、数据库、浏览器自动化）。你可以把它理解为"开源版的 Claude Code"——同样的 Agent 能力，但你可以自由选择模型、自由扩展工具、完全掌控数据。

本教程共 8 章：第 1 章建立认知，第 2 章安装配置，第 3 章掌握 TUI 交互，第 4 章深入模型选择，第 5 章探索 MCP 扩展，第 6 章实战开发场景，第 7 章进阶技巧，第 8 章生产化与团队协作。教程面向有终端使用经验的开发者，从零开始带你掌握 OpenCode 的全部能力。

---

## 🗺️ 章节规划

### 第1章：认识 OpenCode

#### 1.1 OpenCode 是什么？终端原生 AI 编程助手
- **OpenCode 的定位**：开源 AI Coding Agent，运行在终端，支持多模型、MCP 扩展、LSP 集成
- **与 Cursor / GitHub Copilot / Claude Code 的对比**：
  - Cursor：IDE 内嵌、闭源、订阅制、模型选择有限
  - GitHub Copilot：编辑器插件、闭源、订阅制、仅 OpenAI 模型
  - Claude Code：终端 Agent、闭源、仅 Claude 模型、无 MCP
  - OpenCode：终端 Agent、开源、75+ 模型、MCP 扩展、免费
- **OpenCode 的核心能力**：
  - 多模型支持（75+ LLM 提供商，含本地 Ollama）
  - 终端 TUI 界面（文件引用、斜杠命令、快捷键）
  - Agent 工具系统（文件读写、命令执行、代码搜索、Git 操作）
  - MCP 协议扩展（连接 GitHub、数据库、浏览器等外部工具）
  - LSP 集成（代码智能感知、跳转定义、类型检查）
  - 会话管理（多会话并行、会话分享、自动压缩）
- **谁适合用 OpenCode**：后端开发者、DevOps 工程师、开源贡献者、偏好终端工作流的开发者
- **常见误区**：OpenCode 不是 IDE 插件——它是独立的终端应用，不依赖任何编辑器

#### 1.2 OpenCode 的工作原理：Agent + 工具 + 上下文
- **Agent 架构**：Plan 模式（规划）和 Build 模式（执行）的双模式设计
- **工具系统**：AI 可以调用的能力——文件读写、命令执行、代码搜索、Git 操作
- **上下文管理**：项目文件扫描、LSP 代码智能、会话历史、Auto Compact 自动压缩
- **数据流**：用户输入 → 上下文收集 → LLM 推理 → 工具调用 → 结果返回
- **与 Claude Code 的架构对比**：同样的 Agent 模式，但 OpenCode 开源且可扩展

#### 1.3 5 分钟快速体验
- **安装**：`curl -fsSL https://opencode.ai/install | bash`
- **启动**：`cd your-project && opencode`
- **初始化**：`/init` 命令分析项目并生成 AGENTS.md
- **第一次对话**：问项目问题、请求代码修改、执行命令
- **常见误区**：首次使用忘记 `/init`——没有项目上下文，AI 的回答会很泛泛

### 第2章：安装与配置

#### 2.1 安装方式全览
- **一键安装脚本**（推荐）：`curl -fsSL https://opencode.ai/install | bash`
- **Homebrew**：`brew install anomalyco/tap/opencode`（macOS/Linux）
- **NPM**：`npm install -g opencode-ai`
- **Go Install**：`go install github.com/sst/opencode@latest`
- **AUR**：`paru -S opencode-bin`（Arch Linux）
- **Windows**：下载二进制文件
- **版本管理**：指定版本安装、升级
- **常见安装问题**：Go 版本不兼容、权限问题、网络问题

#### 2.2 配置文件详解
- **配置文件位置**：全局 `~/.config/opencode/opencode.json`、项目级 `.opencode/opencode.json`
- **配置文件结构**：
  - `providers`：模型提供商和 API Key
  - `models`：默认模型、大模型、快速模型的配置
  - `agents`：Agent 配置（模型、最大 token、工具权限）
  - `mcp`：MCP 服务器配置
  - `tui`：界面主题和快捷键
  - `data`：数据存储目录
- **环境变量配置**：`ANTHROPIC_API_KEY`、`OPENAI_API_KEY`、`GEMINI_API_KEY` 等
- **分层配置**：全局配置 + 项目配置的合并策略
- **常见误区**：把 API Key 硬编码在配置文件里并提交到 Git——应该用环境变量引用 `$ANTHROPIC_API_KEY`

#### 2.3 模型提供商配置
- **Anthropic（Claude）**：API Key 配置、Claude Pro/Max 订阅登录
- **OpenAI**：API Key 配置、ChatGPT Plus/Pro 订阅登录
- **Google Gemini**：API Key 配置、VertexAI 配置
- **DeepSeek**：API Key 配置、代码生成场景推荐
- **Groq**：高速推理场景
- **Ollama 本地模型**：安装 Ollama、下载模型、配置本地端点
- **OpenRouter**：统一接入多模型的代理服务
- **AWS Bedrock / Azure OpenAI**：企业级部署
- **常见误区**：以为 OpenCode 只能用 Claude——实际上支持 75+ 提供商

### 第3章：TUI 交互与核心操作

#### 3.1 TUI 界面详解
- **界面布局**：对话区、文件变更区、MCP 状态栏、会话列表
- **Plan 模式 vs Build 模式**：Tab 键切换
  - Plan：只规划不执行，适合复杂任务的方案设计
  - Build：直接执行文件修改和命令，适合快速迭代
- **快捷键大全**：Ctrl+C（取消）、Ctrl+L（清屏）、Ctrl+J（换行）、Tab（模式切换）、Esc（返回）
- **主题配置**：tokyo-night、catppuccin 等内置主题
- **常见误区**：一直用 Build 模式不切 Plan——复杂任务应该先 Plan 再 Build

#### 3.2 文件引用与上下文控制
- **@ 文件引用**：`@filename.ts`（引用文件）、`@src/components/`（引用目录）、`@**/*.py`（glob 模式）
- **$ 命令引用**：`$ ls -la`（执行命令并返回输出）、`$$ cat package.json`（执行并加入上下文）
- **斜杠命令大全**：
  - `/init`：初始化项目
  - `/clear`：清空会话
  - `/compact`：压缩对话历史
  - `/models`：切换模型
  - `/sessions`：管理会话
  - `/share`：分享会话链接
  - `/cost`：查看使用成本
  - `/context`：管理上下文
  - `/undo` / `/redo`：撤销/重做修改
- **上下文管理策略**：什么时候该 /clear、什么时候该 /compact、什么时候开新会话
- **常见误区**：上下文太长导致 AI 迷失——长对话应该定期 /compact 或开新会话

#### 3.3 Agent 工具系统
- **内置工具全景**：
  - 文件读写：读取、创建、编辑文件
  - 命令执行：运行 shell 命令
  - 代码搜索：grep、文件搜索
  - Git 操作：status、diff、log
  - LSP 集成：跳转定义、查找引用、类型检查
- **工具权限控制**：在 agents 配置中启用/禁用特定工具
- **安全模型**：工作目录限制、命令白名单、确认机制
- **常见误区**：让 AI 直接执行危险命令（如 `rm -rf`）——应该配置 confirm_destructive: true

### 第4章：模型选择与策略

#### 4.1 模型选择指南
- **按任务选模型**：
  - 复杂架构设计 → Claude Sonnet / GPT-4o
  - 代码生成与调试 → DeepSeek Coder / Claude Sonnet
  - 快速问答与简单修改 → Claude Haiku / GPT-4o-mini
  - 长上下文处理 → Gemini 1.5 Pro（100 万 token 上下文）
  - 隐私敏感代码 → Ollama 本地模型
- **模型配置**：default / big / fast 三档模型配置
- **动态切换**：`/models` 命令在会话中切换模型
- **成本优化**：简单任务用 fast 模型、复杂任务用 big 模型
- **常见误区**：所有任务都用最贵的模型——简单任务用 Haiku 就够了

#### 4.2 Ollama 本地模型
- **为什么用本地模型**：隐私保护、零成本、离线可用
- **安装 Ollama**：`curl -fsSL https://ollama.ai/install | bash`
- **推荐模型**：
  - llama3.2:3b（2GB，快速通用）
  - deepseek-coder:6.7b（4GB，代码生成）
  - codestral:7b（4GB，代码补全）
- **配置 OpenCode 使用 Ollama**：`"ollama": {"baseUrl": "http://localhost:11434"}`
- **本地模型的局限**：推理能力不如云端大模型、长上下文处理能力有限
- **常见误区**：本地模型能替代 Claude/GPT-4——目前本地模型适合辅助任务，复杂任务还是需要云端大模型

#### 4.3 Auto Compact 与上下文管理
- **Auto Compact 机制**：对话接近上下文窗口 95% 时自动压缩
- **手动 /compact**：主动压缩对话历史
- **会话管理**：多会话并行、会话切换、会话分享
- **上下文窗口的利用策略**：如何有效利用有限的上下文窗口
- **常见误区**：Auto Compact 会丢失重要信息——实际上它会保留关键上下文的摘要

### 第5章：MCP 协议与工具扩展

#### 5.1 MCP 是什么？Model Context Protocol 详解
- **MCP 的定位**：AI 工具扩展的标准协议——让 AI 能连接外部工具和数据源
- **MCP 的架构**：Client-Server 模式——OpenCode 是 Client，外部工具是 Server
- **MCP 的能力**：
  - Tools：AI 可以调用的函数（如创建 GitHub Issue、查询数据库）
  - Resources：AI 可以读取的数据（如文档、API 文档）
  - Prompts：预定义的提示模板
- **与 OpenAI Function Calling 的对比**：MCP 是开放协议，不绑定任何模型
- **常见误区**：MCP 只是 OpenCode 的功能——MCP 是 Anthropic 提出的开放协议，很多 AI 工具都在支持

#### 5.2 配置 MCP 服务器
- **MCP 配置方式**：在 opencode.json 中配置 mcp 字段
- **本地 MCP 服务器**：`"type": "local"`，通过命令行启动
- **远程 MCP 服务器**：`"type": "remote"`，通过 HTTP/SSE 连接
- **常用 MCP 服务器**：
  - filesystem：文件系统访问
  - github：GitHub API 集成
  - postgres：数据库查询
  - playwright：浏览器自动化
  - context7：上下文增强
- **MCP 权限控制**：在 agents 配置中控制哪些 Agent 可以使用哪些 MCP 工具
- **常见误区**：配置了太多 MCP 服务器导致启动慢——只启用项目需要的 MCP

#### 5.3 MCP 实战：GitHub 集成
- **配置 GitHub MCP**：安装 @modelcontextprotocol/server-github
- **使用场景**：
  - 创建和管理 Issue
  - 提交 Pull Request
  - 搜索代码和仓库
  - 查看 CI/CD 状态
- **GitHub Token 配置**：`GITHUB_TOKEN` 环境变量
- **安全注意事项**：Token 权限最小化原则

#### 5.4 MCP 实战：数据库与浏览器自动化
- **PostgreSQL MCP**：让 AI 直接查询数据库、分析数据
- **Playwright MCP**：让 AI 操作浏览器、截图、填写表单
- **自定义 MCP 服务器**：用 TypeScript/Python 开发自己的 MCP 工具
- **常见误区**：MCP 工具没有权限控制——实际上可以在 agents 配置中精细控制

### 第6章：实战开发场景

#### 6.1 代码生成与重构
- **从零创建项目**：用 Plan 模式规划 → Build 模式执行
- **功能添加**：描述需求 → AI 分析项目 → 生成代码 → 审查 diff
- **代码重构**：指定重构目标 → AI 分析依赖 → 批量修改 → 验证
- **/undo 和 /redo**：撤销不满意的修改、重做之前的修改
- **常见误区**：AI 生成的代码不审查直接用——应该每次都审查 diff

#### 6.2 调试与问题排查
- **错误诊断**：粘贴错误信息 → AI 分析原因 → 建议修复
- **日志分析**：`$ cat logs/error.log` 引入日志 → AI 分析
- **性能优化**：描述性能问题 → AI 分析瓶颈 → 建议优化方案
- **测试驱动调试**：让 AI 先写测试 → 运行测试 → 根据失败信息修复

#### 6.3 Git 工作流集成
- **代码审查**：`$ git diff` → AI 审查变更
- **提交信息生成**：AI 根据 diff 生成规范的 commit message
- **分支管理**：创建分支、合并冲突解决
- **PR 描述生成**：AI 根据 commit 历史生成 PR 描述
- **常见误区**：让 AI 直接 git push——应该由人工确认后再 push

### 第7章：进阶技巧

#### 7.1 AGENTS.md 与项目定制
- **AGENTS.md 的作用**：项目的"AI 说明书"——告诉 AI 项目的架构、规范、约定
- **/init 命令**：自动分析项目并生成 AGENTS.md
- **手动定制 AGENTS.md**：添加项目特定的编码规范、架构说明、常用命令
- **AGENTS.md 最佳实践**：结构清晰、重点突出、定期更新
- **常见误区**：AGENTS.md 写得太长——AI 的注意力有限，应该只写最关键的信息

#### 7.2 自定义命令与工作流
- **自定义斜杠命令**：在配置文件中定义常用命令模板
- **命名参数**：`/fix-issue $issue_number` 等参数化命令
- **工作流自动化**：结合 CLI 非交互模式 `opencode -p "prompt"` 实现自动化
- **CI/CD 集成**：在 pipeline 中使用 OpenCode 做代码审查
- **常见误区**：自定义命令太复杂——命令应该简单明确，复杂逻辑让 AI 自己推理

#### 7.3 多会话与协作
- **多会话并行**：同时处理多个任务（如一个调试、一个重构）
- **会话分享**：`/share` 生成分享链接，方便团队协作
- **远程开发**：`/connect` 连接远程服务器
- **团队协作最佳实践**：AGENTS.md 共享、MCP 配置共享、会话模板

### 第8章：生产化与团队部署

#### 8.1 安全最佳实践
- **API Key 管理**：环境变量、不提交到 Git、定期轮换
- **工具权限控制**：最小权限原则、确认机制
- **MCP 安全**：只启用信任的 MCP 服务器、MCP 完整性校验
- **代码隐私**：敏感项目使用本地模型、配置数据不外传
- **常见误区**：认为开源就安全——开源意味着可审计，但配置不当仍然有风险

#### 8.2 团队配置标准化
- **项目级配置**：`.opencode/opencode.json` 提交到 Git
- **AGENTS.md 模板**：团队共享的项目规范
- **MCP 配置共享**：团队统一的 MCP 工具集
- **模型策略统一**：团队推荐的模型选择和成本控制策略
- **新人 Onboarding**：一键安装 + 项目配置 + AGENTS.md

#### 8.3 OpenCode 的局限性与替代方案
- **已知局限**：
  - Windows 支持不如 macOS/Linux 完善
  - 无 IDE 可视化界面（纯终端）
  - 依赖外部 LLM 的推理能力
  - MCP 生态仍在发展中
  - 长对话的上下文管理仍有挑战
- **何时用 OpenCode vs Cursor vs Claude Code**：
  - 偏好终端 → OpenCode
  - 偏好 IDE → Cursor
  - 只用 Claude → Claude Code
  - 需要开源和自定义 → OpenCode
- **OpenCode 的未来**：MCP 生态成熟、更多内置工具、企业级功能

---

## 🎯 学习路径建议

```
终端开发者（1天）:
  第1-3章 → 安装配置、掌握 TUI 交互
  → 能用 OpenCode 进行日常编码辅助

效率工程师（1-2天）:
  第4-5章 → 模型选择、MCP 扩展
  → 能根据场景选模型、配置 MCP 工具

团队负责人（1天）:
  第7-8章 → 项目定制、团队标准化
  → 能为团队建立 OpenCode 工作流规范
```

---

## 📚 与其他 AI 编程工具的对照

| 维度 | OpenCode | Cursor | GitHub Copilot | Claude Code | Aider |
|------|----------|--------|---------------|-------------|-------|
| 开源 | ✅ | ❌ | ❌ | ❌ | ✅ |
| 运行环境 | 终端 TUI | IDE | IDE 插件 | 终端 CLI | 终端 CLI |
| 模型选择 | 75+ 提供商 | 有限 | 仅 OpenAI | 仅 Claude | 多模型 |
| MCP 支持 | ✅ | 有限 | ❌ | ❌ | ❌ |
| LSP 集成 | ✅ | ✅ | ✅ | ❌ | ❌ |
| 本地模型 | ✅ Ollama | ❌ | ❌ | ❌ | ✅ |
| 会话管理 | ✅ 多会话 | ❌ | ❌ | ❌ | ❌ |
| 价格 | 免费（模型费自付） | $20/月 | $10/月 | 按量付费 | 免费 |
| 适合人群 | 终端开发者 | IDE 开发者 | 通用开发者 | Claude 用户 | Git 重度用户 |

---

*本大纲遵循以下写作原则：每节以白板式通俗介绍开场 → 深入原理与面试级知识点 → 配合代码示例（"比如下面的程序..."）→ 覆盖常见用法与误区 → 结尾总结要点衔接下一节。*
