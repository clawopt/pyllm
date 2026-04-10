# 1.3 5 分钟快速体验

> **别看文档了，先动手——5 分钟从安装到第一次 AI 对话，让你亲身感受 OpenCode 的 Agent 能力。**

---

## 这一节在讲什么？

前面两节讲了 OpenCode 是什么、它的工作原理是什么，但"纸上得来终觉浅"——不亲手试一试，你很难理解"终端里的 AI Agent"到底意味着什么。这一节带你从零开始，5 分钟内完成安装、启动、初始化、第一次对话。不需要深入配置，不需要理解原理，先跑起来再说。

---

## 第一步：安装 OpenCode

OpenCode 提供了多种安装方式，推荐使用官方安装脚本——一条命令搞定：

```bash
# macOS / Linux：官方安装脚本（推荐）
curl -fsSL https://opencode.ai/install | bash
```

如果你用的是 macOS 并且装了 Homebrew，也可以用 brew 安装：

```bash
# Homebrew 安装（macOS / Linux）
brew install sst/tap/opencode
```

其他安装方式：

```bash
# NPM 安装
npm install -g opencode-ai

# Go Install（需要 Go 1.22+）
go install github.com/sst/opencode@latest

# Arch Linux（AUR）
paru -S opencode-bin
```

安装完成后，验证一下：

```bash
opencode --version
# 输出类似：opencode version 0.4.x
```

如果你看到版本号输出，说明安装成功了。

---

## 第二步：配置模型提供商

OpenCode 需要至少一个 LLM 提供商才能工作。最简单的方式是用 `opencode auth login` 命令交互式配置：

```bash
opencode auth login
```

你会看到一个交互式菜单，列出所有支持的提供商：

```
┌  Add credential
│◆  Select provider
│  ● Anthropic (recommended)
│  ○ OpenAI
│  ○ Google
│  ○ GitHub Copilot
│  ○ Amazon Bedrock
│  ○ Azure
│  ○ DeepSeek
│  ○ Groq
│  ○ OpenRouter
│  ...
└
```

选择一个你已有 API Key 的提供商，输入 Key 即可。如果你有 GitHub Copilot 订阅，选 GitHub Copilot 可以直接用，不需要额外付费。

或者，你也可以通过环境变量配置：

```bash
# Anthropic Claude
export ANTHROPIC_API_KEY="sk-ant-..."

# OpenAI
export OPENAI_API_KEY="sk-..."

# Google Gemini
export GEMINI_API_KEY="AIza..."

# DeepSeek
export DEEPSEEK_API_KEY="sk-..."
```

建议把环境变量写在 `~/.zshrc` 或 `~/.bashrc` 里，这样每次打开终端都能自动加载。

---

## 第三步：启动 OpenCode

找一个你正在开发的项目（或者随便一个代码目录），进入项目目录后启动 OpenCode：

```bash
cd /path/to/your/project
opencode
```

你会看到 OpenCode 的 TUI 界面——上半部分是对话区，下半部分是输入区，右下角显示当前模式（Build）。

```
┌─────────────────────────────────────────────────────────┐
│  OpenCode                                    Build      │
│─────────────────────────────────────────────────────────│
│                                                         │
│  Welcome to OpenCode! How can I help you today?        │
│                                                         │
│                                                         │
│                                                         │
│─────────────────────────────────────────────────────────│
│  > _                                                    │
└─────────────────────────────────────────────────────────┘
```

---

## 第四步：初始化项目

第一次在项目里使用 OpenCode，强烈建议先运行 `/init` 命令——它会分析你的项目结构，生成一个 AGENTS.md 文件，作为 AI 的"项目说明书"：

```
/init
```

OpenCode 会扫描项目目录，识别框架、语言、依赖、目录结构，然后生成类似这样的 AGENTS.md：

```markdown
# Project: my-web-app

## Overview
This is a Next.js application using TypeScript and Tailwind CSS.

## Structure
- `src/app/` - Next.js App Router pages
- `src/components/` - React components
- `src/lib/` - Utility functions
- `prisma/` - Database schema and migrations

## Commands
- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run test` - Run tests
- `npm run lint` - Run linter

## Conventions
- Use TypeScript for all new files
- Follow existing component patterns in src/components/
- Use Tailwind CSS for styling
```

这个文件非常重要——它让 AI 知道你的项目是什么、怎么构建、有什么规范。没有它，AI 只能靠猜测来理解你的项目，回答质量会大打折扣。

---

## 第五步：第一次对话

现在你可以开始跟 OpenCode 对话了。试试这几个例子，感受一下 Agent 的能力：

**问项目问题**：

```
这个项目的认证是怎么实现的？
```

AI 会读取你的项目文件，找到认证相关的代码，然后给你一个基于项目实际情况的回答——而不是泛泛的"认证一般有 JWT、Session 等方式"。

**请求代码修改**：

```
给 /api/users 接口添加分页功能，每页 20 条
```

AI 会读取路由文件，理解现有的接口结构，然后直接修改代码。你会在文件变更区看到 AI 修改了哪些文件、改了什么内容。

**执行命令**：

```
运行测试，看看有没有失败的
```

AI 会调用 RUN 工具执行测试命令，分析输出结果，如果有失败的测试，它会尝试修复。

**切换到 Plan 模式**：

按 Tab 键切换到 Plan 模式，然后输入：

```
我想给这个项目添加一个用户通知系统，帮我规划一下实现方案
```

AI 会分析项目结构，给出一个实现方案，但不会修改任何文件。你审查方案后，再按 Tab 切回 Build 模式，让它执行。

---

## 几个实用技巧

快速体验阶段，记住这几个技巧就够了：

**1. 用 @ 引用文件**：当你想让 AI 关注特定文件时，用 `@filename` 引用它：

```
看一下 @src/auth/login.ts 这个文件的逻辑有没有问题
```

**2. 用 $ 执行命令**：当你想让 AI 看到某个命令的输出时，用 `$ command`：

```
$ npm run build 的输出有什么错误？帮我修复
```

**3. 用 /undo 撤销修改**：如果 AI 的修改不是你想要的，用 `/undo` 撤销：

```
/undo
```

**4. 用 /compact 压缩对话**：对话太长时，AI 的回答质量会下降，用 `/compact` 压缩：

```
/compact
```

---

## 常见误区

**误区一：首次使用不运行 /init**

这是最常见的错误。没有 AGENTS.md，AI 对你的项目一无所知，回答会非常泛泛——就像你问一个陌生人"帮我改一下代码"一样，他不知道你的项目是什么、用什么框架、有什么规范。`/init` 只需要运行一次，但它的价值贯穿整个项目开发周期。

**误区二：一上来就问很复杂的问题**

第一次对话，建议从简单的问题开始——"这个项目用了什么框架？"、"这个文件是做什么的？"——让 AI 先建立对项目的理解，再逐步深入。一上来就问"帮我重构整个认证系统"，AI 可能会因为上下文不足而给出不准确的方案。

**误区三：忘记切换模式**

很多用户一直用 Build 模式，从来不切 Plan。对于简单任务这没问题，但对于复杂任务，直接 Build 可能导致 AI 做出不理想的修改，然后你需要反复 /undo。养成习惯：复杂任务先 Plan，简单任务直接 Build。

**误区四：API Key 配置在项目目录的配置文件里并提交到 Git**

这会泄露你的 API Key！API Key 应该通过环境变量配置，或者写在全局配置文件 `~/.config/opencode/opencode.json` 里，而不是项目级的 `.opencode.json` 里。如果你一定要在项目级配置文件里写 Key，确保 `.opencode/` 目录在 `.gitignore` 里。

---

## 小结

5 分钟快速体验到此结束。你完成了安装、配置模型、启动 OpenCode、初始化项目、第一次对话——这就是 OpenCode 的基本使用流程。当然，这只是冰山一角：模型选择、MCP 扩展、自定义命令、团队协作等高级功能，我们会在后续章节逐步展开。下一章我们从安装与配置开始，把基础打牢。
