# 09-3 第三方工具生态

## Ollama 工具全景图

除了 Open WebUI 和官方 WebUI，Ollama 还有一个庞大且快速增长的第三方工具生态系统。这些工具覆盖了从桌面客户端到 IDE 插件、从移动端到服务端集成的各种场景。

```
┌─────────────────────────────────────────────────────────────┐
│              Ollama 第三方工具生态                              │
│                                                             │
│  ┌─────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐         │
│  │ 桌面端   │ │ 终端 TUI  │ │ IDE      │ │ 移动端   │         │
│  ├─────────┤ ├──────────┤ ├──────────┤ ├──────────┤         │
│  │OpenWebUI│ │LibreChat │ │Continue  │ │TypingMind│         │
│  │LobeChat │ │Msty     │ │Jan       │ │Chatbox  │         │
│  │GPT4All  │ │Harbor   │ │Cursor   │ │          │         │
│  └─────────┘ └──────────┘ └──────────┘ └──────────┘         │
│                                                             │
│  ┌─────────┐ ┌──────────┐ ┌──────────┐                   │
│  │ 服务端   │ │ 开发框架  │ │ 可视化    │                   │
│  ├─────────┤ ├──────────┤ ├──────────┤                   │
│  │Dify     │ │Flowise  │ │LocalAI  │                   │
│  │Zai      │ │LangChain│ │          │                   │
│  └─────────┘ └──────────┘ └──────────┘                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 桌面端 GUI 客户端

### LibreChat：多后端统一聊天

**特点**：开源、支持同时连接多个后端（Ollama / OpenAI / Azure / 自部署）

```bash
# Docker 部署
docker run -d --name librechat \
  -p 3000:3080 \
  -v librechat_data:/data \
  ghcr.io/danielperera84/librechat:latest

# 访问 http://localhost:3000
# Settings → Endpoints → 添加 Ollama:
# URL: http://host.docker.internal:11434/v1
# API Key: ollama (随意填)
```

**适用**：需要同时使用云端和本地模型、团队内部统一入口

### Lobe Chat：现代化 + 插件体系

**特点**：界面精美、插件丰富（搜索/图像生成/长文档/TTS）、国内开发活跃

```bash
# 安装
npm install -g @lobechat/cli
lobechat

# 或 Docker
docker run -d --name lobechat -p 3210:3210 ghcr.io/lobehub/lobe-chat
```

### Chatbox AI：跨平台桌面客户端

**特点**：原生桌面应用（Windows/macOS/Linux）、轻量快速、支持多种 AI 后端

```bash
# 下载地址: https://github.com/Bin-Huang/chatbox
# 各平台有对应的安装包
```

## 终端 TUI 工具

### Msty：极简终端 UI

```bash
# 安装
cargo install msty

# 使用
msty                    # 启动交互界面
msty --model qwen2.5:7b   # 指定模型
```

**特点**：极简设计、零依赖、适合 SSH 远程服务器上的轻量交互

### Harbor：TUI 多模型管理

```bash
pip install harbor-cli
harbor                  # 启动 TUI 界面
```

**特点**：可以同时管理多个模型、在 TUI 中切换对话

## IDE 集成（简要介绍，下一节详述）

| IDE | 工具 | 核心功能 |
|-----|------|---------|
| **VS Code** | Continue | 对话、代码解释、补全、自定义命令 |
| **JetBrains** | Continue | 同 VS Code |
| **Vim / Neovim** | ollama.nvim | `:Ollama` 命令直接在编辑器中对话 |
| **Emacs** | ellama-mode | M-x ellama-start |

## 服务端平台

### Dify：低代码 LLM 应用平台

**特点**：可视化编排工作流、内置 RAG、用户管理、API 发布——**企业级首选**

```bash
docker compose up -d
# 访问 http://localhost/install
# 初始化后添加 Ollama 作为 LLM Provider
```

### Flowise：可视化 LangChain 编排

**特点**：拖拽式节点连接，构建 AI Pipeline，无需写代码

## 快速选型指南

```
你的需求是什么？

├─ 个人开发者，想要一个好用的聊天界面
│  ├── 追求美观、功能全 → Lobe Chat / Open WebUI
│  ├── 跨平台桌面应用 → Chatbox AI
│  └─ 极简主义者 → Msty (TUI)
│
├─ 团队共享，需要统一入口和权限管理
│  ├── 有技术团队可维护 → Dify
│  ├── 想要快速搭建 → LibreChat + Nginx
│  └─ 预算有限 → Ollama 内置 WebUI
│
├─ 在 IDE 中开发，不想切换窗口
│  ├── VS Code → Continue 插件
│  ├── JetBrains → Continue 插件
│  └── Vim/Neovim → ollama.nvim
│
└─ 想做产品给外部用户
   └── Open WebUI (唯一成熟选择)
```

## 本章小结

这一节快速浏览了 Ollama 的第三方工具生态：

1. **桌面 GUI 三巨头**：Open WebUI（功能最全）、Lobe Chat（最漂亮）、Chatbox AI（跨平台）
2. **终端 TUI**：Msty 和 Harbor 为 SSH 环境提供轻量交互能力
3. **IDE 集成**：VS Code/JetBrains/Vim 都有成熟的 Ollama 插件
4. **服务端平台**：Dify 和 Flowise 让非技术人员也能构建 AI 应用
5. **选型核心原则**：个人用看偏好，团队用看需求，产品用 Open WebUI

下一节我们将深入 IDE 集成的具体操作。
