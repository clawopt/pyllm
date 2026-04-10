# 2.3 模型提供商配置

> **OpenCode 支持 75+ 个 LLM 提供商——选哪个？怎么配？这一节帮你把模型这件事搞清楚。**

---

## 这一节在讲什么？

OpenCode 最大的优势之一就是"模型自由"——你可以用 Claude、GPT-4、Gemini、DeepSeek、Ollama 本地模型等 75+ 个提供商。但"自由"也意味着"选择困难"——每个提供商怎么配置？哪个模型适合什么场景？怎么在它们之间切换？这一节我们把主流的模型提供商逐一讲解，帮你建立清晰的模型选择策略。

---

## Anthropic（Claude）

Anthropic 的 Claude 系列是 OpenCode 推荐的默认模型，也是目前 AI 编程领域表现最好的模型之一。

**配置方式一：API Key**

```bash
export ANTHROPIC_API_KEY="sk-ant-api03-..."
```

或者在配置文件中：

```json
{
  "providers": {
    "anthropic": {
      "apiKey": "$ANTHROPIC_API_KEY"
    }
  }
}
```

**配置方式二：Claude Pro/Max 订阅登录**

如果你有 Claude Pro 或 Claude Max 订阅，可以直接用订阅登录，不需要单独买 API Key：

```bash
opencode auth login
# 选择 Anthropic → 按提示完成 OAuth 登录
```

**推荐模型**：

| 模型 | 特点 | 适用场景 | 价格 |
|------|------|---------|------|
| claude-sonnet-4-20250514 | 性能与速度的最佳平衡 | 日常编码、代码生成、调试 | $3/M input, $15/M output |
| claude-haiku-4-20250414 | 速度极快，成本低 | 简单问答、快速修改 | $0.25/M input, $1.25/M output |
| claude-opus-4-20250514 | 最强推理能力 | 复杂架构设计、难题攻坚 | $15/M input, $75/M output |

**在 OpenCode 中指定模型**：

```json
{
  "agents": {
    "coder": {
      "model": "anthropic/claude-sonnet-4-20250514"
    }
  }
}
```

或者在 TUI 中用 `/models` 命令切换。

---

## OpenAI（GPT 系列）

OpenAI 的 GPT 系列是最广泛使用的 LLM，OpenCode 原生支持。

**配置方式**：

```bash
export OPENAI_API_KEY="sk-..."
```

```json
{
  "providers": {
    "openai": {
      "apiKey": "$OPENAI_API_KEY"
    }
  }
}
```

**ChatGPT Plus/Pro 订阅登录**：

```bash
opencode auth login
# 选择 OpenAI → 按提示完成 OAuth 登录
```

**推荐模型**：

| 模型 | 特点 | 适用场景 |
|------|------|---------|
| gpt-4o | 综合能力强，多模态 | 日常编码、代码审查 |
| gpt-4o-mini | 速度快，成本低 | 简单任务、快速问答 |
| o3 | 推理能力强 | 复杂逻辑、算法问题 |

---

## Google Gemini

Google 的 Gemini 系列有一个杀手锏——超长上下文窗口。Gemini 1.5 Pro 支持 100 万 token 的上下文，这意味着你可以把整个项目塞进去让 AI 分析。

**配置方式**：

```bash
export GEMINI_API_KEY="AIza..."
```

```json
{
  "providers": {
    "google": {
      "apiKey": "$GEMINI_API_KEY"
    }
  }
}
```

**VertexAI 配置**（企业级）：

```bash
export VERTEXAI_PROJECT="your-project-id"
export VERTEXAI_LOCATION="us-central1"
```

**推荐模型**：

| 模型 | 特点 | 适用场景 |
|------|------|---------|
| gemini-2.5-pro | 超长上下文（100万 token） | 大型项目分析、长文档处理 |
| gemini-2.5-flash | 速度快 | 日常编码 |

**Gemini 的独特价值**：当你需要 AI 理解整个项目（而不是单个文件）时，Gemini 的 100 万 token 上下文窗口是其他模型无法比拟的。比如，你可以让 AI "审查整个项目的 API 设计是否一致"——这需要 AI 同时看到所有路由文件，Claude 和 GPT-4 的上下文窗口装不下。

---

## DeepSeek

DeepSeek 是中国团队开发的 LLM，在代码生成方面表现突出，而且价格非常便宜。

**配置方式**：

```bash
export DEEPSEEK_API_KEY="sk-..."
```

```json
{
  "providers": {
    "deepseek": {
      "apiKey": "$DEEPSEEK_API_KEY"
    }
  }
}
```

**推荐模型**：

| 模型 | 特点 | 适用场景 |
|------|------|---------|
| deepseek-chat | 通用对话，代码能力强 | 日常编码、代码生成 |
| deepseek-reasoner | 推理能力强 | 复杂逻辑、算法问题 |

DeepSeek 的优势在于性价比——同等代码生成质量下，价格只有 Claude 的 1/10 左右。如果你的预算有限，DeepSeek 是很好的选择。

---

## GitHub Copilot

如果你已经有 GitHub Copilot 订阅（$10/月或 $19/月），可以直接在 OpenCode 里用 Copilot 的模型，不需要额外付费。

**配置方式**：

```bash
opencode auth login
# 选择 GitHub Copilot
# OpenCode 会给你一个设备码，在 github.com/login/device 输入完成授权
```

授权完成后，你可以在 TUI 里用 `/models` 命令选择 Copilot 提供的模型。

**Copilot 的独特价值**：如果你已经订阅了 Copilot，这是"零额外成本"使用 OpenCode 的方式——不需要单独买 API Key，不需要按量付费。

---

## Groq

Groq 提供超高速的 LLM 推理服务——它的 LPU（Language Processing Unit）硬件让推理速度比 GPU 快一个数量级。

**配置方式**：

```bash
export GROQ_API_KEY="gsk_..."
```

```json
{
  "providers": {
    "groq": {
      "apiKey": "$GROQ_API_KEY"
    }
  }
}
```

**推荐模型**：

| 模型 | 特点 | 适用场景 |
|------|------|---------|
| llama-3.3-70b-versatile | 速度快，质量不错 | 快速问答、简单修改 |
| mixtral-8x7b-32768 | 长上下文 | 中等复杂度任务 |

Groq 的优势在于速度——响应几乎是实时的，适合需要快速迭代的场景。但模型质量不如 Claude 和 GPT-4，不适合复杂任务。

---

## OpenRouter

OpenRouter 是一个统一的 LLM 代理服务——你只需要一个 API Key，就能访问几乎所有主流模型。

**配置方式**：

```bash
export OPENROUTER_API_KEY="sk-or-..."
```

```json
{
  "providers": {
    "openrouter": {
      "apiKey": "$OPENROUTER_API_KEY"
    }
  }
}
```

使用时指定模型：

```json
{
  "agents": {
    "coder": {
      "model": "openrouter/anthropic/claude-sonnet-4-20250514"
    }
  }
}
```

OpenRouter 的优势在于"一站式"——不需要为每个提供商单独注册和充值。但价格通常比直接用提供商的 API 稍贵（有中间商差价）。

---

## Ollama 本地模型

Ollama 让你在本地运行 LLM，不需要网络，不需要 API Key，数据完全不出本机。

**安装 Ollama**：

```bash
curl -fsSL https://ollama.ai/install | bash
```

**下载模型**：

```bash
# 通用模型（2GB）
ollama pull llama3.2:3b

# 代码生成模型（4GB）
ollama pull deepseek-coder:6.7b

# 代码补全模型（4GB）
ollama pull codestral:7b
```

**配置 OpenCode 使用 Ollama**：

```json
{
  "providers": {
    "ollama": {}
  },
  "agents": {
    "coder": {
      "model": "ollama/deepseek-coder:6.7b"
    }
  }
}
```

Ollama 默认运行在 `http://localhost:11434`，OpenCode 会自动连接。如果你的 Ollama 运行在其他端口，可以通过环境变量指定：

```bash
export LOCAL_ENDPOINT="http://localhost:11434"
```

**本地模型的局限**：推理能力不如云端大模型，长上下文处理能力有限，生成速度取决于本地硬件。适合辅助任务（代码补全、简单修改），不适合复杂任务（架构设计、大型重构）。

---

## AWS Bedrock / Azure OpenAI

企业用户通常需要通过 AWS Bedrock 或 Azure OpenAI 来使用 LLM——这两个平台提供了企业级的安全合规能力。

**AWS Bedrock 配置**：

```bash
export AWS_ACCESS_KEY_ID="AKIA..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_REGION="us-east-1"
```

```json
{
  "agents": {
    "coder": {
      "model": "bedrock/anthropic.claude-sonnet-4-20250514-v1:0"
    }
  }
}
```

**Azure OpenAI 配置**：

```bash
export AZURE_OPENAI_ENDPOINT="https://xxx.openai.azure.com/"
export AZURE_OPENAI_API_KEY="..."
export AZURE_OPENAI_API_VERSION="2024-02-15-preview"
```

```json
{
  "agents": {
    "coder": {
      "model": "azure/gpt-4o"
    }
  }
}
```

---

## 模型选择策略

面对这么多模型，怎么选？这里有一个简单的决策树：

```
你的需求                          → 推荐模型
─────────────────────────────────────────────────
日常编码，追求质量                 → Claude Sonnet 4
日常编码，追求速度                 → Claude Haiku 4 / GPT-4o-mini
复杂架构设计                       → Claude Opus 4 / o3
代码生成，预算有限                 → DeepSeek Chat
需要理解整个项目                   → Gemini 2.5 Pro（100万上下文）
快速问答，追求响应速度             → Groq Llama 3.3
隐私敏感，数据不能出本机           → Ollama 本地模型
已有 Copilot 订阅，不想额外付费    → GitHub Copilot
企业合规要求                       → AWS Bedrock / Azure OpenAI
```

---

## 常见误区

**误区一：以为 OpenCode 只能用 Claude**

这是最常见的误解。OpenCode 支持 75+ 个提供商，Claude 只是默认推荐。你可以用 GPT-4、Gemini、DeepSeek、Ollama 本地模型等任何支持的模型。甚至可以在同一个会话里切换模型——简单问题用 Haiku，复杂问题切 Sonnet。

**误区二：所有任务都用最贵的模型**

Claude Opus 4 的价格是 Sonnet 4 的 5 倍，但不是所有任务都需要 Opus 的推理能力。简单问答用 Haiku 就够了，日常编码用 Sonnet 性价比最高，只有复杂的架构设计和难题攻坚才需要 Opus。合理分配模型能大幅降低成本。

**误区三：本地模型能替代云端大模型**

目前还不能。Ollama 的 7B 参数模型在代码生成质量上跟 Claude Sonnet 有明显差距。本地模型适合辅助任务——代码补全、简单修改、快速问答——但复杂任务还是需要云端大模型。如果你非常在意隐私，可以日常用本地模型，复杂任务再切到云端。

**误区四：Copilot 订阅不能在 OpenCode 里用**

可以。OpenCode 支持 GitHub Copilot 的 OAuth 登录——你只需要运行 `opencode auth login`，选择 GitHub Copilot，完成设备授权，就能在 OpenCode 里使用 Copilot 的模型。不需要额外买 API Key。

---

## 小结

这一节我们逐一讲解了 OpenCode 支持的主流模型提供商：Anthropic Claude（推荐默认）、OpenAI GPT-4、Google Gemini（超长上下文）、DeepSeek（高性价比）、GitHub Copilot（零额外成本）、Groq（超高速）、Ollama（本地隐私）、AWS Bedrock / Azure OpenAI（企业合规）。模型选择的核心原则是"按需选择"——日常用 Sonnet，简单用 Haiku，复杂用 Opus，隐私用 Ollama。下一章我们进入 OpenCode 的 TUI 交互，学习怎么高效地跟 AI 对话。
