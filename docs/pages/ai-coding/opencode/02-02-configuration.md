# 2.2 配置文件详解

> **OpenCode 的配置文件是你掌控 AI 行为的"遥控器"——模型选哪个、工具开哪些、MCP 连什么，全在这里配置。**

---

## 这一节在讲什么？

安装好 OpenCode 之后，下一步就是配置。OpenCode 的配置文件看起来就是一个 JSON 文件，但里面的每个字段都直接影响 AI 的行为——用哪个模型、最多生成多少 token、能调用哪些工具、连接哪些 MCP 服务器。这一节我们把配置文件的结构、每个字段的作用、分层配置的合并策略都讲清楚，帮你从"能用"升级到"会用"。

---

## 配置文件的位置

OpenCode 的配置文件有三个层级，按优先级从低到高：

| 层级 | 路径 | 作用域 | 优先级 |
|------|------|--------|--------|
| 全局配置 | `~/.config/opencode/opencode.json` | 所有项目 | 最低 |
| 用户配置 | `~/.opencode.json` | 所有项目 | 中 |
| 项目配置 | `.opencode/opencode.json`（项目根目录） | 当前项目 | 最高 |

项目级配置会覆盖全局配置——这意味着你可以在全局配置里设置默认模型，然后在特定项目里覆盖为另一个模型。

比如，你的全局配置用 Claude Sonnet，但某个项目需要用 DeepSeek Coder（因为代码主要是 Python），你只需要在项目级配置里覆盖 model 字段即可。

---

## 配置文件完整结构

下面是一个完整的配置文件示例，每个字段都有注释说明：

```json
{
  "data": {
    "directory": ".opencode"
  },
  "providers": {
    "anthropic": {
      "apiKey": "$ANTHROPIC_API_KEY",
      "disabled": false
    },
    "openai": {
      "apiKey": "$OPENAI_API_KEY",
      "disabled": false
    },
    "copilot": {
      "disabled": false
    },
    "groq": {
      "apiKey": "$GROQ_API_KEY",
      "disabled": false
    },
    "openrouter": {
      "apiKey": "$OPENROUTER_API_KEY",
      "disabled": false
    }
  },
  "agents": {
    "coder": {
      "model": "anthropic/claude-sonnet-4-20250514",
      "maxTokens": 16384
    },
    "task": {
      "model": "anthropic/claude-sonnet-4-20250514",
      "maxTokens": 16384
    }
  },
  "mcp": {
    "github": {
      "type": "local",
      "command": ["npx", "-y", "@modelcontextprotocol/server-github"],
      "enabled": true,
      "env": {
        "GITHUB_TOKEN": "$GITHUB_TOKEN"
      }
    }
  },
  "theme": "tokyo-night",
  "autoCompact": true,
  "shell": {
    "path": "/bin/zsh",
    "args": ["-l"]
  }
}
```

让我们逐个字段来理解：

---

### data：数据存储

```json
{
  "data": {
    "directory": ".opencode"
  }
}
```

`data.directory` 指定 OpenCode 存储会话数据、消息历史的目录。默认是项目根目录下的 `.opencode/`。这个目录里包含 SQLite 数据库文件，存储了你的对话记录和会话信息。

**注意**：`.opencode/` 目录应该加入 `.gitignore`，不要提交到 Git 仓库——它包含的是本地会话数据，不是项目配置。

---

### providers：模型提供商

```json
{
  "providers": {
    "anthropic": {
      "apiKey": "$ANTHROPIC_API_KEY",
      "disabled": false
    },
    "openai": {
      "apiKey": "$OPENAI_API_KEY",
      "disabled": false
    }
  }
}
```

`providers` 是你配置 LLM 提供商的地方。每个提供商有两个关键字段：

- **apiKey**：API 密钥。强烈建议用 `$ENV_VAR` 的方式引用环境变量，而不是硬编码密钥值。OpenCode 会自动从环境变量中读取。
- **disabled**：是否禁用该提供商。如果你有某个提供商的 Key 但暂时不想用，可以设为 true。

支持的提供商包括：`anthropic`、`openai`、`google`、`copilot`、`groq`、`openrouter`、`deepseek`、`bedrock`、`azure` 等。

---

### agents：Agent 配置

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

`agents` 配置 OpenCode 的 Agent 行为。目前有两个内置 Agent：

- **coder**：主要的编码 Agent，负责理解需求、修改代码、执行命令
- **task**：任务 Agent，用于子任务和辅助操作

每个 Agent 的配置项：

- **model**：使用的模型，格式是 `provider/model-name`
- **maxTokens**：单次生成的最大 token 数。默认 5000，建议设为 16384 或更高，让 AI 有足够的空间生成完整的代码修改

模型的格式是 `提供商/模型名`，例如：

```
anthropic/claude-sonnet-4-20250514
anthropic/claude-haiku-4-20250414
openai/gpt-4o
openai/gpt-4o-mini
google/gemini-2.5-pro
deepseek/deepseek-chat
groq/llama-3.3-70b-versatile
ollama/deepseek-coder:6.7b
```

---

### mcp：MCP 服务器配置

```json
{
  "mcp": {
    "github": {
      "type": "local",
      "command": ["npx", "-y", "@modelcontextprotocol/server-github"],
      "enabled": true,
      "env": {
        "GITHUB_TOKEN": "$GITHUB_TOKEN"
      }
    },
    "supabase": {
      "type": "remote",
      "url": "https://mcp.supabase.com/mcp",
      "enabled": true,
      "headers": {
        "Authorization": "Bearer $SUPABASE_TOKEN"
      }
    }
  }
}
```

MCP 配置是 OpenCode 最强大的扩展点。每个 MCP 服务器有以下字段：

- **type**：`local`（本地进程）或 `remote`（远程 HTTP/SSE）
- **command**：本地 MCP 服务器的启动命令（type 为 local 时）
- **url**：远程 MCP 服务器的 URL（type 为 remote 时）
- **enabled**：是否启用
- **env**：传递给 MCP 服务器的环境变量
- **headers**：远程 MCP 服务器的 HTTP 头（type 为 remote 时）

MCP 的详细配置我们会在第 5 章深入讲解，这里只需要知道配置的位置和基本格式。

---

### theme：主题配置

```json
{
  "theme": "tokyo-night"
}
```

OpenCode 内置了多个主题，你可以根据喜好选择：

- `tokyo-night`：深色主题，护眼
- `catppuccin`：柔和的暖色调
- `dracula`：经典的 Dracula 配色
- `nord`：北欧风格冷色调
- `gruvbox`：复古暖色调

---

### autoCompact：自动压缩

```json
{
  "autoCompact": true
}
```

`autoCompact` 控制是否在对话接近上下文窗口上限时自动压缩。默认是 true，建议保持开启——否则长对话会触发模型的上下文限制，导致 AI 无法继续对话。

---

### shell：Shell 配置

```json
{
  "shell": {
    "path": "/bin/zsh",
    "args": ["-l"]
  }
}
```

`shell` 配置 OpenCode 执行命令时使用的 Shell。默认使用 `SHELL` 环境变量指定的 Shell，你可以在这里覆盖。

`args` 中的 `-l` 表示登录 Shell——这会加载 `.zshrc` / `.bashrc` 中的环境变量和别名，确保 AI 执行的命令跟你手动执行的一致。

---

## 环境变量配置

除了配置文件，OpenCode 还支持通过环境变量配置 API Key。这是最安全的方式——Key 不会出现在配置文件里，不会被提交到 Git：

```bash
# Anthropic Claude
export ANTHROPIC_API_KEY="sk-ant-..."

# OpenAI
export OPENAI_API_KEY="sk-..."

# Google Gemini
export GEMINI_API_KEY="AIza..."

# GitHub Copilot（不需要 Key，用 OAuth 登录）
# 运行 opencode auth login 选择 GitHub Copilot

# DeepSeek
export DEEPSEEK_API_KEY="sk-..."

# Groq
export GROQ_API_KEY="gsk_..."

# AWS Bedrock
export AWS_ACCESS_KEY_ID="AKIA..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_REGION="us-east-1"

# Azure OpenAI
export AZURE_OPENAI_ENDPOINT="https://xxx.openai.azure.com/"
export AZURE_OPENAI_API_KEY="..."

# 本地模型（Ollama 等）
export LOCAL_ENDPOINT="http://localhost:11434"
```

建议把这些环境变量写在 `~/.zshrc` 或 `~/.bashrc` 里：

```bash
# ~/.zshrc
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
export GEMINI_API_KEY="AIza..."
```

---

## 分层配置的合并策略

当全局配置和项目配置有冲突时，OpenCode 的合并策略是"项目级覆盖全局级"：

```
全局配置（~/.config/opencode/opencode.json）
  → 设置默认模型为 claude-sonnet-4-20250514
  → 设置主题为 tokyo-night

项目配置（.opencode/opencode.json）
  → 覆盖模型为 deepseek/deepseek-chat
  → 添加项目专属的 MCP 服务器

最终生效的配置：
  → 模型：deepseek/deepseek-chat（项目级覆盖）
  → 主题：tokyo-night（全局级，项目级未覆盖）
  → MCP：全局的 + 项目专属的（合并）
```

这个分层机制非常适合团队协作——全局配置设置个人偏好（主题、默认模型），项目配置设置团队规范（项目专属模型、MCP 服务器、Agent 配置）。项目配置可以提交到 Git，确保团队成员使用一致的配置。

---

## 常见误区

**误区一：把 API Key 硬编码在配置文件里并提交到 Git**

这是最危险的错误。API Key 一旦提交到 Git，即使你后来删除了，它仍然在 Git 历史里。任何有仓库访问权限的人都能看到你的 Key，可能导致 API 费用被恶意消耗。

正确做法：用 `$ENV_VAR` 引用环境变量，或者用 `opencode auth login` 交互式配置（Key 存储在 `~/.local/share/opencode/auth.json`，不在项目目录里）。

```json
// ❌ 错误：硬编码 API Key
{
  "providers": {
    "anthropic": {
      "apiKey": "sk-ant-api03-xxxxx"
    }
  }
}

// ✅ 正确：引用环境变量
{
  "providers": {
    "anthropic": {
      "apiKey": "$ANTHROPIC_API_KEY"
    }
  }
}
```

**误区二：项目级配置文件提交了 .opencode/ 目录**

`.opencode/` 目录里既有配置文件（应该提交），也有会话数据（不应该提交）。正确的做法是只提交配置文件，忽略会话数据：

```gitignore
# .gitignore
.opencode/data/
.opencode/auth.json
# 但保留 .opencode/opencode.json
```

或者把项目配置放在 `.opencode.json`（项目根目录），而不是 `.opencode/opencode.json`，这样更清晰。

**误区三：maxTokens 设置太低**

很多用户保持默认的 maxTokens=5000，这对于简单的问答够用，但如果 AI 需要生成较长的代码修改，5000 token 可能不够——AI 会在代码写到一半时被截断。建议设为 16384 或更高。

**误区四：配置了太多 MCP 服务器**

每个 MCP 服务器启动时都需要初始化，配置太多会导致 OpenCode 启动变慢。只启用项目需要的 MCP 服务器——前端项目用 shadcn MCP，后端项目用数据库 MCP，不需要全部都配。

---

## 小结

这一节我们详细讲解了 OpenCode 的配置文件结构：`providers` 配置模型提供商，`agents` 配置 Agent 行为，`mcp` 配置 MCP 服务器，`theme` 设置主题，`autoCompact` 控制自动压缩，`shell` 配置命令执行环境。配置文件有三个层级（全局、用户、项目），项目级覆盖全局级。最重要的是——**永远不要把 API Key 硬编码在配置文件里**，用环境变量引用。下一节我们深入模型提供商的配置，帮你选到最适合的模型。
