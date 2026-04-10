# 8.1 安全最佳实践

> **开源不等于安全——OpenCode 的安全性取决于你怎么配置。API Key 管理、工具权限控制、MCP 安全，每一项都是你的责任。**

---

## 这一节在讲什么？

OpenCode 是开源的，这意味着你可以审计它的代码、验证它不会做不该做的事。但"可审计"不等于"自动安全"——如果你把 API Key 硬编码在配置文件里并提交到 Git，如果你给 AI 不受限制的命令执行权限，如果你随便添加来源不明的 MCP 服务器，你的项目仍然面临安全风险。这一节我们把 OpenCode 使用中的安全最佳实践系统化地讲清楚，帮你建立安全意识。

---

## API Key 管理

API Key 是 OpenCode 使用中最敏感的信息——它直接关联你的 API 账户和费用。

### 规则一：永远不要硬编码 API Key

```json
// ❌ 危险：硬编码 API Key
{
  "providers": {
    "anthropic": {
      "apiKey": "sk-ant-api03-xxxxxxxxxxxxxxxxxxxx"
    }
  }
}

// ✅ 安全：引用环境变量
{
  "providers": {
    "anthropic": {
      "apiKey": "$ANTHROPIC_API_KEY"
    }
  }
}
```

硬编码的 API Key 一旦提交到 Git，即使你后来删除了，它仍然在 Git 历史里。任何有仓库访问权限的人都能看到你的 Key。

### 规则二：环境变量写在 shell 配置文件里

```bash
# ~/.zshrc 或 ~/.bashrc
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
export GEMINI_API_KEY="AIza..."
```

确保 shell 配置文件的权限是 600（仅所有者可读写）：

```bash
chmod 600 ~/.zshrc
```

### 规则三：定期轮换 API Key

建议每 90 天轮换一次 API Key。如果怀疑 Key 泄露，立即到提供商的控制台撤销并生成新 Key。

### 规则四：使用 `opencode auth login` 代替手动配置

`opencode auth login` 会把凭证存储在 `~/.local/share/opencode/auth.json`，不在项目目录里，不会被提交到 Git。这是最安全的配置方式。

### 规则五：.gitignore 中排除敏感文件

```gitignore
# .gitignore
.opencode/data/
.opencode/auth.json
.opencode.json
*.env
```

---

## 工具权限控制

OpenCode 的 AI 能调用 READ、WRITE、RUN、SEARCH、DIAGNOSTICS 等工具。这些工具的能力需要根据场景限制。

### 最小权限原则

只给 AI 完成当前任务所需的最小权限：

```
场景                     推荐工具配置
─────────────────────────────────────────────
代码审查（只看不改）     READ ✓  WRITE ✗  RUN ✗  SEARCH ✓  DIAGNOSTICS ✓
代码修改（日常编码）     READ ✓  WRITE ✓  RUN ✓  SEARCH ✓  DIAGNOSTICS ✓
调试（需要运行命令）     READ ✓  WRITE ✓  RUN ✓  SEARCH ✓  DIAGNOSTICS ✓
文档生成（只读分析）     READ ✓  WRITE ✗  RUN ✗  SEARCH ✓  DIAGNOSTICS ✗
```

### 确认机制

对于潜在危险的操作，开启确认机制：

```json
{
  "agents": {
    "coder": {
      "confirmDestructive": true
    }
  }
}
```

这会让 AI 在执行破坏性操作（如删除文件、执行 `rm` 命令）前要求你确认。

### 工作目录限制

OpenCode 的文件操作和命令执行都限制在工作目录内。但你可以进一步限制——比如在 Docker 容器中运行 OpenCode，限制它只能访问容器内的文件。

---

## MCP 安全

MCP 服务器给了 AI 连接外部工具的能力——这既是强大的功能，也是潜在的安全风险。

### 只启用信任的 MCP 服务器

不要随便添加来源不明的 MCP 服务器。每个 MCP 服务器都有执行操作的能力——恶意的 MCP 服务器可能窃取你的数据或执行危险操作。

```
MCP 服务器信任等级：

  ✅ 官方 MCP Server：
  - @modelcontextprotocol/server-github
  - @modelcontextprotocol/server-postgres
  - @modelcontextprotocol/server-filesystem
  → 由 Anthropic 官方维护，代码可审计

  ✅ 知名公司的 MCP Server：
  - @playwright/mcp（Microsoft）
  - @upstash/context7-mcp（Upstash）
  - shadcn mcp（shadcn/ui）
  → 由知名公司维护，有社区监督

  ⚠️ 社区 MCP Server：
  - GitHub 上个人开发者发布的 MCP Server
  → 使用前审查代码，确认没有恶意行为

  ❌ 来源不明的 MCP Server：
  - 没有源码的 MCP Server
  - 没有社区评价的 MCP Server
  → 不要使用
```

### MCP 权限最小化

每个 MCP 服务器应该只授予必要的最小权限：

```
GitHub MCP：
- 只读场景：Token 只授予 Contents:只读 + Issues:只读
- 读写场景：Token 授予 Issues:读写 + Pull requests:读写
- 不要授予 Contents:读写（避免 AI 直接推送代码）

PostgreSQL MCP：
- 使用只读数据库用户
- 不要用超级用户连接
- 不要连接生产数据库
```

### MCP 环境变量安全

MCP 配置中的环境变量引用（如 `$GITHUB_TOKEN`）是安全的——OpenCode 会在运行时从环境中读取，不会把实际值写入配置文件。但如果你在 MCP 的 `env` 字段中硬编码了值，它就会被保存在配置文件中：

```json
// ❌ 危险：硬编码 Token
{
  "mcp": {
    "github": {
      "env": {
        "GITHUB_TOKEN": "ghp_xxxxxxxxxxxxxxxxxxxx"
      }
    }
  }
}

// ✅ 安全：引用环境变量
{
  "mcp": {
    "github": {
      "env": {
        "GITHUB_TOKEN": "$GITHUB_TOKEN"
      }
    }
  }
}
```

---

## 代码隐私

当你使用云端 LLM 时，你的代码会被发送到 LLM 提供商的服务器进行处理。这意味着：

- Anthropic 能看到你发送给 Claude 的代码
- OpenAI 能看到你发送给 GPT-4 的代码
- Google 能看到你发送给 Gemini 的代码

虽然这些提供商都声明不会使用 API 数据训练模型，但如果你处理的是高度敏感的代码（如加密算法、安全协议、商业机密），你应该考虑：

**1. 使用本地模型**

Ollama 本地模型完全在你的电脑上运行，代码不会离开本机：

```json
{
  "agents": {
    "coder": {
      "model": "ollama/deepseek-coder:6.7b"
    }
  }
}
```

**2. 使用企业级部署**

AWS Bedrock 和 Azure OpenAI 提供了企业级的数据隔离和合规保证——你的数据不会用于训练模型，且受到企业级的安全保护。

**3. 避免发送敏感文件**

用 `@` 引用文件时，避免引用包含敏感信息的文件（如 `.env`、密钥文件、配置文件中的密码）。只引用跟任务直接相关的代码文件。

---

## 常见误区

**误区一：认为开源就安全**

开源意味着"可审计"——你可以审查代码确认它不会做不该做的事。但"可审计"不等于"自动安全"——如果你配置不当（硬编码 API Key、不受限的工具权限、恶意的 MCP 服务器），你的项目仍然面临安全风险。安全是配置和使用的责任，不仅仅是代码的责任。

**误区二：AI 不会执行危险命令**

大部分时候 AI 不会主动执行危险命令，但它不是 100% 可靠。AI 可能会：
- 执行 `rm -rf` 删除文件（如果你让它"清理临时文件"）
- 执行 `DROP TABLE` 删除数据库表（如果你让它"清理测试数据"）
- 执行 `sudo` 命令（如果你让它"安装系统依赖"）

开启 `confirmDestructive: true`，让 AI 在执行破坏性操作前确认。

**误区三：MCP 服务器是安全的，因为它们是官方的**

官方 MCP Server 的代码是安全的，但配置不当仍然有风险。比如，PostgreSQL MCP 用超级用户连接、GitHub MCP 授予了所有权限——这些配置问题不是 MCP Server 的责任，而是你的责任。

**误区四：本地模型不需要考虑安全**

本地模型确实不会把代码发送到云端，但其他安全风险仍然存在——AI 可能修改错误的文件、执行危险的命令、删除重要的数据。工具权限控制和确认机制在使用本地模型时同样重要。

---

## 小结

这一节我们系统化了 OpenCode 的安全最佳实践：API Key 管理（环境变量引用、定期轮换、auth.json 存储）、工具权限控制（最小权限原则、确认机制、工作目录限制）、MCP 安全（只启用信任的 MCP、权限最小化、环境变量安全）、代码隐私（本地模型、企业级部署、避免发送敏感文件）。核心原则是"安全是配置和使用的责任"——开源让你可审计，但配置不当仍然有风险。下一节我们学习团队配置标准化。
