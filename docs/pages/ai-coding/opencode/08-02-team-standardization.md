# 8.2 团队配置标准化

> **10 个人用 OpenCode，10 种不同的配置——代码风格不一致、模型选择混乱、MCP 工具各异。团队配置标准化，让 AI 编程从"个人工具"升级为"团队基础设施"。**

---

## 这一节在讲什么？

OpenCode 的灵活性是它的优势——每个人可以自由选择模型、配置工具、定制命令。但在团队协作中，过度的灵活性反而会导致混乱——不同成员用不同的模型，AI 生成的代码风格不一致；不同成员配置不同的 MCP，协作时工具能力不统一。这一节我们学习怎么通过项目级配置、AGENTS.md 模板、MCP 配置共享和模型策略统一，把 OpenCode 从"个人工具"升级为"团队基础设施"。

---

## 项目级配置：团队统一的起点

项目级配置（`.opencode/opencode.json`）是团队标准化的核心——它定义了项目级别的模型选择、Agent 配置和 MCP 服务器，提交到 Git 后，所有团队成员自动使用相同的配置。

### 推荐的项目级配置模板

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
  },
  "mcp": {
    "github": {
      "type": "local",
      "command": ["npx", "-y", "@modelcontextprotocol/server-github"],
      "enabled": true,
      "env": {
        "GITHUB_TOKEN": "$GITHUB_TOKEN"
      }
    },
    "context7": {
      "type": "local",
      "command": ["npx", "-y", "@upstash/context7-mcp@latest"],
      "enabled": true
    }
  },
  "autoCompact": true,
  "commands": {
    "review": {
      "description": "Review current code changes",
      "prompt": "Review the current git diff. Focus on security, performance, and correctness. Follow our code conventions in AGENTS.md."
    },
    "test-and-fix": {
      "description": "Run tests and fix failures",
      "prompt": "Run the test suite. If any tests fail, fix the code and run tests again until all pass."
    }
  }
}
```

这个配置模板做了几件事：
1. **统一模型**：所有成员默认使用 Claude Sonnet 4
2. **统一 MCP**：所有成员使用 GitHub MCP 和 Context7 MCP
3. **统一命令**：所有成员使用相同的自定义命令
4. **环境变量引用**：敏感信息通过 `$ENV_VAR` 引用，不硬编码

### 全局配置 vs 项目配置的分工

```
全局配置（~/.config/opencode/opencode.json）：
  → 个人偏好：主题、默认 shell
  → API Key：通过 opencode auth login 配置
  → 不提交到 Git

项目配置（.opencode/opencode.json）：
  → 团队规范：模型选择、Agent 配置
  → 团队工具：MCP 服务器、自定义命令
  → 提交到 Git，所有成员共享
```

---

## AGENTS.md 模板

AGENTS.md 是团队共享的"AI 说明书"——它确保所有成员的 AI 都遵循相同的代码规范和项目约定。

### 团队 AGENTS.md 的关键内容

```markdown
# Project: team-project

## Architecture
- Frontend: React + TypeScript + Tailwind CSS
- Backend: Node.js + Express + Prisma
- Database: PostgreSQL
- Auth: JWT with refresh token rotation

## Code Conventions
- Use TypeScript strict mode
- Use async/await, no .then() chains
- Error handling: use AppError class from src/errors/
- API responses: { data, error } format
- Naming: camelCase for variables, PascalCase for classes
- No console.log in production code, use logger

## Database Conventions
- All tables: id (UUID), createdAt, updatedAt
- Soft delete: deletedAt field
- Multi-tenancy: all queries must include tenantId

## Testing Conventions
- Unit tests: co-located with source file
- API tests: tests/api/ directory
- Mock external services
- Minimum 80% coverage for new code

## Common Pitfalls
- DON'T forget tenantId in database queries
- DON'T use getServerSession in client components
- DON'T hard-delete records
- DON'T skip error handling in API routes
- DON'T commit .env files
```

### AGENTS.md 的维护

AGENTS.md 应该由团队共同维护——当有人发现新的常见错误、调整代码规范、添加新的约定时，都应该更新 AGENTS.md。

建议在 code review 时检查 AGENTS.md 是否需要更新——如果 reviewer 发现了一个新的常见错误模式，应该在 AGENTS.md 的 Common Pitfalls 中添加。

---

## MCP 配置共享

项目级配置中的 MCP 部分可以提交到 Git，确保团队成员使用相同的 MCP 工具集。

### MCP 配置的共享策略

```
共享的内容（提交到 Git）：
  - MCP Server 的类型和启动命令
  - MCP Server 的 enabled 状态
  - 环境变量引用（$GITHUB_TOKEN）

不共享的内容（个人配置）：
  - 环境变量的实际值（每个人的 Token 不同）
  - 个人额外的 MCP Server（如个人项目专用的）
```

### 团队 MCP 配置示例

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
    "postgres": {
      "type": "local",
      "command": ["npx", "-y", "@modelcontextprotocol/server-postgres"],
      "enabled": true,
      "env": {
        "DATABASE_URL": "$DATABASE_URL"
      }
    }
  }
}
```

每个团队成员需要自己设置环境变量：

```bash
# 每个成员的 ~/.zshrc
export GITHUB_TOKEN="ghp_..."        # 每人自己的 Token
export DATABASE_URL="postgresql://..." # 开发库连接字符串
```

---

## 模型策略统一

团队应该统一模型选择策略，避免不同成员用不同模型导致代码风格不一致。

### 推荐的团队模型策略

```
团队模型策略：

  默认模型（日常编码）：
  → Claude Sonnet 4
  → 性能与成本的最佳平衡

  快速模型（简单任务）：
  → Claude Haiku 4
  → 代码审查、简单修改、快速问答

  不推荐团队使用的模型：
  → GPT-4o（代码风格可能跟 Claude 不一致）
  → Ollama 本地模型（质量不稳定）
  → Opus（成本太高，日常不需要）

  特殊场景（需团队负责人批准）：
  → Gemini 2.5 Pro（需要超长上下文时）
  → DeepSeek（预算紧张时）
```

### 模型策略的配置

在项目级配置中设置默认模型：

```json
{
  "agents": {
    "coder": {
      "model": "anthropic/claude-sonnet-4-20250514",
      "maxTokens": 16384
    }
  }
}
```

团队成员可以在 TUI 中用 `/models` 临时切换模型，但默认模型是统一的。

---

## 新人 Onboarding

新成员加入团队时，OpenCode 的 onboarding 流程应该简单明确：

```bash
# 1. 安装 OpenCode
curl -fsSL https://opencode.ai/install | bash

# 2. 配置 API Key（推荐 Claude Pro 订阅）
opencode auth login

# 3. 克隆项目
git clone <repo-url>
cd <project>

# 4. 设置环境变量（团队文档中列出需要的变量）
export GITHUB_TOKEN="ghp_..."
export DATABASE_URL="postgresql://..."

# 5. 启动 OpenCode（项目配置和 AGENTS.md 已在仓库中）
opencode

# 6. 开始使用
"帮我了解一下这个项目的架构"
```

### Onboarding 文档

团队应该维护一份 OpenCode onboarding 文档，包含：

- 安装步骤
- API Key 获取方式（公司是否有统一的 API Key？还是个人自行购买？）
- 需要设置的环境变量列表
- 项目级配置的说明
- AGENTS.md 的阅读指南
- 常用自定义命令的使用方法
- 安全注意事项

---

## 常见误区

**误区一：团队配置太严格，不允许个人自定义**

项目级配置设置了团队统一的"底线"，但不应该完全禁止个人自定义。全局配置允许个人设置主题、额外的 MCP Server 等个人偏好。团队配置和个人配置是互补的——团队配置确保一致性，个人配置保留灵活性。

**误区二：AGENTS.md 写一次就不用管了**

项目在演进，AGENTS.md 也应该跟着更新。建议每次 sprint 结束后检查 AGENTS.md 是否需要更新——有没有新的常见错误？代码规范有没有调整？目录结构有没有变化？

**误区三：所有团队都用相同的配置**

不同项目的需求不同——前端项目需要 shadcn MCP，后端项目需要 PostgreSQL MCP，DevOps 项目需要 Docker MCP。团队配置应该按项目定制，而不是一刀切。

**误区四：新人不需要了解 OpenCode 配置**

新人应该了解项目的 OpenCode 配置——特别是 AGENTS.md 中的代码规范和常见陷阱。这些信息不仅对 AI 有用，对新人理解项目规范也很有帮助。建议把"阅读 AGENTS.md"加入新人的 onboarding 清单。

---

## 小结

这一节我们学习了团队配置标准化的四个方面：项目级配置统一模型选择、Agent 配置和 MCP 服务器；AGENTS.md 模板统一代码规范和项目约定；MCP 配置共享确保团队使用相同的工具集；模型策略统一避免代码风格不一致。团队配置的核心原则是"团队统一底线，个人保留灵活性"——项目配置确保一致性，全局配置保留个人偏好。下一节我们讨论 OpenCode 的局限性和替代方案。
