# 5.2 配置 MCP 服务器

> **配置 MCP 就像安装 VS Code 扩展——找到你需要的 MCP Server，加几行 JSON，重启 OpenCode，AI 就多了新能力。**

---

## 这一节在讲什么？

上一节我们理解了 MCP 是什么、它为什么重要。这一节我们动手——学习怎么在 OpenCode 中配置 MCP 服务器。OpenCode 支持两种 MCP 服务器类型：本地（local）和远程（remote），配置方式都很简单。我们会逐一讲解配置格式、常用 MCP Server 的配置示例、权限控制，以及配置过程中的常见问题。

---

## MCP 配置格式

MCP 服务器配置在 opencode.json 的 `mcp` 字段中：

```json
{
  "mcp": {
    "server-name": {
      "type": "local | remote",
      "command": ["..."],
      "url": "...",
      "enabled": true,
      "env": {},
      "headers": {}
    }
  }
}
```

每个 MCP 服务器的配置项：

| 字段 | 说明 | 必填 |
|------|------|------|
| type | `local`（本地进程）或 `remote`（远程 HTTP/SSE） | 是 |
| command | 本地 MCP Server 的启动命令（type 为 local 时） | local 时必填 |
| url | 远程 MCP Server 的 URL（type 为 remote 时） | remote 时必填 |
| enabled | 是否启用，默认 true | 否 |
| env | 传递给 MCP Server 的环境变量 | 否 |
| headers | 远程 MCP Server 的 HTTP 头（type 为 remote 时） | 否 |

---

## 本地 MCP 服务器

本地 MCP 服务器通过命令行启动——OpenCode 会自动启动和管理这些进程。

### GitHub MCP Server

最常用的 MCP Server 之一，让 AI 能操作 GitHub：

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
    }
  }
}
```

配置前需要设置 GitHub Token：

```bash
export GITHUB_TOKEN="ghp_..."
```

Token 需要的权限取决于你想让 AI 做什么——创建 Issue 需要 `issues` 权限，操作 PR 需要 `pull_requests` 权限。建议使用最小权限原则，只授予必要的权限。

### Filesystem MCP Server

让 AI 能访问指定目录的文件系统：

```json
{
  "mcp": {
    "filesystem": {
      "type": "local",
      "command": ["npx", "-y", "@modelcontextprotocol/server-filesystem", "/path/to/allowed/directory"],
      "enabled": true
    }
  }
}
```

注意：Filesystem MCP Server 的最后一个参数是允许访问的目录路径——AI 只能访问这个目录及其子目录下的文件。这是安全控制的重要手段。

### PostgreSQL MCP Server

让 AI 能查询 PostgreSQL 数据库：

```json
{
  "mcp": {
    "postgres": {
      "type": "local",
      "command": ["npx", "-y", "@modelcontextprotocol/server-postgres", "postgresql://user:pass@localhost:5432/dbname"],
      "enabled": true
    }
  }
}
```

**安全警告**：数据库连接字符串包含用户名和密码，不要硬编码在配置文件里。建议用环境变量引用：

```json
{
  "mcp": {
    "postgres": {
      "type": "local",
      "command": ["npx", "-y", "@modelcontextprotocol/server-postgres", "$DATABASE_URL"],
      "enabled": true
    }
  }
}
```

### Context7 MCP Server

提供智能上下文增强——让 AI 能获取最新的文档和 API 信息：

```json
{
  "mcp": {
    "context7": {
      "type": "local",
      "command": ["npx", "-y", "@upstash/context7-mcp@latest"],
      "enabled": true
    }
  }
}
```

### Shadcn MCP Server

前端开发利器——让 AI 能直接添加 shadcn/ui 组件：

```json
{
  "mcp": {
    "shadcn": {
      "type": "local",
      "command": ["npx", "-y", "shadcn", "mcp"],
      "enabled": true
    }
  }
}
```

---

## 远程 MCP 服务器

远程 MCP 服务器通过 HTTP/SSE 连接——不需要在本地启动进程，直接连接远程服务。

### Supabase MCP Server

让 AI 能操作 Supabase 数据库：

```json
{
  "mcp": {
    "supabase": {
      "type": "remote",
      "url": "https://mcp.supabase.com/mcp",
      "enabled": true,
      "headers": {
        "Authorization": "Bearer $SUPABASE_ACCESS_TOKEN"
      }
    }
  }
}
```

获取 Supabase Access Token：
1. 登录 Supabase Dashboard
2. 进入 Account Settings → Access Tokens
3. 生成新 Token（以 `sbp_` 开头）

### 自定义远程 MCP Server

如果你自己开发了 MCP Server 并部署在远程，可以这样配置：

```json
{
  "mcp": {
    "my-custom-server": {
      "type": "remote",
      "url": "https://my-mcp-server.example.com/mcp",
      "enabled": true,
      "headers": {
        "Authorization": "Bearer $MY_MCP_TOKEN"
      }
    }
  }
}
```

---

## MCP 权限控制

OpenCode 允许你在 Agent 配置中控制哪些 Agent 可以使用哪些 MCP 工具：

```json
{
  "agents": {
    "coder": {
      "model": "anthropic/claude-sonnet-4-20250514",
      "maxTokens": 16384,
      "mcp": {
        "github": true,
        "postgres": false
      }
    }
  }
}
```

上面的配置让 coder Agent 可以使用 GitHub MCP，但不能使用 PostgreSQL MCP。这在团队协作中很有用——你可以给不同的 Agent 配置不同的 MCP 权限。

---

## 用 opencode mcp 命令管理

OpenCode 提供了 CLI 命令来管理 MCP 服务器：

```bash
# 列出所有 MCP 服务器及连接状态
opencode mcp list

# 交互式添加 MCP 服务器
opencode mcp add

# 查看 MCP 服务器的认证状态
opencode mcp auth
```

在 TUI 中，你也可以通过状态栏查看 MCP 服务器的连接状态——绿色圆点表示已连接，红色表示连接失败。

---

## 配置多个 MCP 服务器

你可以同时配置多个 MCP 服务器，OpenCode 会自动管理它们的启动和连接：

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
      "command": ["npx", "-y", "@modelcontextprotocol/server-postgres", "$DATABASE_URL"],
      "enabled": true
    },
    "context7": {
      "type": "local",
      "command": ["npx", "-y", "@upstash/context7-mcp@latest"],
      "enabled": true
    },
    "shadcn": {
      "type": "local",
      "command": ["npx", "-y", "shadcn", "mcp"],
      "enabled": true
    }
  }
}
```

**注意**：每个 MCP 服务器启动时都需要初始化，配置太多会导致 OpenCode 启动变慢。建议只启用项目需要的 MCP 服务器——前端项目用 shadcn MCP，后端项目用数据库 MCP，不需要全部都配。

---

## 常见配置问题

**问题一：MCP 服务器启动失败**

```
MCP: github ✗ (connection failed)
```

常见原因：
- npx 没有安装或不在 PATH 中
- 网络问题导致 npx 下载包失败
- 环境变量未设置（如 GITHUB_TOKEN）

解决方法：

```bash
# 检查 npx 是否可用
npx --version

# 手动测试 MCP Server 是否能启动
npx -y @modelcontextprotocol/server-github

# 检查环境变量
echo $GITHUB_TOKEN
```

**问题二：MCP 服务器启动慢**

每个本地 MCP 服务器都需要 npx 下载和启动，首次启动可能需要 10~30 秒。后续启动会快一些（npx 有缓存）。如果你觉得启动太慢，可以考虑：

- 只启用必要的 MCP 服务器
- 使用 `npm install -g` 全局安装 MCP Server，避免每次 npx 下载
- 使用远程 MCP 服务器（不需要本地启动）

**问题三：数据库连接字符串包含敏感信息**

不要把数据库密码硬编码在配置文件里。用环境变量引用：

```json
{
  "mcp": {
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

---

## 常见误区

**误区一：配置了太多 MCP 服务器导致启动慢**

每个 MCP 服务器启动时都需要初始化，配置 5 个以上 MCP 服务器可能导致 OpenCode 启动需要 30 秒以上。只启用项目需要的 MCP——前端项目用 shadcn，后端项目用数据库，不需要全部都配。

**误区二：MCP 服务器不需要权限控制**

MCP 服务器给了 AI 强大的能力——GitHub MCP 可以创建 Issue 和 PR，PostgreSQL MCP 可以执行任意 SQL。如果不做权限控制，AI 可能会执行你不希望的操作。建议：只启用必要的 MCP，在 Agent 配置中限制 MCP 权限，敏感操作开启确认机制。

**误区三：本地 MCP 服务器不需要网络**

本地 MCP 服务器确实不需要网络来"连接"（不像远程 MCP），但 npx 启动时需要从 npm 下载包。如果你在离线环境，需要提前全局安装 MCP Server：

```bash
npm install -g @modelcontextprotocol/server-github
```

然后在配置中直接使用已安装的命令。

**误区四：MCP 配置只能放在全局配置里**

不是。MCP 配置也可以放在项目级配置（`.opencode/opencode.json`）里——这样不同的项目可以配置不同的 MCP 服务器。比如，项目 A 用 GitHub MCP，项目 B 用 PostgreSQL MCP，互不干扰。

---

## 小结

这一节我们学习了 MCP 服务器的配置方式：本地 MCP 通过 `command` 启动进程，远程 MCP 通过 `url` 连接服务。常用的 MCP Server 包括 GitHub（操作 Issue/PR）、PostgreSQL（查询数据库）、Filesystem（文件系统访问）、Context7（上下文增强）、Shadcn（UI 组件）。配置时注意安全——用环境变量引用敏感信息，只启用必要的 MCP，做好权限控制。下一节我们实战 GitHub MCP，看看 AI 怎么帮你操作 GitHub。
