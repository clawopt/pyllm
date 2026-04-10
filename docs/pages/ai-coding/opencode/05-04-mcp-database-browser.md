# 5.4 MCP 实战：数据库与浏览器自动化

> **让 AI 直接查数据库、操作浏览器——MCP 把 AI 的能力从"读写文件"扩展到"操作一切"。**

---

## 这一节在讲什么？

上一节我们实战了 GitHub MCP，这一节我们继续探索两个强大的 MCP 场景——数据库和浏览器自动化。PostgreSQL MCP 让 AI 能直接查询数据库、分析数据、生成迁移脚本；Playwright MCP 让 AI 能操作浏览器、截图、填写表单、自动化测试。这两个 MCP 把 AI 的能力从"读写本地文件"扩展到"操作外部服务"，是 OpenCode 区别于其他 AI 编程工具的关键能力。

---

## PostgreSQL MCP：让 AI 查询数据库

### 配置

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

设置环境变量：

```bash
export DATABASE_URL="postgresql://user:password@localhost:5432/mydb"
```

### 使用场景

**查看数据库结构**：

```
这个数据库有哪些表？每张表的结构是什么？
```

AI 会调用 PostgreSQL MCP 的 `list_tables` 和 `describe_table` 工具，列出所有表及其字段、类型、约束。

**查询数据**：

```
查看最近 7 天注册的用户数量
```

AI 会生成 SQL 查询并执行：

```sql
SELECT COUNT(*) FROM users WHERE created_at > NOW() - INTERVAL '7 days';
```

**分析数据**：

```
分析一下 orders 表的数据分布，看看有没有异常
```

AI 会执行多个查询来分析数据分布、查找异常值、统计缺失率。

**生成迁移脚本**：

```
给 users 表添加一个 phone 字段，类型是 VARCHAR(20)，可以为空
```

AI 会生成迁移 SQL：

```sql
ALTER TABLE users ADD COLUMN phone VARCHAR(20);
```

**调试数据问题**：

```
用户 ID 为 42 的订单为什么显示"待支付"？
```

AI 会查询相关表的数据，分析状态流转，找出问题原因。

### 安全注意事项

PostgreSQL MCP 给了 AI 执行任意 SQL 的能力——这包括 SELECT，也包括 INSERT、UPDATE、DELETE。为了安全：

**1. 使用只读数据库用户**

```bash
# 创建只读用户
CREATE USER opencode_readonly WITH PASSWORD 'xxx';
GRANT SELECT ON ALL TABLES IN SCHEMA public TO opencode_readonly;

# 连接字符串使用只读用户
export DATABASE_URL="postgresql://opencode_readonly:xxx@localhost:5432/mydb"
```

**2. 避免在生产数据库上使用**

PostgreSQL MCP 适合在开发数据库上使用，不建议直接连接生产数据库。如果必须连接生产库，务必使用只读用户。

**3. 敏感数据脱敏**

AI 查询的数据会出现在对话历史中——如果你的数据库包含敏感信息（用户密码、身份证号等），AI 可能会在回答中展示这些数据。建议在开发库中使用脱敏数据。

---

## Playwright MCP：让 AI 操作浏览器

### 配置

```json
{
  "mcp": {
    "playwright": {
      "type": "local",
      "command": ["npx", "-y", "@playwright/mcp@latest"],
      "enabled": true
    }
  }
}
```

### 使用场景

**截图和页面分析**：

```
打开 http://localhost:3000，截个图，看看页面有没有布局问题
```

AI 会调用 Playwright MCP 的 `navigate` 和 `screenshot` 工具，打开页面并截图。截图会显示在对话区，AI 会分析页面布局是否有问题。

**自动化测试**：

```
测试一下登录功能：打开登录页面，输入用户名 admin 和密码 123456，点击登录按钮，检查是否跳转到首页
```

AI 会依次调用 Playwright MCP 的工具：
1. `navigate`：打开登录页面
2. `fill`：输入用户名和密码
3. `click`：点击登录按钮
4. `screenshot`：截图确认跳转结果

**表单填写**：

```
帮我在 http://localhost:3000/admin/users 页面创建一个新用户，
用户名是 testuser，邮箱是 test@example.com，角色是 editor
```

AI 会操作浏览器填写表单并提交。

**调试前端问题**：

```
打开 http://localhost:3000/dashboard，看看控制台有没有错误
```

AI 会打开页面并检查浏览器控制台的错误信息。

### Playwright MCP 的工具

| 工具 | 功能 |
|------|------|
| navigate | 打开 URL |
| screenshot | 截取页面截图 |
| click | 点击元素 |
| fill | 填写输入框 |
| select | 选择下拉选项 |
| hover | 鼠标悬停 |
| evaluate | 执行 JavaScript |

---

## Supabase MCP：云端数据库

如果你的项目使用 Supabase，可以用 Supabase MCP 替代 PostgreSQL MCP——它提供了更丰富的 Supabase 特定功能：

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

Supabase MCP 的优势：
- 不需要本地启动进程（远程 MCP）
- 支持 Supabase 特有的功能（RLS 策略、Edge Functions 等）
- 自动处理认证和连接

使用示例：

```
查看 users 表的 RLS 策略
```

```
创建一个迁移，给 orders 表添加 status 字段
```

---

## 自定义 MCP 服务器

如果现成的 MCP Server 不能满足你的需求，你可以自己开发。MCP Server 可以用 TypeScript 或 Python 编写。

### TypeScript MCP Server 示例

```typescript
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";

const server = new McpServer({
  name: "my-custom-mcp",
  version: "1.0.0",
});

server.tool(
  "hello",
  "Say hello to someone",
  { name: z.string().describe("Name to greet") },
  async ({ name }) => ({
    content: [{ type: "text", text: `Hello, ${name}!` }],
  })
);

const transport = new StdioServerTransport();
await server.connect(transport);
```

### 配置自定义 MCP Server

```json
{
  "mcp": {
    "my-custom": {
      "type": "local",
      "command": ["node", "/path/to/my-mcp-server.js"],
      "enabled": true
    }
  }
}
```

自定义 MCP Server 的开发超出了本教程的范围，但核心思路很简单：定义工具名称、参数和执行逻辑，然后通过 MCP 协议暴露给 AI。

---

## 常见误区

**误区一：MCP 工具没有权限控制**

有。你可以在 Agent 配置中精细控制哪些 Agent 可以使用哪些 MCP 工具。比如，让 coder Agent 可以使用 GitHub MCP，但不能使用 PostgreSQL MCP。这是安全控制的重要手段。

**误区二：PostgreSQL MCP 会修改生产数据**

如果你用的是只读数据库用户，PostgreSQL MCP 只能执行 SELECT 查询，不能修改数据。安全的关键是使用最小权限的数据库用户——不要用超级用户连接。

**误区三：Playwright MCP 只能测试网页**

不只是测试。Playwright MCP 可以做任何浏览器自动化——填写表单、抓取数据、截图、执行 JavaScript。你可以用它来调试前端问题、自动化重复操作、甚至做简单的爬虫。

**误区四：远程 MCP 不如本地 MCP 安全**

不一定。远程 MCP 的安全性取决于远程服务的安全措施——Supabase MCP 使用 OAuth 认证，安全性不比本地 MCP 差。关键是你信任谁——本地 MCP 信任的是你本地运行的代码，远程 MCP 信任的是远程服务的安全策略。

---

## 小结

这一节我们实战了数据库和浏览器自动化 MCP：PostgreSQL MCP 让 AI 能查询数据库、分析数据、生成迁移脚本（注意使用只读用户和开发库）；Playwright MCP 让 AI 能操作浏览器、截图、填写表单、自动化测试；Supabase MCP 提供了更丰富的云端数据库功能；自定义 MCP Server 让你能开发自己的工具扩展。MCP 把 AI 的能力从"读写本地文件"扩展到"操作一切"——这是 OpenCode 区别于其他 AI 编程工具的核心优势。下一章我们进入实战开发场景，看看 OpenCode 在真实项目中的使用方式。
