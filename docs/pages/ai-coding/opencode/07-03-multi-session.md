# 7.3 多会话与协作

> **一个会话修 bug，一个会话加功能，一个会话做代码审查——OpenCode 的多会话能力让你同时推进多个任务。**

---

## 这一节在讲什么？

实际开发中，你很少只做一件事——你可能同时在修一个 bug、开发一个新功能、审查同事的代码。如果用单会话的 AI 工具，你需要频繁切换上下文，效率很低。OpenCode 的多会话能力让你同时维护多个独立的对话，每个会话有独立的上下文和历史。这一节我们学习多会话并行、会话分享、远程开发，以及团队协作的最佳实践。

---

## 多会话并行

OpenCode 支持同时维护多个会话，每个会话有独立的上下文、对话历史和文件变更。

### 会话管理

在 TUI 中用 `/sessions` 命令管理会话：

```
/sessions
```

这会打开会话管理界面，显示所有会话的列表：

```
┌──────────────────────────────────────────────────────────┐
│  Sessions                                                 │
│                                                           │
│  ● fix-login-bug          15 min ago    12 messages       │
│  ● add-pagination         2 hours ago   28 messages       │
│  ○ code-review            yesterday     8 messages        │
│                                                           │
│  [n] New session  [d] Delete  [Enter] Switch              │
└──────────────────────────────────────────────────────────┘
```

你可以：
- 切换到已有会话（Enter）
- 创建新会话（n）
- 删除旧会话（d）

### 启动时指定会话

```bash
# 继续上次的会话
opencode -c

# 指定会话 ID
opencode -s <session-id>

# 带初始提示词启动
opencode --prompt "Help me fix the login bug"
```

### 多会话的使用策略

```
多会话的最佳实践：

  场景 1：同时处理多个任务
  - 会话 A：修 bug（上下文聚焦在 bug 相关文件）
  - 会话 B：加功能（上下文聚焦在功能相关文件）
  - 会话 C：代码审查（只读模式，不改代码）

  场景 2：同一任务的不同阶段
  - 会话 A：Plan 模式，规划方案
  - 会话 B：Build 模式，执行修改
  （不建议——同一个任务用一个会话更连贯）

  场景 3：不同项目
  - 会话 A：项目 X 的开发
  - 会话 B：项目 Y 的开发
  （不建议——不同项目应该在不同目录启动 OpenCode）
```

**建议**：多会话最适合"同时处理多个独立任务"——每个任务一个会话，上下文互不干扰。不建议用多会话处理同一个任务的不同阶段——同一个任务用一个会话更连贯，AI 能看到完整的修改历史。

---

## 会话分享

OpenCode 的 `/share` 命令可以生成当前对话的分享链接：

```
/share
```

这会创建一个包含完整对话历史的链接，复制到剪贴板。你可以把这个链接发给同事，让他们看到你和 AI 的完整对话。

### 会话分享的使用场景

**1. 知识分享**

你用 OpenCode 解决了一个复杂问题，想把解决过程分享给团队：

```
/share
→ 把链接发到团队群聊
→ 同事可以看到 AI 的分析过程和最终方案
```

**2. 代码审查辅助**

你让 AI 审查了一段代码，想把审查结果分享给 PR 作者：

```
/share
→ 把链接贴在 PR 评论里
→ PR 作者可以看到 AI 的审查意见
```

**3. 新人培训**

新人不知道怎么用 OpenCode，你可以分享一个示例对话：

```
/share
→ 新人看到你是怎么跟 AI 对话的
→ 学习提问技巧和上下文控制方法
```

### 会话分享的隐私注意

分享链接包含完整的对话历史——包括你引用的文件内容、执行的命令输出等。分享前请注意：

- 不要分享包含 API Key 或密码的对话
- 不要分享包含公司敏感代码的对话
- 检查对话中是否有不应该公开的信息

---

## 远程开发

OpenCode 支持通过 `opencode serve` 和 `opencode attach` 进行远程开发——在一台服务器上运行 OpenCode，从另一台机器连接。

### 启动远程服务器

```bash
# 在远程服务器上启动 OpenCode 服务
opencode serve --port 4096 --hostname 0.0.0.0
```

### 从本地连接

```bash
# 从本地连接到远程 OpenCode
opencode attach http://remote-server:4096
```

### 远程开发的使用场景

**1. 在开发服务器上使用**

你的项目运行在远程开发服务器上，你从本地 SSH 连接。与其在 SSH 会话里用 OpenCode（可能受限于终端大小和网络延迟），不如用 `opencode serve` + `opencode attach` 的方式——在远程服务器上运行 OpenCode，从本地 TUI 连接。

**2. 避免 MCP 冷启动**

MCP 服务器启动需要时间（10~30 秒）。如果你频繁开关 OpenCode，每次都要等 MCP 启动。用 `opencode serve` 让 OpenCode 一直运行，MCP 服务器只启动一次，后续连接直接使用。

**3. 团队共享**

在共享的开发服务器上运行 `opencode serve`，团队成员可以用 `opencode attach` 连接。但注意——每个用户应该有独立的会话，不要共享同一个会话。

---

## Web 界面

OpenCode 还提供了 Web 界面——用浏览器访问 OpenCode：

```bash
opencode web --port 4096
```

然后在浏览器中打开 `http://localhost:4096`，你会看到一个基于 Web 的 OpenCode 界面。

Web 界面的优势：
- 不需要终端模拟器——在任何有浏览器的设备上都能用
- 支持更丰富的 Markdown 渲染——代码高亮、表格、图片
- 适合在平板或手机上使用

---

## 团队协作最佳实践

### AGENTS.md 共享

AGENTS.md 提交到 Git，确保团队成员的 AI 都有相同的项目上下文。当有人更新 AGENTS.md 时，其他人 pull 后自动获得更新。

### MCP 配置共享

项目级的 MCP 配置（`.opencode/opencode.json` 中的 `mcp` 字段）也可以提交到 Git，确保团队成员使用相同的 MCP 工具集。

但注意——MCP 配置中的环境变量引用（如 `$GITHUB_TOKEN`）不会泄露 Token，因为每个团队成员的 Token 是不同的。只有 MCP Server 的类型和启动命令是共享的。

### 会话模板

团队可以约定一些常用的会话模板——比如"代码审查模板"、"bug 修复模板"、"功能开发模板"。这些模板可以写在 AGENTS.md 或团队文档中：

```markdown
## Session Templates

### Bug Fix Template
1. Describe the bug with error message and stack trace
2. Use $ to include relevant command output
3. Use @ to reference the affected file
4. Ask AI to analyze the root cause
5. Review AI's fix before applying

### Feature Development Template
1. Start in Plan mode
2. Describe the feature with acceptance criteria
3. Reference existing similar features with @
4. Review AI's plan
5. Switch to Build mode and implement
6. Run tests to verify
```

### 新人 Onboarding

新成员加入团队时，可以用以下步骤快速上手 OpenCode：

```bash
# 1. 安装 OpenCode
curl -fsSL https://opencode.ai/install | bash

# 2. 配置 API Key
opencode auth login

# 3. 克隆项目
git clone <repo-url>
cd <project>

# 4. 启动 OpenCode（AGENTS.md 已在仓库中）
opencode

# 5. 运行 /init（如果 AGENTS.md 需要更新）
/init

# 6. 开始使用
"帮我了解一下这个项目的架构"
```

---

## 常见误区

**误区一：一个会话处理所有任务**

这是最常见的低效用法。一个会话里同时修 bug、加功能、做审查，上下文互相干扰，AI 的回答质量会下降。建议"一事一会话"——每个独立任务开一个新会话。

**误区二：分享会话链接不检查隐私**

分享链接包含完整的对话历史，包括引用的文件内容。分享前一定要检查对话中是否有敏感信息——API Key、密码、公司内部代码等。

**误区三：远程开发不需要安全措施**

`opencode serve` 默认监听所有网络接口（`0.0.0.0`），这意味着同一网络中的任何人都能连接。在生产环境中，应该限制访问——使用防火墙、VPN 或 SSH 隧道：

```bash
# 只监听本地（通过 SSH 隧道访问）
opencode serve --hostname 127.0.0.1

# 然后从本地通过 SSH 隧道连接
ssh -L 4096:localhost:4096 user@remote-server
opencode attach http://localhost:4096
```

**误区四：多会话会消耗更多 API 费用**

多会话本身不会增加 API 费用——费用取决于你发送了多少 token，不取决于你开了多少会话。但多会话可能导致你更频繁地使用 AI，间接增加费用。建议根据实际需要开新会话，不要为了"多任务"而开太多闲置会话。

---

## 小结

这一节我们学习了 OpenCode 的多会话与协作能力：多会话并行让你同时处理多个独立任务，`/share` 分享对话给团队成员，`opencode serve` + `opencode attach` 支持远程开发，`opencode web` 提供 Web 界面。团队协作的最佳实践包括：AGENTS.md 共享、MCP 配置共享、会话模板、新人 Onboarding 流程。核心原则是"一事一会话"——每个独立任务开一个新会话，保持上下文干净。下一章我们进入生产化与团队部署。
