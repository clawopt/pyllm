# 3.3 Agent 工具系统

> **AI 编程助手的能力边界，取决于它能调用哪些工具。理解 OpenCode 的工具系统，你才能知道 AI 能做什么、不能做什么、怎么让它做得更好。**

---

## 这一节在讲什么？

前面两节我们学了 TUI 界面和上下文控制，现在要深入 OpenCode 的"引擎室"——Agent 工具系统。OpenCode 的 AI 不是只会聊天的机器人，它能读取文件、修改代码、执行命令、搜索代码、获取 LSP 诊断——这些都是通过工具系统实现的。理解工具系统，你就能知道 AI 为什么有时候"很聪明"（因为它调用了正确的工具），有时候"很蠢"（因为它缺少必要的工具或权限）。

---

## 内置工具全景

OpenCode 的内置工具是 AI 操作项目的基本能力。每个工具对应一种操作类型：

### READ：读取文件内容

AI 用 READ 工具读取项目文件，理解代码内容。这是 AI 最基础的能力——没有 READ，AI 就像一个盲人，看不到你的代码。

```
AI 的内部调用过程（你看不到，但它在后台发生）：

  用户："这个函数是做什么的？"
  → AI 决定调用 READ 工具
  → READ 读取文件内容
  → AI 根据文件内容回答问题
```

READ 工具不需要你手动触发——当你问关于代码的问题时，AI 会自动决定是否需要读取文件。但你可以通过 `@` 引用来"建议"AI 读取特定文件，提高效率。

### WRITE：创建或修改文件

AI 用 WRITE 工具创建新文件或修改已有文件。这是 AI "动手干活"的核心能力。

```
AI 修改文件的过程：

  1. AI 读取目标文件（READ）
  2. AI 分析需要修改的部分
  3. AI 生成修改后的文件内容
  4. AI 调用 WRITE 工具写入修改
  5. OpenCode 在对话区显示变更摘要
```

WRITE 工具在 Plan 模式下不可用——这是 Plan 模式的核心约束：AI 只能"说"不能"做"。

### RUN：执行 shell 命令

AI 用 RUN 工具执行 shell 命令。这让 AI 能运行测试、安装依赖、查看日志、启动服务等。

```
AI 执行命令的典型场景：

  - 运行测试：npm test, pytest, go test ./...
  - 安装依赖：npm install, pip install, go mod tidy
  - 查看日志：cat logs/error.log, docker logs container
  - 构建项目：npm run build, cargo build, make
  - Git 操作：git status, git diff, git log
```

RUN 工具是"双刃剑"——它让 AI 能做很多事，但也可能执行危险命令。OpenCode 有安全机制来控制这一点，我们后面会讲。

### SEARCH：搜索代码

AI 用 SEARCH 工具在项目中搜索代码。当 AI 需要找到某个函数的定义、某个类的引用、某个配置的位置时，它会调用 SEARCH。

```
AI 搜索代码的典型场景：

  - 查找函数定义："authenticate 函数在哪里定义的？"
  - 查找引用："哪些地方调用了 sendEmail？"
  - 搜索模式："项目中有没有使用 deprecated API 的地方？"
  - 查找配置："数据库连接字符串在哪里配置的？"
```

SEARCH 工具支持 glob 模式和正则表达式，AI 会根据需要选择合适的搜索方式。

### DIAGNOSTICS：代码诊断（LSP）

AI 用 DIAGNOSTICS 工具获取 LSP（Language Server Protocol）提供的代码诊断信息——类型错误、lint 警告、未使用的变量等。

```
AI 使用 LSP 诊断的典型场景：

  - 修改代码后检查是否引入了新的类型错误
  - 分析项目中的 lint 警告
  - 查找未使用的导入和变量
  - 验证修改后的代码是否通过类型检查
```

DIAGNOSTICS 是 OpenCode 区别于其他终端 AI Agent 的关键能力——Claude Code 和 Aider 都没有 LSP 集成。有了 LSP，AI 能在修改代码后立即检查是否引入了错误，而不是等你手动运行编译器才发现问题。

---

## 工具调用流程

理解 AI 的工具调用流程，能帮你更好地使用 OpenCode。一个典型的多工具调用流程如下：

```
用户："给 /api/users 添加分页功能"

  第 1 轮：AI 分析任务
  → 需要了解当前的路由文件
  → 调用 READ：读取 src/routes/users.ts

  第 2 轮：AI 分析代码
  → 当前接口没有分页参数
  → 需要了解数据库查询是怎么写的
  → 调用 READ：读取 src/db/queries.ts

  第 3 轮：AI 制定修改方案
  → 修改路由文件添加 page 和 limit 参数
  → 修改数据库查询添加 LIMIT 和 OFFSET
  → 调用 WRITE：修改 src/routes/users.ts
  → 调用 WRITE：修改 src/db/queries.ts

  第 4 轮：AI 验证修改
  → 调用 DIAGNOSTICS：检查是否有类型错误
  → 如果有错误，继续修复
  → 如果没有错误，返回最终结果
```

这个过程 AI 自主完成，你只需要输入一句话。但理解这个过程很重要——因为如果 AI 的某一步出了问题（比如 READ 了错误的文件），你可以通过提供更精确的上下文（`@` 引用）来纠正它。

---

## 工具权限控制

OpenCode 允许你在配置文件中控制 Agent 能使用哪些工具。这是安全控制的重要手段：

```json
{
  "agents": {
    "coder": {
      "model": "anthropic/claude-sonnet-4-20250514",
      "maxTokens": 16384,
      "tools": {
        "READ": true,
        "WRITE": true,
        "RUN": true,
        "SEARCH": true,
        "DIAGNOSTICS": true
      }
    }
  }
}
```

**只读模式**：如果你只想让 AI 分析代码但不修改，可以禁用 WRITE：

```json
{
  "agents": {
    "coder": {
      "tools": {
        "READ": true,
        "WRITE": false,
        "RUN": false,
        "SEARCH": true,
        "DIAGNOSTICS": true
      }
    }
  }
}
```

这在代码审查场景很有用——让 AI 看代码、找问题、给建议，但不让它改代码。

**禁用命令执行**：如果你担心 AI 执行危险命令，可以禁用 RUN：

```json
{
  "agents": {
    "coder": {
      "tools": {
        "RUN": false
      }
    }
  }
}
```

---

## 安全模型

OpenCode 的安全模型围绕三个原则设计：

**1. 工作目录限制**

AI 的文件操作（READ/WRITE）和命令执行（RUN）都限制在项目工作目录内。AI 不能读取或修改工作目录之外的文件，不能在工作目录之外执行命令。这防止了 AI 意外修改系统文件或其他项目。

**2. 确认机制**

对于潜在危险的操作（如删除文件、执行 `rm` 命令），OpenCode 会要求你确认。你可以在配置中控制确认级别：

- 所有命令都需要确认
- 只有破坏性命令需要确认
- 不需要确认（信任 AI）

**3. 最小权限原则**

默认情况下，AI 只拥有完成当前任务所需的最小权限。你可以通过工具权限配置进一步限制 AI 的能力——比如禁用 RUN 工具、禁用 WRITE 工具。

---

## MCP 工具扩展

除了内置工具，OpenCode 还通过 MCP 协议支持外部工具。MCP 工具跟内置工具的使用方式一样——AI 会根据任务需要自主决定是否调用 MCP 工具。

比如，如果你配置了 GitHub MCP，AI 就能：

```
AI 使用 MCP 工具的示例：

  - 创建 GitHub Issue："帮我创建一个 Issue 记录这个 bug"
  - 提交 Pull Request："把这些修改提交为 PR"
  - 搜索 GitHub 代码："在 GitHub 上搜索类似的实现"
  - 查看 CI 状态："最近的 CI 跑过了吗？"
```

MCP 工具的详细配置和使用我们会在第 5 章深入讲解。

---

## 常见误区

**误区一：让 AI 直接执行危险命令**

比如 `rm -rf`、`DROP TABLE`、`sudo` 等。虽然 OpenCode 有确认机制，但如果你习惯性地按 Enter 确认，可能会造成不可逆的损失。建议在配置中开启破坏性命令的确认：

```json
{
  "agents": {
    "coder": {
      "confirmDestructive": true
    }
  }
}
```

**误区二：禁用太多工具导致 AI 无能为力**

有些用户出于安全考虑，禁用了 WRITE 和 RUN，只保留 READ。这样 AI 就变成了一个"只看不做的顾问"——它能分析代码、找问题，但不能修改代码、不能运行测试。对于代码审查场景这是合理的，但对于日常编码场景，你需要让 AI 有足够的工具来完成任务。

**误区三：AI 的工具调用总是正确的**

大部分时候 AI 的工具选择是合理的，但不是 100% 可靠。有时候 AI 会在不该修改文件的时候修改文件，或者在不该执行命令的时候执行命令。这就是为什么 Plan 模式和 /undo 命令很重要——它们给你"后悔药"。

**误区四：LSP 诊断是万能的**

LSP 能检测类型错误和 lint 警告，但它不能检测逻辑错误、性能问题、安全漏洞。AI 通过 DIAGNOSTICS 工具获取的信息只是"编译器能看到的问题"，不是"所有的问题"。代码审查仍然需要人工参与。

---

## 小结

这一节我们深入了 OpenCode 的 Agent 工具系统：五个内置工具（READ 读取文件、WRITE 修改文件、RUN 执行命令、SEARCH 搜索代码、DIAGNOSTICS LSP 诊断）、工具调用流程（多轮迭代，AI 自主决策）、工具权限控制（按需启用/禁用）、安全模型（工作目录限制、确认机制、最小权限）。理解工具系统，你就理解了 AI 的能力边界——它能做什么、不能做什么、怎么让它做得更好。下一章我们深入模型选择与策略，帮你根据任务选择最合适的模型。
