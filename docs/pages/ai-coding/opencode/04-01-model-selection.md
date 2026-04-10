# 4.1 模型选择指南

> **用 Haiku 回答"这个变量什么意思"要 $0.001，用 Opus 回答同样的问题要 $0.06——60 倍的价差，答案质量几乎一样。选对模型，是控制成本的第一步。**

---

## 这一节在讲什么？

OpenCode 支持 75+ 个 LLM 提供商，每个提供商又有多个模型。面对这么多选择，很多用户要么"全部用最贵的"（浪费钱），要么"全部用最便宜的"（质量差），要么"随便选一个不动了"（没发挥 OpenCode 的多模型优势）。这一节我们要建立一个清晰的模型选择框架——按任务类型选模型、按成本优化策略、按场景动态切换。掌握这个框架后，你能在保证质量的前提下，把 API 成本降低 50% 以上。

---

## 按任务类型选模型

不同的任务对模型能力的要求差异巨大。一个简单的变量重命名，Haiku 就能搞定；一个复杂的架构重构，需要 Sonnet 甚至 Opus。让我们按任务复杂度来分级：

### 简单任务 → 用 fast 模型

简单任务的特点是：目标明确、上下文需求少、不需要复杂推理。

```
典型简单任务：
- "这个函数是做什么的？"
- "把变量名 userId 改成 accountId"
- "这段代码有什么 lint 错误？"
- "给这个接口加个注释"
- "这个文件有多少行代码？"

推荐模型：
- Claude Haiku 4（$0.25/M input, $1.25/M output）
- GPT-4o-mini（$0.15/M input, $0.60/M output）
- Groq Llama 3.3（免费额度，速度极快）
```

### 日常编码 → 用 default 模型

日常编码任务的特点是：需要理解上下文、生成代码、修改文件、运行测试。

```
典型日常任务：
- "给这个接口添加分页功能"
- "修复这个 TypeScript 类型错误"
- "重构这个函数，提取公共逻辑"
- "给这个组件添加 loading 状态"

推荐模型：
- Claude Sonnet 4（$3/M input, $15/M output）——综合最佳
- GPT-4o（$2.50/M input, $10/M output）——备选
- DeepSeek Chat（$0.27/M input, $1.10/M output）——高性价比
```

### 复杂任务 → 用 big 模型

复杂任务的特点是：需要深度推理、跨文件理解、架构级决策。

```
典型复杂任务：
- "设计一个支持多租户的数据隔离方案"
- "重构整个认证系统，从 Session 迁移到 JWT"
- "分析这个性能瓶颈的根本原因"
- "设计一个可扩展的插件架构"

推荐模型：
- Claude Opus 4（$15/M input, $75/M output）——最强推理
- o3（$15/M input, $60/M output）——推理能力强
- Gemini 2.5 Pro（$1.25/M input, $10/M output）——超长上下文
```

### 特殊场景 → 用特定模型

```
隐私敏感代码：
→ Ollama 本地模型（数据不出本机）

需要理解整个项目：
→ Gemini 2.5 Pro（100 万 token 上下文）

追求极致响应速度：
→ Groq Llama 3.3（推理延迟 < 100ms）

已有 Copilot 订阅：
→ GitHub Copilot 模型（零额外成本）
```

---

## 模型配置：三档模型策略

OpenCode 支持在配置文件中设置三档模型——default、big、fast——让你根据任务复杂度快速切换：

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

在 TUI 中，你可以用 `/models` 命令随时切换模型：

```
/models
```

这会打开一个模型选择界面，列出所有可用的模型。你可以根据当前任务的复杂度选择合适的模型。

**启动时指定模型**：

```bash
# 用 Claude Sonnet 启动
opencode -m anthropic/claude-sonnet-4-20250514

# 用 DeepSeek 启动
opencode -m deepseek/deepseek-chat

# 用 Ollama 本地模型启动
opencode -m ollama/deepseek-coder:6.7b
```

**非交互模式指定模型**：

```bash
# 用 GPT-4o 执行任务
opencode run -m openai/gpt-4o "Review the code in src/api/"

# 用 Gemini 分析整个项目
opencode run -m google/gemini-2.5-pro "Analyze the architecture of this project"
```

---

## 成本优化策略

API 调用成本是使用 OpenCode 的主要支出。以下是几个实用的成本优化策略：

**策略一：简单任务用便宜模型**

这是最直接的省钱方式。一个简单问答用 Haiku 只需要 $0.001，用 Sonnet 需要 $0.01，用 Opus 需要 $0.06——同样的答案，60 倍的价差。

**策略二：先 Plan 再 Build**

Plan 模式下 AI 不调用 WRITE 和 RUN 工具，token 消耗比 Build 模式少 30%~50%。先用 Plan 模式确认 AI 的理解正确，再切 Build 模式执行——这比盲目 Build 后 /undo 更省钱。

**策略三：精准提供上下文**

上下文越长，API 费用越高（按 token 计费）。精准引用文件（`@filename`）比让 AI 自己搜索更省 token——因为 AI 搜索时会读取多个文件，产生大量中间 token。

**策略四：用 /cost 监控费用**

```
/cost
```

定期查看当前会话的 token 使用量，了解你的消费模式。如果你发现某个会话特别贵，分析一下是哪个环节消耗了大量 token——可能是上下文太长，或者 AI 进行了太多的工具调用。

**策略五：非交互模式用便宜模型**

`opencode run` 的非交互模式通常处理的是简单任务（代码审查、格式化、简单修改），用便宜模型就够了：

```bash
# 代码审查用 Haiku
opencode run -m anthropic/claude-haiku-4-20250414 "Review @src/api/auth.ts"

# 简单修改用 DeepSeek
opencode run -m deepseek/deepseek-chat "Add JSDoc comments to @src/utils.ts"
```

---

## 模型质量对比

以下是主流模型在编程任务上的表现对比（基于社区评测和实际使用经验）：

| 模型 | 代码生成 | 调试能力 | 架构设计 | 长上下文 | 速度 | 成本 |
|------|---------|---------|---------|---------|------|------|
| Claude Sonnet 4 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 中 |
| Claude Opus 4 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | 高 |
| Claude Haiku 4 | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 低 |
| GPT-4o | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 中 |
| Gemini 2.5 Pro | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 中 |
| DeepSeek Chat | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | 极低 |
| Ollama 7B | ⭐⭐ | ⭐⭐ | ⭐ | ⭐⭐ | ⭐⭐⭐⭐ | 免费 |

---

## 常见误区

**误区一：所有任务都用最贵的模型**

这是最常见的浪费。简单问答用 Opus 跟用 Haiku 的答案质量几乎一样，但价格差 60 倍。养成习惯：简单任务用 fast 模型，日常编码用 default 模型，复杂任务才用 big 模型。

**误区二：便宜模型一定差**

不一定。DeepSeek Chat 在代码生成方面的表现接近 GPT-4o，但价格只有其 1/10。Claude Haiku 在简单任务上的表现也足够好。便宜模型"差"的是复杂推理能力，不是所有能力。

**误区三：模型切换很麻烦**

不麻烦。在 TUI 里输入 `/models` 就能切换，或者用 `-m` 参数在启动时指定。OpenCode 的多模型支持就是为了让切换变得简单——你不需要为不同的模型安装不同的工具。

**误区四：本地模型完全不能用**

Ollama 的 7B 模型在简单任务上（代码补全、简单修改、快速问答）是可以用的，只是复杂任务表现不佳。如果你非常在意隐私，日常用本地模型 + 复杂任务切云端，是一个合理的策略。

---

## 小结

这一节我们建立了模型选择的框架：按任务复杂度选模型（简单→fast、日常→default、复杂→big），按成本优化策略（简单任务用便宜模型、先 Plan 再 Build、精准提供上下文），按特殊场景选特定模型（隐私→Ollama、长上下文→Gemini、速度→Groq）。核心原则是"按需选择"——不是所有任务都需要最强的模型，选对模型能在保证质量的前提下大幅降低成本。下一节我们深入 Ollama 本地模型的配置和使用。
