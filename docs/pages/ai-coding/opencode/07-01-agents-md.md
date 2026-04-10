# 7.1 AGENTS.md 与项目定制

> **AGENTS.md 是你给 AI 写的"项目说明书"——写得好，AI 就像入职三年的老员工；写得差，AI 就像第一天上班的实习生。**

---

## 这一节在讲什么？

你有没有遇到过这种情况——每次跟 OpenCode 对话，都要先解释一遍"我们项目用的是 Express + TypeScript，路由在 src/routes/ 下，测试用 Jest..."？AGENTS.md 就是解决这个问题的——它是项目的"AI 说明书"，告诉 AI 项目的架构、规范、约定、常用命令。有了 AGENTS.md，AI 不需要你每次重复解释项目背景，就能给出符合项目规范的回答。这一节我们学习 AGENTS.md 的作用、自动生成方式、手动定制技巧和最佳实践。

---

## AGENTS.md 的作用

AGENTS.md 是一个 Markdown 文件，放在项目根目录下。当 OpenCode 启动时，它会自动读取 AGENTS.md 的内容，作为 AI 的"项目上下文"。

```
没有 AGENTS.md 的对话：

  你："帮我添加一个新接口"
  AI："好的，请问你用的是什么框架？路由文件在哪里？代码风格是什么？"

有 AGENTS.md 的对话：

  你："帮我添加一个新接口"
  AI："好的，根据项目结构，我会在 src/routes/ 下创建新路由，
      使用 Express Router，遵循现有的错误处理模式..."
```

AGENTS.md 的核心价值是**减少重复解释**——你不需要每次对话都告诉 AI 项目的基本信息，AI 已经"知道"了。

---

## /init 命令：自动生成 AGENTS.md

OpenCode 提供了 `/init` 命令，自动分析项目并生成 AGENTS.md：

```
/init
```

`/init` 会扫描项目目录，识别：
- 项目语言和框架
- 目录结构
- 依赖列表
- 构建和测试命令
- 代码风格配置（ESLint、Prettier 等）

生成的 AGENTS.md 示例：

```markdown
# Project: my-web-app

## Overview
This is a Next.js application using TypeScript and Tailwind CSS.

## Structure
- `src/app/` - Next.js App Router pages
- `src/components/` - React components
- `src/lib/` - Utility functions
- `prisma/` - Database schema and migrations

## Commands
- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run test` - Run tests
- `npm run lint` - Run linter

## Conventions
- Use TypeScript for all new files
- Follow existing component patterns in src/components/
- Use Tailwind CSS for styling
```

这个自动生成的文件已经包含了基本信息，但通常需要手动补充更多细节。

---

## 手动定制 AGENTS.md

自动生成的 AGENTS.md 是"骨架"，你需要手动补充"血肉"。以下是一个更完整的 AGENTS.md 模板：

```markdown
# Project: my-web-app

## Overview
This is a Next.js 14 application using TypeScript, Tailwind CSS, and Prisma.
The app is a task management tool with multi-tenant support.

## Architecture
- Frontend: Next.js App Router + React Server Components
- Backend: Next.js API Routes + Prisma ORM
- Database: PostgreSQL (via Prisma)
- Auth: NextAuth.js with JWT strategy
- Deployment: Vercel

## Directory Structure
- `src/app/` - Next.js App Router pages and layouts
- `src/app/api/` - API route handlers
- `src/components/` - React components (use shadcn/ui)
- `src/lib/` - Utility functions and shared logic
- `src/lib/auth.ts` - Authentication configuration
- `src/lib/db.ts` - Prisma client singleton
- `prisma/` - Database schema and migrations

## Commands
- `npm run dev` - Start development server (port 3000)
- `npm run build` - Build for production
- `npm run test` - Run tests with Jest
- `npm run lint` - Run ESLint
- `npx prisma migrate dev` - Run database migration
- `npx prisma studio` - Open Prisma Studio

## Code Conventions
- Use TypeScript for all new files, no .js files
- Use `async/await` for all async operations, no `.then()`
- Error handling: use try-catch with custom AppError class
- API responses: always return `{ data, error }` format
- Components: use shadcn/ui components, don't create custom UI primitives
- Styling: use Tailwind CSS, no inline styles or CSS modules
- Imports: use `@/` alias for src/ imports

## Database Conventions
- All tables must have `id`, `createdAt`, `updatedAt` fields
- Use UUID for primary keys
- Soft delete: use `deletedAt` field, never hard delete
- Multi-tenancy: all queries must include `where: { tenantId }`

## Testing Conventions
- Unit tests: co-locate with source file (e.g., `auth.ts` → `auth.test.ts`)
- API tests: use supertest in `tests/api/` directory
- Always mock external services in tests
- Minimum coverage: 80% for new code

## Common Pitfalls
- Don't forget to add `tenantId` filter in database queries
- Don't use `getServerSession()` in client components
- Don't import server-only code in client components
- Always run `npx prisma generate` after schema changes
```

这个模板包含了 `/init` 不会自动生成的关键信息：
- **架构说明**：让 AI 知道技术栈和架构决策
- **代码规范**：让 AI 遵循项目的编码风格
- **数据库规范**：让 AI 按照项目的数据约定操作
- **测试规范**：让 AI 按照项目的测试策略写测试
- **常见陷阱**：让 AI 避免项目中的常见错误

---

## AGENTS.md 的最佳实践

**1. 结构清晰，重点突出**

AGENTS.md 不是越长越好——AI 的注意力有限，太长的文件会让 AI 在大量信息中找不到重点。建议控制在 100 行以内，只写最关键的信息。

```
AGENTS.md 的优先级：

  必须写（高优先级）：
  - 项目架构和技术栈
  - 目录结构
  - 常用命令
  - 代码规范（最关键的 3-5 条）

  建议写（中优先级）：
  - 数据库规范
  - 测试规范
  - 常见陷阱

  可选写（低优先级）：
  - 详细的 API 文档
  - 部署流程
  - 团队成员列表
```

**2. 定期更新**

项目在演进，AGENTS.md 也应该跟着更新。当你添加了新的技术栈、修改了目录结构、调整了代码规范时，记得更新 AGENTS.md。过时的 AGENTS.md 比没有更危险——它会让 AI 基于错误的信息做决策。

**3. 提交到 Git**

AGENTS.md 应该提交到 Git 仓库——这样团队成员都能共享同一份项目说明。当你更新 AGENTS.md 时，commit message 应该说明更新了什么：

```
docs: update AGENTS.md with new testing conventions
```

**4. 写"不要做什么"比写"要做什么"更有效**

AI 经常犯的错误是"做了不该做的事"——比如在客户端组件里用了服务端 API、在查询中忘了加 tenantId 过滤。把这些"不要做"的事情写在 AGENTS.md 里，比写"要做"的事情更能避免错误：

```markdown
## Common Pitfalls
- DON'T use getServerSession() in client components
- DON'T forget tenantId filter in database queries
- DON'T hard-delete records, use soft delete
- DON'T create custom UI primitives, use shadcn/ui
```

---

## 常见误区

**误区一：AGENTS.md 写得太长**

AI 的上下文窗口是有限的，AGENTS.md 太长会占用大量上下文空间，留给对话和文件内容的空间就少了。建议控制在 100 行以内，只写最关键的信息。如果你有很多规范要写，考虑把它们拆分成多个文件，只在 AGENTS.md 中引用关键部分。

**误区二：/init 生成的 AGENTS.md 不需要修改**

`/init` 生成的 AGENTS.md 只包含基本信息——项目结构、命令、框架。它不包含代码规范、数据库约定、常见陷阱等关键信息。你需要手动补充这些内容，才能让 AI 真正"理解"你的项目。

**误区三：AGENTS.md 只需要写一次**

项目在演进，AGENTS.md 也应该跟着更新。当你添加了新的技术栈、修改了目录结构、调整了代码规范时，记得更新 AGENTS.md。建议每次 sprint 结束后检查一下 AGENTS.md 是否需要更新。

**误区四：AGENTS.md 里写 API Key 和密码**

绝对不要。AGENTS.md 提交到 Git 后，所有有仓库访问权限的人都能看到。API Key、密码、连接字符串等敏感信息应该用环境变量配置，不要写在 AGENTS.md 里。

---

## 小结

这一节我们学习了 AGENTS.md——项目的"AI 说明书"：`/init` 命令自动生成基础版本，手动补充架构说明、代码规范、数据库规范、常见陷阱等关键信息。AGENTS.md 的最佳实践是"结构清晰、重点突出、定期更新、提交到 Git"。写好 AGENTS.md，AI 就像"入职三年的老员工"——不需要你每次解释项目背景，就能给出符合项目规范的回答。下一节我们学习自定义命令与工作流自动化。
