# 6.3 Git 工作流集成

> **OpenCode 不是 Git 客户端，但它能让 AI 帮你审查代码、生成 commit message、解决合并冲突——Git 工作流中的"脏活累活"交给 AI。**

---

## 这一节在讲什么？

Git 是开发者的日常工具，但 Git 工作流中有很多"脏活累活"——写 commit message、审查代码变更、解决合并冲突、生成 PR 描述。这些任务不需要高深的编程能力，但需要仔细阅读代码和理解上下文——这正是 AI 擅长的。这一节我们学习怎么用 OpenCode 辅助 Git 工作流，让 AI 帮你处理这些繁琐但重要的任务。

---

## 代码审查：让 AI 审查你的变更

代码审查是保证代码质量的重要环节，但人工审查耗时且容易遗漏。OpenCode 可以作为"第一道审查"——先让 AI 审查，再由人工复审。

**审查当前变更**：

```
$ git diff 的改动看起来对吗？有没有潜在的问题？
```

AI 会：
1. 执行 `git diff` 获取当前变更
2. 逐文件分析变更内容
3. 检查潜在问题：安全漏洞、性能问题、逻辑错误、代码风格不一致
4. 给出审查意见

**审查特定文件**：

```
审查一下 @src/api/auth.ts 的最近修改，重点关注安全性
```

**审查某个 commit**：

```
$ git show HEAD 的改动有没有问题？
```

**AI 代码审查的典型输出**：

```
审查结果：

✅ 正确的修改：
- JWT token 过期时间的配置从硬编码改为环境变量
- 添加了 refresh token 的轮换逻辑

⚠️ 潜在问题：
1. src/api/auth.ts 第 45 行：密码比较使用了 === 而不是 bcrypt.compare，
   这是时序攻击漏洞，应该用恒定时间比较
2. src/api/auth.ts 第 78 行：错误信息 "Invalid credentials" 太笼统，
   无法区分是用户名错误还是密码错误（但注意：不要在错误信息中泄露
   具体是哪个错误，这是安全最佳实践）

💡 建议：
- 考虑添加 rate limiting 防止暴力破解
- token 过期时间建议从环境变量读取，而不是硬编码 24h
```

---

## 提交信息生成：让 AI 写 commit message

写好的 commit message 是一门艺术——它需要简洁、准确、遵循团队规范。AI 可以根据代码变更自动生成规范的 commit message。

**生成 commit message**：

```
$ git diff 的改动，帮我生成一个 commit message，遵循 Conventional Commits 规范
```

AI 会分析变更内容，生成类似这样的 commit message：

```
feat(auth): add refresh token rotation

- Add refresh token endpoint at POST /auth/refresh
- Implement token rotation with 7-day expiry
- Store refresh tokens in database with user association
- Add automatic cleanup of expired tokens
```

**生成更详细的 commit message**：

```
帮我生成一个详细的 commit message，包含变更原因和影响范围
```

```
feat(auth): add refresh token rotation to improve security

Problem: Access tokens with 24h expiry pose a security risk if
compromised. Users had to re-authenticate frequently.

Solution: Implement refresh token rotation (RFC 6819 section 5.2.2).
Refresh tokens are single-use and rotated on each refresh request.

Changes:
- Add POST /auth/refresh endpoint
- Add RefreshToken model in Prisma schema
- Add token rotation logic in auth.service.ts
- Add cleanup cron job for expired tokens

Impact: Users stay logged in longer with better security.
Breaking change: None, existing auth flow unchanged.
```

---

## 分支管理

OpenCode 的 AI 可以帮你管理 Git 分支——虽然它不是 Git 客户端，但它能通过 RUN 工具执行 Git 命令。

**创建分支**：

```
帮我创建一个新分支 feature/user-notifications，基于 main 分支
```

**查看分支状态**：

```
$ git branch -a 列出所有分支，看看有没有需要清理的
```

**合并冲突解决**：

这是 Git 工作流中最让人头疼的问题。OpenCode 可以帮你分析冲突并建议解决方案：

```
合并 main 分支到 feature/user-notifications 时有冲突，
帮我看看冲突文件 @src/routes/index.ts 和 @src/services/user.service.ts，
分析一下该怎么解决
```

AI 会：
1. 读取冲突文件，分析冲突标记（`<<<<<<<`、`=======`、`>>>>>>>`）
2. 理解两边的修改意图
3. 建议合并方案——保留哪边的修改，或者如何合并两边的修改
4. 执行合并并验证

**注意**：合并冲突的解决需要非常谨慎——AI 可能不理解业务逻辑的细微差别。建议让 AI 先分析冲突，给出建议，你审查后再让它执行。

---

## PR 描述生成

如果你配置了 GitHub MCP，AI 可以帮你创建 PR 并生成描述：

```
帮我创建一个 PR，标题和描述根据我的修改自动生成
```

AI 会：
1. 用 `$ git diff main...HEAD` 查看当前分支的所有变更
2. 用 `$ git log main...HEAD --oneline` 查看 commit 历史
3. 分析变更内容，生成 PR 标题和描述
4. 调用 GitHub MCP 创建 PR

生成的 PR 描述通常包含：
- 变更摘要
- 修改的文件列表
- 测试说明
- 截图（如果是前端修改）
- Breaking changes（如果有）

---

## Git 工作流的最佳实践

**1. 让 AI 审查代码，但不要完全依赖**

AI 的代码审查是"第一道防线"，但不能替代人工审查。AI 擅长发现安全漏洞、性能问题和代码风格问题，但对业务逻辑的理解不如人类。最佳实践：AI 先审 → 人工复审。

**2. commit message 生成后要审查**

AI 生成的 commit message 大部分时候是好的，但可能遗漏重要的变更或误解修改意图。审查 AI 生成的 commit message，确保它准确描述了你的修改。

**3. 合并冲突让 AI 分析但不直接解决**

合并冲突涉及业务逻辑的选择——AI 可能不理解为什么要保留某段代码。让 AI 分析冲突并给出建议，但最终决策由你来做。

**4. 不要让 AI 直接 git push**

```bash
# ❌ 危险：让 AI 直接推送
"帮我把代码 push 到远程仓库"

# ✅ 安全：让 AI 创建 PR，你审查后合并
"帮我创建一个 PR"
```

直接 push 意味着代码直接进入远程仓库，没有审查环节。通过 PR 的方式，代码在合并前有审查的机会。

---

## 常见误区

**误区一：让 AI 直接 git push**

这是最常见的危险操作。AI 可能推送了不完整或有 bug 的代码，而且 git push 是不可逆的（虽然可以 revert，但会留下混乱的提交历史）。建议通过 PR 的方式提交代码——先创建 PR，审查后再合并。

**误区二：AI 的代码审查能替代人工审查**

不能。AI 的审查是"模式匹配"——它基于训练数据中的常见问题模式来发现潜在问题。但业务逻辑的正确性、用户体验的合理性、架构决策的适当性，这些需要人类来判断。AI 审查 + 人工审查，才是完整的代码审查流程。

**误区三：commit message 不重要，AI 生成就行**

commit message 是代码历史的"地图"——好的 commit message 让你能快速定位问题引入的时间点，理解每次修改的原因。AI 生成的 commit message 通常质量不错，但你仍然应该审查它是否准确描述了你的修改。

**误区四：AI 能解决所有合并冲突**

简单的文本冲突 AI 确实能解决——比如两个人修改了同一个文件的不同部分。但涉及业务逻辑选择的冲突（比如两个人对同一个函数有不同的实现方案），AI 无法判断哪个方案更正确。这种冲突需要人类根据业务需求来决定。

---

## 小结

这一节我们学习了 OpenCode 在 Git 工作流中的应用：代码审查（`$ git diff` + AI 分析）、commit message 生成（遵循 Conventional Commits 规范）、分支管理（创建分支、解决冲突）、PR 描述生成（配合 GitHub MCP）。核心原则是"AI 辅助，人工决策"——让 AI 处理繁琐的文本工作（审查、生成描述），但关键决策（push 代码、解决业务冲突、最终审查）由人类来做。下一章我们进入进阶技巧。
