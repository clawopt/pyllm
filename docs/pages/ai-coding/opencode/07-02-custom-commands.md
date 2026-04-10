# 7.2 自定义命令与工作流

> **重复的操作交给自定义命令——一行命令完成"审查代码 → 生成 commit → 创建 PR"的完整流程。**

---

## 这一节在讲什么？

你每天可能要做很多重复操作——审查代码变更、生成 commit message、创建 PR、运行测试并修复错误。这些操作每次都要输入一大段提示词，既繁琐又容易遗漏步骤。OpenCode 的自定义命令和工作流自动化功能，让你把这些重复操作封装成一行命令——输入 `/review` 就能完成代码审查，输入 `/deploy-check` 就能检查部署前的所有事项。这一节我们学习自定义斜杠命令、命名参数、CLI 非交互模式和 CI/CD 集成。

---

## 自定义斜杠命令

OpenCode 允许你在配置文件中定义自定义斜杠命令——每个命令是一个预定义的提示词模板。

### 配置方式

在 opencode.json 中添加 `commands` 字段：

```json
{
  "commands": {
    "review": {
      "description": "Review current code changes",
      "prompt": "Review the current git diff. Focus on: 1) Security issues 2) Performance problems 3) Logic errors 4) Code style consistency. Format your review as a checklist."
    },
    "fix-lint": {
      "description": "Fix all lint errors",
      "prompt": "Run the linter, analyze all errors, and fix them one by one. After fixing, run the linter again to confirm all errors are resolved."
    },
    "test-and-fix": {
      "description": "Run tests and fix failures",
      "prompt": "Run the test suite. If any tests fail, analyze the failures, fix the code, and run the tests again until all pass."
    }
  }
}
```

使用时在 TUI 中输入 `/review`、`/fix-lint`、`/test-and-fix` 即可。

### 常用自定义命令示例

**代码审查命令**：

```json
{
  "review": {
    "description": "Review code changes for the current branch",
    "prompt": "Review all changes in the current branch compared to main. Use $ git diff main...HEAD to see the changes. Focus on security, performance, and correctness. Generate a summary and a detailed review."
  }
}
```

**部署前检查命令**：

```json
{
  "deploy-check": {
    "description": "Pre-deployment checklist",
    "prompt": "Run the following checks and report results: 1) Run linter 2) Run type check 3) Run tests 4) Check for TODO/FIXME comments 5) Check for console.log statements 6) Verify environment variables are documented"
  }
}
```

**文档生成命令**：

```json
{
  "doc": {
    "description": "Generate documentation for a file",
    "prompt": "Read the file specified by the user and generate comprehensive JSDoc/TSDoc comments for all exported functions and classes. Follow the existing documentation style in the project."
  }
}
```

---

## 命名参数

自定义命令支持命名参数——用 `$参数名` 的方式在提示词模板中引用参数：

```json
{
  "fix-issue": {
    "description": "Fix a GitHub issue",
    "prompt": "Look at GitHub issue #$issue_number. Understand the problem, create a plan, and implement the fix. After fixing, create a PR that references the issue."
  }
}
```

使用时传入参数值：

```
/fix-issue issue_number=42
```

OpenCode 会把 `$issue_number` 替换为 `42`，最终发送给 AI 的提示词是：

```
Look at GitHub issue #42. Understand the problem, create a plan,
and implement the fix. After fixing, create a PR that references the issue.
```

命名参数让自定义命令更灵活——同一个命令模板可以处理不同的输入。

---

## CLI 非交互模式

OpenCode 的 `opencode run` 命令支持非交互模式——你可以在命令行中直接指定提示词，AI 执行完后返回结果。这在脚本和自动化场景中非常有用。

**基本用法**：

```bash
opencode run "Review the code in src/api/auth.ts"
```

**指定模型**：

```bash
opencode run -m anthropic/claude-haiku-4-20250414 "Add JSDoc comments to @src/utils.ts"
```

**附加文件**：

```bash
opencode run -f src/main.ts -f package.json "Analyze this project's dependencies"
```

**JSON 输出**（适合脚本处理）：

```bash
opencode run --format json "List all TypeScript files in the project"
```

**继续上次会话**：

```bash
opencode run -c "What else needs to be done?"
```

**从 stdin 读取输入**：

```bash
echo "Count lines of code" | opencode run "Analyze"
```

---

## 工作流自动化

结合 `opencode run` 的非交互模式，你可以创建各种自动化工作流：

### Git Pre-commit Hook

在每次 commit 前自动运行代码审查：

```bash
# .git/hooks/pre-commit
#!/bin/bash

echo "Running AI code review..."
opencode run -m anthropic/claude-haiku-4-20250414 "Review the staged changes. Only report critical issues." 

if [ $? -ne 0 ]; then
  echo "AI review found critical issues. Please fix before committing."
  exit 1
fi
```

### 每日代码质量报告

```bash
# daily-report.sh
#!/bin/bash

opencode run -m anthropic/claude-haiku-4-20250414 \
  --title "Daily Code Quality Report" \
  "Analyze the codebase and generate a report covering:
  1. Number of TODO/FIXME comments
  2. Test coverage summary
  3. Dependency updates available
  4. Potential security issues
  Format as a markdown report."
```

### PR 自动审查

```bash
# pr-review.sh
#!/bin/bash

PR_NUMBER=$1
opencode run \
  -m anthropic/claude-sonnet-4-20250514 \
  --title "PR #$PR_NUMBER Review" \
  "Review the changes in PR #$PR_NUMBER. Focus on:
  1. Code correctness
  2. Security issues
  3. Performance implications
  4. Test coverage
  Generate a detailed review comment."
```

---

## CI/CD 集成

`opencode run` 可以在 CI/CD pipeline 中使用，实现自动化的代码审查和质量检查：

### GitHub Actions 示例

```yaml
name: AI Code Review
on:
  pull_request:
    types: [opened, synchronize]

jobs:
  ai-review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Install OpenCode
        run: curl -fsSL https://opencode.ai/install | bash

      - name: AI Review
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        run: |
          opencode run \
            -m anthropic/claude-sonnet-4-20250514 \
            "Review the changes in this PR. Focus on security and correctness." \
            >> review-comment.md

      - name: Post Review Comment
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            const comment = fs.readFileSync('review-comment.md', 'utf8');
            github.rest.issues.createComment({
              owner: context.repo.owner,
              repo: context.repo.repo,
              issue_number: context.issue.number,
              body: comment
            });
```

---

## 常见误区

**误区一：自定义命令太复杂**

自定义命令应该是简单明确的——一个命令做一件事。如果你把"审查代码 + 修复问题 + 运行测试 + 创建 PR"全部塞进一个命令，AI 很可能在某个步骤出错，导致后续步骤全部失败。更好的做法是拆分成多个简单命令，按需组合使用。

**误区二：非交互模式能替代 TUI**

不能。`opencode run` 适合一次性任务（代码审查、格式化、简单修改），但对于需要多轮对话的复杂任务（架构设计、大型重构），TUI 的交互体验更好。选择哪种模式取决于任务的性质。

**误区三：CI/CD 中用最贵的模型**

CI/CD 中的代码审查通常是批量运行的——每个 PR 都触发一次。如果用 Claude Opus，每次审查可能花费 $0.5~$1，一天 10 个 PR 就是 $5~$10。建议在 CI/CD 中用 Haiku 或 Sonnet——审查质量足够，成本大幅降低。

**误区四：自定义命令不需要维护**

项目在演进，自定义命令也需要更新。当你修改了代码规范、添加了新的检查步骤、调整了工作流时，记得更新对应的自定义命令。过时的命令比没有命令更危险——它会让 AI 基于过时的规范做决策。

---

## 小结

这一节我们学习了 OpenCode 的自定义命令和工作流自动化：自定义斜杠命令封装重复操作，命名参数让命令更灵活，`opencode run` 的非交互模式适合脚本和 CI/CD 集成。自定义命令的核心原则是"简单明确"——一个命令做一件事，复杂工作流通过组合多个简单命令实现。下一节我们学习多会话与协作技巧。
