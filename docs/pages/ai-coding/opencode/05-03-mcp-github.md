# 5.3 MCP 实战：GitHub 集成

> **配置了 GitHub MCP 后，你可以对 OpenCode 说"帮我创建一个 Issue"——AI 会直接调用 GitHub API 完成操作，你不需要离开终端。**

---

## 这一节在讲什么？

前面两节我们理解了 MCP 的原理和配置方式，这一节我们动手实战——用 GitHub MCP 让 AI 直接操作 GitHub。GitHub MCP 是最常用的 MCP Server 之一，它让 AI 能创建 Issue、提交 PR、搜索代码、查看 CI 状态——所有这些操作都不需要你离开终端，只需要跟 OpenCode 说一句话。这一节我们从配置开始，逐步演示 GitHub MCP 的各种使用场景。

---

## 配置 GitHub MCP

首先，确保你已经安装了 GitHub MCP Server 并配置了 Token：

**第一步：创建 GitHub Token**

1. 访问 [GitHub Settings → Developer settings → Personal access tokens → Fine-grained tokens](https://github.com/settings/personal-access-tokens/new)
2. 创建新 Token，授予以下权限：
   - `Issues`：读写（创建和管理 Issue）
   - `Pull requests`：读写（创建和管理 PR）
   - `Contents`：只读（搜索代码）
   - `Actions`：只读（查看 CI 状态）
3. 复制生成的 Token（以 `ghp_` 开头）

**第二步：设置环境变量**

```bash
export GITHUB_TOKEN="ghp_xxxxxxxxxxxxxxxxxxxx"
```

建议写在 `~/.zshrc` 或 `~/.bashrc` 里，避免每次手动设置。

**第三步：配置 opencode.json**

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

**第四步：验证连接**

启动 OpenCode 后，检查状态栏是否显示 GitHub MCP 已连接：

```
MCP: github ●
```

绿色圆点表示连接成功。如果是红色圆点，检查 Token 是否正确、网络是否正常。

---

## 使用场景一：创建和管理 Issue

配置好 GitHub MCP 后，你可以直接让 AI 创建 Issue：

```
帮我创建一个 Issue，标题是"登录页面在移动端布局错乱"，
描述一下这个问题：在 iPhone 12 上，登录页面的输入框和按钮重叠
```

AI 会调用 GitHub MCP 的 `create_issue` 工具，自动创建 Issue。你可以在对话区看到 AI 的操作结果——Issue 编号、URL 等。

你也可以让 AI 查看和管理 Issue：

```
看看这个仓库有哪些 open 的 Issue？
```

```
把 #42 这个 Issue 分配给我
```

```
给 #42 添加 bug 标签
```

---

## 使用场景二：提交 Pull Request

GitHub MCP 让 AI 能帮你创建 PR：

```
我刚才修改了认证逻辑，帮我创建一个 PR，
标题是"fix: 修复 JWT token 过期后未自动刷新的问题"，
描述一下修改内容
```

AI 会：
1. 用 `$ git diff` 查看当前的代码变更
2. 分析变更内容，生成 PR 描述
3. 调用 GitHub MCP 创建 PR

你也可以让 AI 查看现有的 PR：

```
看看 #15 这个 PR 的 review 评论
```

```
列出所有待 review 的 PR
```

---

## 使用场景三：搜索代码

GitHub MCP 的搜索能力比 OpenCode 内置的 SEARCH 工具更强大——它可以搜索整个 GitHub 上的代码：

```
在 GitHub 上搜索有没有类似的认证实现
```

```
搜索这个仓库里所有使用了 deprecated API 的地方
```

```
查找这个组织下所有使用了 React 18 的仓库
```

---

## 使用场景四：查看 CI/CD 状态

```
最近的 CI 跑过了吗？
```

```
看看 main 分支的 workflow 有没有失败
```

AI 会调用 GitHub MCP 的 Actions 相关工具，查看 CI/CD 的运行状态。如果发现有失败的 workflow，你可以直接让 AI 分析失败原因：

```
CI 失败了，帮我看看日志，找出失败原因
```

---

## GitHub Token 权限最小化

GitHub Token 的权限应该遵循最小权限原则——只授予必要的权限，不要给 Token 所有的权限。

```
权限选择策略：

  只需要查看代码和 Issue：
  → Contents: 只读
  → Issues: 只读

  需要创建 Issue 和 PR：
  → Contents: 只读
  → Issues: 读写
  → Pull requests: 读写

  需要推送代码：
  → Contents: 读写
  （谨慎使用，建议通过 PR 而不是直接推送）

  需要 CI/CD 操作：
  → Actions: 只读
  （通常不需要写权限）
```

---

## 安全注意事项

**1. Token 不要硬编码在配置文件里**

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

**2. 定期轮换 Token**

GitHub Token 应该定期更换——建议每 90 天轮换一次。如果 Token 泄露，立即到 GitHub Settings 中撤销。

**3. 使用 Fine-grained Token 而不是 Classic Token**

Fine-grained Token 可以限制访问的仓库和权限范围，比 Classic Token 更安全。创建 Token 时选择 "Fine-grained tokens"。

**4. 不要让 AI 直接 git push**

虽然 GitHub MCP 可以推送代码，但建议通过 PR 的方式提交修改——先创建 PR，再由人工审查后合并。这样可以在代码进入主分支之前进行审查。

---

## 常见误区

**误区一：GitHub MCP 只能操作公开仓库**

不是。GitHub MCP 可以操作你有权限访问的任何仓库——包括私有仓库。Token 的权限决定了你能操作哪些仓库和执行哪些操作。

**误区二：AI 创建的 Issue/PR 质量不高**

AI 生成的 Issue 和 PR 描述质量取决于你提供的上下文。如果你只说"帮我创建一个 Issue"，AI 只能根据对话历史生成泛泛的描述。如果你提供了详细的问题描述、复现步骤、预期行为，AI 生成的 Issue 质量会很高。

**误区三：GitHub MCP 可以替代 Git 操作**

不能。GitHub MCP 操作的是 GitHub API（Issue、PR、搜索等），不是 Git 本身（commit、push、merge 等）。Git 操作仍然由 OpenCode 的内置 RUN 工具完成。两者是互补的——RUN 处理本地 Git 操作，GitHub MCP 处理远程 GitHub 操作。

**误区四：GitHub MCP 会自动提交代码**

不会。GitHub MCP 不会自动 commit 和 push——这些操作需要你明确指示。AI 可能会建议你提交代码，但不会在没有你确认的情况下执行。

---

## 小结

这一节我们实战了 GitHub MCP：配置 Token 和 MCP Server，用 AI 创建和管理 Issue、提交 Pull Request、搜索代码、查看 CI 状态。GitHub MCP 让 AI 能直接操作 GitHub API，你不需要离开终端就能完成这些操作。安全方面，遵循 Token 权限最小化、环境变量引用、定期轮换、使用 Fine-grained Token 四个原则。下一节我们继续实战——数据库和浏览器自动化 MCP。
