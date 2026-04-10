# 2.1 安装方式全览

> **安装 OpenCode 有七八种方式，选哪种？一句话：官方脚本最省心，Homebrew 最规范，NPM 最通用。**

---

## 这一节在讲什么？

上一章我们快速体验了 OpenCode，用的是最简单的安装方式。但实际工作中，你可能用的是 Arch Linux，可能需要指定版本，可能在 CI/CD 环境里安装，可能在公司内网里安装——不同的场景需要不同的安装方式。这一节我们把所有安装方式过一遍，帮你选到最适合的那一种，同时解决常见的安装问题。

---

## 官方安装脚本（推荐）

这是最简单的安装方式，一条命令搞定，支持 macOS 和 Linux：

```bash
curl -fsSL https://opencode.ai/install | bash
```

这个脚本会自动检测你的操作系统和架构（x86_64 / ARM64），下载对应的二进制文件，安装到 `/usr/local/bin/opencode`。

如果你想安装指定版本，可以通过环境变量指定：

```bash
# 安装指定版本
curl -fsSL https://opencode.ai/install | VERSION=0.4.0 bash
```

安装完成后验证：

```bash
opencode --version
# 输出：opencode version 0.4.x
```

**优点**：最简单，一条命令搞定，自动检测平台
**缺点**：需要网络访问 opencode.ai，某些企业内网可能无法使用

---

## Homebrew 安装（macOS / Linux）

如果你已经装了 Homebrew，这是最规范的方式——版本管理、卸载、升级都很方便：

```bash
# 添加 OpenCode 的 tap 并安装
brew install sst/tap/opencode
```

注意：OpenCode 有两个 Homebrew formula——`sst/tap/opencode` 是官方维护的，更新最快；`opencode` 是 Homebrew 社区维护的，更新可能滞后。推荐用官方 tap。

升级也很简单：

```bash
brew upgrade opencode
```

卸载：

```bash
brew uninstall opencode
```

**优点**：版本管理方便，升级/卸载/回滚一条命令
**缺点**：需要先安装 Homebrew

---

## NPM 安装（跨平台）

如果你有 Node.js 环境，可以用 NPM 安装——这是最跨平台的方式，macOS、Linux、Windows 都支持：

```bash
npm install -g opencode-ai
```

注意包名是 `opencode-ai`，不是 `opencode`——后者是一个完全不相关的 NPM 包。

也可以用其他 Node.js 包管理器：

```bash
# pnpm
pnpm install -g opencode-ai

# yarn
yarn global add opencode-ai

# bun
bun install -g opencode-ai
```

**优点**：跨平台，Windows 也能用
**缺点**：依赖 Node.js 环境，安装速度比二进制方式慢

---

## Go Install（开发者方式）

OpenCode 是用 Go 写的，如果你有 Go 1.22+ 环境，可以直接从源码编译安装：

```bash
go install github.com/sst/opencode@latest
```

这会下载源码、编译、安装到 `$GOPATH/bin/opencode`。

指定版本：

```bash
go install github.com/sst/opencode@v0.4.0
```

**优点**：从源码编译，可以自定义编译选项
**缺点**：编译需要时间，需要 Go 环境

---

## Arch Linux（AUR）

Arch Linux 用户可以通过 AUR 安装：

```bash
# 使用 paru
paru -S opencode-bin

# 使用 yay
yay -S opencode-bin
```

`opencode-bin` 是预编译的二进制包，安装快。如果你想从源码编译，可以用 `opencode` 包（不带 `-bin` 后缀）。

---

## Windows 安装

Windows 的安装方式在不断完善中，目前推荐的方式：

```bash
# Chocolatey
choco install opencode

# Scoop
scoop install opencode

# NPM
npm install -g opencode-ai
```

你也可以直接从 GitHub Releases 页面下载 Windows 二进制文件。

---

## Docker 安装

如果你想在容器里运行 OpenCode：

```bash
docker run -it --rm ghcr.io/anomalyco/opencode
```

注意：Docker 方式需要挂载项目目录和配置文件，否则 OpenCode 无法访问你的代码：

```bash
docker run -it --rm \
  -v $(pwd):/workspace \
  -v ~/.config/opencode:/root/.config/opencode \
  -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
  ghcr.io/anomalyco/opencode
```

---

## 版本管理与升级

无论你用哪种方式安装，升级到最新版本都很重要——OpenCode 更新频繁，新版本通常包含重要的 bug 修复和新功能：

```bash
# 官方脚本安装的升级
curl -fsSL https://opencode.ai/install | bash

# Homebrew 升级
brew upgrade opencode

# NPM 升级
npm update -g opencode-ai

# OpenCode 内置升级命令
opencode upgrade
```

你也可以在 TUI 里检查是否有新版本——OpenCode 会在状态栏提示你升级。

---

## 常见安装问题

**问题一：安装脚本下载失败**

```bash
curl: (7) Failed to connect to opencode.ai
```

这通常是网络问题——你可能需要设置代理：

```bash
export https_proxy=http://your-proxy:port
curl -fsSL https://opencode.ai/install | bash
```

**问题二：权限不足**

```bash
Permission denied: /usr/local/bin/opencode
```

安装脚本需要写入 `/usr/local/bin/`，这个目录通常需要 sudo 权限。你可以用 sudo 运行安装脚本，或者手动指定安装目录：

```bash
# 用 sudo 安装
curl -fsSL https://opencode.ai/install | sudo bash

# 或者安装到用户目录
curl -fsSL https://opencode.ai/install | INSTALL_DIR=$HOME/.local/bin bash
```

**问题三：Go 版本不兼容**

```bash
go: github.com/sst/opencode@latest: module requires Go >= 1.22
```

`go install` 方式需要 Go 1.22 或更高版本。升级 Go 或者换用其他安装方式。

**问题四：macOS 提示"无法验证开发者"**

```bash
"opencode" cannot be opened because the developer cannot be verified.
```

这是 macOS 的安全机制。在"系统设置 → 隐私与安全性"里点击"仍要打开"，或者在终端里执行：

```bash
xattr -d com.apple.quarantine /usr/local/bin/opencode
```

---

## 安装方式选择指南

```
你的情况                          → 推荐安装方式
─────────────────────────────────────────────────
macOS / Linux，想最快上手          → 官方脚本
macOS，已装 Homebrew               → Homebrew
Windows                           → NPM 或 Scoop
Arch Linux                        → AUR
需要指定版本                       → 官方脚本 + VERSION 变量
CI/CD 环境                        → 官方脚本或 NPM
内网环境，无法访问外网              → 下载二进制文件手动安装
想从源码编译                       → Go Install
容器化部署                         → Docker
```

---

## 小结

这一节我们过完了 OpenCode 的所有安装方式。对于大多数开发者，官方脚本 `curl -fsSL https://opencode.ai/install | bash` 是最简单的选择；Homebrew 用户用 `brew install sst/tap/opencode` 更规范；Windows 用户用 NPM 或 Scoop。安装完成后，下一节我们来详细讲解 OpenCode 的配置文件——这是你掌控 OpenCode 行为的核心入口。
