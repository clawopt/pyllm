# 技能管理命令

安装了技能之后，你需要知道如何管理它们。这一章介绍OpenClaw的技能管理命令。

## 安装技能

**从ClawHub安装：**

```bash
# 基本安装
openclaw hub install <skill_name>

# 安装指定版本
openclaw hub install <skill_name>@1.2.0

# 安装到项目目录
openclaw hub install <skill_name> --scope project

# 安装到用户目录
openclaw hub install <skill_name> --scope user
```

**从本地目录安装：**

```bash
# 安装本地技能
openclaw skill install ./path/to/skill/

# 从Git仓库安装
openclaw skill install https://github.com/user/skill-name.git
```

**批量安装：**

```bash
# 从配置文件安装
openclaw hub install --file skills.txt

# skills.txt 内容
agent-browser
file_ops
web_search
summarizer
```

**安装示例：**

```bash
$ openclaw hub install agent-browser

Downloading agent-browser@1.5.2...
████████████████████████████████████ 100%

Verifying package signature... ✓
Checking dependencies... ✓
Installing to ~/.openclaw/skills/agent-browser/... ✓

✓ agent-browser@1.5.2 installed successfully

Documentation: https://hub.openclaw.ai/skills/agent-browser
```

## 查看已安装技能

**列出所有技能：**

```bash
openclaw hub list
```

**输出示例：**

```
已安装技能（共7个）：

┌─────────────────┬─────────┬──────────┬─────────────────────┐
│ 技能名称        │ 版本    │ 来源     │ 描述                │
├─────────────────┼─────────┼──────────┼─────────────────────┤
│ 🟢 agent-browser│ 1.5.2   │ ClawHub  │ 浏览器自动化        │
│ 🟢 file_ops     │ 2.1.0   │ Built-in │ 文件操作            │
│ 🟢 web_search   │ 1.3.0   │ ClawHub  │ 联网搜索            │
│ 🟡 image_gen    │ 0.9.5   │ ClawHub  │ 图像生成            │
│ 🟡 email        │ 1.0.0   │ ClawHub  │ 邮件发送            │
│ 🟡 summarizer   │ 1.2.0   │ ClawHub  │ 内容摘要            │
│ ⚪ ontology     │ 0.5.0   │ ClawHub  │ 结构化记忆          │
└─────────────────┴─────────┴──────────┴─────────────────────┘

🟢 官方认证  🟡 社区验证  ⚪ 新发布
```

**查看技能详情：**

```bash
openclaw hub show agent-browser
```

**输出示例：**

```
技能详情：agent-browser
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

名称：agent-browser
版本：1.5.2
作者：OpenClaw Team
许可：MIT
评分：⭐ 4.9 (1,234 评价)
下载量：89,234

描述：
官方浏览器自动化技能，支持网页导航、元素操作、内容提取、截图等功能。

权限要求：
  • network:access (网络访问)
  • file:write (文件写入，用于截图保存)
  • process:spawn (进程创建，用于启动浏览器)

依赖：
  • playwright >= 1.40.0

安装时间：2026-03-28 10:30:00
安装位置：~/.openclaw/skills/agent-browser/

文档：https://hub.openclaw.ai/skills/agent-browser
仓库：https://github.com/openclaw/skill-browser
```

## 技能搜索与筛选

**搜索技能：**

```bash
# 关键词搜索
openclaw hub search <keyword>

# 按分类搜索
openclaw hub search --category <category>

# 按标签搜索
openclaw hub search --tag <tag>
```

**搜索选项：**

```bash
# 按评分排序
openclaw hub search "browser" --sort rating

# 按下载量排序
openclaw hub search "browser" --sort downloads

# 按更新时间排序
openclaw hub search "browser" --sort updated

# 限制结果数量
openclaw hub search "browser" --limit 5

# 显示详细信息
openclaw hub search "browser" --verbose
```

**搜索示例：**

```bash
$ openclaw hub search "email" --sort rating --limit 5

搜索 "email" 的结果：

┌─────────────────┬─────────┬──────────┬─────────────────────┐
│ 技能名称        │ 评分    │ 下载量   │ 描述                │
├─────────────────┼─────────┼──────────┼─────────────────────┤
│ 🟢 email        │ ⭐ 4.8  │ 34,567   │ 官方邮件发送技能    │
│ 🟡 gmail_client │ ⭐ 4.6  │ 12,345   │ Gmail专用客户端     │
│ 🟡 mail_template│ ⭐ 4.5  │ 8,901    │ 邮件模板管理        │
│ ⚪ email_parser │ ⭐ 4.3  │ 2,345    │ 邮件内容解析        │
│ ⚪ newsletter   │ ⭐ 4.1  │ 1,234    │ 订阅邮件发送        │
└─────────────────┴─────────┴──────────┴─────────────────────┘
```

**查看热门技能：**

```bash
openclaw hub trending
```

**输出示例：**

```
本周热门技能：

1. 🟢 agent-browser    ⭐ 4.9  📥 5,678  浏览器自动化
2. 🟡 code_reviewer    ⭐ 4.7  📥 3,456  AI代码审查
3. 🟢 file_ops         ⭐ 4.8  📥 2,890  文件操作
4. 🟡 video_downloader ⭐ 4.5  📥 2,345  视频下载
5. 🟡 notion_sync      ⭐ 4.6  📥 1,987  Notion同步
```

## 更新与卸载技能

**更新技能：**

```bash
# 更新单个技能
openclaw hub update <skill_name>

# 更新所有技能
openclaw hub update --all

# 更新到指定版本
openclaw hub update <skill_name>@1.3.0
```

**更新示例：**

```bash
$ openclaw hub update agent-browser

Checking for updates...
Current version: 1.5.0
Latest version: 1.5.2

Updating agent-browser...
████████████████████████████████████ 100%

✓ agent-browser updated to 1.5.2

Changelog:
  - Fix: 修复了某些网站的元素定位问题
  - Feat: 新增多标签页管理功能
  - Perf: 优化了内存使用
```

**卸载技能：**

```bash
# 卸载技能
openclaw hub uninstall <skill_name>

# 卸载并删除配置
openclaw hub uninstall <skill_name> --purge
```

**卸载示例：**

```bash
$ openclaw hub uninstall image_gen

确认卸载 image_gen@0.9.5？
  - 这将删除技能文件
  - 相关配置将被保留
  - 已生成的图像文件不会被删除

继续？[y/N] y

✓ image_gen 已卸载
```

## 技能配置

**查看技能配置：**

```bash
openclaw skill config show <skill_name>
```

**编辑技能配置：**

```bash
openclaw skill config edit <skill_name>
```

**设置配置项：**

```bash
openclaw skill config set <skill_name> <key> <value>

# 示例
openclaw skill config set agent-browser headless true
openclaw skill config set email provider gmail
```

**配置示例：**

```bash
$ openclaw skill config show email

技能配置：email
━━━━━━━━━━━━━━━━━━━━━━━━

provider: gmail
email: your@gmail.com
signature: |
  --
  Sent by OpenClaw
  
default_subject: "(无主题)"

配置文件：~/.openclaw/skills/email/config.yaml
```

---

掌握了技能管理命令，你就可以灵活地扩展OpenClaw的能力了。下一章，我们将学习如何开发自己的自定义技能。
