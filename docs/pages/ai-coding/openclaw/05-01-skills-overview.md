# Skills技能系统概述

OpenClaw的强大不仅来自于大模型的智能，更来自于它能实际"动手做事"的能力。这种能力通过Skills技能系统实现。这一章，我们来深入了解这个让小龙虾"学会干活"的核心机制。

## 什么是Skill

Skill（技能）是OpenClaw的能力扩展单元。每个Skill封装了一类具体的操作能力，比如浏览网页、操作文件、发送邮件等。

**Skill的本质是一个纯文本配置文件：**

```
~/.openclaw/skills/
├── browser/
│   ├── SKILL.md           # 技能定义文件
│   ├── templates/         # 提示词模板
│   └── scripts/           # 执行脚本
├── file_ops/
│   └── SKILL.md
└── web_search/
    └── SKILL.md
```

**一个简单的SKILL.md示例：**

```markdown
---
name: file_ops
version: 1.0.0
description: 文件操作技能，支持读写、移动、重命名文件
author: OpenClaw Team
tags: [file, core]
dependencies: []
permissions:
  - file:read
  - file:write
  - file:delete
---

# 文件操作技能

## 能力描述
提供文件系统操作能力，包括：
- 读取文件内容
- 写入/创建文件
- 移动/重命名文件
- 删除文件
- 创建目录

## 使用场景
- 整理文件夹
- 批量重命名
- 日志文件管理
- 配置文件修改

## 调用示例

### 读取文件
```json
{
  "action": "file_ops.read",
  "params": {
    "path": "~/Documents/notes.txt"
  }
}
```

### 写入文件
```json
{
  "action": "file_ops.write",
  "params": {
    "path": "~/Documents/output.txt",
    "content": "Hello, OpenClaw!"
  }
}
```

## 注意事项
- 删除操作需要用户确认
- 敏感文件路径受USER.md限制
```

**Skill与大模型的关系：**

```
用户指令："帮我整理下载文件夹"
        ↓
    Gateway（理解意图）
        ↓
    规划任务步骤
        ↓
    ┌───────────────────────────────┐
    │  Step 1: 扫描目录              │
    │  调用 Skill: file_ops.scan    │
    ├───────────────────────────────┤
    │  Step 2: 分类文件              │
    │  调用 Skill: file_ops.classify│
    ├───────────────────────────────┤
    │  Step 3: 移动文件              │
    │  调用 Skill: file_ops.move    │
    └───────────────────────────────┘
        ↓
    返回执行结果
```

大模型负责理解意图、规划步骤、决策调用哪个Skill；Skill负责执行具体的操作。两者配合，实现从"想法"到"行动"的闭环。

## 技能加载优先级

OpenClaw支持多层次的技能配置，按优先级从高到低：

**1. 项目级技能（最高优先级）**

位于当前工作目录下：

```
~/Projects/myapp/
└── .openclaw/
    └── skills/
        └── custom_skill/
            └── SKILL.md
```

项目级技能只在当前项目中生效，适合项目特定的定制需求。比如一个Python项目可以定义专门的代码规范检查技能。

**2. 用户级技能**

位于用户配置目录下：

```
~/.openclaw/skills/
├── browser/
├── file_ops/
└── my_custom_skill/
```

用户级技能对所有项目生效，是你个人定制的技能库。

**3. 内置技能（最低优先级）**

OpenClaw内置的核心技能：

```
/usr/lib/openclaw/skills/
├── core/
│   ├── file_ops/
│   ├── shell/
│   └── http/
└── utils/
    ├── calculator/
    └── datetime/
```

内置技能不可修改，但可以被用户级或项目级技能覆盖。

**优先级示例：**

假设内置技能、用户级技能、项目级技能都定义了同名技能`file_ops`：

```
内置: /usr/lib/openclaw/skills/file_ops/      ← 被覆盖
用户: ~/.openclaw/skills/file_ops/             ← 被覆盖
项目: ~/Projects/myapp/.openclaw/skills/file_ops/  ← 最终使用
```

**查看技能加载顺序：**

```bash
openclaw skills priority file_ops

# 输出
Skill: file_ops
Priority order:
  1. Project: ~/Projects/myapp/.openclaw/skills/file_ops/ (loaded)
  2. User: ~/.openclaw/skills/file_ops/ (exists, skipped)
  3. Built-in: /usr/lib/openclaw/skills/file_ops/ (exists, skipped)
```

## ClawHub技能市场

OpenClaw拥有一个活跃的技能市场——ClawHub，社区用户可以分享和下载技能。

**访问ClawHub：**

```bash
# 命令行访问
openclaw hub browse

# 或访问网站
https://hub.openclaw.ai
```

**ClawHub统计数据：**

| 指标 | 数据 |
|------|------|
| 收录技能 | 13,729+ |
| 活跃开发者 | 2,300+ |
| 月下载量 | 500,000+ |
| 官方认证技能 | 156 |

**技能分类：**

```
ClawHub技能分类
├── 核心能力
│   ├── 文件操作 (file_ops)
│   ├── 网络请求 (http)
│   └── 系统命令 (shell)
├── 浏览器自动化
│   ├── 网页抓取 (web_scraper)
│   ├── 表单填写 (form_filler)
│   └── 截图工具 (screenshot)
├── 内容处理
│   ├── 文档摘要 (summarizer)
│   ├── 翻译工具 (translator)
│   └── 格式转换 (converter)
├── 通信集成
│   ├── 邮件发送 (email)
│   ├── 消息推送 (notification)
│   └── 日历管理 (calendar)
├── 开发工具
│   ├── 代码生成 (code_gen)
│   ├── Git操作 (git_ops)
│   └── 数据库操作 (database)
└── AI增强
    ├── 图像生成 (image_gen)
    ├── 语音合成 (tts)
    └── 知识管理 (ontology)
```

**技能质量标识：**

| 标识 | 含义 |
|------|------|
| 🟢 官方认证 | OpenClaw官方维护，安全可靠 |
| 🟡 社区验证 | 下载量>1000，评分>4.0 |
| ⚪ 新发布 | 发布时间<30天 |
| 🔴 安全警告 | 存在已知安全问题（谨慎使用） |

**搜索技能：**

```bash
# 搜索关键词
openclaw hub search "browser"

# 按分类搜索
openclaw hub search --category "automation"

# 按评分排序
openclaw hub search "file" --sort rating

# 查看热门技能
openclaw hub trending
```

**搜索结果示例：**

```
搜索 "browser" 的结果（共23个技能）：

┌─────────────────────┬─────────┬──────────┬─────────────────────┐
│ 技能名称            │ 评分    │ 下载量   │ 描述                │
├─────────────────────┼─────────┼──────────┼─────────────────────┤
│ 🟢 agent-browser    │ ⭐ 4.9  │ 89,234   │ 官方浏览器自动化    │
│ 🟡 web_scraper      │ ⭐ 4.7  │ 45,678   │ 网页内容抓取        │
│ 🟡 form_filler      │ ⭐ 4.5  │ 23,456   │ 自动填写网页表单    │
│ ⚪ browser_recorder │ ⭐ 4.3  │ 1,234    │ 录制浏览器操作      │
└─────────────────────┴─────────┴──────────┴─────────────────────┘
```

---

理解了技能系统的基本概念，下一章我们将介绍一些必备的核心技能，让你的小龙虾快速具备实战能力。
