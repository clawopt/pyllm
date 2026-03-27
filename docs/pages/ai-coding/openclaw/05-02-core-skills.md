# 必备核心技能

ClawHub上有上万个技能，哪些是真正值得安装的？这一章，我们精选了7个必备核心技能，覆盖了日常使用中最常见的场景。

## 浏览器自动化：agent-browser

这是OpenClaw官方维护的浏览器自动化技能，也是最强大的技能之一。

**安装：**

```bash
openclaw hub install agent-browser
```

**核心能力：**

| 功能 | 描述 |
|------|------|
| 网页导航 | 打开URL、前进、后退、刷新 |
| 元素操作 | 点击、输入、选择、拖拽 |
| 内容提取 | 抓取文本、图片、链接、表格 |
| 截图 | 全页面或指定区域截图 |
| 表单填写 | 自动填写登录表单、提交数据 |
| 多标签页 | 管理多个浏览器标签 |

**使用示例：**

```
用户：帮我打开GitHub，搜索Python爬虫项目，按star数排序

OpenClaw：[执行浏览器自动化]
  → 启动浏览器
  → 导航到 https://github.com
  → 在搜索框输入 "Python crawler"
  → 点击搜索
  → 选择 "Sort by: Most stars"
  → 抓取前10个项目信息

搜索结果：
1. scrapy/scrapy ⭐ 52,234
   链接：https://github.com/scrapy/scrapy
   
2. psf/requests ⭐ 51,456
   链接：https://github.com/psf/requests
   
...
```

**高级用法：**

```bash
# 登录网站（需要保存登录状态）
openclaw skill run agent-browser --action login \
  --url "https://example.com/login" \
  --username-field "#username" \
  --password-field "#password" \
  --submit-button "#login-btn"

# 批量截图
openclaw skill run agent-browser --action screenshot \
  --urls "urls.txt" \
  --output "~/screenshots/"
```

## 文件管理：file_ops

文件管理是最基础也是最常用的技能。

**安装：**

```bash
openclaw hub install file_ops
```

**核心能力：**

| 功能 | 命令 |
|------|------|
| 读取文件 | `file_ops.read` |
| 写入文件 | `file_ops.write` |
| 移动文件 | `file_ops.move` |
| 复制文件 | `file_ops.copy` |
| 删除文件 | `file_ops.delete` |
| 创建目录 | `file_ops.mkdir` |
| 列出目录 | `file_ops.list` |
| 搜索文件 | `file_ops.search` |
| 压缩文件 | `file_ops.compress` |
| 解压文件 | `file_ops.extract` |

**使用示例：**

```
用户：帮我整理下载文件夹，按文件类型分类

OpenClaw：[执行文件整理]
  → 扫描 ~/Downloads/ 目录
  → 发现 156 个文件
  → 按类型分类：
     - 图片 (45个) → Images/
     - 文档 (32个) → Documents/
     - 压缩包 (28个) → Archives/
     - 代码 (23个) → Code/
     - 其他 (28个) → Others/
  → 创建分类目录
  → 移动文件

✓ 整理完成，共处理 156 个文件
```

## 联网搜索：web_search

让OpenClaw具备实时信息检索能力。

**安装：**

```bash
openclaw hub install web_search
```

**支持的搜索引擎：**

- Google（默认）
- Bing
- DuckDuckGo
- 百度
- 必应中国

**使用示例：**

```
用户：帮我搜索最新的AI新闻，汇总成报告

OpenClaw：[执行联网搜索]
  → 搜索 "AI news 2026 March"
  → 获取前10条结果
  → 访问每篇文章提取摘要
  → 过滤重复内容
  → 生成汇总报告

AI行业动态汇总 - 2026年3月28日

1. OpenAI发布GPT-5，推理能力大幅提升
   来源：TechCrunch
   摘要：新模型在数学推理和代码生成方面表现优异...
   
2. DeepSeek开源新版本，成本降低80%
   来源：GitHub Blog
   摘要：最新版本优化了推理效率...
   
...
```

## 图像生成：image_gen

AI绘画能力，支持多种图像生成模型。

**安装：**

```bash
openclaw hub install image_gen
```

**支持的模型：**

| 模型 | 特点 |
|------|------|
| DALL-E 3 | OpenAI官方，创意性强 |
| Stable Diffusion | 开源，可本地部署 |
| Midjourney | 艺术风格独特 |
| 通义万相 | 阿里云，中文理解好 |

**使用示例：**

```
用户：帮我生成一张科技感的logo图片

OpenClaw：[生成图像]
  → 解析需求：科技感logo
  → 优化提示词：minimalist tech logo, blue gradient, circuit pattern, modern design
  → 调用 DALL-E 3
  → 生成 4 张候选图

已生成图像：
1. /workspace/images/logo_1.png
2. /workspace/images/logo_2.png
3. /workspace/images/logo_3.png
4. /workspace/images/logo_4.png

需要我调整风格或重新生成吗？
```

## 邮件发送：email

集成Gmail和其他邮件服务，自动发送邮件。

**安装：**

```bash
openclaw hub install email
```

**配置：**

```bash
# 配置Gmail（需要应用专用密码）
openclaw skill config email \
  --provider gmail \
  --email your@gmail.com \
  --password "your_app_password"

# 配置SMTP服务器
openclaw skill config email \
  --provider smtp \
  --host smtp.example.com \
  --port 587 \
  --username your@email.com \
  --password "your_password"
```

**使用示例：**

```
用户：把这周的工作总结发送给老板

OpenClaw：[执行邮件发送]
  → 读取 ~/Notes/weekly_report.md
  → 收件人：boss@company.com（从联系人获取）
  → 主题：周报 - 2026年第13周
  → 格式化为HTML邮件
  → 发送

✓ 邮件已发送至 boss@company.com
```

## 内容摘要：summarizer

一键提炼网页、文档、视频的核心内容。

**安装：**

```bash
openclaw hub install summarizer
```

**支持的内容类型：**

- 网页文章
- PDF文档
- Word文档
- YouTube视频（提取字幕）
- 音频文件（语音转文字后摘要）

**使用示例：**

```
用户：帮我总结这篇文章 https://example.com/long-article

OpenClaw：[执行内容摘要]
  → 访问网页
  → 提取正文内容（约5000字）
  → 生成摘要

文章摘要：
━━━━━━━━━━━━━━━━━━━━━━━━

核心观点：
本文讨论了AI Agent的发展趋势，指出从"推理式思考"向"智能体式思考"的转变是必然方向。

关键要点：
1. 传统聊天机器人只能"表达"，无法"交付"
2. AI Agent通过与外部环境交互实现闭环
3. 2026年将是Agent爆发之年

作者建议：
- 开发者应关注Agent开发框架
- 企业应布局Agent应用场景
- 用户应培养"任务委托"的使用习惯

原文链接：https://example.com/long-article
━━━━━━━━━━━━━━━━━━━━━━━━
```

## 结构化记忆：ontology

高级知识管理技能，让OpenClaw能够建立和维护知识图谱。

**安装：**

```bash
openclaw hub install ontology
```

**核心能力：**

| 功能 | 描述 |
|------|------|
| 实体提取 | 从文本中识别人物、地点、事件等实体 |
| 关系建立 | 建立实体之间的关系 |
| 知识查询 | 基于知识图谱回答问题 |
| 记忆持久化 | 长期存储和检索信息 |

**使用示例：**

```
用户：记住，我下周三要和张三开会讨论项目A

OpenClaw：[存储到知识图谱]
  → 提取实体：
     - 事件：会议
     - 人物：张三
     - 项目：项目A
     - 时间：下周三（2026-04-02）
  → 建立关系：
     - 你 → 会议 → 张三
     - 会议 → 讨论 → 项目A
  → 存储

✓ 已记住：2026-04-02 与张三讨论项目A

用户：我下周有什么安排？

OpenClaw：[查询知识图谱]
根据记忆，你下周有以下安排：

周三（4月2日）：
- 与张三开会讨论项目A

需要我添加提醒吗？
```

---

以上7个技能覆盖了日常使用的大部分场景。安装完成后，你的小龙虾已经具备了相当强的实战能力。下一章，我们将学习如何管理这些技能。
