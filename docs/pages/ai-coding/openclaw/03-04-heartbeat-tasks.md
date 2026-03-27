# 定时任务配置

OpenClaw不仅能响应你的即时指令，还能按照预设的时间表自动执行任务。通过HEARTBEAT.md文件，你可以配置各种定时任务，让AI助手真正成为"永不休息"的数字伙伴。

## HEARTBEAT.md文件概述

HEARTBEAT.md定义了OpenClaw的定时任务：

```
~/.openclaw/agents/default/HEARTBEAT.md
```

**基本结构：**

```markdown
# 定时任务配置

## 任务列表

### 任务名称
- **触发时间**：Cron表达式
- **执行内容**：任务描述
- **通知方式**：结果通知渠道
- **失败处理**：重试策略
```

**查看当前定时任务：**

```bash
# 查看所有定时任务
openclaw heartbeat list

# 查看任务详情
openclaw heartbeat show <task_name>

# 查看下次执行时间
openclaw heartbeat schedule
```

## Cron表达式语法

OpenClaw使用标准的Cron表达式定义任务触发时间。

**基本格式：**

```
┌───────────── 分钟 (0 - 59)
│ ┌───────────── 小时 (0 - 23)
│ │ ┌───────────── 日 (1 - 31)
│ │ │ ┌───────────── 月 (1 - 12)
│ │ │ │ ┌───────────── 星期 (0 - 6, 0=周日)
│ │ │ │ │
* * * * *
```

**常用表达式示例：**

| 表达式 | 含义 |
|--------|------|
| `*/30 * * * *` | 每30分钟 |
| `0 * * * *` | 每小时整点 |
| `0 9 * * *` | 每天早上9点 |
| `0 9 * * 1-5` | 工作日早上9点 |
| `0 18 * * 5` | 每周五下午6点 |
| `0 0 1 * *` | 每月1号凌晨 |
| `0 9,12,18 * * *` | 每天9点、12点、18点 |

**特殊字符：**

| 字符 | 含义 | 示例 |
|------|------|------|
| `*` | 任意值 | `* * * * *` = 每分钟 |
| `,` | 列举多个值 | `0 9,12,18 * * *` = 每天9、12、18点 |
| `-` | 范围 | `0 9-17 * * 1-5` = 工作日9-17点每小时 |
| `/` | 间隔 | `*/15 * * * *` = 每15分钟 |

**在线Cron生成器：**

如果不确定如何编写Cron表达式，可以使用在线工具：
- https://crontab.guru/
- https://cron.qqe2.com/

OpenClaw也内置了自然语言转Cron的功能：

```bash
# 自然语言转Cron
openclaw cron parse "每天早上9点"
# 输出：0 9 * * *

openclaw cron parse "每30分钟"
# 输出：*/30 * * * *

openclaw cron parse "工作日下午5点"
# 输出：0 17 * * 1-5
```

## 示例：每30分钟抓取Hacker News

让我们配置一个定时任务，每30分钟自动抓取Hacker News的热门内容。

**配置方式一：命令行配置**

```bash
# 创建定时任务
openclaw heartbeat add \
  --name "hacker_news_fetch" \
  --schedule "*/30 * * * *" \
  --task "抓取Hacker News前10条热门内容，提取标题、链接和评论数" \
  --notify "feishu" \
  --save-to "~/Notes/hacker_news/"
```

**配置方式二：编辑HEARTBEAT.md**

```markdown
# 定时任务配置

## 任务列表

### Hacker News 热门抓取
- **任务ID**：hacker_news_fetch
- **触发时间**：`*/30 * * * *`（每30分钟）
- **执行内容**：
  1. 访问 https://news.ycombinator.com/
  2. 获取前10条热门内容
  3. 提取标题、链接、评论数、分数
  4. 保存到 `~/Notes/hacker_news/` 目录
  5. 如果有重要内容，发送飞书通知
- **通知方式**：飞书（仅当发现重要内容时）
- **失败处理**：重试3次，间隔5分钟
- **超时时间**：5分钟
```

**任务执行效果：**

```
[2026-03-28 10:00:00] ⏰ 触发定时任务：hacker_news_fetch
[2026-03-28 10:00:01] 🌐 访问 Hacker News...
[2026-03-28 10:00:03] 📊 解析页面内容...
[2026-03-28 10:00:05] 💾 保存到文件：~/Notes/hacker_news/2026-03-28_10-00.md
[2026-03-28 10:00:05] ✅ 任务完成

热门内容已保存：
1. Show HN: I built a tool to... (324 points, 156 comments)
2. The state of AI in 2026 (289 points, 234 comments)
3. ...
```

**保存的文件格式：**

```markdown
# Hacker News 热门 - 2026-03-28 10:00

抓取时间：2026-03-28 10:00:05

## 热门列表

1. **Show HN: I built a tool to automate code reviews**
   - 链接：https://github.com/xxx/xxx
   - 分数：324
   - 评论：156
   - 标签：show

2. **The state of AI in 2026**
   - 链接：https://example.com/ai-2026
   - 分数：289
   - 评论：234
   - 标签：article

...
```

## 示例：每天9点汇总行业动态

配置一个每天早上9点自动汇总行业动态的任务。

**HEARTBEAT.md配置：**

```markdown
# 定时任务配置

## 任务列表

### 每日行业动态汇总
- **任务ID**：daily_industry_news
- **触发时间**：`0 9 * * 1-5`（工作日早上9点）
- **执行内容**：
  1. 抓取以下信息源的最新内容：
     - Hacker News 前10条
     - GitHub Trending (Python方向) 前5条
     - 掘金热门文章 前5条
     - V2EX 最新主题 前5条
  2. 过滤与以下关键词相关的内容：
     - AI、机器学习、深度学习
     - Python、JavaScript、Go
     - 开源、架构、性能
  3. 生成汇总报告（Markdown格式）
  4. 发送到飞书和邮箱
- **通知方式**：飞书 + 邮件
- **保存位置**：`~/Notes/daily_reports/`
- **失败处理**：重试2次，如果仍失败则发送告警
```

**命令行配置：**

```bash
openclaw heartbeat add \
  --name "daily_industry_news" \
  --schedule "0 9 * * 1-5" \
  --task "汇总行业动态：Hacker News、GitHub Trending、掘金、V2EX，过滤AI/编程相关内容" \
  --notify "feishu,email" \
  --filter "AI,机器学习,Python,JavaScript,Go,开源,架构" \
  --save-to "~/Notes/daily_reports/"
```

**生成的汇总报告：**

```markdown
# 每日行业动态 - 2026年3月28日

生成时间：2026-03-28 09:00:15

## 🔥 Hacker News 热门

1. **OpenAI releases GPT-5 with reasoning capabilities**
   - 来源：Hacker News
   - 链接：https://news.ycombinator.com/item?id=xxx
   - 讨论：456条评论
   - 摘要：OpenAI发布了具备推理能力的新模型...

2. **Why I switched from Python to Go**
   - 来源：Hacker News
   - 链接：https://example.com/python-to-go
   - 讨论：234条评论
   - 摘要：一位资深开发者分享了他的技术选型经历...

## 📦 GitHub Trending (Python)

1. **langchain-ai/langchain** ⭐ 89,234
   - 描述：Building applications with LLMs
   - 今日增长：+234 stars

2. **pytorch/pytorch** ⭐ 78,456
   - 描述：Tensors and Dynamic neural networks
   - 今日增长：+156 stars

## 📝 掘金热门

1. **深入理解Transformer架构**
   - 作者：张三
   - 阅读量：12,345
   - 链接：https://juejin.cn/post/xxx

## 💬 V2EX 热门

1. **大家都在用什么AI编程工具？**
   - 回复：156
   - 链接：https://v2ex.com/t/xxx

---
*由 OpenClaw 自动生成*
```

## 更多定时任务示例

**每周工作报告：**

```markdown
### 每周工作报告
- **任务ID**：weekly_report
- **触发时间**：`0 17 * * 5`（每周五下午5点）
- **执行内容**：
  1. 汇总本周完成的任务
  2. 统计代码提交记录
  3. 生成周报文档
  4. 发送到工作邮箱
```

**每日备份：**

```markdown
### 项目每日备份
- **任务ID**：daily_backup
- **触发时间**：`0 2 * * *`（每天凌晨2点）
- **执行内容**：
  1. 压缩 `~/Projects/` 目录
  2. 上传到云存储
  3. 清理7天前的备份
  4. 发送备份完成通知
```

**网站监控：**

```markdown
### 网站健康检查
- **任务ID**：website_health_check
- **触发时间**：`*/5 * * * *`（每5分钟）
- **执行内容**：
  1. 检查以下网站可访问性：
     - https://myapp.example.com
     - https://api.example.com/health
  2. 如果响应异常，发送告警
  3. 记录响应时间
```

**股票价格提醒：**

```markdown
### 股票价格监控
- **任务ID**：stock_price_alert
- **触发时间**：`*/30 9-15 * * 1-5`（交易时间每30分钟）
- **执行内容**：
  1. 获取关注股票的最新价格
  2. 如果价格超过设定阈值，发送提醒
  3. 记录价格变化
```

## 管理定时任务

```bash
# 列出所有任务
openclaw heartbeat list

# 查看任务详情
openclaw heartbeat show daily_industry_news

# 手动触发任务
openclaw heartbeat run daily_industry_news

# 暂停任务
openclaw heartbeat pause daily_industry_news

# 恢复任务
openclaw heartbeat resume daily_industry_news

# 删除任务
openclaw heartbeat delete daily_industry_news

# 查看任务执行历史
openclaw heartbeat history daily_industry_news

# 查看下次执行时间
openclaw heartbeat next
```

**任务执行历史：**

```bash
$ openclaw heartbeat history daily_industry_news

最近10次执行记录：

┌─────────────────────┬─────────┬──────────┬─────────────────┐
│ 执行时间            │ 状态    │ 耗时     │ 结果            │
├─────────────────────┼─────────┼──────────┼─────────────────┤
│ 2026-03-28 09:00:00 │ ✅ 成功 │ 15.3s    │ 汇总25条内容    │
│ 2026-03-27 09:00:00 │ ✅ 成功 │ 12.8s    │ 汇总23条内容    │
│ 2026-03-26 09:00:00 │ ✅ 成功 │ 14.1s    │ 汇总28条内容    │
│ 2026-03-25 09:00:00 │ ❌ 失败 │ 30.0s    │ 网络超时        │
│ 2026-03-24 09:00:00 │ ✅ 成功 │ 11.5s    │ 汇总21条内容    │
└─────────────────────┴─────────┴──────────┴─────────────────┘

成功率：80% (4/5)
平均耗时：13.4s
```

---

通过HEARTBEAT.md，OpenClaw可以自动执行各种周期性任务，真正成为你的"数字员工"。结合SOUL.md和USER.md，你已经完成了OpenClaw的全面定制。接下来，让我们进入实战环节，看看OpenClaw如何解决真实场景中的问题。
