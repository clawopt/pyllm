# 日报自动生成与发送

对于打工人来说，写日报是一件既重要又繁琐的事情。这一章，我们用OpenClaw来实现日报的自动生成和发送。

## 场景描述

每天下班前，你需要：
1. 回顾今天做了什么
2. 查看代码提交记录
3. 整理会议笔记
4. 规划明天的工作
5. 发送邮件给领导

用OpenClaw，这些可以全部自动化。

## 读取代码提交记录

OpenClaw可以自动读取你的Git提交记录，作为日报的素材。

**配置Git技能：**

```bash
openclaw hub install git_ops
```

**获取今日提交记录：**

```bash
# 获取今天的提交记录
openclaw skill run git_ops.today-commits \
  --repo ~/Projects/myapp \
  --author "your_name"
```

**输出示例：**

```
今日提交记录（2026-03-28）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

提交1: feat: 添加用户登录API
  - 文件：src/api/auth.py
  - 时间：09:30
  - 详情：实现JWT认证，添加登录/登出接口

提交2: fix: 修复分页查询bug
  - 文件：src/services/query.py
  - 时间：14:15
  - 详情：修复了分页参数错误导致的数据重复问题

提交3: docs: 更新API文档
  - 文件：docs/api.md
  - 时间：16:30
  - 详情：补充登录接口的参数说明

统计：
  - 提交数：3
  - 修改文件：3
  - 新增代码：+156行
  - 删除代码：-23行
```

**多仓库汇总：**

```bash
# 汇总多个仓库的提交
openclaw skill run git_ops.multi-repo-commits \
  --repos "~/Projects/work/*" \
  --today
```

## 汇总会议笔记

如果你有记录会议笔记的习惯，OpenClaw可以自动汇总。

**会议笔记存储位置：**

```
~/Notes/meetings/
├── 2026-03-28_产品评审.md
├── 2026-03-28_技术讨论.md
└── ...
```

**汇总今日会议：**

```bash
openclaw skill run file_ops.read \
  --path "~/Notes/meetings/" \
  --filter "$(date +%Y-%m-%d)*" \
  --summarize
```

**会议笔记模板：**

```markdown
# 产品评审会议 - 2026-03-28 10:00

## 参会人员
张三、李四、王五

## 会议内容
1. 讨论了新功能的优先级
2. 确定了下周的开发计划
3. 评审了UI设计稿

## 待办事项
- [ ] 张三：完成API设计文档
- [ ] 李四：准备测试用例
- [ ] 王五：更新项目排期

## 决议
- 新功能A优先级提升为P0
- 下周一前完成设计评审
```

## 图文并茂输出

OpenClaw可以生成格式美观的日报，包含图表和统计。

**日报生成提示词：**

```
帮我生成今日工作日报，素材如下：

## Git提交记录
[粘贴提交记录]

## 会议笔记
[粘贴会议笔记]

## 其他事项
[手动补充]

## 输出要求
1. 格式：Markdown
2. 包含工作统计图表（用ASCII或描述）
3. 按项目分类
4. 包含明日计划
5. 语言简洁专业
```

**生成的日报示例：**

```markdown
# 工作日报 - 2026年3月28日

张三 | 前端开发组

## 今日工作

### 项目A：用户系统重构

**代码开发**
- ✅ 完成用户登录API开发（JWT认证）
- ✅ 修复分页查询bug
- 📝 更新API文档

**代码统计**
```
提交数：3次
新增代码：156行
修改文件：3个
```

### 会议与沟通

**产品评审会议（10:00-11:30）**
- 讨论新功能优先级
- 评审UI设计稿
- 确定下周开发计划

**技术讨论（15:00-16:00）**
- 讨论性能优化方案
- 确定缓存策略

## 工作进度

```
用户系统重构：████████░░ 80%
API文档更新：██████████ 100%
性能优化：███░░░░░░░ 30%
```

## 明日计划

1. 完成用户注册API开发
2. 编写单元测试
3. 参与技术评审会议

## 需要协调

- 设计稿需要周五前确认
- 测试环境服务器申请

---
*由 OpenClaw 自动生成*
```

## 邮件发送

日报生成后，自动发送邮件。

**配置邮件技能：**

```bash
openclaw hub install email
openclaw skill config email \
  --provider smtp \
  --host smtp.company.com \
  --port 587 \
  --username your@company.com \
  --password "${EMAIL_PASSWORD}"
```

**发送日报：**

```bash
openclaw skill run email.send \
  --to "leader@company.com" \
  --subject "工作日报 - 张三 - $(date +%Y-%m-%d)" \
  --body-file ~/Reports/daily_$(date +%Y-%m-%d).md \
  --format html
```

## 定时任务配置

将日报生成配置为定时任务，每天下班前自动执行。

**HEARTBEAT.md配置：**

```markdown
### 每日工作日报
- **任务ID**：daily_report
- **触发时间**：`0 18 * * 1-5`（工作日18:00）
- **执行内容**：
  1. 获取今日Git提交记录
  2. 汇总今日会议笔记
  3. 生成日报文档
  4. 发送邮件
- **通知方式**：飞书（发送完成后通知）
```

**完整的定时任务脚本：**

```bash
# 创建日报任务
openclaw heartbeat add \
  --name "daily_report" \
  --schedule "0 18 * * 1-5" \
  --task "
    # 获取Git提交
    commits=$(openclaw skill run git_ops.today-commits --repo ~/Projects/work)
    
    # 获取会议笔记
    meetings=$(openclaw skill run file_ops.read --path '~/Notes/meetings/' --filter 'today')
    
    # 生成日报
    report=$(openclaw chat \"生成日报，提交记录：\$commits，会议：\$meetings\")
    
    # 保存日报
    echo \"\$report\" > ~/Reports/daily_\$(date +%Y-%m-%d).md
    
    # 发送邮件
    openclaw skill run email.send --to leader@company.com --subject \"日报 - \$(date +%Y-%m-%d)\" --body \"\$report\"
  " \
  --notify feishu
```

## 周报自动生成

类似地，可以配置周报自动生成。

**周报配置：**

```markdown
### 每周工作周报
- **任务ID**：weekly_report
- **触发时间**：`0 17 * * 5`（每周五17:00）
- **执行内容**：
  1. 汇总本周Git提交记录
  2. 统计代码变更量
  3. 汇总本周完成的任务
  4. 生成周报文档
  5. 发送邮件
```

**周报模板：**

```markdown
# 工作周报 - 2026年第13周

张三 | 前端开发组

## 本周工作总结

### 项目进度

| 项目 | 进度 | 状态 |
|------|------|------|
| 用户系统重构 | 80% | 进行中 |
| API文档更新 | 100% | 已完成 |
| 性能优化 | 30% | 进行中 |

### 代码统计

```
本周提交：15次
新增代码：856行
删除代码：234行
修改文件：23个
```

### 主要成果

1. 完成用户登录/注册API
2. 修复3个线上bug
3. 完成API文档更新

## 下周计划

1. 完成用户系统重构
2. 开始性能优化工作
3. 参与技术分享会

## 需要支持

- 测试环境资源不足
- 设计资源需要协调
```

---

通过这个案例，你可以看到OpenClaw如何将日常重复性工作自动化。下一章，我们来看如何搭建一个7×24小时的信息中枢。
