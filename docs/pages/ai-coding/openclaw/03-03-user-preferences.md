# 设置用户偏好

如果说SOUL.md定义了AI"是谁"，那么USER.md就定义了AI"如何服务你"。通过USER.md，你可以设置个人偏好、操作限制和安全边界，让OpenClaw更懂你的需求。

## USER.md文件概述

USER.md同样位于Agent配置目录下：

```
~/.openclaw/agents/default/USER.md
```

**基本结构：**

```markdown
# 用户偏好设置

## 基本信息
[用户的基本信息和偏好]

## 操作限制
[禁止或限制的操作]

## 确认机制
[需要用户确认的操作类型]

## 常用配置
[常用路径、账号、设置等]

## 特殊要求
[其他个性化需求]
```

**查看当前USER配置：**

```bash
# 查看默认USER
cat ~/.openclaw/agents/default/USER.md

# 或使用命令
openclaw user show
```

## 禁止操作清单

安全是AI助手的首要原则。USER.md允许你定义明确的操作边界。

**文件操作限制：**

```markdown
# 用户偏好设置

## 操作限制

### 禁止的文件操作
- **禁止删除**以下目录及其内容：
  - `~/Documents/重要文档`
  - `~/Projects/production-*`
  - `~/.ssh/`
  - `~/.config/`
  
- **禁止修改**以下文件：
  - `~/.bashrc`
  - `~/.zshrc`
  - `~/Projects/*/config/secrets.yaml`
  
- **禁止访问**以下目录：
  - `~/Private/`
  - `~/Financial/`

### 文件操作规则
- 删除文件前必须移动到回收站（`~/.Trash/`）
- 修改配置文件前必须备份
- 批量操作（超过10个文件）需要确认
```

**系统操作限制：**

```markdown
## 系统操作限制

### 禁止的系统操作
- **禁止执行**以下命令：
  - `rm -rf /`
  - `sudo` 相关命令（除非明确授权）
  - `chmod 777`
  - 任何涉及磁盘格式化的命令
  
- **禁止修改**系统配置：
  - 防火墙规则
  - 网络配置
  - 用户权限

### 网络操作限制
- 禁止上传文件到未授权的服务器
- 禁止发送敏感信息到外部API
- 仅允许访问白名单域名：
  - `*.aliyun.com`
  - `*.github.com`
  - `api.openai.com`
```

**代码操作限制：**

```markdown
## 代码操作限制

### Git操作规则
- 禁止强制推送到 `main` 和 `master` 分支
- 禁止删除远程分支（除非明确授权）
- 提交前必须检查 `.gitignore`

### 数据库操作规则
- 生产数据库：仅允许SELECT查询
- 禁止执行 DROP、TRUNCATE 命令
- 批量更新（超过100条）需要确认

### 部署操作规则
- 禁止直接部署到生产环境
- 部署前必须通过测试环境验证
- 回滚脚本必须提前准备
```

**配置禁止操作：**

```bash
# 添加禁止路径
openclaw user forbid path ~/Documents/重要文档

# 添加禁止命令
openclaw user forbid command "rm -rf"

# 查看禁止列表
openclaw user forbid list
```

## 确认机制

对于潜在风险操作，OpenClaw会在执行前请求你的确认。

**配置确认级别：**

```markdown
## 确认机制

### 确认级别设置
- **级别1（自动执行）**：安全的日常操作
  - 创建文件
  - 读取文件
  - 格式转换
  
- **级别2（提示确认）**：有潜在影响的操作
  - 删除单个文件
  - 修改配置文件
  - 发送邮件
  
- **级别3（必须确认）**：高风险操作
  - 批量删除
  - 系统配置修改
  - Git push
  - 数据库修改

### 确认方式
- 即时消息确认（默认）
- 邮件确认（用于重要操作）
- 超时自动取消（默认5分钟）
```

**具体确认规则：**

```markdown
## 操作确认规则

### 文件操作
| 操作类型 | 确认要求 |
|---------|---------|
| 创建文件 | 无需确认 |
| 读取文件 | 无需确认 |
| 修改文件 < 10KB | 无需确认 |
| 修改文件 >= 10KB | 提示确认 |
| 删除文件 | 必须确认 |
| 批量操作 > 10个文件 | 必须确认 |

### 网络操作
| 操作类型 | 确认要求 |
|---------|---------|
| HTTP GET | 无需确认（白名单域名） |
| HTTP POST | 提示确认 |
| 文件上传 | 必须确认 |
| API调用（付费） | 必须确认 |

### 代码操作
| 操作类型 | 确认要求 |
|---------|---------|
| 本地commit | 无需确认 |
| Git push | 必须确认 |
| 创建分支 | 无需确认 |
| 删除分支 | 必须确认 |
| 合并PR | 必须确认 |
```

**确认对话示例：**

```
用户：删除所有日志文件

OpenClaw：⚠️ 这是一个需要确认的操作

我将执行以下操作：
- 删除目录：~/Projects/myapp/logs/
- 文件数量：23个.log文件
- 总大小：156MB

请确认是否继续？
[确认] [取消] [查看文件列表]

用户：[确认]

OpenClaw：✓ 已删除23个日志文件，释放156MB空间
```

**配置确认命令：**

```bash
# 设置确认级别
openclaw user confirm-level set delete always  # 删除操作总是确认

# 设置超时时间
openclaw user confirm-timeout 300  # 5分钟超时

# 临时跳过确认（危险）
openclaw user confirm-skip --for 1h  # 1小时内跳过确认
```

## 常用配置

USER.md还可以存储你的常用设置，让OpenClaw更了解你的工作环境。

**路径配置：**

```markdown
## 常用路径

### 项目目录
- 工作项目：`~/Projects/work/`
- 个人项目：`~/Projects/personal/`
- 学习笔记：`~/Notes/`

### 常用文件
- 待办事项：`~/Notes/todo.md`
- 工作日志：`~/Notes/work-log/`
- 配置模板：`~/Templates/`

### 输出目录
- 下载文件：`~/Downloads/`
- 导出文件：`~/Exports/`
- 临时文件：`/tmp/openclaw/`
```

**工具配置：**

```markdown
## 开发环境

### 编程语言
- 主要语言：Python 3.11
- 次要语言：JavaScript, Go

### 代码规范
- Python：遵循PEP 8，使用Black格式化
- JavaScript：使用ESLint + Prettier
- 提交信息：遵循Conventional Commits

### 常用工具
- 包管理：pip, npm
- 版本控制：Git
- 容器：Docker
- 编辑器：VS Code
```

**账号配置（非敏感）：**

```markdown
## 账号信息

### GitHub
- 用户名：your-username
- 常用仓库：your-org/main-project

### 邮箱
- 工作邮箱：work@example.com
- 个人邮箱：personal@example.com

### 时区
- 时区：Asia/Shanghai (UTC+8)
- 工作时间：周一至周五 09:00-18:00
```

**偏好设置：**

```markdown
## 个人偏好

### 语言
- 界面语言：中文
- 代码注释：中文
- 技术文档：英文

### 输出格式
- 表格：Markdown格式
- 代码：带语法高亮
- 列表：使用 `-` 符号

### 通知设置
- 任务完成：通知
- 需要确认：通知
- 错误发生：立即通知

### 其他偏好
- 温度单位：摄氏度
- 货币单位：人民币（CNY）
- 日期格式：YYYY-MM-DD
```

## 特殊要求

你还可以在USER.md中添加任何特殊要求。

**隐私保护：**

```markdown
## 隐私保护

### 敏感信息处理
- 不在日志中记录密码、Token
- 不在会话中保存敏感数据
- 敏感文件使用加密存储

### 数据保留
- 会话记录保留：30天
- 任务记录保留：90天
- 日志文件保留：7天
```

**工作习惯：**

```markdown
## 工作习惯

### 响应时间
- 工作时间：立即响应
- 非工作时间：记录待处理，次日处理

### 任务优先级
1. 紧急：立即处理
2. 重要：当天处理
3. 一般：按计划处理

### 报告格式
- 每日报告：18:00自动生成
- 周报：每周五17:00生成
- 格式：Markdown，发送到邮箱
```

## 管理USER配置

```bash
# 编辑USER文件
openclaw user edit

# 查看完整配置
openclaw user show

# 验证配置
openclaw user validate

# 重置为默认
openclaw user reset

# 导出配置
openclaw user export > user_backup.md

# 导入配置
openclaw user import < user_backup.md
```

---

通过SOUL.md和USER.md的配合，你已经完全定义了AI的身份和行为边界。下一章，我们将学习如何配置定时任务，让OpenClaw自动执行周期性工作。
