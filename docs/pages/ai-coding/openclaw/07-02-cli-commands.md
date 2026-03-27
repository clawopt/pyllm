# 常用CLI命令

OpenClaw提供了丰富的命令行工具，让你可以在终端中完成所有管理操作。这一章，我们来学习最常用的CLI命令。

## 健康检查

`openclaw doctor`是最常用的诊断命令，可以快速检查系统状态。

**基本用法：**

```bash
openclaw doctor
```

**输出示例：**

```
Running OpenClaw diagnostics...

System
  ✓ Node.js: v22.1.0 (requires >= 22.0.0)
  ✓ Platform: linux x64
  ✓ Memory: 7.8GB available
  ✓ Disk: 156GB free

OpenClaw
  ✓ Version: v1.2.3
  ✓ Workspace: ~/.openclaw/workspace
  ✓ Config: ~/.openclaw/config.yaml

Model
  ✓ Provider: alibaba
  ✓ Model: qwen-plus
  ✓ API Key: configured
  ✓ Connection: OK (latency: 245ms)

Channels
  ✓ Web: running on http://0.0.0.0:18789
  ✓ Feishu: connected
  ⚠ Telegram: not configured

Skills
  ✓ 7 skills installed
  ✓ No security issues detected

Service
  ✓ Status: running
  ✓ PID: 12345
  ✓ Uptime: 2 days, 5 hours

All checks passed! OpenClaw is healthy.
```

**自动修复：**

```bash
# 自动修复发现的问题
openclaw doctor --fix
```

**详细输出：**

```bash
# 显示更详细的诊断信息
openclaw doctor --verbose
```

## 状态查看

**查看整体状态：**

```bash
openclaw status
```

**输出示例：**

```
OpenClaw Status
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Service: 🟢 running (PID: 12345)
Uptime: 2 days, 5 hours, 15 minutes
Version: v1.2.3

Model: qwen-plus (alibaba)
Channels: web, feishu
Skills: 7 installed

Today's Usage:
  Messages: 45
  Tokens: 12,345
  API Calls: 23

Memory: 156MB
CPU: 0.5%

Endpoints:
  Web UI: http://localhost:18789
  API: http://localhost:18789/api/v1
```

**查看网关状态：**

```bash
openclaw gateway status
```

**输出示例：**

```
Gateway Status
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Status: 🟢 running
Mode: daemon

Model:
  Primary: qwen-plus (alibaba)
  Fallback: deepseek-chat (deepseek)

Connections:
  Active: 2
  Total today: 45

Performance:
  Avg response time: 1.2s
  Success rate: 99.2%
  Tokens/min: 234

Queue:
  Pending: 0
  Processing: 1
```

## 网关控制

**启动网关：**

```bash
# 前台启动（调试用）
openclaw gateway start

# 后台启动
openclaw gateway start --daemon

# 指定端口
openclaw gateway start --port 8080
```

**停止网关：**

```bash
openclaw gateway stop
```

**重启网关：**

```bash
openclaw gateway restart
```

**重载配置：**

```bash
# 不重启服务的情况下重载配置
openclaw gateway reload
```

## 模型管理

**查看可用模型：**

```bash
openclaw models list
```

**输出示例：**

```
Available Models
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Configured Models:
  🟢 qwen-plus (alibaba) - current
  🟢 deepseek-chat (deepseek) - fallback
  ⚪ gpt-4 (openai) - not configured

Supported Providers:
  • alibaba - 通义千问系列
  • deepseek - DeepSeek系列
  • openai - GPT系列
  • anthropic - Claude系列
  • google - Gemini系列
  • zhipu - 智谱GLM系列
```

**切换模型：**

```bash
# 切换主模型
openclaw models set qwen-max

# 设置备选模型
openclaw models set-fallback deepseek-chat

# 清除备选模型
openclaw models clear-fallback
```

**测试模型：**

```bash
# 测试当前模型连接
openclaw models test

# 测试指定模型
openclaw models test qwen-max
```

**查看模型配置：**

```bash
openclaw models show
```

**输出示例：**

```
Model Configuration
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Primary Model:
  Provider: alibaba
  Model: qwen-plus
  API Key: sk-****...****xxxx
  Base URL: https://dashscope.aliyuncs.com/compatible-mode/v1
  
  Parameters:
    Temperature: 0.7
    Max Tokens: 4096
    Top P: 0.9

Fallback Model:
  Provider: deepseek
  Model: deepseek-chat
  API Key: sk-****...****yyyy
```

## 渠道列表

**查看已配置渠道：**

```bash
openclaw channels list
```

**输出示例：**

```
Configured Channels
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Channel      Status      Messages Today    Last Active
─────────────────────────────────────────────────────
web          🟢 running  23               just now
feishu       🟢 running  15               5 min ago
telegram     ⚪ disabled  0               never
dingtalk     ⚪ disabled  0               never
```

**启用/禁用渠道：**

```bash
# 启用渠道
openclaw channels enable telegram

# 禁用渠道
openclaw channels disable dingtalk
```

**测试渠道：**

```bash
# 测试渠道连接
openclaw channels test feishu

# 发送测试消息
openclaw channels test feishu --send-message "测试消息"
```

## 记忆搜索

OpenClaw会记住你的对话和偏好，你可以搜索这些记忆。

**搜索记忆：**

```bash
# 搜索关键词
openclaw memory search "项目"

# 搜索特定时间范围
openclaw memory search "项目" --from "2026-03-01" --to "2026-03-28"

# 限制结果数量
openclaw memory search "项目" --limit 10
```

**输出示例：**

```
Memory Search Results for "项目"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Found 5 memories:

1. [2026-03-28 10:30] 用户偏好将项目文件存放在 ~/Projects/
   Source: conversation

2. [2026-03-27 15:20] 用户正在开发一个Python数据分析项目
   Source: conversation

3. [2026-03-26 09:15] 项目名称：my_app，使用FastAPI框架
   Source: user_preference

4. [2026-03-25 14:00] 用户提到项目部署在阿里云
   Source: conversation

5. [2026-03-24 11:30] 项目代码仓库：github.com/user/my_app
   Source: conversation
```

**查看所有记忆：**

```bash
# 查看所有记忆
openclaw memory list

# 按类型过滤
openclaw memory list --type preference
openclaw memory list --type conversation
```

**清除记忆：**

```bash
# 清除特定记忆
openclaw memory delete <memory_id>

# 清除所有记忆（危险操作）
openclaw memory clear --confirm
```

## 文档查询

OpenClaw内置了文档查询功能，可以快速查找帮助信息。

**查询文档：**

```bash
# 查询主题
openclaw docs "安装"
openclaw docs "配置"
openclaw docs "技能"

# 查看完整文档目录
openclaw docs --index
```

**输出示例：**

```
$ openclaw docs "安装"

OpenClaw 文档 - 安装
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

安装方式：

1. npm安装（推荐）
   npm install -g @openclaw/cli

2. Homebrew安装（MacOS）
   brew tap openclaw/tap
   brew install openclaw

3. Docker安装
   docker pull openclaw/openclaw:latest

系统要求：
  • Node.js >= 22.0.0
  • 内存 >= 4GB
  • 磁盘 >= 10GB

详细文档：https://docs.openclaw.ai/installation
```

## 其他常用命令

**版本信息：**

```bash
openclaw --version
openclaw -v
```

**帮助信息：**

```bash
# 查看帮助
openclaw --help
openclaw -h

# 查看子命令帮助
openclaw gateway --help
openclaw models --help
```

**日志查看：**

```bash
# 查看日志
openclaw logs

# 实时查看日志
openclaw logs -f

# 过滤日志
openclaw logs --filter "error"
openclaw logs --channel feishu

# 限制行数
openclaw logs --lines 100
```

**配置管理：**

```bash
# 查看配置路径
openclaw config path

# 编辑配置
openclaw config edit

# 验证配置
openclaw config validate

# 重置配置
openclaw config reset --confirm
```

---

掌握这些CLI命令，你就可以高效地管理OpenClaw了。下一章，我们来学习常见问题的排查方法。
