# 配置文件详解

OpenClaw的所有配置都存储在用户目录下的`.openclaw`文件夹中。理解这些配置文件的位置和作用，是定制化你的AI助手的第一步。

## 配置文件位置

OpenClaw的主配置文件位于用户主目录下：

```
~/.openclaw/
├── openclaw.json          # 主配置文件
├── workspace/             # 工作区目录
├── credentials/           # 凭证存储
├── agents/                # Agent配置
│   └── default/
│       ├── SOUL.md        # AI身份定义
│       ├── USER.md        # 用户偏好
│       └── sessions/      # 会话记录
├── logs/                  # 日志文件
└── cache/                 # 缓存数据
```

**查看配置目录：**

```bash
# 列出配置目录结构
tree ~/.openclaw

# 或使用openclaw命令
openclaw config path
# 输出：/home/user/.openclaw
```

**主配置文件结构：**

`openclaw.json`是OpenClaw的核心配置文件，采用JSON格式：

```json
{
  "version": "1.2.3",
  "gateway": {
    "model": {
      "provider": "alibaba",
      "name": "qwen-plus",
      "apiKey": "${ALIBABA_API_KEY}",
      "baseUrl": "https://dashscope.aliyuncs.com/compatible-mode/v1",
      "temperature": 0.7,
      "maxTokens": 4096
    },
    "memory": {
      "enabled": true,
      "maxHistory": 100,
      "summaryThreshold": 50
    },
    "planning": {
      "maxSteps": 10,
      "timeout": 300000
    }
  },
  "channels": {
    "web": {
      "enabled": true,
      "port": 18789,
      "host": "0.0.0.0"
    },
    "feishu": {
      "enabled": true,
      "appId": "cli_xxxxx",
      "appSecret": "${FEISHU_APP_SECRET}"
    }
  },
  "skills": {
    "enabled": ["file_ops", "browser", "calendar", "github"],
    "config": {
      "browser": {
        "headless": true,
        "userDataDir": "~/.openclaw/browser_data"
      }
    }
  },
  "logging": {
    "level": "info",
    "file": "~/.openclaw/logs/openclaw.log",
    "maxSize": "10MB",
    "maxFiles": 5
  }
}
```

**编辑配置文件：**

```bash
# 使用内置编辑器（推荐）
openclaw config edit

# 直接编辑
vim ~/.openclaw/openclaw.json

# 使用jq工具修改特定字段
jq '.gateway.model.temperature = 0.5' ~/.openclaw/openclaw.json > tmp.json && mv tmp.json ~/.openclaw/openclaw.json
```

**配置验证：**

```bash
# 验证配置文件格式
openclaw config validate

# 预期输出
✓ openclaw.json: valid JSON
✓ All required fields present
✓ API keys configured
✓ No deprecated settings
```

## 工作区目录

`workspace`目录是OpenClaw执行任务时的工作空间，存储临时文件、任务输出、缓存数据等。

**目录结构：**

```
~/.openclaw/workspace/
├── temp/                  # 临时文件
├── downloads/             # 下载文件
├── exports/               # 导出文件
├── cache/                 # 任务缓存
└── tasks/                 # 任务记录
    ├── 2026-03-28/
    │   ├── task_001.json
    │   ├── task_002.json
    │   └── ...
    └── ...
```

**任务记录文件示例：**

```json
// ~/.openclaw/workspace/tasks/2026-03-28/task_001.json
{
  "taskId": "task_20260328_001",
  "createdAt": "2026-03-28T10:30:00Z",
  "status": "completed",
  "userInput": "帮我整理下载文件夹",
  "plan": [
    {"step": 1, "action": "scan", "description": "扫描下载目录"},
    {"step": 2, "action": "classify", "description": "分类文件"},
    {"step": 3, "action": "move", "description": "移动到目标目录"}
  ],
  "execution": [
    {"step": 1, "status": "success", "duration": 0.5, "output": "找到23个文件"},
    {"step": 2, "status": "success", "duration": 1.2, "output": "分为5个类别"},
    {"step": 3, "status": "success", "duration": 2.3, "output": "移动完成"}
  ],
  "result": "已整理23个文件到5个分类目录",
  "tokens": {"input": 234, "output": 156}
}
```

**工作区管理：**

```bash
# 查看工作区大小
openclaw workspace size
# 输出：Workspace size: 156MB

# 清理临时文件
openclaw workspace clean --temp

# 清理缓存
openclaw workspace clean --cache

# 清理旧任务记录（保留最近30天）
openclaw workspace clean --tasks --keep-days 30

# 完全重置工作区（危险操作）
openclaw workspace reset
```

**自定义工作区位置：**

如果你的系统盘空间有限，可以将工作区移动到其他位置：

```bash
# 停止服务
openclaw stop

# 移动工作区
mv ~/.openclaw/workspace /data/openclaw_workspace

# 创建软链接
ln -s /data/openclaw_workspace ~/.openclaw/workspace

# 或在配置文件中指定
jq '.workspace.path = "/data/openclaw_workspace"' ~/.openclaw/openclaw.json
```

## 凭证存储

`credentials`目录用于安全存储敏感信息，如API密钥、密码、Token等。

**目录结构：**

```
~/.openclaw/credentials/
├── api_keys.enc           # 加密的API密钥
├── passwords.enc          # 加密的密码
├── tokens.enc             # 加密的Token
└── master.key             # 主密钥（首次运行时生成）
```

**安全机制：**

OpenClaw使用AES-256-GCM加密算法保护凭证：

1. 首次运行时生成随机主密钥（master.key）
2. 主密钥用于加密所有凭证文件
3. 主密钥本身通过用户密码派生（如果设置了密码）

**凭证管理命令：**

```bash
# 添加API密钥
openclaw credentials add api_key openai sk-xxxx

# 查看已存储的凭证（不显示值）
openclaw credentials list
# 输出：
# api_keys:
#   - openai (added: 2026-03-28)
#   - alibaba (added: 2026-03-28)

# 获取凭证值（需要密码）
openclaw credentials get api_key openai

# 删除凭证
openclaw credentials delete api_key openai

# 导出凭证（加密备份）
openclaw credentials export > credentials_backup.enc

# 导入凭证
openclaw credentials import < credentials_backup.enc
```

**在配置中使用凭证：**

```json
{
  "gateway": {
    "model": {
      "apiKey": "${ALIBABA_API_KEY}"  // 引用环境变量
    }
  }
}
```

或使用凭证存储：

```json
{
  "gateway": {
    "model": {
      "apiKeyRef": "api_key:alibaba"  // 引用凭证存储
    }
  }
}
```

**安全建议：**

- 不要将`credentials/`目录提交到版本控制
- 定期更换API密钥
- 设置强密码保护主密钥
- 备份主密钥到安全位置

## 会话记录

OpenClaw会保存每次对话的完整记录，用于上下文理解和记忆功能。

**目录结构：**

```
~/.openclaw/agents/
├── default/               # 默认Agent
│   ├── SOUL.md
│   ├── USER.md
│   └── sessions/
│       ├── 2026-03-28/
│       │   ├── session_001.json
│       │   ├── session_002.json
│       │   └── ...
│       └── summary.json   # 会话摘要
├── code_assistant/        # 自定义Agent：代码助手
│   ├── SOUL.md
│   ├── USER.md
│   └── sessions/
└── content_writer/        # 自定义Agent：内容小编
    ├── SOUL.md
    ├── USER.md
    └── sessions/
```

**会话记录格式：**

```json
// ~/.openclaw/agents/default/sessions/2026-03-28/session_001.json
{
  "sessionId": "sess_20260328_001",
  "channel": "feishu",
  "userId": "ou_xxxxx",
  "startTime": "2026-03-28T10:30:00Z",
  "endTime": "2026-03-28T10:35:00Z",
  "messages": [
    {
      "role": "user",
      "content": "帮我整理下载文件夹",
      "timestamp": "2026-03-28T10:30:00Z"
    },
    {
      "role": "assistant",
      "content": "好的，我来帮你整理。首先扫描下载目录...",
      "timestamp": "2026-03-28T10:30:01Z",
      "metadata": {
        "taskId": "task_20260328_001",
        "tokens": {"input": 234, "output": 156}
      }
    },
    {
      "role": "user",
      "content": "好的，执行吧",
      "timestamp": "2026-03-28T10:30:15Z"
    },
    {
      "role": "assistant",
      "content": "已完成整理，共处理23个文件...",
      "timestamp": "2026-03-28T10:30:30Z"
    }
  ],
  "summary": "用户请求整理下载文件夹，AI扫描并分类了23个文件"
}
```

**会话管理命令：**

```bash
# 查看最近的会话
openclaw sessions list --recent 10

# 搜索会话内容
openclaw sessions search "整理文件"

# 导出会话记录
openclaw sessions export --format markdown > sessions.md

# 清理旧会话（保留最近100条）
openclaw sessions clean --keep 100

# 查看会话统计
openclaw sessions stats
# 输出：
# Total sessions: 1234
# Total messages: 5678
# Average session length: 4.6 messages
# Most active day: 2026-03-15 (45 sessions)
```

**会话摘要：**

OpenClaw会自动生成会话摘要，用于长期记忆：

```json
// ~/.openclaw/agents/default/sessions/summary.json
{
  "lastUpdated": "2026-03-28T10:35:00Z",
  "totalSessions": 1234,
  "keyMemories": [
    {
      "date": "2026-03-28",
      "content": "用户偏好将下载文件按类型分类存储",
      "importance": 0.8
    },
    {
      "date": "2026-03-27",
      "content": "用户常用Python进行数据分析",
      "importance": 0.7
    }
  ],
  "userPatterns": [
    "经常在上午10点-12点使用",
    "偏好简洁的回复风格",
    "常用功能：文件管理、代码编写"
  ]
}
```

---

理解了配置文件的结构，接下来我们可以开始定制OpenClaw。下一章，我们将学习如何定义AI的身份和回答风格。
