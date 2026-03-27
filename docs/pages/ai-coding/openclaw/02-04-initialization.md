# 初始化配置

部署完成后，OpenClaw还不能直接使用——它需要知道使用哪个大模型、通过什么渠道接收消息、以及你的个人偏好设置。这一章，我们通过Onboarding向导完成这些初始化配置。

## 运行Onboarding向导

OpenClaw提供了一个交互式的初始化向导，会引导你完成所有必要的配置。

```bash
# 启动向导（推荐首次使用）
openclaw onboard --install-daemon
```

这个命令会：

1. 创建必要的目录结构
2. 引导你配置大模型API
3. 设置通信渠道
4. 安装后台常驻服务
5. 验证配置正确性

**向导交互流程：**

```
$ openclaw onboard --install-daemon

Welcome to OpenClaw! This wizard will help you set up your AI assistant.

Step 1/5: Workspace Setup
─────────────────────────
Creating workspace directory: /home/user/openclaw/workspace
✓ Workspace created

Step 2/5: Model Configuration
─────────────────────────────
Select your LLM provider:
  1) OpenAI (GPT-4, GPT-3.5)
  2) Alibaba Cloud (Qwen series)
  3) DeepSeek
  4) Zhipu AI (GLM series)
  5) Anthropic (Claude)
  6) Custom (OpenAI-compatible API)

Enter choice [1-6]: 2

You selected: Alibaba Cloud (Qwen series)
Enter your API Key: ************

Select model:
  1) qwen-max (Recommended for complex tasks)
  2) qwen-plus (Balanced performance and cost)
  3) qwen-turbo (Fast and economical)

Enter choice [1-3]: 2
✓ Model configured: qwen-plus

Step 3/5: Channel Configuration
───────────────────────────────
Select communication channels (space to select, enter to confirm):
  [ ] Telegram
  [x] Feishu (Lark)
  [ ] WeChat Work
  [ ] DingTalk
  [ ] Discord
  [ ] Web UI only

You selected: Feishu

Configure Feishu:
  App ID: cli_xxxxxxxxxxxx
  App Secret: ************

✓ Feishu channel configured

Step 4/5: Service Installation
──────────────────────────────
Installing OpenClaw as a system service...
✓ Service installed: openclaw.service
✓ Service started

Step 5/5: Verification
──────────────────────
Running diagnostics...
✓ API Key valid
✓ Model accessible
✓ Channel connected
✓ Service running

🎉 OpenClaw is ready!

Access your assistant:
  Web UI: http://localhost:18789
  Feishu: Search "OpenClaw" in Feishu

Commands:
  openclaw status    - Check service status
  openclaw logs      - View logs
  openclaw stop      - Stop service
  openclaw restart   - Restart service
```

## 配置大模型API Key

如果你跳过了向导，或者需要修改模型配置，可以手动编辑配置文件。

**配置文件位置：**

```
~/openclaw/
├── config.yaml          # 主配置文件
├── credentials/         # 加密存储的凭证
└── workspace/           # 工作数据
```

**编辑配置文件：**

```bash
# 使用内置编辑命令
openclaw config edit

# 或直接编辑
vim ~/openclaw/config.yaml
```

**配置示例：**

```yaml
# ~/openclaw/config.yaml

model:
  provider: alibaba      # openai, alibaba, deepseek, anthropic, custom
  model: qwen-plus       # 具体模型名称
  api_key: ${ALIBABA_API_KEY}  # 建议使用环境变量
  base_url: https://dashscope.aliyuncs.com/compatible-mode/v1  # 可选，自定义端点
  
  # 模型参数
  temperature: 0.7
  max_tokens: 4096
  
  # 备用模型（可选）
  fallback:
    provider: deepseek
    model: deepseek-chat
    api_key: ${DEEPSEEK_API_KEY}

# 多模型配置（高级用法）
models:
  reasoning:
    provider: openai
    model: gpt-4
  chat:
    provider: deepseek
    model: deepseek-chat
  local:
    provider: ollama
    model: qwen2:7b
    base_url: http://localhost:11434
```

**使用环境变量存储API Key（推荐）：**

```bash
# 添加到 ~/.bashrc 或 ~/.zshrc
export ALIBABA_API_KEY="sk-xxxxxxxxxxxx"
export OPENAI_API_KEY="sk-xxxxxxxxxxxx"
export DEEPSEEK_API_KEY="sk-xxxxxxxxxxxx"

# 使配置生效
source ~/.bashrc
```

**验证模型配置：**

```bash
# 测试API连接
openclaw test model

# 预期输出
Testing model connection...
Provider: alibaba
Model: qwen-plus
API Key: sk-****...****xxxx

Sending test request...
✓ Connection successful
✓ Model responded: "Hello! I'm Qwen, how can I help you?"
```

## 设置通信渠道

OpenClaw支持多种通信渠道，你可以配置一个或多个。

**飞书配置：**

```yaml
# ~/openclaw/config.yaml

channels:
  feishu:
    enabled: true
    app_id: "cli_xxxxxxxxxx"
    app_secret: "${FEISHU_APP_SECRET}"
    encrypt_key: "${FEISHU_ENCRYPT_KEY}"  # 可选，用于消息加密
    verification_token: "${FEISHU_VERIFY_TOKEN}"  # 可选
```

获取飞书应用凭证：

1. 访问 https://open.feishu.cn/app
2. 创建企业自建应用
3. 在"凭证与基础信息"获取App ID和App Secret
4. 配置事件订阅，URL填写：`http://你的服务器IP:18789/webhook/feishu`
5. 添加权限：`im:message`、`im:message:send_as_bot`

**Telegram配置：**

```yaml
channels:
  telegram:
    enabled: true
    bot_token: "${TELEGRAM_BOT_TOKEN}"
```

获取Telegram Bot Token：

1. 在Telegram中搜索 @BotFather
2. 发送 `/newbot` 创建新机器人
3. 按提示设置名称
4. 获取Bot Token

**企业微信配置：**

```yaml
channels:
  wecom:
    enabled: true
    corp_id: "wxXXXXXXXX"
    agent_id: 1000001
    secret: "${WECOM_SECRET}"
    token: "${WECOM_TOKEN}"      # 可选
    encoding_aes_key: "${WECOM_AES_KEY}"  # 可选
```

**钉钉配置：**

```yaml
channels:
  dingtalk:
    enabled: true
    app_key: "dingxxxxxxxxx"
    app_secret: "${DINGTALK_SECRET}"
```

**Web UI配置：**

Web UI默认启用，无需额外配置。如需自定义：

```yaml
channels:
  web:
    enabled: true
    port: 18789
    host: "0.0.0.0"  # 绑定地址
    auth:
      enabled: true
      username: "admin"
      password: "${WEB_PASSWORD}"
```

**启用认证（强烈推荐）：**

如果你将OpenClaw暴露在公网，务必启用认证：

```yaml
auth:
  enabled: true
  type: basic  # basic, jwt, oauth
  users:
    - username: admin
      password_hash: "$2a$10$..."  # bcrypt hash
```

生成密码hash：

```bash
openclaw util hash-password "your_password"
```

## 安装后台常驻服务

OpenClaw需要作为后台服务持续运行。向导会自动安装，你也可以手动操作。

**MacOS (launchd)：**

```bash
# 安装服务
openclaw service install

# 启动服务
openclaw service start

# 查看状态
openclaw service status

# 停止服务
openclaw service stop

# 卸载服务
openclaw service uninstall
```

服务文件位置：`~/Library/LaunchAgents/com.openclaw.plist`

**Linux (systemd)：**

```bash
# 安装服务（需要sudo）
sudo openclaw service install --system

# 启动服务
sudo systemctl start openclaw

# 开机自启
sudo systemctl enable openclaw

# 查看状态
sudo systemctl status openclaw

# 查看日志
journalctl -u openclaw -f
```

服务文件位置：`/etc/systemd/system/openclaw.service`

**Windows (nssm)：**

```powershell
# 安装服务（需要管理员权限）
openclaw service install

# 启动服务
net start OpenClaw

# 停止服务
net stop OpenClaw
```

## 验证运行状态

完成所有配置后，进行最终验证。

**使用doctor命令：**

```bash
openclaw doctor
```

**预期输出：**

```
Running OpenClaw diagnostics...

System
  ✓ Node.js: v22.1.0
  ✓ Platform: linux x64
  ✓ Memory: 7.8GB available

OpenClaw
  ✓ Version: v1.2.3
  ✓ Workspace: /home/user/openclaw/workspace
  ✓ Config: /home/user/openclaw/config.yaml

Model
  ✓ Provider: alibaba
  ✓ Model: qwen-plus
  ✓ API Key: configured
  ✓ Connection: OK (latency: 245ms)

Channels
  ✓ Feishu: connected
    - App ID: cli_xxxxx
    - Events: message, message_read
  ✓ Web UI: running on http://0.0.0.0:18789

Service
  ✓ Status: running
  ✓ PID: 12345
  ✓ Uptime: 2 hours, 15 minutes
  ✓ Memory: 156MB
  ✓ CPU: 0.5%

Storage
  ✓ Workspace: 23MB used
  ✓ Logs: 5MB used
  ✓ Credentials: encrypted

All checks passed! OpenClaw is healthy.
```

**查看服务状态：**

```bash
openclaw status
```

**输出示例：**

```
OpenClaw Status
───────────────
Service: running (PID: 12345)
Uptime: 2h 15m 32s
Memory: 156 MB
CPU: 0.5%

Model: qwen-plus (alibaba)
Channels: feishu, web
Requests today: 47
Tokens used: 125,432

Web UI: http://localhost:18789
API: http://localhost:18789/api/v1
```

**查看实时日志：**

```bash
openclaw logs -f
```

**日志示例：**

```
[2026-03-28 10:30:15] INFO  [gateway] Received message from feishu
[2026-03-28 10:30:15] INFO  [gateway] Intent: file_management
[2026-03-28 10:30:16] INFO  [skills] Executing: file_ops.scan
[2026-03-28 10:30:17] INFO  [skills] Found 23 files in Downloads
[2026-03-28 10:30:18] INFO  [gateway] Response sent to feishu
[2026-03-28 10:30:18] INFO  [gateway] Tokens: 234 in, 156 out
```

**测试对话：**

```bash
# 命令行测试
openclaw chat "你好，请介绍一下你自己"

# 预期输出
你好！我是OpenClaw，你的个人AI助手。我可以帮你：
- 管理文件和整理文档
- 自动化浏览器操作
- 编写和执行代码
- 管理日程和提醒
- 收集和汇总信息

有什么我可以帮你的吗？
```

---

至此，OpenClaw已经完全配置完成，可以正常使用了。你可以：

- 打开浏览器访问 `http://你的IP:18789` 使用Web界面
- 在飞书/钉钉/企微中找到你的机器人开始对话
- 使用命令行 `openclaw chat "你的问题"` 进行交互

下一章开始，我们将深入OpenClaw的各项功能，学习如何让它真正成为你的效率利器。
