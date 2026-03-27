# 渠道配置方法

上一章我们概览了支持的消息平台，这一章详细介绍各平台的配置方法。

## Telegram Bot配置详解

**创建Bot完整流程：**

```bash
# 1. 在Telegram中搜索 @BotFather
# 2. 发送命令
/newbot

# 3. BotFather会提示你输入Bot名称
# 例如：OpenClaw助手

# 4. 输入Bot用户名（必须以bot结尾）
# 例如：my_openclaw_bot

# 5. 获取Token
# BotFather会返回类似：
# Done! Congratulations on your new bot...
# Use this token to access the HTTP API:
# 1234567890:ABCdefGHIjklMNOpqrsTUVwxyz-123456
# Keep your token secure...
```

**配置OpenClaw：**

```bash
# 方式一：命令行配置
openclaw channel add telegram \
  --token "1234567890:ABCdefGHIjklMNOpqrsTUVwxyz-123456" \
  --webhook-url "https://your-domain.com/webhook/telegram"

# 方式二：编辑配置文件
vim ~/.openclaw/config.yaml
```

```yaml
channels:
  telegram:
    enabled: true
    token: "${TELEGRAM_BOT_TOKEN}"
    webhook:
      enabled: true
      url: "https://your-domain.com/webhook/telegram"
      secret_token: "your_webhook_secret"  # 可选，用于验证请求
    
    # 消息设置
    parse_mode: "MarkdownV2"  # HTML, Markdown, MarkdownV2
    
    # 白名单
    allowlist:
      - 123456789  # 用户ID
    
    # 命令设置
    commands:
      - command: "start"
        description: "开始使用"
      - command: "help"
        description: "获取帮助"
      - command: "status"
        description: "查看状态"
```

**设置Webhook（生产环境必需）：**

```bash
# 如果你有公网域名和HTTPS证书
openclaw channel telegram set-webhook \
  --url "https://your-domain.com/webhook/telegram"

# 删除Webhook（使用轮询模式）
openclaw channel telegram delete-webhook
```

**测试Bot：**

```bash
# 发送测试消息
openclaw channel telegram test \
  --chat-id "123456789" \
  --message "Hello from OpenClaw!"

# 检查Bot状态
openclaw channel telegram status
```

## 飞书机器人配置详解

**创建飞书应用：**

```bash
# 1. 访问开发者后台
open https://open.feishu.cn/app

# 2. 点击"创建企业自建应用"
# 3. 填写应用信息：
#    - 应用名称：OpenClaw助手
#    - 应用描述：你的AI助手小龙虾
#    - 应用图标：上传图片
```

**配置权限：**

在"权限管理"页面添加：

```
必需权限：
✓ im:message:receive_as_bot - 接收消息
✓ im:message:send_as_bot - 发送消息
✓ im:message - 消息基础权限
✓ im:chat:read - 读取群组信息

可选权限：
○ im:chat.member:read - 读取群成员
○ im:resource - 上传下载资源
○ contact:user.base:readonly - 获取用户基本信息
```

**配置事件订阅：**

```yaml
# 在"事件订阅"页面配置：

请求地址：
http://你的服务器IP:18789/webhook/feishu

订阅事件：
✓ im.message.receive_v1 - 接收消息
○ im.message.message_read_v1 - 消息已读
○ im.chat.member.added_v1 - 群成员增加
```

**OpenClaw配置：**

```yaml
channels:
  feishu:
    enabled: true
    
    # 应用凭证
    app_id: "cli_a1b2c3d4e5f6g7h8"
    app_secret: "${FEISHU_APP_SECRET}"
    
    # 加密配置（推荐）
    encrypt_key: "${FEISHU_ENCRYPT_KEY}"
    verification_token: "${FEISHU_VERIFY_TOKEN}"
    
    # 消息设置
    message:
      # 自动回复确认
      auto_reply: true
      # 消息类型
      supported_types:
        - text
        - post
        - image
        - file
    
    # 白名单
    allowlist:
      - "ou_xxxxx"  # 用户open_id
      - "oc_xxxxx"  # 群组chat_id
```

**获取凭证：**

```bash
# 设置环境变量
export FEISHU_APP_SECRET="your_app_secret"
export FEISHU_ENCRYPT_KEY="your_encrypt_key"
export FEISHU_VERIFY_TOKEN="your_verify_token"

# 或使用OpenClaw凭证管理
openclaw credentials add feishu_app_secret "your_app_secret"
openclaw credentials add feishu_encrypt_key "your_encrypt_key"
```

**发布应用：**

```bash
# 1. 创建版本
# 在"版本管理与发布"页面点击"创建版本"

# 2. 填写版本信息
# 版本号：1.0.0
# 更新说明：首次发布

# 3. 提交审核（企业内部应用无需审核）

# 4. 发布
# 审核通过后点击"发布"

# 5. 添加到通讯录
# 在"应用可用范围"中添加部门或个人
```

## 企业微信应用配置详解

**创建应用：**

```bash
# 1. 登录企业微信管理后台
open https://work.weixin.qq.com/wework_admin/frame

# 2. 进入"应用管理" → "自建" → "创建应用"

# 3. 填写应用信息：
#    - 应用名称：OpenClaw助手
#    - 应用logo：上传图片
#    - 可见范围：选择部门或成员

# 4. 获取凭证：
#    - AgentId: 1000001
#    - Secret: xxxxxxxxxxxxxxxxxxxx
```

**配置API接收：**

```yaml
# 在应用详情页找到"API接收消息"

# 设置URL：
http://你的服务器IP:18789/webhook/wecom

# 设置Token（自定义，用于验证）：
mytoken123456

# 设置EncodingAESKey（点击随机获取）：
abcdefghijklmnopqrstuvwxyz0123456789ABCDEFG
```

**OpenClaw配置：**

```yaml
channels:
  wecom:
    enabled: true
    
    # 企业信息
    corp_id: "wxXXXXXXXXXXXXXXXX"
    
    # 应用信息
    agent_id: 1000001
    secret: "${WECOM_SECRET}"
    
    # 消息加密
    token: "${WECOM_TOKEN}"
    encoding_aes_key: "${WECOM_AES_KEY}"
    
    # 消息设置
    message:
      # 发送者显示名称
      sender_name: "OpenClaw助手"
      # 消息类型
      supported_types:
        - text
        - markdown
        - news
        - file
```

**获取企业ID：**

```bash
# 在"我的企业"页面底部
# 企业ID：wxXXXXXXXXXXXXXXXX
```

**配置可信IP：**

```bash
# 在"企业可信IP"中添加你的服务器IP
# 只有白名单IP才能调用API

# 查看服务器公网IP
curl ifconfig.me
```

## 多渠道同时配置

OpenClaw支持同时配置多个渠道：

```yaml
# ~/.openclaw/config.yaml

channels:
  # Web界面（默认启用）
  web:
    enabled: true
    port: 18789
    host: "0.0.0.0"
  
  # Telegram
  telegram:
    enabled: true
    token: "${TELEGRAM_BOT_TOKEN}"
  
  # 飞书
  feishu:
    enabled: true
    app_id: "cli_xxxxx"
    app_secret: "${FEISHU_APP_SECRET}"
  
  # 企业微信
  wecom:
    enabled: true
    corp_id: "wxXXXXXXXX"
    agent_id: 1000001
    secret: "${WECOM_SECRET}"
    token: "${WECOM_TOKEN}"
    encoding_aes_key: "${WECOM_AES_KEY}"
  
  # 钉钉
  dingtalk:
    enabled: false  # 暂不启用
    app_key: "dingxxxxx"
    app_secret: "${DINGTALK_SECRET}"
```

**管理多渠道：**

```bash
# 查看所有渠道状态
openclaw channel list

# 启用/禁用渠道
openclaw channel enable telegram
openclaw channel disable dingtalk

# 重启特定渠道
openclaw channel restart feishu

# 查看渠道日志
openclaw logs --channel telegram
```

---

配置完成后，你的OpenClaw就可以通过多个平台接收消息了。下一章，我们将学习如何管理多账户，让不同用户拥有独立的会话和配置。
