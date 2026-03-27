# 支持的消息平台

OpenClaw的强大之处在于你可以通过各种消息平台与它交互，而不局限于Web界面。这一章，我们来了解OpenClaw支持的消息平台。

## 平台概览

OpenClaw支持两大类消息平台：

**国际平台：**

| 平台 | 特点 | 适用场景 |
|------|------|---------|
| Telegram | 机器人生态成熟，配置简单 | 个人使用、海外用户 |
| WhatsApp | 用户基数大，普及率高 | 个人助理、客户服务 |
| Discord | 社区功能强，支持频道 | 开发者社区、团队协作 |
| Slack | 企业级协作，集成丰富 | 团队办公、企业用户 |
| iMessage | 苹果生态原生支持 | 苹果用户个人使用 |

**国内平台：**

| 平台 | 特点 | 适用场景 |
|------|------|---------|
| 飞书 | 企业协作，功能全面 | 企业办公、团队协作 |
| 钉钉 | 阿里生态，企业普及 | 企业办公、国内企业 |
| 企业微信 | 微信生态，用户熟悉 | 企业办公、客户服务 |
| QQ | 年轻用户多，群功能强 | 社区运营、个人使用 |

## 平台对比

**配置难度：**

```
简单 ←─────────────────────────────────→ 复杂
Telegram | Discord | 飞书 | 钉钉 | 企微 | WhatsApp
```

**功能完整度：**

```
完整 ←─────────────────────────────────→ 基础
飞书 | 钉钉 | Discord | Telegram | 企微 | WhatsApp
```

**推荐选择：**

| 用户类型 | 推荐平台 | 理由 |
|---------|---------|------|
| 个人用户（海外） | Telegram | 配置最简单，功能完整 |
| 个人用户（国内） | 飞书 | 免费，功能全面，体验好 |
| 企业用户 | 钉钉/企微 | 与现有办公系统集成 |
| 开发者 | Discord | 社区功能强，适合技术交流 |
| 团队协作 | Slack | 企业级集成，功能丰富 |

## Telegram配置

Telegram是配置最简单的平台，适合个人用户快速上手。

**步骤一：创建Bot**

1. 在Telegram中搜索 `@BotFather`
2. 发送 `/newbot` 命令
3. 按提示设置Bot名称（如：`MyOpenClawBot`）
4. 设置Bot用户名（必须以`bot`结尾，如：`my_openclaw_bot`）
5. 获取Bot Token（格式：`123456789:ABCdefGHIjklMNOpqrsTUVwxyz`）

**步骤二：配置OpenClaw**

```bash
# 添加Telegram渠道
openclaw channel add telegram \
  --token "123456789:ABCdefGHIjklMNOpqrsTUVwxyz"
```

或编辑配置文件：

```yaml
# ~/.openclaw/config.yaml

channels:
  telegram:
    enabled: true
    token: "${TELEGRAM_BOT_TOKEN}"
```

**步骤三：启动Bot**

```bash
# 重启服务
openclaw restart

# 或启动时指定渠道
openclaw start --channel telegram
```

**步骤四：开始对话**

在Telegram中搜索你的Bot，发送 `/start` 开始对话。

**Telegram特殊功能：**

```bash
# 设置Bot命令菜单
openclaw channel telegram set-commands \
  "help:获取帮助" \
  "status:查看状态" \
  "clear:清除对话"

# 设置Bot描述
openclaw channel telegram set-description "我是你的AI助手小龙虾"

# 设置欢迎消息
openclaw channel telegram set-welcome "欢迎使用OpenClaw！发送任何消息开始对话。"
```

## 飞书配置

飞书是国内用户的首选平台，功能全面且免费。

**步骤一：创建飞书应用**

1. 访问 https://open.feishu.cn/app
2. 点击"创建企业自建应用"
3. 填写应用名称（如：OpenClaw助手）
4. 选择应用图标
5. 创建完成后获取 `App ID` 和 `App Secret`

**步骤二：配置权限**

在应用管理页面，添加以下权限：

| 权限 | 用途 |
|------|------|
| `im:message` | 接收消息 |
| `im:message:send_as_bot` | 以机器人身份发送消息 |
| `im:message:read` | 读取消息内容 |
| `im:chat` | 获取群组信息 |

**步骤三：配置事件订阅**

1. 在"事件订阅"页面开启订阅
2. 配置请求地址：
   ```
   http://你的服务器IP:18789/webhook/feishu
   ```
3. 订阅以下事件：
   - `im.message.receive_v1`（接收消息）

**步骤四：配置OpenClaw**

```bash
openclaw channel add feishu \
  --app-id "cli_xxxxxxxxxx" \
  --app-secret "your_app_secret"
```

或编辑配置文件：

```yaml
# ~/.openclaw/config.yaml

channels:
  feishu:
    enabled: true
    app_id: "cli_xxxxxxxxxx"
    app_secret: "${FEISHU_APP_SECRET}"
    encrypt_key: "${FEISHU_ENCRYPT_KEY}"  # 可选
    verification_token: "${FEISHU_VERIFY_TOKEN}"  # 可选
```

**步骤五：发布应用**

1. 在"版本管理与发布"页面创建版本
2. 提交审核（企业内部应用无需审核）
3. 发布后，在飞书中搜索应用名称即可使用

**飞书特殊功能：**

```bash
# 发送富文本卡片
openclaw channel feishu send-card \
  --to "ou_xxxxx" \
  --title "任务完成" \
  --content "已处理23个文件"

# 发送消息到群组
openclaw channel feishu send \
  --to "oc_xxxxx" \
  --message "大家好！"
```

## 钉钉配置

钉钉是阿里生态的企业办公平台，在国内企业中普及率高。

**步骤一：创建钉钉应用**

1. 访问 https://open-dev.dingtalk.com/
2. 选择"应用开发" → "企业内部开发"
3. 点击"创建应用"
4. 填写应用信息
5. 获取 `AppKey` 和 `AppSecret`

**步骤二：配置权限**

添加以下权限：

| 权限 | 用途 |
|------|------|
| `qyapi_get_member` | 获取用户信息 |
| `qyapi_get_dept` | 获取部门信息 |
| `qyapi_message_corp_send` | 发送企业消息 |

**步骤三：配置回调地址**

1. 在"开发管理" → "消息推送"中配置
2. 设置HTTP回调地址：
   ```
   http://你的服务器IP:18789/webhook/dingtalk
   ```

**步骤四：配置OpenClaw**

```yaml
# ~/.openclaw/config.yaml

channels:
  dingtalk:
    enabled: true
    app_key: "dingxxxxxxxxx"
    app_secret: "${DINGTALK_APP_SECRET}"
    agent_id: "123456789"
```

**钉钉特殊功能：**

```bash
# 发送工作通知
openclaw channel dingtalk send-work-notice \
  --to "user123" \
  --message "您有一个待审批任务"

# @指定用户
openclaw channel dingtalk send \
  --to "chat123" \
  --message "@user123 请查看"
```

## 企业微信配置

企业微信与微信生态打通，用户接受度高。

**步骤一：创建应用**

1. 访问 https://work.weixin.qq.com/wework_admin/frame
2. 进入"应用管理" → "自建"
3. 点击"创建应用"
4. 填写应用信息
5. 获取 `AgentId` 和 `Secret`

**步骤二：配置API接收**

1. 在应用详情页找到"API接收消息"
2. 设置URL：
   ```
   http://你的服务器IP:18789/webhook/wecom
   ```
3. 设置Token和EncodingAESKey（用于消息加密）

**步骤三：配置可信域名**

在"企业可信IP"中添加你的服务器IP。

**步骤四：配置OpenClaw**

```yaml
# ~/.openclaw/config.yaml

channels:
  wecom:
    enabled: true
    corp_id: "wxXXXXXXXX"
    agent_id: 1000001
    secret: "${WECOM_SECRET}"
    token: "${WECOM_TOKEN}"
    encoding_aes_key: "${WECOM_AES_KEY}"
```

**企业微信特殊功能：**

```bash
# 发送文本卡片
openclaw channel wecom send-card \
  --to "user123" \
  --title "审批通知" \
  --description "您有一个待审批任务" \
  --url "https://example.com/approval"
```

## 配对白名单设置

为了安全，你可以设置配对白名单，只允许特定用户使用你的OpenClaw。

**Telegram白名单：**

```yaml
channels:
  telegram:
    enabled: true
    token: "${TELEGRAM_BOT_TOKEN}"
    allowlist:
      - 123456789  # 用户ID
      - 987654321
```

获取用户ID：

```bash
# 用户发送任意消息后
openclaw logs --filter "telegram" | grep "user_id"
```

**飞书白名单：**

```yaml
channels:
  feishu:
    enabled: true
    app_id: "cli_xxxxx"
    app_secret: "${FEISHU_APP_SECRET}"
    allowlist:
      - "ou_xxxxx"  # 用户open_id
      - "ou_yyyyy"
```

**通配符支持：**

```yaml
allowlist:
  - "ou_*"  # 允许所有企业成员
  - "ou_manager_*"  # 允许特定前缀的用户
```

---

配置好消息渠道后，你就可以随时随地通过熟悉的聊天工具与OpenClaw交互了。下一章，我们将学习多账户管理，让OpenClaw能够服务多个用户。
