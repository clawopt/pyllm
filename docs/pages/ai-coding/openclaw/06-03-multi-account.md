# 多账户管理

如果你想让家人、同事或团队成员也能使用你的OpenClaw实例，多账户管理功能必不可少。这一章，我们学习如何配置和管理多用户场景。

## 账户体系概述

OpenClaw的账户体系分为三个层级：

```
┌─────────────────────────────────────────────────────┐
│                   OpenClaw实例                       │
├─────────────────────────────────────────────────────┤
│  账户1 (默认账户)                                    │
│  ├── 配置：SOUL.md, USER.md                         │
│  ├── 凭证：API Keys, 渠道凭证                        │
│  └── 数据：会话记录, 记忆数据                        │
├─────────────────────────────────────────────────────┤
│  账户2 (家庭成员)                                    │
│  ├── 配置：独立的SOUL.md, USER.md                   │
│  ├── 凭证：可共享或独立                              │
│  └── 数据：隔离的会话和记忆                          │
├─────────────────────────────────────────────────────┤
│  账户3 (同事)                                        │
│  └── ...                                            │
└─────────────────────────────────────────────────────┘
```

**关键概念：**

| 概念 | 说明 |
|------|------|
| 默认账户 | 首次运行时创建，拥有管理员权限 |
| 普通账户 | 由默认账户创建，权限受限 |
| 凭证共享 | API Key等可跨账户共享 |
| 数据隔离 | 会话、记忆等数据完全隔离 |

## 创建账户

**命令行创建：**

```bash
# 创建新账户
openclaw account create \
  --name "family_member" \
  --display-name "家庭成员" \
  --channels telegram,feishu \
  --permissions "chat,file_ops"

# 输出
Creating account...
✓ Account created: family_member
✓ Account ID: acc_xxxxxxxxxxxx
✓ API Key: sk-acc-xxxxxxxxxxxx

请保存API Key，它不会再次显示。
```

**交互式创建：**

```bash
openclaw account create --interactive

# 交互流程
账户名称：colleague_zhang
显示名称：张同事
允许的渠道（逗号分隔）：feishu
允许的权限（逗号分隔）：chat,file_ops,web_search
是否共享API Key？[Y/n] Y
设置使用限额？[y/N] n

✓ 账户创建成功
```

**配置文件创建：**

```yaml
# ~/.openclaw/accounts/colleague_zhang/config.yaml

account:
  id: "acc_xxxxxxxxxxxx"
  name: "colleague_zhang"
  display_name: "张同事"
  created_at: "2026-03-28T10:00:00Z"

permissions:
  - chat
  - file_ops
  - web_search

channels:
  - feishu

limits:
  daily_messages: 1000
  daily_tokens: 100000

credentials:
  use_shared: true  # 使用共享的API Key
```

## 账户管理

**查看账户列表：**

```bash
openclaw account list

# 输出
账户列表：
┌──────────────────┬─────────────┬──────────────┬─────────────┐
│ 账户名称         │ 显示名称    │ 渠道         │ 状态        │
├──────────────────┼─────────────┼──────────────┼─────────────┤
│ default (管理员) │ 默认账户    │ 全部         │ 🟢 活跃     │
│ family_member    │ 家庭成员    │ telegram     │ 🟢 活跃     │
│ colleague_zhang  │ 张同事      │ feishu       │ 🟢 活跃     │
│ test_user        │ 测试用户    │ web          │ ⚪ 已禁用   │
└──────────────────┴─────────────┴──────────────┴─────────────┘
```

**查看账户详情：**

```bash
openclaw account show family_member

# 输出
账户详情：family_member
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

基本信息：
  账户ID：acc_xxxxxxxxxxxx
  显示名称：家庭成员
  创建时间：2026-03-28 10:00:00
  状态：活跃

权限：
  ✓ chat - 对话
  ✓ file_ops - 文件操作
  ✗ browser - 浏览器自动化
  ✗ shell - 系统命令

渠道：
  ✓ telegram

使用统计（今日）：
  消息数：45
  Token数：12,345
  API调用：23次

限额：
  每日消息：1000（已用4.5%）
  每日Token：100,000（已用12.3%）
```

**修改账户配置：**

```bash
# 修改权限
openclaw account update family_member \
  --add-permission browser \
  --remove-permission file_ops

# 修改限额
openclaw account update family_member \
  --daily-messages 500 \
  --daily-tokens 50000

# 禁用/启用账户
openclaw account disable test_user
openclaw account enable test_user
```

**删除账户：**

```bash
# 删除账户（会删除所有数据）
openclaw account delete test_user --confirm

# 输出
⚠️ 警告：此操作将删除账户 test_user 的所有数据：
  - 会话记录：23条
  - 记忆数据：156条
  - 配置文件：1个

确认删除？[y/N] y

✓ 账户 test_user 已删除
```

## 凭证共享与隔离

**共享凭证（默认）：**

新账户默认使用共享的API Key，无需额外配置：

```yaml
# ~/.openclaw/config.yaml

credentials:
  shared:
    openai_api_key: "${OPENAI_API_KEY}"
    alibaba_api_key: "${ALIBABA_API_KEY}"
```

**独立凭证：**

为特定账户配置独立的API Key：

```bash
# 为账户设置独立凭证
openclaw account credentials set family_member \
  --provider alibaba \
  --api-key "sk-family-xxxx"
```

或编辑配置：

```yaml
# ~/.openclaw/accounts/family_member/config.yaml

credentials:
  use_shared: false
  providers:
    alibaba:
      api_key: "${FAMILY_ALIBABA_KEY}"
```

**凭证优先级：**

```
账户独立凭证 > 共享凭证 > 默认凭证
```

## 消息路由

当多个用户通过不同渠道发送消息时，OpenClaw需要正确路由消息到对应账户。

**自动路由：**

OpenClaw会根据消息来源自动识别账户：

```
Telegram用户 (ID: 123456789)
    → 匹配账户 family_member
    → 使用 family_member 的配置和权限

飞书用户 (open_id: ou_xxxxx)
    → 匹配账户 colleague_zhang
    → 使用 colleague_zhang 的配置和权限
```

**配置路由规则：**

```yaml
# ~/.openclaw/config.yaml

routing:
  # 自动路由规则
  rules:
    - channel: telegram
      user_id: "123456789"
      account: family_member
    
    - channel: feishu
      user_id: "ou_xxxxx"
      account: colleague_zhang
  
  # 默认账户（未匹配时使用）
  default_account: default
  
  # 拒绝未授权用户
  reject_unauthorized: true
```

**查看路由状态：**

```bash
openclaw routing list

# 输出
消息路由规则：
┌────────────┬──────────────────┬──────────────────┬─────────────┐
│ 渠道       │ 用户标识         │ 账户             │ 状态        │
├────────────┼──────────────────┼──────────────────┼─────────────┤
│ telegram   │ 123456789        │ family_member    │ 🟢 匹配     │
│ feishu     │ ou_xxxxx         │ colleague_zhang  │ 🟢 匹配     │
│ feishu     │ ou_yyyyy         │ (未授权)         │ 🔴 拒绝     │
│ *          │ *                │ default          │ ⚪ 默认     │
└────────────┴──────────────────┴──────────────────┴─────────────┘
```

## 会话隔离

每个账户的会话数据完全隔离，互不影响。

**会话存储结构：**

```
~/.openclaw/
├── accounts/
│   ├── default/
│   │   └── sessions/
│   │       └── 2026-03-28/
│   ├── family_member/
│   │   └── sessions/
│   │       └── 2026-03-28/
│   └── colleague_zhang/
│       └── sessions/
│           └── 2026-03-28/
```

**会话管理：**

```bash
# 查看账户会话
openclaw sessions list --account family_member

# 清除账户会话
openclaw sessions clear --account family_member

# 导出账户会话
openclaw sessions export --account family_member --output family_sessions.json
```

**记忆隔离：**

每个账户有独立的记忆存储：

```bash
# 查看账户记忆
openclaw memory show --account family_member

# 清除账户记忆
openclaw memory clear --account family_member
```

## 使用限额

为每个账户设置使用限额，防止资源滥用。

**配置限额：**

```yaml
# ~/.openclaw/accounts/family_member/config.yaml

limits:
  # 每日消息数
  daily_messages: 1000
  
  # 每日Token数
  daily_tokens: 100000
  
  # 每日API调用次数
  daily_api_calls: 500
  
  # 并发会话数
  concurrent_sessions: 3
  
  # 文件操作限制
  file_ops:
    max_file_size: 10MB
    daily_deletes: 10
```

**查看使用情况：**

```bash
openclaw account usage family_member --today

# 输出
账户使用情况：family_member（今日）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

消息：
  已发送：45 / 1000 (4.5%)

Token：
  已使用：12,345 / 100,000 (12.3%)

API调用：
  已调用：23 / 500 (4.6%)

文件操作：
  读取：12次
  写入：5次
  删除：0次（限额10次）

预估费用：¥0.12
```

**超限处理：**

```yaml
# 超限时的处理方式
limits:
  on_exceed:
    action: "reject"  # reject(拒绝) 或 warn(警告)
    message: "今日使用额度已达上限，请明天再试"
```

---

通过多账户管理，你可以安全地与他人共享OpenClaw实例，同时保持数据和权限的隔离。至此，我们已经完成了消息渠道配置的学习。接下来，你可以开始探索OpenClaw的更多高级功能了。
