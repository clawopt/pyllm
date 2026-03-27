# API Key配置

选择了合适的模型提供商后，下一步是获取并配置API Key。这一章，我们详细介绍各厂商的API Key获取流程和配置方法。

## 阿里云百炼Coding Plan

阿里云百炼是OpenClaw国内用户的首选平台，提供了专门的Coding Plan订阅方案。

### 订阅方案

**免费版：**
- 每月100万tokens免费额度
- 支持Qwen-Turbo、Qwen-Plus
- 适合个人体验

**Coding Plan（推荐）：**
- 价格：¥9.9/月
- 每月500万tokens额度
- 支持所有Qwen模型
- 优先响应速度
- 适合日常开发使用

**专业版：**
- 价格：¥99/月
- 每月5000万tokens额度
- 支持所有模型
- 企业级支持
- 适合团队使用

### 获取API Key

**步骤一：注册阿里云账号**

1. 访问 https://www.aliyun.com/
2. 点击"免费注册"
3. 使用手机号完成注册
4. 完成实名认证（需要身份证）

**步骤二：开通百炼服务**

1. 访问 https://bailian.console.aliyun.com/
2. 点击"立即开通"
3. 选择订阅方案（建议先选免费版体验）
4. 确认开通

**步骤三：创建API Key**

1. 进入百炼控制台
2. 点击左侧"API-KEY管理"
3. 点击"创建 API Key"
4. 输入Key名称（如：openclaw-main）
5. 复制保存API Key（**只显示一次**）

```
⚠️ 重要提示
API Key创建后只显示一次，请务必保存到安全的地方。
如果忘记，只能删除重建。
```

**步骤四：领取代金券**

阿里云经常有新用户活动：

1. 访问 https://www.aliyun.com/activity
2. 搜索"百炼"或"大模型"
3. 领取新用户代金券
4. 常见活动：
   - 新用户送100万tokens
   - 首月Coding Plan半价

### 配置到OpenClaw

**方式一：使用命令配置**

```bash
# 设置模型
openclaw models set qwen-plus --api-key sk-xxxxxxxxxxxx

# 验证配置
openclaw models test
```

**方式二：编辑配置文件**

```bash
# 编辑配置
openclaw config edit
```

添加以下内容：

```json
{
  "gateway": {
    "model": {
      "provider": "alibaba",
      "model": "qwen-plus",
      "apiKey": "sk-xxxxxxxxxxxx",
      "baseUrl": "https://dashscope.aliyuncs.com/compatible-mode/v1"
    }
  }
}
```

**方式三：使用环境变量（推荐）**

```bash
# 添加到 ~/.bashrc 或 ~/.zshrc
export ALIBABA_API_KEY="sk-xxxxxxxxxxxx"

# 使配置生效
source ~/.bashrc
```

配置文件中使用变量引用：

```json
{
  "gateway": {
    "model": {
      "apiKey": "${ALIBABA_API_KEY}"
    }
  }
}
```

## 各厂商控制台入口

以下是各厂商的控制台入口和代金券领取方式。

### 国内厂商

| 厂商 | 控制台地址 | 代金券/活动 |
|------|-----------|------------|
| 阿里云百炼 | https://bailian.console.aliyun.com/ | 新用户100万tokens |
| DeepSeek | https://platform.deepseek.com/ | 新用户500万tokens |
| 智谱AI | https://open.bigmodel.cn/ | 新用户100万tokens |
| 月之暗面 | https://platform.moonshot.cn/ | 新用户15元体验金 |
| 百度智能云 | https://console.bce.baidu.com/qianfan/ | 新用户免费试用 |
| 字节豆包 | https://console.volcengine.com/ark | 新用户免费试用 |

### 国际厂商

| 厂商 | 控制台地址 | 代金券/活动 |
|------|-----------|------------|
| OpenAI | https://platform.openai.com/ | 新用户$5（需充值激活） |
| Anthropic | https://console.anthropic.com/ | 新用户$5 |
| Google AI | https://aistudio.google.com/ | 免费1500次/天 |

### 领取代金券技巧

1. **关注官方公众号**：各厂商经常在公众号发放优惠码
2. **加入开发者社区**：微信群、Discord经常有活动
3. **关注技术博客**：掘金、知乎等平台有专属优惠
4. **节假日活动**：双11、618等大促期间优惠力度大

## 配置命令详解

OpenClaw提供了便捷的模型配置命令。

### 基本命令

```bash
# 查看当前模型配置
openclaw models show

# 设置模型
openclaw models set <model_name> [options]

# 测试模型连接
openclaw models test

# 列出支持的模型
openclaw models list

# 查看模型用量
openclaw models usage
```

### 设置模型示例

**阿里云百炼：**

```bash
openclaw models set qwen-plus \
  --provider alibaba \
  --api-key sk-xxxxxxxxxxxx \
  --base-url https://dashscope.aliyuncs.com/compatible-mode/v1
```

**DeepSeek：**

```bash
openclaw models set deepseek-chat \
  --provider deepseek \
  --api-key sk-xxxxxxxxxxxx \
  --base-url https://api.deepseek.com/v1
```

**OpenAI：**

```bash
openclaw models set gpt-4o \
  --provider openai \
  --api-key sk-xxxxxxxxxxxx
```

**本地Ollama：**

```bash
openclaw models set qwen2.5:7b \
  --provider ollama \
  --base-url http://localhost:11434
```

### 测试模型连接

```bash
# 测试当前配置的模型
openclaw models test

# 测试指定模型
openclaw models test --model qwen-plus

# 详细测试（包含响应时间）
openclaw models test --verbose
```

**输出示例：**

```
Testing model connection...

Provider: alibaba
Model: qwen-plus
API Key: sk-****...****xxxx
Base URL: https://dashscope.aliyuncs.com/compatible-mode/v1

Sending test request...
✓ Connection successful
✓ Response time: 1.23s
✓ Model responded: "你好！我是通义千问，很高兴为您服务。有什么我可以帮助您的吗？"

Token usage: 15 input, 23 output
Estimated cost: ¥0.00015
```

### 查看用量统计

```bash
openclaw models usage
```

**输出示例：**

```
Model Usage Statistics (This Month)

Provider: alibaba
Model: qwen-plus

┌─────────────┬──────────────┬──────────────┐
│ Metric      │ Value        │ Cost         │
├─────────────┼──────────────┼──────────────┤
│ Input       │ 1,234,567    │ ¥4.94        │
│ Output      │ 567,890      │ ¥6.81        │
│ Total       │ 1,802,457    │ ¥11.75       │
└─────────────┴──────────────┴──────────────┘

Requests: 1,234
Average tokens/request: 1,461

Remaining quota: 3,197,543 tokens
```

## 多模型配置

你可以同时配置多个模型，用于不同场景。

### 配置文件示例

```json
{
  "models": {
    "default": {
      "provider": "alibaba",
      "model": "qwen-plus",
      "apiKey": "${ALIBABA_API_KEY}"
    },
    "coding": {
      "provider": "anthropic",
      "model": "claude-3-5-sonnet-20241022",
      "apiKey": "${ANTHROPIC_API_KEY}"
    },
    "cheap": {
      "provider": "deepseek",
      "model": "deepseek-chat",
      "apiKey": "${DEEPSEEK_API_KEY}"
    },
    "local": {
      "provider": "ollama",
      "model": "qwen2.5:7b",
      "baseUrl": "http://localhost:11434"
    }
  }
}
```

### 使用指定模型

```bash
# 使用默认模型
openclaw chat "你好"

# 使用指定模型
openclaw chat --model coding "帮我写一个排序算法"

# 在任务中指定模型
openclaw task run --model cheap "整理下载文件夹"
```

## API Key安全最佳实践

### 不要硬编码

```bash
# ❌ 错误：直接写在配置文件里
"apiKey": "sk-xxxxxxxxxxxx"

# ✅ 正确：使用环境变量
"apiKey": "${ALIBABA_API_KEY}"
```

### 定期轮换

```bash
# 每3个月更换一次API Key
# 1. 在控制台创建新Key
# 2. 更新环境变量
# 3. 删除旧Key
```

### 设置使用限额

在各大厂商控制台设置每日/每月消费上限：

- 阿里云：费用中心 → 消费限额
- DeepSeek：账户设置 → 消费限额
- OpenAI：Usage limits → Set limits

### 监控异常使用

```bash
# 查看最近的API调用
openclaw models logs --recent 100

# 设置异常告警
openclaw alerts add --type "usage_spike" --threshold 10000
```

---

配置好API Key后，你的OpenClaw已经可以正常工作了。下一章，我们将学习如何优化模型配置，在性能和成本之间找到最佳平衡。
