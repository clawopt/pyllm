# 模型策略与成本优化

选择模型不只是选一个"最好的"，而是在性能、成本、速度之间找到平衡。这一章，我们介绍如何通过模型策略配置，实现最优的成本效益。

## Fallback备选链配置

Fallback机制允许你配置多个模型作为备选，当主模型不可用或响应超时时，自动切换到备选模型。

### 为什么需要Fallback

**场景一：模型服务故障**

```
主模型：Claude 3.5 Sonnet
状态：服务暂时不可用
↓ 自动切换
备选模型：GPT-4o
状态：正常
```

**场景二：API限流**

```
主模型：DeepSeek
状态：请求频率超限
↓ 自动切换
备选模型：Qwen-Plus
状态：正常
```

**场景三：网络问题**

```
主模型：OpenAI（海外）
状态：连接超时
↓ 自动切换
备选模型：Qwen-Plus（国内）
状态：正常
```

### 配置Fallback链

**配置文件示例：**

```json
{
  "gateway": {
    "model": {
      "provider": "anthropic",
      "model": "claude-3-5-sonnet-20241022",
      "apiKey": "${ANTHROPIC_API_KEY}",
      "fallback": [
        {
          "provider": "openai",
          "model": "gpt-4o",
          "apiKey": "${OPENAI_API_KEY}"
        },
        {
          "provider": "alibaba",
          "model": "qwen-plus",
          "apiKey": "${ALIBABA_API_KEY}"
        },
        {
          "provider": "deepseek",
          "model": "deepseek-chat",
          "apiKey": "${DEEPSEEK_API_KEY}"
        }
      ]
    }
  }
}
```

**Fallback执行逻辑：**

```
请求 → 主模型 → 成功 → 返回结果
           ↓ 失败/超时
       备选1 → 成功 → 返回结果
           ↓ 失败/超时
       备选2 → 成功 → 返回结果
           ↓ 失败/超时
       备选3 → 成功 → 返回结果
           ↓ 全部失败
       返回错误
```

### Fallback配置选项

```json
{
  "fallback": {
    "enabled": true,
    "maxRetries": 3,           // 每个模型最大重试次数
    "retryDelay": 1000,        // 重试间隔（毫秒）
    "timeout": 30000,          // 单次请求超时（毫秒）
    "onError": "next",         // 错误时行为：next(下一个)/retry(重试)/fail(失败)
    "logFallback": true        // 记录Fallback日志
  }
}
```

### 按场景配置Fallback

**代码编写场景：**

```json
{
  "model": {
    "provider": "anthropic",
    "model": "claude-3-5-sonnet-20241022",
    "fallback": [
      {"provider": "openai", "model": "gpt-4o"},
      {"provider": "deepseek", "model": "deepseek-chat"}
    ]
  }
}
```

**日常对话场景：**

```json
{
  "model": {
    "provider": "deepseek",
    "model": "deepseek-chat",
    "fallback": [
      {"provider": "alibaba", "model": "qwen-plus"},
      {"provider": "zhipu", "model": "glm-4-flash"}
    ]
  }
}
```

## 价格对比

了解各模型的价格差异，是成本优化的基础。

### 国内模型价格对比

| 模型 | 输入价格 | 输出价格 | 百万tokens成本 |
|------|---------|---------|---------------|
| DeepSeek-V3 | ¥1 | ¥2 | ¥3 |
| Qwen-Turbo | ¥2 | ¥6 | ¥8 |
| Qwen-Plus | ¥4 | ¥12 | ¥16 |
| GLM-4-Flash | 免费 | 免费 | ¥0 |
| Kimi-8k | ¥12 | ¥12 | ¥24 |
| Qwen-Max | ¥120 | ¥120 | ¥240 |

### 国际模型价格对比

| 模型 | 输入价格 | 输出价格 | 百万tokens成本 |
|------|---------|---------|---------------|
| GPT-4o-mini | $0.15 | $0.60 | $0.75 (~¥5.4) |
| Claude 3.5 Haiku | $0.80 | $4.00 | $4.80 (~¥34.6) |
| GPT-4o | $2.50 | $10.00 | $12.50 (~¥90) |
| Claude 3.5 Sonnet | $3.00 | $15.00 | $18.00 (~¥129.6) |
| Claude 3 Opus | $15.00 | $75.00 | $90.00 (~¥648) |

### 成本计算示例

**场景：每天处理10万tokens，每月300万tokens**

| 模型 | 月成本 |
|------|--------|
| DeepSeek | ¥9 |
| Qwen-Plus | ¥48 |
| GPT-4o-mini | ¥16.2 |
| Claude 3.5 Sonnet | ¥388.8 |

**结论：DeepSeek比Claude便宜43倍！**

## 五套推荐配置方案

根据不同的使用场景和预算，我们推荐以下五套配置方案。

### 方案一：极致省钱

适合：个人用户、学习体验、预算有限

```json
{
  "model": {
    "provider": "deepseek",
    "model": "deepseek-chat",
    "apiKey": "${DEEPSEEK_API_KEY}",
    "fallback": [
      {"provider": "alibaba", "model": "qwen-turbo"},
      {"provider": "zhipu", "model": "glm-4-flash"}
    ]
  }
}
```

**特点：**
- 主模型：DeepSeek（¥1/百万tokens输入）
- 备选：Qwen-Turbo、GLM-4-Flash（免费额度）
- 月成本：<¥10（中等使用量）

### 方案二：国内平衡

适合：国内日常使用、追求性价比

```json
{
  "model": {
    "provider": "alibaba",
    "model": "qwen-plus",
    "apiKey": "${ALIBABA_API_KEY}",
    "fallback": [
      {"provider": "deepseek", "model": "deepseek-chat"},
      {"provider": "zhipu", "model": "glm-4"}
    ]
  }
}
```

**特点：**
- 主模型：Qwen-Plus（能力强、稳定）
- 备选：DeepSeek（便宜）、GLM-4（备用）
- 月成本：¥50-100（中等使用量）

### 方案三：国际平衡

适合：海外用户、追求国际模型能力

```json
{
  "model": {
    "provider": "openai",
    "model": "gpt-4o",
    "apiKey": "${OPENAI_API_KEY}",
    "fallback": [
      {"provider": "anthropic", "model": "claude-3-5-haiku-20241022"},
      {"provider": "google", "model": "gemini-2.0-flash"}
    ]
  }
}
```

**特点：**
- 主模型：GPT-4o（综合能力强）
- 备选：Claude Haiku（快速）、Gemini（免费额度）
- 月成本：$20-50（中等使用量）

### 方案四：混合最优

适合：专业用户、追求最佳性价比

```json
{
  "models": {
    "default": {
      "provider": "deepseek",
      "model": "deepseek-chat"
    },
    "coding": {
      "provider": "anthropic",
      "model": "claude-3-5-sonnet-20241022"
    },
    "analysis": {
      "provider": "openai",
      "model": "gpt-4o"
    },
    "quick": {
      "provider": "alibaba",
      "model": "qwen-turbo"
    }
  }
}
```

**使用策略：**
- 简单任务 → DeepSeek/Qwen-Turbo（便宜）
- 代码编写 → Claude Sonnet（能力强）
- 复杂分析 → GPT-4o（综合）
- 快速响应 → Qwen-Turbo（速度快）

**月成本：¥100-200（混合使用）**

### 方案五：企业稳定

适合：企业用户、追求稳定可靠

```json
{
  "model": {
    "provider": "anthropic",
    "model": "claude-3-5-sonnet-20241022",
    "apiKey": "${ANTHROPIC_API_KEY}",
    "fallback": [
      {"provider": "openai", "model": "gpt-4o"},
      {"provider": "alibaba", "model": "qwen-max"},
      {"provider": "deepseek", "model": "deepseek-chat"}
    ]
  },
  "fallback": {
    "maxRetries": 3,
    "timeout": 60000,
    "logFallback": true
  }
}
```

**特点：**
- 主模型：Claude Sonnet（能力最强）
- 三层备选：确保高可用
- 详细日志：便于问题排查
- 月成本：¥500-1000+

## 成本监控与优化

### 设置消费告警

```bash
# 设置每日消费上限告警
openclaw alerts add \
  --type daily_spend \
  --threshold 10 \
  --unit CNY \
  --notify feishu

# 设置月度消费上限
openclaw alerts add \
  --type monthly_spend \
  --threshold 200 \
  --unit CNY \
  --notify email
```

### 查看成本分析

```bash
# 查看本月成本
openclaw cost report

# 按模型分解成本
openclaw cost breakdown --by model

# 按日期查看趋势
openclaw cost trend --days 30
```

**输出示例：**

```
Cost Report - March 2026

Total Spend: ¥87.50

By Model:
┌─────────────────┬──────────┬────────┐
│ Model           │ Tokens   │ Cost   │
├─────────────────┼──────────┼────────┤
│ qwen-plus       │ 2.1M     │ ¥33.60 │
│ deepseek-chat   │ 15.0M    │ ¥45.00 │
│ claude-sonnet   │ 0.1M     │ ¥8.90  │
└─────────────────┴──────────┴────────┘

Cost Trend:
Week 1: ¥18.50
Week 2: ¥22.30
Week 3: ¥25.10
Week 4: ¥21.60

Recommendations:
• 38% of requests could use cheaper models
• Consider using qwen-turbo for simple tasks
```

### 优化建议

1. **简单任务用便宜模型**：日常对话、简单查询用DeepSeek
2. **复杂任务用强模型**：代码编写、深度分析用Claude
3. **设置合理超时**：避免长时间等待浪费成本
4. **定期审查用量**：发现异常及时处理

---

通过合理的模型策略配置，你可以在保证服务质量的同时，大幅降低使用成本。下一章，我们将介绍如何部署本地模型，实现完全离线使用。
