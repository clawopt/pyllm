# 支持的AI模型提供商

OpenClaw本身不具备大模型能力，它需要连接外部的大语言模型（LLM）才能工作。这一章，我们介绍OpenClaw支持的所有模型提供商，帮助你选择最适合的"大脑"。

## 国际厂商

国际厂商的模型通常能力最强，但价格较高，且国内访问可能需要代理。

### Anthropic Claude

Claude系列以长文本处理和代码能力著称，是OpenClaw官方推荐的模型之一。

**模型列表：**

| 模型 | 上下文长度 | 特点 | 适用场景 |
|------|-----------|------|---------|
| Claude 3.5 Sonnet | 200K | 综合能力最强 | 复杂任务、代码编写 |
| Claude 3.5 Haiku | 200K | 速度快、成本低 | 简单任务、快速响应 |
| Claude 3 Opus | 200K | 推理能力最强 | 复杂推理、深度分析 |

**价格参考：**

| 模型 | 输入价格 | 输出价格 |
|------|---------|---------|
| Claude 3.5 Sonnet | $3.00/百万tokens | $15.00/百万tokens |
| Claude 3.5 Haiku | $0.80/百万tokens | $4.00/百万tokens |
| Claude 3 Opus | $15.00/百万tokens | $75.00/百万tokens |

**配置方式：**

```yaml
model:
  provider: anthropic
  model: claude-3-5-sonnet-20241022
  api_key: ${ANTHROPIC_API_KEY}
```

**获取API Key：**

1. 访问 https://console.anthropic.com/
2. 注册账号（需要海外手机号或邮箱）
3. 进入API Keys页面创建密钥
4. 新用户赠送$5额度

### OpenAI GPT

GPT系列是最知名的大模型，生态成熟，文档完善。

**模型列表：**

| 模型 | 上下文长度 | 特点 | 适用场景 |
|------|-----------|------|---------|
| GPT-4o | 128K | 多模态、速度快 | 通用场景 |
| GPT-4o-mini | 128K | 轻量级、成本低 | 简单任务 |
| GPT-4-turbo | 128K | 推理能力强 | 复杂任务 |

**价格参考：**

| 模型 | 输入价格 | 输出价格 |
|------|---------|---------|
| GPT-4o | $2.50/百万tokens | $10.00/百万tokens |
| GPT-4o-mini | $0.15/百万tokens | $0.60/百万tokens |
| GPT-4-turbo | $10.00/百万tokens | $30.00/百万tokens |

**配置方式：**

```yaml
model:
  provider: openai
  model: gpt-4o
  api_key: ${OPENAI_API_KEY}
```

**获取API Key：**

1. 访问 https://platform.openai.com/
2. 注册账号
3. 进入API Keys → Create new secret key
4. 最低充值$5起

### Google Gemini

Gemini是Google推出的大模型，多模态能力强，与Google生态深度整合。

**模型列表：**

| 模型 | 上下文长度 | 特点 |
|------|-----------|------|
| Gemini 2.0 Flash | 1M | 速度快、免费额度大 |
| Gemini 1.5 Pro | 2M | 超长上下文 |
| Gemini 1.5 Flash | 1M | 平衡性能与成本 |

**价格参考：**

| 模型 | 输入价格 | 输出价格 |
|------|---------|---------|
| Gemini 2.0 Flash | 免费（有限额） | 免费（有限额） |
| Gemini 1.5 Pro | $1.25/百万tokens | $5.00/百万tokens |
| Gemini 1.5 Flash | $0.075/百万tokens | $0.30/百万tokens |

**配置方式：**

```yaml
model:
  provider: google
  model: gemini-2.0-flash
  api_key: ${GOOGLE_API_KEY}
```

**获取API Key：**

1. 访问 https://aistudio.google.com/app/apikey
2. 使用Google账号登录
3. 创建API Key
4. 免费额度：每天1500次请求

## 国内厂商

国内厂商的模型访问稳定，价格实惠，中文能力强，是大多数国内用户的首选。

### 通义千问（阿里云百炼）

通义千问是阿里云推出的大模型，中文能力强，性价比高。

**模型列表：**

| 模型 | 上下文长度 | 特点 | 适用场景 |
|------|-----------|------|---------|
| Qwen-Max | 32K | 能力最强 | 复杂任务 |
| Qwen-Plus | 128K | 平衡性能与成本 | 通用场景 |
| Qwen-Turbo | 128K | 速度快、成本低 | 简单任务 |
| Qwen-Long | 1M | 超长上下文 | 长文档处理 |

**价格参考：**

| 模型 | 输入价格 | 输出价格 |
|------|---------|---------|
| Qwen-Max | ¥0.12/千tokens | ¥0.12/千tokens |
| Qwen-Plus | ¥0.004/千tokens | ¥0.012/千tokens |
| Qwen-Turbo | ¥0.002/千tokens | ¥0.006/千tokens |
| Qwen-Long | ¥0.0005/千tokens | ¥0.002/千tokens |

**配置方式：**

```yaml
model:
  provider: alibaba
  model: qwen-plus
  api_key: ${ALIBABA_API_KEY}
  base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
```

**获取API Key：**

1. 访问 https://bailian.console.aliyun.com/
2. 使用阿里云账号登录
3. 进入"模型广场" → 选择Qwen系列
4. 开通服务并创建API Key
5. 新用户赠送100万tokens

### DeepSeek

DeepSeek以极致性价比著称，推理能力强，是国内用户的热门选择。

**模型列表：**

| 模型 | 上下文长度 | 特点 | 适用场景 |
|------|-----------|------|---------|
| DeepSeek-V3 | 64K | 综合能力强 | 通用场景 |
| DeepSeek-Chat | 64K | 对话优化 | 日常对话 |
| DeepSeek-Reasoner | 64K | 推理专用 | 复杂推理 |

**价格参考：**

| 模型 | 输入价格 | 输出价格 |
|------|---------|---------|
| DeepSeek-V3 | ¥1/百万tokens | ¥2/百万tokens |
| DeepSeek-Chat | ¥1/百万tokens | ¥2/百万tokens |

**配置方式：**

```yaml
model:
  provider: deepseek
  model: deepseek-chat
  api_key: ${DEEPSEEK_API_KEY}
  base_url: https://api.deepseek.com/v1
```

**获取API Key：**

1. 访问 https://platform.deepseek.com/
2. 注册账号（支持微信扫码）
3. 进入API Keys页面创建密钥
4. 新用户赠送500万tokens

### Kimi（月之暗面）

Kimi以超长上下文著称，适合处理长文档。

**模型列表：**

| 模型 | 上下文长度 | 特点 |
|------|-----------|------|
| moonshot-v1-8k | 8K | 基础版本 |
| moonshot-v1-32k | 32K | 中等长度 |
| moonshot-v1-128k | 128K | 超长上下文 |

**价格参考：**

| 模型 | 输入价格 | 输出价格 |
|------|---------|---------|
| moonshot-v1-8k | ¥0.012/千tokens | ¥0.012/千tokens |
| moonshot-v1-32k | ¥0.024/千tokens | ¥0.024/千tokens |
| moonshot-v1-128k | ¥0.06/千tokens | ¥0.06/千tokens |

**配置方式：**

```yaml
model:
  provider: moonshot
  model: moonshot-v1-8k
  api_key: ${MOONSHOT_API_KEY}
  base_url: https://api.moonshot.cn/v1
```

### 智谱GLM

智谱AI的GLM系列，中文能力强，学术背景深厚。

**模型列表：**

| 模型 | 上下文长度 | 特点 |
|------|-----------|------|
| GLM-4 | 128K | 综合能力强 |
| GLM-4-Flash | 128K | 速度快、免费额度大 |
| GLM-4-Plus | 128K | 推理能力强 |

**价格参考：**

| 模型 | 输入价格 | 输出价格 |
|------|---------|---------|
| GLM-4 | ¥0.1/千tokens | ¥0.1/千tokens |
| GLM-4-Flash | 免费（有限额） | 免费（有限额） |
| GLM-4-Plus | ¥0.05/千tokens | ¥0.05/千tokens |

**配置方式：**

```yaml
model:
  provider: zhipu
  model: glm-4
  api_key: ${ZHIPU_API_KEY}
  base_url: https://open.bigmodel.cn/api/paas/v4
```

### 文心一言（百度）

百度推出的大模型，与百度生态深度整合。

**模型列表：**

| 模型 | 特点 |
|------|------|
| ERNIE-4.0-8K | 能力最强 |
| ERNIE-3.5-8K | 平衡性能与成本 |
| ERNIE-Speed-8K | 速度快 |

**配置方式：**

```yaml
model:
  provider: baidu
  model: ernie-4.0-8k
  api_key: ${BAIDU_API_KEY}
  secret_key: ${BAIDU_SECRET_KEY}
```

### 豆包（字节跳动）

字节跳动推出的大模型，性价比高。

**模型列表：**

| 模型 | 特点 |
|------|------|
| doubao-pro-32k | 综合能力强 |
| doubao-lite-32k | 轻量级 |

**配置方式：**

```yaml
model:
  provider: bytedance
  model: doubao-pro-32k
  api_key: ${BYTEDANCE_API_KEY}
  base_url: https://ark.cn-beijing.volces.com/api/v3
```

## 本地模型

如果你对数据隐私有极高要求，或者需要在断网环境下使用，可以部署本地模型。

### Ollama + Qwen/Llama

Ollama是目前最流行的本地模型运行工具，支持多种开源模型。

**支持的模型：**

| 模型 | 参数量 | 内存需求 | 特点 |
|------|--------|---------|------|
| qwen2.5:7b | 7B | 8GB | 中文能力强 |
| qwen2.5:14b | 14B | 16GB | 能力更强 |
| llama3.1:8b | 8B | 8GB | 英文能力强 |
| deepseek-coder:6.7b | 6.7B | 8GB | 代码专用 |

**配置方式：**

```yaml
model:
  provider: ollama
  model: qwen2.5:7b
  base_url: http://localhost:11434
```

**优势：**
- 完全离线可用
- 数据不出本地
- 无使用限制
- 免费使用

**劣势：**
- 需要较高硬件配置
- 能力不如云端大模型
- 首次下载模型较慢

---

选择模型时，建议考虑以下因素：

1. **预算**：DeepSeek性价比最高，Claude能力最强但价格也最高
2. **场景**：代码任务选Claude/GPT，中文任务选通义/DeepSeek
3. **隐私**：敏感数据选本地模型
4. **网络**：国内用户优先选国内厂商

下一章，我们将详细介绍如何获取和配置API Key。
