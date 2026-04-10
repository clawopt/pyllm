# 1.1 DeepSeek 是什么？开源、低价、强推理的国产大模型

> **如果 GPT-4 是 AI 领域的"iPhone"，那 DeepSeek 就是"小米"——同样的体验，1/30 的价格，还能自己拆开改。**

---

## 这一节在讲什么？

你可能已经用过 ChatGPT、Claude 或者 Gemini，它们都很强，但有一个共同点——**贵**。GPT-4o 的 API 每百万 input token 要 $2.50，Claude Sonnet 要 $3，如果你的应用每天处理几百万 token，月账单可能上千美元。DeepSeek 的出现打破了这个局面——它以 GPT-4 级别的代码生成能力和接近 o3 的推理能力，只收 GPT-4 1/10 的价格。更重要的是，它是开源的——你可以下载模型权重，部署在自己的服务器上，数据完全不出机房。这一节我们把 DeepSeek 的定位、核心优势和与其他模型的对比讲清楚，帮你理解为什么 DeepSeek 在 2025 年引发了全球关注。

---

## DeepSeek 的定位：开源、低价、强推理

DeepSeek 是中国 AI 公司深度求索（DeepSeek AI）开发的大语言模型系列。它的核心定位可以用三个关键词概括：

**开源（Open Source）**：DeepSeek 的模型权重以 MIT 许可证开源——这是最宽松的开源许可证之一，允许你自由使用、修改、分发，甚至用于商业产品。你可以把 DeepSeek 的模型下载到自己的服务器上运行，可以基于它微调自己的模型，可以用它的推理数据训练其他模型。相比之下，GPT-4、Claude、Gemini 都是闭源的——你只能通过 API 调用，无法看到模型内部，无法自己部署。

**低价（Low Cost）**：DeepSeek 的 API 价格极低——`deepseek-chat`（V3.2）每百万 input token 仅 $0.27，是 Claude Sonnet 的 1/11、GPT-4o 的 1/9。更厉害的是，DeepSeek 支持前缀缓存——如果你的请求前缀跟之前的请求相同，缓存命中的 token 价格再降 75%，低至 $0.07/M。这意味着在多轮对话场景下，DeepSeek 的实际成本可能只有标价的 1/4。

**强推理（Strong Reasoning）**：DeepSeek-R1 是专门为推理设计的模型——它在数学、逻辑、编程等需要"深度思考"的任务上表现卓越。在 AIME 2025 数学竞赛测试中，R1 达到了 87.5% 的准确率，接近 OpenAI o3 的水平。在编程基准 Aider 上，R1 达到了 71.6% 的通过率，略超 Claude Opus 4 的 70.6%——而价格只有 Opus 的 1/30。

---

## 与 GPT-4 / Claude / Gemini 的对比

理解 DeepSeek 最好的方式是把它放在主流大模型的生态里对比：

| 维度 | DeepSeek V3.2 | DeepSeek R1 | GPT-4o | Claude Sonnet 4 | Gemini 2.5 Pro |
|------|--------------|-------------|--------|-----------------|----------------|
| 开源 | ✅ MIT | ✅ MIT | ❌ | ❌ | ❌ |
| 输入价格 | $0.27/M | $0.55/M | $2.50/M | $3/M | $1.25/M |
| 输出价格 | $1.10/M | $2.19/M | $10/M | $15/M | $10/M |
| 缓存折扣 | ✅ 75% off | ✅ 75% off | ❌ | ❌ | ❌ |
| 上下文 | 128K | 64K | 128K | 200K | 1M |
| 代码生成 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 推理能力 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 中文能力 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| 本地部署 | ✅ 蒸馏版 | ✅ 蒸馏版 | ❌ | ❌ | ❌ |

从这个对比中你能看出 DeepSeek 的独特价值：

- **价格优势**：V3.2 的 input 价格是 GPT-4o 的 1/9，output 价格是 1/9。R1 虽然比 V3 贵一倍，但仍然只有 GPT-4o 的 1/5。
- **推理优势**：R1 在数学和逻辑推理上接近 o3 水平，这是 GPT-4o 和 Claude Sonnet 做不到的。
- **开源优势**：唯一可以本地部署的主流大模型，适合有隐私和合规需求的企业。
- **中文优势**：DeepSeek 的中文能力在所有模型中最强——这对中文用户来说是硬性优势。

但 DeepSeek 也有短板：Function Calling 能力不如 Claude，长上下文不如 Gemini，英文能力不如 GPT-4o。我们会在第 8 章详细讨论这些局限。

---

## DeepSeek 的核心优势

让我们深入理解 DeepSeek 的五个核心优势：

### 1. 开源——MIT 许可证，可商用、可修改、可蒸馏

DeepSeek 的模型权重以 MIT 许可证发布，这意味着你可以：

- **商业使用**：把 DeepSeek 集成到你的商业产品中，不需要支付许可费
- **修改模型**：基于 DeepSeek 微调你自己的行业模型
- **蒸馏训练**：用 DeepSeek-R1 的推理数据训练更小的模型——DeepSeek 官方就是这么做的，他们用 R1 的 80 万条推理样本微调 Qwen 和 Llama，得到了性能接近 o1-mini 的蒸馏模型

```python
# 开源意味着你可以直接下载模型权重
# 从 HuggingFace 下载 DeepSeek-R1
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
```

### 2. 低价——1/30 的价格，同等的代码质量

让我们算一笔账：假设你每天用 AI 生成代码，每天消耗 100 万 input token 和 50 万 output token：

```
月度成本对比（30 天）：

  DeepSeek V3.2：
    input:  100万 × 30天 × $0.27/M = $8.1
    output: 50万 × 30天 × $1.10/M = $16.5
    合计：$24.6/月

  Claude Sonnet 4：
    input:  100万 × 30天 × $3/M = $90
    output: 50万 × 30天 × $15/M = $225
    合计：$315/月

  GPT-4o：
    input:  100万 × 30天 × $2.50/M = $75
    output: 50万 × 30天 × $10/M = $150
    合计：$225/月

  DeepSeek 的成本是 Claude 的 1/13，GPT-4o 的 1/9
```

如果考虑缓存命中（多轮对话场景下缓存命中率通常 50%+），DeepSeek 的实际成本更低。

### 3. 强推理——R1 接近 o3 水平

DeepSeek-R1 的推理能力是它最引人注目的优势。在几个关键基准测试上：

| 基准测试 | DeepSeek R1 | OpenAI o1 | GPT-4o | Claude Sonnet |
|----------|------------|-----------|--------|---------------|
| AIME 2025（数学） | 87.5% | ~88% | 59.4% | ~65% |
| GPQA（科学推理） | 81.0% | ~78% | 68.4% | ~70% |
| Aider（编程） | 71.6% | ~72% | ~49% | ~65% |

R1 的推理能力接近 o1，远超 GPT-4o 和 Claude Sonnet——而价格只有 o1 的一小部分。

### 4. OpenAI 兼容——迁移零成本

DeepSeek 的 API 完全兼容 OpenAI 格式——你只需要改 `base_url` 和 `api_key`，代码零修改：

```python
# 从 OpenAI 迁移到 DeepSeek，只需改两行
from openai import OpenAI

# 之前用 OpenAI
# client = OpenAI(api_key="sk-...", base_url="https://api.openai.com/v1")

# 现在用 DeepSeek
client = OpenAI(api_key="sk-...", base_url="https://api.deepseek.com")

# 其余代码完全不变
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### 5. 中英双语——中文能力突出

DeepSeek 的训练数据包含大量中文语料，中文理解和生成能力在所有主流模型中最强。这对中文用户来说是硬性优势——无论是中文代码注释、中文文档生成、还是中文业务逻辑理解，DeepSeek 都比 GPT-4o 和 Claude 更准确。

---

## 谁适合用 DeepSeek

**适合用 DeepSeek 的人**：
- 预算有限的开发者——用 1/10 的价格获得接近 GPT-4 的代码生成能力
- 需要开源模型的企业——可以私有化部署，数据不出机房
- 中文场景用户——中文理解和生成能力最强
- AI Agent 构建者——Function Calling + 低成本 = 高性价比 Agent
- 需要强推理的用户——R1 的数学和逻辑推理接近 o3

**不太适合用 DeepSeek 的人**：
- 需要最强 Function Calling 的用户——Claude Sonnet 在工具调用上更可靠
- 需要超长上下文的用户——Gemini 2.5 Pro 的 1M 上下文无可替代
- 需要多模态能力的用户——DeepSeek 目前主要是文本模型

---

## 常见误区

**误区一：DeepSeek 只是"便宜的 GPT 替代品"**

不是。DeepSeek 有三个 GPT 不具备的独特优势：开源（可私有化部署）、强推理（R1 接近 o3）、中文能力突出。它不是"便宜的替代品"，而是在某些场景下（推理、中文、成本敏感）是更好的选择。

**误区二：开源意味着质量差**

恰恰相反。DeepSeek-R1 在 AIME 2025 上达到 87.5%，Aider 编程测试达到 71.6%——这些成绩超过了很多闭源模型。开源不等于低质量，DeepSeek 证明了开源模型可以达到甚至超过闭源模型的水平。

**误区三：DeepSeek 只能做中文**

不是。DeepSeek 的英文能力也很强——在 HumanEval、MMLU-Pro 等英文基准上表现优秀。只是中文能力相对更强，不代表英文能力弱。

**误区四：671B 参数意味着需要超级计算机**

DeepSeek 采用 MoE（Mixture of Experts）架构，671B 是总参数量，但每次推理只激活 37B 参数——这相当于一个 37B 参数的 Dense 模型的推理成本。而且 DeepSeek 提供了 1.5B 到 70B 的蒸馏模型，普通消费级 GPU 就能运行。

---

## 小结

这一节我们建立了对 DeepSeek 的基本认知：它是一个开源、低价、强推理的大模型，API 价格是 Claude 的 1/30，推理能力接近 o3，中文能力最强，兼容 OpenAI 格式。DeepSeek 不是"便宜的 GPT 替代品"——它在推理、中文、开源生态上有独特优势。下一节我们深入 DeepSeek 的模型家族，看看 V3 和 R1 两条产品线各有什么特点。
