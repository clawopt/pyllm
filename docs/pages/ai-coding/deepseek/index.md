# DeepSeek 教程大纲

> **开源、低价、强推理——DeepSeek，用 1/30 的价格跑出 GPT-4 级别的代码和推理能力**

---

## 📖 教程概述

如果你关注 AI 领域，你一定听说过 DeepSeek——这个来自中国的开源大模型，以极低的成本实现了接近 GPT-4 和 Claude 的性能，在全球开发者社区引发了巨大反响。DeepSeek 的核心优势可以概括为三个关键词：**开源、低价、强推理**。它的模型权重以 MIT 许可证开源，允许商业使用和修改；API 价格只有 Claude 的 1/30、GPT-4 的 1/20；而 DeepSeek-R1 的推理能力在数学和编程基准测试上接近 OpenAI o3。

DeepSeek 目前提供两条产品线：**DeepSeek-V3 系列**（通用模型，代号 `deepseek-chat`）和 **DeepSeek-R1 系列**（推理模型，代号 `deepseek-reasoner`）。V3 系列擅长日常对话、代码生成、内容创作，速度快、成本低；R1 系列擅长数学推理、逻辑分析、复杂编程，会"深度思考"后再回答。两条线共享同一个基座模型（671B 参数的 MoE 架构），通过不同的后训练策略分化出不同的能力侧重。

本教程共 8 章：第 1 章建立认知，第 2 章快速上手，第 3 章深入 API 接口，第 4 章掌握推理模式，第 5 章学习 Function Calling 与 Agent 构建，第 6 章实战代码生成与调试，第 7 章本地部署，第 8 章生产化与成本优化。教程面向有 Python 基础的开发者，从零开始带你掌握 DeepSeek 的全部能力。

---

## 🗺️ 章节规划

### 第1章：认识 DeepSeek

#### 1.1 DeepSeek 是什么？开源、低价、强推理的国产大模型
- **DeepSeek 的定位**：中国团队深度求索开发的开源大模型，MIT 许可证，671B 参数 MoE 架构，激活参数仅 37B
- **与 GPT-4 / Claude / Gemini 的对比**：
  - GPT-4o：闭源，$2.50/M input，代码能力强但贵
  - Claude Sonnet 4：闭源，$3/M input，代码和推理均衡
  - Gemini 2.5 Pro：闭源，$1.25/M input，超长上下文
  - DeepSeek V3.2：开源，$0.27/M input，代码能力强，价格极低
  - DeepSeek R1：开源，$0.55/M input，推理能力接近 o3
- **DeepSeek 的核心优势**：
  - 开源（MIT 许可证，可商用、可修改、可蒸馏）
  - 低价（API 价格是 Claude 的 1/30）
  - 强推理（R1 在 AIME 2025 达到 87.5%，接近 o3）
  - OpenAI 兼容（API 格式完全兼容 OpenAI SDK，迁移零成本）
  - 中英双语（中文能力突出，代码生成质量高）
- **谁适合用 DeepSeek**：预算有限的开发者、需要开源模型的企业、中文场景用户、AI Agent 构建者
- **常见误区**：DeepSeek 不只是"便宜的 GPT 替代品"——它的推理能力（R1）和开源生态是独特优势

#### 1.2 DeepSeek 的模型家族：V3 与 R1
- **V3 系列（deepseek-chat）**：通用模型，非思考模式，速度快、成本低
  - DeepSeek-V3 → V3-0324 → V3.1 → V3.1-Terminus → V3.2-Exp → V3.2
  - 擅长：代码生成、内容创作、日常对话、Function Calling
  - 上下文：128K token
- **R1 系列（deepseek-reasoner）**：推理模型，思考模式，深度推理后再回答
  - DeepSeek-R1 → R1-0528
  - 擅长：数学推理、逻辑分析、复杂编程、算法设计
  - 上下文：64K（API）/ 128K（开源版）
  - 特点：输出包含 `<think)>` 思考过程，可观察推理链
- **混合推理架构（V3.1+）**：一个模型同时支持思考模式和非思考模式
  - `deepseek-chat` = V3.2 的非思考模式
  - `deepseek-reasoner` = V3.2 的思考模式
- **蒸馏模型**：从 R1 蒸馏到小模型（1.5B/7B/8B/14B/32B/70B），保留推理能力，降低硬件需求
- **常见误区**：以为 `deepseek-chat` 只能聊天——它也能写代码、做 Function Calling；以为 R1 只能做数学——它的代码生成和调试能力也很强

#### 1.3 DeepSeek 的技术架构：MoE + 混合推理
- **MoE（Mixture of Experts）架构**：671B 总参数，仅激活 37B，高效且低成本
- **混合推理架构**：V3.1+ 将推理和对话融合到一个模型中，通过 `thinking` 参数切换模式
- **与 Dense 模型的对比**：Dense 模型每次推理激活所有参数，MoE 只激活相关专家，效率更高
- **蒸馏技术**：用 R1 的 80 万条推理样本微调 Qwen/Llama，得到小参数蒸馏模型
- **常见误区**：以为 671B 参数意味着需要 671B 的显存——MoE 架构只激活 37B 参数，推理成本远低于同等参数的 Dense 模型

### 第2章：快速上手

#### 2.1 注册与获取 API Key
- **注册 DeepSeek 平台**：platform.deepseek.com
- **获取 API Key**：注册后自动获得免费额度
- **API 定价**：
  - deepseek-chat（V3.2）：$0.27/M input, $1.10/M output（缓存命中 $0.07/M）
  - deepseek-reasoner（R1）：$0.55/M input, $2.19/M output（缓存命中 $0.14/M）
- **免费额度**：新用户赠送 500 万 token 免费额度
- **充值方式**：支持支付宝、微信支付
- **常见误区**：以为 DeepSeek API 需要科学上网——国内可直接访问 api.deepseek.com

#### 2.2 Hello World：第一次 API 调用
- **安装 OpenAI SDK**：`pip install openai`（DeepSeek 兼容 OpenAI 格式）
- **第一次调用**：
  ```python
  from openai import OpenAI
  client = OpenAI(api_key="your-key", base_url="https://api.deepseek.com")
  response = client.chat.completions.create(
      model="deepseek-chat",
      messages=[{"role": "user", "content": "用 Python 写一个快速排序"}]
  )
  print(response.choices[0].message.content)
  ```
- **流式输出**：`stream=True` 实时获取生成内容
- **从 OpenAI 迁移**：只需改 `base_url` 和 `api_key`，代码零修改
- **常见误区**：以为需要安装 DeepSeek 专用 SDK——直接用 OpenAI SDK 即可，只需改 base_url

#### 2.3 网页版与 App 使用
- **chat.deepseek.com**：免费网页版，支持 V3.2 和 R1 模型
- **深度思考模式**：开启后使用 R1 推理，适合复杂问题
- **文件上传**：支持 PDF、Word、Excel、图片等文件分析
- **联网搜索**：Web/App 端支持联网搜索
- **常见误区**：网页版免费但 API 收费——网页版有使用频率限制，API 按量计费无限制

### 第3章：API 接口详解

#### 3.1 Chat Completions 接口
- **接口地址**：`POST https://api.deepseek.com/chat/completions`
- **请求参数详解**：
  - `model`：`deepseek-chat` 或 `deepseek-reasoner`
  - `messages`：对话消息列表（system / user / assistant / tool）
  - `temperature`：采样温度（0-2），代码生成建议 0.1-0.3，创意写作建议 0.7-1.0
  - `top_p`：核采样参数
  - `max_tokens`：最大生成 token 数（默认 4096，Beta 可设 8192）
  - `frequency_penalty` / `presence_penalty`：频率/存在惩罚
  - `stop`：停止序列
  - `stream`：是否流式输出
- **响应格式**：choices、usage（prompt_tokens / completion_tokens / total_tokens）
- **缓存命中**：DeepSeek API 支持前缀缓存，缓存命中的 token 价格降低 75%
- **常见误区**：`max_tokens` 设太小导致输出被截断——代码生成建议设为 4096 以上

#### 3.2 流式输出与 Token 统计
- **流式输出**：`stream=True`，逐 token 返回，适合实时展示
- **流式响应格式**：`data: {"choices":[{"delta":{"content":"..."}}]}` → `data: [DONE]`
- **Token 统计**：`usage` 字段显示 prompt_tokens / completion_tokens / total_tokens
- **缓存命中统计**：`prompt_cache_hit_tokens` 显示缓存命中的 token 数
- **成本计算**：实际成本 = (总 token - 缓存命中 token) × 正常价格 + 缓存命中 token × 缓存价格
- **常见误区**：流式输出时 usage 可能为 null——需要在最后一个 chunk 或设置 `stream_options={"include_usage": True}` 获取

#### 3.3 JSON Output 与结构化输出
- **启用 JSON Output**：`response_format={"type": "json_object"}`
- **使用要求**：必须在 system/user 消息中指导模型输出 JSON 格式
- **代码示例**：让 DeepSeek 输出结构化的代码审查结果
- **常见误区**：设置了 JSON Output 但提示词中没有要求 JSON——模型可能输出非 JSON 内容或格式不正确

#### 3.4 FIM 补全与对话前缀续写
- **FIM（Fill-In-the-Middle）补全**：提供前缀和后缀，模型补全中间内容
  - 接口：`POST https://api.deepseek.com/beta/completions`
  - 适用场景：代码补全、故事续写
- **对话前缀续写**：指定 assistant 消息的前缀，模型按前缀续写
  - 设置 `base_url` 为 `https://api.deepseek.com/beta`
  - 最后一条消息 `role: "assistant"`, `prefix: True`
  - 适用场景：强制代码块输出、续写被截断的内容
- **常见误区**：FIM 和对话前缀续写是 Beta 功能——需要设置 `base_url` 为 beta 端点，且接口可能变化

### 第4章：推理模式与思考过程

#### 4.1 DeepSeek-R1 的思考模式
- **思考模式的工作原理**：模型先在 `<think)>` 标签内进行内部推理，再生成最终回答
- **思考过程的可视化**：API 返回 `reasoning_content` 字段，包含完整的思考链
- **思考模式的触发**：
  - 使用 `deepseek-reasoner` 模型自动触发
  - V3.1+ 通过 `thinking` 参数控制：`{"type": "enabled"}` 或 `{"type": "disabled"}`
- **思考 token 的计费**：思考过程消耗的 token 也计入费用
- **常见误区**：以为思考过程不收费——R1 的思考 token 也按 output 价格计费，复杂问题可能消耗大量 token

#### 4.2 推理模式 vs 非推理模式的选择
- **什么时候用推理模式（R1）**：
  - 数学问题求解、逻辑推理
  - 复杂算法设计
  - 代码调试与根因分析
  - 多步骤推理任务
- **什么时候用非推理模式（V3）**：
  - 日常代码生成
  - 内容创作和翻译
  - 简单问答
  - Function Calling 和 Agent 任务
- **成本对比**：R1 的 output 价格是 V3 的 2 倍，且思考 token 额外消耗
- **常见误区**：所有任务都用 R1——简单任务用 V3 更快更便宜，R1 的推理能力在简单任务上是浪费

#### 4.3 max_tokens 与思考过程的配合
- **R1 的 max_tokens 含义**：限制模型单次输出的总长度（包括思考过程）
  - 默认 32K，最大 64K
  - 思考过程可能消耗 10K-30K token，留给最终回答的空间可能不足
- **V3 的 max_tokens 含义**：限制模型生成的 token 数（不包括输入）
  - 默认 4096，Beta 最大 8192
- **设置建议**：
  - R1 简单问题：max_tokens=8192
  - R1 复杂问题：max_tokens=32768
  - V3 代码生成：max_tokens=4096
  - V3 长文生成：max_tokens=8192（Beta）
- **常见误区**：R1 的 max_tokens 设太小导致思考过程被截断，最终回答为空——建议 R1 至少设 8192

### 第5章：Function Calling 与 Agent 构建

#### 5.1 Function Calling 详解
- **Function Calling 的工作原理**：模型根据用户问题决定是否调用工具，返回工具调用请求，你的代码执行工具后把结果返回给模型
- **定义工具**：用 JSON Schema 描述函数签名和参数
- **工具调用流程**：
  1. 发送 messages + tools 给 API
  2. 模型返回 `finish_reason: "tool_calls"` + `tool_calls` 列表
  3. 你的代码执行工具，将结果以 `role: "tool"` 追加到 messages
  4. 再次请求 API，模型生成最终回复
- **并行工具调用**：模型可能在一次响应中请求调用多个工具
- **最多 128 个 Function**：一次请求最多传入 128 个工具定义
- **代码示例**：天气查询 + 计算器 Agent
- **常见误区**：模型只是"建议"调用工具，真正的执行权在你的代码里——不要盲目执行模型返回的参数

#### 5.2 构建 AI Agent
- **Agent 循环**：用户消息 → 模型推理 → 工具调用 → 执行工具 → 结果返回 → 继续推理 → 最终回答
- **多轮工具调用**：Agent 可能需要多轮工具调用才能完成任务
- **工具注册表模式**：用字典映射工具名到实际函数
- **代码示例**：完整的 DeepSeek Agent 实现
- **错误处理**：工具执行失败时的重试和降级策略
- **常见误区**：Agent 不需要限制工具调用轮数——无限循环可能导致 API 费用失控，建议设置最大轮数

#### 5.3 DeepSeek + RAG 实战
- **RAG 架构**：用户问题 → 检索相关文档 → 构建增强提示词 → DeepSeek 生成回答
- **代码示例**：用 DeepSeek + Chroma 构建 RAG 系统
- **Function Calling + RAG**：用工具调用封装检索逻辑，让 Agent 自主决定何时检索
- **常见误区**：RAG 检索的文档太多塞给模型——应该只检索 top-k 最相关的片段，避免上下文溢出

### 第6章：代码生成与调试实战

#### 6.1 代码生成最佳实践
- **Prompt 设计**：明确语言、框架、需求、约束
- **System Prompt 模板**：定义 AI 的角色和输出规范
- **代码生成示例**：
  - 生成 REST API（Express + TypeScript）
  - 生成数据处理脚本（Python + Pandas）
  - 生成测试用例（Jest / Pytest）
- **Temperature 设置**：代码生成用 0.1-0.3（确定性高），创意代码用 0.5-0.7
- **常见误区**：提示词太笼统——"帮我写个登录功能"不如"用 Express + JWT 写一个 POST /auth/login 接口，接收 email 和 password"

#### 6.2 代码调试与错误修复
- **错误诊断**：粘贴错误信息 + 代码上下文，让 DeepSeek 分析根因
- **R1 模式的优势**：推理模型会"深度思考"错误原因，给出更准确的分析
- **代码审查**：让 DeepSeek 审查代码的安全性和性能
- **测试驱动调试**：先让 DeepSeek 写复现测试，再根据失败信息修复
- **常见误区**：只给错误信息不给代码——DeepSeek 需要看到相关代码才能分析根因

#### 6.3 DeepSeek 在 CI/CD 中的应用
- **非交互式调用**：用 Python 脚本调用 DeepSeek API
- **代码审查自动化**：在 GitHub Actions 中集成 DeepSeek
- **PR 描述生成**：根据 diff 自动生成 PR 描述
- **成本控制**：CI/CD 场景用 V3 而非 R1，控制 API 费用
- **常见误区**：CI/CD 中用 R1 做代码审查——成本太高，V3 的审查质量已经足够

### 第7章：本地部署

#### 7.1 蒸馏模型概览
- **为什么需要蒸馏模型**：满血版 671B 需要数百 GB 显存，普通硬件无法运行
- **蒸馏模型列表**：
  - DeepSeek-R1-Distill-Qwen-1.5B（3GB，入门级）
  - DeepSeek-R1-Distill-Qwen-7B（5GB，推荐）
  - DeepSeek-R1-Distill-Llama-8B（5GB，推荐）
  - DeepSeek-R1-Distill-Qwen-14B（15GB，性能级）
  - DeepSeek-R1-Distill-Qwen-32B（25GB，发烧级）
  - DeepSeek-R1-Distill-Llama-70B（50GB+，工作站级）
- **蒸馏模型的性能**：7B 蒸馏版接近 o1-mini 的推理能力
- **常见误区**：蒸馏模型 = 低质量——蒸馏模型继承了 R1 的推理能力，在同等参数量下远超普通模型

#### 7.2 Ollama 本地部署
- **安装 Ollama**：`curl -fsSL https://ollama.ai/install | bash`
- **下载模型**：
  ```bash
  ollama pull deepseek-r1:7b      # 推理模型 7B
  ollama pull deepseek-coder:6.7b  # 代码模型 6.7B
  ```
- **运行模型**：`ollama run deepseek-r1:7b`
- **API 服务**：`ollama serve` 启动 REST API，兼容 OpenAI 格式
- **硬件要求**：
  - 1.5B：4GB VRAM，CPU 可跑
  - 7B/8B：8GB VRAM（RTX 4060）
  - 14B：12GB VRAM（RTX 4070）
  - 32B：24GB VRAM（RTX 4090）
- **常见误区**：本地模型能替代云端 API——7B 蒸馏版在复杂任务上仍有差距，适合辅助任务

#### 7.3 vLLM 生产级部署
- **vLLM 简介**：高性能 LLM 推理引擎，支持并发、量化、长上下文
- **安装 vLLM**：`pip install vllm`
- **启动推理服务**：
  ```bash
  python -m vllm.entrypoints.openai.api_server \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --tensor-parallel-size 1 \
    --max-model-len 32768
  ```
- **多卡部署**：`--tensor-parallel-size` 控制并行 GPU 数量
- **量化部署**：AWQ/GPTQ 量化降低显存需求
- **性能优化**：Flash Attention、KV Cache、Continuous Batching
- **常见误区**：vLLM 只能部署蒸馏模型——vLLM 也能部署满血版 671B，但需要 8 卡 H100 级别的硬件

### 第8章：生产化与成本优化

#### 8.1 API 成本优化策略
- **缓存命中优化**：DeepSeek API 支持前缀缓存，缓存命中的 token 价格降低 75%
  - 保持 system prompt 一致，提高缓存命中率
  - 多轮对话中复用历史消息，避免重复发送
- **模型选择策略**：
  - 简单任务用 V3（$0.27/M input）
  - 复杂推理用 R1（$0.55/M input）
  - 不要所有任务都用 R1——R1 的思考 token 额外消耗
- **max_tokens 控制**：合理设置 max_tokens，避免生成过多无用内容
- **批量处理**：合并多个小请求为一个大请求，减少 API 调用次数
- **成本对比**：同等代码生成质量下，DeepSeek V3 的成本是 Claude Sonnet 的 1/11
- **常见误区**：缓存命中是自动的——需要保持请求的前缀一致才能命中缓存

#### 8.2 企业级部署方案
- **阿里云百炼**：通过 DashScope API 调用 DeepSeek，国内访问稳定
- **AWS Bedrock / Azure**：海外企业级部署
- **私有化部署**：vLLM + 开源模型权重，数据完全不出机房
- **多区域部署**：国内用阿里云百炼，海外用 DeepSeek 官方 API
- **常见误区**：企业必须私有化部署——如果数据合规允许，API 调用比私有化部署成本低得多

#### 8.3 DeepSeek 的局限性与替代方案
- **已知局限**：
  - R1 的 Function Calling 能力不如 Claude Sonnet（Tau-bench 53.5% vs Claude 80%+）
  - 长上下文处理不如 Gemini（64K vs 1M）
  - 幻觉问题虽有改善但仍存在
  - 英文能力不如中文能力突出
  - API 偶有不稳定（高峰期可能限流）
- **何时用 DeepSeek vs 其他模型**：
  - 预算有限 → DeepSeek V3
  - 需要强推理 → DeepSeek R1
  - 需要超长上下文 → Gemini 2.5 Pro
  - 需要最强 Function Calling → Claude Sonnet 4
  - 需要开源 + 本地部署 → DeepSeek 蒸馏版 + Ollama
- **DeepSeek 的未来**：V3.2 持续优化 Agent 能力、蒸馏模型生态成熟、更多企业级功能
- **常见误区**：DeepSeek 能完全替代 GPT-4/Claude——在 Function Calling、长上下文、多模态等场景仍有差距

---

## 🎯 学习路径建议

```
API 开发者（1天）:
  第1-3章 → 认识 DeepSeek、快速上手、API 接口
  → 能用 DeepSeek API 进行代码生成和对话

效率工程师（1-2天）:
  第4-5章 → 推理模式、Function Calling、Agent 构建
  → 能构建 AI Agent 和 RAG 系统

全栈开发者（2-3天）:
  第6-7章 → 代码生成实战、本地部署
  → 能在本地部署 DeepSeek 并集成到开发流程

团队负责人（1天）:
  第8章 → 成本优化、企业部署、局限性
  → 能为团队制定 DeepSeek 使用策略
```

---

## 📚 与其他 AI 编程模型的对照

| 维度 | DeepSeek V3.2 | DeepSeek R1 | GPT-4o | Claude Sonnet 4 | Gemini 2.5 Pro |
|------|--------------|-------------|--------|-----------------|----------------|
| 开源 | ✅ MIT | ✅ MIT | ❌ | ❌ | ❌ |
| 输入价格 | $0.27/M | $0.55/M | $2.50/M | $3/M | $1.25/M |
| 输出价格 | $1.10/M | $2.19/M | $10/M | $15/M | $10/M |
| 上下文 | 128K | 64K | 128K | 200K | 1M |
| 代码生成 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 推理能力 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Function Calling | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 中文能力 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| 本地部署 | ✅ 蒸馏版 | ✅ 蒸馏版 | ❌ | ❌ | ❌ |
| 缓存折扣 | ✅ 75% off | ✅ 75% off | ❌ | ❌ | ❌ |

---

*本大纲遵循以下写作原则：每节以白板式通俗介绍开场 → 深入原理与面试级知识点 → 配合代码示例（"比如下面的程序..."）→ 覆盖常见用法与误区 → 结尾总结要点衔接下一节。*
