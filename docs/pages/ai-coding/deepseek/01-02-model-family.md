# 1.2 DeepSeek 的模型家族：V3 与 R1

> **V3 是"快枪手"，R1 是"深思者"——同一个基座，两种性格，用哪个取决于你的任务有多难。**

---

## 这一节在讲什么？

DeepSeek 不是"一个模型"——它是一个模型家族，目前有两条主要产品线：**V3 系列**和 **R1 系列**。V3 是通用模型，代号 `deepseek-chat`，擅长日常对话、代码生成、内容创作，速度快、成本低；R1 是推理模型，代号 `deepseek-reasoner`，擅长数学推理、逻辑分析、复杂编程，会"深度思考"后再回答。两条线共享同一个 671B 参数的 MoE 基座，通过不同的后训练策略分化出不同的能力侧重。此外还有蒸馏模型——从 R1 蒸馏到小参数模型，保留推理能力但降低硬件需求。这一节我们把 DeepSeek 的模型家族讲清楚，帮你建立"遇到什么任务选什么模型"的直觉。

---

## V3 系列（deepseek-chat）：快枪手

V3 系列是 DeepSeek 的通用模型，通过 `deepseek-chat` 这个模型名调用。它的特点是**快、便宜、全能**——日常编码、内容创作、Function Calling 都能胜任。

### V3 的演进历程

```
DeepSeek-V3（2024.12）
  → V3-0324（2025.03）：推理能力增强，前端代码生成优化
    → V3.1（2025.08）：混合推理架构，Agent 能力增强
      → V3.1-Terminus（2025.09）：语言一致性修复
        → V3.2-Exp（2025.09）：实验版
          → V3.2（2025.12）：当前最新版
```

每次升级，API 的调用方式都不变——你只需要指定 `model="deepseek-chat"`，DeepSeek 会自动使用最新版本。这意味着你不需要修改代码就能享受到模型升级带来的性能提升。

### V3 的核心能力

| 能力 | 说明 |
|------|------|
| 代码生成 | 支持 Python、JavaScript、TypeScript、Go、Rust 等 30+ 语言 |
| 内容创作 | 中英文写作、翻译、摘要、改写 |
| Function Calling | 支持工具调用，最多 128 个 Function |
| JSON Output | 强制输出 JSON 格式 |
| 上下文长度 | 128K token |
| 价格 | $0.27/M input, $1.10/M output |

### V3 的适用场景

```
✅ 适合 V3 的场景：
- 日常代码生成（写接口、写组件、写脚本）
- 代码审查和优化建议
- 内容创作和翻译
- Function Calling 和 Agent 任务
- 简单问答和知识查询

❌ 不适合 V3 的场景：
- 复杂数学推理（用 R1）
- 多步骤逻辑分析（用 R1）
- 需要深度思考的算法设计（用 R1）
```

---

## R1 系列（deepseek-reasoner）：深思者

R1 系列是 DeepSeek 的推理模型，通过 `deepseek-reasoner` 这个模型名调用。它的特点是**深度推理、思维链可见、数学和逻辑能力强**——它会先在内部"思考"，然后再给出回答。

### R1 的演进历程

```
DeepSeek-R1（2025.01）：首次发布，推理能力接近 o1
  → R1-0528（2025.05）：推理能力大幅增强，幻觉降低 45-50%
```

R1 的升级同样不需要修改代码——指定 `model="deepseek-reasoner"` 即可。

### R1 的思考过程

R1 最大的特点是"思考过程可见"——在生成最终回答之前，它会先输出一段 `<think)>` 标签内的思考过程：

```
<think)>
让我分析一下这个问题...

首先，我需要理解题目的要求...
然后，我考虑可能的解决方案...
方案 A 的优缺点是...
方案 B 的优缺点是...
综合考虑，方案 B 更优，因为...
</think)>

最终回答：...
```

在 API 调用中，思考过程通过 `reasoning_content` 字段返回，你可以选择展示或隐藏它：

```python
response = client.chat.completions.create(
    model="deepseek-reasoner",
    messages=[{"role": "user", "content": "证明根号2是无理数"}]
)

# 思考过程
print(response.choices[0].message.reasoning_content)

# 最终回答
print(response.choices[0].message.content)
```

### R1 的核心能力

| 能力 | 说明 |
|------|------|
| 数学推理 | AIME 2025 达到 87.5%，接近 o3 |
| 逻辑分析 | GPQA 达到 81.0% |
| 代码调试 | Aider 达到 71.6%，略超 Claude Opus |
| Function Calling | 支持（R1-0528 新增） |
| 上下文长度 | 64K（API）/ 128K（开源版） |
| 价格 | $0.55/M input, $2.19/M output |

### R1 的适用场景

```
✅ 适合 R1 的场景：
- 数学问题求解（方程、证明、计算）
- 复杂逻辑推理（多步骤分析）
- 代码调试和根因分析
- 算法设计和优化
- 需要深度思考的决策支持

❌ 不适合 R1 的场景：
- 简单代码生成（V3 更快更便宜）
- 日常对话和问答（V3 足够）
- Function Calling 密集的 Agent（V3 的工具调用更可靠）
```

---

## 混合推理架构：一个模型，两种模式

从 V3.1 开始，DeepSeek 引入了**混合推理架构**——一个模型同时支持思考模式和非思考模式。这意味着 `deepseek-chat` 和 `deepseek-reasoner` 其实共享同一个基座模型，只是通过不同的参数控制推理深度：

```
DeepSeek-V3.2 基座模型
  ├── deepseek-chat → 非思考模式（快速、低成本）
  └── deepseek-reasoner → 思考模式（深度推理、高成本）
```

在 V3.1+ 的 API 中，你可以通过 `thinking` 参数手动控制模式：

```python
# 非思考模式（等同 deepseek-chat）
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "写一个快速排序"}],
    extra_body={"thinking": {"type": "disabled"}}
)

# 思考模式（等同 deepseek-reasoner）
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "证明快速排序的平均时间复杂度是 O(n log n)"}],
    extra_body={"thinking": {"type": "enabled"}}
)
```

混合推理架构的好处是**灵活性**——你不需要在两个模型之间切换，同一个模型可以根据任务需要选择不同的推理深度。

---

## 蒸馏模型：小参数，大能力

DeepSeek-R1 的满血版有 671B 参数，需要数百 GB 显存才能运行——普通开发者根本用不起。为了解决这个问题，DeepSeek 用 R1 的 80 万条推理样本微调了 Qwen 和 Llama 系列的小模型，得到了**蒸馏模型**：

| 蒸馏模型 | 基座 | 参数量 | 模型大小 | 推荐硬件 | 推理能力 |
|---------|------|--------|---------|---------|---------|
| R1-Distill-Qwen-1.5B | Qwen2.5 | 1.5B | 3GB | CPU 可跑 | 入门级 |
| R1-Distill-Qwen-7B | Qwen2.5 | 7B | 5GB | RTX 4060 | 接近 o1-mini |
| R1-Distill-Llama-8B | Llama3 | 8B | 5GB | RTX 4060 | 接近 o1-mini |
| R1-Distill-Qwen-14B | Qwen2.5 | 14B | 15GB | RTX 4070 | 性能级 |
| R1-Distill-Qwen-32B | Qwen2.5 | 32B | 25GB | RTX 4090 | 发烧级 |
| R1-Distill-Llama-70B | Llama3 | 70B | 50GB+ | 多卡工作站 | 工作站级 |

蒸馏模型的关键洞察是：**推理能力可以通过数据蒸馏传递**——用大模型的推理数据训练小模型，小模型也能获得接近大模型的推理能力。7B 蒸馏版在数学测试上超过了 Qwen3-235B（一个 235B 参数的模型），这证明了蒸馏的高效性。

蒸馏模型都采用 MIT 许可证开源，你可以从 HuggingFace 或 ModelScope 下载。

---

## 模型选择决策树

面对这么多模型，怎么选？这里有一个简单的决策树：

```
你的任务是什么？

  → 日常代码生成 / 对话 / Function Calling
    → deepseek-chat（V3.2）
    → 快速、便宜、全能

  → 数学推理 / 逻辑分析 / 复杂调试
    → deepseek-reasoner（R1）
    → 深度思考、推理强

  → 需要本地部署 / 数据不出本机
    → 蒸馏模型 + Ollama
    → 根据硬件选参数量

  → 预算极其有限
    → deepseek-chat（V3.2）
    → $0.27/M input，最便宜的主流模型

  → 需要超长上下文（>128K）
    → 不适合 DeepSeek，考虑 Gemini 2.5 Pro（1M 上下文）
```

---

## 常见误区

**误区一：`deepseek-chat` 只能聊天**

不是。`deepseek-chat` 是 V3 系列的模型名，"chat"只是表示它是对话模型（Chat Model），不是"只能聊天"。它能写代码、做 Function Calling、生成 JSON——跟 GPT-4o 的 `gpt-4o` 模型名一样，"chat"只是 API 的接口类型。

**误区二：R1 只能做数学**

不是。R1 的推理能力在数学上最突出，但它的代码调试、逻辑分析、算法设计能力也很强。R1 的 Aider 编程测试通过率 71.6%，略超 Claude Opus——这说明 R1 不只是"数学模型"，它是一个通用的强推理模型。

**误区三：蒸馏模型质量差**

恰恰相反。蒸馏模型继承了 R1 的推理能力，在同等参数量下远超普通模型。7B 蒸馏版在 AIME 2024 上超过了 Qwen3-8B（+10.0%），与 Qwen3-235B 相当——一个 7B 的模型达到了 235B 模型的推理水平，这就是蒸馏的力量。

**误区四：V3 和 R1 是两个完全不同的模型**

从 V3.1 开始，它们共享同一个基座模型，只是后训练策略不同。V3 是"非思考模式"，R1 是"思考模式"——就像同一个人的"快思考"和"慢思考"。

---

## 小结

这一节我们梳理了 DeepSeek 的模型家族：V3 系列（`deepseek-chat`）是快枪手——快速、便宜、全能，适合日常编码和 Agent 任务；R1 系列（`deepseek-reasoner`）是深思者——深度推理、思维链可见，适合数学和逻辑任务；蒸馏模型让普通硬件也能运行 DeepSeek 的推理能力。选择模型的核心原则是"按需选择"——简单任务用 V3，复杂推理用 R1，本地部署用蒸馏版。下一节我们深入 DeepSeek 的技术架构，理解 MoE 和混合推理是怎么工作的。
