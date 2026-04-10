# 8.1 API 成本优化策略

> **DeepSeek 已经很便宜了，但优化后还能再省 50%——缓存命中 + 模型选择 + max_tokens 控制三管齐下。**

---

## 这一节在讲什么？

DeepSeek 的 API 价格已经是主流模型中最低的，但"便宜"不等于"不需要优化"。一个每天处理 100 万 token 的应用，优化前月费可能 $50，优化后可能只要 $15。这一节我们学习三个成本优化策略：缓存命中优化、模型选择策略、max_tokens 控制。

---

## 缓存命中优化

DeepSeek API 支持前缀缓存——缓存命中的 token 价格降低 75%（V3: $0.07/M vs $0.27/M，R1: $0.14/M vs $0.55/M）。

### 提高缓存命中率的策略

**策略一：固定 System Prompt**

```python
# ✅ 好的做法：所有请求使用相同的 system prompt
SYSTEM_PROMPT = "你是一个专业的编程助手，用中文回答。"

# ❌ 差的做法：每次请求都改 system prompt
system_prompt = f"你是一个编程助手，当前时间：{datetime.now()}"
```

**策略二：多轮对话复用历史消息**

```python
# ✅ 好的做法：多轮对话中，之前的消息全部保留
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "什么是快速排序？"},
    {"role": "assistant", "content": "快速排序是一种..."},
    {"role": "user", "content": "能写一个 Python 实现吗？"}  # 新消息
]
# 前三条消息的 token 都能命中缓存！

# ❌ 差的做法：每轮只发最新消息
messages = [
    {"role": "user", "content": "能写一个 Python 实现吗？"}
]
# 没有历史消息，缓存不会命中
```

**策略三：使用相同的 Prompt 模板**

```python
# ✅ 好的做法：使用模板，保持前缀一致
template = "请审查以下代码，重点关注安全性和性能：\n\n{code}"
prompt = template.format(code=user_code)

# ❌ 差的做法：每次用不同的格式
prompt1 = "帮我看看这段代码有没有问题：" + code
prompt2 = "审查代码：" + code
prompt3 = "代码审查请求：" + code
```

---

## 模型选择策略

```
任务复杂度          → 推荐模型       → 预估成本（1000 token）
─────────────────────────────────────────────────────
简单问答            → V3             → $0.001
代码生成            → V3             → $0.002
代码审查            → V3             → $0.002
数学推理            → R1             → $0.005
复杂调试            → R1             → $0.008
算法设计            → R1             → $0.010

❌ 浪费：所有任务都用 R1
   → 简单问答也花 $0.005，是 V3 的 5 倍

✅ 优化：按任务复杂度选模型
   → 简单任务用 V3，复杂推理用 R1
   → 综合成本降低 50%+
```

---

## max_tokens 控制

```python
# ❌ 浪费：max_tokens 设太大
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "1+1等于几？"}],
    max_tokens=8192  # AI 只需要 5 个 token，但 max_tokens 设了 8192
)
# 虽然不会多收费（按实际 token 计费），但大 max_tokens 可能导致模型生成不必要的冗长内容

# ✅ 优化：根据任务设置合理的 max_tokens
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "1+1等于几？"}],
    max_tokens=100  # 简单问答，100 token 足够
)
```

---

## 成本对比

```
优化前 vs 优化后（每天 100 万 token 的应用）：

  优化前：
  - 所有任务都用 R1
  - 不利用缓存
  - max_tokens 统一设 8192
  → 月费：~$150

  优化后：
  - 70% 任务用 V3，30% 用 R1
  - 缓存命中率 50%
  - max_tokens 按任务设置
  → 月费：~$30

  节省 80%！
```

---

## 常见误区

**误区一：缓存命中是自动的**

缓存命中需要请求前缀一致。如果你每次请求都改 system prompt 或消息顺序，缓存就不会命中。保持前缀一致是提高缓存命中率的关键。

**误区二：DeepSeek 已经很便宜了，不需要优化**

"便宜"是相对的。一个每天处理 1000 万 token 的应用，优化前月费 $1500，优化后 $300——省下的 $1200 够买一张 RTX 4090 了。

**误区三：max_tokens 设越大越好**

max_tokens 只是上限，模型不会为了"用满"它而生成更多内容。但设太大可能让模型生成不必要的冗长回答，浪费 output token。

**误区四：R1 的思考过程不需要优化**

R1 的思考 token 按 output 价格计费，可能占总费用的 50% 以上。对于简单任务，用 V3 可以完全避免思考 token 的消耗。

---

## 小结

这一节我们学习了三个成本优化策略：缓存命中优化（固定 system prompt、复用历史消息、使用模板）、模型选择策略（简单用 V3、复杂用 R1）、max_tokens 控制（按任务设置合理值）。综合优化可以降低 50-80% 的 API 费用。下一节我们学习企业级部署方案。
