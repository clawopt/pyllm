# 4.3 max_tokens 与思考过程的配合

> **R1 的 max_tokens 包含思考过程——设太小，AI "想"到一半就被掐断了，最终回答可能为空。**

---

## 这一节在讲什么？

max_tokens 是控制 AI 输出长度的参数，但在 R1 和 V3 中的含义不同。V3 的 max_tokens 只限制最终回答的长度；R1 的 max_tokens 限制的是思考过程 + 最终回答的总长度。如果你不理解这个区别，可能会遇到"R1 输出为空"的诡异问题——不是因为 AI 不想回答，而是思考过程把 max_tokens 用完了，留给最终回答的空间为零。这一节我们详细讲解 max_tokens 在两种模式下的差异和设置建议。

---

## V3 的 max_tokens

V3 的 max_tokens 限制模型生成的 token 数（不包括输入）：

```
V3 的 max_tokens 含义：

  输入消息（不计入 max_tokens）
  → 模型生成（计入 max_tokens）
  → 如果生成超过 max_tokens，输出被截断

  示例：
  max_tokens=4096
  → 模型最多生成 4096 token 的回答
  → 如果回答需要 5000 token，输出会在 4096 处被截断
  → finish_reason 为 "length"
```

V3 的 max_tokens 设置建议：

| 场景 | 推荐 max_tokens | 说明 |
|------|----------------|------|
| 简单问答 | 1024 | 短回答足够 |
| 代码生成 | 4096 | 大部分代码不超过 4096 token |
| 长文生成 | 8192（Beta） | 需要使用 Beta 端点 |

---

## R1 的 max_tokens

R1 的 max_tokens 限制的是**思考过程 + 最终回答的总长度**：

```
R1 的 max_tokens 含义：

  输入消息（不计入 max_tokens）
  → 思考过程（计入 max_tokens）  ← 关键区别！
  → 最终回答（计入 max_tokens）
  → 如果思考过程 + 最终回答超过 max_tokens，输出被截断

  示例：
  max_tokens=8192
  思考过程消耗 7000 token
  → 留给最终回答的空间只有 1192 token
  → 如果最终回答需要 2000 token，输出会被截断

  极端情况：
  max_tokens=4096
  思考过程消耗 4000+ token
  → 留给最终回答的空间为 0
  → 最终回答为空字符串！
```

这就是为什么有些用户会遇到"R1 输出为空"的问题——不是模型不想回答，而是思考过程把 max_tokens 用完了。

---

## R1 的 max_tokens 设置建议

| 场景 | 推荐 max_tokens | 说明 |
|------|----------------|------|
| 简单推理 | 8192 | 思考过程通常 2K-5K token |
| 中等推理 | 16384 | 思考过程通常 5K-10K token |
| 复杂推理 | 32768 | 思考过程可能 10K-20K token |
| 极端复杂 | 65536 | 数学证明、复杂算法设计 |

**关键原则：R1 的 max_tokens 至少设 8192，宁可设大也不要设小。**

```python
# ❌ 危险：R1 的 max_tokens 设太小
response = client.chat.completions.create(
    model="deepseek-reasoner",
    messages=[{"role": "user", "content": "证明根号2是无理数"}],
    max_tokens=4096  # 思考过程可能就用完了！
)

# ✅ 安全：R1 的 max_tokens 设大一些
response = client.chat.completions.create(
    model="deepseek-reasoner",
    messages=[{"role": "user", "content": "证明根号2是无理数"}],
    max_tokens=16384  # 给思考过程和最终回答留足空间
)
```

---

## 检测思考过程是否被截断

你可以通过 `finish_reason` 判断输出是否被截断：

```python
response = client.chat.completions.create(
    model="deepseek-reasoner",
    messages=[{"role": "user", "content": "..."}],
    max_tokens=8192
)

finish_reason = response.choices[0].finish_reason

if finish_reason == "length":
    print("⚠️ 输出被截断！思考过程可能占用了太多 token")
    print("建议增大 max_tokens 或简化问题")
elif finish_reason == "stop":
    print("✅ 输出完整")
```

---

## 常见误区

**误区一：R1 的 max_tokens 只限制最终回答**

不是。R1 的 max_tokens 限制的是思考过程 + 最终回答的总长度。如果你按 V3 的习惯设 max_tokens=4096，R1 的思考过程可能就把 4096 token 用完了，最终回答为空。

**误区二：max_tokens 设越大越好**

设太大不会影响质量，但会影响费用——max_tokens 只是上限，模型不会为了"用满" max_tokens 而生成更多内容。但设太大可能让你放松警惕，不注意实际的 token 消耗。建议根据任务复杂度选择合适的值。

**误区三：R1 输出为空是 API 的 bug**

不是 bug，是 max_tokens 设太小。R1 的思考过程消耗了大量 token，留给最终回答的空间不足。解决方法是增大 max_tokens。

**误区四：思考过程的 token 不计入费用**

思考过程的 token 按 output 价格计费——跟最终回答的 token 价格一样。R1 的思考过程可能消耗 10K-30K token，这会显著增加 API 费用。使用 R1 时要特别注意 token 消耗。

---

## 小结

这一节我们学习了 max_tokens 在 V3 和 R1 中的区别：V3 的 max_tokens 只限制最终回答，R1 的 max_tokens 限制思考过程 + 最终回答的总长度。R1 的 max_tokens 至少设 8192，复杂推理设 16384-32768。如果 R1 输出为空，检查 max_tokens 是否设太小。下一章我们进入 Function Calling 与 Agent 构建。
