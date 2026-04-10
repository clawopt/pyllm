# 6.2 代码调试与错误修复

> **R1 的推理能力在调试场景下特别有用——它会"深度思考"错误原因，给出更准确的根因分析。**

---

## 这一节在讲什么？

调试是开发者最耗时的活动之一。DeepSeek 在调试场景下有两个优势：V3 快速分析简单错误，R1 深度推理复杂错误的根因。这一节我们学习用 DeepSeek 进行错误诊断、代码审查和测试驱动调试。

---

## 错误诊断

### 简单错误：用 V3

```python
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{
        "role": "user",
        "content": """以下代码报错：TypeError: Cannot read properties of undefined (reading 'id')
        
function getUser(req, res) {
    const user = users.find(u => u.email === req.body.email);
    return user.id;  // 第 3 行报错
}

请分析原因并修复"""
    }],
    temperature=0.1
)
```

V3 能快速识别问题：`users.find()` 可能返回 `undefined`，直接访问 `user.id` 会报错。修复方案是添加空值检查。

### 复杂错误：用 R1

```python
response = client.chat.completions.create(
    model="deepseek-reasoner",
    messages=[{
        "role": "user",
        "content": """以下代码在并发场景下偶尔出现数据不一致的问题：

async function transfer(from, to, amount) {
    const sender = await db.findAccount(from);
    const receiver = await db.findAccount(to);
    if (sender.balance < amount) throw new Error("Insufficient");
    sender.balance -= amount;
    receiver.balance += amount;
    await db.updateAccount(sender);
    await db.updateAccount(receiver);
}

请分析并发问题的根因，并给出修复方案"""
    }],
    max_tokens=16384
)
```

R1 会深度推理：两个 `findAccount` 操作之间存在时间窗口，另一个并发请求可能在这期间修改了余额，导致数据不一致。R1 的思考过程会详细分析竞态条件的各种场景，最终给出加锁或使用事务的修复方案。

---

## 代码审查

```python
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{
        "role": "system",
        "content": "你是一个代码审查专家。请审查代码的安全性、性能和正确性。"
    }, {
        "role": "user",
        "content": "审查以下代码：\n" + code
    }],
    response_format={"type": "json_object"},
    temperature=0.1
)
```

---

## 测试驱动调试

```python
# 第一步：让 DeepSeek 写复现测试
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{
        "role": "user",
        "content": """用户报告了一个 bug：当购物车中有 0 元商品时，总价计算错误。
请写一个 Jest 测试来复现这个 bug。"""
    }]
)

# 第二步：运行测试确认失败

# 第三步：让 DeepSeek 根据失败信息修复
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{
        "role": "user",
        "content": "测试失败了，错误信息：Expected 100, received 0。请修复 totalPrice 函数。"
    }]
)
```

---

## 常见误区

**误区一：只给错误信息不给代码**

DeepSeek 需要看到相关代码才能分析根因。只给错误信息，AI 只能给出泛泛的建议。

**误区二：所有调试都用 R1**

简单错误用 V3 就够了——更快更便宜。R1 的推理能力在复杂调试场景下才有价值。

**误区三：AI 的修复方案直接用**

AI 的修复可能引入新 bug。建议修复后运行测试，确认没有引入新问题。

**误区四：R1 的思考过程不需要看**

思考过程是 R1 调试的核心价值——它告诉你 AI 是怎么推导出根因的。如果 AI 的修复方案不对，你可以从思考过程中找到推理出错的步骤。

---

## 小结

这一节我们学习了用 DeepSeek 进行代码调试：简单错误用 V3 快速分析，复杂错误用 R1 深度推理根因；代码审查用 V3 + JSON Output 结构化输出；测试驱动调试先写复现测试再修复。核心原则是"给 AI 足够的上下文"——错误信息 + 相关代码 + 复现步骤，缺一不可。下一节我们看 DeepSeek 在 CI/CD 中的应用。
