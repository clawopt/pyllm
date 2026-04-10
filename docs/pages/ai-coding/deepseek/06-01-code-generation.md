# 6.1 代码生成最佳实践

> **好的 Prompt 让 DeepSeek 从"随便写写"变成"精准输出"——关键在于明确语言、框架、需求和约束。**

---

## 这一节在讲什么？

DeepSeek 的代码生成能力很强，但很多用户抱怨"生成的代码不符合预期"——问题通常不在模型，而在 Prompt。一个模糊的 Prompt（"帮我写个登录功能"）会让 AI 猜测你的意图；一个精确的 Prompt（"用 Express + JWT 写一个 POST /auth/login 接口"）能让 AI 给出精准的输出。这一节我们学习代码生成的 Prompt 设计技巧、System Prompt 模板和 Temperature 设置策略。

---

## Prompt 设计原则

### 原则一：明确编程语言和框架

```
❌ 差的 Prompt：
"写一个登录功能"

✅ 好的 Prompt：
"用 Express + TypeScript + JWT 写一个登录接口"
```

### 原则二：明确需求和约束

```
❌ 差的 Prompt：
"写一个 API"

✅ 好的 Prompt：
"写一个 REST API，包含以下接口：
1. POST /auth/register - 用户注册
2. POST /auth/login - 用户登录，返回 JWT token
3. GET /users/me - 获取当前用户信息（需要认证）
使用 bcrypt 哈希密码，JWT 有效期 24 小时"
```

### 原则三：提供参考实现

```
✅ 最好的 Prompt：
"按照以下风格写一个 POST /auth/login 接口：
{参考代码}
要求：使用相同的错误处理模式、相同的响应格式"
```

### 原则四：指定输出格式

```
✅ 指定输出格式：
"只输出代码，不要解释。代码放在 ```python 代码块中。"
```

---

## System Prompt 模板

System Prompt 定义了 AI 的角色和输出规范，对代码生成质量影响很大：

```python
CODE_ASSISTANT_PROMPT = """你是一个专业的编程助手。请遵循以下规范：

1. 使用用户指定的编程语言和框架
2. 代码必须包含类型注解（如果语言支持）
3. 必须包含错误处理
4. 必须包含必要的注释
5. 遵循 SOLID 原则
6. 输出格式：先给出完整代码，再给出简要说明
7. 如果有多种实现方式，给出推荐方案并说明原因
"""
```

---

## Temperature 设置策略

```
代码生成的 Temperature 设置：

  0.0-0.1：完全确定性，适合重复性任务（如代码格式化）
  0.1-0.3：低随机性，适合代码生成（推荐默认值）
  0.3-0.5：中等随机性，适合需要创造性的代码（如算法设计）
  0.5+：高随机性，代码生成不推荐（可能产生语法错误）
```

```python
# 代码生成：低 temperature
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[...],
    temperature=0.1
)

# 算法设计：中等 temperature
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[...],
    temperature=0.4
)
```

---

## 代码生成示例

### 生成 REST API

```python
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": CODE_ASSISTANT_PROMPT},
        {"role": "user", "content": """用 Express + TypeScript 写一个 REST API：
1. POST /api/users - 创建用户
2. GET /api/users/:id - 获取用户
3. PUT /api/users/:id - 更新用户
4. DELETE /api/users/:id - 删除用户
使用 Prisma ORM，SQLite 数据库，包含输入验证和错误处理"""}
    ],
    temperature=0.1,
    max_tokens=4096
)
```

### 生成测试用例

```python
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "你是一个测试工程师，擅长编写全面的测试用例。"},
        {"role": "user", "content": """为以下函数编写 Jest 测试用例，覆盖正常情况和边界情况：

function divide(a: number, b: number): number {
    if (b === 0) throw new Error("Division by zero");
    return a / b;
}"""}
    ],
    temperature=0.1
)
```

---

## 常见误区

**误区一：Prompt 越长越好**

不是。Prompt 应该精确而不是冗长——无关的信息会干扰模型的注意力。只提供跟任务直接相关的信息。

**误区二：代码生成不需要 System Prompt**

System Prompt 对代码生成质量影响很大——它定义了 AI 的角色、输出规范和代码风格。没有 System Prompt，AI 可能输出不符合你期望的代码。

**误区三：Temperature 设 0 最好**

Temperature=0 意味着完全确定性——每次生成相同的代码。这在某些场景下是优点（如代码格式化），但在需要创造性的场景下（如算法设计、架构设计）可能过于保守。建议代码生成用 0.1-0.3。

**误区四：一次生成整个项目**

DeepSeek 一次生成的代码量有限（max_tokens 限制）。大型项目应该分模块生成——先生成项目骨架，再逐个生成模块。

---

## 小结

这一节我们学习了代码生成的最佳实践：Prompt 设计四原则（明确语言框架、明确需求约束、提供参考实现、指定输出格式）、System Prompt 模板、Temperature 设置策略（代码生成用 0.1-0.3）。核心原则是"Prompt 越精确，输出越精准"。下一节我们学习代码调试与错误修复。
