# 上下文记忆配置

大模型本身是无状态的——每次对话都是全新的开始。但OpenClaw可以通过上下文记忆功能，让AI"记住"之前的对话内容。这一章，我们来学习如何配置上下文记忆。

## 为什么需要上下文记忆

**没有记忆的对话：**

```
用户：我叫张三
AI：你好，张三！

用户：我叫什么名字？
AI：抱歉，我不知道您的名字。
```

**有记忆的对话：**

```
用户：我叫张三
AI：你好，张三！

用户：我叫什么名字？
AI：您叫张三。
```

**上下文记忆的作用：**

1. **连续对话**：理解上下文，进行连贯的多轮对话
2. **个性化**：记住用户偏好，提供个性化服务
3. **任务延续**：记住之前的任务状态，继续执行

## 开启上下文记忆功能

**配置开启：**

```yaml
# ~/.openclaw/config.yaml

memory:
  enabled: true
  
  # 记忆类型
  type: hybrid  # short_term, long_term, hybrid
  
  # 存储位置
  storage: ~/.openclaw/memory/
```

**记忆类型对比：**

| 类型 | 描述 | 适用场景 |
|------|------|---------|
| short_term | 短期记忆，仅当前会话有效 | 单次对话 |
| long_term | 长期记忆，跨会话持久化 | 个人偏好 |
| hybrid | 混合模式，结合两者优点 | 推荐使用 |

## 配置上下文窗口大小

上下文窗口决定了AI能"记住"多少内容。

**配置窗口大小：**

```yaml
# ~/.openclaw/config.yaml

memory:
  # 短期记忆配置
  short_term:
    # 最大消息数
    max_messages: 50
    
    # 最大Token数
    max_tokens: 4000
    
    # 滑动窗口策略
    strategy: sliding  # sliding, summary
    
  # 长期记忆配置
  long_term:
    # 是否启用
    enabled: true
    
    # 存储上限
    max_entries: 1000
    
    # 记忆提取数量
    retrieval_count: 10
```

**窗口大小的影响：**

```
窗口太小：
  优点：响应快，成本低
  缺点：记不住太多内容

窗口太大：
  优点：记住更多上下文
  缺点：响应慢，成本高
```

**推荐配置：**

| 场景 | 消息数 | Token数 |
|------|--------|---------|
| 简单问答 | 10 | 2000 |
| 日常对话 | 30 | 4000 |
| 复杂任务 | 50 | 8000 |
| 长文档处理 | 100 | 16000 |

## 记忆管理

**查看记忆：**

```bash
# 查看当前会话记忆
openclaw memory show --session

# 查看长期记忆
openclaw memory show --long-term

# 搜索记忆
openclaw memory search "项目"
```

**输出示例：**

```
长期记忆（共23条）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. [2026-03-28] 用户姓名：张三
2. [2026-03-27] 用户偏好Python编程
3. [2026-03-26] 项目名称：my_app，使用FastAPI
4. [2026-03-25] 用户邮箱：zhangsan@example.com
...

短期记忆（当前会话，共5条）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. [10:30] 用户询问项目进度
2. [10:32] 用户提到明天有会议
3. [10:35] 用户请求生成报告
...
```

**编辑记忆：**

```bash
# 添加记忆
openclaw memory add "用户喜欢简洁的回答风格"

# 删除记忆
openclaw memory delete mem_123

# 清除短期记忆
openclaw memory clear --short-term

# 清除所有记忆
openclaw memory clear --all --confirm
```

**记忆重要性：**

```yaml
# ~/.openclaw/config.yaml

memory:
  # 重要性评估
  importance:
    enabled: true
    
    # 重要信息关键词
    keywords:
      - "姓名"
      - "邮箱"
      - "偏好"
      - "项目"
      - "截止日期"
    
    # 自动提取规则
    extract:
      - pattern: "我叫(.+)"
        field: "name"
      - pattern: "我的邮箱是(.+)"
        field: "email"
      - pattern: "我喜欢(.+)"
        field: "preference"
```

## 记忆压缩与摘要

当对话过长时，OpenClaw会自动压缩记忆。

**配置压缩策略：**

```yaml
# ~/.openclaw/config.yaml

memory:
  compression:
    enabled: true
    
    # 触发条件
    trigger:
      # 消息数超过阈值
      max_messages: 30
      
      # Token数超过阈值
      max_tokens: 3000
    
    # 压缩方式
    method: summary  # summary, key_points
    
    # 压缩后保留的消息数
    keep_recent: 5
```

**压缩效果：**

```
压缩前（30条消息）：
  用户：帮我写一个登录功能
  AI：好的，我来帮你...
  用户：添加验证码
  AI：已添加...
  ...（更多对话）
  用户：测试一下
  AI：测试结果...

压缩后：
  摘要：用户请求开发登录功能，包括验证码、密码加密等，
        已完成代码编写和测试，测试通过。
  最近5条消息：
    用户：测试一下
    AI：测试结果...
    ...
```

## 记忆持久化

**配置持久化：**

```yaml
# ~/.openclaw/config.yaml

memory:
  persistence:
    enabled: true
    
    # 存储格式
    format: json  # json, sqlite
    
    # 自动保存间隔（秒）
    auto_save: 60
    
    # 备份配置
    backup:
      enabled: true
      interval: 86400  # 每天备份
      max_backups: 7
```

**导入导出记忆：**

```bash
# 导出记忆
openclaw memory export > memory_backup.json

# 导入记忆
openclaw memory import < memory_backup.json

# 从其他Agent导入
openclaw memory import --from work_assistant
```

## 隐私与安全

**敏感信息过滤：**

```yaml
# ~/.openclaw/config.yaml

memory:
  privacy:
    # 敏感信息检测
    sensitive_detection: true
    
    # 敏感信息类型
    sensitive_types:
      - password
      - api_key
      - credit_card
      - id_card
    
    # 处理方式
    action: mask  # mask, exclude, encrypt
```

**效果：**

```
用户：我的密码是abc123

记忆存储：
  用户提供了密码（已脱敏）：***
```

**记忆访问控制：**

```yaml
memory:
  access_control:
    # 记忆隔离
    isolate_by_user: true
    
    # 记忆共享
    shared_memories:
      - name: common_knowledge
        users: ["user1", "user2"]
```

---

通过上下文记忆配置，OpenClaw可以记住你的偏好和历史对话，提供更个性化的服务。下一章，我们来学习数据分析与优化。
