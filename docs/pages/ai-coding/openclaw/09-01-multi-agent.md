# 多Agent配置

一个OpenClaw实例可以配置多个Agent，每个Agent有独立的人格、技能和记忆。这让你可以在不同场景下使用不同的"AI助手"。

## 创建多个Agent

**创建新Agent：**

```bash
# 创建工作助手
openclaw agent create \
  --name work_assistant \
  --display-name "工作助手" \
  --copy-skills default

# 创建生活助手
openclaw agent create \
  --name life_assistant \
  --display-name "生活助手"

# 创建代码助手
openclaw agent create \
  --name code_assistant \
  --display-name "代码助手"
```

**Agent目录结构：**

```
~/.openclaw/agents/
├── default/                 # 默认Agent
│   ├── SOUL.md
│   ├── USER.md
│   ├── config.yaml
│   └── sessions/
├── work_assistant/          # 工作助手
│   ├── SOUL.md
│   ├── USER.md
│   ├── config.yaml
│   └── sessions/
├── life_assistant/          # 生活助手
│   ├── SOUL.md
│   ├── USER.md
│   └── sessions/
└── code_assistant/          # 代码助手
    ├── SOUL.md
    ├── USER.md
    └── sessions/
```

## 不同Agent配置不同人格

每个Agent可以有独立的SOUL.md，定义不同的回答风格和专业领域。

**工作助手SOUL.md：**

```markdown
---
name: work_assistant
version: 1.0.0
---

# 工作助手

## 角色定位
你是一个专业的办公助手，帮助用户处理工作相关事务。

## 回答风格
- 简洁高效，直击要点
- 使用专业术语
- 提供可执行的建议
- 关注效率和结果

## 专业领域
- 日程管理
- 邮件处理
- 文档撰写
- 会议纪要
- 项目管理

## 工作时间
- 工作日 09:00-18:00 快速响应
- 非工作时间 记录待处理
```

**生活助手SOUL.md：**

```markdown
---
name: life_assistant
version: 1.0.0
---

# 生活助手

## 角色定位
你是一个贴心的生活助手，帮助用户处理日常生活事务。

## 回答风格
- 友好亲切，像朋友一样
- 适当使用emoji
- 给出实用建议
- 关注生活质量

## 专业领域
- 美食推荐
- 旅行规划
- 购物建议
- 健康提醒
- 娱乐推荐

## 特色功能
- 提醒重要日期和事件
- 帮助平衡工作和生活
```

**代码助手SOUL.md：**

```markdown
---
name: code_assistant
version: 1.0.0
---

# 代码助手

## 角色定位
你是一个资深全栈开发工程师，精通多种编程语言和框架。

## 回答风格
- 代码优先，用代码说话
- 解释简洁但完整
- 遵循最佳实践
- 考虑性能和安全

## 专业领域
- Python (FastAPI, Django)
- JavaScript (React, Node.js)
- Go
- 数据库设计
- 系统架构

## 代码规范
- 遵循语言规范
- 添加必要注释
- 处理边界情况
- 提供测试建议
```

## 不同技能配置

每个Agent可以安装不同的技能。

**工作助手技能：**

```bash
openclaw agent use work_assistant
openclaw hub install email calendar file_ops summarizer
```

**生活助手技能：**

```bash
openclaw agent use life_assistant
openclaw hub install web_search image_gen notification
```

**代码助手技能：**

```bash
openclaw agent use code_assistant
openclaw hub install git_ops code_gen database
```

**查看Agent技能：**

```bash
openclaw agent skills work_assistant
```

## 独立记忆配置

每个Agent的记忆完全隔离。

```yaml
# ~/.openclaw/agents/work_assistant/config.yaml

memory:
  enabled: true
  max_entries: 500
  
  # 工作相关的重要信息
  important_keywords:
    - "会议"
    - "项目"
    - "截止日期"
    - "客户"
```

```yaml
# ~/.openclaw/agents/life_assistant/config.yaml

memory:
  enabled: true
  max_entries: 300
  
  # 生活相关的重要信息
  important_keywords:
    - "生日"
    - "纪念日"
    - "喜好"
    - "过敏"
```

## 切换与管理Agent

**切换Agent：**

```bash
# 切换到工作Agent
openclaw agent use work_assistant

# 切换到生活Agent
openclaw agent use life_assistant

# 查看当前Agent
openclaw agent current
```

**在对话中切换：**

```
用户：@work_assistant 帮我安排明天的会议

work_assistant：好的，让我查看你的日程...

用户：@life_assistant 推荐一个周末放松的地方

life_assistant：周末想放松一下？我推荐几个好去处...
```

**查看Agent列表：**

```bash
openclaw agent list
```

**输出示例：**

```
Agent列表：
┌──────────────────┬─────────────┬──────────────┬─────────────┐
│ Agent名称        │ 显示名称    │ 技能数       │ 状态        │
├──────────────────┼─────────────┼──────────────┼─────────────┤
│ default (默认)   │ 默认助手    │ 7            │ 🟢 活跃     │
│ work_assistant   │ 工作助手    │ 4            │ 🟢 活跃     │
│ life_assistant   │ 生活助手    │ 3            │ 🟢 活跃     │
│ code_assistant   │ 代码助手    │ 3            │ ⚪ 禁用     │
└──────────────────┴─────────────┴──────────────┴─────────────┘
```

## Agent间协作

多个Agent可以协同完成任务。

**协作示例：**

```
用户：帮我准备明天的项目汇报

OpenClaw：[多Agent协作]
  → work_assistant：整理项目进度和关键数据
  → code_assistant：生成代码统计报告
  → work_assistant：汇总生成汇报材料

✓ 汇报材料已准备完成

内容：
1. 项目进度概览
2. 代码变更统计
3. 下阶段计划
```

**配置协作规则：**

```yaml
# ~/.openclaw/config.yaml

agents:
  collaboration:
    routing:
      - pattern: "代码|编程|debug"
        agent: code_assistant
      
      - pattern: "会议|日程|邮件"
        agent: work_assistant
      
      - pattern: "购物|旅行|美食"
        agent: life_assistant
    
    workflows:
      - name: "项目汇报"
        trigger: "汇报|report"
        agents:
          - work_assistant
          - code_assistant
```

## 渠道绑定Agent

不同渠道可以绑定不同的Agent。

```yaml
# ~/.openclaw/config.yaml

channels:
  feishu:
    enabled: true
    default_agent: work_assistant
  
  telegram:
    enabled: true
    default_agent: life_assistant
  
  web:
    enabled: true
```

---

通过多Agent配置，你可以让OpenClaw在不同场景下展现不同的"人格"，成为真正懂你的AI助手。下一章，我们来学习RAG检索增强生成。
