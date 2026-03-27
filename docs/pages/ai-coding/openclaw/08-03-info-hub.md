# 7×24小时信息中枢

在信息爆炸的时代，如何高效地获取和筛选信息是一个挑战。这一章，我们用OpenClaw搭建一个7×24小时运行的信息中枢。

## 场景描述

你希望：
- 每天早上自动收到行业动态汇总
- 重要新闻第一时间推送
- 定期监控特定关键词
- 所有信息自动归档可查

OpenClaw可以成为你的"信息秘书"，全天候为你收集、筛选、推送信息。

## 定时抓取行业动态

**配置信息源：**

```yaml
# ~/.openclaw/config.yaml

info_hub:
  sources:
    # 技术资讯
    - name: hacker_news
      type: rss
      url: https://hnrss.org/frontpage
      schedule: "*/30 * * * *"  # 每30分钟
      filters:
        - "AI"
        - "Python"
        - "LLM"
    
    - name: github_trending
      type: api
      url: https://api.github.com/search/repositories
      schedule: "0 9 * * *"  # 每天9点
      params:
        q: "stars:>100 created:>2026-03-01"
        language: "python"
    
    # 国内资讯
    - name: infoq
      type: web
      url: https://www.infoq.cn/
      schedule: "0 8,12,18 * * *"  # 每天8点、12点、18点
      selectors:
        title: ".article-title"
        link: "a.article-link"
```

**创建信息抓取技能：**

```markdown
# ~/.openclaw/skills/info_hub/SKILL.md

---
name: info_hub
version: 1.0.0
description: 信息聚合与推送
dependencies:
  - web_search
  - agent-browser
permissions:
  - network:access
  - file:write
---

# 信息中枢技能

## 能力描述
定时抓取多个信息源，智能筛选，生成汇总报告并推送。

## 工作流程
1. 按配置的时间表抓取各信息源
2. 提取标题、链接、摘要
3. 应用关键词过滤
4. 去重处理
5. 生成汇总报告
6. 推送到指定渠道

## 配置示例
```yaml
sources:
  - name: hacker_news
    type: rss
    url: https://hnrss.org/frontpage
    filters: ["AI", "Python"]
```
```

**执行抓取：**

```bash
# 手动触发抓取
openclaw skill run info_hub.fetch --source hacker_news

# 抓取所有源
openclaw skill run info_hub.fetch --all
```

## AI生成摘要与建议

抓取到的信息，OpenClaw会自动生成摘要和建议。

**摘要生成提示词：**

```
请对以下信息进行摘要和分析：

## 原始信息
[信息列表]

## 要求
1. 每条信息用一句话概括
2. 标注重要程度（高/中/低）
3. 给出阅读建议
4. 发现信息之间的关联

## 输出格式
### 重要信息（必读）
- [标题] - [一句话摘要] - [重要性原因]

### 一般信息（选读）
- [标题] - [一句话摘要]

### 信息关联
[发现的主题或趋势]

### 阅读建议
[个性化的阅读建议]
```

**生成的摘要示例：**

```markdown
# 行业动态汇总 - 2026年3月28日 09:00

## 🔴 重要信息（必读）

### OpenAI发布GPT-5，推理能力大幅提升
- 摘要：新模型在数学推理和代码生成方面表现优异，成本降低40%
- 重要性：行业重大突破，可能影响产品选型
- 来源：TechCrunch
- 链接：https://...

### DeepSeek开源新版本，推理效率提升3倍
- 摘要：新版本优化了推理效率，支持更长上下文
- 重要性：国产模型的重要进展，值得关注
- 来源：GitHub
- 链接：https://...

## 🟡 一般信息（选读）

### Python 3.13发布预览版
- 摘要：新增JIT编译器，性能提升明显
- 链接：https://...

### 微软发布Copilot新功能
- 摘要：支持更多IDE和编程语言
- 链接：https://...

## 📊 信息关联

今日信息主要围绕"AI模型能力提升"主题：
- GPT-5和DeepSeek都强调了推理能力的改进
- 成本降低是共同趋势
- 开源与闭源的差距在缩小

## 💡 阅读建议

1. 优先阅读GPT-5发布文章，了解最新技术动态
2. DeepSeek的更新对国内用户更有参考价值
3. Python 3.13的JIT功能值得技术关注
```

## 推送至指定渠道

**配置推送渠道：**

```yaml
# ~/.openclaw/config.yaml

info_hub:
  push:
    # 早报推送
    morning:
      time: "08:00"
      channels:
        - feishu
        - email
      template: "daily_briefing"
    
    # 重要信息即时推送
    instant:
      channels:
        - feishu
      criteria:
        importance: "high"
    
    # 周报推送
    weekly:
      time: "Friday 18:00"
      channels:
        - email
      template: "weekly_summary"
```

**推送模板：**

```markdown
<!-- daily_briefing.md -->
# 📰 每日资讯 - {{date}}

{{#each items}}
### {{title}}
{{summary}}

🔗 [阅读原文]({{link}}) | 📌 {{source}}
{{/each}}

---
*由 OpenClaw 自动推送 | 回复"详情"查看完整内容*
```

**飞书推送效果：**

```
┌─────────────────────────────────────┐
│ OpenClaw助手                        │
├─────────────────────────────────────┤
│ 📰 每日资讯 - 2026年3月28日         │
│                                     │
│ 🔴 OpenAI发布GPT-5                  │
│ 新模型推理能力大幅提升，成本降低40% │
│ 🔗 阅读原文 | 📌 TechCrunch         │
│                                     │
│ 🔴 DeepSeek开源新版本               │
│ 推理效率提升3倍，支持更长上下文     │
│ 🔗 阅读原文 | 📌 GitHub             │
│                                     │
│ ─────────────────────────           │
│ 由 OpenClaw 自动推送                │
└─────────────────────────────────────┘
```

## 关键词监控

设置特定关键词，当出现相关信息时即时推送。

**配置关键词监控：**

```yaml
# ~/.openclaw/config.yaml

info_hub:
  monitors:
    - keywords:
        - "OpenAI"
        - "GPT"
      sources:
        - hacker_news
        - infoq
      push: instant
      message: "🚨 OpenAI相关新闻"
    
    - keywords:
        - "竞品A"
        - "竞品B"
      sources:
        - all
      push: instant
      message: "👀 竞品动态"
    
    - keywords:
        - "安全漏洞"
        - "CVE"
      sources:
        - github_advisories
      push: instant
      message: "⚠️ 安全告警"
```

**监控触发示例：**

```
[09:15] OpenClaw助手
⚠️ 安全告警

发现包含"CVE"的信息：

Python requests库存在高危漏洞
- CVE-2026-12345
- 影响版本：< 2.32.0
- 建议：立即升级

🔗 https://github.com/psf/requests/security/advisories
```

## 信息归档与检索

所有抓取的信息会自动归档，支持全文检索。

**归档结构：**

```
~/.openclaw/workspace/info_archive/
├── 2026/
│   ├── 03/
│   │   ├── 28/
│   │   │   ├── hacker_news.json
│   │   │   ├── github_trending.json
│   │   │   └── daily_summary.md
│   │   └── ...
│   └── ...
└── index.db  # 全文索引数据库
```

**检索信息：**

```bash
# 搜索历史信息
openclaw info search "GPT-5"

# 按时间范围搜索
openclaw info search "Python" --from "2026-03-01" --to "2026-03-28"

# 按来源搜索
openclaw info search "AI" --source hacker_news
```

**检索结果：**

```
搜索 "GPT-5" 的结果（共3条）：

┌─────────────────────────────────────────────────────────────┐
│ 1. OpenAI发布GPT-5，推理能力大幅提升                        │
│    时间：2026-03-28 09:00                                   │
│    来源：TechCrunch                                         │
│    摘要：新模型在数学推理和代码生成方面表现优异...          │
├─────────────────────────────────────────────────────────────┤
│ 2. GPT-5 vs Claude 3：谁更强？                              │
│    时间：2026-03-27 14:30                                   │
│    来源：Hacker News                                        │
│    摘要：详细对比两款模型的各项能力...                      │
├─────────────────────────────────────────────────────────────┤
│ 3. 如何迁移到GPT-5 API                                      │
│    时间：2026-03-26 10:15                                   │
│    来源：OpenAI Blog                                        │
│    摘要：迁移指南和兼容性说明...                            │
└─────────────────────────────────────────────────────────────┘
```

---

通过这个案例，你可以看到OpenClaw如何成为你的"信息秘书"，帮你从信息海洋中筛选出有价值的内容。下一章，我们来看如何用OpenClaw搭建个人知识库。
