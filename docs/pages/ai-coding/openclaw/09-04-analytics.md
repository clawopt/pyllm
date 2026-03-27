# 数据分析与优化

OpenClaw提供了丰富的数据分析功能，帮助你了解使用情况并优化体验。这一章，我们来学习如何利用这些功能。

## 生成分析报告

**生成使用报告：**

```bash
# 生成今日报告
openclaw analytics report --today

# 生成周报
openclaw analytics report --week

# 生成月报
openclaw analytics report --month

# 自定义时间范围
openclaw analytics report --from "2026-03-01" --to "2026-03-28"
```

**报告示例：**

```markdown
# OpenClaw 使用报告

报告周期：2026年3月1日 - 2026年3月28日

## 使用概览

| 指标 | 数值 | 环比变化 |
|------|------|---------|
| 总消息数 | 1,234 | +15% |
| 总Token数 | 456,789 | +12% |
| 活跃天数 | 25 | +3天 |
| 平均每日消息 | 49 | +5 |

## 使用分布

### 按时间段
```
00-06: ████ 12%
06-12: ████████████ 35%
12-18: ██████████ 30%
18-24: ██████ 23%
```

### 按渠道
```
Web:     ████████████ 45%
Feishu:  ██████████ 38%
Telegram: ████ 17%
```

### 按功能
```
对话:    ████████████ 40%
文件操作: ██████ 20%
代码生成: █████ 17%
搜索:    ████ 13%
其他:    ███ 10%
```

## 成本分析

| 项目 | 用量 | 费用 |
|------|------|------|
| 阿里云API | 456,789 tokens | ¥45.68 |
| DeepSeek API | 123,456 tokens | ¥0.17 |
| **总计** | - | **¥45.85** |

## 效率提升

- 自动化任务节省时间：约 15 小时
- 代码生成节省时间：约 8 小时
- 信息检索节省时间：约 5 小时
- **总计节省：约 28 小时**

## 建议

1. 建议启用Fallback模型，降低成本
2. 高峰期（上午）使用量较大，可考虑错峰
3. 文件操作使用频繁，建议配置自动化规则
```

## 自动优化建议

OpenClaw会根据使用数据自动给出优化建议。

**查看优化建议：**

```bash
openclaw analytics suggestions
```

**输出示例：**

```
优化建议
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💰 成本优化

1. 启用Fallback模型
   当前：100%使用qwen-plus
   建议：qwen-plus → deepseek-chat（备选）
   预计节省：¥15-20/月
   
2. 调整模型选择策略
   发现：80%的请求不需要高级模型
   建议：简单问题使用qwen-turbo
   预计节省：¥10-15/月

⚡ 性能优化

1. 减少上下文窗口
   当前：max_tokens=8000
   建议：max_tokens=4000
   预计提升：响应速度+30%

2. 启用缓存
   发现：15%的问题重复
   建议：启用回答缓存
   预计提升：响应速度+20%

🔧 功能优化

1. 配置自动化规则
   发现：每天10点执行相同任务
   建议：配置定时任务自动执行
   
2. 技能整合
   发现：频繁组合使用file_ops和email
   建议：创建组合技能简化操作
```

**应用优化建议：**

```bash
# 自动应用所有建议
openclaw analytics suggestions --apply-all

# 选择性应用
openclaw analytics suggestions --apply 1,3,5

# 查看应用效果
openclaw analytics suggestions --preview
```

## A/B测试配置

对于不确定的配置调整，可以使用A/B测试。

**创建A/B测试：**

```bash
openclaw experiment create \
  --name "model_comparison" \
  --description "对比qwen-plus和qwen-turbo的效果" \
  --variants 2 \
  --duration 7  # 天
```

**配置测试参数：**

```yaml
# ~/.openclaw/experiments/model_comparison.yaml

experiment:
  name: model_comparison
  description: 对比qwen-plus和qwen-turbo的效果
  
  variants:
    - name: control
      weight: 50  # 50%流量
      config:
        model: qwen-plus
    
    - name: treatment
      weight: 50  # 50%流量
      config:
        model: qwen-turbo
  
  metrics:
    - response_time
    - user_satisfaction
    - cost_per_request
  
  duration: 7  # 天
```

**查看测试结果：**

```bash
openclaw experiment results model_comparison
```

**输出示例：**

```
A/B测试结果：model_comparison
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

测试周期：2026-03-21 - 2026-03-28
样本量：1,234次请求

┌─────────────┬─────────────┬─────────────┐
│ 指标        │ qwen-plus   │ qwen-turbo  │
├─────────────┼─────────────┼─────────────┤
│ 响应时间    │ 1.2s        │ 0.8s ✓      │
│ 用户满意度  │ 4.5/5       │ 4.3/5       │
│ 单次成本    │ ¥0.08       │ ¥0.02 ✓     │
│ 错误率      │ 0.5%        │ 0.8%        │
└─────────────┴─────────────┴─────────────┘

结论：
- qwen-turbo在响应时间和成本上占优
- qwen-plus在用户满意度上略高
- 建议：简单问题使用turbo，复杂问题使用plus

推荐操作：
openclaw config set model.strategy hybrid
```

**应用测试结果：**

```bash
# 应用胜出方案
openclaw experiment apply model_comparison --winner treatment

# 或手动调整
openclaw config set model.primary qwen-turbo
openclaw config set model.fallback qwen-plus
```

## 监控与告警

**配置监控：**

```yaml
# ~/.openclaw/config.yaml

monitoring:
  enabled: true
  
  # 监控指标
  metrics:
    - response_time
    - error_rate
    - token_usage
    - cost
  
  # 告警规则
  alerts:
    - name: high_error_rate
      condition: error_rate > 5%
      action: notify
      channels: [feishu, email]
    
    - name: high_cost
      condition: daily_cost > 10
      action: notify
      channels: [feishu]
    
    - name: slow_response
      condition: avg_response_time > 5s
      action: notify
      channels: [feishu]
```

**告警示例：**

```
[OpenClaw告警]

⚠️ 错误率过高

时间：2026-03-28 10:30
指标：error_rate
当前值：6.2%
阈值：5%

最近错误：
1. API超时 (3次)
2. 模型响应异常 (2次)

建议操作：
1. 检查API状态
2. 切换到备用模型

[查看详情] [忽略] [禁用告警]
```

**查看监控面板：**

```bash
# 终端监控
openclaw monitor

# 或访问Web监控
open http://localhost:18789/monitor
```

---

通过数据分析与优化，你可以持续改进OpenClaw的使用体验，让它更高效、更经济地为你服务。至此，我们已经完成了OpenClaw教程的全部内容。希望这个强大的AI助手能够真正提升你的工作效率！
