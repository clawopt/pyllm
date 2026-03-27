# 自定义Skill开发

当ClawHub上的技能不能满足你的需求时，你可以开发自己的技能。OpenClaw的技能开发非常简单，只需要编写一个SKILL.md文件。

## Skill目录结构

一个完整的技能目录结构如下：

```
my_custom_skill/
├── SKILL.md              # 必需：技能定义文件
├── config.yaml           # 可选：默认配置
├── templates/            # 可选：提示词模板
│   └── prompt.tmpl
├── scripts/              # 可选：执行脚本
│   └── main.py
├── tests/                # 可选：测试文件
│   └── test_skill.py
└── README.md             # 可选：文档
```

**最简单的技能只需要一个SKILL.md文件：**

```
simple_skill/
└── SKILL.md
```

## SKILL.md编写规范

SKILL.md由两部分组成：YAML frontmatter（元数据）和Markdown正文（能力描述）。

**基本模板：**

```markdown
---
name: my_skill
version: 1.0.0
description: 技能的简短描述
author: Your Name
email: your@email.com
license: MIT
repository: https://github.com/user/my_skill
tags: [tag1, tag2]
categories: [category1]
dependencies:
  - skill_name: version
permissions:
  - permission_type
---

# 技能名称

## 能力描述
详细描述这个技能能做什么。

## 使用场景
列出适用的场景。

## 调用示例
提供JSON格式的调用示例。

## 配置说明
说明需要配置的参数。

## 注意事项
使用时需要注意的问题。
```

**完整示例：股票查询技能**

```markdown
---
name: stock_query
version: 1.0.0
description: 查询A股、港股、美股实时行情
author: Developer
email: dev@example.com
license: MIT
repository: https://github.com/example/skill-stock
tags: [stock, finance, query]
categories: [finance]
dependencies: []
permissions:
  - network:access
---

# 股票查询技能

## 能力描述
查询股票实时行情信息，支持：
- A股（沪深两市）
- 港股
- 美股

返回信息包括：
- 当前价格
- 涨跌幅
- 成交量
- 市值

## 使用场景
- 查看持仓股票行情
- 监控股票价格变动
- 获取市场概况

## 调用示例

### 查询单只股票
```json
{
  "action": "stock_query.single",
  "params": {
    "code": "600519",
    "market": "cn"
  }
}
```

返回：
```json
{
  "code": "600519",
  "name": "贵州茅台",
  "price": 1856.00,
  "change": 23.50,
  "changePercent": 1.28,
  "volume": 2345678,
  "marketCap": 2328765432100
}
```

### 批量查询
```json
{
  "action": "stock_query.batch",
  "params": {
    "codes": ["600519", "000858", "AAPL"],
    "markets": ["cn", "cn", "us"]
  }
}
```

## 配置说明
无需额外配置，使用免费API。

如需更高频率查询，可配置付费API：
```yaml
api_key: your_api_key
api_endpoint: https://api.example.com
```

## 注意事项
- 免费API有频率限制（每分钟60次）
- 盘前盘后数据可能有延迟
- 仅供参考，不构成投资建议
```

## 技能实现方式

OpenClaw支持多种技能实现方式：

**1. 纯提示词技能**

最简单的技能类型，只通过提示词指导大模型完成任务：

```markdown
---
name: translator
version: 1.0.0
description: 多语言翻译
---

# 翻译技能

## 系统提示
你是一个专业的翻译助手。当用户提供文本时：
1. 自动检测源语言
2. 翻译为目标语言（默认中文）
3. 保持原文的语气和风格

## 示例
用户：Hello, how are you?
助手：你好，你好吗？（英语 → 中文）

用户：今天天气很好
助手：The weather is nice today.（中文 → 英语）
```

**2. 脚本技能**

通过Python脚本实现具体逻辑：

```
stock_query/
├── SKILL.md
└── scripts/
    └── main.py
```

```python
# scripts/main.py

import requests
from typing import Dict, Any

def single(code: str, market: str = "cn") -> Dict[str, Any]:
    """查询单只股票"""
    url = f"https://api.example.com/stock/{market}/{code}"
    response = requests.get(url, timeout=10)
    data = response.json()
    
    return {
        "code": code,
        "name": data["name"],
        "price": data["price"],
        "change": data["change"],
        "changePercent": data["change_percent"],
        "volume": data["volume"],
        "marketCap": data["market_cap"]
    }

def batch(codes: list, markets: list = None) -> list:
    """批量查询"""
    results = []
    for i, code in enumerate(codes):
        market = markets[i] if markets and i < len(markets) else "cn"
        results.append(single(code, market))
    return results
```

**3. 工具调用技能**

通过定义工具接口，让大模型调用：

```markdown
---
name: calculator
version: 1.0.0
description: 数学计算器
tools:
  - name: calculate
    description: 执行数学表达式计算
    parameters:
      type: object
      properties:
        expression:
          type: string
          description: 数学表达式，如 "2 + 3 * 4"
      required: [expression]
---

# 计算器技能

## 工具定义
使用 calculate 工具执行数学计算。

## 示例
用户：计算 123 * 456
调用：calculate(expression="123 * 456")
返回：56088
```

## 发布到ClawHub

开发完成后，你可以将技能发布到ClawHub供其他人使用。

**1. 准备发布**

```bash
# 验证技能格式
openclaw skill validate ./my_skill/

# 运行测试
openclaw skill test ./my_skill/

# 打包技能
openclaw skill pack ./my_skill/
```

**2. 创建账号**

```bash
# 注册ClawHub账号
openclaw hub register

# 登录
openclaw hub login
```

**3. 发布技能**

```bash
# 发布技能
openclaw hub publish ./my_skill/

# 输出
Validating skill... ✓
Packing skill... ✓
Uploading to ClawHub... ✓

Your skill is now under review.
Review typically takes 1-2 business days.

Track status: https://hub.openclaw.ai/skills/my_skill
```

**4. 更新技能**

```bash
# 更新版本号
# 编辑 SKILL.md 中的 version 字段

# 发布更新
openclaw hub publish ./my_skill/ --update
```

**发布规范：**

| 要求 | 说明 |
|------|------|
| 名称 | 小写字母、数字、下划线，3-30字符 |
| 版本 | 遵循语义化版本规范 |
| 描述 | 10-200字符，清晰说明功能 |
| 许可证 | 推荐使用MIT、Apache-2.0等开源许可证 |
| 文档 | 必须包含使用说明和示例 |

---

通过自定义技能，你可以无限扩展OpenClaw的能力。下一章，我们将讨论一个重要但常被忽视的话题——技能安全。
