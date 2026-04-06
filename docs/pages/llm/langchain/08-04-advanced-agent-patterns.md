---
title: 高级代理模式：多代理协作与长期记忆
description: 多代理团队协作、Agent 的反思机制、长期记忆管理、工具使用优化策略
---
# 高级代理模式：多代理协作与长期记忆

前三节我们学习了 ReAct 模式的基础用法——单个 Agent 使用工具自主完成任务。这一节我们将探索更高级的 Agent 架构：**多个 Agent 协作完成复杂任务**、**让 Agent 具备自我反思和长期记忆能力**。

## 多代理协作：团队模式

到目前为止我们构建的所有 Agent 都是"单打独斗"的——一个 Agent 完成所有工作。但在现实世界中，复杂的任务往往需要不同专长的"角色"来配合。就像一个项目团队有产品经理、开发、测试、运维一样，多代理系统也由**各司其职的子 Agent** 组成。

### 为什么需要多代理

考虑这样一个任务："调研一家竞品公司并撰写分析报告"

单一 Agent 需要做的：
1. 搜索公司基本信息
2. 搜索产品功能和定价
3. 搜索用户评价和市场反馈
4. 搜索最新融资情况
5. 整理所有信息写成结构化报告

这 5 个步骤虽然可以由同一个 Agent 完成，但存在几个问题：
- **上下文窗口压力**：所有搜索结果都要保留在上下文中，容易超限
- **工具冲突风险**：搜索/计算/文件操作混在一起，出错时难以定位
- **无法并行**：串行执行 5 次搜索很慢

### 两阶段架构：Manager + Worker

最经典的多代理架构是 **管理者-工作者（Manager-Worker）** 模式：

```python
"""
multi_agent_demo.py — 简化的多代理协作示例
"""
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain.agents import create_react_agent

chat = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# === 定义各 Worker Agent 的工具 ===

@tool
def search_company_info(company: str) -> str:
    """搜索公司的基本信息（名称、成立时间、规模等）"""
    info = {
        "OpenAI": "成立于2015年，总部旧金山，员工约2000人",
        "Anthropic": "成立于2021年，总部旧金山，员工约800人",
        "Google": "成立于1998年，总部山景城，员工约18万人"
    }
    return info.get(company, f"未找到{company}的信息")

@tool
def search_products(company: str) -> str:
    """搜索公司的产品线和定价信息"""
    products = {
        "OpenAI": "GPT-4o($20/M输入), GPT-4($60/M), Whisper(免费)",
        "Anthropic": "Claude 3.5 Sonnet($3/M), Claude 3 Opus($15/M)",
        "Google": "Gemini Pro(免费), Gemini Ultra($7/M输入)"
    }
    return products.get(company, f"未找到{company}的产品信息")

@tool
def search_reviews(company: str) -> str:
    """搜索用户评价和市场反馈"""
    reviews = {
        "OpenAI": "好评率85%，主要优点是API易用，缺点是偶尔不稳定",
        "Anthropic": "好评率90%，以长上下文和安全著称，价格偏高",
        "Google": "好评率80%，生态强大，但隐私政策受争议"
    }
    return reviews.get(company, f"未找到{company}的评价")

@tool
def write_report(content: str) -> str:
    """将内容写入报告文件"""
    filepath = f"reports/{'report'}.md"
    with open(filepath, "w") as f:
        f.write(content)
    return f"报告已保存: {filepath}"

# === Manager Agent（不直接用工具，只做调度）===

manager_prompt = ChatPromptTemplate.from_template("""
你是一个研究项目经理。你的任务是协调各个研究员完成竞品分析。

你管理的团队成员：
- researcher_basic: 可搜索公司基本信息和产品信息
- researcher_reviews: 可搜索用户评价和市场反馈
- writer: 负责将最终报告写入文件

请按以下流程工作：
1. 先确定要调研的目标公司
2. 分配任务给合适的研究员
3. 收集所有研究结果
4. 让 writer 整理成最终报告并保存

当前目标公司：{company}
""")

manager_chat = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# === Worker Agent 1: 基础信息研究员 ===
worker1_prompt = ChatPromptTemplate.from_template(
    "你是基础信息研究员。请查找以下公司的基本信息和产品线：\n{task}"
)
worker1 = create_react_agent(
    model=chat,
    tools=[search_company_info, search_products],
    prompt=worker1_prompt
)

# === Worker Agent 2: 评价研究员 ===
worker2_prompt = ChatPromptTemplate.from_template(
    "你是市场评价研究员。请查找以下公司的用户评价：\n{task}"
)
worker2 = create_react_agent(
    model=chat,
    tools=[search_reviews],
    prompt=worker2_prompt
)

# === 使用方式 ===

def run_multi_agent_analysis(company: str):
    """运行多代理协作分析"""

    # Step 1: Manager 规划任务
    manager_chain = manager_prompt | manager_chat
    plan = manager_chain.invoke({"company": company})

    print(f"\n📋 Manager 计划:\n{plan.content}\n")

    # Step 2: 并行调用 Worker
    basic_info = worker1.invoke({"task": f"调研 {company}"})
    review_info = worker2.invoke({"task": f"{company} 的用户评价"})

    print(f"📊 基础信息:\n{basic_info['output']}\n")
    print(f"⭐ 用户评价:\n{review_info['output']}\n")

    # Step 3: Manager 综合结果并写报告
    final_input = f"""
    基础信息：
    {basic_info['output']}

    用户评价：
    {review_info['output']}

    请综合以上信息，为{company}写一份简洁的竞品分析报告。
    包含：公司概况、核心产品、优劣势分析、市场定位。
    """

    report = manager_chat.invoke(final_input)

    # Step 4: 保存报告
    report_content = report.content
    save_path = write_report(report_content)

    return report_content


if __name__ == "__main__":
    result = run_multi_agent_analysis("OpenAI")
    print(f"\n{'='*50}")
    print(result)
```

运行效果：

```
📋 Manager 计划:
我将协调团队完成 OpenAI 的竞品分析...
分配给 researcher_basic 搜索基本信息和产品，
分配给 researcher_reviews 搜索用户评价...

📊 基础信息:
OpenAI 成立于2015年，总部位于旧金山...
主要产品包括 GPT-4o ($20/M tokens)、GPT-4 ...

⭐ 用户评价:
OpenAI 好评率约85%...
主要优势在于 API 易于集成...

==================================================
## OpenAI 竞品分析报告

### 公司概况
OpenAI 是目前最具影响力的 AI 研究公司...

### 核心产品
| 产品 | 定价 | 特点 |
|------|------|------|
...

### 优劣势分析
**优势**: 品牌效应强、开发者生态完善...
**劣势**: API 偶尔不稳定、成本较高...
```

### 关键设计要点

**1. Manager 不直接操作工具**

注意 `manager_chain` 只用了 `manager_chat`（纯 LLM），没有给它任何 `tools`。Manager 的职责是**规划和调度**，不是亲自干活。这是多代理系统的核心原则——**职责分离**。

**2. Worker 各自拥有专用工具**

每个 Worker Agent 只拿到自己需要的工具——`worker1` 有搜索公司和产品的工具，`worker2` 只有搜索评价的工具。这种**最小权限原则**既安全又高效。

**3. Worker 之间通过 Manager 间接通信**

Worker 之间不会直接对话。它们各自把结果返回给 Manager，由 Manager 统合后再分发下一步任务。这避免了通信混乱。

## Agent 反思机制

另一个高级能力是让 Agent **检查自己的输出质量并在不满意时自我修正**。

### 基本反思模式

```python
reflect_prompt = ChatPromptTemplate.from_template("""
你是一个有自我审查能力的助手。

请先回答用户的问题。
然后，检查你的回答是否满足以下标准：
1. 是否准确回答了问题？
2. 信息是否有依据（不是编造的）？
3. 语言是否简洁清晰？

如果以上任一标准未满足，请重新给出改进后的答案。
如果都已满足，直接输出原答案即可。

问题：{question}

你的初始回答：
""")

reflective_chain = reflect_prompt | chat | StrOutputParser()

result = reflective_chain.invoke({
    "question": "什么是 RAG？",
    "初始回答: RAG 是一种检索增强生成技术..."   # 先给出初始答案
})
```

这个 Chain 会先输出一个初步答案，然后 LLM 自检质量，如果发现问题就自动修正后重新输出。

### 带评分的反思

更精细的做法是让 Agent 给自己的回答打分：

```python
self_critique_prompt = ChatPromptTemplate.from_template("""
对以下回答进行自我评估：

问题：{question}
回答：{answer}

请从以下维度打分（1-10分）：
- 准确性：回答是否事实正确？
- 完整性：是否完整覆盖了问题的各个方面？
- 清晰度：表达是否易于理解？

如果总分低于 7 分，请给出改进后的版本。
如果 7 分以上，输出原回答即可。

评估格式：
准确:X/10 | 完整:X/10 | 清晰:X/10 | 总分:X/10
改进版：（如需要）
""")
```

反思机制在以下场景特别有价值：
- **代码生成**：Agent 写完代码后自检是否有语法错误或逻辑漏洞
- **数学推理**：Agent 解完题后验证计算步骤是否正确
- **事实检索**：Agent 回答后确认引用的信息是否来自检索结果而非编造

## 长期记忆管理

前面第 5 章学的 Memory（对话记忆）是**会话级别**的短期记忆。而 Agent 的**长期记忆**指的是跨会话持久化的知识和经验积累。

### 经验存储

```python
import json
import os
from langchain_core.tools import tool

MEMORY_FILE = "agent_memory.json"

def load_memory() -> list[dict]:
    """加载历史经验"""
    if not os.path.exists(MEMORY_FILE):
        return []
    with open(MEMORY_FILE, "r") as f:
        return json.load(f)

def save_memory(experience: dict):
    """保存一条新经验"""
    memories = load_memory()
    memories.append(experience)
    with open(MEMORY_FILE, "w") as f:
        json.dump(memories, f, ensure_ascii=False, indent=2)

@tool
def recall_similar_experience(query: str) -> str:
    """查询历史上类似场景的处理经验"""
    memories = load_memory()
    for m in memories:
        if m.get("task_type") in query.lower() or query.lower() in m.get("task", ""):
            return (
                f"找到类似经验: 场景='{m.get('scenario')}', "
                f"方案='{m.get('solution')}', "
                f"结果='{m.get('outcome')}'"
            )
    return "没有找到相关历史经验"
```

这样 Agent 在处理新问题时可以先查一下"我以前有没有遇到过类似的情况"，从而避免重复犯错。

### 工具使用优化策略

高级 Agent 应该学会**选择最优工具组合**而不是盲目尝试：

```python
@tool
def smart_search(query: str, strategy: str = "auto") -> str:
    """
    智能搜索引擎，自动选择最优搜索策略
    
    Args:
        query: 搜索关键词
        strategy: 搜索策略 ("broad"/"precise"/"auto")
    
    road: 广泛搜索，返回概览性结果
    precise: 精确搜索，返回具体细节
    auto: 根据查询类型自动判断
    """
    if strategy == "auto":
        if any(kw in query.lower() for kw in ["是什么", "定义", "区别"]):
            strategy = "road"
        elif any(kw in query.lower() for kw in ["多少", "排名", "价格", "具体"]):
            strategy = "precise"
    
    # 根据策略调整搜索参数
    k = 5 if strategy == "road" else 3
    
    # 执行搜索（模拟）
    results = {
        "road": ["OpenAI 是 AI 公司...", "RAG 是检索增强..."],
        "precise": ["OpenAI 成立于 2015 年...", "GPT-4o 价格 $20/M..."]
    }
    
    return results.get(strategy, [])[-1] if results.get(strategy) else "无结果"
```

这种"知道何时用什么工具、怎么用"的能力是把普通 Agent 变成**生产级 Agent**的关键一步。

到这里，第八章的全部 5 个小节就结束了。下一章我们将进入实战项目部分——构建真正的端到端应用系统。
