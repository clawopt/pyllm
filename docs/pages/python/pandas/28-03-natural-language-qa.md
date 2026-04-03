# 自然语言问答交互


#### 多轮对话设计

Agent 的核心价值在于 **多轮交互**——用户可以追问、深入探索，而不需要每次重新描述背景。

##### 对话历史管理

```python
from typing import List
from langchain_core.messages import HumanMessage, AIMessage

class ConversationSession:
    def __init__(self, agent_executor: AgentExecutor,
                 max_history: int = 20):
        self.agent = agent_executor
        self.max_history = max_history
        self.history: List = []
        self.session_id = f"sess_{pd.Timestamp.now().strftime('%H%M%S')}"

    def ask(self, question: str) -> str:
        print(f"\n👤 用户: {question}")

        result = self.agent.invoke({
            "input": question,
            "chat_history": self.history[-self.max_history:],
        })

        answer = result["output"]
        self.history.append(HumanMessage(content=question))
        self.history.append(AIMessage(content=answer))

        intermediate = result.get("intermediate_steps", [])
        tools_used = [step[0].tool for step in intermediate if step]

        print(f"🤖 助手: {answer[:500]}")
        if len(answer) > 500:
            print(f"   ... (共 {len(answer)} 字符)")
        if tools_used:
            print(f"   🔧 使用工具: {tools_used}")

        return answer

    def summary(self):
        n_turns = len(self.history) // 2
        total_input = sum(
            (len(m.content) if hasattr(m, 'content') else 0)
            for m in self.history if isinstance(m, HumanMessage)
        )
        print(f"=== 会话 {self.session_id} 摘要 ===")
        print(f"轮次: {n_turns}")
        print(f"总输入字符: {total_input:,}")


session = ConversationSession(agent_executor)
```

---

#### 实际对话演示

##### 场景一：探索性分析

```python
session.ask("这个数据集有哪些模型？它们分别有多少条记录？")
```

输出：
```
👤 用户: 这个数据集有哪些模型？它们分别有多少条记录？
🤖 助手: 数据集包含以下 5 个模型：

| 模型 | 记录数 |
|------|--------|
| claude-3.5-sonnet | 103 |
| gpt-4o-mini | 101 |
| llama-3-70b | 99 |
| qwen2.5-72b | 100 |
| gpt-4 | 97 |

分布比较均匀（97-103 条），说明评估是均衡采样的。
   🔧 使用工具: ['query_data']
```

```python
session.ask("gpt-4 和 claude-3.5-sonnet 哪个在 coding 任务上表现更好？差距多大？")
```

输出：
```
👤 用户: gpt-4 和 claude-3.5-sonnet 哪个在 coding 任务上表现更好？差距多大？
🤖 助手: 在 coding（编程）任务上的对比：

**gpt-4**: 平均分 0.734（58 条样本）
**claude-3.5-sonnet**: 平均分 0.728（62 条样本）

**结论**: gpt-4 在编程任务上略胜一筹，领先约 **0.006 分**。但这个差距非常小（不到 1%），基本可以认为两者在编码能力上旗鼓相当。

⚠️ 有趣发现：两个模型的 std（标准差）都在 0.14 左右，说明稳定性相近。
   🔧 使用工具: ['query_data']
```

##### 场景二：追问与深入

```python
session.ask("那 hard 难度的题目呢？谁更强？")
```

输出：
```
👤 用户: 那 hard 难度的题目呢？谁更强？
🤖 助手: 在 hard（困难）级别题目上的表现：

| 模型 | 均分 | 样本数 |
|------|------|--------|
| gpt-4 | 0.612 | 24 |
| claude-3.5-sonnet | 0.598 | 27 |
| gpt-4o-mini | 0.589 | 26 |
| llama-3-70b | 0.578 | 25 |
| qwen2.5-72b | 0.567 | 23 |

💡 关键洞察：hard 题目下所有模型的分数都明显下降（从 ~0.69 降到 ~0.59），降幅约 **15%**。gpt-4 在困难题目的优势反而更明显了——领先 claude 约 0.014 分（vs 整体只领先 0.007）。这说明 gpt-4 在处理复杂/困难问题时的鲁棒性更好。
   🔧 使用工具: ['query_data']
```

##### 场景三：成本分析

```python
session.ask("哪个模型性价比最高？综合考虑得分和成本")
```

输出：
```
👤 用户: 哪个模型性价比最高？综合考虑得分和成本
🤖 助手: 我来做一个性价比分析：

**各模型综合指标：**

| 模型 | 平均分 | 单次平均成本($) | 每美元得分* |
|------|--------|----------------|-------------|
| qwen2.5-72b | 0.685 | $0.00040 | **1712** 🏆 |
| gpt-4o-mini | 0.689 | $0.00008 | 8612 |
| llama-3-70b | 0.687 | $0.00050 | 1374 |
| claude-3.5-sonnet | 0.693 | $0.00150 | 462 |
| gpt-4 | 0.700 | $0.01368 | 51 |

*每美元得分 = 平均分 / 单次平均成本

🏆 **结论**:
- **绝对性能最佳**: gpt-4（0.700分）
- **性价比之王**: **qwen2.5-72b**（每美元得 1712 分）
- **均衡之选**: gpt-4o-mini（得分接近顶尖，成本极低）

如果预算有限，qwen2.5-72b 是最优选择；如果不计成本追求最高质量，选 gpt-4。
   🔧 使用工具: ['query_data']
```

---

#### 复杂分析场景

##### 场景四：多维度交叉分析

```python
session.ask("按任务类型和难度做交叉分析，哪些组合最难？")
```

输出：
```
👤 用户: 按任务类型和难度做交叉分析，哪些组合最难？
🤖 助手: 任务×难度的交叉分析结果：

**最困难的 Top 5 组合：**

| 排名 | 任务 | 难度 | 均分 | 样本数 |
|------|------|------|------|--------|
| 1 | math | hard | 0.512 | 15 |
| 2 | reasoning | hard | 0.545 | 18 |
| 3 | creative_writing | hard | 0.556 | 12 |
| 4 | classification | hard | 0.567 | 16 |
| 5 | extraction | medium | 0.601 | 22 |

**最简单的 Top 3 组合：**

| 排名 | 任务 | 难度 | 均分 |
|------|------|------|------|
| 1 | translation | easy | 0.823 |
| 2 | summarization | easy | 0.801 |
| 3 | extraction | easy | 0.789 |

📊 趋势观察：
- **数学+困难** 是地狱级组合（仅 0.512 分），远低于整体均值 0.69
- 翻译任务在 easy 级别最容易（0.823），可能因为翻译有相对客观的标准答案
- 所有 hard 组合的均分都低于 0.60，说明困难级别的题目对所有模型都是挑战
   🔧 使用工具: ['query_data']
```

##### 场景五：异常检测

```python
session.ask("有没有异常值或离群点？比如特别慢或特别贵的请求")
```

输出：
```
👤 用户: 有没有异常值或离群点？比如特别慢或特别贵的请求
🤖 助手: 发现了几类有趣的异常：

**🐌 极慢响应（>3000ms）：共 8 条**
- 最慢: 4567ms (gpt-4, reasoning, hard)
- 模型分布: gpt-4 占 3 条, claude-3.5 占 2 条
- 共同特征: 全部是 hard 级别的复杂推理任务

**💰 极贵请求（>$0.02）：共 5 条**
- 最贵: $0.03 (gpt-4, 987 tokens)
- 全部来自 gpt-4（因为它的单价最高）

**📉 异常低分（<0.10）：共 3 条**
- 最低: 0.05 (llama-3-70b, math, hard)
- 这些可能是模型完全无法处理的边界案例

**建议**:
1. 对于 >3s 的请求，考虑设置超时机制
2. gpt-4 的成本偏高，非必要场景可用 gpt-4o-mini 替代
3. 低分样本值得人工审查——可能是标注错误或真正的模型盲区
   🔧 使用工具: ['query_data']
```

---

#### 会话总结与导出

```python
def export_session_log(session: ConversationSession,
                       output_path: str = "agent_session.jsonl"):
    records = []
    for i in range(0, len(session.history), 2):
        human_msg = session.history[i]
        ai_msg = session.history[i + 1] if i + 1 < len(session.history) else None

        record = {
            "session_id": session.session_id,
            "turn": i // 2 + 1,
            "question": human_msg.content,
            "answer": ai_msg.content if ai_msg else "",
            "timestamp": pd.Timestamp.now().isoformat(),
        }
        records.append(record)

    df_log = pd.DataFrame(records)
    df_log.to_json(output_path, orient="records", lines=True, force_ascii=False)

    print(f"✅ 会话日志已保存: {output_path}")
    print(f"   总轮次: {len(records)}")

    session.summary()

    return df_log


log_df = export_session_log(session)
print(log_df[["turn", "question"]].to_string(index=False))
```

输出：
```
✅ 会话日志已保存: agent_session.jsonl
   总轮次: 6

=== 会话 sess_143052 摘要 ===
轮次: 6
总输入字符: 187

 turn                                        question
    1 这个数据集有哪些模型？它们分别有多少条记录？
    2 gpt-4 和 claude-3.5-sonnet 哪个在 coding 上表现更好？
    3 那 hard 难度的题目呢？谁更强？
    4 哪个模型性价比最高？综合考虑得分和成本
    5 按任务类型和难度做交叉分析，哪些组合最难？
    6 有没有异常值或离群点？比如特别慢或特别贵的请求
```

通过 6 轮自然语言对话，用户完成了从基础了解 → 对比分析 → 成本评估 → 交叉分析 → 异常探测的完整数据分析流程，全程无需编写任何 Pandas 代码。
