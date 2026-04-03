---
title: Agent 记忆与上下文管理
description: 多轮对话中的状态保持、长对话的 Token 管理、数据摘要策略
---
# Agent 的记忆：让多轮对话有上下文

默认情况下，Pandas DataFrame Agent 是**无状态的**——每次调用 `invoke()` 都是独立的，Agent 不记得之前问过什么。这在实际使用中是个问题：用户可能先问"有哪些列"，然后问"按第一列分组"，但 Agent 不知道"第一列"是什么。

## 解决方案

```python
from langchain_core.messages import HumanMessage, AIMessage

class ConversationAgent:
    def __init__(self, df):
        self.df = df
        self.agent = create_pandas_dataframe_agent(
            llm=ChatOpenAI(model='gpt-4o-mini', temperature=0),
            df=df,
            verbose=False,
        )
        self.history = []
    
    def chat(self, question):
        self.history.append(HumanMessage(content=question))
        
        result = self.agent.invoke({
            'input': question,
            'chat_history': self.history[-10:],  # 只保留最近 10 轮
        })
        
        answer = result['output']
        self.history.append(AIMessage(content=answer))
        return answer
```

核心技巧是维护一个 `history` 列表，把每轮的用户问题和 AI 回答都存进去，然后在下次调用时通过 `chat_history` 参数传给 Agent。**只保留最近 N 轮（如 10 轮）** 是为了控制 token 消耗——完整的对话历史可能很快超出 context window。
