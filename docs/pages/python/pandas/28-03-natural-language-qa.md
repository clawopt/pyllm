---
title: 自然语言问答
description: 中文查询支持、多轮对话、复杂分析任务、结果格式化
---
# 自然语言问答

```python
class PandasQA:
    def __init__(self, df):
        self.agent = create_pandas_dataframe_agent(
            llm=ChatOpenAI(model='gpt-4o-mini', temperature=0),
            df=df,
            verbose=False,
        )
    
    def ask(self, question):
        try:
            r = self.agent.invoke(question)
            return r['output']
        except Exception as e:
            return f"处理失败: {str(e)[:200]}"
    
    def batch(self, questions):
        results = {}
        for q in questions:
            results[q] = self.ask(q)
            print(f"✓ {q[:40]}")
        return results

qa = PandasQA(df)
answers = qa.batch([
    "数据概况？",
    "按模型分组统计平均分和延迟？",
    "延迟超过1000ms的占比？",
])
```
