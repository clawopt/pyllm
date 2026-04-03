---
title: LLM 辅助智能清洗
description: 调用 LLM API 批量判断数据质量、语义去重、成本控制与缓存
---
# LLM 智能清洗

```python
import pandas as pd

class LLMAssistedCleaner:
    def __init__(self, df, batch_size=50):
        self.df = df.copy()
        self.batch_size = batch_size
    
    def judge_quality(self, text_col='user_message'):
        results = []
        for i in range(0, len(self.df), self.batch_size):
            batch = self.df[text_col].iloc[i:i+self.batch_size]
            texts = '\n---\n'.join(f"[{j}] {t}" for j, t in enumerate(batch))
            
            prompt = f"""判断以下文本是否有实质信息量。回复 JSON:
{{"verdicts": [{{"idx":<原始索引>, "keep": true/false, "reason": "一句话"}}]}}
文本:
{texts}"""
            
            results.append(prompt)
        return results
    
    def semantic_dedup(self, col1='user_message', col2='assistant_message',
                       threshold=0.95):
        combined = self.df[col1] + ' ||| ' + self.df[col2]
        hash_col = combined.apply(lambda x: hash(x) % 10000)
        
        deduped = self.df.drop_duplicates(subset=[hash_col], keep='first')
        print(f"语义去重: {len(self.df):,} → {len(deduped):,}")
        return deduped.reset_index(drop=True)

cleaner = LLMAssistedCleaner(df)
```
