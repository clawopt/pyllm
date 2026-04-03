---
title: LLM 数据清洗全流程
description: 从原始语料到 SFT 训练就绪数据的完整流水线、质量检查点、自动化脚本
---
# LLM 数据清洗：完整流水线

前面所有章节的知识在这里汇聚成一个完整的端到端解决方案：**从原始对话语料到 SFT 训练就绪数据**。

## 流水线架构

```
原始 CSV/JSONL (500万行)
    │
    ├─ 1. 加载优化 (usecols + dtype)
    │
    ├─ 2. 完整性检查 (prompt/response 非空)
    │
    ├─ 3. 质量过滤 (score >= 4.0)
    │
    ├─ 4. 长度过滤 (50 <= tokens <= 2000)
    │
    ├─ 5. 字符串清洗 (去HTML/去多余空白)
    │
    ├─ 6. 去重 (按 prompt+response)
    │
    └─ 7. 输出 JSONL/Parquet → 训练集
```

## 完整实现

```python
import pandas as pd
import numpy as np

class SFTPipeline:
    def __init__(self):
        self.stats = {}
    
    def run(self, input_path, output_path):
        chunks = []
        for chunk in pd.read_json(input_path, lines=True, chunksize=250_000,
                                    dtype={'prompt': 'string', 'response': 'string',
                                           'quality': 'float32'}):
            c = self._process(chunk)
            if len(c) > 0:
                chunks.append(c)
        
        result = pd.concat(chunks, ignore_index=True)
        result.to_parquet(output_path, index=False)
        
        print(f"\n完成! {output_path}")
        print(f"输出 {len(result):,} 条高质量训练样本")
        return result
    
    def _process(self, df):
        df = df[
            df['prompt'].notna() & (df['prompt'].str.len() >= 3) &
            df['response'].notna() & (df['response'].str.len() >= 10) &
            (df['quality'] >= 4.0)
        ].copy()
        
        df['clean_prompt'] = (
            df['prompt']
            .str.replace(r'<[^>]+>', '', regex=True)
            .str.strip()
            .str[:2048]
        )
        
        return df.sort_values('quality', ascending=False)\
                  .drop_duplicates(subset=['clean_prompt'], keep='first')

pipeline = SFTPipeline()
result = pipeline.run('raw_conversations.jsonl', 'sft_train_v3.parquet')
```

这个流水线把前面学到的所有技术整合在一起：分块读取、dtype 优化、布尔索引筛选、字符串正则清洗、排序去重、Parquet 输出。**它是你在实际项目中可以直接拿去用的模板**。
