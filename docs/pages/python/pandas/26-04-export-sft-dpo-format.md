---
title: 导出 SFT/DPO 训练格式
description: JSONL 格式转换、messages 字段构建、train/val 划分、Parquet 输出
---
# 导出训练格式

```python
import pandas as pd

class SFTExporter:
    def __init__(self, df):
        self.df = df
    
    def build_messages(self):
        self.df['messages'] = self.df.apply(
            lambda r: [
                {"role": "user", "content": str(r.get('user_message', ''))},
                {"role": "assistant", "content": str(r.get('assistant_message', ''))}
            ], axis=1
        )
        return self
    
    def split_and_export(self, val_ratio=0.05, output_dir='./sft_output'):
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        n = len(self.df)
        val_n = int(n * val_ratio)
        
        train = self.df.iloc[val_n:]
        val = self.df.iloc[:val_n]
        
        train[['messages']].to_json(
            f'{output_dir}/train.jsonl', orient='records', lines=True,
            force_ascii=False, indent=None
        )
        val[['messages']].to_json(
            f'{output_dir}/val.jsonl', orient='records', lines=True,
            force_ascii=False, indent=None
        )
        
        print(f"导出完成:")
        print(f"  Train: {len(train):,} 条 → {output_dir}/train.jsonl")
        print(f"  Val:   {len(val):,} 条 → {output_dir}/val.jsonl")

exporter = SFTExporter(df)
exporter.build_messages().split_and_export()
```
