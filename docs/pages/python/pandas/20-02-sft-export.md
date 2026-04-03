---
title: SFT 流水线（续）：分层抽样与导出
description: 分层抽样策略 / messages 格式转换 / 数据集拆分 / 完整运行示例
---
# SFT 分层抽样与导出


## 第五步：分层抽样

```python
    def stratified_sample(self, tier_weights=None, total_size=10000,
                           min_per_tier=50, seed=42):
        """按质量等级分层抽样"""
        if self.scored is None:
            raise ValueError("请先调用 score_quality()")

        df = self.scored.copy()
        rng = np.random.RandomState(seed)

        if tier_weights is None:
            tier_weights = {'S': 0.15, 'A': 0.35, 'B': 0.30, 'C': 0.15, 'D': 0.05}

        sampled_parts = []
        for tier, weight in tier_weights.items():
            tier_data = df[df['quality_tier'] == tier]
            n_tier = max(int(total_size * weight), min_per_tier)
            n_tier = min(n_tier, len(tier_data))

            if n_tier > 0:
                sample = tier_data.sample(n=n_tier, random_state=rng)
                sampled_parts.append(sample)

        self.sampled = pd.concat(sampled_parts, ignore_index=True)

        self._log('stratified_sample',
                   f"抽样 {len(self.sampled):,} 条 (目标 {total_size:,})")
        return self
```

## 第六步：转换为训练格式

```python
    def to_training_format(self, format_type='messages'):
        """转换为训练格式"""
        if self.sampled is None:
            raise ValueError("请先调用 stratified_sample()")

        df = self.sampled.copy()

        if format_type == 'messages':
            records = []
            for _, row in df.iterrows():
                record = {
                    'messages': [
                        {'role': 'user', 'content': row.get('instruction', '')},
                        {'role': 'assistant', 'content': row.get('response', '')},
                    ],
                }
                if 'source' in df.columns and pd.notna(row.get('source')):
                    record['metadata'] = {
                        'source': row['source'],
                        'quality_score': float(row.get('quality_score', 0)),
                        'tier': str(row.get('quality_tier', '')),
                    }
                records.append(record)

            self.final = pd.DataFrame(records)
            self._log('format_messages', f"转换为 messages 格式: {len(self.final)} 条")

        elif format_type == 'sharegpt':
            records = []
            for _, row in df.iterrows():
                records.append({
                    'conversations': [
                        {'from': 'human', 'value': row.get('instruction', '')},
                        {'from': 'gpt', 'value': row.get('response', '')},
                    ]
                })
            self.final = pd.DataFrame(records)
            self._log('format_sharegpt', f"转换为 ShareGPT 格式: {len(self.final)} 条")

        return self

    def export_jsonl(self, output_path):
        """导出为 JSONL 格式（可直接用于训练）"""
        if self.final is None:
            raise ValueError("请先调用 to_training_format()")

        with open(output_path, 'w') as f:
            for _, row in self.final.iterrows():
                f.write(row.to_json(force_ascii=False) + '\n')

        file_size = __import__('os').path.getsize(output_path) / 1024 / 1024
        self._log('export_jsonl',
                   f"导出 {len(self.final)} 条 → {output_path} ({file_size:.1f} MB)")
        return output_path
```

## 完整运行示例

```python
import pandas as pd
import numpy as np

np.random.seed(42)
n = 5000

raw_sft = pd.DataFrame({
    'instruction': [
        f'{"问题" * (i % 20 + 1)}_{i}' for i in range(n)
    ] + [None] * int(n*0.02) + ['短'] * int(n*0.01),
    'response': [
        f'{"回答" * (i % 10 + 1)}_{i}' for i in range(len(raw_sft))
    ] if 'response' not in locals() else None,
    'source': np.random.choice(
        ['web_crawl', 'textbook', 'wiki', 'forum', 'paper',
         'synthetic', 'github'], n
    ),
})

if 'response' not in raw_sft.columns or raw_sft['response'].isna().any():
    n_actual = len(raw_sft)
    raw_sft['response'] = [
        f'这是对问题的详细回答，包含充分的信息和解释。' +
        f' 补充说明{i}' * (i % 5 + 1)
        for i in range(n_actual)
    ]

pipeline = SFTPipeline(name="llm_sft_v1")

result = (
    pipeline
    .load(raw_sft)
    .clean()
    .score_quality()
    .stratified_sample(total_size=2000)
    .to_training_format(format_type='messages')
)

print(f"\n{'='*55}")
print(f"流水线完成！最终数据集: {len(pipeline.final)} 条")
print(f"{'='*55}")

if pipeline.final is not None:
    print(f"\n质量分布:")
    if '__metadata' in pipeline.final.columns or 'metadata' in pipeline.final.columns:
        meta_col = 'metadata' if 'metadata' in pipeline.final.columns else '__metadata'
        print(f"  带 metadata 的记录: "
              f"{pipeline.final[meta_col].notna().sum()}/{len(pipeline.final)}")
```
