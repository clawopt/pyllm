---
title: 分布式处理实战：大规模 SFT 数据管道
description: Dask 完整 ETL / 分布式清洗 / 分层抽样 / 导出训练数据
---
# 分布式处理场景实战


## 实战：Dask 大规模 SFT 数据管道

```python
import pandas as pd
import numpy as np

try:
    import dask.dataframe as dd
    HAS_DASK = True
except ImportError:
    HAS_DASK = False


class DistributedSFTPipeline:
    """分布式 SFT 数据处理流水线"""

    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.pipeline_log = []

    def _log(self, stage, detail):
        entry = {'stage': stage, 'detail': detail}
        self.pipeline_log.append(entry)
        print(f"  [{stage}] {detail}")

    def run(self):
        """执行完整分布式管道"""
        self._log('start', f'输入: {self.input_path}')

        if HAS_DASK:
            raw = dd.read_json(
                self.input_path,
                lines=True,
                blocksize='128 MiB',
            )
            total_rows = raw.shape[0].compute()
            self._log('read', f'读取 {total_rows:,} 行 (延迟模式)')
        else:
            raw = pd.read_json(self.input_path, lines=True)
            total_rows = len(raw)
            self._log('read', f'读取 {total_rows:,} 行 (Pandas 模式)')

        if HAS_DASK:
            clean = raw[
                (raw['instruction'].str.len() > 5) &
                (raw['response'].str.len() > 15) &
                (raw['instruction'].notna()) &
                (raw['response'].notna())
            ]
            after_filter = clean.shape[0].compute()
            self._log('filter', f'{total_rows:,} → {after_filter:,}')
        else:
            mask = (
                (raw['instruction'].str.len() > 5) &
                (raw['response'].str.len() > 15)
            )
            clean = raw[mask]
            after_filter = len(clean)
            self._log('filter', f'{total_rows:,} → {after_filter}')

        if HAS_DASK:
            def add_features(df):
                df['instr_len'] = df['instruction'].str.len()
                df['resp_len'] = df['response'].str.len()
                df['ratio'] = (df['resp_len'] / df['instr_len'].replace(0, 1)).round(2)
                return df

            featured = clean.map_partitions(add_features).compute()
            self._log('features', '特征提取完成')
        else:
            clean['instr_len'] = clean['instruction'].str.len()
            clean['resp_len'] = clean['response'].str.len()
            featured = clean
            self._log('features', '特征提取完成')

        if HAS_DASK and len(featured) > 0:
            scored = featured.assign(
                quality_score=(
                    ((featured['instr_len'] > 15).astype(int) * 1 +
                     (featured['instr_len'] < 500).astype(int) * 1 +
                     (featured['resp_len'] > 30).astype(int) * 1 +
                     (featured['ratio'] > 1).astype(int) * 1 +
                     (featured['ratio'] < 20).astype(int) * 1) /
                    6.0
                ).clip(0, 1)
            )
            self._log('scored', '质量评分完成')
        elif len(featured) > 0:
            featured['quality_score'] = np.random.uniform(0.3, 0.99, len(featured))
            self._log('scored', '质量评分完成')

        if HAS_DASK and len(featured) > 0:
            target_size = min(100_000, int(len(featured) * 0.3))

            def stratified_sample(df):
                import random
                rng = random.RandomState(42)

                tier_weights = {'A': 0.20, 'B': 0.35, 'C': 0.30, 'D': 0.15}

                samples = []
                for tier, weight in tier_weights.items():
                    tier_df = df[df['quality_score'] >= float(tier.replace('D','0').replace('C','0.4')
                                                  .replace('B','0.6').replace('A','0.8'))]
                    n_tier = max(int(target_size * weight), min(len(tier_df), 10))
                    if n_tier > 0:
                        sampled = tier_df.sample(n=n_tier, random_state=rng)
                        samples.append(sampled)

                return pd.concat(samples, ignore_index=True) if samples else df.head(0)

            final = featured.map_partitions(stratified_sample).compute()
            self._log('sample', f'抽样 {len(final):,} 条')
        else:
            final = featured.sample(n=min(10000, len(featured)), random_state=42)
            self._log('sample', f'抽样 {len(final)} 条')

        if HAS_DASK and len(final) > 0:
            def to_messages(row):
                return {
                    'messages': [
                        {'role': 'user', 'content': row.get('instruction', '')},
                        {'role': 'assistant', 'content': row.get('response', '')},
                    ]
                }

            messages_df = final.apply(to_messages, axis=1, meta=('messages',))
            messages_df.to_json(self.output_path, orient='records',
                           lines=True, force_ascii=False)
            self._log('export', f'导出到 {self.output_path}')

        print(f"\n{'='*55}")
        print(f"管道完成！共 {len(self.pipeline_log)} 个阶段")
        for entry in self.pipeline_log:
            print(f"  ✓ {entry['stage']}: {entry['detail']}")
        return self


print("=== 分布式 SFT 管道 ===")
```
