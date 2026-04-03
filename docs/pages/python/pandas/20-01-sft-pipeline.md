---
title: 项目一：SFT 数据集端到端处理流水线
description: 完整的 SFT 数据清洗 / 质量评估 / 特征工程 / 分层抽样 / 导出为训练格式
---
# SFT 数据集处理流水线


## 项目概述

构建一个完整的 **SFT（Supervised Fine-Tuning）数据集处理流水线**，从原始数据到可直接用于训练的 JSONL 格式。

```
原始数据 → 清洗 → 质量评分 → 特征提取 → 分层抽样 → 格式转换 → 训练集
```

## 第一步：定义完整的数据结构

```python
import pandas as pd
import numpy as np
from datetime import datetime

class SFTPipeline:
    """完整的 SFT 数据处理流水线"""

    EXPECTED_COLUMNS = {
        'instruction': 'str',
        'response': 'str',
        'source': 'category',
        'domain': 'category',
        'language': 'str',
    }

    def __init__(self, name="default_pipeline"):
        self.name = name
        self.raw = None
        self.cleaned = None
        self.scored = None
        self.featured = None
        self.sampled = None
        self.final = None
        self.report = {}
        self.history = []

    def _log(self, stage, detail):
        entry = {'stage': stage, 'detail': detail,
                'timestamp': datetime.now()}
        self.history.append(entry)
        print(f"  [{stage}] {detail}")
        return self
```

## 第二步：数据加载与初始检查

```python
    def load(self, df):
        """加载原始数据"""
        self.raw = df.copy()
        self._log('load', f"加载 {len(df):,} 行 × {len(df.columns)} 列")

        null_counts = self.raw.isnull().sum()
        high_null = null_counts[null_counts > len(df) * 0.1]
        if len(high_null) > 0:
            self._log('load_warning', f"高缺失列: {dict(high_null)}")

        dupes = self.raw.duplicated().sum()
        if dupes > 0:
            self._log('load_warning', f"重复行: {dupes}")

        return self
```

## 第三步：多阶段清洗

```python
    def clean(self):
        """执行完整清洗流程"""
        if self.raw is None:
            raise ValueError("请先调用 load()")

        df = self.raw.copy()

        before = len(df)
        mask = True
        if 'instruction' in df.columns:
            mask &= df['instruction'].notna() & (df['instruction'].str.len() > 3)
        if 'response' in df.columns:
            mask &= df['response'].notna() & (df['response'].str.len() > 10)
        df = df[mask].copy()
        self._log('clean_nulls', f"{before} → {len(df)} (移除 {before - len(df)})")

        if 'instruction' in df.columns:
            before_dedup = len(df)
            df['_instr_hash'] = df['instruction'].str.strip().str.lower()
            df = df.drop_duplicates(subset=['_instr_hash'], keep='first')
            df = df.drop(columns=['_instr_hash'])
            self._log('dedup_instructions',
                       f"{before_dedup} → {len(df)} (去重 {before_dedup - len(df)})")

        if 'instruction' in df.columns:
            df['instruction'] = (
                df['instruction']
                .str.strip()
                .str.replace(r'\s+', ' ', regex=True)
                .str.replace(r'\u200b|\ufeff', '', regex=True)  # 零宽字符
            )

        if 'instruction' in df.columns:
            df['instr_len'] = df['instruction'].str.len()
            before_len = len(df)
            df = df[(df['instr_len'] >= 5) & (df['instr_len'] <= 4000)]
            self._log('filter_length',
                       f"{before_len} → {len(df)} (长度过滤)")

        if 'response' in df.columns:
            df['resp_len'] = df['response'].str.len()
            before_len = len(df)
            df = df[(df['resp_len'] >= 15) & (df['resp_len'] <= 8000)]
            self._log('filter_response_length',
                       f"{before_len} → {len(df)} (响应长度过滤)")

        self.cleaned = df.reset_index(drop=True)
        return self
```

## 第四步：质量评分

```python
    def score_quality(self):
        """计算每条数据的质量分数"""
        if self.cleaned is None:
            raise ValueError("请先调用 clean()")

        df = self.cleaned.copy()

        score_components = pd.Series(0.0, index=df.index)

        if 'instr_len' in df.columns:
            ideal_len_range = (50, 500)
            in_ideal = (
                (df['instr_len'] >= ideal_len_range[0]) &
                (df['instr_len'] <= ideal_len_range[1])
            ).astype(float)
            score_components += in_ideal * 15

            has_question_word = df['instruction'].str.contains(
                r'什么|如何|为什么|怎么|解释|比较|区别|是否|能否|哪些|哪个',
                case=False, regex=True
            ).astype(float) * 8

            not_too_short = (df['instr_len'] > 15).astype(float) * 7
            score_components += has_question_word + not_too_short

        if 'resp_len' in df.columns:
            good_ratio = (
                ((df['resp_len'] / df['instr_len'].replace(0, 1)) > 2) &
                ((df['resp_len'] / df['instr_len'].replace(0, 1)) < 20)
            ).astype(float) * 15

            sufficient_detail = (df['resp_len'] > 50).astype(float) * 12
            not_excessive = (df['resp_len'] < 4000).astype(float) * 13
            score_components += good_ratio + sufficient_detail + not_excessive

        premium_sources = ['paper', 'textbook', 'human_annotated']
        if 'source' in df.columns:
            source_bonus = df['source'].isin(premium_sources).astype(float) * 10
            known_source = df['source'].notna().astype(float) * 5
            score_components += source_bonus + known_source

        max_possible = 85.0
        final_score = (score_components / max_possible).clip(0, 1).round(3)

        df['quality_score'] = final_score
        df['quality_tier'] = pd.cut(
            final_score,
            bins=[0, 0.4, 0.6, 0.75, 0.9, 1.01],
            labels=['D', 'C', 'B', 'A', 'S'],
        )

        self.scored = df
        tier_dist = df['quality_tier'].value_counts().sort_index()

        self._log('quality_scoring',
                   f"平均分: {final_score.mean():.3f}")
        self._log('tier_distribution',
                   f"S={tier_dist.get('S',0)} A={tier_dist.get('A',0)} "
                   f"B={tier_dist.get('B',0)} C={tier_dist.get('C',0)} D={tier_dist.get('D',0)}")
        return self
```
