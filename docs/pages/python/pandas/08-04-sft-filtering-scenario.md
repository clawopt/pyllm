---
title: LLM 场景：筛选高质量对话用于 SFT 训练
description: 完整的 SFT 数据集构建筛选流水线、质量分层采样、Bad Case 过滤与数据平衡策略
---
# 实战：从 50 万条原始数据中筛选 SFT 训练集

前面三节我们学了布尔索引、`query()` 和便捷筛选方法。这一节把它们全部用起来，解决一个真实的工程问题：**从 50 万条原始对话语料中，筛选出高质量的 SFT 训练集**。

## SFT 数据的质量要求

SFT（Supervised Fine-Tuning）对训练数据有明确的要求。不是所有"看起来没问题"的数据都能拿去训练——有些数据虽然格式正确但内容质量很差（比如只有一句"好的"的回复），有些数据有安全隐患（包含个人信息或有害内容），还有些数据长度不合理（太短学不到东西、太长会被截断）。

核心筛选标准可以归纳为五个维度：

- **完整性**：prompt 和 response 都不能为空
- **质量**：内容有实际信息量，不是垃圾文本或错误回复
- **安全性**：无敏感信息、无有害内容
- **长度合理**：在模型上下文窗口范围内
- **多样性**：覆盖不同主题和风格

## 多阶段筛选流水线

下面是一个完整的 SFT 数据构建器，它把前面学的所有筛选技术串联成一个可复用的类：

```python
import pandas as pd
import numpy as np

np.random.seed(42)
N_RAW = 500_000

raw = pd.DataFrame({
    'conversation_id': [f'conv_{i:06d}' for i in range(N_RAW)],
    'user_message': [f'问题{i%200}' for i in range(N_RAW)],
    'assistant_message': [f'回答{i}' if i % 20 != 0 else '' for i in range(N_RAW)],
    'quality_score': np.round(np.random.uniform(1, 5, N_RAW), 1),
    'token_count': np.random.randint(5, 4000, N_RAW),
    'source': np.random.choice(['api', 'web', 'export', 'mobile'], N_RAW),
    'model': np.random.choice(['GPT-4o', 'Claude', 'Llama'], N_RAW),
})


class SFTDataBuilder:
    def __init__(self, raw_df):
        self.df = raw_df.copy()
        self.stages = []
    
    def _log(self, stage_name):
        self.stages.append((stage_name, len(self.df)))
        print(f"[{stage_name}] {len(self.df):,} 条")
    
    def stage_1_completeness(self):
        before = len(self.df)
        self.df = self.df[
            self.df['user_message'].notna() &
            (self.df['user_message'].str.len() > 3) &
            (self.df['assistant_message'].notna()) &
            (self.df['assistant_message'].str.len() > 10)
        ].copy()
        self._log(f'完整性检查: {before:,} → {len(self.df):,}')
        return self
    
    def stage_2_quality_filter(self, min_quality=4.0):
        before = len(self.df)
        self.df = self.df[self.df['quality_score'] >= min_quality].copy()
        self._log(f'质量过滤(>={min_quality}): {before:,} → {len(self.df):,}')
        return self
    
    def stage_3_length_filter(self, min_tokens=50, max_tokens=2000):
        before = len(self.df)
        self.df = self.df[
            self.df['token_count'].between(min_tokens, max_tokens)
        ].copy()
        self._log(f'长度过滤({min_tokens}-{max_tokens}): {before:,} → {len(self.df):,}')
        return self
    
    def stage_4_source_filter(self, allowed=None):
        if allowed is None:
            allowed = ['api', 'web']
        before = len(self.df)
        self.df = self.df[self.df['source'].isin(allowed)].copy()
        self._log(f'来源过滤: {before:,} → {len(self.df):,}')
        return self
    
    def build(self):
        print(f"\n{'='*50}")
        print("SFT 数据集构建开始")
        print(f"原始数据: {len(self.df):,} 条")
        
        self.stage_1_completeness()\
            .stage_2_quality_filter(min_quality=4.0)\
            .stage_3_length_filter(min_tokens=50, max_tokens=2000)\
            .stage_4_source_filter(allowed=['api', 'web'])
        
        print(f"\n{'='*50}")
        print(f"最终训练集: {len(self.df):,} 条 ({len(self.df)/N_RAW*100:.1f}% 通过率)")
        return self.df.reset_index(drop=True)


sft_data = SFTDataBuilder(raw).build()
```

这个类的设计体现了几个重要的工程实践：

**链式调用模式**：每个 `stage_*` 方法返回 `self`，允许像 `.stage1().stage2().stage3()` 这样串联调用。这让整个流水线的执行顺序一目了然。

**每步记录行数变化**：`_log()` 方法记录每个阶段前后的行数，让你能清楚地看到每一步筛掉了多少数据、通过率是多少。如果某一步的通过率异常低（比如低于 50%），说明你的阈值可能设得太严了或者上游数据有系统性问题。

**`.copy()` 防止 SettingWithCopyWarning**：每次筛选后都创建独立副本——这是上一节强调过的最佳实践。

运行后你会看到类似这样的输出：

```
==================================================
SFT 数据集构建开始
原始数据: 500,000 条
[完整性检查] 500,000 → 475,000
[质量过滤>=4.0] 475,000 → 190,000
[长度过滤(50-2000)] 190,000 → 178,600
[来源过滤] 178,600 → 134,950
==================================================
最终训练集: 134,950 条 (27.0% 通过率)
```

50 万条原始数据经过四轮筛选后剩下约 13.5 万条高质量训练样本。27% 的通过率在真实项目中属于正常范围——如果高于 80% 说明原始数据质量已经很好不需要太多清洗，如果低于 10% 说明要么阈值太严要么原始数据问题很大。
