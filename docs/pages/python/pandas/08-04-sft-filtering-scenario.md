---
title: LLM 场景：筛选高质量对话用于 SFT 训练
description: 完整的 SFT 数据集构建筛选流水线、质量分层采样、Bad Case 过滤与数据平衡策略
---
# SFT 数据筛选实战


## SFT 数据筛选的核心目标

SFT（Supervised Fine-Tuning）对训练数据有明确的质量要求：

| 维度 | 要求 | 原因 |
|------|------|------|
| **完整性** | prompt 和 response 都不能为空 | 空样本会导致训练崩溃或质量下降 |
| **质量** | 内容有信息量，不是垃圾/错误 | 垃圾数据会污染模型输出 |
| **多样性** | 覆盖不同主题、风格和难度 | 单一分布导致模型泛化差 |
| **安全性** | 无敏感信息、无有害内容 | 安全风险 |
| **长度合理** | 在上下文窗口范围内 | 超长会被截断，过短无意义 |

## 完整 SFT 数据构建流水线

```python
import pandas as pd
import numpy as np

np.random.seed(42)
N_RAW = 500_000

raw = pd.DataFrame({
    'conversation_id': [f'conv_{i:06d}' for i in range(N_RAW)],
    'user_message': [
        f'{"问题"+str(i%200)+"}" if i%3!=0 else ""}' 
        f'ERROR: timeout' if i%500==0 else 
        f'<html>bad</html>' if i%800==0 else 
        f'正常问题 {i}'
        for i in range(N_RAW)
    ],
    'assistant_message': [
        f'回答{i}' if i%20!=0 else '' for i in range(N_RAW)
    ] + [None] * int(N_RAW*0.02),
    'turn_count': np.random.choice([None] + list(range(1, 8)), N_RAW),
    'quality_score': np.round(
        np.random.choice([None] + list(np.arange(1.0, 5.1, 0.5)), N_RAW), 1
    ),
    'token_count': np.random.randint(5, 4000, N_RAW).astype(float),
    'source': np.random.choice([None, 'api', 'web', 'export', 'mobile'], N_RAW),
    'model': np.random.choice(['GPT-4o', 'Claude', 'Llama'], N_RAW),
    'timestamp': pd.date_range('2025-01-01', periods=N_RAW, freq='min'),
})


class SFTDataBuilder:
    """SFT 数据集构建器"""
    
    def __init__(self, raw_df):
        self.df = raw_df.copy()
        self.stages = []
        self.stats = {}
    
    def _log(self, stage_name, before, after):
        removed = before - after
        pct = (removed / before * 100) if before > 0 else 0
        entry = {
            'stage': stage_name,
            'before': before,
            'after': after,
            'removed': removed,
            'pct_removed': round(pct, 2)
        }
        self.stages.append(entry)
        print(f"  [{stage_name}] {before:,} → {after:,} "
              f"(去除 {removed:,}, {pct:.1f}%)")
        return after
    
    def stage1_integrity(self):
        """Stage 1: 数据完整性检查"""
        before = len(self.df)
        
        m = (
            self.df['user_message'].notna() &
            (self.df['user_message'].str.strip() != '') &
            self.df['assistant_message'].notna() &
            (self.df['assistant_message'].str.len() > 5) &
            self.df['quality_score'].notna()
        )
        
        self.df = self.df[m].copy()
        return self._log('Stage 1 完整性', before, len(self.df))
    
    def stage2_content_quality(self):
        """Stage 2: 内容质量过滤"""
        before = len(self.df)
        
        no_error = ~self.df['user_message'].str.contains(
            r'ERROR|HTTP \d+|<html>|timeout', case=False, na=False
        )
        
        min_prompt_len = 8
        min_response_len = 15
        length_ok = (
            self.df['user_message'].str.len() >= min_prompt_len
        ) & (
            self.df['assistant_message'].str.len() >= min_response_len
        )
        
        not_trivial = ~self.df['user_message'].str.contains(r'^[A-Za-z]+\d*$', regex=True, na=False)
        
        combined = no_error & length_ok & not_trivial
        
        self.df = self.df[combined].copy()
        return self._log('Stage 2 内容质量', before, len(self.df))
    
    def stage3_length_constraints(self):
        """Stage 3: 长度约束"""
        before = len(self.df)
        
        max_total_tokens = 3500  # 模型上下文窗口 - system prompt - overhead
        min_single = 30
        
        length_ok = (
            self.df['token_count'].between(min_single, max_total_tokens) &
            (self.df['user_message'].str.len() / 4 <= max_total_tokens * 0.7)  # prompt 不超70%
        )
        
        ratio_ok = (
            self.df['assistant_message'].str.len() >= 
            self.df['user_message'].str.len() * 0.3
        )  # response 至少是 prompt 的 30%
        
        self.df = self.df[length_ok & ratio_ok].copy()
        return self._log('Stage 3 长度约束', before, len(self.df))
    
    def stage4_source_filtering(self):
        """Stage 4: 来源过滤"""
        before = len(self.df)
        
        valid_sources = ['api', 'web']  # 排除 mobile/export（质量通常较低）
        source_ok = self.df['source'].isin(valid_sources)
        
        self.df = self.df[source_ok].copy()
        return self._log('Stage 4 来源过滤', before, len(self.df))
    
    def stage5_stratified_sampling(self, n_target=50_000, random_state=42):
        """Stage 5: 分层采样到目标数量"""
        before = len(self.df)
        
        if before <= n_target:
            print(f"  [Stage 5] 数据量已满足 ({before:,} <= {n_target:,})")
            return self.df
        
        self.df['quality_tier'] = pd.cut(
            self.df['quality_score'],
            bins=[0, 3.0, 4.0, 4.5, 5.0],
            labels=['low', 'medium', 'good', 'excellent']
        )
        
        sampled_dfs = []
        for tier in ['excellent', 'good', 'medium']:
            tier_data = self.df[self.df['quality_tier'] == tier]
            
            if len(tier_data) > 0:
                n_from_this = max(int(n_target * {
                    'excellent': 0.40,
                    'good': 0.35,
                    'medium': 0.25,
                }.get(tier, 0.15)), min(len(tier_data) // 10))
                
                sampled = tier_data.sample(n=n_from_this, random_state=random_state)
                sampled_dfs.append(sampled)
                
                print(f"    {tier}: {len(tier_data):,} → 取 {len(sampled):,}")
        
        low_data = self.df[self.df['quality_tier'] == 'low']
        if len(low_data) > 0 and len(pd.concat(sampled_dfs)) < n_target * 0.95:
            remaining = n_target - len(pd.concat(sampledfs))
            sampled_low = low_data.sample(n=min(remaining, len(low_data)), random_state=random_state)
            sampled_dfs.append(sampled_low)
            print(f"    low: {len(low_data):,} → 取 {len(sampled_low):,}")
        
        self.df = pd.concat(sampledfs, ignore_index=True).drop(columns=['quality_tier'])
        return self._log('Stage 5 分层采样', before, len(self.df))
    
    def build_messages_format(self):
        """转换为 messages 格式"""
        def to_messages(row):
            messages = []
            messages.append({'role': 'user', 'content': row['user_message']})
            messages.append({'role': 'assistant', 'content': row['assistant_message']})
            return messages
        
        self.df['messages'] = self.df.apply(to_messages, axis=1)
        return self.df
    
    def get_report(self):
        """生成处理报告"""
        report = pd.DataFrame(self.stages)
        total_original = report.iloc[0]['before']
        final_count = report.iloc[-1]['after']
        
        print("\n" + "=" * 60)
        print("  📋 SFT 数据集构建报告")
        print("=" * 60)
        print(report.to_string(index=False))
        print("-" * 60)
        print(f"\n  最终数据集: {final_count:,} 条 "
              f"(保留率: {final_count/total_original*100:.1f}%)")
        
        return self.df


builder = SFTDataBuilder(raw)

sft_df = (
    builder
    .stage1_integrity()
    .stage2_content_quality()
    .stage3_length_constraints()
    .stage4_source_filtering()
    .stage5_stratified_sampling(n_target=80_000)
    .build_messages_format()
    .get_report()
)

output_cols = ['conversation_id', 'messages', 'source', 'quality_score',
               'token_count', 'model']
sft_df[output_cols].to_json(
    'output/sft_train_v1.jsonl',
    orient='records',
    lines=True,
    force_ascii=False
)
print(f"\n✅ 已导出到 output/sft_train_v1.jsonl")
```
