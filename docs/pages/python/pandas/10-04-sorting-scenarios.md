---
title: 排序实战场景
description: 模型评估排序、SFT 数据质量排序、RAG 检索结果重排、时间序列数据排序
---
# 排序场景实战


## 场景一：模型评估报告自动生成

```python
import pandas as pd
import numpy as np

np.random.seed(42)

models = ['GPT-4o', 'Claude-3.5-Sonnet', 'Llama-3.1-70B', 'Qwen2.5-72B',
          'DeepSeek-V3', 'Gemini-1.5-Pro', 'Mistral-Large', 'Yi-Large']

benchmarks = ['MMLU', 'HumanEval', 'MATH', 'GPQA', 'BBH', 'IFEval']

eval_results = []
for model in models:
    base_score = {'GPT-4o': 88, 'Claude-3.5-Sonnet': 89, 'Llama-3.1-70B': 84,
                  'Qwen2.5-72B': 83, 'DeepSeek-V3': 87, 'Gemini-1.5-Pro': 86,
                  'Mistral-Large': 82, 'Yi-Large': 80}.get(model, 78)
    for bm in benchmarks:
        eval_results.append({
            'model': model,
            'benchmark': bm,
            'score': round(base_score + np.random.randn() * (8 if bm == 'MATH' else 4), 1),
        })

eval_df = pd.DataFrame(eval_results)

pivot_df = eval_df.pivot_table(
    index='model',
    columns='benchmark',
    values='score',
    aggfunc='max'
)
pivot_df['mean'] = pivot_df.mean(axis=1).round(1)
pivot_df = pivot_df.sort_values('mean', ascending=False)

print("=== LLM Benchmark 排行榜 ===")
print(pivot_df.round(1).to_string())

print(f"\n各维度最佳模型:")
for col in benchmarks:
    best = pivot_df[col].idxmax()
    print(f"  {col}: {best} ({pivot_df.loc[best, col]:.1f})")
```

### 带权重的综合排名

```python
import pandas as pd

weights = {
    'MMLU': 0.20,
    'HumanEval': 0.25,
    'MATH': 0.15,
    'GPQA': 0.15,
    'BBH': 0.15,
    'IFEval': 0.10,
}

normalized = pivot_df[benchmarks].copy()
for col in benchmarks:
    col_min, col_max = normalized[col].min(), normalized[col].max()
    if col_max > col_min:
        normalized[col] = (normalized[col] - col_min) / (col_max - col_min)

weighted_scores = sum(normalized[col] * w for col, w in weights.items())
pivot_df['weighted_score'] = (weighted_scores * 100).round(1)
pivot_df['final_rank'] = pivot_df['weighted_score'].rank(ascending=False, method='dense').astype(int)

print("\n=== 加权综合排名 ===")
print(pivot_df[['mean', 'weighted_score', 'final_rank']].sort_values('final_rank'))
```

## 场景二：SFT 数据集按质量分层

```python
import pandas as pd
import numpy as np

np.random.seed(42)

n = 1000
sft_data = pd.DataFrame({
    'instruction': [f'指令_{i}' for i in range(n)],
    'response_len': np.random.exponential(300, n).astype(int) + 20,
    'instruction_len': np.random.exponential(50, n).astype(int) + 10,
    'has_code': np.random.choice([0, 1], n, p=[0.7, 0.3]),
    'has_math': np.random.choice([0, 1], n, p=[0.85, 0.15]),
    'source_quality': np.random.choice([5, 4, 3, 2, 1], n, p=[0.2, 0.35, 0.3, 0.12, 0.03]),
    'reward_score': np.random.beta(3, 6, n),
})

def compute_sft_tier(row):
    """计算 SFT 数据质量等级"""
    score = 0

    if row['response_len'] < 20:
        return 'reject_too_short'
    if row['response_len'] > 8000:
        score += 0
    elif row['response_len'] > 500:
        score += 2
    elif row['response_len'] > 100:
        score += 1

    if row['instruction_len'] < 5:
        return 'reject_bad_instruction'
    if row['instruction_len'] > 30:
        score += 1

    score += row['has_code'] * 1
    score += row['has_math'] * 1
    score += (row['source_quality'] - 1) * 0.5
    score += row['reward_score'] * 3

    if score >= 6:
        return 'tier_1_excellent'
    elif score >= 4:
        return 'tier_2_good'
    elif score >= 2.5:
        return 'tier_3_acceptable'
    else:
        return 'tier_4_marginal'

sft_data['tier'] = sft_data.apply(compute_sft_tier, axis=1)

tier_order = ['tier_1_excellent', 'tier_2_good', 'tier_3_acceptable', 'tier_4_marginal']
valid_data = sft_data[~sft_data['tier'].str.startswith('reject')]

print("=== SFT 数据质量分层 ===")
tier_summary = valid_data['tier'].value_counts().reindex(tier_order).fillna(0).astype(int)
for tier, count in tier_summary.items():
    bar = '█' * int(count / 20)
    label = tier.replace('tier_', '').replace('_', ' ').title()
    print(f"  {label:<20s} | {bar:<25s} | {count:>5d} ({count/len(valid_data)*100:.1f}%)")

rejected = sft_data[sft_data['tier'].str.startswith('reject')]
print(f"\n被拒绝: {len(rejected)} 条")
print(f"  - 指令过短: {(rejected['tier']=='reject_bad_instruction').sum()}")
print(f"  - 回复过短: {(rejected['tier']=='reject_too_short').sum()}")

train_split = {}
for tier in tier_order:
    subset = valid_data[valid_data['tier'] == tier]
    train_split[tier] = subset.sample(
        min(len(subset), {'tier_1_excellent': 200, 'tier_2_good': 400,
                         'tier_3_acceptable': 300, 'tier_4_marginal': 100}.get(tier, len(subset))),
        random_state=42
    )

final_train = pd.concat(train_split.values(), ignore_index=True)
print(f"\n最终训练集: {len(final_train)} 条")
```

## 场景三：RAG 检索结果重排（Reranking）

```python
import pandas as pd
import numpy as np

class Reranker:
    """检索结果重排器"""

    def __init__(self):
        self.weights = {
            'semantic_sim': 0.40,
            'keyword_match': 0.20,
            'freshness': 0.10,
            'authority': 0.15,
            'diversity': 0.15,
        }

    def rerank(self, results_df, query_keywords=None):
        df = results_df.copy()

        for feat, weight in self.weights.items():
            if feat in df.columns:
                df[f'{feat}_norm'] = self._normalize(df[feat])
            else:
                df[f'{feat}_norm'] = 0.5

        df['rerank_score'] = sum(
            df[f'{w}_norm'] * v for w, v in self.weights.items()
        )

        df = df.sort_values('rerank_score', ascending=False)
        df.insert(0, 'rerank_rank', range(1, len(df)+1))

        norm_cols = [c for c in df.columns if c.endswith('_norm')]
        df.drop(columns=norm_cols, inplace=True)

        return df

    def _normalize(self, series):
        min_val, max_val = series.min(), series.max()
        if max_val > min_val:
            return (series - min_val) / (max_val - min_val)
        return pd.Series(0.5, index=series.index)


np.random.seed(42)
n = 20
retrieved = pd.DataFrame({
    'chunk_id': [f'ch_{i:04d}' for i in range(n)],
    'title': [f'Relevant doc about topic {i}' for i in range(n)],
    'semantic_sim': np.random.uniform(0.3, 0.95, n),
    'keyword_match': np.random.uniform(0.1, 0.9, n),
    'freshness': np.random.uniform(0.2, 1.0, n),
    'authority': np.random.uniform(0.3, 1.0, n),
    'token_count': np.random.randint(150, 600, n),
})

reranker = Reranker()

original_order = retrieved[['chunk_id', 'semantic_sim']].copy()
original_order['orig_rank'] = range(1, n+1)

reranked = reranker.rerank(retrieved)

print("=== RAG 重排对比（Top 10）===")
print(f"{'Chunk':<10s} {'原始排名':>6s} {'重排后':>6s} {'语义相似':>8s} "
      f"{'关键词':>7s} {'重排分':>8s}")
print("-" * 60)
for _, row in reranked.head(10).iterrows():
    orig = original_order[original_order['chunk_id']==row['chunk_id']]['orig_rank'].values
    orig_r = orig[0] if len(orig) > 0 else '?'
    arrow = "↑" if orig_r > row['rerank_rank'] else ("↓" if orig_r < row['rerank_rank'] else "→")
    print(f"{row['chunk_id']:<10s} #{orig_r:>4d} → #{row['rerank_rank']:<4d} {arrow}  "
          f"{row.get('semantic_sim',0):.3f}   {row.get('keyword_match',0):.3f}   "
          f"{row.get('rerank_score',0):.3f}")
```

## 场景四：对话日志的时间排序与去重

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

base_time = datetime(2025, 1, 15, 9, 0, 0)
n_logs = 200

logs = pd.DataFrame({
    'session_id': [f'sess_{i//5}' for i in range(n_logs)],
    'timestamp': [base_time + timedelta(seconds=np.random.randint(0, 7200)) for _ in range(n_logs)],
    'user_query': [f'用户问题 {i % 30}' for i in range(n_logs)],
    'model_response_len': np.random.randint(50, 2000, n_logs),
    'latency_ms': np.random.randint(200, 8000, n_logs),
    'status': np.random.choice(['success', 'error', 'timeout'], n_logs, p=[0.92, 0.05, 0.03]),
})

logs_sorted = logs.sort_values(['session_id', 'timestamp'])

last_messages = logs_sorted.drop_duplicates(subset=['session_id'], keep='last')
print(f"总日志: {len(logs)} 条 → 独立会话: {last_messages['session_id'].nunique()} 个")

logs['hour'] = logs['timestamp'].dt.hour
hourly_stats = logs.groupby('hour').agg(
    total=('user_query', 'count'),
    success=('status', lambda x: (x == 'success').sum()),
    avg_latency=('latency_ms', 'mean'),
    avg_response_len=('model_response_len', 'mean'),
).round(1)

hourly_stats['success_rate'] = (hourly_stats['success'] / hourly_stats['total'] * 100).round(1)
print("\n=== 每小时对话统计 ===")
print(hourly_stats)

timeouts = logs[logs['status'] == 'timeout'].nlargest(10, 'latency_ms')
if len(timeouts) > 0:
    print(f"\n=== 超时请求 Top 10（按延迟降序）===")
    print(timeouts[['timestamp', 'session_id', 'latency_ms', 'user_query']])
```
