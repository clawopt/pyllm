---
title: 分组聚合实战场景
description: SFT 数据分层统计 / API 调用成本分析 / 模型评估报告生成 / RAG 检索日志聚合
---
# GroupBy 场景实战


## 场景一：SFT 数据集的完整分组统计

```python
import pandas as pd
import numpy as np

class SFTGroupStats:
    """SFT 数据集多维度分组统计"""

    def __init__(self, df):
        self.df = df.copy()

    def by_source_stats(self):
        """按数据来源统计"""
        stats = self.df.groupby('source').agg(
            total=('instruction', 'count'),
            avg_instr_len=('instr_len', 'mean'),
            avg_resp_len=('response_len', 'mean'),
            median_reward=('reward_score', 'median'),
            mean_reward=('reward_score', 'mean'),
            reward_std=('reward_score', 'std'),
            high_quality_rate=('reward_score', lambda x: (x >= 0.7).mean()),
            low_quality_rate=('reward_score', lambda x: (x < 0.3).mean()),
        ).round(3)

        stats['high_pct'] = (stats['high_quality_rate'] * 100).round(1)
        stats['low_pct'] = (stats['low_quality_rate'] * 100).round(1)
        return stats.sort_values('mean_reward', ascending=False)

    def by_length_tier_stats(self):
        """按长度层级统计"""
        self.df['length_tier'] = pd.cut(
            self.df['response_len'],
            bins=[0, 100, 500, 2000, 8000, float('inf')],
            labels=['XS', 'S', 'M', 'L', 'XL']
        )
        return self.df.groupby('length_tier', observed=True).agg(
            count=('instruction', 'count'),
            avg_reward=('reward_score', 'mean'),
            median_instr_len=('instr_len', 'median'),
        ).round(3)

    def cross_analysis(self):
        """来源 × 质量等级交叉表"""
        self.df['quality_tier'] = pd.cut(
            self.df['reward_score'],
            bins=[0, 0.3, 0.5, 0.7, 1.01],
            labels=['低', '中低', '中高', '高']
        )
        ct = pd.crosstab(self.df['source'], self.df['quality_tier'],
                         margins=True, margins_name='合计')
        return ct

    def full_report(self):
        print("=" * 60)
        print("SFT 数据集分组统计报告")
        print("=" * 60)

        print("\n【按来源统计】")
        source_stats = self.by_source_stats()
        print(source_stats.to_string())

        print("\n\n【按长度分布】")
        length_stats = self.by_length_tier_stats()
        print(length_stats.to_string())

        print("\n\n【来源 × 质量交叉】")
        cross = self.cross_analysis()
        print(cross.to_string())

        best_source = source_stats.index[0]
        worst_source = source_stats.index[-1]
        print(f"\n📊 关键发现:")
        print(f"   最佳来源: {best_source} (均值奖励={source_stats.loc[best_source, 'mean_reward']:.3f})")
        print(f"   最差来源: {worst_source} (均值奖励={source_stats.loc[worst_source, 'mean_reward']:.3f})")

        total = len(self.df)
        high_qual = (self.df['reward_score'] >= 0.7).sum()
        print(f"   高质量样本占比: {high_qual}/{total} ({high_qual/total*100:.1f}%)")


np.random.seed(42)
n = 1500
sft_df = pd.DataFrame({
    'source': np.random.choice(['web_crawl', 'textbook', 'wiki', 'forum', 'paper',
                                 'synthetic', 'human_annotated'], n,
                               p=[0.25, 0.15, 0.15, 0.12, 0.10, 0.13, 0.10]),
    'instruction': [f'指令_{i}' for i in range(n)],
    'instr_len': np.random.exponential(40, n).astype(int) + 5,
    'response_len': np.random.exponential(250, n).astype(int) + 20,
    'reward_score': np.random.beta(2.5, 4, n),
})

source_rewards = {
    'paper': 0.08, 'human_annotated': 0.06, 'textbook': 0.04,
    'synthetic': 0.02, 'wiki': 0.0, 'web_crawl': -0.03, 'forum': -0.05,
}
for src, adj in source_rewards.items():
    mask = sft_df['source'] == src
    sft_df.loc[mask, 'reward_score'] = (
        sft_df.loc[mask, 'reward_score'] + adj + np.random.randn(mask.sum()) * 0.08
    ).clip(0.01, 0.99)

analyzer = SFTGroupStats(sft_df)
analyzer.full_report()
```

## 场景二：API 调用成本与性能分析

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class APICostAnalyzer:
    """LLM API 调用成本分析器"""

    PRICING = {
        'gpt-4o': {'input': 2.5, 'output': 10.0},
        'gpt-4o-mini': {'input': 0.15, 'output': 0.6},
        'claude-sonnet': {'input': 3.0, 'output': 15.0},
        'claude-haiku': {'input': 0.8, 'output': 4.0},
        'deepseek-chat': {'input': 0.14, 'output': 0.28},
        'qwen-plus': {'input': 0.4, 'output': 1.2},
    }

    def __init__(self, logs_df):
        self.df = logs_df.copy()
        self._compute_cost()

    def _compute_cost(self):
        costs = []
        for _, row in self.df.iterrows():
            pricing = self.PRICING.get(row['model'], {'input': 1.0, 'output': 3.0})
            cost = (
                row['prompt_tokens'] / 1e6 * pricing['input'] +
                row['completion_tokens'] / 1e6 * pricing['output']
            )
            costs.append(round(cost, 6))
        self.df['cost_usd'] = costs

    def model_summary(self):
        summary = self.df.groupby('model').agg(
            calls=('model', 'count'),
            success=('status', lambda x: (x == 'success').sum()),
            success_rate=('status', lambda x: (x == 'success').mean()),
            total_prompt_tokens=('prompt_tokens', 'sum'),
            total_completion_tokens=('completion_tokens', 'sum'),
            total_tokens=('total_tokens', 'sum'),
            total_cost=('cost_usd', 'sum'),
            avg_latency=('latency_ms', 'mean'),
            p99_latency=('latency_ms', lambda x: x.quantile(0.99)),
            avg_cost_per_call=('cost_usd', 'mean'),
            cost_per_1M_tokens=('cost_usd', lambda x: x.sum() / (df['total_tokens'].sum()/1e6) if df['total_tokens'].sum() > 0 else 0),
        ).round(4)

        summary['success_pct'] = (summary['success_rate'] * 100).round(1)
        return summary.sort_values('total_cost', ascending=False)

    def hourly_trend(self):
        self.df['hour'] = self.df['timestamp'].dt.hour
        hourly = self.df.groupby(['hour', 'model']).agg(
            calls=('model', 'count'),
            total_cost=('cost_usd', 'sum'),
            avg_latency=('latency_ms', 'mean'),
        ).round(4)
        hourly_pivot = hourly.unstack(fill_value=0)
        return hourly_pivot

    def cost_breakdown(self):
        input_cost = (self.df['prompt_tokens'].sum() / 1e6 *
                      sum(self.PRICING[m]['input'] for m in self.df['model']))
        output_cost = (self.df['completion_tokens'].sum() / 1e6 *
                       sum(self.PRICING[m]['output'] for m in self.df['model']))
        return {
            '输入 Token 成本': round(input_cost, 2),
            '输出 Token 成本': round(output_cost, 2),
            '总成本': round(input_cost + output_cost, 2),
            '总调用量': len(self.df),
            '总 Token 数': f"{self.df['total_tokens'].sum():,}",
            '均次成本': f"${self.df['cost_usd'].mean():.4f}",
        }

    def find_expensive_calls(self, threshold=0.5):
        expensive = self.df[self.df['cost_usd'] > threshold].nlargest(10, 'cost_usd')
        return expensive[['timestamp', 'model', 'prompt_tokens',
                          'completion_tokens', 'latency_ms', 'cost_usd']]


np.random.seed(42)
base_time = datetime(2025, 3, 20)
n = 800

models_list = list(APICostAnalyzer.PRICING.keys())
model_weights = [0.18, 0.35, 0.12, 0.15, 0.12, 0.08]

api_logs = pd.DataFrame({
    'timestamp': [base_time + timedelta(minutes=np.random.randint(0, 1440)) for _ in range(n)],
    'model': np.random.choice(models_list, n, p=model_weights),
    'prompt_tokens': np.random.randint(50, 4000, n),
    'completion_tokens': np.random.randint(20, 5000, n),
    'latency_ms': np.random.exponential(600, n).astype(int) + 100,
    'status': np.random.choice(['success', 'rate_limited', 'error'], n, p=[0.94, 0.04, 0.02]),
})
api_logs['total_tokens'] = api_logs['prompt_tokens'] + api_logs['completion_tokens']

analyzer = APICostAnalyzer(api_logs)

print("=== 各模型调用统计 ===")
print(analyzer.model_summary().to_string())

print(f"\n=== 成本分解 ===")
breakdown = analyzer.cost_breakdown()
for k, v in breakdown.items():
    print(f"  {k}: {v}")

expensive = analyzer.find_expensive_calls(threshold=0.30)
if len(expensive) > 0:
    print(f"\n=== 高成本调用 Top 10 (> $0.30) ===")
    print(expensive.to_string(index=False))
```

## 场景三：模型评估数据的综合分组分析

```python
import pandas as pd
import numpy as np

def generate_eval_dataset(n=500, seed=42):
    np.random.seed(seed)
    models = ['GPT-4o', 'Claude-3.5-Sonnet', 'Llama-3.1-70B',
              'Qwen2.5-72B', 'DeepSeek-V3', 'Gemini-1.5-Pro']
    tasks = ['code_gen', 'math_solve', 'reading_comp', 'translation',
             'reasoning', 'instruction_follow', 'creative_writing', 'factual_qa']

    base_scores = {
        'GPT-4o': 88, 'Claude-3.5-Sonnet': 89, 'Llama-3.1-70B': 84,
        'Qwen2.5-72B': 83, 'DeepSeek-V3': 86, 'Gemini-1.5-Pro': 87,
    }
    task_modifiers = {
        'code_gen': 5, 'math_solve': -6, 'reading_comp': 2, 'translation': 1,
        'reasoning': 3, 'instruction_follow': 4, 'creative_writing': -1, 'factual_qa': 0,
    }

    rows = []
    for _ in range(n):
        model = np.random.choice(models)
        task = np.random.choice(tasks)
        base = base_scores[model]
        modifier = task_modifiers.get(task, 0)
        score = round(np.clip(base + modifier + np.random.randn() * 5, 35, 99), 1)
        rows.append({
            'model': model,
            'task': task,
            'score': score,
            'latency_s': round(abs(np.random.randn()) * 8 + 1.5, 2),
            'tokens_used': int(np.random.randint(300, 10000)),
        })
    return pd.DataFrame(rows)


eval_df = generate_eval_dataset()

model_summary = eval_df.groupby('model').agg(
    n=('score', 'count'),
    mean=('score', 'mean'),
    std=('score', 'std'),
    median=('score', 'median'),
    q25=('score', lambda x: x.quantile(0.25)),
    q75=('score', lambda x: x.quantile(0.75)),
    min_val=('score', 'min'),
    max_val=('score', 'max'),
    pass_80=('score', lambda x: (x >= 80).mean()),
    pass_90=('score', lambda x: (x >= 90).mean()),
).round(2)

model_summary['pass80%'] = (model_summary['pass_80'] * 100).round(1)
model_summary['pass90%'] = (model_summary['pass_90'] * 100).round(1)
model_summary['iqr'] = (model_summary['q75'] - model_summary['q25']).round(1)

print("=== 模型总体表现 ===")
display_cols = ['n', 'mean', 'median', 'std', 'iqr', 'q25', 'q75',
                'min_val', 'max_val', 'pass80%', 'pass90%']
print(model_summary[display_cols].sort_values('mean', ascending=False))

task_matrix = eval_df.pivot_table(
    index='model',
    columns='task',
    values='score',
    aggfunc='mean'
).round(1)

rank_matrix = task_matrix.rank(ascending=False, method='dense').astype(int)

print(f"\n=== 任务排名矩阵 ===")
print(rank_matrix)

print(f"\n各模型最强/最弱项:")
for model in rank_matrix.index:
    best_task = rank_matrix.loc[model].idxmin()
    worst_task = rank_matrix.loc[model].idxmax()
    best_rank = rank_matrix.loc[model].min()
    worst_rank = rank_matrix.loc[model].max()
    print(f"  {model:<22s} 最强: {best_task:<18s}(#{best_rank})  "
          f"最弱: {worst_task:<18s}(#{worst_rank})")

stability = model_summary[['std', 'mean']].copy()
stability['cv'] = (stability['std'] / stability['mean']).round(3)
stability = stability.sort_values('cv')
print(f"\n稳定性排名（CV 越小越稳定）:")
for model, row in stability.iterrows():
    bar = '█' * int(row['cv'] * 50)
    print(f"  {model:<22s} CV={row['cv']:.3f} {bar}")
```

## 场景四：RAG 检索日志的聚合分析

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class RAGLogAnalyzer:
    """RAG 检索日志分析器"""

    def __init__(self, log_df):
        self.logs = log_df.copy()

    def query_type_distribution(self):
        """查询类型分布"""
        dist = self.logs.groupby('query_type').agg(
            count=('query_id', 'count'),
            avg_retrieved=('num_retrieved', 'mean'),
            avg_top_score=('top_similarity', 'mean'),
            avg_latency_ms=('latency_ms', 'mean'),
        ).round(2)
        dist['pct'] = (dist['count'] / dist['count'].sum() * 100).round(1)
        return dist.sort_values('count', ascending=False)

    def source_effectiveness(self):
        """各来源文档的有效性"""
        source_stats = self.logs.explode('retrieved_sources').groupby(
            'retrieved_sources'
        ).agg(
            appearances=('query_id', 'count'),
            avg_position=('retrieved_positions', lambda x:
                          np.mean([p for p in x if isinstance(p, (int, float))] if len(x) > 0 else 0)),
            avg_similarity=('retrieved_similarities', lambda x:
                            np.mean([s for s in x if isinstance(s, (int, float))] if len(x) > 0 else 0)),
        ).round(3)

        source_stats['hit_rate'] = source_stats['appearances'] / len(self.logs)
        return source_stats.sort_values('appearances', ascending=False)

    def time_pattern(self):
        """时间模式分析"""
        self.logs['hour'] = self.logs['timestamp'].dt.hour
        self.logs['dow'] = self.logs['timestamp'].dt.day_name()

        hourly = self.logs.groupby('hour').agg(
            queries=('query_id', 'count'),
            avg_latency=('latency_ms', 'mean'),
            avg_relevance=('top_similarity', 'mean'),
        ).round(2)

        dow_stats = self.logs.groupby('dow').agg(
            queries=('query_id', 'count'),
            success_rate=('is_successful', 'mean'),
        ).round(3)

        dow_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        dow_stats = dow_stats.reindex(dow_order)
        dow_stats['success_pct'] = (dow_stats['success_rate'] * 100).round(1)

        return hourly, dow_stats

    def full_report(self):
        print("=" * 60)
        print("RAG 检索日志分析报告")
        print("=" * 60)

        print(f"\n总检索次数: {len(self.logs)}")
        print(f"时间范围: {self.logs['timestamp'].min()} ~ {self.logs['timestamp'].max()}")

        print("\n【查询类型分布】")
        type_dist = self.query_type_distribution()
        print(type_dist.to_string())

        hourly, dow = self.time_pattern()
        print("\n【小时级趋势（Top 10 活跃时段）】")
        print(hourly.nlargest(10, 'queries').to_string())

        print("\n【星期分布】")
        print(dow.to_string())


np.random.seed(42)
n_logs = 300
base_time = datetime(2025, 3, 15)

rag_logs = pd.DataFrame({
    'query_id': [f'q_{i:04d}' for i in range(n_logs)],
    'timestamp': [base_time + timedelta(hours=np.random.randint(0, 168)) for _ in range(n_logs)],
    'query_type': np.random.choice(['factual', 'how_to', 'comparison', 'analysis',
                                     'coding', 'troubleshooting'], n_logs,
                                   p=[0.25, 0.20, 0.15, 0.15, 0.15, 0.10]),
    'num_retrieved': np.random.randint(3, 15, n_logs),
    'top_similarity': np.random.uniform(0.3, 0.98, n_logs),
    'latency_ms': np.random.exponential(150, n_logs).astype(int) + 50,
    'is_successful': np.random.choice([1, 0], n_logs, p=[0.88, 0.12]),
    'retrieved_sources': [
        np.random.choice(['api_docs', 'wiki', 'blog', 'paper', 'kb'],
                         np.random.randint(3, 8)).tolist()
        for _ in range(n_logs)
    ],
    'retrieved_similarities': [
        np.round(np.random.uniform(0.4, 0.98, np.random.randint(3, 8)), 3).tolist()
        for _ in range(n_logs)
    ],
    'retrieved_positions': [
        list(range(1, np.random.randint(4, 9)))
        for _ in range(n_logs)
    ],
})

analyzer = RAGLogAnalyzer(rag_logs)
analyzer.full_report()
```
