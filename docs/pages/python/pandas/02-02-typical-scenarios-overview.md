---
title: 典型场景全景图
description: 千万级 token 语料清洗、RAG 知识库元数据管理、LLM Agent 数据分析工具链、模型评估与结果对比的完整场景解析
---
# 典型应用场景总览


## 场景全景

在大模型开发中，Pandas 的应用场景可以归纳为四大类。本章将逐一展开，每个场景都附带**可运行的代码**和**生产级注意事项**。

```
┌─────────────────────────────────────────────────────┐
│              Pandas × LLM 四大典型场景                │
│                                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│  │ 场景一    │  │ 场景二    │  │ 场景三    │          │
│  │ 语料清洗  │  │ RAG元数据 │  │ Agent    │          │
│  │          │  │ 管理      │  │ 数据分析  │          │
│  │ 千万级    │  │ 知识库    │  │ 工具链    │          │
│  │ token     │  │ 分块索引  │  │ PandasAI │          │
│  └──────────┘  └──────────┘  └──────────┘          │
│                                                     │
│  ┌──────────────────────────────────┐               │
│  │ 场景四：模型评估与结果对比分析      │               │
│  └──────────────────────────────────┘               │
└─────────────────────────────────────────────────────┘
```

## 场景一：千万级 token 语料清洗与结构化

### 问题背景

假设你要构建一个中文对话模型的 SFT 数据集，数据来源包括：

- 用户在产品中的真实对话日志（CSV，约 300 万条）
- 从论坛爬取的问答对（JSON，约 200 万条）
- 内部标注团队的标注数据（Excel，约 50 万条）
- 开源数据集（Parquet 格式，约 100 万条）

总计约 **650 万条原始数据**，目标输出 **80 万条高质量 SFT 训练数据**。

### 完整流水线代码

```python
import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 60)
print("Step 1: 加载多源数据")
print("=" * 60)

df_logs = pd.read_csv('data/user_conversations.csv',
    usecols=['conversation_id', 'user_id', 'timestamp', 'user_message',
             'assistant_message', 'turn_count', 'session_duration'],
    parse_dates=['timestamp'],
    dtype={'conversation_id': 'str', 'user_id': 'str'}
)
print(f"  对话日志: {len(df_logs):,} 条")

df_crawl = pd.read_json('data/forum_qa.jsonl', lines=True,
    columns=['question', 'answer', 'source_url', 'upvotes', 'created_at'])
print(f"  爬取数据: {len(df_crawl):,} 条")

df_label = pd.read_excel('data/annotations.xlsx',
    sheet_name='Sheet1',
    usecols=['id', 'prompt', 'response', 'labeler_id', 'quality_score',
             'category', 'annotated_at'])
print(f"  标注数据: {len(df_label):,} 条")

df_open = pd.read_parquet('data/open_dataset.parquet',
    columns=['instruction', 'output', 'source', 'task_type'])
print(f"  开源数据: {len(df_open):,} 条")

print("\n" + "=" * 60)
print("Step 2: 统一数据 Schema")
print("=" * 60)

def normalize_logs(df):
    return df.rename(columns={
        'user_message': 'prompt',
        'assistant_message': 'response'
    }).assign(
        source='product_logs',
        quality_score=5.0,
        category='conversation'
    )[['prompt', 'response', 'source', 'quality_score', 'category']]

def normalize_crawl(df):
    return df.rename(columns={
        'question': 'prompt',
        'answer': 'response'
    }).assign(
        source='forum_crawl',
        quality_score=np.where(df['upvotes'] > 10, 4.0, 3.0),
        category='qa_pair'
    )[['prompt', 'response', 'source', 'quality_score', 'category']]

def normalize_label(df):
    return df.rename(columns={
        'prompt': 'prompt',
        'response': 'response'
    }).assign(
        source='human_annotated',
        category='category'
    )[['prompt', 'response', 'source', 'quality_score', 'category']]

def normalize_open(df):
    return df.rename(columns={
        'instruction': 'prompt',
        'output': 'response'
    }).assign(
        source='open_source',
        quality_score=4.0,
        category=lambda x: x['task_type'].fillna('general')
    )[['prompt', 'response', 'source', 'quality_score', 'category']]

normalized_logs = normalize_logs(df_logs)
normalized_crawl = normalize_crawl(df_crawl)
normalized_label = normalize_label(df_label)
normalized_open = normalize_open(df_open)

all_data = pd.concat(
    [normalized_logs, normalized_crawl, normalized_label, normalized_open],
    ignore_index=True
)

print(f"  合并后总计: {len(all_data):,} 条")
print(f"  各来源分布:")
print(all_data['source'].value_counts())

print("\n" + "=" * 60)
print("Step 3: 质量过滤")
print("=" * 60)

before_count = len(all_data)

mask_valid = (
    all_data['prompt'].notna() &
    all_data['response'].notna() &
    (all_data['prompt'].astype(str).str.len() >= 5) &
    (all_data['response'].astype(str).str.len() >= 10) &
    (~all_data['prompt'].str.contains(r'^[\s\W]+$', regex=True, na=False)) &
    (~all_data['response'].str.contains(r'^(?:sorry|抱歉|无法|不知道)', regex=True, case=False, na=False))
)
all_data = all_data[mask_valid].copy()
print(f"  完整性过滤: {before_count:,} → {len(all_data):,} "
      f"(去除 {before_count - len(all_data):,}, {(1 - len(all_data)/before_count)*100:.1f}%)")

def clean_text(text):
    if pd.isna(text) or not isinstance(text, str):
        return ''
    import re
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'&[a-z]+;', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[\u200b-\u200f\u2028-\u2029]', '', text)
    return text.strip()

all_data['prompt_clean'] = all_data['prompt'].apply(clean_text)
all_data['response_clean'] = all_data['response'].apply(clean_text)

def estimate_tokens(text):
    if not text:
        return 0
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    other_chars = len(text) - chinese_chars
    words = len(text.split())
    return int(chinese_chars * 1.5 + max(other_chars * 0.5, words * 1.0))

all_data['prompt_tokens'] = all_data['prompt_clean'].apply(estimate_tokens)
all_data['response_tokens'] = all_data['response_clean'].apply(estimate_tokens)

length_mask = (
    (all_data['prompt_tokens'].between(10, 4096)) &
    (all_data['response_tokens'].between(20, 8192)) &
    ((all_data['prompt_tokens'] + all_data['response_tokens']).between(50, 12000))
)
all_data = all_data[length_mask].copy()
print(f"  长度过滤后: {len(all_data):,} 条")

all_data['content_hash'] = (all_data['prompt_clean'] + '<SEP>' + all_data['response_clean']).apply(
    lambda x: hash(x) & 0xFFFFFFFFFFFFFFFF
)
before_dedup = len(all_data)
all_data = all_data.drop_duplicates(subset=['content_hash'], keep='first')
print(f"  去重后: {len(all_data):,} 条 (去除 {before_dedup - len(all_data):,} 重复)")

print("\n" + "=" * 60)
print("Step 4: 分层采样至目标规模")
print("=" * 60)

TARGET_SIZE = 800_000

all_data['stratum'] = all_data['source'] + '_' + all_data['category']

stratum_counts = all_data['stratum'].value_counts()
total = stratum_counts.sum()
sampling_ratio = min(TARGET_SIZE / total, 1.0)

sampled = all_data.groupby('stratum', group_keys=False).apply(
    lambda g: g.sample(n=max(int(len(g) * sampling_ratio), 100),
                       random_state=42) if len(g) > 100 else g
).reset_index(drop=True)

if len(sampled) > TARGET_SIZE:
    sampled = sampled.sample(n=TARGET_SIZE, random_state=42).reset_index(drop=True)

print(f"  最终训练集大小: {len(sampled):,} 条")
print(f"\n  最终数据分布:")
print(sampled[['source', 'category']].value_counts().head(15))

print("\n" + "=" * 60)
print("Step 5: 导出为训练格式")
print("=" * 60)

def build_sft_messages(row):
    messages = [
        {'role': 'user', 'content': row['prompt_clean']},
        {'role': 'assistant', 'content': row['response_clean']}
    ]
    return messages

sampled['messages'] = sampled.apply(build_sft_messages, axis=1)

export_cols = ['messages', 'source', 'category', 'prompt_tokens', 'response_tokens']
sampled[export_cols].to_json(
    'output/sft_train_v1.jsonl',
    orient='records',
    lines=True,
    force_ascii=False
)

print(f"  已导出到: output/sft_train_v1.jsonl")

report = {
    'total_raw': before_count,
    'after_cleaning': len(all_data),
    'final_size': len(sampled),
    'sources': sampled['source'].value_counts().to_dict(),
    'categories': sampled['category'].value_counts().to_dict(),
    'avg_prompt_tokens': float(sampled['prompt_tokens'].mean()),
    'avg_response_tokens': float(sampled['response_tokens'].mean()),
}

pd.DataFrame([report]).to_json('output/cleaning_report.json', orient='records', force_ascii=False, indent=2)
print(f"  清洗报告已保存到: output/cleaning_report.json")
```

### 性能要点

| 操作 | 650 万条耗时 | 内存占用 | 优化建议 |
|------|------------|---------|---------|
| `read_csv` | ~8s | ~2GB | 使用 `dtype` 指定类型减少内存 |
| `concat` | ~2s | ~4GB | 注意各源列名必须一致 |
| 文本清洗 (`apply`) | ~45s | ~4GB | 考虑用 `str` 向量化加速 |
| `drop_duplicates` | ~12s | ~4GB | Hash 列减少比较开销 |
| `groupby` 采样 | ~8s | ~4GB | 分层抽样保证分布均衡 |
| `to_json` 导出 | ~18s | ~6GB | JSONL 是逐行写入的 |

## 场景二：RAG 知识库的文档元数据管理

### 问题背景

你的 RAG 系统需要管理来自以下来源的知识库文档：

- 产品 API 文档（Markdown，200 个文件）
- 技术博客文章（HTML，500 篇）
- 内部 Wiki 页面（Confluence 导出，300 个页面）
- PDF 白皮书（20 份）

每个文档需要被分块、嵌入、存入向量数据库，同时维护完整的元数据。

### 元数据管理系统

```python
import pandas as pd
import hashlib
from datetime import datetime
from pathlib import Path

class KnowledgeBaseManager:
    def __init__(self, metadata_path: str = 'data/kb_metadata.parquet'):
        self.metadata_path = metadata_path
        if Path(metadata_path).exists():
            self.df = pd.read_parquet(metadata_path)
        else:
            self.df = self._empty_schema()
    
    def _empty_schema(self):
        return pd.DataFrame({
            'chunk_id': pd.Series(dtype='string'),
            'doc_source': pd.Series(dtype='string'),
            'doc_title': pd.Series(dtype='string'),
            'chunk_index': pd.Series(dtype='Int32'),
            'chunk_text': pd.Series(dtype='string'),
            'token_count': pd.Series(dtype='Int16'),
            'embedding_model': pd.Series(dtype='string'),
            'vector_db_id': pd.Series(dtype='string'),
            'status': pd.CategoricalDtype(categories=['pending', 'indexed', 'archived', 'error']),
            'last_updated': pd.Series(dtype='datetime64[ns]'),
            'content_hash': pd.Series(dtype='string'),
            'category': pd.Series(dtype='string'),
            'access_level': pd.CategoricalDtype(categories=['public', 'internal', 'restricted']),
        })
    
    def add_document(self, doc_source: str, doc_title: str, chunks: list[str],
                    category: str, access_level: str = 'public'):
        new_rows = []
        for idx, chunk_text in enumerate(chunks):
            content_hash = hashlib.md5(chunk_text.encode()).hexdigest()
            token_est = len(chunk_text.split()) + sum(1 for c in chunk_text if '\u4e00' <= c <= '\u9fff')
            
            new_rows.append({
                'chunk_id': f"{Path(doc_source).stem}_ch{idx:03d}",
                'doc_source': doc_source,
                'doc_title': doc_title,
                'chunk_index': idx,
                'chunk_text': chunk_text,
                'token_count': min(token_est, 32767),
                'embedding_model': None,
                'vector_db_id': None,
                'status': 'pending',
                'last_updated': pd.Timestamp.now(),
                'content_hash': content_hash,
                'category': category,
                'access_level': access_level,
            })
        
        new_df = pd.DataFrame(new_rows)
        self.df = pd.concat([self.df, new_df], ignore_index=True)
        return len(new_rows)
    
    def mark_indexed(self, chunk_ids: list[str], vector_db_ids: list[str],
                    embedding_model: str = 'text-embedding-3-small'):
        mask = self.df['chunk_id'].isin(chunk_ids)
        id_map = dict(zip(chunk_ids, vector_db_ids))
        
        self.df.loc[mask, 'vector_db_id'] = (
            self.df.loc[mask, 'chunk_id'].map(id_map)
        )
        self.df.loc[mask, 'embedding_model'] = embedding_model
        self.df.loc[mask, 'status'] = 'indexed'
        self.df.loc[mask, 'last_updated'] = pd.Timestamp.now()
    
    def detect_changes(self, new_chunks: dict[str, str]) -> list[str]:
        changed_ids = []
        for chunk_id, new_text in new_chunks.items():
            new_hash = hashlib.md5(new_text.encode()).hexdigest()
            existing = self.df[self.df['chunk_id'] == chunk_id]
            
            if len(existing) == 0:
                changed_ids.append(chunk_id)
            elif existing.iloc[0]['content_hash'] != new_hash:
                changed_ids.append(chunk_id)
        
        return changed_ids
    
    def get_stats(self) -> pd.DataFrame:
        stats = {
            '指标': ['总分块数', '已索引', '待处理', '已归档', '错误', '唯一文档数'],
            '数值': [
                len(self.df),
                (self.df['status'] == 'indexed').sum(),
                (self.df['status'] == 'pending').sum(),
                (self.df['status'] == 'archived').sum(),
                (self.df['status'] == 'error').sum(),
                self.df['doc_source'].nunique()
            ]
        }
        return pd.DataFrame(stats)
    
    def save(self):
        self.df.to_parquet(self.metadata_path, index=False)


kb = KnowledgeBaseManager()

chunks_api = ["API 文档第 {} 章...".format(i) for i in range(25)]
kb.add_document('docs/api_reference.md', 'API 参考文档', chunks_api,
               category='api', access_level='public')

chunks_blog = ["博客文章段落 {}...".format(i) for i in range(30)]
kb.add_document('blogs/llm-rag-guide.html', 'LLM RAG 实战指南', chunks_blog,
               category='tutorial', access_level='public')

print(kb.get_stats())
```

## 场景三：LLM Agent 数据分析工具链（Pandas Agent）

### 什么是 Pandas Agent

Pandas Agent 的核心思想是：

> **将自然语言问题 → LLM 理解意图 → 自动生成 Pandas 代码 → 执行并返回结果**

这意味着非技术人员也可以通过自然语言完成数据分析。

### 快速上手示例

```python
import pandas as pd
from pandasai import PandaAI

df = pd.DataFrame({
    'model': ['GPT-4o', 'Claude-3.5-Sonnet', 'Gemini-1.5-Pro',
              'Llama-3.1-405B', 'Qwen2.5-72B', 'DeepSeek-V3'],
    'context_window': [128000, 200000, 1000000, 131072, 131072, 65536],
    'price_input_per_1m': [2.50, 3.00, 1.25, 0.27, 0.27, 0.14],
    'price_output_per_1m': [10.00, 15.00, 5.00, 1.10, 1.10, 0.56],
    'benchmark_mmlu': [88.7, 89.2, 86.8, 85.3, 84.7, 83.5],
    'benchmark_humanity': [92.1, 91.8, 90.2, 88.5, 87.9, 86.2],
    'supports_function_calling': [True, True, True, True, True, True],
    'supports_vision': [True, True, True, False, False, True],
    'training_data_cutoff': ['2024-04', '2024-05', '2024-11', '2024-03', '2024-09', '2024-12'],
})

agent = PandaAI(llm="gpt-4o", verbose=True)

result = agent.chat(df, "哪个模型性价比最高？请综合考虑 MMLU 得分和输入价格")

result = agent.chat(df, "画一个各模型 context window 对比的横向柱状图")

result = agent.chat(df, "支持视觉能力且 MMLU 超过 85 的模型有哪些？它们的平均上下文窗口是多少？")
```

### 生产环境注意事项

```python
import pandas as pd
from pandasai import PandaAI
from pandasai_docker import DockerSandbox

agent = PandaAI(
    llm="gpt-4o",
    sandbox=DockerSandbox()  # 防止恶意代码执行
)

safe_df = df.drop(columns=['api_key', 'internal_notes'])  # 移除敏感列
safe_df['price_input_per_1m'] = safe_df['price_input_per_1m'].round(2)

result = agent.chat(
    safe_df,
    "只做统计分析，不要修改任何数据。告诉我各模型的价格区间分布。",
    config={
        'max_iterations': 3,         # 限制最大迭代次数
        'verbose': True,             # 打印生成的代码供审查
        'enable_cache': True,        # 缓存相同查询的结果
    }
)
```

## 场景四：模型评估与结果对比分析

### 多模型对比框架

```python
import pandas as pd
import numpy as np

np.random.seed(2026)

models = ['GPT-4o', 'Claude-3.5-Sonnet', 'Gemini-1.5-Pro', 'DeepSeek-V3', 'Qwen2.5-72B']
domains = ['代码生成', '数学推理', '知识问答', '文本创作', '逻辑推理', '摘要总结']
difficulties = ['简单', '中等', '困难']

n_samples = 300

eval_df = pd.DataFrame({
    'sample_id': range(1, n_samples + 1),
    'domain': np.random.choice(domains, n_samples),
    'difficulty': np.random.choice(difficulties, n_samples),
})

for model in models:
    base_score = {'GPT-4o': 4.2, 'Claude-3.5-Sonnet': 4.3, 'Gemini-1.5-Pro': 4.0,
                  'DeepSeek-V3': 3.9, 'Qwen2.5-72B': 3.8}[model]
    noise = np.random.normal(0, 0.8, n_samples)
    difficulty_penalty = eval_df['difficulty'].map({'简单': 0.3, '中等': 0, '困难': -0.5})
    domain_bonus = eval_df['domain'].map({
        '代码生成': {'GPT-4o': 0.3, 'Claude-3.5-Sonnet': 0.4}.get(model, 0),
        '数学推理': {'GPT-4o': 0.2, 'Claude-3.5-Sonnet': 0.3}.get(model, 0),
    }) if model in ['GPT-4o', 'Claude-3.5-Sonnet'] else 0
    
    eval_df[f'{model}_score'] = np.clip(base_score + noise + difficulty_penalty + (domain_bonus or 0), 1, 5).round(2)

score_cols = [f'{m}_score' for m in models]
overall_ranking = eval_df[score_cols].mean().sort_values(ascending=False)
print("=== 总体得分排名 ===")
print(overall_ranking.round(3))

domain_breakdown = eval_df.groupby('domain')[score_cols].mean().round(2)
print("\n=== 各领域表现 ===")
print(domain_breakdown.T)

difficulty_breakdown = eval_df.groupby('difficulty')[score_cols].mean().round(2)
print("\n=== 各难度表现 ===")
print(difficulty_breakdown.T)

eval_df['min_score'] = eval_df[score_cols].min(axis=1)
eval_df['max_score'] = eval_df[score_cols].max(axis=1)
eval_df['score_range'] = eval_df['max_score'] - eval_df['min_score']

bad_cases = eval_df[
    (eval_df['min_score'] < 2.5) &
    (eval_df['difficulty'] == '困难')
].sort_values('min_score')

print(f"\n=== 困难题型中的 Bad Case ({len(bad_cases)} 条) ===")
print(bad_cases[['domain', 'difficulty'] + score_cols].head(10))

pairwise_results = []
for i, m1 in enumerate(models):
    for m2 in models[i+1:]:
        col1, col2 = f'{m1}_score', f'{m2}_score'
        wins = (eval_df[col1] > eval_df[col2]).sum()
        ties = (eval_df[col1] == eval_df[col2]).sum()
        losses = (eval_df[col1] < eval_df[col2]).sum()
        pairwise_results.append({
            'Model_A': m1, 'Model_B': m2,
            'Win': wins, 'Tie': ties, 'Loss': losses,
            'Win_Rate': round(wins / n_samples * 100, 1)
        })

pairwise_df = pd.DataFrame(pairwise_results)
print("\n=== 配对比较 (Win Rate %) ===")
print(pairwise_df.pivot_table(index='Model_A', columns='Model_B', values='Win_Rate').round(1))
```

## 四大场景的能力矩阵

| 能力需求 | 场景一：语料清洗 | 场景二：RAG 元数据 | 场景三：Agent | 场景四：评估分析 |
|----------|:-:|:-:|:-:|:-:|
| 大数据量处理 | ★★★★★ | ★★★★☆ | ★★☆☆☆ | ★★★★☆ |
| 字符串操作 | ★★★★★ | ★★★☆☆ | ★★☆☆☆ | ★★★☆☆ |
| 去重/去噪 | ★★★★★ | ★★★★★ | ★☆☆☆☆ | ★★★☆☆ |
| 分层抽样 | ★★★★★ | ★☆☆☆☆ | ★☆☆☆☆ | ★★★★★ |
| 哈希/变更检测 | ★★★☆☆ | ★★★★★ | ☆☆☆☆☆ | ☆☆☆☆☆ |
| 统计聚合 | ★★★☆☆ | ★★★★☆ | ★★★★★ | ★★★★★ |
| 可视化集成 | ★★☆☆☆ | ★★☆☆☆ | ★★★★★ | ★★★★★ |
| 与 LLM 结合 | ★★☆☆☆ | ☆☆☆☆☆ | ★★★★★ | ★★★☆☆ |

每种场景都有其最适合的 Pandas 工具组合——后续章节会逐一深入。
