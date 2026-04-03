---
title: RAG 知识库完整工作流
description: 端到端 RAG 管理系统 / 检索增强生成数据准备 / 知识图谱构建 / 导出与部署
---
# RAG 工作流实战


## 完整的 RAG 知识库管理系统

```python
import pandas as pd
import numpy as np
from datetime import datetime

class RAGKnowledgeSystem:
    """完整的 RAG 知识库管理系统"""

    def __init__(self):
        self.kb = self._create_empty_kb()
        self.retrieval_log = []
        self.version = 1

    def _create_empty(self):
        return pd.DataFrame({
            'chunk_id': pd.Series(dtype='str'),
            'content': pd.Series(dtype='object'),
            'source_doc': pd.Series(dtype='str'),
            'source_type': pd.Categorical(
                categories=['api_docs', 'wiki', 'blog', 'paper',
                             'github', 'internal']
            ),
            'chunk_index': pd.Series(dtype='int32'),
            'token_count': pd.Series(dtype='int32'),
            'embedding': pd.Series(dtype='object'),
            'embedding_status': pd.Categorical(
                categories=['pending', 'done', 'failed']
            ),
            'quality_score': pd.Series(dtype='float32'),
            'created_at': pd.Series(dtype='datetime64[ns]'),
            'version': pd.Series(dtype='int16'),
            'is_active': pd.Series(dtype='bool'),
            'tags': pd.Series(dtype='object'),
        })

    def ingest_batch(self, documents):
        """批量导入文档"""
        now = datetime.now()
        records = []
        base_id = len(self.kb) + 1

        for doc in documents:
            for i, chunk_text in enumerate(doc['chunks']):
                records.append({
                    'chunk_id': f'ch_{base_id:06d}',
                    'content': chunk_text,
                    'source_doc': doc['name'],
                    'source_type': doc.get('type', 'wiki'),
                    'chunk_index': i,
                    'token_count': len(chunk_text.split()),
                    'embedding': None,
                    'embedding_status': 'pending',
                    'quality_score': round(np.random.uniform(0.5, 1.0), 3),
                    'created_at': now,
                    'version': self.version,
                    'is_active': True,
                    'tags': [],
                })
                base_id += 1

        new_df = pd.DataFrame(records)
        self.kb = pd.concat([self.kb, new_df], ignore_index=True)

        return {
            'added': len(records),
            'total': len(self.kb),
            'sources': doc['name'] if len(documents) == 1 else f"{len(documents)} docs",
        }

    def search(self, query_vec, top_k=5, min_sim=0.3, sources=None, max_tokens=4000):
        """检索"""
        active = self.kb[self.kb['is_active']].copy()
        if sources:
            active = active[active['source_type'].isin(sources)]

        emb_col = [c for c in active.columns if c == 'embedding']
        if not emb_col or active['embedding'].dropna().empty:
            return pd.DataFrame(columns=self.kb.columns)

        valid = active[active['embedding'].notna()].copy()
        embs = np.array(valid['embedding'].tolist())
        q_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
        e_norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
        sims = (embs / e_norms) @ q_norm
        valid['_sim'] = sims

        candidates = valid[valid['_sim'] >= min_sim]
        result = candidates.nlargest(min(top_k, len(candidates)), '_sim').copy()

        total_toks = 0
        kept = []
        for idx, row in result.iterrows():
            tc = row.get('token_count', 200)
            if total_toks + tc <= max_tokens:
                kept.append(idx)
                total_toks += tc

        final = result.loc[kept].copy()
        final.insert(0, 'rank', range(1, len(final)+1))
        return final.drop(columns=['_sim'])

    def get_stats(self):
        """知识库统计"""
        active = self.kb[self.kb['is_active']]
        return {
            '总 chunks': len(self.kb),
            '活跃 chunks': len(active),
            '来源数': active['source_doc'].nunique(),
            '来源类型分布': dict(active['source_type'].value_counts().to_dict()),
            'Embedding 覆盖率': f"{(active['embedding_status']=='done').sum()}/{len(active)*100:.1f}%",
            '平均 Token/Chunk': round(active['token_count'].mean(), 0) if len(active) > 0 else 0,
            '版本号': self.version,
        }


rag = RAGKnowledgeSystem()

docs_to_import = [
    {
        'name': 'pandas_guide.pdf',
        'type': 'paper',
        'chunks': [
            'Pandas 是基于 NumPy 构建的数据分析库，提供 DataFrame 数据结构...',
            'DataFrame 可以看作是由 Series 组成的字典...',
            'groupby 操作将数据拆分为组、对每组应用函数、再合并结果...',
            'merge() 支持内连接、外连接、左连接和右连接四种模式...',
        ],
    },
    {
        'name': 'llm_api_docs.html',
        'type': 'api_docs',
        'chunks': [
            'OpenAI API 支持 GPT-4o 和 GPT-4o-mini 两款主要模型...',
            'Claude API 提供最高 200K tokens 的上下文窗口...',
            'DeepSeek API 以极低价格提供接近 GPT-4 级别的性能...',
        ],
    },
]

for doc in docs_to_import:
    result = rag.ingest_batch([doc])
    print(f"✓ {result}")

print(f"\n{'='*50}")
stats = rag.get_stats()
for k, v in stats.items():
    print(f"  {k}: {v}")
```

## 导出与备份

```python
import pandas as pd

def export_knowledge_base(kb_df, output_dir='/tmp'):
    """导出知识库为多种格式"""
    import os
    os.makedirs(output_dir, exist_ok=True)

    export_df = kb_df.copy()
    if 'embedding' in export_df.columns:
        export_df = export_df.drop(columns=['embedding'])
    export_df.to_parquet(f'{output_dir}/kb_backup.parquet', index=False)
    size_parquet = os.path.getsize(f'{output_dir}/kb_backup.parquet') / 1024

    csv_cols = [c for c in kb_df.columns if c not in ['embedding']]
    kb_df[csv_cols].to_csv(f'{output_dir}/kb_export.csv', index=False)
    size_csv = os.path.getsize(f'{output_dir}/kb_export.csv') / 1024

    jsonl_data = kb_df[csv_cols].to_dict(orient='records')
    with open(f'{output_dir}/kb_chunks.jsonl', 'w') as f:
        for record in jsonl_data:
            f.write(pd.json.dumps(record, ensure_ascii=False) + '\n')

    print(f"导出完成:")
    print(f"  Parquet: {size_parquet:.1f} KB")
    print(f"  CSV:     {size_csv:.1f} KB")

    return {'parquet': f'{output_dir}/kb_backup.parquet',
            'csv': f'{output_dir}/kb_export.csv'}
```
