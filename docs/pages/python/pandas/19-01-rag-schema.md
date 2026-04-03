---
title: RAG 知识库数据结构设计
description: Chunk 元数据管理 / 向量存储 / 来源追踪 / 版本控制 / 知识图谱基础
---
# RAG 知识库 Schema 设计


## RAG 与 Pandas 的关系

在 RAG（Retrieval-Augmented Generation）系统中，**Pandas 扮演知识库元数据管理者**的角色：

```
RAG 系统架构:
┌──────────────┐    ┌──────────────────┐    ┌─────────────┐
│ 用户查询     │ →  │ 向量数据库(检索)   │ →  │ LLM (生成)   │
└──────────────┘    └────────┬─────────┘    └─────────────┘
                             │
                    Pandas 管理的元数据
                    - chunk_id, content
                    - source, metadata
                    - embedding_status
                    - quality_score, version
```

## Chunk 数据结构设计

```python
import pandas as pd
import numpy as np

class KnowledgeBaseSchema:
    """RAG 知识库标准 Schema"""

    COLUMNS = {
        'chunk_id': 'str',           # 唯一标识符
        'content': 'str',            # 原始文本内容
        'source_doc': 'str',         # 来源文档名
        'source_type': 'category',   # 类型: api/wiki/blog/paper
        'chunk_index': 'int32',      # 文档内序号
        'token_count': 'int32',      # token 数量
        'char_count': 'int32',       # 字符数
        'embedding_status': 'category', # pending/done/failed
        'quality_score': 'float32',   # 质量分数 (0-1)
        'created_at': 'datetime64[ns]',
        'updated_at': 'datetime64[ns]',
        'version': 'int16',
        'is_active': 'bool',
        'tags': 'object',             # 标签列表
        'metadata': 'object',         # 额外 JSON 元数据
    }

    @staticmethod
    def create_empty(n=0):
        data = {col: [] for col in KnowledgeBaseSchema.COLUMNS}
        for col, dtype in KnowledgeBaseSchema.COLUMNS.items():
            if dtype == 'category':
                if col == 'source_type':
                    data[col] = pd.Categorical([], categories=[
                        'api_docs', 'wiki', 'blog', 'paper', 'github',
                        'slack', 'email', 'internal'
                    ])
                elif col == 'embedding_status':
                    data[col] = pd.Categorical([], categories=[
                        'pending', 'done', 'failed', 'skipped'
                    ])
                else:
                    data[col] = pd.Categorical([])
            elif dtype == 'bool':
                data[col] = pd.Series(dtype='boolean')
            elif dtype == 'object':
                data[col] = pd.Series(dtype='object')
            elif 'int' in dtype:
                data[col] = pd.Series(dtype=dtype)
            elif 'float' in dtype:
                data[col] = pd.Series(dtype=dtype)
            elif 'datetime' in dtype:
                data[col] = pd.Series(dtype=dtype)
            else:
                data[col] = pd.Series()

        return pd.DataFrame(data)


kb = KnowledgeBaseSchema.create_empty()
print(f"空知识库 Schema: {len(kb.columns)} 列")
print(kb.dtypes)
```

## 批量导入文档数据

```python
import pandas as pd
import numpy as np
from datetime import datetime

class DocumentIngestor:
    """文档导入器"""

    def __init__(self, kb_df):
        self.kb = kb_df.copy() if kb_df is not None else KnowledgeBaseSchema.create_empty()
        self.next_id = self._get_next_id()

    def _get_next_id(self):
        if len(self.kb) == 0:
            return 1
        max_id = self.kb['chunk_id'].str.extract(r'(\d+)').astype(float).max()
        return int(max_id) + 1 if not np.isnan(max_id) else 1

    def ingest_document(self, doc_name, chunks, source_type='wiki'):
        """导入一篇文档的所有 chunk"""
        records = []
        now = datetime.now()

        for i, chunk_text in enumerate(chunks):
            records.append({
                'chunk_id': f'ch_{self.next_id + i:06d}',
                'content': chunk_text,
                'source_doc': doc_name,
                'source_type': source_type,
                'chunk_index': int(i),
                'token_count': len(chunk_text.split()),
                'char_count': len(chunk_text),
                'embedding_status': 'pending',
                'quality_score': np.random.uniform(0.5, 1.0).round(3),
                'created_at': now,
                'updated_at': now,
                'version': 1,
                'is_active': True,
                'tags': [],
                'metadata': {},
            })

        new_df = pd.DataFrame(records)
        self.kb = pd.concat([self.kb, new_df], ignore_index=True)
        self.next_id += len(chunks)

        return {
            'doc_name': doc_name,
            'chunks_imported': len(chunks),
            'total_chunks': len(self.kb),
        }


ingestor = DocumentIngestor(None)

docs = [
    ('pandas_guide.pdf', [
        'Pandas 是 Python 最流行的数据分析库...',
        'DataFrame 是二维表格结构，由行和列组成...',
        'groupby 操作遵循 Split-Apply-Combine 范式...',
        'merge() 用于类似 SQL JOIN 的表连接操作...',
    ], 'paper'),
    ('api_reference.html', [
        'GPT-4o API 支持多模态输入...',
        'Claude API 提供长上下文窗口达200K tokens...',
        'DeepSeek API 以高性价比著称...',
    ], 'api_docs'),
]

for doc_name, chunks, src_type in docs:
    result = ingestor.ingest_document(doc_name, chunks, src_type)
    print(f"✓ {result}")

print(f"\n知识库总计: {len(ingestor.kb)} 个 chunks")
print(f"来源分布:")
print(ingestor.kb['source_type'].value_counts())
```
