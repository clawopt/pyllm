---
title: 端到端 RAG 管道
description: 从原始文档到可检索知识库的完整流水线、质量检查点、自动化输出
---
# 端到端管道

```python
import pandas as pd

class RAGPipeline:
    def __init__(self):
        self.stages = []
    
    def run(self, docs_df):
        n = len(docs_df)
        print(f"\n{'='*50}")
        print(f"RAG 管道启动 | {n:,} 份文档")
        print(f"{'='*50}")
        
        chunks = self._chunk(docs_df)
        meta = self._enrich(chunks)
        clean = self._clean(meta)
        output = self._export(clean)
        
        return output
    
    def _chunk(self, df):
        chunks = []
        for _, row in df.iterrows():
            n_chunks = max(1, int(row.get('size_kb', 100) / 50))
            for i in range(n_chunks):
                chunks.append({
                    'doc_id': row['doc_id'],
                    'chunk_index': i,
                    'text': f"[{row['title']}] chunk {i}",
                })
        result = pd.DataFrame(chunks)
        self.stages.append(f"分块: {len(df)} → {len(result)}")
        return result
    
    def _enrich(self, df):
        df['token_count'] = df['text'].str.len() // 4
        self.stages.append(f"元数据丰富: {len(df)} 行")
        return df
    
    def _clean(self, df):
        df = df[df['token_count'].between(20, 2000)].copy()
        self.stages.append(f"清洗: → {len(df)} 行")
        return df
    
    def _export(self, df):
        df.to_parquet('rag_knowledge_base.parquet', index=False)
        self.stages.append(f"导出: rag_knowledge_base.parquet ({len(df)} chunks)")
        for s in self.stages:
            print(f"  ✓ {s}")
        return df

pipeline = RAGPipeline()
result = pipeline.run(docs)
```
