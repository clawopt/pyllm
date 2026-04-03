---
title: 知识库管理与维护
description: 增量更新 / 版本管理 / 质量监控 / 去重 / 导出与备份
---
# RAG 知识库管理


## 增量更新知识库

```python
import pandas as pd
import numpy as np
from datetime import datetime

class KnowledgeBaseManager:
    """RAG 知识库管理器"""

    def __init__(self, kb_df=None):
        self.kb = kb_df if kb_df is not None else pd.DataFrame()
        self.version = 1

    def add_chunks(self, chunks_df):
        """批量添加新 chunks"""
        now = datetime.now()

        chunks_df = chunks_df.copy()
        chunks_df['version'] = self.version
        chunks_df['created_at'] = now
        chunks_df['updated_at'] = now
        chunks_df['is_active'] = True
        chunks_df['embedding_status'] = 'pending'

        if 'chunk_id' not in chunks_df.columns:
            base_id = len(self.kb) + 1
            chunks_df.insert(0, 'chunk_id',
                             [f'ch_{base_id+i:06d}' for i in range(len(chunks_df))])

        self.kb = pd.concat([self.kb, chunks_df], ignore_index=True)
        return len(chunks_df)

    def update_chunk_content(self, chunk_id, new_content):
        """更新指定 chunk 的内容"""
        mask = self.kb['chunk_id'] == chunk_id
        if not mask.any():
            raise ValueError(f"Chunk '{chunk_id}' 不存在")

        self.kb.loc[mask, 'content'] = new_content
        self.kb.loc[mask, 'updated_at'] = datetime.now()
        self.kb.loc[mask, 'version'] += 1
        self.kb.loc[mask, 'embedding_status'] = 'pending'
        return True

    def deactivate_by_source(self, source_doc):
        """停用某文档的所有 chunk（软删除）"""
        mask = self.kb['source_doc'] == source_doc
        count = mask.sum()
        self.kb.loc[mask, 'is_active'] = False
        return int(count)

    def get_active_kb(self):
        """获取当前活跃的知识库"""
        return self.kb[self.kb['is_active'] == True].copy()


np.random.seed(42)
kb = KnowledgeBaseManager()

initial = pd.DataFrame({
    'content': [f'文档A 第{i}段内容' for i in range(5)],
    'source_doc': ['doc_a.pdf']*5,
    'source_type': ['paper']*5,
    'token_count': np.random.randint(100, 500, 5),
})
n_added = kb.add_chunks(initial)
print(f"初始导入: {n_added} 个 chunks")

additional = pd.DataFrame({
    'content': [f'文档B 第{i}段内容' for i in range(3)],
    'source_doc': ['doc_b.html']*3,
    'source_type': ['api_docs']*3,
    'token_count': np.random.randint(150, 600, 3),
})
n_added2 = kb.add_chunks(additional)
print(f"追加导入: {n_added2} 个 chunks")

kb.update_chunk_content('ch_000001', '更新后的文档A第1段内容...')
print("✓ 已更新 ch_000001")

deactivated = kb.deactivate_by_source('doc_a.pdf')
print(f"已停用 doc_a.pdf 的 {deactivated} 个 chunks")

active = kb.get_active_kb()
print(f"\n活跃 chunks: {len(active)}")
print(active[['chunk_id', 'source_doc', 'is_active']].head(8))
```

## 知识库去重

```python
import pandas as pd
from difflib import SequenceMatcher

class KBDeduplicator:
    """知识库去重器"""

    @staticmethod
    def exact_dedup(kb_df):
        """精确去重（完全相同的内容）"""
        before = len(kb_df)
        deduped = kb_df.drop_duplicates(subset=['content'], keep='first')
        removed = before - len(deduped)
        print(f"精确去重: {before} → {len(deduped)} (移除 {removed})")
        return deduped

    @staticmethod
    def fuzzy_dedup(kb_df, threshold=0.9):
        """模糊去重（相似度高于阈值的）"""
        before = len(kb_df)
        contents = kb_df['content'].tolist()
        keep_mask = [True] * len(contents)
        seen = []

        for i, content in enumerate(contents):
            if not keep_mask[i]:
                continue
            is_dup = False
            for seen_content in seen:
                ratio = SequenceMatcher(None, content[:200],
                                          seen_content[:200]).ratio()
                if ratio > threshold:
                    keep_mask[i] = False
                    is_dup = True
                    break
            if not is_dup:
                seen.append(content)

        result = kb_df[keep_mask].reset_index(drop=True)
        removed = before - len(result)
        print(f"模糊去重(>{threshold}): {before} → {len(result)} (移除 {removed})")
        return result


test_kb = pd.DataFrame({'content': [
    'Pandas 是 Python 数据分析库',
    'Pandas 是 Python 数据分析库',
    'DataFrame 是二维表格结构',
    'Dataframe 是二维表格结构',
    'groupby 用于分组聚合操作',
    'merge() 实现表连接',
] * 10})

exact_clean = KBDeduplicator.exact_dedup(test_kb)
fuzzy_clean = KBDeduplicator.fuzzy_dedup(test_kb, threshold=0.85)
```

## 知识库质量报告

```python
class KBQualityReporter:
    """知识库质量报告生成器"""

    @staticmethod
    def generate(kb_df):
        df = kb_df.copy()
        total = len(df)
        active = df[df['is_active'] == True]

        report = {
            '总 chunk 数': total,
            '活跃 chunk 数': len(active),
            '覆盖率': f"{len(active)/total*100:.1f}%" if total > 0 else "N/A",
            'Embedding 完成': int((active['embedding_status'] == 'done').sum()),
            'Embedding 待处理': int((active['embedding_status'] == 'pending').sum()),
            'Embedding 失败': int((active['embedding_status'] == 'failed').sum()),
        }

        if 'token_count' in active.columns and len(active) > 0:
            report.update({
                '平均 Token 数': round(active['token_count'].mean(), 0),
                'Token 总计': int(active['token_count'].sum()),
                '最大 Token 数': int(active['token_count'].max()),
                '最小 Token 数': int(active['token_count'].min()),
            })

        if 'source_type' in active.columns:
            report['来源分布'] = dict(
                active['source_type'].value_counts().to_dict()
            )

        if 'quality_score' in active.columns:
            qs = active['quality_score']
            report.update({
                '平均质量分': round(qs.mean(), 3),
                '低质量(<0.5)': int((qs < 0.5).sum()),
                '高质量(>0.8)': int((qs > 0.8).sum()),
            })

        print("=" * 50)
        print("RAG 知识库质量报告")
        print("=" * 50)
        for k, v in report.items():
            if isinstance(v, dict):
                print(f"\n{k}:")
                for sk, sv in v.items():
                    print(f"  {sk}: {sv}")
            else:
                print(f"  {k:<20s}: {v}")

        return report


np.random.seed(42)
sample_kb = pd.DataFrame({
    'chunk_id': [f'ch_{i:04d}' for i in range(50)],
    'content': [f'Content {i}...' for i in range(50)],
    'source_type': pd.Categorical(
        np.random.choice(['wiki', 'api_docs', 'blog', 'paper'], 50)
    ),
    'token_count': np.random.randint(50, 2000, 50),
    'quality_score': np.random.beta(2, 4, 50).round(3),
    'is_active': np.random.choice([True]*45 + [False]*5, 50),
    'embedding_status': np.Categorical(
        np.random.choice(['done', 'done', 'done', 'pending', 'failed'], 50)
    ),
})

KBQualityReporter.generate(sample_kb)
```
