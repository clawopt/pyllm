---
title: 多模态数据统一管理框架
description: 图像/音频/文本统一存储 / 多模态 DataFrame 设计 / 跨模态关联查询 / 导出为训练格式
---
# 多模态数据统一框架


## 统一多模态 Schema

```python
import pandas as pd
import numpy as np
from datetime import datetime


class MultimodalDataFrame:
    """多模态统一数据管理器"""

    SCHEMA = {
        'item_id': 'str',
        'modality': 'category',  # image / audio / text / video / mixed
        'source_path': 'str',
        'content': 'object',        # 文本内容 / 图片路径 / 音频路径
        'embedding': 'object',      # 向量表示 (可选)
        'metadata': 'object',       # 额外 JSON 信息
        'quality_score': 'float32',
        'created_at': 'datetime64[ns]',
        'tags': 'object',
        'is_active': 'bool',
    }

    MODALITIES = ['image', 'audio', 'text', 'video', 'mixed']

    def __init__(self):
        self.df = self._create_empty()

    def _create_empty(self):
        data = {}
        for col, dtype in self.SCHEMA.items():
            if dtype == 'category':
                data[col] = pd.Categorical(categories=self.MODALITIES)
            elif dtype == 'bool':
                data[col] = pd.Series(dtype='boolean')
            elif dtype == 'object':
                data[col] = pd.Series(dtype='object')
            elif 'datetime' in dtype:
                data[col] = pd.Series(dtype=dtype)
            else:
                data[col] = pd.Series(dtype=dtype)
        return pd.DataFrame(data)

    def add_image(self, item_id, filepath, caption='', tags=None,
                    quality=1.0):
        return self._add_item(item_id, 'image', filepath,
                            {'caption': caption}, tags, quality)

    def add_audio(self, item_id, filepath, transcript='',
                     language='zh', tags=None, quality=1.0):
        return self._add_item(item_id, 'audio', filepath,
                            {'transcript': transcript, 'language': language},
                            tags, quality)

    def add_text(self, item_id, content, source='',
                  tags=None, quality=1.0):
        return self._add_item(item_id, 'text', content,
                            {'source': source}, tags, quality)

    def _add_item(self, item_id, modality, content_obj,
                   metadata, tags, quality):
        record = {
            'item_id': item_id,
            'modality': modality,
            'source_path': content_obj if modality in ['image', 'audio']
                        else str(content_obj),
            'content': content_obj,
            'embedding': None,
            'metadata': metadata or {},
            'quality_score': round(max(0, min(1, float(quality))), 4),
            'created_at': datetime.now(),
            'tags': tags or [],
            'is_active': True,
        }
        new_df = pd.DataFrame([record])
        self.df = pd.concat([self.df, new_df], ignore_index=True)
        return record

    def query(self, modality=None, tags=None, min_quality=0,
              text_query=None):
        """跨模态查询"""
        df = self.df[self.df['is_active']].copy()

        if modality:
            df = df[df['modality'] == modality]

        if min_quality > 0:
            df = df[df['quality_score'] >= min_quality]

        if tags:
            mask = df['tags'].apply(
                lambda t: any(tag in t for tag in tags)
            )
            df = df[mask]

        if text_query and len(df) > 0:
            searchable_cols = [c for c in df.columns
                             if c in ['source_path', 'content']]
            combined = (
                df[searchable_cols].astype(str)
                .agg(' '.join, axis=1).str.lower()
            )
            keywords = text_query.lower().split()
            for kw in keywords:
                combined = combined[combined.str.contains(kw, na=False)]
                df = df[combined]

        return df.reset_index(drop=True)

    def get_stats(self):
        """统计摘要"""
        active = self.df[self.df['is_active']]
        print("=== 多模态数据概览 ===")
        print(f"总条目数:     {len(active)}")
        print(f"模态分布:")
        print(active['modality'].value_counts().to_string())
        print(f"平均质量分:  {active['quality_score'].mean():.3f}")
        print(f"高质量(>0.8): {(active['quality_score']>0.8).sum()}")
        return active

    def export_for_training(self, output_format='jsonl',
                           modalities=['image', 'text']):
        """导出为训练格式"""
        df = self.query(modalities=modalities if isinstance(modalities, list) else [modalities])

        records = []
        for _, row in df.iterrows():
            record = {}
            if row['modality'] == 'image':
                record = {
                    'id': row['item_id'],
                    'image': row['source_path'],
                    'caption': row.get('metadata', {}).get('caption', ''),
                    **{k: v for k, v in row.get('metadata', {}).items()
                       if k != 'caption'},
                }
            elif row['modality'] == 'text':
                record = {
                    'id': row['item_id'],
                    'text': row['content'],
                    'source': row.get('metadata', {}).get('source', ''),
                    **row.get('metadata', {}),
                }
            elif row['modality'] == 'audio':
                record = {
                    'id': row['item_id'],
                    'audio': row['source_path'],
                    'transcript': row.get('metadata', {}).get('transcript', ''),
                    'language': row.get('metadata', {}).get('language', ''),
                }
            records.append(record)

        export_df = pd.DataFrame(records)
        return export_df


mm_df = MultimodalDataFrame()

mm_df.add_text('t_001', 'Transformer 是基于自注意力机制的架构', source='wiki')
mm_df.add_text('t_002', 'RAG 通过检索增强生成质量', source='paper')
mm_df.add_image('img_001', '/data/images/arch.png', caption='模型架构图')
mm_df.add_audio('aud_001', '/data/audio/interview_01.mp3',
               transcript='今天我们讨论了 LLM 的未来发展方向')
mm_df.add_text('t_003', 'LoRA 是参数高效的微调方法')

mm_df.get_stats()

results = mm_df.query(text_query='注意力')
print(f"\n搜索结果: {len(results)} 条")
for _, r in results.head(5).iterrows():
    print(f"  [{r['modality']:>5s}] {r['item_id']} | "
          f"{str(r['content'])[:50]}")
```
