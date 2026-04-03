---
title: 多模态数据处理实战
description: VLM 数据集构建 / 多模态 RAG / 音频日志分析 / 图像质量评估
---
# 多模态场景实战


## 实战一：VLM 训练数据集构建

```python
import pandas as pd
import numpy as np
from datetime import datetime


class VLMDataBuilder:
    """VLM (Vision-Language Model) 训练数据集构建器"""

    def __init__(self):
        self.pairs = pd.DataFrame(columns=[
            'pair_id', 'image_path', 'caption', 'source',
            'image_width', 'image_height', 'caption_len',
            'has_complex_structure', 'is_valid',
            'quality_tier', 'split',
        ])

    def add_pairs(self, image_paths, captions, source='mixed',
                   validate_images=True):
        """批量添加图文对"""
        records = []
        for i, (img_path, caption) in enumerate(zip(image_paths, captions)):
            record = {
                'pair_id': f'vlm_{len(self.pairs)+i+1:06d}',
                'image_path': img_path,
                'caption': caption,
                'source': source,
                'image_width': 0, 'image_height': 0,
                'caption_len': len(caption),
                'has_complex_structure': int(
                    len(caption) > 50 and
                    any(kw in caption.lower() for kw in
                        ['因为', '由于', '包含', '由...组成', '分为', '首先', '其次'])
                ),
                'is_valid': True,
                'quality_tier': None,
                'split': None,
            }

            if validate_images:
                try:
                    from PIL import Image
                    with Image.open(img_path) as img:
                        record['image_width'] = img.width
                        record['image_height'] = img.height
                        if img.width < 32 or img.height < 32:
                            record['is_valid'] = False
                except:
                    record['is_valid'] = False

            if record['is_valid']:
                score = 0
                if 20 <= record['caption_len'] <= 200:
                    score += 2
                elif 200 < record['caption_len'] <= 500:
                    score += 3
                elif record['caption_len'] > 10:
                    score += 1

                if record['has_complex_structure']:
                    score += 2

                ratio_ok = 0.5 <= (record['image_width'] /
                              max(record['image_height'], 1)) <= 2.0
                if ratio_ok and record.get('image_width', 0) > 0:
                    score += 1

                total = 8
                pct = score / total
                if pct >= 0.75:
                    record['quality_tier'] = 'A'
                elif pct >= 0.5:
                    record['quality_tier'] = 'B'
                else:
                    record['quality_tier'] = 'C'

            records.append(record)

        new_df = pd.DataFrame(records)
        self.pairs = pd.concat([self.pairs, new_df], ignore_index=True)
        return {
            'added': len(records),
            'total': len(self.pairs),
            'valid': sum(1 for r in records if r.get('is_valid')),
        }

    def auto_split(self, train=0.7, val=0.15, seed=42):
        """自动划分数据集"""
        valid = self.pairs[self.pairs['is_valid']].copy()
        n = len(valid)

        rng = np.random.RandomState(seed)
        shuffled = valid.sample(frac=1, random_state=rng).reset_index(drop=True)

        t_end = int(n * train)
        v_end = t_end + int(n * val)

        shuffled.loc[:t_end, 'split'] = 'train'
        shuffled.loc[t_end:v_end, 'split'] = 'val'
        shuffled.loc[v_end:, 'split'] = 'test'

        self.pairs.loc[shuffled.index] = shuffled
        return {
            'train': int((shuffled['split']=='train').sum()),
            'val': int((shuffled['split']=='val').sum()),
            'test': int((shuffled['split']=='test').sum()),
        }

    def export_jsonl(self, output_path):
        """导出为 VLM 训练格式 JSONL"""
        valid = self.pairs[self.pairs['is_valid']].copy()

        with open(output_path, 'w') as f:
            for _, row in valid.iterrows():
                entry = {
                    'id': row['pair_id'],
                    'image': row['image_path'],
                    'conversations': [
                        {'role': 'user', 'value': '描述这张图片'},
                        {'role': 'assistant', 'value': row['caption']},
                    ],
                    'metadata': {
                        'source': row['source'],
                        'tier': str(row['quality_tier']),
                        'split': row['split'],
                    }
                }
                f.write(pd.json.dumps(entry, ensure_ascii=False) + '\n')

        print(f"✓ 导出 {len(valid)} 条到 {output_path}")
        return output_path

    def report(self):
        """生成报告"""
        print("=" * 55)
        print("VLM 数据集报告")
        print("=" * 55)

        valid = self.pairs[self.pairs['is_valid']]
        total = len(self.pairs)

        print(f"\n总图文对:     {total}")
        print(f"有效对:       {len(valid)} ({len(valid)/max(total,1)*100:.1f}%)")

        if len(valid) > 0:
            print(f"\n质量分布:")
            tier_dist = valid['quality_tier'].value_counts()
            for t in ['A', 'B', 'C']:
                count = tier_dist.get(t, 0)
                bar = '█' * count
                print(f"  {t}: {bar} ({count})")

            print(f"\n数据集划分:")
            split_dist = valid['split'].value_counts() if 'split' in valid.columns \
                       else pd.Series({'未划分': len(valid)})
            for s in ['train', 'val', 'test']:
                c = split_dist.get(s, 0)
                print(f"  {s}: {c} ({c/len(valid)*100:.1f}%)")

            print(f"\nCaption 长度统计:")
            print(f"  平均: {valid['caption_len'].mean():.0f}")
            print(f"  中位数: {valid['caption_len'].median():.0f}")
            print(f"  最短: {valid['caption_len'].min()}")
            print(f"  最长: {valid['caption_len'].max()}")

        return valid


builder = VLMDataBuilder()

images = [f'/data/vlm/img_{i}.jpg' for i in range(30)]
captions = [
    f'这是一张关于机器学习的图片，图中展示了{["神经网络", "决策树", "聚类算法"][i%3]}的原理图',
    * 30
]

result = builder.add_pairs(images, captions, source='synthetic')
builder.auto_split()
builder.report()
```

## 实战二：多模态 RAG 知识库

```python
class MultimodalRAG:
    """多模态 RAG 系统"""

    def __init__(self):
        self.knowledge = MultimodalDataFrame()

    def add_knowledge(self, item_id, modality, content, metadata=None):
        if modality == 'text':
            self.knowledge.add_text(item_id, content, **(metadata or {}))
        elif modality == 'image':
            self.knowledge.add_image(item_id, content, **(metadata or {}))
        elif modality == 'audio':
            self.knowledge.add_audio(item_id, content, **(metadata or {}))

    def retrieve(self, query_text, top_k=5):
        """多模态检索"""
        text_results = self.knowledge.query(
            modality='text',
            text_query=query_text,
        )
        image_results = self.knowledge.query(modality='image')

        combined = pd.concat([text_results, image_results],
                           ignore_index=True)

        if len(combined) == 0:
            return []

        keywords = query_text.lower().split()

        def relevance_score(row):
            score = 0
            searchable = str(row.get('content', '')).lower() + \
                         str(row.get('source_path', '')).lower()
            for kw in keywords[:5]:
                if kw in searchable:
                    score += 3
            return score

        combined['_relevance'] = combined.apply(relevance_score, axis=1)
        result = combined.nlargest(top_k, '_relevance').drop(columns=['_relevance'])

        return result


rag = MultimodalRAG()
rag.add_knowledge('k001', 'text', '注意力机制计算 Query、Key、Value 的相似度矩阵')
rag.add_knowledge('k002', 'text', 'Transformer 的 Encoder-Decoder 架构详解')
rag.add_knowledge('k003', 'image', '/data/architecture.png',
           caption='模型架构对比图', tags=['architecture'])
rag.add_knowledge('k004', 'audio', '/data/podcast.mp3',
           transcript='本期播客讨论了大模型的最新进展')

results = rag.retrieve('Transformer 注意力机制')
print(f"检索到 {len(results)} 条结果")
```
