---
title: 多模态数据统一管理
description: 图文/音视频/文本混合数据集的元数据管理、跨模态关联分析
---
# 统一管理：多模态数据集

现代 LLM 项目经常需要同时处理多种模态的数据——图文对用于 VLM 训练，音频+文本用于语音模型。Pandas 可以用统一的元数据表来管理所有模态。

```python
import pandas as pd
import numpy as np

n = 2000
multi_df = pd.DataFrame({
    'sample_id': [f'multi_{i:05d}' for i in range(n)],
    'modality': np.random.choice(['image-text', 'audio-text', 'video', 'text-only'], n),
    'image_path': [f'/data/img/{i}.jpg' if np.random.random() > 0.3 else None for i in range(n)],
    'audio_path': [f'/data/aud/{i}.wav' if np.random.random() > 0.5 else None for i in range(n)],
    'text': [f'Sample text {i}' for i in range(n)],
    'language': np.random.choice(['zh', 'en', 'mixed'], n),
})

has_image = multi_df['image_path'].notna().sum()
has_audio = multi_df['audio_path'].notna().sum()
has_both = (multi_df['image_path'].notna() & multi_df['audio_path'].notna()).sum()

print(f"总样本: {n:,}")
print(f"有图像: {has_image:,} | 有音频: {has_audio:,} | 两者都有: {has_both:,}")
print(f"\n按模态分布:\n{multi_df['modality'].value_counts()}")

cross = pd.crosstab(multi_df['image_path'].notna(),
                     multi_df['audio_path'].notna(),
                     margins=True)
cross.index = ['无图', '有图', '合计']
cross.columns = ['无音频', '有音频', '合计']
print(f"\n模态交叉表:\n{cross}")
```
