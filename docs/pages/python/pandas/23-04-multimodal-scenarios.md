---
title: 多模态实战场景
description: VLM 训练数据集构建、多模态 RAG 知识库管理、跨模态检索结果分析
---
# 多模态实战：VLM 数据集构建

```python
import pandas as pd
import numpy as np

n = 5000
vlm_data = pd.DataFrame({
    'image_id': [f'img_{i:06d}' for i in range(n)],
    'image_path': [f'/data/vlm/{i%2000}.jpg' for i in range(n)],
    'caption_zh': [f'这是一张关于主题{i%100}的图片' for i in range(n)],
    'caption_en': [f'A photo about topic {i%100}' for i in range(n)],
    'category': np.random.choice(['nature', 'tech', 'food', 'people', 'object'], n),
    'split': np.random.choice(['train']*80 + ['val']*10 + ['test']*10, n),
})

print("=== VLM 数据集概览 ===")
print(f"总样本: {n:,}")
print(f"\n划分分布:\n{vlm_data['split'].value_counts()}")
print(f"\n类别分布:\n{vlm_data['category'].value_counts()}")

train = vlm_data[vlm_data['split'] == 'train']
val = vlm_data[vlm_data['split'] == 'val']
print(f"\n训练/验证/测试 = {len(train)}:{len(val)}:{len(vlm_data[vlm_data['split']=='test'])}")

avg_caption_len = vlm_data['caption_en'].str.len().mean()
print(f"英文 caption 平均长度: {avg_caption_len:.0f} 字符")
```
