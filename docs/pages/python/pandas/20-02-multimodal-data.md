---
title: 多模态数据处理
description: 图像元数据管理、VLM 训练数据构建、图文对数据的 Pandas 处理
---
# 多模态数据：处理图像与文本的混合数据

LLM 的能力已经从纯文本扩展到了多模态——GPT-4o 能看图、Gemini 能理解视频。在训练和评估这些多模态模型时，你需要处理**图文对（image-text pairs）**的数据集。Pandas 虽然不能直接处理图像像素，但非常适合管理这些数据的**元数据**。

## VLM 训练数据管理

```python
import pandas as pd
import numpy as np

n = 10_000
vlm_df = pd.DataFrame({
    'image_id': [f'img_{i:06d}' for i in range(n)],
    'image_path': [f'/data/images/{i%1000}.jpg' for i in range(n)],
    'width': np.random.randint(256, 1024, n),
    'height': np.random.randint(256, 1024, n),
    'caption': [f'A photo showing scene {i%50}' for i in range(n)],
    'source': np.random.choice(['LAION', 'COCO', 'custom'], n),
    'split': np.random.choice(['train', 'val', 'test'], n, p=[0.8, 0.1, 0.1]),
})

print(f"总样本: {n:,}")
print(f"\n按来源分布:\n{vlm_df['source'].value_counts()}")
print(f"\n按划分分布:\n{vlm_df['split'].value_counts()}")

aspect_ratio = vlm_df['width'] / vlm_df['height']
vlm_df['is_landscape'] = aspect_ratio > 1.2
vlm_df['resolution_tier'] = pd.cut(
    aspect_ratio,
    bins=[0, 0.7, 1.3, float('inf')],
    labels=['tall', 'square', 'wide'],
)
```

关键操作：
- `split` 列控制 train/val/test 划分——这是 ML 数据集的标准做法
- `aspect_ratio` 和 `resolution_tier` 用于检测数据集中是否存在某种分辨率偏差
- `image_path` 存的是文件路径而非像素数据——Pandas 管理元数据，实际图像由 DataLoader 按需加载
