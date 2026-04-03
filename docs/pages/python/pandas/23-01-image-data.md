---
title: 图像数据与 Pandas
description: 图像元数据管理、EXIF 信息提取、VLM 训练数据构建、图像质量检查
---
# 图像数据：Pandas 管理视觉数据的元数据

Pandas 不能直接处理图像像素（那是 PIL/OpenCV/Numpy 的活），但它在**图像数据集的管理和分析**中扮演着关键角色——存储文件路径、尺寸、格式、标注信息等元数据，以及做数据集级别的统计分析。

## VLM 训练数据管理

```python
import pandas as pd
import numpy as np

n = 5000
img_df = pd.DataFrame({
    'image_id': [f'img_{i:06d}' for i in range(n)],
    'file_path': [f'/data/images/{i%1000}.jpg' for i in range(n)],
    'width': np.random.randint(256, 1024, n),
    'height': np.random.randint(256, 768, n),
    'caption': [f'Photo of scene {i%100}' for i in range(n)],
    'source': np.random.choice(['LAION', 'COCO', 'custom', 'web'], n),
})

img_df['aspect_ratio'] = (img_df['width'] / img_df['height']).round(2)
img_df['megapixels'] = (img_df['width'] * img_df['height'] / 1e6).round(2)
img_df['resolution'] = img_df['width'].astype(str) + 'x' + img_df['height'].astype(str)

print(f"总样本: {n:,}")
print(f"\n分辨率分布:\n{img_df['resolution'].value_counts().head()}")
print(f"\n宽高比统计:")
print(img_df['aspect_ratio'].describe())
```

## 数据质量检查

```python
def image_health_check(df):
    issues = []
    
    extreme_aspect = df[(df['aspect_ratio'] > 3) | (df['aspect_ratio'] < 0.33)]
    if len(extreme_aspect) > 0:
        issues.append(f"极端宽高比: {len(extreme_aspect)} 张")
    
    tiny = df[df['megapixels'] < 0.05]
    if len(tiny) > 0:
        issues.append(f"过小图片 (<50KB): {len(tiny)} 张")
    
    source_counts = df['source'].value_counts()
    if source_counts.min() < len(df) * 0.01:
        rare = source_counts[source_counts < len(df) * 0.01]
        issues.append(f"稀少来源 (<1%): {dict(rare)}")
    
    return issues

issues = image_health_check(img_df)
for issue in issues:
    print(f"⚠️ {issue}")
if not issues:
    print("✅ 图像数据集健康")
```
