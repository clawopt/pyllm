---
title: 图像数据在 Pandas 中的处理
description: PIL/Pillow 集成 / 图像元数据管理 / 批量图像处理 / 特征提取与存储
---
# 图像数据管理


## 为什么 LLM 开发需要处理图像

- **多模态 LLM**（GPT-4o、Gemini、Claude）支持图像输入
- **VLM（视觉语言模型）** 训练需要图文对齐的数据集
- **RAG 系统中** 图片作为知识库的一部分
- **SFT 数据清洗** 需要验证图片是否有效

## 基础：图像元数据 DataFrame

```python
import pandas as pd
import numpy as np
from pathlib import Path

class ImageMetadataManager:
    """图像元数据管理器"""

    def __init__(self):
        self.meta = pd.DataFrame(columns=[
            'image_id', 'filepath', 'filename', 'extension',
            'width', 'height', 'channels',
            'file_size_kb', 'format', 'mode',
            'has_exif', 'category', 'caption',
            'is_valid', 'error_msg',
        ])

    def scan_directory(self, directory, extensions=None):
        """扫描目录，收集图像元数据"""
        from PIL import Image

        if extensions is None:
            extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}

        path = Path(directory)
        if not path.exists():
            return {'success': False, 'error': f'目录不存在: {directory}'}

        image_files = []
        for ext in extensions:
            image_files.extend(path.rglob(f'*{ext}'))
            image_files.extend(path.rglob(f'*{ext.upper()}'))

        records = []
        for img_path in image_files:
            try:
                with Image.open(img_path) as img:
                    record = {
                        'image_id': f"img_{len(records):05d}",
                        'filepath': str(img_path),
                        'filename': img_path.name,
                        'extension': img_path.suffix.lower(),
                        'width': img.width,
                        'height': img.height,
                        'channels': len(img.getbands()),
                        'file_size_kb': round(img_path.stat().st_size / 1024, 1),
                        'format': img.format or 'unknown',
                        'mode': img.mode,
                        'has_exif': hasattr(img, '_getexif'),
                        'category': self._guess_category(img_path),
                        'caption': '',
                        'is_valid': True,
                        'error_msg': '',
                    }
                records.append(record)
            except Exception as e:
                records.append({
                    'image_id': f"img_{len(records):05d}",
                    'filepath': str(img_path),
                    'filename': img_path.name,
                    'is_valid': False,
                    'error_msg': str(e)[:100],
                })

        if records:
            new_df = pd.DataFrame(records)
            self.meta = pd.concat([self.meta, new_df], ignore_index=True)

        valid = self.meta['is_valid'].sum()
        invalid = (~self.meta['is_valid']).sum()

        return {
            'success': True,
            'total_scanned': len(records),
            'valid_images': int(valid),
            'invalid_images': int(invalid),
            'total_size_kb': round(self.meta['file_size_kb'].sum(), 1) if valid > 0 else 0,
        }

    @staticmethod
    def _guess_category(path):
        parent = path.parent.name.lower()
        if any(k in parent for k in ['train', 'val', 'test']):
            return 'dataset'
        elif any(k in parent for k in ['screenshot', 'photo', 'pic']):
            return 'photograph'
        elif any(k in parent for k in ['chart', 'plot', 'graph']):
            return 'chart'
        elif any(k in parent for k in ['doc', 'pdf', 'scan']):
            return 'document'
        elif any(k in parent for k in ['icon', 'logo', 'avatar']):
            return 'icon'
        return 'other'

    def summary(self):
        """统计摘要"""
        if len(self.meta) == 0:
            return "无数据"

        valid = self.meta[self.meta['is_valid'] == True]
        print("=== 图像库概览 ===")
        print(f"总图片数:     {len(self.meta)}")
        print(f"有效图片:     {len(valid)}")
        print(f"无效图片:     {(~self.meta['is_valid']).sum()}")
        print(f"总大小:       {self.meta['file_size_kb'].sum():.0f} KB")

        if len(valid) > 0:
            print(f"\n尺寸分布:")
            size_dist = valid.groupby(['width', 'height']).size().reset_index(name='count')
            print(size_dist.nlargest(5, 'count').to_string(index=False))

            print(f"\n格式分布:")
            print(valid['format'].value_counts().to_string())

            print(f"\n分类分布:")
            print(valid['category'].value_counts().to_string())

        return valid


mgr = ImageMetadataManager()
result = mgr.scan_directory('/tmp/images')  # 替换为实际路径
print(result)

if result.get('success'):
    mgr.summary()
```

## 批量图像特征提取

```python
import pandas as pd
import numpy as np
from PIL import Image
import hashlib


class ImageFeatureExtractor:
    """图像特征提取器"""

    @staticmethod
    def extract_basic_features(image_path):
        """提取基础特征（不依赖 ML 模型）"""
        try:
            with Image.open(image_path) as img:
                features = {
                    'aspect_ratio': round(img.width / max(img.height, 1), 4),
                    'pixel_count': img.width * img.height,
                    'megapixels': round((img.width * img.height) / 1e6, 3),
                    'is_landscape': img.width > img.height,
                    'size_category': ImageFeatureExtractor._categorize_size(
                        img.width, img.height
                    ),
                    'color_mode': img.mode,
                    'has_alpha': img.mode == 'RGBA',
                    'file_hash': ImageFeatureExtractor._hash_file(image_path),
                }
                return features
        except Exception as e:
            return {'error': str(e)}

    @staticmethod
    def extract_color_stats(image_path):
        """颜色统计特征"""
        try:
            with Image.open(image_path).convert('RGB') as img:
                arr = np.array(img)
                return {
                    'mean_r': round(arr[:,:,0].mean(), 1),
                    'mean_g': round(arr[:,:,1].mean(), 1),
                    'mean_b': round(arr[:,:,2].mean(), 1),
                    'brightness': round(np.mean(arr), 1),
                    'std_dev': round(np.std(arr), 1),
                    'is_grayscale': bool(
                        (arr[:,:,0] == arr[:,:,1]).all() and
                        (arr[:,:,1] == arr[:,:,2]).all()
                    ),
                }
        except Exception as e:
            return {'error': str(e)}

    @staticmethod
    def batch_extract(meta_df, filepath_col='filepath'):
        """批量提取特征"""
        features_list = []

        for idx, row in meta_df.iterrows():
            fpath = row[filepath_col]
            basic = ImageFeatureExtractor.extract_basic_features(fpath)
            color = ImageFeatureExtractor.extract_color_stats(fpath)

            combined = {**basic, **color_features}
            combined['image_id'] = row['image_id']
            features_list.append(combined)

        feature_df = pd.DataFrame(features_list)
        return feature_df

    @staticmethod
    def _categorize_size(w, h):
        pixels = w * h
        if pixels < 10000:
            return 'tiny'
        elif pixels < 50000:
            return 'small'
        elif pixels < 200000:
            return 'medium'
        elif pixels < 800000:
            return 'large'
        else:
            return 'xlarge'

    @staticmethod
    def _hash_file(path):
        hasher = hashlib.md5()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hasher.update(chunk)
        return hasher.hexdigest()[:12]


meta_df = pd.DataFrame({
    'image_id': [f'img_{i}' for i in range(20)],
    'filepath': [f'/tmp/images/img_{i}.jpg' for i in range(20)],
})

```

## LLM 场景：VLM 训练数据集准备

```python
import pandas as pd
import numpy as np


class VLMTrainingDataPreparer:
    """VLM (Vision-Language Model) 训练数据准备"""

    def __init__(self):
        self.data = None

    def load_image_text_pairs(self, image_dir, metadata_csv):
        """加载图文对数据"""
        images_meta = ImageMetadataManager()
        scan_result = images_meta.scan_directory(image_dir)

        text_data = pd.read_csv(metadata_csv)
        merged = images_meta.meta.merge(
            text_data,
            on='filename',
            how='inner' if 'text' in text_data.columns else 'left'
        )
        self.data = merged
        return self

    def validate_pairs(self, min_caption_len=10):
        """验证图文对质量"""
        if self.data is None or len(self.data) == 0:
            return "无数据"

        df = self.data.copy()

        valid_img = df['is_valid'] == True

        caption_col = [c for c in df.columns if 'caption' in c.lower() or 'text' in c.lower()]
        if caption_col:
            cap_col = caption_col[0]
            df['_cap_len'] = df[cap_col].str.len()
            valid_cap = df['_cap_len'] >= min_caption_len
        else:
            valid_cap = pd.Series([True] * len(df))

        reasonable_size = (
            (df['width'] >= 32) & (df['height'] >= 32) &
            (df['width'] <= 7680) & (df['height'] <= 4320)
        )

        df['is_pair_valid'] = valid_img & valid_cap & reasonable_size.fill(True)

        stats = {
            'total': len(df),
            'valid_pairs': df['is_pair_valid'].sum(),
            'invalid_image': (~valid_img).sum(),
            'short_caption': (~valid_cap).sum(),
            'bad_size': (~reasonable_size.fill(True)).sum(),
        }

        print("=== VLM 数据验证 ===")
        for k, v in stats.items():
            pct = v/len(df)*100 if len(df) > 0 else 0
            print(f"  {k}: {v} ({pct:.1f}%)")

        self.data = df[df['is_pair_valid']]
        return stats

    def split_for_training(self, train_ratio=0.8, val_ratio=0.1, seed=42):
        """划分训练/验证/测试集"""
        if self.data is None or len(self.data) == 0:
            return "无数据"

        n = len(self.data)
        rng = np.random.RandomState(seed)

        shuffled = self.data.sample(frac=1, random_state=rng).reset_index(drop=True)

        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        splits = {
            'train': shuffled.iloc[:train_end],
            'val': shuffled.iloc[train_end:val_end],
            'test': shuffled.iloc[val_end:],
        }

        print("\n=== 数据集划分 ===")
        for name, split_df in splits.items():
            print(f"  {name}: {len(split_df)} ({len(split_df)/n*100:.1f}%)")

        return splits


preparer = VLMTrainingDataPreparer()
print("VLM 训练数据准备流程已就绪")
```
