---
title: 音频数据与 Pandas
description: 音频元数据管理、时长/采样率/格式统计、ASR 数据集构建
---
# 音频数据：语音场景的元数据管理

在 ASR（自动语音识别）和 TTS（文本转语音）相关的 LLM 项目中，你需要管理大量音频文件的元数据——时长、采样率、格式、转录文本、说话人信息等。Pandas 是做这件事的完美工具。

```python
import pandas as pd
import numpy as np

n = 3000
audio_df = pd.DataFrame({
    'file_id': [f'audio_{i:05d}' for i in range(n)],
    'path': [f'/data/audio/{i%500}.wav' for i in range(n)],
    'duration_s': np.random.uniform(1, 60, n).round(2),
    'sample_rate': np.random.choice([16000, 22050, 44100], n),
    'channels': np.random.choice([1, 2], n),
    'speaker': [f'sp_{i%50}' for i in range(n)],
    'transcript': [f'Transcript text {i}' for i in range(n)],
})

audio_df['duration_min'] = (audio_df['duration_s'] / 60).round(2)
long_audio = audio_df[audio_df['duration_s'] > 30]
print(f"长音频 (>30s): {len(long_audio)} / {n} ({len(long_audio)/n*100:.1f}%)")

sr_dist = audio_df['sample_rate'].value_counts()
print(f"\n采样率分布:\n{sr_dist}")

avg_duration = audio_df.groupby('speaker')['duration_s'].mean().sort_values(ascending=False)
print(f"\n平均说话时长 Top5:\n{avg_duration.head()}")
```
