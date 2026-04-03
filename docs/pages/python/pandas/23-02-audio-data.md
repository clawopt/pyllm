---
title: 音频数据在 Pandas 中的处理
description: 音频元数据 / 特征提取 / Whisper 转录文本 / ASR 数据集管理
---
# 音频数据管理


## LLM 场景中的音频数据

- **ASR（自动语音识别）** 训练/评估
- **TTS（文本转语音）** 输出质量检查
- **多模态模型** 音频-文本对齐数据
- **会议记录分析** 语音转文字后的处理

## 音频元数据 DataFrame

```python
import pandas as pd
import numpy as np

class AudioMetadataManager:
    """音频文件元数据管理器"""

    def __init__(self):
        self.meta = pd.DataFrame(columns=[
            'audio_id', 'filepath', 'filename',
            'duration_sec', 'sample_rate', 'channels',
            'format', 'bit_depth', 'file_size_kb',
            'category', 'language', 'speaker',
            'has_transcript', 'transcript_len',
            'is_valid', 'error_msg',
        ])

    def add_from_directory(self, directory, extensions=None,
                             use_ffmpeg=True):
        """扫描音频目录"""
        if extensions is None:
            extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']

        import os
        from pathlib import Path

        path = Path(directory)
        audio_files = []
        for ext in extensions:
            audio_files.extend(path.rglob(f'*{ext}'))
            audio_files.extend(path.rglob(f'*{ext.upper()}'))

        records = []
        for af in sorted(audio_files):
            info = self._probe_audio(af, use_ffmpeg)
            records.append(info)

        if records:
            new_df = pd.DataFrame(records)
            self.meta = pd.concat([self.meta, new_df], ignore_index=True)

        return {
            'total': len(records),
            'valid': sum(1 for r in records if r.get('is_valid', False)),
            'total_duration_sec': sum(r.get('duration_sec', 0) for r in records),
            'total_size_kb': sum(r.get('file_size_kb', 0) for r in records),
        }

    def _probe_audio(self, path, use_ffmpeg=True):
        """探测音频文件信息"""
        record = {
            'audio_id': f"audio_{len(self.meta):05d}",
            'filepath': str(path),
            'filename': path.name,
            'extension': path.suffix.lower(),
            'is_valid': True,
            'error_msg': '',
        }

        try:
            if use_ffmpeg:
                import subprocess
                result = subprocess.run(
                    ['ffprobe', '-v', 'quiet', '-print_format', 'json',
                     '-show_format', '-show_streams', str(path)],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    import json
                    probe = json.loads(result.stdout)

                    fmt = probe.get('format', {})
                    streams = [s for s in probe.get('streams', [])
                               if s.get('codec_type') == 'audio']

                    if streams:
                        s = streams[0]
                        record.update({
                            'duration_sec': round(float(fmt.get('duration', 0)), 2),
                            'sample_rate': int(s.get('sample_rate', 0)),
                            'channels': int(s.get('channels', 0)),
                            'bit_depth': int(s.get('bits_per_sample', 0)),
                            'format': fmt.get('format_name', path.suffix[1:]),
                        })
                    else:
                        record['is_valid'] = False
                        record['error_msg'] = '无音频流'
                else:
                    record['is_valid'] = False
                    record['error_msg'] = f'ffprobe 错误 (code {result.returncode})'
            else:
                stat = path.stat()
                record['file_size_kb'] = round(stat.st_size / 1024, 1)
                record['format'] = path.suffix[1:]
                record['duration_sec'] = None

            record['file_size_kb'] = round(
                record.get('file_size_kb') or (path.stat().st_size / 1024), 1
            )

        except Exception as e:
            record['is_valid'] = False
            record['error_msg'] = str(e)[:100]

        return record

    def summary(self):
        """统计摘要"""
        valid = self.meta[self.meta['is_valid']]
        print("=== 音频库概览 ===")
        print(f"总文件数:     {len(self.meta)}")
        print(f"有效文件:     {len(valid)}")
        print(f"总时长:       {valid['duration_sec'].sum():.0f} 秒 "
              f"({valid['duration_sec'].sum()/3600:.1f} 小时)")
        print(f"总大小:       {valid['file_size_kb'].sum():.0f} KB")

        if len(valid) > 0:
            print(f"\n格式分布:")
            print(valid['format'].value_counts().to_string())
            print(f"\n采样率分布:")
            print(valid['sample_rate'].value_counts().to_string())

        return valid


audio_mgr = AudioMetadataManager()
result = audio_mgr.add_from_directory('/tmp/audio')
print(result)
if result and result.get('total', 0) > 0:
    audio_mgr.summary()
```

## Whisper 集成：批量转写

```python
import pandas as pd
import numpy as np


class BatchTranscriber:
    """批量音频转录管理器"""

    def __init__(self, model_size='base'):
        self.model_size = model_size
        self.results = pd.DataFrame(columns=[
            'audio_id', 'filepath', 'transcript',
            'duration_sec', 'word_count',
            'language_detected', 'confidence',
            'processing_time_s', 'status',
        ])
        self._whisper_model = None

    def _load_whisper(self):
        try:
            import whisper
            self._whisper_model = whisper.load_model(self.model_size)
            return True
        except ImportError:
            return False

    def transcribe_batch(self, meta_df, filepath_col='filepath'):
        """批量转录"""
        if not self._load_whisper():
            return {'success': False, 'error': '需要安装: pip install openai-whisper'}

        records = []
        for idx, row in meta_df.iterrows():
            fpath = row[filepath_col]
            start_time = time.time()

            try:
                result = self._whisper_model.transcribe(str(fpath))
                transcript = result["text"].strip()
                language = result.get("language", "unknown")

                records.append({
                    'audio_id': row.get('audio_id', f'aud_{idx}'),
                    'filepath': fpath,
                    'transcript': transcript,
                    'duration_sec': round(result.get('duration', 0), 2),
                    'word_count': len(transcript.split()),
                    'language_detected': language,
                    'confidence': round(result.get('avg_logprob', 0), 4),
                    'processing_time_s': round(time.time() - start_time, 2),
                    'status': 'success',
                })

            except Exception as e:
                records.append({
                    'audio_id': row.get('audio_id', f'aud_{idx}'),
                    'filepath': fpath,
                    'transcript': '',
                    'status': f'failed: {str(e)[:50]}',
                })

        new_df = pd.DataFrame(records)
        self.results = pd.concat([self.results, new_df], ignore_index=True)

        success = (self.results['status'] == 'success').sum()
        total = len(self.results)

        return {
            'success': True,
            'processed': total,
            'successful': int(success),
            'failed': total - int(success),
            'total_transcript_chars': int(
                self.results.loc[self.results['status']=='success', 'transcript']
                .str.len().sum()
            ),
        }


import time

transcriber = BatchTranscriber(model_size='base')
```
