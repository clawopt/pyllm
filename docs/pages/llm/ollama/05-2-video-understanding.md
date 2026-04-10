# 05-2 视频理解

## Ollama 与视频：现状与可行方案

先说一个重要的事实：**Ollama 目前不原生支持视频输入**。你不能像传图片那样把一个 `.mp4` 文件直接丢给模型然后说"分析这段视频"。这是因为视频理解需要额外的模型架构（时序建模、3D 卷积、Video Transformer 等），而 Ollama 底层依赖的 llama.cpp 推理引擎目前只处理图像和文本两种模态。

但这并不意味着你无法用 Ollama 做视频相关的任务。**通过"视频 → 关键帧序列 → 视觉模型批量分析 → 综合理解"的工程化方案**，你可以实现 80% 以上的视频理解需求。这一节将详细讲解这个方案的每一个环节。

## 方案架构总览

```
┌─────────────────────────────────────────────────────────────┐
│              Ollama 视频理解流水线                           │
│                                                             │
│  输入: video.mp4 (30秒, 24fps = 720帧)                      │
│       │                                                     │
│       ▼                                                     │
│  [FFmpeg 帧抽取]                                            │
│  │  策略: 每秒1帧 + 场景变化检测                             │
│  │  输出: frame_001.jpg ~ frame_030.jpg (约30张)            │
│       │                                                     │
│       ▼                                                     │
│  [Ollama 批量视觉分析]                                       │
│  │  模型: minicpm-v / llava                                 │
│  │  每帧生成描述 + 时间戳标注                                │
│  │  输出: [{frame: 5, desc: "一个人走进房间"}, ...]         │
│       │                                                     │
│       ▼                                                     │
│  [时序聚合与综合理解]                                        │
│  │  将逐帧描述送入 LLM 进行时序推理                          │
│  │  识别事件、动作、因果关系、情感变化                       │
│       │                                                     │
│       ▼                                                     │
│  输出: 结构化视频分析报告                                    │
│      - 场景列表                                             │
│      - 事件时间线                                           │
│      - 内容摘要                                             │
│      - 关键帧标注                                           │
└─────────────────────────────────────────────────────────────┘
```

## 第一步：视频帧抽取

FFmpeg 是视频处理的瑞士军刀，我们将用它来实现智能帧抽取：

```python
#!/usr/bin/env python3
"""视频帧抽取工具 - 支持多种策略"""

import subprocess
import json
import os
from pathlib import Path
from datetime import timedelta

class VideoFrameExtractor:
    """基于 FFmpeg 的视频帧提取器"""
    
    def __init__(self, output_dir="./frames"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def get_video_info(self, video_path):
        """获取视频基本信息"""
        
        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_format", "-show_streams",
            video_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        info = json.loads(result.stdout)
        
        video_stream = next(
            (s for s in info["streams"] if s["codec_type"] == "video"),
            None
        )
        
        if not video_stream:
            raise ValueError("未找到视频流")
        
        duration = float(info["format"]["duration"])
        fps_str = video_stream.get("r_frame_rate", "30/1")
        fps_parts = fps_str.split("/")
        fps = int(fps_parts[0]) // int(fps_parts[1]) if len(fps_parts) == 2 else 30
        
        width = video_stream.get("width", 0)
        height = video_stream.get("height", 0)
        
        total_frames = int(duration * fps)
        
        return {
            "duration_sec": round(duration, 2),
            "duration_str": str(timedelta(seconds=int(duration))),
            "fps": fps,
            "resolution": f"{width}x{height}",
            "total_frames": total_frames,
            "file_size_mb": round(int(info["format"]["size"]) / (1024*1024), 1)
        }
    
    def extract_uniform(self, video_path, fps_target=1, max_width=800):
        """
        均匀采样策略：每秒抽取 N 帧
        
        Args:
            video_path: 视频文件路径
            fps_target: 每秒目标抽帧数 (默认1fps)
            max_width: 最大宽度（缩小以节省处理时间）
        """
        
        info = self.get_video_info(video_path)
        expected_frames = int(info["duration_sec"] * fps_target)
        
        print(f"\n📹 视频信息:")
        print(f"   时长: {info['duration_str']} ({info['duration_sec']}s)")
        print(f"   分辨率: {info['resolution']}")
        print(f"   帧率: {info['fps']} fps")
        print(f"   总帧数: {info['total_frames']:,}")
        print(f"   抽取策略: 均匀 {fps_target} fps")
        print(f"   预计输出: ~{expected_frames} 帧")
        
        output_pattern = str(self.output_dir / "frame_%04d.jpg")
        
        cmd = [
            "ffmpeg", "-i", video_path,
            "-vf", f"fps={fps_target},scale={max_width}:-2",
            "-q:v", "2",           # JPEG 质量 (2=高质量, 31=最低)
            "-start_number", "0",
            output_pattern,
            "-y"                   # 覆盖已有文件
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg 错误: {result.stderr}")
        
        # 统计实际输出的帧数
        actual_frames = len(list(self.output_dir.glob("frame_*.jpg")))
        
        return {
            "strategy": "uniform",
            "fps": fps_target,
            "expected_frames": expected_frames,
            "actual_frames": actual_frames,
            "output_dir": str(self.output_dir),
            "frames": sorted(self.output_dir.glob("frame_*.jpg"))
        }
    
    def extract_scene_change(self, video_path, threshold=0.3, 
                              max_width=800):
        """
        场景变化检测策略：只在场景切换时抽取帧
        
        这种方式比均匀抽样更智能——在动作密集的场景中会自动增加密度，
        在静态场景中减少冗余帧。
        """
        
        info = self.get_video_info(video_path)
        
        output_pattern = str(self.output_dir / "scene_%04d.jpg")
        
        cmd = [
            "ffmpeg", "-i", video_path,
            "-vf", f"select='gt(scene,{threshold})',scale={max_width}:-2",
            "-vs", "vfr",          # 可变帧率模式
            "-q:v", "2",
            "-start_number", "0",
            "-frame_pts", "1",     # 使用原始时间戳作为文件名编号
            output_pattern,
            "-y"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg 错误: {result.stderr}")
        
        actual_frames = len(list(self.output_dir.glob("scene_*.jpg")))
        
        return {
            "strategy": "scene_change",
            "threshold": threshold,
            "actual_frames": actual_frames,
            "output_dir": str(self.output_dir),
            "frames": sorted(self.output_dir.glob("scene_*.jpg"))
        }
    
    def extract_keyframes(self, video_path, num_frames=16, max_width=800):
        """
        关键帧策略：固定数量的代表性帧
        
        适用于只需要概览性理解的场景，大幅减少处理量。
        """
        
        info = self.get_video_info(video_path)
        duration = info["duration_sec"]
        
        # 计算等间隔的时间点
        interval = duration / (num_frames + 1)
        
        timestamps = []
        for i in range(1, num_frames + 1):
            ts = i * interval
            timestamps.append(ts)
        
        # 为每个时间点抽取一帧
        frames = []
        for i, ts in enumerate(timestamps):
            output_path = self.output_dir / f"key_{i:04d}.jpg"
            
            cmd = [
                "ffmpeg", "-ss", str(ts),
                "-i", video_path,
                "-vframes", "1",
                "-vf", f"scale={max_width}:-2",
                "-q:v", "2",
                "-y", str(output_path)
            ]
            
            subprocess.run(cmd, capture_output=True, text=True)
            
            if output_path.exists():
                frames.append({
                    "path": output_path,
                    "timestamp": round(ts, 2),
                    "time_str": str(timedelta(seconds=int(ts)))
                })
        
        return {
            "strategy": "keyframes",
            "num_requested": num_frames,
            "actual_frames": len(frames),
            "output_dir": str(self.output_dir),
            "frames": frames
        }


if __name__ == "__main__":
    import sys
    
    video = sys.argv[1] if len(sys.argv) > 1 else "./test_video.mp4"
    extractor = VideoFrameExtractor("./output_frames")
    
    # 先看视频信息
    info = extractor.get_video_info(video)
    print(json.dumps(info, indent=2, ensure_ascii=False))
    
    # 选择策略
    strategy = sys.argv[2] if len(sys.argv) > 2 else "keyframes"
    
    if strategy == "uniform":
        result = extractor.extract_uniform(video, fps_target=1)
    elif strategy == "scene":
        result = extractor.extract_scene_change(video, threshold=0.3)
    else:
        result = extractor.extract_keyframes(video, num_frames=12)
    
    print(f"\n✅ 抽取完成: {result['actual_frames']} 帧")
```

## 第二步：逐帧视觉分析

有了帧图片之后，下一步是让视觉模型逐一分析每一帧：

```python
#!/usr/bin/env python3
"""视频帧批量分析与时序理解"""

import requests
import base64
import json
import time
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/api/chat"
VISION_MODEL = "minicpm-v"

def analyze_frame(image_path, frame_index, timestamp=""):
    """单帧图像分析"""
    
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    
    payload = {
        "model": VISION_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text",
                 "text": f"""这是视频的第 {frame_index} 帧（时间点: {timestamp}）。
请用简洁的语言描述这个画面中的：
1. 主要对象/人物及其状态
2. 正在发生的动作或事件
3. 场景环境（室内/室外、光线等）
4. 文字/文字内容（如果有）

输出格式（严格 JSON）：
{{
  "timestamp": "{timestamp}",
  "objects": ["对象1", "对象2"],
  "action": "正在做什么",
  "environment": "环境描述",
  "text_content": "画面中的文字（无则填null）",
  "notable_changes": "相对于前一帧的变化（第一帧填'初始帧'）"
}}"""},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            ]
        }],
        "stream": False,
        "options": {"temperature": 0.2}
    }
    
    start = time.time()
    resp = requests.post(OLLAMA_URL, json=payload, timeout=60)
    elapsed = time.time() - start
    
    raw = resp.json()["message"]["content"]
    
    try:
        import re
        json_match = re.search(r'\{[\s\S]*\}', raw)
        if json_match:
            data = json.loads(json_match.group())
            data["analysis_time"] = round(elapsed, 1)
            return data
    except:
        pass
    
    return {
        "timestamp": timestamp,
        "raw_description": raw,
        "analysis_time": round(elapsed, 1)
    }


def analyze_video_frames(frame_dir, strategy_output=None):
    """批量分析所有视频帧"""
    
    frame_dir = Path(frame_dir)
    
    if strategy_output and isinstance(strategy_output, dict):
        frames_info = strategy_output.get("frames", [])
    else:
        # 自动发现帧文件
        frames_info = [{"path": f, "timestamp": ""} 
                       for f in sorted(frame_dir.glob("*.jpg"))]
    
    results = []
    total_time = 0
    
    print(f"\n🔍 开始分析 {len(frames_info)} 帧...")
    print(f"{'='*60}\n")
    
    for i, finfo in enumerate(frames_info, 1):
        frame_path = finfo["path"] if isinstance(finfo, dict) else finfo
        timestamp = finfo.get("timestamp", finfo.get("time_str", f"{i}s"))
        
        print(f"[{i}/{len(frames_info)}] 分析: {frame_path.name}", end=" ", flush=True)
        
        try:
            result = analyze_frame(str(frame_path), i, timestamp)
            results.append(result)
            total_time += result.get("analysis_time", 0)
            print(f"✅ ({result.get('analysis_time', '?')}s)")
        except Exception as e:
            print(f"❌ ({e})")
            results.append({"error": str(e)})
    
    avg_time = total_time / len(results) if results else 0
    
    summary = {
        "total_frames": len(results),
        "successful": sum(1 for r in results if "error" not in r),
        "total_time": round(total_time, 1),
        "avg_per_frame": round(avg_time, 1),
        "frame_results": results
    }
    
    print(f"\n{'='*60}")
    print(f"📊 分析完成:")
    print(f"   总帧数: {summary['total_frames']}")
    print(f"   成功: {summary['successful']}")
    print(f"   总耗时: {summary['total_time']}s")
    print(f"   平均每帧: {summary['avg_per_frame']}s")
    
    return summary


def generate_video_summary(frame_analysis_results):
    """基于逐帧分析结果生成视频摘要"""
    
    # 构建时序描述文本
    timeline_text = "以下是视频中各帧的分析结果（按时间顺序）：\n\n"
    
    for i, r in enumerate(frame_analysis_results.get("frame_results", [])):
        if "error" in r:
            continue
        
        timeline_text += f"[帧{i+1}, 时间:{r.get('timestamp','?')}]\n"
        timeline_text += f"- 对象: {', '.join(r.get('objects', []))}\n"
        timeline_text += f"- 动作: {r.get('action', 'N/A')}\n"
        timeline_text += f"- 环境: {r.get('environment', 'N/A')}\n"
        timeline_text += f"- 变化: {r.get('notable_changes', 'N/A')}\n\n"
    
    # 用 LLM 生成综合摘要
    summary_prompt = f"""基于以下视频逐帧分析结果，请生成一份完整的视频理解报告。

{timeline_text}

请按以下格式输出报告：

# 视频分析报告

## 基本信息
- **时长**: [估算]
- **场景类型**: [如：会议演示/户外活动/屏幕录制/...]

## 内容摘要
[2-3段话概括视频的核心内容]

## 事件时间线
| 时间 | 事件 | 详情 |
|------|------|------|
| ... | ... | ... |

## 关键发现
1. [最重要的发现]
2. [次重要的发现]
3. [值得注意的细节]

## 对象追踪
| 对象 | 出现时段 | 行为概述 |
|------|---------|---------|
| ... | ... | ... |

## 结论与建议
[如果这是一个需要评审的视频，给出你的评价和建议]"""
    
    payload = {
        "model": "qwen2.5:7b",  # 用纯文本模型做最终汇总
        "messages": [{"role": "user", "content": summary_prompt}],
        "stream": False,
        "options": {"temperature": 0.4}
    }
    
    resp = requests.post(OLLAMA_URL, json=payload, timeout=180)
    return resp.json()["message"]["content"]


if __name__ == "__main__":
    import sys
    
    frame_dir = sys.argv[1] if len(sys.argv) > 1 else "./output_frames"
    
    # Step 1: 逐帧分析
    analysis = analyze_video_frames(frame_dir)
    
    # 保存中间结果
    with open("./video_analysis_raw.json", "w") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    
    # Step 2: 生成综合报告
    print("\n\n📝 生成综合报告...\n")
    report = generate_video_summary(analysis)
    print(report)
    
    # 保存报告
    with open("./video_report.md", "w") as f:
        f.write(report)
    print("\n✅ 报告已保存到 video_report.md")
```

## 第三步：完整端到端流程

把帧抽取和分析串联起来：

```bash
#!/bin/bash
# ============================================================
#  video-analyze.sh
#  视频 → 帧抽取 → Ollama 分析 → 报告生成 一站式脚本
#  用法: ./video-analyze.sh <video_file> [strategy]
#  策略: uniform / scene / keyframes (默认 keyframes)
# ============================================================

VIDEO_FILE=$1
STRATEGY=${2:-"keyframes"}
OUTPUT_DIR="./video_analysis_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$OUTPUT_DIR/frames"

echo "============================================="
echo "  Ollama 视频分析工具"
echo "  视频: $VIDEO_FILE"
echo "  策略: $STRATEGY"
echo "  输出: $OUTPUT_DIR"
echo "============================================="

# 阶段 1: 帧抽取
echo ""
echo "[阶段 1/3] 帧抽取..."

case $STRATEGY in
    uniform)
        ffmpeg -i "$VIDEO_FILE" \
            -vf "fps=1,scale=800:-2" \
            -q:v 2 \
            "$OUTPUT_DIR/frames/frame_%04d.jpg" -y
        ;;
    scene)
        ffmpeg -i "$VIDEO_FILE" \
            -vf "select='gt(scene,0.3)',scale=800:-2" \
            -vs vfr -q:v 2 \
            "$OUTPUT_DIR/frames/scene_%04d.jpg" -y
        ;;
    *)
        # keyframes: 提取关键帧
        DURATION=$(ffprobe -v error -show_entries format=duration \
            -of default=noprint_wrappers=1:nokey=1 "$VIDEO_FILE")
        NUM_FRAMES=12
        INTERVAL=$(echo "$DURATION / ($NUM_FRAMES + 1)" | bc -l)
        
        for i in $(seq 1 $NUM_FRAMES); do
            TS=$(echo "$i * $INTERVAL" | bc -l)
            ffmpeg -ss "$TS" -i "$VIDEO_FILE" \
                -vframes 1 -vf "scale=800:-2" -q:v 2 \
                "$OUTPUT_DIR/frames/key_$(printf '%04d' $i).jpg" -y \
                2>/dev/null
        done
        ;;
esac

FRAME_COUNT=$(ls -1 "$OUTPUT_DIR/frames/"*.jpg 2>/dev/null | wc -l)
echo "  ✅ 抽取完成: $FRAME_COUNT 帧"

# 阶段 2: Ollama 分析
echo ""
echo "[阶段 2/3] Ollama 视觉分析..."
python3 analyze_frames.py "$OUTPUT_DIR/frames"
echo "  ✅ 分析完成"

# 阶段 3: 报告生成
echo ""
echo "[阶段 3/3] 生成报告..."
python3 generate_report.py "$OUTPUT_DIR/video_analysis_raw.json" \
    > "$OUTPUT_DIR/video_report.md"
echo "  ✅ 报告已生成"

echo ""
echo "============================================="
echo "  ✅ 全部完成！"
echo "  结果目录: $OUTPUT_DIR"
echo "  报告文件: $OUTPUT_DIR/video_report.md"
echo "============================================="
```

## 性能与限制

### 处理速度估算

```
视频长度    抽帧策略     帧数量    单帧耗时    总耗时
─────────────────────────────────────────────────
30 秒     keyframes    12 帧     ~3-5s       ~40-60s
1 分钟     keyframes    16 帧     ~3-5s       ~60-80s  
5 分钟     uniform      300 帧   ~3-5s       ~15-25分钟
10 分钟    scene        ~80 帧   ~3-5s       ~4-7分钟
30 分钟    keyframes    20 帧    ~3-5s       ~1-2分钟
```

### 内存需求

视频理解本质上是在跑视觉模型，内存需求参考上一节的表格。额外需要注意的是**批处理时的峰值内存**——如果你同时加载多帧进行分析（并行请求），内存占用会成倍增长。建议串行处理。

### 替代方案对比

| 方案 | 优点 | 缺点 | 适用场景 |
|------|------|------|---------|
| FFmpeg + Ollama（本节方案） | 完全本地、隐私安全 | 速度慢、精度有限 | 敏感视频分析 |
| Video-LLaMA / VideoChat | 原生视频理解 | 需要额外部署 | 研究实验 |
| 云 API (GPT-4V + 视频上传) | 效果最好 | 数据出域、费用高 | 非敏感内容 |
| 专用视频模型 (InternVideo) | 时序建模强 | 资源需求极大 | 专业视频分析 |

## 本章小结

这一节我们构建了一套完整的视频理解方案：

1. **Ollama 不原生支持视频**，但可以通过"帧抽取 + 批量视觉分析 + 时序聚合"的工程方案实现
2. **三种帧抽取策略**：均匀采样（固定 fps）、场景变化检测（自适应密度）、关键帧（固定数量）
3. **FFmpeg 是核心工具**，`select='gt(scene,threshold)'` 过滤器可以实现智能场景检测
4. **两阶段处理**：先用视觉模型做逐帧描述，再用文本 LLM 做时序综合推理
5. **性能瓶颈在于串行处理**——30 分钟的视频用 keyframes 策略可在 1-2 分钟内分析完毕

下一节我们将通过三个完整的多模态实战项目来巩固所学知识。
