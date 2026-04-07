# 音频与语音模型：Whisper、Wav2Vec2 与 HuBERT

## 这一节讲什么？

语音是人类最自然的交互方式之一，但长期以来，自动语音识别（ASR）和语音合成（TTS）都是高度专业化的领域——需要复杂的声学模型、语言模型、声学特征提取器等。直到 **Whisper (2022)** 的出现，情况发生了根本性的改变：**一个端到端的 Transformer 模型实现了 SOTA 的多语言语音识别**。

这一节我们将学习：
1. **语音处理的基础流程**——从波形到文本的完整链路
2. **Whisper 架构详解**——Encoder-Decoder + Mel 频谱图
3. **Wav2Vec2 / HuBERT**——自监督语音预训练
4. **HF 实战**——使用 Whisper 进行转录和 TTS

---

## 一、语音处理的完整链路

## 1.1 从声音到文字：ASR 的基本流程

```python
def asr_pipeline_overview():
    """自动语音识别 (ASR) 流程概览"""

    pipeline = """
╔════════════════════════════════════════════════════════════╗
║              自动语音识别 (ASR) 完整处理流程                    ║
╠════════════════════════════════════════════════════════════╣
║                                                            ║
║  输入: 音频文件 (.wav / .mp3 / .flac / 麦克风输入)          ║
║      ↓                                                     ║
║  ┌──────────────┐                                           ║
║  │ Step 1:      │  预加重/降噪/VAD (可选)                  ║
║  │ 音频预处理    │  • 归一化音量                                  ║
║  │              │  • 去除静音段                                  ║
║  └──────┬───────┘                                           ║
║         ↓                                                    ║
║  ┌──────────────┐                                           ║
║  │ Step 2:      │  特征提取器 (Feature Extractor)            ║
║  │ 声学特征提取  │  • MFCC (传统方法, 手工设计特征)              ║
║  │              │  • Filter Bank (学习到的卷积滤波器组)       ║
║  └──────┬───────┘                                           ║
║         ↓                                                    ║
║  ┌──────────────┐                                           ║
║  │ Step 3:      │  编码器 (Encoder)                         ║
║  │ 声学编码器    │  • 将声学特征转换为高层语义表示             ║
║  │              │  • 通常基于 Transformer Encoder               ║
║  └──────┬───────┘                                           ║
║         ↓                                                    ║
║  ┌──────────────┐                                           ║
║  │ Step 4:      │  解码器 (Decoder)                         ║
║  │ 文本解码器    │  • 自回归生成文本 token 序列                 ║
║  │              │  • 通常也是 Transformer Decoder               ║
║  └──────┬───────┘                                           ║
║         ↓                                                    ║
║  输出: 识别出的文本 ("Hello, how are you today?")              ║
║                                                            ║
│  ⭐ Whisper 的创新:                                            ║
│  • Step 1+2 合一: 直接用 log-Mel 频谱图作为输入!           ║
│  • 端端到端: 一个 Transformer 模型完成全部工作                ║
│  • 多任务: 同时做 ASR + 翻译 + 语种识别 + 时间戳预测     ║
╚════════════════════════════════════════════════════════════╝
"""
    print(pipeline)

asr_pipeline_overview()
```

## 1.2 Mel 频谱图：音频的"图像表示"

为什么 Whisper 用 **Mel 频谱图（Mel Spectrogram）** 而不是直接用原始波形？因为 Mel 频谱图是声音在感知上更接近人耳的处理方式：

```python
def mel_spectrogram_intuition():
    """Mel 频谱图的直觉理解"""

    intuition = """
┌───────────────────────────────────────────────────────────────┐
│                  Mel Spectrogram 直觉理解                   │
├───────────────────────────────────────────────────────────────┤
│                                                              │
│  人耳如何处理声音?                                          │
│                                                              │
│  耳朵 (耳蜗):                                                │
│  ┌──────────────────────────┐                                 │
│  │  ██████████████████████  ← 基底膜 (Basilar Membrane)     │
│  │  ░░░░░▓▓▓▓░░░░░░░░░░░░  ← 毛细胞 (Hair Cells)        │
│  │  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  ← 支持细胞 (Supporting Cells)   │
│  └──────────────────────────┘                                 │
│                                                              │
│  不同频率的声音刺激不同位置的毛细胞                          │
│  → 我们对频率敏感, 不是对时间波形敏感!                      │
│                                                              │
│  Mel Spectrogram 做的事情:                                    │
│  • 横轴: 时间帧 (通常每秒 80 帧, 即 12.5ms/帧)              │
│  • 纵轴: Mel 频率刻度 (非线性映射, 低频分辨率高)          │
│  • 颜色深浅: 能量强度 (分贝值)                               │
│                                                              │
│  示例 ("Hello"):                                            │
│                                                              │
│  频率(Hz)│ 0    100   200   ...   8000                           │
│  ────────┼─────┼─────┼─────┼─────┼─────→                       │
│  t=0ms   ████░░░░░░░░░░░░░░░░░░                              │
│  t=12ms  ██░░░░███░░░░░░░░░░░░░░                              │
│  t=24ms  █░░░░░░░░███░░░░░░░░░░░                              │
│  t=36ms  ████████░░░░░░░░░░░░░                                │
│  t=48ms  █░░░░░░░░░░███████░░░                                │
│  t=60ms  ░░░░░░░░░░░░░░░░░░░░░                                │
│                                                              │
│  "H" 和 "e" 是元音, 它们在低频区有强能量                      │
│  "l" 和 "o" 在中高频区有能量                                   │
│                                                              │
│  对比原始波形:                                              │
│  原始波形 = 所有频率的正弦波叠加                             │
│  信息冗余大, 计算效率低                                     │
│  Mel 频谱图 = 人耳感知友好的压缩表示                        │
└───────────────────────────────────────────────────────────────┘
"""
    print(intuition)

mel_spectrogram_intuition()
```

---

## 二、Whisper：OpenAI 的语音识别奇迹

## 2.1 Whisper 为什么这么重要？

```python
def whisper_significance():
    """Whisper 的重要性分析"""

    significance = """
┌───────────────────────────────────────────────────────────────┐
│                    Whisper 的革命性意义                     │
├───────────────────────────────────────────────────────────────┤
│                                                              │
│  Before Whisper (2022 之前):                                  │
│  • ASR 系统 = 声学模型 + 语言模型 + 外部工具链               │
│  • 每个组件都需要单独训练和调优                               │
│  • 多语言支持需要为每种语言训练专门的模型                   │
│  • 弱口音/背景噪音/技术术语鲁棒性差                            │
│  • 开源的高质量 ASR 方案几乎不存在                           │
│                                                              │
│  After Whisper:                                              │
│  ✅ 端到端 Transformer (无需外部组件)                        │
│  ✅ 68 万小时弱监督多语言数据训练                              │
│  ✅ 支持 99 种语言 (包括许多低资源语言)                        │
│  ✅ 极强的鲁棒性 (口音/噪音/技术术语)                          │
│  ✅ 同时支持 ASR + 翻译 + 语种识别 + 时间戳                  │
│  ✅ 完全开源 (Apache 2.0 License)                             │
│                                                              │
│  关键架构:                                                  │
│  Encoder: 处理 log-Mel 频谱图 (类似 ViT 处理图像 patch)          │
│  Decoder: 自回归生成文本 token                                │
│  Cross-Attention: Encoder 输出作为 Decoder 的 K/V              │
│                                                              │
│  数据规模影响:                                              │
│  • tiny: 39M params, ~1GB VRAM, 快但精度一般                   │
│  • base: 74M params, ~1GB VRAM, 平衡之选                       │
│  • small: 244M params, ~2GB VRAM, 推荐大多数场景               │
│  • medium: 769M params, ~5GB VRAM, 高精度需求                   │
│  • large: 1550M params, ~10GB VRAM, 追求极致效果                │
│                                                              │
└───────────────────────────────────────────────────────────────┘
"""
    print(significance)

whisper_significance()
```

## 2.2 Whisper 架构深度解析

```python
def whisper_architecture():
    """Whisper 架构详细解析"""

    arch = """
┌───────────────────────────────────────────────────────────────────┐
│                    Whisper Architecture                     │
├───────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input:                                                      │
│  log-Mel Spectrogram                                        │
│  Shape: (batch=1, n_mels=80, seq_len=3000)                    │
│  (n_mels = 80 个 Mel 频率通道; seq_len 取决于音频长度)       │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              Conv2D Stem (2层卷积)                      │   │
│  │  ↓ 将 Mel 频谱图的维度调整到 hidden_size              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                         ↓                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              Positional Encoding (正弦位置编码)         │   │
│  │  + Sinusoidal Encoding Table (可学习的或固定的)           │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                         ↓                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              Transformer Encoder × N 层                   │   │
│  │                                                       │   │
│  │  For each layer:                                         │   │
│  │    ├─ Multi-Head Self-Attention (Q,K,V from input)       │   │
│  │    ├─ Cross-Attention (K,V from encoder output)          │   │
│  │    └─ Feed-Forward Network (GELU activation)                 │   │
│  │                                                       │   │
│  │  注意: Encoder 的输出会传递给 Decoder 的每一层!           │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                         ↓                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              Transformer Decoder × N 层                   │   │
│  │                                                       │   │
│  │  For each layer:                                         │   │
│  │    ├─ Masked Self-Attention (Causal Mask!)                  │   │
│  │    ├─ Cross-Attention (接收 Encoder 的 K,V)               │   │
│  │    └─ Feed-Forward Network                                │   │
│  │                                                       │   │
│  │  输出: next_token_logits (vocab_size 维向量)               │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                         ↓                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              Projection Layer (LM Head)                   │   │
│  │  → logits over vocabulary (51865 for multilingual model)     │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                         ↓                                       │
│  Output: text tokens (通过 tokenizer 解码为最终文本)           │
│                                                                  │
│  多任务能力 (通过添加不同的 Head):                             │
│  • task="transcribe": 标准语音转文字                             │
│  • task="translate": 翻译                                      │
│  • task="identify": 语种识别                                  │
│  • with_timestamps=True: 额测每个词的时间戳                   │
│                                                                  │
└───────────────────────────────────────────────────────────────────┘
"""
    print(arch)

whisper_architecture()
```

## 2.3 Whisper 实战

```python
def whisper_demo():
    """Whisper 语音识别实战"""

    try:
        import torch
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        
        print("=" * 60)
        print("Whisper 语音识别实战")
        print("=" * 60)
        
        # 加载模型
        model_name = "openai/whisper-tiny"
        
        processor = WhisperProcessor.from_pretrained(model_name)
        model = WhisperForConditionalGeneration.from_pretrained(model_name)
        
        print(f"\n[1] 加载模型: {model_name}")
        print(f"    参数量: {sum(p.numel() for p in model.parameters()):,}")

        # 方法 1: 使用 audio file (推荐)
        print("\n[2] 使用音频文件进行转录:")
        
        # 如果有本地音频文件
        import os
        if os.path.exists("audio.mp3"):
            input_file = "audio.mp3"
        else:
            # 创建一个示例音频 (使用 pyttsx3 或 torchaudio)
            print("    ℹ️  未找到本地音频文件")
            print("    提示: 可使用以下方式提供音频:")
            print("      - 本地 .wav/.mp3/.flac 文件")
            print("      - 麦克风实时录音")
            return

        # 处理音频
        inputs = processor(input_file, return_tensors="pt", sampling_rate=16000)
        
        # 生成
        forced_decoder_ids = processor.get_decoder_prompt_ids(
            language="zh", task="transcribe"
        )
        
        predicted_ids = model.generate(
            inputs.input_features,
            forced_decoder_ids=forced_decoder_ids,
            max_new_tokens=256,
        )
        
        transcription = processor.decode(predicted_ids, skip_special_tokens=True)
        print(f"\n[3] 转录结果:")
        print(f"    '{transcription.strip()}'")

        # 方法 2: 使用 numpy array (适合流式/程序化输入)
        print("\n[4] 使用 NumPy 数组输入 (模拟):")
        import numpy as np
        
        # 模拟 Mel 频谱图 (batch=1, 80 channels, 3000 frames)
        dummy_mel = np.random.randn(1, 80, 3000).astype(np.float32)
        
        input_features = processor(dummy_mel, return_tensors="pt")
        predicted_ids = model.generate(
            input_features=input_features,
            forced_decoder_ids=forced_decoder_ids,
            max_new_tokens=50,
        )
        transcription = processor.decode(predicted_ids, skip_special_tokens=True)
        print(f"    (模拟输出) '{transcription[:50]}...'")

    except ImportError:
        print("\n⚠️  部分库未安装。运行:")
        print("   pip install openai-whisper transformers torch torchaudio")

whisper_demo()
```

---

## 三、Wav2Vec2 与 HuBERT：自监督语音预训练

## 3.1 语音领域的 BERT 时刻

就像 NLP 有 BERT/GPT，视觉领域有 ViT/MAE，语音领域也有自己的预训练基础模型：

```python
def speech_pretraining_models():
    """语音预训练模型对比"""

    models = [
        {
            "名称": "Wav2Vec 2.0",
            "类型": "自监督 (Contrastive)",
            "核心思想": "对比学习: 同一句话的不同增强版本应该相似;"
                      "不同句子应该不相似",
            "预训练数据": "9.3 万小时未标注音频",
            "特点": "Facebook/Meta 出品; "
                      "支持 9 种语言的 XLS-R-53B;"
                      "可作为通用语音 backbone",
        },
        {
            "名称": "HuBERT (Microsoft)",
            "类型": "自监督 (Masked)",
            "核心思想": "类似 MAE 的掩码建模:"
                      "随机 mask 掉部分音频帧,"
                      "让模型重建被遮挡的部分",
            "预训练数据": "60 万小时 (Common Crawl)",
            "特点": "微软出品; "
                      "在 SUPERB 上刷新多项 SOTA;"
                      "支持 100+ 种语言",
        },
        {
            "名称": "WavLM / WavLM v2",
            "类型": "自监督 (Generative)",
            "核心思想": "连接编码器和语言模型:"
                      "将音频编码后用 GPT-style Decoder 生成文本",
            "预训练": "94 万小时 (WebAlign)",
            "特点": "Carnegie Mellon / NVIDIA 出品;"
                      "在多个基准上超越 Wav2Vec2;"
                      "生成式预训练的新范式",
        },
        {
            "名称": "Data2Vec / APC (vq-wav2vec)",
            "类型": "自监督 (VQ-VAE)",
            "核心思想": "离散化: 将连续的音频特征量化为"
                      "有限 codebook 中的索引, 再重建原始信号",
            "特点": "Meta 出品; "
                      "极度压缩的语音表示;"
                      "适合边缘设备部署",
        },
    ]

    print("=" * 70)
    print("语音预训练模型一览")
    print("=" * 70)
    for m in models:
        print(f"\n🎤 {m['名称']}")
        for k, val in m.items():
            if k != "名称":
                print(f"   {k}: {val}")

speech_pretraining_models()
```

---

## 四、本章小结

这一节我们学习了语音处理和多模态模型中的音频组件：

| 模型 | 类型 | 核心贡献 | 应用 |
|------|------|---------|------|
| **Whisper** | 端到端 ASR | 68万小时弱监督多语言训练 | 语音识别/翻译/语种 |
| **Wav2Vec 2.0** | 自监督 Backbone | 对比学习, XLS-R-53B | 语音检索/分类 |
| **HuBERT** | 自监督预训练 | Masked Modeling | 语音理解/分类 |
| **WavLM v2** | 生成式预训练 | Encoder + GPT Decoder | 语音到文本 |

**核心要点**：
1. **Mel 频谱图是音频的"图像"**——它把时域信号转换为时频二维表示，非常适合 Transformer 处理
2. **Whisper 是语音领域的 GPT-3 时刻**——它统一了 ASR、翻译、语种识别等多个任务
3. **自监督学习是语音预训练的主流方向**——标注音频数据极其昂贵，自监督是唯一可行的规模化路径
4. **音频 + 文本 + 图像 = 多模态融合的基础三件套**

下一节我们将学习 **多模态融合模型**——CLIP、BLIP-2、Flamingo 等。
