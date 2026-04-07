# 10-04 视频理解

## 从图像到视频：时间维度的加入

如果你已经理解了上一节中视觉 Transformer（ViT）如何把一张图片切分成 patch、编码成 token 序列，再送入标准 Transformer 处理，那么视频理解的核心问题其实就变得很直观了——视频不就是把多张图片按时间顺序排列在一起吗？但这个"按时间顺序"恰恰是视频理解最关键也最难的地方，因为视频不仅仅是"一堆图片的集合"，它还包含了**运动（motion）、时序依赖（temporal dependency）、因果性（causality）**这些图像模型完全不需要考虑的信息。

举个简单的例子，你给一个纯视觉模型看一张"一个人在踢足球"的图片，它能识别出人和球；但你给它看一段连续的视频帧序列，它就能理解"这个人先跑向球→抬脚→触球→球飞出去"这一整个动作链条，甚至能预测下一秒球会往哪个方向飞。这就是**视频理解**要解决的核心问题：如何在空间特征的基础上建模时间维度上的变化和关联。

在 Hugging Face Transformers 生态中，视频理解主要分为两大类任务：一类是**纯视频理解**（如动作识别、视频分类），另一类是**视频-语言理解**（如视频问答、视频描述生成）。这两类任务的代表模型分别是 InternVideo / VideoMAE 和 VideoLLaVA / AViD。下面我们一层层展开。

## 视频数据的基本处理方式

## 帧采样（Frame Sampling）

视频本质上是一个三维张量 `(T, H, W, C)`，其中 T 是时间维度（帧数）。但在实际处理中，我们不可能把每一帧都送进模型——一段 30 秒的 30fps 视频有 900 帧，全部处理既不现实也没必要。因此第一步就是**帧采样**，也就是从原始视频中挑选出最具代表性的若干帧。

最常见的采样策略有两种：

```python
import torch
from transformers import AutoImageProcessor, AutoModelForVideoClassification
from PIL import Image
import numpy as np


def sample_frames_uniform(video_path: str, num_frames: int = 8) -> list[Image.Image]:
    """均匀采样帧 - 最基础也是最常用的方法"""
    import cv2
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 均匀选取 num_frames 个索引
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
    cap.release()
    return frames


def sample_frames_adaptive(
    video_path: str,
    num_frames: int = 8,
    strategy: str = "uniform"
) -> list[Image.Image]:
    """
    多种采样策略：
    - uniform: 均匀采样（默认）
    - head_heavy: 前端密集采样（适合动作识别，动作通常发生在前半段）
    - keyframe: 基于场景切换检测的关键帧采样
    """
    import cv2
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if strategy == "uniform":
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    elif strategy == "head_heavy":
        # 前半段采更多帧，后半段稀疏
        half = num_frames // 2
        front_indices = np.linspace(0, total_frames // 2 - 1, num_frames - half, dtype=int)
        back_indices = np.linspace(total_frames // 2, total_frames - 1, half, dtype=int)
        indices = np.concatenate([front_indices, back_indices])
    
    elif strategy == "keyframe":
        # 基于帧差检测场景切换
        prev_frame = None
        keyframes = []
        min_gap = total_frames // (num_frames * 2)  # 最小间隔
        
        for i in range(total_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if prev_frame is not None:
                diff = cv2.absdiff(prev_frame, gray)
                diff_score = np.mean(diff)
                
                # 如果与上一个关键帧差距足够大且间隔足够
                if (diff_score > 30 and 
                    (len(keyframes) == 0 or i - keyframes[-1] > min_gap)):
                    keyframes.append(i)
                    
                    if len(keyframes) >= num_frames:
                        break
            
            prev_frame = gray
        
        # 如果关键帧不够，用均匀采样补齐
        if len(keyframes) < num_frames:
            remaining = num_frames - len(keyframes)
            extra = np.linspace(0, total_frames - 1, remaining + 2, dtype=int)[1:-1]
            indices = sorted(set(list(keyframes) + list(extra)))
        else:
            indices = keyframes[:num_frames]
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")
    
    frames = []
    for idx in indices[:num_frames]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
    cap.release()
    return frames
```

这里有一个非常容易踩的坑：**采样帧数的选择直接决定了模型的性能上限**。太少帧（比如 4 帧）可能丢失关键的中间动作状态，太多帧（比如 32 帧）又会带来巨大的计算开销和显存压力。大多数预训练视频模型在设计时就固定了一个"最优帧数"，比如 VideoMAE 默认 16 帧、InternVideo 默认 8 帧。如果你随意修改这个数字而不做相应的微调，效果往往会显著下降。

## 时间维度的嵌入方式

拿到采样后的帧序列后，下一个问题是：怎么让模型知道这些帧是有先后顺序的？这涉及两种主流的时间编码方案：

```
┌─────────────────────────────────────────────────────────────┐
│                     方案一：3D Patch Embedding               │
│                                                             │
│  输入: (B, T, C, H, W)                                      │
│       ↓                                                      │
│  3D Conv: kernel=(t, ph, pw)  把时间和空间一起卷积           │
│       ↓                                                      │
│  输出: (B, T*H/ph*W/pw, D)  每个时空patch变成一个token       │
│                                                             │
│  代表: VideoMAE, ViViT                                       │
├─────────────────────────────────────────────────────────────┤
│                     方案二：2D Patch + Time Embedding         │
│                                                             │
│  输入: (B, T, C, H, W)                                      │
│       ↓  对每帧独立做 2D Patch Embedding                      │
│  每帧 → (H/ph*W/pw, D) 的 token 序列                        │
│       ↓  在 token 序列上拼接可学习的时间 embedding            │
│  输出: (B, T*N_patches, D)                                  │
│                                                             │
│  代表: InternVideo, TimeSformer                              │
└─────────────────────────────────────────────────────────────┘
```

方案一（3D Conv）的优势是能同时捕获局部的时空模式，比如"某个区域从第 t 帧到第 t+3 帧发生了什么变化"；方案二（2D+Time Embed）的优势是复用了成熟的 2D 预训练权重（比如 ImageNet 预训练的 ViT），迁移成本更低。目前工业界更倾向于后者，因为它可以充分利用海量的图像预训练知识。

## 纯视频理解模型：VideoMAE 与 InternVideo

## VideoMAE：视频版的 Masked Autoencoder

我们在视觉 Transformer 一节中介绍过 MAE（Masked Autoencoder）的思想——随机遮住图片的大部分 patch，然后让模型去重建被遮住的部分。VideoMAE 就是把这个思想自然地扩展到了视频领域：**不仅遮住空间上的 patch，还遮住时间上的帧**。

```python
from transformers import VideoMAEConfig, VideoMAEForVideoClassification
import torch


def explain_videomae_architecture():
    """
    VideoMAE 架构详解：
    
    输入视频: (batch, num_frames, channels, height, width)
             例如: (2, 16, 3, 224, 224)  →  16帧224x224的视频
    
    Step 1: Tubelet Embedding (3D Patch Embedding)
    ─────────────────────────────────────────
    用 3D 卷积核 (2, 16, 16) 将时空体积分割成 tubelets:
    - 时间维度: 每 2 帧一组
    - 空间维度: 每个 16x16 区域一组
    - 每个 tubelet 展平后投影到 hidden_size 维度
    
    对于 16帧 224x224 视频:
      时间方向: 16 // 2 = 8 个 tubelet
      空间方向: (224//16)² = 14² = 196 个 tubelet
      总共: 8 × 196 = 1568 个 tokens (+ [CLS])
    
    Step 2: Masking (高比例掩码)
    ─────────────────────────────────────────
    VideoMAE 使用 90% ~ 95% 的掩码率！远高于图像 MAE 的 75%
    为什么视频可以用更高掩码率？
    → 因为相邻帧之间高度冗余，即使遮掉大部分，模型也能从
      时间相关性中推断出缺失内容
    
    Step 3: Encoder (只处理可见部分)
    ─────────────────────────────────────────
    只将未被掩码的 ~5-10% tokens 送入 Transformer Encoder
    这使得预训练极其高效
    
    Step 4: Decoder (重建所有位置)
    ─────────────────────────────────────────
    Decoder 接收 encoder 输出 + mask tokens (可学习的占位向量)
    输出每个位置的像素值重建结果
    
    Loss: MSE(原始像素值, 重建像素值)，只计算被掩码位置
    """
    config = VideoMAEConfig(
        num_frames=16,          # 输入帧数
        image_size=224,         # 每帧分辨率
        patch_size=16,          # 空间 patch 大小
        tubelet_size=2,         # 时间方向的 patch 大小（每几帧一组）
        hidden_size=768,        # Transformer 隐藏层大小
        num_hidden_layers=12,   # Transformer 层数
        num_attention_heads=12, # 注意力头数
        mask_ratio=0.95,        # 掩码率 95%！
        num_labels=400,         # Kinetics-400 动作类别数
    )
    
    model = VideoMAEForVideoClassification(config)
    
    print(f"VideoMAE 参数量: {sum(p.numel()() for p in model.parameters()) / 1e6:.1f}M")
    print(f"输入形状: (batch, {config.num_frames}, 3, {config.image_size}, {config.image_size})")
    
    # 计算实际的 token 数量
    t_tokens = config.num_frames // config.tubelet_size
    h_tokens = config.image_size // config.patch_size
    w_tokens = config.image_size // config.patch_size
    num_patches = t_tokens * h_tokens * w_tokens
    print(f"Tubelet 数量: {t_tokens}(T) × {h_tokens}(H) × {w_tokens}(W) = {num_patches}")
    print(f"实际输入 Encoder 的 tokens (~5% 可见): ~{int(num_patches * 0.05)}")


explain_videomae_architecture()
```

运行上面的代码你会看到，VideoMAE 一个核心设计决策是使用 **95% 的超高掩码率**。这个数字看起来疯狂——你遮住了 95% 的内容，模型怎么能学会任何东西？但正是因为视频中相邻帧之间存在极强的冗余信息（一个人跑步的第 1~3 帧几乎一模一样），所以模型可以通过时间上的上下文来推断被遮住的区域。这种高掩码率带来的好处是 **Encoder 只需要处理极少量 token（约 5%）**，这使得预训练速度比传统方法快很多倍。

## InternVideo：大规模视频-语言预训练

如果说 VideoMAE 是"纯视觉"路线的代表，那 **InternVideo** 走的就是"视频-语言联合预训练"的路线。它不仅仅学习视频内部的表示，还通过大量的视频-文本对来学习视频内容和自然语言之间的对应关系。

```python
from transformers import AutoModel, AutoProcessor
import torch


def internvideo_classification_demo():
    """
    使用 InternVideo 进行视频分类（动作识别）
    
    InternVideo 在 Kinetics-400/600/700、Something-Something v2 等
    数据集上达到了 SOTA 性能。
    """
    
    model_name = "CVPR/InternVideo2Stage1_K400_1K_224p"
    
    try:
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        
        # 模拟视频输入：(num_frames, C, H, W)
        video_tensor = torch.randn(8, 3, 224, 224)  # 8帧
        
        inputs = processor(
            videos=list(video_tensor),  # processor 需要 list of tensors
            return_tensors="pt",
        )
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        
        predicted_class_idx = logits.argmax(-1).item()
        print(f"Predicted class index: {predicted_class_idx}")
        print(f"Logits shape: {logits.shape}")  # (batch, num_classes)
        
    except Exception as e:
        print(f"Note: InternVideo may require specific setup. Error: {e}")
        print("Below is the conceptual usage pattern:")


internvideo_classification_demo()


def internvideo_feature_extraction():
    """
    InternVideo 作为视频特征提取器
    
    很多时候我们不只需要分类结果，还需要视频的特征向量
    用于下游任务（如检索、聚类、异常检测等）
    """
    
    model_name = "CVPR/InternVideo2Stage1_K400_1K_224p"
    
    # InternVideo 提取特征的典型流程
    class InternVideoFeatureExtractor:
        def __init__(self, model_name: str, device: str = "cuda"):
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
            ).to(device).eval()
            self.device = device
        
        def extract_features(
            self,
            video_frames: list,  # list of PIL Images or tensors
            pooling: str = "mean",  # mean / cls / max
            normalize: bool = True,
        ) -> torch.Tensor:
            """
            提取视频级别的特征向量
            
            Args:
                video_frames: 采样的视频帧列表
                pooling: 池化策略
                    - 'mean':对所有patch token求平均（最常用）
                    - 'cls': 取[CLS] token
                    - 'max': 最大池化
                normalize: 是否L2归一化（用于余弦相似度计算）
            """
            inputs = self.processor(
                videos=video_frames,
                return_tensors="pt",
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                
                # 获取最后一层的隐藏状态
                last_hidden = outputs.last_hidden_state  # (1, seq_len, dim)
                
                if pooling == "cls":
                    feature = last_hidden[:, 0, :]  # [CLS] token
                elif pooling == "mean":
                    feature = last_hidden.mean(dim=1)  # 平均池化
                elif pooling == "max":
                    feature = last_hidden.max(dim=1).values  # 最大池化
                else:
                    raise ValueError(f"Unknown pooling: {pooling}")
                
                if normalize:
                    feature = torch.nn.functional.normalize(feature, p=2, dim=-1)
                
                return feature
        
        def compute_similarity(self, feat_a: torch.Tensor, feat_b: torch.Tensor) -> float:
            """计算两个视频特征之间的余弦相似度"""
            return (feat_a @ feat_b.T).item()
        
        def find_similar_videos(
            self,
            query_video_frames: list,
            database_features: dict,  # {video_id: feature_tensor}
            top_k: int = 5,
        ) -> list[tuple[str, float]]:
            """
            视频相似度检索
            
            给定一个查询视频，从数据库中找出最相似的top-k个视频
            """
            query_feat = self.extract_features(query_video_frames)
            
            similarities = {}
            for vid, db_feat in database_features.items():
                sim = self.compute_similarity(query_feat, db_feat)
                similarities[vid] = sim
            
            ranked = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
            return ranked[:top_k]


internvideo_feature_extraction()
```

## TimeSformer：时间注意力机制的设计选择

在视频 Transformer 中，如何让不同帧之间的信息交互是一个核心架构问题。**TimeSformer（Time-space Transformer）** 提出了几种不同的时间注意力设计方案：

```
┌──────────────────────────────────────────────────────────────────┐
│              TimeSformer 的四种时间注意力变体                       │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Space-only Attention (空间注意力)                             │
│     每一帧独立做自注意力，帧之间不交互                              │
│     → 快速，但无法建模时序关系                                     │
│                                                                  │
│  2. Joint Space-Time Attention (联合时空注意力)                    │
│     把所有帧的所有 patch 展平成一个超长序列，统一做自注意力           │
│     → 表达能力最强，但 O((T×N)²) 复杂度极高                         │
│                                                                  │
│  3. Divided Space-Time Attention (分离时空注意力) ★ 最常用         │
│     先做空间注意力（每帧内部），再做时间注意力（跨帧同位置）           │
│     → 平衡了效率和表达能力                                        │
│                                                                  │
│     具体流程:                                                     │
│     Input: (T, N, D)  T帧，每帧N个patch                           │
│       ↓                                                           │
│     Spatial Attn: 对每个 t ∈ [T], 在 N 个 patch 上做 Self-Attn   │
│       ↓  (T, N, D)                                               │
│     Temporal Attn: 对每个 n ∈ [N], 在 T 帧的同位置上做 Self-Attn  │
│       ↓  (T, N, D)                                               │
│     FFN → Output                                                 │
│                                                                  │
│  4. Sparse Local Attention (稀疏局部注意力)                        │
│     只关注邻近帧（±k 帧窗口内）                                    │
│     → 适合长视频，降低复杂度                                       │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

其中 **Divided Space-Time Attention** 是最常用的方案，它把一次复杂的联合注意力分解为两次相对简单独立的操作，既保留了跨帧交互的能力，又避免了二次方爆炸的问题。

## 视频-语言模型：VideoLLaVA

## 从 LLaVA 到 VideoLLaVA

我们在多模态融合一节中介绍了 LLaVA（Large Language and Vision Assistant）——它通过一个视觉编码器（如 CLIP-ViT）提取图像特征，再通过一个投影层把视觉特征映射到 LLM 的词嵌入空间，最后由 LLM 生成文本回答。**VideoLLaVA** 的思路完全一样，唯一的区别是把输入从"单张图片"换成了"多帧视频序列"。

```python
class ConceptualVideoLLaVAArchitecture:
    """
    VideoLLaVA 架构概念实现
    
    核心思路: 把视频的每一帧当作一张独立图片，
    分别提取特征后拼接，再送给 LLM 理解
    
    ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐
    │ Frame 1  │  │ Frame 2  │  │ Frame t  │  │ Frame T  │
    └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘
         │             │             │             │
         ▼             ▼             ▼             ▼
    ┌─────────────────────────────────────────────────┐
    │           Vision Encoder (CLIP-ViT-L/14)        │
    │    每帧独立编码 → 每帧得到 (N_patches, D) 特征    │
    └──────────────────────┬──────────────────────────┘
                           │
                           ▼
    ┌─────────────────────────────────────────────────┐
    │           Temporal Aggregation                   │
    │    方式A: 直接拼接所有帧特征 → (T×N, D)           │
    │    方式B: 帧级平均池化 → (N, D)                  │
    │    方式C: 时序 Transformer 编码 → (T', D)        │
    └──────────────────────┬──────────────────────────┘
                           │
                           ▼
    ┌─────────────────────────────────────────────────┐
    │           Projection Layer (MLP)                 │
    │    映射到 LLM 的 embedding 维度                   │
    └──────────────────────┬──────────────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │   LLM (Vicuna/Llama)   │
              │   + Text Instructions   │
              └────────────┬───────────┘
                           │
                           ▼
                    文本回答输出
    """
    pass


def videollava_inference_demo():
    """
    VideoLLaVA 实际推理示例
    
    VideoLLaVA 支持多种视频理解任务：
    - 视频描述生成 (Video Captioning)
    - 视频问答 (Video QA)
    - 视频对话 (Video Conversation)
    """
    
    from transformers import pipeline
    
    # 使用 Hugging Face pipeline 进行视频对话
    # 注意: VideoLLaVA 可能需要特定版本的 transformers 和额外依赖
    try:
        video_chat = pipeline(
            "video-text-to-text",
            model="LanguageBind/Video-LLaVA-7B",
            trust_remote_code=True,
            device_map="auto",
        )
        
        # 模拟视频输入（实际使用时传入真实视频路径或帧序列）
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "video"},
                    {
                        "type": "text",
                        "text": "Describe what is happening in this video."
                    },
                ],
            }
        ]
        
        output = video_chat(conversation)
        print("VideoLLaVA Response:", output)
        
    except Exception as e:
        print(f"VideoLLaVA requires specific environment setup.")
        print(f"Conceptual API pattern shown below:\n")


videollava_inference_demo()


def videollava_practical_usage():
    """
    VideoLLaVA 的实用代码模板
    
    即使无法直接运行完整模型，我们可以展示其标准调用模式
    """
    
    # ========== 场景1: 视频内容摘要 ==========
    video_summary_prompt = """
    请用中文总结这段视频的主要内容，包括：
    1. 视频中出现的主要人物/物体
    2. 发生的主要事件或动作
    3. 整体的氛围或情感倾向
    回答控制在200字以内。
    """
    
    # ========== 场景2: 视频问答 ==========
    video_qa_prompts = {
        "action_recognition": "视频中的人在做什么动作？请详细描述。",
        "temporal_reasoning": "视频中的事件是按照什么顺序发生的？",
        "causal_reasoning": "为什么视频中的人会做出这样的反应？",
        "counting": "视频中一共出现了几个人？",
        "attribute_query": "视频中的物体是什么颜色的？",
    }
    
    # ========== 场景3: 关键时刻定位 ==========
    temporal_localization_prompt = """
    请分析这段视频，找出以下关键时刻对应的帧号范围：
    1. 动作开始的时间点
    2. 动作达到高潮的时间点
    3. 动作结束的时间点
    如果格式为 JSON: {"start": X, "climax": Y, "end": Z}
    """
    
    # ========== 场景4: 视频质量评估 ==========
    quality_assessment_prompt = """
    从以下几个维度评估这段视频的质量：
    1. 画面清晰度和稳定性
    2. 内容的连贯性和完整性
    3. 信息密度和观赏价值
    给出1-10分的评分并说明理由。
    """
    
    print("VideoLLaVA Prompt Templates Ready")
    print(f"Available scenarios: summary, qa ({len(video_qa_prompts)} types), "
          f"localization, quality")


videollava_practical_usage()
```

VideoLLaVA 的一个重要设计细节是**时间聚合策略**的选择。如果采用"直接拼接所有帧特征"的方式，当帧数较多时（比如 16 帧，每帧 196 个 patch），就会产生 3136 个视觉 token，这对 LLM 来说是一个巨大的负担（上下文长度限制 + 计算成本）。因此很多改进版本会先用一个轻量级的时序模块（如平均池化或一个小型 Transformer）把多帧压缩成更紧凑的表示，再送给 LLM。

## AViD：自适应视频-语言对齐

**AViD（Adaptive Video Discriminator）** 采用了另一种思路来解决视频-语言对齐的问题。它不是简单地"把所有帧都编码进去"，而是引入了一个**自适应帧选择机制**，让模型自己决定哪些帧是重要的、哪些帧可以跳过。

```python
class AdaptiveFrameSelector(torch.nn.Module):
    """
    AViD 式的自适应帧选择器
    
    核心思想: 不是均匀采样固定数量的帧，
    而是根据视频内容的复杂度动态决定需要多少帧
    
    对于一个"一个人静止站立"的视频，可能 2-3 帧就够了；
    对于一个"快速打篮球"的视频，可能需要 16+ 帧才能捕捉到关键动作。
    """
    
    def __init__(
        self,
        vision_encoder: torch.nn.Module,
        max_frames: int = 32,
        min_frames: int = 4,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.max_frames = max_frames
        self.min_frames = min_frames
        
        # 帧重要性评分网络
        self.frame_scorer = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim // 2, 1),
        )
        
        # 自适应阈值网络（根据全局特征动态调整阈值）
        self.threshold_net = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool1d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(hidden_dim, 1),
            torch.nn.Sigmoid(),  # 输出 0~1 之间的阈值
        )
    
    def forward(
        self,
        all_frame_features: torch.Tensor,  # (T, D) 所有候选帧的特征
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            all_frame_features: 所有候选帧的视觉特征
        
        Returns:
            selected_features: 选出的重要帧特征
            importance_scores: 每帧的重要性分数
        """
        T, D = all_frame_features.shape
        
        # Step 1: 计算每帧的重要性分数
        scores = self.frame_scorer(all_frame_features).squeeze(-1)  # (T,)
        scores = torch.sigmoid(scores)  # 归一化到 0~1
        
        # Step 2: 根据全局特征计算自适应阈值
        global_feature = all_frame_features.mean(dim=0, keepdim=True)  # (1, D)
        adaptive_threshold = self.threshold_net(global_feature.T).squeeze()  # ()
        
        # Step 3: 选择重要性超过阈值的帧
        selected_mask = scores > adaptive_threshold
        selected_indices = torch.where(selected_mask)[0]
        
        # 保证至少选 min_frames 帧
        if len(selected_indices) < self.min_frames:
            _, topk_indices = scores.topk(self.min_frames)
            selected_indices = topk_indices
        
        # 保证最多选 max_frames 帧
        if len(selected_indices) > self.max_frames:
            _, topk_indices = scores.topk(self.max_frames)
            selected_indices = topk_indices
        
        # 提取选中帧的特征
        selected_features = all_frame_features[selected_indices]  # (T_sel, D)
        selected_scores = scores[selected_indices]
        
        return selected_features, selected_scores


class AViDStyleModel(torch.nn.Module):
    """
    AViD 风格的视频-语言模型完整流程
    """
    
    def __init__(self, vision_dim: int = 768, llm_dim: int = 4096):
        super().__init__()
        
        # Vision Encoder (冻结的 CLIP-ViT)
        self.vision_encoder = ...  # 冻结的视觉编码器
        
        # 自适应帧选择器
        self.frame_selector = AdaptiveFrameSelector(
            vision_encoder=self.vision_encoder,
            max_frames=16,
            min_frames=4,
            hidden_dim=vision_dim,
        )
        
        # 投影层：视觉 → LLM 空间
        self.projector = torch.nn.Sequential(
            torch.nn.Linear(vision_dim, llm_dim),
            torch.nn.GELU(),
            torch.nn.Linear(llm_dim, llm_dim),
        )
        
        # LLM
        self.llm = ...  # 语言模型
    
    def forward(
        self,
        video_frames: torch.Tensor,  # (T_raw, C, H, W) 原始帧
        input_ids: torch.Tensor,     # 文本 input_ids
        attention_mask: torch.Tensor = None,
    ):
        # Step 1: 提取所有帧的初步特征
        all_features = []  # 收集每帧特征
        for t in range(video_frames.shape[0]):
            frame_feat = self.vision_encoder(video_frames[t:t+1])  # (1, D)
            all_features.append(frame_feat.squeeze(0))
        all_features = torch.stack(all_features)  # (T_raw, D)
        
        # Step 2: 自适应选择重要帧
        selected_features, importance_scores = self.frame_selector(all_features)
        # selected_features: (T_sel, D)
        
        # Step 3: 投影到 LLM 空间
        visual_tokens = self.projector(selected_features)  # (T_sel, llm_dim)
        
        # Step 4: 拼接文本 token 和视觉 token
        text_embeddings = self.llm.get_input_embeddings()(input_ids)
        combined_embeddings = torch.cat([visual_tokens, text_embeddings], dim=1)
        
        # Step 5: LLM 生成
        outputs = self.llm(
            inputs_embeds=combined_embeddings,
            attention_mask=attention_mask,
        )
        
        return outputs, importance_scores
```

AViD 这种自适应选择的思路在实际应用中非常有价值。想象一下你在做一个视频监控系统的异常检测——大部分时间的监控画面都是静止的或者只有微小变化（不需要太多帧），只有发生异常的那几秒才包含丰富的动作信息（需要密集采样）。自适应帧选择可以让模型在这种场景下既节省计算资源又不丢失关键信息。

## 视频理解的常见任务与应用

## 任务一：动作识别（Action Recognition）

这是视频理解中最经典的任务——给定一段视频，判断其中发生了什么动作。比如 Kinetics-400 数据集就包含了 400 种人类日常动作类别（跑步、游泳、弹吉他、做饭等等）。

```python
def action_recognition_pipeline():
    """
    动作识别完整流水线
    
    从视频文件到最终分类结果的完整流程
    """
    import cv2
    from PIL import Image
    import torch
    from transformers import AutoImageProcessor, AutoModelForVideoClassification
    
    # Step 1: 配置
    model_name = "MCG-NJU/videomae-base-finetuned-kinetics"
    num_frames = 16  # VideoMAE 默认采样 16 帧
    
    # Step 2: 加载模型和处理器
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForVideoClassification.from_pretrained(model_name)
    model.eval()
    
    # Step 3: 从视频中采样帧
    def read_video_frames(video_path: str, num_frames: int = 16) -> list[Image.Image]:
        """带错误处理的视频帧读取"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"Video info: {total_frames} frames, {fps:.1f} fps, {duration:.1f}s")
        
        # 均匀采样
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(rgb))
        
        cap.release()
        
        if len(frames) < num_frames:
            print(f"Warning: Only got {len(frames)} frames, expected {num_frames}")
            # 用最后一帧补齐
            while len(frames) < num_frames:
                frames.append(frames[-1])
        
        return frames
    
    # Step 4: 推理
    def classify_video(video_path: str, top_k: int = 5) -> list[dict]:
        """
        对视频进行动作分类
        
        Returns:
            top-k 结果列表，每项包含 label, score, description
        """
        frames = read_video_frames(video_path, num_frames)
        
        inputs = processor(
            images=frames,
            return_tensors="pt",
        )
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
        
        # Top-K 结果
        top_probs, top_indices = probs.topk(top_k)
        
        results = []
        for score, idx in zip(top_probs.tolist(), top_indices.tolist()):
            label = model.config.id2label[idx]
            results.append({
                "label": label,
                "score": round(score, 4),
            })
        
        return results
    
    # 使用示例
    # results = classify_video("path/to/video.mp4", top_k=5)
    # for r in results:
    #     print(f"{r['label']}: {r['score']*100:.1f}%")
    
    return classify_video


action_recognition_pipeline()
```

## 任务二：视频描述生成（Video Captioning）

给定一段视频，自动生成一段自然语言描述。这比动作识别难得多，因为不仅要识别"发生了什么"，还要用通顺的语言表达出来。

```python
def video_captioning_with_template():
    """
    视频描述生成的结构化输出模板
    
    由于完整的视频-语言模型环境配置较复杂，
    这里提供一个实用的 prompt engineering 模板
    """
    
    caption_templates = {
        "descriptive": """
请详细描述这段视频的内容。要求：
1. 开头概括视频主题（一句话）
2. 按时间顺序描述主要事件
3. 描述关键人物的外貌、动作、表情
4. 描述环境和背景
5. 总结视频传达的情感或信息
""",
        
        "concise": """
用两句话概括这段视频的内容。第一句描述场景和主体，第二句描述发生的动作或事件。
""",
        
        "structured": """
请按以下JSON格式输出视频描述：
{
    "scene": "场景描述",
    "subjects": ["主体1", "主体2"],
    "actions": ["动作1", "动作2"],
    "objects": ["物体1", "物体2"],
    "timeline": ["时间点1的事件", "时间点2的事件"],
    "mood": "整体氛围",
    "summary": "一句话总结"
}
""",
        
        "social_media": """
为这段视频写一条社交媒体文案。要求：
- 有吸引力，能在3秒内抓住读者注意
- 包含适当的 emoji
- 长度控制在100字以内
- 适合发布在微博/小红书/抖音平台
""",
    }
    
    return caption_templates


def video_captioning_post_processing():
    """
    视频描述的后处理技巧
    
    模型输出的原始描述往往需要清洗和优化
    """
    
    import re
    
    def clean_caption(raw_caption: str) -> str:
        """清理模型输出的描述文本"""
        # 移除重复短语
        caption = re.sub(r'(.{10,}?)\1+', r'\1', raw_caption)
        # 移除开头/结尾的无意义词
        caption = caption.strip().lstrip('嗯啊呃这个那个').strip()
        # 确保以标点结尾
        if caption and caption[-1] not in '。！？.!':
            caption += '。'
        return caption
    
    def enhance_caption(caption: str, style: str = "natural") -> str:
        """增强描述的可读性"""
        enhancements = {
            "natural": lambda c: c,  # 保持原样
            "detailed": lambda c: c.replace("。", "，具体来说，")[:-1] + "。",
            "casual": lambda c: c.replace("进行", "").replace("实施", ""),
        }
        return enhancements.get(style, lambda c: c)(caption)
    
    return clean_caption, enhance_caption


video_captioning_with_template()
video_captioning_post_processing()
```

## 任务三：视频问答（VideoQA）

用户针对视频内容提出问题，模型基于视频内容给出回答。这是视频理解能力的综合测试——模型需要同时理解视觉内容和语言问题，并在两者之间建立正确的对应关系。

```python
class VideoQASystem:
    """
    视频 QA 系统
    
    支持多种类型的视频问答
    """
    
    QUESTION_TEMPLATES = {
        # === 基础感知类 ===
        "what": "视频中{subject}是什么？",
        "who": "视频中的人是谁？（如果能认出的话）",
        "where": "视频拍摄地点在哪里？",
        "when": "视频中的事件发生在什么时间段？",
        
        # === 动作行为类 ===
        "doing_what": "{subject}正在做什么？",
        "how": "{subject}是怎么{action}的？请详细描述动作过程。",
        "why_action": "{subject}为什么{action}？根据视频内容推断原因。",
        
        # === 时序推理类 ===
        "order": "视频中事件的正确顺序是什么？",
        "before_after": "在{event_A}{之前/之后}发生了什么？",
        "duration": "{action}持续了多长时间？",
        
        # === 因果推理类 ===
        "cause": "导致{result}的原因是什么？",
        "effect": "如果{condition}没有发生，接下来可能会怎样？",
        "prediction": "根据当前的趋势，接下来最可能发生什么？",
        
        # === 计数统计类 ===
        "count": "视频中一共有{target}几个？",
        "frequency": "{action}在视频中出现了几次？",
        
        # === 情感态度类 ===
        "emotion": "视频中{subject}的情绪是怎样的？",
        "attitude": "视频整体传达的态度/立场是什么？",
        
        # === 细节定位类 ===
        "locate": "{object}出现在视频的什么位置？大约在第几秒？",
        "find_moment": "{description}的时刻出现在视频的哪部分？",
    }
    
    @classmethod
    def generate_question(cls, q_type: str, **kwargs) -> str:
        """根据类型和参数生成问题"""
        template = cls.QUESTION_TEMPLATES.get(q_type)
        if template is None:
            available = ", ".join(cls.QUESTION_TEMPLATES.keys())
            raise ValueError(f"Unknown question type: {q_type}. Available: {available}")
        return template.format(**kwargs)


def build_videoqa_dataset_example():
    """
    构建 VideoQA 数据集示例格式
    
    实际开发中，你可能需要构建自己的 VideoQA 数据集
    """
    
    example_dataset = [
        {
            "video_id": "vid_001",
            "video_path": "/data/videos/cooking_001.mp4",
            "duration_seconds": 120,
            "questions": [
                {
                    "question_id": "q_001",
                    "question": "视频里的人在做什么菜？",
                    "answer": "番茄炒蛋",
                    "answer_type": "entity",
                    "difficulty": "easy",
                    "requires_temporal_reasoning": False,
                    "relevant_segment": [10, 45],  # 相关时间段（秒）
                },
                {
                    "question_id": "q_002",
                    "question": "炒蛋的时候为什么要加一点水？",
                    "answer": "让鸡蛋更嫩滑",
                    "answer_type": "explanation",
                    "difficulty": "hard",
                    "requires_temporal_reasoning": False,
                    "requires_causal_reasoning": True,
                    "relevant_segment": [30, 40],
                },
                {
                    "question_id": "q_003",
                    "question": "这个人先洗了番茄还是先打了鸡蛋？",
                    "answer": "先打了鸡蛋",
                    "answer_type": "ordering",
                    "difficulty": "medium",
                    "requires_temporal_reasoning": True,
                    "relevant_segment": [5, 20],
                },
            ],
        },
    ]
    
    return example_dataset


build_videoqa_dataset_example()
```

## 视频理解中的常见陷阱与最佳实践

## 陷阱一：帧采样偏差

这是视频理解中最隐蔽也最容易忽视的问题。假设你的训练数据中所有"做饭"类别的视频都是 30fps 的，而测试时遇到了一个 60fps 的视频。如果你用固定的帧数（比如 16 帧）做均匀采样，那么 30fps 的视频覆盖了约 16/30 = 0.53 秒的内容，而 60fps 的视频只覆盖了 16/60 = 0.27 秒——同样的采样策略在不同帧率的视频上覆盖了完全不同的时间跨度！

```python
def frame_sampling_bias_analysis():
    """
    分析帧采样偏差问题及其解决方案
    """
    
    def calculate_coverage(
        total_frames: int,
        fps: float,
        sampled_frames: int,
    ) -> dict:
        """
        计算采样覆盖的时间信息
        
        Args:
            total_frames: 视频总帧数
            fps: 帧率
            sampled_frames: 采样帧数
        
        Returns:
            包含覆盖率等信息的字典
        """
        duration = total_frames / fps
        time_per_sample = duration / sampled_frames
        
        return {
            "total_duration_sec": round(duration, 2),
            "sampled_frames": sampled_frames,
            "time_coverage_per_sample_sec": round(time_per_sample, 3),
            "total_time_covered_sec": round(duration, 2),  # 均匀采样覆盖全长
            "sampling_density_fps": round(sampled_frames / duration, 2),
        }
    
    # 对比不同帧率下的采样情况
    scenarios = [
        ("30fps 正常视频", 900, 30.0, 16),
        ("60fps 高帧率视频", 1800, 60.0, 16),
        ("24fps 电影视频", 576, 24.0, 16),
        ("15fps 低帧率视频", 300, 15.0, 16),
    ]
    
    print("=" * 70)
    print("帧采样偏差分析（固定采样 16 帧）")
    print("=" * 70)
    
    for name, total_frames, fps, num_samples in scenarios:
        info = calculate_coverage(total_frames, fps, num_samples)
        print(f"\n{name}:")
        print(f"  总时长: {info['total_duration_sec']}s | "
              f"采样密度: {info['sampling_density_fps']} 帧/s")
    
    # 解决方案
    solutions = """
    
    【解决方案】
    
    1. 基于时间的采样（而非基于帧数的采样）:
       不采样固定 N 帧，而是每隔固定时间间隔（如 0.5 秒）采一帧
       
    2. 自适应采样密度:
       根据视频时长动态调整采样帧数
       - 短视频(<5s): 采样 8 帧
       - 中等视频(5-30s): 采样 16 帧  
       - 长视频(>30s): 采样 32 帧 + 分段处理
       
    3. 多尺度采样:
       同时使用粗粒度（少帧）和细粒度（多帧）采样，
       融合两者的特征
    """
    print(solutions)


frame_sampling_bias_analysis()
```

## 陷阱二：短视频 vs 长视频的处理差异

短视频（几秒钟的 TikTok 片段）和长视频（几十分钟的电影或直播录像）的处理策略应该完全不同，但很多人会用同一套参数硬套。

```python
def video_length_aware_processing():
    """
    根据视频长度自适应的处理策略
    """
    
    def get_processing_config(duration_seconds: float) -> dict:
        """
        根据视频长度返回最优处理配置
        
        这是生产环境中常用的策略选择函数
        """
        if duration_seconds < 10:
            # 短视频：尽可能多地保留信息
            return {
                "num_frames": min(int(duration_seconds * 3), 16),  # 每秒约3帧
                "sampling_strategy": "uniform",
                "resolution": 224,
                "use_temporal_model": False,  # 太短不需要复杂的时序建模
                "description": "短视频模式 - 高采样密度",
            }
        elif duration_seconds < 60:
            # 中等视频：平衡效率和效果
            return {
                "num_frames": 16,
                "sampling_strategy": "uniform",
                "resolution": 224,
                "use_temporal_model": True,
                "description": "中等视频模式 - 标准配置",
            }
        elif duration_seconds < 600:
            # 长视频：分段处理 + 关键帧检测
            return {
                "num_frames": 32,
                "sampling_strategy": "keyframe",  # 使用关键帧检测
                "segment_duration": 30,  # 每30秒一段
                "resolution": 224,
                "use_temporal_model": True,
                "merge_strategy": "attention_pooling",  # 段间融合策略
                "description": "长视频模式 - 分段+关键帧",
            }
        else:
            # 超长视频：层次化处理
            return {
                "num_frames": 8,  # 每段的帧数
                "segment_duration": 60,
                "num_segments": min(int(duration_seconds / 60), 20),  # 最多20段
                "hierarchical": True,  # 先粗粒度扫描，再细粒度分析重点段落
                "resolution": 224,
                "description": "超长视频模式 - 层次化处理",
            }
    
    # 测试不同长度的视频
    test_durations = [3, 15, 120, 1800]  # 3秒、15秒、2分钟、30分钟
    
    print("视频长度自适应处理策略:")
    print("-" * 50)
    for dur in test_durations:
        config = get_processing_config(dur)
        print(f"\n{dur}秒视频 → {config.pop('description')}")
        for k, v in config.items():
            print(f"  {k}: {v}")


video_length_aware_processing()
```

## 陷阱三：视觉特征与文本不对齐

在视频-语言模型中，一个常见的问题是视觉特征和文本指令在语义层面"各说各话"。比如你问"视频中穿红衣服的人做了什么？"，模型可能正确识别出了动作，但把人搞混了——这是因为视觉特征中没有足够强的"身份绑定"信息。

```python
def visual_text_alignment_tips():
    """
    视觉-文本对齐的最佳实践
    """
    
    tips = """
    【提高视觉-文本对齐准确性的技巧】
    
    1. Prompt 中明确指代关系:
       ❌ "他在做什么？"  →  模型不知道"他"是谁
       ✅ "视频中左侧穿蓝色衬衫的男士正在做什么？"
       
    2. 利用空间锚点:
       ❌ "那个东西掉了"  →  有歧义
       ✅ "画面中央偏右位置掉落的物体是什么？"
       
    3. 分步提问代替复合问题:
       ❌ "穿白衣服的人和穿黑衣服的人分别做了什么，谁先完成的？"
       ✅ 先问: "穿白衣服的人做了什么？"
          再问: "穿黑衣服的人做了什么？"  
          最后问: "谁的动作先完成？"
          
    4. 提供上下文框架:
       在 prompt 中给出期望的回答结构，
       引导模型按逻辑顺序组织视觉信息
       
    5. 多角度交叉验证:
       同一个问题换个问法再问一遍，
       如果答案不一致，说明模型在该视频上的理解不够稳定
    """
    print(tips)


visual_text_alignment_tips()
```

## 性能优化：视频推理加速

视频理解模型的推理瓶颈主要来自两个方面：**视觉编码器的逐帧处理**和**大量视觉 token 送入 LLM 的开销**。下面是一些实用的优化策略。

## 策略一：帧级缓存与增量更新

如果两个视频查询请求针对的是同一个视频（只是问题不同），我们完全可以复用视觉编码的结果，避免重复计算。

```python
class VideoFeatureCache:
    """
    视频特征缓存系统
    
    核心思想: 同一个视频的视觉特征只需要计算一次，
    后续所有针对该视频的问答都可以直接复用
    """
    
    def __init__(self, max_cache_size: int = 100):
        self.cache: dict[str, torch.Tensor] = {}
        self.max_cache_size = max_cache_size
        self.access_order: list[str] = []  # LRU 追踪
    
    def _evict_if_needed(self):
        """LRU 淘汰策略"""
        while len(self.cache) >= self.max_cache_size:
            oldest_key = self.access_order.pop(0)
            del self.cache[oldest_key]
    
    def get_or_compute(
        self,
        video_id: str,
        compute_fn,  # callable that returns features
        **compute_kwargs,
    ) -> torch.Tensor:
        """
        获取视频特征（命中缓存则直接返回，否则计算并存入缓存）
        """
        if video_id in self.cache:
            # 更新访问顺序（移到末尾）
            self.access_order.remove(video_id)
            self.access_order.append(video_id)
            return self.cache[video_id]
        
        # 缓存未命中，执行计算
        features = compute_fn(**compute_kwargs)
        
        self._evict_if_needed()
        self.cache[video_id] = features
        self.access_order.append(video_id)
        
        return features
    
    def clear(self):
        """清空缓存"""
        self.cache.clear()
        self.access_order.clear()
    
    def stats(self) -> dict:
        """缓存统计信息"""
        return {
            "cached_videos": len(self.cache),
            "max_capacity": self.max_cache_size,
            "utilization": f"{len(self.cache)/self.max_cache_size*100:.1f}%",
        }


class OptimizedVideoLLMService:
    """
    优化后的视频 LLM 服务
    
    整合特征缓存 + 批量推理 + 异步预处理
    """
    
    def __init__(self, model_name: str, cache_size: int = 100):
        self.feature_cache = VideoFeatureCache(max_cache_size=cache_size)
        # 模型初始化...
    
    async def process_video_query(
        self,
        video_path: str,
        question: str,
    ) -> str:
        """
        处理视频查询请求（带缓存优化）
        """
        import hashlib
        
        # 用视频路径的哈希作为缓存 key
        video_id = hashlib.md5(video_path.encode()).hexdigest()[:12]
        
        # 获取或计算视觉特征（带缓存）
        video_features = self.feature_cache.get_or_compute(
            video_id=video_id,
            compute_fn=self._extract_video_features,
            video_path=video_path,
        )
        
        # 只有文本部分需要重新处理
        answer = self._generate_answer(
            video_features=video_features,
            question=question,
        )
        
        return answer
    
    def _extract_video_features(self, video_path: str) -> torch.Tensor:
        """提取视频特征（会被缓存的方法）"""
        # ... 实际的视觉编码逻辑 ...
        pass
    
    def _generate_answer(
        self,
        video_features: torch.Tensor,
        question: str,
    ) -> str:
        """基于缓存的视觉特征生成回答"""
        # ... LLM 推理逻辑 ...
        pass


OptimizedVideoLLMService
```

## 策略二：视觉 Token 压缩

对于高分辨率或长时间视频，视觉 token 的数量可能非常庞大。一种有效的做法是在送入 LLM 之前，用一个轻量级的模块对视觉 token 进行压缩。

```python
class VisualTokenCompressor(torch.nn.Module):
    """
    视觉 Token 压缩器
    
    目标: 将 (T*N, D_vision) 的视觉 token 压缩到 (K, D_llm)
    其中 K << T*N（通常 K=32~256），D_llm 是 LLM 的隐藏维度
    """
    
    def __init__(
        self,
        input_dim: int = 768,    # 视觉特征维度
        output_dim: int = 4096,  # LLM 维度
        num_output_tokens: int = 64,  # 压缩后的 token 数
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.num_output_tokens = num_output_tokens
        
        # 可学习的压缩 query（类似 Perceiver Resampler 的做法）
        self.compress_queries = torch.nn.Parameter(
            torch.randn(num_output_tokens, output_dim) * 0.02
        )
        
        # Cross-Attention: compress_queries attend to visual tokens
        self.cross_attention = torch.nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
            kdim=input_dim,
            vdim=input_dim,
        )
        
        # FFN
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(output_dim, output_dim * 4),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(output_dim * 4, output_dim),
            torch.nn.Dropout(dropout),
        )
        
        # Layer Norm
        self.layer_norm = torch.nn.LayerNorm(output_dim)
        
        # 投影层（如果输入输出维度不一致）
        if input_dim != output_dim:
            self.input_proj = torch.nn.Linear(input_dim, input_dim)
        else:
            self.input_proj = torch.nn.Identity()
    
    def forward(
        self,
        visual_tokens: torch.Tensor,  # (B, T*N, input_dim)
        attention_mask: torch.Tensor = None,  # (B, T*N)
    ) -> torch.Tensor:
        """
        压缩视觉 token
        
        Args:
            visual_tokens: 原始视觉 token 序列
            attention_mask: 可选的注意力掩码
        
        Returns:
            compressed_tokens: 压缩后的 token (B, num_output_tokens, output_dim)
        """
        B, N, D = visual_tokens.shape
        
        # 扩展压缩 queries 到 batch 维度
        queries = self.compress_queries.unsqueeze(0).expand(B, -1, -1)
        
        # Cross-Attention: queries 关注 visual_tokens
        projected_visual = self.input_proj(visual_tokens)
        
        compressed, _ = self.cross_attention(
            query=queries,
            key=projected_visual,
            value=projected_visual,
            key_padding_mask=attention_mask,
        )
        
        # FFN + Residual + LayerNorm
        compressed = compressed + self.ffn(compressed)
        compressed = self.layer_norm(compressed)
        
        return compressed


def compression_ratio_analysis():
    """
    分析不同配置下的压缩比率
    """
    
    configs = [
        {"name": "无压缩", "input_tokens": 1568, "output_tokens": 1568},
        {"name": "轻度压缩", "input_tokens": 1568, "output_tokens": 256},
        {"name": "中度压缩", "input_tokens": 1568, "output_tokens": 64},
        {"name": "重度压缩", "input_tokens": 1568, "output_tokens": 32},
        {"name": "极端压缩", "input_tokens": 1568, "output_tokens": 8},
    ]
    
    print("视觉 Token 压缩效率分析")
    print("=" * 55)
    print(f"{'配置':<12} {'输入Tokens':>12} {'输出Tokens':>12} {'压缩比':>8} {'信息损失风险':>12}")
    print("-" * 55)
    
    for cfg in configs:
        ratio = cfg["output_tokens"] / cfg["input_tokens"] * 100
        risk = "低" if ratio > 50 else "中" if ratio > 5 else "较高"
        print(f"{cfg['name']:<12} {cfg['input_tokens']:>12} {cfg['output_tokens']:>12} "
              f"{ratio:>7.1f}% {risk:>12}")


compression_ratio_analysis()
```

## 面试常考问题

**Q1: 视频理解和图像理解的本质区别是什么？**
视频理解多了时间维度，核心挑战在于建模时序依赖和运动信息。技术上体现为：输入从 `(B,C,H,W)` 变成 `(B,T,C,H,W)`，需要额外的时序建模模块（时间注意力、时间卷积、光流等），数据量需求更大（标注成本更高），计算量也更大（多帧并行处理）。

**Q2: 为什么 VideoMAE 可以用 95% 这么高的掩码率？**
因为视频相邻帧之间高度冗余。图像 MAE 遮住 75% 的 patch 后，模型只能靠空间上下文推断；但视频 MAE 遮住 95% 后，模型还可以靠时间维度上的前后帧信息来推断被遮住的内容。这也意味着视频的自监督信号比图像更丰富。

**Q3: Divided Space-Time Attention 相比 Joint Space-Time Attention 有什么优劣？**
Joint Attention 把所有帧所有 patch 展平做一次自注意力，复杂度是 `O((T×N)²)`，理论上表达能力最强但计算代价过高。Divided Attention 先做空间注意力（每帧内部，`O(N²)`，可并行化），再做时间注意力（跨帧同位置，`O(T²)`），总复杂度降为 `O(T×N² + N×T²)`，在实践中通常是更好的权衡。

**Q4: VideoLLaVA 如何处理视频的多帧输入？**
基本思路是"每帧独立编码 → 拼接/聚合 → 投影到 LLM 空间"。关键设计决策在于聚合方式：直接拼接（保留最多信息但 token 多）、平均池化（高效但丢失时序细节）、时序 Transformer（效果好但增加参数量）。生产环境中常用的是 Q-Former 或 Perceiver Resampler 这类轻量级聚合模块。

**Q5: 如何选择合适的视频帧采样数量？**
取决于任务类型和视频特点。动作识别通常 8-16 帧足够（Kinetics benchmark 的标准设置）；视频描述和问答建议 16-32 帧以捕捉更多细节；实时应用（如监控）可能只需 4-8 帧以保证延迟。关键是采样策略要与模型预训练时的设置一致，随意改变帧数会导致性能下降。
