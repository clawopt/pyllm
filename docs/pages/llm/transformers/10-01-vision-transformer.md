# 视觉 Transformer：ViT、SigLIP、DiT 与 SAM

## 这一节讲什么？

Transformer 最初是为文本设计的（"Attention Is All You Need", 2017），但它的强大远不止于 NLP。2021 年的一篇论文 **"An Image is Worth 16x16 Words"**（ViT）彻底改变了计算机视觉领域——它证明了：**只要把图片切成一个个小方块（patch），每个 patch 当作一个 token，就可以直接用标准的 Transformer 来处理图像！**

这一节我们将学习：
1. **ViT 的核心思想**——从 CNN 到纯 Transformer 的范式转移
2. **Vision Transformer 变体**——Swin、DeiT、DiT 等重要改进
3. **视觉-语言模型**——CLIP、SigLIP 如何连接图像和文本
4. **分割一切模型 (SAM)**——Meta 的分割革命
5. **HF 实战**——使用 ViT 进行图像分类和特征提取

---

## 一、从 CNN 到 Vision Transformer：为什么需要改变？

## 1.1 CNN 的局限性

```python
def cnn_limitations():
    """CNN 在视觉任务上的局限性"""

    limitations = [
        {
            "局限": "局部感受野",
            "解释": "CNN 的卷积核只能看到局部区域。"
                   "虽然深层网络通过堆叠可以扩大感受野,"
                   "但每一层仍然是局部的, 需要很多层才能建立全局联系",
            "类比": "就像用放大镜看画, 每次只看到一小块",
        },
        {
            "局限": "归纳偏置过强",
            "解释": "CNN 天然假设了平移不变性、局部性等先验知识。"
                   "这些在自然图像上通常成立, 但在某些场景下反而限制了模型能力"
                   "(如需要理解全局结构的任务)",
            "类比": "CNN 就像一个有固定思维模式的人, 很难跳出框架",
        },
        {
            "局限": "难以建模长距离依赖",
            "解释": "两个角落在图像对角线上的像素, "
                   "在 CNN 中需要经过很多层才能'看到对方'",
            "信息传递效率低",
            "类比": "传声筒游戏, 距离越远声音越小",
        },
        {
            "局限": "数据效率问题",
            "解释": "CNN 需要 ImageNet 级别的大规模数据才能训练好 "
                   "(数百万张标注图片)。小数据集上效果明显不如人类",
            "类比": "小孩看几张图就能学会认猫狗, CNN 却要百万张",
        },
    ]

    print("=" * 70)
    print("CNN 的四大局限性")
    print("=" * 70)
    for l in limitations:
        print(f"\n⚠️  {l['局限']}")
        print(f"   原因: {l['解释']}")
        print(f"   类比: {l['类比']}")

cnn_limitations()
```

## 1.2 ViT 的核心洞察：图片 = 一串 Token

```python
def vit_core_idea():
    """ViT 的核心思想"""

    idea = """
┌───────────────────────────────────────────────────────────────┐
│                  ViT: 图片即序列                             │
├───────────────────────────────────────────────────────────────┤
│                                                              │
│   输入图像 (如 224×224 RGB):                               │
│                                                              │
│   ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐   │
│   │ ▓ │ ▓ │ ▓ │ ▓ │ ▓ │ ▓ │ ▓ │ ▓ │ ▓ │ ▓ │ ▓ │ ▓ │ ▓ │   │
│   ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤   │
│   │ ▓ │ ▓ │ ▓ │ ▓ │ ▓ │ ▓ │ ▓ │ ▓ │ ▓ │ ▓ │ ▓ │ ▓ │ ▓ │   │
│   ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤   │
│   │...│   │   │   │   │   │   │   │   │   │   │   │   │   │
│   └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘   │
│                                                              │
│   Step 1: 切分成 Patch (Patch)                              │
│     • 每个 Patch 大小: 16×16 像素                           │
│     • 总共: (224/16)² = 196 个 Patch                         │
│     • 每个 Patch 展平为向量: 16×16×3 = 768 维               │
│                                                              │
│   Step 2: 类似于 Word Embedding, 加上位置编码              │
│     • Patch Embedding: 将每个 768 维向量映射到 hidden_dim    │
│     • Position Embedding: 告诉模型每个 Patch 的位置           │
│     • [CLS] token: 可学习的特殊 token, 用于分类                 │
│                                                              │
│   Step 3: 送入标准 Transformer Encoder!                    │
│     • 和 BERT/GPT 用的是完全一样的架构!                      │
│     • Multi-Head Self-Attention + FFN × N 层                │
│                                                              │
│   Step 4: 取 [CLS] 位置的输出做分类                            │
│                                                              │
│   关键公式:                                                  │
│     z₀ = [CLS]; zᵢ = [Patch₁; ...; Patchₙ] + E_pos             │
│     z'ₗ = Transformer(z')                                   │
│     y = MLP(z'_L)                                           │
│                                                              │
└───────────────────────────────────────────────────────────────┘

💡 为什么叫 "An Image is Worth 16x16 Words"? 
   → 因为每个 16×16 的 patch 就像 NLP 中的一个 word/token!
   → 224×224 的图 → 196 个 tokens → 和一个中等长度的句子差不多!
"""
    print(idea)

vit_core_idea()
```

## 1.3 ViT vs CNN：直观对比

```python
def vit_vs_cnn_comparison():
    """ViT vs CNN 直观对比"""

    comparison = """
╔════════════════════════════════════════════════════════════╗
║                    Vision Transformer vs CNN                    ║
╠════════════════════════════════════════════════════════════╣
║                                                              ║
║  维度                    CNN                          ViT          ║
║  ───────────────────────────────────────────────────────────  ║
║  全局感受野              需要很深的网络              天然具备       ║
║  长距离建模              困难 (信息逐层衰减)         直接 Attention ║
║  归纳偏置               强 (平移不变等)               弱 (数据驱动)   ║
║  计算方式                滑动窗口 (局部)                全局 Self-Attn ║
║  数据需求                大量数据 (ImageNet)           可少可多       ║
║  可解释性                较差 (特征图难解读)           Attention Map ║
║  迁移能力                差 (需重新训练)              好 (预训练强)  ║
║                                                              ║
║  适用场景:                                                   ║
║  • 图像分类              ✅ CNN 仍主流 (效率高)      ✅ ViT 有竞争力  ║
║  • 目标检测              ✅ YOLO/Faster R-CNN       ⚠️ DETR 可行     ║
║  • 语义分割              ✅ U-Net/DeepLab           ✅ SETR/SAM     ║
║  • 多模态理解              ❌ 不擅长                     ✅ CLIP/VLM   ║
║  • 视觉问答/推理          ❌ 不支持                     ✅ LLaVA 等     ║
║                                                              ║
╚════════════════════════════════════════════════════════════╝
"""
    print(comparison)

vit_vs_cnn_comparison()
```

---

## 二、ViT 架构详解与 HF 实战

## 2.1 ViT 模型结构拆解

```python
from transformers import ViTModel, ViTConfig, ViTImageProcessor
import torch


def vit_architecture_deep_dive():
    """ViT 架构深度解析"""

    # 加载 ViT 配置
    config = ViTConfig.from_pretrained("google/vit-base-patch16-224")
    
    print("=" * 65)
    print("ViT-Base 架构配置解析")
    print("=" * 65)
    
    key_params = [
        ("hidden_size (D)", config.hidden_size, "Transformer 隐藏维度"),
        ("num_hidden_layers (L)", config.num_hidden_layers, "Encoder 层数"),
        ("num_attention_heads (H)", config.num_attention_heads, "注意力头数"),
        ("intermediate_size (FFN)", config.intermediate_size, "FFN 中间层维度"),
        ("image_size", config.image_size, "输入图像尺寸"),
        ("patch_size", config.patch_size, "每个 Patch 的大小"),
        ("num_patches", (config.image_size // config.patch_size)**2, "Patch 数量 (= 序列长度)"),
        ("num_channels", config.num_channels, "输入通道数 (RGB=3)"),
    ]

    for name, value, desc in key_params:
        print(f"  {name:<30} {value:<10}  ({desc})")

    print(f"\n📐 关键尺寸关系:")
    print(f"   head_dim = D / H = {config.hidden_size} / {config.num_attention_heads}")
    print(f"   FFN 扩展比 = intermediate / D = {config.intermediate_size} / {config.hidden_size:.1f}x")
    print(f"   Patch 数量 = ({config.image_size}/{config.patch_size})² = {(config.image_size // config.patch_size)**2}")

vit_architecture_deep_dive()
```

## 2.2 使用 ViT 进行图像分类

```python
def vit_classification_demo():
    """使用 ViT 进行图像分类的完整流程"""

    from transformers import ViTForImageClassification, AutoImageProcessor
    import urllib.request
    from PIL import Image
    
    print("=" * 60)
    print("ViT 图像分类实战")
    print("=" * 60)

    # 加载模型和处理器
    model_name = "google/vit-base-patch16-224"
    model = ViTForImageClassification.from_pretrained(model_name)
    processor = AutoImageProcessor.from_pretrained(model_name)

    print(f"\n[1] 加载模型: {model_name}")
    print(f"    标签数: {model.config.num_labels} (ImageNet-1K)")
    print(f"    参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 加载示例图像 (使用 URL)
    url = "http://images.cocodataset.org/2017/val/000000039769.jpg"
    try:
        urllib.request.urlretrieve(url, "cat.jpg")
        image = Image.open("cat.jpg").convert("RGB")
        
        print(f"\n[2] 加载测试图像: {image.size}")

        # 预处理
        inputs = processor(images=image, return_tensors="pt")

        # 推理
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_idx = logits.argmax(-1).item()
            score = torch.softmax(logits, dim=-1)[0, predicted_idx].item()
            label = model.config.id2label[predicted_idx]

        print(f"\n[3] 分类结果:")
        print(f"    预测标签: {label}")
        print(f"    置信度: {score:.4f}")
        
        # 显示 Top-5 预测
        top5 = torch.topk(logits.softmax(dim=-1)[0], 5)
        print(f"\n    Top-5 预测:")
        for idx, prob in zip(top5.indices.tolist(), top5.values.tolist()):
            print(f"      {model.config.id2label[idx]}: {prob:.4f}")

    except Exception as e:
        print(f"\n⚠️  图像加载失败: {e}")
        print("   请确保网络连接正常, 或使用本地图像")

vit_classification_demo()
```

## 2.3 提取 ViT 特征用于下游任务

```python
def vit_feature_extraction():
    """提取 ViT 中间层特征用于下游任务"""

    from transformers import ViTModel, AutoImageProcessor
    from PIL import Image
    import torch

    print("=" * 60)
    print("ViT 特征提取实战")
    print("=" * 60)

    model = ViTModel.from_pretrained("google/vit-base-patch16-224")
    processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")

    # 创建一张虚拟图像
    dummy_image = Image.new("RGB", (224, 224), (255, 255, 255))
    inputs = processor(images=dummy_image, return_tensors="pt")

    # 提取特征 (获取所有层的隐藏状态)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    last_hidden_state = outputs.last_hidden_state  # (1, 197, 768)
    all_hidden_states = outputs.hidden_states  # tuple of 13 tensors

    print(f"\n[CLS] token 特征:")
    print(f"    形状: {last_hidden_state.shape}")  # (1, 197, 768)
    print(f"    用途: 线性分类/检索的图像特征向量")

    print(f"\n所有中间层隐藏状态:")
    print(f"    共 {len(all_hidden_states)} 层 (embeddings + 12 encoder layers)")
    for i, h in enumerate(all_hidden_states):
        print(f"      Layer {i}: {h.shape}")

    print(f"\n💡 特征提取的应用场景:")
    print(f"   • 图像检索 (以图搜图): 用 [CLS] 特征计算相似度")
    print(f"   • 视觉问答 (VQA): 将图像特征与文本特征做 Cross-Attention")
    print(f"   • 目标检测/分割: 将 Patch 特征解码为位置级预测")
    print(f"   • 多模态对齐: 与 CLIP/SigLIP 的文本特征对齐")

vit_feature_extraction()
```

---

## 三、ViT 变体家族

## 3.1 重要变体一览

```python
def vit_variants():
    """Vision Transformer 重要变体"""

    variants = [
        {
            "名称": "ViT (原始)",
            "年份": "2020",
            "改进": "首次将纯 Transformer 用于图像分类",
            "关键创新": "Patch Embedding + Position Encoding",
            "局限性": "需要大量预训练数据; 小数据集效果不如 CNN",
        },
        {
            "名称": "DeiT (Data-efficient ViT)",
            "年份": "2021",
            "改进": "用知识蒸馏让 ViT 在 ImageNet-1K 上超越 CNN",
            "关键创新": "Teacher (CNN/Ti-ViT) + Student (ViT) 蒸馏;"
                   "Token 标签化辅助蒸馏",
            "优势": "证明 ViT 可以在小数据上达到 SOTA",
        },
        {
            "名称": "Swin Transformer",
            "年份": "2021",
            "改进": "层次化窗口注意力 (Hierarchical Window Attention)",
            "关键创新": "不再全局 Self-Attention, 改为局部 Window 内 Attention;"
                   "Shifted Window 实现跨窗口连接;"
                   "类似 CNN 的金字塔结构 (分辨率逐层降低)",
            "优势": "计算复杂度 O(N×W) 而非 O(N²); 成为视觉主干网络新选择",
        },
        {
            "名称": "MAE (Masked Autoencoder)",
            "年份": "2021",
            "改进": "视觉领域的 BERT 式预训练 (自监督)",
            "关键创新": "随机 Mask 75% 的 Patch, 让模型重建被遮挡的部分;"
                   "不需要任何标注数据即可完成预训练!",
            "意义": "证明了自监督学习在视觉领域的巨大潜力",
        },
        {
            "名称": "DiT (Diffusion Transformer)",
            "年份": "2023",
            "改进": "将 Transformer 用于扩散模型的 U-Net 骨干",
            "关键创新": "用 ViT 替代 CNN 做 UNet 的 encoder/decoder;"
                   "配合 Classifier-Free Guidance 实现高质量生成",
            "成就": "Stable Diffusion / DALL-E 3 / Sora 的基础架构",
        },
        {
            "名称": "SAM (Segment Anything Model)",
            "年份": "2023",
            "改进": "统一的零样本/少样本分割模型",
            "关键创新": "Promptable Encoder (图像编码器) + Mask Decoder;"
                   "Prompt 可以是点、框、文本或掩码;"
                   "SA-1B 数据集 (1100 万张 mask)",
            "影响": "CV 领域的 GPT-3 时刻; 分割任务大一统",
        },
    ]

    print("=" * 75)
    print("Vision Transformer 重要变体")
    print("=" * 75)
    for v in variants:
        print(f"\n{'🔷 '}{v['名称']}")
        for k, val in v.items():
            if k != "名称":
                print(f"   {k}: {val}")

vit_variants()
```

---

## 四、视觉-语言多模态模型：CLIP 与 SigLIP

## 4.1 CLIP：连接图像与文本的桥梁

CLIP (Contrastive Language-Image Pre-training) 是 OpenAI 于 2021 年发布的里程碑式工作，它开创了**对比学习**连接视觉和语言的范式：

```python
def clip_intuition():
    """CLIP 的核心直觉"""

    intuition = """
┌───────────────────────────────────────────────────────────────┐
│                    CLIP: 对比学习连接图文                       │
├───────────────────────────────────────────────────────────────�
│                                                              │
│  核心思想:                                                    │
│  不需要人工标注 "这是一只猫"!                                 │
│  只需要知道哪些图片和哪些文字经常一起出现!                   │
│                                                              │
│  训练数据 (来自互联网):                                        │
│  Pair 1: ["A photo of a cat", 🐱图片]  ← 正确配对 (positive)    │
│  Pair 2: ["A photo of a dog", 🐱图片]  ← 错误配对 (negative)   │
│  Pair 3: ["A beautiful sunset", 🌅图片]  ← 正确配对 (positive)    │
│  ...                                                        │
│                                                              │
│  训练目标:                                                    │
│  对于正确配对的 (image, text) pair:                              │
│    → 让 image embedding 和 text embedding 在向量空间中尽量接近        │
│  对于错误配对的 pair:                                          │
│    → 让它们的 embedding 尽量远离                                  │
│                                                              │
│  Loss 函数: InfoNCE Loss (Info Noise Contrastive Estimation)  │
│  L = -log[exp(sim(img, txt_pos) / Σ exp(sim(img, txt_all))]      │
│                                                              │
│  结果:                                                      │
│  • Image Encoder 学到了丰富的视觉语义表示                        │
│  • Text Encoder 学到了能描述图像的语言表示                        │
│  • 两者共享同一个嵌入空间 → 可以直接计算图文相似度              │
│                                                              │
│  零样本能力:                                                  │
│  "A photo of [new concept]" → 即使没见过这个概念,                 │
│  CLIP 也能给出合理的匹配结果 (因为学会了通用的视觉-语言对齐)    │
│                                                              │
└───────────────────────────────────────────────────────────────┘
"""
    print(intuition)

clip_intuition()
```

## 4.2 CLIP / SigLIP 实战

```python
def clip_demo():
    """CLIP / SigLIP 多模态推理实战"""

    try:
        from transformers import CLIPModel, CLIPProcessor, CLIPTextModel, CLIPTokenizer
        from PIL import Image
        import torch

        print("=" * 60)
        print("CLIP / SigLIP 多模态实战")
        print("=" * 60)

        # 方式 1: CLIP Pipeline (最简单)
        from transformers import pipeline
        
        pipe = pipeline("zero-shot-image-classification",
                        model="openai/clip-vit-large-patch14")
        
        results = pipe(
            "a photo of a golden retriever playing in the snow",
            candidate_labels=["dog", "cat", "bird", "retriever"],
            images=["https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/Golden_Retriever_tivia8.jpg/220px-Golden_Retriever_tivia8.jpg"],
        )
        
        print("\n[CLIP Zero-Shot Classification]:")
        for r in results:
            print(f"  {r['label']}: {r['score']:.4f}")

        # 方式 2: 提取特征计算相似度
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # 编码图像
        image = processor(
            images=["https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/Golden_Retriever_tivia8.jpg/220px-Golden_Retriever_tivia8.jpg"],
            return_tensors="pt",
        ).pixel_values
        
        # 编码文本
        text = tokenizer(["a dog running in the park", "a cat sleeping on a sofa"], 
                       padding=True, truncation=True, return_tensors="pt")
        
        with torch.no_grad():
            image_features = model.get_image_features(image)
            text_features = model.get_text_features(text)
            
            # 计算余弦相似度
            similarity = (image_features @ text_features.T).softmax(dim=-1)
        
        print("\n[CLIP 相似度矩阵]:")
        print(f"  'dog running...' ↔ 🐱: {similarity[0][0]:.4f}")
        print(f"  'dog running...' ↔ 🐱: {similarity[0][1]:.4f}")
        print(f"  'cat sleeping...' ↔ 🐱: {similarity[1][0]:.4f}")
        print(f"  'cat sleeping...' ↔ 🐱: {similarity[1][1]:.4f}")

    except ImportError as e:
        print(f"\n⚠️ 部分库未安装: {e}")
        print("   运行: pip install transformers Pillow torch")

clip_demo()
```

---

## 五、本章小结

这一节我们学习了视觉 Transformer 的完整生态：

| 模型 | 类型 | 核心贡献 | 应用 |
|------|------|---------|------|
| **ViT** | 分类 | 首次用纯 Transformer 处理图像 | 图像分类、特征提取 |
| **DeiT** | 分类 | 知识蒸馏让 ViT 在小数据上超越 CNN | 高效 ViT 训练 |
| **Swin** | 主干网络 | 层次化窗口注意力，O(NW) 复杂度 | 新一代视觉骨干网络 |
| **MAE** | 预训练 | 自监督掩码重建，无需标注数据 | 自监督视觉预训练 |
| **DiT** | 生成 | ViT 作为 Diffusion U-Net | Stable Diffusion 基础 |
| **SAM** | 分割 | Promptable 统一分割模型 | 万物皆可分割 |
| **CLIP** | 多模态 | 对齐图文到同一空间 | 零样本分类、跨模态检索 |

**核心要点**：
1. **ViT 的革命性在于**：把图像切成 patch 当 token，直接套用 NLP 的 Transformer 架构
2. **CNN 仍有其价值**：在纯视觉任务（分类/检测）上效率更高；但多模态时代 ViT 系列更通用
3. **CLIP 是多模态的基石**：对比学习让模型学会了对齐图文语义
4. **当前趋势是"大统一"**：一个基础模型同时处理文本、图像、视频、音频

下一节我们将学习 **音频/语音模型**——Whisper、Wav2Vec2 等。
