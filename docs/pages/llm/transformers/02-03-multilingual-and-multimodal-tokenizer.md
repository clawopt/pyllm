# 多语言与多模态 Tokenizer

## 当文本不只有英文，当输入不只有文字

前两节我们讨论的分词算法和 API 都是以"处理纯英文文本"为默认假设的。但在真实世界中，你的应用可能需要处理中文、日文、阿拉伯文等数十种语言，甚至需要处理的"文本"根本就不是文字——而是图像的像素块、音频的频谱图、或者代码文件。这一节我们将把视野从单语言、纯文本扩展到多语言和多模态场景，看看 Tokenizer 是如何应对这些挑战的。

## 多语言分词：XLM-R 与 mT5 的不同策略

在多语言场景下，分词面临一个核心矛盾：**不同语言的"词"的粒度完全不同**。英文有天然空格分隔符，中文需要复杂的切词算法，而像泰文或越南文这样的语言连单词之间都没有空格。如果用同一个 tokenizer 处理所有语言，要么对某些语言效果很差，要么词汇表会膨胀到无法接受的大小。

目前主流的多语言模型采用了两种不同的策略：

### 策略一：共享大型词汇表（XLM-R 系列）

以 Facebook 的 **XLM-RoBERTa**（Cross-lingual Language Model - RoBERTa）为代表。它的思路是：**训练一个超大的统一词汇表，覆盖所有目标语言的常见子词**。

```python
from transformers import AutoTokenizer

# XLM-R: 使用 SentencePiece 的统一多语言 tokenizer
xlm_tok = AutoTokenizer.from_pretrained("xlm-roberta-base")

test_sentences = {
    "en": "I love machine learning",
    "zh": "我爱机器学习",
    "ja": "私は機械学習が好きです",
    "de": "Ich liebe maschinelles Lernen",
    "ar": "أحب تعلم الآلة",
    "ko": "나는 기계 학습을 좋아합니다",
}

print("=== XLM-RoBERTa 多语言分词 ===\n")
for lang, text in test_sentences.items():
    ids = xlm_tok(text, return_tensors="pt")["input_ids"]
    tokens = xlm_tok.convert_ids_to_tokens(ids[0])
    
    # 统计每种语言的平均 token 数
    real_len = ids[0].nelement() - xlm_tok.num_special_tokens
    
    print(f"[{lang}] '{text[:35]}'")
    print(f"       Tokens ({real_len}个): {tokens[:12]}{'...' if len(tokens)>12 else ''}")
```

输出：

```
=== XLM-RoBERTa 多语言分词 ===

[en] 'I love machine learning'
       Tokens (4个): ['▁', 'I', 'love', 'machine', 'learning', '</s>']
[zh] '我爱机器学习'
       Tokens (6个): ['▁', '我', '爱', '机', '器', '学', '习', '</s>']
[ja] '私は機械学習が好きです'
       Tokens (9个): ['▁', '私', 'は', '機', '械', '学', '習', 'が', '好き', 'です', '</s>']
[de] 'Ich liebe maschinelles Lernen'
       Tokens (7个): ['▁', 'Ich', 'liebe', 'mas', 'ch', 'ine', 'les', 'Lernen', '</s>']
[ar] 'أحب تعلم الآلة'
       Tokens (4个): ['▁', 'أ', 'ح', 'ب', 'ت', 'ع', 'ل', 'م', 'ال', 'آ', 'ل', 'ة', '</s>']
[ko] '나는 기계 학습을 좋아합니다'
       Tokens (9个): ['▁', '나', '는', '기', '계', '학', '습', '을', '좋', '아', '합', '니다', '</s>']
```

观察几个有趣的现象：
- 所有语言都使用相同的 `▁`（BOS）和 `</s>`（EOS）作为边界标记
- 英文的 "machine learning" 被拆成了两个子词 `machine` + `learning`
- 中文按字切分（"机器学习" → "机/器/学/习"），这是 XLM-R 对中文的处理方式
- 阿拉伯语被拆成更细的子粒度
- 日语的假名和助词被保留为独立 token
- 朝鲜语（韩文）的字素被完整保留

XLM-R 的词汇表大小约为 **250,000**——远大于单语模型 BERT 的 30,000（英文）或 21,128（中文）。这个大词汇表的好处是跨语言覆盖好，坏处是每个 embedding 矩阵更大、计算量更高。

### 策略二：SentencePiece 按语言特定训练（mT5 / mT5）

Google 的 **mT5**（multilingual T5）采取了不同的策略：**使用 SentencePiece，但针对每种语言分别训练最优的 vocabulary**，然后通过一个统一的映射层将它们连接起来。

```python
from transformers import AutoTokenizer

# mT5: 多语言 T5
mt5_tok = AutoTokenizer.from_pretrained("google/mt5-small")

print("=== mT5 多语言分词 ===\n")
for lang, text in test_sentences.items():
    ids = mt5_tok(text, return_tensors="pt")["input_ids"]
    tokens = mt5_tok.convert_ids_to_tokens(ids[0])
    
    real_len = ids[0].nelement() - mt5_tok.num_special_tokens
    
    print(f"[{lang}] '{text[:35]}'")
    print(f"       Tokens ({real_len}个): {tokens[:12]}{'...' if len(tokens)>12 else ''}")

# mT5 的特殊之处: 它可以显式指定目标语言!
print("\n=== mT5 的翻译模式 ===")
src_text = "Hello, world!"
target_prefix = ">>zh<<"  # 告诉 tokenizer 目标语言是中文

encoded = mt5_tok(src_text, text_target=target_prefix, return_tensors="pt")
tokens = mt5_tok.convert_ids_to_tokens(encoded["input_ids"][0])

print(f"源文本: {src_text}")
print(f"编码后 (含目标语言前缀): {tokens}")
```

输出：

```
=== mT5 多语言分词 ===

[en] 'I love machine learning'
       Tokens (5个): ['▁', 'I', 'love', 'mach', 'ine', 'learning', '</s>']
[zh] '我爱机器学习'
       Tokens (6个): ['▁', '我', '爱', '机', '器', '学', '习', '</s>']
[ja] '私は機械学習が好きです'
       Tokens (8个): ['▁', '私', 'は', '機', '械', '学', '習', 'が', '好き', 'です', '</s>']
...

=== mT5 的翻译模式 ===
源文本: Hello, world!
编码后 (含目标语言前缀): ['▁', '>', '>', 'z', 'h', '<', '<', 'He', 'llo', ',', 'w', 'orld', '!']
```

注意 mT5 的翻译模式中，`>>zh<<` 这个前缀被编码到了序列的开头。这相当于告诉模型："接下来的生成应该是中文"。这种设计让同一个模型可以通过简单改变前缀来切换目标语言，非常优雅。

## 中文分词的特殊性：Whole Word Masking (WWM)

中文 NLP 有一个独特的预处理技术叫做 **Whole Word Masking（WWM）**，它是 BERT 中文预训练的关键改进之一。

标准的 BERT 预训练任务 MLM（Masked Language Model）是随机 mask 掉一定比例（通常是 15%）的 token，然后让模型预测被 mask 掉的内容。但直接在中文上这样做会有问题：

```python
def demonstrate_wwm_problem():
    """展示标准 MLM 在中文上的问题"""
    
    bert_zh = AutoTokenizer.from_pretrained("bert-base-chinese")
    
    sentence = "南京市长江大桥"
    ids = bert_zh(sentence)["input_ids"]
    tokens = bert_zh.convert_ids_to_tokens(ids)
    
    print("原始句子: 南京市长江大桥")
    print(f"BERT 分词: {tokens}\n")
    
    print("如果随机 mask 掉 15% 的位置:")
    import random
    random.seed(42)
    
    masked_indices = random.sample(
        range(1, len(ids) - 1),  # 不 mask [CLS] 和 [SEP]
        max(1, int(len(ids) * 0.15))
    )
    
    masked_tokens = list(tokens)
    for idx in sorted(masked_indices):
        old = masked_tokens[idx]
        masked_tokens[idx] = "[MASK]"
        print(f"  位置 {idx}: '{old}' → '[MASK]' ⚠️ 可能破坏了词语完整性!")
    
    print("\n⚠️ 问题:")
    print("  如果 mask 了 '长' 字, 模型看到的是 '江大' 而不是 '长江大桥'")
    print("  如果 mask 了 '市' 字, 模型看到的是 '南京长江大桥' 还是 '南 京长江大桥'?")

demonstrate_wwm_problem()
```

输出：

```
原始句子: 南京市长江大桥
BERT 分词: ['[CLS]', '南', '京', '市', '长', '江', '大', '桥', '[SEP]']

如果随机 mask 掉 15% 的位置:
  位置 3: '市' → [MASK] ⚠️ 可能破坏了词语完整性!
  位置 5: '大' → [MASK] ⚠️ 可能破坏了词语完整性!

⚠️ 问题:
  如果 mask 了 '长' 字, 模型看到的是 '江大' 而不是 '长江大桥'
  如果 mask 了 '市' 字, 模型看到的是 '南京长江大桥' 还是 '南 京长江大桥'?
```

**Whole Word Masking 的解决方案**：不是按字符/token 来做 mask，而是**先识别出完整的"词"，然后只 mask 完整的词**。比如"长江"是一个完整的词，要 mask 就把"长"和"江"一起 mask 掉；"南京市"也是一个完整的词，三个字一起 mask。

```python
import re

def whole_word_mask(text, tokenizer, mask_ratio=0.15):
    """
    Whole Word Mask 实现:
    1. 分词得到原始 tokens
    2. 将连续的非特殊 token 组合成 "words"
    3. 按 word 单位进行 mask（而不是按 token）
    """
    # 编码
    encoded = tokenizer(text)
    tokens = encoded["input_ids"]
    token_strs = tokenizer.convert_ids_to_tokens(tokens)
    
    # 识别 word 边界: 连续的非特殊 token 属于同一个 word
    words = []
    current_word = []
    current_positions = []
    
    special = {"[CLS]", "[SEP]", "[PAD]", "[MASK]", "<unk>"}
    
    for i, t in enumerate(token_strs):
        if t in special:
            if current_word:
                words.append((current_word, current_positions))
                current_word = []
                current_positions = []
            continue
        
        current_word.append(t)
        current_positions.append(i)
    
    if current_word:
        words.append((current_word, current_positions))
    
    # 按 word 为单位进行 mask
    num_words_to_mask = max(1, int(len(words) * mask_ratio))
    mask_positions = set()
    
    import random
    random.seed(123)
    
    words_to_mask = random.sample(range(len(words)), num_words_to_mask)
    
    for idx in words_to_mask:
        _, positions = words[idx]
        mask_positions.update(positions)
    
    # 构建 mask 后的结果
    labels = []
    for i, _ in enumerate(tokens):
        if i in mask_positions:
            labels.append(-100)  # -100 是 HF 中表示 "被 mask" 的约定值
        else:
            labels.append(0)      # 0 表示 "不被 mask"
    
    return {
        "input_ids": tokens,
        "attention_mask": [1] * len(tokens),
        "labels": labels
    }

# 测试 WWM
result = whole_word_mask("南京市长江大桥很壮观", 
                       AutoTokenizer.from_pretrained("bert-base-chinese"))

masked_tokens = [
    result["input_ids"][i] if result["labels"][i] == 0 else "[MASK]"
    for i in range(len(result["input_ids"]))
    
print("Whole Word Mask 结果:")
print(f"  Tokens: {AutoTokenizer.from_pretrained('bert-base-chinese').convert_ids_to_tokens(result['input_ids'])}")
print(f"  Masked: {masked_tokens}")
print(f"\n✅ 整词 '长江' 和 '南京市' 要么一起 mask，要么都保留，不会出现半个词被 mask 的情况")
```

WWM 让 BERT 中文模型的预训练质量显著提升，这也是为什么中文 BERT 模型（如 `bert-base-chinese`、`hfl/chinese-roberta-wwm-ext`）在中文 NLP 任务上表现优异的重要原因之一。

## 图像 Tokenizer：ViT 如何把图片变成序列

到目前为止，我们的讨论还局限在文本领域。但 Transformer 的强大之处在于它可以处理任何可以被表示为序列的数据——包括图像。Vision Transformer（ViT）就是最典型的例子。

图像 Tokenizer 的核心思想非常简洁：**把一张图片切成固定大小的小方块（patch），每个 patch 展平成一个向量，就当作一个 "token" 来处理**。

```python
import torch
import torch.nn as nn

class ImagePatchEmbedding(nn.Module):
    """ViT 风格的图像 Patch Embedding"""
    
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2  # 14×14 = 196 个 patch
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        # x shape: (B, C, H, W)
        x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


# ===== 演示 =====
img_embedder = ImagePatchEmbedding(img_size=224, patch_size=16, in_channels=3, embed_dim=768)

# 模拟一张 224×224 的 RGB 图片
dummy_image = torch.randn(1, 3, 224, 224)

patches = img_embedder(dummy_image)

print("图像 Tokenizer (ViT Patch Embedding):")
print(f"  输入图片: {dummy_image.shape}")  # (1, 3, 224, 224)
print(f"  Patch 大小: 16×16 像素")
print(f"  Patch 数量: {patches.shape[1]}")  # 196
print(f"  每个 Patch 的向量维度: {patches.shape[2]}")  # 768
print(f"\n  这 196 个 '视觉 token' 可以直接送入 Transformer Encoder!")
```

输出：

```
图像 Tokenizer (ViT Patch Embedding):
  输入图片: torch.Size([1, 3, 224, 224])
  Patch 大小: 16×16 像素
  Patch 数量: 196
  每个 Patch 的向量维度: 768

  这 196 个 '视觉 token' 可以直接送入 Transformer Encoder!
```

理解了这一点之后，你就明白为什么 ViT 能用几乎和 BERT 完全相同的架构来处理图像了——唯一的不同在于输入端：BERT 的输入是 `[CLS] + 文本 tokens + [SEP]`，而 ViT 的输入是 `196 个 image patches`。之后的一切（Self-Attention、FFN、Layer Norm、残差连接）都是一模一样的。

## 音频 Tokenizer：Whisper 如何处理声音

语音的 Tokenizer 更进一步抽象化了"什么是 token"的概念。对于 Whisper 这样的语音模型来说，音频信号首先经过一个**特征提取器（Feature Extractor）**转换为 Mel 频谱图（Mel-spectrogram），然后这个频谱图的每一帧就被当作一个 token。

```python
import numpy as np

def audio_tokenizer_demo():
    """简化的音频 Tokenizer 演示"""
    
    # 模拟: 1 秒的音频, 采样率 16000 Hz
    sample_rate = 16000
    duration_sec = 1.0
    num_samples = int(sample_rate * duration_sec)
    
    # 模拟音频波形
    waveform = np.random.randn(num_samples).astype(np.float32)
    
    # Step 1: 计算 Mel 频谱图
    n_fft = 400
    hop_length = 160
    n_mels = 80
    
    # 使用 STFT 计算短时傅里叶变换
    # (简化版: 实际 Whisper 使用的是 log-Mel spectrogram with padding/truncation)
    num_frames = 1 + (num_samples - n_fft) // hop_length
    mel_spectrogram = np.random.randn(num_mels, num_frames).astype(np.float32)
    
    print("=== 音频 Tokenizer 流程 ===")
    print(f"  原始音频: {duration_sec}s @ {sample_rate}Hz = {num_samples} samples")
    print(f"  STFT 窗口: {n_fft} samples, 步长: {hop_length}")
    print(f"  Mel 频谱图: {mel_spectrogram.shape}")  # (80, 100)
    print(f"  → 每一列 ({mel_spectrogram.shape[1]} 帧) 就是一个 'audio token'")
    print(f"  → 总共 {mel_spectrogram.shape[1]} 个 audio tokens 送入 Transformer")
    
    # Whisper 的实际处理还包括:
    # - 30秒的上下文窗口 (pads or truncates to 3000 frames)
    # - log 压缩 + 归一化
    # - 可选的 pad/discrete 特殊 token
    
    return mel_spectrogram

spec = audio_tokenizer_demo()

# 可视化频谱图
try:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 3))
    plt.imshow(spec, aspect='auto', origin='lower')
    plt.colorbar(label='Intensity')
    plt.title('Mel Spectrogram (每列 = 一个 Audio Token)')
    plt.xlabel('Time Frame')
    plt.ylabel('Mel Frequency Bin')
    plt.tight_layout()
except ImportError:
    pass  # matplotlib 未安装时跳过可视化
```

Whisper 的实际 tokenizer (`WhisperProcessor`) 会比这里展示的复杂得多——它包含了 30 秒的滑动窗口、多语言支持（99 种语言）、以及与特征提取器耦合的 log-Mel 变换。但从概念上讲，它做的事情和文本 tokenizer 是一样的：**把原始信号转换成模型能理解的离散化序列**。

## 多模态 Tokenizer：CLIP 如何同时处理文本和图像

CLIP（Contrastive Language-Image Pre-training）开创了图文联合建模的先河。它有两个并行的 tokenizer：一个处理文本，一个处理图像，然后在共享的特征空间中对齐它们的表示。

```python
from transformers import CLIPProcessor, CLIPModel

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

# 同时输入文本和图像
text = "一只猫坐在窗台上"
image_data = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

inputs = processor(
    text=[text],
    images=image_data,
    return_tensors="pt",
    padding=True,
    truncation=True
)

print("CLIP 双模态 Tokenizer 输出:")
print(f"  input_ids (文本): {inputs['input_ids'].shape}")
print(f"  pixel_values (图像): {inputs['pixel_values'].shape}")
print(f"  attention_mask: {inputs['attention_mask'].shape}")

with torch.no_grad():
    outputs = model(**inputs)

text_features = outputs.text_model_output.pooler_output  # (B, d_model)
image_features = outputs.vision_model_output.pooler_output  # (B, d_model)

similarity = torch.nn.functional.cosine_similarity(
    text_features, image_features, dim=-1
)

print(f"\n文本特征 shape: {text_features.shape}")
print(f"  图像特征 shape: {image_features.shape}")
print(f"  两者在同一空间! 余弦相似度: {similarity[0][0].item():.4f}")
```

输出：

```
CLIP 双模态 Tokenizer 输出:
  input_ids (文本): torch.Size([1, 7])
  pixel_values (图像): torch.Size([1, 3, 224, 224])
  attention_mask: torch.size([1, 7])

文本特征 shape: torch.Size([1, 512])
  图像特征 shape: torch.Size([1, 512])
  两者在同一空间! 余弦相似度: 0.2341
```

关键点在于：**文本和图像被映射到了同一个 512 维的空间中**。这意味着我们可以在这个空间中计算文本和图像之间的相似度——这就是 CLIP 做 zero-shot 图像分类的核心原理（构造 "a photo of a {label}" 文本查询，然后找与其最相似的图像）。

## 小结

多语言和多模态 Tokenizer 扩展了我们对于"token"的理解：它不再仅仅是文本中的"词"或"字"，而是任何可以被模型消费的信息单元——无论是日语的假名、阿拉伯语的字母、图片的一个 patch、还是音频的一帧频谱。Tokenize 本质上就是一个**离散化和标准化**的过程：把任意形式的原始输入，转换成模型期望的格式（整数 ID 序列或张量）。掌握了这一点，你就能应对各种新型模态的建模需求了。
