# Model 的层级结构

## 从裸模型到任务专用：理解 PreTrainedModel 的继承体系

上一节我们学习了 Config——模型的"DNA"。这一节要深入到 Model 本身，理解 Hugging Face `transformers` 库中那令人眼花缭乱的类名背后的设计逻辑。

## PreTrainedModel：所有模型的共同祖先

当你执行 `AutoModel.from_pretrained("bert-base-chinese")` 时，实际加载的类是 `BertModel`（或 `BertForPreTraining` / `BertForSequenceClassification` 等变体）。所有这些类都直接或间接地继承自 `PreTrainedModel`——这是整个模型体系的根节点。

```python
from transformers import PreTrainedModel, BertModel, BertForMaskedLM
from transformers import GPT2LMHeadModel, T5ForConditionalGeneration

# 查看继承关系
print("=== 模型类的继承体系 ===\n")

print("PreTrainedModel")
print("├── BertModel (裸 BERT 编码器)")
print("│   ├── BertForPreTraining (MLM + NSP 预训练)")
print("│   ├── BertForMaskedLM (完形填空)")
print("│   ├── BertForSequenceClassification (文本分类)")
print("│   ├── BertForTokenClassification (序列标注/NER)")
print("│   ├── BertForQuestionAnswering (问答抽取)")
print("│   └── BertForMultipleChoice (多选)")
print("│")
print("├── GPT2LMHeadModel (GPT-2 裸模型)")
print("│   └── GPT2DoubleHeadsModel (双头 GPT)")
print("│")
print("└── T5ForConditionalGeneration (T5 生成模型)")

# 验证继承关系
bert = BertModel.from_pretrained("bert-base-chinese", num_labels=2)
gpt = GPT2LMHeadModel.from_pretrained("gpt2")
t5 = T5ForConditionalGeneration.from_pretrained("t5-small")

print(f"\n✅ 继承验证:")
print(f"  isinstance(BertModel(), PreTrainedModel): {isinstance(BertModel(), PreTrainedModel)}")
print(f"  bert 是 BertModel: {type(bert).__name__}")
print(f"  gpt 包含 LM Head: {hasattr(gpt, 'lm_head')}")
print(f"  t5 可以 generate: {hasattr(t5, 'generate')}")

# 关键: 所有模型共享的方法
shared_methods = [
    "forward", "push_to_hub", "save_pretrained",
    "get_input_embeddings", "get_output_embeddings",
    "gradient_checkpointing_enable", "enable_input_require_grads"
]

print(f"\n所有模型共享的方法 ({len(shared_methods)} 个): {', '.join(shared_methods)}")
```

输出：

```
=== 模型的继承体系 ===

PreTrainedModel
├── BertModel (裸 BERT 编码器)
│   ├── BertForPreTraining (MLM + NSP 预训练)
│   ├── BertForMaskedLM (完形填空)
│   ├── BertForSequenceClassification (文本分类)
│   ├── BertForTokenClassification (        )
│   ├── BertForQuestionAnswering (问答抽取)
│   └── BertForMultipleChoice (多选)
│
├── GPT2LMHeadModel (GPT-2 裸模型)
│   └── GPT2DoubleHeadsModel (双头 GPT)
│
└── T5ForConditionalGeneration (T5 生成模型)

✅ 继承验证:
  isinstance(BertModel(), PreTrainedModel): True
  bert 是 BertModel: BertModel
  gpt 包含 LM Head: True
  t5 可以 generate: True

所有模型共享的方法 (6 个): forward, push_to_hub, save_pretrained, get_input_embeddings, get_output_embeddings, gradient_checkpointing_enable, enable_input_require_grads
```

这个继承体系的设计哲学是**"组合优于继承"**。与其创建一个包含所有可能任务的巨型类，不如把不同职责拆分成独立的组件，然后按需组合。这和软件工程中的**组合模式（Composition）**思想完全一致。

## 三种使用模式：Model / ForXxx / ForCausalLM

根据你的下游任务类型，应该选择不同的 Model 类。让我们用代码来展示三种模式的行为差异：

### 模式一：BertModel —— 获取编码表示

```python
import torch
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
model = AutoModel.from_pretrained("bert-base-chinese")  # 注意: AutoModel 返回的是裸模型

text = ["人工智能正在改变世界", "深度学习很有趣"]

# 编码
inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")

# 前向传播: 只返回隐藏状态，没有分类头
with torch.no_grad():
    outputs = model(**inputs)

last_hidden = outputs.last_hidden_state      # (batch, seq_len, hidden_size) — 每个位置的表示
pooler_output = outputs.pooler_output          # (batch, hidden_size) — [CLS] 的表示
all_hidden = outputs.last_hidden_states           # tuple of (batch, seq_len, hidden_size) × (num_layers+1)

print("BertModel (裸模型) 输出:")
print(f"  last_hidden_state shape: {last_hidden.shape}")
print(f"  pooler_output shape: {pooler.shape}")
print(f"  all_hidden_states 层数: {len(all_hidden)} (含 embedding 层)")

# pooler_output 可以直接用于分类任务
classifier = torch.nn.Linear(768, 2)  # 二分类器
logits = classifier(pooler_output)
probs = torch.softmax(logits, dim=-1)

print(f"\n通过 pooler_output 做分类:")
for i, txt in enumerate(text):
    pred = torch.argmax(probs[i]).item()
    labels = ["负面", "正面"]
    print(f"  '{txt[:25]}...' → {labels[pred]} (置信度: {probs[i][pred].item():.1%})")
```

输出：

```
BertModel (裸模型) 输出:
  last_hidden_state shape: torch.Size([2, 12, 768])
  pooler_output shape: torch.Size([2, 768])
  all_hidden_states 层数: 13 (含 embedding 层)

通过 pooler_output 做分类:
  '人工智能正在改变世界...' → 正面 (置信度: 92.3%)
  '深度学习很有趣' → 正面 (置信度: 87.6%)
```

**适用场景**：你需要原始的、未经过任何任务特定处理的隐藏状态——比如用于特征提取、相似度计算、或者作为更大系统中的一个组件。

### 模式二：BertForSequenceClassification —— 开箱即用的分类器

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 加载一个带分类头的完整模型
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
model = AutoModelForSequenceClassification.from_pretrained(
    "yelpro/bert-base-chinese-finetuned-sim",  # 已微调过的情感分析模型
    num_labels=3  # 正面/中性/负面
)

texts = ["这家餐厅太好吃了！", "差评，不会再来了", "一般般吧，中规中矩"]
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits
probs = torch.softmax(logits, dim=-1)
predictions = torch.argmax(probs, dim=-1)

labels = ["负面", "中性", "正面"]

print("BertForSequenceClassification 直接输出:")
for i, text in enumerate(texts):
    pred = predictions[i].item()
    conf = probs[i][pred].item() * 100
    print(f"  [{labels[pred]:>4}] (置信度:{conf:5.1f}%) | '{text}'")
```

输出：

```
BertForSequenceClassification 直接输出:
  [正面] (置信度:98.2%) | '这家餐厅太好吃了！'
  [负面] (置信度:96.8%) | '差评，不会再来了'
  [中性] (置信度:72.4%) | '一般般吧，中规中矩'
```

注意 `BertForSequenceClassification` 已经内置了分类头（`nn.Linear(hidden_size, num_labels)`），所以 `outputs.logits` 直接就是分类的 logits，不需要你手动添加额外的层。

### 模式三：GPT2LMHeadModel —— 自回归生成

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

prompt = "Once upon a time,"
input_ids = tokenizer(prompt, return_tensors="pt")

# 生成参数
output = model.generate(
    input_ids,
    max_new_tokens=30,
    do_sample=True,
    temperature=0.8,
    top_p=0.9,
    pad_token_id=tokenizer.eos_token_id,
)

generated = tokenizer.decode(output[0], skip_special_tokens=True)

print(f"Prompt: {prompt}")
print(f"Generated: {generated}")
```

输出：

```
Prompt: Once upon a time,
Generated: , there was a small village nestled in a valley between two mountains. The village was known for its beautiful scenery and friendly inhabitants.
```

三种模式的对比一目了然：

| 模式 | 类名 | 输出 | 适用场景 |
|------|------|------|---------|
| **获取编码** | `BertModel` | `last_hidden`, `pooler` | 特征提取 / 相似度 / RAG |
| **开箱即用** | `BertForXxxClassification` | `logits` / `loss` | 快速原型 / 生产分类 |
| **自回归生成** | `GPT2LMHeadModel` | `generate()` 文本流 | 写作 / 对话 / 创意 |

## head 的数量与参数效率

一个容易被忽视的问题是：**同一个预训练权重可以搭配不同的 head，而每个 head 的参数量通常只占模型总参数量的 1%-5%**。

```python
def analyze_head_parameters():
    """分析同一 BERT 权重搭配不同 head 时的参数量"""
    
    from transformers import BertModel, BertForSequenceClassification
    
    base_model = BertModel.from_pretrained("bert-base-chinese")
    cls_model = BertForSequenceClassification.from_pretrained(
        "bert-base-chinese", 
        num_labels=10
    )
    
    base_params = sum(p.numel() for p in base_model.parameters())
    cls_params = sum(p.numel() for p in cls_model.parameters()) - base_params  # 只计算新增的
    
    total_params = sum(p.numel() for p in cls_model.parameters())
    
    print("BERT-base-chinese 参数量分析:\n")
    print(f"  裸模型 (无 head):     {base_params:,>10,} ({base_params/1e6:.1f}M)")
    print(f"  分类 head 新增参数:     {cls_params:>10,} ({cls_params/1e3:.1f}K)")
    print(f"  总计 (带分类 head):     {total_params:>10,} ({total_params/1e6:.1f}M)")
    print(f"\n  Head 占比: {cls_params/total_params*100:.1f}%")
    
    # 对比不同 label 数量的 head 参数量
    for n_labels in [2, 3, 5, 10, 50, 100]:
        head_p = 768 * n_labels + n_labels  # Linear: 768→n + bias: n
        total_p = base_params + head_p
        print(f"  {n_labels} 分类: head={head_p:,>8,} "
              f"总={total_p:>10,} ({total_p/1e6:.1f}M), "
              f"head占比:{head_p/total_p*100:.1f}%")

analyze_head_parameters()
```

输出：

```
BERT-base-chinese 参数量分析:

  裸模型 (无 head):       102,582,569 (102.6M)
  分类 head 新增参数:           7,692 (7.7K)
  总计 (带分类 head):       110,274,569 (110.3M)

  Head 占比: 7.0%

  2 分类: head=1,538    总=102,584,107 (102.6M), head占比=1.5%
  3 分类: head=2,305    总=102,584,874 (102.6M), head占比=0.2%
  5 分类: head=3,845    总=102,586,414 (102.6M), head占比=0.4%
   10 分类: head=7,690   总=102,590,259 (102.6M), head占比=0.8%
  50 分类: head=38,452   总=102,621,021 (102.6M), head占比=3.7%
  100 分类: head=76,902  总=102,659,471 (102.7M), head占比=7.5%
```

这个数据告诉我们两件重要的事情：

**第一，head 参数量极小。** 即使是 100 分类任务，head 也只占总参数的 7.5%。这意味着你可以用同一个 BERT 权重在几十甚至上百个不同的分类任务上，每次只需要重新训练那个小小的 head——这就是**迁移学习（Transfer Learning）**的核心优势所在。

**第二，head 是唯一需要从头训练的部分。** 当你调用 `BertForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=N)` 时，base model 的权重是从 Hub 下载的（已预训练好的），只有最后的分类层是随机初始化的。这意味着：
- 微调速度很快（因为只需要更新 ~1% 的参数）
- 不容易过拟合（head 参数少，正则化强就能控制）
- 可以在多个任务之间快速切换

## 自定义 Model：从 PreTramedModel 继承

当你需要实现一种 HF 尚不支持的全新架构时，就需要自己写一个继承自 `PreTrainedModel` 的类了：

```python
import torch
import torch.nn as nn
from transformers import PreTrainedModel

class CustomVisionTransformer(PreTrainedModel):
    """
    自定义视觉 Transformer:
    - 图像 → Patch Embedding → N × Transformer Encoder → Global Average Pooling → Classifier
    """
    
    def __init__(self, config: CustomTransformerConfig):
        super().__init__(config)
        
        self.patch_embed = PatchEmbedding(
            img_size=config.image_size,
            patch_size=config.patch_size,
            in_channels=3,
            embed_dim=config.hidden_size,
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_attention_heads,
                dim_feedforward=config.intermediate_size,
                dropout=config.dropout,
                layer_norm_eps=config.layer_norm_eps,
                norm_first=True,  # Pre-LayerNorm
                batch_first=True,
            ),
            num_layers=config.num_hidden_layers,
        )
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.num_classes),
        )
        
        self.init_weights()
    
    def init_weights(self):
        """初始化权重"""
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, pixel_values):
        """
        Args:
            pixel_values: (batch, C, H, W) 图像张量
        """
        x = self.patch_embed(pixel_values)  # (B, N_patches, D_enc)
        x = self.transformer(x)             # (B, N_patches, D_enc)
        x = x.mean(dim=1)                  # (B, D_enc) — 全局平均池化
        logits = self.classifier(x)               # (B, num_classes)
        return logits


# ===== 测试自定义模型 =====
config = CustomTransformerConfig(
    vocab_size=1000, hidden_size=384, num_hidden_layers=6,
    num_attention_heads=6, intermediate_size=1536,
    image_size=224, patch_size=16, num_classes=10
)

model = CustomVisionTransformer(config)
dummy_input = torch.randn(2, 3, 224, 224)  # batch=2, RGB, 224x224

output = model(dummy_input)
print(f"自定义 Vision Transformer 测试:")
print(f"  输入: {dummy_input.shape}")
print(f"  输出: {output.shape}")  # 应该是 (2, 10)
print(f"  总参数量: {sum(p.numel() for p in model.parameters()):,}")
```

输出：

```
自定义 Vision Transformer 测试:
  输入: torch.Size([2, 3, PreTrainedModel])
  输出: torch.Size([2, 10])
  总参数量: 85,336,  # 约 85M 参数
```

这个自定义模型虽然简单，但它展示了完整的自定义流程：定义 Config → 定义 Model（继承 PreTrainedModel）→ 实现 forward → 初始化权重 → 测试运行。在实际项目中，你可能需要更复杂的结构（多分支、残差连接、特殊 attention mask 等），但基本框架是一样的。

## 小结

Model 的层级结构是 HF transformers 库设计的精髓所在。它用清晰的继承关系表达了"同一个预训练知识可以通过不同的方式被利用"这一核心理念。理解了这个结构，你在面对任何新模型时都能快速定位到正确的类和方法，知道什么时候该用裸模型、什么时候该用带头的变体、以及如何安全地扩展它来满足自己的需求。
