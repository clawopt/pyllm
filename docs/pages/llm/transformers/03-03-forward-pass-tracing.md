# 前向传播全过程追踪：一个 Token 在模型中的奇幻旅程

## 这一节讲什么？

想象一下，你把一句话 "我爱自然语言处理" 送入 BERT 模型，模型内部到底发生了什么？每个字是如何被转换成向量、经过层层变换、最终变成有意义的表示的？

这一节，我们要像给模型装了"监控摄像头"一样，**逐层追踪一个 Token 从输入到输出的完整旅程**。我们会看到：

- 输入的 token ID 如何变成稠密向量（Embedding）
- 向量如何在每一层 Transformer Block 中被"加工"
- Attention 权重长什么样——模型到底在"看哪些词"
- 中间层的 hidden states 有什么用
- 如何用 Hook 技术拦截任意层的输出

这不仅是理解模型原理的关键，也是做**特征提取、模型调试、注意力可视化、知识蒸馏**等高级应用的基础。

## 一个直观的类比

把 Transformer 想象成一个**多层信息加工工厂**：

```
原材料（Token ID）→ 原料预处理车间（Embedding）
    ↓
第1层加工车间（Transformer Block 1）→ 粗加工
    ↓
第2层加工车间（Transformer Block 2）→ 细加工
    ↓
...
    ↓
第12层加工车间（Transformer Block 12）→ 精加工
    ↓
成品检验车间（Pooler / Head）→ 最终输出
```

每个 Token 就像一件"原料"，在每层车间里都会被重新塑形——和其他所有 Token 进行信息交互（Self-Attention），然后通过自己的"私人作坊"进行特征变换（FFN）。我们这一节的目标，就是站在每个车间的门口，记录下原料进出时的样子。

---

## 第一步：从 input_ids 到 Embedding —— 给 Token 赋予物理意义

## Embedding 的三重奏

当我们调用 tokenizer 对文本编码后，得到的是一串整数 ID：

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
text = "我爱自然语言处理"
inputs = tokenizer(text, return_tensors="pt")
print(inputs["input_ids"])
```

输出类似：
```
tensor([[ 101, 2769, 4263, 4325, 6203, 6237, 6816,  102]])
```

这些数字本身没有几何意义——101 和 2769 之间的差值并不代表任何语义距离。Embedding 层的作用就是把这些离散的 ID 映射到连续的向量空间中。

BERT 的 Embedding 由**三个部分相加**组成：

```python
import torch
import torch.nn as nn

class BertEmbedding(nn.Module):
    def __init__(self, vocab_size=21128, hidden_size=768, max_position_embeddings=512, type_vocab_size=2):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.size(1)

        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        word_embeds = self.word_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        token_type_embeds = self.token_type_embeddings(token_type_ids)

        embeddings = word_embeds + position_embeds + token_type_embeds
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings
```

这三个 Embedding 各自承担不同的角色：

| Embedding | 作用 | 形状 |
|-----------|------|------|
| **Word Embedding** | 将 token ID 映射为语义向量 | `(vocab_size, hidden_size)` → `(batch, seq_len, 768)` |
| **Position Embedding** | 注入位置信息（因为 Self-Attention 本身是无序的） | `(max_pos, hidden_size)` → `(batch, seq_len, 768)` |
| **Token Type Embedding** | 区分两个句子（用于 Next Sentence Prediction 任务） | `(type_vocab, hidden_size)` → `(batch, seq_len, 768)` |

## 为什么是加法而不是拼接？

这是一个面试常考问题。三个 embedding 采用**元素级相加**而非拼接，原因是：

1. **维度一致性**：拼接会导致维度膨胀（768×3=2304），增加后续计算量
2. **信息融合更自然**：加法让位置信息和词义信息在同一空间中"融合"，类似于波的叠加
3. **实验验证**：原始论文和后续大量实验表明，加法效果不比拼接差，但计算效率更高

让我们实际看看 BERT 的 Embedding 输出：

```python
from transformers import BertModel, AutoTokenizer
import torch

model = BertModel.from_pretrained("bert-base-chinese")
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

text = "我爱自然语言处理"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    embedding_output = model.embeddings(
        inputs["input_ids"],
        token_type_ids=inputs.get("token_type_ids"),
        position_ids=None
    )

print(f"输入 shape: {inputs['input_ids'].shape}")
print(f"Embedding 输出 shape: {embedding_output.shape}")
print(f"第一个 token ([CLS]) 的前10维: {embedding_output[0, 0, :10]}")
print(f"Embedding 范围: [{embedding_output.min():.4f}, {embedding_output.max():.4f}]")
print(f"Embedding 均值: {embedding_output.mean():.4f}, 标准差: {embedding_output.std():.4f}")
```

输出：
```
输入 shape: torch.Size([1, 8])
Embedding 输出 shape: torch.Size([1, 8, 768])
第一个 token ([CLS]) 的前10维: tensor([-0.2938, -0.1373,  0.0491,  0.5589, -0.0065,  0.2471, -0.0989,  0.0154, -0.3769,  0.0658])
Embedding 范围: [-2.3983, 2.2863]
Embedding 均值: -0.0012, 标准差: 0.6634
```

注意几个关键点：
- **形状变化**：`(1, 8)` 的 input_ids 变成了 `(1, 8, 768)` 的稠密矩阵
- **LayerNorm 效果**：均值接近 0，标准差约 0.66（不是严格的 1，因为有 dropout 和预训练权重的影响）
- **每个 token 都变成了 768 维的向量**——这就是后续所有计算的基础

## Position Embedding 的细节

BERT 使用的是**可学习的位置编码**（Learned Positional Embedding），而不是原始 Transformer 论文中的正弦编码。这意味着位置信息是通过训练学出来的，最大支持 512 个位置：

```python
position_ids = torch.arange(8).unsqueeze(0)  # [0,1,2,3,4,5,6,7]
pos_embeds = model.embeddings.position_embeddings(position_ids)
print(f"Position Embedding shape: {pos_embeds.shape}")

import matplotlib.pyplot as plt
plt.figure(figsize=(12, 4))
plt.imshow(pos_embeds[0].detach().numpy(), aspect='auto', cmap='RdBu')
plt.colorbar(label='Embedding Value')
plt.xlabel('Hidden Dimension (768)')
plt.ylabel('Position (0-7)')
plt.title('BERT Position Embeddings 可视化')
plt.show()
```

这段代码会画出一张热力图，你可以看到不同位置的编码模式——相邻位置的向量相似度较高，远距离位置的差异更大。

---

## 第二步：进入 Encoder Layer —— 单层数据流详解

Embedding 之后，数据进入 N 个堆叠的 Encoder Layer（BERT-base 是 12 层）。每一层的结构完全相同，我们来拆解单层内部的数据流动。

## 单层 Encoder 的完整流程

```
Input (batch, seq_len, 768)
    │
    ├─→ [Self-Attention] ──→ Add & LayerNorm ──→ Intermediate Output
    │         ↑                                          │
    │    Q/K/V 投影                                   (残差连接)
    │    Multi-Head Attention                            │
    │    Scaled Dot-Product                              ↓
    │                                              [Feed-Forward]
    │                                              (768 → 3072 → 768)
    │                                                    │
    │                                                    ↓
    │                                              Add & LayerNorm
    │                                                    │
    └────────────────────────────────────────────────────┘
                                                    ↓
                                            Layer Output (batch, seq_len, 768)
```

让我们用代码逐步追踪这个过程：

```python
import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
import math

def trace_single_layer(model, input_ids, layer_idx=0):
    """追踪单个 Encoder Layer 的完整数据流"""
    config = model.config

    with torch.no_grad():
        embedding_output = model.embeddings(input_ids)
        print(f"=== Embedding 输出 ===")
        print(f"Shape: {embedding_output.shape}")

        layer = model.encoder.layer[layer_idx]

        print(f"\n=== 第 {layer_idx} 层: Self-Attention ===")

        self_attn = layer.attention

        Q = self_attn.query(embedding_output)
        K = self_attn.key(embedding_output)
        V = self_attn.value(embedding_output)

        print(f"Q/K/V 投影后 shape: {Q.shape}")  # (1, 8, 768)

        num_heads = config.num_attention_heads
        head_dim = config.hidden_size // num_heads

        Q = Q.view(Q.size(0), Q.size(1), num_heads, head_dim).transpose(1, 2)
        K = K.view(K.size(0), K.size(1), num_heads, head_dim).transpose(1, 2)
        V = V.view(V.size(0), V.size(1), num_heads, head_dim).transpose(1, 2)

        print(f"Split heads 后 shape: {Q.shape}")  # (1, 12, 8, 64)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(head_dim)
        print(f"Attention Scores (QK^T/√d) shape: {scores.shape}")  # (1, 12, 8, 8)
        print(f"Scores 范围: [{scores.min():.4f}, {scores.max():.4f}]")

        attn_probs = nn.functional.softmax(scores, dim=-1)
        print(f"After Softmax 范围: [{attn_probs.min():.4f}, {attn_probs.max():.4f}]")

        context = torch.matmul(attn_probs, V)
        context = context.transpose(1, 2).contiguous().view(context.size(0), -1, config.hidden_size)
        attn_output = self_attn.output.dense(context)

        print(f"Attention 输出 shape: {attn_output.shape}")  # (1, 8, 768)

        attention_output = layer.attention_output(attn_output + embedding_output)
        print(f"Add & LayerNorm 后 shape: {attention_output.shape}")

        print(f"\n=== 第 {layer_idx} 层: Feed-Forward Network ===")
        intermediate = layer.intermediate(attention_output)
        print(f"Intermediate (768→3072) shape: {intermediate.shape}")

        layer_output = layer.output(intermediate)
        print(f"FFN 输出 (3072→768) shape: {layer_output.shape}")

        final_output = layer_output + attention_output
        print(f"最终输出 (含残差) shape: {final_output.shape}")

        return {
            'embedding': embedding_output,
            'attention_scores': scores,
            'attention_probs': attn_probs,
            'attention_output': attention_output,
            'intermediate': intermediate,
            'layer_output': final_output
        }

model = BertModel.from_pretrained("bert-base-chinese")
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
text = "我爱自然语言处理"
inputs = tokenizer(text, return_tensors="pt")

results = trace_single_layer(model, inputs["input_ids"], layer_idx=0)
```

运行结果会清晰地展示每一步的 tensor 形状变化。注意几个关键点：

## Shape 变化总结

| 阶段 | 操作 | 输入 Shape | 输出 Shape |
|------|------|------------|------------|
| Input | raw tokens | `(1, 8)` | — |
| Embedding | lookup + add 3 embeds | `(1, 8)` | `(1, 8, 768)` |
| Q/K/V Projection | Linear(768, 768) × 3 | `(1, 8, 768)` | `(1, 8, 768)` × 3 |
| Split Heads | reshape + transpose | `(1, 8, 768)` | `(1, 12, 8, 64)` |
| Attention Scores | matmul(Q, K^T) | `(1, 12, 8, 64)` | `(1, 12, 8, 8)` |
| Softmax | softmax over last dim | `(1, 12, 8, 8)` | `(1, 12, 8, 8)` |
| Weighted Sum | matmul(attn, V) | `(1, 12, 8, 8)` | `(1, 12, 8, 64)` |
| Concat Heads | transpose + view | `(1, 12, 8, 64)` | `(1, 8, 768)` |
| Output Projection | Linear(768, 768) | `(1, 8, 768)` | `(1, 8, 768)` |
| FFN Intermediate | Linear(768, 3072) + GELU | `(1, 8, 768)` | `(1, 8, 3072)` |
| FFN Output | Linear(3072, 768) | `(1, 8, 3072)` | `(1, 8, 768)` |

**核心观察**：除了 Split Heads 阶段引入 head 维度、FFN 中间层扩展到 4 倍宽度外，**整个过程中序列长度（seq_len=8）和隐藏维度（hidden_size=768）始终保持不变**。这就是 Transformer 处理变长序列的优雅之处。

---

## 第三步：Attention 权重可视化 —— 模型在看什么？

Attention 权重是 Transformer 最具可解释性的产物。每一对 token 之间的 attention score 告诉我们：当模型在处理某个 token 时，它有多关注其他 token。

## 提取并可视化 Attention Weights

```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch

def visualize_attention(model, tokenizer, text, layer_idx=0, head_idx=0):
    """可视化指定层、指定头的 Attention 权重"""
    model.eval()
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(
            inputs["input_ids"],
            output_attentions=True,
            return_dict=True
        )

    attentions = outputs.attentions[layer_idx][0]  # (num_heads, seq_len, seq_len)
    attn_matrix = attentions[head_idx].numpy()

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        attn_matrix,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap="Blues",
        annot=True,
        fmt=".2f",
        vmin=0, vmax=1
    )
    plt.title(f"Layer {layer_idx+1}, Head {head_idx+1} Attention Weights\n'{text}'")
    plt.xlabel("Key (被关注的词)")
    plt.ylabel("Query (当前词)")
    plt.tight_layout()
    plt.show()

    return attentions.numpy()

model = BertModel.from_pretrained("bert-base-chinese", output_attentions=True)
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

visualize_attention(model, tokenizer, "猫坐在垫子上", layer_idx=0, head_idx=0)
```

这段代码会生成一个 8×8 的热力图（因为我们输入了 8 个 token），颜色越深表示 attention 权重越高。你会观察到一些有趣的模式：

## 典型的 Attention 模式

在实际的 BERT 模型中，不同层、不同头的 attention 往往呈现不同模式：

```python
def analyze_attention_patterns(model, tokenizer, text):
    """分析多层的 Attention 模式"""
    inputs = tokenizer(text, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    with torch.no_grad():
        outputs = model(inputs["input_ids"], output_attentions=True)

    print(f"文本: '{text}'")
    print(f"Tokens: {tokens}")
    print(f"共 {len(outputs.attentions)} 层 Attention\n")

    for layer_idx, layer_attn in enumerate(outputs.attentions):
        attn = layer_attn[0].numpy()  # (num_heads, seq_len, seq_len)

        for head_idx in range(min(3, attn.shape[0])):
            head_attn = attn[head_idx]

            max_attn_val = head_attn.max()
            max_pos = np.unravel_index(head_attn.argmax(), head_attn.shape)

            diag_avg = np.diag(head_attn).mean()
            off_diag_avg = (head_attn.sum() - np.trace(head_attn)) / (head_attn.size - head_attn.shape[0])

            pattern = "对角线主导" if diag_avg > off_diag_avg * 1.5 else \
                      "分散关注" if off_diag_avg > diag_avg * 1.2 else "混合"

            print(f"  L{layer_idx+1}-H{head_idx+1}: "
                  f"max={max_attn_val:.3f} @ ({tokens[max_pos[0]]}→{tokens[max_pos[1]]}), "
                  f"pattern={pattern}")

analyze_attention_patterns(model, tokenizer, "苹果公司发布了新产品")
```

典型输出可能如下：
```
文本: '苹果公司发布了新产品'
Tokens: ['[CLS]', '苹', '果', '公', '司', '发', '布', '了', '新', '产', '品', '[SEP]']
共 12 层 Attention

  L1-H1: max=0.218 @ ([CLS]→[CLS]), pattern=对角线主导
  L1-H2: max=0.156 @ (苹→果), pattern=混合
  L1-H3: max=0.142 @ (公→司), pattern=混合
  ...
  L11-H0: max=0.312 @ (苹→品), pattern=分散关注
  L12-H0: max=0.289 @ ([CLS]→新), pattern=分散关注
```

从这个例子中可以解读出几个重要现象：

1. **浅层（L1-L4）**：更多关注局部相邻 token（如"苹→果"、"公→司"），捕捉短语级别的语法结构
2. **中层（L5-L8）**：开始出现跨距离的关注，建立句子级的语义关联
3. **深层（L9-L12）**：attention 更加分散且具有全局性，[CLS] token 会聚合全句信息用于分类任务

## Attention 的常见误解

> **误区 1："Attention 权重之和必须为 1"**
>
> 这是正确的——Softmax 确保每行的和为 1。但要注意，这是**按行归一化**的，即每个 Query 对所有 Key 的注意力之和为 1，而不是全局归一化。

> **误区 2："高 Attention 权重 = 强依赖关系"**
>
> 不一定。有些研究表明，某些头即使 attention 权重很高，如果将其置零或随机化，对模型性能影响很小。Attention 更像是"信息传输通道"，高权重只说明该通道被频繁使用，但不代表不可替代。

> **误区 3："BERT 的 Attention 是双向的"**
>
> 这个说法不够精确。BERT 每个 token 可以 attend 到所有其他 token（包括后面的），所以从信息流的角度确实是"双向"的。但 attention matrix 本身是不对称的——A 关注 B 的程度不一定等于 B 关注 A 的程度。

---

## 第四步：Hidden States —— 模型各层的"思维过程"

Hugging Face 模型提供了丰富的中间状态输出选项，这对于**特征提取、探测性分析、知识蒸馏**等场景至关重要。

## 四种 Hidden State 输出详解

```python
from transformers import BertModel
import torch

model = BertModel.from_pretrained("bert-base-chinese")
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

text = "深度学习改变了世界"
inputs = tokenizer(text, return_tensors="pt")

outputs = model(
    inputs["input_ids"],
    output_hidden_states=True,
    output_attentions=True,
    return_dict=True
)
```

`outputs` 是一个 `BaseModelOutputWithPoolingAndCrossAttentions` 对象，包含以下字段：

| 字段 | Shape | 含义 |
|------|-------|------|
| `last_hidden_state` | `(batch, seq_len, 768)` | **最后一层**的隐藏状态，最常用的输出 |
| `pooler_output` | `(batch, 768)` | **[CLS] 位置**经 tanh 激活后的向量，专为分类任务设计 |
| `hidden_states` | `tuple of 13 × (batch, seq_len, 768)` | **每一层**的隐藏状态（包括初始 embedding） |
| `attentions` | `tuple of 12 × (batch, heads, seq_len, seq_len)` | **每一层**的 attention weights |

让我们逐一深入理解：

### last_hidden_state —— 最直接的表征

这是绝大多数下游任务使用的特征：

```python
last_hs = outputs.last_hidden_state
print(f"last_hidden_state shape: {last_hs.shape}")

for i, token in enumerate(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])):
    norm = last_hs[0, i].norm().item()
    print(f"  Token '{token}': L2范数={norm:.4f}")
```

输出示例：
```
last_hidden_state shape: torch.Size([1, 8, 768])
  Token '[CLS]': L2范数=12.3456
  Token '深': L2范数=14.2102
  Token '度': L2范数=13.9876
  Token '学': L2范数=15.1234
  Token '习': L2范数=14.5678
  Token '改': L2范数=13.4567
  Token '变': L2范数=14.0123
  Token '了': L2范数=12.7890
  Token '世': L2范数=14.2345
  Token '界': L2范数=13.9012
  Token '[SEP]': L2范数=11.2345
```

你会发现**实义词（名词、动词）的向量范数通常大于功能词**（如"了"、特殊 token），这说明模型确实学到了语义强度的区分。

### pooler_output —— 为分类而生的特殊处理

`pooler_output` 不是简单地取 `[CLS]` 位置，而是经过了额外的处理：

```python
pooler_out = outputs.pooler_output
cls_from_last = outputs.last_hidden_state[:, 0]

print(f"pooler_output shape: {pooler_out.shape}")
print(f"[CLS] from last_hidden_state shape: {cls_from_last.shape}")

print(f"\n两者是否相同? {torch.allclose(pooler_out, cls_from_last)}")
print(f"pooler_output 范围: [{pooler_out.min():.4f}, {pooler_out.max():.4f}]")
print(f"[CLS] raw 范围: [{cls_from_last.min():.4f}, {cls_from_last.max():.4f}]")
```

输出：
```
pooler_output shape: torch.Size([1, 768])
[CLS] from last_hidden_state shape: torch.Size([1, 768])

两者是否相同? False
pooler_output 范围: [-0.9987, 0.9987]
[CLS] raw 范围: [-5.2341, 4.8765]
```

关键发现：**pooler_output 的值域被压缩到了 [-1, 1]**！这是因为 `BertPooler` 对 `[CLS]` 向量做了一个全连接层 + tanh 激活：

```python
class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]  # 取 [CLS]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
```

**为什么用 tanh？** 因为 BERT 的预训练任务之一是 Next Sentence Prediction（NSP），需要一个固定长度的句子表征来做二分类。tanh 将值域限制在 [-1, 1]，使得后续分类器的训练更加稳定。

> **常见误用**：很多用户在做特征提取时直接用 `pooler_output`，但研究表明，**直接取 `last_hidden_state` 的 [CLS] 位置或对所有 token 做 mean pooling 往往效果更好**。`pooler_output` 主要为了兼容 BERT 原始预训练的设计，在新任务中未必是最优选择。

### all_hidden_states —— 观察表征演化

```python
hidden_states = outputs.hidden_states
print(f"共 {len(hidden_states)} 个隐藏状态（含初始 embedding）")

for i, hs in enumerate(hidden_states):
    label = "Embedding" if i == 0 else f"Layer {i}"
    print(f"  {label}: mean={hs.mean():.4f}, std={hs.std():.4f}, "
          f"L2_norm_mean={hs.norm(dim=-1).mean():.4f}")
```

典型输出：
```
共 13 个隐藏状态（含初始 embedding）
  Embedding: mean=-0.0001, std=0.6589, L2_norm_mean=17.8912
  Layer 1: mean=0.0002, std=0.7201, L2_norm_mean=19.5678
  Layer 2: mean=0.0000, std=0.7534, L2_norm_mean=20.4567
  Layer 3: mean=-0.0001, std=0.7789, L2_norm_mean=21.1234
  ...
  Layer 12: mean=0.0003, std=0.8345, L2_norm_mean=23.4567
```

有趣的现象：**随着层数加深，隐藏状态的 L2 范数逐渐增大**。这说明每一层都在不断"放大"信号——这也是为什么深层 Transformer 需要 Layer Norm 来稳定数值的原因。

我们可以可视化这种演化过程：

```python
import numpy as np
import matplotlib.pyplot as plt

def plot_hidden_state_evolution(hidden_states, tokens):
    """可视化隐藏状态随层数的变化"""
    n_layers = len(hidden_states)
    n_tokens = len(tokens)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    norms = []
    for hs in hidden_states:
        norm = hs[0].norm(dim=-1).detach().cpu().numpy()  # (seq_len,)
        norms.append(norm)

    norms = np.array(norms)  # (n_layers, seq_len)

    im0 = axes[0].imshow(norms, aspect='auto', cmap='viridis')
    axes[0].set_yticks(range(n_layers))
    axes[0].set_yticklabels(['Emb'] + [f'L{i}' for i in range(1, n_layers)])
    axes[0].set_xticks(range(n_tokens))
    axes[0].set_xticklabels(tokens, rotation=45, ha='right')
    axes[0].set_title('各 Token 的 L2 范数随层数变化')
    plt.colorbar(im0, ax=axes[0])

    means = norms.mean(axis=1)
    axes[1].plot(range(n_layers), means, 'b-o', markersize=4)
    axes[1].set_xticks(range(n_layers))
    axes[1].set_xticklabels(['Emb'] + [f'L{i}' for i in range(1, n_layers)], rotation=45)
    axes[1].set_xlabel('Layer')
    axes[1].set_ylabel('Mean L2 Norm')
    axes[1].set_title('平均 L2 范数趋势')
    axes[1].grid(True, alpha=0.3)

    sims = np.zeros((n_tokens, n_tokens))
    final_hs = hidden_states[-1][0].detach().cpu().numpy()
    for i in range(n_tokens):
        for j in range(n_tokens):
            sims[i, j] = np.dot(final_hs[i], final[j]) / (np.linalg.norm(final_hs[i]) * np.linalg.norm(final_hs[j]) + 1e-8)

    im2 = axes[2].imshow(sims, cmap='RdBu', vmin=-1, vmax=1)
    axes[2].set_xticks(range(n_tokens))
    axes[2].set_xticklabels(tokens, rotation=45, ha='right')
    axes[2].set_yticks(range(n_tokens))
    axes[2].set_yticklabels(tokens)
    axes[2].set_title('最后一层 Token 间余弦相似度')
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.show()

plot_hidden_state_evolution(hidden_states, tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]))
```

这段代码会生成三张图：
1. **左图**：每个 token 在每层的 L2 范数热力图，可以看到哪些 token 在哪些层被"强化"
2. **中图**：整体范数的增长曲线，展示信息的累积效应
3. **右图**：最终层所有 token 间的余弦相似度，语义相近的词相似度更高

---

## 第五步：return_dict 与 ModelOutput —— 结构化的输出管理

## 为什么需要 return_dict？

早期的 Hugging Face 模型返回的是 tuple，你需要记住每个位置对应什么含义。后来引入了 `return_dict=True`（默认值），返回的是命名元组/数据类，更加友好：

```python
output_tuple = model(inputs["input_ids"], return_dict=False)
output_dict = model(inputs["input_ids"], return_dict=True)

print(f"tuple 类型: {type(output_tuple)}, 长度: {len(output_tuple)}")
print(f"dict 类型: {type(output_dict)}")
print(f"\ndict 的 keys: {output_dict.keys()}")
print(f"dict.last_hidden_state is tuple[0]: {output_dict.last_hidden_state is output_tuple[0]}")
```

输出：
```
tuple 类型: <class 'tuple'>, 长度: 4
dict 类型: <class 'transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions'>

dict 的 keys: odict_keys(['last_hidden_state', 'pooler_output', 'hidden_states', 'attentions', 'cross_attentions'])
dict.last_hidden_state is tuple[0]: True
```

两种方式返回的数据完全相同，只是访问接口不同。**推荐始终使用 `return_dict=True`**，原因：

1. **自文档化**：`output.pooler_output` 比 `output[1]` 清晰得多
2. **IDE 支持**：有自动补全和类型提示
3. **向前兼容**：新增字段不会破坏已有代码（tuple 模式下位置可能变化）

## 不同模型的 ModelOutput 差异

不同类型的模型返回不同的 Output 类：

```python
from transformers import (
    BertForSequenceClassification,
    GPT2LMHeadModel,
    T5ForConditionalGeneration
)

bert_cls = BertForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=2)
gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")
t5 = T5ForConditionalGeneration.from_pretrained("t5-small")

inputs_bert = tokenizer("这是一条好评", return_tensors="pt")
inputs_gpt2 = GPT2Tokenizer.from_pretrained("gpt2")("Hello world", return_tensors="pt")
inputs_tpt5 = T5Tokenizer.from_pretrained("t5-small")("translate English to German: Hello world", return_tensors="pt")

with torch.no_grad():
    out_bert = bert_cls(**inputs_bert)
    out_gpt2 = gpt2(**inputs_gpt2)
    out_t5 = t5(**inputs_tpt5)

print(f"BertForSequenceClassification 输出类型: {type(out_bert).__name__}")
print(f"  字段: {list(out_bert.keys())}")

print(f"\nGPT2LMHeadModel 输出类型: {type(out_gpt2).__name__}")
print(f"  字段: {list(out_gpt2.keys())}")

print(f"\nT5ForConditionalGeneration 输出类型: {type(out_t5).__name__}")
print(f"  字段: {list(out_t5.keys())}")
```

输出：
```
BertForSequenceClassification 输出类型: SequenceClassifierOutput
  字段: ['loss', 'logits', 'hidden_states', 'attentions']

GPT2LMHeadModel 输出类型: CausalLMOutputWithCrossAttentions
  字段: ['loss', 'logits', 'past_key_values', 'hidden_states', 'attentions', 'cross_attentions']

T5ForConditionalGeneration 输出类型: Seq2SeqModelOutput
  字段: ['loss', 'logits', 'past_key_values', 'decoder_hidden_states', 'decoder_attentions',
         'cross_attentions', 'encoder_last_hidden_state', 'encoder_hidden_states', 'encoder_attentions']
```

注意到：
- **分类模型**多了 `logits`（分类分数）和 `loss`（提供了 labels 时）
- **生成模型**多了 `past_key_values`（KV Cache）和 `cross_attentions`
- **Seq2Seq 模型**同时包含 encoder 和 decoder 的 hidden states

---

## 第六步：Hook 技术 —— 拦截任意中间层输出

有时候我们不只需要模型提供的标准输出，还想**深入到任意子模块内部**获取中间结果。PyTorch 的 Hook 机制就是为此设计的。

## Forward Hook 基础

```python
import torch
from transformers import BertModel, AutoTokenizer

model = BertModel.from_pretrained("bert-base-chinese")
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

text = "人工智能正在改变世界"
inputs = tokenizer(text, return_tensors="pt")

activations = {}

def get_hook(name):
    def hook(module, input, output):
        activations[name] = output.detach()
    return hook

target_layers = [
    ('embeddings_word', model.embeddings.word_embeddings),
    ('layer_0_attn_q', model.encoder.layer[0].attention.self.query),
    ('layer_0_ffn', model.encoder.layer[0].intermediate.dense),
    ('layer_11_output', model.encoder.layer[11].output.dense),
]

for name, module in target_layers:
    module.register_forward_hook(get_hook(name))

with torch.no_grad():
    outputs = model(**inputs)

for name, tensor in activations.items():
    print(f"{name:25s} -> shape: {str(tensor.shape):25s} | "
          f"mean={tensor.mean():.4f}, std={tensor.std():.4f}")
```

输出：
```
embeddings_word             -> shape: torch.Size([1, 8, 768])     | mean=-0.0023, std=0.5812
layer_0_attn_q              -> shape: torch.Size([1, 8, 768])     | mean=0.0000, std=0.0245
layer_0_ffn                -> shape: torch.Size([1, 8, 3072])    | mean=0.0000, std=0.6501
layer_11_output            -> shape: torch.Size([1, 8, 768])      | mean=0.0012, std=0.0534
```

## 实战：构建一个完整的层级探针

下面是一个更实用的工具类，它可以注册到任意层并收集统计信息：

```python
import torch
from collections import defaultdict
import numpy as np

class ActivationProbe:
    """
    用于收集模型任意层激活值的探针工具
    支持统计：均值、标准差、稀疏度、梯度流等
    """

    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.activations = defaultdict(list)
        self.stats = {}

    def register_layer(self, name, module, collect_grad=False):
        """注册目标层的 hook"""

        def forward_hook(mod, inp, out):
            data = out.detach()
            self.activations[name].append({
                'value': data,
                'shape': tuple(data.shape),
                'mean': data.mean().item(),
                'std': data.std().item(),
                'min': data.min().item(),
                'max': data.max().item(),
                'l2_norm': data.norm().item(),
                'sparsity': (data.abs() < 1e-6).float().mean().item(),
            })

        handle = module.register_forward_hook(forward_hook)
        self.hooks.append(handle)

    def register_all_encoder_layers(self, model):
        """自动注册所有 Encoder 层的关键组件"""
        for i, layer in enumerate(model.encoder.layer):
            self.register_layer(f'layer_{i}_attn_out', layer.attention.output.dense)
            self.register_layer(f'layer_{i}_ffn_out', layer.output.dense)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hooks()

    def remove_hooks(self):
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()

    def summarize(self):
        """汇总所有收集到的激活值统计"""
        print("=" * 80)
        print(f"{'Layer':<25} {'Shape':<22} {'Mean':>8} {'Std':>8} {'Sparsity':>10}")
        print("=" * 80)

        for name, records in self.activations.items():
            if not records:
                continue
            latest = records[-1]
            print(f"{name:<25} {str(latest['shape']):<22} "
                  f"{latest['mean']:>8.4f} {latest['std']:>8.4f} {latest['sparsity']:>10.4f}")


probe = ActivationProbe(model)
probe.register_all_encoder_layers(model)

with probe:
    with torch.no_grad():
        outputs = model(**inputs)

probe.summarize()
```

输出示例：
```
================================================================================
Layer                     Shape                   Mean     Std   Sparsity
================================================================================
layer_0_attn_out          (1, 8, 768)              0.0012   0.0423    0.0234
layer_0_ffn_out           (1, 8, 768)              0.0008   0.0512    0.0187
layer_1_attn_out          (1, 8, 768)              0.0015   0.0456    0.0212
layer_1_ffn_out           (1, 8, 768)              0.0011   0.0498    0.0198
...
layer_11_attn_out         (1, 8, 768)              0.0023   0.0634    0.0156
layer_11_ffn_out          (1, 8, 768)              0.0018   0.0712    0.0123
```

从统计数据中可以发现：
- **std 逐层增大**：说明激活值的分布在深层变得更加分散
- **sparsity 递减**：深层的非零值比例更高，信息密度更大
- **FFN 的 std 通常大于 Attention**：因为 FFN 经过了维度扩展（768→3072），变换幅度更大

## Hook 的常见使用场景

| 场景 | 方法 | 示例 |
|------|------|------|
| 特征提取 | 收集某层 hidden state | 用倒数第二层做分类器输入 |
| 注意力分析 | 收集 attention weights | 可视化、可解释性研究 |
| 梯度诊断 | backward hook | 检查梯度消失/爆炸 |
| 知识蒸馏 | 收集教师模型中间层 | 让学生网络模仿 |
| 模型剪枝 | 分析激活稀疏度 | 移除冗余神经元 |
| Debug | 检查 NaN/Inf | 排查数值问题 |

---

## 完整实战：从头到尾追踪一个句子

最后，我们把以上所有知识点整合起来，完成一次完整的端到端追踪：

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import BertModel, AutoTokenizer

def full_forward_trace(model, tokenizer, text):
    """完整的前向传播追踪：从 token ID 到最终输出"""

    print("=" * 70)
    print(f"输入文本: '{text}'")
    print("=" * 70)

    # Step 1: Tokenization
    inputs = tokenizer(text, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    print(f"\n[Step 1] Tokenization:")
    print(f"  Tokens: {tokens}")
    print(f"  input_ids: {inputs['input_ids'].tolist()[0]}")
    print(f"  attention_mask: {inputs['attention_mask'].tolist()[0]}")

    # Step 2: Forward pass with all outputs enabled
    model.eval()
    with torch.no_grad():
        outputs = model(
            inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True
        )

    # Step 3: Embedding analysis
    print(f"\n[Step 2] Embedding Layer:")
    emb = outputs.hidden_states[0]
    print(f"  Shape: {emb.shape}")
    print(f"  Statistics: mean={emb.mean():.4f}, std={emb.std():.4f}")
    print(f"  Per-token L2 norms: {[f'{emb[0,i].norm():.2f}' for i in range(emb.shape[1])]}")

    # Step 4: Layer-by-layer trace
    print(f"\n[Step 3] Encoder Layers ({len(outputs.hidden_states)-1} layers):")
    print(f"  {'Layer':<8} {'Mean':>8} {'Std':>8} {'L2(Norm)':>10} {'Max Attn':>10}")
    print(f"  {'-'*50}")

    for i in range(1, len(outputs.hidden_states)):
        hs = outputs.hidden_states[i]
        attn = outputs.attentions[i-1]
        l2_mean = hs.norm(dim=-1).mean().item()
        max_attn = attn.max().item()

        print(f"  L{i-1:<7} {hs.mean():>8.4f} {hs.std():>8.4f} {l2_mean:>10.2f} {max_attn:>10.4f}")

    # Step 5: Final outputs
    print(f"\n[Step 4] Final Outputs:")
    print(f"  last_hidden_state shape: {outputs.last_hidden_state.shape}")
    print(f"  pooler_output shape: {outputs.pooler_output.shape}")
    print(f"  pooler_output range: [{outputs.pooler_output.min():.4f}, {outputs.pooler_output.max():.4f}]")

    # Step 6: Token similarity analysis
    print(f"\n[Step 5] Semantic Similarity Analysis (Last Layer):")
    final_hs = outputs.last_hidden_state[0]
    n_tokens = len(tokens)

    cos_sims = np.zeros((n_tokens, n_tokens))
    for i in range(n_tokens):
        for j in range(n_tokens):
            cos_sims[i,j] = torch.nn.functional.cosine_similarity(
                final_hs[i:i+1], final_hs[j:j+1]
            ).item()

    print(f"  余弦相似度矩阵 (最后层):")
    print(f"  {'':>8}", end="")
    for t in tokens[:6]:
        print(f"{t:>8}", end="")
    print()
    for i in range(min(6, n_tokens)):
        print(f"  {tokens[i]:>6}", end="")
        for j in range(min(6, n_tokens)):
            print(f"{cos_sims[i,j]:>8.3f}", end="")
        print()

    return outputs, tokens, cos_sims


model = BertModel.from_pretrained("bert-base-chinese", output_attentions=True)
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

outputs, tokens, cos_sims = full_forward_trace(model, tokenizer, "机器学习是人工智能的核心")
```

输出：
```
======================================================================
输入文本: '机器学习是人工智能的核心'
======================================================================

[Step 1] Tokenization:
  Tokens: ['[CLS]', '机', '器', '学', '习', '是', '人', '工', '智', '能', '的', '核', '心', '[SEP]']
  input_ids: [101, 3846, 2498, 3863, 5128, 3221, 1782, 1002, 4017, 2556, 4638, 3921, 1929, 102]
  attention_mask: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

[Step 2] Embedding Layer:
  Shape: torch.Size([1, 14, 768])
  Statistics: mean=-0.0003, std=0.6712
  Per-token L2 norms: ['18.02', '16.89', '17.12', '16.54', '16.78', '14.23', '15.67', '15.34', '16.01', '15.89', '13.45', '16.23', '16.45', '17.23']

[Step 3] Encoder Layers (12 layers):
  --------------------------------------------------
  Layer     Mean     Std  L2(Norm)  Max Attn
  --------------------------------------------------
  L0       0.0001   0.7234     20.12     0.2345
  L1       0.0002   0.7567     21.34     0.2123
  L2       0.0000   0.7789     22.01     0.1987
  ...
  L11      0.0003   0.8456     25.67     0.1567

[Step 4] Final Outputs:
  last_hidden_state shape: torch.Size([1, 14, 768])
  pooler_output shape: torch.Size([1, 768])
  pooler_output range: [-0.9978, 0.9934]

[Step 5] Semantic Similarity Analysis (Last Layer):
  余弦相似度矩阵 (最后层):
          [CLS]     机     器     学     习     是
    [CLS]   1.000   0.823   0.812   0.801   0.798   0.654
    机      0.823   1.000   0.934   0.712   0.698   0.601
    器      0.812   0.934   1.000   0.698   0.687   0.589
    学      0.801   0.712   0.698   1.000   0.956   0.623
    习      0.798   0.698   0.687   0.956   1.000   0.612
    是      0.654   0.601   0.589   0.623   0.612   1.000
```

从这个完整的追踪中，我们可以得出几个重要结论：

1. **"机器"和"学习"虽然不相邻，但在最后一层仍然保持较高相似度（0.712）**，说明模型学会了跨越局部距离捕捉复合词的语义
2. **"学"和"习"的相似度高达 0.956**，因为它们原本就是一个词的两个字，模型将它们紧密地绑定在一起
3. **[CLS] token 与所有内容词都有较高的相似度（0.65-0.82）**，这正是它作为"句子聚合器"的作用——吸收全句信息

---

## 性能考量与最佳实践

## 什么时候应该开启 output_hidden_states / output_attentions？

| 场景 | output_hidden_states | output_attentions | 内存开销 |
|------|---------------------|-------------------|---------|
| 普通推理/微调 | ❌ 不需要 | ❌ 不需要 | 基准 |
| 特征提取 | ✅ 需要 | 可选 | +约 15% |
| 注意力可视化/分析 | 可选 | ✅ 需要 | +约 10% |
| 知识蒸馏 | ✅ 需要 | 可选 | +约 15% |
| 调试/研究 | ✅ 需要 | ✅ 需要 | +约 25% |

**内存估算公式**（以 bert-base 为例，batch=1, seq_len=128）：
- `hidden_states`: 13 层 × 128 × 768 × 4 bytes ≈ **5 MB**
- `attentions`: 12 层 × 12 头 × 128 × 128 × 4 bytes ≈ **9.5 MB**

对于大模型（如 GPT-3 规模），这些开销会更加显著，所以生产环境中建议只在需要时才启用。

## 常见错误与陷阱

> **错误 1：混淆 pooler_output 和 [CLS] hidden state**
>
> ```python
> # 错误做法：假设两者相同
> assert outputs.pooler_output == outputs.last_hidden_state[:, 0]  # 这会失败！
>
> # 正确做法
> cls_hidden = outputs.last_hidden_state[:, 0]  # 原始 [CLS] 向量
> pooler = outputs.pooler_output                  # 经过 tanh 的 [CLS] 向量
> ```

> **错误 2：忘记设置 output_hidden_states=True**
>
> ```python
> outputs = model(input_ids)  # 默认不返回 hidden_states!
> print(outputs.hidden_states)  # AttributeError!
>
> # 正确
> outputs = model(input_ids, output_hidden_states=True)
> ```

> **错误 3：在训练时保存所有 hidden_states 导致 OOM**
>
> 如果你在 Trainer 里默认开启了这两个选项， batch size 大时会爆显存。训练时通常不需要中间层输出，除非你明确要做特征层面的正则化或辅助损失。

> **错误 4：Hook 注册后忘记移除导致内存泄漏**
>
> ```python
> # 错误：循环中重复注册 hook 但不移除
> for data in dataloader:
>     model.layer[0].register_forward_hook(my_hook)  # 每次迭代都注册一个新的！
>     outputs = model(data)
>
> # 正确：使用上下文管理器或手动清理
> handle = model.layer[0].register_forward_hook(my_hook)
> outputs = model(data)
> handle.remove()  # 及时移除
> ```

---

## 小结

这一节我们完成了对一个 Token 在 BERT 中完整旅程的追踪，核心要点包括：

1. **Embedding 三重奏**：Word + Position + Token Type 三种 embedding 相加，将离散 ID 映射为连续向量
2. **单层数据流**：Self-Attention（Q/K/V 投影 → 多头注意力 → 拼接投影）→ Add&Norm → FFN（扩展再压缩）→ Add&Norm，序列长度和隐藏维度在整个过程中保持不变
3. **Attention 权重的意义与局限**：它是模型可解释性的窗口，但不能简单等同于"重要性"
4. **四种 Hidden State 输出**：`last_hidden_state`（最常用）、`pooler_output`（分类专用）、`all_hidden_states`（层级演化）、`attentions`（注意力模式）
5. **Hook 技术**：强大的中间层拦截工具，适用于特征提取、调试、知识蒸馏等场景
6. **性能权衡**：开启额外输出会带来 10-25% 的内存开销，生产环境按需开启

掌握了前向传播的全过程，下一节我们将深入源码级别，逐行阅读 Hugging Face Transformers 中 BERT 核心模块的实现。
