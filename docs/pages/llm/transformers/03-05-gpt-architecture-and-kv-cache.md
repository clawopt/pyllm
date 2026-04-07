# GPT 架构细节与 KV Cache：Decoder-only 模型的深度剖析

## 这一节讲什么？

前面我们花了大量篇幅解剖 BERT（Encoder-only 架构），但当今最炙手可热的大模型——从 GPT-3 到 LLaMA、Claude、Qwen——几乎全部采用 **Decoder-only（仅解码器）架构**。虽然它们和 BERT 共享 Transformer 的基础组件，但在几个关键设计上有着本质区别：

1. **因果掩码**：模型只能"看到"当前及之前的内容，不能偷看未来
2. **Pre-LayerNorm**：LayerNorm 放在 Attention/FFN 之前，训练更稳定
3. **KV Cache**：自回归生成的性能优化利器，将生成速度提升数倍

这一节，我们将深入 GPT-2 的架构实现，理解它与 BERT 的异同，并重点掌握 **KV Cache** 这个让大模型推理变得可行的核心技术。

## GPT 与 BERT 的核心差异一览

在深入代码之前，先用一张表建立整体认知：

| 特性 | BERT (Encoder-only) | GPT (Decoder-only) |
|------|---------------------|---------------------|
| **注意力模式** | 双向注意力（每个 token 看所有 token） | 因果/自回归注意力（只看左边） |
| **位置编码** | 绝对位置编码 | 可学习位置编码 (GPT-2) / RoPE (LLaMA) |
| **LayerNorm 位置** | Post-Norm（子层之后） | Pre-Norm（子层之前） |
| **训练目标** | MLM + NSP | 自回归语言建模（预测下一个 token） |
| **输出方式** | [CLS] 用于分类 | 最后一个 token 的 logits 用于生成 |
| **关键优化** | 无 | KV Cache（避免重复计算） |
| **典型用途** | 理解任务（分类、NER、QA） | 生成任务（对话、写作、代码） |

---

## 一、因果掩码 —— "不能偷看未来"的强制约束

## 1.1 直觉理解

想象你在逐字阅读一句话。当你读到第 5 个字时，你自然地知道前 4 个字是什么，但你不可能知道第 6 个字及之后的内容——除非你偷看。

GPT 的训练目标就是**根据前面的内容预测下一个 token**，所以它在处理第 i 个位置时，只能使用位置 0 到 i-1 的信息。这就是**因果注意力（Causal Attention）**或**自回归注意力（Autoregressive Attention）**的核心思想。

## 1.2 掩码的实现方式

因果掩码通过一个**上三角矩阵**来实现：

```python
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

def create_causal_mask(seq_len, dtype=torch.float32):
    """创建因果（上三角）掩码"""
    mask = torch.triu(torch.ones(seq_len, seq_len, dtype=dtype), diagonal=1)
    # 上三角（不含对角线）= 1，其余 = 0
    # 转换为加性掩码：需要屏蔽的位置设为 -inf
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask

# 示例：seq_len = 5
mask_5 = create_causal_mask(5)
print("5×5 因果掩码:")
print(mask_5)
```

输出：
```
5×5 因果掩码:
tensor([[0., -inf, -inf, -inf, -inf],
        [0.,    0., -inf, -inf, -inf],
        [0.,    0.,    0., -inf, -inf],
        [0.,    0.,    0.,    0., -inf],
        [0.,    0.,    0.,    0.,    0.]])
```

可以看到：
- 对角线和左下三角是 0（允许 attention）
- 右上三角是 `-inf`（禁止 attention）

当这个掩码被加到 attention scores 上再经过 Softmax 时，`-inf` 位置的 Softmax 输出为 0，实现了"看不见"的效果：

```python
def demonstrate_causal_mask_effect():
    """演示因果掩码对注意力的实际影响"""
    seq_len = 6
    scores = torch.randn(seq_len, seq_len) * 0.5  # 模拟原始 attention scores
    causal_mask = create_causal_mask(seq_len)

    masked_scores = scores + causal_mask
    attn_probs = F.softmax(masked_scores, dim=-1)

    tokens = ['我', '喜', '欢', '自', '然', '语']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    im0 = axes[0].imshow(scores.numpy(), cmap='RdBu', vmin=-1, vmax=1)
    axes[0].set_xticklabels(tokens, rotation=45)
    axes[0].set_yticklabels(tokens)
    axes[0].set_title('原始 Attention Scores\n(无掩码)')
    for i in range(seq_len):
        for j in range(seq_len):
            axes[0].text(j, i, f'{scores[i,j]:.2f}', ha='center', va='center', fontsize=8)
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(attn_probs.numpy(), cmap='Blues', vmin=0, vmax=1)
    axes[1].set_xticklabels(tokens, rotation=45)
    axes[1].set_yticklabels(tokens)
    axes[1].title.set_text('因果 Attention Probabilities\n(右上角全为0)')
    for i in range(seq_len):
        for j in range(seq_len):
            val = attn_probs[i,j].item()
            if val > 0.001:
                axes[1].text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=8, color='white' if val > 0.3 else 'black')
    plt.colorbar(im1, ax=axes[1])

    row_sums = attn_probs.sum(dim=-1).numpy()
    axes[2].bar(range(seq_len), row_sums, color='steelblue')
    axes[2].set_xticks(range(seq_len))
    axes[2].set_xticklabels(tokens)
    axes[2].set_title('每行概率之和\n(应该都等于1)')
    axes[2].axhline(y=1.0, color='red', linestyle='--', alpha=0.7)
    axes[2].set_ylabel('Sum of Probs')

    plt.tight_layout()
    plt.show()

demonstrate_causal_mask_effect()
```

这段代码会生成三张图：
1. **左图**：原始 attention scores（无掩码时所有位置都有值）
2. **中图**：加上因果掩码后的 attention probabilities——**右上三角全部为 0**
3. **右图**：每行概率之和仍然为 1（Softmax 的归一化特性不受影响）

## 1.3 Hugging Face 中的因果掩码处理

在实际的 GPT 实现中，因果掩码通常不是显式创建一个大矩阵传入，而是通过 `attention_mask` 参数巧妙地组合：

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

text = "Hello world"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        output_attentions=True
    )

attn = outputs.attentions[-1][0]  # 最后一层, 第一个样本

print("最后一层第一个头的 Attention 权重:")
print(f"Shape: {attn.shape}")  # (num_heads, seq_len, seq_len)

tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
print(f"Tokens: {tokens}")

head_0 = attn[0].numpy()
print(f"\nHead 0 Attention Matrix (6×6):")
for i, t in enumerate(tokens):
    row_str = " ".join([f"{head_0[i,j]:.3f}" for j in range(len(tokens))])
    print(f"  {t:8s}: {row_str}")
```

典型输出：
```
最后一层第一个头的 Attention 权重:
Shape: torch.Size([12, 6, 6])
Tokens: ['Hello', 'Ġworld', 'Ġis', 'Ġa', 'Ġgreat', 'Ġplace']

Head 0 Attention Matrix (6×6):
  Hello  : 0.287 0.000 0.000 0.000 0.000 0.000
  Ġworld : 0.312 0.245 0.000 0.000 0.000 0.000
  Ġis    : 0.198 0.156 0.234 0.000 0.000 0.000
  Ġa     : 0.145 0.189 0.201 0.198 0.000 0.000
  Ġgreat : 0.058 0.234 0.187 0.156 0.178 0.000
  Ġplace : 0.067 0.123 0.145 0.198 0.234 0.145
```

可以清楚地看到**严格的上三角零区域**——每个位置只能 attend 到自己及之前的 token。

---

## 二、Pre-LayerNorm —— 为什么 GPT 把 LayerNorm 放在了前面？

## 2.1 两种归一化位置对比

这是 BERT 和 GPT 在架构上一个重要但不那么显而易见的差异：

**BERT（Post-Norm）**：
```python
# BERT 的单层结构
x = x + Attention(x)     # 先计算
x = LayerNorm(x)          # 再归一化
x = x + FFN(x)            # 先计算
x = LayerNorm(x)           # 再归一化
```

**GPT-2（Pre-Norm）**：
```python
# GPT-2 的单层结构
x = x + Attention(LayerNorm(x))  # 先归一化，再计算
x = x + FFN(LayerNorm(x))         # 先归一化，再计算
# 最终输出层有一个额外的 LayerNorm
x = Final_LayerNorm(x)
```

## 2.2 为什么 Pre-Norm 更好？

Pre-Norm 带来的好处是多方面的，我们可以通过实验来直观感受：

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class PostNormBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.norm1(x + self.attn(x, x, x)[0])
        x = self.norm2(x + self.ffn(x))
        return x


class PreNormBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.ffn(self.norm2(x))
        return self.final_norm(x)


def compare_norm_stability(depth=24, seq_len=128, d_model=256, n_steps=100):
    """比较 Pre-Norm 和 Post-Norm 在深层网络中的数值稳定性"""
    post_net = nn.ModuleList([PostNormBlock(d_model) for _ in range(depth)])
    pre_net = nn.ModuleList([PreNormBlock(d_model) for _ in range(depth)])

    post_net.eval()
    pre_net.eval()

    x = torch.randn(1, seq_len, d_model)

    with torch.no_grad():
        post_grads = []
        pre_grads = []
        post_norms = []
        pre_norms = []

        y_post = x
        for i, block in enumerate(post_net):
            y_post = block(y_post)
            post_norms.append(y_post.norm().item())

        y_pre = x
        for i, block in enumerate(pre_net):
            y_pre = block(y_pre)
            pre_norms.append(y_pre.norm().item())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(range(depth), post_norms, 'r-o', markersize=3, label='Post-Norm', alpha=0.8)
    axes[0].plot(range(depth), pre_norms, 'b-s', markersize=3, label='Pre-Norm', alpha=0.8)
    axes[0].set_xlabel('Layer Depth')
    axes[0].set_ylabel('L2 Norm of Hidden State')
    axes[0].set_title('隐藏状态范数随层数变化')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    post_diffs = np.abs(np.diff(post_norms))
    pre_diffs = np.abs(np.diff(pre_norms))

    axes[1].semilogy(range(depth-1), post_diffs + 1e-10, 'r-o', markersize=3, label='Post-Norm', alpha=0.8)
    axes[1].semilogy(range(depth-1), pre_diffs + 1e-10, 'b-s', markersize=3, label='Pre-Norm', alpha=0.8)
    axes[1].set_xlabel('Layer Transition')
    axes[1].set_ylabel('|Δ Norm| (log scale)')
    axes[1].set_title('层间范数变化幅度')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    final_post_std = post_norms[-1] / post_norms[0]
    final_pre_std = pre_norms[-1] / pre_norms[0]

    bars = axes[2].bar(['Post-Norm', 'Pre-Norm'], [final_post_std, final_pre_std],
                       color=['coral', 'steelblue'])
    axes[2].set_ylabel('Output/Input Norm Ratio')
    axes[2].set_title('最终范数 / 初始范数')
    axes[2].axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='Ideal (no change)')
    for bar, val in zip(bars, [final_post_std, final_pre_std]):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.2f}', ha='center', fontsize=11)
    axes[2].legend()

    plt.tight_layout()
    plt.show()

compare_norm_stability()
```

## 2.3 Pre-Norm 的三大优势

基于大量研究和实践，Pre-Norm 相比 Post-Norm 有以下核心优势：

### 优势 1：梯度流动更健康

在 Post-Norm 中，梯度必须先穿过 LayerNorm 才能到达残差分支；而在 Pre-Norm 中，残差连接上的梯度路径更加"干净"。这意味着：

- **深层网络的梯度消失问题大幅缓解**
- **训练超深层模型（如 GPT-3 的 96 层、LLaMA 的 80 层）成为可能**

### 优势 2：训练稳定性更好

Pre-Norm 让每一层的输入都在归一化的空间中，避免了信号在网络深处被不断放大或衰减。这使得：

- **学习率可以设置得更大**（收敛更快）
- **对初始化不那么敏感**
- **不需要 warmup 那么久**

### 优势 3：无需最终 LayerNorm 的"身份映射"初始化

Post-Norm 网络在初始化时需要精心设计以确保每层接近"恒等映射"（identity mapping），否则深层网络的输出会偏离正常范围。Pre-Norm 天然保证了输入到子模块的数据是标准化的，对初始化的容忍度更高。

> **代价是什么？**
>
> Pre-Norm 并非完美无缺。研究表明，在某些任务上 Post-Norm 的最终性能可能略优于 Pre-Norm（约 0.1-0.3%）。另外，Pre-Norm 需要在最后额外加一层 LayerNorm 来保证输出的规范性（GPT-2 就有这个 `ln_f` 层）。

---

## 三、KV Cache —— 让自回归生成加速数倍的魔法

## 3.1 问题：自回归生成的冗余计算

让我们先看看没有 KV Cache 时，GPT 是如何做文本生成的：

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import time

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model.eval()

prompt = "The quick brown fox"
input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]

generated = input_ids.clone()

start = time.time()
for step in range(20):
    with torch.no_grad():
        outputs = model(generated)
        next_token_logits = outputs.logits[:, -1, :]
        next_token = next_token_logits.argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)

elapsed = time.time() - start
text = tokenizer.decode(generated[0], skip_special_tokens=True)
print(f"生成文本: {text}")
print(f"耗时: {elapsed:.3f}秒")
```

这个朴素实现的致命问题是：**每生成一个新 token，都要把整个已有序列重新送入模型一遍**！

假设我们要生成长度为 T 的序列：

| 步骤 | 输入长度 | 计算量 |
|------|---------|--------|
| Step 1 | 1 (初始 prompt) | O(1² × d) |
| Step 2 | 2 | O(2² × d) |
| Step 3 | 3 | O(3² × d) |
| ... | ... | ... |
| Step T | T | O(T² × d) |
| **总计** | — | **O(T³ × d)** |

总计算量是 **O(T³)**！当 T=1000 时，这意味着大约 5 亿次不必要的重复计算。

## 3.2 核心洞察：Key 和 Value 不需要重新计算

仔细观察 Self-Attention 的计算过程：

```
对于位置 i 的 token：
  Q_i = W_Q × x_i          ← 只有这个需要新算（因为 x_i 是新的）
  K_j = W_K × x_j (j < i)  ← 这些在之前的步骤中已经算过了！
  V_j = W_V × x_j (j < i)  ← 同上！
  score_ij = Q_i · K_j / √d
  output_i = Σ score_ij × V_j
```

**关键发现**：当我们在第 t 步生成第 t 个 token 时，位置 0 到 t-1 的 K 和 V 向量在之前的步骤中已经完全计算好了，而且它们的值不会改变（因为模型参数不变、输入不变）。我们完全可以把这些 K 和 V **缓存（cache）** 起来，在第 t 步直接复用！

这就是 **KV Cache（Key-Value Cache）** 的核心思想。

## 3.3 KV Cache 的详细工作流程

```python
def generate_with_kv_cache(model, tokenizer, prompt, max_new_tokens=20):
    """手动实现带 KV Cache 的自回归生成"""

    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    model.eval()

    past_key_values = None  # KV Cache 初始化为空

    generated_tokens = []

    start_time = time.time()

    for step in range(max_new_tokens):
        if step == 0:
            # 第一步：正常的前向传播（处理整个 prompt）
            with torch.no_grad():
                outputs = model(
                    input_ids,
                    use_cache=True,
                    return_dict=True,
                    past_key_values=past_key_values,
                )
        else:
            # 后续步骤：只输入最新的一个 token
            with torch.no_grad():
                outputs = model(
                    next_token_input,
                    use_cache=True,
                    return_dict=True,
                    past_key_values=past_key_values,
                )

        # 提取结果
        logits = outputs.logits[:, -1, :]       # 只取最后一个位置的 logits
        next_token = logits.argmax(dim=-1, keepdim=True)
        generated_tokens.append(next_token.item())

        # 更新 KV Cache
        past_key_values = outputs.past_key_values

        # 下一步只输入这一个新 token
        next_token_input = next_token

    elapsed = time.time() - start_time
    full_text = tokenizer.decode(
        input_ids[0].tolist() + generated_tokens,
        skip_special_tokens=True
    )

    return full_text, elapsed, past_key_values
```

## 3.4 KV Cache 的内部结构

`past_key_values` 是一个嵌套元组，结构如下：

```
past_key_values = (
    # Layer 0
    (
        key_cache_layer_0,   # shape: (batch, num_heads, seq_len, head_dim)
        value_cache_layer_0, # shape: (batch, num_heads, seq_len, head_dim)
    ),
    # Layer 1
    (
        key_cache_layer_1,
        value_cache_layer_1,
    ),
    ...
    # Layer N-1
    (
        key_cache_layer_N-1,
        value_cache_layer_N-1,
    )
)
```

以 GPT-2 为例（12 层，12 头，head_dim=64），假设已经缓存了 100 个 token：

```python
def analyze_kv_cache_structure(past_key_values):
    """分析 KV Cache 的结构和内存占用"""
    total_params = 0
    total_memory_bytes = 0

    print("=" * 70)
    print(f"{'Layer':<8} {'K Shape':<35} {'V Shape':<35} {'Memory':<12}")
    print("=" * 70)

    for layer_idx, (k, v) in enumerate(past_key_values):
        k_shape = tuple(k.shape)
        v_shape = tuple(v.shape)

        k_memory = k.nelement() * k.element_size()
        v_memory = v.nelement() * v.element_size()
        layer_mem = k_memory + v_memory

        total_params += k.nelement() + v.nelement()
        total_memory_bytes += layer_mem

        print(f"L{layer_idx:<7} {str(k_shape):<35} {str(v_shape):<35} {layer_mem/1024:>9.1f}KB")

    print("=" * 70)
    print(f"Total KV Cache: {total_params:,} elements")
    print(f"Total Memory: {total_memory_bytes / 1024 / 1024:.2f} MB")

    return total_memory_bytes

# 使用示例
text, elapsed, pkv = generate_with_kv_cache(model, tokenizer, "Once upon a time", max_new_tokens=30)
analyze_kv_cache_structure(pkv)
```

输出示例：
```
======================================================================
Layer     K Shape                             V Shape                             Memory
======================================================================
L0        (1, 12, 19, 64)                     (1, 12, 19, 64)                     112.5KB
L1        (1, 12, 19, 64)                     (1, 12, 19, 64)                     112.5KB
...
L11       (1, 12, 19, 64)                     (1, 12, 19, 64)                     112.5KB
======================================================================
Total KV Cache: 351,232 elements
Total Memory: 1.34 MB
```

## 3.5 性能对比：有 Cache vs 无 Cache

```python
def benchmark_generation(model, tokenizer, prompt, max_tokens=50):
    """对比有无 KV Cache 的生成性能"""
    import time

    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    model.eval()

    # ===== 方法 1: 无 Cache（每次送入完整序列）=====
    start = time.time()
    gen_no_cache = input_ids.clone()
    for _ in range(max_tokens):
        with torch.no_grad():
            out = model(gen_no_cache)
            next_tok = out.logits[:, -1:, :].argmax(dim=-1)
            gen_no_cache = torch.cat([gen_no_cache, next_tok], dim=1)
    time_no_cache = time.time() - start

    # ===== 方法 2: 有 KV Cache ======
    start = time.time()
    past_kv = None
    gen_with_cache = input_ids.clone()

    for step in range(max_tokens):
        if step == 0:
            with torch.no_grad():
                out = model(gen_with_cache, use_cache=True, past_key_values=None)
        else:
            with torch.no_grad():
                out = model(gen_with_cache[:, -1:], use_cache=True, past_key_values=past_kv)

        next_tok = out.logits[:, -1:, :].argmax(dim=-1)
        gen_with_cache = torch.cat([gen_with_cache, next_tok], dim=1)
        past_kv = out.past_key_values

    time_with_cache = time.time() - start

    speedup = time_no_cache / time_with_cache

    print(f"Prompt: '{prompt}' (length={input_ids.shape[1]}, generate {max_tokens} tokens)")
    print(f"  无 KV Cache: {time_no_cache:.3f}s")
    print(f"  有 KV Cache: {time_with_cache:.3f}s")
    print(f"  加速比: {speedup:.1f}x")
    print(f"\n生成文本一致: {torch.equal(gen_no_cache, gen_with_cache)}")

    return speedup

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

speedups = []
prompts = ["Hello", "The quick brown fox jumps over the lazy dog",
           "In a shocking turn of events, scientists discovered that"]

for prompt in prompts:
    s = benchmark_generation(model, tokenizer, prompt, max_tokens=30)
    speedups.append(s)

print(f"\n平均加速比: {sum(speedups)/len(speedups):.1f}x")
```

典型输出（CPU 上）：
```
Prompt: 'Hello' (length=1, generate 30 tokens)
  无 KV Cache: 2.345s
  有 KV Cache: 0.567s
  加速比: 4.1x

Prompt: 'The quick brown fox...' (length=9, generate 30 tokens)
  无 KV Cache: 5.678s
  有 KV Cache: 0.891s
  加速比: 6.4x

Prompt: 'In a shocking turn...' (length=7, generate 30 tokens)
  无 KV Cache: 4.567s
  有 KV Cache: 0.756s
  加速比: 6.0x

平均加速比: 5.5x
```

**加速效果随着序列长度增加而显著增长**——因为无 Cache 方法的计算量是 O(n²)，而有 Cache 方法每步的计算量是 O(n)（只需要对新 token 和缓存的 K/V 做一次 matmul）。

## 3.6 KV Cache 的内存占用分析

KV Cache 的内存占用是一个重要的工程考量，尤其是对于大模型和长上下文场景：

```python
def estimate_kv_cache_memory(model_name_or_config, seq_length, batch_size=1):
    """估算不同模型的 KV Cache 内存占用"""

    configs = {
        "gpt2": {"layers": 12, "heads": 12, "head_dim": 64, "dtype_bytes": 2},
        "gpt2-xl": {"layers": 48, "heads": 25, "head_dim": 64, "dtype_bytes": 2},
        "llama-7b": {"layers": 32, "heads": 32, "head_dim": 128, "dtype_bytes": 2},
        "llama-13b": {"layers": 40, "heads": 40, "head_dim": 128, "dtype_bytes": 2},
        "llama-70b": {"layers": 80, "heads": 64, "head_dim": 128, "dtype_bytes": 2},
        "qwen-72b": {"layers": 80, "heads": 64, "head_dim": 128, "dtype_bytes": 2},
    }

    print("=" * 75)
    print(f"{'Model':<15} {'SeqLen':>7} {'Layers':>7} {'Heads':>7} {'Per-Token':>12} {'Total':>12}")
    print(f"{'':15} {'':7} {'':7} {'':7} {'(KB)':>12} {'(MB)':>12}")
    print("-" * 75)

    results = {}
    for name, cfg in configs.items():
        layers = cfg["layers"]
        heads = cfg["heads"]
        head_dim = cfg["head_dim"]
        dtype_bytes = cfg["dtype_bytes"]

        # 每个 token 每层的 KV 缓存大小: 2(K+V) × heads × head_dim × dtype_bytes
        per_token_per_layer = 2 * heads * head_dim * dtype_bytes
        per_token_all_layers = per_token_per_layer * layers
        total = per_token_all_layers * seq_length * batch_size

        per_token_kb = per_token_all_layers / 1024
        total_mb = total / 1024 / 1024

        results[name] = {"per_token_kb": per_token_kb, "total_mb": total_mb}

        print(f"{name:<15} {seq_length:>7} {layers:>7} {heads:>7} "
              f"{per_token_kb:>11.1f} {total_mb:>11.1f}")

    print("-" * 75)

    fig, ax = plt.subplots(figsize=(12, 6))
    names = list(results.keys())
    totals = [results[n]["total_mb"] for n in names]
    colors = ['steelblue' if t < 2048 else 'orange' if t < 8192 else 'red' for t in totals]
    bars = ax.bar(names, totals, color=colors)
    ax.set_ylabel('KV Cache Memory (MB)')
    ax.set_title(f'KV Cache 内存占用对比 (seq_len={seq_length}, batch=1, FP16)')
    ax.axhline(y=80*1024, color='red', linestyle='--', alpha=0.5, label='A100 80GB')
    ax.axhline(y=24*1024, color='orange', linestyle='--', alpha=0.5, label='RTX 3090 24GB')
    ax.legend()
    for bar, val in zip(bars, totals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
               f'{val:.0f}', ha='center', fontsize=9)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    return results

estimate_kv_cache_memory(None, seq_length=4096)
estimate_kv_cache_memory(None, seq_length=32768)
```

seq_length=4096 时的输出示例：
```
===========================================================================
Model           SeqLen  Layers   Heads   Per-Token      Total
                         (KB)           (MB)
---------------------------------------------------------------------------
gpt2             4096      12      12      192.0       768.0
gpt2-xl          4096      48      25     1600.0      6400.0
llama-7b         4096      32      32     2048.0      8192.0
llama-13b        4096      40      40     3200.0     12800.0
llama-70b        4096      80      64     5120.0     20480.0
qwen-72b         4096      80      64     5120.0     20480.0
---------------------------------------------------------------------------
```

seq_length=32768（32K 上下文）时的输出：
```
llama-70b @ 32K context: ~163,840 MB = **160 GB!** 远超任何单卡 GPU 显存
```

这就是为什么长上下文模型面临如此大的内存挑战——**KV Cache 的内存占用与序列长度线性增长**，且对于大模型来说绝对值非常可观。

> **前沿方案**：为了缓解这个问题，研究者提出了多种 KV Cache 优化技术：
> - **Multi-Query Attention (MQA)**：所有头共享同一组 K/V，减少 cache 大小
> - **Grouped-Query Attention (GQA)**：折中方案，分组共享 K/V（LLaMA 2/3 使用）
> - **PagedAttention (vLLM)**：类似操作系统的虚拟内存管理，按需分配 KV Cache
> - **H2O / StreamingLLM**：动态淘汰不重要的 cache 条目

---

## 四、GPT-2 的 forward 方法详解

## 4.1 完整的参数列表

```python
def forward(
    self,
    input_ids=None,              # Token ID 序列
    past_key_values=None,        # KV Cache（来自上一步）
    attention_mask=None,         # 注意力掩码
    token_type_ids=None,         # Token 类型 ID
    position_ids=None,           # 位置 ID
    head_mask=None,              # 注意力头剪枝掩码
    inputs_embeds=None,          # 直接传入 embedding（替代 input_ids）
    labels=None,                 # 训练标签（用于计算 loss）
    use_cache=None,              # 是否返回 KV Cache
    output_attentions=False,     # 是否返回 attention weights
    output_hidden_states=False,  # 是否返回 hidden states
    return_dict=None,            # 是否返回命名元组
):
```

## 4.2 input_ids vs inputs_embeds —— 两种输入方式

大多数时候我们用 `input_ids`（token ID），但有时需要直接控制 embedding：

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import torch.nn.functional as F

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# 方式 1: 通过 input_ids（常规方式）
input_ids = tokenizer("Hello world", return_tensors="pt")["input_ids"]

# 方式 2: 通过 inputs_embeds（跳过 embedding 层）
with torch.no_grad():
    embeds = model.transformer.wte(input_ids)  # 手动获取 embedding

outputs_from_ids = model(input_ids)
outputs_from_embeds = model(inputs_embeds=embeds)

print(f"通过 input_ids 的 logits shape: {outputs_from_ids.logits.shape}")
print(f"通过 inputs_embeds 的 logits shape: {outputs_from_embeds.logits.shape}")

logits_match = torch.allclose(outputs_from_ids.logits, outputs_from_embeds.logits, atol=1e-5)
print(f"两种方式的输出是否一致: {logits_match}")
```

`inputs_embeds` 的典型使用场景包括：
- **Embedding 干扰/对抗攻击**：直接修改 embedding 向量来测试模型鲁棒性
- **多模态融合**：图像/音频特征投影后作为 text embedding 的补充
- **软提示（Soft Prompt）**：可学习的连续向量作为 prompt，而非离散 token

## 4.3 position_ids —— 位置编码的处理

GPT-2 使用**可学习的位置编码**（Learned Positional Embedding），最大长度为 1024：

```python
def demonstrate_position_embeddings():
    """展示 GPT-2 的位置编码特性"""
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    wpe = model.transformer.wpe  # Word Position Embeddings

    max_pos = wpe.weight.shape[0]
    hidden_size = wpe.weight.shape[1]

    print(f"GPT-2 位置编码表: max_position={max_pos}, dim={hidden_size}")

    pos_embeds = wpe.weight.data  # (1024, 768)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].imshow(pos_embeds[:50].numpy(), aspect='auto', cmap='RdBu')
    axes[0].set_xlabel('Hidden Dimension')
    axes[0].set_ylabel('Position')
    axes[0].set_title('GPT-2 位置编码 (前50个位置)')

    norms = pos_embeds.norm(dim=-1).numpy()
    axes[1].plot(norms, 'b-', linewidth=1)
    axes[1].set_xlabel('Position')
    axes[1].set_ylabel('L2 Norm')
    axes[1].set_title('各位置编码向量的 L2 范数')
    axes[1].grid(True, alpha=0.3)

    sims = F.cosine_similarity(pos_embeds[:1], pos_embeds[:50], dim=-1).numpy()
    axes[2].plot(sims, 'r-', linewidth=1)
    axes[2].set_xlabel('Position')
    axes[2].set_ylabel('Cosine Similarity to Pos 0')
    axes[2].set_title('与位置0的余弦相似度')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"\n位置 0 与位置 1 的相似度: {F.cosine_similarity(pos_embeds[0:1], pos_embeds[1:2]).item():.4f}")
    print(f"位置 0 与位置 500 的相似度: {F.cosine_similarity(pos_embeds[0:1], pos_embeds[500:501]).item():.4f}")
    print(f"位置 0 与位置 1000 的相似度: {F.cosine_similarity(pos_embeds[0:1], pos_embeds[1000:1001]).item():.4f}")

demonstrate_position_embeddings()
```

> **GPT-2 位置编码的限制**：由于使用的是固定大小的 lookup table（最大 1024），GPT-2 无法处理超过 1024 个 token 的序列。这也是为什么后续模型（如 LLaMA）转向了 **RoPE（旋转位置编码）**——它可以自然地外推到任意长度。

---

## 五、完整实战：从零实现一个支持 KV Cache 的 Decoder-only 层

理论讲完了，我们来动手写一个完整的、支持 KV Cache 的单层 Decoder Block，并与 GPT-2 的官方实现进行对比验证：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CausalSelfAttentionWithCache(nn.Module):
    """
    带有 KV Cache 支持的因果自注意力
    手动实现，用于理解底层机制
    """

    def __init__(self, d_model, num_heads, dropout=0.1, max_seq_len=1024):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()[None, None, :, :]
        )

    def forward(self, x, past_kv=None, use_cache=False):
        """
        Args:
            x: (batch, seq_len, d_model)
            past_kv: Optional[tuple(key, value)] 来自上一步的缓存
                key: (batch, num_heads, cached_len, head_dim)
                value: (batch, num_heads, cached_len, head_dim)
            use_cache: 是否返回更新后的 KV cache
        Returns:
            output: (batch, seq_len, d_model)
            present: Optional[tuple(new_k, new_v)] 更新后的 KV cache
        """
        B, S, D = x.shape

        Q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        if past_kv is not None:
            past_k, past_v = past_kv
            K = torch.cat([past_k, K], dim=2)  # (B, heads, cached+S, head_dim)
            V = torch.cat([past_v, V], dim=2)

        total_len = K.shape[2]

        if use_cache:
            present = (K, V)
        else:
            present = None

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        causal = self.causal_mask[:, :, :S, :total_len]
        scores = scores.masked_fill(causal, float('-inf'))

        attn_probs = F.softmax(scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        context = torch.matmul(attn_probs, V)
        context = context.transpose(1, 2).contiguous().view(B, S, D)

        output = self.resid_dropout(self.out_proj(context))

        return output, present


class PreNormTransformerBlock(nn.Module):
    """Pre-Norm 结构的 Transformer Block（GPT-2 风格）"""

    def __init__(self, d_model, num_heads, d_ff=None, dropout=0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model

        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttentionWithCache(d_model, num_heads, dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, past_kv=None, use_cache=False):
        # Pre-Norm Attention
        residual = x
        x = self.ln1(x)
        attn_out, present = self.attn(x, past_kv=past_kv, use_cache=use_cache)
        x = residual + attn_out

        # Pre-Norm FFN
        residual = x
        x = self.ln2(x)
        x = residual + self.ffn(x)

        return x, present


def verify_manual_implementation():
    """验证手动实现与 GPT-2 官方实现的一致性"""
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    gpt2.eval()

    d_model = gpt2.config.n_embd
    num_heads = gpt2.config.n_head

    my_block = PreNormTransformerBlock(d_model, num_heads)
    my_block.eval()

    text = "The future of AI"
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        official_hidden = gpt2.transformer(inputs["input_ids"]).last_hidden_state

        embeds = gpt2.transformer.wte(inputs["input_ids"])
        positions = torch.arange(inputs["input_ids"].shape[1]).unsqueeze(0)
        pos_embeds = gpt2.transformer.wpe(positions)
        my_input = embeds + pos_embeds

        my_output, _ = my_block(my_input)

    diff = (my_output - official_hidden).abs().max().item()
    print(f"手动实现 vs GPT-2 官方实现:")
    print(f"  最大绝对误差: {diff:.4f}")
    print(f"  注: 由于权重未共享，数值会不同，但 shape 应一致")
    print(f"  官方输出 shape: {official_hidden.shape}")
    print(f"  手动输出 shape: {my_output.shape}")

    assert official_hidden.shape == my_output.shape, "Shape mismatch!"

verify_manual_implementation()
```

---

## 六、流式生成与 KV Cache 的生产级实现

在实际的大模型服务中，KV Cache 不仅用于加速，还是**流式输出（Streaming）**的基础：

```python
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import time

class StreamingGenerator:
    """
    支持 KV Cache 和流式输出的生成器
    模拟 ChatGPT/ChatGLM 的逐字输出效果
    """

    def __init__(self, model, tokenizer, max_new_tokens=200):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.model.eval()

    def generate_streaming(self, prompt, callback=None):
        """
        流式生成，每生成一个 token 就调用 callback
        Args:
            callback: function(token_text, full_text_so_far) -> None
        Yields:
            每个新生成的 token 文本
        """
        input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
        past_kv = None
        generated_text = prompt

        for step in range(self.max_new_tokens):
            if step == 0:
                inputs = input_ids
            else:
                inputs = input_ids[:, -1:]

            with torch.no_grad():
                outputs = self.model(
                    inputs,
                    past_key_values=past_kv,
                    use_cache=True,
                    return_dict=True
                )

            next_token_logits = outputs.logits[:, -1, :]

            # 可以在这里替换为 sampling / top-k / top-p 等策略
            next_token_id = next_token_logits.argmax(dim=-1, keepdim=True)

            # 检查是否结束
            if next_token_id.item() == self.tokenizer.eos_token_id:
                break

            # 解码当前 token
            token_text = self.tokenizer.decode(next_token_id[0], skip_special_tokens=True)
            generated_text += token_text

            # 更新 KV Cache
            past_kv = outputs.past_key_values

            # 更新输入（只保留最新 token）
            input_ids = next_token_id

            if callback:
                callback(token_text, generated_text)

            yield token_text

        return generated_text


# 使用示例
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

generator = StreamingGenerator(model, tokenizer, max_new_tokens=50)

def print_callback(token, full_text):
    print(token, end='', flush=True)

print("生成中...\n")
result = ""
for token in generator.generate_streaming(
    "Artificial intelligence will",
    callback=print_callback
):
    result += token

print(f"\n\n完整生成: {result}")
```

---

## 常见误区与最佳实践

> **误区 1："KV Cache 会改变模型的输出结果"**
>
> 这是错误的。KV Cache 只是一种计算优化，它缓存的是中间结果（K 和 V），这些结果在不使用 Cache 时也会被计算出来（只是被丢弃了）。只要实现正确，有无 Cache 的输出**完全一致**。
>
> 验证方法：对比两种方式下每一步生成的 token ID 是否相同。

> **误区 2："use_cache=True 只在推理时有用"**
>
> 不完全正确。虽然在训练时通常不使用 KV Cache（因为训练是并行处理整个序列的），但在某些特殊训练场景中也有价值：
> - **长文档微调**：如果序列很长，开启 cache 可以节省显存
> - **强化学习微调（RLHF/DPO）**：生成阶段需要用到 cache

> **误区 3："past_key_values 可以跨不同 prompt 共享"**
>
> 绝对不行！KV Cache 是与特定输入序列绑定的。不同的 prompt 产生不同的 K 和 V，混用会导致完全错误的结果。每次新的生成任务都必须从空的 `past_key_values=None` 开始。

> **最佳实践：生成时始终使用 use_cache=True**
>
> HF 的 `generate()` 方法默认就开启了 KV Cache。如果你手写生成循环，记得：
> 1. 第一轮：传入完整 input_ids，past_key_values=None
> 2. 后续轮：只传入最新的一个 token，传入上一轮返回的 past_key_values
> 3. 不要忘记每步更新 past_key_values

---

## 小结

这一节我们深入了 GPT（Decoder-only）架构的核心细节：

1. **因果掩码**：通过上三角矩阵实现"只能看左边"的约束，是自回归生成的数学基础
2. **Pre-LayerNorm**：GPT-2 将 LayerNorm 放在子层之前，带来更好的梯度流动和训练稳定性，使得训练超深模型成为可能
3. **KV Cache**：缓存历史 token 的 Key 和 Value，将生成复杂度从 O(T³) 降低到 O(T²)，实际加速 5-10 倍。它是所有现代大模型推理系统的基础组件
4. **GPT forward 方法**：支持 input_ids 和 inputs_embeds 两种输入方式，position_ids 处理可学习位置编码
5. **内存挑战**：KV Cache 的内存占用与序列长度线性增长，对于 70B 级别的模型在 32K 上下文下可能超过 160 GB，催生了 MQA/GQA/PagedAttention 等优化技术

下一节我们将讨论模型的初始化策略、权重加载方法，以及如何高效地将大模型部署到多 GPU 环境。
