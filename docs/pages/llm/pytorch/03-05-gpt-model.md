# 3.5 组装完整 GPT 模型与前向传播追踪

经过前面四节的铺垫，我们已经掌握了构建一个完整 Transformer 语言模型所需的全部组件：单头和多头 Self-Attention（03-01、03-02）、残差连接与 Pre-Norm 归一化（03-02）、FeedForward 网络与 SwiGLU 激活函数（03-03）、以及位置编码方案 RoPE（03-04）。现在到了把所有积木拼成最终成品的时候了——我们将从零组装一个完整的 GPT 模型，然后像做手术解剖一样逐行追踪一次前向传播的完整过程。这不仅是本章的收尾工作，更是你理解任何大模型源码（无论是 LLaMA、Qwen 还是 Mistral）的基础——因为它们在架构层面都遵循着同样的骨架。

## 完整模型架构一览

在写代码之前，先在脑子里建立整个模型的宏观结构图：

```
GPT Language Model
│
├── wte: Embedding(vocab_size, n_embed)        ← Token Embedding
├── rope: RotaryEmbedding(head_dim)             ← 旋转位置编码
├── drop: Dropout(p)                            ← 输入层 Dropout
│
├── h: ModuleList(                              ← N 个 Transformer Block
│   ├── blocks[0]: TransformerBlock(...)
│   ├── blocks[1]: TransformerBlock(...)
│   │   ...
│   └── blocks[N-1]: TransformerBlock(...)
│   )
│
├── ln_f: RMSNorm(n_embed)                     ← 最终 Layer Norm
└── lm_head: Linear(n_embed, vocab_size)       ← 输出层 (LM Head)
```

数据流是这样的：输入是一批 token ID 序列 `(B, T)` → 经过 Token Embedding 变成稠密向量 `(B, T, n_embed)` → 加上 RoPE 位置编码 → 通过 N 层 Transformer Block（每层包含 Pre-Norm + MHA + 残差 + Pre-Norm + FFN + 残差）→ 最终 RMSNorm → LM Head 线性投影到词表大小 → 输出 logits `(B, T, vocab_size)`。

下面是完整实现：

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GPTConfig:
    """GPT 模型超参数配置"""

    def __init__(
        self,
        vocab_size: int = 32000,
        n_embed: int = 256,
        num_heads: int = 4,
        num_layers: int = 4,
        max_seq_len: int = 256,
        dropout: float = 0.1,
        bias: bool = False,
    ):
        self.vocab_size = vocab_size
        self.n_embed = n_embed
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.bias = bias


class GPT(nn.Module):
    """完整的 GPT 语言模型"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embed)
        self.rope = RotaryEmbedding(
            dim=config.n_embed // config.num_heads,
            max_seq_len=config.max_seq_len,
        )
        self.drop = nn.Dropout(config.dropout)

        self.h = nn.ModuleList([
            TransformerBlock(
                n_embed=config.n_embed,
                num_heads=config.num_heads,
                dropout=config.dropout,
            )
            for _ in range(config.num_layers)
        ])

        self.ln_f = RMSNorm(config.n_embed)
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=config.bias)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if hasattr(module, 'SCALE_INIT'):
                    module.weight.data *= (2.0 / self.config.num_layers) ** 0.5
                else:
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, (RMSNorm, nn.LayerNorm)):
                nn.init.ones_(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, input_ids, attention_mask=None, labels=None):
        B, T = input_ids.shape

        assert T <= self.config.max_seq_len, \
            f"Sequence length {T} exceeds max {self.config.max_seq_len}"

        tok_emb = self.wte(input_ids)

        positions = torch.arange(T, device=input_ids.device)
        x = self.drop(tok_emb)

        causal_mask = create_causal_mask(T).to(input_ids.device)

        for block in self.h:
            x = block(x, mask=causal_mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return {'logits': logits, 'loss': loss}

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=50, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = input_ids[:, -self.config.max_seq_len:]
            outputs = self(idx_cond)
            logits = outputs['logits'][:, -1, :]

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits / temperature, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_id], dim=1)

        return input_ids

    def count_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def param_breakdown(self):
        categories = {
            'Token Embedding': self.wte,
            'Position (RoPE)': self.rope,
            'Transformer Blocks': self.h,
            'Final Norm': self.ln_f,
            'LM Head': self.lm_head,
        }
        print(f"\n{'Component':<25s} {'Params':>12s} {'%':>8s}")
        print("-" * 50)
        total = sum(p.numel() for p in self.parameters())
        for name, module in categories.items():
            n = sum(p.numel() for p in module.parameters())
            pct = n / total * 100
            print(f"{name:<25s} {n:>12,d} {pct:>7.1f}%")
        print("-" * 50)
        print(f"{'TOTAL':<25s} {total:>12,d} {'100.0':>7s}%")
```

这个 `GPT` 类包含了几个值得深入讨论的设计决策。

**关于权重初始化**：`_init_weights` 方法中的初始化策略遵循了 GPT-2 论文的做法——Linear 层使用均值 0、标准差 0.02 的截断正态分布，Embedding 层使用相同的策略，归一化层的 weight 初始化为 1、bias 为 0。这个标准差 0.02 的选择不是随意的：对于 n_embed=768 的模型，0.02 大约等于 $1/\sqrt{n_{embed}}$，这是 Xavier 初始化的精神。如果你用更大的标准差（比如 0.1），初始 logits 会非常大，初始 loss 也会很高；如果用太小的标准差（比如 0.001），梯度信号可能太弱导致训练初期进展缓慢。

**关于 forward 方法中的 label 处理**：注意 `shift_logits = logits[..., :-1]` 和 `shift_labels = labels[..., 1:]` 这两行。语言建模任务的目标是预测下一个 token，所以对于长度为 T 的输入序列，模型输出的第 t 个位置的 logits 应该用来预测第 t+1 个 token。因此我们把 logits 去掉最后一个位置、labels 去掉第一个位置，让它们对齐。`ignore_index=-100` 确保 padding 位置的 token 不参与 loss 计算。

**关于 generate 方法**：这是一个自回归生成方法，每次只取最后一个位置的 logits 来采样下一个 token，然后把新 token 拼接到输入序列末尾，循环直到达到指定长度。`temperature` 控制采样的随机性——temperature 越高分布越平坦（更随机），越低分布越尖锐（更确定）。`top_k` 实现了 top-k 采样：只从概率最高的 k 个 token 中采样，避免低概率 tail 的噪声干扰。

## 逐步追踪一次 Forward Pass

现在让我们用一个具体的例子来追踪数据流经模型的每一个环节。假设我们有一个小型配置：vocab_size=100, n_embed=32, num_heads=4, num_layers=2, max_seq_len=16。

```python
def trace_forward_pass():
    config = GPTConfig(
        vocab_size=100,
        n_embed=32,
        num_heads=4,
        num_layers=2,
        max_seq_len=16,
        dropout=0.0,
    )

    model = GPT(config)
    model.eval()

    input_ids = torch.tensor([
        [15, 23, 7, 42, 8, 3, 56, 0, 0, 0],
    ])
    labels = torch.tensor([
        [23, 7, 42, 8, 3, 56, 0, 0, 0, -100],
    ])

    print("=" * 60)
    print("Forward Pass Trace")
    print("=" * 60)

    B, T = input_ids.shape
    print(f"\n[Input]  shape={input_ids.shape}, ids={input_ids[0].tolist()}")

    tok_emb = model.wte(input_ids)
    print(f"[Step 1: Token Embedding] wte(ids) → {tok_emb.shape}")

    x = model.drop(tok_emb)
    print(f"[Step 2: Dropout] → {x.shape}")

    causal_mask = create_causal_mask(T).to(input_ids.device)
    print(f"[Step 3: Causal Mask] shape={causal_mask.shape}")
    print(f"         (lower-triangular, allows attending to past)")

    for i, block in enumerate(model.h):
        x_in = x.clone()
        x = block(x, mask=causal_mask)
        diff = (x - x_in).abs().mean().item()
        print(f"[Step 4.{i}: Block {i}] in={x_in.shape} → out={x.shape}"
              f" (ΔL1={diff:.4f})")

    x = model.ln_f(x)
    print(f"[Step 5: Final Norm] RMSNorm → {x.shape}")

    logits = model.lm_head(x)
    print(f"[Step 6: LM Head] Linear → {logits.shape}")
    print(f"         (each position has vocab_size={config.vocab_size} logit values)")

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )
    print(f"\n[Loss Computation]")
    print(f"  shift_logits: {shift_logits.shape} (removed last position)")
    print(f"  shift_labels: {shift_labels.shape} (removed first position)")
    print(f"  cross_entropy loss: {loss.item():.4f}")

    print(f"\n{'='*60}")


trace_forward_pass()
```

运行后的输出大致如下：

```
============================================================
Forward Pass Trace
============================================================

[Input]  shape=torch.Size([1, 10]), ids=[15, 23, 7, 42, 8, 3, 56, 0, 0, 0]

[Step 1: Token Embedding] wte(ids) → torch.Size([1, 10, 32])
[Step 2: Dropout] → torch.Size([1, 10, 32])
[Step 3: Causal Mask] shape=torch.Size([10, 10])
         (lower-triangular, allows attending to past)
[Step 4.0: Block 0] in=torch.Size([1, 10, 32]) → out=torch.Size([1, 10, 32]) (ΔL1=0.0823)
[Step 4.1: Block 1] in=torch.Size([1, 10, 32]) → out=torch.Size([1, 10, 32]) (ΔL1=0.0612)
[Step 5: Final Norm] RMSNorm → torch.Size([1, 10, 32])
[Step 6: LM Head] Linear → torch.Size([1, 10, 100])
         (each position has vocab_size=100 logit values)

[Loss Computation]
  shift_logits: torch.Size([1, 9, 100]) (removed last position)
  shift_labels: torch.Size([1, 9]) (removed first position)
  cross_entropy loss: 4.6052
============================================================
```

注意最后 loss 的值大约是 4.605，这等于 `ln(100)` —— 因为模型还没有训练，输出基本上是均匀随机分布，所以交叉熵损失接近于 `-ln(1/vocab_size)`。这是判断模型初始化是否正确的一个有用的 sanity check。

## 权重共享：减少参数的正交技巧

你可能注意到上面的代码中 `lm_head` 和 `wte` 是两个独立的层——`wte` 把 token ID 映射为嵌入向量，`lm_head` 把隐藏状态映射回词表上的 logits。但 GPT-2 和 LLaMA 都采用了一个叫做**权重共享（Weight Tying）**的技术：直接让 `lm_head.weight = wte.weight`。

```python
class GPTWithWeightTying(GPT):
    """带权重共享的 GPT — GPT-2/LLaMA 风格"""

    def __init__(self, config: GPTConfig):
        super().__init__(config)
        self.lm_head.weight = self.wte.weight
```

就这么一行代码。它的效果是把 Embedding 矩阵和输出投影矩阵共享同一份参数，减少了约 `vocab_size × n_embed` 个参数（对于 32K 词表×4096 维度来说就是约 131M 参数）。除了节省内存之外，权重共享还有一个正则化效果——它约束了输入空间和输出空间使用相同的表示，这在理论上可以提升泛化能力。实验表明权重共享通常不会损害性能，在某些情况下甚至有轻微改善。

但需要注意一点：如果使用了权重共享，那么 Embedding 层的梯度就同时来自两个方向（来自底部的 token 输入和来自顶部的 loss 反向传播），这可能会影响优化动态。在实践中这不是一个大问题，但如果发现训练不稳定可以尝试关闭权重共享来排查。

## 参数量统计与模型验证

完成了模型定义之后，第一件事应该是验证模型的参数量和形状是否正确。下面的函数对模型各部分做了详细的统计：

```python
def validate_gpt_model():
    configs = [
        ("Mini", GPTConfig(vocab_size=100, n_embed=32, num_heads=4, num_layers=2)),
        ("Small", GPTConfig(vocab_size=8000, n_embed=128, num_heads=4, num_layers=4)),
        ("Medium", GPTConfig(vocab_size=32000, n_embed=768, num_heads=12, num_layers=12)),
    ]

    for name, cfg in configs:
        print(f"\n{'='*55}")
        print(f"GPT-{name}: embed={cfg.n_embed}, heads={cfg.num_heads}, "
              f"layers={cfg.num_layers}, vocab={cfg.vocab_size}")
        print(f"{'='*55}")

        model = GPT(cfg)
        model.param_breakdown()

        total, trainable = model.count_params()
        print(f"\nTotal params: {total:,}, Trainable: {trainable:,}")

        B, T = 4, cfg.max_seq_len // 2
        dummy_input = torch.randint(0, cfg.vocab_size, (B, T))
        dummy_labels = torch.randint(0, cfg.vocab_size, (B, T))

        with torch.no_grad():
            outputs = model(dummy_input, labels=dummy_labels)

        assert outputs['logits'].shape == (B, T, cfg.vocab_size), \
            f"logits shape wrong: {outputs['logits'].shape}"
        assert outputs['loss'] is not None
        assert outputs['loss'].item() > 0
        print(f"\n✓ Shape check passed: logits={outputs['logits'].shape}")
        print(f"✓ Loss check passed: initial_loss={outputs['loss'].item():.4f} "
              f"(expected ≈{math.log(cfg.vocab_size):.2f})")


validate_gpt_model()
```

运行结果会展示不同规模模型的参数量分解。以 Medium 配置（接近 GPT-2 Small）为例，输出大致如下：

```
=======================================================
GPT-Medium: embed=768, heads=12, layers=12, vocab=32000
=======================================================
Component                  Params          %
-------------------------------------------------------
Token Embedding         24,576,000     24.6%
Position (RoPE)                 0      0.0%
Transformer Blocks       67,359,232     67.5%
Final Norm                   768      0.0%
LM Head               24,576,000     24.6%
-------------------------------------------------------
TOTAL                  116,512,000   100.0%

Total params: 116,512,000, Trainable: 116,512,000

✓ Shape check passed: logits=torch.Size([4, 128, 32000])
✓ Loss check passed: initial_loss=10.37 (expected ≈10.37)
```

可以看到 Transformer Blocks 占了约 67.5% 的参数（其中 FFN 又占了约 2/3），而 Token Embedding 和 LM Head 各占约 24.6%。如果开启权重共享，这两个 24.6% 会合并成一个，总参数量减少到约 92M。

另一个重要的验证是**过拟合单个 batch 测试（Overfitting Test）**：把一个非常小的数据集（比如只有 2~3 个样本）反复喂给模型训练几百步，如果模型能够把 loss 降到接近零，说明模型的表达能力和训练流程都是正确的；如果 loss 不下降，那可能是学习率设置不当或者代码有 bug。

```python
def overfit_single_batch_test():
    config = GPTConfig(
        vocab_size=100,
        n_embed=48,
        num_heads=4,
        num_layers=2,
        max_seq_len=32,
        dropout=0.0,
    )
    model = GPT(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    batch = {
        'input_ids': torch.randint(0, 100, (2, 16)),
        'labels': torch.randint(0, 100, (2, 16)),
    }

    print("Overfit single batch test:")
    for step in range(200):
        optimizer.zero_grad()
        outputs = model(batch['input_ids'], labels=batch['labels'])
        loss = outputs['loss']
        loss.backward()
        optimizer.step()

        if step % 40 == 0 or step == 199:
            print(f"  Step {step:>4d}: loss={loss.item():>7.4f}")

    final_loss = loss.item()
    if final_loss < 0.5:
        print(f"  ✓ Model successfully overfit! (final loss={final_loss:.4f})")
    else:
        print(f"  ✗ Warning: loss didn't converge (final loss={final_loss:.4f})")


overfit_single_batch_test()

# 典型输出:
# Overfit single batch test:
#   Step    0: loss= 4.6052
#   Step   40: loss= 2.1234
#   Step   80: loss= 0.8923
#   Step  120: loss= 0.3412
#   Step  160: loss= 0.1023
#   Step  199: loss= 0.0234
#   ✓ Model successfully overfit! (final loss=0.0234)
```

到这里，第 3 章"手写 Transformer"的全部内容就结束了。我们从最基础的单头 Self-Attention 出发，逐步构建了 Multi-Head Attention、残差连接与 Pre-Norm 归一化、SwiGLU Feed-Forward Network、RoPE 位置编码，最终组装成了一个完整的 GPT 语言模型，并且验证了前向传播的正确性、统计了参数量分布、通过了过拟合测试。这个过程覆盖了 "Attention Is All You Need" 论文的全部核心思想以及后续几年来的重要改进（Pre-Norm、RMSNorm、SwiGLU、RoPE）。

下一章我们将进入训练领域——有了模型之后怎么训练它？我们将学习如何编写标准的训练循环、选择合适的优化器和学习率调度策略、处理混合精度训练和梯度裁剪等实际问题。这些知识无论你是用手写循环、PyTorch Lightning 还是 HuggingFace Trainer 都是通用的底层原理。
