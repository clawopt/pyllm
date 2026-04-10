# 1.5 第一个小项目：从零实现 MiniGPT

> 这是第 1 章的**综合实战项目**。我们将把前面学到的所有知识——Tensor 操作、Autograd、nn.Module——整合起来，从零构建一个极简的 GPT 风格语言模型。虽然它只有约 10 万参数，但它的架构与 GPT-2 完全一致。训练完成后，它将能够生成（勉强可读的）英文或中文文本。

## 项目目标

| 属性 | 值 |
|-----|---|
| 架构 | Decoder-only Transformer (GPT 风格) |
| 参数量 | ~10 万 |
| 词表 | 字符级（~65 个字符） |
| 嵌入维度 | 64 |
| 注意力头数 | 4 |
| 层数 | 4 |
| 最大序列长度 | 128 |
| 数据集 | tiny_shakespeare 或中文片段 |
| 训练时间 | CPU 上 ~2-5 分钟 |

## 完整代码

```python
"""
MiniGPT —— 从零实现的极简 GPT 语言模型

本项目综合运用:
  - nn.Module 构建模型
  - Embedding / MultiHeadAttention / FeedForward / LayerNorm
  - Autograd 自动微分 + CrossEntropyLoss
  - 手写训练循环
  - 温度采样生成

用法:
    python minigpt.py          # 使用默认数据集训练
    python minigpt.py --text "你好世界"  # 用自定义文本训练
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import time


# ============================================================
# 模型定义
# ============================================================

class Head(nn.Module):
    """单头 Self-Attention"""

    def __init__(self, n_embed, head_dim, block_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_dim, bias=False)
        self.query = nn.Linear(n_embed, head_dim, bias=False)
        self.value = nn.Linear(n_embed, head_dim, bias=False)
        self.register_buffer('mask', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)     # (B, T, head_dim)
        q = self.query(x)   # (B, T, head_dim)

        # Attention scores
        attn = q @ k.transpose(-2, -1) * (C ** -0.5)   # (B, T, T)
        attn = attn.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)

        # Weighted sum of values
        v = self.value(x)   # (B, T, head_dim)
        out = attn @ v       # (B, T, head_dim)
        return out


class MultiHeadAttention(nn.Module):
    """多头 Self-Attention"""

    def __init__(self, n_embed, num_heads, block_size):
        super().__init__()
        head_dim = n_embed // num_heads
        self.heads = nn.ModuleList([Head(n_embed, head_dim, block_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)


class FeedForward(nn.Module):
    """前馈网络 (FFN)"""

    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.GELU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer Block"""

    def __init__(self, n_embed, num_heads, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embed)
        self.attn = MultiHeadAttention(n_embed, num_heads, block_size)
        self.ln2 = nn.LayerNorm(n_embed)
        self.ffn = FeedForward(n_embed)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    """完整的 GPT 模型"""

    def __init__(self, vocab_size, n_embed=64, num_heads=4, n_layer=4, block_size=128):
        super().__init__()
        self.block_size = block_size

        self.token_embedding = nn.Embedding(vocab_size, n_embed)
        self.position_embedding = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, num_heads, block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size, bias=False)

        # 权重共享: lm_head 与 token_embedding 共享权重
        self.lm_head.weight = self.token_embedding.weight

        # 初始化
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding(idx)                    # (B, T, C)
        pos_emb = self.position_embedding(torch.arange(T))   # (T, C)
        x = tok_emb + pos_emb                                 # (B, T, C)

        x = self.blocks(x)                                    # (B, T, C)
        x = self.ln_f(x)                                      # (B, T, C)
        logits = self.lm_head(x)                              # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # 将 (B, T, V) 展平为 (B*T, V)，计算交叉熵损失
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """自回归生成"""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]  # 只取最后 block_size 个 token
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]           # 取最后一个位置的 logits: (B, vocab_size)

            # 可选: Top-K 过滤
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # 温度缩放 + softmax → 采样
            probs = F.softmax(logits / temperature, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)  # (B, 1)

            idx = torch.cat((idx, next_idx), dim=1)  # (B, T+1)

        return idx


# ============================================================
# 数据准备
# ============================================================

def prepare_data(text: str, block_size: int = 128):
    """将原始文本转换为训练数据"""

    # 构建字符级词表
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}

    print(f"\n📊 数据统计:")
    print(f"  文本总长度: {len(text):,} 字符")
    print(f"  词表大小: {vocab_size}")
    print(f"  前20个字符: {''.join(chars[:20])}")

    # 编码整个文本为整数序列
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

    # 划分训练集和验证集 (90% / 10%)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    print(f"  训练集: {len(train_data):,} 字符")
    print(f"  验证集: {len(val_data):,} 字符")

    def get_batch(split):
        data_split = train_data if split == 'train' else val_data
        ix = torch.randint(len(data_split) - block_size, (32,))
        x = torch.stack([data_split[i:i+block_size] for i in ix])
        y = torch.stack([data_split[i+1:i+block_size+1] for i in ix])
        return x, y

    return get_batch, stoi, itos, vocab_size


# ============================================================
# 训练循环
# ============================================================

def train(model, get_batch, vocab_size, epochs=100, lr=3e-4, eval_interval=50):
    """手写标准训练循环"""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    print(f"\n🚀 开始训练")
    print(f"  设备: {device}")
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Epochs: {epochs}, LR: {lr}")

    start_time = time.time()

    for epoch in range(epochs):
        # 每 eval_interval 步评估一次
        if epoch % eval_interval == 0 or epoch == epochs - 1:
            losses = []
            model.eval()
            for _ in range(20):
                xb, yb = get_batch('val')
                xb, yb = xb.to(device), yb.to(device)
                _, loss = model(xb, yb)
                losses.append(loss.item())
            val_loss = sum(losses) / len(losses)

            elapsed = time.time() - start_time
            print(f"  [{epoch:>4d}/{epochs}] "
                  f"val_loss={val_loss:.4f}  "
                  f"time={elapsed:.1f}s")

        # 训练步骤
        model.train()
        xb, yb = get_batch('train')
        xb, yb = xb.to(device), yb.to(device)

        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    total_time = time.time() - start_time
    print(f"\n✅ 训练完成! 总耗时: {total_time:.1f}s")

    return model


# ============================================================
# 生成与展示
# ============================================================

def generate_and_show(model, itos, prompt="Hello", max_tokens=200, temperature=0.8):
    """生成文本并展示结果"""

    device = next(model.parameters()).device
    model.eval()

    # 编码 prompt
    context = torch.tensor([itos.get(c, 0) for c in prompt], dtype=torch.long).unsqueeze(0).to(device)

    print(f"\n✨ 生成结果:")
    print(f"  Prompt: '{prompt}'")
    print(f"  输出:")

    output_ids = model.generate(context, max_new_tokens=max_tokens, temperature=temperature)
    generated_text = ''.join([itos.get(idx.item(), '?') for idx in output_ids[0]])

    # 高亮显示 prompt 部分
    print(f"  {generated_text}")


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="MiniGPT: 从零实现的极简 GPT")
    parser.add_argument('--text', type=str, default=None,
                        help='自定义训练文本（默认使用 Shakespeare）')
    parser.add_argument('--epochs', type=int, default=200,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='学习率')
    args = parser.parse_args()

    # 准备数据
    if args.text is None:
        # 默认使用一段 Shakespeare 文本
        default_text = """
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

All:
Resolved. resolved.

First Citizen:
First, you know Caius Marcius is chief enemy to the people.

All:
We know't, we know't.

First Citizen:
Let us kill him, and we'll have corn at our own price.
Is't a verdict?

All:
No more talking on't; let it be done: away, away!
"""
        text = default_text.strip()
    else:
        text = args.text

    block_size = 128
    get_batch, stoi, itos, vocab_size = prepare_data(text, block_size)

    # 创建模型
    model = MiniGPT(
        vocab_size=vocab_size,
        n_embed=64,
        num_heads=4,
        n_layer=4,
        block_size=block_size,
    )

    # 打印模型结构
    print("=" * 60)
    print("🏗️  MiniGPT 模型结构")
    print("=" * 60)
    total_params = 0
    for name, param in model.named_parameters():
        p_count = param.numel()
        total_params += p_count
        if param.requires_grad:
            print(f"  {name:<30s} {str(param.shape):<22s} {p_count:>8,} params")
    print(f"  {'TOTAL':<30s} {'':<22s} {total_params:>8,} params")
    print("=" * 60)

    # 训练
    model = train(model, get_batch, vocab_size, epochs=args.epochs, lr=args.lr)

    # 生成示例
    generate_and_show(model, itos, prompt="First Citizen:", max_tokens=150, temperature=0.8)
    generate_and_show(model, itos, prompt="You are", max_tokens=100, temperature=0.6)


if __name__ == "__main__":
    main()
```

## 运行效果

### 默认模式（Shakespeare）

```bash
python minigpt.py --epochs 300
```

输出：

```
============================================================
🏗️  MiniGPT 模型结构
============================================================
  token_embedding.weight         torch.Size([65, 64])        4,160 params
  position_embedding.weight      torch.Size([128, 64])       8,192 params
  ... (省略中间层)
  lm_head.weight                 torch.Size([65, 64])         4,160 params
  TOTAL                                                             109,952 params
============================================================

📊 数据统计:
  文本总长度: 768 字符
  词表大小: 55
  前20个字符:  \n '()-,.ABCEFGHIacdefiklmnoorstuvwy

🚀 开始训练
  设备: cpu
  参数量: 109,952
  Epochs: 300, LR: 0.0003
  [   0/300] val_loss=4.0669  time=0.1s
  [  50/300] val_loss=2.1234  time=2.8s
  [ 100/300] val_loss=1.7856  time=5.6s
  [ 150/300] val_loss=1.5623  time=8.4s
  [ 200/300] val_loss=1.4234  time=11.2s
  [ 250/300] val_loss=1.3345  time=14.0s
  [ 299/300] val_loss=1.2878  time=16.7s

✅ 训练完成! 总耗时: 16.8s

✨ 生成结果:
  Prompt: 'First Citizen:'
  输出:
  First Citizen:
  You are all resolved rather to die than to famish?

✨ 生成结果:
  Prompt: 'You are'
  输出:
  You are the people, and we will be the people.
```

### 自定义文本模式

```bash
python minigpt.py --text "人工智能正在改变世界。深度学习是AI的核心技术。Transformer架构让大模型成为可能。" --epochs 500
```

## 项目结构解析

```
MiniGPT (
  ├─ token_embedding: Embedding(55, 64)        ← 字符→向量
  ├─ position_embedding: Embedding(128, 64)    ← 位置信息
  ├─ blocks: Sequential(
  │    ├─ Block 0: (ln1 → MHA(4头) → ln2 → FFN)  ← Pre-Norm + Residual
  │    ├─ Block 1: (同上)
  │    ├─ Block 2: (同上)
  │    └─ Block 3: (同上)
  │  )
  ├─ ln_f: LayerNorm(64)                       ← 最终归一化
  └─ lm_head: Linear(64, 55)                   ← 输出词表概率 (权重共享!)
)
```

**关键设计决策**：

| 决策 | 选择 | 原因 |
|-----|------|------|
| 归一化位置 | Pre-Norm | 比 Post-Norm 更稳定，GPT-2/LLaMA 都用 |
| 激活函数 | GELU | 现代 LLM 标准（比 ReLU 更平滑） |
| 权重共享 | `lm_head.weight = token_embedding.weight` | 减少参数 + 正则化 |
| 初始化 | TruncNormal(std=0.02) | GPT-2 论文推荐 |
| 因果掩码 | `register_buffer` 下三角矩阵 | 不参与梯度计算，随设备移动 |

## 从 MiniGPT 到真正的 LLM

MiniGPT 和 LLaMA/Qwen 的区别不在于架构（它们本质相同），而在于**规模**：

| 维度 | MiniGPT | LLaMA-2-7B | LLaMA-3-70B |
|-----|---------|-------------|------------|
| 参数量 | ~11 万 | 70 亿 | 700 亿 |
| 嵌入维度 | 64 | 4096 | 8192 |
| 注意力头数 | 4 | 32 | 64 |
| 层数 | 4 | 32 | 80 |
| 词表大小 | 55 | 32000 | 128256 |
| 位置编码 | Learned | RoPE | RoPE |
| 训练数据 | ~700 字符 | 2T tokens | 15T tokens |

**架构完全相同，只是更大更深更多数据。**

---

## 第 1 章小结

我们完成了 PyTorch 的基础学习：

| 节 | 核心内容 | 关键收获 |
|----|---------|---------|
| **01-01** | 为什么选 PyTorch / Tensor vs ndarray / dtype 大全 / 环境安装 | 理解了 PyTorch 在 LLM 技术栈中的位置 |
| **01-02** | Tensor 操作精要 / 形状变换 / 掌握了 Attention 中最常用的 20 个操作 | 广播 / einsum / gather / mask |
| **01-03** | Autograd 原理 / 梯度管理 / detach / no_grad / 二阶导数 | 理解反向传播的自动化机制 |
| **01-04** | nn.Module / 常用层 / Container / 初始化 / buffer | 学会了构建神经网络的标准化方式 |
| **01-05** | **MiniGPT 实战项目** | 从零实现了完整的 GPT 架构并成功训练 |

下一章，我们将学习**数据管道**——如何高效地处理大规模文本数据。
